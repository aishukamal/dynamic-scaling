"""Kubernetes actuator for the per-replica-pod layout (GPU-M2 stage 2). stdlib only.

Scale-out: patch the RayCluster sampler worker group ``replicas += 1``, wait until the control
actor's ``/status`` node inventory shows a rollout node with a free GPU (the new pod joined Ray),
then let the caller POST ``/scale``. Scale-in: the caller POSTs ``/scale`` with ``release=true``
(the plugin destroys the newest replica, freeing its GPU); once a rollout node is fully free the
actuator maps its IP to the worker pod and patches ``replicas -= 1`` together with
``scaleStrategy.workersToDelete=[pod]`` so KubeRay removes THAT pod and not an active one.

Every transition is returned as a dict of timestamps so the controller writes it to the JSONL.
"""

from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.request
from typing import Any

SA_DIR = "/var/run/secrets/kubernetes.io/serviceaccount"


class K8sClient:
    def __init__(self, base_url: str | None = None, token: str | None = None, ca_file: str | None = None, timeout_s: float = 15.0) -> None:
        host = os.environ.get("KUBERNETES_SERVICE_HOST", "kubernetes.default.svc")
        port = os.environ.get("KUBERNETES_SERVICE_PORT", "443")
        self.base_url = (base_url or f"https://{host}:{port}").rstrip("/")
        self.token = token if token is not None else open(os.path.join(SA_DIR, "token")).read().strip()
        ca = ca_file or os.path.join(SA_DIR, "ca.crt")
        self.ctx = ssl.create_default_context(cafile=ca) if os.path.exists(ca) else ssl.create_default_context()
        self.timeout_s = timeout_s

    def _req(self, method: str, path: str, body: Any = None, ctype: str = "application/json") -> Any:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base_url + path, data=data, method=method)
        req.add_header("Authorization", f"Bearer {self.token}")
        req.add_header("Accept", "application/json")
        if data is not None:
            req.add_header("Content-Type", ctype)
        with urllib.request.urlopen(req, timeout=self.timeout_s, context=self.ctx) as r:  # noqa: S310
            raw = r.read().decode()
            return json.loads(raw) if raw else {}

    def get_raycluster(self, ns: str, name: str) -> dict:
        return self._req("GET", f"/apis/ray.io/v1/namespaces/{ns}/rayclusters/{name}")

    def json_patch_raycluster(self, ns: str, name: str, ops: list[dict]) -> dict:
        return self._req("PATCH", f"/apis/ray.io/v1/namespaces/{ns}/rayclusters/{name}", ops, ctype="application/json-patch+json")

    def list_pods(self, ns: str, selector: str) -> list[dict]:
        return list(self._req("GET", f"/api/v1/namespaces/{ns}/pods?labelSelector={urllib.request.quote(selector)}").get("items", []))

    def delete_pod(self, ns: str, name: str) -> dict:
        return self._req("DELETE", f"/api/v1/namespaces/{ns}/pods/{name}")


class RayClusterGroup:
    """One worker group of a RayCluster (by groupName)."""

    def __init__(self, k8s: K8sClient, ns: str, name: str, group: str) -> None:
        self.k8s = k8s
        self.ns = ns
        self.name = name
        self.group = group

    def _index_and_spec(self) -> tuple[int, dict]:
        rc = self.k8s.get_raycluster(self.ns, self.name)
        for i, g in enumerate(rc["spec"].get("workerGroupSpecs", [])):
            if g.get("groupName") == self.group:
                return i, g
        raise KeyError(f"worker group {self.group} not found in RayCluster {self.ns}/{self.name}")

    def replicas(self) -> int:
        return int(self._index_and_spec()[1].get("replicas", 0))

    def set_replicas(self, n: int, workers_to_delete: list[str] | None = None) -> dict:
        i, g = self._index_and_spec()
        ops = [{"op": "replace" if "replicas" in g else "add", "path": f"/spec/workerGroupSpecs/{i}/replicas", "value": int(n)}]
        if "maxReplicas" in g and int(g["maxReplicas"]) < int(n):
            ops.append({"op": "replace", "path": f"/spec/workerGroupSpecs/{i}/maxReplicas", "value": int(n)})
        if workers_to_delete is not None:
            ops.append({"op": "add", "path": f"/spec/workerGroupSpecs/{i}/scaleStrategy", "value": {"workersToDelete": list(workers_to_delete)}})
        return self.k8s.json_patch_raycluster(self.ns, self.name, ops)

    def pods(self) -> list[dict]:
        return self.k8s.list_pods(self.ns, f"ray.io/cluster={self.name},ray.io/group={self.group}")

    def pod_by_ip(self, ip: str) -> dict | None:
        for p in self.pods():
            if (p.get("status") or {}).get("podIP") == ip:
                return p
        return None


def rollout_nodes(status: dict) -> list[dict]:
    return [n for n in (status.get("nodes") or []) if isinstance(n, dict) and n.get("rollout_node", 0) and n.get("gpu_total", 0)]


def replicas_per_ip(status: dict) -> dict[str, int]:
    """Standalone replicas (active + parked) per node IP, from the plugin's own pool status."""
    pool = status.get("pool") or {}
    out: dict[str, int] = {}
    for r in list(pool.get("active") or []) + list(pool.get("parked") or []):
        ip = str(r.get("address") or "").split(":")[0]
        if ip:
            out[ip] = out.get(ip, 0) + 1
    return out


def gpus_free(n: dict, status: dict) -> float:
    """Free GPUs on a rollout node: Ray's per-node availability when reported, else gpu_total minus the
    replicas the plugin itself has on that node (the first live run exported gpu_available=null)."""
    ga = n.get("gpu_available")
    if ga is not None:
        return float(ga)
    return float(n.get("gpu_total", 0)) - replicas_per_ip(status).get(str(n.get("ip")), 0)


def free_rollout_nodes(status: dict) -> list[dict]:
    """Rollout nodes with at least one unreserved GPU."""
    return [n for n in rollout_nodes(status) if gpus_free(n, status) >= 1.0]


def fully_free_rollout_nodes(status: dict) -> list[dict]:
    return [n for n in rollout_nodes(status) if float(n.get("gpu_total", 0)) > 0 and gpus_free(n, status) >= float(n.get("gpu_total", 0))]


class PodActuator:
    """Two-phase scale moves driven from the controller loop (non-blocking state machine).

    pending = {"kind": "out"|"in", "desired": N, "t_decision", "t_patch", "replicas_after", ...}
    """

    def __init__(self, group: RayClusterGroup, node_wait_timeout_s: float = 1200.0, release: bool = True) -> None:
        self.group = group
        self.node_wait_timeout_s = node_wait_timeout_s
        self.release = release
        self.pending: dict[str, Any] | None = None
        self.history: list[dict[str, Any]] = []

    # -------------------------------------------------------------- scale-out
    def begin_scale_out(self, desired: int, status: dict, now: float) -> dict[str, Any]:
        rec: dict[str, Any] = {"kind": "out", "desired": desired, "t_decision": now}
        if free_rollout_nodes(status):
            rec.update(pod_needed=False, ready=True, t_node_ready=now)
        else:
            before = self.group.replicas()
            self.group.set_replicas(before + 1)
            rec.update(pod_needed=True, ready=False, replicas_before=before, replicas_after=before + 1, t_patch=now)
        self.pending = rec
        return rec

    def poll_scale_out(self, status: dict, now: float) -> dict[str, Any] | None:
        """Returns the record once a free GPU is visible (caller then POSTs /scale), else None."""
        rec = self.pending
        if rec is None or rec["kind"] != "out":
            return None
        if not rec.get("ready"):
            pods = self.group.pods()
            running = [p["metadata"]["name"] for p in pods if (p.get("status") or {}).get("phase") == "Running"]
            rec["pods_running"] = len(running)
            if "t_pod_running" not in rec and len(running) >= rec.get("replicas_after", 0):
                rec["t_pod_running"] = now
            if free_rollout_nodes(status):
                rec.update(ready=True, t_node_ready=now, node_wait_s=now - rec.get("t_patch", rec["t_decision"]))
            elif now - rec.get("t_patch", rec["t_decision"]) > self.node_wait_timeout_s:
                rec.update(error="node wait timeout", t_end=now)
                self.history.append(rec)
                self.pending = None
                return rec
        if rec.get("ready"):
            return rec
        return None

    def finish(self, extra: dict[str, Any] | None = None) -> None:
        if self.pending is not None:
            self.pending.update(extra or {}, t_end=time.time())
            self.history.append(self.pending)
        self.pending = None

    # -------------------------------------------------------------- scale-in
    def begin_scale_in(self, desired: int, now: float) -> dict[str, Any]:
        rec: dict[str, Any] = {"kind": "in", "desired": desired, "t_decision": now, "release": self.release, "ready": False}
        self.pending = rec
        return rec

    def poll_scale_in(self, status: dict, now: float) -> dict[str, Any] | None:
        """After the plugin destroyed the replica, remove exactly the pod whose GPU is now free."""
        rec = self.pending
        if rec is None or rec["kind"] != "in":
            return None
        active = len((status.get("pool") or {}).get("active") or [])
        if active > rec["desired"]:
            if now - rec["t_decision"] > self.node_wait_timeout_s:
                rec.update(error="scale-in did not converge", t_end=now)
                self.history.append(rec)
                self.pending = None
                return rec
            return None
        rec.setdefault("t_replica_gone", now)
        free = fully_free_rollout_nodes(status)
        if not free:
            if now - rec["t_replica_gone"] > 120:
                rec.update(error="no fully free rollout node after destroy", t_end=now)
                self.history.append(rec)
                self.pending = None
                return rec
            return None
        node = free[0]
        pod = self.group.pod_by_ip(node.get("ip") or "")
        if pod is None:
            rec.update(error=f"no pod with ip {node.get('ip')}", t_end=now)
            self.history.append(rec)
            self.pending = None
            return rec
        name = pod["metadata"]["name"]
        before = self.group.replicas()
        self.group.set_replicas(max(0, before - 1), workers_to_delete=[name])
        rec.update(ready=True, pod_deleted=name, node_ip=node.get("ip"), replicas_before=before, replicas_after=before - 1, t_patch=now, t_end=now)
        self.history.append(rec)
        self.pending = None
        return rec
