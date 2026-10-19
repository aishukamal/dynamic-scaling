# Copyright 2026 llm-d
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Minimal client for GKE agent-sandbox Sandboxes used by SWE RL rollouts.

Creates Sandboxes in a namespace excluded from the managed
``secure-sandbox-policy`` (default ``agents-system``) so task images can run
as root, while keeping gVisor, no SA token, and dropped capabilities - see
configs/swe_sandbox_example.yaml for the rationale.

Implementation notes (learned the hard way, see docs/swe_bench_guide.md):
- Write files via exec + base64, never `kubectl cp`: the sandbox drops
  CAP_CHOWN so tar exits nonzero even though content lands.
- The Sandbox controller names the backing pod identically to the Sandbox.
"""

from __future__ import annotations

import base64
import hashlib
import re
import shlex
import threading
import time
from uuid import uuid4

from kubernetes import client as k8s_client  # type: ignore[import-not-found]
from kubernetes import config as k8s_config  # type: ignore[import-not-found]
from kubernetes.client.rest import ApiException  # type: ignore[import-not-found]
from kubernetes.stream import stream as k8s_stream  # type: ignore[import-not-found]

GROUP = "agents.x-k8s.io"
VERSION = "v1beta1"  # agent-sandbox v1.0.x serves v1beta1 only (local PoC patch)
PLURAL = "sandboxes"
CONTAINER = "sandbox"

EXEC_TIMEOUT_DEFAULT = 330  # generous vs the 5-minute grading cap

# TPU-M3 loop v3: under 1000-2000 concurrent trajectories the per-turn exec websockets plus creates/deletes overwhelmed the
# API server path and ~45-65 % of trajectories died with ApiException(0) (no HTTP response) and were salvaged as short
# zero-reward samples. Every API call now goes through a per-process semaphore (SWE_API_CONCURRENCY, default 32) and
# transient failures (status 0 / 429 / 5xx, websocket errors) are retried with exponential backoff.
import os as _os

# TPU-M3 loop v3.1: the single cap starved the control plane. Exec websockets hold a slot for the whole tool command (up to
# 90 s), so at a version boundary (hundreds of fresh sandboxes) the readiness polls of wait_ready queued behind long execs
# and timed out ("not Running after 600s") on pods that Kubernetes had reported Ready within seconds (tpum3-dynr2, 13 % of
# trajectories, then a full stall). Execs and control-plane calls (create / pod read / delete) now have separate caps.
_API_SEM = threading.BoundedSemaphore(int(_os.getenv("SWE_API_CONCURRENCY", "32")))  # exec websockets
_CTL_SEM = threading.BoundedSemaphore(int(_os.getenv("SWE_CTL_CONCURRENCY", "16")))  # create / read / delete
_TRANSIENT_STATUS = {0, 429, 500, 502, 503, 504}
_API_RETRIES = int(_os.getenv("SWE_API_RETRIES", "5"))


def _transient(e: BaseException) -> bool:
    if isinstance(e, ApiException):
        return int(getattr(e, "status", 0) or 0) in _TRANSIENT_STATUS
    return not isinstance(e, SandboxError)  # websocket layer raises bare Exceptions / WebSocketException


def _api_call(what: str, fn, *args, retries: int | None = None, **kwargs):  # noqa: ANN001,ANN201
    """Call a kubernetes API function under the concurrency cap, retrying transient failures with backoff."""
    n = _API_RETRIES if retries is None else retries
    last: BaseException | None = None
    for attempt in range(n + 1):
        try:
            with _CTL_SEM:
                return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            last = e
            if not _transient(e) or attempt >= n:
                raise
            time.sleep(min(60.0, 2.0 * (2**attempt)))
    raise RuntimeError(f"{what}: unreachable") from last


class SandboxError(RuntimeError):
    pass


_thread_local = threading.local()


def get_thread_client(namespace: str = "agents-system") -> SandboxClient:
    """One SandboxClient per (thread, namespace).

    The kubernetes client's GKE exec-plugin auth races when an ApiClient is
    shared across threads (observed as intermittent NoneType decode errors),
    so every worker thread gets its own client.
    """
    cache: dict[str, SandboxClient] | None = getattr(_thread_local, "clients", None)
    if cache is None:
        cache = _thread_local.clients = {}
    if namespace not in cache:
        cache[namespace] = SandboxClient(namespace=namespace)
    return cache[namespace]


def make_name(prefix: str, instance_id: str) -> str:
    """Unique RFC-1123 sandbox name for one use (trajectory, grading, ...)."""
    slug = re.sub(r"[^a-z0-9-]", "-", instance_id.lower()).strip("-")[:30]
    digest = hashlib.sha1(instance_id.encode()).hexdigest()[:4]  # noqa: S324
    return f"{prefix}-{slug}-{digest}-{uuid4().hex[:6]}"


class SandboxClient:
    def __init__(self, namespace: str = "agents-system") -> None:
        try:
            k8s_config.load_incluster_config()
        except Exception:  # noqa: BLE001
            k8s_config.load_kube_config()
        self.namespace = namespace
        self.custom = k8s_client.CustomObjectsApi()
        self.core = k8s_client.CoreV1Api()
        self.last_pod_ready_s: float | None = None  # loop v3.2 diagnostics (set by wait_ready)
        self.last_wait_polls = 0
        self.last_wait_sem_s = 0.0

    def create(  # noqa: PLR0913
        self,
        name: str,
        image: str,
        *,
        cpu_request: str = "500m",
        memory_request: str = "1Gi",
        cpu_limit: str = "2",
        memory_limit: str = "4Gi",
    ) -> None:
        manifest = {
            "apiVersion": f"{GROUP}/{VERSION}",
            "kind": "Sandbox",
            "metadata": {"name": name, "namespace": self.namespace},
            "spec": {
                "podTemplate": {
                    "spec": {
                        "runtimeClassName": "gvisor",
                        "automountServiceAccountToken": False,
                        "nodeSelector": {"sandbox.gke.io/runtime": "gvisor"},
                        "tolerations": [
                            {
                                "key": "sandbox.gke.io/runtime",
                                "operator": "Equal",
                                "value": "gvisor",
                                "effect": "NoSchedule",
                            }
                        ],
                        "containers": [
                            {
                                "name": CONTAINER,
                                "image": image,
                                "command": ["sleep", "infinity"],
                                "securityContext": {"capabilities": {"drop": ["ALL"]}},
                                "resources": {
                                    "requests": {"cpu": cpu_request, "memory": memory_request},
                                    "limits": {"cpu": cpu_limit, "memory": memory_limit},
                                },
                            }
                        ],
                    }
                }
            },
        }
        _api_call("create", self.custom.create_namespaced_custom_object, GROUP, VERSION, self.namespace, PLURAL,
                  manifest)

    def wait_ready(self, name: str, timeout: float = 300) -> float:
        """Block until the backing pod is Running; returns elapsed seconds."""
        start = time.monotonic()
        polls = 0
        sem_wait = 0.0
        last = "no poll completed"
        while time.monotonic() - start < timeout:
            try:
                t0 = time.monotonic()
                with _CTL_SEM:
                    sem_wait += time.monotonic() - t0
                    pod = self.core.read_namespaced_pod(name, self.namespace)
                polls += 1
                last = f"phase={pod.status.phase}"
                if pod.status.phase == "Running":
                    # loop v3.2 diagnostics: server-side create-to-ready (Ready condition transition minus CR/pod creation)
                    # next to the client-observed wait, so a slow wait can be attributed to the pod or to the client path.
                    try:
                        ready_at = next(c.last_transition_time for c in (pod.status.conditions or [])
                                        if c.type == "Ready" and c.status == "True")
                        self.last_pod_ready_s = (ready_at - pod.metadata.creation_timestamp).total_seconds()
                    except Exception:  # noqa: BLE001
                        self.last_pod_ready_s = None
                    self.last_wait_polls = polls
                    self.last_wait_sem_s = sem_wait
                    return time.monotonic() - start
                if pod.status.phase in ("Failed", "Succeeded"):  # noqa: PLR6201
                    raise SandboxError(f"{name}: pod reached terminal phase {pod.status.phase}")
            except ApiException as e:
                polls += 1
                last = f"ApiException status={e.status}"
                # 404 = pod not created yet; transient API failures (status 0 / 429 / 5xx) are waited out, not raised
                if e.status != 404 and not _transient(e):  # noqa: PLR2004
                    raise
            time.sleep(3)
        # loop v3.1: say what the poll loop saw, so a timeout can be attributed (starved poll vs. pod really not Running)
        raise SandboxError(f"{name}: not Running after {timeout}s (polls={polls} sem_wait_s={sem_wait:.0f} last={last})")

    def exec(
        self, name: str, cmd: str, timeout: float = EXEC_TIMEOUT_DEFAULT, retries: int = 2
    ) -> tuple[int, str]:
        """Run a shell command; returns (returncode, combined output). rc 124 on timeout.

        Retries the whole exec on transient websocket errors (the kubernetes
        stream client intermittently dies with e.g. NoneType frame data under
        concurrency) - callers' commands must therefore be idempotent.
        """
        retries = max(retries, _API_RETRIES)
        last: Exception | None = None
        for attempt in range(retries + 1):
            try:
                with _API_SEM:  # the websocket is held for the whole command: this caps concurrent execs per process
                    return self._exec_once(name, cmd, timeout)
            except Exception as e:  # noqa: BLE001,PERF203 - websocket layer raises bare Exceptions
                last = e
                time.sleep(min(60.0, 2.0 * (2**attempt)))
        raise SandboxError(f"{name}: exec failed after {retries + 1} attempts: {type(last).__name__}: {last}")

    def _exec_once(self, name: str, cmd: str, timeout: float) -> tuple[int, str]:
        resp = k8s_stream(
            self.core.connect_get_namespaced_pod_exec,
            name,
            self.namespace,
            container=CONTAINER,
            command=["/bin/sh", "-c", cmd],
            stderr=True,
            stdin=False,
            stdout=True,
            tty=False,
            _preload_content=False,
        )
        chunks: list[str] = []
        deadline = time.monotonic() + timeout
        while resp.is_open() and time.monotonic() < deadline:
            resp.update(timeout=1)
            if resp.peek_stdout():
                chunks.append(resp.read_stdout())
            if resp.peek_stderr():
                chunks.append(resp.read_stderr())
        if resp.is_open():
            resp.close()
            return 124, "".join(chunks)
        try:
            rc = resp.returncode
        except Exception:  # noqa: BLE001
            rc = -1
        return rc if rc is not None else -1, "".join(chunks)

    # Exec commands travel in the API request; chunk large payloads to stay
    # comfortably under apiserver URL limits.
    _WRITE_CHUNK = 24 * 1024

    def write_file(self, name: str, path: str, content: str | bytes) -> None:
        """Write file content via exec + base64 (CAP_CHOWN-safe, quote-safe)."""
        data = content.encode() if isinstance(content, str) else content
        quoted = shlex.quote(path)
        chunks = [data[i : i + self._WRITE_CHUNK] for i in range(0, len(data),
            self._WRITE_CHUNK)] or [b""]
        for i, chunk in enumerate(chunks):
            b64 = base64.b64encode(chunk).decode()
            prefix = f"mkdir -p $(dirname {quoted}) && " if i == 0 else ""
            op = ">" if i == 0 else ">>"
            rc, out = self.exec(name, f"{prefix}echo '{b64}' | base64 -d {op} {quoted}")
            if rc != 0:
                raise SandboxError(f"{name}: write_file({path}) chunk {i} rc={rc}: {out[-500:]}")

    def delete(self, name: str) -> None:
        try:
            _api_call("delete", self.custom.delete_namespaced_custom_object, GROUP, VERSION, self.namespace, PLURAL,
                      name, retries=3)
        except ApiException as e:
            if e.status != 404:  # noqa: PLR2004
                raise
