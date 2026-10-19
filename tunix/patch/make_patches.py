#!/usr/bin/env python3
"""Produces the two patched files shipped by ConfigMap tunix-deepswe-patch (console observability + gVisor runtime class).
No algorithm change. Each anchor must match exactly once."""
import pathlib, subprocess, sys
SRC = pathlib.Path.home() / "workspaces/tunix-poc"
OUT = pathlib.Path(__file__).resolve().parent

def patch(rel, edits):
    s = (SRC / rel).read_text()
    for old, new in edits:
        assert s.count(old) == 1, (rel, old[:60], s.count(old))
        s = s.replace(old, new)
    dst = OUT / pathlib.Path(rel).name
    dst.write_text(s)
    diff = subprocess.run(["diff", "-u", str(SRC / rel), str(dst)], capture_output=True, text=True).stdout
    (OUT / (pathlib.Path(rel).name + ".observability.patch")).write_text(diff)
    print(rel, "->", dst, f"(+{diff.count(chr(10)+'+')-1} lines)")

# 1) sandbox_utils.py: let SANDBOX_RUNTIME_CLASS (e.g. gvisor) flow into ClusterConfig.runtime_class (the SDK supports it,
#    the recipe just never sets it; with the r2egym scaffold there is no TemplateSpec to carry it). GKE's gvisor RuntimeClass
#    adds the sandbox.gke.io/runtime nodeSelector + toleration by itself.
patch("examples/deepswe/sandbox_utils.py", [
    ("""            ClusterConfig(
                name="default",
                namespace=fleet_ns,
                node_selector=node_sel,
                in_cluster=in_cluster,
            )""",
     """            ClusterConfig(
                name="default",
                namespace=fleet_ns,
                node_selector=node_sel,
                in_cluster=in_cluster,
                runtime_class=os.getenv("SANDBOX_RUNTIME_CLASS") or None,
            )"""),
])

# 2) run_deepswe_dist.py: console METRIC / MEMBERSHIP / FTSNAP lines (same as the TUNIX-BASELINE gsm8k patch) + per-step
#    sandbox pod/warm-pool counts from the Kubernetes API (read-only list calls).
patch("tunix/experimental/examples/deepswe_dist/run_deepswe_dist.py", [
    ("""  context.ipc.discovery.on_register(
      functools.partial(
          cluster.register_worker_from_hostname,
          rpc_timeout_s=args.rpc_timeout_s,
      )
  )
""",
     """  context.ipc.discovery.on_register(
      functools.partial(
          cluster.register_worker_from_hostname,
          rpc_timeout_s=args.rpc_timeout_s,
      )
  )

  # --- TUNIX-DEEPSWE observability patch (console-only; no algorithm change) ---
  import collections as _collections  # pylint: disable=g-import-not-at-top
  import time as _time  # pylint: disable=g-import-not-at-top

  def _console_metric(name, value, **kw):
    try:
      logging.info("METRIC step=%s %s=%s", kw.get("step"), name, float(value))
    except Exception:  # pylint: disable=broad-exception-caught
      logging.info("METRIC step=%s %s=%r", kw.get("step"), name, value)

  jax.monitoring.register_scalar_listener(_console_metric)

  def _membership_listener(worker_id, state):
    logging.info(
        "MEMBERSHIP t=%.3f worker_id=%s state=%s",
        _time.time(),
        worker_id,
        getattr(state, "value", state),
    )

  cluster.registry.add_state_listener(_membership_listener)

  _k8s = {}

  def _sandbox_snapshot():
    try:
      if "core" not in _k8s:
        from kubernetes import client as _kc  # pylint: disable=g-import-not-at-top
        from kubernetes import config as _kcfg  # pylint: disable=g-import-not-at-top

        _kcfg.load_incluster_config()
        _k8s["core"] = _kc.CoreV1Api()
        _k8s["custom"] = _kc.CustomObjectsApi()
        _k8s["ns"] = (
            os.getenv("SANDBOX_NAMESPACE") or os.getenv("NAMESPACE") or "default"
        )
      ns = _k8s["ns"]
      pods = _k8s["core"].list_namespaced_pod(ns, label_selector="app=agent-sandbox-rl").items
      phases = _collections.Counter(p.status.phase for p in pods)
      ready = sum(
          1
          for p in pods
          if p.status.phase == "Running"
          and all(cs.ready for cs in (p.status.container_statuses or []))
      )
      pools = _k8s["custom"].list_namespaced_custom_object(
          "extensions.agents.x-k8s.io", "v1beta1", ns, "sandboxwarmpools"
      ).get("items", [])
      pool_replicas = sum(int((p.get("spec") or {}).get("replicas") or 0) for p in pools)
      pool_available = sum(
          int((p.get("status") or {}).get("availableReplicas") or 0) for p in pools
      )
      claims = _k8s["custom"].list_namespaced_custom_object(
          "extensions.agents.x-k8s.io", "v1beta1", ns, "sandboxclaims"
      ).get("items", [])
      return (
          f"pods_total={len(pods)} pods={dict(phases)} pods_ready={ready}"
          f" warmpools={len(pools)} pool_replicas={pool_replicas}"
          f" pool_available={pool_available} claims={len(claims)}"
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      return f"sandbox_snapshot_error={e!r}"[:300]

  def _ft_snapshot(step):
    try:
      eng = cluster.engine
      states = {
          wid: cluster.registry.state(wid).value
          for wid in cluster.registry.worker_ids(include_evicted=True)
      }
      n_active = len(getattr(eng, "_rollout_workers", []) or []) if eng else -1
      ftm = eng.fault_tolerance_metrics if eng else {}
      logging.info(
          "FTSNAP step=%s t=%.3f active_rollout_workers=%d states=%s ft=%s",
          step,
          _time.time(),
          n_active,
          states,
          ftm,
      )
      logging.info("SANDBOXSNAP step=%s t=%.3f %s", step, _time.time(), _sandbox_snapshot())
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.info("FTSNAP step=%s error=%r", step, e)
  # --- end patch ---
"""),
    ("""        on_step_begin=lambda step: logging.info(
            ">>> DeepSWE step %d starting | policy_version=%d",
            step,
            step,
        ),
        on_step_end=lambda step, result: (
            logging.info(
                "<<< DeepSWE step %d finished | train_result=%s",
                step,
                result,
            ),
""",
     """        on_step_begin=lambda step: (
            logging.info(
                ">>> DeepSWE step %d starting | policy_version=%d | t=%.3f",
                step,
                step,
                _time.time(),
            ),
            _ft_snapshot(step),
        ),
        on_step_end=lambda step, result: (
            logging.info(
                "<<< DeepSWE step %d finished | train_result=%s | t=%.3f",
                step,
                result,
                _time.time(),
            ),
            _ft_snapshot(step),
"""),
])
