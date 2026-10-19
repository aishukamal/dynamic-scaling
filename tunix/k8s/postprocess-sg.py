#!/usr/bin/env python3
"""Post-processes manifests rendered by tunix deepswe_dist/k8s_launcher.sh (DRY_RUN=true) for the eu GKE cluster.

Text-level edits only (stdlib):
  * drop priorityClassName; serviceAccount xpk-sa -> default; drop the orchestrator's exclusive-topology annotation
  * TPU pods: topology selector 2x2 -> 4x8 (hosts carry the slice label) + kubernetes.io/hostname pin (one 4-chip host per JobSet)
  * orchestrator + rollout: mount ConfigMap tunix-deepswe-patch at /patch and copy the two patched files over the in-image
    ones before launch (observability + SANDBOX_RUNTIME_CLASS); orchestrator additionally mounts hostPath /tmp (dataset at
    /tmp/tunix-data/r2e-cal on the CPU node) and gets WANDB_MODE=disabled.
Usage: postprocess.py <role: orchestrator|trainer|rollout> <node-or-empty> < in.yaml > out.yaml
"""
import os, re, sys

role = sys.argv[1]
node = sys.argv[2] if len(sys.argv) > 2 else ""
s = sys.stdin.read()
s = "\n".join(l for l in s.split("\n") if not l.startswith("kubectl ")).lstrip("\n")
if s.startswith("---\n"):
    s = s[4:]
s = re.sub(r"^\s*priorityClassName: .*\n", "", s, flags=re.M)
s = s.replace("serviceAccountName: xpk-sa", "serviceAccountName: default")
EXTRA = (" && " + os.environ["SG_EXTRA_CP"]) if os.environ.get("SG_EXTRA_CP") else ""  # TUNIX-REWARD2 extra patch files
CP = ("cp /patch/sandbox_utils.py /app/examples/deepswe/sandbox_utils.py && "
      "cp /patch/run_deepswe_dist.py /app/tunix/experimental/examples/deepswe_dist/run_deepswe_dist.py && "
      "cp /patch/tunix_exporter.py /app/tunix/experimental/examples/deepswe_dist/tunix_exporter.py" + EXTRA + " || exit 75")
if role in ("trainer", "rollout"):
    assert "cloud.google.com/gke-tpu-topology: 2x2" in s, "expected 2x2 topology selector"
    topo = os.environ.get("SG_TPU_TOPOLOGY_LABEL_" + role.upper()) or os.environ.get("SG_TPU_TOPOLOGY_LABEL", "4x8")  # per-role override (mixed pools)  # 4x8 for the v6e-4x8 hosts; 2x2 for single-host 2x2 nodes (v6e-4x1-long)
    s = s.replace("cloud.google.com/gke-tpu-topology: 2x2",
                  f"cloud.google.com/gke-tpu-topology: {topo}" + (f"\n              kubernetes.io/hostname: {node}" if node else ""))
TOL = """            tolerations:
            - {key: google.com/tpu, operator: Exists, effect: NoSchedule}
            - {key: cloud.google.com/gke-spot, operator: Equal, value: "true", effect: NoSchedule}
"""
if role in ("trainer", "rollout") and os.environ.get("SG_SPOT") == "1":
    # spot pool v6e-4x1-long carries the gke-spot taint; the launcher template leaves tolerations commented out
    assert s.count("            # tolerations:\n") == 1, "expected the commented tolerations block"
    s = s.replace("            # tolerations:\n", TOL + "            # tolerations:\n", 1)
if role == "rollout" and os.environ.get("SG_ENABLE_THINKING") == "1":
    # generation-side deviation approved by the user (reward probe / long arms): run_rollout_node --enable_thinking
    n = s.count("--in_flight_weight_updates=")
    assert n == 1, f"expected one in_flight_weight_updates flag, got {n}"
    s = re.sub(r"(--in_flight_weight_updates=\S+)", r"\1 --enable_thinking", s, count=1)
if role == "rollout":
    assert s.count("                  (\n") == 1, "expected one subshell open line"
    s = s.replace("                  (\n", "                  (\n                    " + CP + "\n", 1)
    assert s.count("              volumeMounts:\n") == 1 and s.count("            volumes:\n") == 1
    s = s.replace("              volumeMounts:\n", "              volumeMounts:\n              - name: patch\n                mountPath: /patch\n", 1)
    s = s.replace("            volumes:\n", "            volumes:\n            - name: patch\n              configMap:\n                name: tunix-deepswe-patch-ctl\n", 1)
if role == "orchestrator" and os.environ.get("SG_ORCH_CPU"):
    # the launcher hardcodes a 24-core request for the orchestrator; several orchestrators share the one CPU node
    assert s.count('cpu: "24"') >= 1
    s = s.replace('cpu: "24"', f'cpu: "{os.environ["SG_ORCH_CPU"]}"')
if role == "orchestrator":
    s = re.sub(r"^\s*alpha\.jobset\.sigs\.k8s\.io/exclusive-topology: .*\n", "", s, flags=re.M)
    s = s.replace("python -m tunix.experimental.distributed.runtime.main",
                  "WANDB_MODE=disabled python -m tunix.experimental.distributed.runtime.main", 1)
    assert s.count("                  (\n") == 1, "expected one subshell open line"
    s = s.replace("                  (\n", "                  (\n                    " + CP + "\n", 1)
    tail = """              volumeMounts:
              - name: patch
                mountPath: /patch
              - name: tmp
                mountPath: /tmp
            volumes:
            - name: patch
              configMap:
                name: tunix-deepswe-patch-ctl
            - name: tmp
              hostPath:
                path: /tmp
                type: DirectoryOrCreate
"""
    s = s.rstrip("\n") + "\n" + tail
sys.stdout.write(s)
