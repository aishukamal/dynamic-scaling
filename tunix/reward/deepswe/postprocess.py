#!/usr/bin/env python3
"""TUNIX-REWARD: post-processes manifests rendered by deepswe_dist/k8s_launcher.sh (DRY_RUN=true) for the eu cluster.
Same edits as the TUNIX-DEEPSWE agent's postprocess.py, plus: ConfigMap tunix-reward-patch (5 files) copied over the
in-image files on the orchestrator and rollout pods (swe_agent/openhands_utils/adapter = logging-only patches).
Usage: postprocess.py <role: orchestrator|trainer|rollout> <node-or-empty> < in.yaml > out.yaml"""
import os, re, sys
role = sys.argv[1]; node = sys.argv[2] if len(sys.argv) > 2 else ""
s = sys.stdin.read()
s = "\n".join(l for l in s.split("\n") if not l.startswith("kubectl ")).lstrip("\n")
if s.startswith("---\n"): s = s[4:]
s = re.sub(r"^\s*priorityClassName: .*\n", "", s, flags=re.M)
s = s.replace("serviceAccountName: xpk-sa", "serviceAccountName: default")
CP = ("cp /patch/sandbox_utils.py /app/examples/deepswe/sandbox_utils.py && "
      "cp /patch/run_deepswe_dist.py /app/tunix/experimental/examples/deepswe_dist/run_deepswe_dist.py && "
      "cp /patch/swe_agent.py /app/examples/deepswe/swe_agent.py && "
      "cp /patch/openhands_utils.py /app/examples/deepswe/openhands_utils.py && "
      "cp /patch/inprocess_vllm_sampler_adapter.py /app/tunix/experimental/rollout/inprocess_vllm_sampler_adapter.py || exit 75")
if role in ("trainer", "rollout"):
    assert "cloud.google.com/gke-tpu-topology: 2x2" in s
    s = s.replace("cloud.google.com/gke-tpu-topology: 2x2",
                  "cloud.google.com/gke-tpu-topology: 4x8" + (f"\n              kubernetes.io/hostname: {node}" if node else ""))
if role == "rollout":
    assert s.count("                  (\n") == 1
    s = s.replace("                  (\n", "                  (\n                    " + CP + "\n", 1)
    assert s.count("              volumeMounts:\n") == 1 and s.count("            volumes:\n") == 1
    s = s.replace("              volumeMounts:\n", "              volumeMounts:\n              - name: patch\n                mountPath: /patch\n", 1)
    s = s.replace("            volumes:\n", "            volumes:\n            - name: patch\n              configMap:\n                name: tunix-reward-patch\n", 1)
if role == "orchestrator":
    s = re.sub(r"^\s*alpha\.jobset\.sigs\.k8s\.io/exclusive-topology: .*\n", "", s, flags=re.M)
    if os.environ.get("ORCH_CPU"):
        s = s.replace('cpu: "24"', 'cpu: "%s"' % os.environ["ORCH_CPU"], 1)
    s = s.replace("python -m tunix.experimental.distributed.runtime.main",
                  "WANDB_MODE=disabled python -m tunix.experimental.distributed.runtime.main", 1)
    assert s.count("                  (\n") == 1
    s = s.replace("                  (\n", "                  (\n                    " + CP + "\n", 1)
    s = s.rstrip("\n") + "\n" + """              volumeMounts:
              - name: patch
                mountPath: /patch
              - name: tmp
                mountPath: /tmp
            volumes:
            - name: patch
              configMap:
                name: tunix-reward-patch
            - name: tmp
              hostPath:
                path: /tmp
                type: DirectoryOrCreate
"""
sys.stdout.write(s)
