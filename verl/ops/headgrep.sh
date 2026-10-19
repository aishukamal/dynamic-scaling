#!/usr/bin/env bash
# headgrep.sh <job> <python regex> [context_lines] [max_lines] : greps the Ray job log inside the head pod (script copied in, no inline shell)
CTX=gke_aishuk-test_europe-west4-b_tpu-stress-test-eu; K="/opt/homebrew/bin/kubectl --context $CTX --request-timeout=120s"
HEAD=$($K get pods -l ray.io/node-type=head,ray.io/cluster=${CLUSTER:-ray-tpum3} -o jsonpath='{.items[0].metadata.name}')
cat > /Users/aishuk/workspaces/rl-pool-autoscaler-ops/tpu-m3/.headgrep.py <<'PY'
import re, subprocess, sys
job, pat, ctx, mx = sys.argv[1], re.compile(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
ansi = re.compile(r"\x1b\[[0-9;]*m")
noise = re.compile(r"platform_plugins|register_tpu_platform|VLLM_PLUGINS|Patched|loaded successfully|Platform plugin torchtpu|tpu_info.py|Triton|libnccl|debug_manager|telemetry|UserWarning|cupy")
out = subprocess.run(["ray", "job", "logs", job], capture_output=True, text=True, errors="replace").stdout
lines = [ansi.sub("", l) for l in out.splitlines()]
hits = [i for i, l in enumerate(lines) if pat.search(l) and not noise.search(l)]
shown = set(); n = 0
for i in hits:
    for j in range(max(0, i - ctx), min(len(lines), i + ctx + 1)):
        if j in shown or noise.search(lines[j]): continue
        shown.add(j); print(lines[j][:330]); n += 1
        if n >= mx: break
    if n >= mx: break
print(f"-- total lines {len(lines)}, hits {len(hits)}")
st = subprocess.run(["ray", "job", "status", job], capture_output=True, text=True).stdout.strip().splitlines()
print(st[-1] if st else "status?")
PY
$K cp /Users/aishuk/workspaces/rl-pool-autoscaler-ops/tpu-m3/.headgrep.py "$HEAD:/home/ray/headgrep.py" >/dev/null
$K exec "$HEAD" -- python3 /home/ray/headgrep.py "$1" "$2" "${3:-0}" "${4:-60}"
