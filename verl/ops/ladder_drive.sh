#!/usr/bin/env bash
# ladder_drive.sh <job> <stage pod holding the pool jsonl> <pool jsonl path> : quasi-static ladder through the PRODUCT PATH.
#   S=1 until N1 versions are logged at n_active=1 (default 7), then scale_out_manual.sh 2 (RayCluster +1 host -> Ray join -> POST
#   /scale -> init_standalone), then once >= N2 settled versions at n_active=2 exist (settled = after the first 2 at that size,
#   default 6) the GATE is evaluated: PASS if mean(step_s S=2 settled) <= (1 - GATE_FRAC) x mean(step_s S=1 settled v3..) AND
#   mean(slot_util S=1 settled) >= GATE_UTIL; PASS -> scale_out_manual.sh 3 and stop after N3 settled versions at 3 (default 6; the amended gate accepts a flat S=3);
#   FAIL -> ray job stop + 'GATE FAILED' line. Every poll line is a measurement (UTC, version, active, util, idle).
CTX=gke_aishuk-test_europe-west4-b_tpu-stress-test-eu; K="/opt/homebrew/bin/kubectl --context $CTX --request-timeout=40s"
JOB=$1; SPOD=$2; POOL=$3; N1=${N1:-8}; N2=${N2:-6}; N3=${N3:-6}; GATE_FRAC=${GATE_FRAC:-0.3333}; GATE_UTIL=${GATE_UTIL:-0.85}
D="$HOME/workspaces/rl-pool-autoscaler-ops/tpu-m3"; H="${CONTROL_HOST:-127.0.0.1}"
HEAD=$($K get pods -l ray.io/node-type=head,ray.io/cluster=ray-tpum3 -o jsonpath='{.items[0].metadata.name}')
P2=0; P3=0; GATED=""
while true; do
  M=$($K exec "$HEAD" -- python3 /home/ray/rlpool/ctl.py "$H" metrics 2>/dev/null)
  V=$(echo "$M" | awk '/^rl_pool_current_param_version /{print int($2)}')
  A=$(echo "$M" | awk '/^rl_pool_active_standalone_replicas /{print $2}')
  U=$(echo "$M" | awk '/^rl_pool_rollout_resource_utilization /{print $2}')
  I=$(echo "$M" | awk '/^rl_pool_trainer_idle_ratio /{print $2}')
  $K cp "$SPOD:$POOL" "$D/results/pool-ladder.partial.jsonl" >/dev/null 2>&1
  S=$(python3 - "$N1" "$N2" "$N3" "$GATE_FRAC" "$GATE_UTIL" <<'PY'
import json, sys, statistics as st
n1, n2, n3, frac, umin = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
steps = {}
try:
    for line in open("/Users/aishuk/workspaces/rl-pool-autoscaler-ops/tpu-m3/results/pool-ladder.partial.jsonl"):
        if line.startswith("{"):
            try: r = json.loads(line)
            except json.JSONDecodeError: continue
            if r.get("kind") == "step": steps[r["step"]] = r
except FileNotFoundError:
    pass
by = {}
for v in sorted(steps):
    by.setdefault(steps[v]["n_active"], []).append(steps[v])
def settled(n, skip):
    return by.get(n, [])[skip:]
s1, s2, s3 = settled(1, 2), settled(2, 2), settled(3, 2)
out = {"v1": len(by.get(1, [])), "s2": len(s2), "s3": len(s3)}
if s1 and s2:
    m1 = st.mean(r["metrics"]["timing_s/step"] for r in s1); m2 = st.mean(r["metrics"]["timing_s/step"] for r in s2)
    u1 = st.mean((r.get("util") or {}).get("decode_slot_util") or 0.0 for r in s1)
    out.update(m1=round(m1, 1), m2=round(m2, 1), ratio=round(m2 / m1, 3), u1=round(u1, 3), gate="PASS" if (m2 <= (1 - frac) * m1 and u1 >= umin) else "FAIL",
               s1v=[r["step"] for r in s1], s2v=[r["step"] for r in s2])
print(json.dumps(out))
PY
)
  echo "$(date -u +%FT%TZ) version=${V:-none} active=${A:-none} util=${U:-none} idle=${I:-none} posted=$P2$P3 gate=${GATED:-pending} $S"
  V1=$(echo "$S" | python3 -c "import sys,json; print(json.load(sys.stdin)['v1'])"); S2=$(echo "$S" | python3 -c "import sys,json; print(json.load(sys.stdin)['s2'])"); S3=$(echo "$S" | python3 -c "import sys,json; print(json.load(sys.stdin)['s3'])")
  if [ "$P2" = 0 ] && [ "${V1:-0}" -ge "$N1" ]; then echo "$(date -u +%FT%TZ) SCALE-OUT 1->2 (product path)"; "$D/scale_out_manual.sh" 2; P2=1; fi
  if [ "$P2" = 1 ] && [ -z "$GATED" ] && [ "${S2:-0}" -ge "$N2" ]; then
    GATED=$(echo "$S" | python3 -c "import sys,json; print(json.load(sys.stdin).get('gate','FAIL'))")
    echo "$(date -u +%FT%TZ) GATE RESULT: $GATED $S"
    if [ "$GATED" = PASS ]; then echo "$(date -u +%FT%TZ) SCALE-OUT 2->3 (product path)"; "$D/scale_out_manual.sh" 3; P3=1
    else echo "$(date -u +%FT%TZ) GATE FAILED: stopping $JOB"; $K exec "$HEAD" -- ray job stop "$JOB" 2>&1 | tail -1; break; fi
  fi
  if [ "$P3" = 1 ] && [ "${S3:-0}" -ge "$N3" ]; then echo "$(date -u +%FT%TZ) ladder complete: stopping $JOB"; $K exec "$HEAD" -- ray job stop "$JOB" 2>&1 | tail -1; break; fi
  sleep 30
done
