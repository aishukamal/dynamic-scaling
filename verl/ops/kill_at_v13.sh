#!/usr/bin/env bash
# kill_at_v13.sh : waits for dyn3's version-12 step record, then (inside version 13's generation) SIGKILLs chip 1's vLLM worker of the
# NEWEST active replica (highest rank in the last step record). Writes results/kill-tpum3-dyn3.log.
D="$HOME/workspaces/rl-pool-autoscaler-ops/tpu-m3"; cd "$D"; P=results/pool-tpum3-dyn3.partial.jsonl
while ! grep -q '"kind": "step", "step": 12,' "$P" 2>/dev/null && ! python3 -c "
import json,sys
ok=any(json.loads(l).get('kind')=='step' and json.loads(l).get('step')==12 for l in open('$P') if l.startswith('{'))
sys.exit(0 if ok else 1)"; do sleep 20; done
echo "$(date -u +%FT%T.%3NZ) version 12 logged; waiting 45 s into version 13's generation" | tee -a results/kill-tpum3-dyn3.log
sleep 45
R=$(python3 -c "
import json
last=[json.loads(l) for l in open('$P') if l.startswith('{') and '\"kind\": \"step\"' in l][-1]
print(max(last['active']))")
echo "$(date -u +%FT%T.%3NZ) newest active replica rank=$R; killing chip 1 holder" | tee -a results/kill-tpum3-dyn3.log
CLUSTER=ray-tpum3 JOB=tpum3-dyn3 ./kill_dyn3.sh "$R" 1 2>&1 | tee -a results/kill-tpum3-dyn3.log
