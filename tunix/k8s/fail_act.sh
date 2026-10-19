#!/bin/bash
# fail_act.sh <worker-id e.g. tc-roll-1> : deletes the worker's POD (not the JobSet) and records timestamps.
set -uo pipefail
W=$1; K=~/workspaces/rl-pool-autoscaler-ops/tunix-slotgate/k8s/kctl.sh; A=~/workspaces/rl-pool-autoscaler-ops/tunix-recipe2/logs/actions.log
POD=$($K get pods -n default -o name | grep "pod/${W}-" | head -1 | cut -d/ -f2)
NODE=$($K get pod -n default $POD -o jsonpath='{.spec.nodeName}')
ORCH=$($K get pods -n default -o name | grep "pod/${W%%-roll-*}-orch" | head -1 | cut -d/ -f2)
STEP=$($K logs -n default $ORCH 2>/dev/null | grep -o ">>> Step [0-9]*" | tail -1 | awk '{print $3}')
T0=$(python3 -c 'import time;print(f"{time.time():.3f}")')
echo "$T0 $(date -u +%FT%TZ) FAILURE_ACT delete pod $POD (worker $W on $NODE) at step_in_progress=$STEP" | tee -a $A
$K delete pod -n default $POD --wait=false 2>&1 | tee -a $A
for i in $(seq 1 40); do sleep 3; NEW=$($K get pods -n default -o name | grep "pod/${W}-" | grep -v "$POD" | head -1 | cut -d/ -f2); [ -n "$NEW" ] && break; done
echo "$(python3 -c 'import time;print(f"{time.time():.3f}")') $(date -u +%FT%TZ) replacement pod $NEW phase=$($K get pod -n default $NEW -o jsonpath='{.status.phase}' 2>/dev/null) (+$(python3 -c "import time;print(f'{time.time()-$T0:.1f}')") s after delete)" | tee -a $A
