#!/bin/bash
# kill_after_step.sh <prefix> <step>: when the controller log shows last_completed_step >= <step>, delete the pod of the newest
# controller-added ACTIVE rollout worker (highest template index among ACTIVE workers other than roll-0); records to actions.log
PFX=$1; STEP=$2; K=~/workspaces/rl-pool-autoscaler-ops/tunix-slotgate/k8s/kctl.sh; A=~/workspaces/rl-pool-autoscaler-ops/tunix-recipe2/logs/actions.log
while true; do
  line=$($K logs deploy/tunix-controller-$PFX -n default --tail=1 2>/dev/null | tail -1)
  W=$(printf '%s' "$line" | python3 -I -c "
import sys,json
try: r=json.loads(sys.stdin.read()); s=r.get('status',{})
except Exception: sys.exit(0)
last=s.get('last_completed_step'); ws=s.get('workers',{})
act=[w for w,v in ws.items() if v.startswith('active') and not w.endswith('-roll-0')]
if last is not None and last>=$STEP and act: print(sorted(act)[-1])
")
  if [ -n "$W" ]; then
    echo "$(date -u +%FT%TZ) KILL TRIGGER: $PFX last_completed_step>=$STEP, target $W" | tee -a $A
    ~/workspaces/rl-pool-autoscaler-ops/tunix-slotgate/k8s/fail_act.sh $W 2>&1 | tee -a $A
    exit 0
  fi
  sleep 5
done
