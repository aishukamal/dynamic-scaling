#!/bin/bash
# sandbox_live_logger.sh <run>: every 60 s append UTC time + the run's sandbox pods by state (unschedulable = phase Pending with PodScheduled=False; creating = Pending but scheduled)
RUN=$1; D=~/workspaces/rl-pool-autoscaler-ops/tunix-recipe2/logs/$RUN; K=~/workspaces/rl-pool-autoscaler-ops/tunix-slotgate/k8s/kctl.sh
while [ ! -f $D/live/.stop ]; do
  $K get pods -n agents-system -o json 2>/dev/null | python3 -I -c "
import json,sys,collections,datetime
d=json.load(sys.stdin); c=collections.Counter()
for p in d['items']:
    n=p['metadata']['name']
    if '-$RUN-' not in n and not n.startswith('sandbox-claim'): continue
    ph=p['status']['phase']
    if ph=='Pending':
        conds={x['type']:x for x in p['status'].get('conditions',[])}; c['unschedulable' if conds.get('PodScheduled',{}).get('status')=='False' else 'creating']+=1
    else: c[ph]+=1
print(datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), dict(c))" >> $D/sandbox_live.txt
  sleep 60
done
