#!/bin/bash
# node_logger.sh <run>: every 60 s append UTC time, gVisor node count per pool, sandbox pod phases to logs/<run>/nodes_timeline.txt
RUN=$1; D=~/workspaces/rl-pool-autoscaler-ops/tunix-recipe2/logs/$RUN; K=~/workspaces/rl-pool-autoscaler-ops/tunix-slotgate/k8s/kctl.sh
while [ ! -f $D/live/.stop ]; do
  echo "$(date -u +%FT%TZ) nodes=$($K get nodes -l sandbox.gke.io/runtime=gvisor -L cloud.google.com/gke-nodepool --no-headers 2>/dev/null | awk '{print $NF}' | sort | uniq -c | awk '{printf "%s:%s ", $2, $1}') pods=$($K get pods -n agents-system --no-headers 2>/dev/null | awk '{print $3}' | sort | uniq -c | awk '{printf "%s:%s ", $2, $1}')" >> $D/nodes_timeline.txt
  sleep 60
done
