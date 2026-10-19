#!/bin/bash
# collect_logs.sh <run-name> <job-prefix>   -> ~/workspaces/rl-pool-autoscaler-ops/tunix/logs/<run-name>/
set -uo pipefail
RUN=$1; PFX=$2; K=~/workspaces/rl-pool-autoscaler-ops/tunix-slotgate/k8s/kctl.sh
OUT=~/workspaces/rl-pool-autoscaler-ops/tunix-recipe2/logs/$RUN; mkdir -p "$OUT"
$K get pods -n default -o wide | grep -E "^${PFX}-" > "$OUT/pods.txt"
$K get jobsets -n default | grep -E "^${PFX}-" > "$OUT/jobsets.txt"
for pod in $($K get pods -n default -o name | grep -E "pod/${PFX}-" | cut -d/ -f2); do
  $K logs "$pod" -n default --timestamps > "$OUT/$pod.log" 2>&1
  $K logs "$pod" -n default --timestamps --previous > "$OUT/$pod.previous.log" 2>/dev/null || rm -f "$OUT/$pod.previous.log"
done
$K get events -n default --sort-by=.lastTimestamp | grep -E "${PFX}-" > "$OUT/events.txt" 2>/dev/null
ls -la "$OUT" | tail -n +2
