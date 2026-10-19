#!/bin/bash
# follow_logs.sh <run-name> <job-prefix>: streams `kubectl logs -f --timestamps` of every <prefix>-* pod into logs/<run>/live/
# as soon as it is Running (worker pods are stopped by the orchestrator at run end, so post-hoc collection loses them).
set -uo pipefail
RUN=$1; PFX=$2; K=~/workspaces/rl-pool-autoscaler-ops/tunix-reward/k8s/kctl.sh
OUT=~/workspaces/rl-pool-autoscaler-ops/tunix-reward/logs/$RUN/live; mkdir -p "$OUT"
while true; do
  for pod in $($K get pods -n default -o name 2>/dev/null | grep -E "pod/${PFX}-" | cut -d/ -f2); do
    if [ ! -f "$OUT/.following-$pod" ]; then
      ph=$($K get pod "$pod" -n default -o jsonpath='{.status.phase}' 2>/dev/null)
      if [ "$ph" = "Running" ] || [ "$ph" = "Succeeded" ] || [ "$ph" = "Failed" ]; then
        touch "$OUT/.following-$pod"
        # retry loop: `logs -f` on a container that is still creating exits at once with an error; re-attach until the
        # pod is gone (the file is truncated on each attempt, so only the final successful stream is kept)
        ( for i in $(seq 1 30); do $K logs -f "$pod" -n default --timestamps > "$OUT/$pod.log.try" 2>&1; sz=$(stat -f%z "$OUT/$pod.log.try" 2>/dev/null || echo 0); if [ "$sz" -gt 2048 ]; then mv "$OUT/$pod.log.try" "$OUT/$pod.log"; break; fi; sleep 10; done ) &
      fi
    fi
  done
  [ -f "$OUT/.stop" ] && break
  sleep 10
done
wait
