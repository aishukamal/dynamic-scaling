#!/bin/bash
# apply_run.sh <rendered-dir> [orchestrator|trainer|rollout-N ...]  (default: orchestrator trainer rollout-*)
set -euo pipefail
D=$1; shift || true
K=~/workspaces/rl-pool-autoscaler-ops/tunix-recipe2/k8s/kctl.sh
parts=("$@"); [ ${#parts[@]} -eq 0 ] && parts=(orchestrator trainer $(cd "$D" && ls rollout-*.yaml | sed 's/.yaml//'))
for p in "${parts[@]}"; do
  echo "$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ 2>/dev/null || date -u +%Y-%m-%dT%H:%M:%SZ) apply $p" | tee -a "$D/apply.log"
  $K apply -f "$D/$p.yaml" | tee -a "$D/apply.log"
done
