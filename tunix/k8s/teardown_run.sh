#!/bin/bash
# teardown_run.sh <job-prefix>: deletes the run's JobSets + its sandbox resources (label app.kubernetes.io/created-by=<prefix>-orch)
set -uo pipefail
PFX=$1; K=~/workspaces/rl-pool-autoscaler-ops/tunix-recipe2/k8s/kctl.sh; NS=${SANDBOX_NAMESPACE:-agents-system}
for js in $($K get jobsets -n default -o name | grep -E "jobset.jobset.x-k8s.io/${PFX}-" ); do $K delete "$js" -n default --wait=false; done
for i in $(seq 1 60); do n=$($K get pods -n default -o name | grep -c -E "pod/${PFX}-"); [ "$n" = 0 ] && break; sleep 5; done
$K get pods -n default -o wide | grep -E "^${PFX}-" || echo "no ${PFX}-* pods left"
for r in sandboxwarmpools sandboxclaims sandboxtemplates sandboxes; do $K delete $r -n $NS -l "app.kubernetes.io/created-by=${PFX}-orch" --wait=false --ignore-not-found=true 2>/dev/null; done
sleep 10; $K get pods -n $NS --no-headers 2>/dev/null | wc -l | xargs echo "sandbox pods left in $NS:"
