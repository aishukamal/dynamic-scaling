#!/usr/bin/env bash
# cleanup_sandboxes.sh : deletes the SWE loop's leftover Sandbox CRs (names swe-* and grade-*, created by integration/verl/swe) in agents-system,
# in batches, --wait=false. Logs to logs/cleanup_sandboxes.out. Only these two prefixes (the loop's make_name("swe", ...) and the pristine grader).
CTX=gke_aishuk-test_europe-west4-b_tpu-stress-test-eu; K="/opt/homebrew/bin/kubectl --context $CTX --request-timeout=300s"
D="$HOME/workspaces/rl-pool-autoscaler-ops/tpu-m3"
echo "$(date -u +%FT%TZ) start"
$K get sandboxes -n agents-system -o name 2>/dev/null | grep -E "^sandbox.agents.x-k8s.io/(swe|grade)-" > "$D/results/sandboxes-to-delete.txt"
echo "$(date -u +%FT%TZ) to delete: $(wc -l < "$D/results/sandboxes-to-delete.txt")"
split -l 400 "$D/results/sandboxes-to-delete.txt" "$D/results/.sbx-batch-"
for B in "$D"/results/.sbx-batch-*; do
  $K delete -n agents-system --wait=false $(tr '\n' ' ' < "$B") >/dev/null 2>&1; echo "$(date -u +%FT%TZ) batch $(basename "$B") done"
done
rm -f "$D"/results/.sbx-batch-*
sleep 60
echo "$(date -u +%FT%TZ) remaining sandbox CRs (swe/grade): $($K get sandboxes -n agents-system -o name 2>/dev/null | grep -cE '/(swe|grade)-'); pods: $($K get pods -n agents-system --no-headers 2>/dev/null | grep -cE '^(swe|grade)-')"
