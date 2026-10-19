#!/usr/bin/env bash
# sandbox_monitor.sh <tag> [interval_s] : every interval, counts sandbox pods in ns agents-system by phase (all pods, and separately
# ours = names starting with swe- as the SWE loop creates them), Pending/waiting reasons, gVisor node counts -> results/sandboxes-<tag>.jsonl.
# Read-only. Stop with: kill $(cat results/sandbox_monitor-<tag>.pid)
CTX=gke_aishuk-test_europe-west4-b_tpu-stress-test-eu; K="/opt/homebrew/bin/kubectl --context $CTX --request-timeout=40s"
TAG=$1; IV=${2:-20}; D="$HOME/workspaces/rl-pool-autoscaler-ops/tpu-m3/results"; mkdir -p "$D"; echo $$ > "$D/sandbox_monitor-$TAG.pid"
while true; do
  T=$(date -u +%FT%TZ)
  ALLP=$($K get pods -n agents-system --no-headers -o custom-columns=N:.metadata.name,P:.status.phase,W:.status.containerStatuses[0].state.waiting.reason,NODE:.spec.nodeName 2>/dev/null)
  PH=$(echo "$ALLP" | awk 'NF{c[$2]++} END{for(k in c) printf "\"%s\":%d,", k, c[k]}')
  SW=$(echo "$ALLP" | awk '$1 ~ /^swe-/ {c[$2]++} END{for(k in c) printf "\"%s\":%d,", k, c[k]}')
  SWW=$(echo "$ALLP" | awk '$1 ~ /^swe-/ && $3 != "<none>" && $3 != "" {c[$3]++} END{for(k in c) printf "\"%s\":%d,", k, c[k]}')
  SWU=$(echo "$ALLP" | awk '$1 ~ /^swe-/ && $2 == "Pending" && ($4 == "<none>" || $4 == "") {n++} END{printf "%d", n+0}')
  NE=$($K get nodes -l cloud.google.com/gke-nodepool=swe-gvisor-pool-eu --no-headers 2>/dev/null | wc -l | tr -d ' ')
  NC=$($K get nodes -l cloud.google.com/gke-nodepool=swe-gvisor-pool-eu-c --no-headers 2>/dev/null | wc -l | tr -d ' ')
  echo "{\"t\":\"$T\",\"phase\":{${PH%,}},\"swe\":{${SW%,}},\"swe_waiting\":{${SWW%,}},\"swe_unscheduled\":$SWU,\"nodes_eu\":$NE,\"nodes_eu_c\":$NC}" >> "$D/sandboxes-$TAG.jsonl"
  sleep "$IV"
done
