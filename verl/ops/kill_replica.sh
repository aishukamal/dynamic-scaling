#!/usr/bin/env bash
# kill_replica.sh <chip> <sampler host short> : SIGKILL the vLLM worker process (RayWorkerWrapper) holding /dev/vfio/<chip> on that sampler host (CLUSTER env)
set -euo pipefail
CTX=gke_aishuk-test_europe-west4-b_tpu-stress-test-eu; K="/opt/homebrew/bin/kubectl --context $CTX --request-timeout=60s"
H="$2"; case "$H" in gke-*) ;; *) H="gke-tpu-dac1e83d-$H";; esac
W=$($K get pods -l ray.io/cluster=${CLUSTER:-ray-tpum3},ray.io/group=tpu-group-1 -o wide --no-headers | awk -v h="$H" '$7==h {print $1}')
[ -n "$W" ] || { echo "no sampler worker pod on host $2"; exit 1; }
$K cp "$(dirname "$0")/vfio_map.py" "$W:/tmp/vfio_map_tpum2.py"
MAP=$($K exec "$W" -c ray-worker -- python3 /tmp/vfio_map_tpum2.py 2>/dev/null)
PID=$(echo "$MAP" | python3 -c "import sys,json; m=json.load(sys.stdin); c='/dev/vfio/$1'; x=[e for e in m if c in e['vfio']]; print(x[0]['pid'] if x else '')")
echo "map: $MAP" | cut -c1-900
[ -n "$PID" ] || { echo "no holder for chip $1"; exit 1; }
echo "$(date -u +%FT%T.%3NZ) KILL chip=$1 pid=$PID"; date -u +%s.%N
$K exec "$W" -c ray-worker -- kill -9 "$PID"
echo "$(date -u +%FT%T.%3NZ) kill sent"
