#!/usr/bin/env bash
# scale_out_manual.sh <N> : the K8s actuator's scale-out by hand (quasi-static ladder, product path): patch the RayCluster
# sampler group replicas to N -> wait for the new worker pod Running -> wait until /status shows a rollout node with 4 free
# TPUs (joined Ray) -> POST /scale N (the hook spawns a TP=4 replica there via init_standalone). Prints UTC timestamps.
set -u
CTX=gke_aishuk-test_europe-west4-b_tpu-stress-test-eu; K="/opt/homebrew/bin/kubectl --context $CTX --request-timeout=60s"
N=$1; H="${CONTROL_HOST:-127.0.0.1}"
HEAD=$($K get pods -l ray.io/node-type=head,ray.io/cluster=${CLUSTER:-ray-tpum3} -o jsonpath='{.items[0].metadata.name}')
IDX=$($K get raycluster ${CLUSTER:-ray-tpum3} -o json | python3 -c "import sys,json; g=json.load(sys.stdin)['spec']['workerGroupSpecs']; print([x['groupName'] for x in g].index('tpu-group-1'))")
echo "$(date -u +%FT%T.%3NZ) PATCH replicas=$N (group index $IDX)"
$K patch raycluster ${CLUSTER:-ray-tpum3} --type=json -p "[{\"op\":\"replace\",\"path\":\"/spec/workerGroupSpecs/$IDX/replicas\",\"value\":$N}]"
T_PATCH=$(date +%s.%N)
while true; do
  R=$($K get pods -l ray.io/cluster=${CLUSTER:-ray-tpum3},ray.io/group=tpu-group-1 --no-headers 2>/dev/null | awk '$3=="Running"' | wc -l | tr -d ' ')
  [ "$R" -ge "$N" ] && break; sleep 3
done
T_POD=$(date +%s.%N); echo "$(date -u +%FT%T.%3NZ) POD RUNNING ($N sampler pods) +$(python3 -c "print(round($T_POD-$T_PATCH,1))") s"
$K get pods -l ray.io/cluster=${CLUSTER:-ray-tpum3},ray.io/group=tpu-group-1 -o wide --no-headers | awk '{print "  ", $1, $3, $7}'
while true; do
  FREE=$($K exec "$HEAD" -- python3 /home/ray/rlpool/ctl.py "$H" status 2>/dev/null | python3 -c "
import sys,json; st=json.load(sys.stdin); print(sum(1 for n in st.get('nodes',[]) if n.get('rollout_node') and (n.get('gpu_available') or 0) >= 4))" 2>/dev/null)
  [ "${FREE:-0}" -ge 1 ] && break; sleep 3
done
T_NODE=$(date +%s.%N); echo "$(date -u +%FT%T.%3NZ) RAY NODE FREE (rollout node with 4 TPU available) +$(python3 -c "print(round($T_NODE-$T_PATCH,1))") s after patch"
$K exec "$HEAD" -- python3 /home/ray/rlpool/ctl.py "$H" scale "$N"
echo "$(date -u +%FT%T.%3NZ) POSTED /scale $N (+$(python3 -c "print(round($(date +%s.%N)-$T_PATCH,1))") s after patch)"
