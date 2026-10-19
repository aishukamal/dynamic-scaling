#!/usr/bin/env bash
# poll.sh <submission-id> [lines] [cut]
CTX=gke_aishuk-test_europe-west4-b_tpu-stress-test-eu; K="/opt/homebrew/bin/kubectl --request-timeout=55s --context $CTX"
SUB="$1"; N="${2:-30}"; C="${3:-300}"
HEAD=$($K get pods -l ray.io/node-type=head,ray.io/cluster=${CLUSTER:-ray-tpum3} -o jsonpath='{.items[0].metadata.name}')
$K exec "$HEAD" -- bash -c "ray job logs $SUB 2>/dev/null | grep -aE 'SIGSEGV|Traceback|ActorDied|Error|error|ran out of memory|RESOURCE_EXHAUSTED|\[POOL\]|RAIDEN|destination sampler|step:[0-9]+ -|KV cache size|init engine|Aborted [0-9]+ requests|TPU rollout process addresses|TPUM3 layout|PARITY|Mesh|SliceBuilder|Target Topology|swe_agent|Sandbox|sandbox|No available servers|Error in _run_prompt|dead_detected|spawn|Compil|recompile' | grep -avE 'Error 404|libnccl|UserWarning|tpu_info.py|importing.py|cupy|install_library|accelerator-type|swe_agent rollout error on None' | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-$C | tail -$N; echo lines=\$(ray job logs $SUB 2>/dev/null | wc -l); ray job status $SUB 2>/dev/null | tail -1"
date -u +%T
