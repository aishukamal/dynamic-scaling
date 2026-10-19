#!/usr/bin/env bash
# metrics.sh [status] : GET /metrics (or /status) from the control actor via the head pod (python client, no curl in the image)
CTX=gke_aishuk-test_europe-west4-b_tpu-stress-test-eu; K="/opt/homebrew/bin/kubectl --context $CTX --request-timeout=60s"
HEAD=$($K get pods -l ray.io/node-type=head,ray.io/cluster=${CLUSTER:-ray-tpum3} -o jsonpath='{.items[0].metadata.name}')
$K exec "$HEAD" -- python3 /home/ray/rlpool/ctl.py "${CONTROL_HOST:-127.0.0.1}" "${1:-metrics}" | grep -v "^#"; date -u +%T
