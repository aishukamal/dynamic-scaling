#!/usr/bin/env bash
# prep_head.sh <cluster> : copies the controller package, ctl.py and the in-pod helper scripts to the cluster's head pod
set -euo pipefail
CTX=gke_aishuk-test_europe-west4-b_tpu-stress-test-eu; K="/opt/homebrew/bin/kubectl --context $CTX --request-timeout=120s"
C=$1; D="$HOME/workspaces/rl-pool-autoscaler-ops/tpu-m3"
HEAD=$($K get pods -l ray.io/node-type=head,ray.io/cluster=$C -o jsonpath='{.items[0].metadata.name}')
$K exec "$HEAD" -- mkdir -p /home/ray/rlpool
$K cp "$HOME/workspaces/GPU-CR/rl-pool-autoscaler/code/rl_pool_autoscaler" "$HEAD:/home/ray/rlpool/rl_pool_autoscaler"
for f in ctl.py filter_log.py taskrunner_ip.sh .headgrep.py .headtail.py .lasttail.py .wdtail.py .trajsum.py; do $K cp "$D/$f" "$HEAD:/home/ray/${f#.}"; done
$K cp "$D/ctl.py" "$HEAD:/home/ray/rlpool/ctl.py"
$K exec "$HEAD" -- env PYTHONPATH=/home/ray/rlpool python3 -c "import rl_pool_autoscaler.controller.main; print('controller import ok on', '$C')"
