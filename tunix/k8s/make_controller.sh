#!/bin/bash
# make_controller.sh <run-dir> <prefix> <base-worker-ids csv> <template indexes csv e.g. 1,2>
# Writes <run-dir>/controller.yaml (Deployment) and (re)creates ConfigMaps: controller code + rollout templates.
set -euo pipefail
D=$1; PFX=$2; BASE=$3; IDX=$4
K=~/workspaces/rl-pool-autoscaler-ops/tunix-slotgate/k8s/kctl.sh
CODE=/Users/aishuk/workspaces/GPU-CR/rl-pool-autoscaler/code/rl_pool_autoscaler
IMAGE=${TUNIX_IMAGE:-us-west1-docker.pkg.dev/aishuk-test/tpu-poc-repo/tunix-poc:mlperf-65d021b-raiden-20261014225303}
$K create configmap tunix-controller-code -n default --dry-run=client -o yaml \
  --from-file=pkg_init.py=$CODE/__init__.py --from-file=prom.py=$CODE/prom.py \
  --from-file=ctl_init.py=$CODE/controller/__init__.py --from-file=policy.py=$CODE/controller/policy.py \
  --from-file=tunix_backend.py=$CODE/controller/tunix_backend.py --from-file=tunix_main.py=$CODE/controller/tunix_main.py \
  | $K apply -f -
args=(); tpl=""
for i in ${IDX//,/ }; do args+=(--from-file=rollout-$i.yaml=$D/rollout-$i.yaml); tpl="${tpl:+$tpl,}${PFX}-roll-$i=${PFX}-roll-$i=/templates/rollout-$i.yaml"; done
$K create configmap tunix-controller-templates-$PFX -n default --dry-run=client -o yaml "${args[@]}" | $K apply -f -
NTPL=$(echo "$IDX" | tr ',' '\n' | wc -l | tr -d ' '); NBASE=$(echo "$BASE" | tr ',' '\n' | wc -l | tr -d ' ')
cat > $D/controller.yaml <<EOY
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tunix-controller-$PFX
  namespace: default
  labels: {app: tunix-controller, run-prefix: "$PFX"}
spec:
  replicas: 1
  strategy: {type: Recreate}
  selector: {matchLabels: {app: tunix-controller, run-prefix: "$PFX"}}
  template:
    metadata:
      labels: {app: tunix-controller, run-prefix: "$PFX"}
    spec:
      serviceAccountName: tunix-controller
      nodeSelector: {cloud.google.com/gke-nodepool: pathways-head-jax}
      tolerations:
      - {key: cloud.google.com/gke-nodepool, operator: Exists, effect: NoSchedule}
      containers:
      - name: controller
        image: $IMAGE
        imagePullPolicy: IfNotPresent
        command: ["python3", "-m", "rl_pool_autoscaler.controller.tunix_main"]
        env:
        - {name: PYTHONPATH, value: /code}
        - {name: PYTHONUNBUFFERED, value: "1"}
        - {name: STATUS_URL, value: "http://${PFX}-orch-metrics.default.svc.cluster.local:${EXPORTER_PORT:-8787}"}
        - {name: POLL_INTERVAL_S, value: "${POLL_INTERVAL_S:-10}"}
        - {name: MIN_WORKERS, value: "${MIN_WORKERS:-1}"}
        - {name: MAX_WORKERS, value: "${MAX_WORKERS:-$((NTPL + NBASE))}"}
        - {name: IDLE_HI, value: "${IDLE_HI:-0.40}"}
        - {name: IDLE_LO, value: "${IDLE_LO:-0.15}"}
        - {name: DWELL_STEPS, value: "${DWELL_STEPS:-2}"}
        - {name: MIN_ACTION_INTERVAL_S, value: "${MIN_ACTION_INTERVAL_S:-120}"}
        - {name: MQ_HI_FRAC, value: "0.0"}
        - {name: TEMPLATES, value: "$tpl"}
        - {name: BASE_WORKERS, value: "$BASE"}
        - {name: JOIN_TIMEOUT_S, value: "${JOIN_TIMEOUT_S:-420}"}
        - {name: EVICT_TIMEOUT_S, value: "${EVICT_TIMEOUT_S:-180}"}
        - {name: GUARD_ENABLED, value: "${GUARD_ENABLED:-1}"}
        - {name: GUARD_WINDOW, value: "${GUARD_WINDOW:-2}"}
        - {name: GUARD_MIN_GAIN, value: "${GUARD_MIN_GAIN:-0.10}"}
        - {name: UTIL_HI, value: "${UTIL_HI:-}"}
        - {name: UTIL_LO, value: "${UTIL_LO:-}"}
        - {name: UTIL_STAT, value: "${UTIL_STAT:-peak}"}
        - {name: UTIL_WINDOW, value: "${UTIL_WINDOW:-}"}
        - {name: QUEUE_MIN, value: "${QUEUE_MIN:-}"}
        - {name: DECISIONS_JSONL, value: /var/log/rl-pool/decisions-$PFX.jsonl}
        - {name: DRY_RUN, value: "${DRY_RUN:-0}"}
        - {name: NAMESPACE, value: default}
        resources:
          requests: {cpu: "200m", memory: "512Mi"}
          limits: {cpu: "1", memory: "1Gi"}
        volumeMounts:
        - {name: code, mountPath: /code}
        - {name: templates, mountPath: /templates}
        - {name: decisions, mountPath: /var/log/rl-pool}
      volumes:
      - name: code
        configMap:
          name: tunix-controller-code
          items:
          - {key: pkg_init.py, path: rl_pool_autoscaler/__init__.py}
          - {key: prom.py, path: rl_pool_autoscaler/prom.py}
          - {key: ctl_init.py, path: rl_pool_autoscaler/controller/__init__.py}
          - {key: policy.py, path: rl_pool_autoscaler/controller/policy.py}
          - {key: tunix_backend.py, path: rl_pool_autoscaler/controller/tunix_backend.py}
          - {key: tunix_main.py, path: rl_pool_autoscaler/controller/tunix_main.py}
      - name: templates
        configMap: {name: tunix-controller-templates-$PFX}
      - name: decisions
        hostPath: {path: /tmp/tunix-controller, type: DirectoryOrCreate}
EOY
echo "wrote $D/controller.yaml (templates: $tpl; base: $BASE; max=$((NTPL + NBASE)))"
