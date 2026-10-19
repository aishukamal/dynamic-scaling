# Kubernetes-side controller image (NOT the verl image). stdlib-only Python; no extra deps.
# Build (Cloud Build, not locally):
#   gcloud builds submit --project aishuk-test --tag us-west1-docker.pkg.dev/aishuk-test/tpu-poc-repo/rl-pool-controller:v1 .
FROM python:3.11-slim

WORKDIR /app
COPY rl_pool_autoscaler/__init__.py rl_pool_autoscaler/prom.py /app/rl_pool_autoscaler/
COPY rl_pool_autoscaler/controller /app/rl_pool_autoscaler/controller
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app
RUN useradd -r -u 10001 controller && mkdir -p /var/log/rl-pool && chown controller /var/log/rl-pool
USER 10001
CMD ["python", "-m", "rl_pool_autoscaler.controller.main"]
