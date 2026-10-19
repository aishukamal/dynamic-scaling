# tunix-poc:mlperf-65d021b-raiden = base image + tpu_sync_jax (Raiden) wheel built from google/tpu-sync main
# (the wheel Tunix pins lives in a private bucket). Mirrors Dockerfile's INSTALL_RAIDEN=true step
# (scripts/install_raiden.sh: force-reinstall --no-deps + import check; protos were already compiled in the base).
FROM us-west1-docker.pkg.dev/aishuk-test/tpu-poc-repo/tunix-poc:mlperf-65d021b-base
COPY wheels/ /app/raiden_wheels/
RUN python3 -m pip install --force-reinstall --no-deps /app/raiden_wheels/*.whl && python3 -c "import tpu_sync; print('tpu_sync ok')"
