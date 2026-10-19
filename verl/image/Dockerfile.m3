# verl-tpu-colocated:v8-rc10-swe1 = :v8-rc10 (by digest; pinned TPU stack untouched) + the py-inference-scheduler SWE agent loop
# package at /opt/swe_work (branch swe-poc-local, see SWE_LOOP_SOURCE) + its only missing runtime deps, installed with --no-deps
# (a plain `pip install kubernetes` would upgrade aiohttp, pydantic, certifi, typing_extensions, yarl). The build fails if any
# pinned package version changes or the loop does not import.
FROM us-west1-docker.pkg.dev/aishuk-test/tpu-poc-repo/verl-tpu-colocated@sha256:cfba89fa6f8d35aef0d70ecd419be26197f77e28681c43df7ad38983e508a53b
USER root
COPY swe_work /opt/swe_work
COPY check_m3.py /opt/check_m3.py
RUN pip freeze > /opt/freeze_before.txt && \
    pip install --no-cache-dir --no-deps kubernetes==37.0.1 websocket-client==1.9.2 durationpy==0.11 && \
    pip freeze > /opt/freeze_after.txt && \
    python3 /opt/check_m3.py
RUN set -e; (echo "127.0.0.1 metadata.google.internal" >> /etc/hosts) || true; \
    export TPU_SKIP_MDS_QUERY=1 TPU_ACCELERATOR_TYPE=v5e VERL_PLATFORM=tpu PYTHONPATH=/opt/swe_work; \
    cd /opt/swe_work && python3 -c "import kubernetes, kubernetes.stream; import integration.verl.swe.swe_agent_loop as m; from verl.experimental.agent_loop.agent_loop import _agent_loop_registry; assert 'swe_agent' in _agent_loop_registry, list(_agent_loop_registry); print('SWE loop import OK; kubernetes', kubernetes.__version__)"
WORKDIR /app
