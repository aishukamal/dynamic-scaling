# tunix-poc:mlperf-65d021b-deepswe-raiden = raiden image + the repo Dockerfile's INSTALL_DEEPSWE_DEPS=true block
# (kubernetes, gym, swebench, openhands-sdk, k8s-agent-sandbox, agentic-sandbox-client, agent_sandbox_rl, r2egym + the 3 seds),
# then the same numpy/jax/flax re-pin the repo Dockerfile does and the Raiden wheel re-installed (already in /app/raiden_wheels).
FROM us-west1-docker.pkg.dev/aishuk-test/tpu-poc-repo/tunix-poc:mlperf-65d021b-raiden-20261014225303
RUN uv pip install kubernetes gym swebench==3.0.2 'openhands-sdk>=1.44.1' 'k8s-agent-sandbox>=0.5.1' httpx docker fire simple-parsing unidiff && \
    uv pip install --no-deps git+https://github.com/kubernetes-sigs/agent-sandbox.git#subdirectory=clients/python/agentic-sandbox-client && \
    uv pip install --no-deps git+https://github.com/kubernetes-sigs/agent-sandbox.git#subdirectory=examples/agent-sandbox-rl && \
    uv pip install --no-deps git+https://github.com/kubernetes-sigs/agent-sandbox.git#subdirectory=clients/integrations/openhands && \
    uv pip install --no-deps git+https://github.com/r2e-gym/r2e-gym.git@0d94c4eb9431cd195c55a7ea3abd54006c9a1735 && \
    sed -i 's/create_repo, upload_folder, HfFolder/create_repo, upload_folder/' /opt/venv/lib/python3.12/site-packages/r2egym/agenthub/utils/utils.py && \
    sed -i 's/self.commit = ParsedCommit(\*\*json.loads(self.commit_json))/self.commit = ParsedCommit(\*\*(json.loads(self.commit_json) if isinstance(self.commit_json, str) else self.commit_json))/' /opt/venv/lib/python3.12/site-packages/r2egym/agenthub/runtime/docker.py && \
    sed -i 's/sel = c.resources.managed_selector()/sel = self.run_selector()/' /opt/venv/lib/python3.12/site-packages/agent_sandbox_rl/fleet.py
RUN uv pip install numpy==2.3.5 'jax==0.11.0' 'flax==0.12.7' && \
    python3 -m pip install --force-reinstall --no-deps /app/raiden_wheels/*.whl && \
    uv pip install --no-deps -e /app
