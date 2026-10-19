#!/usr/bin/env bash
# inpod_check.sh : run inside the inspection pod. Installs the three packages with --no-deps, asserts the pinned set is unchanged,
# imports the SWE loop against the image's verl. No host writes.
set -u
cd /home/ray && tar -xzf swe_work.tgz
pip freeze 2>/dev/null | grep -i -E "^(aiohttp|pydantic|pydantic_core|typing_extensions|certifi|yarl|torch|vllm|ray|transformers)==" | sort > /home/ray/pins_before.txt
pip install --no-cache-dir --no-deps kubernetes==37.0.1 websocket-client==1.9.2 durationpy==0.11 2>&1 | tail -1
pip freeze 2>/dev/null | grep -i -E "^(aiohttp|pydantic|pydantic_core|typing_extensions|certifi|yarl|torch|vllm|ray|transformers)==" | sort > /home/ray/pins_after.txt
if diff /home/ray/pins_before.txt /home/ray/pins_after.txt; then echo PINS-UNCHANGED; else echo PINS-CHANGED; fi
cat /home/ray/pins_after.txt
(echo "127.0.0.1 metadata.google.internal" >> /etc/hosts) 2>/dev/null || true
export TPU_SKIP_MDS_QUERY=1 VERL_PLATFORM=tpu PYTHONPATH=/home/ray/swe_work
cd /home/ray
timeout 600 python3 - <<'PY' 2>&1 | grep -v -i "warn" | tail -12
import kubernetes, kubernetes.stream
from kubernetes import client, config
print("kubernetes", kubernetes.__version__)
import integration.verl.swe.swe_agent_loop as m
from verl.experimental.agent_loop.agent_loop import _agent_loop_registry, AgentLoopBase
print("registry:", list(_agent_loop_registry))
print("has apply_chat_template on base:", hasattr(AgentLoopBase, "apply_chat_template"), "| shim on SWE loop:", hasattr(m.SWEAgentLoop, "apply_chat_template"))
print("IMPORT-OK")
PY
