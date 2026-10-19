#!/usr/bin/env bash
# inpod_import_overlay.sh <tgz basename> : extract an overlay tarball under /home/ray/ovl-<name> and import the TPU engine modules from it
set -u
B=$1; N=${B%.tgz}; mkdir -p /home/ray/ovl-$N && tar -xzf /home/ray/$B -C /home/ray/ovl-$N
(echo "127.0.0.1 metadata.google.internal" >> /etc/hosts) 2>/dev/null || true
export TPU_SKIP_MDS_QUERY=1 VERL_PLATFORM=tpu PYTHONPATH=/home/ray/ovl-$N
cd /home/ray
python3 - <<'PY' 2>&1 | grep -v -i "warn\|importing.py\|libnccl\|cupyx\|pallas\|tpu_info\|Triton" | tail -25
import importlib, traceback, verl_hardware_plugin
print("plugin from:", verl_hardware_plugin.__file__)
for m in ["verl_hardware_plugin.accelerators.tpu.engines.tpu_pool_actor",
          "verl_hardware_plugin.accelerators.tpu.engines.tpu_pool_control",
          "verl_hardware_plugin.accelerators.tpu.engines.tpu_checkpoint_engine",
          "verl_hardware_plugin.accelerators.tpu.engines.torchtitan_tpu"]:
    try:
        importlib.import_module(m); print("OK  ", m)
    except Exception as e:
        print("FAIL", m, type(e).__name__, str(e)[:300]); traceback.print_exc(limit=3)
from verl.workers.engine.base import EngineRegistry
print("registry language_model backends:", {b: list(v.keys()) for b, v in EngineRegistry._engines.get("language_model", {}).items()})
PY
