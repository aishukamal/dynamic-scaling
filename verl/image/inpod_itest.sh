#!/usr/bin/env bash
# inpod_itest.sh : runs the SWE loop integration check (scripted model, real eu sandbox + pristine grading) inside the inspection pod
set -u
export PYTHONPATH=/home/ray/swe_work TPU_SKIP_MDS_QUERY=1 VERL_PLATFORM=tpu HF_HUB_OFFLINE=0 SWE_SANDBOX_NAMESPACE=agents-system
cd /home/ray
date -u
timeout 1500 python3 -m integration.verl.swe.integration_check --parquet /home/ray/train_calibrated.parquet --instance aiohttp-f0d74880deec 2>&1 | grep -v -i "warn\|importing.py\|libnccl\|cupyx\|pallas"
echo "exit=${PIPESTATUS[0]}"; date -u
