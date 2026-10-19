#!/bin/bash
# Sequenced TPU capacity expansion on tpu-stress-test-eu (pool ops serialize per cluster).
P=aishuk-test; CL=tpu-stress-test-eu; Z=europe-west4-b; LOG=~/workspaces/rl-pool-autoscaler-ops/capacity/expand.log
SCOPES=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/trace.append
wait_idle(){ while gcloud container operations list --project $P --filter="targetLink~$CL AND status=RUNNING" --format="value(name)" 2>/dev/null | grep -q .; do sleep 30; done; }
run(){ echo "$(date -u +%FT%TZ) RUN: $*" >> $LOG; wait_idle; "$@" >> $LOG 2>&1; echo "$(date -u +%FT%TZ) EXIT $? : $1 $2 $3" >> $LOG; }
run gcloud container clusters resize $CL --node-pool v6e-4x1-long --num-nodes 16 --zone $Z --project $P --quiet
run gcloud container node-pools create v6e-8-spot-b --cluster $CL --zone $Z --node-locations europe-west4-b --project $P --machine-type ct6e-standard-8t --num-nodes 4 --spot --disk-size 500 --scopes $SCOPES --quiet
run gcloud container node-pools create v6e-4-spot-b --cluster $CL --zone $Z --node-locations europe-west4-b --project $P --machine-type ct6e-standard-4t --num-nodes 8 --spot --disk-size 500 --scopes $SCOPES --quiet
run gcloud container node-pools create v6e-8-spot-c --cluster $CL --zone $Z --node-locations europe-west4-c --project $P --machine-type ct6e-standard-8t --num-nodes 4 --spot --disk-size 500 --scopes $SCOPES --quiet
run gcloud container node-pools create v6e-4-spot-c --cluster $CL --zone $Z --node-locations europe-west4-c --project $P --machine-type ct6e-standard-4t --num-nodes 8 --spot --disk-size 500 --scopes $SCOPES --quiet
echo "$(date -u +%FT%TZ) DONE" >> $LOG
