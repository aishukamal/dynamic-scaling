#!/usr/bin/env bash
# taskrunner_ip.sh <job> : run in the head pod: IP of the node running the job's TaskRunnerV1 (from the job log)
ray job logs "$1" 2>/dev/null | grep -a -m1 -o "TaskRunnerV1 pid=[0-9]*, ip=[0-9.]*" | sed 's/.*ip=//'
