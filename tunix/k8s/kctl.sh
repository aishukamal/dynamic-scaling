#!/bin/bash
exec /opt/homebrew/bin/kubectl --context gke_aishuk-test_europe-west4-b_tpu-stress-test-eu "$@"
