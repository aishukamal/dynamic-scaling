#!/usr/bin/env bash
# Pure-Python tests (no verl/ray/torch needed). Inside the verl image the verl-seam tests run too.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD:$PWD/tests${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m unittest discover -s tests -p 'test_*.py' -v "$@"
