#!/usr/bin/env bash
set -euo pipefail

python -m cmpfe.cli \
  --config config/videos.yaml \
  --data-root data \
  --output-root outputs \
  "$@"
