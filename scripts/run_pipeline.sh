#!/usr/bin/env bash
set -euo pipefail

python3 -m cmpfe.cli \
  --config config/videos.yaml \
  --data-root data \
  --output-root outputs \
  "$@"
