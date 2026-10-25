#!/bin/sh
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
exec python3 "$PROJECT_DIR/scripts/checkpoints.py" package "$@"
