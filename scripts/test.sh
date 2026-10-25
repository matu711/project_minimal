#!/bin/sh
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
TEST_DEPS=$(mktemp -d "${TMPDIR:-/tmp}/tuning-test-deps.XXXXXX")
trap 'rm -rf "$TEST_DEPS"' 0
uv pip install --target "$TEST_DEPS" --no-deps pytest==8.3.4 pluggy==1.6.0 iniconfig==2.1.0
export PYTHONPATH="$TEST_DEPS:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu
python3 -m pytest "$PROJECT_DIR/tests" "$@"
