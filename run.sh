#!/bin/sh
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
case "${1-}" in
    ''|--attach) ;;
    *) echo "Usage: sh run.sh [--attach]" >&2; exit 2 ;;
esac
. "$PROJECT_DIR/scripts/docker-common.sh"
check_docker
prepare_directories
container_id=$(compose ps --all --quiet tuning)
if [ -n "$container_id" ]; then
    state=$(docker inspect --format '{{.State.Status}}' "$container_id")
    case "$state" in
        running) ;;
        exited|created) compose start tuning ;;
        *) echo "コンテナの状態が $state です。状態を確認してください。" >&2; exit 1 ;;
    esac
else
    if ! docker image inspect tuning:local >/dev/null 2>&1; then
        echo "共有用イメージがありません。先に sh \"$PROJECT_DIR/build.sh\" を実行してください。" >&2
        exit 1
    fi
    compose up --detach --no-build --pull never tuning
fi
if [ "${1-}" != '--attach' ]; then
    if command -v terminator >/dev/null 2>&1 && [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then
        exec terminator --working-directory "$PROJECT_DIR" -x sh "$PROJECT_DIR/run.sh" --attach
    fi
    echo 'TerminatorまたはGUI環境が見つからないため、現在の端末で接続します。' >&2
fi
exec docker compose --project-directory "$PROJECT_DIR" --file "$PROJECT_DIR/compose.yaml" \
    --project-name project-minimal-tuning exec --workdir /workspace/project_minimal tuning sh
