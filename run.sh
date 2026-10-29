#!/bin/sh
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
case "${1-}" in
    ''|--here|--attach) ;;
    *) echo "Usage: sh run.sh [--here]" >&2; exit 2 ;;
esac
[ "$#" -le 1 ] || { echo "Usage: sh run.sh [--here]" >&2; exit 2; }
. "$PROJECT_DIR/scripts/docker-common.sh"
# Check access before opening a window; preserve Docker's diagnostic output.
if ! command -v docker >/dev/null 2>&1; then
    echo 'Dockerがありません。Dockerをインストールしてください。' >&2
    exit 1
fi
if ! docker compose version >/dev/null || ! docker info >/dev/null; then
    echo 'Dockerを利用できません。上記の原因（権限、daemon、Composeなど）を確認してください。' >&2
    exit 1
fi
if [ -z "${1-}" ]; then
    if ! command -v terminator >/dev/null 2>&1; then
        echo 'Terminatorがありません。必要なら sudo apt install terminator を実行してください。現在の端末で接続します。' >&2
    elif [ -n "${SSH_CONNECTION:-}${SSH_CLIENT:-}${SSH_TTY:-}" ] ||
         [ -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then
        echo 'SSH接続中またはGUI環境がないため、現在の端末で接続します。' >&2
    else
        # A separate process avoids reusing an existing Terminator window.
        # Only the child executes the container startup/attachment below.
        (
            # Snap IDEs can export GTK/Python paths that load Snap's glibc
            # into the host Terminator. Clean only this launch environment.
            case "${SNAP:-}${LD_LIBRARY_PATH:-}${LD_PRELOAD:-}${GTK_PATH:-}${GIO_MODULE_DIR:-}${GI_TYPELIB_PATH:-}${PYTHONHOME:-}${PYTHONPATH:-}" in
                *'/snap/'*)
                    unset LD_LIBRARY_PATH LD_PRELOAD GTK_PATH GTK_EXE_PREFIX \
                        GTK_DATA_PREFIX GTK_IM_MODULE_FILE GIO_MODULE_DIR \
                        GI_TYPELIB_PATH PYTHONHOME PYTHONPATH
                    ;;
            esac
            exec terminator --no-dbus --title 'PiPER Fine-tuning' \
                --working-directory "$PROJECT_DIR" -x sh "$PROJECT_DIR/run.sh" --here
        ) </dev/null &
        terminal_pid=$!
        # Catch immediate GUI/display failures without blocking on the session.
        sleep 1
        if ! kill -0 "$terminal_pid" 2>/dev/null; then
            if wait "$terminal_pid"; then
                exit 0
            fi
            echo 'Terminatorの起動に失敗しました。上記の原因を確認するか、sh run.sh --here を使用してください。' >&2
            exit 1
        fi
        exit 0
    fi
fi
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
# Inspect the actual service identity, including when reusing a container that
# was created with a different primary GID. Only account registration uses root.
identity=$(compose exec -T tuning python3 /workspace/project_minimal/scripts/container_user.py --probe)
if [ -n "$identity" ]; then
    compose exec -T --user 0 tuning python3 /workspace/project_minimal/scripts/container_user.py \
        --identity "$identity" --username "$(id -un)"
    # Confirm NSS now resolves the service UID/GID before opening Bash.
    remaining=$(compose exec -T tuning python3 /workspace/project_minimal/scripts/container_user.py --probe)
    [ -z "$remaining" ] || { echo 'コンテナ内のユーザー名登録に失敗しました。' >&2; exit 1; }
fi
exec docker compose --project-directory "$PROJECT_DIR" --file "$PROJECT_DIR/compose.yaml" \
    --project-name project-minimal-tuning exec --workdir /workspace/project_minimal tuning bash
