# Shared POSIX sh helpers. PROJECT_DIR is set by the caller.
export TUNING_UID="$(id -u)"
export TUNING_GID="$(id -g)"
compose() {
    docker compose --project-directory "$PROJECT_DIR" --file "$PROJECT_DIR/compose.yaml" \
        --project-name project-minimal-tuning "$@"
}
check_docker() {
    command -v docker >/dev/null 2>&1 || { echo 'Dockerをインストールしてください。' >&2; exit 1; }
    docker compose version >/dev/null
    docker info >/dev/null
}
prepare_directories() {
    mkdir -p "$PROJECT_DIR/datasets/mcap" "$PROJECT_DIR/datasets/lerobot" \
        "$PROJECT_DIR/checkpoints" "$PROJECT_DIR/exports" "$PROJECT_DIR/logs" \
        "$PROJECT_DIR/models" "$PROJECT_DIR/assets" "$PROJECT_DIR/.cache/tuning"
}
