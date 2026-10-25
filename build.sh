#!/bin/sh
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
. "$PROJECT_DIR/scripts/docker-common.sh"
check_docker
prepare_directories
if ! compose build tuning; then
    echo 'Dockerイメージのビルドに失敗しました。上記のエラーを確認してください。' >&2
    exit 1
fi
echo 'ビルド完了。sh run.sh でコンテナに入れます。'
