#!/bin/bash
# 動作確認用（感情2個×トピック2個×ストーリー2本 = 8件のみ生成）
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_DIR"

IMAGE_NAME="emotion-vectors"
mkdir -p data/stories data/activations logs

echo "=== Dockerイメージをビルド ==="
docker build -t "$IMAGE_NAME" -f notebooks/labpc/Dockerfile .

echo ""
echo "=== テスト実行（感情2個・トピック2個・ストーリー2本）==="
docker run --rm \
    --gpus "device=0" \
    --name "${IMAGE_NAME}-test" \
    -v "$PROJECT_DIR/data":/app/data \
    -e HF_TOKEN="$HF_TOKEN" \
    "$IMAGE_NAME" \
    python src/generate.py \
        --emotion-slice 0:2 \
        --n-topics 2 \
        --n-stories 2

echo ""
echo "=== 生成結果確認 ==="
ls data/stories/
echo ""
echo "テスト完了。問題なければ本番実行:"
echo "  bash notebooks/labpc/run_docker.sh all"
