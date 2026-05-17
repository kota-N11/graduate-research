#!/bin/bash
# Dockerを使って研究室PCでストーリー生成を実行
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_DIR"

IMAGE_NAME="emotion-vectors"
DATA_DIR="$PROJECT_DIR/data/stories"
mkdir -p "$DATA_DIR" logs

# イメージをビルド（初回のみ数分かかる）
echo "=== Dockerイメージをビルド ==="
docker build -t "$IMAGE_NAME" -f notebooks/labpc/Dockerfile .

# GPU数を確認
N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
N=$(python3 -c "import json; print(len(json.load(open('config/emotions.json'))['emotions']))")
CHUNK=$(python3 -c "print(($N + $N_GPUS - 1) // $N_GPUS)")

echo "GPU数: $N_GPUS / 感情数: $N"
echo ""

# GPU数に応じて並列実行
for i in $(seq 0 $((N_GPUS - 1))); do
    START=$((i * CHUNK))
    END=$(python3 -c "print(min($((i * CHUNK + CHUNK)), $N))")
    echo "GPU${i}: 感情 ${START}-$((END-1)) を起動"

    docker run -d \
        --gpus "\"device=$i\"" \
        --name "${IMAGE_NAME}-gpu${i}" \
        -v "$DATA_DIR":/app/data/stories \
        -e HF_TOKEN="$HF_TOKEN" \
        "$IMAGE_NAME" \
        python src/generate.py --emotion-slice "${START}:${END}"
done

echo ""
echo "実行中。ログ確認:"
echo "  docker logs -f ${IMAGE_NAME}-gpu0"
echo ""
echo "停止・クリーンアップ:"
echo "  docker stop \$(docker ps -q --filter name=${IMAGE_NAME})"
echo "  docker rm \$(docker ps -aq --filter name=${IMAGE_NAME})"
