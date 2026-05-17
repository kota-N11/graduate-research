#!/bin/bash
# Dockerを使って研究室PCでストーリー生成 → 活性化抽出を実行
# 使い方: bash notebooks/labpc/run_docker.sh [generate|extract|all]
set -euo pipefail

MODE="${1:-all}"  # generate / extract / all
PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_DIR"

IMAGE_NAME="emotion-vectors"
mkdir -p data/stories data/activations data/vectors logs

# 開始時刻を記録（monitor.sh が参照する）
echo "$(date +%s)" > logs/start_time

# イメージをビルド（初回のみ数分かかる）
echo "=== Dockerイメージをビルド ==="
docker build -t "$IMAGE_NAME" -f notebooks/labpc/Dockerfile .

# GPU数・感情数を確認
N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
N=$(python3 -c "import json; print(len(json.load(open('config/emotions.json'))['emotions']))")
CHUNK=$(python3 -c "print(($N + $N_GPUS - 1) // $N_GPUS)")

echo "GPU数: $N_GPUS / 感情数: $N / モード: $MODE"
echo ""

run_parallel() {
    local CMD="$1"
    local STEP="$2"

    for i in $(seq 0 $((N_GPUS - 1))); do
        START=$((i * CHUNK))
        END=$(python3 -c "print(min($((i * CHUNK + CHUNK)), $N))")
        echo "GPU${i}: 感情 ${START}-$((END-1))"

        docker run -d \
            --gpus "\"device=$i\"" \
            --name "${IMAGE_NAME}-${STEP}-gpu${i}" \
            -v "$PROJECT_DIR/data":/app/data \
            -e HF_TOKEN="$HF_TOKEN" \
            "$IMAGE_NAME" \
            python $CMD --emotion-slice "${START}:${END}"
    done

    echo "ログ確認: docker logs -f ${IMAGE_NAME}-${STEP}-gpu0"
    docker wait $(docker ps -q --filter name="${IMAGE_NAME}-${STEP}")
    echo "[$(date '+%Y-%m-%d %H:%M')] ${STEP} 完了"
    docker rm $(docker ps -aq --filter name="${IMAGE_NAME}-${STEP}")
}

# ステップ1: ストーリー生成
if [[ "$MODE" == "generate" || "$MODE" == "all" ]]; then
    echo "=== Step 1: ストーリー生成 ==="
    run_parallel "src/generate.py" "generate"
    echo ""
fi

# ステップ2: 活性化抽出
if [[ "$MODE" == "extract" || "$MODE" == "all" ]]; then
    echo "=== Step 2: 活性化抽出 ==="
    run_parallel "src/extract.py" "extract"
    echo ""
fi

echo "=== 全工程完了 ==="
echo "データ保存先: $PROJECT_DIR/data/"
