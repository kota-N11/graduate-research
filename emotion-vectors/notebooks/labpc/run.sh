#!/bin/bash
# 研究室PCでストーリー生成を実行
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_DIR"
mkdir -p logs

# GPU数を確認
N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "GPU数: $N_GPUS"

# 感情数を取得
N=$(python3 -c "import json; print(len(json.load(open('config/emotions.json'))['emotions']))")
echo "感情数: $N"
echo ""

if [ "$N_GPUS" -ge 2 ]; then
    # 複数GPUの場合: 分割して並列実行
    CHUNK=$(python3 -c "print(($N + $N_GPUS - 1) // $N_GPUS)")
    echo "${N_GPUS}GPU並列で実行"
    for i in $(seq 0 $((N_GPUS - 1))); do
        START=$((i * CHUNK))
        END=$(python3 -c "print(min($((i * CHUNK + CHUNK)), $N))")
        CUDA_VISIBLE_DEVICES=$i python src/generate.py --emotion-slice ${START}:${END} > logs/gpu${i}.log 2>&1 &
        echo "  GPU${i}: 感情 ${START}-$((END-1))"
    done
    wait
else
    # 1GPUの場合: そのまま実行
    echo "1GPUで実行"
    python src/generate.py > logs/gpu0.log 2>&1
fi

echo ""
echo "[$(date '+%Y-%m-%d %H:%M')] 完了"
