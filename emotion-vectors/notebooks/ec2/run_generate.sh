#!/bin/bash
# ストーリー生成を4GPU並列で実行
# 使い方: tmux new -s generate → bash scripts/run_generate.sh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
mkdir -p logs

# 感情数を取得して4分割
N=$(python3 -c "import json; print(len(json.load(open('config/emotions.json'))['emotions']))")
CHUNK=$(python3 -c "print(($N + 3) // 4)")
C1=$CHUNK
C2=$((CHUNK * 2))
C3=$((CHUNK * 3))

echo "[$(date '+%Y-%m-%d %H:%M')] 開始: ${N}感情 → 4GPU分割（各${CHUNK}感情）"
echo "  GPU0: 0-$((C1-1))   GPU1: ${C1}-$((C2-1))   GPU2: ${C2}-$((C3-1))   GPU3: ${C3}-$((N-1))"
echo ""

# 4プロセス並列起動
CUDA_VISIBLE_DEVICES=0 python src/generate.py --emotion-slice 0:${C1}   > logs/gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python src/generate.py --emotion-slice ${C1}:${C2} > logs/gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python src/generate.py --emotion-slice ${C2}:${C3} > logs/gpu2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python src/generate.py --emotion-slice ${C3}:${N}  > logs/gpu3.log 2>&1 &

echo "ログ確認:"
echo "  tail -f logs/gpu0.log     # GPU0のみ"
echo "  tail -f logs/gpu*.log     # 全GPU"
echo ""
echo "切断するには Ctrl+B → D  (tmux detach)"
echo "再接続:    tmux attach -t generate"
echo ""

wait
echo "[$(date '+%Y-%m-%d %H:%M')] 全GPU完了"
