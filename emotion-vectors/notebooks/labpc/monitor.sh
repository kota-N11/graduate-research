#!/bin/bash
# 生成・抽出の進捗と推定終了時間を表示
# 使い方: bash notebooks/labpc/monitor.sh [間隔秒数（省略時は一回だけ表示）]

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
INTERVAL="${1:-0}"

START_FILE="$PROJECT_DIR/logs/start_time"
if [ ! -f "$START_FILE" ]; then
    echo "警告: start_time が見つかりません。run_docker.sh を先に実行してください。"
    exit 1
fi
START_TS=$(cat "$START_FILE")

N_EMOTIONS=$(python3 -c "import json; print(len(json.load(open('$PROJECT_DIR/config/emotions.json'))['emotions']))")
N_TOPICS=$(python3 -c "import json; print(len(json.load(open('$PROJECT_DIR/config/topics.json'))['topics']))")
TOTAL=$((N_EMOTIONS * N_TOPICS))

calc_eta() {
    local DONE=$1
    local ELAPSED=$2
    if [ "$DONE" -gt 0 ] && [ "$ELAPSED" -gt 0 ]; then
        python3 -c "
import datetime
now = $(date +%s)
rate = $DONE / $ELAPSED
remaining = ($TOTAL - $DONE) / rate if rate > 0 else 0
h = int(remaining // 3600)
m = int((remaining % 3600) // 60)
eta = datetime.datetime.fromtimestamp(now + remaining).strftime('%m/%d %H:%M')
rate_h = round(rate * 3600, 1)
print(f'{rate_h}件/時間 | 残り {h}時間{m}分 | 完了予定 {eta}')
"
    else
        echo "計算中..."
    fi
}

show_progress() {
    local NOW=$(date +%s)
    local ELAPSED=$((NOW - START_TS))
    local ELAPSED_H=$((ELAPSED / 3600))
    local ELAPSED_M=$(((ELAPSED % 3600) / 60))

    local STORIES=$(find "$PROJECT_DIR/data/stories" -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
    local ACTIVATIONS=$(find "$PROJECT_DIR/data/activations" -name '*.npy' 2>/dev/null | wc -l | tr -d ' ')

    local STORIES_PCT=$(python3 -c "print(round($STORIES / $TOTAL * 100, 1))")
    local ACT_PCT=$(python3 -c "print(round($ACTIVATIONS / $N_EMOTIONS * 100, 1))")

    echo "=============================="
    echo "$(date '+%Y-%m-%d %H:%M:%S')  経過: ${ELAPSED_H}時間${ELAPSED_M}分"
    echo ""
    echo "[Step 1] ストーリー生成"
    echo "  $STORIES / $TOTAL 件 ($STORIES_PCT%)"
    echo "  $(calc_eta $STORIES $ELAPSED)"
    echo ""
    echo "[Step 2] 活性化抽出"
    echo "  $ACTIVATIONS / $N_EMOTIONS 感情 ($ACT_PCT%)"
    echo "  $(calc_eta $ACTIVATIONS $ELAPSED)"
    echo "=============================="
}

if [ "$INTERVAL" -gt 0 ]; then
    while true; do
        clear
        show_progress
        sleep "$INTERVAL"
    done
else
    show_progress
fi
