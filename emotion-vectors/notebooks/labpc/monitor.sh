#!/bin/bash
# 生成進捗・推定終了時間を表示
# 使い方: bash notebooks/labpc/monitor.sh [間隔秒数（省略時は一回だけ表示）]

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
STORIES_DIR="$PROJECT_DIR/data/stories"
INTERVAL="${1:-0}"

# 開始時刻ファイル（初回実行時に記録）
START_FILE="$PROJECT_DIR/logs/start_time"
if [ ! -f "$START_FILE" ]; then
    date +%s > "$START_FILE"
fi
START_TS=$(cat "$START_FILE")

show_progress() {
    NOW=$(date +%s)
    ELAPSED=$((NOW - START_TS))

    # 完了ファイル数・目標数
    TOTAL_TARGET=$(python3 -c "
import json
e = len(json.load(open('$PROJECT_DIR/config/emotions.json'))['emotions'])
t = len(json.load(open('$PROJECT_DIR/config/topics.json'))['topics'])
print(e * t)
")
    DONE=$(find "$STORIES_DIR" -name '*.json' 2>/dev/null | wc -l | tr -d ' ')

    # 経過・推定残り時間
    if [ "$DONE" -gt 0 ]; then
        RATE=$(python3 -c "print(round($DONE / ($ELAPSED / 3600), 1))")  # 件/時間
        REMAINING=$(python3 -c "
remaining = ($TOTAL_TARGET - $DONE) / ($DONE / $ELAPSED) if $DONE > 0 else 0
h = int(remaining // 3600)
m = int((remaining % 3600) // 60)
print(f'{h}時間{m}分')
")
        ETA=$(python3 -c "
import datetime
eta_ts = $NOW + ($TOTAL_TARGET - $DONE) * ($ELAPSED / max($DONE, 1))
print(datetime.datetime.fromtimestamp(eta_ts).strftime('%m/%d %H:%M'))
")
    else
        RATE=0
        REMAINING="計算中..."
        ETA="計算中..."
    fi

    ELAPSED_H=$((ELAPSED / 3600))
    ELAPSED_M=$(((ELAPSED % 3600) / 60))
    PCT=$(python3 -c "print(round($DONE / $TOTAL_TARGET * 100, 1))")

    echo "=============================="
    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    echo "進捗    : $DONE / $TOTAL_TARGET 件 ($PCT%)"
    echo "経過    : ${ELAPSED_H}時間${ELAPSED_M}分"
    echo "速度    : ${RATE} 件/時間"
    echo "残り時間: $REMAINING"
    echo "推定完了: $ETA"
    echo "=============================="
}

if [ "$INTERVAL" -gt 0 ]; then
    # 定期更新モード
    while true; do
        clear
        show_progress
        sleep "$INTERVAL"
    done
else
    # 一回だけ表示
    show_progress
fi
