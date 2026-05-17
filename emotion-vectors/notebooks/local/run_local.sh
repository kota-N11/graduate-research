#!/bin/bash
# ローカル（Mac M2 Pro / MPS）でストーリー生成を実行
# 使い方: bash notebooks/local/run_local.sh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_DIR"

echo "Device: MPS (Mac M2 Pro)"
echo "$(python3 -c "import json; d=json.load(open('config/emotions.json')); print(f'感情数: {len(d[\"emotions\"])}')")"
echo ""

uv run python src/generate.py "$@"
