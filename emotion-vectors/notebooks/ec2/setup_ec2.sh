#!/bin/bash
# EC2初回セットアップ（1回だけ実行）
set -euo pipefail

echo "=== EC2 セットアップ ==="

# 依存関係インストール
echo "[1/4] パッケージインストール..."
pip install -q 'transformers>=4.40' accelerate tqdm huggingface_hub

# vault をclone
echo "[2/4] コードをclone..."
VAULT_DIR="$HOME/vault"
if [ ! -d "$VAULT_DIR" ]; then
    git clone git@github.com:kota-N11/vault.git "$VAULT_DIR"
else
    echo "  既存のvaultを pull"
    git -C "$VAULT_DIR" pull
fi

# ディレクトリ準備
echo "[3/4] ディレクトリ準備..."
PROJECT="$VAULT_DIR/school/5/eary/Graduate/emotion-vectors"
mkdir -p "$PROJECT/logs" "$PROJECT/data/stories"

# HuggingFace ログイン
echo "[4/4] HuggingFace ログイン..."
huggingface-cli login

echo ""
echo "セットアップ完了。"
echo "次のコマンドで生成開始:"
echo "  tmux new -s generate"
echo "  bash $PROJECT/scripts/run_generate.sh"
