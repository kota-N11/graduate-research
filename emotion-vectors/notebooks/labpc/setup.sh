#!/bin/bash
# 研究室PCセットアップ（初回のみ）
set -euo pipefail

echo "=== GPU確認 ==="
nvidia-smi || echo "警告: GPUが見つかりません"

echo ""
echo "=== vault をclone ==="
# vaultはPrivateリポジトリなのでGitHubのPersonal Access Tokenが必要
# https://github.com/settings/tokens でTokenを発行して以下を実行
# git clone https://{YOUR_TOKEN}@github.com/kota-N11/vault.git
echo "以下を実行してください（TOKENは自分のGitHub Personal Access Tokenに置き換え）:"
echo "  git clone https://TOKEN@github.com/kota-N11/vault.git"
echo ""
read -p "cloneが完了したらEnterを押してください..."

# ディレクトリ移動
PROJECT="$HOME/vault/school/5/eary/Graduate/emotion-vectors"
cd "$PROJECT"
mkdir -p logs data/stories

echo ""
echo "=== 依存関係インストール ==="
pip install 'transformers>=4.40' accelerate tqdm huggingface_hub

echo ""
echo "=== HuggingFace ログイン ==="
huggingface-cli login

echo ""
echo "セットアップ完了。次: bash notebooks/labpc/run.sh"
