# 卒業研究 進捗状態
<!-- Claude: セッション開始時に必ずこのファイルを読め。セッション終了前に必ず更新せよ。 -->

最終更新: 2026-05-21

## 現在のフェーズ

**emotion-vectors — Step 1: ストーリー生成（EC2で実行中）**

## 全体マップ

```
[実行中] Step 1: ストーリー生成  171感情 × 100トピック × 12ストーリー ≈ 205,200
                                 ETA 約15日（g4dn.xlarge / T4）
[ ] Step 3: 活性化抽出
[ ] Step 5: ベクトル計算
[ ] Step 6: 妥当性検証（A/B/C）
[ ] Part 2: 幾何構造分析
[ ] Part 3: steering で因果性検証
```

## 使用環境

- 実行環境: AWS EC2 g4dn.xlarge（NVIDIA T4 / CUDA）Auto Scaling Group 管理
- ローカル: MacBook Pro M2 Pro 32GB（コード編集・確認用）
- モデル: Gemma 2 2B-it（EC2 上でローカル生成）
- ライブラリ: TransformerLens, HuggingFace transformers
- パッケージ管理: uv（`uv run python -u src/generate.py`）
- 使用層: layer 17（全26層の約2/3）

## 今すぐやること（次のアクション）

生成完了まで待ちつつ並行して進める：
1. 活性化抽出・ベクトル計算・検証コードの整備
2. R勉強の続き

## 重要な決定事項

- **フルスケールで実施**（縮小なし）: 171 × 100 × 12 ≈ 205,200 ストーリー
- EC2 は Auto Scaling Group（ASG）管理 → スポット中断後に自動再起動
- ノイズ除去（PCA projection）は最初は省略
- generate.py: `**[N]**` セパレータ対応済み（2026-05-21 修正）
- extract.py: CUDA 対応済み（2026-05-21 修正）
- ログ確認: `tail -f ~/emotion-vectors/results/logs/generate.log`
- 12未満ファイル削除: `cd ~/emotion-vectors && python3 delete_incomplete.py`
- 不完全ファイル修復: `cd ~/emotion-vectors && python3 fix.py`

## ブロッカー

なし

## R勉強の進捗

ACF（自己相関係数）・自己共分散まで完了（0420.md参照）。
次章: 未確認（本を確認すること）
