# 研究室PC GPU使用申請

## 申請者
中島 航太（大枝研究室）

## 目的
卒業研究「大規模言語モデルにおける感情概念の表現に関する研究」の実験データ生成。

Anthropicの論文「Emotion Concepts and their Function in a Large Language Model」(2026) を再現するため、感情ストーリーの大規模生成が必要。

## 使用内容

### 生成するデータ
- 171感情 × 100トピック × 12ストーリー = **約205,200件**のテキストデータ
- 生成モデル: Gemma 2 2B-it（Google DeepMind、オープンソース）

### インストールするソフトウェア
Dockerコンテナ内に閉じるため、**ホストOSには何もインストールしない**。

| ライブラリ | 用途 |
|---|---|
| PyTorch 2.3 | 深層学習フレームワーク |
| Transformers | Gemmaモデルの実行 |
| Accelerate | GPU最適化 |
| HuggingFace Hub | モデルダウンロード |

### ダウンロードするデータ
| データ | サイズ | 保存先 |
|---|---|---|
| Gemma 2 2B モデル | 約5GB | Dockerキャッシュ or 指定ディレクトリ |
| 生成ストーリー | 約2GB | 指定ディレクトリ |

## セキュリティ対策

- **Docker コンテナで完全隔離**（ホストOSのPython環境・システムには一切影響なし）
- HuggingFaceのアクセストークンは環境変数で渡し、ファイルには保存しない
- 使用終了後はコンテナ・イメージを完全削除

```bash
# 使用後のクリーンアップ
docker stop $(docker ps -q --filter name=emotion-vectors)
docker rm $(docker ps -aq --filter name=emotion-vectors)
docker rmi emotion-vectors
```

## GPU使用計画

| 項目 | 内容 |
|---|---|
| 使用期間 | 約1〜2週間（連続稼働） |
| 使用GPU | 搭載GPU全台 |
| 使用時間帯 | 他の利用者がいない時間帯を優先 |
| 中断への対応 | チェックポイント実装済み、随時中断・再開可能 |

## 実行方法

 # まず動作確認（数分で終わる）
  bash notebooks/labpc/test_docker.sh

  # 個別実行
  bash notebooks/labpc/run_docker.sh generate  # 生成のみ
  bash notebooks/labpc/run_docker.sh extract   # 抽出のみ

時間測定
  bash notebooks/labpc/monitor.sh 30

コピー
rsync -av --exclude='.venv'　pcのパス　usbのパス