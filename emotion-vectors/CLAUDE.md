# Emotion Vector Replication — 作業ガイド

このディレクトリは Anthropic の論文「Emotion Concepts and their Function in a Large Language Model」(2026) を **縮小規模で再現する**実装プロジェクト。

論文HTML: `../PreviousResearch/Emotion Concepts and their Function in a Large Language Model.html`
論文URL: https://transformer-circuits.pub/2026/emotions/index.html

## 当面のフェーズ目標

**Part 1 までを再現する。具体的には：**
1. **感情ストーリーの生成**（縮小スケール）
2. **モデル活性化の収集**（residual stream）
3. **感情ベクトルの計算**（平均差分法）
4. **妥当性の検証**（活性化が期待通りに発火するか）

→ ここまでで **「縮小モデルでも感情ベクトル/プローブが特定できる」**を示すのがフェーズゴール。
Part 2（幾何構造分析）と Part 3（steering で行動への因果性検証）はその後。

## ローカル環境の前提

- マシン: MacBook Pro M2 Pro / 32GB unified memory
- バックエンド: PyTorch **MPS**（CUDA 不可）
- 想定モデル: **Gemma 2 2B** または **Gemma 3 4B**（Geminiと同系譜なのが選定理由）
- ライブラリ: TransformerLens（フック・steering）、HuggingFace transformers（モデル）

---

## 論文の手法（Part 1）— 詳細仕様

### Step 1: 感情ストーリーの生成

**論文オリジナル**:
- 171個の感情語
- 100トピック × 12ストーリー × 171感情 = **約20万ストーリー**
- 生成モデル: Claude Sonnet 4.5

**生成プロンプト**（論文Appendix line 1746より、英文ママ）:

```
Write {n_stories} different stories based on the following premise.

Topic: {topic}

The story should follow a character who is feeling {emotion}.

Format the stories like so:

[story 1]

[story 2]

[story 3]

etc.

The paragraphs should each be a fresh start, with no continuity. Try to make them diverse and not use the same turns of phrase. Across the different stories, use a mix of third-person narration and first-person narration.

IMPORTANT: You must NEVER use the word '{emotion}' or any direct synonyms of it in the stories. Instead, convey the emotion ONLY through:

- The character's actions and behaviors
- Physical sensations and body language
- Dialogue and tone of voice
- Thoughts and internal reactions
- Situational context and environmental descriptions

The emotion should be clearly conveyed to the reader through these indirect means, but never explicitly named.
```

**重要な仕掛け**: 感情語そのもの・同義語を**禁止**することで、モデルが「行動・身体感覚・文脈」で感情を表現せざるを得なくする → ベクトルが単語表面ではなく**概念**を捉えるようにする。

### Step 2: 縮小スケール案（卒研用）

| パラメータ | 論文 | 縮小案 | 縮小比 |
|---|---|---|---|
| 感情数 | 171 | **20** | 1/8.5 |
| トピック数 | 100 | **10** | 1/10 |
| トピックあたりストーリー | 12 | **5** | 1/2.4 |
| **合計ストーリー** | **~205,200** | **1,000** | **約1/200** |

**選定する20感情の候補**（valence × arousal の網羅性を優先）:
- 高valence・高arousal: happy, excited, proud, thrilled
- 高valence・低arousal: calm, content, loving, peaceful
- 低valence・高arousal: angry, afraid, desperate, panicked
- 低valence・低arousal: sad, gloomy, lonely, depressed
- 中立/特殊: surprised, confused, nostalgic, guilty

**生成モデルの選択肢**:
- A) ローカルモデル（Gemma 2 2B）に書かせる → 自己一貫性、低コスト
- B) Claude/Gemini API に書かせる → 質が高い、お金かかる

**推奨**: A でやってみて、質が低ければ B に切り替え。

**100トピックリスト**（論文Appendix line 1542以降）から、研究テーマに合うものを10個選定する。リスト全文は論文HTML参照。

### Step 3: 活性化収集

各ストーリーをモデルに再入力して残差ストリーム活性化を取得する。

**論文の方法（line 177）**:
- residual stream activations を各層で取得
- **50トークン目以降**でトークン位置を平均（最初は感情がまだ表出されてないため除外）
- 同じ感情のストーリー群で平均
- **全感情の平均**を引いて差分を取る → emotion vector

**層の選択（line 185）**:
- 論文では「モデルの **約2/3 の深さ** の層」を使用
- Gemma 2 2B は26層 → **層17あたり**から始める
- Gemma 3 4B は34層 → **層22あたり**から始める

**TransformerLens の操作イメージ**:
```python
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gemma-2-2b")
_, cache = model.run_with_cache(story_text)

# 残差ストリーム取得（層17の post position）
activations = cache["blocks.17.hook_resid_post"]  # shape: [seq_len, d_model]

# 50トークン目以降を平均
mean_activation = activations[50:].mean(dim=0)  # shape: [d_model]
```

### Step 4: ノイズ除去（任意・後段で）

論文 line 179：
- 中立な対話データで活性化を取り、上位主成分（分散の50%を説明）を取得
- それらを感情ベクトルから projection で取り除く

**中立対話の生成プロンプト**（論文Appendix line 1780より）:
```
Write {n_stories} different dialogues based on the following topic.
[...]
CRITICAL REQUIREMENT: These dialogues must be completely neutral and emotionless.
- NO emotional content whatsoever
- The Person should not express any feelings
- The AI should not express any feelings
[...]
```

縮小版では最初は省略してもOK、結果が汚いなら追加する。

### Step 5: ベクトル計算

```python
import numpy as np

# emotion_activations: dict[str, list[np.ndarray]]
# 各感情ごとに、ストーリーから得た mean_activation のリスト

emotion_means = {e: np.stack(acts).mean(axis=0) for e, acts in emotion_activations.items()}
overall_mean = np.stack(list(emotion_means.values())).mean(axis=0)

emotion_vectors = {e: m - overall_mean for e, m in emotion_means.items()}
```

→ これが感情ベクトル。

### Step 6: 妥当性検証（3つの角度）

論文と同じ3つの方法で検証する：

#### 検証 A: 別データでの活性化（line 189）
- 別のテキスト（適当なストーリー、ニュース等）を流す
- 各トークン位置で `activation · desperate_vector` を計算
- 「絶望的な内容のトークン」で値が高くなることを確認

#### 検証 B: Logit lens（line 192-198）
- ベクトルを出力空間に投影：`logits = unembed(emotion_vector)`
- 上位/下位トークンを見る
- 例（論文より）:
  - desperate → "desperate", "urgent", "bankrupt" が上位
  - sad → "grief", "tears", "lonely"
  - calm → "leis", "relax", "thought", "enjoyed"

#### 検証 C: 暗黙的感情プロンプト（line 254）
- 感情語を含まない、感情を喚起するプロンプトで活性化を測る
- 例：
  - 「娘が初めて歩いた」→ happy, proud が活性化するはず
  - 「30年の結婚記念日」→ loving が活性化するはず
- Assistant: の `:` トークンでの活性化を測定する

---

## 実行コストの見積もり（M2 Pro / Gemma 2 2B）

| タスク | 時間目安 |
|---|---|
| 1,000ストーリー生成（Gemma 2Bで自己生成） | 2〜4時間 |
| 1,000ストーリーで活性化抽出 | 1〜2時間 |
| ベクトル計算 | 数分 |
| 検証 A・B・C | 30分〜1時間 |
| **合計** | **半日〜1日** |

**ローカル完結可能**。AWS は Part 2/Part 3 で必要になったら検討。

---

## ディレクトリ構造（予定）

```
replication/
├── CLAUDE.md            ← このファイル
├── README.md            ← 進捗・結果まとめ（後で）
├── config/
│   ├── emotions.json    ← 縮小版20感情リスト
│   └── topics.json      ← 縮小版10トピック
├── prompts/
│   ├── story_gen.txt    ← ストーリー生成プロンプト
│   └── neutral_gen.txt  ← 中立対話プロンプト
├── data/
│   ├── stories/         ← 生成ストーリー（gitignore）
│   ├── activations/     ← 活性化（gitignore、サイズ大）
│   └── vectors/         ← 計算後の感情ベクトル
├── src/
│   ├── generate.py      ← Step 1: ストーリー生成
│   ├── extract.py       ← Step 3: 活性化収集
│   ├── compute.py       ← Step 5: ベクトル計算
│   └── validate.py      ← Step 6: 妥当性検証
├── notebooks/
│   └── exploration.ipynb
└── results/
    ├── figures/
    └── logs/
```

---

## 注意事項

### モデル選定の根拠
- **Gemma 一択**の理由：
  - Gemini と同系譜（DeepMind 開発）→「Geminiの開放された兄弟」と論じられる
  - TransformerLens 対応
  - 2B/4B の選択肢あり、M2 Pro で動く
  - Gemma Scope（SAE）も Google から公開されている → 後段で SAE比較研究に拡張可能

### 既知の限界（論文 Discussion line 1385）
- **線形仮定の限界**: 論文自身が認めている。「複雑な感情の混合」「キャラクター固有の状態」は線形プローブで取れない可能性
- **off-policy データの偏り**: 合成ストーリーは、自然な会話と感情表現が違う可能性
- → これらは**卒研の Limitations セクションに書く** + **SAE比較で部分的に検証できる**

### 論文との違い（縮小に伴う妥協点）
- スケールが約1/200 → 統計的検出力は落ちる
- ノイズ除去（PCA projection）は最初は省略
- 検証はランダムサンプルベースで質的評価中心

### 関連メモ
- 卒研テーマの位置づけ: `../個人勉強/研究テーマまとめ.md`
- 論文の事前精査: 過去の会話で 171感情・blackmail 22%・64アクティビティ等の数値を検証済み
