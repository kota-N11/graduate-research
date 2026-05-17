"""Step 3: Emotion vector analysis.

Computes emotion vectors (mean-difference method) and evaluates:
  1. Cosine similarity matrix between all emotion vectors
  2. Correlation between vector distances and valence/arousal distances (from emotions.json)
  3. Nearest / farthest emotion pairs

Run:  uv run python src/analyze.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
ACT_DIR = ROOT / "data" / "activations"
CONFIG_DIR = ROOT / "config"


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def main() -> None:
    npy_files = sorted(ACT_DIR.glob("*.npy"))
    if not npy_files:
        raise SystemExit("No activation files found. Run extract.py first.")

    emotions = [f.stem for f in npy_files]
    acts = {e: np.load(f) for e, f in zip(emotions, npy_files)}

    print(f"Loaded activations for {len(emotions)} emotions")
    for e, a in acts.items():
        print(f"  {e:<15} shape={a.shape}")

    # 感情ベクトル = 各感情の平均 − 全感情の平均
    emo_means = {e: a.mean(axis=0) for e, a in acts.items()}
    global_mean = np.stack(list(emo_means.values())).mean(axis=0)
    emo_vecs = {e: emo_means[e] - global_mean for e in emotions}

    # コサイン類似度行列
    n = len(emotions)
    sim_matrix = np.zeros((n, n))
    for i, ei in enumerate(emotions):
        for j, ej in enumerate(emotions):
            sim_matrix[i, j] = cosine_sim(emo_vecs[ei], emo_vecs[ej])

    print("\n--- Cosine similarity matrix (emotion vectors) ---")
    header = f"{'':15}" + "".join(f"{e[:6]:>8}" for e in emotions)
    print(header)
    for i, ei in enumerate(emotions):
        row = f"{ei:<15}" + "".join(f"{sim_matrix[i,j]:8.3f}" for j in range(n))
        print(row)

    # 最も近い・遠いペア
    pairs = [(sim_matrix[i, j], emotions[i], emotions[j])
             for i in range(n) for j in range(i+1, n)]
    pairs.sort()

    print("\n--- Most similar pairs (top 5) ---")
    for sim, a, b in pairs[-5:][::-1]:
        print(f"  {a} ↔ {b}: {sim:.3f}")

    print("\n--- Most distant pairs (top 5) ---")
    for sim, a, b in pairs[:5]:
        print(f"  {a} ↔ {b}: {sim:.3f}")

    # emotions.jsonのvalence/arousalとの相関チェック
    emo_config = json.loads((CONFIG_DIR / "emotions.json").read_text())["emotions"]
    valence_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    arousal_map = {"high": 1.0, "low": -1.0}

    va_coords = {}
    for e in emo_config:
        if e["name"] in emotions:
            va_coords[e["name"]] = np.array([
                valence_map[e["valence"]],
                arousal_map[e["arousal"]],
            ])

    common = [e for e in emotions if e in va_coords]
    if len(common) > 3:
        va_dists, vec_sims = [], []
        for i, ei in enumerate(common):
            for j, ej in enumerate(common):
                if i >= j:
                    continue
                va_dist = np.linalg.norm(va_coords[ei] - va_coords[ej])
                vec_sim = cosine_sim(emo_vecs[ei], emo_vecs[ej])
                va_dists.append(va_dist)
                vec_sims.append(vec_sim)

        corr = float(np.corrcoef(va_dists, vec_sims)[0, 1])
        print(f"\n--- Valence/Arousal distance vs Vector cosine similarity ---")
        print(f"  Pearson r = {corr:.3f}  (n={len(va_dists)} pairs)")
        print(f"  Expected: negative r (distant in VA space → low cosine sim)")
        if corr < -0.3:
            print("  → 有意な負の相関あり。感情ベクトルはvalence×arousal構造を反映している")
        elif corr < 0:
            print("  → 弱い負の相関。部分的に構造を反映")
        else:
            print("  → 相関なし/正。ベクトルがvalence×arousal構造を反映していない可能性")


if __name__ == "__main__":
    main()
