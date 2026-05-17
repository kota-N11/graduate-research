"""Step 2: Activation extraction.

For each story, extracts the residual stream at TARGET_LAYER via TransformerLens,
mean-pools over token positions, and saves per-emotion activation matrices.

Output: data/activations/{emotion}.npy  shape=[n_stories, d_model]

Run:  uv run python src/extract.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from transformer_lens import HookedTransformer

ROOT = Path(__file__).parent.parent
STORIES_DIR = ROOT / "data" / "stories"
OUT_DIR = ROOT / "data" / "activations"

MODEL_NAME = "gemma-2-2b-it"
TARGET_LAYER = 17
MAX_TOKENS = 512  # truncate long stories to limit VRAM


def extract_activation(model: HookedTransformer, text: str, device: torch.device) -> np.ndarray:
    tokens = model.to_tokens(text, truncate=True)
    if tokens.shape[1] > MAX_TOKENS:
        tokens = tokens[:, :MAX_TOKENS]

    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=f"blocks.{TARGET_LAYER}.hook_resid_post",
        )

    acts = cache[f"blocks.{TARGET_LAYER}.hook_resid_post"]  # [1, seq_len, d_model]
    mean_act = acts[0].float().mean(dim=0)  # [d_model]
    return mean_act.cpu().numpy()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading {MODEL_NAME}...")
    t0 = time.time()
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=str(device), dtype=torch.bfloat16)
    model.eval()
    print(f"Loaded in {time.time() - t0:.1f}s")

    emotions = sorted([d.name for d in STORIES_DIR.iterdir() if d.is_dir()])
    print(f"Emotions: {len(emotions)}")

    total_stories = 0
    t_start = time.time()

    for emo_idx, emotion in enumerate(emotions):
        out_path = OUT_DIR / f"{emotion}.npy"
        if out_path.exists():
            print(f"[{emo_idx+1}/{len(emotions)}] {emotion} — skipped (exists)")
            continue

        emo_dir = STORIES_DIR / emotion
        topic_files = sorted(emo_dir.glob("topic*.json"))

        acts_list = []
        for topic_file in topic_files:
            data = json.loads(topic_file.read_text())
            for story in data["stories"]:
                act = extract_activation(model, story, device)
                acts_list.append(act)

        acts_array = np.stack(acts_list)  # [n_stories, d_model]
        np.save(out_path, acts_array)

        total_stories += len(acts_list)
        elapsed = time.time() - t_start
        remaining = len(emotions) - emo_idx - 1
        eta = (elapsed / (emo_idx + 1)) * remaining
        print(f"[{emo_idx+1}/{len(emotions)}] {emotion}: {len(acts_list)} stories, "
              f"shape={acts_array.shape} — ETA {eta/60:.1f}min")

    print(f"\nDone. {total_stories} activations extracted.")
    print(f"Total time: {(time.time() - t_start)/60:.1f}min")


if __name__ == "__main__":
    main()
