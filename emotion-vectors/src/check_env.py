"""Environment sanity check.

Verifies:
  1. PyTorch MPS backend is available
  2. Gemma 2 2B-it loads via TransformerLens on MPS
  3. run_with_cache returns the expected residual stream shape
  4. Reports forward-pass latency and process RSS

Run:  uv run python src/check_env.py
"""
from __future__ import annotations

import os
import time

import psutil
import torch

MODEL_NAME = "gemma-2-2b-it"
TARGET_LAYER = 17  # ~2/3 depth of 26 layers (per CLAUDE.md)
DUMMY_TEXT = (
    "She sat alone in the empty kitchen, staring at the cold cup of tea, "
    "her hands trembling as the silence pressed in from every wall."
)


def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2


def main() -> None:
    print("=" * 60)
    print("Step 1: PyTorch / MPS")
    print("-" * 60)
    print(f"torch version       : {torch.__version__}")
    print(f"MPS available       : {torch.backends.mps.is_available()}")
    print(f"MPS built           : {torch.backends.mps.is_built()}")
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable — abort.")

    print()
    print("=" * 60)
    print(f"Step 2: Loading {MODEL_NAME}")
    print("-" * 60)
    rss_before = rss_mb()
    t0 = time.time()

    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        device="mps",
        dtype=torch.bfloat16,
    )
    load_time = time.time() - t0
    rss_after = rss_mb()
    print(f"load time           : {load_time:.1f}s")
    print(f"RSS delta           : {rss_after - rss_before:.0f} MB "
          f"(before {rss_before:.0f} → after {rss_after:.0f})")
    print(f"n_layers            : {model.cfg.n_layers}")
    print(f"d_model             : {model.cfg.d_model}")

    print()
    print("=" * 60)
    print("Step 3: Forward pass + cache")
    print("-" * 60)
    tokens = model.to_tokens(DUMMY_TEXT)
    print(f"input tokens shape  : {tuple(tokens.shape)}")

    t0 = time.time()
    _, cache = model.run_with_cache(tokens)
    fwd_time = time.time() - t0

    hook_name = f"blocks.{TARGET_LAYER}.hook_resid_post"
    acts = cache[hook_name]
    print(f"forward time        : {fwd_time:.2f}s")
    print(f"{hook_name} : {tuple(acts.shape)}  dtype={acts.dtype}")

    mean_act = acts[0, 50:].float().mean(dim=0) if acts.shape[1] > 50 else acts[0].float().mean(dim=0)
    print(f"mean activation L2  : {mean_act.norm().item():.3f}")

    print()
    print("=" * 60)
    print("OK — environment ready.")


if __name__ == "__main__":
    main()
