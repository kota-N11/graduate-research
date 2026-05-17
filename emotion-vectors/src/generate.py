"""Step 1: Story generation.

Generates ~1000 stories using Gemma 2 2B-it on MPS.
Output: data/stories/{emotion}/{topic_idx}_{story_idx}.txt

Run:  uv run python src/generate.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent.parent
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data" / "stories"

MODEL_NAME = "google/gemma-2-2b-it"
N_STORIES_PER_TOPIC = 5
MAX_NEW_TOKENS = 2048

STORY_PROMPT = """\
Write {n} different stories based on the following premise.

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

The emotion should be clearly conveyed to the reader through these indirect means, but never explicitly named.\
"""


def load_config() -> tuple[list[str], list[str]]:
    emotions = [e["name"] for e in json.loads((CONFIG_DIR / "emotions.json").read_text())["emotions"]]
    topics = json.loads((CONFIG_DIR / "topics.json").read_text())["topics"]
    return emotions, topics


def split_stories(raw: str, n: int, min_chars: int = 150) -> list[str]:
    """Split model output into individual stories by common separator patterns."""
    import re
    parts = re.split(
        r"(?:\[story\s*\d+\]"         # [story N]
        r"|\*\*?#{1,3}\s+[^\n]+"      # **## Title or ## Title (bold+header)
        r"|#{2,3}\s+[^\n]+"           # ## Title or ### Title
        r"|\*\*Story\s*\d+\*\*"       # **Story N**
        r"|\*{3,}|---+)",             # *** or ---
        raw,
        flags=re.IGNORECASE,
    )
    stories = [p.strip() for p in parts if len(p.strip()) >= min_chars]
    return stories[:n]


def generate_stories(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    emotion: str,
    topic: str,
    n: int,
    device: torch.device,
) -> list[str]:
    prompt = STORY_PROMPT.format(n=n, topic=topic, emotion=emotion)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(generated, skip_special_tokens=True)
    return split_stories(raw, n)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--emotion-slice", type=str, default=None,
                        help="感情リストの範囲 例: '0:43'。省略時は全感情")
    args = parser.parse_args()

    os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS", "1")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    emotions, topics = load_config()
    if args.emotion_slice:
        start, end = (int(x) for x in args.emotion_slice.split(":"))
        emotions = emotions[start:end]

    print(f"Emotions: {len(emotions)}, Topics: {len(topics)}, Stories/topic: {N_STORIES_PER_TOPIC}")
    print(f"Total target: {len(emotions) * len(topics) * N_STORIES_PER_TOPIC} stories")

    print(f"\nLoading {MODEL_NAME}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map=str(device),
    )
    model.eval()
    print(f"Loaded in {time.time() - t0:.1f}s")

    total = len(emotions) * len(topics)
    done = 0
    skipped = 0
    t_start = time.time()

    for emotion in emotions:
        emo_dir = DATA_DIR / emotion
        emo_dir.mkdir(parents=True, exist_ok=True)

        for topic_idx, topic in enumerate(topics):
            out_path = emo_dir / f"topic{topic_idx:02d}.json"
            if out_path.exists():
                skipped += 1
                done += 1
                continue

            stories = generate_stories(model, tokenizer, emotion, topic, N_STORIES_PER_TOPIC, device)

            out_path.write_text(json.dumps({
                "emotion": emotion,
                "topic": topic,
                "topic_idx": topic_idx,
                "stories": stories,
            }, ensure_ascii=False, indent=2))

            done += 1
            elapsed = time.time() - t_start
            eta = elapsed / done * (total - done)
            print(f"[{done}/{total}] {emotion} / topic{topic_idx:02d} "
                  f"({len(stories)} stories) — ETA {eta/60:.1f}min")

    print(f"\nDone. {done - skipped} generated, {skipped} skipped (already existed).")
    print(f"Total time: {(time.time() - t_start)/60:.1f}min")


if __name__ == "__main__":
    main()
