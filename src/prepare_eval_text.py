"""
prepare_eval_text.py

Downloads WikiText-2 (the standard text used for perplexity benchmarks) and
writes a fixed slice of it to data/eval_text.txt.

Run ONCE, then commit data/eval_text.txt to the repo. Every precision is scored
on that exact same file -- that's what makes the comparison controlled.

    python src/prepare_eval_text.py
"""

from pathlib import Path
from datasets import load_dataset

TARGET_CHARS = 20000    # ~5k tokens: enough for a stable estimate, quick to score

def main():
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # Keep only substantial lines (drop blanks and section headers like " = Title = ")
    lines = [t for t in ds["text"] if len(t.strip()) > 100 and not t.strip().startswith("=")]

    text = "\n\n".join(lines)[:TARGET_CHARS]

    Path("data").mkdir(exist_ok=True)
    out = Path("data/eval_text.txt")
    out.write_text(text, encoding="utf-8")
    print(f"Wrote {len(text)} chars -> {out}")
    print("Commit this file so the eval is reproducible.")

if __name__ == "__main__":
    main()
