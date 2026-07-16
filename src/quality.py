"""
quality.py  (Milestone 4, Phase 2)

Does quantization make the model WORSE? We measure PERPLEXITY: feed the model a
fixed passage and, at every token, ask what probability it assigned to the word
that actually came next. Low perplexity = confident and correct.

Perplexity is the right tool here because it is continuous (every token gives
signal), deterministic (no sampling luck), and needs no generation -- just
prefill. A task eval would need thousands of questions to detect a small drop.

Run ONE precision at a time (only one vLLM server fits in 8 GB):
    python src/quality.py vllm        # fp16
    python src/quality.py vllm-int8
    python src/quality.py vllm-int4
"""

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def chunk_text(text, chunk_chars=2000):
    """Split the passage into fixed-size chunks so each fits the context window.
    Chunking by characters (not tokens) keeps this simple; the SAME chunks are
    used for every precision, which is all the comparison requires."""
    return [text[i:i + chunk_chars] for i in range(0, len(text), chunk_chars)]


def score_chunk(url, model, chunk):
    """Ask vLLM for the log-probability of every token in this chunk."""
    payload = {
        "model": model,
        "prompt": chunk,
        "max_tokens": 1,        # we want scoring, not generation (1 is the minimum)
        "temperature": 0,
        "prompt_logprobs": 0,   # 0 = return the actual token's logprob, no alternatives
        "echo": False,
    }
    session = requests.Session()
    session.trust_env = False
    response = session.post(url, json=payload, timeout=300)
    response.raise_for_status()
    data = response.json()

    entries = data["choices"][0]["prompt_logprobs"]
    logprobs = []
    for entry in entries:
        if entry is None:
            continue            # the first token has no preceding context to predict it
        (info,) = entry.values()   # prompt_logprobs=0 -> exactly one entry per position
        logprobs.append(info["logprob"])
    return logprobs


def perplexity(logprobs):
    """perplexity = exp( - average log-probability per token )"""
    if not logprobs:
        return float("nan")
    return math.exp(-sum(logprobs) / len(logprobs))


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/quality.py <engine>   (vllm | vllm-int8 | vllm-int4)")
        return
    wanted = sys.argv[1]

    config = load_config()
    engine_cfg = next((e for e in config["engines"] if e["name"] == wanted), None)
    if engine_cfg is None:
        print(f"No engine '{wanted}' in config.")
        return

    # prompt_logprobs lives on the /v1/completions endpoint, not the chat one.
    url = engine_cfg["url"].replace("/v1/chat/completions", "/v1/completions")
    model = engine_cfg["model"]

    text = Path("data/eval_text.txt").read_text(encoding="utf-8")
    chunks = chunk_text(text)
    print(f"Perplexity eval: {wanted}  (model: {model})")
    print(f"Text: {len(text)} chars -> {len(chunks)} chunks")
    print("-" * 60)

    all_logprobs = []
    for i, chunk in enumerate(chunks, 1):
        lps = score_chunk(url, model, chunk)
        all_logprobs.extend(lps)
        print(f"  chunk {i:>2}: {len(lps):>5} tokens | chunk ppl {perplexity(lps):8.3f}")

    ppl = perplexity(all_logprobs)
    print("-" * 60)
    print(f"TOKENS SCORED : {len(all_logprobs)}")
    print(f"PERPLEXITY    : {ppl:.4f}")

    Path("results").mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    record = {"engine": wanted, "model": model, "timestamp_utc": ts,
              "eval_text_chars": len(text), "chunks": len(chunks),
              "tokens_scored": len(all_logprobs), "perplexity": ppl}
    with open(Path("results") / f"quality-{wanted}.json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"Saved -> results/quality-{wanted}.json")


if __name__ == "__main__":
    main()
