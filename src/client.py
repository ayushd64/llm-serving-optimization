"""
client.py  (upgraded: now measures WHERE the time goes)

One job: send ONE request to Ollama and bring back a full timing breakdown --
separating loading, prefill (reading the prompt), and generation, alongside our
own wall-clock view. Comparing those two views is how we hunt hidden costs.
"""

import json
import time
from dataclasses import dataclass

import requests

# A shared session: reuses the TCP connection between requests (a little faster)
# and ignores any system HTTP proxy, so nothing on the machine can silently
# distort our measurements. Reproducibility hygiene for a benchmark tool.
_session = requests.Session()
_session.trust_env = False


NANOSECONDS_PER_SECOND = 1_000_000_000


@dataclass
class RequestResult:
    # --- our wall-clock stopwatch (what a user actually feels) ---
    ttft_s: float          # send -> first visible answer token
    total_s: float         # send -> whole answer finished

    # --- Ollama's OWN reported breakdown (converted ns -> seconds) ---
    load_s: float          # loading the model (~0 once warm)
    prefill_s: float       # reading/processing the prompt
    gen_s: float           # generating the answer tokens
    ollama_total_s: float  # Ollama's own total for the whole job

    # --- counts and speed ---
    prompt_tokens: int
    output_tokens: int
    gen_tps: float         # output_tokens / gen_s  (true generation speed)

    # --- sanity checks ---
    thinking_chars: int    # >0 means the model secretly "thought" (should be 0)
    response_text: str


def run_single_request(ollama_url, model, prompt, options, think):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "think": think,
        "options": options,
    }

    start = time.perf_counter()
    first_response_time = None
    response_text = ""
    thinking_text = ""
    final_chunk = {}

    with _session.post(ollama_url, json=payload, stream=True) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)

            # capture any hidden "thinking" text separately (should stay empty)
            thinking_text += chunk.get("thinking") or ""

            piece = chunk.get("response", "")
            if piece:
                if first_response_time is None:
                    first_response_time = time.perf_counter()   # first VISIBLE token
                response_text += piece

            if chunk.get("done"):
                final_chunk = chunk          # last packet carries all the numbers
                break

    end = time.perf_counter()

    ns = NANOSECONDS_PER_SECOND
    output_tokens = final_chunk.get("eval_count", 0)
    gen_s = final_chunk.get("eval_duration", 0) / ns

    return RequestResult(
        ttft_s=(first_response_time - start) if first_response_time else 0.0,
        total_s=end - start,
        load_s=final_chunk.get("load_duration", 0) / ns,
        prefill_s=final_chunk.get("prompt_eval_duration", 0) / ns,
        gen_s=gen_s,
        ollama_total_s=final_chunk.get("total_duration", 0) / ns,
        prompt_tokens=final_chunk.get("prompt_eval_count", 0),
        output_tokens=output_tokens,
        gen_tps=(output_tokens / gen_s) if gen_s > 0 else 0.0,
        thinking_chars=len(thinking_text),
        response_text=response_text,
    )
