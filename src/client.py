"""
client.py

Talks to serving engines. Each engine speaks a different dialect (Ollama has its
own API; vLLM speaks the OpenAI format), so we define ONE interface --
run(prompt) -> RequestResult -- and write a small ADAPTER per engine. The
benchmark never needs to know which engine it's using.

Fairness rule: we time BOTH engines with OUR OWN wall-clock stopwatch, not each
engine's self-reported timings, so both are measured with the same ruler.
"""

import json
import time
from dataclasses import dataclass

import requests


def _make_session():
    """A session that ignores the system proxy and allows many parallel connections."""
    session = requests.Session()
    session.trust_env = False
    adapter = requests.adapters.HTTPAdapter(pool_connections=128, pool_maxsize=128)
    session.mount("http://", adapter)
    return session


@dataclass
class RequestResult:
    ttft_s: float          # wall-clock: send -> first token
    total_s: float         # wall-clock: send -> done
    gen_tps: float         # decode speed: tokens-after-first / time-after-first
    prompt_tokens: int     # engine-reported prompt token count
    output_tokens: int     # engine-reported generated token count
    response_text: str


def _decode_speed(output_tokens, ttft_s, total_s):
    """Tokens generated AFTER the first, over the time spent after the first token.
    Computed identically for every engine, so comparisons are apples-to-apples."""
    window = total_s - ttft_s
    return (max(output_tokens - 1, 0) / window) if window > 0 else 0.0


class EngineClient:
    """The interface every engine adapter must implement."""
    def run(self, prompt, temperature, max_tokens) -> RequestResult:
        raise NotImplementedError


class OllamaClient(EngineClient):
    """Adapter for Ollama's native /api/generate streaming format."""
    def __init__(self, url, model):
        self.url, self.model = url, model
        self.session = _make_session()         # ignore system proxy (our earlier fix)

    def run(self, prompt, temperature, max_tokens):
        payload = {
            "model": self.model, "prompt": prompt, "stream": True, "think": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        start = time.perf_counter()
        first, text, final = None, "", {}
        with self.session.post(self.url, json=payload, stream=True) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                piece = chunk.get("response", "")
                if piece:
                    if first is None:
                        first = time.perf_counter()
                    text += piece
                if chunk.get("done"):
                    final = chunk
                    break
        end = time.perf_counter()
        ttft = (first - start) if first else 0.0
        out_tok = final.get("eval_count", 0)
        return RequestResult(ttft, end - start, _decode_speed(out_tok, ttft, end - start),
                             final.get("prompt_eval_count", 0), out_tok, text)


class VLLMClient(EngineClient):
    """Adapter for vLLM's OpenAI-format /v1/chat/completions streaming."""
    def __init__(self, url, model):
        self.url, self.model = url, model
        self.session = requests.Session()
        self.session.trust_env = False

    def run(self, prompt, temperature, max_tokens):
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature, "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},   # ask vLLM to report token counts
        }
        start = time.perf_counter()
        first, text, usage = None, "", {}
        with self.session.post(self.url, json=payload, stream=True) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                # vLLM streams "Server-Sent Events": each line is  data: {...json...}
                # and the stream ends with  data: [DONE]
                line = line.decode("utf-8") if isinstance(line, bytes) else line
                if not line.startswith("data:"):
                    continue
                payload_str = line[len("data:"):].strip()
                if payload_str == "[DONE]":
                    break
                chunk = json.loads(payload_str)
                if chunk.get("usage"):                    # final chunk carries token counts
                    usage = chunk["usage"]
                choices = chunk.get("choices", [])
                if choices:
                    piece = choices[0].get("delta", {}).get("content") or ""
                    if piece:
                        if first is None:
                            first = time.perf_counter()
                        text += piece
        end = time.perf_counter()
        ttft = (first - start) if first else 0.0
        out_tok = usage.get("completion_tokens", 0)
        return RequestResult(ttft, end - start, _decode_speed(out_tok, ttft, end - start),
                             usage.get("prompt_tokens", 0), out_tok, text)


def make_client(engine_config):
    """Factory: build the right adapter from a config entry."""
    name = engine_config["name"]
    if name == "ollama":
        return OllamaClient(engine_config["url"], engine_config["model"])
    if name == "vllm":
        return VLLMClient(engine_config["url"], engine_config["model"])
    raise ValueError(f"Unknown engine: {name}")

