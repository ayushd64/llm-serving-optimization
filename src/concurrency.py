"""
concurrency.py  (Milestone 5)

Hammer ONE engine with many simultaneous requests and watch how it copes as the
load rises -- a "concurrency sweep". For each level C (1, 2, 4, ...), we keep C
requests in flight at once, fire a fixed number, and measure:
  - throughput : requests/sec and output-tokens/sec (total work done)
  - latency    : p50 / p95 / p99 of end-to-end time (the tail users feel)

Each engine is swept SEPARATELY (8 GB can't hold two models), then compared.

Run from the project root:
    python src/concurrency.py vllm      # sweep vLLM
    python src/concurrency.py ollama    # sweep Ollama
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import yaml

from client import make_client
from metrics import percentile


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_at_concurrency(client, prompt, temp, max_tokens, concurrency, num_requests):
    """Fire num_requests total, keeping `concurrency` in flight at once."""
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(
            lambda _: client.run(prompt, temp, max_tokens), range(num_requests)))
    wall = time.perf_counter() - start

    latencies = [r.total_s for r in results]
    total_out = sum(r.output_tokens for r in results)
    p50 = percentile(latencies, 50)
    rps = num_requests / wall if wall > 0 else 0.0
    implied_concurrency = rps * p50          # Little's Law: C = throughput x latency
    return {
        "concurrency": concurrency,
        "requests": num_requests,
        "wall_s": wall,
        "throughput_rps": num_requests / wall if wall > 0 else 0.0,
        "throughput_tps": total_out / wall if wall > 0 else 0.0,
        "latency_p50": percentile(latencies, 50),
        "latency_p95": percentile(latencies, 95),
        "latency_p99": percentile(latencies, 99),
        "implied_concurrency": implied_concurrency,
        "concurrency_achieved_pct": (implied_concurrency / concurrency * 100) if concurrency else 0.0,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/concurrency.py <engine>   (ollama | vllm)")
        return
    wanted = sys.argv[1]

    config = load_config()
    engine_cfg = next((e for e in config["engines"] if e["name"] == wanted), None)
    if engine_cfg is None:
        print(f"No engine '{wanted}' in config.")
        return

    prompt = config["prompt"]
    temp = config["temperature"]
    max_tokens = config["max_tokens"]
    levels = config["concurrency_levels"]
    # num_requests = config["requests_per_level"]
    configured_requests = config["requests_per_level"]

    client = make_client(engine_cfg)
    print(f"Concurrency sweep: {wanted}  (model: {engine_cfg['model']})")
    print(f"Levels: {levels} | {configured_requests} requests each | temperature {temp}")
    print("=" * 70)

    print("Warming up...")
    for _ in range(config.get("warmup_requests", 5)):
        client.run(prompt, temp, max_tokens)

    # print(f"\n{'conc':>4} {'req/s':>8} {'tok/s':>9} {'p50':>8} {'p95':>8} {'p99':>8}")
    print(f"\n{'conc':>4} {'sent':>6} {'req/s':>8} {'tok/s':>9} {'p50':>8} {'p95':>8} {'p99':>8} {'real C':>8}")

    print("-" * 70)
    rows = []
    for c in levels:
        # row = run_at_concurrency(client, prompt, temp, max_tokens, c, num_requests)
        # Each level must send far more requests than its concurrency, or the
        # pool starves and the real concurrency silently caps at num_requests.
        n = configured_requests if configured_requests > 0 else max(60, c * 10)
        row = run_at_concurrency(client, prompt, temp, max_tokens, c, n)
        rows.append(row)
        print(f"{c:>4} {n:>6} {row['throughput_rps']:>8.1f} {row['throughput_tps']:>9.1f} "
              f"{row['latency_p50']:>7.2f}s {row['latency_p95']:>7.2f}s "
              f"{row['latency_p99']:>7.2f}s {row['implied_concurrency']:>8.1f}")



    Path("results").mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    record = {"engine": wanted, "model": engine_cfg["model"], "timestamp_utc": ts,
              "settings": {"temperature": temp, "max_tokens": max_tokens,
                           "requests_per_level": configured_requests}, "sweep": rows}
    for path in (Path("results") / f"sweep-{wanted}.json",
                 Path("results") / f"sweep-{wanted}-{ts}.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
    print("-" * 70)
    print(f"Saved sweep -> results/sweep-{wanted}.json")


if __name__ == "__main__":
    main()
