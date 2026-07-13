"""
benchmark.py

Benchmarks EACH configured engine (warmup + trials) and prints them side by side.
An engine whose server isn't running is skipped with a warning, so you can run
with one or both servers up.

Run from the project root:  python src/benchmark.py
"""

import json
import sys
import statistics
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

from client import make_client
from metrics import summarize



def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def benchmark_engine(client, prompt, temperature, max_tokens, warmup_runs, trial_runs):
    for _ in range(warmup_runs):                 # warm up (discarded)
        client.run(prompt, temperature, max_tokens)
    results = []
    for i in range(trial_runs):                  # measured trials
        r = client.run(prompt, temperature, max_tokens)
        results.append(r)
        print(f"  trial {i+1:>2}: {r.gen_tps:6.1f} tok/s | TTFT {r.ttft_s:5.2f}s | "
              f"{r.output_tokens} out tok | {r.prompt_tokens} prompt tok")
    return results


def summarize_results(results):
    return {"gen_tps": summarize([r.gen_tps for r in results]),
            "ttft_s":  summarize([r.ttft_s for r in results]),
            "total_s": summarize([r.total_s for r in results])}


def print_comparison(all_summaries):
    if not all_summaries:
        print("No engines benchmarked -- are the servers running?")
        return
    print("SIDE-BY-SIDE (medians across trials)")
    header = f"{'metric':<28}" + "".join(f"{n:>16}" for n in all_summaries)
    print(header)
    print("-" * len(header))
    rows = [("generation speed (tok/s)", "gen_tps", "median"),
            ("  spread (stdev tok/s)",   "gen_tps", "stdev"),
            ("time to first token (s)",  "ttft_s", "median"),
            ("end-to-end time (s)",      "total_s", "median")]
    for label, metric, stat in rows:
        line = f"{label:<28}"
        for n in all_summaries:
            line += f"{all_summaries[n]['summary'][metric][stat]:>16.2f}"
        print(line)



def save_engine_result(name, model, config, results):
    """Save one engine's run to a stable 'latest-<engine>.json' plus a timestamped copy."""
    Path("results").mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    record = {
        "engine": name, "model": model, "timestamp_utc": ts,
        "prompt": config["prompt"],
        "settings": {k: config[k] for k in
                     ("temperature", "max_tokens", "warmup_runs", "trial_runs")},
        "summary": summarize_results(results),
        "trials": [vars(r) for r in results],
        "sample_response": results[0].response_text[:200],
    }
    for path in (Path("results") / f"latest-{name}.json",
                 Path("results") / f"{name}-{ts}.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
    print(f"\nSaved {name} -> results/latest-{name}.json")


def main():
    config = load_config()
    prompt, temp = config["prompt"], config["temperature"]
    max_tokens = config["max_tokens"]
    warmup_runs, trial_runs = config["warmup_runs"], config["trial_runs"]

    # Run ONE engine per invocation (we can't fit both in 8 GB at once).
    wanted = sys.argv[1] if len(sys.argv) > 1 else None
    engines = [e for e in config["engines"] if wanted is None or e["name"] == wanted]
    if not engines:
        names = [e["name"] for e in config["engines"]]
        print(f"Usage: python src/benchmark.py <engine>   options: {names}")
        return

    for cfg in engines:
        name = cfg["name"]
        print(f"### {name}  (model: {cfg['model']})")
        client = make_client(cfg)
        try:
            results = benchmark_engine(client, prompt, temp, max_tokens,
                                       warmup_runs, trial_runs)
        except requests.exceptions.RequestException as e:
            print(f"  !! cannot reach {name} ({e.__class__.__name__}). Is it running?")
            continue
        save_engine_result(name, cfg["model"], config, results)




if __name__ == "__main__":
    main()
