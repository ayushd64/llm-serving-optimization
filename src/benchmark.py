"""
benchmark.py  (upgraded: adds a "where did the time go?" breakdown)

Reads config, warms up, runs measured trials, summarizes with spread, then
reconciles OUR stopwatch against Ollama's own timing so TTFT is never a mystery
bucket again. Saves the raw data.

Run FROM THE PROJECT ROOT:
    python src/benchmark.py
"""

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import yaml

from client import run_single_request
from metrics import summarize


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def print_summary(label, s):
    print(label)
    print(f"  median {s['median']:7.2f} | mean {s['mean']:7.2f} | "
          f"min {s['min']:7.2f} | max {s['max']:7.2f} | "
          f"stdev {s['stdev']:6.2f}  (n={s['count']})")


def median_of(results, attr):
    """Median of one field across all trial results."""
    return statistics.median([getattr(r, attr) for r in results])


def save_results(config, results, summaries):
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_path = results_dir / f"{config['model'].replace(':', '-')}-{timestamp}.json"
    record = {
        "timestamp_utc": timestamp,
        "config": config,
        "summaries": summaries,
        "trials": [vars(r) for r in results],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"Raw results saved to: {out_path}")


def main():
    config = load_config()
    url, model, prompt = config["ollama_url"], config["model"], config["prompt"]
    options, think = config["options"], config["think"]
    warmup_runs, trial_runs = config["warmup_runs"], config["trial_runs"]

    print(f"Model  : {model}")
    print(f"Prompt : {prompt}")
    print(f"Warmup : {warmup_runs} runs (discarded) | Trials: {trial_runs} runs (measured)")
    print("-" * 64)

    print("Warming up...")
    for _ in range(warmup_runs):
        run_single_request(url, model, prompt, options, think)

    print("Measuring...")
    results = []
    for i in range(trial_runs):
        r = run_single_request(url, model, prompt, options, think)
        results.append(r)
        print(f"  trial {i + 1:>2}: {r.gen_tps:6.1f} tok/s | "
              f"TTFT {r.ttft_s:5.2f}s | prefill {r.prefill_s:5.2f}s | "
              f"{r.output_tokens} tok | load {r.load_s:.2f}s")

    summaries = {
        "gen_tps":  summarize([r.gen_tps for r in results]),
        "ttft_s":   summarize([r.ttft_s for r in results]),
        "total_s":  summarize([r.total_s for r in results]),
    }

    print("-" * 64)
    print_summary("Generation speed (tokens/sec):", summaries["gen_tps"])
    print_summary("Time to first token (seconds):", summaries["ttft_s"])
    print_summary("End-to-end time (seconds):",     summaries["total_s"])

    # --- "Where did the time go?" : our clock vs Ollama's own clock ---
    load_m    = median_of(results, "load_s")
    prefill_m = median_of(results, "prefill_s")
    gen_m     = median_of(results, "gen_s")
    ototal_m  = median_of(results, "ollama_total_s")
    wall_m    = median_of(results, "total_s")
    gap       = wall_m - ototal_m

    print("-" * 64)
    print("Where did the time go? (medians)")
    print(f"  Ollama load     : {load_m:6.2f}s")
    print(f"  Ollama prefill  : {prefill_m:6.2f}s")
    print(f"  Ollama generate : {gen_m:6.2f}s")
    print(f"  Ollama total    : {ototal_m:6.2f}s   <- everything OLLAMA times")
    print(f"  Our wall total  : {wall_m:6.2f}s   <- everything WE see")
    print(f"  Overhead gap    : {gap:6.2f}s   <- lives OUTSIDE Ollama's timing")

    max_think = max(r.thinking_chars for r in results)
    print(f"  Thinking check  : "
          f"{'OK (off)' if max_think == 0 else f'WARNING {max_think} chars'}")
    print("-" * 64)

    save_results(config, results, summaries)


if __name__ == "__main__":
    main()
