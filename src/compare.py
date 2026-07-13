"""
compare.py

We can't hold two 16-bit models in 8 GB at once, so each engine is benchmarked
separately (python src/benchmark.py <engine>). This reads each engine's most
recent saved result and prints them side by side.

Run from the project root:  python src/compare.py
"""

import json
from pathlib import Path


def load_latest(name):
    path = Path("results") / f"latest-{name}.json"
    return json.load(open(path, encoding="utf-8")) if path.exists() else None


def main():
    records = {n: load_latest(n) for n in ("ollama", "vllm")}
    records = {n: r for n, r in records.items() if r is not None}
    if not records:
        print("No saved results yet. Run:  python src/benchmark.py <engine>")
        return

    print("SIDE-BY-SIDE (medians across trials; each engine measured separately)\n")
    header = f"{'metric':<28}" + "".join(f"{n:>16}" for n in records)
    print(header)
    print("-" * len(header))
    rows = [("generation speed (tok/s)", "gen_tps", "median"),
            ("  spread (stdev tok/s)",   "gen_tps", "stdev"),
            ("time to first token (s)",  "ttft_s", "median"),
            ("end-to-end time (s)",      "total_s", "median")]
    for label, metric, stat in rows:
        line = f"{label:<28}"
        for n in records:
            line += f"{records[n]['summary'][metric][stat]:>16.2f}"
        print(line)

    print("\nmeasured at:")
    for n, rec in records.items():
        print(f"  {n:<8} {rec['timestamp_utc']}  model={rec['model']}")


if __name__ == "__main__":
    main()
