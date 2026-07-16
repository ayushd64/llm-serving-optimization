"""
plots.py

Regenerates every figure in the README from the RAW result files in results/.
Nothing here is hand-typed: change the data, re-run, the plots update.

    python src/plots.py

Reads:  results/sweep-*.json, results/quality-*.json, manifest.json
Writes: plots/*.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # render to file, no interactive window needed
import matplotlib.pyplot as plt

RESULTS = Path("results")
PLOTS = Path("plots")
BLUE, GREEN, GRAY, RED, AMBER = "#2a78d6", "#00a67d", "#8a8a8a", "#e34948", "#eda100"


def load(name):
    """Load a results file, or return None if that experiment wasn't run."""
    path = RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def series(record, field):
    """Pull (concurrency, field) pairs out of a sweep record."""
    xs = [r["concurrency"] for r in record["sweep"]]
    ys = [r[field] for r in record["sweep"]]
    return xs, ys


def plot_throughput():
    """Headline: throughput vs concurrency, every engine/config on one axis."""
    specs = [("sweep-vllm.json", "vLLM fp16", BLUE, "-", "o"),
             ("sweep-vllm-int4.json", "vLLM int4 (AWQ)", GREEN, "-", "s"),
             ("sweep-ollama-p16.json", "Ollama NUM_PARALLEL=16", AMBER, "--", "^"),
             ("sweep-ollama-p8.json", "Ollama NUM_PARALLEL=8", GRAY, "--", "v"),
             ("sweep-ollama-p32.json", "Ollama NUM_PARALLEL=32 (VRAM spill)", RED, ":", "x")]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for fname, label, color, ls, marker in specs:
        rec = load(fname)
        if rec is None:
            continue
        xs, ys = series(rec, "throughput_tps")
        ax.plot(xs, ys, label=label, color=color, linestyle=ls, marker=marker, lw=2)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("concurrency (requests in flight)")
    ax.set_ylabel("output throughput (tokens/sec)")
    ax.set_title("Throughput vs concurrency — RTX 3070 (8 GB, 45 W), Qwen2.5-1.5B-Instruct")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOTS / "throughput_vs_concurrency.png", dpi=150)
    plt.close(fig)


def plot_latency():
    """The other half of the story: what the tail latency costs."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for fname, label, color in [("sweep-vllm.json", "vLLM fp16", BLUE),
                                ("sweep-vllm-int4.json", "vLLM int4", GREEN),
                                ("sweep-ollama-p16.json", "Ollama P=16", AMBER)]:
        rec = load(fname)
        if rec is None:
            continue
        xs, p50 = series(rec, "latency_p50")
        _, p99 = series(rec, "latency_p99")
        ax.plot(xs, p50, label=f"{label} p50", color=color, marker="o", lw=2)
        ax.plot(xs, p99, label=f"{label} p99", color=color, marker="x", ls="--", lw=1, alpha=0.7)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("concurrency (requests in flight)")
    ax.set_ylabel("end-to-end latency (seconds)")
    ax.set_title("Latency vs concurrency (p50 solid, p99 dashed)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(PLOTS / "latency_vs_concurrency.png", dpi=150)
    plt.close(fig)


def plot_quantization():
    """Memory / speed / quality across precisions -- the whole trade-off in one figure."""
    man = json.loads(Path("manifest.json").read_text(encoding="utf-8"))
    labels, keys = ["FP16", "INT8\n(GPTQ)", "INT4\n(AWQ)"], ["fp16", "int8", "int4"]
    names = ["vllm", "vllm-int8", "vllm-int4"]

    weights = [man["vllm_memory"][k]["weights_gib"] for k in keys]
    cache = [man["vllm_memory"][k]["kv_cache_tokens"] / 1000 for k in keys]
    speed = [man["single_request_tok_s"][k] for k in keys]
    ppl = [load(f"quality-{n}.json")["perplexity"] for n in names]

    fig, axes = plt.subplots(1, 4, figsize=(13, 4))
    for ax, vals, title, ylab, color in [
            (axes[0], weights, "Weights (smaller better)", "GiB", BLUE),
            (axes[1], cache, "KV cache (bigger better)", "thousand tokens", GREEN),
            (axes[2], speed, "Speed @ C=1 (bigger better)", "tokens/sec", AMBER),
            (axes[3], ppl, "Perplexity (smaller better)", "perplexity", RED)]:
        bars = ax.bar(labels, vals, color=color, alpha=0.85)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylab, fontsize=9)
        ax.grid(alpha=0.3, axis="y")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}" if v < 100 else f"{v:.0f}",
                    ha="center", va="bottom", fontsize=8)
    axes[3].set_ylim(min(ppl) * 0.95, max(ppl) * 1.05)   # zoom: the deltas are small
    fig.suptitle("Quantization trade-off — Qwen2.5-1.5B-Instruct on RTX 3070", fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOTS / "quantization_tradeoff.png", dpi=150)
    plt.close(fig)


def plot_int4_decay():
    """The subtle result: INT4's advantage decays as the server gets busy."""
    fp16, int4 = load("sweep-vllm.json"), load("sweep-vllm-int4.json")
    if not (fp16 and int4):
        return
    f = {r["concurrency"]: r["throughput_tps"] for r in fp16["sweep"]}
    i = {r["concurrency"]: r["throughput_tps"] for r in int4["sweep"]}
    xs = sorted(set(f) & set(i))
    ratio = [i[c] / f[c] for c in xs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ratio, color=GREEN, marker="o", lw=2.5)
    ax.axhline(1.0, color=GRAY, ls="--", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("concurrency (requests in flight)")
    ax.set_ylabel("INT4 throughput / FP16 throughput")
    ax.set_title("INT4's advantage decays with load\n(memory-bound at C=1 → compute-bound at C=128)")
    ax.grid(alpha=0.3)
    for x, r in zip(xs, ratio):
        ax.annotate(f"{r:.2f}x", (x, r), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "int4_advantage_decay.png", dpi=150)
    plt.close(fig)


def plot_max_num_seqs():
    """The correction: the 'saturation cliff' was a default config limit."""
    man = json.loads(Path("manifest.json").read_text(encoding="utf-8"))
    runs = man["max_num_seqs_experiment"]["runs"]
    labels = [f"C={r['max_concurrency']}\nmax_num_seqs={r['max_num_seqs']}" for r in runs]
    tps = [r["output_tok_s"] for r in runs]
    tpot = [r["tpot_ms"] for r in runs]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = [BLUE, RED, GREEN]
    for ax, vals, title, ylab in [(axes[0], tps, "Output throughput", "tokens/sec"),
                                  (axes[1], tpot, "Time per output token", "ms (lower better)")]:
        bars = ax.bar(labels, vals, color=colors, alpha=0.85)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylab, fontsize=9)
        ax.grid(alpha=0.3, axis="y")
        ax.tick_params(labelsize=8)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center",
                    va="bottom", fontsize=9)
    fig.suptitle("The 'saturation cliff' was a default config limit, not hardware\n"
                 "(measured with vllm bench serve — independent of our harness)", fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOTS / "max_num_seqs_correction.png", dpi=150)
    plt.close(fig)


def main():
    PLOTS.mkdir(exist_ok=True)
    plot_throughput()
    plot_latency()
    plot_quantization()
    plot_int4_decay()
    plot_max_num_seqs()
    for p in sorted(PLOTS.glob("*.png")):
        print(f"  wrote {p}  ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
