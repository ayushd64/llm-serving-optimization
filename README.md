# LLM Inference Serving: An Honest Benchmark on Constrained Hardware

Benchmarking **vLLM** and **Ollama** serving `Qwen2.5-1.5B-Instruct` on a single
8 GB laptop GPU — measuring throughput, latency, memory, and **quality**, with a
methodology designed to catch its own mistakes.

**Headline finding: every performance ceiling I found in this study turned out to be
a default configuration limit, not an architectural one.** Both engines' apparent
"saturation points" moved when I changed a single setting. The engines were rarely
the bottleneck; the defaults were.

---

## Hardware & software

| | |
|---|---|
| GPU | NVIDIA RTX 3070 **Laptop** (8 GB VRAM, **45 W** power limit) |
| CPU / RAM | Intel i7 / 16 GB |
| OS | Windows 11 + WSL2 (Ubuntu) |
| Driver / CUDA | 581.95 / 13.0 |
| vLLM | 0.22.1 (in WSL2) |
| Ollama | native Windows |
| Model | `Qwen2.5-1.5B-Instruct` (fp16 / GPTQ-Int8 / AWQ-Int4) |

The 45 W laptop variant matters: it has substantially less bandwidth and compute than
a 220 W desktop 3070. Absolute numbers here should not be compared to desktop results.

Full versions and hand-recorded measurements: [`manifest.json`](manifest.json).

---

## Results

### 1. Throughput vs concurrency

![throughput](plots/throughput_vs_concurrency.png)

| config | peak throughput | at concurrency |
|---|---|---|
| vLLM fp16 | **2,197 tok/s** | 128 |
| vLLM int4 (AWQ) | **2,410 tok/s** | 128 |
| Ollama `NUM_PARALLEL=16` | **426 tok/s** | 32 |
| Ollama `NUM_PARALLEL=8` | 326 tok/s | 32 |
| Ollama `NUM_PARALLEL=32` | 184 tok/s | 32 (VRAM spill) |
| Ollama default | ~31 tok/s | flat at all levels |

At single-request load the engines are **comparable** (vLLM 33 tok/s vs Ollama 35 tok/s).
vLLM's advantage is entirely a concurrency story: continuous batching.

**The number I almost published was wrong three times.** Ollama's default config
serialises requests (flat 31 tok/s → a 73× "advantage" for vLLM). Setting
`OLLAMA_NUM_PARALLEL=8` closed that to 6.7×. Sweeping further to `=16` closed it to
**5.2×** — and `=32` *reduced* throughput because 16 slots × 4096 context exceeded
8 GB and forced an 18% CPU offload (verified via `ollama ps`).

**Ollama's ceiling on this hardware is VRAM-bound, not architectural.**

### 2. The "saturation cliff" was a default

![correction](plots/max_num_seqs_correction.png)

Our sweeps showed a dramatic throughput collapse at C=256 (2,197 → 610 tok/s, latency
2.8s → 21s). I proposed two explanations and **both were wrong:**

- *"KV cache exhaustion"* — refuted: vLLM's own logs reported `GPU KV cache usage: 3-8%`
  during the collapse. Doubling the cache via INT4 (87k → 158k tokens) did not move the cliff.
- *"Our load generator is the bottleneck"* — refuted: `vllm bench serve`, an independent
  tool, reproduced the collapse (12.65 req/s vs our 12.2 req/s).

The actual cause: **vLLM's default `--max-num-seqs=256`.** Offering exactly 256 requests
to a scheduler capped at 256 causes thrashing.

| C=256 | `max_num_seqs=256` (default) | `max_num_seqs=512` |
|---|---|---|
| output throughput | 734 tok/s | **1,796 tok/s** (+145%) |
| TPOT | 342 ms | **114 ms** (−67%) |
| TTFT | 625 ms | 1,679 ms (**+169% — the cost**) |

This is a **trade**, not a free win: 2.4× throughput bought with 2.7× worse
time-to-first-token. True saturation was never located — it is above C=256.

### 3. Quantization: memory, speed, and quality

![quantization](plots/quantization_tradeoff.png)

| precision | weights | KV cache | tok/s @ C=1 | perplexity | Δ vs fp16 |
|---|---|---|---|---|---|
| FP16 | 2.98 GiB | 87,024 tok | 33.4 | 14.498 | — |
| **INT8 (GPTQ)** | 1.74 GiB | 134,192 tok | 89.8 | 14.511 | **+0.09%** |
| INT4 (AWQ) | 1.10 GiB | 158,000 tok | 122.3 | 15.734 | **+8.52%** |

**INT8 is effectively free**: 2.7× faster, 42% less VRAM, perplexity indistinguishable
from FP16. The +0.09% is within chunk-level noise — 3 of 10 chunks were *better* than fp16.

**INT4 is not free.** +8.5% perplexity, and the degradation is **systematic, not noise**:
**10 of 10 chunks got worse** (+3.7% to +13.4%). If quantization were harmless you would
expect ~half the chunks to drift in each direction, as INT8 did.

Perplexity measured on a fixed 4,530-token WikiText-2 slice, identical across all three
precisions, via vLLM's `prompt_logprobs`. Deterministic — no sampling variance.

### 4. Quantization's benefit decays with load

![decay](plots/int4_advantage_decay.png)

| concurrency | 1 | 8 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| INT4 speedup vs FP16 | **3.70×** | 2.58× | 1.88× | 1.57× | **1.10×** |

At C=1 decoding is **memory-bandwidth-bound** — the GPU spends its time reading weights,
so 2.7× fewer bytes means ~3.7× faster. At C=128 it is **compute-bound** — weight reads
amortise across the batch and the dequantisation overhead consumes nearly the entire
advantage.

**Quantization's speed benefit is inversely proportional to how busy your server is.**
I predicted the opposite before measuring, and was wrong.

---

## Methodology

- **Warmup + repeated trials.** All single-request numbers are 10 trials after 2 warmups,
  reported with spread (typical stdev < 2%).
- **Determinism.** `temperature=0` throughout; every trial produces identical output and
  therefore identical work.
- **One ruler for all engines.** TTFT and latency are timed by our own wall clock, not by
  each engine's self-reported timings, so engines are measured identically. Token *counts*
  come from each engine's own reporting (`eval_count` / `usage.completion_tokens`).
- **Little's Law validation.** Every sweep row checks `concurrency ≈ throughput × latency`
  and reports the achieved concurrency. **This caught a silent bug**: with a fixed 60
  requests per level, levels above C=64 could never reach their target concurrency — five
  "different" levels were secretly the same experiment. The apparent plateau was an
  artifact. Request counts now auto-scale to 10× concurrency.
- **Independent cross-validation.** Key results re-measured with `vllm bench serve`.

---

## Limitations

Stated plainly, because under-measured results are worse than no results.

1. **Single prompt.** Every sweep used one identical 22-token prompt. vLLM reported an
   **82% prefix-cache hit rate** as a result — throughput numbers are inflated by caching
   a real workload wouldn't get. `vllm bench serve` with random prompts gave **1,703 tok/s
   at C=128 vs our 2,197** — suggesting **~30% inflation**. No prompt-length distribution
   was tested; prefill behaviour is therefore uncharacterised.
2. **Closed-loop load generation.** N workers each fire the next request on completion.
   This couples offered load to server speed and suffers coordinated omission. Open-loop
   (Poisson arrivals at fixed RPS) is the rigorous approach and was not implemented.
3. **Engines benchmarked sequentially, not concurrently.** Two fp16 1.5B models exceed
   8 GB, so both could never be resident. Runs are minutes apart, not simultaneous.
4. **Different environments.** vLLM ran in WSL2 (Linux); Ollama ran natively on Windows.
   Same physical GPU, different stacks.
5. **Bit-width and algorithm are conflated.** INT8 is GPTQ; INT4 is AWQ. "INT4 is 8.5%
   worse" partly measures AWQ-vs-GPTQ, not purely 8-vs-4 bits. Isolating this needs
   `GPTQ-Int4`, which was not run.
6. **Quantization damage may be size-specific.** 8.5% is higher than the 1–3% AWQ papers
   typically report — plausibly because 1.5B models have less redundancy to absorb rounding
   error. **These results should not be extrapolated to 7B+.**
7. **Perplexity is a proxy.** +8.5% perplexity does not mean 8.5% worse answers. No
   downstream task accuracy (GSM8K, MMLU) was measured.
8. **Non-standard perplexity protocol.** A 20k-char WikiText-2 slice scored in independent
   chunks, not the full test set with sliding-window context. Absolute values are **not**
   comparable to published figures; only the deltas are meaningful.
9. **Single-run sweeps.** Concurrency sweeps were run once, not repeated. Observed
   run-to-run variance at C=64 was **~35%** (26.1 vs 40.3 req/s across identical runs) —
   sweep numbers are far less reproducible than the single-request trials suggest.
10. **True saturation never located.** With `max_num_seqs=512` the cliff disappeared;
    no higher concurrency was tested. The real ceiling is unknown.
11. **Ollama's parallelism not exhaustively swept.** Tested 1/8/16/32. `=16` was best,
    `=32` spilled to CPU. Intermediate values untested.
12. **Unexplained: Ollama reports a fixed ~0.38 s `load_duration` per request** even with
    the model verified resident (`ollama ps` = 100% GPU), pinned (`keep_alive=30m`), and
    no other GPU process. Ruled out: eviction, GPU contention. Cause unknown.
13. **Unexplained: Ollama's reported memory scales non-linearly** with `NUM_PARALLEL`
    (3.3 GB at 16, 7.7 GB at 32), suggesting lazy allocation or a silent cap.
14. **No cost-per-million-tokens.** Not computed. Would be modelled, not measured, since
    this ran on owned hardware.
15. **A ~2 s measurement bug existed and was fixed mid-project.** On Windows, `localhost`
    resolved to IPv6 first and stalled ~2 s per request before falling back. This silently
    doubled every latency number until caught by reconciling our wall clock against the
    engine's self-reported timings. All published numbers use `127.0.0.1`. **Any result
    from before this fix would have been wrong and looked fine.**

---

## What I would do next

1. **Open-loop load generation** with Poisson arrivals — the single biggest methodological gap.
2. **Prompt-length distributions** (short/medium/long) and random prompts, to kill the
   prefix-cache confound and characterise prefill.
3. **Repeat sweeps 3×** and report spread; 35% single-run variance is not publishable.
4. **`GPTQ-Int4`** to separate bit-width from algorithm.
5. **Downstream task eval** (GSM8K subset) to translate perplexity into user-facing quality.
6. **Sweep `max_num_seqs`** properly and map the throughput/TTFT Pareto frontier.
7. **A larger GPU** to test whether the quantization findings hold at 7B+.

---

## Reproduce

```bash
git clone <repo> && cd llm-serving-optimization
python -m venv .venv && .venv\Scripts\Activate.ps1     # Windows
pip install -r requirements.txt

# start ONE engine (8 GB fits one model at a time)
vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000 --dtype float16 \
  --max-model-len 4096 --gpu-memory-utilization 0.80        # in WSL2

python src/benchmark.py vllm        # single-request: TTFT, tok/s, spread
python src/concurrency.py vllm      # concurrency sweep + Little's Law check
python src/quality.py vllm          # perplexity
python src/plots.py                 # regenerate every figure from results/
```

Every figure in this README is generated by `src/plots.py` from the raw JSON in
`results/`. Nothing is hand-transcribed.

## Repo layout

```
config.yaml        all knobs; no hardcoded values
manifest.json      hardware, versions, hand-recorded log measurements
data/              fixed perplexity eval text (committed for reproducibility)
src/
  client.py        engine adapters (Ollama + vLLM) behind one interface
  metrics.py       summary stats + percentiles
  benchmark.py     single-request harness
  concurrency.py   concurrency sweep + Little's Law validation
  quality.py       perplexity via prompt_logprobs
  plots.py         regenerates all figures from results/
results/           raw JSON, one file per run (immutable)
plots/             generated figures
```
