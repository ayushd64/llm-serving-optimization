Known limitation — fixed per-request overhead (Ollama, Windows/WDDM). 
On this setup, Ollama reports a consistent ~0.38s load_duration on every request, even when the model is verified resident in VRAM (ollama ps shows 100% GPU), explicitly pinned (keep_alive=30m), and the GPU is not shared with any other process. Ruled out: model eviction (persists across rapid back-to-back requests), and GPU contention (persists after closing all other GPU apps). Concluded to be an inherent per-request cost of the Ollama runner on Windows. Impact: it inflates end-to-end latency and TTFT by ~0.38s but is cleanly separable — it lives inside Ollama's reported load_duration, so generation-speed (tok/s) and prefill numbers are unaffected. When we compare serving engines in later milestones, we report generation throughput and prefill/decode separately precisely so this fixed offset doesn't distort the comparison. 


Milestone 3 — engine comparison caveats. 
(1) Engines were benchmarked sequentially, not concurrently: two fp16 1.5B models exceed 8 GB VRAM, so both could not be resident at once. 
(2) Environments differ — vLLM ran in Linux (WSL2), Ollama ran natively on Windows; same physical GPU, but not an identical stack. 
(3) A fixed ~2s per-request overhead from IPv6 localhost resolution on the Windows→WSL2 bridge was found and eliminated by using 127.0.0.1; all reported vLLM numbers use the fixed path. 
(4) This is single-request load only — vLLM's throughput advantages are expected to appear under concurrency (Milestone 5), not here. 
