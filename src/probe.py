"""
probe.py  (throwaway diagnostic -- not part of the benchmark)

We confirmed ~2s of overhead lives OUTSIDE Ollama's own timing. This isolates
the cause by timing a trivial "list models" request (no model work at all) four
ways: localhost vs 127.0.0.1, and with vs without the system proxy. Whichever
combination is fast points straight at the culprit.
"""

import time
import requests


def time_get(url, use_env_proxy):
    session = requests.Session()
    session.trust_env = use_env_proxy   # False = ignore system HTTP proxy settings
    start = time.perf_counter()
    session.get(url, timeout=30)
    return time.perf_counter() - start


tests = [
    ("localhost  + system proxy", "http://localhost:11434/api/tags", True),
    ("localhost  + NO proxy",     "http://localhost:11434/api/tags", False),
    ("127.0.0.1  + system proxy", "http://127.0.0.1:11434/api/tags", True),
    ("127.0.0.1  + NO proxy",     "http://127.0.0.1:11434/api/tags", False),
]

print("Timing a trivial 'list models' request (no model work at all):\n")
for label, url, use_env in tests:
    best = min(time_get(url, use_env) for _ in range(3))   # best of 3, to skip flukes
    print(f"  {label:28} {best * 1000:8.1f} ms")
