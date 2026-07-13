"""
probe_load.py  (throwaway diagnostic)

Investigate the ~0.39s 'load' that shows on every warm request. We fire several
TINY requests back-to-back (num_predict=1, so almost no generation) with the
model explicitly pinned in memory (keep_alive=30m). Then we watch load_duration:

  - If it stays ~0.39s on every request  -> a genuine fixed PER-REQUEST cost.
  - If it's high once then drops to ~0     -> the model was being evicted/reloaded.
"""

import time
import requests

URL = "http://127.0.0.1:11434/api/generate"
MODEL = "qwen3.5:2b"
NANO = 1_000_000_000

session = requests.Session()
session.trust_env = False


def tiny_request():
    payload = {
        "model": MODEL,
        "prompt": "Hi",
        "stream": False,
        "think": False,
        "keep_alive": "30m",                                  # pin the model in memory
        "options": {"temperature": 0, "num_predict": 1},      # generate ~1 token only
    }
    data = session.post(URL, json=payload).json()
    return (data.get("load_duration", 0) / NANO,
            data.get("total_duration", 0) / NANO)


print("Firing 8 tiny back-to-back requests (model pinned with keep_alive=30m):\n")
for i in range(8):
    load_s, total_s = tiny_request()
    print(f"  req {i + 1}: load {load_s:6.3f}s | total {total_s:6.3f}s")
    time.sleep(0.2)   # small gap, but nowhere near the 30m keep-alive window
