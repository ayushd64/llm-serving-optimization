"""
first_contact.py

Milestone 1: send a prompt to our locally-running Ollama model and print the
reply as it streams in, one piece at a time. This proves our setup works and
lets us SEE the model generating its answer token by token.
"""


import json         # to build and read the small data packets we exchange
import time         # to measure how long things take
import requests     # to send the request to the Ollama server


# --- Settings (later we will move these into a proper config file) ---
OLLAMA_URL = "http://localhost:11434/api/generate"      # where ollama listens
MODEL_NAME = "qwen3.5:9b"                               # the model we pulled
PROMPT = "Explain what a GPU is in two sentences."      # what we ask


def main():
    # The request body: what we want, in the format Ollama expects.
    payload = {
        "model": MODEL_NAME,
        "prompt": PROMPT,
        "stream": True,     # ask Ollama to send the answer piece-by-piece
    }

    print(f"Prompt: {PROMPT}\n")
    print("Model reply:\n")

    start = time.perf_counter()     # stopwatch: the moment we send the request
    first_token_time = None         # will hold WHEN the first piece arrives
    piece_count = 0

    # Send the request; stream=True lets us read the reply as it trickles in.
    with requests.post(OLLAMA_URL, json=payload, stream=True) as response:
        for line in response.iter_lines():
            if not line:
                continue                            # skip any empty lines
            chunk = json.loads(line)                # each line is one small packet
            piece = chunk.get("response", "")       # the bit of text it contains

            if first_token_time is None and piece:
                first_token_time = time.perf_counter()      # first piece landed

            print(piece, end="", flush=True)        # print it with no line break
            piece_count += 1

            if chunk.get("done"):                   # Ollama says when it's finished
                break

    end = time.perf_counter()

    # --- A first, rough timing summary ---
    ttft = (first_token_time - start) if first_token_time else 0.0
    total = end - start
    print("\n\n--- rough timing ---")
    print(f"Time to first token : {ttft:.3f} s")
    print(f"Total time          : {total:.3f} s")
    print(f"Pieces received     : {piece_count}")


if __name__ == "__main__":
    main()