"""Minimal Ollama client that logs tokens and latency for the cost analysis."""
from __future__ import annotations

import os
import subprocess
import time

import requests


def _server_up(host):
    try:
        requests.get(host, timeout=2)
        return True
    except requests.exceptions.RequestException:
        return False


def setup_ollama(models, host="http://localhost:11434", wait_s=120, log_path="/tmp/ollama.log",
                 num_parallel=1, max_loaded_models=3):
    """Install and start Ollama in Colab, then pull the given models. Safe to re-run."""
    import shutil

    # 1. Install (the official script needs zstd to unpack the release archive).
    if shutil.which("ollama") is None:
        print("Installing Ollama ...")
        subprocess.run("apt-get update -qq && apt-get install -y -qq pciutils zstd > /dev/null",
                       shell=True, check=False)
        r = subprocess.run("curl -fsSL https://ollama.com/install.sh | sh",
                           shell=True, capture_output=True, text=True)
        if shutil.which("ollama") is None:
            raise RuntimeError("Ollama installation failed:\n" + (r.stdout + r.stderr)[-2000:])

    # 2. Start the server in the background (no shell, so the process handle stays valid).
    if _server_up(host) and num_parallel > 1:
        print("Note: server already running; restart the Colab session if OLLAMA_NUM_PARALLEL must change.")
    if not _server_up(host):
        print(f"Starting Ollama server (parallel={num_parallel}) ...")
        env = dict(os.environ, OLLAMA_KEEP_ALIVE="30m", OLLAMA_HOST=host.split("//")[-1],
                   OLLAMA_NUM_PARALLEL=str(num_parallel), OLLAMA_MAX_LOADED_MODELS=str(max_loaded_models))
        logf = open(log_path, "w")
        subprocess.Popen(["ollama", "serve"], stdout=logf, stderr=subprocess.STDOUT, env=env,
                         start_new_session=True)
        for _ in range(wait_s):
            if _server_up(host):
                break
            time.sleep(1)
        else:
            tail = open(log_path).read()[-2000:]
            raise RuntimeError(f"Ollama server did not start within {wait_s}s. Log tail:\n{tail}")

    # 3. Pull models (skips layers that are already present).
    for m in models:
        print(f"Pulling {m} ...")
        r = subprocess.run(["ollama", "pull", m], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"Could not pull {m}:\n{r.stderr[-1000:]}")
    ver = subprocess.run(["ollama", "--version"], capture_output=True, text=True).stdout.strip()
    print("Ollama ready |", ver, "| models:", ", ".join(models))


class OllamaClient:
    def __init__(self, host="http://localhost:11434", num_ctx=2048, num_predict=256, retries=3):
        self.url = host.rstrip("/") + "/api/chat"
        self.num_ctx, self.num_predict, self.retries = num_ctx, num_predict, retries

    def chat(self, model, messages, temperature=0.0, seed=0, schema=None) -> dict:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "seed": seed,
                        "num_ctx": self.num_ctx, "num_predict": self.num_predict},
        }
        if schema is not None:
            payload["format"] = schema
        last_err = None
        for attempt in range(self.retries):
            t0 = time.perf_counter()
            try:
                r = requests.post(self.url, json=payload, timeout=300)
                r.raise_for_status()
                d = r.json()
                return {
                    "content": d.get("message", {}).get("content", ""),
                    "prompt_tokens": d.get("prompt_eval_count", 0),
                    "completion_tokens": d.get("eval_count", 0),
                    "latency_s": time.perf_counter() - t0,
                    # time Ollama spent loading the model into GPU memory (part of latency_s)
                    "load_s": d.get("load_duration", 0) / 1e9,
                    "error": None,
                }
            except Exception as e:  # network hiccup, timeout, model reload
                last_err = repr(e)
                time.sleep(2 * (attempt + 1))
        return {"content": "", "prompt_tokens": 0, "completion_tokens": 0,
                "latency_s": 0.0, "load_s": 0.0, "error": last_err}
