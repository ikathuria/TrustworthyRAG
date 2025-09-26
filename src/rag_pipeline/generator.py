"""
generator.py
Wrapper around an LLM (Ollama via subprocess fallback).
Exposes a simple `generate(prompt, max_tokens=...)` interface.
"""

from typing import Optional
import subprocess
import json
import shlex
import os

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_RUN_CMD = os.getenv("OLLAMA_RUN_CMD", "ollama")  # path to ollama binary


class OllamaWrapper:
    """
    Minimal wrapper that calls the `ollama run <model> "<prompt>"` command using subprocess.
    This keeps the rest of your codebase independent of how you run the model.
    If you have a Python client for Ollama, replace this class.
    """

    def __init__(self, model: str = OLLAMA_MODEL, binary: str = OLLAMA_RUN_CMD, extra_args: Optional[str] = None):
        self.model = model
        self.binary = binary
        self.extra_args = extra_args or ""

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.0) -> str:
        """
        Runs ollama via subprocess. Returns model text (stdout).
        Note: this is a simple approach — for production, use a proper HTTP or Python client.
        """
        print("\n[DEBUG] Generating with prompt (first 500 chars):")
        print(prompt[:500] + ("..." if len(prompt) > 500 else ""))

        # Build command safely
        cmd = f'{shlex.quote(self.binary)} run {shlex.quote(self.model)} {self.extra_args} "{prompt}"'
        print(f"\n[DEBUG] Running command: {cmd}")
        try:
            proc = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=120)
            if proc.returncode != 0:
                raise RuntimeError(f"Ollama run failed: {proc.stderr.strip()}")
            return proc.stdout.strip()
        except Exception as e:
            raise RuntimeError(f"OllamaWrapper generate error: {e}") from e


def build_prompt(query: str, contexts: list, instructions: Optional[str] = None) -> str:
    """
    Build a RAG prompt that includes top retrieved contexts.
    """
    instructions = instructions or "You are an assistant who is very good at searching documents. Use the provided context to answer the question concisely and cite sources where relevant. Use only the information from the context. If the answer is not contained within the context, say 'I don't know'."
    ctx_text = ""
    for i, c in enumerate(contexts, 1):
        ctx_text += f"source {i}: {c['text']}\n"
    prompt = f"""{instructions}

Query: {query}

Context:
{ctx_text}

Answer:
"""

    return prompt
