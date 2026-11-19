"""Local LLM loading and inference (llama-cpp-python)."""
import math
import os
from pathlib import Path
from typing import Any, List, Optional

_llm = None
_model_path: Optional[str] = None


def load_model(model_path: str | Path, n_ctx: int = 4096,
               n_threads: Optional[int] = None, n_gpu_layers: int = 0,
               verbose: bool = False) -> Any:
    """Load a GGUF model with llama-cpp-python."""
    global _llm, _model_path
    if (not isinstance(n_ctx, int) or isinstance(n_ctx, bool) or n_ctx < 1 or
        (n_threads is not None and (not isinstance(n_threads, int) or isinstance(n_threads, bool) or n_threads < 1)) or
        not isinstance(n_gpu_layers, int) or isinstance(n_gpu_layers, bool) or n_gpu_layers < 0):
        raise ValueError("n_ctx and n_threads must be positive integers; n_gpu_layers must be a non-negative integer")
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise ImportError(
            "llama-cpp-python is required. Install with pip install llama-cpp-python. "
            "On Termux you may need to build from source."
        ) from exc

    resolved = Path(model_path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Model not found: {resolved}")
    threads = n_threads if n_threads is not None else max(1, (os.cpu_count() or 4) // 2)
    instance = Llama(model_path=str(resolved), n_ctx=n_ctx, n_threads=threads,
                     n_gpu_layers=n_gpu_layers, verbose=verbose)
    _llm, _model_path = instance, str(resolved)
    return instance


def is_loaded() -> bool:
    return _llm is not None


def loaded_model_path() -> Optional[str]:
    """Return the resolved path for the currently loaded model, if any."""
    return _model_path


def generate(prompt: str, system: str = "You are a helpful assistant that answers questions about documents.",
             max_tokens: int = 512, temperature: float = 0.7,
             stop: Optional[List[str]] = None) -> str:
    """Generate text; only use completion fallback when chat API is absent."""
    if _llm is None:
        raise RuntimeError("No model loaded. Call load_model() first.")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if not isinstance(system, str):
        raise TypeError("system must be a string")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be a finite non-negative number")
    if stop is not None and (not isinstance(stop, list) or any(not isinstance(item, str) for item in stop)):
        raise ValueError("stop must be a list of strings")
    stop_sequences = stop or []
    if hasattr(_llm, "create_chat_completion"):
        output = _llm.create_chat_completion(
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=temperature, stop=stop_sequences,
        )
        try:
            content = output["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Model returned an invalid chat-completion response") from exc
    else:
        output = _llm(f"System: {system}\n\nUser: {prompt}\n\nAssistant:",
                      max_tokens=max_tokens, temperature=temperature,
                      stop=stop or ["User:", "System:"])
        try:
            content = output["choices"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Model returned an invalid completion response") from exc
    if not isinstance(content, str):
        raise RuntimeError("Model returned non-text content")
    return content.strip()


def unload() -> None:
    """Release the model (best-effort)."""
    global _llm, _model_path
    _llm = None
    _model_path = None
