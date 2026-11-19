"""Local LLM loading and inference (llama-cpp-python)."""
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
    if n_ctx < 1 or (n_threads is not None and n_threads < 1) or n_gpu_layers < 0:
        raise ValueError("n_ctx and n_threads must be positive; n_gpu_layers cannot be negative")
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
    # Construct first; retain the previously loaded model if construction fails.
    instance = Llama(model_path=str(resolved), n_ctx=n_ctx, n_threads=threads,
                     n_gpu_layers=n_gpu_layers, verbose=verbose)
    _llm, _model_path = instance, str(resolved)
    return instance


def is_loaded() -> bool:
    return _llm is not None


def generate(prompt: str, system: str = "You are a helpful assistant that answers questions about documents.",
             max_tokens: int = 512, temperature: float = 0.7,
             stop: Optional[List[str]] = None) -> str:
    """Generate text; only use completion fallback when chat API is absent."""
    if _llm is None:
        raise RuntimeError("No model loaded. Call load_model() first.")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if max_tokens < 1 or temperature < 0:
        raise ValueError("max_tokens must be positive and temperature cannot be negative")
    if hasattr(_llm, "create_chat_completion"):
        output = _llm.create_chat_completion(
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=temperature, stop=stop or [],
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
