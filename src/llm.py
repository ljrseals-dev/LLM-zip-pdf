"""Local LLM loading and inference (llama-cpp-python)."""

from pathlib import Path
from typing import Optional, List, Dict, Any
import os

_llm = None
_model_path: Optional[str] = None


def load_model(
    model_path: str | Path,
    n_ctx: int = 4096,
    n_threads: Optional[int] = None,
    n_gpu_layers: int = 0,
    verbose: bool = False,
) -> Any:
    """Load a GGUF model with llama-cpp-python.

    Args:
        model_path: Path to .gguf file.
        n_ctx: Context window size.
        n_threads: CPU threads (default: os.cpu_count() // 2 or 4).
        n_gpu_layers: Layers to offload to GPU (0 = CPU only).
        verbose: Print llama.cpp logs.

    Returns:
        Llama instance.
    """
    global _llm, _model_path
    try:
        from llama_cpp import Llama
    except ImportError as e:
        raise ImportError(
            "llama-cpp-python is required. Install with:\n"
            "  pip install llama-cpp-python\n"
            "On Termux you may need to build from source or use a prebuilt wheel."
        ) from e

    model_path = str(Path(model_path).resolve())
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if n_threads is None:
        n_threads = max(1, (os.cpu_count() or 4) // 2)

    _llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        verbose=verbose,
    )
    _model_path = model_path
    return _llm


def is_loaded() -> bool:
    return _llm is not None


def generate(
    prompt: str,
    system: str = "You are a helpful assistant that answers questions about documents.",
    max_tokens: int = 512,
    temperature: float = 0.7,
    stop: Optional[List[str]] = None,
) -> str:
    """Generate a completion from the loaded model."""
    if _llm is None:
        raise RuntimeError("No model loaded. Call load_model() first.")

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]

    # Prefer chat completion if available
    try:
        out = _llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop or [],
        )
        return out["choices"][0]["message"]["content"].strip()
    except Exception:
        # Fallback to plain completion
        full_prompt = f"System: {system}\n\nUser: {prompt}\n\nAssistant:"
        out = _llm(
            full_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop or ["User:", "System:"],
        )
        return out["choices"][0]["text"].strip()


def unload() -> None:
    """Release the model (best-effort)."""
    global _llm, _model_path
    _llm = None
    _model_path = None
