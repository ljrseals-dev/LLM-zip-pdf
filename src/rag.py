"""Simple text chunking and retrieval for document Q&A."""
import re
from typing import List, Tuple


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """Split text into bounded, overlapping chunks without zero-step loops."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if (not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or
            not isinstance(overlap, int) or isinstance(overlap, bool)):
        raise TypeError("chunk_size and overlap must be integers")
    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be in [0, chunk_size)")
    if not text.strip():
        return []

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def simple_retrieve(query: str, chunks: List[str], top_k: int = 4) -> List[Tuple[str, float]]:
    """Rank chunks by query-term overlap; stable ties preserve document order."""
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if not isinstance(chunks, (list, tuple)):
        raise TypeError("chunks must be a list or tuple of strings")
    if any(not isinstance(chunk, str) for chunk in chunks):
        raise TypeError("every chunk must be a string")
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise TypeError("top_k must be an integer")
    if top_k < 0:
        raise ValueError("top_k cannot be negative")
    if not chunks or top_k == 0:
        return []
    query_terms = set(re.findall(r"\w+", query.lower()))
    if not query_terms:
        return [(chunk, 0.0) for chunk in chunks[:top_k]]
    scored = []
    for index, chunk in enumerate(chunks):
        terms = set(re.findall(r"\w+", chunk.lower()))
        scored.append((chunk, len(query_terms & terms) / len(query_terms), index))
    scored.sort(key=lambda item: (-item[1], item[2]))
    return [(chunk, score) for chunk, score, _ in scored[:top_k]]


def build_context(query: str, document_text: str, max_context_chars: int = 3000) -> str:
    """Build a context string bounded to max_context_chars."""
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if not isinstance(document_text, str):
        raise TypeError("document_text must be a string")
    if not isinstance(max_context_chars, int) or isinstance(max_context_chars, bool):
        raise TypeError("max_context_chars must be an integer")
    if max_context_chars < 0:
        raise ValueError("max_context_chars cannot be negative")
    if max_context_chars == 0 or not document_text:
        return ""
    chunks = chunk_text(document_text)
    if not chunks:
        return ""
    selected = []
    total = 0
    for chunk, _score in simple_retrieve(query, chunks, top_k=len(chunks)):
        separator = "\n\n---\n\n" if selected else ""
        remaining = max_context_chars - total - len(separator)
        if remaining <= 0:
            break
        if len(chunk) > remaining:
            if not selected:
                selected.append(chunk[:remaining])
            break
        selected.append(chunk)
        total += len(separator) + len(chunk)
    return "\n\n---\n\n".join(selected)
