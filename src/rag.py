"""Simple text chunking and retrieval for document Q&A."""

from typing import List, Tuple
import re


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks by approximate character count."""
    if not text or not text.strip():
        return []

    # Prefer splitting on paragraph / sentence boundaries
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 1 <= chunk_size:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            # If single paragraph is huge, hard-split
            while len(para) > chunk_size:
                chunks.append(para[:chunk_size])
                para = para[chunk_size - overlap :]
            current = para

    if current:
        chunks.append(current)

    # Add overlap between consecutive chunks for continuity
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-overlap:]
            overlapped.append(prev_tail + "\n" + chunks[i])
        return overlapped
    return chunks


def simple_retrieve(
    query: str,
    chunks: List[str],
    top_k: int = 4,
) -> List[Tuple[str, float]]:
    """Very lightweight keyword-based retrieval (no embeddings required).

    Scores chunks by term overlap. Good enough for small documents offline.
    """
    if not chunks:
        return []

    query_terms = set(re.findall(r"\w+", query.lower()))
    if not query_terms:
        return [(c, 0.0) for c in chunks[:top_k]]

    scored: List[Tuple[str, float]] = []
    for chunk in chunks:
        chunk_terms = set(re.findall(r"\w+", chunk.lower()))
        overlap = len(query_terms & chunk_terms)
        score = overlap / max(1, len(query_terms))
        scored.append((chunk, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def build_context(query: str, document_text: str, max_context_chars: int = 3000) -> str:
    """Build a context string from the most relevant chunks."""
    chunks = chunk_text(document_text)
    if not chunks:
        return document_text[:max_context_chars]

    top = simple_retrieve(query, chunks, top_k=6)
    context_parts = []
    total = 0
    for chunk, score in top:
        if total + len(chunk) > max_context_chars:
            break
        context_parts.append(chunk)
        total += len(chunk)

    return "\n\n---\n\n".join(context_parts) if context_parts else document_text[:max_context_chars]
