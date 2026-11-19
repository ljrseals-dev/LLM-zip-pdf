"""PDF text extraction utilities."""
from pathlib import Path
from typing import List, Optional

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


def _reader(path: str | Path):
    if PdfReader is None:
        raise ImportError("pypdf is required. Install with: pip install pypdf")
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"PDF not found: {source}")
    try:
        return PdfReader(str(source), strict=False)
    except Exception as exc:
        raise ValueError(f"Unable to read PDF {source}: {exc}") from exc


def extract_text_from_pdf(path: str | Path, max_pages: Optional[int] = None) -> str:
    """Extract text, retaining page markers and reporting per-page extraction errors."""
    if max_pages is not None and (not isinstance(max_pages, int) or isinstance(max_pages, bool) or max_pages < 1):
        raise ValueError("max_pages must be a positive integer")
    reader = _reader(path)
    pages = reader.pages if max_pages is None else reader.pages[:max_pages]
    texts: List[str] = []
    for index, page in enumerate(pages, start=1):
        try:
            text = page.extract_text() or ""
            if text.strip():
                texts.append(f"--- Page {index} ---\n{text.strip()}")
        except Exception as exc:
            texts.append(f"--- Page {index} (extraction error: {type(exc).__name__}) ---")
    return "\n\n".join(texts) if texts else "(No extractable text found)"


def get_pdf_page_count(path: str | Path) -> int:
    """Return number of pages in a PDF."""
    return len(_reader(path).pages)
