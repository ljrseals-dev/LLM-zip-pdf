"""PDF text extraction utilities."""

from pathlib import Path
from typing import List, Optional

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


def extract_text_from_pdf(path: str | Path, max_pages: Optional[int] = None) -> str:
    """Extract text from a PDF file.

    Args:
        path: Path to the PDF.
        max_pages: Optional limit on number of pages to process.

    Returns:
        Extracted text as a single string.
    """
    if PdfReader is None:
        raise ImportError("pypdf is required. Install with: pip install pypdf")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    reader = PdfReader(str(path))
    pages = reader.pages
    if max_pages is not None:
        pages = pages[:max_pages]

    texts: List[str] = []
    for i, page in enumerate(pages):
        try:
            text = page.extract_text() or ""
            if text.strip():
                texts.append(f"--- Page {i + 1} ---\n{text.strip()}")
        except Exception as e:
            texts.append(f"--- Page {i + 1} (extraction error: {e}) ---")

    return "\n\n".join(texts) if texts else "(No extractable text found)"


def get_pdf_page_count(path: str | Path) -> int:
    """Return number of pages in the PDF."""
    if PdfReader is None:
        raise ImportError("pypdf is required")
    reader = PdfReader(str(path))
    return len(reader.pages)
