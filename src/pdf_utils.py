"""PDF text extraction with explicit input and output resource limits."""
from pathlib import Path
from typing import List, Optional

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

DEFAULT_MAX_PDF_BYTES = 50 * 1024 * 1024
DEFAULT_MAX_PDF_PAGES = 500
DEFAULT_MAX_TEXT_CHARS = 5_000_000


def _positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")


def _reader(path: str | Path, max_file_bytes: int = DEFAULT_MAX_PDF_BYTES):
    if PdfReader is None:
        raise ImportError("pypdf is required. Install with: pip install pypdf")
    _positive_int("max_file_bytes", max_file_bytes)
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"PDF not found: {source}")
    size = source.stat().st_size
    if size > max_file_bytes:
        raise ValueError(f"PDF exceeds the {max_file_bytes}-byte input limit")
    try:
        reader = PdfReader(str(source), strict=False)
        if reader.is_encrypted:
            raise ValueError("Encrypted PDFs are not supported; provide an unencrypted PDF")
        return reader
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Unable to read PDF {source}: {exc}") from exc


def extract_text_from_pdf(
    path: str | Path,
    max_pages: Optional[int] = DEFAULT_MAX_PDF_PAGES,
    max_file_bytes: int = DEFAULT_MAX_PDF_BYTES,
    max_chars: int = DEFAULT_MAX_TEXT_CHARS,
) -> str:
    """Extract text within bounded file, page and character budgets.

    max_pages=None uses the safe default page cap rather than allowing an
    unbounded extraction. A smaller explicit value extracts only that prefix.
    """
    if max_pages is None:
        max_pages = DEFAULT_MAX_PDF_PAGES
    _positive_int("max_pages", max_pages)
    _positive_int("max_file_bytes", max_file_bytes)
    _positive_int("max_chars", max_chars)
    reader = _reader(path, max_file_bytes=max_file_bytes)
    try:
        page_count = len(reader.pages)
        pages_to_extract = reader.pages[:max_pages]
    except Exception as exc:
        raise ValueError(f"Unable to read PDF page tree: {exc}") from exc

    texts: List[str] = []
    total_chars = 0
    for index, page in enumerate(pages_to_extract, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            texts.append(f"--- Page {index} (extraction error: {type(exc).__name__}) ---")
            continue
        cleaned = text.strip()
        if not cleaned:
            continue
        total_chars += len(cleaned)
        if total_chars > max_chars:
            raise ValueError(f"PDF extracted text exceeds the {max_chars}-character limit")
        texts.append(f"--- Page {index} ---\n{cleaned}")
    if page_count > max_pages:
        texts.append(
            f"[Extraction stopped at configured page limit: {max_pages} of {page_count} pages]"
        )
    return "\n\n".join(texts) if texts else "(No extractable text found)"


def get_pdf_page_count(
    path: str | Path, max_file_bytes: int = DEFAULT_MAX_PDF_BYTES
) -> int:
    """Return the page count for a PDF within the input file-size limit."""
    try:
        return len(_reader(path, max_file_bytes=max_file_bytes).pages)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Unable to read PDF page count: {exc}") from exc
