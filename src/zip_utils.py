"""ZIP archive utilities for upload / download."""

import zipfile
from pathlib import Path
from typing import List, Dict, Optional
import tempfile
import shutil


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".csv", ".json", ".py", ".html"}


def list_zip_contents(zip_path: str | Path) -> List[Dict[str, str | int]]:
    """List files inside a ZIP archive."""
    zip_path = Path(zip_path)
    contents = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if not info.is_dir():
                contents.append({
                    "name": info.filename,
                    "size": info.file_size,
                    "compress_size": info.compress_size,
                })
    return contents


def extract_supported_files(
    zip_path: str | Path,
    dest_dir: Optional[str | Path] = None,
    max_files: int = 50,
) -> List[Path]:
    """Extract supported text/PDF files from ZIP into a directory.

    Returns list of extracted file paths.
    """
    zip_path = Path(zip_path)
    if dest_dir is None:
        dest_dir = Path(tempfile.mkdtemp(prefix="llm_zip_"))
    else:
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

    extracted: List[Path] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name
            ext = Path(name).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue
            if len(extracted) >= max_files:
                break
            target = dest_dir / name
            # Avoid path traversal
            if ".." in info.filename or info.filename.startswith("/"):
                continue
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(target)
    return extracted


def create_results_zip(
    files: Dict[str, str | bytes],
    output_path: str | Path,
) -> Path:
    """Create a ZIP containing result files.

    Args:
        files: Mapping of filename -> content (str or bytes).
        output_path: Where to write the ZIP.

    Returns:
        Path to the created ZIP.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            # Sanitize name
            safe_name = Path(name).name
            if isinstance(content, str):
                zf.writestr(safe_name, content.encode("utf-8"))
            else:
                zf.writestr(safe_name, content)
    return output_path


def cleanup_dir(path: str | Path) -> None:
    """Remove a temporary directory tree."""
    p = Path(path)
    if p.exists() and p.is_dir():
        shutil.rmtree(p, ignore_errors=True)
