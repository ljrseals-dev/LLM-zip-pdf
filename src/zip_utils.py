"""ZIP archive utilities for upload / download.

Extraction is deliberately bounded: archives are untrusted input and may contain
path traversal entries, duplicate basenames, symlinks, or decompression bombs.
"""
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".csv", ".json", ".py", ".html"}
DEFAULT_MAX_FILES = 50
DEFAULT_MAX_FILE_BYTES = 25 * 1024 * 1024
DEFAULT_MAX_TOTAL_BYTES = 100 * 1024 * 1024
DEFAULT_MAX_RATIO = 200


def _safe_member_path(name: str) -> Optional[Path]:
    """Return a safe relative path, or None for absolute/traversal names."""
    normalized = name.replace("\\", "/")
    member = PurePosixPath(normalized)
    if member.is_absolute() or not member.parts or any(p in ("", ".", "..") for p in member.parts):
        return None
    if member.parts[0].endswith(":"):
        return None
    return Path(*member.parts)


def list_zip_contents(zip_path: str | Path) -> List[Dict[str, str | int]]:
    """List regular files in a ZIP archive without extracting them."""
    contents = []
    with zipfile.ZipFile(Path(zip_path), "r") as zf:
        for info in zf.infolist():
            if not info.is_dir():
                contents.append({"name": info.filename, "size": info.file_size,
                                 "compress_size": info.compress_size})
    return contents


def extract_supported_files(
    zip_path: str | Path,
    dest_dir: Optional[str | Path] = None,
    max_files: int = DEFAULT_MAX_FILES,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    max_compression_ratio: int = DEFAULT_MAX_RATIO,
) -> List[Path]:
    """Safely extract supported files, preserving paths and enforcing size limits.

    A caller-supplied destination is created if needed. A generated temporary
    destination is removed if extraction fails; callers own successful temp dirs.
    """
    if max_files < 1 or max_file_bytes < 1 or max_total_bytes < 1 or max_compression_ratio < 1:
        raise ValueError("ZIP extraction limits must be positive")
    zip_path = Path(zip_path)
    owns_destination = dest_dir is None
    destination = Path(tempfile.mkdtemp(prefix="llm_zip_")) if owns_destination else Path(dest_dir)
    destination.mkdir(parents=True, exist_ok=True)
    extracted: List[Path] = []
    total_bytes = 0
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                relative = _safe_member_path(info.filename)
                if relative is None:
                    continue
                if relative.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue
                mode = info.external_attr >> 16
                if stat.S_ISLNK(mode):
                    continue
                if len(extracted) >= max_files:
                    raise ValueError(f"ZIP exceeds the {max_files}-file extraction limit")
                if info.file_size > max_file_bytes:
                    raise ValueError(f"ZIP member exceeds per-file size limit: {info.filename}")
                total_bytes += info.file_size
                if total_bytes > max_total_bytes:
                    raise ValueError("ZIP exceeds total uncompressed size limit")
                if info.file_size and (info.compress_size == 0 or
                    info.file_size / max(1, info.compress_size) > max_compression_ratio):
                    raise ValueError(f"Suspicious compression ratio: {info.filename}")
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                # Refuse collisions rather than silently overwriting earlier entries.
                if target.exists():
                    raise ValueError(f"Duplicate ZIP destination: {info.filename}")
                written = 0
                with zf.open(info, "r") as source, target.open("xb") as output:
                    while True:
                        block = source.read(1024 * 1024)
                        if not block:
                            break
                        written += len(block)
                        if written > max_file_bytes or written + total_bytes - info.file_size > max_total_bytes:
                            raise ValueError(f"ZIP member exceeded extraction limit: {info.filename}")
                        output.write(block)
                if written != info.file_size:
                    raise zipfile.BadZipFile(f"Size mismatch while extracting {info.filename}")
                extracted.append(target)
        return extracted
    except Exception:
        if owns_destination:
            shutil.rmtree(destination, ignore_errors=True)
        raise


def create_results_zip(files: Dict[str, str | bytes], output_path: str | Path) -> Path:
    """Create a ZIP containing result files using safe flat filenames."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            safe_name = Path(str(name).replace("\\", "/")).name
            if safe_name in ("", ".", ".."):
                raise ValueError(f"Invalid result filename: {name!r}")
            if isinstance(content, str):
                zf.writestr(safe_name, content.encode("utf-8"))
            elif isinstance(content, bytes):
                zf.writestr(safe_name, content)
            else:
                raise TypeError(f"Result content for {safe_name!r} must be str or bytes")
    return output_path


def cleanup_dir(path: str | Path) -> None:
    """Remove a temporary directory tree."""
    candidate = Path(path)
    if candidate.exists() and candidate.is_dir():
        shutil.rmtree(candidate, ignore_errors=True)
