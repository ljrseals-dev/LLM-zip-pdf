"""ZIP archive utilities for upload / download.

Extraction is deliberately bounded: archives are untrusted input and may contain
path traversal entries, duplicate paths, symlinks, encrypted files, or bombs.
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
DEFAULT_MAX_ARCHIVE_ENTRIES = 10000


def _safe_member_path(name: str) -> Optional[Path]:
    """Return a safe relative path, or None for absolute/traversal names."""
    normalized = name.replace("\\", "/")
    member = PurePosixPath(normalized)
    if member.is_absolute() or not member.parts or any(p in ("", ".", "..") for p in member.parts):
        return None
    if member.parts[0].endswith(":"):
        return None
    return Path(*member.parts)


def _validate_regular_member(info: zipfile.ZipInfo) -> None:
    """Reject encrypted and non-regular Unix entries before opening them."""
    if info.flag_bits & 0x1:
        raise ValueError(f"Encrypted ZIP members are not supported: {info.filename}")
    mode = info.external_attr >> 16
    kind = stat.S_IFMT(mode)
    if kind not in (0, stat.S_IFREG):
        raise ValueError(f"Non-regular ZIP member is not supported: {info.filename}")


def _safe_destination(destination: Path, relative: Path) -> Path:
    """Resolve a member below destination without following existing symlinks."""
    if destination.is_symlink():
        raise ValueError("ZIP destination must not be a symbolic link")
    root = destination.resolve()
    target = destination / relative
    current = destination
    for part in relative.parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"ZIP destination contains a symbolic link: {current}")
    if target.is_symlink():
        raise ValueError(f"ZIP destination is a symbolic link: {target}")
    resolved_parent = target.parent.resolve()
    if resolved_parent != root and root not in resolved_parent.parents:
        raise ValueError(f"ZIP member escapes destination: {relative}")
    return target


def list_zip_contents(zip_path: str | Path,
                      max_entries: int = DEFAULT_MAX_ARCHIVE_ENTRIES) -> List[Dict[str, str | int]]:
    """List regular files without extracting; cap metadata work for hostile archives."""
    if max_entries < 1:
        raise ValueError("max_entries must be positive")
    contents = []
    with zipfile.ZipFile(Path(zip_path), "r") as zf:
        infos = zf.infolist()
        if len(infos) > max_entries:
            raise ValueError(f"ZIP exceeds the {max_entries}-entry listing limit")
        for info in infos:
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
    try:
        if destination.is_symlink():
            raise ValueError("ZIP destination must not be a symbolic link")
        destination.mkdir(parents=True, exist_ok=True)
        extracted: List[Path] = []
        total_bytes = 0
        with zipfile.ZipFile(zip_path, "r") as zf:
            infos = zf.infolist()
            if len(infos) > DEFAULT_MAX_ARCHIVE_ENTRIES:
                raise ValueError(f"ZIP exceeds the {DEFAULT_MAX_ARCHIVE_ENTRIES}-entry limit")
            for info in infos:
                if info.is_dir():
                    continue
                relative = _safe_member_path(info.filename)
                if relative is None:
                    continue
                if relative.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue
                _validate_regular_member(info)
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
                target = _safe_destination(destination, relative)
                target.parent.mkdir(parents=True, exist_ok=True)
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
    normalized = []
    seen = set()
    for name, content in files.items():
        safe_name = Path(str(name).replace("\\\\", "/")).name
        if safe_name in ("", ".", ".."):
            raise ValueError(f"Invalid result filename: {name!r}")
        if safe_name in seen:
            raise ValueError(f"Duplicate result filename after flattening: {safe_name}")
        if not isinstance(content, (str, bytes)):
            raise TypeError(f"Result content for {safe_name!r} must be str or bytes")
        seen.add(safe_name)
        normalized.append((safe_name, content))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for safe_name, content in normalized:
            zf.writestr(safe_name, content.encode("utf-8") if isinstance(content, str) else content)
    return output_path

def cleanup_dir(path: str | Path) -> None:
    """Remove a temporary directory tree."""
    candidate = Path(path)
    if candidate.exists() and candidate.is_dir():
        shutil.rmtree(candidate, ignore_errors=True)
