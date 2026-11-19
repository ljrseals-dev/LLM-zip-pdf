"""Bounded ZIP archive utilities for untrusted document uploads."""
import os
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


def _positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")


def _safe_member_path(name: str) -> Optional[Path]:
    """Return a safe relative path, or None for absolute/traversal names."""
    if not isinstance(name, str) or not name or "\x00" in name:
        return None
    normalized = name.replace("\\", "/")
    raw_parts = normalized.split("/")
    if normalized.startswith("/") or any(p in ("", ".", "..") for p in raw_parts):
        return None
    if raw_parts[0].endswith(":") or any(":" in part for part in raw_parts):
        return None
    member = PurePosixPath(normalized)
    if member.is_absolute() or not member.parts:
        return None
    return Path(*member.parts)


def _validate_regular_member(info: zipfile.ZipInfo) -> None:
    """Reject encrypted entries, symlinks, devices, sockets and malformed dirs."""
    if info.flag_bits & 0x1:
        raise ValueError(f"Encrypted ZIP members are not supported: {info.filename}")
    mode = info.external_attr >> 16
    kind = stat.S_IFMT(mode)
    if info.is_dir():
        if kind not in (0, stat.S_IFDIR):
            raise ValueError(f"Non-directory ZIP entry has directory name: {info.filename}")
    elif kind not in (0, stat.S_IFREG):
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
        if current.exists() and not current.is_dir():
            raise ValueError(f"ZIP destination path is not a directory: {current}")
    if target.is_symlink():
        raise ValueError(f"ZIP destination is a symbolic link: {target}")
    resolved_parent = target.parent.resolve()
    if resolved_parent != root and root not in resolved_parent.parents:
        raise ValueError(f"ZIP member escapes destination: {relative}")
    return target


def list_zip_contents(zip_path: str | Path,
                      max_entries: int = DEFAULT_MAX_ARCHIVE_ENTRIES) -> List[Dict[str, str | int]]:
    """List archive entries without extraction; cap metadata work."""
    _positive_int("max_entries", max_entries)
    source = Path(zip_path)
    if not source.is_file():
        raise FileNotFoundError(f"ZIP not found: {source}")
    contents = []
    with zipfile.ZipFile(source, "r") as zf:
        infos = zf.infolist()
        if len(infos) > max_entries:
            raise ValueError(f"ZIP exceeds the {max_entries}-entry listing limit")
        for info in infos:
            _validate_regular_member(info)
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
    """Safely extract supported files, preserving paths and enforcing hard limits.

    Internally-created destinations are removed on failure. For caller-owned
    destinations, only files and directories created by this call are rolled back.
    """
    _positive_int("max_files", max_files)
    _positive_int("max_file_bytes", max_file_bytes)
    _positive_int("max_total_bytes", max_total_bytes)
    _positive_int("max_compression_ratio", max_compression_ratio)
    source = Path(zip_path)
    if not source.is_file():
        raise FileNotFoundError(f"ZIP not found: {source}")
    owns_destination = dest_dir is None
    destination = Path(tempfile.mkdtemp(prefix="llm_zip_")) if owns_destination else Path(dest_dir)
    created_files: List[Path] = []
    created_dirs: List[Path] = []
    try:
        if destination.is_symlink():
            raise ValueError("ZIP destination must not be a symbolic link")
        if not destination.exists():
            destination.mkdir(parents=True)
        if not destination.is_dir():
            raise NotADirectoryError(f"ZIP destination is not a directory: {destination}")
        with zipfile.ZipFile(source, "r") as zf:
            infos = zf.infolist()
            if len(infos) > DEFAULT_MAX_ARCHIVE_ENTRIES:
                raise ValueError(f"ZIP exceeds the {DEFAULT_MAX_ARCHIVE_ENTRIES}-entry limit")
            candidates = []
            seen = set()
            total_declared = 0
            for info in infos:
                _validate_regular_member(info)
                if info.is_dir():
                    continue
                relative = _safe_member_path(info.filename)
                if relative is None:
                    continue
                if relative.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue
                if relative in seen:
                    raise ValueError(f"Duplicate ZIP destination: {info.filename}")
                seen.add(relative)
                if len(candidates) >= max_files:
                    raise ValueError(f"ZIP exceeds the {max_files}-file extraction limit")
                if info.file_size < 0 or info.compress_size < 0:
                    raise ValueError(f"Invalid ZIP member sizes: {info.filename}")
                if info.file_size > max_file_bytes:
                    raise ValueError(f"ZIP member exceeds per-file size limit: {info.filename}")
                if info.file_size and (info.compress_size == 0 or
                    info.file_size / info.compress_size > max_compression_ratio):
                    raise ValueError(f"Suspicious compression ratio: {info.filename}")
                total_declared += info.file_size
                if total_declared > max_total_bytes:
                    raise ValueError("ZIP exceeds total uncompressed size limit")
                candidates.append((info, relative))

            extracted: List[Path] = []
            actual_total = 0
            for info, relative in candidates:
                target = _safe_destination(destination, relative)
                parent = target.parent
                missing = []
                cursor = parent
                while cursor != destination and not cursor.exists():
                    missing.append(cursor)
                    cursor = cursor.parent
                for directory in reversed(missing):
                    directory.mkdir()
                    created_dirs.append(directory)
                if target.exists() or target.is_symlink():
                    raise ValueError(f"ZIP destination already exists: {target}")
                written = 0
                with zf.open(info, "r") as archive_source, target.open("xb") as output:
                    created_files.append(target)
                    while True:
                        block = archive_source.read(1024 * 1024)
                        if not block:
                            break
                        written += len(block)
                        actual_total += len(block)
                        if written > max_file_bytes or actual_total > max_total_bytes:
                            raise ValueError(f"ZIP member exceeded extraction limit: {info.filename}")
                        output.write(block)
                if written != info.file_size:
                    raise zipfile.BadZipFile(f"Size mismatch while extracting {info.filename}")
                extracted.append(target)
            return extracted
    except Exception:
        if owns_destination:
            shutil.rmtree(destination, ignore_errors=True)
        else:
            for path in reversed(created_files):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            for path in reversed(created_dirs):
                try:
                    path.rmdir()
                except OSError:
                    pass
        raise


def create_results_zip(files: Dict[str, str | bytes], output_path: str | Path) -> Path:
    """Atomically create a ZIP with safe, unique flat filenames."""
    if not isinstance(files, dict):
        raise TypeError("files must be a dictionary")
    output_path = Path(output_path)
    if output_path.is_symlink():
        raise ValueError("ZIP output path must not be a symbolic link")
    normalized = []
    seen = set()
    for name, content in files.items():
        if not isinstance(name, str) or not name or "\x00" in name:
            raise ValueError(f"Invalid result filename: {name!r}")
        safe_name = name.replace("\\", "/").rsplit("/", 1)[-1]
        if safe_name in ("", ".", ".."):
            raise ValueError(f"Invalid result filename: {name!r}")
        if safe_name in seen:
            raise ValueError(f"Duplicate result filename after flattening: {safe_name}")
        if not isinstance(content, (str, bytes)):
            raise TypeError(f"Result content for {safe_name!r} must be str or bytes")
        seen.add(safe_name)
        normalized.append((safe_name, content))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".tmp",
                                     dir=str(output_path.parent))
    os.close(fd)
    try:
        with zipfile.ZipFile(temp_name, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for safe_name, content in normalized:
                zf.writestr(safe_name, content.encode("utf-8") if isinstance(content, str) else content)
        if output_path.is_symlink():
            raise ValueError("ZIP output path became a symbolic link")
        os.replace(temp_name, output_path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise
    return output_path


def cleanup_dir(path: str | Path) -> None:
    """Remove a temporary directory tree without following a symlink."""
    candidate = Path(path)
    if candidate.is_symlink():
        candidate.unlink()
    elif candidate.is_dir():
        shutil.rmtree(candidate, ignore_errors=True)
