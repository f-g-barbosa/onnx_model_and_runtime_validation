"""
Path utilities for the validation pipeline.

Provides consistent path handling and resolution.
"""

from pathlib import Path
from typing import Optional


def resolve_path(path: str | Path, base_dir: Optional[Path] = None) -> Path:
    """
    Resolve a path, optionally relative to a base directory.

    Args:
        path: Path string or Path object.
        base_dir: Base directory for relative paths.

    Returns:
        Resolved absolute path.
    """
    path_obj = Path(path)

    if path_obj.is_absolute():
        return path_obj

    if base_dir:
        return (base_dir / path_obj).resolve()

    return path_obj.resolve()


def get_relative_path(file_path: Path, base_dir: Path) -> Path:
    """Get relative path from base directory."""
    try:
        return file_path.relative_to(base_dir)
    except ValueError:
        return file_path


def ensure_posix_path(path: Path) -> str:
    """
    Convert path to POSIX format (forward slashes).

    Useful for cross-platform compatibility in reports.
    """
    return path.as_posix()


def get_stem_and_suffix(file_path: Path) -> tuple[str, str]:
    """Get filename stem and suffix separately."""
    return file_path.stem, file_path.suffix
