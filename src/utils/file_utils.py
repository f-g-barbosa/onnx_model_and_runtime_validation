"""
File system utilities for the validation pipeline.

Provides common file operations, validation, and path handling.
"""

from pathlib import Path
from typing import List, Optional
import json
import csv
from datetime import datetime

from src.core.exceptions import InputDataError


def list_image_files(directory: Path, extensions: Optional[List[str]] = None) -> List[Path]:
    """
    List all image files in a directory.

    Args:
        directory: Directory to search.
        extensions: Image file extensions to look for (default: jpg, png, jpeg).

    Returns:
        List of image file paths.

    Raises:
        InputDataError: If directory doesn't exist.
    """
    if not directory.exists():
        raise InputDataError(f"Directory not found: {directory}")

    if not directory.is_dir():
        raise InputDataError(f"Path is not a directory: {directory}")

    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png", ".bmp"]

    image_files = []
    for ext in extensions:
        image_files.extend(directory.glob(f"*{ext}"))
        image_files.extend(directory.glob(f"*{ext.upper()}"))

    return sorted(image_files)


def save_json_report(data: dict, output_path: Path, pretty: bool = True) -> None:
    """
    Save data as JSON report.

    Args:
        data: Dictionary to save.
        output_path: Output file path.
        pretty: Whether to pretty-print JSON.

    Raises:
        InputDataError: If write fails.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2 if pretty else None, default=str)
    except Exception as e:
        raise InputDataError(f"Failed to save JSON report to {output_path}: {e}")


def load_json_report(input_path: Path) -> dict:
    """
    Load JSON report.

    Args:
        input_path: Input file path.

    Returns:
        Parsed JSON data.

    Raises:
        InputDataError: If read fails.
    """
    try:
        with open(input_path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise InputDataError(f"Failed to load JSON report from {input_path}: {e}")


def save_csv_report(data: List[dict], output_path: Path, fieldnames: Optional[List[str]] = None) -> None:
    """
    Save data as CSV report.

    Args:
        data: List of dictionaries to save.
        output_path: Output file path.
        fieldnames: CSV column names (auto-detected if None).

    Raises:
        InputDataError: If write fails.
    """
    if not data:
        return

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fields = fieldnames or list(data[0].keys())

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)
    except Exception as e:
        raise InputDataError(f"Failed to save CSV report to {output_path}: {e}")


def ensure_directory(directory: Path, description: str = "") -> Path:
    """
    Ensure directory exists, creating if necessary.

    Args:
        directory: Directory path.
        description: Description for error messages.

    Returns:
        Directory path.

    Raises:
        InputDataError: If directory creation fails.
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
        return directory
    except Exception as e:
        msg = f"Failed to create {description or 'directory'}: {e}"
        raise InputDataError(msg)


def get_file_size_mb(file_path: Path) -> float:
    """Get file size in MB."""
    return file_path.stat().st_size / (1024 * 1024)


def validate_file_exists(file_path: Path, description: str = "") -> Path:
    """
    Validate that a file exists.

    Args:
        file_path: File path to check.
        description: Description for error messages.

    Returns:
        File path if it exists.

    Raises:
        InputDataError: If file doesn't exist.
    """
    if not file_path.exists():
        msg = f"{description or 'File'} not found: {file_path}"
        raise InputDataError(msg)
    return file_path
