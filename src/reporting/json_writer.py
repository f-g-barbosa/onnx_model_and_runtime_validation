"""
JSON report writing utilities.

Centralizes JSON output handling.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import json

from src.core.exceptions import ReportGenerationError


def write_json_report(
    data: Dict[str, Any],
    output_path: Path,
    pretty: bool = True,
) -> None:
    """
    Write JSON report to file.

    Args:
        data: Dictionary to save.
        output_path: Output file path.
        pretty: Whether to pretty-print.

    Raises:
        ReportGenerationError: If write fails.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            if pretty:
                json.dump(data, f, indent=2, default=str)
            else:
                json.dump(data, f, default=str)
    except Exception as e:
        raise ReportGenerationError(f"Failed to write JSON report: {e}")


def read_json_report(input_path: Path) -> Dict[str, Any]:
    """
    Read JSON report from file.

    Args:
        input_path: Input file path.

    Returns:
        Parsed JSON data.

    Raises:
        ReportGenerationError: If read fails.
    """
    try:
        with open(input_path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise ReportGenerationError(f"Failed to read JSON report: {e}")
