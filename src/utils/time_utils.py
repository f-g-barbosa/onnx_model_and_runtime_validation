"""
Time utilities for the validation pipeline.

Provides timing and duration utilities for performance monitoring.
"""

from datetime import datetime, timedelta
from typing import Optional


class Timer:
    """Context manager for timing operations."""

    def __init__(self, name: str = ""):
        """Initialize timer."""
        self.name = name
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

    def __enter__(self):
        """Enter context."""
        self.start_time = datetime.utcnow()
        return self

    def __exit__(self, *args):
        """Exit context."""
        self.end_time = datetime.utcnow()

    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        if not self.start_time or not self.end_time:
            return 0.0
        delta = self.end_time - self.start_time
        return delta.total_seconds() * 1000


def format_duration(duration_ms: float) -> str:
    """
    Format duration in milliseconds to human-readable string.

    Args:
        duration_ms: Duration in milliseconds.

    Returns:
        Formatted string (e.g., "1m 23s 456ms").
    """
    total_seconds = duration_ms / 1000
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    parts = []
    if hours > 0:
        parts.append(f"{int(hours)}h")
    if minutes > 0:
        parts.append(f"{int(minutes)}m")
    parts.append(f"{seconds:.1f}s")

    return " ".join(parts)


def get_timestamp_str() -> str:
    """Get current UTC timestamp as ISO format string."""
    return datetime.utcnow().isoformat()


def parse_iso_timestamp(timestamp_str: str) -> datetime:
    """Parse ISO format timestamp string."""
    return datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
