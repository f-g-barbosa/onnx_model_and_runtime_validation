"""
Runtime validation.

Monitors runtime health and performance consistency.
"""

from typing import List, Optional
from statistics import mean, stdev

from src.core.exceptions import ValidationError
from src.logging_utils.logger import StructuredLogger


class RuntimeValidator:
    """Validates runtime performance and consistency."""

    def __init__(self, logger: Optional[StructuredLogger] = None):
        """
        Initialize runtime validator.

        Args:
            logger: Optional logger instance.
        """
        self.logger = logger

    def check_inference_time_stability(
        self,
        inference_times: List[float],
        tolerance_pct: float = 20.0,
    ) -> bool:
        """
        Check if inference times are stable (not varying too much).

        Args:
            inference_times: List of inference times in milliseconds.
            tolerance_pct: Maximum allowed deviation percentage.

        Returns:
            True if times are stable.
        """
        if len(inference_times) < 2:
            return True

        mean_time = mean(inference_times)
        std_dev = stdev(inference_times) if len(inference_times) > 1 else 0

        # Calculate coefficient of variation
        if mean_time > 0:
            cv = (std_dev / mean_time) * 100
        else:
            cv = 0

        is_stable = cv <= tolerance_pct

        if self.logger:
            self.logger.info(
                "Inference time stability check",
                mean_time_ms=mean_time,
                std_dev=std_dev,
                cv_pct=cv,
                is_stable=is_stable,
            )

        return is_stable

    def check_memory_leaks(
        self,
        memory_snapshots: List[float],
    ) -> bool:
        """
        Check for potential memory leaks by monitoring memory usage trend.

        Args:
            memory_snapshots: List of memory usage values over time.

        Returns:
            True if no memory leak detected.
        """
        if len(memory_snapshots) < 3:
            return True

        # Simple linear trend analysis
        diffs = [memory_snapshots[i+1] - memory_snapshots[i] for i in range(len(memory_snapshots) - 1)]
        avg_increase = mean(diffs)

        # If memory keeps increasing significantly, might indicate leak
        threshold_mb_per_step = 50.0
        has_leak = avg_increase > threshold_mb_per_step

        if self.logger:
            self.logger.warning(
                f"Memory leak check: {'POTENTIAL LEAK' if has_leak else 'OK'}",
                avg_increase_per_step_mb=avg_increase,
            )

        return not has_leak
