"""
Baseline vs Candidate comparison.

Compares candidate model performance against baseline.
"""

from typing import Dict, Optional, List
from src.core.types import ComparisonResult
from src.core.exceptions import ValidationError
from src.logging_utils.logger import StructuredLogger


class BaselineComparator:
    """Compares candidate model against baseline model."""

    def __init__(self, logger: Optional[StructuredLogger] = None):
        """
        Initialize comparator.

        Args:
            logger: Optional logger instance.
        """
        self.logger = logger

    def compare_metrics(
        self,
        baseline_name: str,
        candidate_name: str,
        baseline_metrics: Dict[str, float],
        candidate_metrics: Dict[str, float],
        regression_tolerance_pct: float = 2.0,
    ) -> ComparisonResult:
        """
        Compare baseline vs candidate metrics.

        Args:
            baseline_name: Name/version of baseline model.
            candidate_name: Name/version of candidate model.
            baseline_metrics: Baseline model metrics.
            candidate_metrics: Candidate model metrics.
            regression_tolerance_pct: Maximum acceptable regression percentage.

        Returns:
            ComparisonResult with deltas and recommendations.
        """
        deltas = {}
        regressions = []

        for key in baseline_metrics:
            if key not in candidate_metrics:
                continue

            baseline_val = baseline_metrics[key]
            candidate_val = candidate_metrics[key]

            if baseline_val == 0:
                delta_pct = 0.0
            else:
                delta_pct = ((candidate_val - baseline_val) / baseline_val) * 100

            deltas[key] = delta_pct

            # For most metrics, negative delta is better (lower is better)
            # Adjust if metric is "latency" or "error"
            if "latency" in key.lower() or "error" in key.lower():
                # Lower is better
                if delta_pct > regression_tolerance_pct:
                    regressions.append(key)
            else:
                # Higher is usually better (accuracy, precision, etc.)
                if delta_pct < -regression_tolerance_pct:
                    regressions.append(key)

        regression_detected = len(regressions) > 0
        improvement_detected = any(d > 0 for d in deltas.values())

        result = ComparisonResult(
            baseline_model=baseline_name,
            candidate_model=candidate_name,
            baseline_metrics=baseline_metrics,
            candidate_metrics=candidate_metrics,
            deltas=deltas,
            regression_detected=regression_detected,
            improvement_detected=improvement_detected,
        )

        if self.logger:
            self.logger.info(
                "Baseline comparison completed",
                baseline=baseline_name,
                candidate=candidate_name,
                regression_detected=regression_detected,
                improvement_detected=improvement_detected,
                regressions=regressions,
            )

        return result

    def get_recommendation(
        self,
        comparison: ComparisonResult,
        min_accuracy_pct: float = 95.0,
    ) -> str:
        """
        Get promotion recommendation based on comparison.

        Args:
            comparison: Comparison result.
            min_accuracy_pct: Minimum required accuracy.

        Returns:
            Recommendation string.
        """
        if comparison.regression_detected:
            return "REJECT - Performance regression detected"

        if not comparison.improvement_detected:
            return "NO_CHANGE - No significant improvement"

        return "APPROVE - Candidate shows improvement"
