"""
Flagged sample selection for human review.

Identifies and selects images for human review based on validation results.
"""

from typing import List
from src.core.types import ValidationMetrics


class FlaggedSampleSelector:
    """Selects representative flagged samples for human review."""

    def __init__(self):
        """Initialize sample selector."""
        pass

    def select_samples(
        self,
        image_metrics: List[ValidationMetrics],
        num_samples: int = 10,
    ) -> List[str]:
        """
        Select representative flagged samples.

        Args:
            image_metrics: List of image validation metrics.
            num_samples: Number of samples to select.

        Returns:
            List of selected image names.
        """
        # Get all problematic images
        problematic = [m for m in image_metrics if not m.passed_threshold]

        if len(problematic) <= num_samples:
            return [m.image_name for m in problematic]

        # Group by anomaly type and select representative samples
        selected = []
        anomaly_types = {}

        for m in problematic:
            for anomaly in m.anomalies:
                if anomaly not in anomaly_types:
                    anomaly_types[anomaly] = []
                anomaly_types[anomaly].append(m.image_name)

        # Select samples from each anomaly type
        for anomaly, images in anomaly_types.items():
            num_for_type = max(1, num_samples // len(anomaly_types))
            # Select ones with lowest scores for this anomaly
            selected.extend(images[:num_for_type])

        return selected[:num_samples]

    def stratify_samples(
        self,
        image_metrics: List[ValidationMetrics],
        num_samples: int = 10,
    ) -> dict:
        """
        Select samples stratified by confidence score ranges.

        Args:
            image_metrics: List of image validation metrics.
            num_samples: Total number of samples to select.

        Returns:
            Dictionary mapping confidence range to image names.
        """
        # Sort by max score
        sorted_metrics = sorted(image_metrics, key=lambda m: m.max_score)

        # Divide into bins
        num_bins = 4
        bin_size = len(sorted_metrics) // num_bins
        samples_per_bin = num_samples // num_bins

        result = {}
        for i in range(num_bins):
            start_idx = i * bin_size
            end_idx = start_idx + bin_size if i < num_bins - 1 else len(sorted_metrics)

            bin_metrics = sorted_metrics[start_idx:end_idx]
            bin_images = [m.image_name for m in bin_metrics[:samples_per_bin]]

            result[f"bin_{i}"] = bin_images

        return result
