"""
Batch validation.

Validates entire batch of images and produces summary statistics.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
from statistics import mean, stdev
from datetime import datetime

from src.core.types import ValidationMetrics, BatchValidationSummary
from src.core.schemas import ValidationConfig
from src.core.exceptions import ValidationError
from src.logging_utils.logger import StructuredLogger
from src.utils.time_utils import Timer


class BatchValidator:
    """Validates entire batch of images."""

    def __init__(self, config: ValidationConfig, logger: Optional[StructuredLogger] = None):
        """
        Initialize batch validator.

        Args:
            config: Validation configuration.
            logger: Optional logger instance.
        """
        self.config = config
        self.logger = logger

    def summarize_batch(
        self,
        batch_id: str,
        image_metrics: List[ValidationMetrics],
        batch_validation_time_ms: float,
    ) -> BatchValidationSummary:
        """
        Create summary of batch validation.

        Args:
            batch_id: Batch identifier.
            image_metrics: List of validation metrics for each image.
            batch_validation_time_ms: Total time spent validating batch.

        Returns:
            BatchValidationSummary with batch stats.
        """
        total_images = len(image_metrics)
        successful = sum(1 for m in image_metrics if m.passed_threshold)
        failed = total_images - successful

        # Aggregate metrics
        all_detections = [m.num_detections for m in image_metrics]
        all_scores = [m.max_score for m in image_metrics if m.max_score > 0]

        avg_detections = mean(all_detections) if all_detections else 0.0
        avg_max_score = mean(all_scores) if all_scores else 0.0

        # Collect all anomalies
        all_anomalies = []
        for m in image_metrics:
            all_anomalies.extend(m.anomalies)

        summary = BatchValidationSummary(
            batch_id=batch_id,
            total_images=total_images,
            successful=successful,
            failed=failed,
            avg_num_detections=avg_detections,
            avg_max_score=avg_max_score,
            anomalies_detected=list(set(all_anomalies)),
            validation_time_ms=batch_validation_time_ms,
        )

        if self.logger:
            self.logger.info(
                f"Batch validation summary: {batch_id}",
                total_images=total_images,
                successful=successful,
                failed=failed,
                avg_detections=avg_detections,
                avg_max_score=avg_max_score,
            )

        return summary

    def get_problematic_images(
        self,
        image_metrics: List[ValidationMetrics],
        top_k: int = 5,
    ) -> List[str]:
        """
        Get list of images with lowest scores or most anomalies.

        Args:
            image_metrics: List of validation metrics.
            top_k: Number of problematic images to return.

        Returns:
            List of image names, sorted by problematic score.
        """
        scored = [
            (m.image_name, m.max_score, len(m.anomalies))
            for m in image_metrics
        ]

        # Sort by: has anomalies (desc), max_score (asc)
        scored.sort(key=lambda x: (-x[2], x[1]))

        return [name for name, _, _ in scored[:top_k]]
