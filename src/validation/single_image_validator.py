"""
Single image validation.

Validates individual image inference results against policy.
"""

from typing import List, Optional
from src.core.types import InferenceResult, ValidationMetrics, ParsedDetection
from src.core.enums import AnomalyType
from src.core.schemas import ValidationConfig
from src.core.exceptions import ValidationError
from src.logging_utils.logger import StructuredLogger


class SingleImageValidator:
    """Validates single image inference results."""

    def __init__(self, config: ValidationConfig, logger: Optional[StructuredLogger] = None):
        """
        Initialize single image validator.

        Args:
            config: Validation configuration.
            logger: Optional logger instance.
        """
        self.config = config
        self.logger = logger

    def validate(
        self,
        image_name: str,
        parsed_detection: ParsedDetection,
        inference_time_ms: float,
    ) -> ValidationMetrics:
        """
        Validate single image inference result.

        Args:
            image_name: Name of the image.
            parsed_detection: Parsed detection results.
            inference_time_ms: Inference time in milliseconds.

        Returns:
            ValidationMetrics with results and anomalies.
        """
        anomalies: List[str] = []
        num_detections = len(parsed_detection.boxes)

        # Check minimum detections
        if num_detections < self.config.min_detections_per_image:
            anomalies.append(AnomalyType.NO_DETECTIONS.value)

        # Check maximum detections
        if (
            self.config.max_detections_per_image
            and num_detections > self.config.max_detections_per_image
        ):
            anomalies.append(AnomalyType.EXCESSIVE_DETECTIONS.value)

        # Check class constraints
        if self.config.allowed_classes:
            for box in parsed_detection.boxes:
                if box.class_id not in self.config.allowed_classes:
                    anomalies.append(AnomalyType.CLASS_DRIFT.value)
                    break

        # Check latency
        if (
            self.config.latency_threshold_ms
            and inference_time_ms > self.config.latency_threshold_ms
        ):
            anomalies.append(AnomalyType.LATENCY_SPIKE.value)

        # Calculate scores
        scores = [box.score for box in parsed_detection.boxes]
        max_score = max(scores) if scores else 0.0
        min_score = min(scores) if scores else 0.0

        # Check low confidence
        if max_score < self.config.score_threshold:
            anomalies.append(AnomalyType.LOW_CONFIDENCE.value)

        passed = len(anomalies) == 0

        metrics = ValidationMetrics(
            image_name=image_name,
            num_detections=num_detections,
            max_score=max_score,
            min_score=min_score,
            detected_classes=list(set(box.class_id for box in parsed_detection.boxes)),
            passed_threshold=passed,
            anomalies=anomalies,
        )

        if self.logger:
            self.logger.debug(
                f"Image validation: {image_name}",
                passed=passed,
                num_detections=num_detections,
                max_score=max_score,
                anomalies=anomalies,
            )

        return metrics
