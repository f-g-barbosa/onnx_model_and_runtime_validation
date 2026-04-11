"""
Core type definitions for the validation pipeline.

This module provides shared data structures used throughout the system.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime


@dataclass
class ModelMetadata:
    """Metadata about an ONNX model."""
    model_name: str
    model_version: str
    model_path: Path
    framework: str = "ONNX"
    input_shape: Optional[tuple] = None
    output_names: List[str] = field(default_factory=list)
    provider: str = "CPUExecutionProvider"
    onboarded_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ImageMetadata:
    """Metadata about an image processed by the pipeline."""
    image_path: Path
    image_name: str
    original_shape: tuple  # (height, width, channels)
    resized_shape: tuple  # (height, width, channels)
    processed_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class InferenceResult:
    """Result of a single image inference."""
    image_metadata: ImageMetadata
    raw_outputs: List[Any]
    inference_time_ms: float
    model_metadata: ModelMetadata
    batch_id: Optional[str] = None


@dataclass
class DetectionBox:
    """A single detection bounding box."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    class_id: int
    score: float
    class_name: str = ""


@dataclass
class ParsedDetection:
    """Parsed detection results from model outputs."""
    boxes: List[DetectionBox]
    num_detections: int
    raw_scores: List[float] = field(default_factory=list)
    raw_classes: List[int] = field(default_factory=list)


@dataclass
class ValidationMetrics:
    """Metrics from validating a single image."""
    image_name: str
    num_detections: int
    max_score: float
    min_score: float
    detected_classes: List[int]
    passed_threshold: bool
    anomalies: List[str] = field(default_factory=list)


@dataclass
class BatchValidationSummary:
    """Summary of validation for an entire batch."""
    batch_id: str
    total_images: int
    successful: int
    failed: int
    avg_num_detections: float
    avg_max_score: float
    anomalies_detected: List[str] = field(default_factory=list)
    validation_time_ms: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ComparisonResult:
    """Result of comparing baseline vs candidate model."""
    baseline_model: str
    candidate_model: str
    baseline_metrics: Dict[str, float]
    candidate_metrics: Dict[str, float]
    deltas: Dict[str, float]
    regression_detected: bool
    improvement_detected: bool


@dataclass
class PromotionPolicy:
    """Policy rules for promoting a model."""
    model_name: str
    min_accuracy: float
    max_latency_ms: float
    regression_tolerance: float
    manual_review_required: bool
    approval_required_roles: List[str] = field(default_factory=list)


@dataclass
class PromotionRecommendation:
    """Recommendation result for model promotion."""
    model_name: str
    recommendable: bool
    reason: str
    policy_check_results: Dict[str, bool] = field(default_factory=dict)
    requires_manual_review: bool = False
    audit_referral: str = ""
