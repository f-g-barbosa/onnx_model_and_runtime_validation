"""
Enumeration types for the validation pipeline.

Provides standardized status values, states, and classifications.
"""

from enum import Enum


class ValidationStatus(Enum):
    """Status of validation operations."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class ModelStatus(Enum):
    """Status of a model in the governance pipeline."""
    ONBOARDED = "onboarded"
    VALIDATED = "validated"
    BASELINE_SET = "baseline_set"
    CANDIDATE = "candidate"
    APPROVED = "approved"
    DEPRECATED = "deprecated"
    REJECTED = "rejected"


class ImageProcessingStatus(Enum):
    """Status of individual image processing."""
    PENDING = "pending"
    PREPROCESSED = "preprocessed"
    INFERRED = "inferred"
    PARSED = "parsed"
    VALIDATED = "validated"
    ERROR = "error"


class AnomalyType(Enum):
    """Types of anomalies detected during validation."""
    NO_DETECTIONS = "no_detections"
    LOW_CONFIDENCE = "low_confidence"
    EXCESSIVE_DETECTIONS = "excessive_detections"
    CLASS_DRIFT = "class_drift"
    LATENCY_SPIKE = "latency_spike"
    RUNTIME_ERROR = "runtime_error"
    PREPROCESSING_ERROR = "preprocessing_error"


class PromotionGateStatus(Enum):
    """Status of a promotion gate."""
    PASSED = "passed"
    FAILED = "failed"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"


class ReviewDecision(Enum):
    """Human review decision."""
    APPROVED = "approved"
    REJECTED = "rejected"
    PENDING = "pending"


class ExecutionEnvironment(Enum):
    """Environment where the pipeline is running."""
    LOCAL = "local"
    CI_CD = "ci_cd"
    KUBERNETES = "kubernetes"
    BATCH_PROCESSING = "batch_processing"
