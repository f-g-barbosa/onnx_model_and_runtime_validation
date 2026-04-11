"""
Schema dataclasses for configuration and request/response structures.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path


@dataclass
class RuntimeConfig:
    """Runtime configuration for ONNX inference."""
    model_path: Path
    providers: List[str] = field(default_factory=lambda: ["CPUExecutionProvider"])
    input_image_size: tuple = (640, 640)
    verbose: bool = False


@dataclass
class ValidationConfig:
    """Validation thresholds and settings."""
    score_threshold: float = 0.5
    max_detections_per_image: Optional[int] = None
    min_detections_per_image: int = 0
    allowed_classes: Optional[List[int]] = None
    latency_threshold_ms: Optional[float] = None
    check_consistency: bool = True


@dataclass
class PromotionPolicyConfig:
    """Policy configuration for model promotion."""
    min_accuracy_pct: float = 95.0
    max_regression_pct: float = 2.0
    max_latency_ms: Optional[float] = None
    require_human_review: bool = True
    approval_required_roles: List[str] = field(default_factory=lambda: ["model_reviewer"])


@dataclass
class PreprocessingConfig:
    """Image preprocessing configuration."""
    resize_width: int = 640
    resize_height: int = 640
    normalize: bool = False
    normalization_mean: Optional[List[float]] = None
    normalization_std: Optional[List[float]] = None


@dataclass
class OutputConfig:
    """Output and artifact configuration."""
    output_dir: Path
    save_debug_outputs: bool = False
    save_annotated_images: bool = False
    save_batch_report: bool = True
    save_audit_report: bool = True
    debug_dir: Optional[Path] = None
    annotated_dir: Optional[Path] = None


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_level: str = "INFO"
    log_file: Optional[Path] = None
    structured_logs: bool = True
    console_output: bool = True


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""
    runtime: RuntimeConfig
    validation: ValidationConfig
    promotion_policy: PromotionPolicyConfig
    preprocessing: PreprocessingConfig
    output: OutputConfig
    logging: LoggingConfig
    batch_id: str = ""
    dry_run: bool = False
