"""
Configuration loader for YAML-based pipeline configuration.

Handles loading and validating all configuration files.
"""

from pathlib import Path
from typing import Optional
import yaml

from src.core.schemas import (
    RuntimeConfig,
    ValidationConfig,
    PromotionPolicyConfig,
    PreprocessingConfig,
    OutputConfig,
    LoggingConfig,
    PipelineConfig,
)
from src.core.exceptions import ConfigurationError


class ConfigLoader:
    """Loads and validates pipeline configuration from YAML files."""

    def __init__(self, config_dir: Path):
        """
        Initialize config loader.

        Args:
            config_dir: Directory containing YAML configuration files.

        Raises:
            ConfigurationError: If config directory doesn't exist.
        """
        self.config_dir = Path(config_dir)
        if not self.config_dir.exists():
            raise ConfigurationError(f"Config directory not found: {self.config_dir}")

    def load_full_config(self, batch_id: str, dry_run: bool = False) -> PipelineConfig:
        """
        Load complete pipeline configuration from all YAML files.

        Args:
            batch_id: Batch identifier for this run.
            dry_run: Whether this is a dry run.

        Returns:
            Complete PipelineConfig object.

        Raises:
            ConfigurationError: If any configuration file is invalid.
        """
        try:
            runtime_cfg = self._load_runtime_config()
            validation_cfg = self._load_validation_config()
            promotion_cfg = self._load_promotion_policy_config()
            preprocessing_cfg = self._load_preprocessing_config()
            output_cfg = self._load_output_config()
            logging_cfg = self._load_logging_config()

            return PipelineConfig(
                runtime=runtime_cfg,
                validation=validation_cfg,
                promotion_policy=promotion_cfg,
                preprocessing=preprocessing_cfg,
                output=output_cfg,
                logging=logging_cfg,
                batch_id=batch_id,
                dry_run=dry_run,
            )
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {str(e)}")

    def _load_runtime_config(self) -> RuntimeConfig:
        """Load runtime configuration."""
        config_file = self.config_dir / "runtime.yaml"
        data = self._load_yaml(config_file)

        return RuntimeConfig(
            model_path=Path(data["model_path"]),
            providers=data.get("providers", ["CPUExecutionProvider"]),
            input_image_size=tuple(data.get("input_image_size", [640, 640])),
            verbose=data.get("verbose", False),
        )

    def _load_validation_config(self) -> ValidationConfig:
        """Load validation configuration."""
        config_file = self.config_dir / "validation.yaml"
        data = self._load_yaml(config_file)

        return ValidationConfig(
            score_threshold=float(data.get("score_threshold", 0.5)),
            max_detections_per_image=data.get("max_detections_per_image"),
            min_detections_per_image=int(data.get("min_detections_per_image", 0)),
            allowed_classes=data.get("allowed_classes"),
            latency_threshold_ms=data.get("latency_threshold_ms"),
            check_consistency=data.get("check_consistency", True),
        )

    def _load_promotion_policy_config(self) -> PromotionPolicyConfig:
        """Load promotion policy configuration."""
        config_file = self.config_dir / "promotion_policy.yaml"
        data = self._load_yaml(config_file)

        return PromotionPolicyConfig(
            min_accuracy_pct=float(data.get("min_accuracy_pct", 95.0)),
            max_regression_pct=float(data.get("max_regression_pct", 2.0)),
            max_latency_ms=data.get("max_latency_ms"),
            require_human_review=data.get("require_human_review", True),
            approval_required_roles=data.get("approval_required_roles", ["model_reviewer"]),
        )

    def _load_preprocessing_config(self) -> PreprocessingConfig:
        """Load preprocessing configuration."""
        config_file = self.config_dir / "preprocessing.yaml"
        if not config_file.exists():
            return PreprocessingConfig()

        data = self._load_yaml(config_file)
        return PreprocessingConfig(
            resize_width=int(data.get("resize_width", 640)),
            resize_height=int(data.get("resize_height", 640)),
            normalize=data.get("normalize", False),
            normalization_mean=data.get("normalization_mean"),
            normalization_std=data.get("normalization_std"),
        )

    def _load_output_config(self) -> OutputConfig:
        """Load output configuration."""
        config_file = self.config_dir / "output.yaml"
        if not config_file.exists():
            return OutputConfig(output_dir=Path("outputs"))

        data = self._load_yaml(config_file)
        output_dir = Path(data.get("output_dir", "outputs"))

        return OutputConfig(
            output_dir=output_dir,
            save_debug_outputs=data.get("save_debug_outputs", False),
            save_annotated_images=data.get("save_annotated_images", False),
            save_batch_report=data.get("save_batch_report", True),
            save_audit_report=data.get("save_audit_report", True),
            debug_dir=Path(data["debug_dir"]) if "debug_dir" in data else None,
            annotated_dir=Path(data["annotated_dir"]) if "annotated_dir" in data else None,
        )

    def _load_logging_config(self) -> LoggingConfig:
        """Load logging configuration."""
        config_file = self.config_dir / "logging.yaml"
        if not config_file.exists():
            return LoggingConfig()

        data = self._load_yaml(config_file)
        return LoggingConfig(
            log_level=data.get("log_level", "INFO"),
            log_file=Path(data["log_file"]) if "log_file" in data else None,
            structured_logs=data.get("structured_logs", True),
            console_output=data.get("console_output", True),
        )

    @staticmethod
    def _load_yaml(file_path: Path) -> dict:
        """
        Load and parse a YAML file.

        Args:
            file_path: Path to YAML file.

        Returns:
            Parsed YAML content as dict.

        Raises:
            ConfigurationError: If file cannot be loaded or parsed.
        """
        if not file_path.exists():
            raise ConfigurationError(f"Configuration file not found: {file_path}")

        try:
            with open(file_path, "r") as f:
                content = yaml.safe_load(f)
                if content is None:
                    return {}
                return content
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Failed to parse YAML file {file_path}: {str(e)}")
        except Exception as e:
            raise ConfigurationError(f"Failed to read configuration file {file_path}: {str(e)}")
