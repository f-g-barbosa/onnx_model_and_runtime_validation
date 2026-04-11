"""
Structured logging utilities for the validation pipeline.

Provides configurable, audit-friendly logging with both console and file output.
"""

import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Any, Dict

from src.core.schemas import LoggingConfig


class StructuredLogger:
    """Provides structured logging for audit trails and diagnostics."""

    def __init__(self, name: str, config: LoggingConfig):
        """
        Initialize structured logger.

        Args:
            name: Logger name (usually __name__).
            config: LoggingConfig specifying log settings.
        """
        self.name = name
        self.config = config
        self.logger = logging.getLogger(name)

        # Set log level
        log_level = getattr(logging, config.log_level.upper(), logging.INFO)
        self.logger.setLevel(log_level)

        # Clear existing handlers
        self.logger.handlers.clear()

        # Add console handler if enabled
        if config.console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # Add file handler if specified
        if config.log_file:
            config.log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(config.log_file)
            file_handler.setLevel(log_level)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info level message with optional structured data."""
        if kwargs:
            self.logger.info(f"{message} | {self._format_structured(kwargs)}")
        else:
            self.logger.info(message)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug level message with optional structured data."""
        if kwargs:
            self.logger.debug(f"{message} | {self._format_structured(kwargs)}")
        else:
            self.logger.debug(message)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning level message with optional structured data."""
        if kwargs:
            self.logger.warning(f"{message} | {self._format_structured(kwargs)}")
        else:
            self.logger.warning(message)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error level message with optional structured data."""
        if kwargs:
            self.logger.error(f"{message} | {self._format_structured(kwargs)}")
        else:
            self.logger.error(message)

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical level message with optional structured data."""
        if kwargs:
            self.logger.critical(f"{message} | {self._format_structured(kwargs)}")
        else:
            self.logger.critical(message)

    @staticmethod
    def _format_structured(data: Dict[str, Any]) -> str:
        """Format structured data for logging."""
        # Convert non-serializable objects
        clean_data = {}
        for k, v in data.items():
            if isinstance(v, (Path, datetime)):
                clean_data[k] = str(v)
            else:
                clean_data[k] = v

        return json.dumps(clean_data, indent=2)


class AuditLogger:
    """Specialized logger for audit events and policy decisions."""

    def __init__(self, log_dir: Path):
        """
        Initialize audit logger.

        Args:
            log_dir: Directory to store audit logs.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Audit log file
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.audit_log_file = self.log_dir / f"audit_{timestamp}.jsonl"

    def log_event(
        self,
        event_type: str,
        model_name: str,
        status: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log an audit event.

        Args:
            event_type: Type of event (e.g., 'model_onboarded', 'promotion_approved').
            model_name: Name of the model involved.
            status: Status of the event (e.g., 'success', 'failure').
            details: Additional event details.
        """
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "model_name": model_name,
            "status": status,
            "details": details or {},
        }

        # Append to JSONL audit file
        try:
            with open(self.audit_log_file, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            print(f"Failed to write audit log: {e}")

    def log_model_promotion(
        self,
        model_name: str,
        approved: bool,
        reviewer: str,
        policy_results: Dict[str, bool],
        notes: str = "",
    ) -> None:
        """Log a model promotion decision."""
        self.log_event(
            event_type="model_promotion_decision",
            model_name=model_name,
            status="approved" if approved else "rejected",
            details={
                "reviewer": reviewer,
                "policy_results": policy_results,
                "notes": notes,
            },
        )

    def log_validation_run(
        self,
        batch_id: str,
        model_name: str,
        total_images: int,
        successful: int,
        failed: int,
        avg_time_ms: float,
    ) -> None:
        """Log a validation run."""
        self.log_event(
            event_type="validation_run",
            model_name=model_name,
            status="completed",
            details={
                "batch_id": batch_id,
                "total_images": total_images,
                "successful": successful,
                "failed": failed,
                "avg_inference_time_ms": avg_time_ms,
            },
        )
