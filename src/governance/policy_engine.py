"""
Policy engine for model governance.

Enforces policies for model validation and promotion.
"""

from typing import Dict, Optional, List
from src.core.types import PromotionPolicy
from src.core.schemas import PromotionPolicyConfig
from src.core.exceptions import PolicyViolationError
from src.logging_utils.logger import StructuredLogger, AuditLogger


class PolicyEngine:
    """Enforces governance policies for model promotion."""

    def __init__(self, config: PromotionPolicyConfig, logger: Optional[StructuredLogger] = None, audit_logger: Optional[AuditLogger] = None):
        """
        Initialize policy engine.

        Args:
            config: Promotion policy configuration.
            logger: Optional structured logger.
            audit_logger: Optional audit logger.
        """
        self.config = config
        self.logger = logger
        self.audit_logger = audit_logger

    def check_accuracy_gate(
        self,
        model_name: str,
        accuracy_pct: float,
    ) -> bool:
        """
        Check if model meets minimum accuracy threshold.

        Args:
            model_name: Name of model.
            accuracy_pct: Accuracy percentage.

        Returns:
            True if model passes gate.

        Raises:
            PolicyViolationError: If model fails gate.
        """
        if accuracy_pct < self.config.min_accuracy_pct:
            raise PolicyViolationError(
                f"Accuracy {accuracy_pct:.2f}% below minimum {self.config.min_accuracy_pct}%"
            )

        if self.logger:
            self.logger.info(
                f"Accuracy gate passed: {model_name}",
                accuracy_pct=accuracy_pct,
                required=self.config.min_accuracy_pct,
            )

        return True

    def check_regression_gate(
        self,
        model_name: str,
        regression_pct: float,
    ) -> bool:
        """
        Check if regression is within tolerance.

        Args:
            model_name: Name of model.
            regression_pct: Percentage regression vs baseline.

        Returns:
            True if regression is acceptable.

        Raises:
            PolicyViolationError: If regression exceeds tolerance.
        """
        if regression_pct > self.config.max_regression_pct:
            raise PolicyViolationError(
                f"Regression {regression_pct:.2f}% exceeds maximum {self.config.max_regression_pct}%"
            )

        if self.logger:
            self.logger.info(
                f"Regression gate passed: {model_name}",
                regression_pct=regression_pct,
                max_allowed=self.config.max_regression_pct,
            )

        return True

    def check_latency_gate(
        self,
        model_name: str,
        latency_ms: float,
    ) -> bool:
        """
        Check if inference latency is acceptable.

        Args:
            model_name: Name of model.
            latency_ms: Inference latency in milliseconds.

        Returns:
            True if latency is acceptable.

        Raises:
            PolicyViolationError: If latency exceeds threshold.
        """
        if self.config.max_latency_ms is None:
            return True

        if latency_ms > self.config.max_latency_ms:
            raise PolicyViolationError(
                f"Latency {latency_ms:.2f}ms exceeds maximum {self.config.max_latency_ms}ms"
            )

        if self.logger:
            self.logger.info(
                f"Latency gate passed: {model_name}",
                latency_ms=latency_ms,
                max_allowed_ms=self.config.max_latency_ms,
            )

        return True

    def check_all_gates(
        self,
        model_name: str,
        metrics: Dict[str, float],
    ) -> Dict[str, bool]:
        """
        Check all applicable gates.

        Args:
            model_name: Name of model.
            metrics: Dictionary of metric_name -> value.

        Returns:
            Dictionary of gate_name -> passed.
        """
        results = {}

        try:
            if "accuracy" in metrics:
                self.check_accuracy_gate(model_name, metrics["accuracy"])
                results["accuracy"] = True
        except PolicyViolationError:
            results["accuracy"] = False

        try:
            if "regression" in metrics:
                self.check_regression_gate(model_name, metrics["regression"])
                results["regression"] = True
        except PolicyViolationError:
            results["regression"] = False

        try:
            if "latency_ms" in metrics:
                self.check_latency_gate(model_name, metrics["latency_ms"])
                results["latency"] = True
        except PolicyViolationError:
            results["latency"] = False

        if self.logger:
            self.logger.info(
                f"Policy gates evaluated: {model_name}",
                gate_results=results,
            )

        return results
