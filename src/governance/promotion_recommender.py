"""
Promotion recommendation engine.

Generates promotion recommendations based on validation results and policies.
"""

from typing import Dict, Optional
from src.core.types import PromotionRecommendation
from src.core.schemas import PromotionPolicyConfig
from src.core.exceptions import ValidationError
from src.logging_utils.logger import StructuredLogger


class PromotionRecommender:
    """Generates model promotion recommendations."""

    def __init__(self, policy_config: PromotionPolicyConfig, logger: Optional[StructuredLogger] = None):
        """
        Initialize recommendation engine.

        Args:
            policy_config: Promotion policy configuration.
            logger: Optional logger instance.
        """
        self.policy_config = policy_config
        self.logger = logger

    def generate_recommendation(
        self,
        model_name: str,
        policy_check_results: Dict[str, bool],
        metrics: Dict[str, float],
    ) -> PromotionRecommendation:
        """
        Generate promotion recommendation.

        Args:
            model_name: Name of model.
            policy_check_results: Results from policy engine.
            metrics: Model metrics dictionary.

        Returns:
            PromotionRecommendation.
        """
        # Check if all required gates passed
        all_gates_passed = all(policy_check_results.values())

        # Determine if recommendable
        recommendable = all_gates_passed and not self.policy_config.require_human_review

        # Generate reason
        if all_gates_passed:
            if self.policy_config.require_human_review:
                reason = "All technical gates passed. Awaiting human review."
            else:
                reason = "All gates passed. Ready for promotion."
        else:
            failed_gates = [k for k, v in policy_check_results.items() if not v]
            reason = f"Failed gates: {', '.join(failed_gates)}"

        recommendation = PromotionRecommendation(
            model_name=model_name,
            recommendable=recommendable,
            reason=reason,
            policy_check_results=policy_check_results,
            requires_manual_review=self.policy_config.require_human_review,
            audit_referral=f"review_{model_name}" if all_gates_passed else "",
        )

        if self.logger:
            self.logger.info(
                f"Promotion recommendation: {model_name}",
                recommendable=recommendable,
                gate_results=policy_check_results,
                requires_review=self.policy_config.require_human_review,
            )

        return recommendation
