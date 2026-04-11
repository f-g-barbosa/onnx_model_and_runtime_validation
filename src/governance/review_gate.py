"""
Human review gate for model promotion.

Manages human review workflow for governance decisions.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

from src.core.enums import ReviewDecision
from src.core.exceptions import ValidationError
from src.logging_utils.logger import StructuredLogger, AuditLogger


@dataclass
class ReviewRecord:
    """Record of a human review."""
    model_name: str
    reviewer_id: str
    decision: ReviewDecision
    notes: str
    timestamp: datetime
    policy_check_results: Dict[str, bool]


class ReviewGate:
    """Manages human review gate for model promotion."""

    def __init__(self, logger: Optional[StructuredLogger] = None, audit_logger: Optional[AuditLogger] = None):
        """
        Initialize review gate.

        Args:
            logger: Optional structured logger.
            audit_logger: Optional audit logger.
        """
        self.logger = logger
        self.audit_logger = audit_logger
        self.review_records: List[ReviewRecord] = []

    def request_review(
        self,
        model_name: str,
        policy_check_results: Dict[str, bool],
        required_approvers: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Request human review for model promotion.

        Args:
            model_name: Name of model requiring review.
            policy_check_results: Results of automated policy checks.
            required_approvers: List of required approval roles.
            context: Additional context for reviewers.

        Returns:
            Review request ID.
        """
        review_id = f"{model_name}_{datetime.utcnow().isoformat()}"

        if self.logger:
            self.logger.info(
                f"Review requested: {model_name}",
                review_id=review_id,
                required_approvers=required_approvers,
                policy_results=policy_check_results,
            )

        if self.audit_logger:
            self.audit_logger.log_event(
                event_type="review_requested",
                model_name=model_name,
                status="pending",
                details={
                    "review_id": review_id,
                    "required_approvers": required_approvers,
                    "policy_checks": policy_check_results,
                },
            )

        return review_id

    def submit_review(
        self,
        model_name: str,
        reviewer_id: str,
        decision: ReviewDecision,
        notes: str,
        policy_check_results: Dict[str, bool],
    ) -> ReviewRecord:
        """
        Submit a human review result.

        Args:
            model_name: Name of model.
            reviewer_id: ID of reviewer.
            decision: Review decision (approved/rejected).
            notes: Review notes.
            policy_check_results: Policy check results reference.

        Returns:
            ReviewRecord.
        """
        record = ReviewRecord(
            model_name=model_name,
            reviewer_id=reviewer_id,
            decision=decision,
            notes=notes,
            timestamp=datetime.utcnow(),
            policy_check_results=policy_check_results,
        )

        self.review_records.append(record)

        if self.logger:
            self.logger.info(
                f"Review submitted: {model_name}",
                reviewer_id=reviewer_id,
                decision=decision.value,
            )

        if self.audit_logger:
            self.audit_logger.log_model_promotion(
                model_name=model_name,
                approved=(decision == ReviewDecision.APPROVED),
                reviewer=reviewer_id,
                policy_results=policy_check_results,
                notes=notes,
            )

        return record

    def get_review_status(self, model_name: str) -> Optional[ReviewRecord]:
        """Get most recent review status for a model."""
        matching = [r for r in self.review_records if r.model_name == model_name]
        return matching[-1] if matching else None
