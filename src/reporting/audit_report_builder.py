"""
Audit report building.

Produces comprehensive audit reports for governance and compliance.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.core.types import PromotionRecommendation, BatchValidationSummary
from src.utils.file_utils import save_json_report
from src.logging_utils.logger import StructuredLogger


class AuditReportBuilder:
    """Builds comprehensive audit reports."""

    def __init__(self, logger: Optional[StructuredLogger] = None):
        """
        Initialize audit report builder.

        Args:
            logger: Optional logger instance.
        """
        self.logger = logger

    def build_promotion_audit_report(
        self,
        model_name: str,
        model_version: str,
        validation_summary: BatchValidationSummary,
        policy_results: Dict[str, bool],
        promotion_recommendation: PromotionRecommendation,
        baseline_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build comprehensive promotion audit report.

        Args:
            model_name: Name of model.
            model_version: Version of model.
            validation_summary: Batch validation summary.
            policy_results: Policy check results.
            promotion_recommendation: Promotion recommendation.
            baseline_model: Optional baseline model name for comparison.

        Returns:
            Audit report dictionary.
        """
        report = {
            "report_type": "model_promotion_audit",
            "generated_at": datetime.utcnow().isoformat(),
            "model": {
                "name": model_name,
                "version": model_version,
                "baseline_model": baseline_model,
            },
            "validation_results": {
                "batch_id": validation_summary.batch_id,
                "total_images": validation_summary.total_images,
                "successful": validation_summary.successful,
                "failed": validation_summary.failed,
                "success_rate_pct": (validation_summary.successful / validation_summary.total_images * 100)
                    if validation_summary.total_images > 0 else 0,
                "avg_num_detections": validation_summary.avg_num_detections,
                "avg_max_score": validation_summary.avg_max_score,
                "anomalies": validation_summary.anomalies_detected,
            },
            "policy_evaluation": {
                "policy_checks": policy_results,
                "all_checks_passed": all(policy_results.values()),
            },
            "recommendation": {
                "recommendable": promotion_recommendation.recommendable,
                "reason": promotion_recommendation.reason,
                "requires_manual_review": promotion_recommendation.requires_manual_review,
                "audit_referral": promotion_recommendation.audit_referral,
            },
        }

        return report

    def save_audit_report(
        self,
        report: Dict[str, Any],
        output_path: Path,
    ) -> None:
        """
        Save audit report to JSON file.

        Args:
            report: Audit report dictionary.
            output_path: Output file path.
        """
        save_json_report(report, output_path, pretty=True)

        if self.logger:
            self.logger.info(f"Audit report saved: {output_path}")
