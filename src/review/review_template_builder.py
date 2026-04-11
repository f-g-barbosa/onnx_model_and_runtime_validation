"""
Review template building.

Generates templates for human reviewers.
"""

from pathlib import Path
from typing import List, Dict, Optional, Any
from src.core.types import ValidationMetrics
from src.utils.file_utils import save_json_report


class ReviewTemplateBuilder:
    """Builds templates for human review workflow."""

    def __init__(self):
        """Initialize review template builder."""
        pass

    def build_review_template(
        self,
        model_name: str,
        batch_id: str,
        problematic_images:List[str],
        image_details: Dict[str, Any],
        policy_results: Dict[str, bool],
    ) -> Dict[str, Any]:
        """
        Build review template for reviewers.

        Args:
            model_name: Name of model.
            batch_id: Batch identifier.
            problematic_images: List of problematic image names.
            image_details: Details about images.
            policy_results: Policy check results.

        Returns:
            Review template dictionary.
        """
        template = {
            "review_task": {
                "model_name": model_name,
                "batch_id": batch_id,
                "instructions": "Please evaluate the model performance on the flagged images below.",
            },
            "policy_summary": {
                "checks": policy_results,
                "all_passed": all(policy_results.values()),
            },
            "flagged_samples": problematic_images,
            "sample_details": {
                name: image_details.get(name, {})
                for name in problematic_images
            },
            "review_criteria": {
                "accuracy": "Does the model correctly detect objects?",
                "consistency": "Are detections consistent across similar images?",
                "confidence": "Are confidence scores reasonable?",
                "latency": "Is inference time acceptable?",
            },
            "reviewer_decision": {
                "recommendation": "APPROVE | REJECT | HOLD",
                "confidence": "HIGH | MEDIUM | LOW",
                "notes": "",
                "required_approvers": ["model_reviewer", "ml_ops_engineer"],
            },
        }

        return template

    def save_review_template(
        self,
        template: Dict[str, Any],
        output_path: Path,
    ) -> None:
        """
        Save review template to file.

        Args:
            template: Review template dictionary.
            output_path: Output file path.
        """
        save_json_report(template, output_path, pretty=True)
