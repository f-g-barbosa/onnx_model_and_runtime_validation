"""
Report building for batch validation results.

Produces JSON and CSV batch validation reports.
"""

from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

from src.core.types import ValidationMetrics, BatchValidationSummary
from src.utils.file_utils import save_json_report, save_csv_report
from src.logging_utils.logger import StructuredLogger


class BatchReportBuilder:
    """Builds batch validation reports."""

    def __init__(self, logger: Optional[StructuredLogger] = None):
        """
        Initialize report builder.

        Args:
            logger: Optional logger instance.
        """
        self.logger = logger

    def build_batch_report_json(
        self,
        summary: BatchValidationSummary,
        image_metrics: List[ValidationMetrics],
        problematic_images: List[str],
    ) -> Dict[str, Any]:
        """
        Build batch report as JSON-serializable dictionary.

        Args:
            summary: Batch validation summary.
            image_metrics: Metrics for each image.
            problematic_images: List of problematic image names.

        Returns:
            Dictionary for JSON serialization.
        """
        report = {
            "batch_id": summary.batch_id,
            "timestamp": datetime.utcnow().isoformat(),
            "summary": {
                "total_images": summary.total_images,
                "successful": summary.successful,
                "failed": summary.failed,
                "success_rate_pct": (summary.successful / summary.total_images * 100) if summary.total_images > 0 else 0,
                "avg_num_detections": summary.avg_num_detections,
                "avg_max_score": summary.avg_max_score,
                "anomalies_detected": summary.anomalies_detected,
                "validation_time_ms": summary.validation_time_ms,
            },
            "problematic_images": problematic_images,
            "image_details": [
                {
                    "image_name": m.image_name,
                    "num_detections": m.num_detections,
                    "max_score": m.max_score,
                    "min_score": m.min_score,
                    "detected_classes": m.detected_classes,
                    "passed_threshold": m.passed_threshold,
                    "anomalies": m.anomalies,
                }
                for m in image_metrics
            ],
        }

        return report

    def save_batch_report(
        self,
        report: Dict[str, Any],
        output_path: Path,
    ) -> None:
        """
        Save batch report to JSON file.

        Args:
            report: Report dictionary.
            output_path: Output file path.
        """
        save_json_report(report, output_path, pretty=True)

        if self.logger:
            self.logger.info(f"Batch report saved: {output_path}")

    def save_image_metrics_csv(
        self,
        image_metrics: List[ValidationMetrics],
        output_path: Path,
    ) -> None:
        """
        Save image metrics as CSV.

        Args:
            image_metrics: List of image validation metrics.
            output_path: Output file path.
        """
        data = [
            {
                "image_name": m.image_name,
                "num_detections": m.num_detections,
                "max_score": f"{m.max_score:.4f}",
                "min_score": f"{m.min_score:.4f}",
                "detected_classes": ",".join(str(c) for c in m.detected_classes),
                "passed_threshold": m.passed_threshold,
                "anomalies": ";".join(m.anomalies) if m.anomalies else "",
            }
            for m in image_metrics
        ]

        save_csv_report(data, output_path)

        if self.logger:
            self.logger.info(f"Image metrics CSV saved: {output_path}")
