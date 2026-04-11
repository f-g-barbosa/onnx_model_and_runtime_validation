#!/usr/bin/env python3
"""
Bank-Ready AI Model Validation & Governance Pipeline

Main orchestration script for model onboarding, validation, comparison,
governance, and promotion recommendation.

Usage:
    python main.py --batch-id test_batch_001 --dry-run
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

from src.config.loader import ConfigLoader
from src.logging_utils.logger import StructuredLogger, AuditLogger
from src.runtime.session_factory import SessionFactory
from src.runtime.io_inspector import IOInspector
from src.runtime.inference_runner import InferenceRunner
from src.preprocess.image_preprocessor import ImagePreprocessor
from src.postprocess.output_parser import OutputParser
from src.postprocess.detection_visualizer import DetectionVisualizer
from src.onboarding.model_metadata_validator import ModelMetadataValidator, SignatureValidator
from src.validation.single_image_validator import SingleImageValidator
from src.validation.batch_validator import BatchValidator
from src.governance.policy_engine import PolicyEngine
from src.governance.promotion_recommender import PromotionRecommender
from src.reporting.batch_report_builder import BatchReportBuilder
from src.reporting.audit_report_builder import AuditReportBuilder
from src.review.flagged_sample_selector import FlaggedSampleSelector
from src.utils.file_utils import list_image_files, ensure_directory
from src.utils.time_utils import Timer, format_duration
from src.core.exceptions import ValidationPipelineError


def create_logger(config_dir: Path, batch_id: str) -> tuple[StructuredLogger, AuditLogger]:
    """Create structured and audit loggers."""
    config = ConfigLoader(config_dir).load_full_config(batch_id)

    logger = StructuredLogger(__name__, config.logging)
    audit_logger = AuditLogger(config.output.output_dir / "audit")

    return logger, audit_logger


def main():
    """Main pipeline orchestration."""
    parser = argparse.ArgumentParser(
        description="Bank-Ready AI Model Validation & Governance Pipeline"
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs"),
        help="Directory containing YAML configuration files",
    )
    parser.add_argument(
        "--batch-id",
        type=str,
        default=f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        help="Batch identifier for this run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode (no file output)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data"),
        help="Input images directory",
    )
    parser.add_argument(
        "--skip-review",
        action="store_true",
        help="Skip human review gate (for testing)",
    )

    args = parser.parse_args()

    try:
        # Initialize logging
        logger, audit_logger = create_logger(args.config_dir, args.batch_id)

        logger.info(
            "Pipeline started",
            batch_id=args.batch_id,
            config_dir=str(args.config_dir),
            dry_run=args.dry_run,
        )

        # Load configuration
        config_loader = ConfigLoader(args.config_dir)
        config = config_loader.load_full_config(args.batch_id, dry_run=args.dry_run)

        # Setup output directories
        output_dir = config.output.output_dir
        debug_dir = config.output.debug_dir or output_dir / "debugs"
        annotated_dir = config.output.annotated_dir or output_dir / "annotated"

        if not args.dry_run:
            ensure_directory(output_dir, "Output")
            ensure_directory(debug_dir, "Debug")
            ensure_directory(annotated_dir, "Annotated")

        # ===== ONBOARDING =====
        logger.info("Starting model onboarding")

        metadata_validator = ModelMetadataValidator(logger)
        sig_validator = SignatureValidator(logger)

        metadata_validator.validate_onboarding(
            config.runtime.model_path,
            "ssd_mobilenet_v1",
            "1.0.0",
        )

        # Create session
        session_factory = SessionFactory(logger)
        session = session_factory.create_session(
            config.runtime.model_path,
            config.runtime.providers,
        )

        # Validate signature
        sig_validator.validate_signature(session)
        sig_validator.validate_input_shape(session)

        # Inspect IO
        io_inspector = IOInspector(session, logger)
        if config.runtime.verbose:
            io_inspector.print_model_io()

        model_metadata = io_inspector.create_model_metadata(
            "ssd_mobilenet_v1",
            "1.0.0",
            str(config.runtime.model_path),
        )

        logger.info("Model onboarding completed")

        # ===== VALIDATION =====
        logger.info("Starting batch validation")

        input_dir = args.input_dir
        image_files = list_image_files(input_dir)
        logger.info(f"Found {len(image_files)} images for validation")

        if len(image_files) == 0:
            logger.warning(f"No images found in {input_dir}")
            return 1

        # Preprocess
        preprocessor = ImagePreprocessor(config.preprocessing, logger)
        tensors, image_metadata_list, orig_images = preprocessor.batch_preprocess(
            image_files,
            to_rgb=True,
            add_batch_dim=True,
        )

        logger.info(f"Preprocessed {len(tensors)} images")

        # Inference
        inference_runner = InferenceRunner(session, logger)
        input_name = io_inspector.get_input_name()

        inference_results = []
        batch_timer = Timer()
        batch_timer.__enter__()

        for tensor, img_meta, orig_img in zip(tensors, image_metadata_list, orig_images):
            result = inference_runner.run_inference(
                tensor,
                input_name,
                img_meta,
                model_metadata,
                args.batch_id,
            )
            inference_results.append((result, orig_img))

        logger.info(f"Completed {len(inference_results)} inferences")

        # Parse outputs and validate
        output_parser = OutputParser(logger)
        visualizer = DetectionVisualizer(logger)
        single_validator = SingleImageValidator(config.validation, logger)

        image_metrics = []
        for inference_result, orig_image in inference_results:
            # Parse outputs
            parsed = output_parser.parse_ssd_mobilenet_output(
                inference_result.raw_outputs,
                config.validation.score_threshold,
            )

            # Validate
            metrics = single_validator.validate(
                inference_result.image_metadata.image_name,
                parsed,
                inference_result.inference_time_ms,
            )
            image_metrics.append(metrics)

            # Save debug if configured
            if config.output.save_debug_outputs and not args.dry_run:
                debug_img_dir = debug_dir / inference_result.image_metadata.image_path.stem
                debug_img_dir.mkdir(parents=True, exist_ok=True)
                for idx, output in enumerate(inference_result.raw_outputs):
                    np.save(debug_img_dir / f"output_{idx}.npy", output)

            # Draw and save annotated if configured
            if config.output.save_annotated_images and not args.dry_run:
                annotated = visualizer.draw_detections(
                    orig_image,
                    parsed,
                    inference_result.image_metadata.original_shape,
                    (config.preprocessing.resize_width, config.preprocessing.resize_height),
                    config.validation.score_threshold,
                )
                ann_path = annotated_dir / f"{inference_result.image_metadata.image_path.stem}_annotated.jpg"
                visualizer.save_annotated_image(annotated, ann_path)

        batch_timer.__exit__(None, None, None)

        # Batch summary
        batch_validator = BatchValidator(config.validation, logger)
        validation_summary = batch_validator.summarize_batch(
            args.batch_id,
            image_metrics,
            batch_timer.elapsed_ms(),
        )

        problematic = batch_validator.get_problematic_images(image_metrics, top_k=5)
        logger.info(f"Validation completed. Problematic images: {problematic}")

        # ===== GOVERNANCE & PROMOTION =====
        logger.info("Starting governance evaluation")

        # Calculate aggregate metrics
        metrics_dict = {
            "accuracy": validation_summary.avg_max_score * 100,
            "regression": 0.0,
            "latency_ms": validation_summary.validation_time_ms / len(image_metrics) if image_metrics else 0,
        }

        # Policy checks
        policy_engine = PolicyEngine(config.promotion_policy, logger, audit_logger)
        policy_results = policy_engine.check_all_gates("ssd_mobilenet_v1", metrics_dict)

        logger.info("Policy checks completed", results=policy_results)

        # Promotion recommendation
        recommender = PromotionRecommender(config.promotion_policy, logger)
        recommendation = recommender.generate_recommendation(
            "ssd_mobilenet_v1",
            policy_results,
            metrics_dict,
        )

        logger.info(
            f"Promotion recommendation: {recommendation.recommendable}",
            reason=recommendation.reason,
        )

        # ===== REPORTING =====
        logger.info("Generating reports")

        batch_report_builder = BatchReportBuilder(logger)
        batch_report = batch_report_builder.build_batch_report_json(
            validation_summary,
            image_metrics,
            problematic,
        )

        audit_report_builder = AuditReportBuilder(logger)
        audit_report = audit_report_builder.build_promotion_audit_report(
            "ssd_mobilenet_v1",
            "1.0.0",
            validation_summary,
            policy_results,
            recommendation,
        )

        if not args.dry_run:
            batch_report_path = output_dir / f"batch_report_{args.batch_id}.json"
            audit_report_path = output_dir / f"audit_report_{args.batch_id}.json"

            batch_report_builder.save_batch_report(batch_report, batch_report_path)
            audit_report_builder.save_audit_report(audit_report, audit_report_path)

            logger.info("Reports saved successfully")

        logger.info(
            "Pipeline completed successfully",
            total_time_ms=batch_timer.elapsed_ms(),
            batch_id=args.batch_id,
        )

        print("\n" + "=" * 60)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 60)
        print(f"Batch ID: {args.batch_id}")
        print(f"Images processed: {validation_summary.total_images}")
        print(f"Successful: {validation_summary.successful}")
        print(f"Failed: {validation_summary.failed}")
        print(f"Promotion recommendable: {recommendation.recommendable}")
        print(f"Total time: {format_duration(batch_timer.elapsed_ms())}")
        print("=" * 60)

        return 0

    except ValidationPipelineError as e:
        if 'logger' in locals():
            logger.error(f"Pipeline error: {e}")
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        if 'logger' in locals():
            logger.critical(f"Unexpected error: {e}")
        print(f"\nUNEXPECTED ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

    print("\nDone.")


if __name__ == "__main__":
    main()
