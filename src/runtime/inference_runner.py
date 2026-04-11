"""
Inference execution runner.

Handles single-image inference with timing and error handling.
"""

import time
from typing import List, Optional
import numpy as np
import onnxruntime as ort

from src.core.types import InferenceResult, ImageMetadata, ModelMetadata
from src.core.exceptions import InferenceError
from src.logging_utils.logger import StructuredLogger
from src.utils.time_utils import Timer


class InferenceRunner:
    """Executes inference on preprocessed images."""

    def __init__(self, session: ort.InferenceSession, logger: Optional[StructuredLogger] = None):
        """
        Initialize inference runner.

        Args:
            session: ONNX Runtime session.
            logger: Optional logger instance.
        """
        self.session = session
        self.logger = logger

    def run_inference(
        self,
        input_tensor: np.ndarray,
        input_name: str,
        image_metadata: ImageMetadata,
        model_metadata: ModelMetadata,
        batch_id: Optional[str] = None,
    ) -> InferenceResult:
        """
        Run inference on input tensor.

        Args:
            input_tensor: Preprocessed input tensor.
            input_name: Name of input node.
            image_metadata: Metadata about the image.
            model_metadata: Metadata about the model.
            batch_id: Optional batch identifier.

        Returns:
            InferenceResult with outputs and timing.

        Raises:
            InferenceError: If inference fails.
        """
        try:
            with Timer() as timer:
                outputs = self.session.run(None, {input_name: input_tensor})

            inference_time_ms = timer.elapsed_ms()

            if self.logger:
                self.logger.debug(
                    f"Inference completed: {image_metadata.image_name}",
                    inference_time_ms=inference_time_ms,
                    num_outputs=len(outputs),
                )

            return InferenceResult(
                image_metadata=image_metadata,
                raw_outputs=outputs,
                inference_time_ms=inference_time_ms,
                model_metadata=model_metadata,
                batch_id=batch_id,
            )

        except Exception as e:
            error_msg = f"Inference failed for {image_metadata.image_name}: {e}"
            if self.logger:
                self.logger.error(error_msg)
            raise InferenceError(error_msg)

    def batch_inference(
        self,
        input_tensors: List[np.ndarray],
        input_name: str,
        images_metadata: List[ImageMetadata],
        model_metadata: ModelMetadata,
        batch_id: Optional[str] = None,
    ) -> List[InferenceResult]:
        """
        Run inference on multiple images.

        Args:
            input_tensors: List of preprocessed input tensors.
            input_name: Name of input node.
            images_metadata: List of image metadata.
            model_metadata: Metadata about the model.
            batch_id: Optional batch identifier.

        Returns:
            List of InferenceResult objects.
        """
        results = []
        for input_tensor, img_meta in zip(input_tensors, images_metadata):
            try:
                result = self.run_inference(
                    input_tensor=input_tensor,
                    input_name=input_name,
                    image_metadata=img_meta,
                    model_metadata=model_metadata,
                    batch_id=batch_id,
                )
                results.append(result)
            except InferenceError as e:
                if self.logger:
                    self.logger.error(f"Failed to process {img_meta.image_name}: {e}")
                # Continue processing other images

        return results
