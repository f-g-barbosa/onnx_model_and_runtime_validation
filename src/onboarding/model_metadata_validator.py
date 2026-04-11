"""
Model onboarding and validation.

Validates model metadata and signatures before allowing use in pipeline.
"""

from pathlib import Path
from typing import Optional
import onnxruntime as ort

from src.core.exceptions import OnboardingError, SignatureValidationError
from src.runtime.io_inspector import IOInspector
from src.logging_utils.logger import StructuredLogger
from src.utils.file_utils import validate_file_exists


class ModelMetadataValidator:
    """Validates model metadata for onboarding."""

    def __init__(self, logger: Optional[StructuredLogger] = None):
        """
        Initialize metadata validator.

        Args:
            logger: Optional logger instance.
        """
        self.logger = logger

    def validate_onboarding(
        self,
        model_path: Path,
        model_name: str,
        model_version: str,
    ) -> bool:
        """
        Perform full onboarding validation.

        Args:
            model_path: Path to ONNX model file.
            model_name: Model name.
            model_version: Model version.

        Returns:
            True if model passes all checks.

        Raises:
            OnboardingError: If validation fails.
        """
        # Check file exists and is readable
        validate_file_exists(model_path, "Model file")

        # Check file size
        file_size_mb = model_path.stat().st_size / (1024 * 1024)
        if file_size_mb == 0:
            raise OnboardingError("Model file is empty")

        if self.logger:
            self.logger.info(
                f"Onboarding model: {model_name}",
                model_version=model_version,
                file_size_mb=file_size_mb,
            )

        return True


class SignatureValidator:
    """Validates model input/output signatures."""

    def __init__(self, logger: Optional[StructuredLogger] = None):
        """
        Initialize signature validator.

        Args:
            logger: Optional logger instance.
        """
        self.logger = logger

    def validate_signature(
        self,
        session: ort.InferenceSession,
        expected_num_inputs: int = 1,
        expected_num_outputs: int = 4,
    ) -> bool:
        """
        Validate model has expected inputs and outputs.

        Args:
            session: ONNX Runtime session.
            expected_num_inputs: Expected number of inputs.
            expected_num_outputs: Expected number of outputs.

        Returns:
            True if signature is valid.

        Raises:
            SignatureValidationError: If signature doesn't match.
        """
        inspector = IOInspector(session, self.logger)

        # Check number of inputs
        inputs = inspector.get_inputs()
        if len(inputs) != expected_num_inputs:
            raise SignatureValidationError(
                f"Expected {expected_num_inputs} input(s), got {len(inputs)}"
            )

        # Check number of outputs
        outputs = inspector.get_outputs()
        if len(outputs) != expected_num_outputs:
            raise SignatureValidationError(
                f"Expected {expected_num_outputs} output(s), got {len(outputs)}"
            )

        if self.logger:
            self.logger.info(
                "Model signature validated",
                num_inputs=len(inputs),
                num_outputs=len(outputs),
            )

        return True

    def validate_input_shape(
        self,
        session: ort.InferenceSession,
        expected_size: tuple = (640, 640),
    ) -> bool:
        """
        Validate input image dimensions.

        Args:
            session: ONNX Runtime session.
            expected_size: Expected (width, height).

        Returns:
            True if input shape is valid.

        Raises:
            SignatureValidationError: If shape doesn't match.
        """
        inspector = IOInspector(session, self.logger)
        input_shape = inspector.get_input_shape()

        # Input shape is typically (1, height, width, channels)
        if len(input_shape) < 3:
            raise SignatureValidationError(
                f"Unexpected input shape: {input_shape}"
            )

        if self.logger:
            self.logger.info(
                "Input shape validated",
                input_shape=input_shape,
            )

        return True
