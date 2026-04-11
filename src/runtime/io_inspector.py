"""
ONNX model input/output inspection.

Provides utilities to inspect model signatures and I/O specifications.
"""

from typing import List, Dict, Any, Tuple, Optional
import onnxruntime as ort
import numpy as np

from src.core.types import ModelMetadata
from src.core.exceptions import SignatureValidationError
from src.logging_utils.logger import StructuredLogger


class IOInspector:
    """Inspector for model inputs and outputs."""

    def __init__(self, session: ort.InferenceSession, logger: Optional[StructuredLogger] = None):
        """
        Initialize I/O inspector.

        Args:
            session: ONNX Runtime session.
            logger: Optional logger instance.
        """
        self.session = session
        self.logger = logger

    def get_inputs(self) -> List[Dict[str, Any]]:
        """
        Get model input specifications.

        Returns:
            List of input specifications.
        """
        inputs = []
        for inp in self.session.get_inputs():
            inputs.append({
                "name": inp.name,
                "shape": inp.shape,
                "type": inp.type,
            })
        return inputs

    def get_outputs(self) -> List[Dict[str, Any]]:
        """
        Get model output specifications.

        Returns:
            List of output specifications.
        """
        outputs = []
        for out in self.session.get_outputs():
            outputs.append({
                "name": out.name,
                "shape": out.shape,
                "type": out.type,
            })
        return outputs

    def get_input_name(self, index: int = 0) -> str:
        """
        Get name of input at specified index.

        Args:
            index: Input index (default: 0).

        Returns:
            Input name.

        Raises:
            SignatureValidationError: If model has no inputs.
        """
        inputs = self.session.get_inputs()
        if not inputs:
            raise SignatureValidationError("Model has no inputs")
        if index >= len(inputs):
            raise SignatureValidationError(f"Input index {index} out of range")
        return inputs[index].name

    def get_input_shape(self, index: int = 0) -> Tuple:
        """Get shape of input at specified index."""
        inputs = self.session.get_inputs()
        if index >= len(inputs):
            raise SignatureValidationError(f"Input index {index} out of range")
        return tuple(inputs[index].shape)

    def get_output_names(self) -> List[str]:
        """Get all output names."""
        return [out.name for out in self.session.get_outputs()]

    def validate_input_shape(self, shape: Tuple) -> bool:
        """
        Validate input tensor shape against model expectations.

        Args:
            shape: Shape to validate.

        Returns:
            True if shape is valid.

        Raises:
            SignatureValidationError: If shape is invalid.
        """
        expected_shape = self.get_input_shape()

        # Handle dynamic dimensions (None/-1)
        for i, (expected, actual) in enumerate(zip(expected_shape, shape)):
            if expected is not None and expected > 0 and expected != actual:
                raise SignatureValidationError(
                    f"Shape mismatch at dimension {i}: expected {expected}, got {actual}"
                )

        return True

    def print_model_io(self) -> None:
        """Print model inputs and outputs to console."""
        print("\n=== MODEL INPUTS ===")
        for inp in self.get_inputs():
            print(f"  Name:  {inp['name']}")
            print(f"  Shape: {inp['shape']}")
            print(f"  Type:  {inp['type']}")
            print()

        print("\n=== MODEL OUTPUTS ===")
        for out in self.get_outputs():
            print(f"  Name:  {out['name']}")
            print(f"  Shape: {out['shape']}")
            print(f"  Type:  {out['type']}")
            print()

    def create_model_metadata(
        self,
        model_name: str,
        model_version: str,
        model_path: str,
    ) -> ModelMetadata:
        """
        Create ModelMetadata from session inspection.

        Args:
            model_name: Name of the model.
            model_version: Version identifier.
            model_path: Path to model file.

        Returns:
            ModelMetadata instance.
        """
        from src.core.types import ModelMetadata
        from pathlib import Path

        return ModelMetadata(
            model_name=model_name,
            model_version=model_version,
            model_path=Path(model_path),
            input_shape=self.get_input_shape(),
            output_names=self.get_output_names(),
            provider=self.session.get_providers()[0] if self.session.get_providers() else "Unknown",
        )
