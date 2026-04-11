"""
ONNX Runtime session management.

Creates and manages InferenceSession objects with proper error handling.
"""

from pathlib import Path
from typing import List, Optional
import onnxruntime as ort

from src.core.exceptions import InvalidModelError, ModelNotFoundError
from src.logging_utils.logger import StructuredLogger
from src.core.schemas import LoggingConfig


class SessionFactory:
    """Factory for creating and managing ONNX Runtime sessions."""

    def __init__(self, logger: Optional[StructuredLogger] = None):
        """
        Initialize session factory.

        Args:
            logger: Optional logger instance.
        """
        self.logger = logger

    def create_session(
        self,
        model_path: Path,
        providers: Optional[List[str]] = None,
    ) -> ort.InferenceSession:
        """
        Create an ONNX Runtime InferenceSession.

        Args:
            model_path: Path to ONNX model file.
            providers: List of execution providers to use.

        Returns:
            ONNX Runtime InferenceSession.

        Raises:
            ModelNotFoundError: If model file doesn't exist.
            InvalidModelError: If model cannot be loaded.
        """
        model_path = Path(model_path)

        # Validate file exists
        if not model_path.exists():
            raise ModelNotFoundError(f"Model not found: {model_path}")

        if model_path.stat().st_size == 0:
            raise InvalidModelError(f"Model file is empty: {model_path}")

        # Default providers
        if providers is None:
            providers = ["CPUExecutionProvider"]

        if self.logger:
            self.logger.info(f"Loading ONNX model: {model_path}", providers=providers)

        try:
            session = ort.InferenceSession(
                str(model_path),
                providers=providers,
            )

            if self.logger:
                self.logger.info(
                    "ONNX model loaded successfully",
                    actual_providers=session.get_providers(),
                )

            return session

        except Exception as e:
            raise InvalidModelError(f"Failed to load ONNX model {model_path}: {e}")

    def get_available_providers(self) -> List[str]:
        """
        Get list of available execution providers.

        Returns:
            List of available provider names.
        """
        return ort.get_available_providers()
