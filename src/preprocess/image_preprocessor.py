"""
Image preprocessing for model inference.

Handles resizing, normalization, and tensor preparation.
"""

from pathlib import Path
from typing import Tuple, Optional
import numpy as np

from src.core.types import ImageMetadata
from src.core.schemas import PreprocessingConfig
from src.core.exceptions import PreprocessingError
from src.utils.image_utils import (
    load_image,
    resize_image,
    convert_bgr_to_rgb,
    normalize_image,
    get_image_shape,
)
from src.logging_utils.logger import StructuredLogger


class ImagePreprocessor:
    """Preprocesses images for ONNX model inference."""

    def __init__(self, config: PreprocessingConfig, logger: Optional[StructuredLogger] = None):
        """
        Initialize preprocessor.

        Args:
            config: Preprocessing configuration.
            logger: Optional logger instance.
        """
        self.config = config
        self.logger = logger

    def preprocess(
        self,
        image_path: Path,
        to_rgb: bool = True,
        add_batch_dim: bool = True,
        dtype: np.dtype = np.uint8,
    ) -> Tuple[np.ndarray, ImageMetadata, np.ndarray]:
        """
        Preprocess image for inference.

        Args:
            image_path: Path to image file.
            to_rgb: Convert BGR to RGB.
            add_batch_dim: Add batch dimension (N,H,W,C).
            dtype: Output tensor data type.

        Returns:
            Tuple of (input_tensor, image_metadata, original_image).

        Raises:
            PreprocessingError: If preprocessing fails.
        """
        try:
            # Load image
            image = load_image(image_path)
            original_shape = get_image_shape(image)

            # Resize
            resized = resize_image(
                image,
                self.config.resize_width,
                self.config.resize_height,
            )
            resized_shape = get_image_shape(resized)

            # Convert color space if needed
            if to_rgb:
                resized = convert_bgr_to_rgb(resized)

            # Normalize if configured
            if self.config.normalize:
                resized = normalize_image(
                    resized,
                    mean=self.config.normalization_mean,
                    std=self.config.normalization_std,
                )

            # Ensure correct data type and shape
            tensor = resized.astype(dtype)

            # Add batch dimension
            if add_batch_dim:
                tensor = np.expand_dims(tensor, axis=0)  # (1, H, W, C)

            # Create metadata
            metadata = ImageMetadata(
                image_path=image_path,
                image_name=image_path.name,
                original_shape=original_shape,
                resized_shape=resized_shape,
            )

            if self.logger:
                self.logger.debug(
                    f"Image preprocessed: {image_path.name}",
                    original_shape=original_shape,
                    resized_shape=resized_shape,
                    output_shape=tuple(tensor.shape),
                )

            return tensor, metadata, image

        except PreprocessingError:
            raise
        except Exception as e:
            raise PreprocessingError(f"Failed to preprocess image {image_path}: {e}")

    def batch_preprocess(
        self,
        image_paths: list[Path],
        **kwargs,
    ) -> Tuple[list[np.ndarray], list[ImageMetadata], list[np.ndarray]]:
        """
        Preprocess multiple images.

        Args:
            image_paths: List of image file paths.
            **kwargs: Additional arguments passed to preprocess().

        Returns:
            Tuple of (tensors_list, metadata_list, originals_list).
        """
        tensors = []
        metadata_list = []
        originals = []

        for image_path in image_paths:
            try:
                tensor, metadata, original = self.preprocess(image_path, **kwargs)
                tensors.append(tensor)
                metadata_list.append(metadata)
                originals.append(original)
            except PreprocessingError as e:
                if self.logger:
                    self.logger.warning(f"Failed to preprocess {image_path}: {e}")
                continue

        return tensors, metadata_list, originals
