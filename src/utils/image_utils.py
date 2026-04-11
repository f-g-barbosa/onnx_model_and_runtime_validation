"""
Image utilities for the validation pipeline.

Provides image loading, resizing, and format conversion utilities.
"""

from pathlib import Path
from typing import Tuple, Optional
import cv2
import numpy as np

from src.core.exceptions import PreprocessingError


def load_image(image_path: Path) -> np.ndarray:
    """
    Load image from file using OpenCV.

    Args:
        image_path: Path to image file.

    Returns:
        Image as numpy array in BGR format.

    Raises:
        PreprocessingError: If image cannot be loaded.
    """
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            raise PreprocessingError(f"Could not read image: {image_path}")
        return image
    except Exception as e:
        raise PreprocessingError(f"Failed to load image {image_path}: {e}")


def resize_image(
    image: np.ndarray,
    target_width: int,
    target_height: int,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """
    Resize image to target dimensions.

    Args:
        image: Input image array.
        target_width: Target width.
        target_height: Target height.
        interpolation: OpenCV interpolation method.

    Returns:
        Resized image array.

    Raises:
        PreprocessingError: If resize fails.
    """
    try:
        resized = cv2.resize(image, (target_width, target_height), interpolation=interpolation)
        return resized
    except Exception as e:
        raise PreprocessingError(f"Failed to resize image: {e}")


def convert_bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert image from BGR to RGB color space.

    Args:
        image: Image in BGR format.

    Returns:
        Image in RGB format.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def convert_rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """
    Convert image from RGB to BGR color space.

    Args:
        image: Image in RGB format.

    Returns:
        Image in BGR format.
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def normalize_image(
    image: np.ndarray,
    mean: Optional[Tuple[float, ...]] = None,
    std: Optional[Tuple[float, ...]] = None,
) -> np.ndarray:
    """
    Normalize image values.

    Args:
        image: Input image array.
        mean: Mean values for normalization.
        std: Standard deviation values for normalization.

    Returns:
        Normalized image array.
    """
    normalized = image.astype(np.float32) / 255.0

    if mean and std:
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        normalized = (normalized - mean) / std

    return normalized


def add_bounding_box(
    image: np.ndarray,
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    label: str = "",
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw bounding box on image.

    Args:
        image: Image to draw on (will be modified).
        x_min, y_min, x_max, y_max: Box coordinates.
        label: Text label to display.
        color: Box color in BGR format.
        thickness: Line thickness.

    Returns:
        Image with bounding box drawn.
    """
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        cv2.putText(
            image,
            label,
            (x_min, y_min - 5),
            font,
            font_scale,
            color,
            thickness,
        )

    return image


def save_image(image: np.ndarray, output_path: Path) -> None:
    """
    Save image to file.

    Args:
        image: Image numpy array.
        output_path: Output file path.

    Raises:
        PreprocessingError: If save fails.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        success = cv2.imwrite(str(output_path), image)
        if not success:
            raise PreprocessingError(f"Failed to write image: {output_path}")
    except Exception as e:
        raise PreprocessingError(f"Failed to save image to {output_path}: {e}")


def get_image_shape(image: np.ndarray) -> Tuple[int, int, int]:
    """
    Get image shape as (height, width, channels).

    Args:
        image: Image array.

    Returns:
        Tuple of (height, width, channels).
    """
    if len(image.shape) == 2:  # Grayscale
        return (*image.shape, 1)
    return tuple(image.shape)
