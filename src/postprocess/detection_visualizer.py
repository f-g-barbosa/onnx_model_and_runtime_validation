"""
Detection visualization for object detection models.

Draws bounding boxes and annotations on images.
"""

from pathlib import Path
from typing import List, Optional
import numpy as np

from src.core.types import DetectionBox, ParsedDetection
from src.core.exceptions import PostprocessingError
from src.utils.image_utils import add_bounding_box, save_image
from src.logging_utils.logger import StructuredLogger


class DetectionVisualizer:
    """Draws detection boxes and annotations on images."""

    # COCO classes (for reference)
    COCO_CLASSES = {
        0: "person", 1: "bicycle", 2: "car", 3: "motorbike", 4: "aeroplane",
        5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
        10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
        14: "cat", 15: "dog", 16: "horse", 17: "sheep", 18: "cow", 19: "elephant",
    }

    def __init__(self, logger: Optional[StructuredLogger] = None):
        """
        Initialize visualizer.

        Args:
            logger: Optional logger instance.
        """
        self.logger = logger

    def draw_detections(
        self,
        image: np.ndarray,
        parsed_detections: ParsedDetection,
        original_size: tuple,
        resized_size: tuple,
        score_threshold: float = 0.0,
        class_names: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Draw detection boxes on image.

        Args:
            image: Original image (BGR).
            parsed_detections: Parsed detection results.
            original_size: Original image size (H, W).
            resized_size: Model input size (H, W).
            score_threshold: Minimum score to draw.
            class_names: Optional mapping of class_id to class_name.

        Returns:
            Image with drawn detections.

        Raises:
            PostprocessingError: If drawing fails.
        """
        try:
            if class_names is None:
                class_names = self.COCO_CLASSES

            img_copy = image.copy()
            orig_h, orig_w = original_size[:2]
            resized_h, resized_w = resized_size

            for box in parsed_detections.boxes:
                if box.score < score_threshold:
                    continue

                # Scale box coordinates from resized to original size
                x_min = int(box.x_min * orig_w)
                y_min = int(box.y_min * orig_h)
                x_max = int(box.x_max * orig_w)
                y_max = int(box.y_max * orig_h)

                # Clamp to image bounds
                x_min = max(0, min(x_min, orig_w - 1))
                y_min = max(0, min(y_min, orig_h - 1))
                x_max = max(0, min(x_max, orig_w))
                y_max = max(0, min(y_max, orig_h))

                # Get class name
                class_name = class_names.get(box.class_id, f"Class {box.class_id}")
                label = f"{class_name}: {box.score:.2f}"

                # Draw box
                add_bounding_box(
                    img_copy,
                    x_min, y_min, x_max, y_max,
                    label=label,
                    color=(0, 255, 0),
                    thickness=2,
                )

            if self.logger:
                self.logger.debug(
                    "Drew detections on image",
                    num_boxes=len(parsed_detections.boxes),
                )

            return img_copy

        except Exception as e:
            raise PostprocessingError(f"Failed to draw detections: {e}")

    def save_annotated_image(
        self,
        annotated_image: np.ndarray,
        output_path: Path,
    ) -> None:
        """
        Save annotated image to file.

        Args:
            annotated_image: Image with annotations.
            output_path: Output file path.

        Raises:
            PostprocessingError: If save fails.
        """
        try:
            save_image(annotated_image, output_path)
            if self.logger:
                self.logger.info(f"Saved annotated image: {output_path}")
        except Exception as e:
            raise PostprocessingError(f"Failed to save annotated image: {e}")

        ymin, xmin, ymax, xmax = boxes[i]

        x1 = int(xmin * width)
        y1 = int(ymin * height)
        x2 = int(xmax * width)
        y2 = int(ymax * height)

        x1 = clip_box(x1, 0, width - 1)
        y1 = clip_box(y1, 0, height - 1)
        x2 = clip_box(x2, 0, width - 1)
        y2 = clip_box(y2, 0, height - 1)

        print(f"det {i} | score={score:.3f} | class_id={int(classes[i])} | box(px)=({x1}, {y1}, {x2}, {y2})")

        if x2 <= x1 or y2 <= y1:
            print(f"det {i} skipped because box is invalid after clipping")
            continue

        class_id = int(classes[i])

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), box_thickness)

        label = f"class={class_id} score={score:.2f}"
        text_x = x1
        text_y = max(20, y1 - 10)

        cv2.putText(annotated, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0),
                    text_thickness, cv2.LINE_AA)

        drawn += 1

    print(f"Detections drawn: {drawn}")
    return annotated


def save_annotated_image(image: np.ndarray, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "annotated.jpg"

    ok = cv2.imwrite(str(output_path), image)
    if not ok:
        raise RuntimeError(f"Failed to save annotated image to: {output_path}")

    return output_path
