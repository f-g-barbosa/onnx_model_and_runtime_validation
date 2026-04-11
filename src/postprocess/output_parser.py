"""
Model output parsing for object detection models.

Parses raw ONNX outputs into structured detection results.
"""

from typing import List, Tuple, Optional
import numpy as np

from src.core.types import DetectionBox, ParsedDetection
from src.core.exceptions import PostprocessingError
from src.logging_utils.logger import StructuredLogger


class OutputParser:
    """Parses SSD MobileNetV1 detection outputs."""

    def __init__(self, logger: Optional[StructuredLogger] = None):
        """
        Initialize output parser.

        Args:
            logger: Optional logger instance.
        """
        self.logger = logger

    def parse_ssd_mobilenet_output(
        self,
        outputs: List[np.ndarray],
        score_threshold: float = 0.0,
    ) -> ParsedDetection:
        """
        Parse SSD MobileNetV1 model outputs.

        Model outputs:
        - output_0: detection boxes [1, N, 1, 4]
        - output_1: detection classes [1, N, 1, 1]
        - output_2: detection scores [1, N, 1, 1]
        - output_3: num_detections [1]

        Args:
            outputs: List of raw model outputs.
            score_threshold: Minimum score to include detection.

        Returns:
            ParsedDetection with boxes, classes, and scores.

        Raises:
            PostprocessingError: If outputs cannot be parsed.
        """
        try:
            if len(outputs) != 4:
                raise PostprocessingError(f"Expected 4 outputs, got {len(outputs)}")

            boxes_raw = outputs[0]
            classes_raw = outputs[1]
            scores_raw = outputs[2]
            num_dets = outputs[3]

            # Extract number of detections
            num_detections = int(num_dets[0])

            # Extract boxes, classes, scores
            boxes_list = []
            classes_list = []
            scores_list = []

            for i in range(min(num_detections, len(scores_raw[0]))):
                score = float(scores_raw[0, i, 0, 0])

                if score < score_threshold:
                    continue

                # Box coordinates [y_min, x_min, y_max, x_max]
                box = boxes_raw[0, i, 0, :]
                y_min, x_min, y_max, x_max = box

                class_id = int(classes_raw[0, i, 0, 0])

                boxes_list.append(
                    DetectionBox(
                        x_min=float(x_min),
                        y_min=float(y_min),
                        x_max=float(x_max),
                        y_max=float(y_max),
                        class_id=class_id,
                        score=score,
                    )
                )
                classes_list.append(class_id)
                scores_list.append(score)

            if self.logger:
                self.logger.debug(
                    "Parsed detections",
                    num_detections=num_detections,
                    passed_threshold=len(boxes_list),
                )

            return ParsedDetection(
                boxes=boxes_list,
                num_detections=num_detections,
                raw_scores=list(scores_raw[0, :num_detections, 0, 0]),
                raw_classes=list(classes_raw[0, :num_detections, 0, 0].astype(int)),
            )

        except Exception as e:
            raise PostprocessingError(f"Failed to parse model outputs: {e}")

    def parse_generic_output(
        self,
        outputs: List[np.ndarray],
    ) -> dict:
        """
        Parse generic model outputs.

        Args:
            outputs: List of raw model outputs.

        Returns:
            Dictionary with output information.
        """
        parsed = {
            "num_outputs": len(outputs),
            "output_shapes": [tuple(out.shape) for out in outputs],
            "output_dtypes": [str(out.dtype) for out in outputs],
        }

        return parsed
