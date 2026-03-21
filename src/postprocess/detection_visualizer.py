from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def parse_detection_outputs(outputs: list[np.ndarray]):
    """
    Expected output order:
    outputs[0] -> boxes          shape: (1, N, 4)
    outputs[1] -> scores         shape: (1, N)
    outputs[2] -> classes        shape: (1, N)
    outputs[3] -> num_detections shape: (1,)
    """

    if len(outputs) < 4:
        raise RuntimeError(f"Expected at least 4 outputs, got {len(outputs)}")

    boxes = outputs[0][0]
    classes= outputs[1][0]
    scores = outputs[2][0]
    num_detections = int(outputs[3][0])

    return boxes, scores, classes, num_detections


def clip_box(value: int, min_value: int, max_value: int):
    return max(min_value, min(value, max_value))


def draw_detections(image: np.ndarray, boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, num_detections: int,
                    score_threshold: float):
    annotated = image.copy()
    height, width = annotated.shape[:2]

    font_scale = max(0.5, height / 1000)
    text_thickness = max(1, int(height / 500))
    box_thickness = max(2, int(height / 300))

    drawn = 0

    for i in range(num_detections):
        score = float(scores[i])
        if score < score_threshold:
            continue

        print(f"det {i} | score={score:.3f} | class_id={int(classes[i])} | box(normalized)={boxes[i]}")

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

        cv2.putText(annotated, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), text_thickness, cv2.LINE_AA)

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
