from __future__ import annotations

import argparse
from pathlib import Path

from detection_visualizer import (draw_detections, parse_detection_outputs, save_annotated_image)
from run_infer import (create_session, get_input_name, load_image, preprocess_image, print_model_io, run_inference,
                       save_debug_output)


def parse_args():
    parser = argparse.ArgumentParser(description="Run ONNX inference on an image and save annotated detections.")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--input", type=str, required=True, help="Path to input image")
    parser.add_argument("--size", type=int, nargs=2, default=[640, 640], metavar=("WIDTH", "HEIGHT"),
                        help="Resize image to WIDTH HEIGHT before inference")
    parser.add_argument("--score-threshold", type=float, default=0.5, help="Minimum score to draw a detection")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Directory to save raw outputs and annotated image")
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = Path(args.model)
    image_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    image = load_image(str(image_path))
    original_image, input_tensor = preprocess_image(image, args.size[0], args.size[1])

    session = create_session(str(model_path))
    print(f"Available providers: {session.get_providers()}")
    print_model_io(session)

    input_name = get_input_name(session)
    print(f"\nUsing input name: {input_name}")
    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Input tensor dtype: {input_tensor.dtype}")

    outputs = run_inference(session, input_name, input_tensor)
    save_debug_output(outputs, output_dir)

    boxes, scores, classes, num_detections = parse_detection_outputs(outputs)

    print(f"\nboxes shape: {boxes.shape}")
    print(f"scores shape: {scores.shape}")
    print(f"classes shape: {classes.shape}")
    print(f"num_detections: {num_detections}")
    print(f"first box: {boxes[0]}")
    print(f"first 5 scores: {scores[:5]}")
    print(f"first 5 classes: {classes[:5]}")

    annotated = draw_detections(image=original_image, boxes=boxes, scores=scores, classes=classes,
                                num_detections=num_detections, score_threshold=args.score_threshold)

    annotated_path = save_annotated_image(annotated, output_dir)
    print(f"\nAnnotated image saved to: {annotated_path}")
    print("Done.")


if __name__ == "__main__":
    main()
