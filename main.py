import argparse
import json
from pathlib import Path

from src.postprocess.detection_visualizer import save_annotated_image
from src.runtime.run_infer import (create_session, get_input_name, load_image, preprocess_image, print_model_io,
                                   run_inference, save_debug_output)
from src.postprocess.detection_visualizer import (parse_detection_outputs, draw_detections)


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


def process_image(
        image_path,
        session,
        input_name,
        size,
        score_threshold,
        output_dir,
        annotated_output_dir,
):
    image = load_image(str(image_path))
    original_image, input_tensor = preprocess_image(image, size[0], size[1])

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

    annotated = draw_detections(
        image=original_image,
        boxes=boxes,
        scores=scores,
        classes=classes,
        num_detections=num_detections,
        score_threshold=score_threshold,
    )

    annotated_output_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = annotated_output_dir / f"{image_path.stem}_annotated{image_path.suffix}"

    # reaproveitando a função existente salvaria sempre annotated.jpg,
    # então aqui salvamos diretamente com nome por imagem
    import cv2

    ok = cv2.imwrite(str(annotated_path), annotated)
    if not ok:
        raise RuntimeError(f"Failed to save annotated image to: {annotated_path}")

    valid_scores = []
    valid_classes = []

    for i in range(num_detections):
        score = float(scores[i])
        if score < score_threshold:
            continue

        valid_scores.append(score)
        valid_classes.append(int(classes[i]))

    summary = {
        "filename": image_path.name,
        "num_detections": len(valid_scores),
        "top_score": max(valid_scores) if valid_scores else 0.0,
        "detected_classes": sorted(set(valid_classes)),
        "annotated_image": str(annotated_path),
    }

    return summary


def process_image_folder(input_dir, session, input_name, size, score_threshold, output_dir, annotated_output_dir):
    all_summaries = []

    valid_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    image_paths = sorted(
        [path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in valid_suffixes])

    if not image_paths:
        print(f"No valid images found in: {input_dir}")
        return all_summaries

    print(f"Found {len(image_paths)} image(s) in {input_dir}")

    for image_path in image_paths:
        print(f"\nProcessing: {image_path.name}")

        try:
            summary = process_image(image_path=image_path, session=session, input_name=input_name, size=size,
                                    score_threshold=score_threshold, output_dir=output_dir,
                                    annotated_output_dir=annotated_output_dir)

            all_summaries.append(summary)

        except Exception as e:
            print(f"Failed to process {image_path.name}: {e}")

    return all_summaries


def save_summaries_to_json(all_summaries, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)


def build_batch_report(all_summaries):
    total_images = len(all_summaries)
    images_with_detections = sum(1 for s in all_summaries if s["num_detections"] > 0)
    total_detections = sum(s["num_detections"] for s in all_summaries)

    if total_images > 0:
        avg_detections = total_detections / total_images
        max_top_score = max(s["top_score"] for s in all_summaries)
    else:
        avg_detections = 0.0
        max_top_score = 0.0

    return {
        "total_images": total_images,
        "images_with_detections": images_with_detections,
        "total_detections": total_detections,
        "avg_detections_per_image": avg_detections,
        "max_top_score": max_top_score,
    }


def main():
    args = parse_args()

    model_path = Path(args.model)
    input_dir = Path(args.input)
    output_dir = Path(args.output_dir)
    annotated_output_dir = output_dir / "annotated"
    summaries_json_path = output_dir / "summaries.json"
    batch_report_json_path = output_dir / "batch_report.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not input_dir.exists():
        raise FileNotFoundError(f"Input image not found: {input_dir}")

    session = create_session(str(model_path))
    print(f"Available providers: {session.get_providers()}")
    print_model_io(session)

    input_name = get_input_name(session)
    print(f"\nUsing input name: {input_name}")

    all_summaries = process_image_folder(input_dir=input_dir, session=session, input_name=input_name,
                                         size=(args.size[0], args.size[1]), score_threshold=args.score_threshold,
                                         output_dir=output_dir, annotated_output_dir=annotated_output_dir)

    batch_report = build_batch_report(all_summaries)

    save_summaries_to_json(all_summaries, summaries_json_path)
    save_summaries_to_json(batch_report, batch_report_json_path)

    print(f"\nSaved summaries JSON to: {summaries_json_path}")

    print("\nSUMMARY")
    for summary in all_summaries:
        print(summary)
    print("\nDone.")


if __name__ == "__main__":
    main()
