import argparse
from pathlib import Path
from collections import Counter

import cv2

from src.runtime.run_infer import (create_session, get_input_name, preprocess_image, print_model_io, run_inference,
                                   save_debug_output)
from src.postprocess.detection_visualizer import (draw_detections, parse_detection_outputs, save_annotated_image, )
from src.utils.io_utils import save_summaries_to_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ONNX inference on a folder of images and save annotated detections.")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--input", type=str, required=True, help="Path to input image folder")
    parser.add_argument("--size", type=int, nargs=2, default=[640, 640], metavar=("WIDTH", "HEIGHT"),
                        help="Resize each image to WIDTH HEIGHT before inference")
    parser.add_argument("--score-threshold", type=float, default=0.5, help="Minimum score to keep a detection")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Directory to save debug outputs, summaries, and annotated images")
    return parser.parse_args()


def process_image(image_path, session, input_name, size, score_threshold, output_dir, annotated_output_dir):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    original_bgr, input_tensor = preprocess_image(image=image, width=size[0], height=size[1])

    outputs = run_inference(session, input_name, input_tensor)

    image_debug_dir = output_dir / "debugs" / image_path.stem
    image_debug_dir.mkdir(parents=True, exist_ok=True)
    save_debug_output(outputs, image_debug_dir)

    boxes, scores, classes, num_detections = parse_detection_outputs(outputs)

    annotated = draw_detections(image=original_bgr.copy(), boxes=boxes, scores=scores, classes=classes,
                                score_threshold=score_threshold, )

    annotated_output_dir.mkdir(parents=True, exist_ok=True)
    annotated_image_path = annotated_output_dir / f"{image_path.stem}_annotated.jpg"
    save_annotated_image(annotated, annotated_image_path)

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
        "annotated_image": annotated_image_path.as_posix(),
        "debug_dir": image_debug_dir.as_posix(),
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
            summary = process_image(
                image_path=image_path,
                session=session,
                input_name=input_name,
                size=size,
                score_threshold=score_threshold,
                output_dir=output_dir,
                annotated_output_dir=annotated_output_dir,
            )
            all_summaries.append(summary)
        except Exception as e:
            print(f"Failed to process {image_path.name}: {e}")

    return all_summaries


def build_batch_report(all_summaries):
    total_images = len(all_summaries)
    images_with_detections = sum(1 for s in all_summaries if s["num_detections"] > 0)
    images_without_detections = total_images - images_with_detections
    total_detections = sum(s["num_detections"] for s in all_summaries)

    if total_images > 0:
        avg_detections = total_detections / total_images
        max_top_score = max(s["top_score"] for s in all_summaries)
        detection_rate = images_with_detections / total_images
    else:
        avg_detections = 0.0
        max_top_score = 0.0
        detection_rate = 0.0

    class_counter = Counter()
    for summary in all_summaries:
        for cls in summary["detected_classes"]:
            class_counter[int(cls)] += 1

    top_images_by_detection_count = sorted(
        [
            {
                "filename": s["filename"],
                "num_detections": s["num_detections"],
                "top_score": s["top_score"],
            }
            for s in all_summaries
        ],
        key=lambda x: x["num_detections"],
        reverse=True,
    )[:5]

    top_images_by_score = sorted(
        [{"filename": s["filename"], "top_score": s["top_score"], "num_detections": s["num_detections"], }
         for s in all_summaries],
        key=lambda x: x["top_score"],
        reverse=True,
    )[:5]

    return {
        "total_images": total_images,
        "images_with_detections": images_with_detections,
        "images_without_detections": images_without_detections,
        "detection_rate": detection_rate,
        "total_detections": total_detections,
        "avg_detections_per_image": avg_detections,
        "max_top_score": max_top_score,
        "class_frequency": dict(sorted(class_counter.items())),
        "top_images_by_detection_count": top_images_by_detection_count,
        "top_images_by_score": top_images_by_score,
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
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    session = create_session(str(model_path))
    print(f"Available providers: {session.get_providers()}")
    print_model_io(session)

    input_name = get_input_name(session)
    print(f"\nUsing input name: {input_name}")

    all_summaries = process_image_folder(
        input_dir=input_dir,
        session=session,
        input_name=input_name,
        size=(args.size[0], args.size[1]),
        score_threshold=args.score_threshold,
        output_dir=output_dir,
        annotated_output_dir=annotated_output_dir,
    )

    batch_report = build_batch_report(all_summaries)

    save_summaries_to_json(all_summaries, summaries_json_path)
    save_summaries_to_json(batch_report, batch_report_json_path)

    print(f"\nSaved summaries JSON to: {summaries_json_path}")
    print(f"Saved batch report JSON to: {batch_report_json_path}")

    print("\nSUMMARY")
    for summary in all_summaries:
        print(summary)

    print("\nDone.")


if __name__ == "__main__":
    main()
