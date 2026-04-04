from collections import Counter

import cv2

from src.runtime.run_infer import preprocess_image, run_inference, save_debug_output
from src.postprocess.detection_visualizer import (
    parse_detection_outputs,
    draw_detections,
    save_annotated_image,
)


def process_image(
    image_path,
    session,
    input_name,
    size,
    score_threshold,
    output_dir,
    annotated_output_dir,
    save_debug,
    save_annotated,
):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    original_bgr, input_tensor = preprocess_image(
        image=image,
        width=size[0],
        height=size[1],
    )

    outputs = run_inference(session, input_name, input_tensor)

    image_debug_dir = output_dir / "debugs" / image_path.stem
    if save_debug:
        image_debug_dir.mkdir(parents=True, exist_ok=True)
        save_debug_output(outputs, image_debug_dir)
        debug_dir_str = image_debug_dir.as_posix()
    else:
        debug_dir_str = None

    boxes, scores, classes, num_detections = parse_detection_outputs(outputs)

    annotated_image_path = None
    if save_annotated:
        annotated = draw_detections(
            image=original_bgr.copy(),
            boxes=boxes,
            scores=scores,
            classes=classes,
            num_detections=num_detections,
            score_threshold=score_threshold,
        )

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

    return {
        "filename": image_path.name,
        "status": "success",
        "error": None,
        "num_detections": len(valid_scores),
        "top_score": max(valid_scores) if valid_scores else 0.0,
        "detected_classes": sorted(set(valid_classes)),
        "annotated_image": annotated_image_path.as_posix() if annotated_image_path else None,
        "debug_dir": debug_dir_str,
    }


def process_image_folder(
    input_dir,
    session,
    input_name,
    size,
    score_threshold,
    output_dir,
    annotated_output_dir,
    save_debug,
    save_annotated,
):
    all_summaries = []
    valid_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    image_paths = sorted(
        [
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in valid_suffixes
        ]
    )

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
                save_debug=save_debug,
                save_annotated=save_annotated,
            )

            all_summaries.append(summary)

        except Exception as e:
            error_message = str(e)
            print(f"Failed to process {image_path.name}: {error_message}")

            failure_summary = {
                "filename": image_path.name,
                "status": "failed",
                "error": error_message,
                "num_detections": 0,
                "top_score": 0.0,
                "detected_classes": [],
                "annotated_image": None,
                "debug_dir": None,
            }
            all_summaries.append(failure_summary)

    return all_summaries


def build_batch_report(all_summaries, top_k=5):
    total_images = len(all_summaries)
    successful_images = sum(1 for s in all_summaries if s["status"] == "success")
    failed_images = sum(1 for s in all_summaries if s["status"] == "failed")

    images_with_detections = sum(
        1 for s in all_summaries
        if s["status"] == "success" and s["num_detections"] > 0
    )
    images_without_detections = sum(
        1 for s in all_summaries
        if s["status"] == "success" and s["num_detections"] == 0
    )
    total_detections = sum(
        s["num_detections"] for s in all_summaries if s["status"] == "success"
    )

    if successful_images > 0:
        avg_detections = total_detections / successful_images
        max_top_score = max(
            s["top_score"] for s in all_summaries if s["status"] == "success"
        )
        detection_rate = images_with_detections / successful_images
    else:
        avg_detections = 0.0
        max_top_score = 0.0
        detection_rate = 0.0

    class_counter = Counter()
    for summary in all_summaries:
        if summary["status"] != "success":
            continue
        for cls in summary["detected_classes"]:
            class_counter[int(cls)] += 1

    successful_summaries = [s for s in all_summaries if s["status"] == "success"]

    top_images_by_detection_count = sorted(
        [
            {
                "filename": s["filename"],
                "num_detections": s["num_detections"],
                "top_score": s["top_score"],
            }
            for s in successful_summaries
        ],
        key=lambda x: x["num_detections"],
        reverse=True,
    )[:top_k]

    top_images_by_score = sorted(
        [
            {
                "filename": s["filename"],
                "top_score": s["top_score"],
                "num_detections": s["num_detections"],
            }
            for s in successful_summaries
        ],
        key=lambda x: x["top_score"],
        reverse=True,
    )[:top_k]

    return {
        "total_images": total_images,
        "successful_images": successful_images,
        "failed_images": failed_images,
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