import csv
import json


def save_summaries_to_json(data, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_summaries_to_csv(all_summaries, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "filename",
        "status",
        "error",
        "num_detections",
        "top_score",
        "detected_classes",
        "annotated_image",
        "debug_dir"
    ]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for summary in all_summaries:
            row = {
                "filename": summary.get("filename"),
                "status": summary.get("status"),
                "error": summary.get("error"),
                "num_detections": summary.get("num_detections"),
                "top_score": summary.get("top_score"),
                "detected_classes": ",".join(str(cls) for cls in summary.get("detected_classes", [])),
                "annotated_image": summary.get("annotated_image"),
                "debug_dir": summary.get("debug_dir")
            }
            writer.writerow(row)


def save_run_metadata_to_json(metadata, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
