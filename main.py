import argparse
from pathlib import Path

from src.pipeline.batch_validator import build_batch_report, process_image_folder
from src.runtime.run_infer import create_session, get_input_name, print_model_io
from src.utils.io_utils import save_summaries_to_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ONNX inference on a folder of images and save annotated detections."
    )
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--input", type=str, required=True, help="Path to input image folder")
    parser.add_argument(
        "--size", type=int, nargs=2, default=[640, 640], metavar=("WIDTH", "HEIGHT"),
        help="Resize each image to WIDTH HEIGHT before inference")
    parser.add_argument("--score-threshold", type=float, default=0.5, help="Minimum score to keep a detection", )
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Directory to save debug outputs, summaries, and annotated images")
    parser.add_argument("--save-debug", action="store_true", help="Save raw model outputs (.npy) for each image")
    parser.add_argument("--save-annotated", action="store_true", help="Save annotated output images")
    parser.add_argument("--top-k-report", type=int, default=5,
                        help="Number of top images to keep in batch report sections")
    return parser.parse_args()


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
        save_debug=args.save_debug,
        save_annotated=args.save_annotated,
    )

    batch_report = build_batch_report(        all_summaries,        top_k=args.top_k_report)

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
