from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ONNX inference on a single image.")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--input", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=[640, 640],
        metavar=("WIDTH", "HEIGHT"),
        help="Model input size",
    )
    return parser.parse_args()


def load_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def preprocess_image(image: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    original_bgr = image.copy()
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    input_tensor = np.expand_dims(rgb.astype(np.uint8), axis=0)  # NHWC uint8
    return original_bgr, input_tensor


def create_session(model_path: str) -> ort.InferenceSession:
    providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(model_path, providers=providers)


def get_input_name(session: ort.InferenceSession) -> str:
    inputs = session.get_inputs()
    if not inputs:
        raise RuntimeError("Model has no inputs.")
    return inputs[0].name


def run_inference(session: ort.InferenceSession, input_name: str, input_tensor: np.ndarray) -> list[np.ndarray]:
    start = time.perf_counter()
    outputs = session.run(None, {input_name: input_tensor})
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    print(f"Inference time: {elapsed_ms:.2f} ms")
    return outputs


def save_debug_output(outputs: list[np.ndarray], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, output in enumerate(outputs):
        np.save(output_dir / f"output_{idx}.npy", output)
        print(f"Saved output_{idx}.npy with shape {output.shape} and dtype {output.dtype}")


def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    image_path = Path(args.input)
    output_dir = Path("outputs")

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    image = load_image(str(image_path))
    _, input_tensor = preprocess_image(image, args.size[0], args.size[1])

    session = create_session(str(model_path))
    input_name = get_input_name(session)

    print(f"Model input name: {input_name}")
    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Available providers: {session.get_providers()}")

    outputs = run_inference(session, input_name, input_tensor)
    save_debug_output(outputs, output_dir)


if __name__ == "__main__":
    main()