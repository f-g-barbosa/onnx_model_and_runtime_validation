from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def load_image(image_path: str):
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def preprocess_image(image: np.ndarray, width: int, height: int):
    """
    SSD-MobileNetV1 ONNX expects:
    - NHWC
    - uint8
    """

    original_bgr = image.copy()
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    input_tensor = np.expand_dims(rgb.astype(np.uint8), axis=0)  # [1, H, W, 3]
    return original_bgr, input_tensor


def create_session(model_path: str):

    providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(model_path, providers=providers)


def print_model_io(session: ort.InferenceSession):
    print("\n=== MODEL INPUTS ===")
    for inp in session.get_inputs():
        print(f"Name:  {inp.name}")
        print(f"Shape: {inp.shape}")
        print(f"Type:  {inp.type}")
        print("-" * 40)

    print("\n=== MODEL OUTPUTS ===")
    for out in session.get_outputs():
        print(f"Name:  {out.name}")
        print(f"Shape: {out.shape}")
        print(f"Type:  {out.type}")
        print("-" * 40)


def get_input_name(session: ort.InferenceSession):

    inputs = session.get_inputs()

    if not inputs:
        raise RuntimeError("Model has no inputs.")
    return inputs[0].name


def run_inference(session: ort.InferenceSession, input_name: str, input_tensor: np.ndarray):

    start = time.perf_counter()
    outputs = session.run(None, {input_name: input_tensor})
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    print(f"\nInference time: {elapsed_ms:.2f} ms")
    return outputs


def save_debug_output(outputs: list[np.ndarray], output_dir: Path):

    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, output in enumerate(outputs):
        out_path = output_dir / f"output_{idx}.npy"
        np.save(out_path, output)
        print(f"Saved {out_path.name} | shape={output.shape} | dtype={output.dtype}")
