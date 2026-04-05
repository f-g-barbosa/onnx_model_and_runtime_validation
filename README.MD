# ONNX Runtime Validation Pipeline

Real world tooling for ONNX model onboarding, runtime inspection, and batch validation reporting.

## What this project is

This project was built to validate ONNX object detection models in a way that is closer to day to day engineering work than a one-off inference script.

Instead of running a model on a single image and stopping there, this pipeline processes a full folder of inputs, saves per-image artifacts, generates structured summaries, and records run metadata so the execution can be inspected later.

## Why it matters

A model that runs once is not the same thing as a model that is ready to be evaluated, debugged, and onboarded into a larger workflow.

When teams work with ONNX models, they often need to answer practical questions:

- Does the model behave consistently across multiple images?
- What exactly did it output at runtime?
- Which images failed?
- Which images produced detections, and how many?
- Can the run be reproduced later with the same parameters?

This project focuses on those questions.

## What the pipeline does

- Loads an ONNX model with ONNX Runtime
- Inspects model inputs and outputs
- Runs inference over a folder of images
- Saves raw runtime tensors per image
- Saves annotated output images
- Generates per-image summaries in JSON and CSV
- Generates an aggregated batch report
- Saves run metadata for traceability
- Tracks success and failure per image
- Exposes configurable behavior through CLI flags

## What this project demonstrates

This is mainly a systems oriented ML project. The goal is not training a model, but building reliable tooling around model execution.

It demonstrates:

- batch inference  workflow design
- runtime artifact generation
- structured reporting
- failure-tolerant processing
- reproducibility-minded execution
- clean separation between CLI orchestration and pipeline logic

## Tech stack

- Python
- ONNX Runtime
- NumPy
- OpenCV

## Project structure

```text
.
├── main.py
├── models/
├── data/
├── outputs/
├── src/
│   ├── pipelines/
│   │   └── batch_validator.py
│   ├── runtime/
│   │   └── run_infer.py
│   ├── postprocess/
│   │   └── detection_visualizer.py
│   └── utils/
│       └── io_utils.py
