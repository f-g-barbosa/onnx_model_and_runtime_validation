# Model Validation & Governance Pipeline

Real world tooling for ONNX model onboarding, runtime inspection, and batch validation reporting.

**See [ARCHITECTURE.md](ARCHITECTURE.md) for system design and detailed documentation.**

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Settings
Edit YAML files in `configs/`:
- `runtime.yaml` - Model path and execution settings
- `validation.yaml` - Detection thresholds and rules
- `promotion_policy.yaml` - Policy gates for promotion
- `preprocessing.yaml` - Image preprocessing parameters
- `output.yaml` - Output directory configuration
- `logging.yaml` - Logging settings

### 3. Prepare Input Data
Place images in `data/` directory (supported: .jpg, .png, .jpeg, .bmp)

### 4. Run Pipeline

**Debug with VS Code:**
- Press **F5** and select a configuration
- Or use terminal:

```bash
# Standard run
python main.py --config-dir configs --input-dir data

# Dry run (no file output)
python main.py --config-dir configs --input-dir data --dry-run

# Custom batch ID
python main.py --batch-id my_experiment_001
```

## What This Pipeline Does

### Model Onboarding
• Validate ONNX model files
• Inspect model signatures (inputs/outputs)
• Verify execution environment compatibility

### Batch Validation
• Preprocess images (resize, normalize)
• Run inference with profiling
• Parse object detection outputs
• Apply validation rules per image
• Aggregate batch-level statistics

### Governance & Policy
• Automatic policy gates (accuracy, regression, latency)
• Human review workflow
• Decision documentation
• Audit trails with timestamps

### Reporting & Compliance
• Structured JSON batch reports
• Comprehensive audit reports
• Per-image metrics CSV
• Annotated validation images
• JSONL audit event logs

## Output Artifacts

Pipeline generates reports in `outputs/`:
```
batch_report_batch_id.json      # Validation results and metrics
audit_report_batch_id.json      # Governance and policy decisions
debugs/                         # Raw model outputs (.npy files)
annotated/                      # Images with drawn bounding boxes
audit/audit_TIMESTAMP.jsonl     # Event audit trail
```

## Key Concepts

### Configuration-Driven
All thresholds and behavior controlled via YAML - no code changes required.

### Type-Safe
Full type hints and dataclasses for IDE support and runtime safety.

### Structured Logging
JSON-formatted logs for easy parsing and compliance auditing.

### Policy Enforcement
Automatic gates check accuracy, regression, latency against configured policies.

### Human Review
Built-in workflow for governance decisions with audit documentation.

## Architecture

Clean modular design with clear separation of concerns:

```
core/        → Types, enums, exceptions, schemas
config/      → YAML configuration management
logging/     → Structured and audit logging
runtime/     → ONNX session and inference
preprocess/  → Image preprocessing
postprocess/ → Output parsing and visualization
onboarding/  → Model validation
validation/  → Inference validation
governance/  → Policy enforcement
reporting/   → Report generation
review/      → Human review workflow
utils/       → Common utilities
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for complete module documentation.

## Example Usage

### Run Full Validation Pipeline
```bash
python main.py --batch-id my_eperiemnt_001 --input-dir data
```

### Dry Run (Testing)
```bash
python main.py --batch-id my_eperiemnt_001 --input-dir data --dry-run
```

### Debug Mode
```bash
python main.py --batch-id debug_run --input-dir data 2>&1 | head -50
```

## Typical Workflow

1. **Configure** - Set thresholds, model paths, policies in YAML
2. **Prepare** - Add images to input directory
3. **Run** - Execute pipeline with `python main.py`
4. **Review** - Examine JSON reports and annotated images
5. **Decide** - Make promotion decision based on policy results
6. **Audit** - Archive reports for compliance

## Error Handling

Clear, meaningful error messages with custom exception types:

```
ValidationPipelineError (base)
├─ ModelNotFoundError
├─ InvalidModelError
├─ PreprocessingError
├─ InferenceError
├─ ValidationError
├─ PolicyViolationError
└─ ConfigurationError
```

## Performance

• Batch inference with profiling
• Configurable ONNX providers (CPU, GPU)
• Memory-efficient processing
• Latency monitoring per image

## Enterprise Features

- Policy-driven promotion gates
- Role-based approvals
- Audit-ready reports
- Structured logging
- Type safety with hints
- Docker-ready
- Kubernetes-ready

## Testing

```bash
# Unit tests (add tests/ folder)
pytest tests/ -v --cov=src
```

## Extending

See [ARCHITECTURE.md](ARCHITECTURE.md) for extending with custom validators, report builders, and policy checks.

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design, module docs, best practices
- **configs/*.yaml** - Configuration templates and explanations
- **src/** - Inline docstrings in all modules

---

**Last Updated:** April 2026
**Version:** 1.0.0


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
