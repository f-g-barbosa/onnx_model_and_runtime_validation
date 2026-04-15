# Bank-Ready AI Model Validation & Governance Pipeline

A production-grade ONNX model validation and governance system designed for regulated environments (financial institutions, healthcare, aviation, etc.).

## Overview

This pipeline provides end-to-end model management capabilities:
- **Model Onboarding** - Validate and register models
- **Batch Validation** - Run inference on image batches with policy checks  
- **Governance** - Policy-based gates and human review workflows
- **Reporting** - Audit-ready compliance reports

## Architecture

```
src/
├─ core/              # Types, enums, exceptions, schemas
│  ├─ types.py       # Dataclass definitions (ModelMetadata, ValidationMetrics, etc.)
│  ├─ enums.py       # Enumeration types (ValidationStatus, ModelStatus, etc.)
│  ├─ exceptions.py  # Custom exception hierarchy
│  └─ schemas.py     # Configuration dataclasses
│
├─ config/           # Configuration management
│  └─ loader.py      # YAML configuration loader
│
├─ logging_utils/    # Structured logging
│  └─ logger.py      # StructuredLogger and AuditLogger
│
├─ runtime/          # ONNX Runtime management
│  ├─ session_factory.py    # Session creation
│  ├─ io_inspector.py       # Model I/O inspection
│  └─ inference_runner.py   # Inference execution
│
├─ preprocess/       # Image preprocessing
│  └─ image_preprocessor.py # Resize, normalize, tensorize
│
├─ postprocess/      # Output processing
│  ├─ output_parser.py         # Parse model outputs
│  └─ detection_visualizer.py  # Draw boxes, save annotations
│
├─ onboarding/       # Model validation
│  └─ model_metadata_validator.py  # Signature & metadata checks
│
├─ validation/       # Inference validation
│  ├─ single_image_validator.py   # Per-image rules
│  ├─ batch_validator.py          # Batch summary
│  ├─ runtime_validator.py        # Performance monitoring
│  └─ baseline_comparator.py      # A/B comparison
│
├─ governance/       # Policy enforcement
│  ├─ policy_engine.py           # Policy gates
│  ├─ review_gate.py             # Human review workflow
│  └─ promotion_recommender.py   # Recommendations
│
├─ reporting/        # Report generation
│  ├─ batch_report_builder.py    # Batch validation report
│  ├─ audit_report_builder.py    # Comprehensive audit report
│  └─ json_writer.py             # JSON I/O
│
├─ review/          # Review workflow
│  ├─ flagged_sample_selector.py      # Sample selection
│  └─ review_template_builder.py      # Review templates
│
└─ utils/           # Common utilities
   ├─ file_utils.py      # File I/O, directory management
   ├─ image_utils.py     # Image loading, resizing
   ├─ path_utils.py      # Path resolution
   └─ time_utils.py      # Timing, formatting
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Edit YAML files in `configs/`:
- `runtime.yaml` - Model path, ONNX providers
- `validation.yaml` - Detection thresholds
- `promotion_policy.yaml` - Policy gates
- `preprocessing.yaml` - Image resize, normalization
- `output.yaml` - Output directories
- `logging.yaml` - Logging settings

### Run Pipeline

```bash
# Standard run
python main.py --config-dir configs --input-dir data

# Dry run (no file output)
python main.py --config-dir configs --input-dir data --dry-run

# Custom batch ID
python main.py --batch-id my_batch_001
```

### VS Code Debugging

Press **F5** or use the Debug menu:
- "Pipeline: Standard Run" - Normal execution
- "Pipeline: Dry Run" - Validation without output
- "Pipeline: With Debug Output" - Detailed trace

## Key Concepts

### Model Metadata

Every model is catalogued with:
```python
ModelMetadata(
    model_name="ssd_mobilenet_v1",
    model_version="1.0.0",
    model_path=Path("models/ssd_mobilenet_v1_12.onnx"),
    input_shape=(1, 640, 640, 3),
    output_names=["boxes", "classes", "scores", "num_detections"],
    provider="CPUExecutionProvider"
)
```

### Validation Metrics

Per-image validation results:
```python
ValidationMetrics(
    image_name="test.jpg",
    num_detections=5,
    max_score=0.95,
    passed_threshold=True,
    anomalies=[]  # Empty if no issues
)
```

### Policy Gates

Automatic promotion checks:
- **Accuracy Gate** - `accuracy >= min_accuracy_pct`
- **Regression Gate** - `regression <= max_regression_pct`
- **Latency Gate** - `latency_ms <= max_latency_ms`

### Review Workflow

```
Model Ready → Policy Checks → Recommendation
   ↓
   If all pass: APPROVE / HOLD
   If any fail: REJECT / REVIEW
   ↓
Human Review (if required)
   ↓
Final Decision → Audit Log
```

## Configuration YAML Reference

### runtime.yaml
```yaml
model_path: models/ssd_mobilenet_v1_12.onnx
providers:
  - CPUExecutionProvider
input_image_size: [640, 640]
verbose: false
```

### validation.yaml
```yaml
score_threshold: 0.5
max_detections_per_image: null
min_detections_per_image: 0
allowed_classes: null
latency_threshold_ms: null
check_consistency: true
```

### promotion_policy.yaml
```yaml
min_accuracy_pct: 95.0
max_regression_pct: 2.0
max_latency_ms: null
require_human_review: true
approval_required_roles:
  - model_reviewer
  - ml_ops_engineer
```

## Output Artifacts

After pipeline execution in `outputs/`:

```
outputs/
├─ batch_report_batch_id.json       # Detailed batch validation report
├─ audit_report_batch_id.json       # Governance audit trail
├─ debugs/                          # Raw model outputs (.npy)
│  └─ image_name/
│     ├─ output_0.npy
│     └─ ...
├─ annotated/                       # Images with drawn boxes
│  └─ image_name_annotated.jpg
└─ audit/                           # Audit event logs
   └─ audit_TIMESTAMP.jsonl
```

## Example Report

Batch validation report JSON:
```json
{
  "batch_id": "batch_20260104_120000",
  "timestamp": "2026-01-04T12:00:00.000000",
  "summary": {
    "total_images": 100,
    "successful": 95,
    "failed": 5,
    "success_rate_pct": 95.0,
    "avg_num_detections": 3.2,
    "avg_max_score": 0.87
  },
  "problematic_images": ["image_1.jpg", "image_5.jpg"],
  "image_details": [...]
}
```

## Error Handling

Custom exception hierarchy for clear error context:

```python
ValidationPipelineError
├─ ModelNotFoundError
├─ InvalidModelError
├─ OnboardingError
├─ PreprocessingError
├─ InferenceError
├─ PostprocessingError
├─ ValidationError
├─ PolicyViolationError
├─ ConfigurationError
└─ InputDataError
```

## Best Practices

1. **Configuration First** - All thresholds in YAML, not hardcoded
2. **Type Hints** - Full type annotations for IDE support
3. **Logging** - Structured logs for audit trails
4. **Error Context** - Custom exceptions with meaningful messages
5. **Modularity** - Clear separation of concerns
6. **Dataclasses** - Immutable data structures
7. **Testing** - Unit tests for each module (add tests/ folder)

## Docker Integration

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENTRYPOINT ["python", "main.py"]
CMD ["--config-dir", "configs", "--input-dir", "data"]
```

## Kubernetes Integration

Deploy with ConfigMaps for configuration:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: validation-config
data:
  runtime.yaml: |
    model_path: models/...
    ...
```

## Performance Tuning

- **Batch Inference** - Process multiple images in parallel
- **GPU Support** - Set `providers: [CUDAExecutionProvider]` in config
- **Memory Efficiency** - Stream results instead of accumulating
- **Latency Monitoring** - Track inference times per image

## Extending the Pipeline

### Add Custom Validation Rule
```python
class MyCustomValidator(SingleImageValidator):
    def validate(self, ...):
        metrics = super().validate(...)
        # Add custom checks
        if my_condition:
            metrics.anomalies.append("my_anomaly")
        return metrics
```

### Add New Report Type
```python
class MyReportBuilder(BatchReportBuilder):
    def build_custom_report(self, ...):
        # Build custom report logic
        return report_dict
```

## Testing

```bash
# Run unit tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## Monitoring & Logging

Structured logs in `outputs/audit/audit_TIMESTAMP.jsonl`:
```json
{
  "timestamp": "2026-01-04T12:00:01.000000",
  "event_type": "model_promotion_decision",
  "model_name": "ssd_mobilenet_v1",
  "status": "approved",
  "details": {"reviewer": "alice@company.com", "notes": "..."}
}
```

## Compliance & Audit

• Structured audit logs (JSONL format)
• Policy decision trails
• Human review records  
• Model versioning
• Timestamped reports
• Role-based approvals

## License

Internal use only

## Contributors

ML Engineering Team
