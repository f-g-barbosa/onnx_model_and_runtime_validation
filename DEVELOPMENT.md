# Development Guide

Guide for developers extending and maintaining the Bank-Ready AI Model Validation & Governance Pipeline.

## Project Structure

```
project_root/
├── main.py                 # Entry point
├── requirements.txt        # Python dependencies
├── configs/               # YAML configuration files
├── data/                  # Input images
├── models/                # ONNX models
├── outputs/               # Generated reports and artifacts
├── src/                   # Source code
│   ├── __init__.py
│   ├── core/             # Core types and utilities
│   ├── config/           # Configuration loading
│   ├── logging_utils/    # Logging implementation
│   ├── runtime/          # ONNX Runtime integration
│   ├── preprocess/       # Image preprocessing
│   ├── postprocess/      # Output parsing and visualization
│   ├── onboarding/       # Model validation
│   ├── validation/       # Inference validation
│   ├── governance/       # Policy and promotion
│   ├── reporting/        # Report generation
│   ├── review/           # Review workflow
│   └── utils/            # Utilities
├── ARCHITECTURE.md       # System design documentation
├── readme.md            # Quick start guide
└── DEVELOPMENT.md       # This file
```

## Development Workflow

### 1. Setting Up Development Environment

```bash
# Clone repository
git clone <repo-url>
cd project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development tools (optional)
pip install pytest pytest-cov black flake8 mypy
```

### 2. Code Organization Principles

#### Separation of Concerns
- **Core**: Pure data types and enumerations
- **Config**: External configuration management
- **Logic**: Business logic separated from I/O
- **Utils**: Reusable utilities without dependencies

#### Type Safety
```python
# Always use type hints
def process_images(
    image_paths: List[Path],
    config: PreprocessingConfig,
) -> List[np.ndarray]:
    """Process images according to config."""
    ...
```

#### Dataclasses for Data
```python
# Use dataclasses for immutable data structures
@dataclass
class ValidationMetrics:
    image_name: str
    num_detections: int
    passed_threshold: bool
    anomalies: List[str] = field(default_factory=list)
```

#### Error Handling
```python
# Use custom exceptions with context
from src.core.exceptions import PreprocessingError

try:
    image = load_image(path)
except Exception as e:
    raise PreprocessingError(f"Failed to load {path}: {e}")
```

### 3. Adding New Features

#### Example: Adding a Custom Validation Rule

1. **Define the validation logic**:
```python
# src/validation/custom_validator.py

class CustomValidator(SingleImageValidator):
    """Custom validation logic."""
    
    def validate_custom_rule(
        self,
        metrics: ValidationMetrics,
    ) -> bool:
        """Check custom rule."""
        # Implement logic
        return True
```

2. **Integrate into pipeline**:
```python
# In main.py
custom_validator = CustomValidator(config.validation, logger)
for metrics in image_metrics:
    if not custom_validator.validate_custom_rule(metrics):
        metrics.anomalies.append("custom_rule_violation")
```

#### Example: Adding a New Report Type

1. **Create report builder**:
```python
# src/reporting/custom_report.py

class CustomReportBuilder:
    def build_report(self, data: dict) -> dict:
        """Build custom report."""
        return {
            "type": "custom_report",
            "data": data,
        }
```

2. **Generate report in main.py**:
```python
builder = CustomReportBuilder(logger)
report = builder.build_report(data)
builder.save_report(report, output_path)
```

### 4. Configuration Management

#### Adding New Configuration Section

1. **Create schema in src/core/schemas.py**:
```python
@dataclass
class MyNewConfig:
    param1: str
    param2: float = 0.5
```

2. **Update PipelineConfig**:
```python
@dataclass
class PipelineConfig:
    # ... existing fields ...
    my_new_config: MyNewConfig
```

3. **Add loader method in src/config/loader.py**:
```python
def _load_my_new_config(self) -> MyNewConfig:
    config_file = self.config_dir / "my_new_config.yaml"
    data = self._load_yaml(config_file)
    return MyNewConfig(
        param1=data.get("param1", "default"),
        param2=float(data.get("param2", 0.5)),
    )
```

4. **Create YAML file in configs/**:
```yaml
# configs/my_new_config.yaml
param1: "value"
param2: 0.7
```

### 5. Testing

#### Unit Tests Structure

```
tests/
├── test_core/
├── test_config/
├── test_runtime/
├── test_validation/
├── test_governance/
├── test_reporting/
└── test_utils/
```

#### Example Test

```python
# tests/test_validation/test_single_image_validator.py

import pytest
from src.validation.single_image_validator import SingleImageValidator
from src.core.types import ParsedDetection, DetectionBox
from src.core.schemas import ValidationConfig

def test_validation_passes_with_detections():
    config = ValidationConfig(score_threshold=0.5)
    validator = SingleImageValidator(config)
    
    detection = ParsedDetection(
        boxes=[DetectionBox(0.1, 0.1, 0.5, 0.5, 0, 0.9)],
        num_detections=1,
    )
    
    metrics = validator.validate("test.jpg", detection, 100.0)
    
    assert metrics.num_detections == 1
    assert metrics.passed_threshold == True
```

#### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_validation/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_validation/test_single_image_validator.py::test_validation_passes_with_detections -v
```

### 6. Code Quality

#### Type Checking

```bash
# Type check with mypy
mypy src/ --ignore-missing-imports
```

#### Code Formatting

```bash
# Format code with black
black src/ --line-length=100

# Check style with flake8
flake8 src/ --max-line-length=100
```

#### Pre-commit Hooks (Optional)

Create `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
      - id: black
  - repo: https://github.com/PyCQA/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
```

### 7. Logging Best Practices

#### Use Structured Logging

```python
from src.logging_utils.logger import StructuredLogger

logger.info(
    "Image validation completed",
    image_name="test.jpg",
    num_detections=5,
    score=0.95,
    # Include relevant context as kwargs
)
```

#### Audit Logging

```python
from src.logging_utils.logger import AuditLogger

audit_logger.log_model_promotion(
    model_name="my_model",
    approved=True,
    reviewer="alice@company.com",
    policy_results={"accuracy": True, "regression": True},
    notes="Good performance on test set",
)
```

### 8. Performance Optimization

#### Profiling

```python
from src.utils.time_utils import Timer

with Timer() as timer:
    # Code to profile
    process_images(image_list)

print(f"Time: {timer.elapsed_ms()}ms")
```

#### Memory Monitoring

```python
import psutil
import os

process = psutil.Process(os.getpid())
mem_mb = process.memory_info().rss / (1024 * 1024)
print(f"Memory: {mem_mb}MB")
```

### 9. Documentation Standards

#### Module Docstring

```python
"""
Module description.

This module handles [what it does] by [how it does it].

Key classes:
    - MyClass: Does thing A
    
Key functions:
    - my_function: Does thing B
"""
```

#### Function/Class Docstring

```python
def validate_model(
    model_path: Path,
    strict_mode: bool = False,
) -> bool:
    """
    Validate ONNX model for compatibility.
    
    Args:
        model_path: Path to ONNX model file.
        strict_mode: Whether to check all features.
    
    Returns:
        True if model passes validation.
    
    Raises:
        ModelNotFoundError: If model file not found.
        InvalidModelError: If model cannot be loaded.
    """
```

### 10. Debugging Tips

#### Enable Verbose Logging

Set in `configs/logging.yaml`:
```yaml
log_level: DEBUG
```

#### Use VS Code Debugger

Set breakpoints and press F5. Use Debug Console for inspection:
```python
# In debug console
import json
print(json.dumps(some_dict, indent=2))
```

#### Print Model I/O

```python
io_inspector = IOInspector(session, logger)
io_inspector.print_model_io()
```

#### Check Raw Model Outputs

```python
# Enable save_debug_outputs in configs/output.yaml
# Then examine .npy files:
import numpy as np
output = np.load("outputs/debugs/image_name/output_0.npy")
print(output.shape, output.dtype)
```

## Common Tasks

### Add Support for New Model Type

1. Create model-specific output parser in `src/postprocess/`:
```python
class MyModelOutputParser(OutputParser):
    def parse_outputs(self, outputs):
        # Custom parsing logic
        pass
```

2. Update main.py to use new parser
3. Update validation rules in config

### Integrate with External System

1. Create interface in `src/utils/`:
```python
class ExternalSystemClient:
    def push_report(self, report_path: Path) -> bool:
        # Integration logic
        pass
```

2. Call from reporting module

### Add Monitoring/Metrics

1. Extend `StructuredLogger`:
```python
logger.info(
    "Custom metric",
    metric_name="latency",
    metric_value=123.45,
)
```

2. Parse logs with external monitoring tool

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt pytest
      - run: pytest tests/ -v
```

## Deployment Considerations

- Use `--dry-run` for validation before production runs
- Archive `audit/` logs for compliance
- Version models and configurations together
- Monitor `outputs/audit/*.jsonl` for anomalies
- Set up alerts for policy violations
- Regular backup of audit logs

## Troubleshooting

### Import Errors
Ensure `PYTHONPATH` includes project root:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### ONNX Runtime Provider Issues
```python
# Check available providers
import onnxruntime as ort
print(ort.get_available_providers())

# Use CPU as fallback
providers = ["CPUExecutionProvider"]
```

### Memory Issues with Large Batches
- Process images in smaller batches
- Use `--dry-run` for testing
- Monitor with `psutil.Process(os.getpid()).memory_info()`

## Release Checklist

- [ ] Update version in `src/__init__.py`
- [ ] Update README with new features
- [ ] Update ARCHITECTURE.md if structure changed
- [ ] Run full test suite
- [ ] Check code coverage > 80%
- [ ] Format code with black
- [ ] Create release notes
- [ ] Tag commit with version

---

For more information, see [ARCHITECTURE.md](ARCHITECTURE.md)
