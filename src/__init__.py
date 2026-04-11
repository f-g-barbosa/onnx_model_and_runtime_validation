"""
Bank-Ready AI Model Validation & Governance Pipeline

A production-grade ONNX model validation and governance system designed for
regulated environments (e.g., financial institutions).

Main Features:
- Model Onboarding: Comprehensive model validation and signature checking
- Batch Validation: Image-by-image inference with configurable rules
- Policy Enforcement: Automated decision gates for model promotion
- Governance: Human review workflow and audit trails
- Reporting: Structured JSON/CSV reports for compliance

Architecture:
    src/
    ├─ core/       - Types, enums, exceptions, schemas
    ├─ config/     - YAML configuration loaders
    ├─ logging_utils/ - Structured and audit logging
    ├─ runtime/    - ONNX session management
    ├─ preprocess/ - Image preprocessing
    ├─ postprocess/- Model output parsing and visualization
    ├─ onboarding/ - Model validation and onboarding
    ├─ validation/ - Image and batch validation
    ├─ governance/ - Policy engine and promotion gates
    ├─ reporting/  - Report generation
    ├─ review/     - Human review workflow
    └─ utils/      - Common utilities
"""

__version__ = "1.0.0"
__author__ = "ML Engineering Team"

from src.core import *
from src.config import *
from src.logging_utils import *
from src.runtime import *
from src.preprocess import *
from src.postprocess import *
from src.onboarding import *
from src.validation import *
from src.governance import *
from src.reporting import *
from src.review import *
from src.utils import *

__all__ = [
    "__version__",
    "__author__",
]
