"""
Custom exceptions for the validation pipeline.

Provides domain-specific exception types for better error handling and clarity.
"""


class ValidationPipelineError(Exception):
    """Base exception for all pipeline errors."""
    pass


class ModelNotFoundError(ValidationPipelineError):
    """Raised when model file is not found or inaccessible."""
    pass


class InvalidModelError(ValidationPipelineError):
    """Raised when model is invalid or cannot be loaded."""
    pass


class OnboardingError(ValidationPipelineError):
    """Raised during model onboarding validation failures."""
    pass


class SignatureValidationError(ValidationPipelineError):
    """Raised when model signature validation fails."""
    pass


class PreprocessingError(ValidationPipelineError):
    """Raised during image preprocessing failures."""
    pass


class InferenceError(ValidationPipelineError):
    """Raised during inference execution failures."""
    pass


class PostprocessingError(ValidationPipelineError):
    """Raised during output postprocessing failures."""
    pass


class ValidationError(ValidationPipelineError):
    """Raised during validation checks."""
    pass


class PolicyViolationError(ValidationPipelineError):
    """Raised when policy checks fail."""
    pass


class ConfigurationError(ValidationPipelineError):
    """Raised when configuration is invalid or missing."""
    pass


class InputDataError(ValidationPipelineError):
    """Raised when input data is invalid or inaccessible."""
    pass


class ReportGenerationError(ValidationPipelineError):
    """Raised when report generation fails."""
    pass
