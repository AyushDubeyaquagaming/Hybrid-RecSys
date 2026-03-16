class PipelineError(Exception):
    """Base exception for pipeline failures."""


class ExternalServiceError(PipelineError):
    """Raised when an external dependency such as MongoDB is unavailable."""


class DataValidationError(PipelineError):
    """Raised when pipeline inputs or intermediate data are invalid."""


class ModelTrainingError(PipelineError):
    """Raised when model training fails."""


class ArtifactExportError(PipelineError):
    """Raised when artifact export fails."""