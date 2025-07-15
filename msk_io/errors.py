"""Custom exception classes for the MSK-IO application."""

class MSKIOError(Exception):
    def __init__(self, message: str = "An MSK-IO error occurred.", code: int = 500) -> None:
        self.message = message
        self.code = code
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"MSKIOError (Code: {self.code}): {self.message}"

class ConfigurationError(MSKIOError):
    def __init__(self, message: str = "Configuration error.", code: int = 501) -> None:
        super().__init__(message, code)

class DataValidationError(MSKIOError):
    def __init__(self, message: str = "Data validation error.", code: int = 502) -> None:
        super().__init__(message, code)

class ProcessingError(MSKIOError):
    def __init__(self, message: str = "Data processing error.", code: int = 503) -> None:
        super().__init__(message, code)

class ImageProcessingError(ProcessingError):
    def __init__(self, message: str = "Image processing error.", code: int = 504) -> None:
        super().__init__(message, code)

class LLMInferenceError(ProcessingError):
    def __init__(self, message: str = "LLM inference error.", code: int = 505) -> None:
        super().__init__(message, code)

class RetrievalError(MSKIOError):
    def __init__(self, message: str = "Data retrieval error.", code: int = 506) -> None:
        super().__init__(message, code)

class IndexingError(MSKIOError):
    def __init__(self, message: str = "Semantic indexing error.", code: int = 507) -> None:
        super().__init__(message, code)

class AgentOrchestrationError(MSKIOError):
    def __init__(self, message: str = "Agent orchestration error.", code: int = 508) -> None:
        super().__init__(message, code)

class ExternalServiceError(MSKIOError):
    def __init__(self, message: str = "External service error.", code: int = 509) -> None:
        super().__init__(message, code)
