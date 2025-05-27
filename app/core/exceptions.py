"""
Custom exception classes for Cognify application.

Provides structured error handling with proper HTTP status codes and error details.
"""

from typing import Any, Dict, Optional


class CognifyException(Exception):
    """
    Base exception class for Cognify-specific errors.
    """

    def __init__(
        self,
        message: str,
        error_code: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(CognifyException):
    """Exception raised for validation errors."""

    def __init__(self, message: str, field: str = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=400,
            details={"field": field, **(details or {})}
        )


class AuthenticationError(CognifyException):
    """Exception raised for authentication errors."""

    def __init__(self, message: str = "Authentication failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            status_code=401,
            details=details
        )


class AuthorizationError(CognifyException):
    """Exception raised for authorization errors."""

    def __init__(self, message: str = "Access denied", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR",
            status_code=403,
            details=details
        )


class NotFoundError(CognifyException):
    """Exception raised when a resource is not found."""

    def __init__(self, resource: str, identifier: str = None, details: Optional[Dict[str, Any]] = None):
        message = f"{resource} not found"
        if identifier:
            message += f": {identifier}"

        super().__init__(
            message=message,
            error_code="NOT_FOUND",
            status_code=404,
            details={"resource": resource, "identifier": identifier, **(details or {})}
        )


class ConflictError(CognifyException):
    """Exception raised for resource conflicts."""

    def __init__(self, message: str, resource: str = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="CONFLICT",
            status_code=409,
            details={"resource": resource, **(details or {})}
        )


class RateLimitError(CognifyException):
    """Exception raised when rate limits are exceeded."""

    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            status_code=429,
            details={"retry_after": retry_after, **(details or {})}
        )


# Chunking-specific exceptions

class ChunkingError(CognifyException):
    """Base exception for chunking-related errors."""

    def __init__(self, message: str, file_path: str = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="CHUNKING_ERROR",
            status_code=500,
            details={"file_path": file_path, **(details or {})}
        )


class AgentError(ChunkingError):
    """Exception raised when agent processing fails."""

    def __init__(self, message: str, agent_name: str, file_path: str = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Agent {agent_name} failed: {message}",
            file_path=file_path,
            details={"agent_name": agent_name, **(details or {})}
        )
        self.error_code = "AGENT_ERROR"


class QualityThresholdError(ChunkingError):
    """Exception raised when chunking quality is below threshold."""

    def __init__(self, quality_score: float, threshold: float, file_path: str = None, details: Optional[Dict[str, Any]] = None):
        message = f"Chunking quality {quality_score:.3f} below threshold {threshold:.3f}"
        super().__init__(
            message=message,
            file_path=file_path,
            details={"quality_score": quality_score, "threshold": threshold, **(details or {})}
        )
        self.error_code = "QUALITY_THRESHOLD_ERROR"


class UnsupportedLanguageError(ChunkingError):
    """Exception raised for unsupported programming languages."""

    def __init__(self, language: str, file_path: str = None, details: Optional[Dict[str, Any]] = None):
        message = f"Unsupported language: {language}"
        super().__init__(
            message=message,
            file_path=file_path,
            details={"language": language, **(details or {})}
        )
        self.error_code = "UNSUPPORTED_LANGUAGE"


class ParsingError(ChunkingError):
    """Exception raised when code parsing fails."""

    def __init__(self, message: str, language: str, file_path: str = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Parsing failed for {language}: {message}",
            file_path=file_path,
            details={"language": language, **(details or {})}
        )
        self.error_code = "PARSING_ERROR"


# Database-specific exceptions

class DatabaseError(CognifyException):
    """Exception raised for database-related errors."""

    def __init__(self, message: str, operation: str = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            status_code=500,
            details={"operation": operation, **(details or {})}
        )


class VectorDBError(CognifyException):
    """Exception raised for vector database errors."""

    def __init__(self, message: str, operation: str = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="VECTOR_DB_ERROR",
            status_code=500,
            details={"operation": operation, **(details or {})}
        )


class PermissionError(CognifyException):
    """Exception raised for permission/authorization errors."""

    def __init__(self, message: str, resource_type: str = None, resource_id: str = None, required_permission: str = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="PERMISSION_DENIED",
            status_code=403,
            details={
                "resource_type": resource_type,
                "resource_id": resource_id,
                "required_permission": required_permission,
                **(details or {})
            }
        )


class ProcessingError(CognifyException):
    """Exception raised for document processing errors."""

    def __init__(self, message: str, processing_stage: str = None, document_id: str = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="PROCESSING_ERROR",
            status_code=500,
            details={
                "processing_stage": processing_stage,
                "document_id": document_id,
                **(details or {})
            }
        )


# External service exceptions

class ExternalServiceError(CognifyException):
    """Exception raised for external service errors."""

    def __init__(self, service: str, message: str, status_code: int = 502, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"{service} service error: {message}",
            error_code="EXTERNAL_SERVICE_ERROR",
            status_code=status_code,
            details={"service": service, **(details or {})}
        )


class LLMError(ExternalServiceError):
    """Exception raised for LLM service errors."""

    def __init__(self, provider: str, message: str, model: str = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            service=f"LLM ({provider})",
            message=message,
            details={"provider": provider, "model": model, **(details or {})}
        )
        self.error_code = "LLM_ERROR"


class EmbeddingError(ExternalServiceError):
    """Exception raised for embedding service errors."""

    def __init__(self, provider: str, message: str, model: str = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            service=f"Embedding ({provider})",
            message=message,
            details={"provider": provider, "model": model, **(details or {})}
        )
        self.error_code = "EMBEDDING_ERROR"


# Configuration exceptions

class ConfigurationError(CognifyException):
    """Exception raised for configuration errors."""

    def __init__(self, message: str, setting: str = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            status_code=500,
            details={"setting": setting, **(details or {})}
        )


# Utility functions for exception handling

def handle_external_service_error(service: str, error: Exception) -> ExternalServiceError:
    """
    Convert external service errors to standardized format.

    Args:
        service: Name of the external service
        error: Original exception

    Returns:
        ExternalServiceError: Standardized exception
    """
    if hasattr(error, 'status_code'):
        status_code = error.status_code
    elif hasattr(error, 'response') and hasattr(error.response, 'status_code'):
        status_code = error.response.status_code
    else:
        status_code = 502

    return ExternalServiceError(
        service=service,
        message=str(error),
        status_code=status_code,
        details={"original_error": type(error).__name__}
    )


def handle_chunking_error(error: Exception, file_path: str = None, context: Dict[str, Any] = None) -> ChunkingError:
    """
    Convert chunking errors to standardized format.

    Args:
        error: Original exception
        file_path: Path to the file being processed
        context: Additional context information

    Returns:
        ChunkingError: Standardized chunking exception
    """
    if isinstance(error, ChunkingError):
        return error

    return ChunkingError(
        message=str(error),
        file_path=file_path,
        details={
            "original_error": type(error).__name__,
            "context": context or {}
        }
    )
