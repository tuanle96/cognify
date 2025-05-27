"""
Structured logging configuration for Cognify application.

Provides JSON-formatted logging with correlation IDs and performance metrics.
"""

import logging
import sys
from typing import Any, Dict

import structlog
from structlog.types import EventDict, Processor

from app.core.config import get_settings

settings = get_settings()


def add_correlation_id(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """
    Add correlation ID to log events for request tracing.
    """
    # This will be enhanced with actual correlation ID from request context
    event_dict["correlation_id"] = getattr(logger, "_correlation_id", None)
    return event_dict


def add_service_info(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """
    Add service information to log events.
    """
    event_dict["service"] = "cognify"
    event_dict["version"] = settings.VERSION
    event_dict["environment"] = settings.ENVIRONMENT
    return event_dict


def add_performance_metrics(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """
    Add performance metrics to log events when available.
    """
    # Add timing information if available
    if hasattr(logger, "_start_time"):
        import time
        event_dict["duration_ms"] = (time.time() - logger._start_time) * 1000
    
    return event_dict


def filter_sensitive_data(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """
    Filter sensitive data from log events.
    """
    sensitive_keys = {
        "password", "token", "api_key", "secret", "authorization",
        "x-api-key", "x-auth-token", "cookie", "session"
    }
    
    def _filter_dict(data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively filter sensitive data from dictionaries."""
        if not isinstance(data, dict):
            return data
        
        filtered = {}
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                filtered[key] = "[REDACTED]"
            elif isinstance(value, dict):
                filtered[key] = _filter_dict(value)
            elif isinstance(value, list):
                filtered[key] = [_filter_dict(item) if isinstance(item, dict) else item for item in value]
            else:
                filtered[key] = value
        return filtered
    
    # Filter the entire event dict
    return _filter_dict(event_dict)


def setup_logging() -> None:
    """
    Setup structured logging configuration.
    """
    # Configure processors based on environment
    processors: list[Processor] = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        add_service_info,
        add_correlation_id,
        add_performance_metrics,
        filter_sensitive_data,
    ]
    
    if settings.LOG_FORMAT == "json":
        # JSON formatting for production
        processors.extend([
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ])
    else:
        # Human-readable formatting for development
        processors.extend([
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer(colors=True)
        ])
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        context_class=dict,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.LOG_LEVEL),
    )
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(
        logging.INFO if settings.DATABASE_ECHO else logging.WARNING
    )
    
    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)


class PerformanceLogger:
    """
    Context manager for performance logging.
    """
    
    def __init__(self, operation: str, logger: Any = None):
        self.operation = operation
        self.logger = logger or structlog.get_logger()
        self.start_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        self.logger.info(f"Starting {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        duration = (time.time() - self.start_time) * 1000
        
        if exc_type is None:
            self.logger.info(
                f"Completed {self.operation}",
                duration_ms=duration,
                status="success"
            )
        else:
            self.logger.error(
                f"Failed {self.operation}",
                duration_ms=duration,
                status="error",
                error_type=exc_type.__name__,
                error_message=str(exc_val)
            )


class ChunkingLogger:
    """
    Specialized logger for chunking operations.
    """
    
    def __init__(self, file_path: str, language: str, purpose: str):
        self.logger = structlog.get_logger("chunking")
        self.context = {
            "file_path": file_path,
            "language": language,
            "purpose": purpose
        }
    
    def log_strategy_selection(self, strategy: str, factors: Dict[str, Any]):
        """Log chunking strategy selection."""
        self.logger.info(
            "Chunking strategy selected",
            strategy=strategy,
            factors=factors,
            **self.context
        )
    
    def log_agent_decision(self, agent: str, decision: Dict[str, Any], reasoning: str):
        """Log agent chunking decision."""
        self.logger.info(
            "Agent chunking decision",
            agent=agent,
            decision=decision,
            reasoning=reasoning,
            **self.context
        )
    
    def log_quality_assessment(self, quality_score: float, dimensions: Dict[str, float]):
        """Log quality assessment results."""
        self.logger.info(
            "Chunking quality assessment",
            quality_score=quality_score,
            dimensions=dimensions,
            **self.context
        )
    
    def log_performance_metrics(self, metrics: Dict[str, Any]):
        """Log chunking performance metrics."""
        self.logger.info(
            "Chunking performance metrics",
            metrics=metrics,
            **self.context
        )


def get_logger(name: str = None) -> Any:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (optional)
    
    Returns:
        Structured logger instance
    """
    return structlog.get_logger(name)


def get_chunking_logger(file_path: str, language: str, purpose: str) -> ChunkingLogger:
    """
    Get a specialized chunking logger.
    
    Args:
        file_path: Path to the file being chunked
        language: Programming language
        purpose: Chunking purpose
    
    Returns:
        ChunkingLogger instance
    """
    return ChunkingLogger(file_path, language, purpose)
