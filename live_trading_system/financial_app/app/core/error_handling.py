"""
Comprehensive error handling module for the Trading Strategies Application.

Provides a robust error handling system with:
- Standardized error categories
- Detailed error tracking
- Contextual error information
"""

from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum
from typing import (
    Any, 
    Dict, 
    Optional, 
    Union, 
    Callable,  # Add this
    Tuple,     # Add this
    Type       # Add this
)
import functools  # Add this for retry_operation
import time       # Add this for retry_operation
import random
from fastapi import HTTPException, status
from pydantic import BaseModel, Field


# Configure logging
logger = logging.getLogger(__name__)


class ErrorCategory(str, Enum):
    """Standardized error categories for the application."""
    DATABASE = "database"
    DATABASE_CONNECTION = "database_connection"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    BUSINESS_LOGIC = "business_logic"
    EXTERNAL_SERVICE = "external_service"
    RATE_LIMIT = "rate_limit"
    OPERATIONAL = "operational"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class ErrorSeverity(str, Enum):
    """Error severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AppErrorDetail(BaseModel):
    """
    Structured error detail for comprehensive error tracking.
    """
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    message: str
    error_category: ErrorCategory
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    error_code: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    component: Optional[str] = None


class AppError(Exception):
    """
    Base application error class for standardized error handling.
    """
    def __init__(
        self,
        message: str,
        error_category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        component: Optional[str] = None,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    ):
        """
        Initialize a comprehensive application error.
        """
        self.detail = AppErrorDetail(
            message=message,
            error_category=error_category,
            severity=severity,
            error_code=error_code,
            context=context or {},
            component=component
        )
        self.status_code = status_code
        super().__init__(message)

    def log(self, logger_obj: Optional[logging.Logger] = None) -> None:
        """
        Log the error with appropriate context.
        """
        log_method = (logger_obj or logger).error
        log_method(
            f"{self.detail.error_category.upper()} Error: {self.detail.message}",
            extra={
                "error_detail": self.detail.dict(),
                "severity": self.detail.severity,
                "component": self.detail.component
            }
        )

    def to_http_exception(self) -> HTTPException:
        """
        Convert error to a FastAPI HTTPException.
        """
        return HTTPException(
            status_code=self.status_code,
            detail={
                "message": self.detail.message,
                "category": self.detail.error_category,
                "timestamp": self.detail.timestamp.isoformat()
            }
        )


class DatabaseError(AppError):
    """
    Base error for database-related operations.
    """
    def __init__(
        self,
        message: str,
        db_name: Optional[str] = None,
        **kwargs
    ):
        kwargs.setdefault('error_category', ErrorCategory.DATABASE)
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        
        context = kwargs.get('context', {})
        if db_name:
            context['database'] = db_name
        kwargs['context'] = context
        
        super().__init__(message, **kwargs)


class DatabaseConnectionError(DatabaseError):
    """
    Specific error for database connection failures.
    """
    def __init__(
        self,
        message: str,
        db_name: Optional[str] = None,
        connection_details: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        kwargs.setdefault('error_category', ErrorCategory.DATABASE_CONNECTION)
        kwargs.setdefault('severity', ErrorSeverity.CRITICAL)
        kwargs.setdefault('status_code', status.HTTP_503_SERVICE_UNAVAILABLE)
        
        context = kwargs.get('context', {})
        if connection_details:
            context.update(connection_details)
        kwargs['context'] = context
        
        super().__init__(
            message, 
            db_name=db_name, 
            **kwargs
        )


class OperationalError(AppError):
    """
    Error representing operational issues in the system.
    """
    def __init__(
        self,
        message: str,
        **kwargs
    ):
        kwargs.setdefault('error_category', ErrorCategory.OPERATIONAL)
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        kwargs.setdefault('status_code', status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        super().__init__(message, **kwargs)


class ValidationError(AppError):
    """
    Error for data validation failures.
    """
    def __init__(
        self,
        message: str,
        **kwargs
    ):
        kwargs.setdefault('error_category', ErrorCategory.VALIDATION)
        kwargs.setdefault('severity', ErrorSeverity.MEDIUM)
        kwargs.setdefault('status_code', status.HTTP_422_UNPROCESSABLE_ENTITY)
        
        super().__init__(message, **kwargs)


class AuthenticationError(AppError):
    """
    Error for authentication-related failures.
    """
    def __init__(
        self,
        message: str,
        **kwargs
    ):
        kwargs.setdefault('error_category', ErrorCategory.AUTHENTICATION)
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        kwargs.setdefault('status_code', status.HTTP_401_UNAUTHORIZED)
        
        super().__init__(message, **kwargs)

class RateLimitExceededError(AppError):
    """
    Error for rate limit exceeded scenarios.
    """
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        **kwargs
    ):
        kwargs.setdefault('error_category', ErrorCategory.RATE_LIMIT)
        kwargs.setdefault('severity', ErrorSeverity.MEDIUM)
        kwargs.setdefault('status_code', status.HTTP_429_TOO_MANY_REQUESTS)
        
        super().__init__(message, **kwargs)




class ErrorTracker:
    """
    Centralized error tracking and monitoring utility.
    """
    # Global error tracking
    _error_stats = {
        "counters": {},  # Error counters by category
        "last_errors": {},  # Last error by category
        "error_timestamps": {},  # Timestamps of errors by category
    }

    @classmethod
    def track_error(
        cls, 
        error: Union[AppError, Exception], 
        category: Optional[ErrorCategory] = None
    ) -> None:
        """
        Track error occurrence for monitoring and circuit breaker purposes.
        
        Args:
            error: The error that occurred
            category: Optional category override (for non-AppError exceptions)
        """
        # Determine the error category
        if isinstance(error, AppError):
            error_cat = error.detail.error_category
        else:
            error_cat = category or ErrorCategory.UNKNOWN
        
        # Initialize category in tracking dicts if not present
        if error_cat not in cls._error_stats["counters"]:
            cls._error_stats["counters"][error_cat] = 0
            cls._error_stats["last_errors"][error_cat] = None
            cls._error_stats["error_timestamps"][error_cat] = []
        
        # Update error statistics
        cls._error_stats["counters"][error_cat] += 1
        cls._error_stats["last_errors"][error_cat] = error
        
        # Keep last 100 timestamps for rate calculation
        timestamps = cls._error_stats["error_timestamps"][error_cat]
        timestamps.append(datetime.utcnow())
        cls._error_stats["error_timestamps"][error_cat] = timestamps[-100:]

    @classmethod
    def get_error_rate(
        cls, 
        category: ErrorCategory, 
        seconds: int = 60
    ) -> float:
        """
        Calculate the rate of errors in a specific category over a time period.
        
        Args:
            category: Error category to check
            seconds: Time window in seconds
            
        Returns:
            Error rate (errors per second)
        """
        if category not in cls._error_stats["error_timestamps"]:
            return 0.0
        
        # Get timestamps in the specified window
        now = datetime.utcnow()
        window_start = now.timestamp() - seconds
        recent_timestamps = [
            ts for ts in cls._error_stats["error_timestamps"][category]
            if ts.timestamp() > window_start
        ]
        
        # Calculate error rate
        return len(recent_timestamps) / seconds

    @classmethod
    def clear_error_stats(cls) -> None:
        """Reset error statistics (mainly for testing)."""
        cls._error_stats["counters"] = {}
        cls._error_stats["last_errors"] = {}
        cls._error_stats["error_timestamps"] = {}

    @classmethod
    def get_error_stats(cls) -> Dict[str, Any]:
        """
        Retrieve current error statistics.
        
        Returns:
            Dictionary of current error tracking data
        """
        return {
            "counters": cls._error_stats["counters"].copy(),
            "last_errors": {
                k: str(v) for k, v in cls._error_stats["last_errors"].items()
            },
            "error_timestamps": {
                k: [ts.isoformat() for ts in v] 
                for k, v in cls._error_stats["error_timestamps"].items()
            }
        }
# Convenience function for error conversion
def convert_exception(e: Exception) -> AppError:
    """
    Convert standard exceptions to AppError types.
    
    Args:
        e: Exception to convert
    
    Returns:
        Converted AppError
    """
    if isinstance(e, ValueError):
        return ValidationError(str(e))
    elif isinstance(e, PermissionError):
        return AuthenticationError(str(e))
    elif isinstance(e, ConnectionError):
        return DatabaseConnectionError(str(e))
    
    # Default to generic AppError
    return AppError(
        f"Unexpected error: {str(e)}",
        error_category=ErrorCategory.UNKNOWN,
        severity=ErrorSeverity.CRITICAL
    )


def format_error_message(
    template: str,
    **kwargs: Any
) -> str:
    """
    Format error message with consistent structure and variable substitution.
    
    Args:
        template: Message template with placeholders
        **kwargs: Values to substitute in the template
        
    Returns:
        Formatted error message
        
    Example:
        msg = format_error_message(
            "Failed to execute trade for {symbol} at {price}", 
            symbol="AAPL", 
            price=150.75
        )
    """
    try:
        return template.format(**kwargs)
    except KeyError as e:
        logger.warning(f"Missing key {e} in error message template: {template}")
        return template + f" (Missing data: {e})"
    except Exception as e:
        logger.warning(f"Error formatting message template: {e}")
        return template


def create_error_context(
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Create a standardized error context dictionary.
    
    Args:
        **kwargs: Key-value pairs to include in the context
        
    Returns:
        Standardized context dictionary
        
    Example:
        context = create_error_context(
            user_id="123", 
            trade_id="trade_456", 
            symbol="AAPL"
        )
    """
    return {k: v for k, v in kwargs.items() if v is not None}


def log_critical_error(
    error: Union[AppError, Exception],
    context: Optional[Dict[str, Any]] = None,
    logger_obj: Optional[logging.Logger] = None,
    track: bool = True
) -> None:
    """
    Log a critical error with enhanced tracking and alerting.
    
    Args:
        error: The error to log
        context: Additional context for the error
        logger_obj: Optional logger to use (defaults to module logger)
    """
    # Convert to AppError if not already
    app_error = error if isinstance(error, AppError) else convert_exception(error)
    
    # Add context if provided
    if context:
        app_error.detail.context.update(context)
    
    # Log the error
    app_error.log(logger_obj or logger)
    
    if track:
        ErrorTracker.track_error(app_error)
    
    # Optional: Additional alerting mechanism could be added here
    # For example, sending an email or triggering a notification system


def retry_operation(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    allowed_exceptions: Tuple[Type[Exception], ...] = (Exception,)
) -> Callable:
    """
    Decorator for retrying operations with advanced error handling and exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay between retries
        allowed_exceptions: Tuple of exception types that trigger a retry
    
    Returns:
        Decorated function with retry logic
    
    Example:
        @retry_operation(max_retries=3, delay=1.0, allowed_exceptions=(ValueError, ConnectionError))
        def fetch_data():
            # Function implementation
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_retry = 0
            current_delay = delay
            
            while current_retry < max_retries:
                try:
                    return func(*args, **kwargs)
                except allowed_exceptions as e:
                    current_retry += 1
                    
                    # Prepare error context with retry information
                    error_context = {
                        "retry_attempt": current_retry,
                        "max_retries": max_retries,
                        "function_name": func.__name__,
                        "exception_type": type(e).__name__
                    }
                    
                    # Log the error with context
                    log_critical_error(
                        e, 
                        context=error_context,
                        track=True  # Optionally track the error
                    )
                    
                    # Check if we've exhausted retries
                    if current_retry >= max_retries:
                        # Raise the last encountered exception
                        raise
                    
                    # Calculate exponential backoff with jitter
                    jitter = random.uniform(0.8, 1.2)
                    sleep_time = current_delay * jitter
                    
                    # Wait before retrying
                    time.sleep(sleep_time)
                    
                    # Increase delay for next retry
                    current_delay *= backoff_factor
            
            # Fallback error (should never be reached)
            raise RuntimeError(f"Retry failed for function {func.__name__}")
        
        return wrapper
    
    return decorator


__all__ = [
    'AppError',
    'DatabaseError',
    'DatabaseConnectionError',
    'OperationalError',
    'ValidationError',
    'AuthenticationError',
    'RateLimitExceededError',
    'ErrorCategory',
    'ErrorSeverity',
    'ErrorTracker',
    'convert_exception',
    'format_error_message',
    'create_error_context',
    'log_critical_error',
    'retry_operation'
]