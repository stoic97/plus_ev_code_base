"""
Comprehensive error handling module for the Trading Strategies Application.

This module provides:
- Custom exception types for different error scenarios
- Retry decorators with configurable backoff strategies
- Centralized error logging and tracking
- Fallback mechanisms for graceful degradation
"""

import functools
import logging
import random
import time
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, cast

from fastapi import HTTPException, status
from pydantic import BaseModel, Field

from app.core.config import settings

# Set up logging
logger = logging.getLogger(__name__)

# Type variable for function return typing
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


class ErrorSeverity(str, Enum):
    """Severity levels for application errors."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ErrorCategory(str, Enum):
    """Categories of errors for classification and handling."""
    DATABASE = "database"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    BUSINESS_LOGIC = "business_logic"
    EXTERNAL_SERVICE = "external_service"
    MARKET_DATA = "market_data"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class AppErrorDetail(BaseModel):
    """Detailed information about an application error."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    message: str
    error_category: ErrorCategory
    severity: ErrorSeverity
    error_code: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    retry_possible: bool = True
    component: Optional[str] = None


class AppError(Exception):
    """
    Base application error that provides structured error information.
    
    All custom application errors should inherit from this class.
    """
    def __init__(
        self,
        message: str,
        error_category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        retry_possible: bool = True,
        component: Optional[str] = None,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        *args
    ):
        self.detail = AppErrorDetail(
            message=message,
            error_category=error_category,
            severity=severity,
            error_code=error_code,
            context=context or {},
            retry_possible=retry_possible,
            component=component
        )
        self.status_code = status_code
        self.args = args
        super().__init__(message, *args)

    def log_error(self, log_level: int = logging.ERROR) -> None:
        """Log the error with appropriate context."""
        error_info = {
            "error_category": self.detail.error_category,
            "severity": self.detail.severity,
            "error_code": self.detail.error_code,
            "retry_possible": self.detail.retry_possible,
            "component": self.detail.component,
            "context": self.detail.context,
        }

        if self.detail.severity == ErrorSeverity.CRITICAL:
            log_level = logging.CRITICAL
        elif self.detail.severity == ErrorSeverity.HIGH:
            log_level = logging.ERROR
        elif self.detail.severity == ErrorSeverity.MEDIUM:
            log_level = logging.WARNING
        elif self.detail.severity == ErrorSeverity.LOW:
            log_level = logging.INFO
        
        logger.log(log_level, f"{self.detail.message}", extra=error_info)

    def to_http_exception(self) -> HTTPException:
        """Convert to FastAPI HTTPException for API responses."""
        return HTTPException(
            status_code=self.status_code,
            detail={
                "message": self.detail.message,
                "error_code": self.detail.error_code,
                "timestamp": self.detail.timestamp.isoformat(),
                "category": self.detail.error_category,
            }
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            "message": self.detail.message,
            "error_category": self.detail.error_category,
            "severity": self.detail.severity,
            "error_code": self.detail.error_code,
            "timestamp": self.detail.timestamp.isoformat(),
            "context": self.detail.context,
            "retry_possible": self.detail.retry_possible,
            "component": self.detail.component,
        }


class DatabaseError(AppError):
    """Error related to database operations."""
    def __init__(
        self,
        message: str,
        *args,
        db_component: Optional[str] = None,
        **kwargs
    ):
        kwargs.setdefault("error_category", ErrorCategory.DATABASE)
        kwargs.setdefault("component", db_component or "database")
        super().__init__(message, *args, **kwargs)


class ValidationError(AppError):
    """Error related to data validation."""
    def __init__(self, message: str, *args, **kwargs):
        kwargs.setdefault("error_category", ErrorCategory.VALIDATION)
        kwargs.setdefault("severity", ErrorSeverity.MEDIUM)
        kwargs.setdefault("status_code", status.HTTP_422_UNPROCESSABLE_ENTITY)
        super().__init__(message, *args, **kwargs)


class AuthenticationError(AppError):
    """Error related to authentication."""
    def __init__(self, message: str, *args, **kwargs):
        kwargs.setdefault("error_category", ErrorCategory.AUTHENTICATION)
        kwargs.setdefault("severity", ErrorSeverity.MEDIUM)
        kwargs.setdefault("status_code", status.HTTP_401_UNAUTHORIZED)
        super().__init__(message, *args, **kwargs)


class AuthorizationError(AppError):
    """Error related to authorization and permissions."""
    def __init__(self, message: str, *args, **kwargs):
        kwargs.setdefault("error_category", ErrorCategory.AUTHORIZATION)
        kwargs.setdefault("severity", ErrorSeverity.MEDIUM)
        kwargs.setdefault("status_code", status.HTTP_403_FORBIDDEN)
        super().__init__(message, *args, **kwargs)


class BusinessLogicError(AppError):
    """Error related to business logic violations."""
    def __init__(self, message: str, *args, **kwargs):
        kwargs.setdefault("error_category", ErrorCategory.BUSINESS_LOGIC)
        kwargs.setdefault("severity", ErrorSeverity.MEDIUM)
        kwargs.setdefault("status_code", status.HTTP_400_BAD_REQUEST)
        super().__init__(message, *args, **kwargs)


class ExternalServiceError(AppError):
    """Error related to external service calls."""
    def __init__(
        self,
        message: str,
        *args,
        service_name: Optional[str] = None,
        **kwargs
    ):
        kwargs.setdefault("error_category", ErrorCategory.EXTERNAL_SERVICE)
        kwargs.setdefault("component", service_name)
        super().__init__(message, *args, **kwargs)


class MarketDataError(AppError):
    """Error related to market data processing."""
    def __init__(self, message: str, *args, **kwargs):
        kwargs.setdefault("error_category", ErrorCategory.MARKET_DATA)
        kwargs.setdefault("component", "market_data")
        super().__init__(message, *args, **kwargs)


class SystemError(AppError):
    """Error related to system infrastructure."""
    def __init__(self, message: str, *args, **kwargs):
        kwargs.setdefault("error_category", ErrorCategory.SYSTEM)
        kwargs.setdefault("severity", ErrorSeverity.HIGH)
        super().__init__(message, *args, **kwargs)


class NotFoundError(AppError):
    """Error for resources that cannot be found."""
    def __init__(
        self,
        message: str,
        *args,
        resource_type: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get("context", {})
        if resource_type:
            context["resource_type"] = resource_type
        kwargs["context"] = context
        kwargs.setdefault("error_category", ErrorCategory.BUSINESS_LOGIC)
        kwargs.setdefault("severity", ErrorSeverity.LOW)
        kwargs.setdefault("status_code", status.HTTP_404_NOT_FOUND)
        super().__init__(message, *args, **kwargs)


# Retry strategy options
class RetryStrategy(str, Enum):
    """Available retry strategies for the retry decorator."""
    FIXED = "fixed"  # Fixed delay between retries
    EXPONENTIAL = "exponential"  # Exponential backoff
    LINEAR = "linear"  # Linear backoff
    RANDOM = "random"  # Random delay within a range
    FIBONACCI = "fibonacci"  # Fibonacci sequence backoff


def retry(
    max_retries: int = 3,
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
    base_delay: float = 0.1,
    max_delay: float = 10.0,
    exceptions: Union[Type[Exception], List[Type[Exception]]] = Exception,
    logger_name: Optional[str] = None,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
    retry_condition: Optional[Callable[[Exception], bool]] = None,
) -> Callable[[F], F]:
    """
    Decorator for retrying a function if it raises specified exceptions.
    
    Args:
        max_retries: Maximum number of retry attempts
        retry_strategy: Strategy for calculating delay between retries
        base_delay: Base delay in seconds for retry strategies
        max_delay: Maximum delay in seconds between retries
        exceptions: Exception type(s) to catch and retry
        logger_name: Name of logger to use (if None, uses module default)
        on_retry: Optional callback function called before each retry
        retry_condition: Optional function to determine if retry should occur
        
    Returns:
        Decorated function
        
    Example:
        ```
        @retry(max_retries=3, exceptions=[DatabaseError, ConnectionError])
        def fetch_data_from_database():
            # Function implementation
        ```
    """
    retry_logger = logging.getLogger(logger_name or __name__)
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            retries = 0
            
            while True:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    # Check if we've reached max retries
                    if retries >= max_retries:
                        retry_logger.error(
                            f"Maximum retries ({max_retries}) reached for {func.__name__}",
                            exc_info=True
                        )
                        raise
                    
                    # Check if this exception should be retried
                    if retry_condition and not retry_condition(e):
                        retry_logger.warning(
                            f"Retry condition not met for {func.__name__}, not retrying",
                            exc_info=True
                        )
                        raise
                    
                    # For our custom AppError, check if retry is possible
                    if isinstance(e, AppError) and not e.detail.retry_possible:
                        retry_logger.warning(
                            f"Retry not possible for {func.__name__}: {e.detail.message}",
                            exc_info=True
                        )
                        raise
                    
                    # Calculate delay based on strategy
                    delay = calculate_retry_delay(
                        retries + 1,
                        retry_strategy,
                        base_delay,
                        max_delay
                    )
                    
                    # Increment retry counter
                    retries += 1
                    
                    # Log retry attempt
                    retry_logger.warning(
                        f"Retry {retries}/{max_retries} for {func.__name__} after {delay:.2f}s: {str(e)}"
                    )
                    
                    # Call on_retry callback if provided
                    if on_retry:
                        try:
                            on_retry(e, retries)
                        except Exception as callback_err:
                            retry_logger.error(
                                f"Error in on_retry callback: {str(callback_err)}",
                                exc_info=True
                            )
                    
                    # Wait before retrying
                    time.sleep(delay)
        
        return cast(F, wrapper)
    
    return decorator


def calculate_retry_delay(
    retry_count: int,
    strategy: RetryStrategy,
    base_delay: float,
    max_delay: float
) -> float:
    """
    Calculate the delay between retries based on the specified strategy.
    
    Args:
        retry_count: Current retry attempt (starting at 1)
        strategy: The retry strategy to use
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        
    Returns:
        Delay in seconds before the next retry
    """
    delay = 0.0
    
    if strategy == RetryStrategy.FIXED:
        delay = base_delay
    elif strategy == RetryStrategy.EXPONENTIAL:
        delay = base_delay * (2 ** (retry_count - 1))
    elif strategy == RetryStrategy.LINEAR:
        delay = base_delay * retry_count
    elif strategy == RetryStrategy.RANDOM:
        delay = base_delay + (random.random() * base_delay * retry_count)
    elif strategy == RetryStrategy.FIBONACCI:
        # Use a simplified approach for Fibonacci
        fib = [1, 1]
        for i in range(2, retry_count + 1):
            fib.append(fib[i - 1] + fib[i - 2])
        delay = base_delay * fib[retry_count]
    
    # Add jitter to avoid thundering herd problem
    jitter = random.uniform(0.8, 1.2)
    delay *= jitter
    
    # Ensure delay doesn't exceed max_delay
    return min(delay, max_delay)


class Fallback:
    """
    Utility for providing fallback values when operations fail.
    
    Example:
        ```
        # Using as a context manager
        with Fallback(default_value=[]) as fallback:
            result = fallback.execute(fetch_data_from_api)
            if fallback.has_error:
                # Log or handle the stored error
                logger.error(f"API call failed: {fallback.error}")
        ```
    """
    def __init__(self, default_value: Any = None):
        self.default_value = default_value
        self.error = None
        self.has_error = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.error = exc_val
            self.has_error = True
            # Suppress the exception
            return True
        return False
    
    def execute(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Execute a function with fallback handling.
        
        Args:
            func: Function to execute
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Function result or default value if an exception occurs
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.error = e
            self.has_error = True
            return self.default_value


# Convenience function to format and standardize error messages
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
        ```
        msg = format_error_message("Failed to find {resource} with ID {id}", 
                                   resource="user", id="123")
        # Result: "Failed to find user with ID 123"
        ```
    """
    try:
        return template.format(**kwargs)
    except KeyError as e:
        # If there's a missing key, return a still-useful message
        logger.warning(f"Missing key {e} in error message template: {template}")
        return template + f" (Missing data: {e})"
    except Exception as e:
        logger.warning(f"Error formatting message template: {e}")
        return template


# Global error tracking dictionary for circuit breakers and monitoring
error_stats = {
    "counters": {},  # Error counters by category
    "last_errors": {},  # Last error by category
    "error_timestamps": {},  # Timestamps of errors by category
}


def track_error(error: Union[AppError, Exception], category: Optional[ErrorCategory] = None) -> None:
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
    if error_cat not in error_stats["counters"]:
        error_stats["counters"][error_cat] = 0
        error_stats["last_errors"][error_cat] = None
        error_stats["error_timestamps"][error_cat] = []
    
    # Update error statistics
    error_stats["counters"][error_cat] += 1
    error_stats["last_errors"][error_cat] = error
    
    # Keep last 100 timestamps for rate calculation
    timestamps = error_stats["error_timestamps"][error_cat]
    timestamps.append(datetime.utcnow())
    error_stats["error_timestamps"][error_cat] = timestamps[-100:]


def get_error_rate(category: ErrorCategory, seconds: int = 60) -> float:
    """
    Calculate the rate of errors in a specific category over a time period.
    
    Args:
        category: Error category to check
        seconds: Time window in seconds
        
    Returns:
        Error rate (errors per second)
    """
    if category not in error_stats["error_timestamps"]:
        return 0.0
    
    # Get timestamps in the specified window
    now = datetime.utcnow()
    window_start = now.timestamp() - seconds
    recent_timestamps = [
        ts for ts in error_stats["error_timestamps"][category]
        if ts.timestamp() > window_start
    ]
    
    # Calculate error rate
    return len(recent_timestamps) / seconds


def clear_error_stats() -> None:
    """Reset error statistics (mainly for testing)."""
    error_stats["counters"] = {}
    error_stats["last_errors"] = {}
    error_stats["error_timestamps"] = {}


def convert_exception(e: Exception) -> AppError:
    """
    Convert a standard Exception to an appropriate AppError.
    
    Args:
        e: Exception to convert
        
    Returns:
        Converted AppError
    """
    # Map standard exceptions to app-specific errors
    if isinstance(e, ValueError):
        return ValidationError(str(e), context={"original_error": str(e)})
    elif isinstance(e, KeyError):
        return NotFoundError(f"Key not found: {str(e)}", context={"original_error": str(e)})
    elif isinstance(e, TimeoutError):
        return ExternalServiceError(
            f"Operation timed out: {str(e)}",
            context={"original_error": str(e)},
            severity=ErrorSeverity.HIGH
        )
    elif isinstance(e, ConnectionError):
        return ExternalServiceError(
            f"Connection error: {str(e)}",
            context={"original_error": str(e)},
            severity=ErrorSeverity.HIGH
        )
    elif isinstance(e, PermissionError):
        return AuthorizationError(
            f"Permission denied: {str(e)}",
            context={"original_error": str(e)}
        )
    
    # Default case - convert to generic AppError
    return AppError(
        f"Unexpected error: {str(e)}",
        error_category=ErrorCategory.UNKNOWN,
        context={"original_error": str(e), "error_type": type(e).__name__}
    )