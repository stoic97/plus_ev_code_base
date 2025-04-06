"""
Circuit breaker implementation for preventing cascading failures.

This module provides a circuit breaker pattern implementation that:
- Tracks failures of operations
- Opens the circuit after a threshold is reached
- Allows periodic testing in half-open state
- Closes the circuit when operations succeed again

The circuit breaker pattern prevents system overload by failing fast
when underlying services are unhealthy.
"""

import functools
import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar, cast

from app.core.error_handling import AppError, ErrorCategory, ErrorSeverity, track_error

# Set up logging
logger = logging.getLogger(__name__)

# Type variables for typing
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


class CircuitState(str, Enum):
    """Possible states of a circuit breaker."""
    CLOSED = "closed"  # Normal operation, requests go through
    OPEN = "open"      # Failure threshold reached, requests fail fast
    HALF_OPEN = "half_open"  # Testing if service is healthy again


class CircuitBreakerError(AppError):
    """Error raised when a circuit is open."""
    def __init__(self, circuit_name: str, *args, **kwargs):
        kwargs.setdefault("error_category", ErrorCategory.SYSTEM)
        kwargs.setdefault("severity", ErrorSeverity.HIGH)
        kwargs.setdefault("retry_possible", False)
        kwargs.setdefault("context", {}).update({"circuit_name": circuit_name})
        super().__init__(
            f"Circuit '{circuit_name}' is open, requests are failing fast",
            *args,
            **kwargs
        )


class CircuitBreaker:
    """
    Implementation of the circuit breaker pattern.
    
    Tracks failures and prevents operation execution when the circuit is open.
    """
    # Class-level dictionary to track all circuit breakers
    _circuits: Dict[str, "CircuitBreaker"] = {}
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        half_open_timeout: float = 30.0,
        excluded_exceptions: Optional[List[type]] = None,
    ):
        """
        Initialize a new circuit breaker.
        
        Args:
            name: Unique name for this circuit breaker
            failure_threshold: Number of failures before opening the circuit
            reset_timeout: Time in seconds before trying half-open state
            half_open_timeout: Time in seconds between half-open tests
            excluded_exceptions: List of exception types that won't count as failures
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_timeout = half_open_timeout
        self.excluded_exceptions = excluded_exceptions or []
        
        # Current state
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.last_test_time = None
        
        # Register this circuit
        CircuitBreaker._circuits[name] = self
        
        logger.info(f"Circuit breaker '{name}' initialized with threshold {failure_threshold}")
    
    @classmethod
    def get_circuit(cls, name: str) -> "CircuitBreaker":
        """
        Get a circuit breaker by name, or create a new one if it doesn't exist.
        
        Args:
            name: Circuit breaker name
            
        Returns:
            CircuitBreaker instance
        """
        if name not in cls._circuits:
            return CircuitBreaker(name)
        return cls._circuits[name]
    
    @classmethod
    def get_all_circuits(cls) -> Dict[str, "CircuitBreaker"]:
        """Get all registered circuit breakers."""
        return cls._circuits.copy()
    
    @classmethod
    def reset_all(cls) -> None:
        """Reset all circuit breakers to closed state (mainly for testing)."""
        for circuit in cls._circuits.values():
            circuit.reset()
    
    def reset(self) -> None:
        """Reset this circuit breaker to closed state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.last_test_time = None
        logger.info(f"Circuit breaker '{self.name}' has been reset")
    
    def record_failure(self) -> None:
        """Record a failure and potentially open the circuit."""
        now = datetime.utcnow()
        self.failure_count += 1
        self.last_failure_time = now
        
        if self.state == CircuitState.CLOSED and self.failure_count >= self.failure_threshold:
            logger.warning(
                f"Circuit breaker '{self.name}' is now OPEN after {self.failure_count} failures"
            )
            self.state = CircuitState.OPEN
        
        if self.state == CircuitState.HALF_OPEN:
            logger.warning(f"Test failed for circuit breaker '{self.name}', returning to OPEN state")
            self.state = CircuitState.OPEN
    
    def record_success(self) -> None:
        """Record a success and potentially close the circuit."""
        if self.state == CircuitState.HALF_OPEN:
            logger.info(f"Circuit breaker '{self.name}' is now CLOSED after successful test")
            self.state = CircuitState.CLOSED
            self.failure_count = 0
    
    def allow_request(self) -> bool:
        """
        Check if a request should be allowed through the circuit breaker.
        
        Returns:
            True if the request should be allowed, False otherwise
        """
        now = datetime.utcnow()
        
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            # Check if enough time has passed to try half-open
            if (self.last_failure_time is not None and 
                    now - self.last_failure_time > timedelta(seconds=self.reset_timeout)):
                logger.info(f"Circuit breaker '{self.name}' entering HALF-OPEN state for testing")
                self.state = CircuitState.HALF_OPEN
                self.last_test_time = now
                return True
            return False
        
        if self.state == CircuitState.HALF_OPEN:
            # Only allow requests periodically in half-open state
            if (self.last_test_time is not None and 
                    now - self.last_test_time > timedelta(seconds=self.half_open_timeout)):
                logger.info(f"Allowing test request for circuit breaker '{self.name}'")
                self.last_test_time = now
                return True
            return False
        
        # Default case (should not reach here)
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of this circuit breaker."""
        return {
            "name": self.name,
            "state": self.state,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self.last_failure_time,
            "last_test_time": self.last_test_time,
        }
    
    def __call__(self, func: F) -> F:
        """
        Decorator for protecting a function with this circuit breaker.
        
        Args:
            func: Function to protect
            
        Returns:
            Decorated function
            
        Example:
            ```
            @circuit_breaker("database_operations")
            def fetch_data():
                # implementation
            ```
        """
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not self.allow_request():
                raise CircuitBreakerError(self.name)
            
            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                # Don't count excluded exceptions as failures
                if not any(isinstance(e, exc_type) for exc_type in self.excluded_exceptions):
                    self.record_failure()
                    # Track error for monitoring
                    track_error(e)
                raise
        
        return cast(F, wrapper)


def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    reset_timeout: float = 60.0,
    half_open_timeout: float = 30.0,
    excluded_exceptions: Optional[List[type]] = None,
) -> Callable[[F], F]:
    """
    Decorator factory for circuit breaker pattern.
    
    Args:
        name: Circuit breaker name
        failure_threshold: Number of failures before opening circuit
        reset_timeout: Time in seconds before allowing half-open test
        half_open_timeout: Time in seconds between half-open tests
        excluded_exceptions: Exception types that won't count as failures
        
    Returns:
        Decorator function
        
    Example:
        ```
        @circuit_breaker("external_api_calls", failure_threshold=3)
        def call_external_api():
            # implementation
        ```
    """
    cb = CircuitBreaker.get_circuit(name)
    
    # Update settings if different from defaults
    if cb.failure_threshold != failure_threshold:
        cb.failure_threshold = failure_threshold
    if cb.reset_timeout != reset_timeout:
        cb.reset_timeout = reset_timeout
    if cb.half_open_timeout != half_open_timeout:
        cb.half_open_timeout = half_open_timeout
    if excluded_exceptions is not None:
        cb.excluded_exceptions = excluded_exceptions
    
    return cb