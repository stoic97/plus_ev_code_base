"""
Circuit breaker pattern implementation for Kafka consumers.

This module provides a circuit breaker pattern implementation for
preventing cascading failures when services or external systems are unhealthy.
"""

import logging
import threading
import time
from enum import Enum
from typing import Any, Callable, Dict, Optional, Type, TypeVar, cast

# Set up logging
logger = logging.getLogger(__name__)

# Type variable for the function result
T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation, requests are allowed through
    OPEN = "open"           # Circuit is open, requests will fail fast
    HALF_OPEN = "half_open" # Testing if the service is healthy again


class CircuitBreakerError(Exception):
    """Error raised when the circuit is open."""
    
    def __init__(self, name: str, message: Optional[str] = None):
        """
        Initialize a circuit breaker error.
        
        Args:
            name: Name of the circuit breaker
            message: Optional error message
        """
        self.name = name
        super().__init__(message or f"Circuit '{name}' is open")


class CircuitBreaker:
    """
    Circuit breaker implementation for preventing cascading failures.
    
    The circuit breaker pattern monitors for failures and prevents operations
    when a certain threshold is reached, allowing the system to recover.
    """
    
    # Class-level registry of circuit breakers
    _instances: Dict[str, 'CircuitBreaker'] = {}
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_timeout: float = 30.0,
        excluded_exceptions: Optional[list] = None,
    ):
        """
        Initialize a new circuit breaker.
        
        Args:
            name: Unique name for this circuit breaker
            failure_threshold: Number of failures before opening the circuit
            recovery_timeout: Time in seconds to wait before trying half-open state
            half_open_timeout: Time in seconds to wait between half-open tests
            excluded_exceptions: Exceptions that don't count as failures
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_timeout = half_open_timeout
        self.excluded_exceptions = excluded_exceptions or []
        
        # State tracking
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.last_test_time: Optional[float] = None
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Register this instance
        CircuitBreaker._instances[name] = self
        
        logger.info(f"Created circuit breaker '{name}' with threshold {failure_threshold}")
    
    @classmethod
    def get(cls, name: str) -> 'CircuitBreaker':
        """
        Get a circuit breaker by name, creating it if it doesn't exist.
        
        Args:
            name: Circuit breaker name
            
        Returns:
            CircuitBreaker instance
        """
        if name not in cls._instances:
            return CircuitBreaker(name)
        return cls._instances[name]
    
    @classmethod
    def get_all(cls) -> Dict[str, 'CircuitBreaker']:
        """
        Get all circuit breakers.
        
        Returns:
            Dictionary of circuit breakers by name
        """
        return cls._instances.copy()
    
    @classmethod
    def reset_all(cls) -> None:
        """Reset all circuit breakers to closed state."""
        for breaker in cls._instances.values():
            breaker.reset()
    
    def reset(self) -> None:
        """Reset this circuit breaker to closed state."""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.last_failure_time = None
            self.last_test_time = None
            logger.info(f"Reset circuit breaker '{self.name}'")
    
    def record_success(self) -> None:
        """Record a successful operation."""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                # Successful test, close the circuit
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info(f"Circuit '{self.name}' closed after successful test")
    
    def record_failure(self, exception: Optional[Exception] = None) -> None:
        """
        Record a failed operation.
        
        Args:
            exception: Exception that occurred
        """
        # Ignore excluded exceptions
        if exception and any(isinstance(exception, exc_type) for exc_type in self.excluded_exceptions):
            return
            
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitState.CLOSED and self.failure_count >= self.failure_threshold:
                # Too many failures, open the circuit
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit '{self.name}' opened after {self.failure_count} failures")
            elif self.state == CircuitState.HALF_OPEN:
                # Failed test, back to open
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit '{self.name}' reopened after failed test")
    
    def allow_request(self) -> bool:
        """
        Check if a request should be allowed through the circuit breaker.
        
        Returns:
            True if the request should be allowed, False otherwise
        """
        with self._lock:
            now = time.time()
            
            if self.state == CircuitState.CLOSED:
                # Circuit is closed, allow the request
                return True
                
            if self.state == CircuitState.OPEN:
                # Check if recovery timeout has elapsed
                if self.last_failure_time and now - self.last_failure_time > self.recovery_timeout:
                    # Try half-open state
                    self.state = CircuitState.HALF_OPEN
                    self.last_test_time = now
                    logger.info(f"Circuit '{self.name}' entering half-open state")
                    return True
                # Still open, reject the request
                return False
                
            if self.state == CircuitState.HALF_OPEN:
                # Only allow periodic test requests
                if self.last_test_time and now - self.last_test_time < self.half_open_timeout:
                    # Not time for another test yet
                    return False
                # Time for a test request
                self.last_test_time = now
                return True
                
            # Default case (shouldn't happen)
            return True
    
    def execute(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Execute a function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function
            
        Raises:
            CircuitBreakerError: If the circuit is open
            Exception: Any exception raised by the function
        """
        if not self.allow_request():
            raise CircuitBreakerError(self.name)
            
        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure(e)
            raise
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """
        Decorator for protecting a function with this circuit breaker.
        
        Args:
            func: Function to protect
            
        Returns:
            Protected function
            
        Example:
            ```
            @circuit_breaker('database_operations')
            def fetch_data():
                # implementation
            ```
        """
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return self.execute(func, *args, **kwargs)
        return cast(Callable[..., T], wrapper)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of this circuit breaker.
        
        Returns:
            Dictionary with status details
        """
        with self._lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "failure_threshold": self.failure_threshold,
                "last_failure_time": self.last_failure_time,
                "last_test_time": self.last_test_time,
                "recovery_timeout": self.recovery_timeout,
                "half_open_timeout": self.half_open_timeout,
            }


def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    half_open_timeout: float = 30.0,
    excluded_exceptions: Optional[list] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator factory for circuit breaker pattern.
    
    Args:
        name: Circuit breaker name
        failure_threshold: Number of failures before opening the circuit
        recovery_timeout: Time in seconds to wait before trying half-open state
        half_open_timeout: Time in seconds to wait between half-open tests
        excluded_exceptions: Exceptions that don't count as failures
        
    Returns:
        Decorator function
        
    Example:
        ```
        @circuit_breaker('database_operations')
        def fetch_data():
            # implementation
        ```
    """
    # Get or create circuit breaker
    breaker = CircuitBreaker.get(name)
    
    # Update settings if different from defaults
    if breaker.failure_threshold != failure_threshold:
        breaker.failure_threshold = failure_threshold
    if breaker.recovery_timeout != recovery_timeout:
        breaker.recovery_timeout = recovery_timeout
    if breaker.half_open_timeout != half_open_timeout:
        breaker.half_open_timeout = half_open_timeout
    if excluded_exceptions is not None:
        breaker.excluded_exceptions = excluded_exceptions
    
    # Return as decorator
    return breaker


# Add missing import
from functools import wraps