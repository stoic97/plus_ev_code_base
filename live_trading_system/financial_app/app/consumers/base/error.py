"""
Error classification for Kafka consumers.

This module defines a hierarchy of error types for Kafka consumers,
enabling consistent handling of different error scenarios.
"""

from typing import Any, Dict, Optional


class ConsumerError(Exception):
    """Base class for all consumer errors."""
    
    def __init__(
        self, 
        message: str, 
        retry_possible: bool = True,
        max_retries: int = 3,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new consumer error.
        
        Args:
            message: Error message
            retry_possible: Whether retry is possible for this error
            max_retries: Maximum number of retries for this error
            context: Additional context for the error
        """
        self.message = message
        self.retry_possible = retry_possible
        self.max_retries = max_retries
        self.context = context or {}
        super().__init__(message)


class ConnectionError(ConsumerError):
    """Error connecting to Kafka brokers."""
    
    def __init__(
        self, 
        message: str, 
        retry_possible: bool = True,
        max_retries: int = 5,
        context: Optional[Dict[str, Any]] = None
    ):
        """Initialize a connection error."""
        super().__init__(
            message=message,
            retry_possible=retry_possible,
            max_retries=max_retries,
            context=context
        )


class DeserializationError(ConsumerError):
    """Error deserializing a message."""
    
    def __init__(
        self, 
        message: str, 
        retry_possible: bool = False,  # Generally can't retry if message format is wrong
        context: Optional[Dict[str, Any]] = None
    ):
        """Initialize a deserialization error."""
        super().__init__(
            message=message,
            retry_possible=retry_possible,
            max_retries=0,
            context=context
        )


class ProcessingError(ConsumerError):
    """Error processing a message."""
    
    def __init__(
        self, 
        message: str, 
        retry_possible: bool = True,
        max_retries: int = 3,
        context: Optional[Dict[str, Any]] = None
    ):
        """Initialize a processing error."""
        super().__init__(
            message=message,
            retry_possible=retry_possible,
            max_retries=max_retries,
            context=context
        )


class ValidationError(ConsumerError):
    """Error validating message contents."""
    
    def __init__(
        self, 
        message: str, 
        retry_possible: bool = False,  # Usually can't retry for invalid data
        context: Optional[Dict[str, Any]] = None
    ):
        """Initialize a validation error."""
        super().__init__(
            message=message,
            retry_possible=retry_possible,
            max_retries=0,
            context=context
        )


class CommitError(ConsumerError):
    """Error committing offsets."""
    
    def __init__(
        self, 
        message: str, 
        retry_possible: bool = True,
        max_retries: int = 5,
        context: Optional[Dict[str, Any]] = None
    ):
        """Initialize a commit error."""
        super().__init__(
            message=message,
            retry_possible=retry_possible,
            max_retries=max_retries,
            context=context
        )


class ConfigurationError(ConsumerError):
    """Error in consumer configuration."""
    
    def __init__(
        self, 
        message: str, 
        retry_possible: bool = False,  # Config errors rarely resolve with retry
        context: Optional[Dict[str, Any]] = None
    ):
        """Initialize a configuration error."""
        super().__init__(
            message=message,
            retry_possible=retry_possible,
            max_retries=0,
            context=context
        )


def is_retriable_error(error: Exception) -> bool:
    """
    Check if an error is retriable.
    
    Args:
        error: Exception to check
        
    Returns:
        True if the error is retriable, False otherwise
    """
    if isinstance(error, ConsumerError):
        return error.retry_possible
    
    # For non-consumer errors, determine based on error type
    # Some errors should never be retried
    non_retriable_errors = (
        ValueError,
        AttributeError,
        TypeError,
        KeyError
    )
    
    return not isinstance(error, non_retriable_errors)