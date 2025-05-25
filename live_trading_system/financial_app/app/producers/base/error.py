"""
Custom exceptions for the producer system.

This module defines custom exceptions used throughout the producer system
for better error handling and more specific error messages.
"""

class ProducerError(Exception):
    """Base exception for all producer-related errors."""
    pass

class SerializationError(ProducerError):
    """Raised when message serialization fails."""
    pass

class PublishingError(ProducerError):
    """Raised when message publishing fails."""
    pass

class ValidationError(ProducerError):
    """Raised when message validation fails."""
    pass

class ConfigurationError(ProducerError):
    """Raised when producer configuration is invalid."""
    pass

class ConnectionError(ProducerError):
    """Raised when connection to Kafka fails."""
    pass

class BatchProcessingError(ProducerError):
    """Raised when batch processing fails."""
    pass 