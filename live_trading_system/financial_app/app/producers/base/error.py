"""
Custom exceptions for producer operations.
"""

class SerializationError(Exception):
    """Raised when message serialization fails."""
    pass

class PublishingError(Exception):
    """Raised when message publishing fails."""
    pass 