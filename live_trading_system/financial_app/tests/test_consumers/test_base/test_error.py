"""
Unit tests for the error.py module.

This test file covers all critical functionality of the error classification
for Kafka consumers, including inheritance, properties, and error handling.
"""

import unittest
from unittest.mock import MagicMock
from typing import Dict, Any, Type

# Update import path to match the project structure
from app.consumers.base.error import (
    ConsumerError,
    ConnectionError,
    DeserializationError,
    ProcessingError,
    ValidationError,
    CommitError,
    ConfigurationError,
    is_retriable_error
)


class TestConsumerError(unittest.TestCase):
    """Tests for the base ConsumerError class."""
    
    def test_initialization(self):
        """Test basic initialization of ConsumerError."""
        error = ConsumerError("Test error message")
        
        self.assertEqual(error.message, "Test error message")
        self.assertTrue(error.retry_possible)
        self.assertEqual(error.max_retries, 3)
        self.assertEqual(error.context, {})
    
    def test_initialization_with_params(self):
        """Test initialization with custom parameters."""
        context = {"topic": "test-topic", "partition": 0}
        error = ConsumerError(
            message="Custom error",
            retry_possible=False,
            max_retries=5,
            context=context
        )
        
        self.assertEqual(error.message, "Custom error")
        self.assertFalse(error.retry_possible)
        self.assertEqual(error.max_retries, 5)
        self.assertEqual(error.context, context)
    
    def test_inheritance_from_exception(self):
        """Test that ConsumerError inherits from Exception."""
        error = ConsumerError("Test error")
        self.assertIsInstance(error, Exception)


class TestSpecificErrorClasses(unittest.TestCase):
    """Tests for the specific error classes derived from ConsumerError."""
    
    def _test_error_class(
        self, 
        error_class: Type[ConsumerError], 
        expected_retry_possible: bool,
        expected_max_retries: int
    ):
        """Helper method to test specific error classes."""
        error = error_class("Test error message")
        
        self.assertEqual(error.message, "Test error message")
        self.assertEqual(error.retry_possible, expected_retry_possible)
        self.assertEqual(error.max_retries, expected_max_retries)
        self.assertIsInstance(error, ConsumerError)
    
    def test_connection_error(self):
        """Test ConnectionError class."""
        self._test_error_class(ConnectionError, True, 5)
    
    def test_deserialization_error(self):
        """Test DeserializationError class."""
        self._test_error_class(DeserializationError, False, 0)
    
    def test_processing_error(self):
        """Test ProcessingError class."""
        self._test_error_class(ProcessingError, True, 3)
    
    def test_validation_error(self):
        """Test ValidationError class."""
        self._test_error_class(ValidationError, False, 0)
    
    def test_commit_error(self):
        """Test CommitError class."""
        self._test_error_class(CommitError, True, 5)
    
    def test_configuration_error(self):
        """Test ConfigurationError class."""
        self._test_error_class(ConfigurationError, False, 0)
    
    def test_custom_parameters(self):
        """Test that derived classes can override default parameters."""
        # Override non-default parameters in a class that defaults to retriable
        conn_error = ConnectionError(
            message="Custom connection error",
            retry_possible=False,
            max_retries=2,
            context={"broker": "localhost:9092"}
        )
        
        self.assertEqual(conn_error.message, "Custom connection error")
        self.assertFalse(conn_error.retry_possible)
        self.assertEqual(conn_error.max_retries, 2)
        self.assertEqual(conn_error.context, {"broker": "localhost:9092"})
        
        # Override non-default parameters in a class that defaults to non-retriable
        validation_error = ValidationError(
            message="Custom validation error",
            retry_possible=True,  # Override default False
            context={"field": "timestamp"}
        )
        
        self.assertEqual(validation_error.message, "Custom validation error")
        self.assertTrue(validation_error.retry_possible)
        self.assertEqual(validation_error.max_retries, 0)  # Should retain default
        self.assertEqual(validation_error.context, {"field": "timestamp"})


class TestIsRetriableError(unittest.TestCase):
    """Tests for the is_retriable_error function."""
    
    def test_consumer_error_retriable(self):
        """Test retriable ConsumerError detection."""
        error = ConsumerError("Retriable error", retry_possible=True)
        self.assertTrue(is_retriable_error(error))
    
    def test_consumer_error_non_retriable(self):
        """Test non-retriable ConsumerError detection."""
        error = ConsumerError("Non-retriable error", retry_possible=False)
        self.assertFalse(is_retriable_error(error))
    
    def test_derived_error_classes(self):
        """Test retriable detection for derived error classes."""
        # Errors that default to retriable
        self.assertTrue(is_retriable_error(ConnectionError("Connection error")))
        self.assertTrue(is_retriable_error(ProcessingError("Processing error")))
        self.assertTrue(is_retriable_error(CommitError("Commit error")))
        
        # Errors that default to non-retriable
        self.assertFalse(is_retriable_error(DeserializationError("Deserialization error")))
        self.assertFalse(is_retriable_error(ValidationError("Validation error")))
        self.assertFalse(is_retriable_error(ConfigurationError("Configuration error")))
    
    def test_standard_python_errors(self):
        """Test retriable detection for standard Python errors."""
        # Non-retriable standard errors
        self.assertFalse(is_retriable_error(ValueError("Invalid value")))
        self.assertFalse(is_retriable_error(TypeError("Invalid type")))
        self.assertFalse(is_retriable_error(KeyError("Missing key")))
        self.assertFalse(is_retriable_error(AttributeError("Missing attribute")))
        
        # Other standard errors should be retriable
        self.assertTrue(is_retriable_error(RuntimeError("Runtime error")))
        self.assertTrue(is_retriable_error(IOError("IO error")))
        self.assertTrue(is_retriable_error(OSError("OS error")))


class TestErrorContext(unittest.TestCase):
    """Tests for error context functionality."""
    
    def test_empty_context(self):
        """Test initialization with empty context."""
        error = ConsumerError("Error with empty context")
        self.assertEqual(error.context, {})
    
    def test_context_reference(self):
        """Test that the error maintains a reference to the provided context."""
        context = {"key": "value"}
        error = ConsumerError("Error with context", context=context)
        
        # Error should have the initial context value
        self.assertEqual(error.context, {"key": "value"})
        
        # Since the implementation uses the original reference, 
        # modifying original context affects the error's context
        context["key"] = "modified"
        self.assertEqual(error.context, {"key": "modified"})
    
    def test_complex_context(self):
        """Test with a more complex context object."""
        context = {
            "topic": "test-topic",
            "partition": 0,
            "offset": 100,
            "message": {
                "key": "test-key",
                "value": "test-value",
                "headers": [("header1", b"value1")]
            }
        }
        
        error = ProcessingError("Processing failed", context=context)
        self.assertEqual(error.context, context)


class TestErrorHierarchy(unittest.TestCase):
    """Tests for error class hierarchy and inheritance."""
    
    def test_inheritance_chain(self):
        """Test that all errors follow expected inheritance chain."""
        # All error classes should inherit from ConsumerError
        error_classes = [
            ConnectionError,
            DeserializationError,
            ProcessingError,
            ValidationError,
            CommitError,
            ConfigurationError
        ]
        
        for error_class in error_classes:
            error = error_class("Test error")
            self.assertIsInstance(error, ConsumerError)
            self.assertIsInstance(error, Exception)
    
    def test_error_str_representation(self):
        """Test string representation of errors."""
        error = ConsumerError("Test error message")
        self.assertEqual(str(error), "Test error message")
        
        # Check that derived classes maintain the string representation
        conn_error = ConnectionError("Connection failed")
        self.assertEqual(str(conn_error), "Connection failed")


if __name__ == '__main__':
    unittest.main()