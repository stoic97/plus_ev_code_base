"""
Unit tests for the OHLCV consumer.

This module contains tests for the OHLCV consumer class, including message processing,
validation, and callback handling.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from financial_app.app.consumers.market_data.ohlcv_consumer import OHLCVConsumer
from app.consumers.config.settings import KinesisSettings
from app.consumers.base.error import ValidationError, ProcessingError
from app.consumers.base.metrics import get_metrics_registry, MetricsRegistry, ConsumerMetrics

# Test data
VALID_OHLCV_MESSAGE = {
    'symbol': 'AAPL',
    'timestamp': '2024-02-14T10:00:00Z',
    'open': 150.0,
    'high': 151.0,
    'low': 149.0,
    'close': 150.5,
    'volume': 1000,
    'interval': '1m'
}

INVALID_OHLCV_MESSAGE = {
    'symbol': 'AAPL',
    'timestamp': '2024-02-14T10:00:00Z',
    'open': 150.0,
    'high': 151.0,
    'low': 149.0,
    'close': 150.5,
    'interval': '1m'
    # Missing volume
}

MALFORMED_MESSAGE = {
    'invalid': 'data'
}

@pytest.fixture
def mock_kinesis_client():
    """Create a mock Kinesis client."""
    with patch('boto3.client') as mock_client:
        yield mock_client.return_value

@pytest.fixture
def mock_metrics():
    """Create a mock metrics object."""
    metrics = MagicMock(spec=ConsumerMetrics)
    metrics.record_message_processed = Mock()
    metrics.record_message_failed = Mock()
    metrics.get_metrics_summary = Mock(return_value={
        'messages_processed': 0,
        'messages_failed': 0,
        'avg_processing_time_ms': 0.0,
        'error_rate_percent': 0.0
    })
    return metrics

@pytest.fixture
def mock_metrics_registry(mock_metrics):
    """Create a mock metrics registry."""
    registry = MagicMock(spec=MetricsRegistry)
    registry.register_consumer.return_value = mock_metrics
    registry.counter = Mock()
    registry.histogram = Mock()
    registry.gauge = Mock()
    return registry

@pytest.fixture
def ohlcv_consumer(mock_kinesis_client, mock_metrics_registry):
    """Create an OHLCV consumer instance for testing."""
    settings = KinesisSettings()
    with patch('app.consumers.base.metrics.get_metrics_registry', return_value=mock_metrics_registry):
        consumer = OHLCVConsumer(
            stream_name="test-ohlcv-stream",
            settings=settings,
            region_name="us-east-1",
            batch_size=10,
            batch_timeout_ms=1000
        )
        return consumer

class TestOHLCVConsumer:
    """Test cases for the OHLCV consumer."""
    
    def test_initialization(self, ohlcv_consumer):
        """Test consumer initialization."""
        assert ohlcv_consumer.stream_name == "test-ohlcv-stream"
        assert ohlcv_consumer.batch_size == 10
        assert ohlcv_consumer.batch_timeout_ms == 1000
        assert ohlcv_consumer.on_candlestick is None
    
    def test_process_valid_message(self, ohlcv_consumer):
        """Test processing a valid OHLCV message."""
        # Create a callback to capture the processed message
        processed_message = None
        def on_ohlcv(message):
            nonlocal processed_message
            processed_message = message
        
        # Update consumer with callback
        ohlcv_consumer.on_candlestick = on_ohlcv
        
        # Process message
        raw_record = {'Data': b'{}', 'SequenceNumber': '123'}
        ohlcv_consumer.process_message(VALID_OHLCV_MESSAGE, raw_record)
        
        # Verify message was processed correctly
        assert processed_message is not None
        assert processed_message['symbol'] == 'AAPL'
        assert processed_message['open'] == 150.0
        assert processed_message['high'] == 151.0
        assert processed_message['low'] == 149.0
        assert processed_message['close'] == 150.5
        assert processed_message['volume'] == 1000
        assert processed_message['interval'] == '1m'
        assert isinstance(processed_message['timestamp'], datetime)
    
    def test_process_invalid_message(self, ohlcv_consumer):
        """Test processing an invalid OHLCV message."""
        raw_record = {'Data': b'{}', 'SequenceNumber': '123'}
        
        # Process invalid message
        with pytest.raises(ProcessingError) as exc_info:
            ohlcv_consumer.process_message(INVALID_OHLCV_MESSAGE, raw_record)
        
        # Verify error message
        assert "Missing required fields: volume" in str(exc_info.value)
    
    def test_process_message_with_invalid_timestamp(self, ohlcv_consumer):
        """Test processing a message with invalid timestamp format."""
        # Create message with invalid timestamp
        message = VALID_OHLCV_MESSAGE.copy()
        message['timestamp'] = 'invalid-timestamp'
        
        raw_record = {'Data': b'{}', 'SequenceNumber': '123'}
        
        # Process message
        with pytest.raises(ProcessingError) as exc_info:
            ohlcv_consumer.process_message(message, raw_record)
        
        # Verify error message
        assert "Field 'timestamp' has invalid datetime format" in str(exc_info.value)
    
    def test_process_message_with_invalid_numeric_values(self, ohlcv_consumer):
        """Test processing a message with invalid numeric values."""
        # Create message with invalid numeric values
        message = VALID_OHLCV_MESSAGE.copy()
        message['open'] = 'invalid'
        
        raw_record = {'Data': b'{}', 'SequenceNumber': '123'}
        
        # Process message
        with pytest.raises(ProcessingError) as exc_info:
            ohlcv_consumer.process_message(message, raw_record)
        
        # Verify error message
        assert "Field 'open' has non-numeric value" in str(exc_info.value)
    
    def test_callback_execution(self, ohlcv_consumer):
        """Test callback execution with valid message."""
        callback_mock = Mock()
        ohlcv_consumer.on_candlestick = callback_mock
        
        raw_record = {'Data': b'{}', 'SequenceNumber': '123'}
        ohlcv_consumer.process_message(VALID_OHLCV_MESSAGE, raw_record)
        
        # Verify callback was called with the message
        callback_mock.assert_called_once()
        called_message = callback_mock.call_args[0][0]
        assert isinstance(called_message['timestamp'], datetime)
        assert called_message['symbol'] == 'AAPL'
        assert called_message['open'] == 150.0
        assert called_message['interval'] == '1m'
    
    def test_timestamp_conversion(self, ohlcv_consumer):
        """Test timestamp string to datetime conversion."""
        message = VALID_OHLCV_MESSAGE.copy()
        raw_record = {'Data': b'{}', 'SequenceNumber': '123'}
        
        ohlcv_consumer.process_message(message, raw_record)
        
        assert isinstance(message['timestamp'], datetime)
        assert message['timestamp'].isoformat() == '2024-02-14T10:00:00+00:00'
    
    def test_error_handling(self, ohlcv_consumer):
        """Test error handling during message processing."""
        raw_record = {'Data': b'{}', 'SequenceNumber': '123'}
        
        # Test with malformed message
        with pytest.raises(ProcessingError) as exc_info:
            ohlcv_consumer.process_message(MALFORMED_MESSAGE, raw_record)
        assert "Missing required fields" in str(exc_info.value)
    
    def test_metrics_recording(self, ohlcv_consumer, mock_metrics_registry, mock_metrics):
        """Test metrics recording during message processing."""
        raw_record = {'Data': b'{}', 'SequenceNumber': '123'}
        
        # Process a valid message
        ohlcv_consumer.process_message(VALID_OHLCV_MESSAGE, raw_record)
        
        # Verify metrics were recorded
        mock_metrics.record_message_processed.assert_called_once()
        
        # Verify metrics summary
        metrics_summary = mock_metrics.get_metrics_summary()
        assert metrics_summary['messages_processed'] == 0  # Initial value
        assert metrics_summary['messages_failed'] == 0
        assert metrics_summary['avg_processing_time_ms'] == 0.0
        assert metrics_summary['error_rate_percent'] == 0.0
    
    def test_metrics_recording_on_error(self, ohlcv_consumer, mock_metrics_registry, mock_metrics):
        """Test metrics recording when message processing fails."""
        raw_record = {'Data': b'{}', 'SequenceNumber': '123'}
        
        # Process an invalid message
        with pytest.raises(ProcessingError):
            ohlcv_consumer.process_message(INVALID_OHLCV_MESSAGE, raw_record)
        
        # Verify metrics were recorded
        mock_metrics.record_message_failed.assert_called_once()
        
        # Verify metrics summary
        metrics_summary = mock_metrics.get_metrics_summary()
        assert metrics_summary['messages_processed'] == 0
        assert metrics_summary['messages_failed'] == 0  # Initial value
        assert metrics_summary['error_rate_percent'] == 0.0 