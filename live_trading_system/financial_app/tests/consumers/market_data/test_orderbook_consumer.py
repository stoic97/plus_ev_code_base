"""
Unit tests for the OrderBookConsumer class.

This module contains tests for the Kinesis-based orderbook consumer, including:
- Normal message processing
- Edge cases and boundary conditions
- Error handling and invalid inputs
- Metrics recording
- Callback functionality
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch
from typing import Dict, Any
import json

from app.consumers.market_data.orderbook_consumer import OrderBookConsumer
from app.consumers.config.settings import KinesisSettings
from app.consumers.base.error import ValidationError, ProcessingError

# Test data fixtures for different scenarios
@pytest.fixture
def valid_orderbook_message() -> Dict[str, Any]:
    """A valid orderbook message with typical market data."""
    return {
        'symbol': 'AAPL',
        'timestamp': '2024-03-20T10:00:00',
        'bids': [
            [150.0, 100],  # [price, volume]
            [149.9, 200]
        ],
        'asks': [
            [150.1, 150],  # [price, volume]
            [150.2, 300]
        ]
    }

@pytest.fixture
def edge_case_orderbook_message() -> Dict[str, Any]:
    """An orderbook message with edge case values."""
    return {
        'symbol': 'AAPL',
        'timestamp': '2024-03-20T10:00:00',
        'bids': [
            [0.0001, 1],  # Minimum price and volume
            [999999.9999, 999999999]  # Maximum price and volume
        ],
        'asks': [
            [0.0002, 1],
            [1000000.0, 999999999]
        ]
    }

@pytest.fixture
def invalid_orderbook_message() -> Dict[str, Any]:
    """An invalid orderbook message with overlapping bid/ask prices."""
    return {
        'symbol': 'AAPL',
        'timestamp': '2024-03-20T10:00:00',
        'bids': [
            [150.1, 100],  # Invalid: bid price >= ask price
            [149.9, 200]
        ],
        'asks': [
            [150.1, 150],
            [150.2, 300]
        ]
    }

@pytest.fixture
def mock_kinesis_client():
    """Mock Kinesis client for testing."""
    with patch('boto3.client') as mock_client:
        yield mock_client.return_value

@pytest.fixture
def mock_metrics_registry():
    """Mock metrics registry for testing."""
    registry = Mock()
    registry.counter = Mock()
    registry.histogram = Mock()
    registry.register_consumer = Mock(return_value=Mock())
    registry._counters = {}
    registry._histograms = {}
    return registry

@pytest.fixture
def orderbook_consumer(mock_kinesis_client, mock_metrics_registry):
    """Create an OrderBookConsumer instance for testing."""
    settings = KinesisSettings()
    with patch('app.consumers.base.metrics.get_metrics_registry', return_value=mock_metrics_registry), \
         patch('app.consumers.utils.serialization.deserialize_json') as mock_deserialize:
        consumer = OrderBookConsumer(
            stream_name="test-orderbook-stream",
            settings=settings,
            region_name="us-east-1",
            batch_size=10,
            batch_timeout_ms=1000
        )
        consumer._deserialize_message = mock_deserialize
        consumer.metrics_registry = mock_metrics_registry  # Explicitly set the metrics registry
        return consumer

class TestOrderBookConsumer:
    """Test suite for OrderBookConsumer class."""
    
    def test_initialization(self, orderbook_consumer):
        """Test proper initialization of the consumer."""
        assert orderbook_consumer.stream_name == "test-orderbook-stream"
        assert orderbook_consumer.batch_size == 10
        assert orderbook_consumer.batch_timeout_ms == 1000
        assert orderbook_consumer.on_orderbook is None

    class TestNormalOperation:
        """Tests for normal operation scenarios."""
        
        def test_process_valid_message(self, orderbook_consumer, valid_orderbook_message):
            """Test processing of a valid orderbook message."""
            raw_record = {'Data': b'{}', 'SequenceNumber': '123'}
            orderbook_consumer.process_message(valid_orderbook_message, raw_record)
            # No exception should be raised
        
        def test_callback_execution(self, orderbook_consumer, valid_orderbook_message):
            """Test that callback is executed with valid message."""
            callback_mock = Mock()
            orderbook_consumer.on_orderbook = callback_mock
            
            raw_record = {'Data': b'{}', 'SequenceNumber': '123'}
            orderbook_consumer.process_message(valid_orderbook_message, raw_record)
            
            callback_mock.assert_called_once_with(valid_orderbook_message)
        
        def test_timestamp_conversion(self, orderbook_consumer, valid_orderbook_message):
            """Test proper conversion of timestamp string to datetime."""
            raw_record = {'Data': b'{}', 'SequenceNumber': '123'}
            orderbook_consumer.process_message(valid_orderbook_message, raw_record)
            
            assert isinstance(valid_orderbook_message['timestamp'], datetime)
            assert valid_orderbook_message['timestamp'].isoformat() == '2024-03-20T10:00:00'

    class TestEdgeCases:
        """Tests for edge cases and boundary conditions."""
        
        def test_edge_case_values(self, orderbook_consumer, edge_case_orderbook_message):
            """Test processing of orderbook with extreme price and volume values."""
            raw_record = {'Data': b'{}', 'SequenceNumber': '123'}
            orderbook_consumer.process_message(edge_case_orderbook_message, raw_record)
            # Should process without errors
        
        def test_empty_orderbook(self, orderbook_consumer):
            """Test processing of orderbook with no orders."""
            empty_message = {
                'symbol': 'AAPL',
                'timestamp': '2024-03-20T10:00:00',
                'bids': [],
                'asks': []
            }
            raw_record = {'Data': b'{}', 'SequenceNumber': '123'}
            orderbook_consumer.process_message(empty_message, raw_record)
            # Should process without errors
        
        def test_single_level_orderbook(self, orderbook_consumer):
            """Test processing of orderbook with only one price level."""
            single_level_message = {
                'symbol': 'AAPL',
                'timestamp': '2024-03-20T10:00:00',
                'bids': [[150.0, 100]],
                'asks': [[150.1, 150]]
            }
            raw_record = {'Data': b'{}', 'SequenceNumber': '123'}
            orderbook_consumer.process_message(single_level_message, raw_record)
            # Should process without errors

    class TestInvalidInputs:
        """Tests for invalid inputs and error handling."""
        
        def test_invalid_bid_ask_prices(self, orderbook_consumer, invalid_orderbook_message):
            """Test rejection of orderbook with invalid bid/ask price relationship."""
            raw_record = {'Data': b'{}', 'SequenceNumber': '123'}
            with pytest.raises(ProcessingError) as exc_info:
                orderbook_consumer.process_message(invalid_orderbook_message, raw_record)
            assert "Crossed order book" in str(exc_info.value)
        
        def test_missing_required_fields(self, orderbook_consumer):
            """Test rejection of message with missing required fields."""
            raw_record = {'Data': b'{}', 'SequenceNumber': '123'}
            invalid_message = {'symbol': 'AAPL'}  # Missing timestamp, bids, asks
            with pytest.raises(ProcessingError) as exc_info:
                orderbook_consumer.process_message(invalid_message, raw_record)
            assert "Missing required fields" in str(exc_info.value)
        
        def test_invalid_timestamp_format(self, orderbook_consumer):
            """Test rejection of message with invalid timestamp format."""
            raw_record = {'Data': b'{}', 'SequenceNumber': '123'}
            invalid_message = {
                'symbol': 'AAPL',
                'timestamp': 'invalid-timestamp',
                'bids': [[150.0, 100]],
                'asks': [[150.1, 150]]
            }
            with pytest.raises(ProcessingError) as exc_info:
                orderbook_consumer.process_message(invalid_message, raw_record)
            assert "invalid datetime format" in str(exc_info.value)
        
        def test_negative_prices(self, orderbook_consumer):
            """Test rejection of orderbook with negative prices."""
            raw_record = {'Data': b'{}', 'SequenceNumber': '123'}
            invalid_message = {
                'symbol': 'AAPL',
                'timestamp': '2024-03-20T10:00:00',
                'bids': [[-150.0, 100]],
                'asks': [[150.1, 150]]
            }
            with pytest.raises(ProcessingError) as exc_info:
                orderbook_consumer.process_message(invalid_message, raw_record)
            assert "is below minimum value" in str(exc_info.value)
        
        def test_negative_volumes(self, orderbook_consumer):
            """Test rejection of orderbook with negative volumes."""
            raw_record = {'Data': b'{}', 'SequenceNumber': '123'}
            invalid_message = {
                'symbol': 'AAPL',
                'timestamp': '2024-03-20T10:00:00',
                'bids': [[150.0, -100]],
                'asks': [[150.1, 150]]
            }
            with pytest.raises(ProcessingError) as exc_info:
                orderbook_consumer.process_message(invalid_message, raw_record)
            assert "is below minimum value" in str(exc_info.value)
        
        def test_invalid_price_volume_format(self, orderbook_consumer):
            """Test rejection of orderbook with invalid price/volume format."""
            raw_record = {'Data': b'{}', 'SequenceNumber': '123'}
            invalid_message = {
                'symbol': 'AAPL',
                'timestamp': '2024-03-20T10:00:00',
                'bids': [[150.0, 100, 'extra']],  # Wrong format: extra element
                'asks': [[150.1, 150]]
            }
            with pytest.raises(ProcessingError) as exc_info:
                orderbook_consumer.process_message(invalid_message, raw_record)
            assert "Invalid bid at index 0: must be exactly [price, volume] list" in str(exc_info.value)

    class TestMetrics:
        """Tests for metrics recording functionality."""
        
        def test_metrics_recording_success(self, orderbook_consumer, mock_metrics_registry, valid_orderbook_message):
            """Test metrics recording for successful message processing."""
            raw_record = {'Data': b'{}', 'SequenceNumber': '123'}
            orderbook_consumer._deserialize_message.return_value = valid_orderbook_message
            orderbook_consumer._process_record(raw_record)
            
            # Verify metrics were recorded
            mock_metrics_registry.counter.assert_called_with(
                'kinesis_messages_processed_total',
                labels={'stream': orderbook_consumer.stream_name, 'status': 'success'}
            )
            mock_metrics_registry.histogram.assert_called()
        
        def test_metrics_recording_failure(self, orderbook_consumer, mock_metrics_registry):
            """Test metrics recording for failed message processing."""
            raw_record = {
                'Data': json.dumps({
                    'symbol': 'AAPL',
                    'timestamp': '2024-03-20T10:00:00',
                    'bids': [[150.0, 100, 'extra']],  # Invalid bid format
                    'asks': [[150.1, 150]]
                }).encode('utf-8'),
                'SequenceNumber': '123'
            }
            
            # Mock deserialization to return invalid message
            orderbook_consumer._deserialize_message.return_value = {
                'symbol': 'AAPL',
                'timestamp': '2024-03-20T10:00:00',
                'bids': [[150.0, 100, 'extra']],  # Invalid bid format
                'asks': [[150.1, 150]]
            }
            
            with pytest.raises(ProcessingError):
                orderbook_consumer._process_record(raw_record)
            
            # Verify failure metrics were recorded
            mock_metrics_registry.counter.assert_called_with(
                'kinesis_messages_processed_total',
                labels={'stream': orderbook_consumer.stream_name, 'status': 'error'}
            ) 