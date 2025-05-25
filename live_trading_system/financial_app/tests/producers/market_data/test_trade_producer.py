"""
Unit tests for the TradeProducer class.

This module contains comprehensive tests for the TradeProducer class,
including tests for message publishing, batching, error handling, and cleanup.
"""

import json
import time
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch, call

import pytest
from confluent_kafka import KafkaError, Message

from app.producers.market_data.trade_producer import TradeProducer
from app.producers.base.error import SerializationError, PublishingError
from app.producers.config.settings import KafkaSettings

# Test data
VALID_TRADE = {
    "symbol": "BTC-USD",
    "price": 50000.0,
    "volume": 1.5,
    "side": "buy",
    "trade_id": "trade123",
    "timestamp": 1645000000000,
    "source": "test",
    "additional_field": "value"
}

@pytest.fixture
def mock_producer():
    """Create a mock Kafka producer."""
    with patch('confluent_kafka.Producer') as mock:
        producer = MagicMock()
        mock.return_value = producer
        yield producer

@pytest.fixture
def mock_metrics():
    """Create a mock metrics registry."""
    with patch('app.producers.base.metrics.get_metrics_registry') as mock:
        metrics = MagicMock()
        mock.return_value.register_producer.return_value = metrics
        yield metrics

@pytest.fixture
def producer(mock_producer, mock_metrics):
    """Create a TradeProducer instance with mocked dependencies."""
    settings = KafkaSettings()
    settings.TRADES_TOPIC = "test-trades-topic"
    return TradeProducer(settings=settings, batch_size=2, batch_timeout_ms=100)

class TestTradeProducer:
    """Test suite for TradeProducer class."""

    def test_initialization(self, producer, mock_producer, mock_metrics):
        """Test producer initialization."""
        assert producer.topic == "test-trades-topic"
        assert producer.batch_size == 2
        assert producer.batch_timeout_ms == 100
        assert len(producer._batch) == 0
        mock_producer.assert_called_once()
        mock_metrics.assert_called_once()

    def test_publish_trade_basic(self, producer, mock_producer):
        """Test basic trade publishing."""
        producer.publish_trade(
            symbol="BTC-USD",
            price=50000.0,
            volume=1.5
        )
        
        # Verify message was added to batch
        assert len(producer._batch) == 1
        message = producer._batch[0]
        assert message["symbol"] == "BTC-USD"
        assert message["price"] == 50000.0
        assert message["volume"] == 1.5
        assert message["source"] == "producer"
        assert "timestamp" in message

    def test_publish_trade_with_optional_fields(self, producer, mock_producer):
        """Test trade publishing with all optional fields."""
        producer.publish_trade(
            symbol="BTC-USD",
            price=50000.0,
            volume=1.5,
            side="buy",
            trade_id="trade123",
            additional_field="value"
        )
        
        message = producer._batch[0]
        assert message["side"] == "buy"
        assert message["trade_id"] == "trade123"
        assert message["additional_field"] == "value"

    def test_timestamp_handling(self, producer, mock_producer):
        """Test different timestamp formats."""
        # Test with milliseconds
        producer.publish_trade(
            symbol="BTC-USD",
            price=50000.0,
            volume=1.5,
            timestamp=1645000000000
        )
        assert producer._batch[0]["timestamp"] == 1645000000000
        
        # Test with seconds
        producer._batch = []
        producer.publish_trade(
            symbol="BTC-USD",
            price=50000.0,
            volume=1.5,
            timestamp=1645000000
        )
        assert producer._batch[0]["timestamp"] == 1645000000000
        
        # Test with ISO format
        producer._batch = []
        producer.publish_trade(
            symbol="BTC-USD",
            price=50000.0,
            volume=1.5,
            timestamp="2022-02-16T12:00:00Z"
        )
        assert isinstance(producer._batch[0]["timestamp"], int)

    def test_batch_processing(self, producer, mock_producer):
        """Test batch processing functionality."""
        # Fill batch to capacity
        producer.publish_trade(
            symbol="BTC-USD",
            price=50000.0,
            volume=1.5
        )
        producer.publish_trade(
            symbol="ETH-USD",
            price=3000.0,
            volume=2.0
        )
        
        # Verify batch was published
        assert len(producer._batch) == 0
        assert mock_producer.produce.call_count == 2
        mock_producer.flush.assert_called_once()

    def test_batch_timeout(self, producer, mock_producer):
        """Test batch timeout functionality."""
        producer.publish_trade(
            symbol="BTC-USD",
            price=50000.0,
            volume=1.5
        )
        
        # Wait for timeout
        time.sleep(0.2)  # Longer than batch_timeout_ms
        
        # Verify batch was published
        assert len(producer._batch) == 0
        mock_producer.produce.assert_called_once()
        mock_producer.flush.assert_called_once()

    def test_serialization_error(self, producer, mock_producer):
        """Test handling of serialization errors."""
        with patch('app.producers.utils.serialization.serialize_json') as mock_serialize:
            mock_serialize.side_effect = Exception("Serialization failed")
            
            with pytest.raises(PublishingError) as exc_info:
                producer.publish_trade(
                    symbol="BTC-USD",
                    price=50000.0,
                    volume=1.5
                )
            
            assert "Failed to publish trade" in str(exc_info.value)

    def test_publishing_error(self, producer, mock_producer):
        """Test handling of publishing errors."""
        mock_producer.produce.side_effect = KafkaError("Publishing failed")
        
        with pytest.raises(PublishingError) as exc_info:
            producer.publish_trade(
                symbol="BTC-USD",
                price=50000.0,
                volume=1.5
            )
            producer._publish_batch()
        
        assert "Failed to publish trade batch" in str(exc_info.value)

    def test_delivery_report_success(self, producer, mock_metrics):
        """Test successful message delivery report."""
        mock_msg = MagicMock(spec=Message)
        mock_msg.topic.return_value = "test-topic"
        mock_msg.partition.return_value = 0
        
        producer._delivery_report(None, mock_msg)
        mock_metrics.record_message_failed.assert_not_called()

    def test_delivery_report_failure(self, producer, mock_metrics):
        """Test failed message delivery report."""
        mock_msg = MagicMock(spec=Message)
        producer._delivery_report(KafkaError("Delivery failed"), mock_msg)
        mock_metrics.record_message_failed.assert_called_once()

    def test_on_stop(self, producer, mock_producer):
        """Test cleanup on stop."""
        # Add some messages to batch
        producer.publish_trade(
            symbol="BTC-USD",
            price=50000.0,
            volume=1.5
        )
        
        # Call on_stop
        producer.on_stop()
        
        # Verify cleanup
        assert len(producer._batch) == 0
        mock_producer.flush.assert_called_once()

    def test_on_stop_with_error(self, producer, mock_producer):
        """Test cleanup on stop with error."""
        mock_producer.flush.side_effect = KafkaError("Flush failed")
        
        # Should not raise exception
        producer.on_stop()
        
        # Verify flush was attempted
        mock_producer.flush.assert_called_once()

    @pytest.mark.parametrize("invalid_data", [
        {
            "symbol": "BTC-USD",
            "price": -50000.0,  # Invalid price
            "volume": 1.5
        },
        {
            "symbol": "BTC-USD",
            "price": 50000.0,
            "volume": -1.5  # Invalid volume
        },
        {
            "symbol": "",  # Empty symbol
            "price": 50000.0,
            "volume": 1.5
        },
        {
            "symbol": "BTC-USD",
            "price": "invalid",  # Invalid price type
            "volume": 1.5
        },
        {
            "symbol": "BTC-USD",
            "price": 50000.0,
            "volume": "invalid"  # Invalid volume type
        }
    ])
    def test_invalid_trade_data(self, producer, invalid_data):
        """Test handling of invalid trade data."""
        with pytest.raises(PublishingError):
            producer.publish_trade(**invalid_data)

    def test_metrics_recording(self, producer, mock_producer, mock_metrics):
        """Test metrics recording during publishing."""
        # Fill batch to capacity
        producer.publish_trade(
            symbol="BTC-USD",
            price=50000.0,
            volume=1.5
        )
        producer.publish_trade(
            symbol="ETH-USD",
            price=3000.0,
            volume=2.0
        )
        
        # Verify metrics were recorded
        mock_metrics.record_messages_published.assert_called_once()
        args = mock_metrics.record_messages_published.call_args[0]
        assert args[0] == 2  # Number of messages
        assert isinstance(args[1], float)  # Processing time

    def test_message_key_encoding(self, producer, mock_producer):
        """Test that message keys are properly encoded."""
        producer.publish_trade(
            symbol="BTC-USD",
            price=50000.0,
            volume=1.5
        )
        producer._publish_batch()
        
        # Verify key encoding
        call_args = mock_producer.produce.call_args[1]
        assert call_args["key"] == b"BTC-USD" 