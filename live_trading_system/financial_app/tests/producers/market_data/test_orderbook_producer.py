"""
Unit tests for the OrderBookProducer class.

This module contains comprehensive tests for the OrderBookProducer class,
including tests for message publishing, batching, error handling, and cleanup.
"""

import json
import time
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch, call

import pytest
from confluent_kafka import KafkaError, Message

from app.producers.market_data.orderbook_producer import OrderBookProducer
from app.producers.base.error import SerializationError, PublishingError
from app.producers.config.settings import KafkaSettings

# Test data
VALID_ORDERBOOK = {
    "symbol": "BTC-USD",
    "bids": [[50000.0, 1.5], [49900.0, 2.0]],
    "asks": [[50100.0, 1.0], [50200.0, 2.5]],
    "timestamp": 1645000000000,
    "source": "test",
    "depth": 10,
    "spread": 200.0,
    "weighted_mid_price": 50050.0,
    "imbalance": 0.2
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
    """Create an OrderBookProducer instance with mocked dependencies."""
    settings = KafkaSettings()
    settings.ORDERBOOK_TOPIC = "test-orderbook-topic"
    return OrderBookProducer(settings=settings, batch_size=2, batch_timeout_ms=100)

class TestOrderBookProducer:
    """Test suite for OrderBookProducer class."""

    def test_initialization(self, producer, mock_producer, mock_metrics):
        """Test producer initialization."""
        assert producer.topic == "test-orderbook-topic"
        assert producer.batch_size == 2
        assert producer.batch_timeout_ms == 100
        assert len(producer._batch) == 0
        mock_producer.assert_called_once()
        mock_metrics.assert_called_once()

    def test_publish_orderbook_basic(self, producer, mock_producer):
        """Test basic orderbook publishing."""
        producer.publish_orderbook(
            symbol="BTC-USD",
            bids=[[50000.0, 1.5]],
            asks=[[50100.0, 1.0]]
        )
        
        # Verify message was added to batch
        assert len(producer._batch) == 1
        message = producer._batch[0]
        assert message["symbol"] == "BTC-USD"
        assert message["bids"] == [[50000.0, 1.5]]
        assert message["asks"] == [[50100.0, 1.0]]
        assert "timestamp" in message
        assert message["source"] == "producer"

    def test_publish_orderbook_with_optional_fields(self, producer, mock_producer):
        """Test orderbook publishing with all optional fields."""
        producer.publish_orderbook(
            symbol="BTC-USD",
            bids=[[50000.0, 1.5]],
            asks=[[50100.0, 1.0]],
            depth=10,
            spread=200.0,
            weighted_mid_price=50050.0,
            imbalance=0.2
        )
        
        message = producer._batch[0]
        assert message["depth"] == 10
        assert message["spread"] == 200.0
        assert message["weighted_mid_price"] == 50050.0
        assert message["imbalance"] == 0.2

    def test_timestamp_handling(self, producer, mock_producer):
        """Test different timestamp formats."""
        # Test with milliseconds
        producer.publish_orderbook(
            symbol="BTC-USD",
            bids=[[50000.0, 1.5]],
            asks=[[50100.0, 1.0]],
            timestamp=1645000000000
        )
        assert producer._batch[0]["timestamp"] == 1645000000000
        
        # Test with seconds
        producer._batch = []
        producer.publish_orderbook(
            symbol="BTC-USD",
            bids=[[50000.0, 1.5]],
            asks=[[50100.0, 1.0]],
            timestamp=1645000000
        )
        assert producer._batch[0]["timestamp"] == 1645000000000
        
        # Test with ISO format
        producer._batch = []
        producer.publish_orderbook(
            symbol="BTC-USD",
            bids=[[50000.0, 1.5]],
            asks=[[50100.0, 1.0]],
            timestamp="2022-02-16T12:00:00Z"
        )
        assert isinstance(producer._batch[0]["timestamp"], int)

    def test_batch_processing(self, producer, mock_producer):
        """Test batch processing functionality."""
        # Fill batch to capacity
        producer.publish_orderbook(
            symbol="BTC-USD",
            bids=[[50000.0, 1.5]],
            asks=[[50100.0, 1.0]]
        )
        producer.publish_orderbook(
            symbol="ETH-USD",
            bids=[[3000.0, 2.0]],
            asks=[[3100.0, 1.5]]
        )
        
        # Verify batch was published
        assert len(producer._batch) == 0
        assert mock_producer.produce.call_count == 2
        mock_producer.flush.assert_called_once()

    def test_batch_timeout(self, producer, mock_producer):
        """Test batch timeout functionality."""
        producer.publish_orderbook(
            symbol="BTC-USD",
            bids=[[50000.0, 1.5]],
            asks=[[50100.0, 1.0]]
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
                producer.publish_orderbook(
                    symbol="BTC-USD",
                    bids=[[50000.0, 1.5]],
                    asks=[[50100.0, 1.0]]
                )
            
            assert "Failed to publish order book" in str(exc_info.value)

    def test_publishing_error(self, producer, mock_producer):
        """Test handling of publishing errors."""
        mock_producer.produce.side_effect = KafkaError("Publishing failed")
        
        with pytest.raises(PublishingError) as exc_info:
            producer.publish_orderbook(
                symbol="BTC-USD",
                bids=[[50000.0, 1.5]],
                asks=[[50100.0, 1.0]]
            )
            producer._publish_batch()
        
        assert "Failed to publish order book batch" in str(exc_info.value)

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
        producer.publish_orderbook(
            symbol="BTC-USD",
            bids=[[50000.0, 1.5]],
            asks=[[50100.0, 1.0]]
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
        {"symbol": "BTC-USD", "bids": [], "asks": []},  # Empty orderbook
        {"symbol": "BTC-USD", "bids": None, "asks": []},  # Invalid bids
        {"symbol": "BTC-USD", "bids": [], "asks": None},  # Invalid asks
        {"symbol": "", "bids": [[50000.0, 1.5]], "asks": [[50100.0, 1.0]]},  # Empty symbol
    ])
    def test_invalid_orderbook_data(self, producer, invalid_data):
        """Test handling of invalid orderbook data."""
        with pytest.raises(PublishingError):
            producer.publish_orderbook(**invalid_data)

    def test_metrics_recording(self, producer, mock_producer, mock_metrics):
        """Test metrics recording during publishing."""
        # Fill batch to capacity
        producer.publish_orderbook(
            symbol="BTC-USD",
            bids=[[50000.0, 1.5]],
            asks=[[50100.0, 1.0]]
        )
        producer.publish_orderbook(
            symbol="ETH-USD",
            bids=[[3000.0, 2.0]],
            asks=[[3100.0, 1.5]]
        )
        
        # Verify metrics were recorded
        mock_metrics.record_messages_published.assert_called_once()
        args = mock_metrics.record_messages_published.call_args[0]
        assert args[0] == 2  # Number of messages
        assert isinstance(args[1], float)  # Processing time 