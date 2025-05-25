"""
Unit tests for the OHLCVProducer class.

This module contains comprehensive tests for the OHLCVProducer class,
including tests for message publishing, batching, error handling, and cleanup.
"""

import json
import time
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch, call

import pytest
from confluent_kafka import KafkaError, Message

from app.producers.market_data.ohlcv_producer import OHLCVProducer
from app.producers.base.error import SerializationError, PublishingError
from app.producers.config.settings import KafkaSettings

# Test data
VALID_OHLCV = {
    "symbol": "BTC-USD",
    "open": 50000.0,
    "high": 51000.0,
    "low": 49000.0,
    "close": 50500.0,
    "volume": 100.5,
    "interval": "1h",
    "timestamp": 1645000000000,
    "source": "test",
    "vwap": 50200.0,
    "trades_count": 1000,
    "open_interest": 5000.0,
    "adjusted_close": 50400.0
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
    """Create an OHLCVProducer instance with mocked dependencies."""
    settings = KafkaSettings()
    settings.OHLCV_TOPIC = "test-ohlcv-topic"
    return OHLCVProducer(settings=settings, batch_size=2, batch_timeout_ms=100)

class TestOHLCVProducer:
    """Test suite for OHLCVProducer class."""

    def test_initialization(self, producer, mock_producer, mock_metrics):
        """Test producer initialization."""
        assert producer.topic == "test-ohlcv-topic"
        assert producer.batch_size == 2
        assert producer.batch_timeout_ms == 100
        assert len(producer._batch) == 0
        mock_producer.assert_called_once()
        mock_metrics.assert_called_once()

    def test_publish_ohlcv_basic(self, producer, mock_producer):
        """Test basic OHLCV publishing."""
        producer.publish_ohlcv(
            symbol="BTC-USD",
            open_price=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=100.5,
            interval="1h"
        )
        
        # Verify message was added to batch
        assert len(producer._batch) == 1
        message = producer._batch[0]
        assert message["symbol"] == "BTC-USD"
        assert message["open"] == 50000.0
        assert message["high"] == 51000.0
        assert message["low"] == 49000.0
        assert message["close"] == 50500.0
        assert message["volume"] == 100.5
        assert message["interval"] == "1h"
        assert "timestamp" in message
        assert message["source"] == "producer"

    def test_publish_ohlcv_with_optional_fields(self, producer, mock_producer):
        """Test OHLCV publishing with all optional fields."""
        producer.publish_ohlcv(
            symbol="BTC-USD",
            open_price=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=100.5,
            interval="1h",
            vwap=50200.0,
            trades_count=1000,
            open_interest=5000.0,
            adjusted_close=50400.0
        )
        
        message = producer._batch[0]
        assert message["vwap"] == 50200.0
        assert message["trades_count"] == 1000
        assert message["open_interest"] == 5000.0
        assert message["adjusted_close"] == 50400.0

    def test_timestamp_handling(self, producer, mock_producer):
        """Test different timestamp formats."""
        # Test with milliseconds
        producer.publish_ohlcv(
            symbol="BTC-USD",
            open_price=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=100.5,
            interval="1h",
            timestamp=1645000000000
        )
        assert producer._batch[0]["timestamp"] == 1645000000000
        
        # Test with seconds
        producer._batch = []
        producer.publish_ohlcv(
            symbol="BTC-USD",
            open_price=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=100.5,
            interval="1h",
            timestamp=1645000000
        )
        assert producer._batch[0]["timestamp"] == 1645000000000
        
        # Test with ISO format
        producer._batch = []
        producer.publish_ohlcv(
            symbol="BTC-USD",
            open_price=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=100.5,
            interval="1h",
            timestamp="2022-02-16T12:00:00Z"
        )
        assert isinstance(producer._batch[0]["timestamp"], int)

    def test_batch_processing(self, producer, mock_producer):
        """Test batch processing functionality."""
        # Fill batch to capacity
        producer.publish_ohlcv(
            symbol="BTC-USD",
            open_price=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=100.5,
            interval="1h"
        )
        producer.publish_ohlcv(
            symbol="ETH-USD",
            open_price=3000.0,
            high=3100.0,
            low=2900.0,
            close=3050.0,
            volume=50.0,
            interval="1h"
        )
        
        # Verify batch was published
        assert len(producer._batch) == 0
        assert mock_producer.produce.call_count == 2
        mock_producer.flush.assert_called_once()

    def test_batch_timeout(self, producer, mock_producer):
        """Test batch timeout functionality."""
        producer.publish_ohlcv(
            symbol="BTC-USD",
            open_price=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=100.5,
            interval="1h"
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
                producer.publish_ohlcv(
                    symbol="BTC-USD",
                    open_price=50000.0,
                    high=51000.0,
                    low=49000.0,
                    close=50500.0,
                    volume=100.5,
                    interval="1h"
                )
            
            assert "Failed to publish OHLCV data" in str(exc_info.value)

    def test_publishing_error(self, producer, mock_producer):
        """Test handling of publishing errors."""
        mock_producer.produce.side_effect = KafkaError("Publishing failed")
        
        with pytest.raises(PublishingError) as exc_info:
            producer.publish_ohlcv(
                symbol="BTC-USD",
                open_price=50000.0,
                high=51000.0,
                low=49000.0,
                close=50500.0,
                volume=100.5,
                interval="1h"
            )
            producer._publish_batch()
        
        assert "Failed to publish OHLCV batch" in str(exc_info.value)

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
        producer.publish_ohlcv(
            symbol="BTC-USD",
            open_price=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=100.5,
            interval="1h"
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
            "open_price": 50000.0,
            "high": 51000.0,
            "low": 49000.0,
            "close": 50500.0,
            "volume": -100.5,  # Invalid volume
            "interval": "1h"
        },
        {
            "symbol": "BTC-USD",
            "open_price": 50000.0,
            "high": 49000.0,  # High < Open
            "low": 49000.0,
            "close": 50500.0,
            "volume": 100.5,
            "interval": "1h"
        },
        {
            "symbol": "BTC-USD",
            "open_price": 50000.0,
            "high": 51000.0,
            "low": 52000.0,  # Low > High
            "close": 50500.0,
            "volume": 100.5,
            "interval": "1h"
        },
        {
            "symbol": "",  # Empty symbol
            "open_price": 50000.0,
            "high": 51000.0,
            "low": 49000.0,
            "close": 50500.0,
            "volume": 100.5,
            "interval": "1h"
        },
        {
            "symbol": "BTC-USD",
            "open_price": 50000.0,
            "high": 51000.0,
            "low": 49000.0,
            "close": 50500.0,
            "volume": 100.5,
            "interval": "invalid"  # Invalid interval
        }
    ])
    def test_invalid_ohlcv_data(self, producer, invalid_data):
        """Test handling of invalid OHLCV data."""
        with pytest.raises(PublishingError):
            producer.publish_ohlcv(**invalid_data)

    def test_metrics_recording(self, producer, mock_producer, mock_metrics):
        """Test metrics recording during publishing."""
        # Fill batch to capacity
        producer.publish_ohlcv(
            symbol="BTC-USD",
            open_price=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=100.5,
            interval="1h"
        )
        producer.publish_ohlcv(
            symbol="ETH-USD",
            open_price=3000.0,
            high=3100.0,
            low=2900.0,
            close=3050.0,
            volume=50.0,
            interval="1h"
        )
        
        # Verify metrics were recorded
        mock_metrics.record_messages_published.assert_called_once()
        args = mock_metrics.record_messages_published.call_args[0]
        assert args[0] == 2  # Number of messages
        assert isinstance(args[1], float)  # Processing time

    def test_message_key_encoding(self, producer, mock_producer):
        """Test that message keys are properly encoded."""
        producer.publish_ohlcv(
            symbol="BTC-USD",
            open_price=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=100.5,
            interval="1h"
        )
        producer._publish_batch()
        
        # Verify key encoding
        call_args = mock_producer.produce.call_args[1]
        assert call_args["key"] == b"BTC-USD" 