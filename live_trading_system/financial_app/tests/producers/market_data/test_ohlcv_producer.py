"""
Tests for the OHLCV producer.

This module contains tests for the OHLCVProducer class, including
normal operation, edge cases, and error handling.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from app.producers.market_data.ohlcv_producer import OHLCVProducer
from app.producers.base.error import SerializationError, PublishingError
from app.producers.config.settings import KinesisSettings

# Test data
TEST_STREAM_NAME = "test-ohlcv-stream"
TEST_SYMBOL = "AAPL"
TEST_TIMESTAMP = datetime.now()
TEST_INTERVAL = "1m"
TEST_EXCHANGE = "NASDAQ"

# Valid OHLCV data
VALID_OHLCV = {
    "symbol": TEST_SYMBOL,
    "open": 150.0,
    "high": 155.0,
    "low": 148.0,
    "close": 153.0,
    "volume": 1000.0,
    "timestamp": TEST_TIMESTAMP.isoformat(),
    "interval": TEST_INTERVAL,
    "exchange": TEST_EXCHANGE
}

@pytest.fixture
def mock_kinesis_client():
    """Create a mock Kinesis client."""
    with patch('boto3.client') as mock_client:
        client = MagicMock()
        mock_client.return_value = client
        yield client

@pytest.fixture
def producer(mock_kinesis_client):
    """Create a test producer instance."""
    return OHLCVProducer(TEST_STREAM_NAME)

class TestOHLCVProducer:
    """Test suite for OHLCVProducer."""
    
    def test_initialization(self, producer):
        """Test producer initialization with default values."""
        assert producer.stream_name == TEST_STREAM_NAME
        assert isinstance(producer.settings, KinesisSettings)
        assert producer.batch_size == 100
        assert producer.batch_timeout_ms == 1000
    
    def test_initialization_custom_settings(self):
        """Test producer initialization with custom settings."""
        custom_settings = KinesisSettings(
            aws_access_key_id="test-key",
            aws_secret_access_key="test-secret",
            aws_region="us-west-2"
        )
        producer = OHLCVProducer(
            stream_name=TEST_STREAM_NAME,
            settings=custom_settings,
            region_name="us-west-2",
            batch_size=50,
            batch_timeout_ms=500
        )
        assert producer.stream_name == TEST_STREAM_NAME
        assert producer.settings == custom_settings
        assert producer.batch_size == 50
        assert producer.batch_timeout_ms == 500
    
    def test_publish_ohlcv_single(self, producer, mock_kinesis_client):
        """Test publishing a single OHLCV message."""
        # Mock successful response
        mock_kinesis_client.put_records.return_value = {
            'FailedRecordCount': 0,
            'Records': [{'SequenceNumber': '123', 'ShardId': 'shard-1'}]
        }
        
        # Publish message
        producer.publish_ohlcv(
            symbol=TEST_SYMBOL,
            open_price=150.0,
            high_price=155.0,
            low_price=148.0,
            close_price=153.0,
            volume=1000.0,
            timestamp=TEST_TIMESTAMP,
            interval=TEST_INTERVAL,
            exchange=TEST_EXCHANGE
        )
        
        # Force batch publish
        producer._publish_batch()
        
        # Verify Kinesis client call
        mock_kinesis_client.put_records.assert_called_once()
        call_args = mock_kinesis_client.put_records.call_args[1]
        assert call_args['StreamName'] == TEST_STREAM_NAME
        assert len(call_args['Records']) == 1
    
    def test_publish_ohlcv_batch(self, producer, mock_kinesis_client):
        """Test publishing multiple OHLCV messages in a batch."""
        # Mock successful response
        mock_kinesis_client.put_records.return_value = {
            'FailedRecordCount': 0,
            'Records': [
                {'SequenceNumber': '123', 'ShardId': 'shard-1'},
                {'SequenceNumber': '456', 'ShardId': 'shard-2'}
            ]
        }
        
        # Publish multiple messages
        for i in range(2):
            producer.publish_ohlcv(
                symbol=TEST_SYMBOL,
                open_price=150.0 + i,
                high_price=155.0 + i,
                low_price=148.0 + i,
                close_price=153.0 + i,
                volume=1000.0 + i,
                timestamp=TEST_TIMESTAMP + timedelta(minutes=i),
                interval=TEST_INTERVAL,
                exchange=TEST_EXCHANGE
            )
        
        # Force batch publish
        producer._publish_batch()
        
        # Verify Kinesis client call
        mock_kinesis_client.put_records.assert_called_once()
        call_args = mock_kinesis_client.put_records.call_args[1]
        assert call_args['StreamName'] == TEST_STREAM_NAME
        assert len(call_args['Records']) == 2
    
    def test_publish_ohlcv_edge_cases(self, producer, mock_kinesis_client):
        """Test publishing OHLCV messages with edge case values."""
        # Mock successful response
        mock_kinesis_client.put_records.return_value = {
            'FailedRecordCount': 0,
            'Records': [{'SequenceNumber': '123', 'ShardId': 'shard-1'}]
        }
        
        # Test with zero values
        producer.publish_ohlcv(
            symbol=TEST_SYMBOL,
            open_price=0.0,
            high_price=0.0,
            low_price=0.0,
            close_price=0.0,
            volume=0.0,
            timestamp=TEST_TIMESTAMP,
            interval=TEST_INTERVAL,
            exchange=TEST_EXCHANGE
        )
        
        # Test with very large values
        producer.publish_ohlcv(
            symbol=TEST_SYMBOL,
            open_price=1e9,
            high_price=1e9,
            low_price=1e9,
            close_price=1e9,
            volume=1e9,
            timestamp=TEST_TIMESTAMP,
            interval=TEST_INTERVAL,
            exchange=TEST_EXCHANGE
        )
        
        # Force batch publish
        producer._publish_batch()
        
        # Verify Kinesis client call
        mock_kinesis_client.put_records.assert_called_once()
        call_args = mock_kinesis_client.put_records.call_args[1]
        assert call_args['StreamName'] == TEST_STREAM_NAME
        assert len(call_args['Records']) == 2
    
    def test_publish_ohlcv_invalid_inputs(self, producer):
        """Test publishing OHLCV messages with invalid inputs."""
        # Test with negative prices
        with pytest.raises(ValueError):
            producer.publish_ohlcv(
                symbol=TEST_SYMBOL,
                open_price=-150.0,
                high_price=155.0,
                low_price=148.0,
                close_price=153.0,
                volume=1000.0,
                timestamp=TEST_TIMESTAMP,
                interval=TEST_INTERVAL,
                exchange=TEST_EXCHANGE
            )
        
        # Test with negative volume
        with pytest.raises(ValueError):
            producer.publish_ohlcv(
                symbol=TEST_SYMBOL,
                open_price=150.0,
                high_price=155.0,
                low_price=148.0,
                close_price=153.0,
                volume=-1000.0,
                timestamp=TEST_TIMESTAMP,
                interval=TEST_INTERVAL,
                exchange=TEST_EXCHANGE
            )
        
        # Test with invalid interval
        with pytest.raises(ValueError):
            producer.publish_ohlcv(
                symbol=TEST_SYMBOL,
                open_price=150.0,
                high_price=155.0,
                low_price=148.0,
                close_price=153.0,
                volume=1000.0,
                timestamp=TEST_TIMESTAMP,
                interval="invalid",
                exchange=TEST_EXCHANGE
            )
    
    def test_publish_ohlcv_publishing_error(self, producer, mock_kinesis_client):
        """Test handling of publishing errors."""
        # Mock failed response
        mock_kinesis_client.put_records.return_value = {
            'FailedRecordCount': 1,
            'Records': [
                {'ErrorCode': 'ProvisionedThroughputExceededException', 'ErrorMessage': 'Throughput exceeded'}
            ]
        }
        
        # Publish message
        producer.publish_ohlcv(
            symbol=TEST_SYMBOL,
            open_price=150.0,
            high_price=155.0,
            low_price=148.0,
            close_price=153.0,
            volume=1000.0,
            timestamp=TEST_TIMESTAMP,
            interval=TEST_INTERVAL,
            exchange=TEST_EXCHANGE
        )
        
        # Force batch publish and expect error
        with pytest.raises(PublishingError) as exc_info:
            producer._publish_batch()
        
        assert "1 records failed to publish" in str(exc_info.value)
    
    def test_on_stop(self, producer, mock_kinesis_client):
        """Test cleanup on stop."""
        # Add a message to the batch
        producer.publish_ohlcv(
            symbol=TEST_SYMBOL,
            open_price=150.0,
            high_price=155.0,
            low_price=148.0,
            close_price=153.0,
            volume=1000.0,
            timestamp=TEST_TIMESTAMP,
            interval=TEST_INTERVAL,
            exchange=TEST_EXCHANGE
        )
        
        # Mock successful response
        mock_kinesis_client.put_records.return_value = {
            'FailedRecordCount': 0,
            'Records': [{'SequenceNumber': '123', 'ShardId': 'shard-1'}]
        }
        
        # Call on_stop
        producer.on_stop()
        
        # Verify that the batch was published
        mock_kinesis_client.put_records.assert_called_once() 