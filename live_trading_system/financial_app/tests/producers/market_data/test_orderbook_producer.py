"""
Tests for the OrderBook producer.

This module contains tests for the OrderBookProducer class, including
normal operation, edge cases, and error handling.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from app.producers.market_data.orderbook_producer import OrderBookProducer
from app.producers.base.error import SerializationError, PublishingError
from app.producers.config.settings import KinesisSettings

# Test data
TEST_STREAM_NAME = "test-orderbook-stream"
TEST_SYMBOL = "AAPL"
TEST_TIMESTAMP = datetime.now()
TEST_EXCHANGE = "NASDAQ"

# Valid orderbook data
VALID_BIDS = [(150.0, 100.0), (149.0, 200.0), (148.0, 300.0)]
VALID_ASKS = [(151.0, 100.0), (152.0, 200.0), (153.0, 300.0)]

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
    return OrderBookProducer(TEST_STREAM_NAME)

class TestOrderBookProducer:
    """Test suite for OrderBookProducer."""
    
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
        producer = OrderBookProducer(
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
    
    def test_publish_orderbook_single(self, producer, mock_kinesis_client):
        """Test publishing a single orderbook message."""
        # Mock successful response
        mock_kinesis_client.put_records.return_value = {
            'FailedRecordCount': 0,
            'Records': [{'SequenceNumber': '123', 'ShardId': 'shard-1'}]
        }
        
        # Publish message
        producer.publish_orderbook(
            symbol=TEST_SYMBOL,
            bids=VALID_BIDS,
            asks=VALID_ASKS,
            timestamp=TEST_TIMESTAMP,
            exchange=TEST_EXCHANGE,
            depth=3
        )
        
        # Force batch publish
        producer._publish_batch()
        
        # Verify Kinesis client call
        mock_kinesis_client.put_records.assert_called_once()
        call_args = mock_kinesis_client.put_records.call_args[1]
        assert call_args['StreamName'] == TEST_STREAM_NAME
        assert len(call_args['Records']) == 1
        
        # Verify message content
        message = call_args['Records'][0]
        assert message['PartitionKey'] == TEST_SYMBOL
        data = message['Data'].decode('utf-8')
        assert TEST_SYMBOL in data
        assert str(VALID_BIDS[0][0]) in data
        assert str(VALID_ASKS[0][0]) in data
    
    def test_publish_orderbook_batch(self, producer, mock_kinesis_client):
        """Test publishing multiple orderbook messages in a batch."""
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
            producer.publish_orderbook(
                symbol=TEST_SYMBOL,
                bids=[(150.0 + i, 100.0), (149.0 + i, 200.0)],
                asks=[(151.0 + i, 100.0), (152.0 + i, 200.0)],
                timestamp=TEST_TIMESTAMP + timedelta(seconds=i),
                exchange=TEST_EXCHANGE,
                depth=2
            )
        
        # Force batch publish
        producer._publish_batch()
        
        # Verify Kinesis client call
        mock_kinesis_client.put_records.assert_called_once()
        call_args = mock_kinesis_client.put_records.call_args[1]
        assert call_args['StreamName'] == TEST_STREAM_NAME
        assert len(call_args['Records']) == 2
    
    def test_publish_orderbook_edge_cases(self, producer, mock_kinesis_client):
        """Test publishing orderbook messages with edge case values."""
        # Mock successful response
        mock_kinesis_client.put_records.return_value = {
            'FailedRecordCount': 0,
            'Records': [{'SequenceNumber': '123', 'ShardId': 'shard-1'}]
        }
        
        # Test with empty orderbook
        producer.publish_orderbook(
            symbol=TEST_SYMBOL,
            bids=[],
            asks=[],
            timestamp=TEST_TIMESTAMP,
            exchange=TEST_EXCHANGE,
            depth=0
        )
        
        # Test with single level
        producer.publish_orderbook(
            symbol=TEST_SYMBOL,
            bids=[(150.0, 100.0)],
            asks=[(151.0, 100.0)],
            timestamp=TEST_TIMESTAMP,
            exchange=TEST_EXCHANGE,
            depth=1
        )
        
        # Test with very large quantities
        producer.publish_orderbook(
            symbol=TEST_SYMBOL,
            bids=[(150.0, 1e9)],
            asks=[(151.0, 1e9)],
            timestamp=TEST_TIMESTAMP,
            exchange=TEST_EXCHANGE,
            depth=1
        )
        
        # Force batch publish
        producer._publish_batch()
        
        # Verify Kinesis client call
        mock_kinesis_client.put_records.assert_called_once()
        call_args = mock_kinesis_client.put_records.call_args[1]
        assert call_args['StreamName'] == TEST_STREAM_NAME
        assert len(call_args['Records']) == 3
    
    def test_publish_orderbook_invalid_inputs(self, producer):
        """Test publishing orderbook messages with invalid inputs."""
        # Test with negative prices
        with pytest.raises(ValueError):
            producer.publish_orderbook(
                symbol=TEST_SYMBOL,
                bids=[(-150.0, 100.0)],
                asks=[(151.0, 100.0)],
                timestamp=TEST_TIMESTAMP,
                exchange=TEST_EXCHANGE,
                depth=1
            )
        
        # Test with negative quantities
        with pytest.raises(ValueError):
            producer.publish_orderbook(
                symbol=TEST_SYMBOL,
                bids=[(150.0, -100.0)],
                asks=[(151.0, 100.0)],
                timestamp=TEST_TIMESTAMP,
                exchange=TEST_EXCHANGE,
                depth=1
            )
        
        # Test with invalid depth
        with pytest.raises(ValueError):
            producer.publish_orderbook(
                symbol=TEST_SYMBOL,
                bids=VALID_BIDS,
                asks=VALID_ASKS,
                timestamp=TEST_TIMESTAMP,
                exchange=TEST_EXCHANGE,
                depth=10  # Depth greater than actual levels
            )
        
        # Test with crossed orderbook
        with pytest.raises(ValueError):
            producer.publish_orderbook(
                symbol=TEST_SYMBOL,
                bids=[(152.0, 100.0)],  # Bid price higher than ask
                asks=[(151.0, 100.0)],
                timestamp=TEST_TIMESTAMP,
                exchange=TEST_EXCHANGE,
                depth=1
            )
    
    def test_publish_orderbook_publishing_error(self, producer, mock_kinesis_client):
        """Test handling of publishing errors."""
        # Mock failed response
        mock_kinesis_client.put_records.return_value = {
            'FailedRecordCount': 1,
            'Records': [
                {'ErrorCode': 'ProvisionedThroughputExceededException', 'ErrorMessage': 'Throughput exceeded'}
            ]
        }
        
        # Publish message
        producer.publish_orderbook(
            symbol=TEST_SYMBOL,
            bids=VALID_BIDS,
            asks=VALID_ASKS,
            timestamp=TEST_TIMESTAMP,
            exchange=TEST_EXCHANGE,
            depth=3
        )
        
        # Force batch publish and expect error
        with pytest.raises(PublishingError) as exc_info:
            producer._publish_batch()
        
        assert "1 records failed to publish" in str(exc_info.value)
    
    def test_on_stop(self, producer, mock_kinesis_client):
        """Test cleanup on stop."""
        # Add a message to the batch
        producer.publish_orderbook(
            symbol=TEST_SYMBOL,
            bids=VALID_BIDS,
            asks=VALID_ASKS,
            timestamp=TEST_TIMESTAMP,
            exchange=TEST_EXCHANGE,
            depth=3
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