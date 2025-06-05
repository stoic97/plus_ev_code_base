"""
Tests for the base Kinesis consumer.

This module contains tests for the BaseKinesisConsumer class, including
initialization, message processing, error handling, and metrics recording.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

import boto3
from botocore.exceptions import ClientError

from app.consumers.base.kinesis_consumer import BaseKinesisConsumer
from app.consumers.base.error import DeserializationError, ProcessingError
from app.consumers.config.settings import KinesisSettings

# Test data
TEST_STREAM_NAME = "test-stream"
TEST_SHARD_ID = "shard-000000000000"
TEST_RECORD = {
    "Data": json.dumps({
        "symbol": "AAPL",
        "price": 150.0,
        "quantity": 100,
        "timestamp": datetime.now().isoformat()
    }).encode('utf-8'),
    "PartitionKey": "AAPL",
    "SequenceNumber": "1234567890"
}

class TestKinesisConsumer(BaseKinesisConsumer):
    """Test implementation of BaseKinesisConsumer."""
    
    def process_message(self, message: dict, raw_record: dict) -> None:
        """Process a test message."""
        self.last_processed_message = message
        self.last_raw_record = raw_record

@pytest.fixture
def mock_kinesis_client():
    """Create a mock Kinesis client."""
    with patch('boto3.client') as mock_client:
        client = MagicMock()
        mock_client.return_value = client
        yield client

@pytest.fixture
def consumer(mock_kinesis_client):
    """Create a test consumer instance."""
    # Mock stream description
    mock_kinesis_client.describe_stream.return_value = {
        'StreamDescription': {
            'Shards': [{'ShardId': TEST_SHARD_ID}]
        }
    }
    
    # Mock shard iterator
    mock_kinesis_client.get_shard_iterator.return_value = {
        'ShardIterator': 'test-iterator'
    }
    
    # Mock get records
    mock_kinesis_client.get_records.return_value = {
        'Records': [TEST_RECORD],
        'NextShardIterator': 'next-iterator'
    }
    
    return TestKinesisConsumer(TEST_STREAM_NAME)

def test_initialization(consumer, mock_kinesis_client):
    """Test consumer initialization."""
    assert consumer.stream_name == TEST_STREAM_NAME
    assert consumer._shard_ids == [TEST_SHARD_ID]
    assert consumer._shard_iterators[TEST_SHARD_ID] == 'test-iterator'
    
    # Verify Kinesis client calls
    mock_kinesis_client.describe_stream.assert_called_once_with(
        StreamName=TEST_STREAM_NAME
    )
    mock_kinesis_client.get_shard_iterator.assert_called_once_with(
        StreamName=TEST_STREAM_NAME,
        ShardId=TEST_SHARD_ID,
        ShardIteratorType='LATEST'
    )

def test_get_shard_iterator_error(mock_kinesis_client):
    """Test error handling in get_shard_iterator."""
    mock_kinesis_client.get_shard_iterator.side_effect = ClientError(
        {'Error': {'Code': 'InvalidArgumentException', 'Message': 'Invalid shard'}},
        'GetShardIterator'
    )
    
    with pytest.raises(ProcessingError) as exc_info:
        TestKinesisConsumer(TEST_STREAM_NAME)
    
    assert "Failed to get shard iterator" in str(exc_info.value)

def test_deserialize_message(consumer):
    """Test message deserialization."""
    data = json.dumps({"test": "data"}).encode('utf-8')
    result = consumer._deserialize_message(data)
    assert result == {"test": "data"}

def test_deserialize_message_error(consumer):
    """Test deserialization error handling."""
    with pytest.raises(DeserializationError):
        consumer._deserialize_message(b"invalid json")

def test_process_record(consumer):
    """Test record processing."""
    consumer._process_record(TEST_RECORD)
    
    assert consumer.last_processed_message["symbol"] == "AAPL"
    assert consumer.last_processed_message["price"] == 150.0
    assert consumer.last_raw_record == TEST_RECORD

def test_process_record_error(consumer, mock_kinesis_client):
    """Test error handling in record processing."""
    mock_kinesis_client.get_records.return_value = {
        'Records': [{'Data': b'invalid data'}],
        'NextShardIterator': 'next-iterator'
    }
    
    with pytest.raises(ProcessingError):
        consumer.consume()

def test_consume_expired_iterator(consumer, mock_kinesis_client):
    """Test handling of expired iterator."""
    # First call succeeds, second call fails with expired iterator
    mock_kinesis_client.get_records.side_effect = [
        {'Records': [TEST_RECORD], 'NextShardIterator': 'next-iterator'},
        ClientError(
            {'Error': {'Code': 'ExpiredIteratorException', 'Message': 'Iterator expired'}},
            'GetRecords'
        )
    ]
    
    # Should not raise an error, should refresh iterator
    consumer.consume()
    
    # Verify iterator was refreshed
    assert mock_kinesis_client.get_shard_iterator.call_count == 2

def test_consume_client_error(consumer, mock_kinesis_client):
    """Test handling of client errors."""
    mock_kinesis_client.get_records.side_effect = ClientError(
        {'Error': {'Code': 'InternalFailure', 'Message': 'Internal error'}},
        'GetRecords'
    )
    
    with pytest.raises(ProcessingError) as exc_info:
        consumer.consume()
    
    assert "Failed to consume records" in str(exc_info.value)

def test_metrics_recording(consumer):
    """Test metrics recording during message processing."""
    # Process a record
    consumer._process_record(TEST_RECORD)
    
    # Verify metrics were recorded
    assert consumer.metrics.messages_processed > 0
    assert consumer.metrics.messages_failed == 0

def test_on_stop(consumer):
    """Test cleanup on stop."""
    # Should not raise any errors
    consumer.on_stop() 