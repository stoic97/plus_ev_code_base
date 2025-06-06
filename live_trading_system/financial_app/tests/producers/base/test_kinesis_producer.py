"""
Tests for the base Kinesis producer.

This module contains tests for the BaseKinesisProducer class, including
initialization, message publishing, error handling, and metrics recording.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

import boto3
from botocore.exceptions import ClientError

from app.producers.base.kinesis_producer import BaseKinesisProducer
from app.producers.base.error import SerializationError, PublishingError
from app.producers.config.settings import KinesisSettings
from app.producers.base.metrics import get_metrics_registry

# Test data
TEST_STREAM_NAME = "test-stream"
TEST_MESSAGE = {
    "symbol": "AAPL",
    "price": 150.0,
    "quantity": 100,
    "timestamp": datetime.now().isoformat()
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
    # Reset metrics registry before each test
    registry = get_metrics_registry()
    registry._initialize()
    return BaseKinesisProducer(TEST_STREAM_NAME)

def test_initialization(producer):
    """Test producer initialization."""
    assert producer.stream_name == TEST_STREAM_NAME
    assert isinstance(producer.settings, KinesisSettings)
    assert producer.batch_size == 100
    assert producer.batch_timeout_ms == 1000

def test_serialize_message(producer):
    """Test message serialization."""
    result = producer._serialize_message(TEST_MESSAGE)
    assert isinstance(result, bytes)
    assert json.loads(result.decode('utf-8')) == TEST_MESSAGE

def test_serialize_message_error(producer):
    """Test serialization error handling."""
    # Create an object that can't be serialized
    class Unserializable:
        pass
    
    with pytest.raises(SerializationError):
        producer._serialize_message(Unserializable())

def test_publish_message(producer, mock_kinesis_client):
    """Test single message publishing."""
    producer.publish_message(TEST_MESSAGE, "AAPL")
    
    # Verify Kinesis client call
    mock_kinesis_client.put_record.assert_called_once()
    call_args = mock_kinesis_client.put_record.call_args[1]
    assert call_args['StreamName'] == TEST_STREAM_NAME
    assert call_args['PartitionKey'] == "AAPL"
    assert json.loads(call_args['Data'].decode('utf-8')) == TEST_MESSAGE

def test_publish_message_error(producer, mock_kinesis_client):
    """Test error handling in message publishing."""
    mock_kinesis_client.put_record.side_effect = ClientError(
        {'Error': {'Code': 'InternalFailure', 'Message': 'Internal error'}},
        'PutRecord'
    )
    
    with pytest.raises(PublishingError) as exc_info:
        producer.publish_message(TEST_MESSAGE, "AAPL")
    
    assert "Failed to publish message" in str(exc_info.value)

def test_publish_batch(producer, mock_kinesis_client):
    """Test batch message publishing."""
    # Mock successful response
    mock_kinesis_client.put_records.return_value = {
        'FailedRecordCount': 0,
        'Records': [
            {'SequenceNumber': '123', 'ShardId': 'shard-1'},
            {'SequenceNumber': '456', 'ShardId': 'shard-2'}
        ]
    }
    
    messages = [
        (TEST_MESSAGE, "AAPL"),
        (TEST_MESSAGE, "GOOGL")
    ]
    
    producer.publish_batch(messages)
    
    # Verify Kinesis client call
    mock_kinesis_client.put_records.assert_called_once()
    call_args = mock_kinesis_client.put_records.call_args[1]
    assert call_args['StreamName'] == TEST_STREAM_NAME
    assert len(call_args['Records']) == 2

def test_publish_batch_with_failures(producer, mock_kinesis_client):
    """Test batch publishing with some failures."""
    # Mock response with one failed record
    mock_kinesis_client.put_records.return_value = {
        'FailedRecordCount': 1,
        'Records': [
            {'SequenceNumber': '123', 'ShardId': 'shard-1'},
            {'ErrorCode': 'ProvisionedThroughputExceededException', 'ErrorMessage': 'Throughput exceeded'}
        ]
    }
    
    messages = [
        (TEST_MESSAGE, "AAPL"),
        (TEST_MESSAGE, "GOOGL")
    ]
    
    with pytest.raises(PublishingError) as exc_info:
        producer.publish_batch(messages)
    
    assert "1 records failed to publish" in str(exc_info.value)

def test_publish_batch_error(producer, mock_kinesis_client):
    """Test error handling in batch publishing."""
    mock_kinesis_client.put_records.side_effect = ClientError(
        {'Error': {'Code': 'InternalFailure', 'Message': 'Internal error'}},
        'PutRecords'
    )
    
    messages = [(TEST_MESSAGE, "AAPL")]
    
    with pytest.raises(PublishingError) as exc_info:
        producer.publish_batch(messages)
    
    assert "Failed to publish batch" in str(exc_info.value)

def test_metrics_recording(producer, mock_kinesis_client):
    """Test metrics recording during message publishing."""
    # Publish a message
    producer.publish_message(TEST_MESSAGE, "AAPL")
    
    # Verify metrics were recorded
    assert producer.metrics.messages_published > 0
    assert producer.metrics.messages_failed == 0

def test_on_stop(producer):
    """Test cleanup on stop."""
    # Should not raise any errors
    producer.on_stop() 