"""
Unit tests for the BaseConsumer class.

This module contains comprehensive tests for the BaseConsumer abstract base class,
testing its functionality for connection management, message processing,
error handling, shutdown behavior, and statistics tracking.
"""

import json
import threading
import time
import unittest
from unittest.mock import MagicMock, patch, call

import pytest
from confluent_kafka import KafkaError, KafkaException

from app.consumers.base.consumer import BaseConsumer, ConsumerStats
from app.consumers.base.error import DeserializationError, ProcessingError
from app.consumers.config.settings import KafkaSettings


class ConcreteConsumer(BaseConsumer):
    """
    Concrete implementation of BaseConsumer for testing.
    
    Implements the abstract methods required by BaseConsumer.
    """
    
    def __init__(self, **kwargs):
        """Initialize with test configurations."""
        self.deserialized_messages = []
        self.processed_messages = []
        self.deserialize_error = kwargs.pop('deserialize_error', False)
        self.process_error = kwargs.pop('process_error', False)
        super().__init__(**kwargs)
    
    def _deserialize_message(self, msg):
        """Test implementation of message deserialization."""
        if self.deserialize_error:
            raise DeserializationError("Test deserialization error")
        
        try:
            # Simple JSON deserialization for testing
            data = json.loads(msg.value().decode('utf-8'))
            self.deserialized_messages.append(data)
            return data
        except Exception as e:
            raise DeserializationError(f"Failed to deserialize: {e}")
    
    def process_message(self, message, raw_message):
        """Test implementation of message processing."""
        if self.process_error:
            raise ProcessingError("Test processing error")
        
        # Simply store the message for test verification
        self.processed_messages.append(message)


class TestConsumerStats:
    """Tests for the ConsumerStats class."""
    
    def test_init(self):
        """Test initialization of stats object."""
        stats = ConsumerStats(topic="test-topic")
        
        assert stats.topic == "test-topic"
        assert stats.messages_processed == 0
        assert stats.messages_failed == 0
        assert stats.bytes_processed == 0
        assert stats.last_message_timestamp is None
        assert stats.processing_time_ms == 0
        assert stats.avg_processing_time_ms == 0
    
    def test_update_processing_stats(self):
        """Test updating processing statistics."""
        stats = ConsumerStats(topic="test-topic")
        
        # Update with first message
        stats.update_processing_stats(processing_time_ms=10.0, message_size=100)
        
        assert stats.messages_processed == 1
        assert stats.bytes_processed == 100
        assert stats.processing_time_ms == 10.0
        assert stats.avg_processing_time_ms == 10.0
        assert stats.last_message_timestamp is not None
        
        # Update with second message
        stats.update_processing_stats(processing_time_ms=20.0, message_size=200)
        
        assert stats.messages_processed == 2
        assert stats.bytes_processed == 300  # 100 + 200
        assert stats.processing_time_ms == 30.0  # 10 + 20
        assert stats.avg_processing_time_ms == 15.0  # (10 + 20) / 2
    
    def test_record_failure(self):
        """Test recording message failures."""
        stats = ConsumerStats(topic="test-topic")
        
        assert stats.messages_failed == 0
        
        stats.record_failure()
        assert stats.messages_failed == 1
        
        stats.record_failure()
        assert stats.messages_failed == 2


class TestBaseConsumer:
    """Tests for the BaseConsumer class."""
    
    @pytest.fixture
    def mock_consumer(self):
        """Fixture for creating a mock Kafka consumer."""
        mock = MagicMock()
        return mock
    
    @pytest.fixture
    def consumer_settings(self):
        """Fixture for consumer settings."""
        return KafkaSettings(
            BOOTSTRAP_SERVERS=["localhost:9092"],
            MARKET_DATA_TOPIC="market-data",
            GROUP_ID="test-group",
            AUTO_OFFSET_RESET="latest",
            ENABLE_AUTO_COMMIT=False,
            MAX_POLL_INTERVAL_MS=300000,
            SESSION_TIMEOUT_MS=30000,
            REQUEST_TIMEOUT_MS=40000,
        )
    
    @patch('app.consumers.base.consumer.Consumer')
    def test_init(self, mock_consumer_class, consumer_settings):
        """Test consumer initialization."""
        mock_consumer_instance = MagicMock()
        mock_consumer_class.return_value = mock_consumer_instance
        
        consumer = ConcreteConsumer(
            topic="test-topic",
            group_id="test-group",
            settings=consumer_settings
        )
        
        # Verify consumer was created with the right config
        mock_consumer_class.assert_called_once()
        config = mock_consumer_class.call_args[0][0]
        assert config['bootstrap.servers'] == 'localhost:9092'
        assert config['group.id'] == 'test-group'
        assert config['auto.offset.reset'] == 'latest'
        assert config['enable.auto.commit'] is False
        
        # Verify subscription to the topic
        mock_consumer_instance.subscribe.assert_called_once_with(["test-topic"])
        
        # Verify internal state
        assert consumer.topic == "test-topic"
        assert consumer.group_id == "test-group"
        assert consumer._running is False
        assert consumer._shutdown_event.is_set() is False
    
    @patch('app.consumers.base.consumer.Consumer')
    def test_init_with_kafka_exception(self, mock_consumer_class, consumer_settings):
        """Test handling of KafkaException during initialization."""
        mock_consumer_class.side_effect = KafkaException("Test Kafka error")
        
        with pytest.raises(KafkaException):
            ConcreteConsumer(
                topic="test-topic",
                group_id="test-group",
                settings=consumer_settings
            )
    
    @patch('app.consumers.base.consumer.Consumer')
    def test_start_stop_nonblocking(self, mock_consumer_class, consumer_settings):
        """Test starting and stopping the consumer in non-blocking mode."""
        mock_consumer_instance = MagicMock()
        mock_consumer_class.return_value = mock_consumer_instance
        
        consumer = ConcreteConsumer(
            topic="test-topic",
            group_id="test-group",
            settings=consumer_settings
        )
        
        # Test starting in non-blocking mode
        consumer.start(blocking=False)
        
        # Verify thread is started
        assert consumer._running is True
        assert hasattr(consumer, '_thread')
        assert consumer._thread.daemon is True
        
        # Allow thread to run briefly
        time.sleep(0.1)
        
        # Test stopping
        consumer.stop()
        
        # Verify consumer is stopped
        assert consumer._running is False
        assert consumer._shutdown_event.is_set() is True
        mock_consumer_instance.close.assert_called_once()
    
    @patch('app.consumers.base.consumer.Consumer')
    def test_message_processing_success(self, mock_consumer_class, consumer_settings):
        """Test successful message processing."""
        # Create mock message
        mock_message = MagicMock()
        mock_message.error.return_value = None
        mock_message.value.return_value = json.dumps({"key": "value"}).encode('utf-8')
        
        # Setup mock consumer
        mock_consumer_instance = MagicMock()
        mock_consumer_instance.poll.side_effect = [
            mock_message,  # First poll returns message
            None,  # Second poll returns None to exit loop
        ]
        mock_consumer_class.return_value = mock_consumer_instance
        
        # Create consumer and run
        consumer = ConcreteConsumer(
            topic="test-topic",
            group_id="test-group",
            settings=consumer_settings,
            auto_commit=False  # Explicit commit mode
        )
        
        # Important: Don't start/stop the consumer, just patch the _consume_loop
        # This will preserve the consumer object
        with patch.object(consumer, '_consume_loop', autospec=True):
            # Call the process message method directly
            consumer._process_message(mock_message)

            # Verify message was processed
            assert len(consumer.deserialized_messages) == 1
            assert consumer.deserialized_messages[0] == {"key": "value"}
            assert len(consumer.processed_messages) == 1
            assert consumer.processed_messages[0] == {"key": "value"}

            # Verify offset was committed
            mock_consumer_instance.commit.assert_called_once_with(mock_message, asynchronous=False)
            
            # Verify stats were updated
            assert consumer.stats.messages_processed == 1
            assert consumer.stats.messages_failed == 0
    
    @patch('app.consumers.base.consumer.Consumer')
    def test_message_processing_deserialization_error(self, mock_consumer_class, consumer_settings):
        """Test handling of deserialization errors."""
        # Create mock message
        mock_message = MagicMock()
        mock_message.error.return_value = None
        mock_message.value.return_value = "invalid json".encode('utf-8')
        
        # Setup mock consumer
        mock_consumer_instance = MagicMock()
        mock_consumer_class.return_value = mock_consumer_instance
        
        # Create consumer with deserialization error flag
        consumer = ConcreteConsumer(
            topic="test-topic",
            group_id="test-group",
            settings=consumer_settings,
            deserialize_error=True  # Will force a deserialization error
        )
        
        # Process message directly
        consumer._process_message(mock_message)
        
        # Verify error was handled
        assert len(consumer.deserialized_messages) == 0
        assert len(consumer.processed_messages) == 0
        assert consumer.stats.messages_processed == 0
        assert consumer.stats.messages_failed == 1
        
        # Verify offset was not committed
        mock_consumer_instance.commit.assert_not_called()
    
    @patch('app.consumers.base.consumer.Consumer')
    def test_message_processing_processing_error(self, mock_consumer_class, consumer_settings):
        """Test handling of processing errors."""
        # Create mock message
        mock_message = MagicMock()
        mock_message.error.return_value = None
        mock_message.value.return_value = json.dumps({"key": "value"}).encode('utf-8')
        
        # Setup mock consumer
        mock_consumer_instance = MagicMock()
        mock_consumer_class.return_value = mock_consumer_instance
        
        # Create consumer with processing error flag
        consumer = ConcreteConsumer(
            topic="test-topic",
            group_id="test-group",
            settings=consumer_settings,
            process_error=True  # Will force a processing error
        )
        
        # Process message directly
        consumer._process_message(mock_message)
        
        # Verify message was deserialized but not processed
        assert len(consumer.deserialized_messages) == 1
        assert len(consumer.processed_messages) == 0
        assert consumer.stats.messages_processed == 0
        assert consumer.stats.messages_failed == 1
        
        # Verify offset was not committed
        mock_consumer_instance.commit.assert_not_called()
    
    @patch('app.consumers.base.consumer.Consumer')
    def test_handle_kafka_error(self, mock_consumer_class, consumer_settings):
        """Test handling of Kafka errors."""
        # Create mock consumer
        mock_consumer_instance = MagicMock()
        mock_consumer_class.return_value = mock_consumer_instance
        
        # Create consumer
        consumer = ConcreteConsumer(
            topic="test-topic",
            group_id="test-group",
            settings=consumer_settings
        )
        
        # Create a fatal Kafka error
        fatal_error = MagicMock()
        fatal_error.code.return_value = KafkaError.REQUEST_TIMED_OUT
        
        # Handle the error
        consumer._handle_kafka_error(fatal_error)
        
        # Verify error was recorded
        assert consumer.stats.messages_failed == 1
        
        # Create a non-fatal Kafka error
        non_fatal_error = MagicMock()
        non_fatal_error.code.return_value = KafkaError.OFFSET_OUT_OF_RANGE
        
        # Reset stats
        consumer.stats.messages_failed = 0
        
        # Handle the error
        consumer._handle_kafka_error(non_fatal_error)
        
        # Verify error was not recorded as a failure
        assert consumer.stats.messages_failed == 0
    
    @patch('app.consumers.base.consumer.Consumer')
    def test_consume_loop_with_shutdown(self, mock_consumer_class, consumer_settings):
        """Test the consume loop with shutdown handling."""
        # Setup mock consumer
        mock_consumer_instance = MagicMock()
        mock_consumer_class.return_value = mock_consumer_instance

        # Create consumer
        consumer = ConcreteConsumer(
            topic="test-topic",
            group_id="test-group",
            settings=consumer_settings
        )

        # Set up a thread to stop the consumer after a short delay
        def delayed_stop():
            time.sleep(0.1)
            # Directly set the shutdown event in the test
            consumer._shutdown_event.set()
            consumer.stop()

        stop_thread = threading.Thread(target=delayed_stop)
        stop_thread.daemon = True

        # Start the consumer and the stop thread
        with patch.object(consumer, '_process_message') as mock_process:
            stop_thread.start()
            consumer._consume_loop()
            # Add a longer delay to ensure all operations complete
            time.sleep(0.2)

            # Verify the consume loop ended due to shutdown
            assert consumer._running is False
            assert consumer._shutdown_event.is_set() is True
    
    @patch('app.consumers.base.consumer.Consumer')
    def test_get_consumer_metrics(self, mock_consumer_class, consumer_settings):
        """Test retrieving consumer metrics."""
        # Setup mock consumer
        mock_consumer_instance = MagicMock()
        mock_consumer_class.return_value = mock_consumer_instance
        
        # Create consumer with some stats
        consumer = ConcreteConsumer(
            topic="test-topic",
            group_id="test-group",
            settings=consumer_settings
        )
        
        # Set some stats
        consumer.stats.messages_processed = 10
        consumer.stats.messages_failed = 2
        consumer.stats.avg_processing_time_ms = 15.5
        consumer.stats.bytes_processed = 1024
        consumer.stats.last_message_timestamp = time.time()
        
        # Get metrics
        metrics = consumer.get_consumer_metrics()
        
        # Verify metrics
        assert metrics["topic"] == "test-topic"
        assert metrics["group_id"] == "test-group"
        assert metrics["messages_processed"] == 10
        assert metrics["messages_failed"] == 2
        assert metrics["avg_processing_time_ms"] == 15.5
        assert metrics["bytes_processed"] == 1024
        assert metrics["last_message_timestamp"] is not None
        assert metrics["running"] is False
    
    @patch('app.consumers.base.consumer.Consumer')
    def test_maybe_log_stats(self, mock_consumer_class, consumer_settings):
        """Test periodic logging of stats."""
        # Setup mock consumer
        mock_consumer_instance = MagicMock()
        mock_consumer_class.return_value = mock_consumer_instance
        
        # Create consumer with short stats interval
        consumer = ConcreteConsumer(
            topic="test-topic",
            group_id="test-group",
            settings=consumer_settings,
            stats_interval=1  # Log stats every second
        )
        
        # Set stats
        consumer.stats.messages_processed = 5
        consumer.stats.messages_failed = 1
        consumer.stats.avg_processing_time_ms = 10.0
        
        # Reset last stats time to force logging
        consumer._last_stats_time = 0
        
        # Mock the logger
        with patch.object(consumer, '_logger') as mock_logger:
            # Call the log stats method
            consumer._maybe_log_stats()
            
            # Verify logger was called with stats info
            mock_logger.info.assert_called_once()
            log_message = mock_logger.info.call_args[0][0]
            assert "processed=5" in log_message
            assert "failed=1" in log_message
            assert "avg_time=10.00ms" in log_message
            
            # Update last stats time to recent
            consumer._last_stats_time = time.time()
            mock_logger.reset_mock()
            
            # Call again, should not log due to interval
            consumer._maybe_log_stats()
            mock_logger.info.assert_not_called()


if __name__ == "__main__":
    pytest.main()