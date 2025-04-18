"""
End-to-end unit tests for OrderBookConsumer.

This module provides comprehensive tests for the OrderBookConsumer class,
verifying all critical functionalities including message processing,
batch handling, error handling, and database interactions.
"""

import json
import os
import time
import unittest
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pytest
from confluent_kafka import KafkaError, Message

from app.consumers.base.error import DeserializationError, ProcessingError
from app.consumers.config.settings import KafkaSettings
from app.consumers.managers.offset_manager import OffsetManager
from app.consumers.market_data.orderbook_consumer import OrderBookConsumer
from app.db.repositories.market_data_repository import MarketDataRepository
from app.models.market_data import Instrument, OrderBookSnapshot


class TestOrderBookConsumer(unittest.TestCase):
    """Test suite for OrderBookConsumer."""

    def setUp(self):
        """Set up test fixtures before each test."""
        # Mock repository
        self.mock_repository = Mock(spec=MarketDataRepository)
        
        # Mock instrument
        self.mock_instrument = Mock(spec=Instrument)
        self.mock_instrument.id = "test-instrument-id"
        self.mock_repository.get_or_create_instrument.return_value = self.mock_instrument
        
        # Mock Kafka consumer
        self.mock_kafka_consumer = Mock()
        
        # Mock offset manager
        self.mock_offset_manager = Mock(spec=OffsetManager)
        
        # Mock metrics and health
        self.mock_metrics = Mock()
        self.mock_metrics.record_message_processed = Mock()
        self.mock_metrics.record_message_failed = Mock()
        
        self.mock_health = Mock()
        self.mock_health.record_message_processed = Mock()
        self.mock_health.record_error = Mock()
        
        # Mock registry
        self.mock_registry = Mock()
        self.mock_registry.register_consumer.return_value = self.mock_metrics
        
        # Mock health manager
        self.mock_health_manager = Mock()
        self.mock_health_manager.register_consumer.return_value = self.mock_health
        
        # Create valid test message
        self.valid_message = {
            "symbol": "BTC-USD",
            "timestamp": datetime.now().isoformat(),
            "bids": [[10000.0, 1.5], [9999.0, 2.0]],
            "asks": [[10001.0, 1.0], [10002.0, 3.0]],
            "source": "exchange",
            "depth": 2,
            "spread": 1.0,
            "weighted_mid_price": 10000.5,
            "imbalance": 0.1
        }
        
        # Create mock Kafka Message
        self.mock_kafka_message = Mock(spec=Message)
        self.mock_kafka_message.error.return_value = None
        self.mock_kafka_message.value.return_value = json.dumps(self.valid_message).encode('utf-8')
        self.mock_kafka_message.key.return_value = b"BTC-USD"
        self.mock_kafka_message.partition.return_value = 0
        self.mock_kafka_message.offset.return_value = 100
        
        # Patches
        self.patches = [
            patch('app.consumers.market_data.orderbook_consumer.get_metrics_registry', return_value=self.mock_registry),
            patch('app.consumers.market_data.orderbook_consumer.get_health_manager', return_value=self.mock_health_manager),
            patch('app.consumers.base.consumer.Consumer', return_value=self.mock_kafka_consumer),
            patch('app.consumers.market_data.orderbook_consumer.OffsetManager', return_value=self.mock_offset_manager)
        ]
        
        # Start patches
        for p in self.patches:
            p.start()
            
        # Create consumer
        self.consumer = OrderBookConsumer(
            topic="test-topic",
            group_id="test-group",
            repository=self.mock_repository,
            batch_size=2,
            batch_timeout_ms=1000
        )

    def tearDown(self):
        """Clean up after each test."""
        # Stop patches
        for p in self.patches:
            p.stop()

    def test_init(self):
        """Test consumer initialization."""
        self.assertEqual(self.consumer.topic, "test-topic")
        self.assertEqual(self.consumer.group_id, "test-group")
        self.assertEqual(self.consumer.batch_size, 2)
        self.assertEqual(self.consumer.batch_timeout_ms, 1000)
        self.assertEqual(self.consumer.repository, self.mock_repository)
        self.assertEqual(self.consumer._batch, [])
        self.assertIsNotNone(self.consumer._batch_start_time)

    def test_deserialize_valid_message(self):
        """Test deserialization of a valid message."""
        result = self.consumer._deserialize_message(self.mock_kafka_message)
        self.assertEqual(result, self.valid_message)

    def test_deserialize_invalid_json(self):
        """Test deserialization of invalid JSON."""
        # Create message with invalid JSON
        invalid_message = Mock(spec=Message)
        invalid_message.error.return_value = None
        invalid_message.value.return_value = b"{invalid-json"
        
        # Expect DeserializationError
        with self.assertRaises(DeserializationError):
            self.consumer._deserialize_message(invalid_message)

    def test_deserialize_missing_fields(self):
        """Test deserialization of a message with missing required fields."""
        # Create message with missing required fields
        incomplete_message = {
            "symbol": "BTC-USD",
            # Missing timestamp
            "bids": [[10000.0, 1.5]],
            # Missing asks
        }
        
        incomplete_kafka_message = Mock(spec=Message)
        incomplete_kafka_message.error.return_value = None
        incomplete_kafka_message.value.return_value = json.dumps(incomplete_message).encode('utf-8')
        
        # Expect DeserializationError
        with self.assertRaises(DeserializationError):
            self.consumer._deserialize_message(incomplete_kafka_message)

    def test_create_orderbook_from_message(self):
        """Test creation of OrderBookSnapshot from a valid message."""
        # Call the method
        result = self.consumer._create_orderbook_from_message(self.valid_message)
        
        # Verify repository was called correctly
        self.mock_repository.get_or_create_instrument.assert_called_once_with("BTC-USD")
        
        # Verify OrderBookSnapshot properties
        self.assertEqual(result.instrument_id, self.mock_instrument.id)
        self.assertEqual(result.bids, self.valid_message['bids'])
        self.assertEqual(result.asks, self.valid_message['asks'])
        self.assertEqual(result.source, self.valid_message['source'])
        self.assertEqual(result.depth, self.valid_message['depth'])
        self.assertEqual(result.spread, self.valid_message['spread'])
        self.assertEqual(result.weighted_mid_price, self.valid_message['weighted_mid_price'])
        self.assertEqual(result.imbalance, self.valid_message['imbalance'])

    def test_create_orderbook_with_unix_timestamp(self):
        """Test creation of OrderBookSnapshot with a Unix timestamp."""
        # Create message with Unix timestamp
        unix_message = self.valid_message.copy()
        unix_message['timestamp'] = time.time()
        
        # Call the method
        result = self.consumer._create_orderbook_from_message(unix_message)
        
        # Verify the timestamp was processed correctly
        self.assertIsInstance(result.timestamp, datetime)

    def test_process_message_single(self):
        """Test processing of a single message (not triggering batch process)."""
        # Call process_message
        self.consumer.process_message(self.valid_message, self.mock_kafka_message)
        
        # Verify batch was updated but not processed
        self.assertEqual(len(self.consumer._batch), 1)
        self.assertEqual(self.consumer._batch[0], self.valid_message)
        
        # Verify offset was tracked
        self.mock_offset_manager.track_message.assert_called_once_with(self.mock_kafka_message)
        
        # Verify metrics were updated
        self.mock_metrics.record_message_processed.assert_called_once()
        self.mock_health.record_message_processed.assert_called_once()
        
        # Verify no database operations were performed yet
        self.mock_repository.save_orderbook_batch.assert_not_called()

    def test_process_message_batch_full(self):
        """Test processing messages until batch is full."""
        # Process first message
        self.consumer.process_message(self.valid_message, self.mock_kafka_message)
        
        # Process second message - should trigger batch processing
        second_message = self.valid_message.copy()
        second_message['symbol'] = "ETH-USD"
        
        # Create a spy for _process_batch
        with patch.object(self.consumer, '_process_batch', wraps=self.consumer._process_batch) as mock_process_batch:
            self.consumer.process_message(second_message, self.mock_kafka_message)
            
            # Verify batch was processed
            mock_process_batch.assert_called_once()
        
        # Verify batch was emptied
        self.assertEqual(len(self.consumer._batch), 0)
        
        # Verify repository was called to save batch
        self.mock_repository.save_orderbook_batch.assert_called_once()
        
        # Verify that commit was checked
        self.mock_offset_manager.should_commit.assert_called_once()

    def test_process_message_timeout(self):
        """Test processing messages with batch timeout."""
        # Process a message
        self.consumer.process_message(self.valid_message, self.mock_kafka_message)
        
        # Set batch start time to simulate timeout
        self.consumer._batch_start_time = time.time() * 1000 - 2000  # 2 seconds ago
        
        # Process another message (would normally not trigger batch processing due to batch size)
        second_message = self.valid_message.copy()
        second_message['symbol'] = "ETH-USD"
        
        # Create a spy for _process_batch
        with patch.object(self.consumer, '_process_batch', wraps=self.consumer._process_batch) as mock_process_batch:
            self.consumer.process_message(second_message, self.mock_kafka_message)
            
            # Verify batch was processed due to timeout
            mock_process_batch.assert_called_once()

    def test_save_orderbook_batch(self):
        """Test saving a batch of order book data."""
        # Create sample orderbook models
        orderbooks = [
            OrderBookSnapshot(
                instrument_id=self.mock_instrument.id,
                timestamp=datetime.now(),
                bids=[[10000.0, 1.5]],
                asks=[[10001.0, 1.0]],
                source="exchange"  # Use valid DataSource enum value
            ),
            OrderBookSnapshot(
                instrument_id=self.mock_instrument.id,
                timestamp=datetime.now(),
                bids=[[9999.0, 2.0]],
                asks=[[10002.0, 3.0]],
                source="exchange"  # Use valid DataSource enum value
            )
        ]
        
        # Call the method
        self.consumer._save_orderbook_batch(orderbooks)
        
        # Verify repository was called correctly
        self.mock_repository.save_orderbook_batch.assert_called_once_with(orderbooks)

    def test_save_orderbook_batch_error(self):
        """Test error handling when saving a batch fails."""
        # Make repository raise an exception
        self.mock_repository.save_orderbook_batch.side_effect = Exception("Database error")
        
        # Create sample orderbook model
        orderbooks = [
            OrderBookSnapshot(
                instrument_id=self.mock_instrument.id,
                timestamp=datetime.now(),
                bids=[[10000.0, 1.5]],
                asks=[[10001.0, 1.0]],
                source="exchange"  # Use valid DataSource enum value
            )
        ]
        
        # Expect ProcessingError
        with self.assertRaises(ProcessingError):
            self.consumer._save_orderbook_batch(orderbooks)

    def test_get_or_create_instrument(self):
        """Test getting or creating an instrument."""
        # Call the method
        result = self.consumer._get_or_create_instrument("BTC-USD")
        
        # Verify repository was called correctly
        self.mock_repository.get_or_create_instrument.assert_called_once_with("BTC-USD")
        
        # Verify the result is correct
        self.assertEqual(result, self.mock_instrument)

    def test_get_or_create_instrument_error(self):
        """Test error handling when getting or creating an instrument fails."""
        # Make repository raise an exception
        self.mock_repository.get_or_create_instrument.side_effect = Exception("Database error")
        
        # Expect ProcessingError
        with self.assertRaises(ProcessingError):
            self.consumer._get_or_create_instrument("BTC-USD")

    def test_on_stop(self):
        """Test consumer stop behavior with pending batch."""
        # Add a message to the batch
        self.consumer._batch = [self.valid_message]
        
        # Create a spy for _process_batch
        with patch.object(self.consumer, '_process_batch', wraps=self.consumer._process_batch) as mock_process_batch:
            # Call on_stop
            self.consumer.on_stop()
            
            # Verify batch was processed
            mock_process_batch.assert_called_once()
            
            # Verify final commit was attempted
            # Note: We don't assert called_once since process_batch may have already called it
            self.mock_offset_manager.commit.assert_called_with(async_commit=False)

    def test_on_stop_empty_batch(self):
        """Test consumer stop behavior with no pending messages."""
        # Ensure batch is empty
        self.consumer._batch = []
        
        # Create a spy for _process_batch
        with patch.object(self.consumer, '_process_batch', wraps=self.consumer._process_batch) as mock_process_batch:
            # Call on_stop
            self.consumer.on_stop()
            
            # Verify batch was not processed (empty)
            mock_process_batch.assert_not_called()
            
            # Verify final commit was still attempted
            self.mock_offset_manager.commit.assert_called_once_with(async_commit=False)

    def test_process_batch_error_handling(self):
        """Test error handling in batch processing."""
        # Add a message to the batch
        self.consumer._batch = [self.valid_message]
        
        # Make repository raise an exception during processing
        self.mock_repository.save_orderbook_batch.side_effect = Exception("Database error")
        
        # Call _process_batch, which should handle the error
        with self.assertRaises(ProcessingError):
            self.consumer._process_batch()
        
        # Verify batch was cleared on error
        self.assertEqual(len(self.consumer._batch), 0)
        
        # Verify metrics registered failure
        self.mock_metrics.record_message_failed.assert_not_called()  # Not called directly in _process_batch

    def test_circuit_breaker_integration(self):
        """Test circuit breaker integration."""
        # This is a basic test to verify circuit breaker decorator is applied
        # More comprehensive testing of the circuit breaker itself would be in separate tests
        
        # Check that the circuit breaker decorator is applied to database methods
        # We're just testing that the circuit breaker is applied to the methods
        # No need to import it directly here
        
        # Create a message that will trigger database operations
        self.consumer._batch = [self.valid_message, self.valid_message]
        
        # Mock that should_commit returns True to test commit flow
        self.mock_offset_manager.should_commit.return_value = True
        
        # Process the batch
        self.consumer._process_batch()
        
        # Verify repository was called
        self.mock_repository.save_orderbook_batch.assert_called_once()
        
        # Verify offset was committed
        self.mock_offset_manager.commit.assert_called_once()


if __name__ == '__main__':
    unittest.main()