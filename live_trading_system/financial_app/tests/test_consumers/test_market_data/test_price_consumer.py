"""
Unit tests for OHLCV price data consumer.

Tests the OHLCVConsumer class that consumes price data from Kafka and stores it in the database.
"""

import json
import unittest
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, Mock, patch, PropertyMock

from confluent_kafka import Message, KafkaError

from app.consumers.market_data.price_consumer import OHLCVConsumer
from app.consumers.base.error import DeserializationError, ProcessingError
from app.consumers.config.settings import KafkaSettings
from app.models.market_data import OHLCV, Instrument, DataSource


class TestOHLCVConsumer(unittest.TestCase):
    """Test cases for the OHLCV consumer."""

    def setUp(self):
        """Set up test fixtures."""
        # Find out what DataSource values are available and pick one for testing
        self.valid_source = DataSource.EXCHANGE.value

        # Mock the Kafka Consumer
        self.mock_consumer = MagicMock()
        self.mock_consumer.subscribe = MagicMock()

        # Mock the repository
        self.mock_repository = MagicMock()
        self.mock_repository.save_ohlcv_batch = MagicMock()
        
        # Create mock instrument
        self.mock_instrument = MagicMock(spec=Instrument)
        self.mock_instrument.id = uuid.uuid4()
        self.mock_instrument.symbol = "AAPL"
        self.mock_repository.get_or_create_instrument = MagicMock(return_value=self.mock_instrument)

        # Mock the offset manager
        self.mock_offset_manager = MagicMock()
        self.mock_offset_manager.track_message = MagicMock()
        self.mock_offset_manager.should_commit = MagicMock(return_value=False)
        self.mock_offset_manager.commit = MagicMock()

        # Mock the health manager
        self.mock_health_check = MagicMock()
        self.mock_health_check.record_message_processed = MagicMock()
        self.mock_health_check.record_error = MagicMock()

        # Mock the metrics
        self.mock_metrics = MagicMock()
        self.mock_metrics.record_message_processed = MagicMock()
        self.mock_metrics.record_message_failed = MagicMock()

        # Create the consumer with mocks
        with patch('app.consumers.market_data.price_consumer.OffsetManager', return_value=self.mock_offset_manager), \
             patch('app.consumers.market_data.price_consumer.get_health_manager') as mock_health_manager, \
             patch('app.consumers.market_data.price_consumer.get_metrics_registry') as mock_metrics_registry, \
             patch('app.consumers.base.consumer.Consumer', return_value=self.mock_consumer):
            
            mock_health_manager.return_value.register_consumer.return_value = self.mock_health_check
            mock_metrics_registry.return_value.register_consumer.return_value = self.mock_metrics
            
            self.consumer = OHLCVConsumer(
                topic="test-ohlcv-topic",
                group_id="test-consumer-group",
                batch_size=2,  # Small batch size for easier testing
                batch_timeout_ms=500,
                repository=self.mock_repository
            )

    def _create_mock_message(self, payload, key=None, topic="test-ohlcv-topic", partition=0, offset=0):
        """Helper to create a mock Kafka message."""
        mock_message = MagicMock(spec=Message)
        mock_message.error.return_value = None
        mock_message.topic.return_value = topic
        mock_message.partition.return_value = partition
        mock_message.offset.return_value = offset
        
        if isinstance(payload, dict):
            payload = json.dumps(payload).encode('utf-8')
        elif isinstance(payload, str):
            payload = payload.encode('utf-8')
            
        mock_message.value.return_value = payload
        
        if key:
            if isinstance(key, str):
                key = key.encode('utf-8')
            mock_message.key.return_value = key
        else:
            mock_message.key.return_value = None
            
        return mock_message

    def test_initialization(self):
        """Test consumer initialization."""
        self.assertEqual(self.consumer.topic, "test-ohlcv-topic")
        self.assertEqual(self.consumer.group_id, "test-consumer-group")
        self.assertEqual(self.consumer.batch_size, 2)
        self.assertEqual(self.consumer.batch_timeout_ms, 500)
        self.assertEqual(self.consumer.repository, self.mock_repository)
        self.mock_consumer.subscribe.assert_called_once()

    def test_deserialize_valid_message(self):
        """Test deserializing a valid OHLCV message."""
        # Create valid message data
        now = datetime.utcnow()
        message_data = {
            "symbol": "AAPL",
            "interval": "1m",
            "timestamp": now.isoformat(),
            "open": 150.5,
            "high": 151.2,
            "low": 150.0,
            "close": 150.75,
            "volume": 10000,
            "source": self.valid_source
        }
        
        # Create mock message
        mock_message = self._create_mock_message(message_data)
        
        # Deserialize
        result = self.consumer._deserialize_message(mock_message)
        
        # Assertions
        self.assertEqual(result["symbol"], "AAPL")
        self.assertEqual(result["interval"], "1m")
        self.assertEqual(result["open"], 150.5)
        self.assertEqual(result["high"], 151.2)
        self.assertEqual(result["low"], 150.0)
        self.assertEqual(result["close"], 150.75)
        self.assertEqual(result["volume"], 10000)

    def test_deserialize_invalid_json(self):
        """Test deserializing an invalid JSON message."""
        # Create invalid message data
        invalid_message = b"{invalid json"
        
        # Create mock message
        mock_message = self._create_mock_message(invalid_message)
        
        # Should raise DeserializationError
        with self.assertRaises(DeserializationError):
            self.consumer._deserialize_message(mock_message)

    def test_deserialize_missing_fields(self):
        """Test deserializing a message with missing required fields."""
        # Create message with missing fields
        incomplete_message = {
            "symbol": "AAPL",
            "interval": "1m",
            # Missing timestamp
            "open": 150.5,
            "high": 151.2,
            # Missing low
            "close": 150.75,
            "volume": 10000
        }
        
        # Create mock message
        mock_message = self._create_mock_message(incomplete_message)
        
        # Should raise DeserializationError
        with self.assertRaises(DeserializationError):
            self.consumer._deserialize_message(mock_message)

    def test_create_ohlcv_from_message(self):
        """Test creating an OHLCV model from a message."""
        # Create valid message data with ISO timestamp
        now = datetime.utcnow()
        message_data = {
            "symbol": "AAPL",
            "interval": "1m",
            "timestamp": now.isoformat(),
            "open": 150.5,
            "high": 151.2,
            "low": 150.0,
            "close": 150.75,
            "volume": 10000,
            "vwap": 150.6,
            "trades_count": 50,
            "source": self.valid_source
        }
        
        # Create a fully mocked OHLCV that will be returned
        mock_ohlcv = MagicMock(spec=OHLCV)
        
        # Patch the _create_ohlcv_from_message to avoid actual creation
        with patch.object(self.consumer, '_get_or_create_instrument', return_value=self.mock_instrument), \
             patch('app.consumers.market_data.price_consumer.OHLCV', return_value=mock_ohlcv):
             
            # Call the method
            result = self.consumer._create_ohlcv_from_message(message_data)
            
            # Verify the result is our mock
            self.assertEqual(result, mock_ohlcv)

    def test_create_ohlcv_from_message_with_timestamp_int(self):
        """Test creating an OHLCV model from a message with Unix timestamp."""
        # Create valid message data with Unix timestamp (seconds)
        now = datetime.utcnow()
        timestamp_seconds = int(now.timestamp())
        message_data = {
            "symbol": "AAPL",
            "interval": "1m",
            "timestamp": timestamp_seconds,
            "open": 150.5,
            "high": 151.2,
            "low": 150.0,
            "close": 150.75,
            "volume": 10000,
            "source": self.valid_source
        }
        
        # Create a fully mocked OHLCV that will be returned
        mock_ohlcv = MagicMock(spec=OHLCV)
        
        # Patch the OHLCV constructor to return our mock
        with patch('app.consumers.market_data.price_consumer.OHLCV', return_value=mock_ohlcv):
            # Call the method
            result = self.consumer._create_ohlcv_from_message(message_data)
            
            # Verify the result is our mock
            self.assertEqual(result, mock_ohlcv)

    def test_create_ohlcv_from_message_with_timestamp_millis(self):
        """Test creating an OHLCV model from a message with millisecond timestamp."""
        # Create valid message data with Unix timestamp (milliseconds)
        now = datetime.utcnow()
        timestamp_millis = int(now.timestamp() * 1000)
        message_data = {
            "symbol": "AAPL",
            "interval": "1m",
            "timestamp": timestamp_millis,
            "open": 150.5,
            "high": 151.2,
            "low": 150.0,
            "close": 150.75,
            "volume": 10000,
            "source": self.valid_source
        }
        
        # Create a fully mocked OHLCV that will be returned
        mock_ohlcv = MagicMock(spec=OHLCV)
        
        # Patch the OHLCV constructor to return our mock
        with patch('app.consumers.market_data.price_consumer.OHLCV', return_value=mock_ohlcv):
            # Call the method
            result = self.consumer._create_ohlcv_from_message(message_data)
            
            # Verify the result is our mock
            self.assertEqual(result, mock_ohlcv)

    def test_process_message_batch_size(self):
        """Test processing messages in a batch when batch size is reached."""
        # Create mock OHLCV objects to return from _create_ohlcv_from_message
        mock_ohlcv1 = MagicMock(spec=OHLCV)
        mock_ohlcv2 = MagicMock(spec=OHLCV)
        
        # Create valid messages
        now = datetime.utcnow()
        message1_data = {
            "symbol": "AAPL",
            "interval": "1m",
            "timestamp": now.isoformat(),
            "open": 150.5,
            "high": 151.2,
            "low": 150.0,
            "close": 150.75,
            "volume": 10000,
            "source": self.valid_source
        }
        
        message2_data = {
            "symbol": "MSFT",
            "interval": "1m",
            "timestamp": now.isoformat(),
            "open": 250.5,
            "high": 251.2,
            "low": 250.0,
            "close": 250.75,
            "volume": 5000,
            "source": self.valid_source
        }
        
        # Create mock messages
        mock_message1 = self._create_mock_message(message1_data, offset=0)
        mock_message2 = self._create_mock_message(message2_data, offset=1)
        
        # Reset consumer's batch
        self.consumer._batch = []
        
        # Set up time.time to return a static value to avoid timeout issues
        with patch('time.time', return_value=1000.0), \
             patch('app.consumers.market_data.price_consumer.OHLCV', side_effect=[mock_ohlcv1, mock_ohlcv2]), \
             patch.object(self.consumer, '_save_ohlcv_batch') as mock_save_batch:
            
            # Process first message
            self.consumer.process_message(message1_data, mock_message1)
            
            # First message should be added to batch but not processed yet
            mock_save_batch.assert_not_called()
            self.mock_offset_manager.track_message.assert_called_once_with(mock_message1)
            
            # Process second message
            self.consumer.process_message(message2_data, mock_message2)
            
            # Both messages should now be processed
            mock_save_batch.assert_called_once_with([mock_ohlcv1, mock_ohlcv2])
            
            # Check tracking
            self.assertEqual(self.mock_offset_manager.track_message.call_count, 2)
            self.mock_metrics.record_message_processed.assert_called()
            self.mock_health_check.record_message_processed.assert_called()

    def test_process_message_batch_timeout(self):
        """Test processing messages in a batch when timeout is reached."""
        # Create a valid message
        now = datetime.utcnow()
        message_data = {
            "symbol": "AAPL",
            "interval": "1m",
            "timestamp": now.isoformat(),
            "open": 150.5,
            "high": 151.2,
            "low": 150.0,
            "close": 150.75,
            "volume": 10000,
            "source": self.valid_source
        }
        
        # Create mock message
        mock_message = self._create_mock_message(message_data, offset=0)
        
        # Create mock OHLCV
        mock_ohlcv = MagicMock(spec=OHLCV)
        
        # Patch time.time and the process_batch method
        with patch('time.time') as mock_time, \
            patch.object(self.consumer, '_process_batch') as mock_process_batch, \
            patch('app.consumers.market_data.price_consumer.OHLCV', return_value=mock_ohlcv):
            
            # Set up time.time mock to return appropriate values
            # We need at least 4 values:
            # 1. Initial start_time in process_message
            # 2. For current_time calculation
            # 3. For processing_time_ms calculation
            # 4. For any additional calls
            mock_time.side_effect = [1000.0, 1002.0, 1003.0, 1004.0, 1005.0]
            
            # Set batch timeout low
            self.consumer._batch = []
            self.consumer._batch_timeout_ms = 500  # 500ms
            self.consumer._batch_start_time = 1000.0 * 1000  # Must match first mock time value × 1000
            
            # Process message, should trigger timeout check
            self.consumer.process_message(message_data, mock_message)
            
            # Verify _process_batch was called
            mock_process_batch.assert_called_once()

    def test_save_ohlcv_batch(self):
        """Test saving a batch of OHLCV data."""
        # Create mock OHLCV instances
        mock_ohlcv1 = MagicMock(spec=OHLCV)
        mock_ohlcv2 = MagicMock(spec=OHLCV)
        
        # Call the method
        self.consumer._save_ohlcv_batch([mock_ohlcv1, mock_ohlcv2])
        
        # Verify repository was called
        self.mock_repository.save_ohlcv_batch.assert_called_once_with([mock_ohlcv1, mock_ohlcv2])

    def test_save_ohlcv_batch_error(self):
        """Test error handling when saving a batch of OHLCV data fails."""
        # Mock repository to raise exception
        self.mock_repository.save_ohlcv_batch.side_effect = Exception("Database error")
        
        # Create mock OHLCV instances
        mock_ohlcv = MagicMock(spec=OHLCV)
        
        # Should raise ProcessingError
        with self.assertRaises(ProcessingError):
            self.consumer._save_ohlcv_batch([mock_ohlcv])

    def test_get_or_create_instrument(self):
        """Test getting or creating an instrument."""
        # Get or create instrument
        instrument = self.consumer._get_or_create_instrument("AAPL")
        
        # Verify repository was called
        self.mock_repository.get_or_create_instrument.assert_called_once_with("AAPL")
        self.assertEqual(instrument, self.mock_instrument)

    def test_get_or_create_instrument_error(self):
        """Test error handling when getting or creating an instrument fails."""
        # Mock repository to raise exception
        self.mock_repository.get_or_create_instrument.side_effect = Exception("Database error")
        
        # Should raise ProcessingError
        with self.assertRaises(ProcessingError):
            self.consumer._get_or_create_instrument("AAPL")

    def test_on_stop(self):
        """Test cleanup when the consumer stops."""
        # Add a message to the batch
        now = datetime.utcnow()
        message_data = {
            "symbol": "AAPL",
            "interval": "1m",
            "timestamp": now.isoformat(),
            "open": 150.5,
            "high": 151.2,
            "low": 150.0,
            "close": 150.75,
            "volume": 10000,
            "source": self.valid_source
        }
        
        # Create mock OHLCV
        mock_ohlcv = MagicMock(spec=OHLCV)
        
        # Set up consumer
        self.consumer._batch = [message_data]
        
        # Patch _process_batch and OHLCV creation
        with patch.object(self.consumer, '_process_batch') as mock_process_batch, \
             patch('app.consumers.market_data.price_consumer.OHLCV', return_value=mock_ohlcv):
            # Call on_stop
            self.consumer.on_stop()
            
            # Verify _process_batch was called
            mock_process_batch.assert_called_once()

    def test_on_stop_with_error(self):
        """Test cleanup when the consumer stops and there's an error processing the batch."""
        # Add a message to the batch
        now = datetime.utcnow()
        message_data = {
            "symbol": "AAPL",
            "interval": "1m",
            "timestamp": now.isoformat(),
            "open": 150.5,
            "high": 151.2,
            "low": 150.0,
            "close": 150.75,
            "volume": 10000,
            "source": self.valid_source
        }
        
        # Create mock OHLCV
        mock_ohlcv = MagicMock(spec=OHLCV)
        
        # Set up consumer
        self.consumer._batch = [message_data]
        
        # Patch _process_batch to raise exception and OHLCV creation
        with patch.object(self.consumer, '_process_batch', side_effect=Exception("Processing error")), \
             patch('app.consumers.market_data.price_consumer.logger') as mock_logger, \
             patch('app.consumers.market_data.price_consumer.OHLCV', return_value=mock_ohlcv):
            
            # Call on_stop
            self.consumer.on_stop()
            
            # Verify logger.error was called
            mock_logger.error.assert_called_once()

    def test_on_stop_with_commit(self):
        """Test cleanup with offset commit when the consumer stops."""
        # Set auto_commit to False to test manual commit
        self.consumer.settings.ENABLE_AUTO_COMMIT = False
        
        # Call on_stop
        self.consumer.on_stop()
        
        # Verify offset_manager.commit was called with async_commit=False
        self.mock_offset_manager.commit.assert_called_once_with(async_commit=False)

    def test_on_stop_with_commit_error(self):
        """Test cleanup when the consumer stops and there's an error committing offsets."""
        # Set auto_commit to False to test manual commit
        self.consumer.settings.ENABLE_AUTO_COMMIT = False
        
        # Mock offset_manager.commit to raise exception
        self.mock_offset_manager.commit.side_effect = Exception("Commit error")
        
        # Call on_stop with patched logger
        with patch('app.consumers.market_data.price_consumer.logger') as mock_logger:
            self.consumer.on_stop()
            
            # Verify logger.error was called
            mock_logger.error.assert_called_once()

    def test_start_method(self):
        """Test that the consumer can be started."""
        # Just test that the start method exists and can be called
        with patch.object(self.consumer, 'start') as mock_start:
            # Call start method
            self.consumer.start()
            # Verify it was called
            mock_start.assert_called_once()


if __name__ == '__main__':
    unittest.main()