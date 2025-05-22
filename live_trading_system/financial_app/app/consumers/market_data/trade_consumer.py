"""
Trade data consumer for market data ingestion.

This module provides a Kafka consumer for ingesting individual trade data
from Kafka topics and storing it in the database.
"""

import logging
import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from confluent_kafka import Message, Consumer

from app.consumers.base.consumer import BaseConsumer
from app.consumers.base.error import DeserializationError, ProcessingError, ValidationError
from app.consumers.base.metrics import get_metrics_registry
from app.consumers.config.settings import KafkaSettings
from app.consumers.managers.offset_manager import OffsetManager
from app.consumers.managers.health_manager import get_health_manager, HealthCheckConfig
from app.consumers.utils.serialization import deserialize_json
from app.consumers.utils.validation import validate_trade_message
from app.consumers.utils.circuit_breaker import circuit_breaker

from app.models.market_data import Tick, Instrument
from app.db.repositories.market_data_repository import MarketDataRepository

# Set up logging
logger = logging.getLogger(__name__)


class TradeConsumer(BaseConsumer):
    """
    Kafka consumer for individual trade data.
    
    Consumes trade data from a Kafka topic and stores it in the database
    using the Tick model.
    """
    
    def __init__(
        self,
        topic: Optional[str] = None,
        group_id: Optional[str] = None,
        settings: Optional[KafkaSettings] = None,
        batch_size: int = 100,
        batch_timeout_ms: int = 1000,
        repository: Optional[MarketDataRepository] = None,
    ):
        """
        Initialize a new trade consumer.
        
        Args:
            topic: Kafka topic to consume (defaults to TRADE_TOPIC from settings)
            group_id: Consumer group ID (defaults to TRADE_GROUP_ID from settings)
            settings: Kafka configuration settings
            batch_size: Number of messages to process in a batch
            batch_timeout_ms: Maximum time to wait for a full batch in ms
            repository: Market data repository for database operations
        """
        self.settings = settings or KafkaSettings()
        self.topic = topic or self.settings.TRADE_TOPIC
        self.group_id = group_id or self.settings.TRADE_GROUP_ID
        
        # Initialize base consumer
        super().__init__(
            topic=self.topic,
            group_id=self.group_id,
            settings=self.settings,
            auto_commit=self.settings.ENABLE_AUTO_COMMIT
        )
        
        # Batch processing
        self.batch_size = batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self._batch: List[Dict[str, Any]] = []
        self._batch_start_time = time.time() * 1000
        
        # Repository for database operations
        self.repository = repository or MarketDataRepository()
        
        # Set up offset manager
        self.offset_manager = OffsetManager(
            consumer=self._consumer,
            auto_commit=self.settings.ENABLE_AUTO_COMMIT,
            commit_interval_ms=self.settings.COMMIT_INTERVAL_MS,
            commit_threshold=self.batch_size
        )
        
        # Set up metrics
        self.metrics = get_metrics_registry().register_consumer(
            consumer_id=f"trade_{self.group_id}",
            topic=self.topic,
            group_id=self.group_id
        )
        
        # Register with health manager
        self.health_check = get_health_manager().register_consumer(
            consumer_id=f"trade_{self.group_id}",
            consumer=self._consumer,
            offset_manager=self.offset_manager,
            config=HealthCheckConfig(
                max_lag_messages=100000,  # Higher for trades which can be high volume
                critical_lag_messages=500000
            )
        )
        
        logger.info(f"Initialized trade consumer for topic '{self.topic}' with group ID '{self.group_id}'")
    
    def _deserialize_message(self, msg: Message) -> Dict[str, Any]:
        """
        Deserialize a Kafka message containing trade data.
        
        Args:
            msg: Kafka message
            
        Returns:
            Deserialized trade data
            
        Raises:
            DeserializationError: If the message cannot be deserialized
        """
        try:
            # Deserialize using JSON by default
            data = deserialize_json(msg)
            
            # Validate the message structure
            validate_trade_message(data)
            
            return data
        except json.JSONDecodeError as e:
            raise DeserializationError(f"Invalid JSON in trade message: {e}")
        except ValidationError as e:
            raise DeserializationError(f"Invalid trade message: {e}")
        except Exception as e:
            raise DeserializationError(f"Failed to deserialize trade message: {e}")
    
    def process_message(self, message: Dict[str, Any], raw_message: Message) -> None:
        """
        Process a deserialized trade message.
        
        Args:
            message: Deserialized message content
            raw_message: Original Kafka message
            
        Raises:
            ProcessingError: If the message cannot be processed
        """
        start_time = time.time()
        
        try:
            # Add message to batch
            self._batch.append(message)
            
            # Track offset for manual commit
            self.offset_manager.track_message(raw_message)
            
            # Process batch if full or timeout reached
            current_time = time.time() * 1000
            if (len(self._batch) >= self.batch_size or 
                current_time - self._batch_start_time >= self.batch_timeout_ms):
                self._process_batch()
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Update metrics
            self.metrics.record_message_processed(processing_time_ms)
            self.health_check.record_message_processed(processing_time_ms)
            
        except Exception as e:
            self.metrics.record_message_failed()
            self.health_check.record_error()
            raise ProcessingError(f"Failed to process trade message: {e}")
    
    def _process_batch(self) -> None:
        """
        Process a batch of trade messages.
        
        Raises:
            ProcessingError: If the batch cannot be processed
        """
        if not self._batch:
            return
            
        try:
            # Convert messages to Tick models
            tick_data = []
            for message in self._batch:
                # Create Tick model
                tick = self._create_tick_from_message(message)
                tick_data.append(tick)
            
            # Save to database
            self._save_tick_batch(tick_data)
            
            # Commit offsets if needed
            if self.offset_manager.should_commit():
                self.offset_manager.commit()
            
            # Reset batch
            batch_size = len(self._batch)
            self._batch = []
            self._batch_start_time = time.time() * 1000
            
            logger.debug(f"Processed batch of {batch_size} trade messages")
            
        except Exception as e:
            # Clear batch on error to avoid reprocessing bad messages
            self._batch = []
            self._batch_start_time = time.time() * 1000
            
            raise ProcessingError(f"Failed to process trade batch: {e}")
    
    @circuit_breaker("db_operations")
    def _save_tick_batch(self, tick_data: List[Tick]) -> None:
        """
        Save a batch of tick data to the database.
        
        Args:
            tick_data: List of Tick models to save
            
        Raises:
            ProcessingError: If saving to the database fails
        """
        try:
            # Use repository to save batch
            self.repository.save_tick_batch(tick_data)
        except Exception as e:
            raise ProcessingError(f"Failed to save tick batch to database: {e}")
    
    def _create_tick_from_message(self, message: Dict[str, Any]) -> Tick:
        """
        Create a Tick model from a message.
        
        Args:
            message: Deserialized message
            
        Returns:
            Tick model
        
        Raises:
            ProcessingError: If creating the model fails
        """
        try:
            # Get or create instrument
            instrument = self._get_or_create_instrument(message['symbol'])
            
            # Parse timestamp
            if isinstance(message['timestamp'], (int, float)):
                # Convert from milliseconds if necessary
                timestamp = message['timestamp']
                if timestamp > 1e12:  # Assume milliseconds if very large
                    timestamp = timestamp / 1000.0
                timestamp = datetime.fromtimestamp(timestamp)
            else:
                # Parse ISO format
                timestamp = datetime.fromisoformat(message['timestamp'].replace('Z', '+00:00'))
            
            # Create Tick model
            tick = Tick(
                instrument_id=instrument.id,
                timestamp=timestamp,
                price=message['price'],
                volume=message['volume'],
                source=message.get('source', 'kafka'),
                source_timestamp=timestamp
            )
            
            # Add additional fields if present
            if 'trade_id' in message:
                tick.trade_id = message['trade_id']
            if 'side' in message:
                tick.side = message['side']
            
            # Handle additional trade data as JSON
            trade_data = {}
            for key, value in message.items():
                if key not in ['symbol', 'timestamp', 'price', 'volume', 'trade_id', 'side', 'source']:
                    trade_data[key] = value
            
            if trade_data:
                tick.trade_data = trade_data
            
            return tick
            
        except KeyError as e:
            raise ProcessingError(f"Missing required field in trade message: {e}")
        except Exception as e:
            raise ProcessingError(f"Failed to create Tick model: {e}")
    
    @circuit_breaker("db_operations")
    def _get_or_create_instrument(self, symbol: str) -> Instrument:
        """
        Get or create an instrument by symbol.
        
        Args:
            symbol: Instrument symbol
            
        Returns:
            Instrument model
            
        Raises:
            ProcessingError: If getting or creating the instrument fails
        """
        try:
            # Use repository to get or create instrument
            return self.repository.get_or_create_instrument(symbol)
        except Exception as e:
            raise ProcessingError(f"Failed to get or create instrument: {e}")

    def on_stop(self) -> None:
        """
        Perform cleanup when the consumer stops.
        
        This method is called when the consumer is stopping to ensure
        any pending data is processed.
        """
        # Process any remaining messages in the batch
        if self._batch:
            try:
                self._process_batch()
            except Exception as e:
                logger.error(f"Error processing final batch: {e}")
        
        # Final commit if needed
        try:
            if not self.settings.ENABLE_AUTO_COMMIT:
                self.offset_manager.commit(async_commit=False)
        except Exception as e:
            logger.error(f"Error committing final offsets: {e}")


# Import for _create_tick_from_message
from datetime import datetime