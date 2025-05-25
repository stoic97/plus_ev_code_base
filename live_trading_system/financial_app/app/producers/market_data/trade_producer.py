"""
Trade data producer for market data publishing.

This module provides a Kafka producer for publishing individual trade data
to Kafka topics.
"""

import logging
import json
import time
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from confluent_kafka import Producer

from app.producers.base.producer import BaseProducer
from app.producers.base.error import SerializationError, PublishingError
from app.producers.base.metrics import get_metrics_registry
from app.producers.config.settings import KafkaSettings
from app.producers.utils.serialization import serialize_json
from app.producers.utils.validation import validate_trade_message

# Set up logging
logger = logging.getLogger(__name__)

class TradeProducer(BaseProducer):
    """
    Kafka producer for individual trade data.
    
    Publishes trade data to a Kafka topic.
    """
    
    def __init__(
        self,
        topic: Optional[str] = None,
        settings: Optional[KafkaSettings] = None,
        batch_size: int = 100,
        batch_timeout_ms: int = 1000,
    ):
        """
        Initialize a new trade producer.
        
        Args:
            topic: Kafka topic to produce to (defaults to TRADES_TOPIC from settings)
            settings: Kafka configuration settings
            batch_size: Number of messages to batch before sending
            batch_timeout_ms: Maximum time to wait for a full batch in ms
        """
        self.settings = settings or KafkaSettings()
        self.topic = topic or self.settings.TRADES_TOPIC
        
        # Initialize base producer
        super().__init__(
            topic=self.topic,
            settings=self.settings
        )
        
        # Batch processing
        self.batch_size = batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self._batch: List[Dict[str, Any]] = []
        self._batch_start_time = time.time() * 1000
        
        # Set up metrics
        self.metrics = get_metrics_registry().register_producer(
            producer_id=f"trade_producer",
            topic=self.topic
        )
        
        logger.info(f"Initialized trade producer for topic '{self.topic}'")
    
    def _serialize_message(self, data: Dict[str, Any]) -> bytes:
        """
        Serialize trade data to bytes.
        
        Args:
            data: Trade data to serialize
            
        Returns:
            Serialized data as bytes
            
        Raises:
            SerializationError: If the data cannot be serialized
        """
        try:
            # Validate the message structure
            validate_trade_message(data)
            
            # Serialize using JSON
            return serialize_json(data)
        except ValidationError as e:
            raise SerializationError(f"Invalid trade message: {e}")
        except Exception as e:
            raise SerializationError(f"Failed to serialize trade message: {e}")
    
    def publish_trade(
        self,
        symbol: str,
        price: float,
        volume: float,
        side: Optional[str] = None,
        trade_id: Optional[str] = None,
        timestamp: Optional[Union[int, float, str]] = None,
        source: str = "producer",
        **additional_data: Any
    ) -> None:
        """
        Publish a trade to Kafka.
        
        Args:
            symbol: Trading instrument symbol
            price: Trade price
            volume: Trade volume
            side: Trade side ('buy' or 'sell')
            trade_id: Unique trade identifier
            timestamp: Message timestamp (defaults to current time)
            source: Data source identifier
            additional_data: Additional trade data to include
            
        Raises:
            PublishingError: If publishing fails
        """
        try:
            # Create message
            message = {
                "symbol": symbol,
                "price": price,
                "volume": volume,
                "source": source,
            }
            
            # Add optional fields
            if side is not None:
                message["side"] = side
            if trade_id is not None:
                message["trade_id"] = trade_id
            
            # Add timestamp
            if timestamp is None:
                timestamp = int(time.time() * 1000)  # Current time in milliseconds
            elif isinstance(timestamp, (int, float)):
                if timestamp < 1e12:  # Assume seconds if small
                    timestamp = int(timestamp * 1000)
            else:
                # Parse ISO format
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                timestamp = int(dt.timestamp() * 1000)
            message["timestamp"] = timestamp
            
            # Add additional data
            if additional_data:
                message.update(additional_data)
            
            # Add to batch
            self._batch.append(message)
            
            # Process batch if full or timeout reached
            current_time = time.time() * 1000
            if (len(self._batch) >= self.batch_size or 
                current_time - self._batch_start_time >= self.batch_timeout_ms):
                self._publish_batch()
            
        except Exception as e:
            raise PublishingError(f"Failed to publish trade: {e}")
    
    def _publish_batch(self) -> None:
        """
        Publish a batch of trade messages.
        
        Raises:
            PublishingError: If publishing fails
        """
        if not self._batch:
            return
            
        start_time = time.time()
        
        try:
            # Publish each message in the batch
            for message in self._batch:
                # Serialize message
                serialized = self._serialize_message(message)
                
                # Publish to Kafka
                self._producer.produce(
                    topic=self.topic,
                    value=serialized,
                    key=message["symbol"].encode('utf-8'),
                    on_delivery=self._delivery_report
                )
            
            # Flush to ensure delivery
            self._producer.flush()
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Update metrics
            self.metrics.record_messages_published(len(self._batch), processing_time_ms)
            
            # Reset batch
            batch_size = len(self._batch)
            self._batch = []
            self._batch_start_time = time.time() * 1000
            
            logger.debug(f"Published batch of {batch_size} trade messages")
            
        except Exception as e:
            # Clear batch on error to avoid reprocessing bad messages
            self._batch = []
            self._batch_start_time = time.time() * 1000
            
            raise PublishingError(f"Failed to publish trade batch: {e}")
    
    def _delivery_report(self, err: Optional[Exception], msg: Any) -> None:
        """
        Handle delivery reports from Kafka.
        
        Args:
            err: Error if delivery failed
            msg: Message that was delivered
        """
        if err is not None:
            logger.error(f"Message delivery failed: {err}")
            self.metrics.record_message_failed()
        else:
            logger.debug(f"Message delivered to {msg.topic()} [{msg.partition()}]")
    
    def on_stop(self) -> None:
        """
        Perform cleanup when the producer stops.
        
        This method is called when the producer is stopping to ensure
        any pending data is published.
        """
        # Publish any remaining messages in the batch
        if self._batch:
            try:
                self._publish_batch()
            except Exception as e:
                logger.error(f"Error publishing final batch: {e}")
        
        # Flush any remaining messages
        try:
            self._producer.flush()
        except Exception as e:
            logger.error(f"Error flushing producer: {e}") 