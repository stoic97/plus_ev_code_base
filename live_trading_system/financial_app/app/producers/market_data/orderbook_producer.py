"""
Order book data producer for market data publishing.

This module provides a Kafka producer for publishing order book data
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
from app.producers.utils.validation import validate_orderbook_message

# Set up logging
logger = logging.getLogger(__name__)

class OrderBookProducer(BaseProducer):
    """
    Kafka producer for order book data.
    
    Publishes order book snapshots to a Kafka topic.
    """
    
    def __init__(
        self,
        topic: Optional[str] = None,
        settings: Optional[KafkaSettings] = None,
        batch_size: int = 100,
        batch_timeout_ms: int = 1000,
    ):
        """
        Initialize a new order book producer.
        
        Args:
            topic: Kafka topic to produce to (defaults to ORDERBOOK_TOPIC from settings)
            settings: Kafka configuration settings
            batch_size: Number of messages to batch before sending
            batch_timeout_ms: Maximum time to wait for a full batch in ms
        """
        self.settings = settings or KafkaSettings()
        self.topic = topic or self.settings.ORDERBOOK_TOPIC
        
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
            producer_id=f"orderbook_producer",
            topic=self.topic
        )
        
        logger.info(f"Initialized order book producer for topic '{self.topic}'")
    
    def _serialize_message(self, data: Dict[str, Any]) -> bytes:
        """
        Serialize order book data to bytes.
        
        Args:
            data: Order book data to serialize
            
        Returns:
            Serialized data as bytes
            
        Raises:
            SerializationError: If the data cannot be serialized
        """
        try:
            # Validate the message structure
            validate_orderbook_message(data)
            
            # Serialize using JSON
            return serialize_json(data)
        except ValidationError as e:
            raise SerializationError(f"Invalid order book message: {e}")
        except Exception as e:
            raise SerializationError(f"Failed to serialize order book message: {e}")
    
    def publish_orderbook(
        self,
        symbol: str,
        bids: List[List[float]],
        asks: List[List[float]],
        timestamp: Optional[Union[int, float, str]] = None,
        source: str = "producer",
        depth: Optional[int] = None,
        spread: Optional[float] = None,
        weighted_mid_price: Optional[float] = None,
        imbalance: Optional[float] = None,
    ) -> None:
        """
        Publish an order book snapshot to Kafka.
        
        Args:
            symbol: Trading instrument symbol
            bids: List of [price, quantity] pairs for bids
            asks: List of [price, quantity] pairs for asks
            timestamp: Message timestamp (defaults to current time)
            source: Data source identifier
            depth: Order book depth
            spread: Current spread
            weighted_mid_price: Weighted mid price
            imbalance: Order book imbalance
            
        Raises:
            PublishingError: If publishing fails
        """
        try:
            # Create message
            message = {
                "symbol": symbol,
                "bids": bids,
                "asks": asks,
                "source": source,
            }
            
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
            
            # Add optional fields
            if depth is not None:
                message["depth"] = depth
            if spread is not None:
                message["spread"] = spread
            if weighted_mid_price is not None:
                message["weighted_mid_price"] = weighted_mid_price
            if imbalance is not None:
                message["imbalance"] = imbalance
            
            # Add to batch
            self._batch.append(message)
            
            # Process batch if full or timeout reached
            current_time = time.time() * 1000
            if (len(self._batch) >= self.batch_size or 
                current_time - self._batch_start_time >= self.batch_timeout_ms):
                self._publish_batch()
            
        except Exception as e:
            raise PublishingError(f"Failed to publish order book: {e}")
    
    def _publish_batch(self) -> None:
        """
        Publish a batch of order book messages.
        
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
            
            logger.debug(f"Published batch of {batch_size} order book messages")
            
        except Exception as e:
            # Clear batch on error to avoid reprocessing bad messages
            self._batch = []
            self._batch_start_time = time.time() * 1000
            
            raise PublishingError(f"Failed to publish order book batch: {e}")
    
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