"""
OHLCV price data producer for market data publishing.

This module provides a Kafka producer for publishing OHLCV (Open, High, Low, Close, Volume)
price data to Kafka topics.
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
from app.producers.utils.validation import validate_ohlcv_message

# Set up logging
logger = logging.getLogger(__name__)

class OHLCVProducer(BaseProducer):
    """
    Kafka producer for OHLCV (Open, High, Low, Close, Volume) price data.
    
    Publishes OHLCV data to a Kafka topic.
    """
    
    def __init__(
        self,
        topic: Optional[str] = None,
        settings: Optional[KafkaSettings] = None,
        batch_size: int = 100,
        batch_timeout_ms: int = 1000,
    ):
        """
        Initialize a new OHLCV producer.
        
        Args:
            topic: Kafka topic to produce to (defaults to OHLCV_TOPIC from settings)
            settings: Kafka configuration settings
            batch_size: Number of messages to batch before sending
            batch_timeout_ms: Maximum time to wait for a full batch in ms
        """
        self.settings = settings or KafkaSettings()
        self.topic = topic or self.settings.OHLCV_TOPIC
        
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
            producer_id=f"ohlcv_producer",
            topic=self.topic
        )
        
        logger.info(f"Initialized OHLCV producer for topic '{self.topic}'")
    
    def _serialize_message(self, data: Dict[str, Any]) -> bytes:
        """
        Serialize OHLCV data to bytes.
        
        Args:
            data: OHLCV data to serialize
            
        Returns:
            Serialized data as bytes
            
        Raises:
            SerializationError: If the data cannot be serialized
        """
        try:
            # Validate the message structure
            validate_ohlcv_message(data)
            
            # Serialize using JSON
            return serialize_json(data)
        except ValidationError as e:
            raise SerializationError(f"Invalid OHLCV message: {e}")
        except Exception as e:
            raise SerializationError(f"Failed to serialize OHLCV message: {e}")
    
    def publish_ohlcv(
        self,
        symbol: str,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        interval: str,
        timestamp: Optional[Union[int, float, str]] = None,
        source: str = "producer",
        vwap: Optional[float] = None,
        trades_count: Optional[int] = None,
        open_interest: Optional[float] = None,
        adjusted_close: Optional[float] = None,
    ) -> None:
        """
        Publish an OHLCV data point to Kafka.
        
        Args:
            symbol: Trading instrument symbol
            open_price: Opening price
            high: Highest price
            low: Lowest price
            close: Closing price
            volume: Trading volume
            interval: Time interval (e.g., '1m', '5m', '1h', '1d')
            timestamp: Message timestamp (defaults to current time)
            source: Data source identifier
            vwap: Volume-weighted average price
            trades_count: Number of trades
            open_interest: Open interest (for derivatives)
            adjusted_close: Adjusted closing price
            
        Raises:
            PublishingError: If publishing fails
        """
        try:
            # Create message
            message = {
                "symbol": symbol,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "interval": interval,
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
            if vwap is not None:
                message["vwap"] = vwap
            if trades_count is not None:
                message["trades_count"] = trades_count
            if open_interest is not None:
                message["open_interest"] = open_interest
            if adjusted_close is not None:
                message["adjusted_close"] = adjusted_close
            
            # Add to batch
            self._batch.append(message)
            
            # Process batch if full or timeout reached
            current_time = time.time() * 1000
            if (len(self._batch) >= self.batch_size or 
                current_time - self._batch_start_time >= self.batch_timeout_ms):
                self._publish_batch()
            
        except Exception as e:
            raise PublishingError(f"Failed to publish OHLCV data: {e}")
    
    def _publish_batch(self) -> None:
        """
        Publish a batch of OHLCV messages.
        
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
            
            logger.debug(f"Published batch of {batch_size} OHLCV messages")
            
        except Exception as e:
            # Clear batch on error to avoid reprocessing bad messages
            self._batch = []
            self._batch_start_time = time.time() * 1000
            
            raise PublishingError(f"Failed to publish OHLCV batch: {e}")
    
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