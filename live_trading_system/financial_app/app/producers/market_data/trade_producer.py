"""
Kinesis producer for trade data.

This module provides a producer for publishing individual trade data to Kinesis streams.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime

from app.producers.base.kinesis_producer import BaseKinesisProducer
from app.producers.config.settings import KinesisSettings
from app.producers.utils.validation import validate_trade_message

# Set up logging
logger = logging.getLogger(__name__)

class TradeProducer(BaseKinesisProducer):
    """
    Producer for publishing trade data to Kinesis.
    
    This producer handles the publishing of individual trade data to a Kinesis stream,
    including validation and batching of messages.
    """
    
    def __init__(
        self,
        stream_name: str = "market-data-trades",
        settings: Optional[KinesisSettings] = None,
        region_name: Optional[str] = None,
        batch_size: int = 100,
        batch_timeout_ms: int = 1000
    ):
        """
        Initialize a new trade producer.
        
        Args:
            stream_name: Name of the Kinesis stream
            settings: Kinesis configuration settings
            region_name: AWS region name
            batch_size: Number of records to process in a batch
            batch_timeout_ms: Maximum time to wait for a full batch in ms
        """
        super().__init__(stream_name, settings, region_name)
        self.batch_size = batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self._batch: List[Dict[str, Any]] = []
        self._last_batch_time = time.time()
    
    def _serialize_message(self, data: Dict[str, Any]) -> bytes:
        """
        Validate and serialize trade data.
        
        Args:
            data: Trade data to serialize
            
        Returns:
            Serialized data as bytes
            
        Raises:
            SerializationError: If the data cannot be serialized
        """
        # Validate trade data
        validate_trade_message(data)
        return super()._serialize_message(data)
    
    def publish_trade(
        self,
        symbol: str,
        price: float,
        quantity: float,
        side: str,
        timestamp: Optional[datetime] = None,
        trade_id: Optional[str] = None,
        exchange: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """
        Publish a trade message to Kinesis.
        
        Args:
            symbol: Trading symbol
            price: Trade price
            quantity: Trade quantity
            side: Trade side (buy/sell)
            timestamp: Trade timestamp (defaults to current time)
            trade_id: Unique trade identifier
            exchange: Exchange name
            **kwargs: Additional trade data
        """
        # Construct trade message
        message = {
            "symbol": symbol,
            "price": price,
            "quantity": quantity,
            "side": side,
            "timestamp": (timestamp or datetime.now()).isoformat(),
            "trade_id": trade_id,
            "exchange": exchange,
            **kwargs
        }
        
        # Add to batch
        self._batch.append((message, symbol))
        
        # Check if batch should be published
        current_time = time.time()
        if (len(self._batch) >= self.batch_size or
            (self._batch and current_time - self._last_batch_time >= self.batch_timeout_ms / 1000)):
            self._publish_batch()
    
    def _publish_batch(self) -> None:
        """Publish the current batch of messages."""
        if not self._batch:
            return
            
        try:
            # Publish batch
            self.publish_batch(self._batch)
            
            # Clear batch
            self._batch = []
            self._last_batch_time = time.time()
            
        except Exception as e:
            logger.error(f"Failed to publish trade batch: {e}")
            raise
    
    def on_stop(self) -> None:
        """
        Perform cleanup when the producer stops.
        
        This method ensures any remaining messages in the batch are published
        before the producer stops.
        """
        try:
            if self._batch:
                self._publish_batch()
        except Exception as e:
            logger.error(f"Error publishing final batch: {e}")
        finally:
            super().on_stop() 