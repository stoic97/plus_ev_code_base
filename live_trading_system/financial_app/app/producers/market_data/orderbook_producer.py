"""
Kinesis producer for orderbook data.

This module provides a producer for publishing orderbook data to Kinesis streams.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime

from app.producers.base.kinesis_producer import BaseKinesisProducer
from app.producers.config.settings import KinesisSettings
from app.producers.utils.validation import validate_orderbook_message
from app.producers.base.error import SerializationError, PublishingError

# Set up logging
logger = logging.getLogger(__name__)

class OrderBookProducer(BaseKinesisProducer):
    """
    Producer for publishing orderbook data to Kinesis.
    
    This producer handles the publishing of orderbook data to a Kinesis stream,
    including validation and batching of messages.
    """
    
    def __init__(
        self,
        stream_name: str = "market-data-orderbook",
        settings: Optional[KinesisSettings] = None,
        region_name: Optional[str] = None,
        batch_size: int = 100,
        batch_timeout_ms: int = 1000
    ):
        """
        Initialize a new orderbook producer.
        
        Args:
            stream_name: Name of the Kinesis stream
            settings: Kinesis configuration settings
            region_name: AWS region name
            batch_size: Number of records to process in a batch
            batch_timeout_ms: Maximum time to wait for a full batch in ms
        """
        super().__init__(stream_name, settings, region_name, batch_size, batch_timeout_ms)
        self._batch: List[Dict[str, Any]] = []
        self._last_batch_time = time.time()
    
    def _validate_orderbook_data(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        depth: Optional[int] = None
    ) -> None:
        """
        Validate orderbook data.
        
        Args:
            bids: List of bid orders as (price, quantity) tuples
            asks: List of ask orders as (price, quantity) tuples
            depth: Orderbook depth
            
        Raises:
            ValueError: If any validation fails
        """
        # Validate depth
        if depth is not None and depth > len(bids) and depth > len(asks):
            raise ValueError("Depth cannot be greater than the number of price levels")
        
        # Validate bids
        for price, quantity in bids:
            if price <= 0:
                raise ValueError("Bid prices must be positive")
            if quantity <= 0:
                raise ValueError("Bid quantities must be positive")
        
        # Validate asks
        for price, quantity in asks:
            if price <= 0:
                raise ValueError("Ask prices must be positive")
            if quantity <= 0:
                raise ValueError("Ask quantities must be positive")
        
        # Validate orderbook is not crossed
        if bids and asks and bids[0][0] >= asks[0][0]:
            raise ValueError("Orderbook is crossed: best bid price >= best ask price")
        
        # Validate price levels are sorted
        for i in range(len(bids) - 1):
            if bids[i][0] <= bids[i + 1][0]:
                raise ValueError("Bid prices must be in descending order")
        
        for i in range(len(asks) - 1):
            if asks[i][0] >= asks[i + 1][0]:
                raise ValueError("Ask prices must be in ascending order")
    
    def _serialize_message(self, data: Dict[str, Any]) -> bytes:
        """
        Validate and serialize orderbook data.
        
        Args:
            data: Orderbook data to serialize
            
        Returns:
            Serialized data as bytes
            
        Raises:
            SerializationError: If the data cannot be serialized
        """
        # Validate orderbook data
        validate_orderbook_message(data)
        return super()._serialize_message(data)
    
    def publish_orderbook(
        self,
        symbol: str,
        bids: List[Tuple[float, float]],  # List of (price, quantity) tuples
        asks: List[Tuple[float, float]],  # List of (price, quantity) tuples
        timestamp: Optional[datetime] = None,
        exchange: Optional[str] = None,
        depth: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        """
        Publish an orderbook message to Kinesis.
        
        Args:
            symbol: Trading symbol
            bids: List of bid orders as (price, quantity) tuples
            asks: List of ask orders as (price, quantity) tuples
            timestamp: Orderbook timestamp (defaults to current time)
            exchange: Exchange name
            depth: Orderbook depth
            **kwargs: Additional orderbook data
        """
        # Validate orderbook data
        self._validate_orderbook_data(bids, asks, depth)
        
        # Construct orderbook message
        message = {
            "symbol": symbol,
            "bids": [{"price": price, "quantity": qty} for price, qty in bids],
            "asks": [{"price": price, "quantity": qty} for price, qty in asks],
            "timestamp": (timestamp or datetime.now()).isoformat(),
            "exchange": exchange,
            "depth": depth or len(bids),
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
            # Extract messages and partition keys
            messages = [msg for msg, _ in self._batch]
            partition_keys = [key for _, key in self._batch]
            
            # Publish batch
            self.publish_batch(messages, partition_keys[0] if len(set(partition_keys)) == 1 else None)
            
            # Clear batch
            self._batch = []
            self._last_batch_time = time.time()
            
        except Exception as e:
            logger.error(f"Failed to publish orderbook batch: {e}")
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