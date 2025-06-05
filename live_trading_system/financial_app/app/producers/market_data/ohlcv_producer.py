"""
Kinesis producer for OHLCV data.

This module provides a producer for publishing OHLCV (Open, High, Low, Close, Volume)
candlestick data to Kinesis streams.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

from app.producers.base.kinesis_producer import BaseKinesisProducer
from app.producers.config.settings import KinesisSettings
from app.producers.utils.validation import validate_ohlcv_message
from app.producers.base.error import SerializationError, PublishingError

# Set up logging
logger = logging.getLogger(__name__)

class OHLCVProducer(BaseKinesisProducer):
    """
    Producer for publishing OHLCV data to Kinesis.
    
    This producer handles the publishing of OHLCV candlestick data to a Kinesis stream,
    including validation and batching of messages.
    """
    
    def __init__(
        self,
        stream_name: str = "market-data-ohlcv",
        settings: Optional[KinesisSettings] = None,
        region_name: Optional[str] = None,
        batch_size: int = 100,
        batch_timeout_ms: int = 1000
    ):
        """
        Initialize a new OHLCV producer.
        
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
    
    def _validate_ohlcv_data(
        self,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        volume: float,
        interval: str
    ) -> None:
        """
        Validate OHLCV data.
        
        Args:
            open_price: Opening price
            high_price: Highest price
            low_price: Lowest price
            close_price: Closing price
            volume: Trading volume
            interval: Time interval
            
        Raises:
            ValueError: If any validation fails
        """
        # Validate prices
        if open_price < 0 or high_price < 0 or low_price < 0 or close_price < 0:
            raise ValueError("Prices cannot be negative")
        
        # Validate volume
        if volume < 0:
            raise ValueError("Volume cannot be negative")
        
        # Validate price relationships
        if high_price < low_price:
            raise ValueError("High price cannot be less than low price")
        
        if open_price < low_price or open_price > high_price:
            raise ValueError("Open price must be between low and high prices")
        
        if close_price < low_price or close_price > high_price:
            raise ValueError("Close price must be between low and high prices")
        
        # Validate interval
        valid_intervals = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        if interval not in valid_intervals:
            raise ValueError(f"Invalid interval. Must be one of: {valid_intervals}")
    
    def _serialize_message(self, data: Dict[str, Any]) -> bytes:
        """
        Validate and serialize OHLCV data.
        
        Args:
            data: OHLCV data to serialize
            
        Returns:
            Serialized data as bytes
            
        Raises:
            SerializationError: If the data cannot be serialized
        """
        # Validate OHLCV data
        validate_ohlcv_message(data)
        return super()._serialize_message(data)
    
    def publish_ohlcv(
        self,
        symbol: str,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        volume: float,
        timestamp: Optional[datetime] = None,
        interval: str = "1m",
        exchange: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """
        Publish an OHLCV message to Kinesis.
        
        Args:
            symbol: Trading symbol
            open_price: Opening price
            high_price: Highest price
            low_price: Lowest price
            close_price: Closing price
            volume: Trading volume
            timestamp: Candlestick timestamp (defaults to current time)
            interval: Time interval (e.g., "1m", "5m", "1h")
            exchange: Exchange name
            **kwargs: Additional OHLCV data
        """
        # Validate OHLCV data
        self._validate_ohlcv_data(open_price, high_price, low_price, close_price, volume, interval)
        
        # Construct OHLCV message
        message = {
            "symbol": symbol,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume,
            "timestamp": (timestamp or datetime.now()).isoformat(),
            "interval": interval,
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
            # Extract messages and partition keys
            messages = [msg for msg, _ in self._batch]
            partition_keys = [key for _, key in self._batch]
            
            # Publish batch
            self.publish_batch(messages, partition_keys[0] if len(set(partition_keys)) == 1 else None)
            
            # Clear batch
            self._batch = []
            self._last_batch_time = time.time()
            
        except Exception as e:
            logger.error(f"Failed to publish OHLCV batch: {e}")
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