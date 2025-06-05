"""
Kinesis consumer for OHLCV data.

This module provides a consumer for processing OHLCV (Open, High, Low, Close, Volume)
candlestick data from Kinesis streams.
"""

import logging
import time
from typing import Dict, Any, Optional, Callable
from datetime import datetime

from app.consumers.base.kinesis_consumer import BaseKinesisConsumer
from app.consumers.config.settings import KinesisSettings
from app.consumers.utils.validation import validate_ohlcv_message
from app.consumers.base.error import ProcessingError

# Set up logging
logger = logging.getLogger(__name__)

class OHLCVConsumer(BaseKinesisConsumer):
    """
    Consumer for processing OHLCV data from Kinesis.
    
    This consumer handles the processing of OHLCV candlestick data from a Kinesis stream,
    including validation and callback handling.
    """
    
    def __init__(
        self,
        stream_name: str = "market-data-ohlcv",
        settings: Optional[KinesisSettings] = None,
        region_name: Optional[str] = None,
        batch_size: int = 100,
        batch_timeout_ms: int = 1000,
        on_candlestick: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """
        Initialize a new OHLCV consumer.
        
        Args:
            stream_name: Name of the Kinesis stream
            settings: Kinesis configuration settings
            region_name: AWS region name
            batch_size: Number of records to process in a batch
            batch_timeout_ms: Maximum time to wait for a full batch in ms
            on_candlestick: Callback function for processing candlestick messages
        """
        super().__init__(stream_name, settings, region_name, batch_size, batch_timeout_ms)
        self.on_candlestick = on_candlestick
    
    def process_message(self, message: Dict[str, Any], raw_record: Dict[str, Any]) -> None:
        """
        Process an OHLCV message.
        
        Args:
            message: Deserialized OHLCV message
            raw_record: Original Kinesis record
            
        Raises:
            ProcessingError: If processing fails
        """
        start_time = time.time()
        
        try:
            # Validate OHLCV message
            validate_ohlcv_message(message)
            
            # Convert timestamp string to datetime
            if isinstance(message.get('timestamp'), str):
                message['timestamp'] = datetime.fromisoformat(message['timestamp'])
            
            # Call candlestick callback if provided
            if self.on_candlestick:
                self.on_candlestick(message)
            else:
                logger.debug(f"Received OHLCV message: {message}")
            
            # Record successful processing
            processing_time_ms = (time.time() - start_time) * 1000
            self.metrics.record_message_processed(processing_time_ms)
                
        except Exception as e:
            # Record failed processing
            self.metrics.record_message_failed()
            logger.error(f"Failed to process OHLCV message: {e}")
            raise ProcessingError(f"Failed to process OHLCV message: {e}") 