"""
Order book data consumer for market data ingestion.

This module provides a Kinesis consumer for ingesting order book data
from Kinesis streams and storing it in the database.
"""

import logging
import time
from typing import Dict, Any, Optional, Callable
from datetime import datetime

from app.consumers.base.kinesis_consumer import BaseKinesisConsumer
from app.consumers.config.settings import KinesisSettings
from app.consumers.utils.validation import validate_orderbook_message
from app.consumers.base.error import ProcessingError

# Set up logging
logger = logging.getLogger(__name__)


class OrderBookConsumer(BaseKinesisConsumer):
    """
    Consumer for processing orderbook data from Kinesis.
    
    This consumer handles the processing of orderbook data from a Kinesis stream,
    including validation and callback handling for bid and ask orders.
    """
    
    def __init__(
        self,
        stream_name: str = "market-data-orderbook",
        settings: Optional[KinesisSettings] = None,
        region_name: Optional[str] = None,
        batch_size: int = 100,
        batch_timeout_ms: int = 1000,
        on_orderbook: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """
        Initialize a new orderbook consumer.
        
        Args:
            stream_name: Name of the Kinesis stream
            settings: Kinesis configuration settings
            region_name: AWS region name
            batch_size: Number of records to process in a batch
            batch_timeout_ms: Maximum time to wait for a full batch in ms
            on_orderbook: Callback function for processing orderbook messages
        """
        super().__init__(stream_name, settings, region_name, batch_size, batch_timeout_ms)
        self.on_orderbook = on_orderbook
    
    def process_message(self, message: Dict[str, Any], raw_record: Dict[str, Any]) -> None:
        """
        Process an orderbook message.
        
        Args:
            message: Deserialized orderbook message
            raw_record: Original Kinesis record
            
        Raises:
            ProcessingError: If processing fails
        """
        start_time = time.time()
        
        try:
            # Validate orderbook message
            validate_orderbook_message(message)
            
            # Convert timestamp string to datetime
            if isinstance(message.get('timestamp'), str):
                message['timestamp'] = datetime.fromisoformat(message['timestamp'])
            
            # Call orderbook callback if provided
            if self.on_orderbook:
                self.on_orderbook(message)
            else:
                logger.debug(f"Received orderbook message: {message}")
            
            # Record successful processing
            processing_time_ms = (time.time() - start_time) * 1000
            self.metrics.record_message_processed(processing_time_ms)
                
        except Exception as e:
            # Record failed processing
            self.metrics.record_message_failed()
            logger.error(f"Failed to process orderbook message: {e}")
            raise ProcessingError(f"Failed to process orderbook message: {e}")