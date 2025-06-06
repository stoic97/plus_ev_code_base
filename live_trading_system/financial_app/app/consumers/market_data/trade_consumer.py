"""
Kinesis consumer for trade data.

This module provides a consumer for processing individual trade data from Kinesis streams.
"""

import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime

from app.consumers.base.kinesis_consumer import BaseKinesisConsumer
from app.consumers.config.settings import KinesisSettings
from app.consumers.utils.validation import validate_trade_message

# Set up logging
logger = logging.getLogger(__name__)

class TradeConsumer(BaseKinesisConsumer):
    """
    Consumer for processing trade data from Kinesis.
    
    This consumer handles the processing of individual trade data from a Kinesis stream,
    including validation and callback handling.
    """
    
    def __init__(
        self,
        stream_name: str = "market-data-trades",
        settings: Optional[KinesisSettings] = None,
        region_name: Optional[str] = None,
        batch_size: int = 100,
        batch_timeout_ms: int = 1000,
        on_trade: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """
        Initialize a new trade consumer.
        
        Args:
            stream_name: Name of the Kinesis stream
            settings: Kinesis configuration settings
            region_name: AWS region name
            batch_size: Number of records to process in a batch
            batch_timeout_ms: Maximum time to wait for a full batch in ms
            on_trade: Callback function for processing trade messages
        """
        super().__init__(stream_name, settings, region_name, batch_size, batch_timeout_ms)
        self.on_trade = on_trade
    
    def process_message(self, message: Dict[str, Any], raw_record: Dict[str, Any]) -> None:
        """
        Process a trade message.
        
        Args:
            message: Deserialized trade message
            raw_record: Original Kinesis record
            
        Raises:
            ProcessingError: If processing fails
        """
        try:
            # Validate trade message
            validate_trade_message(message)
            
            # Convert timestamp string to datetime
            if isinstance(message.get('timestamp'), str):
                message['timestamp'] = datetime.fromisoformat(message['timestamp'])
            
            # Call trade callback if provided
            if self.on_trade:
                self.on_trade(message)
            else:
                logger.debug(f"Received trade message: {message}")
                
        except Exception as e:
            logger.error(f"Failed to process trade message: {e}")
            raise