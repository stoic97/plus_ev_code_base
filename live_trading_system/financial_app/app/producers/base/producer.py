"""
Base producer class for Kafka producers.

This module provides a base class for all Kafka producers in the system,
implementing common functionality and configuration.
"""

import logging
from typing import Any, Dict, Optional

from confluent_kafka import Producer

from app.producers.config.settings import KafkaSettings

# Set up logging
logger = logging.getLogger(__name__)

class BaseProducer:
    """
    Base class for all Kafka producers.
    
    Provides common functionality for Kafka producers including:
    - Producer initialization
    - Basic message publishing
    - Error handling
    - Metrics tracking
    """
    
    def __init__(
        self,
        topic: str,
        settings: Optional[KafkaSettings] = None,
    ):
        """
        Initialize a new base producer.
        
        Args:
            topic: Kafka topic to produce to
            settings: Kafka configuration settings
        """
        self.settings = settings or KafkaSettings()
        self.topic = topic
        
        # Initialize Kafka producer
        self._producer = Producer({
            'bootstrap.servers': self.settings.BOOTSTRAP_SERVERS,
            'client.id': f"{self.__class__.__name__}-{id(self)}",
            'acks': self.settings.ACKS,
            'retries': self.settings.RETRIES,
            'retry.backoff.ms': self.settings.RETRY_BACKOFF_MS,
            'compression.type': self.settings.COMPRESSION_TYPE,
            'linger.ms': self.settings.LINGER_MS,
            'batch.size': self.settings.BATCH_SIZE,
            'max.in.flight.requests.per.connection': self.settings.MAX_IN_FLIGHT_REQUESTS,
            'enable.idempotence': self.settings.ENABLE_IDEMPOTENCE,
            'transactional.id': self.settings.TRANSACTIONAL_ID,
        })
        
        logger.info(f"Initialized {self.__class__.__name__} for topic '{self.topic}'")
    
    def _serialize_message(self, data: Dict[str, Any]) -> bytes:
        """
        Serialize message data to bytes.
        
        Args:
            data: Message data to serialize
            
        Returns:
            Serialized data as bytes
            
        Raises:
            SerializationError: If the data cannot be serialized
        """
        raise NotImplementedError("Subclasses must implement _serialize_message")
    
    def _delivery_report(self, err: Optional[Exception], msg: Any) -> None:
        """
        Handle delivery reports from Kafka.
        
        Args:
            err: Error if delivery failed
            msg: Message that was delivered
        """
        if err is not None:
            logger.error(f"Message delivery failed: {err}")
        else:
            logger.debug(f"Message delivered to {msg.topic()} [{msg.partition()}]")
    
    def on_stop(self) -> None:
        """
        Perform cleanup when the producer stops.
        
        This method is called when the producer is stopping to ensure
        any pending data is published.
        """
        try:
            self._producer.flush()
        except Exception as e:
            logger.error(f"Error flushing producer: {e}") 