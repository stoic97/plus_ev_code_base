"""
Metrics tracking for Kafka producers.

This module provides functionality for tracking various metrics related to
Kafka producer performance and message delivery.
"""

import logging
import time
from typing import Dict, Optional
from threading import Lock

# Set up logging
logger = logging.getLogger(__name__)

class MetricsRegistry:
    """
    Registry for producer metrics.
    
    Maintains a collection of metrics for different producers and topics.
    """
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialize()
            return cls._instance
    
    def _initialize(self):
        """Initialize the metrics registry."""
        self._producers: Dict[str, ProducerMetrics] = {}
    
    def register_producer(self, producer_id: str, topic: str) -> 'ProducerMetrics':
        """
        Register a new producer for metrics tracking.
        
        Args:
            producer_id: Unique identifier for the producer
            topic: Kafka topic being produced to
            
        Returns:
            ProducerMetrics instance for the producer
        """
        with self._lock:
            if producer_id not in self._producers:
                self._producers[producer_id] = ProducerMetrics(producer_id, topic)
            return self._producers[producer_id]
    
    def get_producer_metrics(self, producer_id: str) -> Optional['ProducerMetrics']:
        """
        Get metrics for a specific producer.
        
        Args:
            producer_id: Producer identifier
            
        Returns:
            ProducerMetrics instance if found, None otherwise
        """
        return self._producers.get(producer_id)

class ProducerMetrics:
    """
    Metrics tracking for a single producer.
    
    Tracks various metrics including:
    - Messages published
    - Failed messages
    - Processing time
    - Batch statistics
    """
    
    def __init__(self, producer_id: str, topic: str):
        """
        Initialize metrics for a producer.
        
        Args:
            producer_id: Unique identifier for the producer
            topic: Kafka topic being produced to
        """
        self.producer_id = producer_id
        self.topic = topic
        self.messages_published = 0
        self.messages_failed = 0
        self.total_processing_time_ms = 0
        self.batches_published = 0
        self.last_publish_time = None
    
    def record_messages_published(self, count: int, processing_time_ms: float) -> None:
        """
        Record successful message publishing.
        
        Args:
            count: Number of messages published
            processing_time_ms: Time taken to process the messages
        """
        self.messages_published += count
        self.total_processing_time_ms += processing_time_ms
        self.batches_published += 1
        self.last_publish_time = time.time()
    
    def record_message_failed(self) -> None:
        """Record a failed message delivery."""
        self.messages_failed += 1
    
    def get_average_processing_time(self) -> float:
        """
        Get average processing time per message.
        
        Returns:
            Average processing time in milliseconds
        """
        if self.messages_published == 0:
            return 0.0
        return self.total_processing_time_ms / self.messages_published
    
    def get_success_rate(self) -> float:
        """
        Get message delivery success rate.
        
        Returns:
            Success rate as a percentage
        """
        total = self.messages_published + self.messages_failed
        if total == 0:
            return 100.0
        return (self.messages_published / total) * 100

def get_metrics_registry() -> MetricsRegistry:
    """
    Get the global metrics registry instance.
    
    Returns:
        MetricsRegistry instance
    """
    return MetricsRegistry() 