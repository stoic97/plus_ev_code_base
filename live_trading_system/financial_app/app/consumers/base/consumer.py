"""
Base Kafka consumer implementation.

This module provides the foundation for all Kafka consumers in the application,
with common functionality for connection management, message processing,
and error handling.
"""

import logging
import signal
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, Callable

from confluent_kafka import Consumer, KafkaError, KafkaException, Message
from pydantic import BaseModel, Field

from app.consumers.base.error import (
    ConsumerError, 
    ConnectionError, 
    DeserializationError, 
    ProcessingError
)
from app.consumers.config.settings import KafkaSettings

# Set up logging
logger = logging.getLogger(__name__)


class ConsumerStats(BaseModel):
    """Statistics for consumer monitoring and performance tracking."""
    
    topic: str = Field(..., description="Kafka topic name")
    messages_processed: int = Field(default=0, description="Total messages processed")
    messages_failed: int = Field(default=0, description="Total messages failed to process")
    bytes_processed: int = Field(default=0, description="Total bytes processed")
    last_message_timestamp: Optional[float] = Field(
        default=None, description="Timestamp of last message processed"
    )
    processing_time_ms: float = Field(
        default=0, description="Total processing time in milliseconds"
    )
    avg_processing_time_ms: float = Field(
        default=0, description="Average processing time per message in milliseconds"
    )
    consumer_lag: Optional[Dict[int, int]] = Field(
        default=None, description="Consumer lag by partition"
    )
    
    def update_processing_stats(self, processing_time_ms: float, message_size: int) -> None:
        """Update processing statistics with a new message."""
        self.messages_processed += 1
        self.bytes_processed += message_size
        self.processing_time_ms += processing_time_ms
        self.last_message_timestamp = time.time()
        
        # Update average processing time
        if self.messages_processed > 0:
            self.avg_processing_time_ms = self.processing_time_ms / self.messages_processed
    
    def record_failure(self) -> None:
        """Record a message processing failure."""
        self.messages_failed += 1


class BaseConsumer(ABC):
    """
    Abstract base class for Kafka consumers.
    
    Provides common functionality for all Kafka consumers including:
    - Connection management
    - Message polling
    - Error handling
    - Shutdown hooks
    - Statistics
    """
    
    def __init__(
        self,
        topic: str,
        group_id: Optional[str] = None,
        settings: Optional[KafkaSettings] = None,
        offset_reset: str = 'latest',
        auto_commit: bool = False,
        stats_interval: int = 60,
    ):
        """
        Initialize a new Kafka consumer.
        
        Args:
            topic: Kafka topic to consume
            group_id: Consumer group ID (defaults to class name if not provided)
            settings: Kafka configuration settings
            offset_reset: Auto offset reset strategy ('latest' or 'earliest')
            auto_commit: Whether to auto-commit offsets
            stats_interval: Interval in seconds to log consumer statistics
        """
        self.topic = topic
        self.group_id = group_id or self.__class__.__name__
        self.settings = settings or KafkaSettings()
        self.offset_reset = offset_reset
        self.auto_commit = auto_commit
        self.stats_interval = stats_interval
        
        # Internal state
        self._consumer = None
        self._running = False
        self._shutdown_event = threading.Event()
        
        # Statistics
        self.stats = ConsumerStats(topic=topic)
        self._last_stats_time = 0
        
        # Set up logging
        self._logger = logger.getChild(self.__class__.__name__)
        
        # Initialize the consumer
        self._init_consumer()
    
    def _init_consumer(self) -> None:
        """Initialize the Kafka consumer instance."""
        config = {
            'bootstrap.servers': ','.join(self.settings.BOOTSTRAP_SERVERS),
            'group.id': self.group_id,
            'auto.offset.reset': self.offset_reset,
            'enable.auto.commit': self.auto_commit,
            'max.poll.interval.ms': self.settings.MAX_POLL_INTERVAL_MS,
            'session.timeout.ms': self.settings.SESSION_TIMEOUT_MS,
            'request.timeout.ms': self.settings.REQUEST_TIMEOUT_MS,
            # Add client ID to improve monitoring
            'client.id': f"{self.group_id}-{self.__class__.__name__}"
        }
        
        try:
            self._consumer = Consumer(config)
            self._consumer.subscribe([self.topic])
            self._logger.info(f"Initialized consumer for topic '{self.topic}' with group ID '{self.group_id}'")
        except KafkaException as e:
            self._logger.error(f"Failed to initialize Kafka consumer: {e}")
            raise  # Re-raise the original exception
    
    def start(self, blocking: bool = True) -> None:
        """
        Start consuming messages from Kafka.
        
        Args:
            blocking: Whether to block the current thread
        """
        if self._running:
            self._logger.warning("Consumer already running")
            return
            
        self._running = True
        self._shutdown_event.clear()
        
        # Register signal handlers for graceful shutdown
        self._register_signal_handlers()
        
        self._logger.info(f"Starting consumer for topic '{self.topic}'")
        
        if blocking:
            try:
                self._consume_loop()
            except KeyboardInterrupt:
                self._logger.info("Consumer interrupted by user")
            finally:
                self.stop()
        else:
            # Start in a separate thread
            self._thread = threading.Thread(target=self._consume_loop)
            self._thread.daemon = True
            self._thread.start()
    
    def stop(self) -> None:
        """Stop the consumer gracefully."""
        if not self._running:
            return
            
        self._logger.info("Stopping consumer...")
        # Set the shutdown flag first
        self._running = False
        
        # Explicitly set the shutdown event
        self._shutdown_event.set()
        
        # Wait for thread to finish if running non-blocking
        if hasattr(self, '_thread') and self._thread.is_alive():
            self._thread.join(timeout=30)
            
        # Clean up resources
        if self._consumer:
            self._consumer.close()
            self._consumer = None
            
        self._logger.info("Consumer stopped")
    
    def _consume_loop(self) -> None:
        """Main message consumption loop."""
        while self._running:
            try:
                # Check for shutdown signal
                if self._shutdown_event.is_set():
                    break
                    
                # Poll for messages
                msg = self._consumer.poll(timeout=1.0)
                
                # No message, continue polling
                if msg is None:
                    continue
                
                # Handle errors
                if msg.error():
                    self._handle_kafka_error(msg.error())
                    continue
                
                # Process the message
                self._process_message(msg)
                
                # Log stats periodically
                self._maybe_log_stats()
                
            except Exception as e:
                self._logger.error(f"Unexpected error in consumer loop: {e}", exc_info=True)
                # Continue running despite errors
    
    def _process_message(self, msg: Message) -> None:
        """
        Process a single Kafka message.
        
        Args:
            msg: Kafka message to process
        """
        start_time = time.time()
        
        try:
            # Deserialize message
            deserialized = self._deserialize_message(msg)
            
            # Process message in derived class
            self.process_message(deserialized, msg)
            
            # Commit offset if not auto-committing
            if not self.auto_commit and self._consumer is not None:
                self._consumer.commit(msg, asynchronous=False)
            
            # Update statistics
            processing_time_ms = (time.time() - start_time) * 1000
            self.stats.update_processing_stats(
                processing_time_ms=processing_time_ms,
                message_size=len(msg.value()) if msg.value() else 0
            )
            
        except DeserializationError as e:
            self._logger.error(f"Failed to deserialize message: {e}")
            self.stats.record_failure()
        except ProcessingError as e:
            self._logger.error(f"Failed to process message: {e}")
            self.stats.record_failure()
        except Exception as e:
            self._logger.error(f"Unexpected error processing message: {e}", exc_info=True)
            self.stats.record_failure()
    
    @abstractmethod
    def _deserialize_message(self, msg: Message) -> Any:
        """
        Deserialize a Kafka message.
        
        Args:
            msg: Kafka message to deserialize
            
        Returns:
            Deserialized message content
            
        Raises:
            DeserializationError: If message cannot be deserialized
        """
        pass
    
    @abstractmethod
    def process_message(self, message: Any, raw_message: Message) -> None:
        """
        Process a deserialized message.
        
        This method must be implemented by derived classes to provide
        specific message processing logic.
        
        Args:
            message: Deserialized message content
            raw_message: Original Kafka message
            
        Raises:
            ProcessingError: If message cannot be processed
        """
        pass
    
    def _handle_kafka_error(self, error: KafkaError) -> None:
        """
        Handle Kafka errors.
        
        Args:
            error: Kafka error to handle
        """
        # Fatal errors that require consumer restart
        if error.code() in (
            KafkaError.UNKNOWN_TOPIC_OR_PART,
            KafkaError.LEADER_NOT_AVAILABLE,
            KafkaError.NOT_LEADER_FOR_PARTITION,
            KafkaError.REQUEST_TIMED_OUT,
        ):
            self._logger.error(f"Kafka error: {error}")
            self.stats.record_failure()
        # Non-fatal errors to log but continue
        else:
            self._logger.warning(f"Kafka error: {error}")
    
    def _register_signal_handlers(self) -> None:
        """Register signal handlers for graceful shutdown."""
        
        def handle_signal(sig, frame):
            self._logger.info(f"Received signal {sig}, shutting down...")
            self.stop()
        
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
    
    def _maybe_log_stats(self) -> None:
        """Log consumer statistics periodically."""
        now = time.time()
        if now - self._last_stats_time >= self.stats_interval:
            self._last_stats_time = now
            self._logger.info(
                f"Consumer stats for {self.topic}: "
                f"processed={self.stats.messages_processed}, "
                f"failed={self.stats.messages_failed}, "
                f"avg_time={self.stats.avg_processing_time_ms:.2f}ms"
            )
    
    def get_consumer_metrics(self) -> Dict[str, Any]:
        """
        Get consumer metrics for monitoring.
        
        Returns:
            Dictionary of consumer metrics
        """
        return {
            "topic": self.topic,
            "group_id": self.group_id,
            "messages_processed": self.stats.messages_processed,
            "messages_failed": self.stats.messages_failed,
            "avg_processing_time_ms": self.stats.avg_processing_time_ms,
            "bytes_processed": self.stats.bytes_processed,
            "last_message_timestamp": self.stats.last_message_timestamp,
            "consumer_lag": self.stats.consumer_lag,
            "running": self._running
        }