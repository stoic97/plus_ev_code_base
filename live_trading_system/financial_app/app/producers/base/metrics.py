"""
Metrics registry for producer operations.
"""

import logging
import threading
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class MetricsRegistry:
    """Registry for producer metrics."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialize()
            return cls._instance
    
    def _initialize(self):
        """Initialize metrics registry."""
        self._counters: Dict[str, int] = {}
        self._histograms: Dict[str, list] = {}
        self._producers: Dict[str, Any] = {}
    
    def register_producer(self, producer_id: str, topic: str) -> 'ProducerMetrics':
        """Register a producer for metrics tracking."""
        if producer_id not in self._producers:
            self._producers[producer_id] = ProducerMetrics(producer_id, topic)
        return self._producers[producer_id]
    
    def counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        key = f"{name}_{str(labels)}"
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + 1
    
    def histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram metric."""
        key = f"{name}_{str(labels)}"
        with self._lock:
            if key not in self._histograms:
                self._histograms[key] = []
            self._histograms[key].append(value)

class ProducerMetrics:
    """Metrics for a single producer."""
    
    def __init__(self, producer_id: str, topic: str):
        self.producer_id = producer_id
        self.topic = topic
        self._registry = get_metrics_registry()
        self._messages_published = 0
        self._messages_failed = 0
    
    @property
    def messages_published(self) -> int:
        """Get the number of successfully published messages."""
        return self._messages_published
    
    @property
    def messages_failed(self) -> int:
        """Get the number of failed messages."""
        return self._messages_failed
    
    def record_messages_published(self, count: int = 1, processing_time_ms: float = 0) -> None:
        """Record published messages."""
        self._messages_published += count
        self._registry.counter(
            "messages_published",
            {"producer": self.producer_id, "topic": self.topic, "status": "success"}
        )
        if processing_time_ms > 0:
            self._registry.histogram(
                "message_processing_time_ms",
                processing_time_ms,
                {"producer": self.producer_id, "topic": self.topic}
            )
    
    def record_message_failed(self, count: int = 1) -> None:
        """Record failed messages."""
        self._messages_failed += count
        self._registry.counter(
            "messages_published",
            {"producer": self.producer_id, "topic": self.topic, "status": "error"}
        )

def get_metrics_registry() -> MetricsRegistry:
    """Get the global metrics registry instance."""
    return MetricsRegistry() 