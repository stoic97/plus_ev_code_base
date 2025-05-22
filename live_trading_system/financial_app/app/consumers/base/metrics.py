"""
Performance metrics for Kafka consumers.

This module provides utilities for tracking and reporting on
consumer performance, including throughput, latency, and lag metrics.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import threading
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any


@dataclass
class MetricSample:
    """A single metric sample with timestamp."""
    value: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class ConsumerMetrics:
    """Metrics for a single consumer instance."""
    
    # Consumer identification
    consumer_id: str
    topic: str
    group_id: str
    
    # Performance metrics
    messages_processed: int = 0
    messages_failed: int = 0
    bytes_processed: int = 0
    processing_times_ms: List[MetricSample] = field(default_factory=list)
    commit_times_ms: List[MetricSample] = field(default_factory=list)
    
    # Time metrics
    start_time: float = field(default_factory=time.time)
    last_message_time: Optional[float] = None
    
    # Lag metrics
    consumer_lag: Dict[int, int] = field(default_factory=dict)
    
    # Runtime status
    is_running: bool = False
    
    # Thread safety
    _lock: threading.RLock = field(default_factory=threading.RLock)

    
    def record_message_processed(self, size_bytes: int, processing_time_ms: float) -> None:
        """
        Record metrics for a successfully processed message.
        
        Args:
            size_bytes: Size of the message in bytes
            processing_time_ms: Time taken to process the message in milliseconds
        """
        with self._lock:
            self.messages_processed += 1
            self.bytes_processed += size_bytes
            self.last_message_time = time.time()
            
            # Keep only recent processing times for moving averages
            self.processing_times_ms.append(MetricSample(processing_time_ms))
            
            # Limit list size to avoid memory growth
            if len(self.processing_times_ms) > 1000:
                self.processing_times_ms = self.processing_times_ms[-1000:]
    
    def record_message_failed(self) -> None:
        """Record a message processing failure."""
        with self._lock:
            self.messages_failed += 1
    
    def record_commit(self, commit_time_ms: float) -> None:
        """
        Record metrics for an offset commit operation.
        
        Args:
            commit_time_ms: Time taken to commit offsets in milliseconds
        """
        with self._lock:
            self.commit_times_ms.append(MetricSample(commit_time_ms))
            
            # Limit list size
            if len(self.commit_times_ms) > 100:
                self.commit_times_ms = self.commit_times_ms[-100:]
    
    def update_consumer_lag(self, partition: int, lag: int) -> None:
        """
        Update consumer lag for a partition.
        
        Args:
            partition: Partition number
            lag: Consumer lag (difference between latest and current offset)
        """
        with self._lock:
            self.consumer_lag[partition] = lag
    
    def get_avg_processing_time_ms(self, window_seconds: int = 60) -> Optional[float]:
        """
        Get average message processing time over the specified window.
        
        Args:
            window_seconds: Time window in seconds to calculate average
            
        Returns:
            Average processing time in milliseconds or None if no data
        """
        with self._lock:
            if not self.processing_times_ms:
                return None
                
            # Filter samples within the time window
            cutoff = time.time() - window_seconds
            recent_samples = [
                sample.value for sample in self.processing_times_ms
                if sample.timestamp >= cutoff
            ]
            
            if not recent_samples:
                return None
                
            return statistics.mean(recent_samples)
    
    def get_throughput(self, window_seconds: int = 60) -> float:
        """
        Calculate message throughput (messages per second).
        
        Args:
            window_seconds: Time window in seconds to calculate throughput
            
        Returns:
            Messages per second over the specified window
        """
        with self._lock:
            # Count messages in the time window
            cutoff = time.time() - window_seconds
            recent_messages = sum(
                1 for sample in self.processing_times_ms
                if sample.timestamp >= cutoff
            )
            
            # Calculate throughput
            elapsed = min(window_seconds, time.time() - self.start_time)
            if elapsed <= 0:
                return 0
                
            return recent_messages / elapsed
    
    def get_total_lag(self) -> int:
        """
        Get total consumer lag across all partitions.
        
        Returns:
            Sum of lag values across all partitions
        """
        with self._lock:
            return sum(self.consumer_lag.values())
    
    def get_p95_processing_time_ms(self, window_seconds: int = 60) -> Optional[float]:
        """
        Get 95th percentile processing time.
        
        Args:
            window_seconds: Time window in seconds
            
        Returns:
            95th percentile processing time in milliseconds or None if no data
        """
        with self._lock:
            # Filter samples within the time window
            cutoff = time.time() - window_seconds
            recent_samples = [
                sample.value for sample in self.processing_times_ms
                if sample.timestamp >= cutoff
            ]
            
            if len(recent_samples) < 5:  # Need enough samples for percentile
                return None
                
            # Calculate 95th percentile
            recent_samples.sort()
            index = int(len(recent_samples) * 0.95)
            return recent_samples[index]
    
    def get_error_rate(self, window_seconds: int = 60) -> float:
        """
        Calculate error rate (percentage of failed messages).
        
        Args:
            window_seconds: Time window in seconds
            
        Returns:
            Error rate as a percentage
        """
        with self._lock:
            # Count total messages in the window
            cutoff = time.time() - window_seconds
            total_in_window = sum(
                1 for sample in self.processing_times_ms
                if sample.timestamp >= cutoff
            )
            
            # We don't track timestamps for failures, so this is approximate
            if total_in_window == 0:
                return 0.0
                
            # Estimate failures in the window based on overall ratio
            if self.messages_processed == 0:
                return 100.0 if self.messages_failed > 0 else 0.0
                
            failure_ratio = self.messages_failed / (self.messages_processed + self.messages_failed)
            estimated_failures = total_in_window * failure_ratio
            
            return (estimated_failures / (total_in_window + estimated_failures)) * 100.0
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of all metrics.
        
        Returns:
            Dictionary with all metric values
        """
        with self._lock:
            uptime_seconds = time.time() - self.start_time
            
            return {
                "consumer_id": self.consumer_id,
                "topic": self.topic,
                "group_id": self.group_id,
                "uptime_seconds": uptime_seconds,
                "messages_processed": self.messages_processed,
                "messages_failed": self.messages_failed,
                "bytes_processed": self.bytes_processed,
                "throughput_msgs_per_sec": self.get_throughput(),
                "throughput_bytes_per_sec": self.bytes_processed / uptime_seconds if uptime_seconds > 0 else 0,
                "avg_processing_time_ms": self.get_avg_processing_time_ms(),
                "p95_processing_time_ms": self.get_p95_processing_time_ms(),
                "error_rate_percent": self.get_error_rate(),
                "total_consumer_lag": self.get_total_lag(),
                "consumer_lag_by_partition": dict(self.consumer_lag),
                "is_running": self.is_running,
                "last_message_time": self.last_message_time,
                "idle_seconds": time.time() - self.last_message_time if self.last_message_time else None
            }


class MetricsRegistry:
    """
    Registry for tracking metrics from multiple consumers.
    
    This class provides a centralized way to track and access metrics
    from all consumer instances in the application.
    """
    
    def __init__(self):
        """Initialize the metrics registry."""
        self._metrics: Dict[str, ConsumerMetrics] = {}
        self._lock = threading.Lock()
    
    def register_consumer(self, consumer_id: str, topic: str, group_id: str) -> ConsumerMetrics:
        """
        Register a new consumer for metrics tracking.
        
        Args:
            consumer_id: Unique identifier for the consumer
            topic: Kafka topic
            group_id: Consumer group ID
            
        Returns:
            ConsumerMetrics instance for the consumer
        """
        with self._lock:
            metrics = ConsumerMetrics(
                consumer_id=consumer_id,
                topic=topic,
                group_id=group_id
            )
            self._metrics[consumer_id] = metrics
            return metrics
    
    def get_consumer_metrics(self, consumer_id: str) -> Optional[ConsumerMetrics]:
        """
        Get metrics for a specific consumer.
        
        Args:
            consumer_id: Consumer identifier
            
        Returns:
            ConsumerMetrics instance or None if not found
        """
        with self._lock:
            return self._metrics.get(consumer_id)
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get metrics summaries for all registered consumers.
        
        Returns:
            Dictionary mapping consumer IDs to metric summaries
        """
        with self._lock:
            return {
                consumer_id: metrics.get_metrics_summary()
                for consumer_id, metrics in self._metrics.items()
            }
    
    def get_all_metrics_by_topic(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get metrics summaries grouped by topic.
        
        Returns:
            Dictionary mapping topics to lists of metric summaries
        """
        with self._lock:
            result = {}
            for metrics in self._metrics.values():
                topic = metrics.topic
                if topic not in result:
                    result[topic] = []
                result[topic].append(metrics.get_metrics_summary())
            return result


# Global metrics registry
metrics_registry = MetricsRegistry()


def get_metrics_registry() -> MetricsRegistry:
    """
    Get the global metrics registry.
    
    Returns:
        Global metrics registry instance
    """
    return metrics_registry