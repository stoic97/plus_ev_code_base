"""
Health monitoring utilities for Kafka consumers.

This module provides utilities for monitoring the health of Kafka consumers,
including lag monitoring, throughput tracking, and alerting.
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Any, Set, Callable, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta

from confluent_kafka import Consumer, KafkaException

from app.consumers.managers.offset_manager import OffsetManager

# Set up logging
logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status of a consumer."""
    HEALTHY = "healthy"         # Consumer is operating normally
    DEGRADED = "degraded"       # Consumer is operating with reduced performance
    UNHEALTHY = "unhealthy"     # Consumer is not operating correctly
    STALLED = "stalled"         # Consumer is not making progress
    UNKNOWN = "unknown"         # Consumer health status is unknown


@dataclass
class HealthCheckConfig:
    """Configuration for consumer health checks."""
    # Lag thresholds
    max_lag_messages: int = 10000
    critical_lag_messages: int = 50000
    
    # Time thresholds
    max_idle_seconds: int = 300  # 5 minutes
    critical_idle_seconds: int = 600  # 10 minutes
    
    # Performance thresholds
    min_throughput_messages_per_second: float = 1.0
    low_throughput_messages_per_second: float = 0.1
    
    # Error thresholds
    max_error_rate_percent: float = 5.0
    critical_error_rate_percent: float = 20.0


class ConsumerHealthCheck:
    """
    Health check for a Kafka consumer.
    
    Monitors consumer health based on various metrics:
    - Consumer lag
    - Processing throughput
    - Error rate
    - Resource usage
    """
    
    def __init__(
        self,
        consumer_id: str,
        consumer: Consumer,
        offset_manager: Optional[OffsetManager] = None,
        config: Optional[HealthCheckConfig] = None,
        health_check_interval_seconds: int = 60
    ):
        """
        Initialize a new consumer health check.
        
        Args:
            consumer_id: Unique identifier for the consumer
            consumer: Kafka consumer instance
            offset_manager: Offset manager for lag tracking
            config: Health check configuration
            health_check_interval_seconds: Interval between health checks
        """
        self.consumer_id = consumer_id
        self.consumer = consumer
        self.offset_manager = offset_manager
        self.config = config or HealthCheckConfig()
        self.health_check_interval_seconds = health_check_interval_seconds
        
        # Health tracking
        self.status = HealthStatus.UNKNOWN
        self.health_issues: List[str] = []
        self.last_message_time: Optional[float] = None
        self.last_health_check_time: float = time.time()
        
        # Performance metrics
        self.messages_processed = 0
        self.messages_processed_last_check = 0
        self.errors_count = 0
        self.errors_last_check = 0
        self.processing_times: List[float] = []  # milliseconds
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Alert callbacks
        self._alert_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
    
    def record_message_processed(self, processing_time_ms: float) -> None:
        """
        Record a successfully processed message.
        
        Args:
            processing_time_ms: Time taken to process the message in milliseconds
        """
        with self._lock:
            self.messages_processed += 1
            self.last_message_time = time.time()
            self.processing_times.append(processing_time_ms)
            
            # Keep only the last 1000 processing times
            if len(self.processing_times) > 1000:
                self.processing_times = self.processing_times[-1000:]
    
    def record_error(self) -> None:
        """Record a message processing error."""
        with self._lock:
            self.errors_count += 1
    
    def add_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """
        Add a callback for health alerts.
        
        Args:
            callback: Function to call with alerts (severity, details)
        """
        with self._lock:
            self._alert_callbacks.append(callback)
    
    def run_health_check(self) -> HealthStatus:
        """
        Run a health check for the consumer.
        
        Returns:
            Current health status
        """
        with self._lock:
            now = time.time()
            
            # Only run health check at configured interval
            if now - self.last_health_check_time < self.health_check_interval_seconds:
                return self.status
                
            # Clear previous issues
            self.health_issues = []
            
            # Calculate time since last message
            idle_seconds = None
            # only skip when no timestamp at all (None), but include 0.0
            if self.last_message_time is not None:
                idle_seconds = now - self.last_message_time

                # Check for stalled consumer
                if idle_seconds > self.config.critical_idle_seconds:
                    self.health_issues.append(f"Consumer stalled for {idle_seconds:.1f} seconds")
                elif idle_seconds > self.config.max_idle_seconds:
                    self.health_issues.append(f"Consumer idle for {idle_seconds:.1f} seconds")

            
            # Calculate throughput
            seconds_since_last_check = now - self.last_health_check_time
            messages_since_last_check = self.messages_processed - self.messages_processed_last_check
            errors_since_last_check = self.errors_count - self.errors_last_check
            
            throughput = messages_since_last_check / seconds_since_last_check if seconds_since_last_check > 0 else 0
            
            # Check throughput
            if throughput < self.config.low_throughput_messages_per_second:
                self.health_issues.append(f"Low throughput: {throughput:.2f} messages/second")
            elif throughput < self.config.min_throughput_messages_per_second:
                self.health_issues.append(f"Below target throughput: {throughput:.2f} messages/second")
            
            # Calculate error rate
            total_attempts = messages_since_last_check + errors_since_last_check
            error_rate = (errors_since_last_check / total_attempts * 100) if total_attempts > 0 else 0
            
            # Check error rate
            if error_rate > self.config.critical_error_rate_percent:
                self.health_issues.append(f"Critical error rate: {error_rate:.1f}%")
            elif error_rate > self.config.max_error_rate_percent:
                self.health_issues.append(f"High error rate: {error_rate:.1f}%")
            
            # Check consumer lag if offset manager is available
            if self.offset_manager:
                lag_by_partition = self.offset_manager.get_consumer_lag()
                total_lag = sum(lag_by_partition.values())
                
                if total_lag > self.config.critical_lag_messages:
                    self.health_issues.append(f"Critical consumer lag: {total_lag} messages")
                elif total_lag > self.config.max_lag_messages:
                    self.health_issues.append(f"High consumer lag: {total_lag} messages")
            
            # Determine overall health status
            if not self.health_issues:
                self.status = HealthStatus.HEALTHY
            elif any("Critical" in issue for issue in self.health_issues):
                self.status = HealthStatus.UNHEALTHY
                
                # Send alerts for unhealthy status
                self._send_alerts("critical", {
                    "consumer_id": self.consumer_id,
                    "status": self.status.value,
                    "issues": self.health_issues,
                    "throughput": throughput,
                    "error_rate": error_rate,
                    "idle_seconds": idle_seconds
                })
            elif "stalled" in self.status.value.lower() or any("stalled" in issue.lower() for issue in self.health_issues):
                self.status = HealthStatus.STALLED
                
                # Send alerts for stalled status
                self._send_alerts("warning", {
                    "consumer_id": self.consumer_id,
                    "status": self.status.value,
                    "issues": self.health_issues,
                    "idle_seconds": idle_seconds
                })
            else:
                self.status = HealthStatus.DEGRADED
            
            # Update tracking variables for next check
            self.messages_processed_last_check = self.messages_processed
            self.errors_last_check = self.errors_count
            self.last_health_check_time = now
            
            # Log health status
            if self.status != HealthStatus.HEALTHY:
                logger.warning(f"Consumer health status: {self.status.value}, issues: {', '.join(self.health_issues)}")
            else:
                logger.info(f"Consumer health status: {self.status.value}")
            
            return self.status
    
    def _send_alerts(self, severity: str, details: Dict[str, Any]) -> None:
        """
        Send alerts to all registered callbacks.
        
        Args:
            severity: Alert severity ('critical', 'warning', 'info')
            details: Alert details
        """
        for callback in self._alert_callbacks:
            try:
                callback(severity, details)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def get_health_info(self) -> Dict[str, Any]:
        """
        Get detailed health information.
        
        Returns:
            Dictionary with health status and metrics
        """
        with self._lock:
            # Make sure health check is up to date
            self.run_health_check()
            
            # Calculate metrics
            avg_processing_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
            
            return {
                "consumer_id": self.consumer_id,
                "status": self.status.value,
                "issues": self.health_issues,
                "messages_processed": self.messages_processed,
                "errors_count": self.errors_count,
                "error_rate": (self.errors_count / (self.messages_processed + self.errors_count) * 100) 
                              if (self.messages_processed + self.errors_count) > 0 else 0,
                "avg_processing_time_ms": avg_processing_time,
                "last_message_time": self.last_message_time,
                "idle_seconds": time.time() - self.last_message_time if self.last_message_time else None,
                "last_health_check_time": self.last_health_check_time
            }


class HealthManager:
    """
    Manager for consumer health checks.
    
    Provides centralized health monitoring for multiple consumers.
    """
    
    def __init__(self):
        """Initialize a new health manager."""
        self._health_checks: Dict[str, ConsumerHealthCheck] = {}
        self._lock = threading.RLock()
        self._alert_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
    
    def register_consumer(
        self,
        consumer_id: str,
        consumer: Consumer,
        offset_manager: Optional[OffsetManager] = None,
        config: Optional[HealthCheckConfig] = None
    ) -> ConsumerHealthCheck:
        """
        Register a consumer for health monitoring.
        
        Args:
            consumer_id: Unique identifier for the consumer
            consumer: Kafka consumer instance
            offset_manager: Offset manager for lag tracking
            config: Health check configuration
            
        Returns:
            ConsumerHealthCheck instance
        """
        with self._lock:
            # Create health check
            health_check = ConsumerHealthCheck(
                consumer_id=consumer_id,
                consumer=consumer,
                offset_manager=offset_manager,
                config=config
            )
            
            # Add alert callbacks
            for callback in self._alert_callbacks:
                health_check.add_alert_callback(callback)
            
            # Store health check
            self._health_checks[consumer_id] = health_check
            
            return health_check
    
    def get_consumer_health(self, consumer_id: str) -> Optional[ConsumerHealthCheck]:
        """
        Get health check for a specific consumer.
        
        Args:
            consumer_id: Consumer identifier
            
        Returns:
            ConsumerHealthCheck instance or None if not found
        """
        with self._lock:
            return self._health_checks.get(consumer_id)
    
    def run_all_health_checks(self) -> Dict[str, HealthStatus]:
        """
        Run health checks for all registered consumers.
        
        Returns:
            Dictionary mapping consumer IDs to health statuses
        """
        results = {}
        with self._lock:
            for consumer_id, health_check in self._health_checks.items():
                results[consumer_id] = health_check.run_health_check()
        return results
    
    def get_overall_health(self) -> HealthStatus:
        """
        Get overall health status for all consumers.
        
        Returns:
            Worst health status among all consumers
        """
        with self._lock:
            if not self._health_checks:
                return HealthStatus.UNKNOWN
                
            # Run all health checks
            statuses = [check.run_health_check() for check in self._health_checks.values()]
            
            # Determine worst status
            if HealthStatus.UNHEALTHY in statuses:
                return HealthStatus.UNHEALTHY
            elif HealthStatus.STALLED in statuses:
                return HealthStatus.STALLED
            elif HealthStatus.DEGRADED in statuses:
                return HealthStatus.DEGRADED
            elif HealthStatus.UNKNOWN in statuses:
                return HealthStatus.UNKNOWN
            else:
                return HealthStatus.HEALTHY
    
    def add_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """
        Add a callback for health alerts.
        
        Args:
            callback: Function to call with alerts (severity, details)
        """
        with self._lock:
            self._alert_callbacks.append(callback)
            
            # Add to existing health checks
            for health_check in self._health_checks.values():
                health_check.add_alert_callback(callback)
    
    def get_health_summary(self) -> Dict[str, Any]:
        """
        Get a summary of health information for all consumers.
        
        Returns:
            Dictionary with health summary
        """
        with self._lock:
            overall_status = self.get_overall_health()
            consumer_statuses = {
                consumer_id: check.status.value
                for consumer_id, check in self._health_checks.items()
            }
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "overall_status": overall_status.value,
                "consumer_count": len(self._health_checks),
                "consumer_statuses": consumer_statuses,
                "unhealthy_consumers": [
                    consumer_id for consumer_id, check in self._health_checks.items()
                    if check.status in [HealthStatus.UNHEALTHY, HealthStatus.STALLED]
                ],
                "degraded_consumers": [
                    consumer_id for consumer_id, check in self._health_checks.items()
                    if check.status == HealthStatus.DEGRADED
                ]
            }
    
    def get_detailed_health_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed health information for all consumers.
        
        Returns:
            Dictionary mapping consumer IDs to health details
        """
        with self._lock:
            return {
                consumer_id: check.get_health_info()
                for consumer_id, check in self._health_checks.items()
            }


# Global health manager
_health_manager = HealthManager()


def get_health_manager() -> HealthManager:
    """
    Get the global health manager.
    
    Returns:
        Global HealthManager instance
    """
    return _health_manager