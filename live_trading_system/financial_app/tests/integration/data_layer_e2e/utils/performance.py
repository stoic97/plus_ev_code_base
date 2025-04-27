"""
Utility functions for measuring performance during end-to-end tests.
"""

import time
import logging
import statistics
from typing import Dict, List, Any, Optional, Callable
from functools import wraps
from contextlib import contextmanager

from financial_app.tests.integration.data_layer_e2e.e2e_config import PERFORMANCE_THRESHOLDS

logger = logging.getLogger(__name__)

class PerformanceTracker:
    """Track performance metrics during testing."""
    
    def __init__(self):
        self.metrics = {}
        self.current_timers = {}
    
    def record_metric(self, metric_name: str, value: float) -> None:
        """
        Record a performance metric.
        
        Args:
            metric_name: Name of the metric
            value: Value to record
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        
        self.metrics[metric_name].append(value)
        
        # Log if value exceeds threshold
        threshold_key = f"max_{metric_name}"
        if threshold_key in PERFORMANCE_THRESHOLDS:
            threshold = PERFORMANCE_THRESHOLDS[threshold_key]
            if value > threshold:
                logger.warning(f"Performance metric {metric_name} exceeded threshold: {value} > {threshold}")
    
    def start_timer(self, timer_name: str) -> None:
        """
        Start a timer for a specific operation.
        
        Args:
            timer_name: Name of the timer
        """
        self.current_timers[timer_name] = time.time()
    
    def stop_timer(self, timer_name: str) -> float:
        """
        Stop a timer and record the duration.
        
        Args:
            timer_name: Name of the timer
            
        Returns:
            float: Duration in milliseconds
        """
        if timer_name not in self.current_timers:
            logger.warning(f"Timer {timer_name} was not started")
            return 0
        
        duration_ms = (time.time() - self.current_timers[timer_name]) * 1000
        self.record_metric(f"{timer_name}_ms", duration_ms)
        del self.current_timers[timer_name]
        
        return duration_ms
    
    def get_metric_stats(self, metric_name: str) -> Dict[str, float]:
        """
        Get statistics for a specific metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            dict: Statistics for the metric
        """
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return {
                "count": 0,
                "min": 0,
                "max": 0,
                "mean": 0,
                "median": 0,
                "p95": 0,
                "p99": 0
            }
        
        values = self.metrics[metric_name]
        sorted_values = sorted(values)
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p95": sorted_values[int(0.95 * len(sorted_values))],
            "p99": sorted_values[int(0.99 * len(sorted_values))]
        }
    
    def get_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for all recorded metrics.
        
        Returns:
            dict: Statistics for all metrics
        """
        return {metric: self.get_metric_stats(metric) for metric in self.metrics}
    
    def check_thresholds(self) -> List[Dict[str, Any]]:
        """
        Check if any metrics exceed their thresholds.
        
        Returns:
            list: List of threshold violations
        """
        violations = []
        
        for metric_name, values in self.metrics.items():
            threshold_key = f"max_{metric_name}"
            if threshold_key in PERFORMANCE_THRESHOLDS:
                threshold = PERFORMANCE_THRESHOLDS[threshold_key]
                stats = self.get_metric_stats(metric_name)
                
                if stats["max"] > threshold:
                    violations.append({
                        "metric": metric_name,
                        "threshold": threshold,
                        "max_value": stats["max"],
                        "mean_value": stats["mean"],
                        "p95_value": stats["p95"]
                    })
        
        return violations


# Context manager for timing operations
@contextmanager
def timed_operation(tracker: PerformanceTracker, operation_name: str):
    """
    Context manager for timing operations.
    
    Args:
        tracker: Performance tracker instance
        operation_name: Name of the operation
    """
    tracker.start_timer(operation_name)
    try:
        yield
    finally:
        duration = tracker.stop_timer(operation_name)
        logger.debug(f"{operation_name} took {duration:.2f}ms")


# Decorator for timing function calls
def timed_function(tracker: PerformanceTracker, operation_name: Optional[str] = None):
    """
    Decorator for timing function calls.
    
    Args:
        tracker: Performance tracker instance
        operation_name: Optional name for the operation (defaults to function name)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = operation_name or func.__name__
            tracker.start_timer(func_name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = tracker.stop_timer(func_name)
                logger.debug(f"{func_name} took {duration:.2f}ms")
        return wrapper
    return decorator


# Batch performance measuring
def measure_batch_processing(tracker: PerformanceTracker, 
                           batch_size: int, 
                           total_items: int, 
                           processing_time_ms: float,
                           operation_name: str = "batch_processing") -> None:
    """
    Record metrics for batch processing operations.
    
    Args:
        tracker: Performance tracker instance
        batch_size: Size of the processed batch
        total_items: Total number of items processed so far
        processing_time_ms: Time taken to process the batch in milliseconds
        operation_name: Name for the operation
    """
    # Record total processing time
    tracker.record_metric(f"{operation_name}_total_ms", processing_time_ms)
    
    # Record per-item processing time
    per_item_ms = processing_time_ms / batch_size if batch_size > 0 else 0
    tracker.record_metric(f"{operation_name}_per_item_ms", per_item_ms)
    
    # Record throughput (items per second)
    throughput = (batch_size / processing_time_ms) * 1000 if processing_time_ms > 0 else 0
    tracker.record_metric(f"{operation_name}_throughput", throughput)
    
    logger.debug(f"Batch processing: {batch_size} items in {processing_time_ms:.2f}ms " 
                f"({per_item_ms:.2f}ms/item, {throughput:.2f} items/sec)")


# Function for measuring latency between events
def measure_latency(tracker: PerformanceTracker, 
                   start_time: float, 
                   end_time: float, 
                   metric_name: str = "latency") -> float:
    """
    Measure and record latency between two events.
    
    Args:
        tracker: Performance tracker instance
        start_time: Start timestamp
        end_time: End timestamp
        metric_name: Name for the latency metric
        
    Returns:
        float: Latency in milliseconds
    """
    latency_ms = (end_time - start_time) * 1000
    tracker.record_metric(f"{metric_name}_ms", latency_ms)
    
    # Check if latency exceeds threshold
    threshold_key = f"max_{metric_name}_ms"
    if threshold_key in PERFORMANCE_THRESHOLDS:
        threshold = PERFORMANCE_THRESHOLDS[threshold_key]
        if latency_ms > threshold:
            logger.warning(f"Latency {metric_name} exceeded threshold: {latency_ms:.2f}ms > {threshold}ms")
    
    return latency_ms


# Function to measure database query performance
def measure_db_query(tracker: PerformanceTracker, 
                    query_name: str,
                    rows_returned: int,
                    query_time_ms: float) -> None:
    """
    Record metrics for database query performance.
    
    Args:
        tracker: Performance tracker instance
        query_name: Name of the query
        rows_returned: Number of rows returned by the query
        query_time_ms: Time taken to execute the query in milliseconds
    """
    tracker.record_metric(f"db_{query_name}_time_ms", query_time_ms)
    
    # Calculate and record per-row time if applicable
    if rows_returned > 0:
        per_row_ms = query_time_ms / rows_returned
        tracker.record_metric(f"db_{query_name}_per_row_ms", per_row_ms)
    
    logger.debug(f"DB query '{query_name}': {rows_returned} rows in {query_time_ms:.2f}ms")