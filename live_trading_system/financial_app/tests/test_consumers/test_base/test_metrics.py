import unittest
import time
from unittest.mock import MagicMock

# Import the module to be tested
from app.consumers.base.metrics import (
    MetricSample,
    ConsumerMetrics,
    MetricsRegistry,
    metrics_registry,
    get_metrics_registry
)


class TestMetricSample(unittest.TestCase):
    """Test the MetricSample dataclass."""

    def test_metric_sample_initialization(self):
        # Test with just value
        sample1 = MetricSample(value=10.5)
        self.assertEqual(sample1.value, 10.5)
        self.assertIsInstance(sample1.timestamp, float)
        self.assertLessEqual(sample1.timestamp, time.time())

        # Test with both value and timestamp
        current_time = time.time()
        sample2 = MetricSample(value=20.5, timestamp=current_time)
        self.assertEqual(sample2.value, 20.5)
        self.assertEqual(sample2.timestamp, current_time)


class TestConsumerMetrics(unittest.TestCase):
    """Test the ConsumerMetrics class."""

    def setUp(self):
        """Set up a test ConsumerMetrics instance."""
        self.consumer_metrics = ConsumerMetrics(
            consumer_id="test-consumer-1",
            topic="test-topic",
            group_id="test-group"
        )

    def test_initialization(self):
        """Test initialization of ConsumerMetrics."""
        self.assertEqual(self.consumer_metrics.consumer_id, "test-consumer-1")
        self.assertEqual(self.consumer_metrics.topic, "test-topic")
        self.assertEqual(self.consumer_metrics.group_id, "test-group")
        self.assertEqual(self.consumer_metrics.messages_processed, 0)
        self.assertEqual(self.consumer_metrics.messages_failed, 0)
        self.assertEqual(self.consumer_metrics.bytes_processed, 0)
        self.assertEqual(len(self.consumer_metrics.processing_times_ms), 0)
        self.assertEqual(len(self.consumer_metrics.commit_times_ms), 0)
        self.assertIsNone(self.consumer_metrics.last_message_time)
        self.assertEqual(self.consumer_metrics.consumer_lag, {})
        self.assertFalse(self.consumer_metrics.is_running)

    def test_record_message_processed(self):
        """Test recording a processed message."""
        self.consumer_metrics.record_message_processed(size_bytes=1024, processing_time_ms=10.5)
        
        self.assertEqual(self.consumer_metrics.messages_processed, 1)
        self.assertEqual(self.consumer_metrics.bytes_processed, 1024)
        self.assertIsNotNone(self.consumer_metrics.last_message_time)
        self.assertEqual(len(self.consumer_metrics.processing_times_ms), 1)
        self.assertEqual(self.consumer_metrics.processing_times_ms[0].value, 10.5)

    def test_record_multiple_messages(self):
        """Test recording multiple messages."""
        for i in range(5):
            self.consumer_metrics.record_message_processed(
                size_bytes=1000 + i,
                processing_time_ms=10.0 + i
            )
        
        self.assertEqual(self.consumer_metrics.messages_processed, 5)
        self.assertEqual(self.consumer_metrics.bytes_processed, 5010)  # 1000+1001+1002+1003+1004
        self.assertEqual(len(self.consumer_metrics.processing_times_ms), 5)
        
        # Check that values were recorded correctly
        processing_times = [sample.value for sample in self.consumer_metrics.processing_times_ms]
        self.assertEqual(processing_times, [10.0, 11.0, 12.0, 13.0, 14.0])

    def test_record_message_failed(self):
        """Test recording a failed message."""
        self.consumer_metrics.record_message_failed()
        self.assertEqual(self.consumer_metrics.messages_failed, 1)
        
        self.consumer_metrics.record_message_failed()
        self.assertEqual(self.consumer_metrics.messages_failed, 2)

    def test_record_commit(self):
        """Test recording commit times."""
        self.consumer_metrics.record_commit(commit_time_ms=5.5)
        
        self.assertEqual(len(self.consumer_metrics.commit_times_ms), 1)
        self.assertEqual(self.consumer_metrics.commit_times_ms[0].value, 5.5)
        
        # Add another commit
        self.consumer_metrics.record_commit(commit_time_ms=6.5)
        self.assertEqual(len(self.consumer_metrics.commit_times_ms), 2)

    def test_update_consumer_lag(self):
        """Test updating consumer lag."""
        self.consumer_metrics.update_consumer_lag(partition=0, lag=100)
        self.assertEqual(self.consumer_metrics.consumer_lag, {0: 100})
        
        # Update existing partition
        self.consumer_metrics.update_consumer_lag(partition=0, lag=50)
        self.assertEqual(self.consumer_metrics.consumer_lag, {0: 50})
        
        # Add another partition
        self.consumer_metrics.update_consumer_lag(partition=1, lag=200)
        self.assertEqual(self.consumer_metrics.consumer_lag, {0: 50, 1: 200})

    def test_get_avg_processing_time_ms_no_data(self):
        """Test getting average processing time with no data."""
        avg_time = self.consumer_metrics.get_avg_processing_time_ms()
        self.assertIsNone(avg_time)

    def test_get_total_lag(self):
        """Test getting total consumer lag."""
        # Empty lag
        self.assertEqual(self.consumer_metrics.get_total_lag(), 0)
        
        # Add some lag values
        self.consumer_metrics.consumer_lag = {0: 100, 1: 200, 2: 300}
        self.assertEqual(self.consumer_metrics.get_total_lag(), 600)

    def test_list_size_limits(self):
        """Test that list sizes are limited to avoid memory growth."""
        # Mock time.time() to avoid creating 1000+ real timestamps
        original_time = time.time
        time.time = MagicMock(return_value=1000.0)
        
        try:
            # Add more than 1000 processing time samples
            for i in range(1100):
                self.consumer_metrics.record_message_processed(
                    size_bytes=100,
                    processing_time_ms=float(i)
                )
            
            # Should be limited to 1000
            self.assertEqual(len(self.consumer_metrics.processing_times_ms), 1000)
            
            # Add more than 100 commit time samples
            for i in range(110):
                self.consumer_metrics.record_commit(commit_time_ms=float(i))
            
            # Should be limited to 100
            self.assertEqual(len(self.consumer_metrics.commit_times_ms), 100)
        finally:
            # Restore original time function
            time.time = original_time


class TestMetricsRegistry(unittest.TestCase):
    """Test the MetricsRegistry class."""

    def setUp(self):
        """Set up a test MetricsRegistry instance."""
        self.registry = MetricsRegistry()

    def test_initialization(self):
        """Test initialization of MetricsRegistry."""
        self.assertEqual(len(self.registry._metrics), 0)

    def test_register_consumer(self):
        """Test registering a consumer."""
        metrics = self.registry.register_consumer(
            consumer_id="test-consumer-1",
            topic="test-topic",
            group_id="test-group"
        )
        
        self.assertIsInstance(metrics, ConsumerMetrics)
        self.assertEqual(metrics.consumer_id, "test-consumer-1")
        self.assertEqual(metrics.topic, "test-topic")
        self.assertEqual(metrics.group_id, "test-group")
        
        # Should be in the registry
        self.assertIn("test-consumer-1", self.registry._metrics)
        self.assertEqual(self.registry._metrics["test-consumer-1"], metrics)

    def test_get_consumer_metrics(self):
        """Test getting metrics for a specific consumer."""
        # Register a consumer
        original_metrics = self.registry.register_consumer(
            consumer_id="test-consumer-1",
            topic="test-topic",
            group_id="test-group"
        )
        
        # Get the metrics
        retrieved_metrics = self.registry.get_consumer_metrics("test-consumer-1")
        self.assertEqual(retrieved_metrics, original_metrics)
        
        # Non-existent consumer
        self.assertIsNone(self.registry.get_consumer_metrics("non-existent"))

    def test_get_all_metrics(self):
        """Test getting metrics for all consumers."""
        # Register a few consumers
        metrics1 = self.registry.register_consumer(
            consumer_id="consumer1",
            topic="topic1",
            group_id="group1"
        )
        metrics2 = self.registry.register_consumer(
            consumer_id="consumer2",
            topic="topic2",
            group_id="group2"
        )
        
        # Record some activity
        metrics1.record_message_processed(size_bytes=100, processing_time_ms=10.0)
        metrics2.record_message_processed(size_bytes=200, processing_time_ms=20.0)
        
        # Get all metrics
        all_metrics = self.registry.get_all_metrics()
        
        # Should have entries for both consumers
        self.assertIn("consumer1", all_metrics)
        self.assertIn("consumer2", all_metrics)
        
        # Check some values
        self.assertEqual(all_metrics["consumer1"]["topic"], "topic1")
        self.assertEqual(all_metrics["consumer1"]["messages_processed"], 1)
        self.assertEqual(all_metrics["consumer2"]["topic"], "topic2")
        self.assertEqual(all_metrics["consumer2"]["messages_processed"], 1)

    def test_get_all_metrics_by_topic(self):
        """Test getting metrics grouped by topic."""
        # Register a few consumers with some shared topics
        self.registry.register_consumer(
            consumer_id="consumer1",
            topic="topic1",
            group_id="group1"
        )
        self.registry.register_consumer(
            consumer_id="consumer2",
            topic="topic2",
            group_id="group2"
        )
        self.registry.register_consumer(
            consumer_id="consumer3",
            topic="topic1",  # Same as consumer1
            group_id="group3"
        )
        
        # Get metrics by topic
        metrics_by_topic = self.registry.get_all_metrics_by_topic()
        
        # Should have entries for both topics
        self.assertIn("topic1", metrics_by_topic)
        self.assertIn("topic2", metrics_by_topic)
        
        # topic1 should have 2 consumers, topic2 should have 1
        self.assertEqual(len(metrics_by_topic["topic1"]), 2)
        self.assertEqual(len(metrics_by_topic["topic2"]), 1)


class TestGlobalRegistry(unittest.TestCase):
    """Test the global metrics registry."""

    def test_global_registry_instance(self):
        """Test that we have a global metrics registry instance."""
        self.assertIsInstance(metrics_registry, MetricsRegistry)
        
        # get_metrics_registry should return the same instance
        self.assertEqual(get_metrics_registry(), metrics_registry)


if __name__ == "__main__":
    unittest.main()