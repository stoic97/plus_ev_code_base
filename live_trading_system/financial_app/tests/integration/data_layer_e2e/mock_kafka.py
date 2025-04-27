"""
Mock Kafka implementations for testing when real Kafka is unavailable.
This provides fallback implementations that mimic the behavior of real Kafka components.
"""

import time
import random
import logging
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple

from financial_app.tests.integration.data_layer_e2e.e2e_config import TEST_INSTRUMENTS

logger = logging.getLogger(__name__)

# In-memory storage for mock Kafka
class MockKafkaStorage:
    """Singleton class to store Kafka messages and state across mock instances"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MockKafkaStorage, cls).__new__(cls)
            cls._instance.topics = {}  # Dict mapping topics to lists of messages
            cls._instance.consumer_groups = set()  # Set of consumer group IDs
            cls._instance.group_offsets = {}  # Dict mapping (group_id, topic, partition) to offset
            cls._instance.lock = threading.RLock()  # Thread-safe operations
        return cls._instance
    
    def add_message(self, topic: str, message: Dict[str, Any], partition: int = 0) -> int:
        """
        Add a message to a topic and return its offset
        
        Args:
            topic: Topic to add message to
            message: Message to add
            partition: Partition to add message to
            
        Returns:
            int: Offset of the new message
        """
        with self.lock:
            if topic not in self.topics:
                self.topics[topic] = {partition: []}
            
            if partition not in self.topics[topic]:
                self.topics[topic][partition] = []
            
            self.topics[topic][partition].append(message)
            return len(self.topics[topic][partition]) - 1
    
    def get_messages(self, topic: str, partition: int, start_offset: int, max_count: int = 10) -> List[Dict[str, Any]]:
        """
        Get messages from a topic starting at a specific offset
        
        Args:
            topic: Topic to get messages from
            partition: Partition to get messages from
            start_offset: Offset to start at
            max_count: Maximum number of messages to return
            
        Returns:
            List of messages
        """
        with self.lock:
            if topic not in self.topics or partition not in self.topics[topic]:
                return []
            
            messages = self.topics[topic][partition]
            end_offset = min(start_offset + max_count, len(messages))
            
            if start_offset >= end_offset:
                return []
                
            return messages[start_offset:end_offset]
    
    def register_consumer_group(self, group_id: str) -> None:
        """Register a consumer group"""
        with self.lock:
            self.consumer_groups.add(group_id)
    
    def get_consumer_groups(self) -> Set[str]:
        """Get all registered consumer groups"""
        with self.lock:
            return self.consumer_groups.copy()
    
    def get_offset(self, group_id: str, topic: str, partition: int) -> int:
        """Get current offset for a consumer group on a topic/partition"""
        with self.lock:
            key = (group_id, topic, partition)
            return self.group_offsets.get(key, 0)
    
    def set_offset(self, group_id: str, topic: str, partition: int, offset: int) -> None:
        """Set offset for a consumer group on a topic/partition"""
        with self.lock:
            key = (group_id, topic, partition)
            self.group_offsets[key] = offset
    
    def get_end_offset(self, topic: str, partition: int) -> int:
        """Get the end offset (length) of a topic/partition"""
        with self.lock:
            if topic not in self.topics or partition not in self.topics[topic]:
                return 0
            return len(self.topics[topic][partition])
    
    def get_beginning_offset(self, topic: str, partition: int) -> int:
        """Get the beginning offset of a topic/partition (always 0)"""
        return 0
    
    def get_partitions(self, topic: str) -> List[int]:
        """Get all partitions for a topic"""
        with self.lock:
            if topic not in self.topics:
                return [0]  # Default partition
            return list(self.topics[topic].keys())


# Mock Kafka Producer implementation
class MockProducer:
    """Mock implementation of Kafka Producer"""
    
    def __init__(self, bootstrap_servers=None, **kwargs):
        self.bootstrap_servers = bootstrap_servers
        self.storage = MockKafkaStorage()
        self.value_serializer = kwargs.get('value_serializer', lambda v: v)
        self.key_serializer = kwargs.get('key_serializer', lambda k: k)
        self.closed = False
        logger.info("Initialized MockProducer")
    
    def send(self, topic: str, value, key=None, partition=None, timestamp_ms=None, headers=None) -> 'MockFuture':
        """
        Send a message to the specified topic
        
        Args:
            topic: Topic to send to
            value: Message value
            key: Message key
            partition: Specific partition (default: random)
            timestamp_ms: Message timestamp in ms
            headers: Message headers
            
        Returns:
            MockFuture: Future object for the send operation
        """
        if self.closed:
            raise Exception("Producer is closed")
        
        # Apply serializers
        serialized_value = self.value_serializer(value)
        serialized_key = self.key_serializer(key) if key is not None else None
        
        # Choose a partition if not specified
        if partition is None:
            partitions = self.storage.get_partitions(topic)
            partition = random.choice(partitions) if partitions else 0
        
        # Add to storage
        offset = self.storage.add_message(topic, serialized_value, partition)
        
        # Return a future that's already done
        return MockFuture(topic, partition, offset)
    
    def flush(self) -> None:
        """Flush all messages (no-op in mock)"""
        pass
    
    def close(self) -> None:
        """Close the producer"""
        self.closed = True


# Mock Kafka Consumer implementation
class MockConsumerManager:
    """Mock implementation of Kafka Consumer Manager"""
    
    def __init__(self, bootstrap_servers=None, group_id=None, topics=None, **kwargs):
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.topics = topics or []
        self.storage = MockKafkaStorage()
        self.auto_offset_reset = kwargs.get('auto_offset_reset', 'latest')
        
        self.running = False
        self.current_offsets = {0: 0, 1: 0}  # Start with two partitions
        self.assigned_partitions = {}  # Dict mapping topic to list of partitions
        self.error_count = 0
        
        # Register the consumer group
        if group_id:
            self.storage.register_consumer_group(group_id)
        
        # Initialize with some data in the partition
        for topic in self.topics:
            self._ensure_topic_has_data(topic)
                
        logger.info(f"Initialized MockConsumerManager with group_id={group_id}, topics={topics}")
    
    def _ensure_topic_has_data(self, topic):
        """Make sure topic has some test data"""
        for i in range(5):  # Add some initial messages to each partition
            instrument = random.choice(TEST_INSTRUMENTS)
            message = {
                'symbol': instrument['symbol'],
                'exchange': instrument['exchange'],
                'timestamp': datetime.now().isoformat(),
                'price': round(random.uniform(100, 1000), 2),
                'volume': random.randint(100, 10000)
            }
            self.storage.add_message(topic, message, partition=0)
            self.storage.add_message(topic, message, partition=1)
    
    def start(self):
        """Start the consumer"""
        self.running = True
        logger.info("MockConsumerManager started")
        return True
    
    def stop(self):
        """Stop the consumer"""
        self.running = False
        logger.info("MockConsumerManager stopped")
        return True
    
    def poll(self, timeout_ms=1000, max_records=10):
        """
        Poll for messages
        
        Args:
            timeout_ms: Poll timeout in milliseconds
            max_records: Maximum number of records to return
            
        Returns:
            List of messages
        """
        if not self.running:
            return []
        
        # Simulate some delay
        time.sleep(0.01)
        
        # Generate some mock messages
        num_messages = random.randint(1, max_records)
        messages = []
        
        for _ in range(num_messages):
            instrument = random.choice(TEST_INSTRUMENTS)
            message = {
                'symbol': instrument['symbol'],
                'exchange': instrument['exchange'],
                'timestamp': datetime.now().isoformat(),
                'price': round(random.uniform(100, 1000), 2),
                'volume': random.randint(100, 10000)
            }
            
            # 5% chance of creating a malformed message for error testing
            if random.random() < 0.05:
                # Create various types of malformed messages
                errors = [
                    lambda m: m.pop('timestamp'),  # Missing timestamp
                    lambda m: m.update(price="not_a_number"),  # Wrong data type
                    lambda m: m.update(volume=-100),  # Invalid value
                ]
                random.choice(errors)(message)
                self.error_count += 1
                
            messages.append(message)
            
            # Update mock offsets - randomly choose a partition
            partition = random.choice([0, 1])
            self.current_offsets[partition] += 1
        
        return messages
    
    def get_error_count(self):
        """Get the number of error messages generated"""
        return self.error_count
    
    def get_current_offsets(self):
        """
        Get current offsets by partition
        
        Returns:
            Dict mapping partition IDs to offsets
        """
        # Add random increment to make sure offsets change
        for partition in self.current_offsets:
            self.current_offsets[partition] += random.randint(1, 5)
        return self.current_offsets.copy()
    
    def list_consumer_groups(self):
        """List all consumer groups"""
        return list(self.storage.get_consumer_groups())


# Mock Future for async operations
class MockFuture:
    """Mock implementation of Kafka Future"""
    
    def __init__(self, topic: str, partition: int, offset: int):
        self.topic = topic
        self.partition = partition
        self.offset = offset
        self._success = True
        self._exception = None
    
    def get(self, timeout=None):
        """Get the result of the future"""
        if self._exception:
            raise self._exception
        return self._create_record_metadata()
    
    def _create_record_metadata(self):
        """Create a mock record metadata object"""
        return MockRecordMetadata(
            topic=self.topic,
            partition=self.partition,
            offset=self.offset
        )
    
    def succeeded(self):
        """Check if the future succeeded"""
        return self._success
    
    def exception(self):
        """Get the exception if the future failed"""
        return self._exception
    
    def add_callback(self, callback):
        """Add a callback to be called when the future completes"""
        if callable(callback):
            callback(self._create_record_metadata())
    
    def add_errback(self, errback):
        """Add an errback to be called if the future fails"""
        pass  # No-op since our mock always succeeds


# Mock RecordMetadata for producer results
class MockRecordMetadata:
    """Mock implementation of Kafka RecordMetadata"""
    
    def __init__(self, topic: str, partition: int, offset: int):
        self.topic = topic
        self.partition = partition
        self.offset = offset
        self.timestamp = int(time.time() * 1000)
    
    def __str__(self):
        return f"RecordMetadata(topic={self.topic}, partition={self.partition}, offset={self.offset})"


# Helper function to generate mock market data
def generate_mock_market_data(num_messages: int = 10, include_errors: bool = False) -> List[Dict[str, Any]]:
    """
    Generate mock market data messages
    
    Args:
        num_messages: Number of messages to generate
        include_errors: Whether to include malformed messages
        
    Returns:
        List of market data messages
    """
    messages = []
    
    for _ in range(num_messages):
        # Choose a random instrument
        instrument = random.choice(TEST_INSTRUMENTS)
        
        # Generate a valid message
        message = {
            'symbol': instrument['symbol'],
            'exchange': instrument['exchange'],
            'timestamp': datetime.now().isoformat(),
            'price': round(random.uniform(100, 1000), 2),
            'volume': random.randint(100, 10000),
            'bid': round(random.uniform(100, 1000), 2),
            'ask': round(random.uniform(100, 1000), 2),
            'data_source': 'mock'
        }
        
        # Add to list
        messages.append(message)
    
    # If requested, include some errors
    if include_errors and num_messages > 5:
        # Choose a few messages to make malformed
        for i in random.sample(range(num_messages), num_messages // 5):
            error_types = [
                lambda m: m.pop('timestamp', None),  # Missing required field
                lambda m: m.update(price="invalid"),  # Invalid data type
                lambda m: m.update(volume=-100),      # Invalid value
            ]
            random.choice(error_types)(messages[i])
    
    return messages


# Function to publish mock market data to the mock Kafka
def publish_mock_market_data(
    topic: str = 'market_data',
    num_messages: int = 100,
    include_errors: bool = True
) -> int:
    """
    Publish mock market data to the mock Kafka
    
    Args:
        topic: Topic to publish to
        num_messages: Number of messages to publish
        include_errors: Whether to include malformed messages
        
    Returns:
        int: Number of messages published
    """
    storage = MockKafkaStorage()
    
    # Generate the messages
    messages = generate_mock_market_data(num_messages, include_errors)
    
    # Publish to storage
    for message in messages:
        partition = random.randint(0, 2)  # Use 3 partitions for variety
        storage.add_message(topic, message, partition)
    
    logger.info(f"Published {num_messages} mock market data messages to {topic}")
    return num_messages