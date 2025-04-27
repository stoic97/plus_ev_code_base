"""
Integration test for Kafka connectivity.
Verifies the application can connect to Kafka brokers and access the required topics.
"""
import os
import unittest
import pytest
from confluent_kafka import Producer, Consumer, KafkaException, KafkaError
from confluent_kafka.admin import AdminClient, NewTopic
import time
import uuid
import logging
from app.core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestKafkaConnectivity(unittest.TestCase):
    """Test Kafka connection, topic access, and basic functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test class by creating Kafka admin client and test topic."""
        # Get Kafka configuration from settings
        cls.bootstrap_servers = settings.kafka.BOOTSTRAP_SERVERS
        cls.market_data_topic = settings.kafka.MARKET_DATA_TOPIC
        cls.signal_topic = "market-data-signal"
        cls.group_id = f"test-group-{uuid.uuid4()}"  # Unique group ID for tests
        
        # Create admin client for topic management
        cls.admin_client = AdminClient({'bootstrap.servers': ','.join(cls.bootstrap_servers)})
        
        # Create test topics if they don't exist
        cls.test_topic = f"test-topic-{uuid.uuid4()}"
        cls._create_test_topic()
        
        # Allow time for topic creation
        time.sleep(2)
        
    @classmethod
    def tearDownClass(cls):
        """Clean up by deleting test topics."""
        try:
            cls._delete_test_topic()
        except Exception as e:
            logger.warning(f"Error cleaning up test topic: {e}")
    
    @classmethod
    def _create_test_topic(cls):
        """Create a test topic for integration testing."""
        try:
            # Create new topic with 1 partition and replication factor 1
            topic = NewTopic(
                cls.test_topic,
                num_partitions=1,
                replication_factor=1
            )
            
            # Create the topic
            futures = cls.admin_client.create_topics([topic])
            
            # Wait for operation to complete
            for topic_name, future in futures.items():
                try:
                    future.result()  # Block until topic is created
                    logger.info(f"Test topic {topic_name} created successfully")
                except Exception as e:
                    # Topic might already exist
                    if "already exists" in str(e):
                        logger.info(f"Test topic {topic_name} already exists")
                    else:
                        logger.error(f"Failed to create topic {topic_name}: {e}")
                        raise
        except Exception as e:
            logger.error(f"Error creating test topic: {e}")
            raise
    
    @classmethod
    def _delete_test_topic(cls):
        """Delete the test topic after tests complete."""
        try:
            # Delete the test topic
            futures = cls.admin_client.delete_topics([cls.test_topic])
            
            # Wait for operation to complete
            for topic_name, future in futures.items():
                try:
                    future.result()  # Block until topic is deleted
                    logger.info(f"Test topic {topic_name} deleted successfully")
                except Exception as e:
                    logger.error(f"Failed to delete topic {topic_name}: {e}")
        except Exception as e:
            logger.error(f"Error deleting test topic: {e}")
            raise
    
    def test_kafka_broker_connection(self):
        """Test basic connectivity to Kafka brokers."""
        try:
            # Create a simple producer to test connection
            producer_config = {
                'bootstrap.servers': ','.join(self.bootstrap_servers),
                'client.id': f'test-client-{uuid.uuid4()}'
            }
            
            producer = Producer(producer_config)
            
            # Simple API call to verify connection
            metadata = producer.list_topics(timeout=10)
            
            # If we get here, connection was successful
            self.assertIsNotNone(metadata, "Failed to get metadata from Kafka")
            logger.info("Successfully connected to Kafka brokers")
        except KafkaException as e:
            self.fail(f"Failed to connect to Kafka brokers: {e}")
    
    def test_topic_accessibility(self):
        """Test that configured topics are accessible."""
        try:
            producer_config = {
                'bootstrap.servers': ','.join(self.bootstrap_servers),
                'client.id': f'test-client-{uuid.uuid4()}'
            }
            
            producer = Producer(producer_config)
            
            # Get metadata to check for topics
            metadata = producer.list_topics(timeout=10)
            
            # Check configured topics
            topics_to_check = [self.market_data_topic, self.signal_topic]
            
            for topic in topics_to_check:
                topic_metadata = metadata.topics.get(topic)
                self.assertIsNotNone(
                    topic_metadata,
                    f"Topic {topic} does not exist or is not accessible"
                )
                logger.info(f"Successfully verified access to topic: {topic}")
        except KafkaException as e:
            self.fail(f"Failed to access Kafka topics: {e}")
    
    def test_produce_consume_message(self):
        """Test producing and consuming a message from the test topic."""
        # Create producer configuration
        producer_config = {
            'bootstrap.servers': ','.join(self.bootstrap_servers),
            'client.id': f'test-producer-{uuid.uuid4()}'
        }
        
        # Create consumer configuration
        consumer_config = {
            'bootstrap.servers': ','.join(self.bootstrap_servers),
            'group.id': self.group_id,
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': True
        }
        
        # Test message
        test_message = f"Test message {uuid.uuid4()}"
        message_received = False
        
        try:
            # Create producer and consumer
            producer = Producer(producer_config)
            consumer = Consumer(consumer_config)
            
            # Subscribe consumer to test topic
            consumer.subscribe([self.test_topic])
            
            # Produce message
            producer.produce(self.test_topic, test_message.encode('utf-8'))
            producer.flush()
            
            # Try to consume the message with timeout
            start_time = time.time()
            timeout = 30  # 30 seconds timeout
            
            while time.time() - start_time < timeout and not message_received:
                msg = consumer.poll(1.0)
                
                if msg is None:
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        logger.info("Reached end of partition")
                    else:
                        logger.error(f"Consumer error: {msg.error()}")
                else:
                    # Decode and check the message
                    received_message = msg.value().decode('utf-8')
                    if received_message == test_message:
                        message_received = True
                        logger.info(f"Successfully received message: {received_message}")
                        break
            
            self.assertTrue(message_received, "Failed to receive the produced message")
        
        except Exception as e:
            self.fail(f"Error in produce-consume test: {e}")
        
        finally:
            # Clean up resources
            try:
                if 'consumer' in locals():
                    consumer.close()
            except Exception as e:
                logger.warning(f"Error closing consumer: {e}")


if __name__ == "__main__":
    unittest.main()