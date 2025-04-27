"""
Utilities for Kafka testing.

This module provides helper functions for setting up Kafka test environments,
including topic creation and verification.
"""

import logging
import time
from typing import List, Dict, Any
from confluent_kafka.admin import AdminClient, NewTopic
from confluent_kafka import KafkaException

# Set up logging
logger = logging.getLogger(__name__)


def ensure_topics_exist(
    bootstrap_servers: List[str],
    topic_names: List[str],
    num_partitions: int = 1,
    replication_factor: int = 1,
    max_retries: int = 3
) -> bool:
    """
    Ensure that Kafka topics exist, creating them if necessary.
    
    Args:
        bootstrap_servers: List of Kafka bootstrap servers
        topic_names: List of topic names to ensure exist
        num_partitions: Number of partitions for each topic
        replication_factor: Replication factor for each topic
        max_retries: Maximum number of retries
        
    Returns:
        True if all topics exist or were created, False otherwise
    """
    # Create admin client
    admin_client = AdminClient({
        'bootstrap.servers': ','.join(bootstrap_servers)
    })
    
    # Check which topics already exist
    try:
        cluster_metadata = admin_client.list_topics(timeout=10)
        existing_topics = set(cluster_metadata.topics.keys())
        logger.info(f"Existing topics: {existing_topics}")
        
        topics_to_create = [name for name in topic_names if name not in existing_topics]
        
        if not topics_to_create:
            logger.info("All topics already exist")
            return True
        
        logger.info(f"Topics to create: {topics_to_create}")
    except Exception as e:
        logger.error(f"Error checking existing topics: {e}")
        return False
    
    # Create new topics
    new_topics = [
        NewTopic(
            name,
            num_partitions=num_partitions,
            replication_factor=replication_factor
        )
        for name in topics_to_create
    ]
    
    retry_count = 0
    while retry_count < max_retries:
        try:
            futures = admin_client.create_topics(new_topics)
            
            # Wait for topic creation to complete
            for topic_name, future in futures.items():
                try:
                    future.result()  # Block until topic is created
                    logger.info(f"Topic '{topic_name}' created successfully")
                except KafkaException as e:
                    if "already exists" in str(e):
                        logger.info(f"Topic '{topic_name}' already exists")
                    else:
                        logger.error(f"Failed to create topic '{topic_name}': {e}")
                        retry_count += 1
                        break
            
            # Verify all topics exist
            cluster_metadata = admin_client.list_topics(timeout=10)
            existing_topics = set(cluster_metadata.topics.keys())
            missing_topics = [name for name in topic_names if name not in existing_topics]
            
            if not missing_topics:
                logger.info(f"All topics exist: {topic_names}")
                return True
            
            logger.warning(f"Some topics still missing: {missing_topics}")
            retry_count += 1
            time.sleep(1)  # Wait before retrying
            
        except Exception as e:
            logger.error(f"Error creating topics: {e}")
            retry_count += 1
            time.sleep(1)  # Wait before retrying
    
    logger.error(f"Failed to ensure topics exist after {max_retries} retries")
    return False