"""
Base Kinesis producer for AWS Kinesis integration.

This module provides a base class for Kinesis producers with common functionality
for publishing messages to Kinesis streams.
"""

import logging
import json
import time
from typing import Any, Dict, Optional, Union, List
from datetime import datetime

import boto3
from botocore.exceptions import ClientError

from app.producers.base.error import SerializationError, PublishingError
from app.producers.base.metrics import get_metrics_registry
from app.producers.config.settings import KinesisSettings
from app.producers.utils.serialization import serialize_json

# Set up logging
logger = logging.getLogger(__name__)

class BaseKinesisProducer:
    """
    Base class for Kinesis producers.
    
    Provides common functionality for publishing messages to Kinesis streams.
    """
    
    def __init__(
        self,
        stream_name: str,
        settings: Optional[KinesisSettings] = None,
        region_name: Optional[str] = None,
        batch_size: int = 100,
        batch_timeout_ms: int = 1000
    ):
        """
        Initialize a new Kinesis producer.
        
        Args:
            stream_name: Name of the Kinesis stream
            settings: Kinesis configuration settings
            region_name: AWS region name
            batch_size: Number of records to process in a batch
            batch_timeout_ms: Maximum time to wait for a full batch in ms
        """
        self.settings = settings or KinesisSettings()
        self.stream_name = stream_name
        self.region_name = region_name or self.settings.AWS_REGION
        self.batch_size = batch_size
        self.batch_timeout_ms = batch_timeout_ms
        
        # Initialize Kinesis client
        self._client = boto3.client(
            'kinesis',
            region_name=self.region_name,
            aws_access_key_id=self.settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=self.settings.AWS_SECRET_ACCESS_KEY
        )
        
        # Set up metrics
        self.metrics = get_metrics_registry().register_producer(
            producer_id=f"kinesis_{stream_name}",
            topic=stream_name
        )
        
        logger.info(f"Initialized Kinesis producer for stream '{stream_name}'")
    
    def _serialize_message(self, data: Dict[str, Any]) -> bytes:
        """
        Serialize message data to bytes.
        
        Args:
            data: Message data to serialize
            
        Returns:
            Serialized data as bytes
            
        Raises:
            SerializationError: If the data cannot be serialized
        """
        try:
            return serialize_json(data)
        except Exception as e:
            raise SerializationError(f"Failed to serialize message: {e}")
    
    def publish_message(
        self,
        data: Dict[str, Any],
        partition_key: Optional[str] = None,
        explicit_hash_key: Optional[str] = None
    ) -> None:
        """
        Publish a message to Kinesis.
        
        Args:
            data: Message data to publish
            partition_key: Partition key for the record
            explicit_hash_key: Explicit hash key for the record
            
        Raises:
            PublishingError: If publishing fails
        """
        try:
            # Serialize message
            serialized = self._serialize_message(data)
            
            # Prepare record
            record = {
                'Data': serialized,
                'StreamName': self.stream_name
            }
            
            # Add partition key if provided
            if partition_key:
                record['PartitionKey'] = partition_key
            
            # Add explicit hash key if provided
            if explicit_hash_key:
                record['ExplicitHashKey'] = explicit_hash_key
            
            # Publish to Kinesis
            start_time = time.time()
            response = self._client.put_record(**record)
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Update metrics
            self.metrics.record_messages_published(1, processing_time_ms)
            
            logger.debug(
                f"Published message to stream '{self.stream_name}' "
                f"with shard ID '{response['ShardId']}'"
            )
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logger.error(f"Failed to publish message: {error_code} - {error_message}")
            self.metrics.record_message_failed()
            raise PublishingError(f"Failed to publish message: {error_message}")
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
            self.metrics.record_message_failed()
            raise PublishingError(f"Failed to publish message: {e}")
    
    def publish_batch(
        self,
        records: List[Dict[str, Any]],
        partition_key: Optional[str] = None
    ) -> None:
        """
        Publish a batch of messages to Kinesis.
        
        Args:
            records: List of message data to publish
            partition_key: Partition key for the records
            
        Raises:
            PublishingError: If publishing fails or if any records fail to publish
        """
        if not records:
            return
            
        try:
            # Prepare records
            kinesis_records = []
            for data in records:
                # Serialize message
                serialized = self._serialize_message(data)
                
                # Create record
                record = {
                    'Data': serialized,
                    'PartitionKey': partition_key or str(time.time_ns())
                }
                kinesis_records.append(record)
            
            # Publish batch to Kinesis
            start_time = time.time()
            response = self._client.put_records(
                Records=kinesis_records,
                StreamName=self.stream_name
            )
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Check for failed records
            failed_count = response.get('FailedRecordCount', 0)
            if failed_count > 0:
                error_message = f"{failed_count} records failed to publish"
                logger.error(error_message)
                self.metrics.record_message_failed(failed_count)
                raise PublishingError(error_message)
            
            # Update metrics for successful records
            success_count = len(records) - failed_count
            if success_count > 0:
                self.metrics.record_messages_published(success_count, processing_time_ms)
            
            logger.debug(
                f"Published batch of {len(records)} messages to stream '{self.stream_name}'"
            )
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logger.error(f"Failed to publish batch: {error_code} - {error_message}")
            self.metrics.record_message_failed(len(records))
            raise PublishingError(f"Failed to publish batch: {error_message}")
        except Exception as e:
            logger.error(f"Failed to publish batch: {e}")
            self.metrics.record_message_failed(len(records))
            raise PublishingError(f"Failed to publish batch: {e}")
    
    def on_stop(self) -> None:
        """
        Perform cleanup when the producer stops.
        
        This method is called when the producer is stopping to ensure
        any pending data is published.
        """
        # No cleanup needed for Kinesis client
        pass 