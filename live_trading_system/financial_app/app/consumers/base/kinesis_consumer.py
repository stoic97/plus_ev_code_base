"""
Base Kinesis consumer for AWS Kinesis integration.

This module provides a base class for Kinesis consumers with common functionality
for consuming messages from Kinesis streams.
"""

import logging
import json
import time
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime

import boto3
from botocore.exceptions import ClientError

from app.consumers.base.error import DeserializationError, ProcessingError
from app.consumers.base.metrics import get_metrics_registry
from app.consumers.config.settings import KinesisSettings
from app.consumers.utils.serialization import deserialize_json
from app.consumers.utils.circuit_breaker import circuit_breaker

# Set up logging
logger = logging.getLogger(__name__)

class BaseKinesisConsumer:
    """
    Base class for Kinesis consumers.
    
    Provides common functionality for consuming messages from Kinesis streams.
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
        Initialize a new Kinesis consumer.
        
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
        
        # Get stream description
        self._stream_description = self._client.describe_stream(
            StreamName=self.stream_name
        )['StreamDescription']
        
        # Get shard IDs
        self._shard_ids = [shard['ShardId'] for shard in self._stream_description['Shards']]
        
        # Initialize shard iterators
        self._shard_iterators = {}
        for shard_id in self._shard_ids:
            self._shard_iterators[shard_id] = self._get_shard_iterator(shard_id)
        
        # Set up metrics
        self.metrics_registry = get_metrics_registry()
        self.metrics = self.metrics_registry.register_consumer(
            consumer_id=f"kinesis_{stream_name}",
            topic=stream_name,
            group_id="kinesis_consumer"
        )
        
        logger.info(f"Initialized Kinesis consumer for stream '{stream_name}'")
    
    def _get_shard_iterator(self, shard_id: str) -> str:
        """
        Get a shard iterator for a specific shard.
        
        Args:
            shard_id: Shard ID
            
        Returns:
            Shard iterator
            
        Raises:
            ProcessingError: If getting the shard iterator fails
        """
        try:
            response = self._client.get_shard_iterator(
                StreamName=self.stream_name,
                ShardId=shard_id,
                ShardIteratorType='LATEST'
            )
            return response['ShardIterator']
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logger.error(f"Failed to get shard iterator: {error_code} - {error_message}")
            raise ProcessingError(f"Failed to get shard iterator: {error_message}")
    
    def _deserialize_message(self, data: bytes) -> Dict[str, Any]:
        """
        Deserialize message data from bytes.
        
        Args:
            data: Message data to deserialize
            
        Returns:
            Deserialized data
            
        Raises:
            DeserializationError: If the data cannot be deserialized
        """
        try:
            return deserialize_json(data)
        except Exception as e:
            raise DeserializationError(f"Failed to deserialize message: {e}")
    
    def _process_record(self, record: Dict[str, Any]) -> None:
        """
        Process a single record.
        
        Args:
            record: Record to process
            
        Raises:
            ProcessingError: If processing fails
        """
        try:
            # Deserialize data
            data = self._deserialize_message(record['Data'])
            
            # Process message
            start_time = time.time()
            self.process_message(data, record)
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Update detailed metrics
            self.metrics.record_message_processed(processing_time_ms)
            
            # Update high-level metrics
            self.metrics_registry.counter(
                'kinesis_messages_processed_total',
                labels={'stream': self.stream_name, 'status': 'success'}
            )
            self.metrics_registry.histogram(
                'kinesis_message_processing_duration_seconds',
                labels={'stream': self.stream_name},
                value=processing_time_ms / 1000  # Convert to seconds
            )
            
        except Exception as e:
            logger.error(f"Failed to process record: {e}")
            # Update detailed metrics
            self.metrics.record_message_failed()
            # Update high-level metrics
            self.metrics_registry.counter(
                'kinesis_messages_processed_total',
                labels={'stream': self.stream_name, 'status': 'error'}
            )
            raise ProcessingError(f"Failed to process record: {e}")
    
    def process_message(self, message: Dict[str, Any], raw_record: Dict[str, Any]) -> None:
        """
        Process a deserialized message.
        
        Args:
            message: Deserialized message content
            raw_record: Original Kinesis record
            
        Raises:
            ProcessingError: If processing fails
        """
        raise NotImplementedError("Subclasses must implement process_message")
    
    @circuit_breaker("kinesis_operations")
    def consume(self) -> None:
        """
        Consume messages from the stream.
        
        This method continuously polls the stream for new records and processes them.
        It handles shard iteration and error recovery.
        
        Raises:
            ProcessingError: If consumption fails
        """
        while True:
            try:
                # Process each shard
                for shard_id in self._shard_ids:
                    # Get records from shard
                    response = self._client.get_records(
                        ShardIterator=self._shard_iterators[shard_id],
                        Limit=self.batch_size
                    )
                    
                    # Update shard iterator
                    self._shard_iterators[shard_id] = response['NextShardIterator']
                    
                    # Process records
                    for record in response['Records']:
                        self._process_record(record)
                    
                    # Sleep if no records were received
                    if not response['Records']:
                        time.sleep(self.batch_timeout_ms / 1000)
                
            except ClientError as e:
                error_code = e.response['Error']['Code']
                error_message = e.response['Error']['Message']
                logger.error(f"Failed to consume records: {error_code} - {error_message}")
                
                # Handle specific error cases
                if error_code == 'ExpiredIteratorException':
                    # Refresh shard iterators
                    for shard_id in self._shard_ids:
                        self._shard_iterators[shard_id] = self._get_shard_iterator(shard_id)
                else:
                    raise ProcessingError(f"Failed to consume records: {error_message}")
            
            except Exception as e:
                logger.error(f"Failed to consume records: {e}")
                raise ProcessingError(f"Failed to consume records: {e}")
    
    def on_stop(self) -> None:
        """
        Perform cleanup when the consumer stops.
        
        This method is called when the consumer is stopping to ensure
        proper cleanup of resources.
        """
        # No cleanup needed for Kinesis client
        pass 