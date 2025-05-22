"""
Kafka consumer configuration settings.
This module defines the configuration settings for Kafka consumers,
using Pydantic for validation and centralized configuration management.
"""
from typing import List, Dict, Any, Optional, Union, ClassVar
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import os
import logging

# Set up logging
logger = logging.getLogger(__name__)

class KafkaSettings(BaseSettings):
    """Kafka configuration settings for consumers."""
    
    # Broker connection
    BOOTSTRAP_SERVERS: List[str] = Field(
        default=["localhost:9092"], 
        description="Kafka bootstrap servers"
    )
    
    # Topic settings
    MARKET_DATA_TOPIC: str = Field(
        default="market-data", 
        description="Topic for market data streaming"
    )
    OHLCV_TOPIC: str = Field(
        default="market-data-ohlcv", 
        description="Topic for OHLCV candlestick data"
    )
    TRADE_TOPIC: str = Field(
        default="market-data-trades", 
        description="Topic for individual trade data"
    )
    ORDERBOOK_TOPIC: str = Field(
        default="market-data-orderbook", 
        description="Topic for order book data"
    )
    
    # Consumer group settings
    GROUP_ID: str = Field(
        default="trading-app", 
        description="Default consumer group ID"
    )
    OHLCV_GROUP_ID: str = Field(
        default="trading-app-ohlcv", 
        description="Consumer group ID for OHLCV data"
    )
    TRADE_GROUP_ID: str = Field(
        default="trading-app-trades", 
        description="Consumer group ID for trade data"
    )
    ORDERBOOK_GROUP_ID: str = Field(
        default="trading-app-orderbook", 
        description="Consumer group ID for orderbook data"
    )
    
    # Consumer behavior
    AUTO_OFFSET_RESET: str = Field(
        default="latest", 
        description="Auto offset reset policy (latest or earliest)"
    )
    ENABLE_AUTO_COMMIT: bool = Field(
        default=False, 
        description="Enable auto commit for consumer offsets"
    )
    MAX_POLL_INTERVAL_MS: int = Field(
        default=300000,  # 5 minutes
        description="Maximum delay between polls in ms"
    )
    MAX_POLL_RECORDS: int = Field(
        default=500, 
        description="Maximum records returned in a single poll"
    )
    SESSION_TIMEOUT_MS: int = Field(
        default=30000,  # 30 seconds
        description="Session timeout in ms"
    )
    REQUEST_TIMEOUT_MS: int = Field(
        default=40000,  # 40 seconds
        description="Request timeout in ms"
    )
    
    # Performance tuning
    CONSUMER_THREADS: int = Field(
        default=1, 
        description="Number of consumer threads per consumer instance"
    )
    COMMIT_INTERVAL_MS: int = Field(
        default=5000,  # 5 seconds
        description="Interval for committing offsets in ms when auto-commit is disabled"
    )
    BATCH_SIZE: int = Field(
        default=100, 
        description="Batch size for processing"
     )
    BATCH_TIMEOUT_MS: int = Field(
        default=1000,  # 1 second
        description="Maximum time to wait for a full batch in ms"
    )
    
    # Error handling
    MAX_RETRIES: int = Field(
        default=3, 
        description="Maximum number of retries for failed messages"
    )
    RETRY_BACKOFF_MS: int = Field(
        default=1000,  # 1 second
        description="Backoff time between retries in ms"
    )
    ERROR_TOPIC: Optional[str] = Field(
        default=None, 
        description="Dead letter queue topic for failed messages"
    )
    
    # Serialization
    VALUE_DESERIALIZER: str = Field(
        default="json", 
        description="Message value deserializer type",
        json_schema_extra={"enum": ["json", "avro", "protobuf", "string", "bytes"]}
    )
    
    # Monitoring
    METRICS_ENABLED: bool = Field(
        default=True, 
        description="Enable consumer metrics collection"
    )
    METRICS_INTERVAL_MS: int = Field(
        default=60000,  # 60 seconds
        description="Interval for logging metrics in ms"
    )
    
    # Configure Pydantic model settings
    model_config = SettingsConfigDict(
        env_prefix="KAFKA_",
        env_file=".env",
        case_sensitive=True,
        extra="ignore"
    )
    
    # Use field validator for BOOTSTRAP_SERVERS specifically
    @field_validator("BOOTSTRAP_SERVERS", mode="before")
    @classmethod
    def parse_bootstrap_servers(cls, value):
        """Parse comma-separated bootstrap servers string."""
        if isinstance(value, str):
            return [server.strip() for server in value.split(",")]
        return value
    
    @model_validator(mode="after")
    def validate_settings(self) -> "KafkaSettings":
        """Validate settings consistency."""
        # Validate offset reset
        offset_reset = self.AUTO_OFFSET_RESET
        if offset_reset not in ["latest", "earliest", "none"]:
            raise ValueError(f"Invalid AUTO_OFFSET_RESET value: {offset_reset}. Must be 'latest', 'earliest', or 'none'")
        
        # Validate bootstrap servers
        bootstrap_servers = self.BOOTSTRAP_SERVERS
        if not bootstrap_servers:
            logger.warning("No Kafka bootstrap servers specified")
        
        # Validate thread count
        consumer_threads = self.CONSUMER_THREADS
        if consumer_threads < 1:
            raise ValueError(f"CONSUMER_THREADS must be at least 1, got {consumer_threads}")
        elif consumer_threads > 1:
            logger.warning(f"Using {consumer_threads} consumer threads - ensure your application handles concurrency correctly")
        
        # Validate batch settings
        batch_size = self.BATCH_SIZE
        if batch_size < 1:
            raise ValueError(f"BATCH_SIZE must be at least 1, got {batch_size}")
        
        return self
    
    def get_consumer_config(self, group_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get consumer configuration dictionary for librdkafka.
        
        Args:
            group_id: Override group ID
            
        Returns:
            Dictionary of configuration values
        """
        return {
            'bootstrap.servers': ','.join(self.BOOTSTRAP_SERVERS),
            'group.id': group_id or self.GROUP_ID,
            'auto.offset.reset': self.AUTO_OFFSET_RESET,
            'enable.auto.commit': self.ENABLE_AUTO_COMMIT,
            'max.poll.interval.ms': self.MAX_POLL_INTERVAL_MS,
            'max.poll.records': self.MAX_POLL_RECORDS,
            'session.timeout.ms': self.SESSION_TIMEOUT_MS,
            'request.timeout.ms': self.REQUEST_TIMEOUT_MS,
        }

# Singleton instance for KafkaSettings
_kafka_settings_instance = None

def get_kafka_settings() -> KafkaSettings:
    """
    Get Kafka settings singleton.
    
    Returns:
        KafkaSettings instance
    """
    global _kafka_settings_instance
    if _kafka_settings_instance is None:
        _kafka_settings_instance = KafkaSettings()
    return _kafka_settings_instance