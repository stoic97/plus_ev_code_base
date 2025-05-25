"""
Kafka configuration settings.

This module provides configuration settings for Kafka producers,
including connection details, performance tuning, and topic names.
"""

import os
from typing import Optional

class KafkaSettings:
    """
    Kafka configuration settings.
    
    Provides configuration for Kafka producers including:
    - Connection settings
    - Performance tuning
    - Topic names
    - Security settings
    """
    
    def __init__(self):
        """Initialize Kafka settings with default values."""
        # Connection settings
        self.BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
        self.CLIENT_ID = os.getenv('KAFKA_CLIENT_ID', 'financial-app-producer')
        
        # Performance settings
        self.ACKS = os.getenv('KAFKA_ACKS', 'all')
        self.RETRIES = int(os.getenv('KAFKA_RETRIES', '3'))
        self.RETRY_BACKOFF_MS = int(os.getenv('KAFKA_RETRY_BACKOFF_MS', '100'))
        self.COMPRESSION_TYPE = os.getenv('KAFKA_COMPRESSION_TYPE', 'snappy')
        self.LINGER_MS = int(os.getenv('KAFKA_LINGER_MS', '5'))
        self.BATCH_SIZE = int(os.getenv('KAFKA_BATCH_SIZE', '16384'))
        self.MAX_IN_FLIGHT_REQUESTS = int(os.getenv('KAFKA_MAX_IN_FLIGHT_REQUESTS', '5'))
        
        # Idempotence and transactions
        self.ENABLE_IDEMPOTENCE = os.getenv('KAFKA_ENABLE_IDEMPOTENCE', 'true').lower() == 'true'
        self.TRANSACTIONAL_ID = os.getenv('KAFKA_TRANSACTIONAL_ID')
        
        # Security settings
        self.SECURITY_PROTOCOL = os.getenv('KAFKA_SECURITY_PROTOCOL', 'PLAINTEXT')
        self.SASL_MECHANISM = os.getenv('KAFKA_SASL_MECHANISM')
        self.SASL_USERNAME = os.getenv('KAFKA_SASL_USERNAME')
        self.SASL_PASSWORD = os.getenv('KAFKA_SASL_PASSWORD')
        self.SSL_CA_LOCATION = os.getenv('KAFKA_SSL_CA_LOCATION')
        self.SSL_CERTIFICATE_LOCATION = os.getenv('KAFKA_SSL_CERTIFICATE_LOCATION')
        self.SSL_KEY_LOCATION = os.getenv('KAFKA_SSL_KEY_LOCATION')
        
        # Topic names
        self.OHLCV_TOPIC = os.getenv('KAFKA_OHLCV_TOPIC', 'market-data.ohlcv')
        self.ORDERBOOK_TOPIC = os.getenv('KAFKA_ORDERBOOK_TOPIC', 'market-data.orderbook')
        self.TRADES_TOPIC = os.getenv('KAFKA_TRADES_TOPIC', 'market-data.trades')
        
        # Schema registry
        self.SCHEMA_REGISTRY_URL = os.getenv('KAFKA_SCHEMA_REGISTRY_URL')
        
        # Monitoring
        self.ENABLE_METRICS = os.getenv('KAFKA_ENABLE_METRICS', 'true').lower() == 'true'
        self.METRICS_PORT = int(os.getenv('KAFKA_METRICS_PORT', '9090'))
    
    def get_security_config(self) -> dict:
        """
        Get security-related configuration.
        
        Returns:
            Dictionary of security configuration settings
        """
        config = {}
        
        if self.SECURITY_PROTOCOL != 'PLAINTEXT':
            config['security.protocol'] = self.SECURITY_PROTOCOL
            
            if self.SECURITY_PROTOCOL in ['SASL_PLAINTEXT', 'SASL_SSL']:
                if self.SASL_MECHANISM:
                    config['sasl.mechanism'] = self.SASL_MECHANISM
                if self.SASL_USERNAME:
                    config['sasl.username'] = self.SASL_USERNAME
                if self.SASL_PASSWORD:
                    config['sasl.password'] = self.SASL_PASSWORD
            
            if self.SECURITY_PROTOCOL in ['SSL', 'SASL_SSL']:
                if self.SSL_CA_LOCATION:
                    config['ssl.ca.location'] = self.SSL_CA_LOCATION
                if self.SSL_CERTIFICATE_LOCATION:
                    config['ssl.certificate.location'] = self.SSL_CERTIFICATE_LOCATION
                if self.SSL_KEY_LOCATION:
                    config['ssl.key.location'] = self.SSL_KEY_LOCATION
        
        return config 