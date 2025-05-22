# tests/test_consumers/test_config/test_settings.py

import os
import pytest
from unittest import mock
from app.consumers.config.settings import KafkaSettings, get_kafka_settings


class TestKafkaSettings:
    """Test Kafka consumer configuration settings."""

    def test_default_values(self):
        """Test default values are set correctly."""
        settings = KafkaSettings()
        
        # Check broker connection defaults
        assert settings.BOOTSTRAP_SERVERS == ["localhost:9092"]
        
        # Check topic defaults
        assert settings.MARKET_DATA_TOPIC == "market-data"
        assert settings.OHLCV_TOPIC == "market-data-ohlcv"
        assert settings.TRADE_TOPIC == "market-data-trades"
        assert settings.ORDERBOOK_TOPIC == "market-data-orderbook"
        
        # Check consumer group defaults
        assert settings.GROUP_ID == "trading-app"
        assert settings.OHLCV_GROUP_ID == "trading-app-ohlcv"
        assert settings.TRADE_GROUP_ID == "trading-app-trades"
        assert settings.ORDERBOOK_GROUP_ID == "trading-app-orderbook"
        
        # Check behavior defaults
        assert settings.AUTO_OFFSET_RESET == "latest"
        assert settings.ENABLE_AUTO_COMMIT is False
        
        # Check performance defaults
        assert settings.CONSUMER_THREADS == 1
        assert settings.BATCH_SIZE == 100
        
        # Check error handling defaults
        assert settings.MAX_RETRIES == 3
        assert settings.ERROR_TOPIC is None

    def test_environment_variable_override(self):
        """Test that environment variables override defaults."""
    with mock.patch.dict(os.environ, {
        "KAFKA_BOOTSTRAP_SERVERS": '["server1:9092","server2:9092"]',  # JSON format
        "KAFKA_MARKET_DATA_TOPIC": "prod-market-data",
        "KAFKA_CONSUMER_THREADS": "2",
        "KAFKA_BATCH_SIZE": "200",
        "KAFKA_ERROR_TOPIC": "error-topic"
    }):
        settings = KafkaSettings()
        
        # Check environment overrides
        assert settings.BOOTSTRAP_SERVERS == ["server1:9092", "server2:9092"]
        assert settings.MARKET_DATA_TOPIC == "prod-market-data" 
        assert settings.CONSUMER_THREADS == 2
        assert settings.BATCH_SIZE == 200
        assert settings.ERROR_TOPIC == "error-topic"
        
        # Values not overridden should stay at defaults
        assert settings.AUTO_OFFSET_RESET == "latest"

    def test_validation_valid_config(self):
        """Test that valid configurations pass validation."""
        # Default configuration should be valid
        settings = KafkaSettings()
        # No exception should be raised

    def test_validation_invalid_offset_reset(self):
        """Test validation rejects invalid offset reset values."""
        with pytest.raises(ValueError) as excinfo:
            # Set invalid offset reset value
            KafkaSettings(AUTO_OFFSET_RESET="invalid")
        
        assert "Invalid AUTO_OFFSET_RESET value" in str(excinfo.value)

    def test_validation_invalid_thread_count(self):
        """Test validation rejects invalid thread counts."""
        with pytest.raises(ValueError) as excinfo:
            # Set invalid thread count
            KafkaSettings(CONSUMER_THREADS=0)
        
        assert "CONSUMER_THREADS must be at least 1" in str(excinfo.value)

    def test_validation_invalid_batch_size(self):
        """Test validation rejects invalid batch sizes."""
        with pytest.raises(ValueError) as excinfo:
            # Set invalid batch size
            KafkaSettings(BATCH_SIZE=0)
        
        assert "BATCH_SIZE must be at least 1" in str(excinfo.value)

    def test_get_consumer_config(self):
        """Test get_consumer_config returns correct configuration."""
        settings = KafkaSettings()
        config = settings.get_consumer_config()
        
        # Check important config values
        assert config['bootstrap.servers'] == "localhost:9092"
        assert config['group.id'] == "trading-app"
        assert config['auto.offset.reset'] == "latest"
        assert config['enable.auto.commit'] is False
        
        # Test with group ID override
        config = settings.get_consumer_config("custom-group")
        assert config['group.id'] == "custom-group"

    def test_singleton_pattern(self):
        """Test that get_kafka_settings returns a singleton instance."""
        settings1 = get_kafka_settings()
        settings2 = get_kafka_settings()
        
        # Should be the same instance
        assert settings1 is settings2