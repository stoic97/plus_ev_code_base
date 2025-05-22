"""
Unit tests for config.py module.

Tests configuration loading, validation, and accessor methods.
"""

import os
from unittest import mock

import pytest
from pydantic import ValidationError

# Import settings and get_settings directly - avoids circular references
from app.core.config import Settings, get_settings


class TestConfig:
    """Test configuration settings and validation."""

    def test_get_settings_singleton(self):
        """Test that get_settings returns a singleton instance."""
        settings1 = get_settings()
        settings2 = get_settings()
        
        # Same instance
        assert settings1 is settings2
    
    def setup_method(self):
        """Set up tests by clearing environment variables that might affect tests."""
        # Store original environ
        self.original_environ = os.environ.copy()
        
        # Clear settings specific vars that might affect tests
        keys_to_clear = [
            k for k in os.environ
            if k.startswith(("APP_", "DB_", "DB__", "KAFKA_", "KAFKA__", "SECURITY_", "SECURITY__", "PERF_", "PERF__"))
        ]
        for key in keys_to_clear:
            if key in os.environ:
                del os.environ[key]
    
    def teardown_method(self):
        """Restore environment after tests."""
        os.environ.clear()
        os.environ.update(self.original_environ)
    
    def test_default_values(self):
        """Test default configuration values."""
        # Create a new instance directly (not using singleton)
        settings = Settings()
        
        # Application metadata
        assert settings.APP_NAME == "Trading Strategies Application"
        assert settings.APP_VERSION == "0.1.0"
        assert settings.DEBUG is False
        assert settings.ENV == "development"
        
        # Kafka settings
        assert settings.kafka.BOOTSTRAP_SERVERS == ["localhost:9092"]
        assert settings.kafka.GROUP_ID == "trading-app"
        
        # Database settings
        assert settings.db.POSTGRES_SERVER == "localhost"
        assert settings.db.MONGODB_MAX_POOL_SIZE == 50  # Test a random setting
        
        # Security settings
        assert settings.security.ALGORITHM == "HS256"
        assert settings.security.ACCESS_TOKEN_EXPIRE_MINUTES == 30
        
        # Performance settings
        assert settings.performance.CACHE_TTL_DEFAULT == 300
    
    def test_environment_variables(self):
        """Test loading configuration from environment variables."""
        # Set up environment variables for this test
        env_vars = {
            "APP_NAME": "Test App",
            "DEBUG": "true",
            "ENV": "production",
            "KAFKA__BOOTSTRAP_SERVERS": "server1:9092,server2:9092",
            "DB__POSTGRES_SERVER": "postgres.example.com",
            "DB__POSTGRES_PASSWORD": "securepassword",
            "SECURITY__ACCESS_TOKEN_EXPIRE_MINUTES": "60",
            "PERF__CACHE_TTL_DEFAULT": "600"
        }
        
        with mock.patch.dict(os.environ, env_vars, clear=True):
            # Create a new settings instance (not using singleton)
            settings = Settings()
            
            # Check if env vars were applied
            assert settings.APP_NAME == "Test App"
            assert settings.DEBUG is True
            assert settings.ENV == "production"
            
            # Test nested settings with delimiter
            assert settings.kafka.BOOTSTRAP_SERVERS == ["server1:9092", "server2:9092"]
            assert settings.db.POSTGRES_SERVER == "postgres.example.com"
            assert settings.db.POSTGRES_PASSWORD == "securepassword"
            assert settings.security.ACCESS_TOKEN_EXPIRE_MINUTES == 60
            assert settings.performance.CACHE_TTL_DEFAULT == 600
    
    def test_uri_validators(self):
        """Test database URI validators."""
        # Create a new settings instance directly (not using singleton)
        settings = Settings()
        
        # Check Postgres URI construction
        postgres_uri = settings.db.POSTGRES_URI
        assert postgres_uri is not None
        
        # Test with more specific pattern to avoid partial matches
        assert f"postgresql://{settings.db.POSTGRES_USER}:{settings.db.POSTGRES_PASSWORD}" in str(postgres_uri)
        assert f"@{settings.db.POSTGRES_SERVER}:{settings.db.POSTGRES_PORT}" in str(postgres_uri)
        
        # Check MongoDB URI construction
        mongodb_uri = settings.db.MONGODB_URI
        assert mongodb_uri is not None
        
        # Test with more specific pattern matching
        if settings.db.MONGODB_USER and settings.db.MONGODB_PASSWORD:
            assert f"mongodb://{settings.db.MONGODB_USER}:{settings.db.MONGODB_PASSWORD}@" in str(mongodb_uri)
        else:
            assert "mongodb://" in str(mongodb_uri)
    
    def test_environment_validation(self):
        """Test validation of environment value."""
        with mock.patch.dict(os.environ, {"ENV": "invalid_env"}):
            with pytest.raises(ValidationError) as exc_info:
                Settings()
            
            # Check that the correct validation error was raised
            assert "Environment must be one of" in str(exc_info.value)
    
    def test_explicit_uri_override(self):
        """Test that explicit URI overrides constructed URI."""
        test_uri = "postgresql://user:pass@custom-host:5432/db"
        
        with mock.patch.dict(os.environ, {"DB__POSTGRES_URI": test_uri}):
            # Create a new settings instance directly
            settings = Settings()
            
            # The explicit URI should be used instead of the constructed one
            assert str(settings.db.POSTGRES_URI) == test_uri