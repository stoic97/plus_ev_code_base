"""
Unit tests for provider settings configuration.

This module tests the configuration validation and functionality of
provider settings classes to ensure they handle various inputs correctly.
"""

import os
import unittest
from unittest.mock import patch
import pytest
from pydantic import ValidationError, AnyHttpUrl
from pydantic_settings import BaseSettings

from app.providers.config.provider_settings import (
    ProviderType,
    BaseProviderSettings,
    FyersSettings,
    ProviderSettings,
    get_provider_settings,
    get_settings_for_provider
)


class TestBaseProviderSettings(unittest.TestCase):
    """Test the base provider settings class."""
    
    def test_default_values(self):
        """Test that default values are set correctly."""
        settings = BaseProviderSettings()
        self.assertEqual(settings.REQUEST_TIMEOUT, 30.0)
        self.assertEqual(settings.CONNECTION_TIMEOUT, 10.0)
        self.assertEqual(settings.MAX_RETRIES, 3)
        self.assertEqual(settings.RATE_LIMIT_ENABLED, True)
        
    def test_validation_positive_integers(self):
        """Test validation of positive integer values."""
        # Should raise validation error for non-positive integers
        with self.assertRaises(ValidationError):
            BaseProviderSettings(MAX_RETRIES=0)
        
        with self.assertRaises(ValidationError):
            BaseProviderSettings(RATE_LIMIT_CALLS=-1)
        
        # Valid values should work
        settings = BaseProviderSettings(MAX_RETRIES=5, RATE_LIMIT_CALLS=200)
        self.assertEqual(settings.MAX_RETRIES, 5)
        self.assertEqual(settings.RATE_LIMIT_CALLS, 200)
    
    def test_from_environment_variables(self):
        """Test loading settings from environment variables."""
        with patch.dict(os.environ, {
            "PROVIDER_REQUEST_TIMEOUT": "60.0",
            "PROVIDER_MAX_RETRIES": "5"
        }):
            settings = BaseProviderSettings()
            self.assertEqual(settings.REQUEST_TIMEOUT, 60.0)
            self.assertEqual(settings.MAX_RETRIES, 5)
            # Other values should still have defaults
            self.assertEqual(settings.CONNECTION_TIMEOUT, 10.0)


class TestFyersSettings(unittest.TestCase):
    """Test the Fyers-specific settings class."""
    
    def setUp(self):
        """Set up test environment with valid Fyers settings."""
        self.valid_settings = {
            "APP_ID": "ABCDE-100",
            "APP_SECRET": "secretkey123",
            "REDIRECT_URI": "https://example.com/callback"
        }
    
    def test_required_fields(self):
        """Test that required fields are enforced."""
        # Missing required fields should raise ValidationError
        with self.assertRaises(ValidationError):
            FyersSettings()
        
        with self.assertRaises(ValidationError):
            FyersSettings(APP_ID="ABCDE-100", REDIRECT_URI="https://example.com/callback")
        
        # Valid settings should work
        settings = FyersSettings(**self.valid_settings)
        self.assertEqual(settings.APP_ID, "ABCDE-100")
        # Compare string representation instead of direct equality for AnyHttpUrl
        self.assertEqual(str(settings.REDIRECT_URI), "https://example.com/callback")
        self.assertEqual(settings.APP_SECRET.get_secret_value(), "secretkey123")
    
    def test_app_id_format_validation(self):
        """Test validation of APP_ID format."""
        # Invalid format
        invalid_settings = self.valid_settings.copy()
        invalid_settings["APP_ID"] = "ABCDE"
        
        with self.assertRaises(ValidationError) as context:
            FyersSettings(**invalid_settings)
        
        self.assertIn("APP_ID must be in the format", str(context.exception))
        
        # Valid format
        valid_settings = self.valid_settings.copy()
        valid_settings["APP_ID"] = "ABCDE-100"
        
        settings = FyersSettings(**valid_settings)
        self.assertEqual(settings.APP_ID, "ABCDE-100")
    
    def test_orderbook_depth_validation(self):
        """Test validation of ORDERBOOK_DEPTH values."""
        # Invalid value
        invalid_settings = self.valid_settings.copy()
        invalid_settings["ORDERBOOK_DEPTH"] = 15
        
        with self.assertRaises(ValidationError) as context:
            FyersSettings(**invalid_settings)
        
        self.assertIn("ORDERBOOK_DEPTH must be one of", str(context.exception))
        
        # Valid values
        for depth in [5, 10, 20]:
            valid_settings = self.valid_settings.copy()
            valid_settings["ORDERBOOK_DEPTH"] = depth
            
            settings = FyersSettings(**valid_settings)
            self.assertEqual(settings.ORDERBOOK_DEPTH, depth)
    
    def test_get_auth_headers(self):
        """Test generation of auth headers."""
        # Without access token
        settings = FyersSettings(**self.valid_settings)
        headers = settings.get_auth_headers()
        
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(headers["Accept"], "application/json")
        self.assertNotIn("Authorization", headers)
        
        # With access token
        settings = FyersSettings(**{**self.valid_settings, "ACCESS_TOKEN": "token123"})
        headers = settings.get_auth_headers()
        
        self.assertEqual(headers["Authorization"], "ABCDE-100:token123")
    
    def test_mask_sensitive_data(self):
        """Test masking of sensitive data for logging."""
        settings = FyersSettings(**{**self.valid_settings, "ACCESS_TOKEN": "token123"})
        masked_data = settings.mask_sensitive_data()
        
        self.assertEqual(masked_data["APP_ID"], "ABCDE-100")
        self.assertEqual(masked_data["APP_SECRET"], "********")
        self.assertEqual(masked_data["ACCESS_TOKEN"], "********")


class TestProviderSettings(unittest.TestCase):
    """Test the aggregated provider settings class."""
    
    @patch.dict(os.environ, {
        "FYERS_APP_ID": "TEST-100",
        "FYERS_APP_SECRET": "test_secret",
        "FYERS_REDIRECT_URI": "https://test.com/callback",
        "PROVIDER_DEFAULT_PROVIDER": "fyers"
    })
    def test_environment_loading(self):
        """Test loading provider settings from environment variables."""
        settings = ProviderSettings()
        
        self.assertEqual(settings.DEFAULT_PROVIDER, ProviderType.FYERS)
        self.assertEqual(settings.FYERS.APP_ID, "TEST-100")
        self.assertEqual(settings.FYERS.APP_SECRET.get_secret_value(), "test_secret")
        # Compare string representation instead of direct equality for AnyHttpUrl
        self.assertEqual(str(settings.FYERS.REDIRECT_URI), "https://test.com/callback")
    
    def test_get_provider_settings(self):
        """Test retrieving settings for specific providers."""
        # Create with minimal valid Fyers settings
        settings = ProviderSettings(
            FYERS=FyersSettings(
                APP_ID="TEST-100",
                APP_SECRET="test_secret",
                REDIRECT_URI="https://test.com/callback"
            )
        )
        
        # Get Fyers settings explicitly
        fyers_settings = settings.get_provider_settings(ProviderType.FYERS)
        self.assertIsInstance(fyers_settings, FyersSettings)
        self.assertEqual(fyers_settings.APP_ID, "TEST-100")
        
        # Get default provider settings
        settings.DEFAULT_PROVIDER = ProviderType.FYERS
        default_settings = settings.get_provider_settings()
        self.assertIsInstance(default_settings, FyersSettings)
        self.assertEqual(default_settings.APP_ID, "TEST-100")
        
        # Test unsupported provider
        with self.assertRaises(ValueError) as context:
            settings.get_provider_settings("unsupported")
        
        self.assertIn("Unsupported provider type", str(context.exception))


@pytest.mark.usefixtures("mock_env_vars")
class TestSingletonPattern:
    """Test the singleton pattern for provider settings."""
    
    @pytest.fixture
    def mock_env_vars(self, monkeypatch):
        """Set up environment variables for testing."""
        monkeypatch.setenv("FYERS_APP_ID", "TEST-100")
        monkeypatch.setenv("FYERS_APP_SECRET", "test_secret")
        monkeypatch.setenv("FYERS_REDIRECT_URI", "https://test.com/callback")
    
    def test_get_provider_settings_singleton(self):
        """Test that get_provider_settings returns a singleton instance."""
        # First call should create the instance
        settings1 = get_provider_settings()
        
        # Second call should return the same instance
        settings2 = get_provider_settings()
        
        assert settings1 is settings2
        assert settings1.FYERS.APP_ID == "TEST-100"
    
    def test_get_settings_for_provider(self):
        """Test the helper function to get settings for a specific provider."""
        # Get settings for default provider
        default_settings = get_settings_for_provider()
        assert isinstance(default_settings, BaseProviderSettings)
        
        # Get settings for specific provider
        fyers_settings = get_settings_for_provider(ProviderType.FYERS)
        assert isinstance(fyers_settings, FyersSettings)
        assert fyers_settings.APP_ID == "TEST-100"
        # Compare string representation for URL
        assert str(fyers_settings.REDIRECT_URI) == "https://test.com/callback"


if __name__ == "__main__":
    unittest.main()