"""
Provider configuration settings.

This module defines settings models for market data providers using Pydantic,
with specific implementations for Fyers and other supported providers.
"""

import os
import logging
from enum import Enum
from typing import Dict, List, Optional, Union, Any, ClassVar
from pydantic import Field, SecretStr, AnyHttpUrl, field_validator, model_validator, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict

# Set up logging
logger = logging.getLogger(__name__)


class ProviderType(str, Enum):
    """Supported market data provider types."""
    FYERS = "fyers"
    ZERODHA = "zerodha"  # For future implementation
    INTERACTIVE_BROKERS = "interactive_brokers"  # For future implementation
    MOCK = "mock"  # For testing


class BaseProviderSettings(BaseSettings):
    """Base settings for all market data providers."""
    
    # Connection settings
    REQUEST_TIMEOUT: float = Field(
        default=30.0,
        description="HTTP request timeout in seconds"
    )
    CONNECTION_TIMEOUT: float = Field(
        default=10.0,
        description="Connection establishment timeout in seconds"
    )
    MAX_RETRIES: int = Field(
        default=3,
        description="Maximum number of retries for failed requests"
    )
    RETRY_BACKOFF: float = Field(
        default=0.5,
        description="Exponential backoff factor for retries"
    )
    
    # Rate limiting
    RATE_LIMIT_ENABLED: bool = Field(
        default=True,
        description="Enable rate limiting for API requests"
    )
    RATE_LIMIT_CALLS: int = Field(
        default=100,
        description="Number of calls allowed in the rate limit window"
    )
    RATE_LIMIT_PERIOD: int = Field(
        default=60,
        description="Rate limit window in seconds"
    )
    
    # WebSocket settings
    WEBSOCKET_PING_INTERVAL: int = Field(
        default=30,
        description="Interval between WebSocket ping messages in seconds"
    )
    WEBSOCKET_RECONNECT_DELAY: int = Field(
        default=5,
        description="Delay before reconnecting WebSocket in seconds"
    )
    WEBSOCKET_MAX_RECONNECTS: int = Field(
        default=5,
        description="Maximum number of WebSocket reconnect attempts"
    )
    
    # Logging and monitoring
    DEBUG_MODE: bool = Field(
        default=False,
        description="Enable debug logging for provider operations"
    )
    CAPTURE_METRICS: bool = Field(
        default=True,
        description="Capture performance metrics for provider operations"
    )
    
    model_config = SettingsConfigDict(
        env_prefix="PROVIDER_",
        case_sensitive=True,
        extra="ignore"
    )
    
    @field_validator("MAX_RETRIES", "RATE_LIMIT_CALLS", "RATE_LIMIT_PERIOD")
    @classmethod
    def validate_positive_integer(cls, v: int) -> int:
        """Validate that the value is a positive integer."""
        if v <= 0:
            raise ValueError("must be a positive integer")
        return v


class FyersSettings(BaseProviderSettings):
    """Fyers-specific provider settings."""
    
    # Authentication
    APP_ID: str = Field(
        ...,  # Required field
        description="Fyers App ID in the format APP_ID-100"
    )
    APP_SECRET: SecretStr = Field(
        ...,  # Required field
        description="Fyers App Secret"
    )
    REDIRECT_URI: AnyHttpUrl = Field(
        ...,  # Required field
        description="Redirect URI for OAuth flow"
    )
    ACCESS_TOKEN: Optional[SecretStr] = Field(
        default=None,
        description="Fyers access token (can be provided directly or generated through auth flow)"
    )
    
    # API endpoints
    API_BASE_URL: AnyHttpUrl = Field(
        default="https://api.fyers.in/api/v2/",
        description="Fyers REST API base URL"
    )
    DATA_API_URL: AnyHttpUrl = Field(
        default="https://api.fyers.in/data-rest/v2/",
        description="Fyers Data API URL for historical data"
    )
    WEBSOCKET_URL: str = Field(
        default="wss://api.fyers.in/socket/v2/",
        description="Fyers WebSocket API URL"
    )
    
    # Fyers-specific rate limits (different endpoints have different limits)
    MARKET_DEPTH_RATE_LIMIT: int = Field(
        default=10,
        description="Rate limit for market depth requests per second"
    )
    HISTORICAL_DATA_RATE_LIMIT: int = Field(
        default=5,
        description="Rate limit for historical data requests per second"
    )
    QUOTES_RATE_LIMIT: int = Field(
        default=15,
        description="Rate limit for quote requests per second"
    )
    
    # WebSocket channel configuration
    ORDERBOOK_DEPTH: int = Field(
        default=10,
        description="Depth of order book data to request (5, 10, or 20)"
    )
    
    # Token management
    AUTO_RENEW_TOKEN: bool = Field(
        default=True,
        description="Automatically renew access token before expiry"
    )
    TOKEN_RENEWAL_MARGIN: int = Field(
        default=300,  # 5 minutes
        description="Seconds before token expiry to trigger renewal"
    )
    
    model_config = SettingsConfigDict(
        env_prefix="FYERS_",
        case_sensitive=True,
        extra="ignore"
    )
    
    @field_validator("APP_ID")
    @classmethod
    def validate_app_id_format(cls, v: str) -> str:
        """Validate that the APP_ID is in the correct format."""
        if "-" not in v:
            raise ValueError("APP_ID must be in the format APP_ID-100")
        return v
    
    @model_validator(mode="after")
    def validate_orderbook_depth(self) -> "FyersSettings":
        """Validate that ORDERBOOK_DEPTH is one of the allowed values."""
        allowed_depths = {5, 10, 20}
        if self.ORDERBOOK_DEPTH not in allowed_depths:
            raise ValueError(f"ORDERBOOK_DEPTH must be one of {allowed_depths}")
        return self
    
    def get_auth_headers(self) -> Dict[str, str]:
        """
        Get authentication headers for API requests.
        
        Returns:
            Dictionary of headers with authentication information
        """
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        if self.ACCESS_TOKEN:
            # Add Authorization header if token is available
            headers["Authorization"] = f"{self.APP_ID}:{self.ACCESS_TOKEN.get_secret_value()}"
        
        return headers
    
    def mask_sensitive_data(self) -> Dict[str, Any]:
        """
        Get a copy of settings with sensitive data masked for logging.
        
        Returns:
            Dictionary with masked sensitive fields
        """
        data = self.model_dump()
        for sensitive_field in ["APP_SECRET", "ACCESS_TOKEN"]:
            if sensitive_field in data and data[sensitive_field]:
                data[sensitive_field] = "********"
        return data


class ProviderSettings(BaseSettings):
    """Aggregated settings for all providers."""
    
    # Default provider to use
    DEFAULT_PROVIDER: ProviderType = Field(
        default=ProviderType.FYERS,
        description="Default market data provider to use"
    )
    
    # Provider-specific settings
    FYERS: FyersSettings = Field(
        default_factory=FyersSettings,
        description="Fyers provider settings"
    )
    
    # Future provider implementations
    # ZERODHA: ZerodhaSettings = Field(...)
    # INTERACTIVE_BROKERS: IBSettings = Field(...)
    
    model_config = SettingsConfigDict(
        env_prefix="PROVIDER_",
        case_sensitive=True,
        extra="ignore"
    )
    
    def get_provider_settings(self, provider_type: Optional[ProviderType] = None) -> BaseProviderSettings:
        """
        Get settings for the specified provider.
        
        Args:
            provider_type: Provider type (uses DEFAULT_PROVIDER if None)
            
        Returns:
            Provider-specific settings
            
        Raises:
            ValueError: If the provider type is not supported
        """
        provider = provider_type or self.DEFAULT_PROVIDER
        
        if provider == ProviderType.FYERS:
            return self.FYERS
        # Add cases for other providers as they are implemented
        # elif provider == ProviderType.ZERODHA:
        #     return self.ZERODHA
        
        raise ValueError(f"Unsupported provider type: {provider}")


# Singleton instance
_provider_settings_instance = None

def get_provider_settings() -> ProviderSettings:
    """
    Get provider settings singleton.
    
    Returns:
        ProviderSettings instance
    """
    global _provider_settings_instance
    if _provider_settings_instance is None:
        _provider_settings_instance = ProviderSettings()
    return _provider_settings_instance


def get_settings_for_provider(provider_type: Optional[ProviderType] = None) -> BaseProviderSettings:
    """
    Get settings for a specific provider type.
    
    Args:
        provider_type: Provider type to get settings for (uses default if None)
        
    Returns:
        Provider-specific settings
    """
    settings = get_provider_settings()
    return settings.get_provider_settings(provider_type)