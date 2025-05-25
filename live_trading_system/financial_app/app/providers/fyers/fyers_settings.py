"""
Fyers Provider Settings.

This module defines Fyers-specific provider settings using Pydantic,
with specialized validation and helper methods for configuration management.
"""

import logging
import re
import hashlib
from typing import Dict, Any, Optional, ClassVar
from pydantic import Field, SecretStr, AnyHttpUrl, field_validator, model_validator, ConfigDict

from app.providers.config.provider_settings import BaseProviderSettings

# Set up logging
logger = logging.getLogger(__name__)


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
    USERNAME: str = Field(
        ...,  # Required field
        description="Fyers username for authentication"
    )
    PASSWORD: SecretStr = Field(
        ...,  # Required field
        description="Fyers password for authentication"
    )
    PIN: Optional[SecretStr] = Field(
        default=None,
        description="Trading PIN if needed for certain operations"
    )
    TOTP_KEY: Optional[SecretStr] = Field(
        default=None,
        description="TOTP secret key for two-factor authentication"
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
    AUTH_BASE_URL: AnyHttpUrl = Field(
        default="https://api-t1.fyers.in/api/v3/",
        description="Fyers authentication API URL"
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
    
    model_config = ConfigDict(
        env_prefix="FYERS_",
        case_sensitive=True,
        extra="ignore"
    )
    
    @field_validator("APP_ID")
    @classmethod
    def validate_app_id_format(cls, v: str) -> str:
        """Validate that the APP_ID is in the correct format."""
        if not re.match(r'^[A-Za-z0-9]+-\d+$', v):
            raise ValueError("APP_ID must be in the format APP_ID-100")
        return v
    
    @field_validator("MARKET_DEPTH_RATE_LIMIT", "HISTORICAL_DATA_RATE_LIMIT", "QUOTES_RATE_LIMIT")
    @classmethod
    def validate_positive_rate_limit(cls, v: int) -> int:
        """Validate that rate limits are positive integers."""
        if v <= 0:
            raise ValueError("Rate limits must be positive integers")
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
    
    def get_endpoint_url(self, endpoint: str, api_type: str = "api") -> str:
        """
        Get full URL for a specific endpoint.
        
        Args:
            endpoint: API endpoint path
            api_type: Type of API ('api', 'auth', 'data', 'ws')
            
        Returns:
            Full URL for the endpoint
        """
        # Strip leading slash if present to avoid double slashes
        endpoint = endpoint.lstrip("/")
        
        # Choose base URL based on API type
        if api_type == "auth":
            base_url = str(self.AUTH_BASE_URL).rstrip("/")
        elif api_type == "data":
            base_url = str(self.DATA_API_URL).rstrip("/")
        elif api_type == "ws":
            base_url = str(self.WEBSOCKET_URL).rstrip("/")
        else:
            base_url = str(self.API_BASE_URL).rstrip("/")
        
        return f"{base_url}/{endpoint}"
    
    def generate_app_id_hash(self) -> str:
        """
        Generate the app ID hash required for token generation.
        
        Returns:
            SHA-256 hash of APP_ID:APP_SECRET
        """
        app_id = self.APP_ID
        app_secret = self.APP_SECRET.get_secret_value()
        data = f"{app_id}:{app_secret}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    def mask_sensitive_data(self) -> Dict[str, Any]:
        """
        Get a copy of settings with sensitive data masked for logging.
        
        Returns:
            Dictionary with masked sensitive fields
        """
        data = self.model_dump()
        sensitive_fields = ["APP_SECRET", "PASSWORD", "PIN", "TOTP_KEY", "ACCESS_TOKEN"]
        
        for field in sensitive_fields:
            if field in data and data[field]:
                data[field] = "********"
        
        return data