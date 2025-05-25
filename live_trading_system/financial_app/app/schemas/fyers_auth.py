"""
Fyers Authentication Schemas

This module defines Pydantic models for Fyers authentication API
request/response validation and documentation.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, HttpUrl


class AuthUrlRequest(BaseModel):
    """Request model for generating auth URL."""
    state: Optional[str] = Field(None, description="Optional state parameter for CSRF protection")


class AuthUrlResponse(BaseModel):
    """Response model for auth URL generation."""
    auth_url: HttpUrl
    state: str
    expires_in: int = Field(3600, description="URL validity in seconds")


class TokenRequest(BaseModel):
    """Request model for token exchange."""
    auth_code: str = Field(..., min_length=1, description="Authorization code from redirect")
    state: Optional[str] = Field(None, description="State parameter for validation")


class TokenResponse(BaseModel):
    """Response model for token endpoints."""
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "Bearer"
    expires_at: datetime
    app_id: str


class TokenValidationRequest(BaseModel):
    """Request model for token validation."""
    token: Optional[str] = Field(None, description="Token to validate (uses stored if None)")


class TokenValidationResponse(BaseModel):
    """Response model for token validation."""
    is_valid: bool
    expires_at: Optional[datetime] = None
    time_remaining: Optional[int] = Field(None, description="Seconds until expiry")


class TokenInfoResponse(BaseModel):
    """Response model for token information."""
    has_token: bool
    has_refresh_token: bool
    expiry: Optional[datetime] = None
    is_valid: bool
    time_remaining: Optional[float] = Field(None, description="Seconds until expiry")


class RefreshTokenRequest(BaseModel):
    """Request model for token refresh."""
    pin: Optional[str] = Field(None, description="Trading PIN if required")


class TokenClearResponse(BaseModel):
    """Response model for token clearing."""
    success: bool
    message: str


class RenewalInstructionsResponse(BaseModel):
    """Response model for renewal instructions."""
    message: str
    steps: List[str]
    auth_url: HttpUrl


class DiscordAlertRequest(BaseModel):
    """Request model for Discord alerts."""
    message: str
    level: str = Field("info", pattern="^(info|warning|error)$")


class DiscordAlertResponse(BaseModel):
    """Response model for Discord alerts."""
    success: bool
    results: Dict[str, Any]


class ExpiryCheckResponse(BaseModel):
    """Response model for expiry checking."""
    is_expiring_soon: bool
    time_remaining: Optional[float] = Field(None, description="Seconds until expiry")
    message: Optional[str] = None
    auth_url: Optional[HttpUrl] = None


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    error_code: Optional[int] = None