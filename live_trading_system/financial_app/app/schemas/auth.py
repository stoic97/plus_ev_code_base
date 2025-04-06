"""
Authentication-related schemas for request/response validation.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field, validator


class TokenResponse(BaseModel):
    """Response model for token endpoints."""
    access_token: str
    token_type: str
    expires_at: datetime
    scopes: List[str]


class UserResponse(BaseModel):
    """User information for responses."""
    username: str
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    scopes: List[str]


class LoginResponse(BaseModel):
    """Response model for login endpoint."""
    user: UserResponse
    access_token: str
    token_type: str
    expires_at: datetime


class RefreshTokenRequest(BaseModel):
    """Request model for token refresh."""
    token: str


class RegisterRequest(BaseModel):
    """Request model for user registration."""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None
    
    @validator("username")
    def username_alphanumeric(cls, v):
        """Validate username is alphanumeric."""
        if not v.isalnum():
            raise ValueError("Username must be alphanumeric")
        return v


class RegisterResponse(BaseModel):
    """Response model for user registration."""
    username: str
    email: EmailStr
    success: bool
    message: str


class PasswordResetRequest(BaseModel):
    """Request model for password reset."""
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """Request model for password reset confirmation."""
    token: str
    password: str = Field(..., min_length=8)