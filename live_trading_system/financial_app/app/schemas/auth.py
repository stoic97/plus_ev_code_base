"""
Authentication-related schemas for request/response validation.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field, field_validator
import re

class TokenResponse(BaseModel):
    """Response model for token endpoints."""
    access_token: str
    refresh_token: str 
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
    password: str = Field(..., min_length=12)  # Enforce minimum length
    full_name: Optional[str] = None
    
    @field_validator("username")
    @classmethod
    def username_alphanumeric(cls, v):
        """Validate username is alphanumeric."""
        if not v.isalnum():
            raise ValueError("Username must be alphanumeric")
        return v
    
    @field_validator("password")
    @classmethod
    def password_complexity(cls, password):
        """
        Validate password complexity:
        - At least 12 characters
        - Contains uppercase, lowercase, numbers, and special characters
        """
        if len(password) < 12:
            raise ValueError("Password must be at least 12 characters long")
        
        # Check complexity requirements
        checks = [
            re.search(r'[A-Z]', password),  # Uppercase
            re.search(r'[a-z]', password),  # Lowercase
            re.search(r'\d', password),     # Digit
            re.search(r'[!@#$%^&*(),.?":{}|<>]', password)  # Special char
        ]
        
        if not all(checks):
            raise ValueError(
                "Password must contain uppercase, lowercase, numbers, and special characters"
            )
        
        # Prevent common weak passwords
        weak_patterns = [
            'password', '123456', 'qwerty', 
            'admin', 'letmein', 'welcome'
        ]
        if any(pattern in password.lower() for pattern in weak_patterns):
            raise ValueError("Password is too common or weak")
        
        return password

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