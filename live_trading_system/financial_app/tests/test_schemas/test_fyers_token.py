"""
Tests for Fyers Authentication Schemas

This module contains unit tests for all Pydantic models used in
Fyers authentication API request/response validation.
"""

import pytest
from datetime import datetime, timezone, timedelta
from typing import Dict, Any
from pydantic import ValidationError

from app.schemas.fyers_auth import (
    AuthUrlRequest,
    AuthUrlResponse,
    TokenRequest,
    TokenResponse,
    TokenValidationRequest,
    TokenValidationResponse,
    TokenInfoResponse,
    RefreshTokenRequest,
    TokenClearResponse,
    RenewalInstructionsResponse,
    DiscordAlertRequest,
    DiscordAlertResponse,
    ExpiryCheckResponse,
    ErrorResponse
)


class TestAuthUrlRequest:
    """Test AuthUrlRequest schema."""
    
    def test_valid_with_state(self):
        """Test valid request with state parameter."""
        data = {"state": "test_state_123"}
        request = AuthUrlRequest(**data)
        assert request.state == "test_state_123"
    
    def test_valid_without_state(self):
        """Test valid request without state parameter."""
        request = AuthUrlRequest()
        assert request.state is None
    
    def test_empty_dict(self):
        """Test with empty dictionary."""
        request = AuthUrlRequest(**{})
        assert request.state is None
    
    def test_serialization(self):
        """Test model serialization."""
        request = AuthUrlRequest(state="test_state")
        data = request.model_dump()
        assert data == {"state": "test_state"}


class TestAuthUrlResponse:
    """Test AuthUrlResponse schema."""
    
    def test_valid_response(self):
        """Test valid auth URL response."""
        data = {
            "auth_url": "https://api.fyers.in/api/v3/generate-authcode",
            "state": "test_state",
            "expires_in": 3600
        }
        response = AuthUrlResponse(**data)
        assert str(response.auth_url) == data["auth_url"]
        assert response.state == data["state"]
        assert response.expires_in == data["expires_in"]
    
    def test_default_expires_in(self):
        """Test default expires_in value."""
        data = {
            "auth_url": "https://api.fyers.in/auth",
            "state": "test_state"
        }
        response = AuthUrlResponse(**data)
        assert response.expires_in == 3600
    
    def test_invalid_url(self):
        """Test invalid URL format."""
        data = {
            "auth_url": "not_a_valid_url",
            "state": "test_state"
        }
        with pytest.raises(ValidationError) as exc_info:
            AuthUrlResponse(**data)
        assert "url" in str(exc_info.value).lower()
    
    def test_missing_required_fields(self):
        """Test missing required fields."""
        with pytest.raises(ValidationError) as exc_info:
            AuthUrlResponse(state="test")
        assert "auth_url" in str(exc_info.value)


class TestTokenRequest:
    """Test TokenRequest schema."""
    
    def test_valid_request(self):
        """Test valid token request."""
        data = {
            "auth_code": "valid_auth_code_123",
            "state": "test_state"
        }
        request = TokenRequest(**data)
        assert request.auth_code == data["auth_code"]
        assert request.state == data["state"]
    
    def test_without_state(self):
        """Test request without state."""
        data = {"auth_code": "valid_auth_code_123"}
        request = TokenRequest(**data)
        assert request.auth_code == data["auth_code"]
        assert request.state is None
    
    def test_empty_auth_code(self):
        """Test empty auth code validation."""
        with pytest.raises(ValidationError) as exc_info:
            TokenRequest(auth_code="")
        assert "at least 1 character" in str(exc_info.value)
    
    def test_missing_auth_code(self):
        """Test missing auth code."""
        with pytest.raises(ValidationError) as exc_info:
            TokenRequest()
        assert "auth_code" in str(exc_info.value)


class TestTokenResponse:
    """Test TokenResponse schema."""
    
    def test_valid_response(self):
        """Test valid token response."""
        expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
        data = {
            "access_token": "eyJ0eXAiOiJKV1Q...",
            "refresh_token": "eyJ0eXAiOiJSRUY...",
            "token_type": "Bearer",
            "expires_at": expires_at,
            "app_id": "TEST123-100"
        }
        response = TokenResponse(**data)
        assert response.access_token == data["access_token"]
        assert response.refresh_token == data["refresh_token"]
        assert response.token_type == data["token_type"]
        assert response.expires_at == data["expires_at"]
        assert response.app_id == data["app_id"]
    
    def test_without_refresh_token(self):
        """Test response without refresh token."""
        expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
        data = {
            "access_token": "eyJ0eXAiOiJKV1Q...",
            "token_type": "Bearer",
            "expires_at": expires_at,
            "app_id": "TEST123-100"
        }
        response = TokenResponse(**data)
        assert response.refresh_token is None
    
    def test_default_token_type(self):
        """Test default token type."""
        expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
        data = {
            "access_token": "eyJ0eXAiOiJKV1Q...",
            "expires_at": expires_at,
            "app_id": "TEST123-100"
        }
        response = TokenResponse(**data)
        assert response.token_type == "Bearer"
    
    def test_missing_required_fields(self):
        """Test missing required fields."""
        with pytest.raises(ValidationError) as exc_info:
            TokenResponse()
        errors = str(exc_info.value)
        assert "access_token" in errors
        assert "expires_at" in errors
        assert "app_id" in errors


class TestTokenValidationRequest:
    """Test TokenValidationRequest schema."""
    
    def test_with_token(self):
        """Test with explicit token."""
        data = {"token": "eyJ0eXAiOiJKV1Q..."}
        request = TokenValidationRequest(**data)
        assert request.token == data["token"]
    
    def test_without_token(self):
        """Test without token (uses stored)."""
        request = TokenValidationRequest()
        assert request.token is None
    
    def test_empty_token(self):
        """Test with empty token."""
        request = TokenValidationRequest(token="")
        assert request.token == ""


class TestTokenValidationResponse:
    """Test TokenValidationResponse schema."""
    
    def test_valid_token_response(self):
        """Test valid token response."""
        expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
        data = {
            "is_valid": True,
            "expires_at": expires_at,
            "time_remaining": 3600
        }
        response = TokenValidationResponse(**data)
        assert response.is_valid is True
        assert response.expires_at == expires_at
        assert response.time_remaining == 3600
    
    def test_invalid_token_response(self):
        """Test invalid token response."""
        data = {"is_valid": False}
        response = TokenValidationResponse(**data)
        assert response.is_valid is False
        assert response.expires_at is None
        assert response.time_remaining is None
    
    def test_missing_required_field(self):
        """Test missing required is_valid field."""
        with pytest.raises(ValidationError) as exc_info:
            TokenValidationResponse()
        assert "is_valid" in str(exc_info.value)


class TestTokenInfoResponse:
    """Test TokenInfoResponse schema."""
    
    def test_complete_token_info(self):
        """Test complete token information."""
        expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
        data = {
            "has_token": True,
            "has_refresh_token": True,
            "expiry": expires_at,
            "is_valid": True,
            "time_remaining": 3600.5
        }
        response = TokenInfoResponse(**data)
        assert response.has_token is True
        assert response.has_refresh_token is True
        assert response.expiry == expires_at
        assert response.is_valid is True
        assert response.time_remaining == 3600.5
    
    def test_no_token_info(self):
        """Test no token information."""
        data = {
            "has_token": False,
            "has_refresh_token": False,
            "is_valid": False
        }
        response = TokenInfoResponse(**data)
        assert response.has_token is False
        assert response.has_refresh_token is False
        assert response.expiry is None
        assert response.is_valid is False
        assert response.time_remaining is None
    
    def test_missing_required_fields(self):
        """Test missing required fields."""
        with pytest.raises(ValidationError) as exc_info:
            TokenInfoResponse()
        errors = str(exc_info.value)
        assert "has_token" in errors
        assert "has_refresh_token" in errors
        assert "is_valid" in errors


class TestRefreshTokenRequest:
    """Test RefreshTokenRequest schema."""
    
    def test_with_pin(self):
        """Test with PIN."""
        data = {"pin": "1234"}
        request = RefreshTokenRequest(**data)
        assert request.pin == "1234"
    
    def test_without_pin(self):
        """Test without PIN."""
        request = RefreshTokenRequest()
        assert request.pin is None
    
    def test_empty_pin(self):
        """Test with empty PIN."""
        request = RefreshTokenRequest(pin="")
        assert request.pin == ""


class TestTokenClearResponse:
    """Test TokenClearResponse schema."""
    
    def test_successful_clear(self):
        """Test successful token clear."""
        data = {
            "success": True,
            "message": "Token cleared successfully"
        }
        response = TokenClearResponse(**data)
        assert response.success is True
        assert response.message == data["message"]
    
    def test_failed_clear(self):
        """Test failed token clear."""
        data = {
            "success": False,
            "message": "No token to clear"
        }
        response = TokenClearResponse(**data)
        assert response.success is False
        assert response.message == data["message"]
    
    def test_missing_required_fields(self):
        """Test missing required fields."""
        with pytest.raises(ValidationError) as exc_info:
            TokenClearResponse()
        errors = str(exc_info.value)
        assert "success" in errors
        assert "message" in errors


class TestRenewalInstructionsResponse:
    """Test RenewalInstructionsResponse schema."""
    
    def test_valid_instructions(self):
        """Test valid renewal instructions."""
        data = {
            "message": "Token needs renewal",
            "steps": [
                "Step 1: Visit auth URL",
                "Step 2: Complete authentication",
                "Step 3: Enter auth code"
            ],
            "auth_url": "https://api.fyers.in/auth"
        }
        response = RenewalInstructionsResponse(**data)
        assert response.message == data["message"]
        assert response.steps == data["steps"]
        assert str(response.auth_url) == data["auth_url"]
    
    def test_empty_steps(self):
        """Test with empty steps list."""
        data = {
            "message": "Token needs renewal",
            "steps": [],
            "auth_url": "https://api.fyers.in/auth"
        }
        response = RenewalInstructionsResponse(**data)
        assert response.steps == []
    
    def test_missing_required_fields(self):
        """Test missing required fields."""
        with pytest.raises(ValidationError) as exc_info:
            RenewalInstructionsResponse()
        errors = str(exc_info.value)
        assert "message" in errors
        assert "steps" in errors
        assert "auth_url" in errors


class TestDiscordAlertRequest:
    """Test DiscordAlertRequest schema."""
    
    def test_valid_info_alert(self):
        """Test valid info alert."""
        data = {
            "message": "Token refreshed successfully",
            "level": "info"
        }
        request = DiscordAlertRequest(**data)
        assert request.message == data["message"]
        assert request.level == data["level"]
    
    def test_valid_warning_alert(self):
        """Test valid warning alert."""
        data = {
            "message": "Token expiring soon",
            "level": "warning"
        }
        request = DiscordAlertRequest(**data)
        assert request.level == "warning"
    
    def test_valid_error_alert(self):
        """Test valid error alert."""
        data = {
            "message": "Token refresh failed",
            "level": "error"
        }
        request = DiscordAlertRequest(**data)
        assert request.level == "error"
    
    def test_default_level(self):
        """Test default level."""
        data = {"message": "Test message"}
        request = DiscordAlertRequest(**data)
        assert request.level == "info"
    
    def test_invalid_level(self):
        """Test invalid level."""
        data = {
            "message": "Test message",
            "level": "invalid_level"
        }
        with pytest.raises(ValidationError) as exc_info:
            DiscordAlertRequest(**data)
        assert "pattern" in str(exc_info.value)
    
    def test_missing_message(self):
        """Test missing message."""
        with pytest.raises(ValidationError) as exc_info:
            DiscordAlertRequest()
        assert "message" in str(exc_info.value)


class TestDiscordAlertResponse:
    """Test DiscordAlertResponse schema."""
    
    def test_successful_alert(self):
        """Test successful alert response."""
        data = {
            "success": True,
            "results": {
                "webhook1": {"status": "sent", "response_code": 200},
                "webhook2": {"status": "sent", "response_code": 200}
            }
        }
        response = DiscordAlertResponse(**data)
        assert response.success is True
        assert response.results == data["results"]
    
    def test_failed_alert(self):
        """Test failed alert response."""
        data = {
            "success": False,
            "results": {
                "webhook1": {"status": "failed", "error": "Connection timeout"}
            }
        }
        response = DiscordAlertResponse(**data)
        assert response.success is False
        assert response.results == data["results"]
    
    def test_empty_results(self):
        """Test with empty results."""
        data = {
            "success": False,
            "results": {}
        }
        response = DiscordAlertResponse(**data)
        assert response.results == {}
    
    def test_missing_required_fields(self):
        """Test missing required fields."""
        with pytest.raises(ValidationError) as exc_info:
            DiscordAlertResponse()
        errors = str(exc_info.value)
        assert "success" in errors
        assert "results" in errors


class TestExpiryCheckResponse:
    """Test ExpiryCheckResponse schema."""
    
    def test_not_expiring(self):
        """Test token not expiring soon."""
        data = {
            "is_expiring_soon": False,
            "time_remaining": 7200.5
        }
        response = ExpiryCheckResponse(**data)
        assert response.is_expiring_soon is False
        assert response.time_remaining == 7200.5
        assert response.message is None
        assert response.auth_url is None
    
    def test_expiring_soon(self):
        """Test token expiring soon."""
        data = {
            "is_expiring_soon": True,
            "time_remaining": 300.0,
            "message": "Token expires in 5 minutes",
            "auth_url": "https://api.fyers.in/auth"
        }
        response = ExpiryCheckResponse(**data)
        assert response.is_expiring_soon is True
        assert response.time_remaining == 300.0
        assert response.message == data["message"]
        assert str(response.auth_url) == data["auth_url"]
    
    def test_expired_token(self):
        """Test expired token."""
        data = {
            "is_expiring_soon": True,
            "time_remaining": 0.0,
            "message": "Token has expired"
        }
        response = ExpiryCheckResponse(**data)
        assert response.time_remaining == 0.0
    
    def test_missing_required_field(self):
        """Test missing required field."""
        with pytest.raises(ValidationError) as exc_info:
            ExpiryCheckResponse()
        assert "is_expiring_soon" in str(exc_info.value)


class TestErrorResponse:
    """Test ErrorResponse schema."""
    
    def test_basic_error(self):
        """Test basic error response."""
        data = {"error": "Authentication failed"}
        response = ErrorResponse(**data)
        assert response.error == data["error"]
        assert response.detail is None
        assert response.error_code is None
    
    def test_detailed_error(self):
        """Test error with details."""
        data = {
            "error": "Token validation failed",
            "detail": "Token has expired",
            "error_code": 401
        }
        response = ErrorResponse(**data)
        assert response.error == data["error"]
        assert response.detail == data["detail"]
        assert response.error_code == data["error_code"]
    
    def test_missing_error_field(self):
        """Test missing required error field."""
        with pytest.raises(ValidationError) as exc_info:
            ErrorResponse()
        assert "error" in str(exc_info.value)


class TestSchemaIntegration:
    """Test schema integration and serialization."""
    
    def test_json_serialization(self):
        """Test JSON serialization of all schemas."""
        # Test with a complex schema
        expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
        token_response = TokenResponse(
            access_token="eyJ0eXAiOiJKV1Q...",
            refresh_token="eyJ0eXAiOiJSRUY...",
            expires_at=expires_at,
            app_id="TEST123-100"
        )
        
        # Serialize to dict
        data = token_response.model_dump()
        assert isinstance(data, dict)
        assert "access_token" in data
        
        # Serialize to JSON
        json_str = token_response.model_dump_json()
        assert isinstance(json_str, str)
        assert "access_token" in json_str
    
    def test_schema_validation_chain(self):
        """Test validation across related schemas."""
        # Create auth URL request
        auth_request = AuthUrlRequest(state="test_state")
        
        # Create auth URL response
        auth_response = AuthUrlResponse(
            auth_url="https://api.fyers.in/auth",
            state=auth_request.state or "generated_state"
        )
        
        # Create token request using state from auth response
        token_request = TokenRequest(
            auth_code="test_code",
            state=auth_response.state
        )
        
        assert token_request.state == auth_response.state
    
    @pytest.mark.parametrize("schema_class,valid_data", [
        (AuthUrlRequest, {}),
        (AuthUrlResponse, {
            "auth_url": "https://api.fyers.in/auth",
            "state": "test"
        }),
        (TokenRequest, {"auth_code": "code123"}),
        (TokenValidationRequest, {}),
        (RefreshTokenRequest, {}),
        (DiscordAlertRequest, {"message": "test"}),
        (ErrorResponse, {"error": "test error"}),
    ])
    def test_schema_instantiation(self, schema_class, valid_data):
        """Test that all schemas can be instantiated with minimal valid data."""
        instance = schema_class(**valid_data)
        assert isinstance(instance, schema_class)