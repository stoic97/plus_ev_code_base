"""
Tests for authentication endpoints.

This module contains tests for all endpoints in app/api/v1/endpoints/auth.py.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, ANY
from datetime import datetime, timedelta

from app.core.database import get_db
from app.core.security import get_current_user, get_current_active_user


from app.api.v1.endpoints.auth import router
from app.core.security import User, Token, Roles
from app.schemas.auth import (
    RegisterRequest,
    LoginResponse,
    TokenResponse,
    UserResponse,
    RefreshTokenRequest,
    PasswordResetRequest
)


# Setup test app
app = FastAPI()
app.include_router(router)
client = TestClient(app)


# Mock user data
TEST_USER = {
    "username": "testuser",
    "email": "test@example.com",
    "password": "StrongP@ssw0rd",  # Strong password for tests
    "full_name": "Test User",
    "hashed_password": "hashed_password_value",
    "roles": [Roles.OBSERVER]
}

# Mock token data
TEST_TOKEN = {
    "access_token": "mock_access_token",
    "refresh_token": "mock_refresh_token",
    "token_type": "bearer"
}


# Fixtures for testing
@pytest.fixture
def mock_db():
    """Mock database session."""
    return MagicMock()


@pytest.fixture
def mock_user():
    """Mock user object."""
    return User(
        username=TEST_USER["username"],
        email=TEST_USER["email"],
        full_name=TEST_USER["full_name"],
        roles=TEST_USER["roles"]
    )


@pytest.fixture
def mock_token():
    """Mock token response."""
    return Token(
        access_token=TEST_TOKEN["access_token"],
        refresh_token=TEST_TOKEN["refresh_token"],
        token_type=TEST_TOKEN["token_type"]
    )


# Helper function to mock dependencies
def override_get_db():
    """Override database dependency."""
    return MagicMock()


def override_get_current_user():
    """Override current user dependency."""
    print("override_get_current_user was called!")
    return User(
        username=TEST_USER["username"],
        email=TEST_USER["email"],
        full_name=TEST_USER["full_name"],
        roles=TEST_USER["roles"]
    )


def override_get_current_active_user():
    """Override current active user dependency."""
    print("override_get_current_active_user was called!")
    return override_get_current_user()


# Apply dependency overrides
app.dependency_overrides = {
    get_db: override_get_db,
    get_current_user: override_get_current_user,
    get_current_active_user: override_get_current_active_user,
}


# Tests for /token endpoint
@patch("app.api.v1.endpoints.auth.authenticate_user")
@patch("app.api.v1.endpoints.auth.create_access_token")
@patch("app.api.v1.endpoints.auth.create_refresh_token")
def test_login_for_access_token_success(
    mock_create_refresh_token, mock_create_access_token, mock_authenticate_user
):
    """Test successful token generation."""
    # Setup mocks
    mock_authenticate_user.return_value = override_get_current_user()
    mock_create_access_token.return_value = TEST_TOKEN["access_token"]
    mock_create_refresh_token.return_value = TEST_TOKEN["refresh_token"]
    
    # Make request
    response = client.post(
        "/token",
        data={
            "username": TEST_USER["username"],
            "password": TEST_USER["password"]
        }
    )
    
    # Assertions
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert "refresh_token" in response.json()
    assert "token_type" in response.json()
    assert "expires_at" in response.json()
    assert response.json()["token_type"] == "bearer"


@patch("app.api.v1.endpoints.auth.authenticate_user")
def test_login_for_access_token_invalid_credentials(mock_authenticate_user):
    """Test token generation with invalid credentials."""
    # Setup mocks
    mock_authenticate_user.return_value = None
    
    # Make request
    response = client.post(
        "/token",
        data={
            "username": "wronguser",
            "password": "wrongpassword"
        }
    )
    
    # Assertions
    assert response.status_code == 401
    assert "detail" in response.json()
    assert "Incorrect username or password" in response.json()["detail"]


# Tests for /login endpoint
@patch("app.api.v1.endpoints.auth.authenticate_user")
@patch("app.api.v1.endpoints.auth.create_access_token")
def test_login_success(mock_create_access_token, mock_authenticate_user):
    """Test successful login."""
    # Setup mocks
    mock_authenticate_user.return_value = override_get_current_user()
    mock_create_access_token.return_value = TEST_TOKEN["access_token"]
    
    # Make request
    response = client.post(
        "/login",
        data={
            "username": TEST_USER["username"],
            "password": TEST_USER["password"]
        }
    )
    
    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert "user" in data
    assert "access_token" in data
    assert "token_type" in data
    assert "expires_at" in data
    assert data["user"]["username"] == TEST_USER["username"]
    assert data["user"]["email"] == TEST_USER["email"]
    assert data["user"]["full_name"] == TEST_USER["full_name"]
    assert data["token_type"] == "bearer"


@patch("app.api.v1.endpoints.auth.authenticate_user")
def test_login_invalid_credentials(mock_authenticate_user):
    """Test login with invalid credentials."""
    # Setup mocks
    mock_authenticate_user.return_value = None
    
    # Make request
    response = client.post(
        "/login",
        data={
            "username": "wronguser",
            "password": "wrongpassword"
        }
    )
    
    # Assertions
    assert response.status_code == 401
    assert "detail" in response.json()
    assert "Incorrect username or password" in response.json()["detail"]


# Tests for /refresh endpoint
@patch("app.api.v1.endpoints.auth.refresh_access_token")
def test_refresh_token_success(mock_refresh_access_token):
    """Test successful token refresh."""
    # Setup mocks
    mock_refresh_access_token.return_value = TEST_TOKEN
    
    # Make request
    response = client.post(
        "/refresh",
        json={"token": TEST_TOKEN["refresh_token"]}
    )
    
    # Assertions
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert "refresh_token" in response.json()
    assert "token_type" in response.json()
    assert "expires_at" in response.json()
    assert response.json()["token_type"] == "bearer"


@patch("app.api.v1.endpoints.auth.refresh_access_token")
def test_refresh_token_invalid(mock_refresh_access_token):
    """Test token refresh with invalid token."""
    # Setup mocks
    mock_refresh_access_token.side_effect = Exception("Invalid token")
    
    # Make request
    response = client.post(
        "/refresh",
        json={"token": "invalid_token"}
    )
    
    # Assertions
    assert response.status_code == 401
    assert "detail" in response.json()
    assert "Invalid token for refresh" in response.json()["detail"]


# Tests for /register endpoint
@patch("app.api.v1.endpoints.auth.verify_password_strength")
@patch("app.api.v1.endpoints.auth.create_user")
def test_register_success(mock_create_user, mock_verify_password_strength):
    """Test successful user registration."""
    # Setup mocks
    mock_verify_password_strength.return_value = True
    mock_create_user.return_value = True
    
    # Make request
    response = client.post(
        "/register",
        json={
            "username": TEST_USER["username"],
            "email": TEST_USER["email"],
            "password": TEST_USER["password"],
            "full_name": TEST_USER["full_name"]
        }
    )
    
    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert data["username"] == TEST_USER["username"]
    assert data["email"] == TEST_USER["email"]
    assert data["success"] is True
    assert "registered successfully" in data["message"]
    
    # Verify create_user called with correct parameters
    mock_create_user.assert_called_once_with(
        db=ANY,
        username=TEST_USER["username"],
        email=TEST_USER["email"],
        password=TEST_USER["password"],
        full_name=TEST_USER["full_name"],
        roles=[Roles.OBSERVER]
    )


@patch("app.api.v1.endpoints.auth.verify_password_strength")
def test_register_weak_password(mock_verify_password_strength):
    """Test registration with weak password."""
    # Setup mocks
    mock_verify_password_strength.return_value = False
    
    # Make request
    response = client.post(
        "/register",
        json={
            "username": TEST_USER["username"],
            "email": TEST_USER["email"],
            "password": "weak",
            "full_name": TEST_USER["full_name"]
        }
    )
    
    # Assertions
    assert response.status_code == 422
    assert "detail" in response.json()
    assert "String should have at least 12 characters" in str(response.json())


@patch("app.api.v1.endpoints.auth.verify_password_strength")
@patch("app.api.v1.endpoints.auth.create_user")
def test_register_existing_user(mock_create_user, mock_verify_password_strength):
    """Test registration with existing username."""
    # Setup mocks
    mock_verify_password_strength.return_value = True
    mock_create_user.return_value = False
    
    # Make request
    response = client.post(
        "/register",
        json={
            "username": TEST_USER["username"],
            "email": TEST_USER["email"],
            "password": TEST_USER["password"],
            "full_name": TEST_USER["full_name"]
        }
    )
    
    # Assertions
    assert response.status_code == 400
    assert "detail" in response.json()
    assert "User registration failed" in response.json()["detail"]


# Tests for /logout endpoint
@patch("app.api.v1.endpoints.auth.log_auth_event")
def test_logout(mock_log_auth_event):
    """Test logout endpoint."""
    # Make request
    response = client.post(
    "/logout",
    headers={"Authorization": f"Bearer {TEST_TOKEN['access_token']}"}
)
    
    # Assertions
    assert response.status_code == 200
    assert "message" in response.json()
    assert "Successfully logged out" in response.json()["message"]
    
    # Verify auth event logged
    mock_log_auth_event.assert_called_once_with(
        db=ANY,
        event_type="logout",
        username=TEST_USER["username"],
        success=True,
        client_ip=ANY
    )


# Tests for /password-reset-request endpoint
@patch("app.api.v1.endpoints.auth.log_auth_event")
def test_password_reset_request(mock_log_auth_event):
    """Test password reset request."""
    # Make request
    response = client.post(
        "/password-reset-request",
        json={"email": TEST_USER["email"]}
    )
    
    # Assertions
    assert response.status_code == 200
    assert "message" in response.json()
    assert "password reset link" in response.json()["message"]
    
    # Verify auth event logged
    mock_log_auth_event.assert_called_once_with(
        db=ANY,
        event_type="password_reset_request",
        username=TEST_USER["email"],
        success=True,
        client_ip=ANY
    )


# Tests for /me endpoint
def test_get_user_me():
    """Test retrieving current user information."""
    # Make request
    # Modify your test to include the authorization header
    response = client.get(
        "/me",
        headers={"Authorization": f"Bearer {TEST_TOKEN['access_token']}"}
        )
    
    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert data["username"] == TEST_USER["username"]
    assert data["email"] == TEST_USER["email"]
    assert data["full_name"] == TEST_USER["full_name"]
    assert data["scopes"] == TEST_USER["roles"]


# Tests for password strength verification function
def test_verify_password_strength_strong_password():
    """Test that strong passwords pass verification."""
    from app.api.v1.endpoints.auth import verify_password_strength
    
    # Test various strong passwords
    assert verify_password_strength("StrongP@ssw0rd") is True
    assert verify_password_strength("C0mpl3x!P@55") is True
    assert verify_password_strength("Th1s!Is@V3ry#Str0ng") is True


def test_verify_password_strength_weak_password():
    """Test that weak passwords fail verification."""
    from app.api.v1.endpoints.auth import verify_password_strength
    
    # Test various weak passwords
    assert verify_password_strength("password") is False  # No uppercase/digits/special
    assert verify_password_strength("Password") is False  # No digits/special
    assert verify_password_strength("password123") is False  # No uppercase/special
    assert verify_password_strength("Pass123") is False  # Too short, no special
    assert verify_password_strength("UPPERCASE123!") is False  # No lowercase