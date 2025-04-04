"""
Unit tests for security.py module.

Tests authentication, authorization, password handling, and user management functions.
"""

import datetime
import ipaddress
import json
import os
import pytest
from datetime import timedelta
from unittest.mock import MagicMock, patch

from fastapi import HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from jose import jwt, JWTError

from app.core.config import Settings
from app.core.database import PostgresDB
from app.core.security import (
    authenticate_user, create_access_token, create_refresh_token,
    get_password_hash, get_user, has_role, is_allowed_ip,
    log_auth_event, verify_password, User, Roles,
    get_current_user, get_current_active_user,
    create_user, update_user_roles, disable_user, enable_user,
    change_password, admin_reset_password
)


# Mocks and fixtures
@pytest.fixture
def mock_db_session():
    """Create a mock database session for testing."""
    session_mock = MagicMock()
    session_mock.__enter__ = MagicMock(return_value=session_mock)
    session_mock.__exit__ = MagicMock(return_value=None)
    return session_mock


@pytest.fixture
def mock_db(mock_db_session):
    """Create a mock PostgresDB instance."""
    db = MagicMock(spec=PostgresDB)
    db.session.return_value = mock_db_session
    return db


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = MagicMock()
    # Create the nested security attribute
    settings.security = MagicMock()
    settings.security.SECRET_KEY = "test_secret_key"
    settings.security.ALGORITHM = "HS256"
    settings.security.ACCESS_TOKEN_EXPIRE_MINUTES = 30
    # Add other required attributes
    return settings


@pytest.fixture
def mock_request():
    """Create a mock FastAPI Request object."""
    request = MagicMock()
    request.client.host = "192.168.1.100"
    return request


@pytest.fixture
def test_user_data():
    """Sample user data for testing."""
    return {
        "username": "testuser",
        "email": "test@example.com",
        "full_name": "Test User",
        "disabled": False,
        "hashed_password": get_password_hash("testpassword"),
        "roles": ["trader", "analyst"]
    }


@pytest.fixture
def test_user(test_user_data):
    """Create a User object for testing."""
    user_dict = {k: v for k, v in test_user_data.items() if k != "hashed_password"}
    return User(**user_dict)


@pytest.fixture
def sample_token_data():
    """Sample token data for testing."""
    return {
        "sub": "testuser",
        "roles": ["trader", "analyst"]
    }


def setup_db_user_query(mock_db_session, test_user_data):
    """Set up mock database session to return test user data."""
    # Create a mock result for fetchone
    mock_row = MagicMock()
    mock_row.username = test_user_data["username"]
    mock_row.email = test_user_data["email"]
    mock_row.full_name = test_user_data["full_name"]
    mock_row.disabled = test_user_data["disabled"]
    mock_row.hashed_password = test_user_data["hashed_password"]
    mock_row.roles = ",".join(test_user_data["roles"])
    
    # Set up the mock session to return our mock row
    mock_db_session.execute.return_value.fetchone.return_value = mock_row


#################################################
# Password Management Tests
#################################################

def test_password_hashing():
    """Test password hashing and verification."""
    password = "securepassword123"
    hashed = get_password_hash(password)
    
    # Hashed password should be different from plain text
    assert hashed != password
    
    # Verification should work with correct password
    assert verify_password(password, hashed)
    
    # Verification should fail with incorrect password
    assert not verify_password("wrongpassword", hashed)


#################################################
# JWT Token Tests
#################################################

def test_create_access_token():
    """Test JWT access token creation."""
    # Create fixed settings for consistent token verification
    test_settings = MagicMock()
    test_settings.security = MagicMock()
    test_settings.security.SECRET_KEY = "fixed_test_secret_key"
    test_settings.security.ALGORITHM = "HS256"
    test_settings.security.ACCESS_TOKEN_EXPIRE_MINUTES = 30
    
    with patch("app.core.security.get_settings", return_value=test_settings):
        data = {"sub": "testuser", "roles": ["admin"]}
        
        # Test with custom expiry
        expires = timedelta(minutes=5)
        token = create_access_token(data, expires)
        assert token is not None
        
        # Decode token to verify payload - use the same test_settings
        payload = jwt.decode(
            token, 
            test_settings.security.SECRET_KEY, 
            algorithms=[test_settings.security.ALGORITHM],
            options={"verify_signature": False}
        )
        
        assert payload["sub"] == "testuser"
        assert payload["roles"] == ["admin"]
        assert "exp" in payload


def test_create_refresh_token(mock_settings):
    """Test JWT refresh token creation."""
    # Create fixed settings for consistent token verification
    test_settings = MagicMock()
    test_settings.security = MagicMock()
    test_settings.security.SECRET_KEY = "fixed_test_secret_key"
    test_settings.security.ALGORITHM = "HS256"
    test_settings.security.ACCESS_TOKEN_EXPIRE_MINUTES = 30

    with patch("app.core.security.get_settings", return_value=mock_settings):
        data = {"sub": "testuser", "roles": ["admin"]}
        
        token = create_refresh_token(data)
        assert token is not None
        
        # Decode token to verify payload
        payload = jwt.decode(
            token, 
            mock_settings.security.SECRET_KEY, 
            algorithms=[mock_settings.security.ALGORITHM],
            options={"verify_signature": False}
        )
        assert payload["sub"] == "testuser"
        assert payload["roles"] == ["admin"]
        assert "exp" in payload


#################################################
# User Authentication Tests
#################################################

def test_get_user(mock_db, mock_db_session, test_user_data):
    """Test retrieving user from database."""
    setup_db_user_query(mock_db_session, test_user_data)
    
    user = get_user(mock_db, test_user_data["username"])
    
    # Verify correct SQL query was executed
    mock_db_session.execute.assert_called_once()
    query_args = mock_db_session.execute.call_args[0][1]
    assert query_args["username"] == test_user_data["username"]
    
    # Verify user data was returned correctly
    assert user is not None
    assert user["username"] == test_user_data["username"]
    assert user["email"] == test_user_data["email"]
    assert user["roles"] == test_user_data["roles"]
    assert "hashed_password" in user


def test_get_user_not_found(mock_db, mock_db_session):
    """Test retrieving non-existent user from database."""
    # Set up mock to return None for fetchone
    mock_db_session.execute.return_value.fetchone.return_value = None
    
    user = get_user(mock_db, "nonexistentuser")
    
    # Verify correct SQL query was executed
    mock_db_session.execute.assert_called_once()
    
    # Verify None was returned
    assert user is None


def test_authenticate_user_success(mock_db, mock_db_session, test_user_data):
    """Test successful user authentication."""
    setup_db_user_query(mock_db_session, test_user_data)
    
    user = authenticate_user(mock_db, test_user_data["username"], "testpassword")
    
    # Verify user was authenticated
    assert user is not None
    assert user.username == test_user_data["username"]
    assert user.roles == test_user_data["roles"]
    
    # Verify no password hash in user object
    assert not hasattr(user, "hashed_password")


def test_authenticate_user_wrong_password(mock_db, mock_db_session, test_user_data):
    """Test authentication with wrong password."""
    setup_db_user_query(mock_db_session, test_user_data)
    
    user = authenticate_user(mock_db, test_user_data["username"], "wrongpassword")
    
    # Verify authentication failed
    assert user is None


def test_authenticate_user_not_found(mock_db, mock_db_session):
    """Test authentication with non-existent user."""
    # Set up mock to return None for fetchone
    mock_db_session.execute.return_value.fetchone.return_value = None
    
    user = authenticate_user(mock_db, "nonexistentuser", "testpassword")
    
    # Verify authentication failed
    assert user is None


#################################################
# Current User Tests
#################################################

@pytest.mark.asyncio
async def test_get_current_user_valid_token(mock_db, mock_db_session, mock_request, test_user_data, sample_token_data, mock_settings):
    """Test getting current user with valid token."""
    setup_db_user_query(mock_db_session, test_user_data)
    
    # Use a fake token, we'll mock the decode function
    token = "fake.jwt.token"
    
    # Patch JWT decode to avoid actual cryptographic verification
    with patch("app.core.security.get_db", return_value=mock_db):
        with patch("app.core.security.jwt.decode") as mock_decode:

            # Set up mock decode to return expected payload
            mock_decode.return_value = {
                "sub": test_user_data["username"],
                "roles": test_user_data["roles"],
                "exp": (datetime.datetime.now(datetime.timezone.utc) + timedelta(minutes=15)).timestamp()
            }
            
            # Patch is_allowed_ip to return True
            with patch("app.core.security.is_allowed_ip", return_value=True):
                # Patch log_auth_event to do nothing
                with patch("app.core.security.log_auth_event"):
                    user = await get_current_user(mock_request, mock_db, token)
                    
                    # Verify user was returned correctly
                    assert user is not None
                    assert user.username == test_user_data["username"]
                    assert user.roles == test_user_data["roles"]


@pytest.mark.asyncio
async def test_get_current_user_invalid_token(mock_db, mock_request, mock_settings):
    """Test getting current user with invalid token."""
    # Patch JWT decode to raise PyJWTError (not JWTError)
    with patch("app.core.security.get_db", return_value=mock_db):
        with patch("app.core.security.jwt.decode") as mock_decode:
            # Fixed test using the JWTError from jose
            mock_decode.side_effect = JWTError("Invalid token")  # or jose.JWTError if imported separately
            
            # Patch is_allowed_ip to return True
            with patch("app.core.security.is_allowed_ip", return_value=True):
                # Patch log_auth_event to do nothing
                with patch("app.core.security.log_auth_event"):
                    with pytest.raises(HTTPException) as exc_info:
                        await get_current_user(mock_request, mock_db, "invalid_token")
                    
                    # Verify correct exception was raised
                    assert exc_info.value.status_code == 401
                    assert "Could not validate credentials" in exc_info.value.detail


@pytest.mark.asyncio
async def test_get_current_user_ip_restricted(mock_db, mock_request, mock_settings):
    """Test getting current user from restricted IP."""
    # Change request IP to restricted one
    mock_request.client.host = "1.2.3.4"
    
    # Patch is_allowed_ip to return False
    with patch("app.core.security.is_allowed_ip", return_value=False):
        # Patch log_auth_event to do nothing
        with patch("app.core.security.log_auth_event"):
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user(mock_request, mock_db, "valid_token")
            
            # Verify correct exception was raised
            assert exc_info.value.status_code == 403
            assert "Access denied from your IP address" in exc_info.value.detail


@pytest.mark.asyncio
async def test_get_current_active_user(test_user):
    """Test getting current active user."""
    user = await get_current_active_user(test_user)
    
    # Verify user was returned unchanged
    assert user is test_user


@pytest.mark.asyncio
async def test_get_current_active_user_disabled(test_user):
    """Test getting current active user when user is disabled."""
    # Create a disabled user
    disabled_user = User(
        username=test_user.username,
        email=test_user.email,
        roles=test_user.roles,
        disabled=True
    )
    
    with pytest.raises(HTTPException) as exc_info:
        await get_current_active_user(disabled_user)
    
    # Verify correct exception was raised
    assert exc_info.value.status_code == 400
    assert "Inactive user" in exc_info.value.detail


#################################################
# Role-Based Access Control Tests
#################################################

@pytest.mark.asyncio
async def test_has_role_authorized(test_user, mock_request):
    """Test role-based authorization with authorized user."""
    # Create role checker for trader role
    role_checker = has_role(["trader"])
    
    # Patch log_auth_event to do nothing
    with patch("app.core.security.log_auth_event"):
        user = await role_checker(mock_request, test_user)
        
        # Verify user was returned unchanged
        assert user is test_user


@pytest.mark.asyncio
async def test_has_role_unauthorized(test_user, mock_request):
    """Test role-based authorization with unauthorized user."""
    # Create role checker for admin role
    role_checker = has_role(["admin"])
    
    # Patch log_auth_event to do nothing and PostgresDB constructor
    with patch("app.core.security.log_auth_event"):
        with patch("app.core.security.PostgresDB"):
            with pytest.raises(HTTPException) as exc_info:
                await role_checker(mock_request, test_user)
            
            # Verify correct exception was raised
            assert exc_info.value.status_code == 403
            assert "Insufficient permissions" in exc_info.value.detail


#################################################
# IP Restriction Tests
#################################################

def test_is_allowed_ip_allowed(mock_settings):
    """Test IP restriction with allowed IP."""
    with patch("app.core.security.get_settings", return_value=mock_settings):
        # Test IP in first allowed range
        assert is_allowed_ip("192.168.1.100")
        
        # Test IP in second allowed range
        assert is_allowed_ip("10.1.2.3")


def test_is_allowed_ip_restricted():
    """Test IP restriction with restricted IP."""
    with patch("app.core.security.settings") as mock_settings:
        # Create a properly structured settings mock
        mock_settings.security.ALLOWED_IP_RANGES = ["192.168.0.0/16", "10.0.0.0/8"]
        
        # Test IP outside allowed ranges
        assert not is_allowed_ip("8.8.8.8")


def test_is_allowed_ip_invalid_ip():
    """Test IP restriction with invalid IP format."""
    with patch("app.core.security.settings") as mock_settings:
        # Create a properly structured settings mock
        mock_settings.security.ALLOWED_IP_RANGES = ["192.168.0.0/16"]
        
        # Test invalid IP format
        assert not is_allowed_ip("invalid-ip")


def test_is_allowed_ip_no_restrictions():
    """Test IP restriction with no restrictions configured."""
    # Create settings with no IP restrictions
    settings = MagicMock(spec=Settings)
    settings.security = MagicMock()
    
    # Remove ALLOWED_IP_RANGES attribute
    del settings.security.ALLOWED_IP_RANGES
    
    with patch("app.core.security.get_settings", return_value=settings):
        # All IPs should be allowed
        assert is_allowed_ip("8.8.8.8")


def test_is_allowed_ip_empty_restrictions(mock_settings):
    """Test IP restriction with empty restrictions list."""
    # Set empty allowed IP ranges
    mock_settings.security.ALLOWED_IP_RANGES = []
    
    with patch("app.core.security.get_settings", return_value=mock_settings):
        # All IPs should be allowed when empty list
        assert is_allowed_ip("8.8.8.8")


#################################################
# Audit Logging Tests
#################################################

def test_log_auth_event(mock_db, mock_db_session):
    """Test logging authentication event."""
    log_auth_event(
        mock_db,
        "login_success",
        "testuser",
        True,
        "192.168.1.100",
        "Test details"
    )
    
    # Verify correct SQL query was executed
    mock_db_session.execute.assert_called_once()
    mock_db_session.commit.assert_called_once()
    
    # Verify query parameters
    query_args = mock_db_session.execute.call_args[0][1]
    assert query_args["event_type"] == "login_success"
    assert query_args["username"] == "testuser"
    assert query_args["success"] is True
    assert query_args["client_ip"] == "192.168.1.100"
    assert query_args["details"] == "Test details"


def test_log_auth_event_exception(mock_db, mock_db_session):
    """Test logging authentication event with exception."""
    # Set up mock to raise exception
    mock_db_session.execute.side_effect = Exception("Database error")
    
    # Patch logger to capture error
    with patch("app.core.security.logger.error") as mock_error:
        log_auth_event(
            mock_db,
            "login_error",
            "testuser",
            False,
            "192.168.1.100"
        )
        
        # Verify error was logged
        mock_error.assert_called_once()
        assert "Error logging auth event" in mock_error.call_args[0][0]


#################################################
# User Management Tests
#################################################

def test_create_user_success(mock_db, mock_db_session):
    """Test creating a new user."""
    # Set up mock to return None for existing user check
    mock_db_session.execute.return_value.fetchone.return_value = None
    
    result = create_user(
        mock_db,
        "newuser",
        "new@example.com",
        "password123",
        "New User",
        ["trader", "analyst"]
    )
    
    # Verify user was created
    assert result is True
    
    # Verify correct SQL queries were executed
    assert mock_db_session.execute.call_count == 2
    mock_db_session.commit.assert_called_once()


def test_create_user_already_exists(mock_db, mock_db_session):
    """Test creating a user that already exists."""
    # Set up mock to return a result for existing user check
    mock_db_session.execute.return_value.fetchone.return_value = MagicMock()
    
    result = create_user(
        mock_db,
        "existinguser",
        "existing@example.com",
        "password123"
    )
    
    # Verify user creation failed
    assert result is False
    
    # Verify only the check query was executed
    assert mock_db_session.execute.call_count == 1
    mock_db_session.commit.assert_not_called()


def test_create_user_exception(mock_db, mock_db_session):
    """Test creating a user with database exception."""
    # Set up mock to return None for existing user check
    mock_db_session.execute.return_value.fetchone.return_value = None
    
    # Set up the mock to return None for existing user check first, then raise an exception
    mock_db_session.execute.side_effect = [
        MagicMock(fetchone=MagicMock(return_value=None)),  # First call returns None for user check
        Exception("Database error")  # Second call raises an exception
    ]
    
    # Patch logger to capture error
    with patch("app.core.security.logger.error") as mock_error:
        result = create_user(
            mock_db,
            "erroruser",
            "error@example.com",
            "password123"
        )
        
        # Verify user creation failed
        assert result is False
        
        # Verify error was logged
        mock_error.assert_called_once()
        assert "Error creating user" in mock_error.call_args[0][0]


def test_update_user_roles_success(mock_db, mock_db_session):
    """Test updating user roles."""
    # Set up mock to return 1 for rowcount (indicating success)
    execute_result = MagicMock()
    execute_result.rowcount = 1
    mock_db_session.execute.return_value = execute_result
    
    result = update_user_roles(
        mock_db,
        "testuser",
        ["admin", "trader"]
    )
    
    # Verify roles were updated
    assert result is True
    
    # Verify correct SQL query was executed
    mock_db_session.execute.assert_called_once()
    mock_db_session.commit.assert_called_once()
    
    # Verify query parameters
    query_args = mock_db_session.execute.call_args[0][1]
    assert query_args["username"] == "testuser"
    assert query_args["roles"] == "admin,trader"


def test_update_user_roles_user_not_found(mock_db, mock_db_session):
    """Test updating roles for non-existent user."""
    # Set up mock to return 0 for rowcount (indicating no rows updated)
    execute_result = MagicMock()
    execute_result.rowcount = 0
    mock_db_session.execute.return_value = execute_result
    
    result = update_user_roles(
        mock_db,
        "nonexistentuser",
        ["admin"]
    )
    
    # Verify update failed
    assert result is False


def test_disable_user_success(mock_db, mock_db_session):
    """Test disabling a user."""
    # Set up mock to return 1 for rowcount (indicating success)
    execute_result = MagicMock()
    execute_result.rowcount = 1
    mock_db_session.execute.return_value = execute_result
    
    result = disable_user(mock_db, "testuser")
    
    # Verify user was disabled
    assert result is True
    
    # Verify correct SQL query was executed
    mock_db_session.execute.assert_called_once()
    mock_db_session.commit.assert_called_once()


def test_enable_user_success(mock_db, mock_db_session):
    """Test enabling a user."""
    # Set up mock to return 1 for rowcount (indicating success)
    execute_result = MagicMock()
    execute_result.rowcount = 1
    mock_db_session.execute.return_value = execute_result
    
    result = enable_user(mock_db, "testuser")
    
    # Verify user was enabled
    assert result is True
    
    # Verify correct SQL query was executed
    mock_db_session.execute.assert_called_once()
    mock_db_session.commit.assert_called_once()


def test_change_password_success(mock_db, mock_db_session, test_user_data):
    """Test changing password."""
    # Set up mocks for authenticate_user
    with patch("app.core.security.authenticate_user") as mock_auth:
        mock_auth.return_value = User(
            username=test_user_data["username"],
            email=test_user_data["email"],
            roles=test_user_data["roles"]
        )
        
        result = change_password(
            mock_db,
            test_user_data["username"],
            "testpassword",
            "newpassword123"
        )
        
        # Verify password was changed
        assert result is True
        
        # Verify correct SQL query was executed
        mock_db_session.execute.assert_called_once()
        mock_db_session.commit.assert_called_once()


def test_change_password_authentication_failed(mock_db):
    """Test changing password with failed authentication."""
    # Mock authenticate_user to return None (authentication failed)
    with patch("app.core.security.authenticate_user", return_value=None):
        result = change_password(
            mock_db,
            "testuser",
            "wrongpassword",
            "newpassword123"
        )
        
        # Verify password change failed
        assert result is False


def test_admin_reset_password_success(mock_db, mock_db_session):
    """Test admin password reset."""
    # Create admin user
    admin_user = User(
        username="adminuser",
        email="admin@example.com",
        roles=[Roles.ADMIN]
    )
    
    # Set up mock to return 1 for rowcount (indicating success)
    execute_result = MagicMock()
    execute_result.rowcount = 1
    mock_db_session.execute.return_value = execute_result
    
    result = admin_reset_password(
        mock_db,
        admin_user,
        "testuser",
        "resetpassword123"
    )
    
    # Verify password was reset
    assert result is True
    
    # Verify correct SQL query was executed
    mock_db_session.execute.assert_called_once()
    mock_db_session.commit.assert_called_once()


def test_admin_reset_password_not_admin(mock_db):
    """Test password reset by non-admin user."""
    # Create non-admin user
    non_admin_user = User(
        username="regularuser",
        email="regular@example.com",
        roles=["trader"]
    )
    
    result = admin_reset_password(
        mock_db,
        non_admin_user,
        "testuser",
        "resetpassword123"
    )
    
    # Verify password reset failed
    assert result is False


def test_admin_reset_password_user_not_found(mock_db, mock_db_session):
    """Test admin password reset for non-existent user."""
    # Create admin user
    admin_user = User(
        username="adminuser",
        email="admin@example.com",
        roles=[Roles.ADMIN]
    )
    
    # Set up mock to return 0 for rowcount (indicating no rows updated)
    execute_result = MagicMock()
    execute_result.rowcount = 0
    mock_db_session.execute.return_value = execute_result
    
    result = admin_reset_password(
        mock_db,
        admin_user,
        "nonexistentuser",
        "resetpassword123"
    )
    
    # Verify password reset failed
    assert result is False