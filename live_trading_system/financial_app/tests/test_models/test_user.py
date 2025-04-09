"""
Unit tests for the user models.

Tests cover authentication, permission handling, API key management,
session control, and other critical user functionality.
"""

import pytest
import uuid
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from sqlalchemy import create_engine, Column, String
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.types import TypeDecorator
from sqlalchemy.dialects.postgresql import UUID

from app.models.user import User, Role, UserSession, APIKey, PasswordReset, AuditLog, user_roles
from app.core.database import Base
from app.core.security import verify_password


# SQLite-compatible UUID type
class GUID(TypeDecorator):
    """Platform-independent GUID type.
    Uses PostgreSQL's UUID type, otherwise uses String(36).
    """
    impl = String(36)
    cache_ok = True
    
    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(UUID())
        else:
            return dialect.type_descriptor(String(36))
    
    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            return str(value)
        else:
            if not isinstance(value, uuid.UUID):
                return str(uuid.UUID(value))
            else:
                return str(value)
    
    def process_result_value(self, value, dialect):
        if value is None:
            return value
        else:
            if not isinstance(value, uuid.UUID):
                value = uuid.UUID(value)
            return value


# Patch the UUID type in the models for testing
def patch_uuid_columns():
    """Patch UUID columns to use the SQLite-compatible GUID type."""
    for table in Base.metadata.tables.values():
        for column in table.columns:
            if isinstance(column.type, UUID):
                column.type = GUID()


# Create a fresh database for each test class
@pytest.fixture(scope='function')
def db_engine():
    """Create an in-memory database engine for testing."""
    # Patch UUID columns before creating tables
    patch_uuid_columns()
    
    # Create engine and tables
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    
    yield engine
    
    # Clean up after each test
    Base.metadata.drop_all(engine)


@pytest.fixture(scope='function')
def db_session(db_engine):
    """Create a database session for each test."""
    TestingSessionLocal = sessionmaker(bind=db_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def admin_role(db_session):
    """Create an admin role for testing."""
    role = Role(
        name="admin",
        description="Administrator role",
        permissions={"permissions": ["user:create", "user:read", "user:update", "user:delete", "kill_switch"]}
    )
    db_session.add(role)
    db_session.commit()
    db_session.refresh(role)
    return role


@pytest.fixture
def trader_role(db_session):
    """Create a trader role for testing."""
    role = Role(
        name="trader",
        description="Trader role",
        permissions={"permissions": ["trade:execute", "trade:read"]}
    )
    db_session.add(role)
    db_session.commit()
    db_session.refresh(role)
    return role


@pytest.fixture
def test_user(db_session):
    """Create a test user without roles."""
    user = User(
        email="test@example.com",
        username="testuser",
        full_name="Test User",
        algorithm_access={"algorithms": ["basic_ml", "trend_following"]},
        environment_access={"environments": ["dev", "test"]},
        trading_limits={"limits": {"equities": 100000.0, "options": 50000.0}}
    )
    user.set_password("password123")
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def admin_user(db_session, admin_role):
    """Create an admin user for testing."""
    user = User(
        email="admin@example.com",
        username="adminuser",
        full_name="Admin User",
        is_superuser=True,
        algorithm_access={"algorithms": ["basic_ml", "trend_following", "deep_learning"]},
        environment_access={"environments": ["dev", "test", "prod"]},
        trading_limits={"limits": {"equities": 500000.0, "options": 250000.0}},
        has_kill_switch_access=True
    )
    user.set_password("adminpass123")
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    
    # Add role in a separate transaction to avoid SQLite issues
    user.roles.append(admin_role)
    db_session.commit()
    db_session.refresh(user)
    
    return user


@pytest.fixture
def trader_user(db_session, trader_role):
    """Create a trader user for testing."""
    user = User(
        email="trader@example.com",
        username="traderuser",
        full_name="Trader User",
        algorithm_access={"algorithms": ["basic_ml"]},
        environment_access={"environments": ["dev"]},
        trading_limits={"limits": {"equities": 100000.0}}
    )
    user.set_password("traderpass123")
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    
    # Add role in a separate transaction
    user.roles.append(trader_role)
    db_session.commit()
    db_session.refresh(user)
    
    return user


@pytest.fixture
def active_session(db_session, test_user):
    """Create an active session for test_user."""
    session = UserSession(
        user_id=test_user.id,
        token_id="test_token_id",
        ip_address="127.0.0.1",
        user_agent="Test Browser",
        expires_at=datetime.utcnow() + timedelta(hours=1),
        environment="test"
    )
    db_session.add(session)
    db_session.commit()
    db_session.refresh(session)
    return session


@pytest.fixture
def api_key(db_session, test_user):
    """Create an API key for test_user."""
    # In a real app, you'd hash the key
    api_key = APIKey(
        user_id=test_user.id,
        name="Test API Key",
        key_prefix="test123",
        key_hash="hashed_api_key",
        permissions={"trade": True, "read": True},
        environment="test",
        rate_limit=100,
        expires_at=datetime.utcnow() + timedelta(days=30)
    )
    db_session.add(api_key)
    db_session.commit()
    db_session.refresh(api_key)
    return api_key


class TestUserModel:
    """Test cases for the User model."""

    def test_password_hashing(self, test_user):
        """Test that passwords are properly hashed and can be verified."""
        # Password should be hashed
        assert test_user.hashed_password != "password123"
        
        # Correct password should verify
        assert test_user.verify_password("password123") is True
        
        # Incorrect password should not verify
        assert test_user.verify_password("wrongpassword") is False

    def test_password_change(self, test_user, db_session):
        """Test changing a user's password."""
        # Get the current password changed timestamp
        original_timestamp = test_user.password_changed_at
        
        # Change the password
        test_user.set_password("newpassword456")
        db_session.commit()
        db_session.refresh(test_user)
        
        # Password should be changed and timestamp updated
        assert test_user.verify_password("newpassword456") is True
        assert test_user.verify_password("password123") is False
        assert test_user.password_changed_at > original_timestamp

    def test_login_tracking(self, test_user, db_session):
        """Test tracking successful and failed login attempts."""
        # Initial state
        assert test_user.failed_login_attempts == 0
        assert test_user.last_login_at is None
        
        # Record a failed login
        test_user.record_failed_login()
        db_session.commit()
        db_session.refresh(test_user)
        
        # Failed login should increment counter
        assert test_user.failed_login_attempts == 1
        assert test_user.last_failed_login_at is not None
        assert test_user.is_locked_out is False
        
        # Record more failed logins to trigger lockout
        for _ in range(4):
            test_user.record_failed_login()
        db_session.commit()
        db_session.refresh(test_user)
        
        # Account should be locked out
        assert test_user.failed_login_attempts == 5
        assert test_user.is_locked_out is True
        
        # Record a successful login
        test_user.record_successful_login()
        db_session.commit()
        db_session.refresh(test_user)
        
        # Successful login should reset counters
        assert test_user.failed_login_attempts == 0
        assert test_user.last_login_at is not None
        assert test_user.is_locked_out is False

    def test_role_permissions(self, admin_user, trader_user, test_user):
        """Test role-based permission checks."""
        # Admin should have admin permissions
        assert admin_user.has_permission("user:create") is True
        assert admin_user.has_permission("user:delete") is True
        
        # Trader should have trader permissions but not admin permissions
        assert trader_user.has_permission("trade:execute") is True
        assert trader_user.has_permission("user:delete") is False
        
        # Regular user should not have special permissions
        assert test_user.has_permission("trade:execute") is False
        assert test_user.has_permission("user:delete") is False
        
        # Superuser should have all permissions
        assert admin_user.has_permission("any:permission") is True

    def test_algorithm_access(self, admin_user, trader_user, test_user):
        """Test algorithm access controls."""
        # All users should have access to basic_ml
        assert admin_user.has_algorithm_access("basic_ml") is True
        assert trader_user.has_algorithm_access("basic_ml") is True
        assert test_user.has_algorithm_access("basic_ml") is True
        
        # Only admin and test_user have access to trend_following
        assert admin_user.has_algorithm_access("trend_following") is True
        assert trader_user.has_algorithm_access("trend_following") is False
        assert test_user.has_algorithm_access("trend_following") is True
        
        # Only admin has access to deep_learning
        assert admin_user.has_algorithm_access("deep_learning") is True
        assert trader_user.has_algorithm_access("deep_learning") is False
        assert test_user.has_algorithm_access("deep_learning") is False
        
        # No one has access to non-existent algorithm (except superuser)
        assert trader_user.has_algorithm_access("nonexistent") is False
        assert test_user.has_algorithm_access("nonexistent") is False
        assert admin_user.has_algorithm_access("nonexistent") is True

    def test_environment_access(self, admin_user, trader_user, test_user):
        """Test environment access controls."""
        # Admin should have access to all environments
        assert admin_user.can_trade_in_environment("dev") is True
        assert admin_user.can_trade_in_environment("test") is True
        assert admin_user.can_trade_in_environment("prod") is True
        
        # Trader should have limited environment access
        assert trader_user.can_trade_in_environment("dev") is True
        assert trader_user.can_trade_in_environment("test") is False
        assert trader_user.can_trade_in_environment("prod") is False
        
        # Test user should have dev and test access
        assert test_user.can_trade_in_environment("dev") is True
        assert test_user.can_trade_in_environment("test") is True
        assert test_user.can_trade_in_environment("prod") is False

    def test_trading_limits(self, admin_user, trader_user, test_user):
        """Test trading limit retrieval."""
        # Check equities limits
        assert admin_user.get_trading_limit("equities") == 500000.0
        assert trader_user.get_trading_limit("equities") == 100000.0
        assert test_user.get_trading_limit("equities") == 100000.0
        
        # Check options limits
        assert admin_user.get_trading_limit("options") == 250000.0
        assert trader_user.get_trading_limit("options") == 0.0  # Default
        assert test_user.get_trading_limit("options") == 50000.0
        
        # Check nonexistent instrument
        assert admin_user.get_trading_limit("futures") == 0.0
        assert trader_user.get_trading_limit("futures") == 0.0
        assert test_user.get_trading_limit("futures") == 0.0

    def test_get_scopes(self, admin_user, trader_user, test_user):
        """Test JWT scope generation."""
        # Admin scopes
        admin_scopes = admin_user.get_scopes()
        assert "user" in admin_scopes
        assert "role:admin" in admin_scopes
        assert "permission:user:create" in admin_scopes
        assert "env:prod" in admin_scopes
        assert "algorithm:deep_learning" in admin_scopes
        assert "kill_switch" in admin_scopes
        assert "superuser" in admin_scopes
        
        # Trader scopes
        trader_scopes = trader_user.get_scopes()
        assert "user" in trader_scopes
        assert "role:trader" in trader_scopes
        assert "permission:trade:execute" in trader_scopes
        assert "env:dev" in trader_scopes
        assert "algorithm:basic_ml" in trader_scopes
        assert "superuser" not in trader_scopes
        assert "kill_switch" not in trader_scopes
        
        # Basic user scopes
        user_scopes = test_user.get_scopes()
        assert "user" in user_scopes
        assert "role:admin" not in user_scopes
        assert "permission:trade:execute" not in user_scopes
        assert "env:dev" in user_scopes
        assert "env:test" in user_scopes
        assert "env:prod" not in user_scopes

    def test_to_dict(self, admin_user):
        """Test user serialization to dictionary."""
        # Basic serialization
        user_dict = admin_user.to_dict()
        assert user_dict["username"] == "adminuser"
        assert user_dict["email"] == "admin@example.com"
        assert user_dict["is_superuser"] is True
        assert "has_kill_switch_access" in user_dict
        assert "hashed_password" not in user_dict
        assert "trading_limits" not in user_dict
        
        # Include sensitive information
        sensitive_dict = admin_user.to_dict(include_sensitive=True)
        assert "trading_limits" in sensitive_dict
        assert "algorithm_access" in sensitive_dict
        assert "environment_access" in sensitive_dict
        assert "hashed_password" not in sensitive_dict  # Still excluded


class TestUserSessionModel:
    """Test cases for the UserSession model."""

    def test_session_active_status(self, active_session, db_session):
        """Test session active status checking."""
        # Session should be active
        assert active_session.is_active is True
        
        # Revoke the session
        active_session.revoke()
        db_session.commit()
        db_session.refresh(active_session)
        
        # Session should no longer be active
        assert active_session.is_active is False
        assert active_session.revoked_at is not None
        
        # Restore but set expiration in past
        active_session.is_revoked = False
        active_session.expires_at = datetime.utcnow() - timedelta(hours=1)
        db_session.commit()
        db_session.refresh(active_session)
        
        # Expired session should not be active
        assert active_session.is_active is False


class TestAPIKeyModel:
    """Test cases for the APIKey model."""

    def test_api_key_active_status(self, api_key, db_session):
        """Test API key active status checking."""
        # API key should be active
        assert api_key.is_active is True
        
        # Revoke the key
        api_key.revoke()
        db_session.commit()
        db_session.refresh(api_key)
        
        # Key should no longer be active
        assert api_key.is_active is False
        assert api_key.revoked_at is not None
        
        # Restore but set expiration in past
        api_key.is_revoked = False
        api_key.expires_at = datetime.utcnow() - timedelta(days=1)
        db_session.commit()
        db_session.refresh(api_key)
        
        # Expired key should not be active
        assert api_key.is_active is False

    def test_api_key_usage_tracking(self, api_key, db_session):
        """Test API key usage tracking."""
        # Initial state
        assert api_key.use_count == 0
        assert api_key.last_used_at is None
        
        # Record usage
        api_key.record_usage()
        db_session.commit()
        db_session.refresh(api_key)
        
        # Usage should be tracked
        assert api_key.use_count == 1
        assert api_key.last_used_at is not None
        
        # Record more usage
        api_key.record_usage()
        api_key.record_usage()
        db_session.commit()
        db_session.refresh(api_key)
        
        # Usage count should increment
        assert api_key.use_count == 3


class TestPasswordResetModel:
    """Test cases for the PasswordReset model."""

    def test_password_reset_token(self, test_user, db_session):
        """Test password reset token validation and usage."""
        # Create a reset token
        reset_token = PasswordReset(
            user_id=test_user.id,
            token_hash="hashed_token_value",
            expires_at=datetime.utcnow() + timedelta(hours=1)
        )
        db_session.add(reset_token)
        db_session.commit()
        db_session.refresh(reset_token)
        
        # Token should be valid
        assert reset_token.is_valid is True
        
        # Use the token
        reset_token.use_token(ip_address="192.168.1.1", user_agent="Test Browser")
        db_session.commit()
        db_session.refresh(reset_token)
        
        # Token should no longer be valid
        assert reset_token.is_valid is False
        assert reset_token.used_at is not None
        assert reset_token.ip_address == "192.168.1.1"
        assert reset_token.user_agent == "Test Browser"
        
        # Create an expired token
        expired_token = PasswordReset(
            user_id=test_user.id,
            token_hash="another_hashed_token",
            expires_at=datetime.utcnow() - timedelta(minutes=5)
        )
        db_session.add(expired_token)
        db_session.commit()
        db_session.refresh(expired_token)
        
        # Expired token should not be valid
        assert expired_token.is_valid is False


class TestAuditLogModel:
    """Test cases for the AuditLog model."""

    def test_audit_log_creation(self, test_user, db_session):
        """Test creating audit log entries."""
        # Create an audit log entry
        log_entry = AuditLog(
            user_id=test_user.id,
            action="login",
            ip_address="127.0.0.1",
            user_agent="Test Browser",
            target_type="user",
            target_id=str(test_user.id),
            environment="test",
            details={"successful": True}
        )
        db_session.add(log_entry)
        db_session.commit()
        db_session.refresh(log_entry)
        
        # Query the log back
        retrieved_log = db_session.query(AuditLog).filter_by(user_id=test_user.id).first()
        
        # Verify log data
        assert retrieved_log is not None
        assert retrieved_log.action == "login"
        assert retrieved_log.environment == "test"
        assert retrieved_log.details["successful"] is True
        assert retrieved_log.user_id == test_user.id