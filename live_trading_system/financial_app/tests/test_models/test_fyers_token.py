"""
Unit tests for Fyers Token model.

This module contains tests for the FyersToken model, including
creation, retrieval, update, deletion, and validation operations.
"""

import pytest
from datetime import datetime, timedelta, timezone
from typing import Any, Generator
from contextlib import contextmanager  # Fixed: was "from contextmanager import contextmanager"

from sqlalchemy import text
from app.models.fyers_token import FyersToken
from app.core.database import Base, DatabaseType, get_db_instance

# Test data
TEST_APP_ID = "TEST_APP-100"
TEST_ACCESS_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IlRlc3QgVXNlciIsImlhdCI6MTUxNjIzOTAyMiwiZXhwIjoxOTExMDIzNDU2fQ.dWWLQBUoc0z1LQxfD5XC8gYGwLT9oZnjUYBXMr9huE0"
TEST_REFRESH_TOKEN = "refresh_token_123"


def get_utc_now() -> datetime:
    """Get current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)


@contextmanager
def get_test_session() -> Generator[Any, None, None]:
    """Get a database session for testing."""
    # Get PostgreSQL database
    db = get_db_instance(DatabaseType.POSTGRESQL)
    
    # Use the synchronous session manager
    with db.session() as session:
        yield session


@pytest.fixture(scope="module")
def setup_database():
    """Set up the test database."""
    # Get PostgreSQL database
    db = get_db_instance(DatabaseType.POSTGRESQL)
    
    # Create the fyers_tokens table if it doesn't exist
    with db.session() as session:
        # Create table using SQLAlchemy text() function
        session.execute(text(f"""
        CREATE TABLE IF NOT EXISTS {FyersToken.__tablename__} (
            id SERIAL PRIMARY KEY,
            app_id VARCHAR(50) NOT NULL UNIQUE,
            access_token TEXT NOT NULL,
            refresh_token TEXT,
            expiry TIMESTAMP WITH TIME ZONE NOT NULL,
            is_active BOOLEAN NOT NULL DEFAULT true,
            token_type VARCHAR(20) NOT NULL DEFAULT 'access_token',
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
            updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
            token_metadata JSONB
        )
        """))
        session.commit()
    
    yield db


@pytest.fixture(scope="function")
def clean_db():
    """Ensure the database is clean before each test."""
    with get_test_session() as session:
        # Delete all tokens using SQLAlchemy text() function
        session.execute(text(f"DELETE FROM {FyersToken.__tablename__}"))
        session.commit()
    yield
    # Clean up after test
    with get_test_session() as session:
        session.execute(text(f"DELETE FROM {FyersToken.__tablename__}"))
        session.commit()


def test_create_table(setup_database):
    """Test that the token table can be created successfully."""
    # The table should already exist from the setup
    with get_test_session() as session:
        # Check if table exists
        result = session.execute(text(
            f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '{FyersToken.__tablename__}')"
        ))
        exists = result.scalar()
        assert exists is True


def test_save_token(clean_db):
    """Test saving a token to the database."""
    expiry = get_utc_now() + timedelta(hours=24)
    
    # Save token
    with get_test_session() as session:
        result = FyersToken.save_token(
            session, TEST_APP_ID, TEST_ACCESS_TOKEN, TEST_REFRESH_TOKEN, expiry
        )
        assert result is True
        
        # Verify token was saved
        token = FyersToken.get_by_app_id(session, TEST_APP_ID)
        
        assert token is not None
        assert token.app_id == TEST_APP_ID
        assert token.access_token == TEST_ACCESS_TOKEN
        assert token.refresh_token == TEST_REFRESH_TOKEN
        # Compare timezone-aware datetimes
        assert abs((token.expiry - expiry).total_seconds()) < 1


def test_get_by_app_id(clean_db):
    """Test retrieving a token by app_id."""
    # Insert test token
    expiry = get_utc_now() + timedelta(hours=24)
    
    with get_test_session() as session:
        FyersToken.save_token(
            session, TEST_APP_ID, TEST_ACCESS_TOKEN, TEST_REFRESH_TOKEN, expiry
        )
        
        # Retrieve token
        token = FyersToken.get_by_app_id(session, TEST_APP_ID)
        
        assert token is not None
        assert token.app_id == TEST_APP_ID
        assert token.access_token == TEST_ACCESS_TOKEN
        assert token.refresh_token == TEST_REFRESH_TOKEN


def test_update_token(clean_db):
    """Test updating an existing token."""
    with get_test_session() as session:
        # Insert initial token
        initial_expiry = get_utc_now() + timedelta(hours=24)
        FyersToken.save_token(
            session, TEST_APP_ID, TEST_ACCESS_TOKEN, TEST_REFRESH_TOKEN, initial_expiry
        )
        
        # Update with new values
        new_access_token = "new_" + TEST_ACCESS_TOKEN
        new_refresh_token = "new_" + TEST_REFRESH_TOKEN
        new_expiry = get_utc_now() + timedelta(hours=48)
        
        result = FyersToken.save_token(
            session, TEST_APP_ID, new_access_token, new_refresh_token, new_expiry
        )
        assert result is True
        
        # Verify token was updated
        token = FyersToken.get_by_app_id(session, TEST_APP_ID)
        
        assert token.access_token == new_access_token
        assert token.refresh_token == new_refresh_token
        # Compare timezone-aware datetimes
        assert abs((token.expiry - new_expiry).total_seconds()) < 1


def test_delete_token(clean_db):
    """Test deleting a token."""
    with get_test_session() as session:
        # Insert test token
        expiry = get_utc_now() + timedelta(hours=24)
        FyersToken.save_token(
            session, TEST_APP_ID, TEST_ACCESS_TOKEN, TEST_REFRESH_TOKEN, expiry
        )
        
        # Verify token exists
        token = FyersToken.get_by_app_id(session, TEST_APP_ID)
        assert token is not None
        
        # Delete token
        result = FyersToken.delete_token(session, TEST_APP_ID)
        assert result is True
        
        # Verify token was deleted
        token = FyersToken.get_by_app_id(session, TEST_APP_ID)
        assert token is None


def test_cleanup_expired_tokens(clean_db):
    """Test cleaning up expired tokens."""
    with get_test_session() as session:
        # Insert expired token
        expired_expiry = get_utc_now() - timedelta(hours=1)
        FyersToken.save_token(
            session, TEST_APP_ID, TEST_ACCESS_TOKEN, TEST_REFRESH_TOKEN, expired_expiry
        )
        
        # Insert valid token
        valid_app_id = "VALID_APP-100"
        valid_expiry = get_utc_now() + timedelta(hours=24)
        FyersToken.save_token(
            session, valid_app_id, TEST_ACCESS_TOKEN, TEST_REFRESH_TOKEN, valid_expiry
        )
        
        # Clean up expired tokens
        deleted_count = FyersToken.cleanup_expired_tokens(session)
        assert deleted_count == 1
        
        # Verify expired token was deleted
        expired_token = FyersToken.get_by_app_id(session, TEST_APP_ID)
        assert expired_token is None
        
        # Verify valid token still exists
        valid_token = FyersToken.get_by_app_id(session, valid_app_id)
        assert valid_token is not None


def test_validation_app_id_format(clean_db):
    """Test validation of app_id format."""
    # Try to create a token with invalid app_id format
    invalid_app_id = "INVALID_APP_ID"  # Missing hyphen
    expiry = get_utc_now() + timedelta(hours=24)
    
    with pytest.raises(ValueError, match="app_id must be in the format APP_ID-100"):
        # Create a token instance directly with invalid app_id
        token = FyersToken(
            app_id=invalid_app_id,
            access_token=TEST_ACCESS_TOKEN,
            refresh_token=TEST_REFRESH_TOKEN,
            expiry=expiry
        )


def test_validation_access_token_empty(clean_db):
    """Test validation that access_token cannot be empty."""
    expiry = get_utc_now() + timedelta(hours=24)
    
    with pytest.raises(ValueError, match="access_token cannot be empty"):
        # Create a token instance with empty access_token
        token = FyersToken(
            app_id=TEST_APP_ID,
            access_token="",  # Empty token
            refresh_token=TEST_REFRESH_TOKEN,
            expiry=expiry
        )


def test_token_expiry_check():
    """Test token expiry functionality."""
    with get_test_session() as session:
        # Create a token that expires in the future
        future_expiry = get_utc_now() + timedelta(hours=1)
        FyersToken.save_token(
            session, TEST_APP_ID, TEST_ACCESS_TOKEN, TEST_REFRESH_TOKEN, future_expiry
        )
        
        token = FyersToken.get_by_app_id(session, TEST_APP_ID)
        assert token is not None
        
        # Check that the token is not yet expired
        assert token.expiry > get_utc_now()
        
        # Clean up
        session.execute(text(f"DELETE FROM {FyersToken.__tablename__}"))
        session.commit()


def test_multiple_tokens():
    """Test handling multiple tokens for different app_ids."""
    with get_test_session() as session:
        # Create tokens for different app_ids
        app_id_1 = "APP1-100"
        app_id_2 = "APP2-100"
        expiry = get_utc_now() + timedelta(hours=24)
        
        FyersToken.save_token(session, app_id_1, TEST_ACCESS_TOKEN, TEST_REFRESH_TOKEN, expiry)
        FyersToken.save_token(session, app_id_2, TEST_ACCESS_TOKEN, TEST_REFRESH_TOKEN, expiry)
        
        # Verify both tokens exist
        token1 = FyersToken.get_by_app_id(session, app_id_1)
        token2 = FyersToken.get_by_app_id(session, app_id_2)
        
        assert token1 is not None
        assert token2 is not None
        assert token1.app_id == app_id_1
        assert token2.app_id == app_id_2
        
        # Clean up
        session.execute(text(f"DELETE FROM {FyersToken.__tablename__}"))
        session.commit()