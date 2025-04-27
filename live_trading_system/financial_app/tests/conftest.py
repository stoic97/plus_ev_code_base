"""
Pytest configuration and fixtures for all tests.
"""

import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

# Set test environment variable
os.environ["TESTING"] = "True"

# Import your application
from app.main import app
from app.core.database import get_db, DatabaseType
from app.core.security import get_current_user, get_current_active_user
from app.consumers.config.settings import KafkaSettings

# Create mock user for authentication tests
def get_mock_user():
    """Return a mock user for testing."""
    from app.core.security import User
    return User(
        username="testuser",
        email="test@example.com",
        full_name="Test User",
        roles=["observer"]
    )

# Create mock database function
def get_mock_db():
    """Return a mock database for testing."""
    mock_db = MagicMock()
    # Make session work as a context manager
    mock_session = MagicMock()
    mock_db.session.return_value.__enter__.return_value = mock_session
    mock_db.session.return_value.__exit__.return_value = None
    return mock_db

@pytest.fixture
def client():
    """
    Create a test client with mocked dependencies.
    This fixture can be used by any test that needs to make API requests.
    """
    # Override dependencies
    app.dependency_overrides[get_db] = get_mock_db
    app.dependency_overrides[get_current_user] = get_mock_user
    app.dependency_overrides[get_current_active_user] = get_mock_user
    
    # Create test client
    with TestClient(app) as test_client:
        yield test_client
    
    # Clean up after tests
    app.dependency_overrides.clear()

@pytest.fixture(scope="session")
def kafka_settings():
    """Return a KafkaSettings instance for testing."""
    return KafkaSettings()