"""
Pytest configuration and fixtures for unit tests.
This conftest.py is specifically for unit tests and uses mocks.
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add the project root to the path for unit tests
test_dir = Path(__file__).parent
project_root = test_dir.parent.parent
sys.path.insert(0, str(project_root))

# Set test environment variable
os.environ["TESTING"] = "True"

# Import your application with the correct path
try:
    from financial_app.app.main import app
    from financial_app.app.core.database import get_db
    from financial_app.app.core.security import get_current_user, get_current_active_user
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    print(f"Warning: Could not import application modules: {e}")

# Create mock user for authentication tests
def get_mock_user():
    """Return a mock user for testing."""
    if IMPORTS_AVAILABLE:
        try:
            from financial_app.app.core.security import User
            return User(
                username="testuser",
                email="test@example.com",
                full_name="Test User",
                roles=["observer"]
            )
        except ImportError:
            pass
    
    # Fallback mock user
    mock_user = MagicMock()
    mock_user.username = "testuser"
    mock_user.email = "test@example.com"
    mock_user.full_name = "Test User"
    mock_user.roles = ["observer"]
    mock_user.disabled = False
    return mock_user

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
    if not IMPORTS_AVAILABLE:
        pytest.skip("Application modules not available")
    
    # Override dependencies
    app.dependency_overrides[get_db] = get_mock_db
    app.dependency_overrides[get_current_user] = get_mock_user
    app.dependency_overrides[get_current_active_user] = get_mock_user
    
    # Create test client
    with TestClient(app) as test_client:
        yield test_client
    
    # Clean up after tests
    app.dependency_overrides.clear()

@pytest.fixture
def mock_db():
    """Provide a mock database instance for direct use in tests."""
    return get_mock_db()

@pytest.fixture
def mock_user():
    """Provide a mock user for direct use in tests."""
    return get_mock_user()

@pytest.fixture
def mock_settings():
    """Create mock settings for unit tests."""
    settings = MagicMock()
    # Create the nested security attribute
    settings.security = MagicMock()
    settings.security.SECRET_KEY = "test_secret_key"
    settings.security.ALGORITHM = "HS256"
    settings.security.ACCESS_TOKEN_EXPIRE_MINUTES = 30
    settings.security.REFRESH_TOKEN_EXPIRE_DAYS = 7
    settings.security.ALLOWED_IP_RANGES = ["192.168.0.0/16", "10.0.0.0/8"]
    
    # Add database settings
    settings.db = MagicMock()
    settings.db.POSTGRES_URI = "postgresql://test:test@localhost:5432/test_db"
    settings.db.POSTGRES_SERVER = "localhost"
    settings.db.POSTGRES_PORT = "5432"
    settings.db.POSTGRES_USER = "test"
    settings.db.POSTGRES_PASSWORD = "test"
    settings.db.POSTGRES_DB = "test_db"
    settings.db.USE_SSL = False
    settings.db.SSL_MODE = "disable"
    
    return settings

@pytest.fixture
def mock_db_session():
    """Create a mock database session for testing."""
    session_mock = MagicMock()
    session_mock.__enter__ = MagicMock(return_value=session_mock)
    session_mock.__exit__ = MagicMock(return_value=None)
    return session_mock

@pytest.fixture
def mock_request():
    """Create a mock FastAPI Request object."""
    request = MagicMock()
    request.client.host = "192.168.1.100"
    request.headers = {}
    request.method = "GET"
    return request

# Unit test specific fixtures
@pytest.fixture
def test_user_data():
    """Sample user data for testing."""
    return {
        "username": "testuser",
        "email": "test@example.com",
        "full_name": "Test User",
        "disabled": False,
        "roles": ["trader", "analyst"]
    }

@pytest.fixture
def sample_token_data():
    """Sample token data for testing."""
    return {
        "sub": "testuser",
        "roles": ["trader", "analyst"]
    }

# Helper function for setting up database query mocks
def setup_db_user_query(mock_db_session, test_user_data):
    """Set up mock database session to return test user data."""
    mock_row = MagicMock()
    mock_row.username = test_user_data["username"]
    mock_row.email = test_user_data["email"]
    mock_row.full_name = test_user_data.get("full_name")
    mock_row.disabled = test_user_data.get("disabled", False)
    mock_row.hashed_password = test_user_data.get("hashed_password")
    mock_row.roles = ",".join(test_user_data.get("roles", []))
    
    mock_db_session.execute.return_value.fetchone.return_value = mock_row
    return mock_row