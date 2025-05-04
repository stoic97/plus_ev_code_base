"""
Pytest configuration file for financial_app tests.

This module configures the test environment and provides fixtures for testing
without requiring connections to actual databases.
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# Add the proper path to the Python path
# The error shows your project structure has duplicated directories
# This will handle the nested structure
current_path = os.path.dirname(os.path.abspath(__file__))
project_parts = current_path.split(os.sep)

# Find the correct path by looking for financial_app
for i in range(len(project_parts), 0, -1):
    potential_path = os.sep.join(project_parts[:i])
    app_path = os.path.join(potential_path, "app")
    if os.path.exists(app_path) and os.path.isdir(app_path):
        if potential_path not in sys.path:
            sys.path.insert(0, potential_path)
        break

# Mock the pydantic_settings import if it's not installed
# This will only be used if the real module can't be imported
try:
    import pydantic_settings
except ImportError:
    # Create a simple mock for BaseSettings and SettingsConfigDict
    mock_settings = MagicMock()
    mock_settings.BaseSettings = type('BaseSettings', (), {})
    mock_settings.SettingsConfigDict = lambda **kwargs: {}
    sys.modules['pydantic_settings'] = mock_settings
    print("Note: Using mocked pydantic_settings module. Install with: pip install pydantic-settings")

@pytest.fixture
def db_session():
    """
    Provide a mocked SQLAlchemy session.
    
    This fixture offers a transactional session that rolls back after each test.
    """
    # Create a mock session
    session = MagicMock()
    
    # Make add and add_all actually store objects
    added_objects = []
    
    def mock_add(obj):
        added_objects.append(obj)
    
    def mock_add_all(objects):
        added_objects.extend(objects)
    
    def mock_commit():
        # When committing, assign IDs to objects if they don't have one
        for i, obj in enumerate(added_objects, 1):
            if hasattr(obj, 'id') and obj.id is None:
                obj.id = i
    
    def mock_flush():
        # Similar to commit but without "permanent" persistence
        for i, obj in enumerate(added_objects, 1):
            if hasattr(obj, 'id') and obj.id is None:
                obj.id = i
    
    def mock_rollback():
        # Clear the added objects on rollback
        added_objects.clear()
    
    session.add = mock_add
    session.add_all = mock_add_all
    session.commit = mock_commit
    session.flush = mock_flush
    session.rollback = mock_rollback
    
    # Mock query builder
    query_mock = MagicMock()
    query_mock.filter.return_value = query_mock
    query_mock.all.return_value = []
    query_mock.first.return_value = None
    query_mock.limit.return_value = query_mock
    query_mock.offset.return_value = query_mock
    query_mock.order_by.return_value = query_mock
    
    session.query = MagicMock(return_value=query_mock)
    
    return session

# Mock database utilities to avoid actual database connections
@pytest.fixture(autouse=True)
def mock_database_connections():
    """Automatically mock all database connections for all tests."""
    with patch('app.core.database.PostgresDB') as postgres_mock, \
         patch('app.core.database.TimescaleDB') as timescale_mock, \
         patch('app.core.database.MongoDB') as mongo_mock, \
         patch('app.core.database.RedisDB') as redis_mock:
        
        # Configure all mocks to behave properly
        for db_mock in [postgres_mock, timescale_mock, mongo_mock, redis_mock]:
            db_mock.return_value.is_connected = True
            db_mock.return_value.check_health.return_value = True
            
        yield {
            'postgres': postgres_mock,
            'timescale': timescale_mock,
            'mongodb': mongo_mock,
            'redis': redis_mock
        }