"""
Configuration file for pytest.

This file sets up the Python path and common fixtures for all tests.
"""
import os
import sys
import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

# Get the root directory of the project
root_dir = Path(__file__).parent.parent
app_dir = root_dir / "app"

# Add the root directory to Python path
sys.path.insert(0, str(root_dir))

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
    mock_settings = Mock()
    mock_settings.BaseSettings = type('BaseSettings', (), {})
    mock_settings.SettingsConfigDict = lambda **kwargs: {}
    sys.modules['pydantic_settings'] = mock_settings
    print("Note: Using mocked pydantic_settings module. Install with: pip install pydantic-settings")

@pytest.fixture
async def db_session():
    """
    Provide a mocked SQLAlchemy session.
    
    This fixture offers a transactional session that rolls back after each test.
    """
    session = AsyncMock()
    
    # Make add and add_all actually store objects
    added_objects = []
    
    async def mock_add(obj):
        added_objects.append(obj)
    
    async def mock_add_all(objects):
        added_objects.extend(objects)
    
    async def mock_commit():
        # When committing, assign IDs to objects if they don't have one
        for i, obj in enumerate(added_objects, 1):
            if hasattr(obj, 'id') and obj.id is None:
                obj.id = i
    
    async def mock_flush():
        # Similar to commit but without "permanent" persistence
        for i, obj in enumerate(added_objects, 1):
            if hasattr(obj, 'id') and obj.id is None:
                obj.id = i
    
    async def mock_rollback():
        # Clear the added objects on rollback
        added_objects.clear()
    
    session.add = mock_add
    session.add_all = mock_add_all
    session.commit = mock_commit
    session.flush = mock_flush
    session.rollback = mock_rollback
    
    # Mock query builder
    query_mock = AsyncMock()
    query_mock.filter.return_value = query_mock
    query_mock.all.return_value = []
    query_mock.first.return_value = None
    query_mock.limit.return_value = query_mock
    query_mock.offset.return_value = query_mock
    query_mock.order_by.return_value = query_mock
    
    session.query = AsyncMock(return_value=query_mock)
    
    return session

@pytest.fixture
async def mock_database_connections():
    """Mock database connections for testing."""
    postgres_mock = AsyncMock()
    redis_mock = AsyncMock()
    
    postgres_mock.is_connected = True
    postgres_mock.check_health.return_value = True
    redis_mock.is_connected = True
    redis_mock.check_health.return_value = True
    
    with patch('app.core.database.PostgresDB', return_value=postgres_mock) as p_mock, \
         patch('app.core.database.RedisDB', return_value=redis_mock) as r_mock:
        yield p_mock, r_mock

@pytest.fixture(autouse=True)
async def cleanup_tasks():
    """Cleanup any remaining tasks after each test."""
    yield
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

@pytest.fixture(scope="session")
def event_loop_policy():
    """Override event loop policy for tests."""
    return asyncio.DefaultEventLoopPolicy()

@pytest.fixture(scope="function")
def event_loop(event_loop_policy):
    """Create an instance of the default event loop for each test case."""
    loop = event_loop_policy.new_event_loop()
    yield loop
    loop.close()