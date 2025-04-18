"""
Shared test fixtures for unit and integration tests.
"""
import pytest
from unittest.mock import MagicMock, patch
from contextlib import contextmanager

# We can import DatabaseType since it doesn't try to connect to any database
from app.core.database import DatabaseType

@pytest.fixture(scope="session", autouse=True)
def mock_db_connections():
    """
    Mock database connections to prevent actual connections during import.
    This is applied before importing MarketDataService.
    """
    with patch('app.core.database.get_db_instance') as mock_get_db:
        # Create mock database instances
        mock_timescale = MagicMock()
        mock_redis = MagicMock()
        mock_postgres = MagicMock()
        
        # Configure the mock to return appropriate mock DB instances
        def side_effect(db_type):
            if db_type == DatabaseType.TIMESCALEDB:
                return mock_timescale
            elif db_type == DatabaseType.REDIS:
                return mock_redis
            elif db_type == DatabaseType.POSTGRESQL:
                return mock_postgres
            else:
                return MagicMock()
                
        mock_get_db.side_effect = side_effect
        
        # Make mock database instances accessible for other fixtures
        mock_get_db.mock_timescale = mock_timescale
        mock_get_db.mock_redis = mock_redis
        mock_get_db.mock_postgres = mock_postgres
        
        # Prevent real connections by properly mocking connection methods
        mock_timescale.connect = MagicMock()
        mock_timescale.is_connected = True
        mock_timescale.engine = MagicMock()
        
        # Create a proper context manager for timescale session
        session_context = MagicMock()
        session_mock = MagicMock()
        session_context.__enter__.return_value = session_mock
        session_context.__exit__ = MagicMock()
        mock_timescale.session.return_value = session_context
        
        # Same for Redis
        mock_redis.connect = MagicMock()
        mock_redis.is_connected = True
        mock_redis.get_json = MagicMock(return_value=None)  # Default to cache miss
        mock_redis.set_json = MagicMock(return_value=True)  # Default success
        
        # Same for Postgres
        mock_postgres.connect = MagicMock()
        mock_postgres.is_connected = True
        mock_postgres.engine = MagicMock()
        mock_postgres.session.return_value = session_context
        
        # Mock the db_session context manager
        with patch('app.services.market_data.db_session') as mock_db_session:
            @contextmanager
            def fake_db_session(db_type):
                yield session_mock
                
            mock_db_session.side_effect = fake_db_session
            
            yield mock_get_db