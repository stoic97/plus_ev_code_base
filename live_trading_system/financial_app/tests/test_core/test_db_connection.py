# tests/test_database_manager.py
import pytest
import os
from unittest.mock import MagicMock, patch

from app.core.database import PostgresDB, MongoDB, RedisDB, TimescaleDB


# Fixtures for mock settings
@pytest.fixture
def mock_settings():
    settings = MagicMock()
    
    # PostgreSQL settings
    settings.db.POSTGRES_URI = "postgresql://postgres:postgres@localhost:5432/test_db"
    settings.db.POSTGRES_MIN_CONNECTIONS = 1
    settings.db.POSTGRES_MAX_CONNECTIONS = 3
    settings.db.POSTGRES_STATEMENT_TIMEOUT = 10000
    
    # MongoDB settings
    settings.db.MONGODB_URI = "mongodb://localhost:27017/test_db"
    settings.db.MONGODB_DB = "test_db"
    settings.db.MONGODB_MAX_POOL_SIZE = 10
    settings.db.MONGODB_MIN_POOL_SIZE = 1
    settings.db.MONGODB_MAX_IDLE_TIME_MS = 10000
    settings.db.MONGODB_CONNECT_TIMEOUT_MS = 5000
    
    # Redis settings
    settings.db.REDIS_HOST = "localhost"
    settings.db.REDIS_PORT = 6379
    settings.db.REDIS_DB = 1
    settings.db.REDIS_PASSWORD = None
    settings.db.REDIS_SSL = False
    settings.db.REDIS_SOCKET_TIMEOUT = 2.0
    settings.db.REDIS_SOCKET_CONNECT_TIMEOUT = 1.0
    settings.db.REDIS_CONNECTION_POOL_SIZE = 10
    settings.db.REDIS_KEY_PREFIX = "test:"
    
    return settings

# Add these fixtures with correct patching paths
@pytest.fixture
def mock_sqlalchemy_engine():
    """Mock SQLAlchemy engine to avoid real connections."""
    with patch('app.core.database.create_engine') as mock_create:
        # Configure mock engine
        mock_engine = MagicMock()
        
        # Mock the event system
        with patch('app.core.database.event') as mock_event:
            # Make listens_for a no-op that returns the function unchanged
            mock_event.listens_for = lambda target, event_name: lambda fn: fn
            
            # Mock sessionmaker
            mock_session_factory = MagicMock()
            mock_session = MagicMock()
            mock_session_factory.return_value = mock_session
            
            # Session context manager mock
            mock_session_ctx = MagicMock()
            mock_session.return_value = mock_session_ctx
            mock_session_ctx.__enter__.return_value = mock_session
            
            # Connection mock
            mock_connection = MagicMock()
            mock_result = MagicMock()
            mock_fetchone = MagicMock()
            
            # Set up the method chain
            mock_fetchone.test = 1
            mock_result.fetchone.return_value = mock_fetchone
            mock_session.execute.return_value = mock_result
            
            # Engine connection
            mock_conn_ctx = MagicMock()
            mock_conn_ctx.__enter__.return_value = mock_connection
            mock_engine.connect.return_value = mock_conn_ctx
            
            # Return values
            mock_create.return_value = mock_engine
            
            # Patch sessionmaker as well
            with patch('app.core.database.sessionmaker', return_value=mock_session_factory):
                yield mock_create


@pytest.fixture
def mock_mongo_client():
    """Mock MongoDB client."""
    with patch('app.core.database.MongoClient') as mock_class:
        # Create mock instance with required attributes
        mock_instance = MagicMock()
        mock_admin = MagicMock()
        mock_db = MagicMock()
        mock_collection = MagicMock()
        
        # Configure the chain
        mock_instance.admin = mock_admin
        mock_instance.server_info.return_value = {"version": "4.4.0"}
        mock_admin.command.return_value = {"ok": 1}
        mock_instance.__getitem__.return_value = mock_db
        mock_db.__getitem__.return_value = mock_collection
        
        # Return the mocked instance when instantiated
        mock_class.return_value = mock_instance
        
        yield mock_class

@pytest.fixture
def mock_redis_client():
    """Mock Redis client."""
    with patch('app.core.database.redis.Redis') as mock_redis:
        # Create mock instances
        mock_instance = MagicMock()
        
        # Configure chain and return values
        mock_instance.ping.return_value = True
        
        # Important: Configure the get/set methods
        mock_get_result = MagicMock()
        mock_get_result.decode.return_value = "test_value"
        mock_instance.get.return_value = mock_get_result
        mock_instance.set.return_value = True
        mock_instance.delete.return_value = 1
        
        mock_redis.return_value = mock_instance
        
        # Also patch connection pool
        with patch('app.core.database.redis.ConnectionPool') as mock_pool:
            mock_pool_instance = MagicMock()
            mock_pool.return_value = mock_pool_instance
            yield mock_redis

class TestDatabaseManager:
    """Basic tests for database manager functionality."""
    
    def test_postgres_connection(self, mock_settings, mock_sqlalchemy_engine):
        """Test PostgreSQL connection using mocks."""
        # Create database manager with test settings
        postgres_db = PostgresDB(settings=mock_settings)
        
        # Patch the _register_event_listeners method to do nothing
        with patch.object(postgres_db, '_register_event_listeners', return_value=None):
            try:
                # Test connection
                postgres_db.connect()
                assert postgres_db.is_connected == True
                assert postgres_db.check_health() == True
                
                # Test basic query execution
                with postgres_db.session() as session:
                    result = session.execute("SELECT 1 as test")
                    assert result.fetchone().test == 1
            
            finally:
                # Clean up
                if postgres_db.is_connected:
                    postgres_db.disconnect()
                    assert postgres_db.is_connected == False
    
    def test_mongo_connection(self, mock_settings, mock_mongo_client):
        """Test MongoDB connection using mocks."""
        # Create database manager with test settings
        mongo_db = MongoDB(settings=mock_settings)
        
        try:
            # Test connection
            mongo_db.connect()
            assert mongo_db.is_connected == True
            assert mongo_db.check_health() == True
            
            # Test basic collection access
            collection = mongo_db.get_collection("test_collection")
            assert collection is not None
        
        finally:
            # Clean up
            if mongo_db.is_connected:
                mongo_db.disconnect()
                assert mongo_db.is_connected == False
    
    def test_redis_connection(self, mock_settings, mock_redis_client):
        """Test Redis connection using mocks."""
        # Create database manager with test settings
        redis_db = RedisDB(settings=mock_settings)
        
        try:
            # Test connection
            redis_db.connect()
            assert redis_db.is_connected == True
            assert redis_db.check_health() == True
            
            # Test basic set/get operations
            redis_db.set("test_key", "test_value")
            assert redis_db.get("test_key") == "test_value"
            redis_db.delete("test_key")
        
        finally:
            # Clean up
            if redis_db.is_connected:
                redis_db.disconnect()
                assert redis_db.is_connected == False