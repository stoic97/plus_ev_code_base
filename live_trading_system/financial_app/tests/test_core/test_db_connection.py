# tests/test_database_manager.py
import pytest
import os
from unittest.mock import MagicMock, patch

from app.core.database import (
    PostgresDB, MongoDB, RedisDB, TimescaleDB,
    DatabaseType, get_db_instance, close_db_connections, db_session,
    get_db, get_mongo_db, get_redis_db, get_timescale_db, cache
)


# Fixtures for mock settings
@pytest.fixture
def mock_settings():
    settings = MagicMock()
    
    # PostgreSQL settings
    settings.db.POSTGRES_URI = "postgresql://postgres:postgres@localhost:5432/test_db"
    settings.db.POSTGRES_MIN_CONNECTIONS = 1
    settings.db.POSTGRES_MAX_CONNECTIONS = 3
    settings.db.POSTGRES_STATEMENT_TIMEOUT = 10000
    
    # TimescaleDB settings
    settings.db.TIMESCALE_URI = "postgresql://postgres:postgres@localhost:5432/timescale_db"
    settings.db.TIMESCALE_MIN_CONNECTIONS = 1
    settings.db.TIMESCALE_MAX_CONNECTIONS = 3
    settings.db.TIMESCALE_STATEMENT_TIMEOUT = 10000
    
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
    
    # Cache settings
    settings.performance = MagicMock()
    settings.performance.CACHE_TTL_DEFAULT = 300
    
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

# Fixture for patching get_settings
@pytest.fixture
def mock_get_settings(mock_settings):
    """Mock get_settings to return our mock settings."""
    with patch('app.core.database.get_settings', return_value=mock_settings):
        yield


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

    def test_timescale_connection(self, mock_settings, mock_sqlalchemy_engine):
        """Test TimescaleDB connection using mocks."""
        # Create database manager with test settings
        timescale_db = TimescaleDB(settings=mock_settings)
        
        # Patch the _register_event_listeners method to do nothing
        with patch.object(timescale_db, '_register_event_listeners', return_value=None):
            try:
                # Test connection
                timescale_db.connect()
                assert timescale_db.is_connected == True
                assert timescale_db.check_health() == True
                
                # Test basic query execution
                with timescale_db.session() as session:
                    result = session.execute("SELECT 1 as test")
                    assert result.fetchone().test == 1
            
            finally:
                # Clean up
                if timescale_db.is_connected:
                    timescale_db.disconnect()
                    assert timescale_db.is_connected == False


class TestDatabaseConnectionManagement:
    """Tests for database connection management functionality."""
    
    def test_get_db_instance_singleton(self, mock_get_settings, mock_sqlalchemy_engine, mock_mongo_client, mock_redis_client):
        """Test that get_db_instance returns a singleton instance."""
        with patch.object(PostgresDB, '_register_event_listeners', return_value=None):
            # Get PostgreSQL instance twice
            db1 = get_db_instance(DatabaseType.POSTGRESQL)
            db2 = get_db_instance(DatabaseType.POSTGRESQL)
            
            # Should be the same instance
            assert db1 is db2
            assert isinstance(db1, PostgresDB)
            assert db1.is_connected
            
            # Get MongoDB instance
            mongo_db = get_db_instance(DatabaseType.MONGODB)
            assert isinstance(mongo_db, MongoDB)
            assert mongo_db.is_connected
            
            # Get Redis instance
            redis_db = get_db_instance(DatabaseType.REDIS)
            assert isinstance(redis_db, RedisDB)
            assert redis_db.is_connected
            
            # Clean up
            close_db_connections()
    
    def test_close_db_connections(self, mock_get_settings, mock_sqlalchemy_engine, mock_mongo_client, mock_redis_client):
        """Test that close_db_connections closes all connections."""
        with patch.object(PostgresDB, '_register_event_listeners', return_value=None):
            # Create some database instances
            postgres_db = get_db_instance(DatabaseType.POSTGRESQL)
            mongo_db = get_db_instance(DatabaseType.MONGODB)
            redis_db = get_db_instance(DatabaseType.REDIS)
            
            # Make sure they're connected
            assert postgres_db.is_connected
            assert mongo_db.is_connected
            assert redis_db.is_connected
            
            # Close all connections
            close_db_connections()
            
            # Instances should still exist but be disconnected
            from app.core.database import _db_instances
            assert len(_db_instances) == 0
    
    def test_db_session_context_manager(self, mock_get_settings, mock_sqlalchemy_engine):
        """Test db_session context manager."""
        with patch.object(PostgresDB, '_register_event_listeners', return_value=None):
            # Use the context manager
            with db_session(DatabaseType.POSTGRESQL) as session:
                # Execute a query
                result = session.execute("SELECT 1 as test")
                assert result.fetchone().test == 1
            
            # Should raise TypeError for non-relational databases
            with pytest.raises(TypeError):
                with db_session(DatabaseType.MONGODB) as session:
                    pass
            
            # Clean up
            close_db_connections()
    
    def test_dependency_injection_functions(self, mock_get_settings, mock_sqlalchemy_engine, mock_mongo_client, mock_redis_client):
        """Test dependency injection functions."""
        with patch.object(PostgresDB, '_register_event_listeners', return_value=None):
            with patch.object(TimescaleDB, '_register_event_listeners', return_value=None):
                # Test get_db
                postgres_db = get_db()
                assert isinstance(postgres_db, PostgresDB)
                assert postgres_db.is_connected
                
                # Test get_timescale_db
                timescale_db = get_timescale_db()
                assert isinstance(timescale_db, TimescaleDB)
                assert timescale_db.is_connected
                
                # Test get_mongo_db
                mongo_db = get_mongo_db()
                assert isinstance(mongo_db, MongoDB)
                assert mongo_db.is_connected
                
                # Test get_redis_db
                redis_db = get_redis_db()
                assert isinstance(redis_db, RedisDB)
                assert redis_db.is_connected
                
                # Clean up
                close_db_connections()

    def test_cache_decorator(self, mock_get_settings, mock_redis_client):
        """Test cache decorator."""
        # Create a test function with the cache decorator
        @cache(ttl=60)
        def test_function(param1, param2):
            # This is a function that might be expensive to call
            return f"Result for {param1} and {param2}"
        
        # Configure Redis get_json and set_json for the test
        redis_instance = get_db_instance(DatabaseType.REDIS)
        redis_instance.get_json = MagicMock(return_value=None)  # First call returns None (cache miss)
        redis_instance.set_json = MagicMock(return_value=True)
        
        # Call the function - should miss cache and set cache
        result1 = test_function("value1", "value2")
        assert result1 == "Result for value1 and value2"
        redis_instance.set_json.assert_called_once()
        
        # Modify the mock to return a cached value
        redis_instance.get_json = MagicMock(return_value="Cached result")
        
        # Call the function again - should hit cache
        result2 = test_function("value1", "value2")
        assert result2 == "Cached result"
        
        # Clean up
        close_db_connections()