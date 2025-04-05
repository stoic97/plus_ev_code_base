"""
Comprehensive database management module for financial applications.

This module provides a unified interface for managing connections to multiple database systems:
- PostgreSQL for relational data (users, accounts, strategies)
- TimescaleDB for time-series data (market prices, indicators)
- MongoDB for flexible document storage (signals, analysis)
- Redis for high-speed caching

Features:
- Connection pooling
- Health monitoring
- Session management
- Transaction utilities
- Caching mechanisms
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime, timedelta
from enum import Enum, auto
from functools import wraps
from typing import Any, Dict, Generator, List, Optional, Tuple, Union, Callable

# Database drivers
import redis
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database as MongoDatabase
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Import our configuration
from app.core.config import get_settings, Settings

# Set up logging
logger = logging.getLogger(__name__)

# Base class for SQLAlchemy models
Base = declarative_base()


#################################################
# Database Type Enum
#################################################

class DatabaseType(Enum):
    """Enum for supported database types."""
    POSTGRESQL = auto()
    TIMESCALEDB = auto()
    MONGODB = auto()
    REDIS = auto()


#################################################
# Abstract Base Database Class
#################################################

class Database(ABC):
    """
    Abstract base class for database connections.
    Defines the interface that all database implementations must follow.
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the database connection.
        
        Args:
            settings: Application settings (uses singleton if not provided)
        """
        self.settings = settings or get_settings()
        self.is_connected = False
        logger.debug(f"Initializing {self.__class__.__name__}")
    
    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the database."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close the database connection."""
        pass
    
    @abstractmethod
    def check_health(self) -> bool:
        """
        Check if the database connection is healthy.
        
        Returns:
            True if connection is healthy, False otherwise
        """
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get database connection status information.
        
        Returns:
            Dictionary with status details
        """
        return {
            "name": self.__class__.__name__,
            "connected": self.is_connected
        }


#################################################
# PostgreSQL Implementation
#################################################

class PostgresDB(Database):
    """
    PostgreSQL database connection manager.
    
    Handles connection pooling, session management, and health checks
    for PostgreSQL databases.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize PostgreSQL connection manager."""
        super().__init__(*args, **kwargs)
        self.engine = None
        self.SessionLocal = None
    
    def connect(self) -> None:
        """
        Establish connection to PostgreSQL database using settings.
        Creates connection pool and session factory.
        """
        try:
            # Create engine with connection pool configuration
            self.engine = create_engine(
                str(self.settings.db.POSTGRES_URI),
                pool_pre_ping=True,  # Verify connections before using them
                pool_size=self.settings.db.POSTGRES_MIN_CONNECTIONS,
                max_overflow=self.settings.db.POSTGRES_MAX_CONNECTIONS - self.settings.db.POSTGRES_MIN_CONNECTIONS,
                pool_recycle=3600,  # Recycle connections after 1 hour
                connect_args={
                    "connect_timeout": 10,  # Connection timeout in seconds
                    "options": f"-c statement_timeout={self.settings.db.POSTGRES_STATEMENT_TIMEOUT}"  # Query timeout
                }
            )
            
            # Set up session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            # Register event listeners
            self._register_event_listeners()
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            self.is_connected = True
            logger.info("PostgreSQL connection established successfully")
        
        except Exception as e:
            self.is_connected = False
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
    
    def disconnect(self) -> None:
        """Close the PostgreSQL connection pool."""
        if self.engine:
            self.engine.dispose()
            self.is_connected = False
            logger.info("PostgreSQL connection closed")
    
    def check_health(self) -> bool:
        """
        Check if the PostgreSQL connection is healthy.
        
        Returns:
            True if connection is healthy, False otherwise
        """
        if not self.engine:
            return False
        
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"PostgreSQL health check failed: {e}")
            return False
    
    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """
        Get a database session as a context manager.
        
        Yields:
            SQLAlchemy session
        
        Example:
            ```
            with postgres_db.session() as db:
                result = db.query(User).filter(User.id == user_id).first()
            ```
        """
        if not self.SessionLocal:
            raise RuntimeError("PostgreSQL connection not initialized. Call connect() first.")
        
        session = self.SessionLocal()
        try:
            yield session
        except Exception as e:
            logger.error(f"Database session error: {e}")
            session.rollback()
            raise
        finally:
            session.close()
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a raw SQL query and return results as dictionaries.
        
        Args:
            query: SQL query string
            params: Query parameters
        
        Returns:
            List of dictionaries with query results
        
        Example:
            ```
            users = postgres_db.execute_query(
                "SELECT id, name FROM users WHERE active = :active",
                {"active": True}
            )
            ```
        """
        with self.session() as session:
            result = session.execute(text(query), params or {})
            # Convert to list of dictionaries using column names as keys
            return [dict(row) for row in result]
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed status information about the database connection."""
        status = super().get_status()
        
        if self.engine:
            # Add connection pool information
            pool = self.engine.pool
            status.update({
                "pool_size": pool.size(),
                "pool_checkedin": pool.checkedin(),
                "pool_overflow": pool.overflow(),
                "pool_checkedout": pool.checkedout(),
            })
        
        return status
    
    def _register_event_listeners(self) -> None:
        """Register SQLAlchemy event listeners for connection management."""
        if not self.engine:
            return
        
        @event.listens_for(self.engine, "checkout")
        def checkout(dbapi_connection, connection_record, connection_proxy):
            logger.debug(f"Connection checkout: {id(dbapi_connection)}")
        
        @event.listens_for(self.engine, "checkin")
        def checkin(dbapi_connection, connection_record):
            logger.debug(f"Connection checkin: {id(dbapi_connection)}")
        
        # Handle connection failures
        @event.listens_for(self.engine, "connect")
        def connect(dbapi_connection, connection_record):
            logger.debug(f"Connection established: {id(dbapi_connection)}")
        
        @event.listens_for(self.engine, "engine_connect")
        def engine_connect(connection):
            logger.debug("Engine connection event")
            # If connection was invalidated (e.g., after timeout), log it
            if connection.invalidated:
                logger.warning("Connection was invalidated, getting new connection")


#################################################
# TimescaleDB Implementation
#################################################

class TimescaleDB(PostgresDB):
    """
    TimescaleDB connection manager.
    
    Extends PostgreSQL functionality with TimescaleDB-specific features
    for time-series data management.
    """
    
    def connect(self) -> None:
        """
        Establish connection to TimescaleDB using settings.
        Creates connection pool and session factory.
        """
        try:
            # Create engine with connection pool configuration
            # Note: We use TimescaleDB specific settings here
            self.engine = create_engine(
                str(self.settings.db.TIMESCALE_URI),
                pool_pre_ping=True,
                pool_size=self.settings.db.TIMESCALE_MIN_CONNECTIONS,
                max_overflow=self.settings.db.TIMESCALE_MAX_CONNECTIONS - self.settings.db.TIMESCALE_MIN_CONNECTIONS,
                pool_recycle=3600,
                connect_args={
                    "connect_timeout": 10,
                    "options": f"-c statement_timeout={self.settings.db.TIMESCALE_STATEMENT_TIMEOUT}"
                }
            )
            
            # Set up session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            # Register event listeners from parent class
            self._register_event_listeners()
            
            # Test connection and verify TimescaleDB extension
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT extname FROM pg_extension WHERE extname = 'timescaledb'"))
                rows = result.fetchall()
                if len(rows) == 0:
                    logger.warning("TimescaleDB extension not found in database!")
            
            self.is_connected = True
            logger.info("TimescaleDB connection established successfully")
        
        except Exception as e:
            self.is_connected = False
            logger.error(f"Failed to connect to TimescaleDB: {e}")
            raise
    
    def get_time_bucket(self, table: str, time_column: str, interval: str,
                      aggregation: str, value_column: str,
                      start_time: datetime, end_time: datetime,
                      filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a TimescaleDB time_bucket query for time-series aggregation.
        
        Args:
            table: Table name
            time_column: Timestamp column name
            interval: Time bucket interval (e.g., '1 hour', '1 day')
            aggregation: Aggregation function (e.g., 'AVG', 'SUM', 'MAX')
            value_column: Column to aggregate
            start_time: Start of time range
            end_time: End of time range
            filters: Additional filter conditions
        
        Returns:
            List of time buckets with aggregated values
        
        Example:
            ```
            # Get hourly average prices for AAPL
            prices = timescale_db.get_time_bucket(
                'market_data', 
                'timestamp', 
                '1 hour',
                'AVG', 
                'price',
                datetime(2023, 1, 1),
                datetime(2023, 1, 2),
                {'symbol': 'AAPL'}
            )
            ```
        """
        # Build WHERE clause from filters
        where_clauses = []
        params = {"start_time": start_time, "end_time": end_time}
        
        # Add time range filter
        where_clauses.append(f"{time_column} >= :start_time AND {time_column} <= :end_time")
        
        # Add additional filters
        if filters:
            for i, (key, value) in enumerate(filters.items()):
                param_name = f"param_{i}"
                where_clauses.append(f"{key} = :{param_name}")
                params[param_name] = value
        
        where_clause = " AND ".join(where_clauses)
        
        # Build query
        query = f"""
        SELECT 
            time_bucket('{interval}', {time_column}) AS bucket,
            {aggregation}({value_column}) AS value
        FROM 
            {table}
        WHERE 
            {where_clause}
        GROUP BY 
            bucket
        ORDER BY 
            bucket ASC
        """
        
        # Execute query
        with self.session() as session:
            result = session.execute(text(query), params)
            return [{"bucket": row.bucket, "value": row.value} for row in result]
    
    def get_ohlcv(self, table: str, symbol_column: str, time_column: str, price_column: str, 
                volume_column: str, interval: str, symbol: str,
                start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """
        Get OHLCV (Open, High, Low, Close, Volume) data for a symbol.
        
        Args:
            table: Table name
            symbol_column: Symbol column name
            time_column: Timestamp column name
            price_column: Price column name
            volume_column: Volume column name
            interval: Time bucket interval (e.g., '1 minute', '1 hour', '1 day')
            symbol: Trading symbol
            start_time: Start of time range
            end_time: End of time range
        
        Returns:
            List of OHLCV data points
            
        Example:
            ```
            # Get daily OHLCV for Apple stock
            ohlcv = timescale_db.get_ohlcv(
                'market_data', 
                'symbol', 
                'timestamp', 
                'price', 
                'volume',
                '1 day', 
                'AAPL',
                datetime(2023, 1, 1),
                datetime(2023, 1, 31)
            )
            ```
        """
        query = f"""
        SELECT 
            time_bucket('{interval}', {time_column}) AS time,
            {symbol_column} AS symbol,
            FIRST({price_column}, {time_column}) AS open,
            MAX({price_column}) AS high,
            MIN({price_column}) AS low,
            LAST({price_column}, {time_column}) AS close,
            SUM({volume_column}) AS volume
        FROM 
            {table}
        WHERE 
            {symbol_column} = :symbol
            AND {time_column} >= :start_time
            AND {time_column} <= :end_time
        GROUP BY 
            time, symbol
        ORDER BY 
            time ASC
        """
        
        params = {
            "symbol": symbol,
            "start_time": start_time,
            "end_time": end_time
        }
        
        with self.session() as session:
            result = session.execute(text(query), params)
            return [
                {
                    "time": row.time,
                    "symbol": row.symbol,
                    "open": row.open,
                    "high": row.high,
                    "low": row.low,
                    "close": row.close,
                    "volume": row.volume
                }
                for row in result
            ]


#################################################
# MongoDB Implementation
#################################################

class MongoDB(Database):
    """
    MongoDB database connection manager.
    
    Provides connection management and utility methods for MongoDB operations.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize MongoDB connection manager."""
        super().__init__(*args, **kwargs)
        self.client = None
        self.db = None
    
    def connect(self) -> None:
        """
        Establish connection to MongoDB using settings.
        Sets up connection pool and database reference.
        """
        try:
            # Create MongoDB client with connection pool settings
            self.client = MongoClient(
                self.settings.db.MONGODB_URI,
                maxPoolSize=self.settings.db.MONGODB_MAX_POOL_SIZE,
                minPoolSize=self.settings.db.MONGODB_MIN_POOL_SIZE,
                maxIdleTimeMS=self.settings.db.MONGODB_MAX_IDLE_TIME_MS,
                connectTimeoutMS=self.settings.db.MONGODB_CONNECT_TIMEOUT_MS,
                serverSelectionTimeoutMS=5000,  # Timeout for server selection
                waitQueueTimeoutMS=1000  # How long to wait for a connection from the pool
            )
            
            # Get database reference
            self.db = self.client[self.settings.db.MONGODB_DB]
            
            # Test connection by accessing server info
            self.client.server_info()
            
            self.is_connected = True
            logger.info("MongoDB connection established successfully")
        
        except Exception as e:
            self.is_connected = False
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def disconnect(self) -> None:
        """Close the MongoDB connection."""
        if self.client:
            self.client.close()
            self.is_connected = False
            logger.info("MongoDB connection closed")
    
    def check_health(self) -> bool:
        """
        Check if the MongoDB connection is healthy.
        
        Returns:
            True if connection is healthy, False otherwise
        """
        if not self.client:
            return False
        
        try:
            self.client.admin.command('ping')
            return True
        except Exception as e:
            logger.error(f"MongoDB health check failed: {e}")
            return False
    
    def get_collection(self, collection_name: str) -> Collection:
        """
        Get a MongoDB collection reference.
        
        Args:
            collection_name: Name of the collection
        
        Returns:
            MongoDB collection reference
        
        Example:
            ```
            signals = mongo_db.get_collection("trading_signals")
            result = signals.find({"strategy_id": "my_strategy"})
            ```
        """
        if not self.db:
            raise RuntimeError("MongoDB connection not initialized. Call connect() first.")
        
        return self.db[collection_name]
    
    def find_one(self, collection_name: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Find a single document in a collection.
        
        Args:
            collection_name: Name of the collection
            query: MongoDB query dictionary
        
        Returns:
            Document dictionary or None if not found
        
        Example:
            ```
            user = mongo_db.find_one("users", {"email": "user@example.com"})
            ```
        """
        collection = self.get_collection(collection_name)
        return collection.find_one(query)
    
    def find_many(self, collection_name: str, query: Dict[str, Any], 
                 sort: Optional[List[tuple]] = None, limit: int = 0) -> List[Dict[str, Any]]:
        """
        Find multiple documents in a collection.
        
        Args:
            collection_name: Name of the collection
            query: MongoDB query dictionary
            sort: Optional sort specification [(field, direction), ...]
            limit: Maximum number of documents to return (0 = no limit)
        
        Returns:
            List of document dictionaries
        
        Example:
            ```
            # Get recent signals sorted by timestamp
            signals = mongo_db.find_many(
                "trading_signals", 
                {"strategy_id": "my_strategy"},
                sort=[("timestamp", -1)],
                limit=100
            )
            ```
        """
        collection = self.get_collection(collection_name)
        cursor = collection.find(query)
        
        if sort:
            cursor = cursor.sort(sort)
        
        if limit > 0:
            cursor = cursor.limit(limit)
        
        return list(cursor)
    
    def insert_one(self, collection_name: str, document: Dict[str, Any]) -> str:
        """
        Insert a single document into a collection.
        
        Args:
            collection_name: Name of the collection
            document: Document to insert
        
        Returns:
            Inserted document ID
        
        Example:
            ```
            signal_id = mongo_db.insert_one("trading_signals", {
                "strategy_id": "my_strategy",
                "symbol": "AAPL",
                "action": "BUY",
                "timestamp": datetime.utcnow()
            })
            ```
        """
        collection = self.get_collection(collection_name)
        result = collection.insert_one(document)
        return str(result.inserted_id)
    
    def insert_many(self, collection_name: str, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Insert multiple documents into a collection.
        
        Args:
            collection_name: Name of the collection
            documents: List of documents to insert
        
        Returns:
            List of inserted document IDs
        """
        collection = self.get_collection(collection_name)
        result = collection.insert_many(documents)
        return [str(id) for id in result.inserted_ids]
    
    def update_one(self, collection_name: str, query: Dict[str, Any], 
                  update: Dict[str, Any], upsert: bool = False) -> int:
        """
        Update a single document in a collection.
        
        Args:
            collection_name: Name of the collection
            query: MongoDB query to find document
            update: Update operations to apply
            upsert: Whether to insert if document doesn't exist
        
        Returns:
            Number of documents modified
        
        Example:
            ```
            modified = mongo_db.update_one(
                "orders",
                {"order_id": "12345"},
                {"$set": {"status": "filled", "filled_at": datetime.utcnow()}}
            )
            ```
        """
        collection = self.get_collection(collection_name)
        result = collection.update_one(query, update, upsert=upsert)
        return result.modified_count
    
    def delete_many(self, collection_name: str, query: Dict[str, Any]) -> int:
        """
        Delete multiple documents from a collection.
        
        Args:
            collection_name: Name of the collection
            query: MongoDB query to find documents to delete
        
        Returns:
            Number of documents deleted
        """
        collection = self.get_collection(collection_name)
        result = collection.delete_many(query)
        return result.deleted_count
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed status information about MongoDB connection."""
        status = super().get_status()
        
        if self.client:
            # Add MongoDB server information if available
            try:
                server_status = self.client.admin.command("serverStatus")
                status.update({
                    "server_version": server_status.get("version", "unknown"),
                    "connections": server_status.get("connections", {}).get("current", 0),
                    "uptime_seconds": server_status.get("uptime", 0)
                })
            except Exception:
                # If server status is not available, just use basic info
                pass
        
        return status


#################################################
# Redis Implementation
#################################################

class RedisDB(Database):
    """
    Redis database connection manager.
    
    Provides connection management and utility methods for Redis operations.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize Redis connection manager."""
        super().__init__(*args, **kwargs)
        self.client = None
        self.connection_pool = None
    
    def connect(self) -> None:
        """
        Establish connection to Redis using settings.
        Sets up connection pool and client.
        """
        try:
            # Create Redis connection pool
            self.connection_pool = redis.ConnectionPool(
                host=self.settings.db.REDIS_HOST,
                port=self.settings.db.REDIS_PORT,
                db=self.settings.db.REDIS_DB,
                password=self.settings.db.REDIS_PASSWORD,
                ssl=self.settings.db.REDIS_SSL,
                socket_timeout=self.settings.db.REDIS_SOCKET_TIMEOUT,
                socket_connect_timeout=self.settings.db.REDIS_SOCKET_CONNECT_TIMEOUT,
                max_connections=self.settings.db.REDIS_CONNECTION_POOL_SIZE
            )
            
            # Create Redis client using connection pool
            self.client = redis.Redis(connection_pool=self.connection_pool)
            
            # Test connection
            self.client.ping()
            
            self.is_connected = True
            logger.info("Redis connection established successfully")
        
        except Exception as e:
            self.is_connected = False
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def disconnect(self) -> None:
        """Close the Redis connection pool."""
        if self.connection_pool:
            self.connection_pool.disconnect()
            self.is_connected = False
            logger.info("Redis connection closed")
    
    def check_health(self) -> bool:
        """
        Check if the Redis connection is healthy.
        
        Returns:
            True if connection is healthy, False otherwise
        """
        if not self.client:
            return False
        
        try:
            self.client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False
    
    def _get_key(self, key: str) -> str:
        """
        Add prefix to key if configured.
        
        Args:
            key: Redis key
        
        Returns:
            Prefixed key
        """
        if self.settings.db.REDIS_KEY_PREFIX:
            return f"{self.settings.db.REDIS_KEY_PREFIX}{key}"
        return key
    
    def get(self, key: str) -> Optional[str]:
        """
        Get a string value from Redis.
        
        Args:
            key: Redis key
        
        Returns:
            String value or None if not found
        """
        if not self.client:
            raise RuntimeError("Redis connection not initialized. Call connect() first.")
        
        value = self.client.get(self._get_key(key))
        return value.decode('utf-8') if value is not None else None
    
    def set(self, key: str, value: str, expiration: Optional[int] = None) -> bool:
        """
        Set a string value in Redis.
        
        Args:
            key: Redis key
            value: String value
            expiration: Optional expiration time in seconds
        
        Returns:
            True if successful
        """
        if not self.client:
            raise RuntimeError("Redis connection not initialized. Call connect() first.")
        
        return self.client.set(self._get_key(key), value, ex=expiration)
    
    def delete(self, key: str) -> int:
        """
        Delete a key from Redis.
        
        Args:
            key: Redis key
        
        Returns:
            Number of keys deleted (0 or 1)
        """
        if not self.client:
            raise RuntimeError("Redis connection not initialized. Call connect() first.")
        
        return self.client.delete(self._get_key(key))
    
    def get_json(self, key: str) -> Optional[Any]:
        """
        Get a JSON value from Redis.
        
        Args:
            key: Redis key
        
        Returns:
            Decoded JSON object or None if not found
        
        Example:
            ```
            market_data = redis_db.get_json("market:summary:AAPL")
            ```
        """
        value = self.get(key)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                logger.warning(f"Failed to decode JSON for key {key}")
        return None
    
    def set_json(self, key: str, value: Any, expiration: Optional[int] = None) -> bool:
        """
        Set a JSON value in Redis.
        
        Args:
            key: Redis key
            value: Python object to encode as JSON
            expiration: Optional expiration time in seconds
        
        Returns:
            True if successful
        
        Example:
            ```
            redis_db.set_json(
                "market:summary:AAPL",
                {"price": 150.25, "change": 2.5, "updated_at": "2023-01-01T14:30:00Z"},
                expiration=300  # 5 minutes
            )
            ```
        """
        json_value = json.dumps(value)
        return self.set(key, json_value, expiration)
    
    def get_hash(self, key: str) -> Dict[str, str]:
        """
        Get all fields of a Redis hash.
        
        Args:
            key: Redis key
        
        Returns:
            Dictionary of field-value pairs
        """
        if not self.client:
            raise RuntimeError("Redis connection not initialized. Call connect() first.")
        
        result = self.client.hgetall(self._get_key(key))
        return {k.decode('utf-8'): v.decode('utf-8') for k, v in result.items()} if result else {}
    
    def set_hash(self, key: str, mapping: Dict[str, Any], expiration: Optional[int] = None) -> bool:
        """
        Set multiple fields of a Redis hash.
        
        Args:
            key: Redis key
            mapping: Dictionary of field-value pairs
            expiration: Optional expiration time in seconds
        
        Returns:
            True if successful
        """
        if not self.client:
            raise RuntimeError("Redis connection not initialized. Call connect() first.")
        
        prefixed_key = self._get_key(key)
        
        # Convert all values to strings
        string_mapping = {k: str(v) for k, v in mapping.items()}
        
        # Use pipeline for atomic operation
        with self.client.pipeline() as pipe:
            pipe.hmset(prefixed_key, string_mapping)
            if expiration:
                pipe.expire(prefixed_key, expiration)
            pipe.execute()
        
        return True
    
    def increment(self, key: str, amount: int = 1) -> int:
        """
        Increment a counter in Redis.
        
        Args:
            key: Redis key
            amount: Amount to increment by
        
        Returns:
            New counter value
        """
        if not self.client:
            raise RuntimeError("Redis connection not initialized. Call connect() first.")
        
        return self.client.incrby(self._get_key(key), amount)
    
    def cache(self, key: str, value: Any, expiration: Optional[int] = None) -> bool:
        """
        Cache a value in Redis.
        Automatically detects if the value should be stored as JSON or string.
        
        Args:
            key: Cache key
            value: Value to cache (string or JSON-serializable object)
            expiration: Optional expiration time in seconds
        
        Returns:
            True if successful
        """
        if isinstance(value, (str, bytes)):
            return self.set(key, value, expiration)
        else:
            return self.set_json(key, value, expiration)
        

#################################################
# Database Connection Management
#################################################

# Database instance singletons
_db_instances: Dict[DatabaseType, Database] = {}


def get_db_instance(db_type: DatabaseType) -> Database:
    """
    Get a database instance of the specified type.
    Uses a singleton pattern to reuse existing connections.
    
    Args:
        db_type: Database type enum value
        
    Returns:
        Database instance of the requested type
    
    Raises:
        ValueError: If an unsupported database type is requested
    """
    global _db_instances
    
    # Return existing instance if available
    if db_type in _db_instances and _db_instances[db_type].is_connected:
        return _db_instances[db_type]
    
    # Otherwise create a new instance
    settings = get_settings()
    
    if db_type == DatabaseType.POSTGRESQL:
        db = PostgresDB(settings=settings)
    elif db_type == DatabaseType.TIMESCALEDB:
        db = TimescaleDB(settings=settings)
    elif db_type == DatabaseType.MONGODB:
        db = MongoDB(settings=settings)
    elif db_type == DatabaseType.REDIS:
        db = RedisDB(settings=settings)
    else:
        raise ValueError(f"Unsupported database type: {db_type}")
    
    # Connect to database
    if not db.is_connected:
        db.connect()
    
    # Store instance in cache
    _db_instances[db_type] = db
    
    return db


def close_db_connections():
    """
    Close all database connections.
    Should be called when the application shuts down.
    """
    global _db_instances
    
    for db_type, db in _db_instances.items():
        try:
            if db.is_connected:
                logger.info(f"Closing {db_type.name} connection")
                db.disconnect()
        except Exception as e:
            logger.error(f"Error closing {db_type.name} connection: {e}")
    
    # Clear instances
    _db_instances = {}


@contextmanager
def db_session(db_type: DatabaseType) -> Generator[Session, None, None]:
    """
    Get a database session as a context manager.
    Only works for relational databases (PostgreSQL and TimescaleDB).
    
    Args:
        db_type: Database type enum value (must be PostgreSQL or TimescaleDB)
        
    Yields:
        SQLAlchemy session
    
    Raises:
        TypeError: If requested for a non-relational database
    
    Example:
        ```
        with db_session(DatabaseType.POSTGRESQL) as session:
            users = session.query(User).all()
        ```
    """
    if db_type not in (DatabaseType.POSTGRESQL, DatabaseType.TIMESCALEDB):
        raise TypeError(f"Cannot get session for non-relational database: {db_type}")
    
    # Get database instance
    db = get_db_instance(db_type)
    
    # We know this is a PostgresDB or TimescaleDB instance
    if isinstance(db, (PostgresDB, TimescaleDB)):
        with db.session() as session:
            yield session
    else:
        # This should never happen due to the check above
        raise TypeError(f"Database instance is not relational: {type(db)}")


def get_db() -> PostgresDB:
    """
    Dependency injection function for FastAPI to get a PostgreSQL database instance.
    
    Returns:
        PostgreSQL database instance
    
    Example:
        ```
        @app.get("/users")
        def get_users(db: PostgresDB = Depends(get_db)):
            with db.session() as session:
                users = session.query(User).all()
                return users
        ```
    """
    return get_db_instance(DatabaseType.POSTGRESQL)


def get_timescale_db() -> TimescaleDB:
    """
    Dependency injection function for FastAPI to get a TimescaleDB database instance.
    
    Returns:
        TimescaleDB database instance
    """
    return get_db_instance(DatabaseType.TIMESCALEDB)


def get_mongo_db() -> MongoDB:
    """
    Dependency injection function for FastAPI to get a MongoDB database instance.
    
    Returns:
        MongoDB database instance
    """
    return get_db_instance(DatabaseType.MONGODB)


def get_redis_db() -> RedisDB:
    """
    Dependency injection function for FastAPI to get a Redis database instance.
    
    Returns:
        Redis database instance
    """
    return get_db_instance(DatabaseType.REDIS)


# Cache decorator for efficient function result caching
def cache(ttl: int = None, key_prefix: str = None):
    """
    Decorator for caching function results in Redis.
    
    Args:
        ttl: Cache time-to-live in seconds (None = use default)
        key_prefix: Prefix for cache keys
        
    Returns:
        Decorated function
        
    Example:
        ```
        @cache(ttl=60)
        def get_market_data(symbol: str):
            # Expensive operation to get market data
            ...
            return data
        ```
    """
    def decorator(func):
        from functools import wraps
        import json
        import inspect
        import hashlib
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get Redis instance
            redis_db = get_db_instance(DatabaseType.REDIS)
            settings = get_settings()
            
            # Determine TTL
            cache_ttl = ttl
            if cache_ttl is None:
                cache_ttl = settings.performance.CACHE_TTL_DEFAULT
            
            # Generate cache key
            prefix = key_prefix or f"{func.__module__}.{func.__name__}"
            
            # Create signature for args and kwargs
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Serialize arguments to a string for hashing
            args_str = json.dumps(
                {k: str(v) for k, v in bound_args.arguments.items()},
                sort_keys=True
            )
            
            # Create hash of arguments
            args_hash = hashlib.md5(args_str.encode()).hexdigest()
            
            # Full cache key
            cache_key = f"{prefix}:{args_hash}"
            
            # Try to get from cache
            cached_value = redis_db.get_json(cache_key)
            if cached_value is not None:
                return cached_value
            
            # Call function
            result = func(*args, **kwargs)
            
            # Store in cache
            redis_db.set_json(cache_key, result, expiration=cache_ttl)
            
            return result
        
        return wrapper
    
    return decorator

