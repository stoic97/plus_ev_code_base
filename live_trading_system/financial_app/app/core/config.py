"""
Configuration module using Pydantic for type validation and centralized settings management.
Implements a singleton pattern to ensure consistent configuration across the application.
"""

import os
import secrets
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

# Main pydantic imports
from pydantic import AnyUrl, Field, field_validator, model_validator

# Settings-specific imports
from pydantic_settings import BaseSettings, SettingsConfigDict

# For database URLs
from pydantic.networks import PostgresDsn, RedisDsn


# Custom settings helper function to handle nested environment variables
def get_environment_value(env_prefix: str, field_name: str, default_value: Any = None) -> Any:
    """
    Get value from environment variables with support for nested delimiters.
    Checks both formats: PREFIX_FIELD and PREFIX__FIELD to support both styles.
    """
    # Check for double underscore format (e.g., "DB__POSTGRES_SERVER")
    double_underscore_key = f"{env_prefix}__{field_name}"
    if double_underscore_key in os.environ:
        return os.environ[double_underscore_key]
    
    # Check for single underscore format (e.g., "DB_POSTGRES_SERVER")
    single_underscore_key = f"{env_prefix}_{field_name}"
    if single_underscore_key in os.environ:
        return os.environ[single_underscore_key]
    
    return default_value


class KafkaSettings(BaseSettings):
    """Kafka configuration settings."""
    
    BOOTSTRAP_SERVERS: List[str] = Field(
        default=["localhost:9092"], 
        description="Kafka bootstrap servers"
    )
    MARKET_DATA_TOPIC: str = Field(
        default="market-data", 
        description="Topic for market data streaming"
    )
    SIGNAL_TOPIC: str = Field(
        default="trading-signals", 
        description="Topic for trading signals"
    )
    GROUP_ID: str = Field(
        default="trading-app", 
        description="Consumer group ID"
    )
    AUTO_OFFSET_RESET: str = Field(
        default="latest", 
        description="Auto offset reset policy (latest or earliest)"
    )
    ENABLE_AUTO_COMMIT: bool = Field(
        default=False, 
        description="Enable auto commit for consumer offsets"
    )
    MAX_POLL_INTERVAL_MS: int = Field(
        default=300000, 
        description="Maximum delay between polls in ms"
    )
    MAX_POLL_RECORDS: int = Field(
        default=500, 
        description="Maximum records returned in a single poll"
    )
    SESSION_TIMEOUT_MS: int = Field(
        default=30000, 
        description="Session timeout in ms"
    )
    REQUEST_TIMEOUT_MS: int = Field(
        default=40000, 
        description="Request timeout in ms"
    )
    
    # Updated Config to model_config
    model_config = SettingsConfigDict(
        env_prefix="KAFKA_",
        case_sensitive=True,
        extra="allow"  # Allow extra attributes for env variables with double underscore format
    )
    
    @field_validator("BOOTSTRAP_SERVERS", mode="before")
    def parse_bootstrap_servers(cls, v: Any) -> List[str]:
        """Convert comma-separated string to list if needed."""
        # Check for environment variable with double underscore format
        env_value = get_environment_value("KAFKA", "BOOTSTRAP_SERVERS")
        if env_value is not None:
            if isinstance(env_value, str):
                if not env_value:
                    return []
                return [server.strip() for server in env_value.split(",")]
        
        # Process the value from regular Pydantic flow
        if isinstance(v, str):
            if not v:
                return []
            return [server.strip() for server in v.split(",")]
        return v


class DatabaseSettings(BaseSettings):
    """Database connection settings."""
    
    # PostgreSQL for trading strategies and orders
    POSTGRES_SERVER: str = Field(
        default="localhost", 
        description="PostgreSQL server hostname"
    )
    POSTGRES_PORT: str = Field(
        default="5432", 
        description="PostgreSQL server port"
    )
    POSTGRES_USER: str = Field(
        default="postgres", 
        description="PostgreSQL username"
    )
    POSTGRES_PASSWORD: str = Field(
        default="postgres", 
        description="PostgreSQL password"
    )
    POSTGRES_DB: str = Field(
        default="trading_strategies", 
        description="PostgreSQL database name"
    )
    POSTGRES_MIN_CONNECTIONS: int = Field(
        default=5, 
        description="Minimum PostgreSQL connections in pool"
    )
    POSTGRES_MAX_CONNECTIONS: int = Field(
        default=20, 
        description="Maximum PostgreSQL connections in pool"
    )
    POSTGRES_STATEMENT_TIMEOUT: int = Field(
        default=30000, 
        description="Statement timeout in ms"
    )
    
    # TimescaleDB for time-series market data
    TIMESCALE_SERVER: str = Field(
        default="localhost", 
        description="TimescaleDB server hostname"
    )
    TIMESCALE_PORT: str = Field(
        default="5432", 
        description="TimescaleDB server port"
    )
    TIMESCALE_USER: str = Field(
        default="postgres", 
        description="TimescaleDB username"
    )
    TIMESCALE_PASSWORD: str = Field(
        default="postgres", 
        description="TimescaleDB password"
    )
    TIMESCALE_DB: str = Field(
        default="market_data", 
        description="TimescaleDB database name"
    )
    TIMESCALE_MIN_CONNECTIONS: int = Field(
        default=5, 
        description="Minimum TimescaleDB connections in pool"
    )
    TIMESCALE_MAX_CONNECTIONS: int = Field(
        default=30, 
        description="Maximum TimescaleDB connections in pool"
    )
    TIMESCALE_STATEMENT_TIMEOUT: int = Field(
        default=20000, 
        description="Statement timeout in ms"
    )
    
    # MongoDB for trading signals
    MONGODB_SERVER: str = Field(
        default="localhost", 
        description="MongoDB server hostname"
    )
    MONGODB_PORT: int = Field(
        default=27017, 
        description="MongoDB server port"
    )
    MONGODB_USER: str = Field(
        default="mongodb", 
        description="MongoDB username"
    )
    MONGODB_PASSWORD: str = Field(
        default="mongodb", 
        description="MongoDB password"
    )
    MONGODB_DB: str = Field(
        default="trading_signals", 
        description="MongoDB database name"
    )
    MONGODB_AUTH_SOURCE: str = Field(
        default="admin", 
        description="MongoDB authentication source"
    )
    MONGODB_MAX_POOL_SIZE: int = Field(
        default=50, 
        description="MongoDB maximum connection pool size"
    )
    MONGODB_MIN_POOL_SIZE: int = Field(
        default=10, 
        description="MongoDB minimum connection pool size"
    )
    MONGODB_MAX_IDLE_TIME_MS: int = Field(
        default=10000, 
        description="MongoDB maximum connection idle time in ms"
    )
    MONGODB_CONNECT_TIMEOUT_MS: int = Field(
        default=20000, 
        description="MongoDB connection timeout in ms"
    )
    
    # Redis for caching
    REDIS_HOST: str = Field(
        default="localhost", 
        description="Redis server hostname"
    )
    REDIS_PORT: int = Field(
        default=6379, 
        description="Redis server port"
    )
    REDIS_DB: int = Field(
        default=0, 
        description="Redis database index"
    )
    REDIS_PASSWORD: Optional[str] = Field(
        default=None, 
        description="Redis password"
    )
    REDIS_SSL: bool = Field(
        default=False, 
        description="Use SSL for Redis connections"
    )
    REDIS_CONNECTION_POOL_SIZE: int = Field(
        default=100, 
        description="Redis connection pool size"
    )
    REDIS_SOCKET_TIMEOUT: float = Field(
        default=2.0, 
        description="Redis socket timeout in seconds"
    )
    REDIS_SOCKET_CONNECT_TIMEOUT: float = Field(
        default=1.0, 
        description="Redis socket connect timeout in seconds"
    )
    REDIS_KEY_PREFIX: str = Field(
        default="trading_app:", 
        description="Redis key prefix for namespacing"
    )
    
    # URI placeholders
    POSTGRES_URI: Optional[PostgresDsn] = None
    TIMESCALE_URI: Optional[PostgresDsn] = None
    MONGODB_URI: Optional[str] = None
    REDIS_URI: Optional[RedisDsn] = None

    model_config = SettingsConfigDict(
        env_prefix="DB_",
        case_sensitive=True,
        extra="allow"  # Allow extra attributes for env variables with double underscore format
    )

    def __init__(self, **data: Any):
        """Custom initialization to check for double underscore environment variables."""
        # Apply values from environment variables with double underscore format
        # This must happen before Pydantic's own initialization
        
        # Apply server and password settings that are used in tests
        server_env = get_environment_value("DB", "POSTGRES_SERVER")
        if server_env is not None:
            data["POSTGRES_SERVER"] = server_env
            
        password_env = get_environment_value("DB", "POSTGRES_PASSWORD")
        if password_env is not None:
            data["POSTGRES_PASSWORD"] = password_env
            
        # Check for explicit URI override with double underscore format
        uri_env = get_environment_value("DB", "POSTGRES_URI")
        if uri_env is not None:
            data["POSTGRES_URI"] = uri_env
            
        super().__init__(**data)

    # Connection strings constructors - properly indented
    @field_validator("POSTGRES_URI", mode="after")
    def assemble_postgres_uri(cls, v: Optional[str], info) -> Any:
        """Assembles PostgreSQL URI if not provided."""
        # Check again for explicit URI with double underscore format
        uri_env = get_environment_value("DB", "POSTGRES_URI")
        if uri_env is not None:
            return uri_env
            
        # Use value if already provided
        if isinstance(v, str) and v:
            return v
            
        # Otherwise build from components
        values = info.data
        
        # Build URL manually instead of using PostgresDsn.build()
        user = values.get("POSTGRES_USER", "")
        password = values.get("POSTGRES_PASSWORD", "")
        server = values.get("POSTGRES_SERVER", "localhost")
        port = values.get("POSTGRES_PORT", "5432")
        db = values.get("POSTGRES_DB", "")
        
        return f"postgresql://{user}:{password}@{server}:{port}/{db}"
    
    @field_validator("TIMESCALE_URI", mode="before")
    def assemble_timescale_uri(cls, v: Optional[str], info) -> Any:
        """Assembles TimescaleDB URI if not provided."""
        if isinstance(v, str) and v:
            return v
            
        values = info.data
        
        # Build URL manually instead of using PostgresDsn.build()
        user = values.get("TIMESCALE_USER", "")
        password = values.get("TIMESCALE_PASSWORD", "")
        server = values.get("TIMESCALE_SERVER", "localhost")
        port = values.get("TIMESCALE_PORT", "5432")
        db = values.get("TIMESCALE_DB", "")
        
        return f"postgresql://{user}:{password}@{server}:{port}/{db}"
    
    @field_validator("MONGODB_URI", mode="before")
    def assemble_mongodb_uri(cls, v: Optional[str], info) -> Any:
        """Assembles MongoDB URI if not provided."""
        if isinstance(v, str) and v:
            return v
            
        values = info.data
        user = values.get("MONGODB_USER", "")
        password = values.get("MONGODB_PASSWORD", "")
        server = values.get("MONGODB_SERVER", "localhost")
        port = values.get("MONGODB_PORT", 27017)
        db = values.get("MONGODB_DB", "")
        auth_source = values.get("MONGODB_AUTH_SOURCE", "admin")
        
        auth_str = ""
        if user and password:
            auth_str = f"{user}:{password}@"
            
        return f"mongodb://{auth_str}{server}:{port}/{db}?authSource={auth_source}"
    
    @field_validator("REDIS_URI", mode="before")
    def assemble_redis_uri(cls, v: Optional[str], info) -> Any:
        """Assembles Redis URI if not provided."""
        if isinstance(v, str) and v:
            return v
            
        values = info.data
        scheme = "rediss" if values.get("REDIS_SSL", False) else "redis"
        password = values.get("REDIS_PASSWORD")
        host = values.get("REDIS_HOST", "localhost")
        port = values.get("REDIS_PORT", 6379)
        db = values.get("REDIS_DB", 0)
        
        auth_str = ""
        if password:
            auth_str = f":{password}@"
            
        return f"{scheme}://{auth_str}{host}:{port}/{db}"


class SecuritySettings(BaseSettings):
    """Security-related configuration settings."""
    
    SECRET_KEY: str = Field(
        default_factory=lambda: secrets.token_urlsafe(32),
        description="Secret key for JWT tokens and other cryptographic operations"
    )
    ALGORITHM: str = Field(
        default="HS256", 
        description="Algorithm used for JWT token encoding/decoding"
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30, 
        description="JWT access token expiry time in minutes"
    )
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(
        default=7, 
        description="JWT refresh token expiry time in days"
    )
    SECURE_COOKIES: bool = Field(
        default=False, 
        description="Set secure flag on cookies (HTTPS only)"
    )
    CORS_ORIGINS: List[str] = Field(
        default=["*"], 
        description="List of origins for CORS"
    )
    
    model_config = SettingsConfigDict(
        env_prefix="SECURITY_",
        case_sensitive=True,
        extra="allow"  # Allow extra attributes for env variables with double underscore format
    )
    
    def __init__(self, **data: Any):
        """Custom initialization to check for double underscore environment variables."""
        # Check for ACCESS_TOKEN_EXPIRE_MINUTES with double underscore format
        expire_mins = get_environment_value("SECURITY", "ACCESS_TOKEN_EXPIRE_MINUTES")
        if expire_mins is not None:
            data["ACCESS_TOKEN_EXPIRE_MINUTES"] = int(expire_mins)
            
        super().__init__(**data)


class PerformanceSettings(BaseSettings):
    """Performance tuning configuration settings."""
    
    WORKERS: int = Field(
        default=4, 
        description="Number of worker processes"
    )
    WORKER_CONNECTIONS: int = Field(
        default=1000, 
        description="Maximum number of connections per worker"
    )
    KEEPALIVE: int = Field(
        default=5, 
        description="Keep-alive connection timeout in seconds"
    )
    TIMEOUT: int = Field(
        default=120, 
        description="Worker silent timeout in seconds"
    )
    GRACEFUL_TIMEOUT: int = Field(
        default=30, 
        description="Graceful worker shutdown timeout in seconds"
    )
    MAX_REQUESTS: int = Field(
        default=10000, 
        description="Maximum requests per worker before restart"
    )
    MAX_REQUESTS_JITTER: int = Field(
        default=1000, 
        description="Random jitter added to max_requests to prevent all workers restarting at once"
    )
    
    # Cache settings
    CACHE_TTL_DEFAULT: int = Field(
        default=300, 
        description="Default cache TTL in seconds for generic items"
    )
    CACHE_TTL_INDICATORS: int = Field(
        default=60, 
        description="Cache TTL in seconds for technical indicators"
    )
    CACHE_TTL_MARKET_STATE: int = Field(
        default=10, 
        description="Cache TTL in seconds for market state"
    )
    
    model_config = SettingsConfigDict(
        env_prefix="PERF_",
        case_sensitive=True,
        extra="allow"  # Allow extra attributes for env variables with double underscore format
    )
    
    def __init__(self, **data: Any):
        """Custom initialization to check for double underscore environment variables."""
        # Check for CACHE_TTL_DEFAULT with double underscore format
        cache_ttl = get_environment_value("PERF", "CACHE_TTL_DEFAULT")
        if cache_ttl is not None:
            data["CACHE_TTL_DEFAULT"] = int(cache_ttl)
            
        super().__init__(**data)


class Settings(BaseSettings):
    """Main application settings that combine all setting categories."""
    
    # Application metadata
    APP_NAME: str = Field(
        default="Trading Strategies Application", 
        description="Application name"
    )
    APP_VERSION: str = Field(
        default="0.1.0", 
        description="Application version"
    )
    DEBUG: bool = Field(
        default=False, 
        description="Enable debug mode"
    )
    ENV: str = Field(
        default="development", 
        description="Environment (development, staging, production)"
    )
    API_PREFIX: str = Field(
        default="/api/v1", 
        description="API route prefix"
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=True,
        extra="allow"  # Allow extra attributes for env variables with nested format
    )

    # In Pydantic v2, we need to define these fields differently
    # They become model fields, not just instance attributes
    kafka: KafkaSettings = Field(default_factory=KafkaSettings)
    db: DatabaseSettings = Field(default_factory=DatabaseSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    performance: PerformanceSettings = Field(default_factory=PerformanceSettings)
    
    @field_validator("ENV")
    def validate_env(cls, v: str) -> str:
        """Validates environment value."""
        allowed = {"development", "staging", "production"}
        if v.lower() not in allowed:
            raise ValueError(f"Environment must be one of {allowed}")
        return v.lower()


# Singleton pattern - ensures only one Settings instance is created
_settings_instance = None

def get_settings() -> Settings:
    """
    Returns singleton instance of application settings.
    Uses module-level variable for singleton pattern to avoid issues with circular imports.
    """
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance

# Export the singleton instance for use throughout the application
settings = get_settings()