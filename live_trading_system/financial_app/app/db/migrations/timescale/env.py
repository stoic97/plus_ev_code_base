import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool, text
from sqlalchemy.engine.url import make_url

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

# Import models and Base
from app.core.database import Base
from app.core.config import get_settings
from app.models import *  # Import all models

# Read in the configuration
config = context.config
settings = get_settings()

# Set up target metadata
target_metadata = Base.metadata


def get_timescale_url():
    """Get TimescaleDB connection URL with proper priority."""
    # Priority: Direct env var > Settings > Config file
    
    # 1. Check direct environment variable
    direct_url = os.getenv('DB__TIMESCALE_URI')
    if direct_url:
        print(f"Using direct TimescaleDB URI from environment")
        return direct_url
    
    # 2. Build from settings
    if settings and hasattr(settings, 'db'):
        url = str(settings.db.TIMESCALE_URI)
        if url and url != 'None':
            print(f"Using TimescaleDB URI from settings")
            return url
    
    # 3. Fallback to config file
    config_url = config.get_main_option("timescaledb.url")
    if config_url:
        print(f"Using TimescaleDB URI from alembic.ini")
        return config_url
    
    raise ValueError("No TimescaleDB connection URL found in environment, settings, or config")


def get_ssl_connect_args(url: str) -> tuple:
    """Parse SSL arguments and return (connect_args, clean_url)."""
    parsed_url = make_url(url)
    connect_args = {}
    
    # Auto-detect SSL for Supabase
    if 'supabase.co' in (parsed_url.host or ''):
        connect_args['sslmode'] = 'require'
    
    # Handle SSL parameters in URL
    if parsed_url.query:
        query_params = dict(parsed_url.query)
        
        if 'sslmode' in query_params:
            connect_args['sslmode'] = query_params.pop('sslmode')
        
        # Create clean URL without SSL params
        parsed_url = parsed_url._replace(query=query_params)
    
    return connect_args, str(parsed_url)


def is_timescaledb_available(connection):
    """Check if TimescaleDB extension is available."""
    try:
        # Check if we're on Supabase (no TimescaleDB)
        if 'supabase.co' in str(connection.engine.url):
            print("Running on Supabase - TimescaleDB features will be skipped")
            return False
        
        result = connection.execute(text("SELECT extname FROM pg_extension WHERE extname = 'timescaledb'"))
        available = result.fetchone() is not None
        
        if available:
            print("TimescaleDB extension detected")
        else:
            print("TimescaleDB extension not found - features will be skipped")
            
        return available
    except Exception as e:
        print(f"Error checking TimescaleDB availability: {e}")
        return False


def run_migrations_offline():
    """Run migrations in 'offline' mode."""
    url = get_timescale_url()
    
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        include_schemas=True,
        version_table='alembic_version_timescale',
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    """Run migrations in 'online' mode."""
    url = get_timescale_url()
    connect_args, clean_url = get_ssl_connect_args(url)
    
    # Build configuration
    configuration = config.get_section(config.config_ini_section) or {}
    configuration['sqlalchemy.url'] = clean_url
    
    # Create engine
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        connect_args=connect_args,
    )
    
    # Check TimescaleDB availability before running migrations
    print("Checking TimescaleDB availability...")
    with connectable.connect() as test_conn:
        timescale_available = is_timescaledb_available(test_conn)
        
        # Set context attribute for migrations to check
        context.timescaledb_available = timescale_available
    
    # Run migration
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            version_table='alembic_version_timescale',
            transaction_per_migration=False,
        )
        
        with context.begin_transaction():
            context.run_migrations()


def run_timescale_migrations():
    """Main entry point for TimescaleDB migrations."""
    print("Running TimescaleDB migrations...")
    
    if context.is_offline_mode():
        run_migrations_offline()
    else:
        run_migrations_online()