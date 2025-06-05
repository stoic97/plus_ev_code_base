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
from app.models import *  # Import all models to ensure they're registered

# Read in the configuration
config = context.config
settings = get_settings()

# Set up target metadata
target_metadata = Base.metadata


def get_postgres_url():
    """Get PostgreSQL connection URL with proper priority."""
    # Priority: Direct env var > Settings > Config file
    
    # 1. Check direct environment variable
    direct_url = os.getenv('DB__POSTGRES_URI')
    if direct_url:
        print(f"Using direct PostgreSQL URI from environment")
        return direct_url
    
    # 2. Build from settings
    if settings and hasattr(settings, 'db'):
        url = str(settings.db.POSTGRES_URI)
        if url and url != 'None':
            print(f"Using PostgreSQL URI from settings")
            return url
    
    # 3. Fallback to config file
    config_url = config.get_main_option("sqlalchemy.url")
    if config_url:
        print(f"Using PostgreSQL URI from alembic.ini")
        return config_url
    
    raise ValueError("No PostgreSQL connection URL found in environment, settings, or config")


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


def is_schema_creating_migration():
    """Check if this is a schema-creating migration."""
    try:
        cmd_opts = context.get_x_argument(as_dictionary=True)
        revision = cmd_opts.get('revision', '')
        
        # Known schema-creating revisions
        schema_revisions = ['91e93a42b21c', 'create_auth_schema']
        return any(rev in str(revision) for rev in schema_revisions)
    except:
        return False


def run_migrations_offline():
    """Run migrations in 'offline' mode."""
    url = get_postgres_url()
    
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        include_schemas=True,
        version_table='alembic_version_postgres',
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    """Run migrations in 'online' mode."""
    url = get_postgres_url()
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
    
    # Pre-create schemas for schema-creating migrations
    if is_schema_creating_migration():
        print("Detected PostgreSQL schema-creating migration")
        try:
            with connectable.execution_options(isolation_level="AUTOCOMMIT").connect() as autocommit_conn:
                print("Pre-creating extension and schema with autocommit")
                autocommit_conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))
                autocommit_conn.execute(text('CREATE SCHEMA IF NOT EXISTS app_auth'))
                print("Successfully pre-created extension and schema")
        except Exception as e:
            print(f"Error pre-creating schema: {e}")
    
    # Run migration
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            version_table='alembic_version_postgres',
            transaction_per_migration=False,
        )
        
        with context.begin_transaction():
            context.run_migrations()


def run_postgres_migrations():
    """Main entry point for PostgreSQL migrations."""
    print("Running PostgreSQL migrations...")
    
    if context.is_offline_mode():
        run_migrations_offline()
    else:
        run_migrations_online()