import os
print(f"Using direct connection: {os.getenv('DB__POSTGRES_URI')}")
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool, text
from sqlalchemy.engine.url import make_url

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Import models and Base
from app.core.database import Base
from app.core.config import get_settings
from app.models import *  # noqa: Import all models to ensure they're registered with Base

# Read in the configuration
config = context.config
settings = get_settings()

# Inject environment variables into config with DB__ prefix to match alembic.ini
section = config.config_ini_section

# PostgreSQL settings with DB__ prefix
config.set_section_option(section, "DB__POSTGRES_USER", settings.db.POSTGRES_USER)
config.set_section_option(section, "DB__POSTGRES_PASSWORD", settings.db.POSTGRES_PASSWORD)
config.set_section_option(section, "DB__POSTGRES_SERVER", settings.db.POSTGRES_SERVER)
config.set_section_option(section, "DB__POSTGRES_PORT", str(settings.db.POSTGRES_PORT))
config.set_section_option(section, "DB__POSTGRES_DB", settings.db.POSTGRES_DB)
config.set_section_option(section, "DB__POSTGRES_SSL_MODE", getattr(settings.db, 'SSL_MODE', 'require'))

# TimescaleDB settings with DB__ prefix
config.set_section_option(section, "DB__TIMESCALE_USER", settings.db.TIMESCALE_USER)
config.set_section_option(section, "DB__TIMESCALE_PASSWORD", settings.db.TIMESCALE_PASSWORD)
config.set_section_option(section, "DB__TIMESCALE_SERVER", settings.db.TIMESCALE_SERVER)
config.set_section_option(section, "DB__TIMESCALE_PORT", str(settings.db.TIMESCALE_PORT))
config.set_section_option(section, "DB__TIMESCALE_DB", settings.db.TIMESCALE_DB)
config.set_section_option(section, "DB__TIMESCALE_SSL_MODE", getattr(settings.db, 'SSL_MODE', 'require'))

# Interpret the config file for logging
fileConfig(config.config_file_name)

# Set up target metadata
target_metadata = Base.metadata

# Database selection based on revision tag
database_selection = {
    'postgres': os.getenv('DB__POSTGRES_URI') or config.get_main_option("sqlalchemy.url"),
    'timescale': os.getenv('DB__TIMESCALE_URI') or config.get_main_option("timescaledb.url")
}

# Custom function to determine which database to target based on revision tags
def get_current_database():
    # Get the revision or autogenerate args
    cmd_opts = context.get_x_argument(as_dictionary=True)
    
    # Get the database argument or default to postgres
    return cmd_opts.get('database', 'postgres')


# Fixed get_ssl_connect_args function in env.py

def get_ssl_connect_args(url: str) -> tuple:
    """
    Parse SSL arguments from connection URL and prepare connect_args.
    
    Args:
        url: Database connection URL with potential SSL parameters
        
    Returns:
        Tuple of (connect_args, clean_url)
    """
    from sqlalchemy.engine.url import make_url
    
    parsed_url = make_url(url)
    connect_args = {}
    
    # Check if SSL should be enabled based on host
    if 'supabase.co' in (parsed_url.host or ''):
        connect_args['sslmode'] = 'require'
    
    # Handle query parameters for SSL
    if parsed_url.query:
        # Create a mutable copy of the query dictionary
        query_params = dict(parsed_url.query)
        
        # Extract SSL-related parameters
        if 'sslmode' in query_params:
            connect_args['sslmode'] = query_params.pop('sslmode')
        
        # Remove SSL parameters from URL to avoid passing them twice
        # Create a new URL with the modified query parameters
        parsed_url = parsed_url._replace(query=query_params)
    
    # Get the clean URL string
    clean_url = str(parsed_url)
    
    return connect_args, clean_url


def run_migrations_offline():
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.
    """
    db_name = get_current_database()
    url = database_selection[db_name]
    
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        include_schemas=True,
    )

    with context.begin_transaction():
        context.run_migrations(database=db_name)


# Fixed run_migrations_online function in env.py

def run_migrations_online():
    """Run migrations in 'online' mode."""
    
    # Get the current database from command line arguments
    db_name = get_current_database()
    
    # Force direct connection URL if available in environment
    if db_name == 'postgres' and os.getenv('DB__POSTGRES_URI'):
        url = os.getenv('DB__POSTGRES_URI')
        print(f"Using direct environment URI for postgres")
    elif db_name == 'timescale' and os.getenv('DB__TIMESCALE_URI'):
        url = os.getenv('DB__TIMESCALE_URI')
        print(f"Using direct environment URI for timescale")
    else:
        url = database_selection[db_name]
        print(f"Using config connection URL for {db_name}")
    
    # Prepare SSL connection arguments
    connect_args, clean_url = get_ssl_connect_args(url)
    
    # Get configuration section
    config_section = config.config_ini_section
    
    # Create a configuration dictionary
    configuration = config.get_section(config_section)
    configuration['sqlalchemy.url'] = clean_url
    
    # Create engine with configuration - skip pooling configuration completely
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        connect_args=connect_args,
    )
    
    # Determine if this is a schema-creating migration
    is_schema_creating = False
    
    # Get revision information (if available)
    revision = context.get_x_argument(as_dictionary=True).get('revision', 'head')
    
    if revision and revision != 'head':
        # Try to get script from revision ID
        try:
            from alembic.script import ScriptDirectory
            script_dir = ScriptDirectory.from_config(config)
            
            script = script_dir.get_revision(revision)
            if script and hasattr(script, 'module') and script.module.__doc__:
                # Check for markers in docstring
                docstring = script.module.__doc__.lower()
                if 'create schema' in docstring or 'auth schema' in docstring:
                    is_schema_creating = True
                    print(f"Detected schema-creating migration: {revision}")
                    
                    # Pre-create schemas with autocommit before running migration
                    with connectable.execution_options(isolation_level="AUTOCOMMIT").connect() as autocommit_conn:
                        print("Pre-creating schema using autocommit")
                        autocommit_conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))
                        autocommit_conn.execute(text('CREATE SCHEMA IF NOT EXISTS app_auth'))
                        print("Successfully pre-created extension and schema")
        except Exception as e:
            print(f"Error checking migration type: {e}")
    
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            version_table=f'alembic_version_{db_name}',
            transaction_per_migration=False,  # Always disable transactions for all migrations
        )
        
        with context.begin_transaction():
            context.run_migrations()