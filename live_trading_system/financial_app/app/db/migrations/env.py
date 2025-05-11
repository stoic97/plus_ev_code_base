import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Import models and Base
from app.core.database import Base
from app.core.config import get_settings
from app.models import *  # noqa: Import all models to ensure they're registered with Base

# Read in the configuration
config = context.config
settings = get_settings()

# Inject environment variables into config
section = config.config_ini_section
config.set_section_option(section, "POSTGRES_USER", settings.db.POSTGRES_USER)
config.set_section_option(section, "POSTGRES_PASSWORD", settings.db.POSTGRES_PASSWORD)
config.set_section_option(section, "POSTGRES_SERVER", settings.db.POSTGRES_SERVER)
config.set_section_option(section, "POSTGRES_PORT", settings.db.POSTGRES_PORT)
config.set_section_option(section, "POSTGRES_DB", settings.db.POSTGRES_DB)

# TimescaleDB settings
config.set_section_option(section, "TIMESCALE_USER", settings.db.TIMESCALE_USER)
config.set_section_option(section, "TIMESCALE_PASSWORD", settings.db.TIMESCALE_PASSWORD)
config.set_section_option(section, "TIMESCALE_SERVER", settings.db.TIMESCALE_SERVER)
config.set_section_option(section, "TIMESCALE_PORT", settings.db.TIMESCALE_PORT)
config.set_section_option(section, "TIMESCALE_DB", settings.db.TIMESCALE_DB)

# Interpret the config file for logging
fileConfig(config.config_file_name)

# Set up target metadata
target_metadata = Base.metadata

# Database selection based on revision tag
database_selection = {
    'postgres': config.get_main_option("sqlalchemy.url"),
    'timescale': config.get_main_option("timescaledb.url")
}

# Custom function to determine which database to target based on revision tags
def get_current_database():
    # Get the revision or autogenerate args
    cmd_opts = context.get_x_argument(as_dictionary=True)
    
    # Get the database argument or default to postgres
    return cmd_opts.get('database', 'postgres')


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


def run_migrations_online():
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.
    """
    db_name = get_current_database()
    url = database_selection[db_name]
    
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        url=url,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, 
            target_metadata=target_metadata,
            compare_type=True,
            include_schemas=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()