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

# Read in the configuration
config = context.config
settings = get_settings()

# Interpret the config file for logging
if config.config_file_name:
    fileConfig(config.config_file_name)

# Set up target metadata
target_metadata = Base.metadata


def get_current_database():
    """Determine which database to target based on command line arguments."""
    try:
        cmd_opts = context.get_x_argument(as_dictionary=True)
        database = cmd_opts.get('database', 'postgres')
        
        # Validate database choice
        valid_databases = ['postgres', 'timescale']
        if database not in valid_databases:
            raise ValueError(f"Invalid database '{database}'. Must be one of: {valid_databases}")
        
        return database
    except (AttributeError, TypeError):
        return 'postgres'


def delegate_to_database_env():
    """
    Delegate to the appropriate database-specific env.py file.
    This is the main router that directs to postgres/ or timescale/ directories.
    """
    database = get_current_database()
    
    # Import and call the appropriate database-specific env
    if database == 'postgres':
        from app.db.migrations.postgres.env import run_postgres_migrations
        run_postgres_migrations()
    elif database == 'timescale':
        from app.db.migrations.timescale.env import run_timescale_migrations
        run_timescale_migrations()
    else:
        raise ValueError(f"No migration handler for database: {database}")


# Route to appropriate database
delegate_to_database_env()