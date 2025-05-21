"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}
Database: ${database | default("postgres")}

"""
from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

# Detect if we're running against TimescaleDB and need hypertable functionality
import re
from sqlalchemy import inspect, text
from sqlalchemy.engine import reflection

# revision identifiers, used by Alembic.
revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}
database = '${database | default("postgres")}'


def create_hypertable(table_name, time_column_name, chunk_time_interval="1 day", 
                     if_not_exists=True, migrate_data=False):
    """Helper function to create a TimescaleDB hypertable."""
    if_not_exists_clause = "IF NOT EXISTS" if if_not_exists else ""
    migrate_data_clause = "WITH (migrate_data=True)" if migrate_data else ""
    
    op.execute(sa.text(f"""
        SELECT create_hypertable(
            '{table_name}', 
            '{time_column_name}',
            {if_not_exists_clause}
            chunk_time_interval => INTERVAL '{chunk_time_interval}'
            {migrate_data_clause}
        )
    """))


def drop_hypertable(table_name, if_exists=True, cascade=False):
    """Helper function to drop a TimescaleDB hypertable."""
    # We don't actually need to do anything special for dropping
    # The standard DROP TABLE command works for hypertables
    pass


def add_cagg(name, hypertable, view_query, time_column, bucket_interval, 
             with_data=True, if_not_exists=True):
    """Helper function to create a TimescaleDB continuous aggregate."""
    if_not_exists_clause = "IF NOT EXISTS" if if_not_exists else ""
    with_data_clause = "WITH DATA" if with_data else "WITH NO DATA"
    
    op.execute(sa.text(f"""
        CREATE MATERIALIZED VIEW {if_not_exists_clause} {name}
        WITH (timescaledb.continuous) AS
        {view_query}
        WITH {with_data_clause}
    """))


def drop_cagg(name, if_exists=True, cascade=False):
    """Helper function to drop a TimescaleDB continuous aggregate."""
    if_exists_clause = "IF EXISTS" if if_exists else ""
    cascade_clause = "CASCADE" if cascade else ""
    
    op.execute(sa.text(f"""
        DROP MATERIALIZED VIEW {if_exists_clause} {name} {cascade_clause}
    """))


def is_timescaledb_available():
    """Check if TimescaleDB extension is available."""
    try:
        connection = op.get_bind()
        result = connection.execute(text("SELECT extname FROM pg_extension WHERE extname = 'timescaledb'"))
        return result.fetchone() is not None
    except Exception:
        return False


def upgrade():
    ${upgrades if upgrades else "pass"}


def downgrade():
    ${downgrades if downgrades else "pass"}