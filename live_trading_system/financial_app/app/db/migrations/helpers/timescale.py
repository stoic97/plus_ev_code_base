"""
TimescaleDB-specific migration helpers.

This module provides specialized functions for working with TimescaleDB
features like hypertables, continuous aggregates, and compression.

MVP version with Supabase compatibility - gracefully handles environments 
without TimescaleDB support.
"""

import logging
from typing import Optional, Union, List
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError, OperationalError
from alembic import op

# Set up logging
logger = logging.getLogger(__name__)


def is_timescaledb_available(op):
    """
    Check if TimescaleDB extension is available in the database.
    Returns False for Supabase or when TimescaleDB is not installed.
    """
    try:
        connection = op.get_bind()
        
        # Check if we're on Supabase (no TimescaleDB support)
        if 'supabase.co' in str(connection.engine.url):
            logger.info("Running on Supabase - TimescaleDB features disabled")
            return False
        
        # Check if TimescaleDB extension exists
        result = connection.execute(text(
            "SELECT extname FROM pg_extension WHERE extname = 'timescaledb'"
        ))
        available = result.fetchone() is not None
        
        if available:
            logger.info("TimescaleDB extension is available")
        else:
            logger.warning("TimescaleDB extension not found - features disabled")
            
        return available
    except Exception as e:
        logger.warning(f"Error checking TimescaleDB availability: {e}")
        return False


def create_hypertable(op, table_name: str, time_column_name: str, schema: Optional[str] = None, 
                     chunk_time_interval: str = "1 day", if_not_exists: bool = True,
                     migrate_data: bool = False, partitioning_column: Optional[str] = None,
                     number_partitions: Optional[int] = None):
    """
    Create a TimescaleDB hypertable from an existing table.
    Skips creation if TimescaleDB is not available.
    
    Args:
        op: Alembic operations object
        table_name: Name of the table to convert to a hypertable
        time_column_name: Name of the timestamp column to use for partitioning
        schema: Optional schema name
        chunk_time_interval: Time interval for chunks (e.g., '1 day', '1 hour')
        if_not_exists: Only create if it doesn't already exist
        migrate_data: Whether to migrate existing data into chunks
        partitioning_column: Optional column for space partitioning
        number_partitions: Optional number of space partitions to create
    """
    if not is_timescaledb_available(op):
        logger.info(f"Skipping hypertable creation for {table_name} - TimescaleDB not available")
        return
    
    try:
        schema_clause = f"'{schema}', " if schema else ""
        if_not_exists_clause = "true" if if_not_exists else "false"
        migrate_data_clause = "migrate_data => TRUE" if migrate_data else ""
        
        # Additional options for space partitioning
        partitioning_options = ""
        if partitioning_column and number_partitions:
            partitioning_options = f"""
                partitioning_column => '{partitioning_column}',
                number_partitions => {number_partitions},
            """
        
        # Build the final SQL command
        query = f"""
            SELECT create_hypertable(
                '{table_name}', 
                '{time_column_name}',
                {schema_clause}
                chunk_time_interval => INTERVAL '{chunk_time_interval}',
                {partitioning_options}
                if_not_exists => {if_not_exists_clause}
                {', ' + migrate_data_clause if migrate_data_clause else ''}
            )
        """
        
        op.execute(text(query))
        logger.info(f"Created hypertable for {table_name}")
        
    except (ProgrammingError, OperationalError) as e:
        logger.error(f"Failed to create hypertable for {table_name}: {e}")
        # Don't re-raise - allow migration to continue


def add_hypertable_compression(op, table_name: str, schema: Optional[str] = None, 
                              compress_after: str = "7 days",
                              compress_segmentby: Optional[Union[List[str], str]] = None, 
                              compress_orderby: Optional[Union[List[str], str]] = None):
    """
    Add native compression to a TimescaleDB hypertable.
    Skips if TimescaleDB is not available.
    
    Args:
        op: Alembic operations object
        table_name: Name of the hypertable
        schema: Optional schema name
        compress_after: When to compress chunks (e.g., '7 days')
        compress_segmentby: Column(s) to use for segmenting data
        compress_orderby: Column(s) to use for ordering data
    """
    if not is_timescaledb_available(op):
        logger.info(f"Skipping compression for {table_name} - TimescaleDB not available")
        return
    
    try:
        full_table_name = f"{schema}.{table_name}" if schema else table_name
        
        # Build compression clauses
        compression_clauses = ["timescaledb.compress = TRUE"]
        
        # Build segment by clause
        if compress_segmentby:
            if isinstance(compress_segmentby, list):
                segmentby_cols = ", ".join(compress_segmentby)
            else:
                segmentby_cols = compress_segmentby
            compression_clauses.append(f"timescaledb.compress_segmentby = '{segmentby_cols}'")
        
        # Build order by clause
        if compress_orderby:
            if isinstance(compress_orderby, list):
                orderby_cols = ", ".join(compress_orderby)
            else:
                orderby_cols = compress_orderby
            compression_clauses.append(f"timescaledb.compress_orderby = '{orderby_cols}'")
        
        compress_params = ", ".join(compression_clauses)
        
        # Set compression settings
        op.execute(text(f"""
            ALTER TABLE {full_table_name} SET (
                {compress_params}
            )
        """))
        
        # Add compression policy
        op.execute(text(f"""
            SELECT add_compression_policy('{full_table_name}', INTERVAL '{compress_after}')
        """))
        
        logger.info(f"Added compression policy for {table_name}")
        
    except (ProgrammingError, OperationalError) as e:
        logger.error(f"Failed to add compression for {table_name}: {e}")
        # Don't re-raise - allow migration to continue


def create_retention_policy(op, table_name: str, retention_period: str = "90 days", 
                          schema: Optional[str] = None):
    """
    Create a data retention policy for a hypertable.
    Skips if TimescaleDB is not available.
    
    Args:
        op: Alembic operations object
        table_name: Name of the hypertable
        retention_period: How long to keep data (e.g., '90 days')
        schema: Optional schema name
    """
    if not is_timescaledb_available(op):
        logger.info(f"Skipping retention policy for {table_name} - TimescaleDB not available")
        return
    
    try:
        full_table_name = f"{schema}.{table_name}" if schema else table_name
        
        op.execute(text(f"""
            SELECT add_retention_policy(
                '{full_table_name}', 
                INTERVAL '{retention_period}'
            )
        """))
        
        logger.info(f"Added retention policy for {table_name}")
        
    except (ProgrammingError, OperationalError) as e:
        logger.error(f"Failed to add retention policy for {table_name}: {e}")
        # Don't re-raise - allow migration to continue


def remove_retention_policy(op, table_name: str, schema: Optional[str] = None):
    """
    Remove a data retention policy from a hypertable.
    Skips if TimescaleDB is not available.
    
    Args:
        op: Alembic operations object
        table_name: Name of the hypertable
        schema: Optional schema name
    """
    if not is_timescaledb_available(op):
        logger.info(f"Skipping retention policy removal for {table_name} - TimescaleDB not available")
        return
    
    try:
        full_table_name = f"{schema}.{table_name}" if schema else table_name
        
        op.execute(text(f"""
            SELECT remove_retention_policy('{full_table_name}')
        """))
        
        logger.info(f"Removed retention policy for {table_name}")
        
    except (ProgrammingError, OperationalError) as e:
        logger.error(f"Failed to remove retention policy for {table_name}: {e}")
        # Don't re-raise - allow migration to continue


def create_time_bucket_index(op, table_name: str, time_column: str, 
                           bucket_interval: str = "1 day", 
                           include_columns: Optional[Union[List[str], str]] = None, 
                           schema: Optional[str] = None):
    """
    Create an optimized index for time_bucket queries.
    Falls back to regular timestamp index if TimescaleDB is not available.
    
    Args:
        op: Alembic operations object
        table_name: Name of the table
        time_column: Name of the timestamp column
        bucket_interval: Bucket interval for the index
        include_columns: Additional columns to include in the index
        schema: Optional schema name
    """
    full_table_name = f"{schema}.{table_name}" if schema else table_name
    index_name = f"idx_{table_name}_{time_column}_bucket"
    
    try:
        if is_timescaledb_available(op):
            # Create TimescaleDB time_bucket index
            include_clause = ""
            if include_columns:
                if isinstance(include_columns, list):
                    include_list = ", ".join(include_columns)
                    include_clause = f"INCLUDE ({include_list})"
                else:
                    include_clause = f"INCLUDE ({include_columns})"
            
            op.execute(text(f"""
                CREATE INDEX {index_name} ON {full_table_name} 
                USING BTREE (time_bucket(INTERVAL '{bucket_interval}', {time_column}))
                {include_clause}
            """))
            
            logger.info(f"Created time_bucket index for {table_name}")
        else:
            # Fall back to regular timestamp index
            op.create_index(
                index_name, 
                table_name, 
                [time_column],
                schema=schema
            )
            logger.info(f"Created regular timestamp index for {table_name}")
            
    except (ProgrammingError, OperationalError) as e:
        logger.error(f"Failed to create index for {table_name}: {e}")
        # Don't re-raise - allow migration to continue


# Remove complex features for MVP
# Continuous aggregates are not included in the MVP version
# We only keep the essential hypertable and compression functions