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


def create_continuous_aggregate(op, view_name: str, hypertable_name: str, query: str,
                              schema: Optional[str] = None, refresh_interval: str = "1 hour",
                              refresh_lag: str = "3 hours", materialized_only: bool = True,
                              with_data: bool = True, if_not_exists: bool = True):
    """
    Create a TimescaleDB continuous aggregate (real-time materialized view).
    Skips creation if TimescaleDB is not available.
    
    Args:
        op: Alembic operations object
        view_name: Name for the continuous aggregate view
        hypertable_name: Name of the source hypertable
        query: SELECT query defining the aggregate (should include time_bucket)
        schema: Optional schema name for the view
        refresh_interval: How often to refresh the view (e.g., '1 hour')
        refresh_lag: How far behind real-time to refresh (e.g., '3 hours')
        materialized_only: Whether to use materialized data only
        with_data: Whether to materialize existing data
        if_not_exists: Only create if it doesn't already exist
    
    Example query:
        "SELECT time_bucket('1 hour', time) as bucket, symbol, avg(price) as avg_price 
         FROM market_data GROUP BY bucket, symbol"
    """
    if not is_timescaledb_available(op):
        logger.info(f"Skipping continuous aggregate creation for {view_name} - TimescaleDB not available")
        return
    
    try:
        # Build full view name with schema
        full_view_name = f"{schema}.{view_name}" if schema else view_name
        
        # Build clauses
        if_not_exists_clause = "IF NOT EXISTS" if if_not_exists else ""
        materialized_clause = "timescaledb.materialized_only=true" if materialized_only else "timescaledb.materialized_only=false"
        with_data_clause = "WITH DATA" if with_data else "WITH NO DATA"
        
        # Create the continuous aggregate view
        create_sql = f"""
            CREATE MATERIALIZED VIEW {if_not_exists_clause} {full_view_name}
            WITH (timescaledb.continuous, {materialized_clause}) AS
            {query}
            {with_data_clause}
        """
        
        op.execute(text(create_sql))
        logger.info(f"Created continuous aggregate view: {full_view_name}")
        
        # Add refresh policy
        policy_sql = f"""
            SELECT add_continuous_aggregate_policy('{full_view_name}',
                start_offset => INTERVAL '{refresh_lag}',
                end_offset => INTERVAL '1 minute',
                schedule_interval => INTERVAL '{refresh_interval}')
        """
        
        op.execute(text(policy_sql))
        logger.info(f"Added refresh policy for continuous aggregate: {full_view_name}")
        
    except (ProgrammingError, OperationalError) as e:
        logger.error(f"Failed to create continuous aggregate {view_name}: {e}")
        # Don't re-raise - allow migration to continue


def drop_continuous_aggregate(op, view_name: str, schema: Optional[str] = None, 
                            if_exists: bool = True, cascade: bool = False):
    """
    Drop a TimescaleDB continuous aggregate.
    Skips if TimescaleDB is not available.
    
    Args:
        op: Alembic operations object
        view_name: Name of the continuous aggregate view to drop
        schema: Optional schema name
        if_exists: Only drop if it exists
        cascade: Whether to cascade the drop
    """
    if not is_timescaledb_available(op):
        logger.info(f"Skipping continuous aggregate drop for {view_name} - TimescaleDB not available")
        return
    
    try:
        full_view_name = f"{schema}.{view_name}" if schema else view_name
        if_exists_clause = "IF EXISTS" if if_exists else ""
        cascade_clause = "CASCADE" if cascade else ""
        
        # First remove the refresh policy
        try:
            op.execute(text(f"""
                SELECT remove_continuous_aggregate_policy('{full_view_name}')
            """))
            logger.info(f"Removed refresh policy for {view_name}")
        except Exception as e:
            logger.warning(f"Could not remove refresh policy for {view_name}: {e}")
        
        # Drop the continuous aggregate
        op.execute(text(f"""
            DROP MATERIALIZED VIEW {if_exists_clause} {full_view_name} {cascade_clause}
        """))
        
        logger.info(f"Dropped continuous aggregate: {full_view_name}")
        
    except (ProgrammingError, OperationalError) as e:
        logger.error(f"Failed to drop continuous aggregate {view_name}: {e}")
        # Don't re-raise - allow migration to continue


def refresh_continuous_aggregate(op, view_name: str, schema: Optional[str] = None,
                               start_time: Optional[str] = None, end_time: Optional[str] = None):
    """
    Manually refresh a continuous aggregate for a specific time range.
    Skips if TimescaleDB is not available.
    
    Args:
        op: Alembic operations object
        view_name: Name of the continuous aggregate view
        schema: Optional schema name
        start_time: Start time for refresh (e.g., '2024-01-01 00:00:00')
        end_time: End time for refresh (e.g., '2024-01-02 00:00:00')
    """
    if not is_timescaledb_available(op):
        logger.info(f"Skipping continuous aggregate refresh for {view_name} - TimescaleDB not available")
        return
    
    try:
        full_view_name = f"{schema}.{view_name}" if schema else view_name
        
        if start_time and end_time:
            # Refresh specific time range
            op.execute(text(f"""
                CALL refresh_continuous_aggregate('{full_view_name}', 
                    TIMESTAMP '{start_time}', TIMESTAMP '{end_time}')
            """))
            logger.info(f"Refreshed continuous aggregate {view_name} from {start_time} to {end_time}")
        else:
            # Refresh entire aggregate
            op.execute(text(f"""
                CALL refresh_continuous_aggregate('{full_view_name}', NULL, NULL)
            """))
            logger.info(f"Refreshed entire continuous aggregate {view_name}")
        
    except (ProgrammingError, OperationalError) as e:
        logger.error(f"Failed to refresh continuous aggregate {view_name}: {e}")
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


def create_compression_policy(op, table_name: str, compress_after: str = "7 days",
                            schema: Optional[str] = None):
    """
    Create a compression policy for a hypertable (alternative to add_hypertable_compression).
    Skips if TimescaleDB is not available.
    
    Args:
        op: Alembic operations object
        table_name: Name of the hypertable
        compress_after: When to compress chunks (e.g., '7 days')
        schema: Optional schema name
    """
    if not is_timescaledb_available(op):
        logger.info(f"Skipping compression policy for {table_name} - TimescaleDB not available")
        return
    
    try:
        full_table_name = f"{schema}.{table_name}" if schema else table_name
        
        op.execute(text(f"""
            SELECT add_compression_policy('{full_table_name}', INTERVAL '{compress_after}')
        """))
        
        logger.info(f"Added compression policy for {table_name}")
        
    except (ProgrammingError, OperationalError) as e:
        logger.error(f"Failed to add compression policy for {table_name}: {e}")
        # Don't re-raise - allow migration to continue


def remove_compression_policy(op, table_name: str, schema: Optional[str] = None):
    """
    Remove a compression policy from a hypertable.
    Skips if TimescaleDB is not available.
    
    Args:
        op: Alembic operations object
        table_name: Name of the hypertable
        schema: Optional schema name
    """
    if not is_timescaledb_available(op):
        logger.info(f"Skipping compression policy removal for {table_name} - TimescaleDB not available")
        return
    
    try:
        full_table_name = f"{schema}.{table_name}" if schema else table_name
        
        op.execute(text(f"""
            SELECT remove_compression_policy('{full_table_name}')
        """))
        
        logger.info(f"Removed compression policy for {table_name}")
        
    except (ProgrammingError, OperationalError) as e:
        logger.error(f"Failed to remove compression policy for {table_name}: {e}")
        # Don't re-raise - allow migration to continue


# Utility function for creating complete time-series table setup
def create_timeseries_table_complete(op, table_name: str, time_column: str, 
                                   schema: Optional[str] = None,
                                   chunk_time_interval: str = "1 day",
                                   compress_after: str = "7 days",
                                   retention_period: Optional[str] = None,
                                   compress_segmentby: Optional[Union[List[str], str]] = None,
                                   compress_orderby: Optional[Union[List[str], str]] = None):
    """
    Complete setup for a time-series table with hypertable, compression, and retention.
    This is a convenience function that combines multiple operations.
    Gracefully handles non-TimescaleDB environments.
    
    Args:
        op: Alembic operations object
        table_name: Name of the table (should already exist)
        time_column: Name of the timestamp column
        schema: Optional schema name
        chunk_time_interval: Time interval for chunks
        compress_after: When to compress chunks
        retention_period: Optional retention period (e.g., '90 days')
        compress_segmentby: Column(s) for compression segmenting
        compress_orderby: Column(s) for compression ordering
    """
    logger.info(f"Setting up time-series table: {table_name}")
    
    # 1. Create hypertable
    create_hypertable(
        op, table_name, time_column, schema=schema,
        chunk_time_interval=chunk_time_interval
    )
    
    # 2. Add compression if TimescaleDB is available
    if compress_segmentby or compress_orderby:
        add_hypertable_compression(
            op, table_name, schema=schema, compress_after=compress_after,
            compress_segmentby=compress_segmentby, compress_orderby=compress_orderby
        )
    else:
        create_compression_policy(op, table_name, compress_after=compress_after, schema=schema)
    
    # 3. Add retention policy if specified
    if retention_period:
        create_retention_policy(op, table_name, retention_period=retention_period, schema=schema)
    
    # 4. Create optimized index
    create_time_bucket_index(op, table_name, time_column, schema=schema)
    
    logger.info(f"Completed time-series setup for {table_name}")