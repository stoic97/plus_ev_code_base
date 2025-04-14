"""
TimescaleDB-specific migration helpers.

This module provides specialized functions for working with TimescaleDB
features like hypertables, continuous aggregates, and compression.
"""

from sqlalchemy import text
from alembic import op


def create_hypertable(op, table_name, time_column_name, schema=None, 
                     chunk_time_interval="1 day", if_not_exists=True,
                     migrate_data=False, partitioning_column=None,
                     number_partitions=None):
    """
    Create a TimescaleDB hypertable from an existing table.
    
    Args:
        table_name: Name of the table to convert to a hypertable
        time_column_name: Name of the timestamp column to use for partitioning
        schema: Optional schema name
        chunk_time_interval: Time interval for chunks (e.g., '1 day', '1 hour')
        if_not_exists: Only create if it doesn't already exist
        migrate_data: Whether to migrate existing data into chunks
        partitioning_column: Optional column for space partitioning
        number_partitions: Optional number of space partitions to create
        op: Alembic operations object
        table_name: Name of the table to convert to a hypertable
        time_column_name: Name of the timestamp column to use for partitioning
    """
    schema_clause = f"'{schema}', " if schema else ""
    if_not_exists_clause = "IF NOT EXISTS" if if_not_exists else ""
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
            {if_not_exists_clause}
            chunk_time_interval => INTERVAL '{chunk_time_interval}',
            {partitioning_options}
            {migrate_data_clause}
        )
    """
    
    op.execute(text(query))


def add_hypertable_compression(op, table_name, schema=None, compress_after="7 days",
                              compress_segmentby=None, compress_orderby=None):
    """
    Add native compression to a TimescaleDB hypertable.
    
    Args:
        table_name: Name of the hypertable
        schema: Optional schema name
        compress_after: When to compress chunks (e.g., '7 days')
        compress_segmentby: Column(s) to use for segmenting data
        compress_orderby: Column(s) to use for ordering data
    """
    full_table_name = f"{schema}.{table_name}" if schema else table_name
    
    # Build segment by clause
    segmentby_clause = ""
    if compress_segmentby:
        if isinstance(compress_segmentby, list):
            segmentby_cols = ",".join([f"'{col}'" for col in compress_segmentby])
            segmentby_clause = f"segmentby => ARRAY[{segmentby_cols}]"
        else:
            segmentby_clause = f"segmentby => ARRAY['{compress_segmentby}']"
    
    # Build order by clause
    orderby_clause = ""
    if compress_orderby:
        if isinstance(compress_orderby, list):
            orderby_cols = ",".join([f"'{col}'" for col in compress_orderby])
            orderby_clause = f"orderby => ARRAY[{orderby_cols}]"
        else:
            orderby_clause = f"orderby => ARRAY['{compress_orderby}']"
    
    # Combine clauses
    compression_clauses = []
    if segmentby_clause:
        compression_clauses.append(segmentby_clause)
    if orderby_clause:
        compression_clauses.append(orderby_clause)
    
    compress_params = ", ".join(compression_clauses)
    
    # Set compression policy
    op.execute(text(f"""
        ALTER TABLE {full_table_name} SET (
            timescaledb.compress = TRUE,
            {compress_params}
        )
    """))
    
    # Add compression policy
    op.execute(text(f"""
        SELECT add_compression_policy('{full_table_name}', INTERVAL '{compress_after}')
    """))


def create_continuous_aggregate(op, view_name, hypertable_name, query, 
                              refresh_interval="1 hour", 
                              refresh_lag="3 hours",
                              materialized_only=True,
                              with_data=True,
                              if_not_exists=True):
    """
    Create a TimescaleDB continuous aggregate.
    
    Args:
        view_name: Name for the continuous aggregate view
        hypertable_name: Name of the source hypertable
        query: SELECT query defining the aggregate
        refresh_interval: How often to refresh the view
        refresh_lag: How far behind real-time to refresh
        materialized_only: Whether to use materialized data only
        with_data: Whether to materialize existing data
        if_not_exists: Only create if it doesn't already exist
    """
    if_not_exists_clause = "IF NOT EXISTS" if if_not_exists else ""
    materialized_clause = "timescaledb.materialized_only=true" if materialized_only else ""
    with_data_clause = "WITH DATA" if with_data else "WITH NO DATA"
    
    # Create the continuous aggregate view
    op.execute(text(f"""
        CREATE MATERIALIZED VIEW {if_not_exists_clause} {view_name}
        WITH (timescaledb.continuous, {materialized_clause}) AS
        {query}
        {with_data_clause}
    """))
    
    # Add refresh policy
    op.execute(text(f"""
        SELECT add_continuous_aggregate_policy('{view_name}',
            start_offset => INTERVAL '{refresh_lag}',
            end_offset => INTERVAL '1 minute',
            schedule_interval => INTERVAL '{refresh_interval}')
    """))


def create_retention_policy(op, table_name, retention_period="90 days", schema=None):
    """
    Create a data retention policy for a hypertable.
    
    Args:
        table_name: Name of the hypertable
        retention_period: How long to keep data (e.g., '90 days')
        schema: Optional schema name
    """
    full_table_name = f"{schema}.{table_name}" if schema else table_name
    
    op.execute(text(f"""
        SELECT add_retention_policy(
            '{full_table_name}', 
            INTERVAL '{retention_period}'
        )
    """))


def remove_retention_policy(op, table_name, schema=None):
    """
    Remove a data retention policy from a hypertable.
    
    Args:
        table_name: Name of the hypertable
        schema: Optional schema name
    """
    full_table_name = f"{schema}.{table_name}" if schema else table_name
    
    op.execute(text(f"""
        SELECT remove_retention_policy('{full_table_name}')
    """))


def create_time_bucket_index(op, table_name, time_column, bucket_interval="1 day", 
                           include_columns=None, schema=None):
    """
    Create an optimized index for time_bucket queries.
    
    Args:
        table_name: Name of the table
        time_column: Name of the timestamp column
        bucket_interval: Bucket interval for the index
        include_columns: Additional columns to include in the index
        schema: Optional schema name
    """
    full_table_name = f"{schema}.{table_name}" if schema else table_name
    index_name = f"idx_{table_name}_{time_column}_bucket"
    
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


def is_timescaledb_available(op):
    """Check if TimescaleDB extension is available."""
    try:
        connection = op.get_bind()
        result = connection.execute(text("SELECT extname FROM pg_extension WHERE extname = 'timescaledb'"))
        return result.fetchone() is not None
    except Exception:
        return False