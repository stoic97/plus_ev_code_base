"""
TimescaleDB-specific migration helpers.

This module provides specialized functions for working with TimescaleDB
features like hypertables, continuous aggregates, and compression.
"""

from typing import Any, Dict, List, Optional, Union
from sqlalchemy import text
from alembic import op
from alembic.operations import Operations, MigrateOperation


def create_hypertable(op, table_name, time_column_name, schema=None, 
                     chunk_time_interval="1 day", if_not_exists=True,
                     migrate_data=False, partitioning_column=None,
                     number_partitions=None):
    """
    Create a TimescaleDB hypertable from an existing table.
    
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
            if_not_exists => {if_not_exists_clause},
            {migrate_data_clause}
        )
    """
    
    op.execute(text(query))


def add_hypertable_compression(op, table_name, schema=None, compress_after="7 days",
                              compress_segmentby=None, compress_orderby=None):
    """
    Add native compression to a TimescaleDB hypertable.
    
    Args:
        op: Alembic operations object
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
            segmentby_clause = f"timescaledb.compress_segmentby = ARRAY[{segmentby_cols}]"
        else:
            segmentby_clause = f"timescaledb.compress_segmentby = ARRAY['{compress_segmentby}']"
    
    # Build order by clause
    orderby_clause = ""
    if compress_orderby:
        if isinstance(compress_orderby, list):
            orderby_cols = ",".join([f"'{col}'" for col in compress_orderby])
            orderby_clause = f"timescaledb.compress_orderby = ARRAY[{orderby_cols}]"
        else:
            orderby_clause = f"timescaledb.compress_orderby = ARRAY['{compress_orderby}']"
    
    # Combine clauses
    compression_clauses = ["timescaledb.compress = TRUE"]
    if segmentby_clause:
        compression_clauses.append(segmentby_clause)
    if orderby_clause:
        compression_clauses.append(orderby_clause)
    
    compress_params = ", ".join(compression_clauses)
    
    # Set compression policy
    op.execute(text(f"""
        ALTER TABLE {full_table_name} SET (
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
        op: Alembic operations object
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
        op: Alembic operations object
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
        op: Alembic operations object
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
        op: Alembic operations object
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
    """Check if TimescaleDB extension is available in the database."""
    try:
        connection = op.get_bind()
        result = connection.execute(text(
            "SELECT extname FROM pg_extension WHERE extname = 'timescaledb'"
        ))
        return result.fetchone() is not None
    except Exception:
        return False


# Custom Alembic operation for creating a hypertable
@Operations.register_operation("create_timescaledb_hypertable")
class CreateHypertableOp(MigrateOperation):
    """Create a TimescaleDB hypertable operation."""
    
    def __init__(
        self,
        table_name: str,
        time_column: str,
        schema: Optional[str] = None,
        chunk_time_interval: str = '1 day',
        if_not_exists: bool = True,
        migrate_data: bool = False,
        partitioning_column: Optional[str] = None,
        number_partitions: Optional[int] = None
    ):
        self.table_name = table_name
        self.time_column = time_column
        self.schema = schema
        self.chunk_time_interval = chunk_time_interval
        self.if_not_exists = if_not_exists
        self.migrate_data = migrate_data
        self.partitioning_column = partitioning_column
        self.number_partitions = number_partitions
    
    @classmethod
    def create_timescaledb_hypertable(
        cls,
        operations: Operations,
        table_name: str,
        time_column: str,
        schema: Optional[str] = None,
        chunk_time_interval: str = '1 day',
        if_not_exists: bool = True,
        migrate_data: bool = False,
        partitioning_column: Optional[str] = None,
        number_partitions: Optional[int] = None
    ) -> None:
        """Create a hypertable using the Alembic Operations API."""
        op = CreateHypertableOp(
            table_name,
            time_column,
            schema,
            chunk_time_interval,
            if_not_exists,
            migrate_data,
            partitioning_column,
            number_partitions
        )
        return operations.invoke(op)


@Operations.implementation_for(CreateHypertableOp)
def create_timescaledb_hypertable(operations, operation):
    """Implementation for create_timescaledb_hypertable operation."""
    create_hypertable(
        operations,
        operation.table_name,
        operation.time_column,
        schema=operation.schema,
        chunk_time_interval=operation.chunk_time_interval,
        if_not_exists=operation.if_not_exists,
        migrate_data=operation.migrate_data,
        partitioning_column=operation.partitioning_column,
        number_partitions=operation.number_partitions
    )


# Custom Alembic operation for adding compression
@Operations.register_operation("add_timescaledb_compression")
class AddCompressionOp(MigrateOperation):
    """Add TimescaleDB compression operation."""
    
    def __init__(
        self,
        table_name: str,
        schema: Optional[str] = None,
        compress_after: str = '7 days',
        compress_segmentby: Optional[Union[List[str], str]] = None,
        compress_orderby: Optional[Union[List[str], str]] = None
    ):
        self.table_name = table_name
        self.schema = schema
        self.compress_after = compress_after
        self.compress_segmentby = compress_segmentby
        self.compress_orderby = compress_orderby
    
    @classmethod
    def add_timescaledb_compression(
        cls,
        operations: Operations,
        table_name: str,
        schema: Optional[str] = None,
        compress_after: str = '7 days',
        compress_segmentby: Optional[Union[List[str], str]] = None,
        compress_orderby: Optional[Union[List[str], str]] = None
    ) -> None:
        """Add compression policy using the Alembic Operations API."""
        op = AddCompressionOp(
            table_name,
            schema,
            compress_after,
            compress_segmentby,
            compress_orderby
        )
        return operations.invoke(op)


@Operations.implementation_for(AddCompressionOp)
def add_timescaledb_compression(operations, operation):
    """Implementation for add_timescaledb_compression operation."""
    add_hypertable_compression(
        operations,
        operation.table_name,
        schema=operation.schema,
        compress_after=operation.compress_after,
        compress_segmentby=operation.compress_segmentby,
        compress_orderby=operation.compress_orderby
    )


# Custom Alembic operation for creating a retention policy
@Operations.register_operation("add_timescaledb_retention_policy")
class AddRetentionPolicyOp(MigrateOperation):
    """Add TimescaleDB retention policy operation."""
    
    def __init__(
        self,
        table_name: str,
        retention_period: str = '90 days',
        schema: Optional[str] = None
    ):
        self.table_name = table_name
        self.retention_period = retention_period
        self.schema = schema
    
    @classmethod
    def add_timescaledb_retention_policy(
        cls,
        operations: Operations,
        table_name: str,
        retention_period: str = '90 days',
        schema: Optional[str] = None
    ) -> None:
        """Add retention policy using the Alembic Operations API."""
        op = AddRetentionPolicyOp(
            table_name,
            retention_period,
            schema
        )
        return operations.invoke(op)


@Operations.implementation_for(AddRetentionPolicyOp)
def add_timescaledb_retention_policy(operations, operation):
    """Implementation for add_timescaledb_retention_policy operation."""
    create_retention_policy(
        operations,
        operation.table_name,
        retention_period=operation.retention_period,
        schema=operation.schema
    )