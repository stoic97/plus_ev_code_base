"""
TimescaleDB integration module.

This module provides specialized functions for TimescaleDB features like hypertables,
continuous aggregates, compression policies, and time-bucket queries.

It extends the basic SQLAlchemy ORM models with TimescaleDB-specific optimizations
for time-series data management.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

from sqlalchemy import DDL, event, text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.declarative import DeclarativeMeta

# Import market data models
from app.models.market_data import OHLCV, Tick, OrderBookSnapshot

# Set up logging
logger = logging.getLogger(__name__)


# ------------------------------------------------
# Hypertable Creation Event Listeners
# ------------------------------------------------

@event.listens_for(OHLCV.__table__, 'after_create')
def create_ohlcv_hypertable(target, connection, **kw):
    """Create TimescaleDB hypertable for OHLCV data"""
    logger.info("Creating TimescaleDB hypertable for OHLCV data")
    connection.execute(DDL(
        "SELECT create_hypertable('ohlcv', 'timestamp', "
        "chunk_time_interval => interval '1 day', if_not_exists => TRUE);"
    ))
    
    # Create compression policy
    connection.execute(DDL(
        "ALTER TABLE ohlcv SET (timescaledb.compress, timescaledb.compress_segmentby = 'instrument_id,interval');"
    ))
    
    # Add retention policy - compress data older than 7 days
    connection.execute(DDL(
        "SELECT add_compression_policy('ohlcv', INTERVAL '7 days');"
    ))


@event.listens_for(Tick.__table__, 'after_create')
def create_tick_hypertable(target, connection, **kw):
    """Create TimescaleDB hypertable for tick data"""
    logger.info("Creating TimescaleDB hypertable for tick data")
    connection.execute(DDL(
        "SELECT create_hypertable('tick', 'timestamp', "
        "chunk_time_interval => interval '1 hour', if_not_exists => TRUE);"
    ))
    
    # Create compression policy
    connection.execute(DDL(
        "ALTER TABLE tick SET (timescaledb.compress, timescaledb.compress_segmentby = 'instrument_id');"
    ))
    
    # Add retention policy - compress data older than 1 day
    connection.execute(DDL(
        "SELECT add_compression_policy('tick', INTERVAL '1 day');"
    ))


@event.listens_for(OrderBookSnapshot.__table__, 'after_create')
def create_orderbook_hypertable(target, connection, **kw):
    """Create TimescaleDB hypertable for order book data"""
    logger.info("Creating TimescaleDB hypertable for order book data")
    connection.execute(DDL(
        "SELECT create_hypertable('order_book_snapshot', 'timestamp', "
        "chunk_time_interval => interval '1 hour', if_not_exists => TRUE);"
    ))
    
    # Create compression policy
    connection.execute(DDL(
        "ALTER TABLE order_book_snapshot SET (timescaledb.compress, "
        "timescaledb.compress_segmentby = 'instrument_id');"
    ))
    
    # Add retention policy - compress data older than 1 day
    connection.execute(DDL(
        "SELECT add_compression_policy('order_book_snapshot', INTERVAL '1 day');"
    ))


# ------------------------------------------------
# TimescaleDB Query Utilities
# ------------------------------------------------

def execute_time_bucket_query(connection: Connection, 
                            table: str, 
                            time_column: str, 
                            interval: str,
                            agg_func: str, 
                            value_column: str,
                            start_time: datetime, 
                            end_time: datetime,
                            filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Execute a time_bucket query for time-series aggregation.
    
    Args:
        connection: SQLAlchemy connection
        table: Table name
        time_column: Timestamp column name
        interval: Time bucket interval (e.g., '1 hour', '1 day')
        agg_func: Aggregation function (e.g., 'AVG', 'SUM', 'MAX')
        value_column: Column to aggregate
        start_time: Start of time range
        end_time: End of time range
        filters: Additional filter conditions
    
    Returns:
        List of time buckets with aggregated values
    """
    # Build WHERE clause from filters
    where_clauses = []
    params = {"start_time": start_time, "end_time": end_time}
    
    # Add time range filter
    where_clauses.append(f"{time_column} >= :start_time AND {time_column} <= :end_time")
    
    # Add additional filters
    if filters:
        for i, (key, value) in enumerate(filters.items()):
            param_name = f"param_{i}"
            where_clauses.append(f"{key} = :{param_name}")
            params[param_name] = value
    
    where_clause = " AND ".join(where_clauses)
    
    # Build query
    query = f"""
    SELECT 
        time_bucket('{interval}', {time_column}) AS bucket,
        {agg_func}({value_column}) AS value
    FROM 
        {table}
    WHERE 
        {where_clause}
    GROUP BY 
        bucket
    ORDER BY 
        bucket ASC
    """
    
    # Execute query
    result = connection.execute(text(query), params)
    return [{"bucket": row.bucket, "value": row.value} for row in result]


def get_ohlcv_from_ticks(connection: Connection,
                        instrument_id: str, 
                        interval: str, 
                        start_time: datetime, 
                        end_time: datetime) -> List[Dict[str, Any]]:
    """
    Generate OHLCV data from tick data for a specified interval.
    
    Args:
        connection: SQLAlchemy connection
        instrument_id: UUID of the instrument
        interval: Time bucket interval (e.g., '1 minute', '1 hour', '1 day')
        start_time: Start of time range
        end_time: End of time range
    
    Returns:
        List of OHLCV data points
    """
    query = f"""
    SELECT 
        time_bucket('{interval}', timestamp) AS time,
        instrument_id,
        FIRST(price, timestamp) AS open,
        MAX(price) AS high,
        MIN(price) AS low,
        LAST(price, timestamp) AS close,
        SUM(volume) AS volume,
        COUNT(*) AS trades_count
    FROM 
        tick
    WHERE 
        instrument_id = :instrument_id
        AND timestamp >= :start_time
        AND timestamp <= :end_time
    GROUP BY 
        time, instrument_id
    ORDER BY 
        time ASC
    """
    
    params = {
        "instrument_id": instrument_id,
        "start_time": start_time,
        "end_time": end_time
    }
    
    result = connection.execute(text(query), params)
    return [
        {
            "time": row.time,
            "instrument_id": row.instrument_id,
            "open": row.open,
            "high": row.high,
            "low": row.low,
            "close": row.close,
            "volume": row.volume,
            "trades_count": row.trades_count
        }
        for row in result
    ]


# ------------------------------------------------
# Continuous Aggregate Management
# ------------------------------------------------

def configure_continuous_aggregates(connection: Connection, table_name: str, interval: str):
    """
    Configure continuous aggregates for automatic materialized views.
    
    Args:
        connection: SQLAlchemy connection
        table_name: Table to create continuous aggregate for (e.g., 'ohlcv')
        interval: Time bucket interval (e.g., '1 hour', '1 day')
    """
    view_name = f"{table_name}_{interval.replace(' ', '_')}_agg"
    
    # Different query depending on table type
    if table_name == 'ohlcv':
        query = f"""
        CREATE MATERIALIZED VIEW {view_name} 
        WITH (timescaledb.continuous) AS
        SELECT 
            time_bucket('{interval}', timestamp) AS bucket,
            instrument_id,
            interval,
            FIRST(open, timestamp) AS open,
            MAX(high) AS high,
            MIN(low) AS low,
            LAST(close, timestamp) AS close,
            SUM(volume) AS volume
        FROM 
            ohlcv
        GROUP BY bucket, instrument_id, interval;
        """
    elif table_name == 'tick':
        query = f"""
        CREATE MATERIALIZED VIEW {view_name}
        WITH (timescaledb.continuous) AS
        SELECT 
            time_bucket('{interval}', timestamp) AS bucket,
            instrument_id,
            AVG(price) AS avg_price,
            SUM(volume) AS volume,
            COUNT(*) AS trade_count
        FROM 
            tick
        GROUP BY bucket, instrument_id;
        """
    else:
        raise ValueError(f"Unsupported table for continuous aggregates: {table_name}")
    
    try:
        connection.execute(DDL(query))
        
        # Add refresh policy - refresh data every hour for data older than 1 hour
        refresh_query = f"""
        SELECT add_continuous_aggregate_policy('{view_name}',
            start_offset => INTERVAL '2 days',
            end_offset => INTERVAL '1 hour',
            schedule_interval => INTERVAL '1 hour');
        """
        
        connection.execute(DDL(refresh_query))
        logger.info(f"Created continuous aggregate view: {view_name}")
    except Exception as e:
        logger.error(f"Failed to create continuous aggregate: {e}")
        raise


# ------------------------------------------------
# Compression Management
# ------------------------------------------------

def compress_chunks(connection: Connection, table_name: str, older_than: str):
    """
    Manually compress chunks older than specified interval.
    
    Args:
        connection: SQLAlchemy connection
        table_name: Table name
        older_than: Interval string (e.g., '1 day', '7 days')
    """
    query = f"""
    SELECT compress_chunk(chunk)
    FROM timescaledb_information.chunks
    WHERE hypertable_name = '{table_name}'
    AND chunk_status = 'Uncompressed'
    AND range_end < NOW() - INTERVAL '{older_than}';
    """
    
    try:
        result = connection.execute(DDL(query))
        compressed_count = result.rowcount
        logger.info(f"Compressed {compressed_count} chunks for {table_name}")
    except Exception as e:
        logger.error(f"Failed to compress chunks: {e}")
        raise


def decompress_chunks(connection: Connection, table_name: str, time_range: Tuple[datetime, datetime]):
    """
    Decompress chunks in a specific time range for faster queries.
    
    Args:
        connection: SQLAlchemy connection
        table_name: Table name
        time_range: (start_time, end_time) tuple
    """
    start_time, end_time = time_range
    
    query = f"""
    SELECT decompress_chunk(chunk)
    FROM timescaledb_information.chunks
    WHERE hypertable_name = '{table_name}'
    AND chunk_status = 'Compressed'
    AND range_start <= '{end_time}'::timestamptz
    AND range_end >= '{start_time}'::timestamptz;
    """
    
    try:
        result = connection.execute(DDL(query))
        decompressed_count = result.rowcount
        logger.info(f"Decompressed {decompressed_count} chunks for {table_name}")
    except Exception as e:
        logger.error(f"Failed to decompress chunks: {e}")
        raise


# ------------------------------------------------
# System Monitoring
# ------------------------------------------------

def get_chunk_statistics(connection: Connection, table_name: str) -> List[Dict[str, Any]]:
    """
    Get statistics about hypertable chunks.
    
    Args:
        connection: SQLAlchemy connection
        table_name: Table name
    
    Returns:
        List of chunk statistics
    """
    query = f"""
    SELECT 
        chunk_name,
        range_start,
        range_end,
        chunk_status,
        pg_size_pretty(before_compression_total_bytes) as before_size,
        pg_size_pretty(after_compression_total_bytes) as after_size
    FROM timescaledb_information.chunks
    WHERE hypertable_name = '{table_name}'
    ORDER BY range_start DESC;
    """
    
    try:
        result = connection.execute(text(query))
        return [dict(row) for row in result]
    except Exception as e:
        logger.error(f"Failed to get chunk statistics: {e}")
        raise


def get_compression_statistics(connection: Connection, table_name: str) -> Dict[str, Any]:
    """
    Get compression statistics for a hypertable.
    
    Args:
        connection: SQLAlchemy connection
        table_name: Table name
    
    Returns:
        Dictionary with compression statistics
    """
    query = f"""
    SELECT 
        hypertable_name,
        pg_size_pretty(SUM(before_compression_total_bytes)) as total_uncompressed,
        pg_size_pretty(SUM(after_compression_total_bytes)) as total_compressed,
        ROUND(SUM(after_compression_total_bytes)::numeric / 
              NULLIF(SUM(before_compression_total_bytes), 0)::numeric, 2) as compression_ratio
    FROM timescaledb_information.chunks
    WHERE hypertable_name = '{table_name}'
    AND chunk_status = 'Compressed'
    GROUP BY hypertable_name;
    """
    
    try:
        result = connection.execute(text(query))
        row = result.fetchone()
        if row:
            return dict(row)
        else:
            return {
                "hypertable_name": table_name,
                "total_uncompressed": "0 bytes",
                "total_compressed": "0 bytes",
                "compression_ratio": 0
            }
    except Exception as e:
        logger.error(f"Failed to get compression statistics: {e}")
        raise


# ------------------------------------------------
# Registration Function
# ------------------------------------------------

def register_timescale_listeners():
    """
    Register all TimescaleDB event listeners.
    This function is a no-op since SQLAlchemy event listeners are registered
    at module import time, but it's included for explicitness and documentation.
    """
    logger.info("TimescaleDB event listeners registered")
    # Event listeners are registered when the module is imported
    # This function is provided for explicitness in application startup
    pass