"""
TimescaleDB integration module.

This module provides functions for TimescaleDB features like hypertables,
compression policies, and time-bucket queries.

MVP version with Supabase compatibility - gracefully handles environments 
without TimescaleDB support.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import DDL, event, text
from sqlalchemy.engine import Connection
from sqlalchemy.exc import ProgrammingError, OperationalError

# Import market data models
from app.models.market_data import OHLCV, Tick, OrderBookSnapshot

# Set up logging
logger = logging.getLogger(__name__)


# ------------------------------------------------
# TimescaleDB Availability Check
# ------------------------------------------------

def is_timescaledb_available(connection: Connection) -> bool:
    """
    Check if TimescaleDB extension is available.
    Returns False for Supabase or when TimescaleDB is not installed.
    """
    try:
        # Check if we're on Supabase (no TimescaleDB support)
        if 'supabase.co' in str(connection.engine.url):
            logger.info("Running on Supabase - TimescaleDB features disabled")
            return False
        
        # Check if TimescaleDB extension exists
        result = connection.execute(text("SELECT extname FROM pg_extension WHERE extname = 'timescaledb'"))
        available = result.fetchone() is not None
        
        if available:
            logger.info("TimescaleDB extension is available")
        else:
            logger.warning("TimescaleDB extension not found - features disabled")
            
        return available
    except Exception as e:
        logger.warning(f"Error checking TimescaleDB availability: {e}")
        return False


# ------------------------------------------------
# Hypertable Creation Event Listeners
# ------------------------------------------------

@event.listens_for(OHLCV.__table__, 'after_create')
def create_ohlcv_hypertable(target, connection, **kw):
    """Create TimescaleDB hypertable for OHLCV data - skips on Supabase"""
    if not is_timescaledb_available(connection):
        logger.info("Skipping OHLCV hypertable creation - TimescaleDB not available")
        return
    
    try:
        logger.info("Creating TimescaleDB hypertable for OHLCV data")
        connection.execute(DDL(
            "SELECT create_hypertable('ohlcv', 'timestamp', "   
            "chunk_time_interval => interval '1 day', if_not_exists => TRUE);"
        ))
        
        # Create compression policy
        connection.execute(DDL(
            "ALTER TABLE ohlcv SET (timescaledb.compress, timescaledb.compress_segmentby = 'instrument_id,interval');"
        ))
        
        # Add compression policy - compress data older than 7 days
        connection.execute(DDL(
            "SELECT add_compression_policy('ohlcv', INTERVAL '7 days');"
        ))
        
        logger.info("OHLCV hypertable created successfully")
        
    except (ProgrammingError, OperationalError) as e:
        logger.error(f"Failed to create OHLCV hypertable: {e}")
        # Don't re-raise - allow table to function as regular PostgreSQL table


@event.listens_for(Tick.__table__, 'after_create')
def create_tick_hypertable(target, connection, **kw):
    """Create TimescaleDB hypertable for tick data - skips on Supabase"""
    if not is_timescaledb_available(connection):
        logger.info("Skipping tick hypertable creation - TimescaleDB not available")
        return
    
    try:
        logger.info("Creating TimescaleDB hypertable for tick data")
        connection.execute(DDL(
            "SELECT create_hypertable('tick', 'timestamp', "
            "chunk_time_interval => interval '1 hour', if_not_exists => TRUE);"
        ))
        
        # Create compression policy
        connection.execute(DDL(
            "ALTER TABLE tick SET (timescaledb.compress, timescaledb.compress_segmentby = 'instrument_id');"
        ))
        
        # Add compression policy - compress data older than 1 day
        connection.execute(DDL(
            "SELECT add_compression_policy('tick', INTERVAL '1 day');"
        ))
        
        logger.info("Tick hypertable created successfully")
        
    except (ProgrammingError, OperationalError) as e:
        logger.error(f"Failed to create tick hypertable: {e}")
        # Don't re-raise - allow table to function as regular PostgreSQL table


@event.listens_for(OrderBookSnapshot.__table__, 'after_create')
def create_orderbook_hypertable(target, connection, **kw):
    """Create TimescaleDB hypertable for order book data - skips on Supabase"""
    if not is_timescaledb_available(connection):
        logger.info("Skipping order book hypertable creation - TimescaleDB not available")
        return
    
    try:
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
        
        # Add compression policy - compress data older than 1 day
        connection.execute(DDL(
            "SELECT add_compression_policy('order_book_snapshot', INTERVAL '1 day');"
        ))
        
        logger.info("Order book hypertable created successfully")
        
    except (ProgrammingError, OperationalError) as e:
        logger.error(f"Failed to create order book hypertable: {e}")
        # Don't re-raise - allow table to function as regular PostgreSQL table


# ------------------------------------------------
# Query Utilities (Works with both TimescaleDB and PostgreSQL)
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
    Execute a time bucket query for time-series aggregation.
    Falls back to date_trunc() if time_bucket() is not available.
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
    
    try:
        # Try TimescaleDB time_bucket function first
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
        
        result = connection.execute(text(query), params)
        return [{"bucket": row.bucket, "value": row.value} for row in result]
        
    except (ProgrammingError, OperationalError) as e:
        # Fall back to PostgreSQL date_trunc if time_bucket is not available
        logger.warning(f"time_bucket not available, falling back to date_trunc: {e}")
        
        # Convert interval to date_trunc precision
        precision_map = {
            '1 minute': 'minute',
            '5 minutes': 'minute',  # Will need post-processing
            '1 hour': 'hour',
            '1 day': 'day',
            '1 week': 'week',
            '1 month': 'month'
        }
        
        precision = precision_map.get(interval, 'hour')
        
        query = f"""
        SELECT 
            date_trunc('{precision}', {time_column}) AS bucket,
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
        
        result = connection.execute(text(query), params)
        return [{"bucket": row.bucket, "value": row.value} for row in result]


def get_ohlcv_from_ticks(connection: Connection,
                        instrument_id: str, 
                        interval: str, 
                        start_time: datetime, 
                        end_time: datetime) -> List[Dict[str, Any]]:
    """
    Generate OHLCV data from tick data for a specified interval.
    Works with both TimescaleDB and standard PostgreSQL.
    """
    params = {
        "instrument_id": instrument_id,
        "start_time": start_time,
        "end_time": end_time
    }
    
    try:
        # Try TimescaleDB-specific query with time_bucket and FIRST/LAST functions
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
        
        result = connection.execute(text(query), params)
        
    except (ProgrammingError, OperationalError) as e:
        # Fall back to standard PostgreSQL query
        logger.warning(f"TimescaleDB functions not available, using standard SQL: {e}")
        
        # Convert interval to date_trunc precision
        precision_map = {
            '1 minute': 'minute',
            '5 minutes': 'minute',  # Will need post-processing
            '1 hour': 'hour',
            '1 day': 'day'
        }
        
        precision = precision_map.get(interval, 'hour')
        
        query = f"""
        SELECT 
            date_trunc('{precision}', timestamp) AS time,
            instrument_id,
            MIN(price) AS open,  -- Approximation: using MIN as open
            MAX(price) AS high,
            MIN(price) AS low,
            MAX(price) AS close,  -- Approximation: using MAX as close
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
# Compression Management (TimescaleDB only)
# ------------------------------------------------

def compress_chunks(connection: Connection, table_name: str, older_than: str):
    """
    Manually compress chunks older than specified interval.
    Only works with TimescaleDB - silently skips on Supabase.
    """
    if not is_timescaledb_available(connection):
        logger.info(f"Skipping chunk compression for {table_name} - TimescaleDB not available")
        return
    
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


# ------------------------------------------------
# Registration Function
# ------------------------------------------------

def register_timescale_listeners():
    """
    Register all TimescaleDB event listeners.
    Event listeners are registered at module import time.
    This function is provided for explicitness in application startup.
    """
    logger.info("TimescaleDB event listeners registered")
    # Event listeners are registered when the module is imported
    pass