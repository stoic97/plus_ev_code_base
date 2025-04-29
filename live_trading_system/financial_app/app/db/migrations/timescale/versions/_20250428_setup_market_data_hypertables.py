"""TimescaleDB hypertable migration for market data tables.

Revision ID: 20250428_setup_market_data_hypertables
Revises: None
Create Date: 2025-04-28
Database: timescale
"""

from alembic import op
import sqlalchemy as sa
from datetime import datetime, timedelta

# Import TimescaleDB helpers
from app.db.migrations.helpers.timescale_support import (
    create_hypertable,
    add_hypertable_compression,
    create_continuous_aggregate,
    create_retention_policy,
    create_time_bucket_index,
    is_timescaledb_available
)

# Import database extensions helper
from app.db.migrations.helpers.db_init import create_extension

# Import models (for reference only)
from app.models.market_data import OHLCV, Tick, OrderBookSnapshot

# revision identifiers, used by Alembic
revision = '20250428_setup_market_data_hypertables'
down_revision = None  # Set this to your previous migration
branch_labels = None
depends_on = None
database = 'timescale'


def upgrade():
    """
    Upgrade database by converting regular tables to TimescaleDB hypertables.
    
    This migration:
    1. Checks and enables the TimescaleDB extension
    2. Converts market data tables to hypertables
    3. Sets up compression policies for efficient storage
    4. Creates time-bucket indexes for efficient querying
    5. Creates continuous aggregates for common time windows
    """
    # Check if TimescaleDB is available
    if not is_timescaledb_available(op):
        # Force disconnect all connections to safely create extension
        op.execute(sa.text(
            "SELECT pg_terminate_backend(pg_stat_activity.pid) "
            "FROM pg_stat_activity "
            "WHERE pg_stat_activity.datname = current_database() "
            "AND pid <> pg_backend_pid();"
        ))
        # Create TimescaleDB extension if not available
        create_extension(op, "timescaledb")
        
    # Convert OHLCV table to hypertable
    convert_ohlcv_to_hypertable()
    
    # Convert Tick table to hypertable
    convert_tick_to_hypertable()
    
    # Convert OrderBookSnapshot table to hypertable
    convert_orderbook_to_hypertable()
    
    # Create continuous aggregates for common time windows
    create_ohlcv_aggregates()
    
    # Set up data retention policies
    setup_retention_policies()


def downgrade():
    """
    Downgrade database by removing TimescaleDB-specific features.
    
    Note: This is mostly symbolic as hypertables can't be easily
    converted back to regular tables without data loss.
    """
    # Drop continuous aggregates
    op.execute(sa.text("DROP MATERIALIZED VIEW IF EXISTS ohlcv_hourly CASCADE"))
    op.execute(sa.text("DROP MATERIALIZED VIEW IF EXISTS ohlcv_daily CASCADE"))
    
    # Remove retention policies
    # Note: We can't actually convert hypertables back to regular tables easily,
    # so we leave them as is, just removing the continuous aggregates and policies.
    op.execute(sa.text("SELECT remove_retention_policy('tick', if_exists => true)"))
    op.execute(sa.text("SELECT remove_retention_policy('ohlcv', if_exists => true)"))
    op.execute(sa.text("SELECT remove_retention_policy('order_book_snapshot', if_exists => true)"))
    
    # Remove compression policies by setting them to NULL
    op.execute(sa.text("SELECT add_compression_policy('ohlcv', older_than => NULL, if_exists => true)"))
    op.execute(sa.text("SELECT add_compression_policy('tick', older_than => NULL, if_exists => true)"))
    op.execute(sa.text("SELECT add_compression_policy('order_book_snapshot', older_than => NULL, if_exists => true)"))


def convert_ohlcv_to_hypertable():
    """
    Convert the OHLCV table to a TimescaleDB hypertable.
    
    OHLCV data is typically lower volume with 1-minute to 1-day intervals,
    so we use a 1-day chunk interval for efficient storage and querying.
    """
    try:
        # Convert to hypertable with 1-day chunks
        create_hypertable(
            op,
            table_name="ohlcv",
            time_column_name="timestamp",
            chunk_time_interval="1 day",
            if_not_exists=True,
            migrate_data=True
        )
        
        # Add compression policy to compress chunks older than 30 days
        add_hypertable_compression(
            op,
            table_name="ohlcv",
            compress_after="30 days",
            compress_segmentby=["instrument_id", "interval"],
            compress_orderby="timestamp"
        )
        
        # Create compound index for common queries
        op.execute(sa.text(
            "CREATE INDEX IF NOT EXISTS ix_ohlcv_instrument_interval_time "
            "ON ohlcv (instrument_id, interval, timestamp DESC)"
        ))
        
        # Create time-bucket index for interval queries
        create_time_bucket_index(
            op,
            table_name="ohlcv",
            time_column="timestamp",
            bucket_interval="1 hour",
            include_columns=["instrument_id", "interval"]
        )
        
    except Exception as e:
        op.execute(sa.text("ROLLBACK"))
        raise Exception(f"Failed to convert OHLCV table to hypertable: {e}")


def convert_tick_to_hypertable():
    """
    Convert the Tick table to a TimescaleDB hypertable.
    
    Tick data is high volume with many entries per second,
    so we use a 4-hour chunk interval to balance between
    query performance and chunk management overhead.
    """
    try:
        # Convert to hypertable with 4-hour chunks
        create_hypertable(
            op,
            table_name="tick",
            time_column_name="timestamp",
            chunk_time_interval="4 hours",
            if_not_exists=True,
            migrate_data=True
        )
        
        # Add compression policy to compress chunks older than 7 days
        add_hypertable_compression(
            op,
            table_name="tick",
            compress_after="7 days",
            compress_segmentby=["instrument_id"],
            compress_orderby="timestamp"
        )
        
        # Create compound index for common queries
        op.execute(sa.text(
            "CREATE INDEX IF NOT EXISTS ix_tick_instrument_time "
            "ON tick (instrument_id, timestamp DESC)"
        ))
        
        # Create index on trade_id for lookups
        op.execute(sa.text(
            "CREATE INDEX IF NOT EXISTS ix_tick_trade_id_time "
            "ON tick (trade_id, timestamp DESC) "
            "WHERE trade_id IS NOT NULL"
        ))
        
    except Exception as e:
        op.execute(sa.text("ROLLBACK"))
        raise Exception(f"Failed to convert Tick table to hypertable: {e}")


def convert_orderbook_to_hypertable():
    """
    Convert the OrderBookSnapshot table to a TimescaleDB hypertable.
    
    OrderBookSnapshot data can be large but less frequent than ticks,
    so we use a 6-hour chunk interval.
    """
    try:
        # Convert to hypertable with 6-hour chunks
        create_hypertable(
            op,
            table_name="order_book_snapshot",
            time_column_name="timestamp",
            chunk_time_interval="6 hours",
            if_not_exists=True,
            migrate_data=True
        )
        
        # Add compression policy to compress chunks older than 3 days
        add_hypertable_compression(
            op,
            table_name="order_book_snapshot",
            compress_after="3 days",
            compress_segmentby=["instrument_id"],
            compress_orderby="timestamp"
        )
        
        # Create compound index for common queries
        op.execute(sa.text(
            "CREATE INDEX IF NOT EXISTS ix_orderbook_instrument_time "
            "ON order_book_snapshot (instrument_id, timestamp DESC)"
        ))
        
        # Create index on spread for analysis
        op.execute(sa.text(
            "CREATE INDEX IF NOT EXISTS ix_orderbook_spread "
            "ON order_book_snapshot (instrument_id, spread, timestamp DESC) "
            "WHERE spread IS NOT NULL"
        ))
        
    except Exception as e:
        op.execute(sa.text("ROLLBACK"))
        raise Exception(f"Failed to convert OrderBookSnapshot table to hypertable: {e}")


def create_ohlcv_aggregates():
    """
    Create continuous aggregates for OHLCV data.
    
    This creates materialized views for common time windows like hourly and daily,
    which are automatically refreshed as new data arrives.
    """
    try:
        # Create hourly aggregate for minute data
        hourly_query = """
        SELECT
            time_bucket('1 hour', timestamp) AS bucket,
            instrument_id,
            interval,
            first(open, timestamp) AS open,
            max(high) AS high,
            min(low) AS low,
            last(close, timestamp) AS close,
            sum(volume) AS volume,
            count(*) AS sample_count
        FROM ohlcv
        WHERE interval = '1m' OR interval = '5m'
        GROUP BY bucket, instrument_id, interval
        """
        
        create_continuous_aggregate(
            op,
            view_name="ohlcv_hourly",
            hypertable_name="ohlcv",
            query=hourly_query,
            refresh_interval="30 minutes",
            refresh_lag="1 hour",
            materialized_only=True,
            with_data=True
        )
        
        # Create daily aggregate for hourly data
        daily_query = """
        SELECT
            time_bucket('1 day', timestamp) AS bucket,
            instrument_id,
            interval,
            first(open, timestamp) AS open,
            max(high) AS high,
            min(low) AS low,
            last(close, timestamp) AS close,
            sum(volume) AS volume,
            count(*) AS sample_count
        FROM ohlcv
        WHERE interval = '1h' OR interval = '30m'
        GROUP BY bucket, instrument_id, interval
        """
        
        create_continuous_aggregate(
            op,
            view_name="ohlcv_daily",
            hypertable_name="ohlcv",
            query=daily_query,
            refresh_interval="1 hour",
            refresh_lag="3 hours",
            materialized_only=True,
            with_data=True
        )
        
        # Create indexes on continuous aggregates
        op.execute(sa.text(
            "CREATE INDEX IF NOT EXISTS ix_ohlcv_hourly_instrument_bucket "
            "ON ohlcv_hourly (instrument_id, interval, bucket DESC)"
        ))
        
        op.execute(sa.text(
            "CREATE INDEX IF NOT EXISTS ix_ohlcv_daily_instrument_bucket "
            "ON ohlcv_daily (instrument_id, interval, bucket DESC)"
        ))
        
    except Exception as e:
        op.execute(sa.text("ROLLBACK"))
        raise Exception(f"Failed to create continuous aggregates: {e}")


def setup_retention_policies():
    """
    Set up data retention policies to manage storage.
    
    This configures automatic data purging of older data to control
    database size and maintain performance.
    """
    try:
        # Set retention policy for tick data (keep 90 days)
        create_retention_policy(
            op,
            table_name="tick",
            retention_period="90 days"
        )
        
        # Set retention policy for orderbook data (keep 30 days)
        create_retention_policy(
            op,
            table_name="order_book_snapshot",
            retention_period="30 days"
        )
        
        # For OHLCV, we keep longer history but might set a policy for very old data
        # Uncommenting this would limit OHLCV data retention
        # create_retention_policy(
        #     op,
        #     table_name="ohlcv",
        #     retention_period="365 days"
        # )
        
        # Log retention policy setup
        op.execute(sa.text(
            "SELECT add_job('log_retention_policy_status', '24 hours', "
            "$$SELECT hypertable_name, older_than FROM timescaledb_information.jobs "
            "WHERE proc_name = 'policy_retention'$$)"
        ))
        
    except Exception as e:
        op.execute(sa.text("ROLLBACK"))
        raise Exception(f"Failed to set up retention policies: {e}")