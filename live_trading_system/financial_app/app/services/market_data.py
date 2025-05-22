"""
Market Data Service

This module provides business logic for retrieving and processing market data from TimescaleDB.
It implements efficient time-series data operations, caching, and data aggregation
for financial market data analysis.

Features:
- OHLCV data retrieval with time range filtering
- Time-series data aggregation at different intervals
- Statistical calculations on market data
- Performance optimization with caching
- Multi-symbol support
- Error handling and logging
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from uuid import UUID

from fastapi import HTTPException, status

from app.core.database import (
    DatabaseType, TimescaleDB, RedisDB, 
    get_db_instance, cache, db_session
)
from app.core.config import get_settings
from app.schemas.market_data import (
    AssetClass, DataSource, TimeInterval, 
    OHLCVResponse, OHLCVFilter, MarketDataStats,
    InstrumentResponse, MarketDataFilter
)

# Set up logging
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()


class MarketDataService:
    """
    Service for retrieving and processing market data from TimescaleDB.
    
    This service provides methods for:
    - OHLCV data retrieval with flexible filtering
    - Time-series data aggregation
    - Market data statistics and analysis
    - Multi-symbol support
    """
    
    def __init__(self):
        """Initialize the market data service."""
        self.timescale_db = get_db_instance(DatabaseType.TIMESCALEDB)
        self.redis_db = get_db_instance(DatabaseType.REDIS)
        self.settings = get_settings()
    
    #################################################
    # Instrument Methods
    #################################################
    
    @cache(ttl=300, key_prefix="instrument")
    def get_instrument_by_id(self, instrument_id: UUID) -> Optional[InstrumentResponse]:
        """
        Get instrument by ID.
        
        Args:
            instrument_id: Instrument UUID
        
        Returns:
            Instrument data or None if not found
        """
        try:
            # Use the object's session method directly for better testability
            with self.timescale_db.session() as session:
                # Query the instrument from the database
                result = session.execute(
                    """
                    SELECT id, symbol, name, asset_class, exchange, currency,
                           expiry_date, contract_size, price_adj_factor, specifications,
                           active, isin, figi, created_at, modified_at
                    FROM instruments
                    WHERE id = :instrument_id
                    """,
                    {"instrument_id": instrument_id}
                ).fetchone()
                
                if not result:
                    return None
                
                # Convert query result to dictionary and then to Pydantic model
                instrument_dict = dict(result)
                
                # Convert asset_class string to enum
                if "asset_class" in instrument_dict:
                    instrument_dict["asset_class"] = AssetClass(instrument_dict["asset_class"])
                
                return InstrumentResponse(**instrument_dict)
        except Exception as e:
            logger.error(f"Error retrieving instrument by ID: {e}")
            return None
    
    @cache(ttl=300, key_prefix="instrument_symbol")
    def get_instrument_by_symbol(self, symbol: str) -> Optional[InstrumentResponse]:
        """
        Get instrument by symbol.
        
        Args:
            symbol: Trading symbol
        
        Returns:
            Instrument data or None if not found
        """
        try:
            # Use the object's session method directly for better testability
            with self.timescale_db.session() as session:
                # Query the instrument from the database
                result = session.execute(
                    """
                    SELECT id, symbol, name, asset_class, exchange, currency,
                           expiry_date, contract_size, price_adj_factor, specifications,
                           active, isin, figi, created_at, modified_at
                    FROM instruments
                    WHERE symbol = :symbol
                    """,
                    {"symbol": symbol}
                ).fetchone()
                
                if not result:
                    return None
                
                # Convert query result to dictionary and then to Pydantic model
                instrument_dict = dict(result)
                
                # Convert asset_class string to enum
                if "asset_class" in instrument_dict:
                    instrument_dict["asset_class"] = AssetClass(instrument_dict["asset_class"])
                
                return InstrumentResponse(**instrument_dict)
        except Exception as e:
            logger.error(f"Error retrieving instrument by symbol: {e}")
            return None
    
    def resolve_instrument_id(self, instrument_id: Optional[UUID] = None, symbol: Optional[str] = None) -> Optional[UUID]:
        """
        Resolve instrument ID from either direct ID or symbol.
        
        Args:
            instrument_id: Optional instrument UUID
            symbol: Optional trading symbol
        
        Returns:
            Resolved instrument UUID or None if not found
        
        Raises:
            ValueError: If neither instrument_id nor symbol is provided
        """
        if instrument_id:
            return instrument_id
        
        if symbol:
            instrument = self.get_instrument_by_symbol(symbol)
            if instrument:
                return instrument.id
            
        if not instrument_id and not symbol:
            raise ValueError("Either instrument_id or symbol must be provided")
            
        return None
    
    #################################################
    # OHLCV Data Retrieval Methods
    #################################################
    
    @cache(ttl=60, key_prefix="ohlcv_latest")
    def get_latest_ohlcv(
        self, 
        instrument_id: Optional[UUID] = None, 
        symbol: Optional[str] = None,
        interval: TimeInterval = TimeInterval.DAY_1
    ) -> Optional[OHLCVResponse]:
        """
        Get the latest OHLCV data point for an instrument.
        
        Args:
            instrument_id: Optional instrument UUID
            symbol: Optional trading symbol
            interval: Time interval for OHLCV data
        
        Returns:
            Latest OHLCV data or None if not found
        
        Raises:
            HTTPException: If instrument not found
        """
        try:
            resolved_id = self.resolve_instrument_id(instrument_id, symbol)
            if not resolved_id:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Instrument not found"
                )
            
            # Use the object's session method directly for better testability
            with self.timescale_db.session() as session:
                # Query the latest OHLCV data
                result = session.execute(
                    """
                    SELECT id, instrument_id, timestamp, source, source_timestamp,
                           open, high, low, close, volume, interval, vwap,
                           trades_count, open_interest, adjusted_close,
                           is_anomaly, anomaly_reason, metadata,
                           created_at, modified_at
                    FROM ohlcv
                    WHERE instrument_id = :instrument_id AND interval = :interval
                    ORDER BY timestamp DESC
                    LIMIT 1
                    """,
                    {
                        "instrument_id": resolved_id,
                        "interval": interval.value
                    }
                ).fetchone()
                
                if not result:
                    return None
                
                # Convert query result to dictionary and then to Pydantic model
                ohlcv_dict = dict(result)
                
                # Convert string values to proper enum types
                if "source" in ohlcv_dict:
                    ohlcv_dict["source"] = DataSource(ohlcv_dict["source"])
                if "interval" in ohlcv_dict:
                    ohlcv_dict["interval"] = TimeInterval(ohlcv_dict["interval"])
                
                return OHLCVResponse(**ohlcv_dict)
        
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        except Exception as e:
            logger.error(f"Error retrieving latest OHLCV data: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error retrieving market data"
            )
    
    def get_ohlcv_data(self, filter_params: OHLCVFilter) -> List[OHLCVResponse]:
        """
        Get OHLCV data based on filter criteria.
        
        Args:
            filter_params: Filter parameters for OHLCV data
        
        Returns:
            List of OHLCV data points
        
        Raises:
            HTTPException: If instrument not found or other errors occur
        """
        try:
            # Resolve instrument ID from either ID or symbol
            resolved_id = self.resolve_instrument_id(
                filter_params.instrument_id, filter_params.symbol
            )
            if not resolved_id:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Instrument not found"
                )
            
            # Default time range if not specified
            end_timestamp = filter_params.end_timestamp or datetime.utcnow()
            
            # Default to 30 days of data if no start time provided
            if not filter_params.start_timestamp:
                start_timestamp = end_timestamp - timedelta(days=30)
            else:
                start_timestamp = filter_params.start_timestamp
            
            # Build query with filters
            query = """
            SELECT id, instrument_id, timestamp, source, source_timestamp,
                open, high, low, close, volume, interval, vwap,
                trades_count, open_interest, adjusted_close,
                is_anomaly, anomaly_reason, metadata,
                created_at, modified_at
            FROM ohlcv
            WHERE instrument_id = :instrument_id
            AND timestamp >= :start_timestamp
            AND timestamp <= :end_timestamp
            """
            
            params = {
                "instrument_id": resolved_id,
                "start_timestamp": start_timestamp,
                "end_timestamp": end_timestamp
            }
            
            # Add interval filter if specified
            if filter_params.interval:
                query += " AND interval = :interval"
                params["interval"] = filter_params.interval.value
            
            # Add source filter if specified
            if filter_params.source:
                query += " AND source = :source"
                params["source"] = filter_params.source.value
            
            # Add volume filter if specified
            if filter_params.min_volume is not None:
                query += " AND volume >= :min_volume"
                params["min_volume"] = filter_params.min_volume
            
            # Handle anomaly filtering
            if filter_params.exclude_anomalies:
                query += " AND (is_anomaly = FALSE OR is_anomaly IS NULL)"
            
            # Add sorting and pagination
            query += " ORDER BY timestamp ASC"
            query += " LIMIT :limit OFFSET :offset"
            params["limit"] = filter_params.limit
            params["offset"] = filter_params.offset
            
            # Execute query
            with self.timescale_db.session() as session:
                results = session.execute(query, params).fetchall()
                
                # Convert query results to OHLCVResponse objects
                ohlcv_data = []
                for row in results:
                    # Handle both dictionary-like objects and MagicMock objects
                    if hasattr(row, '_asdict'):
                        row_dict = row._asdict()
                    elif hasattr(row, '__dict__'):
                        row_dict = {k: getattr(row, k) for k in dir(row) 
                                if not k.startswith('_') and not callable(getattr(row, k))}
                    else:
                        # For MagicMock or other objects in tests
                        row_dict = {k: getattr(row, k) for k in [
                            'id', 'instrument_id', 'timestamp', 'source', 'source_timestamp',
                            'open', 'high', 'low', 'close', 'volume', 'interval', 'vwap',
                            'trades_count', 'open_interest', 'adjusted_close', 
                            'is_anomaly', 'anomaly_reason', 'metadata',
                            'created_at', 'modified_at'
                        ] if hasattr(row, k)}
                    
                    # Convert string values to proper enum types
                    if "source" in row_dict and isinstance(row_dict["source"], str):
                        row_dict["source"] = DataSource(row_dict["source"])
                    if "interval" in row_dict and isinstance(row_dict["interval"], str):
                        row_dict["interval"] = TimeInterval(row_dict["interval"])
                    
                    ohlcv_data.append(OHLCVResponse(**row_dict))
                
                return ohlcv_data
        
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        except Exception as e:
            logger.error(f"Error retrieving OHLCV data: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error retrieving market data"
            )
    
    #################################################
    # Data Aggregation Methods
    #################################################
    
    @cache(ttl=300, key_prefix="aggregated_ohlcv")
    def get_aggregated_ohlcv(
        self,
        instrument_id: Optional[UUID] = None,
        symbol: Optional[str] = None,
        start_timestamp: Optional[datetime] = None,
        end_timestamp: Optional[datetime] = None,
        source_interval: Optional[TimeInterval] = None,
        target_interval: TimeInterval = TimeInterval.DAY_1,
        exclude_anomalies: bool = True
        ) -> List[Dict[str, Any]]:
        """
        Get OHLCV data aggregated to a larger time interval.

        Args:
            instrument_id: Optional instrument UUID
            symbol: Optional trading symbol
            start_timestamp: Start time for data range
            end_timestamp: End time for data range
            source_interval: Source data interval (if None, use the smallest available)
            target_interval: Target aggregation interval
            exclude_anomalies: Whether to exclude anomalous data points

        Returns:
            List of aggregated OHLCV data points

        Raises:
            HTTPException: If instrument not found or other errors occur
        """
        try:
            # Resolve instrument ID from either ID or symbol
            resolved_id = self.resolve_instrument_id(instrument_id, symbol)
            if not resolved_id:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Instrument not found"
                )
            
            # Default time range if not specified
            end_ts = end_timestamp or datetime.utcnow()
            start_ts = start_timestamp or (end_ts - timedelta(days=30))
            
            # Map TimeInterval enum values to SQL interval strings
            interval_map = {
                TimeInterval.MINUTE_1: "1 minute",
                TimeInterval.MINUTE_5: "5 minutes",
                TimeInterval.MINUTE_15: "15 minutes",
                TimeInterval.MINUTE_30: "30 minutes",
                TimeInterval.HOUR_1: "1 hour",
                TimeInterval.HOUR_2: "2 hours",
                TimeInterval.HOUR_4: "4 hours",
                TimeInterval.HOUR_6: "6 hours",
                TimeInterval.HOUR_8: "8 hours",
                TimeInterval.HOUR_12: "12 hours",
                TimeInterval.DAY_1: "1 day",
                TimeInterval.DAY_3: "3 days",
                TimeInterval.WEEK_1: "1 week",
                TimeInterval.MONTH_1: "1 month",
            }
            
            # Ensure we have a valid SQL interval string for the target
            if target_interval not in interval_map:
                raise ValueError(f"Unsupported target interval: {target_interval}")
            
            sql_interval = interval_map[target_interval]
            
            # Use the TimescaleDB instance to leverage specialized time-bucket functionality
            timescale_db = self.timescale_db
            
            # Build the source interval filter
            source_interval_filter = ""
            source_params = {}
            if source_interval:
                source_interval_filter = "AND interval = :source_interval"
                source_params["source_interval"] = source_interval.value
            
            # Build the anomaly filter
            anomaly_filter = ""
            if exclude_anomalies:
                anomaly_filter = "AND (is_anomaly = FALSE OR is_anomaly IS NULL)"
            
            # Construct the query
            query = f"""
            SELECT 
                time_bucket('{sql_interval}', timestamp) AS bucket,
                instrument_id,
                FIRST(open, timestamp) AS open,
                MAX(high) AS high,
                MIN(low) AS low,
                LAST(close, timestamp) AS close,
                SUM(volume) AS volume,
                SUM(CASE WHEN vwap IS NOT NULL THEN volume * vwap ELSE 0 END) / 
                    NULLIF(SUM(CASE WHEN vwap IS NOT NULL THEN volume ELSE 0 END), 0) AS vwap,
                SUM(trades_count) AS trades_count,
                LAST(open_interest, timestamp) AS open_interest,
                LAST(adjusted_close, timestamp) AS adjusted_close,
                '{target_interval.value}' AS interval,
                COUNT(*) AS data_points
            FROM 
                ohlcv
            WHERE 
                instrument_id = :instrument_id
                AND timestamp >= :start_timestamp
                AND timestamp <= :end_timestamp
                {source_interval_filter}
                {anomaly_filter}
            GROUP BY 
                bucket, instrument_id
            ORDER BY 
                bucket ASC
            """
            
            # Combine all parameters
            params = {
                "instrument_id": resolved_id,
                "start_timestamp": start_ts,
                "end_timestamp": end_ts,
                **source_params
            }
            
            # Execute query via the database session
            with self.timescale_db.session() as session:
                results = session.execute(query, params).fetchall()
                
                # Process results
                aggregated_data = []
                for row in results:
                    # Handle both SQLAlchemy Row objects and MagicMock objects
                    if hasattr(row, '_asdict'):
                        data_point = row._asdict()
                    elif hasattr(row, '__dict__'):
                        # For regular objects or mock objects with __dict__
                        data_point = {}
                        for attr in ['bucket', 'instrument_id', 'open', 'high', 'low', 'close', 'volume', 
                                    'vwap', 'trades_count', 'open_interest', 'adjusted_close', 
                                    'interval', 'data_points']:
                            if hasattr(row, attr):
                                data_point[attr] = getattr(row, attr)
                    else:
                        # For MagicMock objects 
                        data_point = {}
                        for attr in dir(row):
                            if not attr.startswith('_') and not callable(getattr(row, attr)):
                                data_point[attr] = getattr(row, attr)
                    
                    # Format the timestamp - handle possible missing bucket key
                    if 'bucket' in data_point:
                        data_point["timestamp"] = data_point.pop("bucket")
                    elif hasattr(row, 'bucket'):
                        data_point["timestamp"] = getattr(row, 'bucket')
                    else:
                        # If bucket not found, skip this row
                        logger.warning("Missing 'bucket' field in aggregated result")
                        continue
                    
                    # Convert interval string to enum
                    if 'interval' in data_point and isinstance(data_point['interval'], str):
                        data_point["interval"] = TimeInterval(data_point["interval"])
                    
                    # Remove internal data_points count
                    data_points = data_point.pop("data_points") if "data_points" in data_point else 0
                    
                    # Add source information
                    data_point["source"] = DataSource.CALCULATED
                    
                    # Add metadata about aggregation
                    data_point["metadata"] = {
                        "aggregated": True,
                        "data_points": data_points,
                        "source_interval": source_interval.value if source_interval else "mixed"
                    }
                    
                    aggregated_data.append(data_point)
                
                return aggregated_data

        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        except Exception as e:
            logger.error(f"Error retrieving aggregated OHLCV data: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error aggregating market data"
            )
    
    #################################################
    # Statistics and Analysis Methods
    #################################################
    
    @cache(ttl=600, key_prefix="market_data_stats")
    def get_market_data_statistics(
        self, 
        instrument_id: Optional[UUID] = None,
        symbol: Optional[str] = None
    ) -> MarketDataStats:
        """
        Get statistics about available market data for an instrument.
        
        Args:
            instrument_id: Optional instrument UUID
            symbol: Optional trading symbol
        
        Returns:
            Statistics about the market data
        
        Raises:
            HTTPException: If instrument not found or other errors occur
        """
        try:
            # Resolve instrument ID and get instrument details
            resolved_id = self.resolve_instrument_id(instrument_id, symbol)
            if not resolved_id:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Instrument not found"
                )
            
            instrument = self.get_instrument_by_id(resolved_id)
            if not instrument:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Instrument not found"
                )
            
            with self.timescale_db.session() as session:
                # Query statistics about OHLCV data
                stats_result = session.execute(
                    """
                    SELECT 
                        MIN(timestamp) AS first_timestamp,
                        MAX(timestamp) AS last_timestamp,
                        COUNT(*) AS record_count,
                        ARRAY_AGG(DISTINCT source) AS data_sources,
                        AVG(CASE WHEN is_anomaly THEN 1.0 ELSE 0.0 END) AS anomaly_percentage
                    FROM ohlcv
                    WHERE instrument_id = :instrument_id
                    """,
                    {"instrument_id": resolved_id}
                ).fetchone()
                
                if not stats_result:
                    # Return empty statistics if no data found
                    return MarketDataStats(
                        instrument_id=resolved_id,
                        symbol=instrument.symbol,
                        first_timestamp=datetime.utcnow(),
                        last_timestamp=datetime.utcnow(),
                        record_count=0,
                        data_sources=[],
                        anomaly_percentage=0.0
                    )
                
                # Convert sources from strings to DataSource enum values
                sources = [DataSource(source) for source in stats_result.data_sources]
                
                return MarketDataStats(
                    instrument_id=resolved_id,
                    symbol=instrument.symbol,
                    first_timestamp=stats_result.first_timestamp,
                    last_timestamp=stats_result.last_timestamp,
                    record_count=stats_result.record_count,
                    data_sources=sources,
                    anomaly_percentage=stats_result.anomaly_percentage
                )
        
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        except Exception as e:
            logger.error(f"Error retrieving market data statistics: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error retrieving market data statistics"
            )
    
    def calculate_price_statistics(
        self,
        instrument_id: Optional[UUID] = None,
        symbol: Optional[str] = None,
        start_timestamp: Optional[datetime] = None,
        end_timestamp: Optional[datetime] = None,
        interval: TimeInterval = TimeInterval.DAY_1
    ) -> Dict[str, Any]:
        """
        Calculate statistics on price data for a given time range.
        
        Args:
            instrument_id: Optional instrument UUID
            symbol: Optional trading symbol
            start_timestamp: Start time for data range
            end_timestamp: End time for data range
            interval: Time interval for OHLCV data
        
        Returns:
            Dictionary of price statistics
        
        Raises:
            HTTPException: If instrument not found or other errors occur
        """
        try:
            # Resolve instrument ID
            resolved_id = self.resolve_instrument_id(instrument_id, symbol)
            if not resolved_id:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Instrument not found"
                )
            
            # Default time range if not specified
            end_ts = end_timestamp or datetime.utcnow()
            start_ts = start_timestamp or (end_ts - timedelta(days=30))
            
            with self.timescale_db.session() as session:
                # Query price statistics
                stats = session.execute(
                    """
                    SELECT 
                        AVG(close) AS avg_price,
                        MIN(low) AS min_price,
                        MAX(high) AS max_price,
                        STDDEV(close) AS std_dev_price,
                        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY close) AS median_price,
                        AVG(volume) AS avg_volume,
                        MAX(volume) AS max_volume,
                        SUM(volume) AS total_volume,
                        COUNT(*) AS data_points,
                        (MAX(high) - MIN(low)) / MIN(low) AS price_range_percent
                    FROM ohlcv
                    WHERE 
                        instrument_id = :instrument_id
                        AND timestamp >= :start_timestamp
                        AND timestamp <= :end_timestamp
                        AND interval = :interval
                        AND (is_anomaly = FALSE OR is_anomaly IS NULL)
                    """,
                    {
                        "instrument_id": resolved_id,
                        "start_timestamp": start_ts,
                        "end_timestamp": end_ts,
                        "interval": interval.value
                    }
                ).fetchone()
                
                if not stats or stats.data_points == 0:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="No data found for the specified parameters"
                    )
                
                # Calculate additional metrics if we have enough data points
                additional_metrics = {}
                if stats.data_points >= 2:
                    # Get first and last price for return calculation
                    price_data = session.execute(
                        """
                        SELECT 
                            (SELECT close FROM ohlcv 
                             WHERE instrument_id = :instrument_id
                             AND timestamp >= :start_timestamp
                             AND timestamp <= :end_timestamp
                             AND interval = :interval
                             AND (is_anomaly = FALSE OR is_anomaly IS NULL)
                             ORDER BY timestamp ASC LIMIT 1) AS first_price,
                            (SELECT close FROM ohlcv 
                             WHERE instrument_id = :instrument_id
                             AND timestamp >= :start_timestamp
                             AND timestamp <= :end_timestamp
                             AND interval = :interval
                             AND (is_anomaly = FALSE OR is_anomaly IS NULL)
                             ORDER BY timestamp DESC LIMIT 1) AS last_price
                        """,
                        {
                            "instrument_id": resolved_id,
                            "start_timestamp": start_ts,
                            "end_timestamp": end_ts,
                            "interval": interval.value
                        }
                    ).fetchone()
                    
                    if price_data and price_data.first_price and price_data.last_price:
                        # Calculate return metrics
                        first_price = price_data.first_price
                        last_price = price_data.last_price
                        
                        price_change = last_price - first_price
                        percent_change = (price_change / first_price) * 100 if first_price else 0
                        
                        additional_metrics = {
                            "first_price": first_price,
                            "last_price": last_price,
                            "price_change": price_change,
                            "percent_change": percent_change
                        }
                
                # Combine basic stats with additional metrics
                result = {
                    "avg_price": stats.avg_price,
                    "min_price": stats.min_price,
                    "max_price": stats.max_price,
                    "std_dev_price": stats.std_dev_price,
                    "median_price": stats.median_price,
                    "avg_volume": stats.avg_volume,
                    "max_volume": stats.max_volume,
                    "total_volume": stats.total_volume,
                    "price_range_percent": stats.price_range_percent * 100 if stats.price_range_percent else 0,
                    "data_points": stats.data_points,
                    **additional_metrics
                }
                
                return result
        
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        except Exception as e:
            logger.error(f"Error calculating price statistics: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error calculating market statistics"
            )
    
    #################################################
    # Multi-Symbol Methods
    #################################################
    
    def get_multi_symbol_ohlcv(
        self,
        symbols: List[str],
        start_timestamp: Optional[datetime] = None,
        end_timestamp: Optional[datetime] = None,
        interval: TimeInterval = TimeInterval.DAY_1,
        exclude_anomalies: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get OHLCV data for multiple symbols.
        
        Args:
            symbols: List of trading symbols
            start_timestamp: Start time for data range
            end_timestamp: End time for data range
            interval: Time interval for OHLCV data
            exclude_anomalies: Whether to exclude anomalous data points
        
        Returns:
            Dictionary mapping symbols to their OHLCV data
        
        Raises:
            HTTPException: If errors occur
        """
        if not symbols:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one symbol must be provided"
            )
        
        result = {}
        
        # Process symbols in batches for efficiency
        batch_size = 5
        for i in range(0, len(symbols), batch_size):
            symbol_batch = symbols[i:i+batch_size]
            
            # Process each symbol in the batch
            for symbol in symbol_batch:
                try:
                    # Get OHLCV data for this symbol
                    filter_params = OHLCVFilter(
                        symbol=symbol,
                        start_timestamp=start_timestamp,
                        end_timestamp=end_timestamp,
                        interval=interval,
                        exclude_anomalies=exclude_anomalies,
                        limit=1000  # Set a reasonable limit
                    )
                    
                    ohlcv_data = self.get_ohlcv_data(filter_params)
                    
                    # Convert OHLCVResponse objects to dictionaries
                    result[symbol] = [data.model_dump() for data in ohlcv_data]
                
                except HTTPException as http_ex:
                    # For multi-symbol requests, we don't want to fail completely
                    # if one symbol has an issue, so we store the error
                    if http_ex.status_code == status.HTTP_404_NOT_FOUND:
                        result[symbol] = {"error": "Instrument not found"}
                    else:
                        result[symbol] = {"error": http_ex.detail}
                
                except Exception as e:
                    logger.error(f"Error retrieving data for symbol {symbol}: {e}")
                    result[symbol] = {"error": "Internal error processing symbol"}
        
        return result
    
    def _calculate_correlation(self, series1: List[float], series2: List[float]) -> float:
        """
        Calculate Pearson correlation coefficient between two series.
        
        Args:
            series1: First data series
            series2: Second data series
            
        Returns:
            Correlation coefficient between -1 and 1
        """
        # Handle empty or single-element series
        if not series1 or not series2 or len(series1) < 2 or len(series2) < 2:
            return 0.0
            
        # Make sure series are the same length
        min_length = min(len(series1), len(series2))
        series1 = series1[:min_length]
        series2 = series2[:min_length]
        
        # Calculate means
        mean1 = sum(series1) / len(series1)
        mean2 = sum(series2) / len(series2)
        
        # Calculate variances and covariance
        var1 = sum((x - mean1) ** 2 for x in series1)
        var2 = sum((x - mean2) ** 2 for x in series2)
        
        # Avoid division by zero
        if var1 == 0 or var2 == 0:
            return 0.0
            
        covar = sum((series1[i] - mean1) * (series2[i] - mean2) for i in range(min_length))
        
        # Calculate correlation
        correlation = covar / ((var1 * var2) ** 0.5)
        
        # Ensure result is within valid range
        return max(min(correlation, 1.0), -1.0)

    def compare_instruments(
        self,
        symbols: List[str],
        start_timestamp: Optional[datetime] = None,
        end_timestamp: Optional[datetime] = None,
        interval: TimeInterval = TimeInterval.DAY_1
    ) -> Dict[str, Any]:
        """
        Compare price performance of multiple instruments.
        
        Args:
            symbols: List of trading symbols
            start_timestamp: Start time for data range
            end_timestamp: End time for data range
            interval: Time interval for OHLCV data
        
        Returns:
            Dictionary with comparison results
        
        Raises:
            HTTPException: If errors occur
        """
        if not symbols or len(symbols) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least two symbols must be provided for comparison"
            )
        
        # Default time range if not specified
        end_ts = end_timestamp or datetime.utcnow()
        start_ts = start_timestamp or (end_ts - timedelta(days=30))
        
        try:
            # Get price statistics for each symbol
            symbol_stats = {}
            for symbol in symbols:
                try:
                    stats = self.calculate_price_statistics(
                        symbol=symbol,
                        start_timestamp=start_ts,
                        end_timestamp=end_ts,
                        interval=interval
                    )
                    
                    symbol_stats[symbol] = stats
                except HTTPException:
                    symbol_stats[symbol] = {"error": "No data available"}
                except Exception as e:
                    logger.error(f"Error calculating statistics for {symbol}: {e}")
                    symbol_stats[symbol] = {"error": "Error processing data"}
            
            # Calculate correlation matrix if we have enough data
            correlation_matrix = {}
            symbols_with_data = [s for s in symbols if isinstance(symbol_stats[s], dict) and "error" not in symbol_stats[s]]
            
            if len(symbols_with_data) >= 2:
                # Get price series for correlation calculation
                price_series = {}
                for symbol in symbols_with_data:
                    try:
                        # Get OHLCV data for this symbol
                        filter_params = OHLCVFilter(
                            symbol=symbol,
                            start_timestamp=start_ts,
                            end_timestamp=end_ts,
                            interval=interval,
                            exclude_anomalies=True
                        )
                        
                        ohlcv_data = self.get_ohlcv_data(filter_params)
                        
                        # Extract just the timestamps and close prices
                        price_series[symbol] = {
                            data.timestamp.isoformat(): float(data.close)
                            for data in ohlcv_data
                        }
                    except Exception as e:
                        logger.error(f"Error getting price series for {symbol}: {e}")
                
                # Unified set of timestamps across all symbols
                all_timestamps = set()
                for symbol, prices in price_series.items():
                    all_timestamps.update(prices.keys())
                
                all_timestamps = sorted(all_timestamps)
                
                # Calculate correlations using normalized returns
                if len(all_timestamps) >= 3:  # Need at least 3 points for meaningful correlation
                    # Calculate returns for each symbol
                    returns = {}
                    for symbol in symbols_with_data:
                        if symbol in price_series:
                            symbol_returns = []
                            prev_price = None
                            for ts in all_timestamps:
                                price = price_series[symbol].get(ts)
                                if price is not None:
                                    if prev_price is not None:
                                        ret = (price / prev_price) - 1
                                        symbol_returns.append(ret)
                                    prev_price = price
                            
                            if symbol_returns:
                                returns[symbol] = symbol_returns
                    
                    # Calculate correlations
                    for symbol1 in returns:
                        correlation_matrix[symbol1] = {}
                        for symbol2 in returns:
                            if symbol1 == symbol2:
                                correlation_matrix[symbol1][symbol2] = 1.0
                            else:
                                # Calculate correlation coefficient
                                corr = self._calculate_correlation(returns[symbol1], returns[symbol2])
                                correlation_matrix[symbol1][symbol2] = corr
            
            # Calculate relative performance
            relative_performance = {}
            for symbol in symbols_with_data:
                if isinstance(symbol_stats[symbol], dict) and "percent_change" in symbol_stats[symbol]:
                    relative_performance[symbol] = symbol_stats[symbol]["percent_change"]
            
            # Find best and worst performers
            best_performer = max(relative_performance.items(), key=lambda x: x[1]) if relative_performance else None
            worst_performer = min(relative_performance.items(), key=lambda x: x[1]) if relative_performance else None
            
            # Combine all results
            return {
                "symbol_stats": symbol_stats,
                "correlation_matrix": correlation_matrix,
                "relative_performance": relative_performance,
                "best_performer": best_performer,
                "worst_performer": worst_performer,
                "time_period": {
                    "start": start_ts.isoformat(),
                    "end": end_ts.isoformat(),
                    "interval": interval.value
                }
            }
        
        except Exception as e:
            logger.error(f"Error comparing instruments: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error comparing instruments"
            )


    def detect_data_gaps(
        self,
        instrument_id: Optional[UUID] = None,
        symbol: Optional[str] = None,
        start_timestamp: Optional[datetime] = None,
        end_timestamp: Optional[datetime] = None,
        interval: TimeInterval = TimeInterval.DAY_1,
        min_gap_hours: float = 24.0
    ) -> List[Dict[str, Any]]:
        """
        Detect gaps in time-series data.
        
        Args:
            instrument_id: Optional instrument UUID
            symbol: Optional trading symbol
            start_timestamp: Start time for data range
            end_timestamp: End time for data range
            interval: Time interval of the data
            min_gap_hours: Minimum gap size to report (in hours)
            
        Returns:
            List of detected gaps with start/end times and gap size
        """
        try:
            # Resolve instrument ID from either ID or symbol
            resolved_id = self.resolve_instrument_id(instrument_id, symbol)
            if not resolved_id:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Instrument not found"
                )
            
            # Default time range if not specified
            end_ts = end_timestamp or datetime.utcnow()
            start_ts = start_timestamp or (end_ts - timedelta(days=30))
            
            with self.timescale_db.session() as session:
                # Get timestamps in order
                results = session.execute(
                    """
                    SELECT timestamp
                    FROM ohlcv
                    WHERE instrument_id = :instrument_id
                    AND interval = :interval
                    AND timestamp >= :start_timestamp
                    AND timestamp <= :end_timestamp
                    ORDER BY timestamp ASC
                    """,
                    {
                        "instrument_id": resolved_id,
                        "interval": interval.value,
                        "start_timestamp": start_ts,
                        "end_timestamp": end_ts
                    }
                ).fetchall()
                
                # Extract timestamps from results
                timestamps = [row.timestamp for row in results]
                
                if len(timestamps) < 2:
                    return []
                
                # Find gaps
                gaps = []
                for i in range(1, len(timestamps)):
                    gap_duration = timestamps[i] - timestamps[i-1]
                    gap_hours = gap_duration.total_seconds() / 3600
                    
                    # Map interval to expected hours
                    expected_hours = {
                        TimeInterval.MINUTE_1: 1/60,
                        TimeInterval.MINUTE_5: 5/60,
                        TimeInterval.MINUTE_15: 15/60,
                        TimeInterval.MINUTE_30: 30/60,
                        TimeInterval.HOUR_1: 1,
                        TimeInterval.HOUR_2: 2,
                        TimeInterval.HOUR_4: 4,
                        TimeInterval.HOUR_6: 6,
                        TimeInterval.HOUR_8: 8,
                        TimeInterval.HOUR_12: 12,
                        TimeInterval.DAY_1: 24,
                        TimeInterval.DAY_3: 72,
                        TimeInterval.WEEK_1: 168,
                        TimeInterval.MONTH_1: 720  # Approximation
                    }.get(interval, 24)
                    
                    # Only report gaps significantly larger than expected
                    if gap_hours > max(expected_hours * 1.5, min_gap_hours):
                        gaps.append({
                            "start": timestamps[i-1].isoformat(),
                            "end": timestamps[i].isoformat(),
                            "gap_hours": gap_hours,
                            "expected_hours": expected_hours,
                            "gap_ratio": gap_hours / expected_hours
                        })
                
                return gaps
                
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        except Exception as e:
            logger.error(f"Error detecting data gaps: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error detecting data gaps"
            )