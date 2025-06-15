"""
Backtesting Data Service for Trading Strategies Application.

This service handles historical market data loading, validation, and processing
for backtesting operations. It parses the local CSV data file and provides
clean, validated OHLCV data for strategy replay and performance analysis.

Key Features:
- Local CSV data loading from /Users/rikkawal/Downloads/Edata.csv
- Timestamp parsing and validation (DD/MM/YY H:MM format)
- OHLCV data integrity validation
- Market hours filtering (9:15 AM - 11:59 PM IST)
- Efficient data querying with pandas
- Data caching for performance optimization
"""

import logging
import pandas as pd
from datetime import datetime, time, timedelta
from typing import Optional, Dict, List, Any, Tuple
from pathlib import Path
import os

from app.core.error_handling import (
    OperationalError,
    ValidationError,
    DatabaseConnectionError
)

# Set up logging
logger = logging.getLogger(__name__)


class BacktestingDataService:
    """
    Service for managing historical market data for backtesting operations.
    
    This service provides a clean interface for loading, validating, and querying
    historical OHLCV data from the local CSV file. It handles data parsing,
    validation, and filtering to ensure high-quality data for backtesting.
    """
    
    def __init__(self, csv_file_path: str = "/Users/rikkawal/Downloads/Edata.csv", 
                 filter_market_hours: bool = False):
        """
        Initialize the backtesting data service.
        
        Args:
            csv_file_path: Path to the historical data CSV file
            filter_market_hours: Whether to filter data to traditional market hours
                                True = Filter to 9:15 AM - 3:30 PM IST (for stocks)
                                False = Use all 24/7 data (for crypto/futures)
        """
        self.csv_file_path = csv_file_path
        self._data_cache: Optional[pd.DataFrame] = None
        self._cache_timestamp: Optional[datetime] = None
        self.cache_ttl_minutes = 60  # Cache data for 60 minutes
        
        # Market session configuration - configurable based on data type
        self.filter_market_hours = filter_market_hours
        if filter_market_hours:
            self.market_open = time(9, 15)  # 9:15 AM IST
            self.market_close = time(15, 30)  # 3:30 PM IST
            logger.info("Market hours filtering enabled (9:15 AM - 3:30 PM IST)")
        else:
            self.market_open = None
            self.market_close = None
            logger.info("24/7 data mode - no market hours filtering")
        
        # Data validation thresholds
        self.max_price_change_percent = 20.0  # Maximum 20% price change per minute
        self.min_volume = 0  # Minimum volume threshold
        
        logger.info(f"BacktestingDataService initialized with data file: {csv_file_path}")
    
    def load_historical_data(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load historical data from CSV file with caching.
        
        Args:
            force_reload: Force reload data even if cached version is available
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
            
        Raises:
            OperationalError: If file cannot be loaded or parsed
        """
        try:
            # Check if we can use cached data
            if not force_reload and self._is_cache_valid():
                logger.info("Using cached historical data")
                return self._data_cache.copy()
            
            logger.info(f"Loading historical data from {self.csv_file_path}")
            
            # Verify file exists
            if not os.path.exists(self.csv_file_path):
                raise OperationalError(f"Historical data file not found: {self.csv_file_path}")
            
            # Load CSV data
            df = pd.read_csv(self.csv_file_path)
            logger.info(f"Loaded {len(df)} rows of raw data")
            
            # Parse and validate data
            df = self._parse_csv_data(df)
            df = self._validate_ohlcv_data(df)
            
            # Skip market hours filtering for 24/7 data
            if self.filter_market_hours and self.market_open and self.market_close:
                df = self._filter_market_hours(df)
            else:
                logger.info("Market hours filtering disabled - using 24/7 data")
            
            # Cache the processed data
            self._data_cache = df.copy()
            self._cache_timestamp = datetime.now()
            
            logger.info(f"Successfully processed {len(df)} rows of historical data")
            logger.info(f"Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            return df.copy()
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            raise OperationalError(f"Failed to load historical data: {str(e)}")
    
    def _parse_csv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse raw CSV data and convert to proper formats.
        
        Args:
            df: Raw DataFrame from CSV
            
        Returns:
            Parsed DataFrame with proper column types
        """
        try:
            # Create a copy to avoid modifying original
            parsed_df = df.copy()
            
            # Expected column mapping
            expected_columns = {
                'timestamp': 'timestamp',
                'open': 'open', 
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
                'last_updated_time': 'last_updated_time'
            }
            
            # Check if all required columns exist
            missing_columns = set(expected_columns.keys()) - set(parsed_df.columns)
            if missing_columns:
                raise ValidationError(f"Missing required columns: {missing_columns}")
            
            # Parse timestamp from "24/12/24 9:00" format to datetime
            parsed_df['timestamp'] = pd.to_datetime(
                parsed_df['timestamp'], 
                format='%d/%m/%y %H:%M',
                errors='coerce'
            )
            
            # Convert price columns to float
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                parsed_df[col] = pd.to_numeric(parsed_df[col], errors='coerce')
            
            # Convert volume to integer
            parsed_df['volume'] = pd.to_numeric(parsed_df['volume'], errors='coerce').astype('Int64')
            
            # Remove rows with any NaN values in critical columns
            critical_columns = ['timestamp'] + price_columns + ['volume']
            initial_count = len(parsed_df)
            parsed_df = parsed_df.dropna(subset=critical_columns)
            dropped_count = initial_count - len(parsed_df)
            
            if dropped_count > 0:
                logger.warning(f"Dropped {dropped_count} rows due to parsing errors")
            
            # Sort by timestamp
            parsed_df = parsed_df.sort_values('timestamp').reset_index(drop=True)
            
            return parsed_df
            
        except Exception as e:
            logger.error(f"Error parsing CSV data: {e}")
            raise ValidationError(f"Failed to parse CSV data: {str(e)}")
    
    def _validate_ohlcv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate OHLCV data integrity and remove invalid rows.
        
        Args:
            df: DataFrame with parsed data
            
        Returns:
            DataFrame with validated data
        """
        try:
            validated_df = df.copy()
            initial_count = len(validated_df)
            
            # OHLC relationship validation: high >= max(open, close) >= min(open, close) >= low
            valid_ohlc = (
                (validated_df['high'] >= validated_df[['open', 'close']].max(axis=1)) &
                (validated_df['low'] <= validated_df[['open', 'close']].min(axis=1)) &
                (validated_df['high'] >= validated_df['low'])
            )
            
            # Volume validation
            valid_volume = validated_df['volume'] >= self.min_volume
            
            # Price change validation (detect potential data errors)
            price_changes = validated_df['close'].pct_change().abs() * 100
            valid_price_change = (price_changes <= self.max_price_change_percent) | price_changes.isna()
            
            # Combine all validation rules
            valid_rows = valid_ohlc & valid_volume & valid_price_change
            
            # Filter to valid rows only
            validated_df = validated_df[valid_rows].reset_index(drop=True)
            
            removed_count = initial_count - len(validated_df)
            if removed_count > 0:
                logger.warning(f"Removed {removed_count} invalid rows during validation")
                logger.warning(f"Validation summary: OHLC errors: {(~valid_ohlc).sum()}, "
                             f"Volume errors: {(~valid_volume).sum()}, "
                             f"Price change errors: {(~valid_price_change).sum()}")
            
            return validated_df
            
        except Exception as e:
            logger.error(f"Error validating OHLCV data: {e}")
            raise ValidationError(f"Failed to validate OHLCV data: {str(e)}")
    
    def _filter_market_hours(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter data to include only market trading hours.
        
        NOTE: Currently disabled for 24/7 trading data (crypto/futures).
        Can be enabled by setting filter_market_hours=True and providing market_open/close times.
        
        Args:
            df: DataFrame with timestamp data
            
        Returns:
            DataFrame filtered to market hours (or unfiltered if disabled)
        """
        try:
            if not self.filter_market_hours or not self.market_open or not self.market_close:
                logger.info("Market hours filtering is disabled - returning all data")
                return df
            
            # Extract time component from timestamps
            market_hours_mask = (
                (df['timestamp'].dt.time >= self.market_open) &
                (df['timestamp'].dt.time <= self.market_close)
            )
            
            filtered_df = df[market_hours_mask].reset_index(drop=True)
            
            removed_count = len(df) - len(filtered_df)
            if removed_count > 0:
                logger.info(f"Filtered out {removed_count} rows outside market hours "
                          f"({self.market_open} - {self.market_close})")
            
            return filtered_df
            
        except Exception as e:
            logger.error(f"Error filtering market hours: {e}")
            raise ValidationError(f"Failed to filter market hours: {str(e)}")
    
    def get_data_for_period(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Get historical data for a specific time period.
        
        Args:
            start_date: Start of the period (inclusive)
            end_date: End of the period (inclusive)
            
        Returns:
            DataFrame with data for the specified period
            
        Raises:
            ValidationError: If date range is invalid
        """
        try:
            if start_date >= end_date:
                raise ValidationError("Start date must be before end date")
            
            # Load data if not already cached
            df = self.load_historical_data()
            
            # Filter by date range
            mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
            period_data = df[mask].reset_index(drop=True)
            
            logger.info(f"Retrieved {len(period_data)} rows for period {start_date} to {end_date}")
            
            if len(period_data) == 0:
                logger.warning(f"No data found for period {start_date} to {end_date}")
            
            return period_data
            
        except Exception as e:
            logger.error(f"Error retrieving data for period: {e}")
            raise OperationalError(f"Failed to get data for period: {str(e)}")
    
    def get_latest_data_point(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recent data point available.
        
        Returns:
            Dictionary with latest OHLCV data or None if no data
        """
        try:
            df = self.load_historical_data()
            
            if len(df) == 0:
                return None
            
            latest_row = df.iloc[-1]
            return {
                'timestamp': latest_row['timestamp'],
                'open': latest_row['open'],
                'high': latest_row['high'],
                'low': latest_row['low'],
                'close': latest_row['close'],
                'volume': latest_row['volume']
            }
            
        except Exception as e:
            logger.error(f"Error getting latest data point: {e}")
            return None
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics about the available historical data.
        
        Returns:
            Dictionary with data summary information
        """
        try:
            df = self.load_historical_data()
            
            if len(df) == 0:
                return {'status': 'no_data', 'message': 'No historical data available'}
            
            summary = {
                'status': 'ok',
                'total_records': len(df),
                'date_range': {
                    'start': df['timestamp'].min().isoformat(),
                    'end': df['timestamp'].max().isoformat(),
                    'days': (df['timestamp'].max() - df['timestamp'].min()).days
                },
                'price_range': {
                    'min_price': float(df[['open', 'high', 'low', 'close']].min().min()),
                    'max_price': float(df[['open', 'high', 'low', 'close']].max().max()),
                    'latest_close': float(df['close'].iloc[-1])
                },
                'volume_stats': {
                    'total_volume': int(df['volume'].sum()),
                    'avg_volume': float(df['volume'].mean()),
                    'max_volume': int(df['volume'].max())
                },
                'data_quality': {
                    'missing_values': int(df.isnull().sum().sum()),
                    'duplicate_timestamps': int(df['timestamp'].duplicated().sum())
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating data summary: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _is_cache_valid(self) -> bool:
        """
        Check if cached data is still valid.
        
        Returns:
            True if cache is valid, False otherwise
        """
        if self._data_cache is None or self._cache_timestamp is None:
            return False
        
        cache_age = datetime.now() - self._cache_timestamp
        return cache_age.total_seconds() < (self.cache_ttl_minutes * 60)
    
    def clear_cache(self) -> None:
        """Clear the data cache to force reload on next access."""
        self._data_cache = None
        self._cache_timestamp = None
        logger.info("Data cache cleared")
    
    def validate_file_accessibility(self) -> Dict[str, Any]:
        """
        Validate that the CSV file is accessible and readable.
        
        Returns:
            Dictionary with validation results
        """
        result = {
            'file_exists': False,
            'file_readable': False,
            'file_size_mb': 0,
            'estimated_rows': 0,
            'status': 'error',
            'message': ''
        }
        
        try:
            # Check if file exists
            file_path = Path(self.csv_file_path)
            result['file_exists'] = file_path.exists()
            
            if not result['file_exists']:
                result['message'] = f"File not found: {self.csv_file_path}"
                return result
            
            # Check if file is readable
            try:
                with open(self.csv_file_path, 'r') as f:
                    f.readline()  # Try to read first line
                result['file_readable'] = True
            except Exception as e:
                result['message'] = f"File not readable: {str(e)}"
                return result
            
            # Get file size
            file_size = file_path.stat().st_size
            result['file_size_mb'] = round(file_size / (1024 * 1024), 2)
            
            # Estimate number of rows (rough approximation)
            if file_size > 0:
                # Assume average row size of 50 bytes
                result['estimated_rows'] = file_size // 50
            
            result['status'] = 'ok'
            result['message'] = 'File is accessible and readable'
            
        except Exception as e:
            result['message'] = f"Error validating file: {str(e)}"
        
        return result


# Utility functions for backtesting data operations
def create_backtesting_data_service(csv_path: Optional[str] = None, 
                                   filter_market_hours: bool = False) -> BacktestingDataService:
    """
    Factory function to create a BacktestingDataService instance.
    
    Args:
        csv_path: Optional custom path to CSV file
        filter_market_hours: Whether to filter to traditional market hours
                            False = 24/7 data (crypto/futures) - DEFAULT
                            True = Stock market hours (9:15 AM - 3:30 PM IST)
        
    Returns:
        Configured BacktestingDataService instance
    """
    if csv_path is None:
        csv_path = "/Users/rikkawal/Downloads/Edata.csv"
    
    return BacktestingDataService(csv_path, filter_market_hours)


def validate_ohlcv_consistency(open_price: float, high_price: float, 
                              low_price: float, close_price: float) -> bool:
    """
    Validate OHLCV data consistency for a single data point.
    
    Args:
        open_price: Opening price
        high_price: High price
        low_price: Low price
        close_price: Closing price
        
    Returns:
        True if data is consistent, False otherwise
    """
    try:
        # High should be >= max(open, close)
        if high_price < max(open_price, close_price):
            return False
        
        # Low should be <= min(open, close)
        if low_price > min(open_price, close_price):
            return False
        
        # High should be >= low
        if high_price < low_price:
            return False
        
        return True
        
    except (TypeError, ValueError):
        return False