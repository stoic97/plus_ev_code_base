"""
Market Data API Endpoints

This module provides FastAPI endpoints for retrieving and analyzing market data.
Endpoints are organized by data type and functionality, with appropriate error handling
and input validation.
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from app.api.dependencies import CommonQueryParams
from app.core.security import User, get_current_active_user, has_role
from app.schemas.market_data import (
    AssetClass, DataSource, TimeInterval, 
    OHLCVResponse, OHLCVFilter, MarketDataStats,
    InstrumentResponse, MarketDataFilter
)
from app.services.market_data import MarketDataService, get_market_data_service

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/market-data", tags=["market-data"])


# Dependency to get market data service
def get_service() -> MarketDataService:
    """Dependency to get market data service instance."""
    return get_market_data_service()


#################################################
# OHLCV Data Endpoints
#################################################

@router.get("/ohlcv", response_model=List[OHLCVResponse])
async def get_ohlcv(
    request: Request,
    symbol: Optional[str] = None,
    instrument_id: Optional[UUID] = None,
    interval: Optional[TimeInterval] = TimeInterval.DAY_1,
    start_timestamp: Optional[datetime] = None,
    end_timestamp: Optional[datetime] = None,
    source: Optional[DataSource] = None,
    min_volume: Optional[Decimal] = None,
    exclude_anomalies: bool = True,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    service: MarketDataService = Depends(get_service),
    current_user: User = Depends(get_current_active_user)
) -> List[OHLCVResponse]:
    """
    Get OHLCV (Open, High, Low, Close, Volume) data for a financial instrument.
    
    Parameters can be used to filter the results by time range, interval, and other criteria.
    
    - **symbol**: Trading symbol (e.g., "AAPL" for Apple)
    - **instrument_id**: UUID of the instrument (alternative to symbol)
    - **interval**: Time interval for OHLCV data (e.g., "1d" for daily data)
    - **start_timestamp**: Start of time range
    - **end_timestamp**: End of time range
    - **source**: Data source filter
    - **min_volume**: Minimum volume filter
    - **exclude_anomalies**: Whether to exclude anomalous data points
    - **limit**: Maximum number of data points to return
    - **offset**: Offset for pagination
    """
    if not symbol and not instrument_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either symbol or instrument_id must be provided"
        )
    
    try:
        # Create filter parameters
        filter_params = OHLCVFilter(
            symbol=symbol,
            instrument_id=instrument_id,
            interval=interval,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            source=source,
            min_volume=min_volume,
            exclude_anomalies=exclude_anomalies,
            limit=limit,
            offset=offset
        )
        
        # Get OHLCV data using service
        results = service.get_ohlcv_data(filter_params)
        return results
    
    except HTTPException as e:
        # Re-raise HTTP exceptions from service
        raise e
    except Exception as e:
        logger.error(f"Error retrieving OHLCV data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving market data"
        )


@router.get("/ohlcv/latest", response_model=OHLCVResponse)
async def get_latest_ohlcv(
    request: Request,
    symbol: Optional[str] = None,
    instrument_id: Optional[UUID] = None,
    interval: TimeInterval = TimeInterval.DAY_1,
    service: MarketDataService = Depends(get_service),
    current_user: User = Depends(get_current_active_user)
) -> OHLCVResponse:
    """
    Get the latest (most recent) OHLCV data point for a financial instrument.
    
    - **symbol**: Trading symbol (e.g., "AAPL" for Apple)
    - **instrument_id**: UUID of the instrument (alternative to symbol)
    - **interval**: Time interval for OHLCV data (e.g., "1d" for daily data)
    """
    if not symbol and not instrument_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either symbol or instrument_id must be provided"
        )
    
    try:
        # Get latest OHLCV data using service
        result = service.get_latest_ohlcv(
            instrument_id=instrument_id,
            symbol=symbol,
            interval=interval
        )
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No data found for the specified parameters"
            )
        
        return result
    
    except HTTPException as e:
        # Re-raise HTTP exceptions from service
        raise e
    except Exception as e:
        logger.error(f"Error retrieving latest OHLCV data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving market data"
        )


@router.get("/ohlcv/aggregated", response_model=List[Dict[str, Any]])
async def get_aggregated_ohlcv(
    request: Request,
    symbol: Optional[str] = None,
    instrument_id: Optional[UUID] = None,
    start_timestamp: Optional[datetime] = None,
    end_timestamp: Optional[datetime] = None,
    source_interval: Optional[TimeInterval] = None,
    target_interval: TimeInterval = TimeInterval.DAY_1,
    exclude_anomalies: bool = True,
    service: MarketDataService = Depends(get_service),
    current_user: User = Depends(get_current_active_user)
) -> List[Dict[str, Any]]:
    """
    Get OHLCV data aggregated to a larger time interval.
    
    This endpoint allows aggregating data from smaller intervals to larger ones
    (e.g., from 1-hour to 1-day bars).
    
    - **symbol**: Trading symbol (e.g., "AAPL" for Apple)
    - **instrument_id**: UUID of the instrument (alternative to symbol)
    - **start_timestamp**: Start of time range
    - **end_timestamp**: End of time range
    - **source_interval**: Source data interval (if None, use the smallest available)
    - **target_interval**: Target aggregation interval
    - **exclude_anomalies**: Whether to exclude anomalous data points
    """
    if not symbol and not instrument_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either symbol or instrument_id must be provided"
        )
    
    try:
        # Get aggregated OHLCV data using service
        results = service.get_aggregated_ohlcv(
            instrument_id=instrument_id,
            symbol=symbol,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            source_interval=source_interval,
            target_interval=target_interval,
            exclude_anomalies=exclude_anomalies
        )
        
        if not results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No data found for the specified parameters"
            )
        
        return results
    
    except HTTPException as e:
        # Re-raise HTTP exceptions from service
        raise e
    except Exception as e:
        logger.error(f"Error retrieving aggregated OHLCV data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error aggregating market data"
        )


#################################################
# Market Data Statistics Endpoints
#################################################

@router.get("/statistics", response_model=MarketDataStats)
async def get_market_data_statistics(
    request: Request,
    symbol: Optional[str] = None,
    instrument_id: Optional[UUID] = None,
    service: MarketDataService = Depends(get_service),
    current_user: User = Depends(get_current_active_user)
) -> MarketDataStats:
    """
    Get statistics about available market data for an instrument.
    
    Returns information about the data range, sources, and quality.
    
    - **symbol**: Trading symbol (e.g., "AAPL" for Apple)
    - **instrument_id**: UUID of the instrument (alternative to symbol)
    """
    if not symbol and not instrument_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either symbol or instrument_id must be provided"
        )
    
    try:
        # Get market data statistics using service
        result = service.get_market_data_statistics(
            instrument_id=instrument_id,
            symbol=symbol
        )
        
        return result
    
    except HTTPException as e:
        # Re-raise HTTP exceptions from service
        raise e
    except Exception as e:
        logger.error(f"Error retrieving market data statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving market data statistics"
        )


@router.get("/price-stats", response_model=Dict[str, Any])
async def calculate_price_statistics(
    request: Request,
    symbol: Optional[str] = None,
    instrument_id: Optional[UUID] = None,
    start_timestamp: Optional[datetime] = None,
    end_timestamp: Optional[datetime] = None,
    interval: TimeInterval = TimeInterval.DAY_1,
    service: MarketDataService = Depends(get_service),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Calculate statistics on price data for a given time range.
    
    Returns metrics such as average price, volatility, and return percentage.
    
    - **symbol**: Trading symbol (e.g., "AAPL" for Apple)
    - **instrument_id**: UUID of the instrument (alternative to symbol)
    - **start_timestamp**: Start of time range
    - **end_timestamp**: End of time range
    - **interval**: Time interval for OHLCV data (e.g., "1d" for daily data)
    """
    if not symbol and not instrument_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either symbol or instrument_id must be provided"
        )
    
    try:
        # Calculate price statistics using service
        result = service.calculate_price_statistics(
            instrument_id=instrument_id,
            symbol=symbol,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            interval=interval
        )
        
        return result
    
    except HTTPException as e:
        # Re-raise HTTP exceptions from service
        raise e
    except Exception as e:
        logger.error(f"Error calculating price statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error calculating market statistics"
        )


#################################################
# Multi-Symbol Endpoints
#################################################

@router.get("/multi-symbol", response_model=Dict[str, Any])
async def get_multi_symbol_ohlcv(
    request: Request,
    symbols: List[str] = Query(..., min_length=1, max_length=20),
    start_timestamp: Optional[datetime] = None,
    end_timestamp: Optional[datetime] = None,
    interval: TimeInterval = TimeInterval.DAY_1,
    exclude_anomalies: bool = True,
    service: MarketDataService = Depends(get_service),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get OHLCV data for multiple symbols in a single request.
    
    Returns a dictionary mapping symbols to their OHLCV data.
    
    - **symbols**: List of trading symbols (e.g., ["AAPL", "MSFT"])
    - **start_timestamp**: Start of time range
    - **end_timestamp**: End of time range
    - **interval**: Time interval for OHLCV data (e.g., "1d" for daily data)
    - **exclude_anomalies**: Whether to exclude anomalous data points
    """
    try:
        # Get multi-symbol OHLCV data using service
        results = service.get_multi_symbol_ohlcv(
            symbols=symbols,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            interval=interval,
            exclude_anomalies=exclude_anomalies
        )
        
        return results
    
    except HTTPException as e:
        # Re-raise HTTP exceptions from service
        raise e
    except Exception as e:
        logger.error(f"Error retrieving multi-symbol data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving multi-symbol data"
        )


@router.get("/compare", response_model=Dict[str, Any])
async def compare_instruments(
    request: Request,
    symbols: List[str] = Query(..., min_length=2, max_length=10),
    start_timestamp: Optional[datetime] = None,
    end_timestamp: Optional[datetime] = None,
    interval: TimeInterval = TimeInterval.DAY_1,
    service: MarketDataService = Depends(get_service),
    current_user: User = Depends(has_role(["admin", "analyst", "trader", "risk_manager"]))
) -> Dict[str, Any]:
    """
    Compare price performance of multiple instruments.
    
    Returns correlation matrix, relative performance, and other comparison metrics.
    
    - **symbols**: List of trading symbols to compare (e.g., ["AAPL", "MSFT"])
    - **start_timestamp**: Start of time range
    - **end_timestamp**: End of time range
    - **interval**: Time interval for OHLCV data (e.g., "1d" for daily data)
    
    Note: This endpoint requires analyst or higher role permissions.
    """
    try:
        # Compare instruments using service
        results = service.compare_instruments(
            symbols=symbols,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            interval=interval
        )
        
        return results
    
    except HTTPException as e:
        # Re-raise HTTP exceptions from service
        raise e
    except Exception as e:
        logger.error(f"Error comparing instruments: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error comparing instruments"
        )


#################################################
# Data Quality Endpoints
#################################################

@router.get("/quality/gaps", response_model=List[Dict[str, Any]])
async def detect_data_gaps(
    request: Request,
    symbol: Optional[str] = None,
    instrument_id: Optional[UUID] = None,
    start_timestamp: Optional[datetime] = None,
    end_timestamp: Optional[datetime] = None,
    interval: TimeInterval = TimeInterval.DAY_1,
    service: MarketDataService = Depends(get_service),
    current_user: User = Depends(has_role(["admin", "analyst", "risk_manager"]))
) -> List[Dict[str, Any]]:
    """
    Detect gaps in time-series data.
    
    Returns a list of gaps with start/end times and estimated missing points.
    
    - **symbol**: Trading symbol (e.g., "AAPL" for Apple)
    - **instrument_id**: UUID of the instrument (alternative to symbol)
    - **start_timestamp**: Start of time range
    - **end_timestamp**: End of time range
    - **interval**: Time interval for OHLCV data (e.g., "1d" for daily data)
    
    Note: This endpoint requires analyst or higher role permissions.
    """
    if not symbol and not instrument_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either symbol or instrument_id must be provided"
        )
    
    try:
        # Detect data gaps using service
        gaps = service.detect_data_gaps(
            instrument_id=instrument_id,
            symbol=symbol,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            interval=interval
        )
        
        return gaps
    
    except HTTPException as e:
        # Re-raise HTTP exceptions from service
        raise e
    except Exception as e:
        logger.error(f"Error detecting data gaps: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error analyzing data quality"
        )