"""
Market Analysis API endpoints for the Trading Strategies Application.

This module provides RESTful endpoints for market analysis including:
- Multi-timeframe analysis with strict alignment requirements
- Market state classification and trading setup identification  
- Break of structure (BOS) detection
- Institutional behavior analysis
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Path, status, Body
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.core.database import get_postgres_db
from app.core.error_handling import (
    DatabaseConnectionError,
    OperationalError,
    ValidationError,
    AuthenticationError,
)
from app.services.strategy_engine import StrategyEngineService
from app.schemas.strategy import (
    TimeframeAnalysisResult,
    MarketStateAnalysis,
    TimeframeValueEnum,
)

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


# Dependency functions
def get_strategy_service(
    db: Session = Depends(get_postgres_db)
) -> StrategyEngineService:
    """
    Dependency to create StrategyEngineService instance.
    
    Args:
        db: Database session from dependency injection
        
    Returns:
        StrategyEngineService instance
    """
    try:
        return StrategyEngineService(db.session())
    except Exception as e:
        logger.error(f"Failed to create StrategyEngineService: {e}")
        raise DatabaseConnectionError("Unable to connect to database service")


def get_current_user_id() -> int:
    """
    Dependency to get current authenticated user ID.
    
    Returns:
        User ID
    """
    # TODO: Implement actual user extraction from auth middleware
    return 1  # Placeholder


# Market Analysis Endpoints

@router.post(
    "/timeframes/{strategy_id}",
    response_model=TimeframeAnalysisResult,
    summary="Analyze timeframes for strategy",
    description="Analyze market data across multiple timeframes according to strategy settings"
)
async def analyze_strategy_timeframes(
    strategy_id: int = Path(..., gt=0, description="Strategy ID"),
    market_data: Dict[str, Dict[str, Any]] = Body(..., description="Market data by timeframe"),
    service: StrategyEngineService = Depends(get_strategy_service),
    user_id: int = Depends(get_current_user_id)
) -> TimeframeAnalysisResult:
    """
    Analyze market data across multiple timeframes for a strategy.
    
    Implements the hierarchical timeframe structure with strict alignment requirements
    as defined in Rikk's trading principles.
    
    Args:
        strategy_id: ID of the strategy to analyze
        market_data: Dictionary of market data by timeframe (e.g., {"1h": {...}, "15m": {...}})
        service: Strategy service instance
        user_id: Current authenticated user ID
        
    Returns:
        Comprehensive timeframe analysis with alignment scores
        
    Raises:
        HTTPException: If strategy not found or access denied
        ValidationError: If market data is invalid
    """
    try:
        logger.info(f"Analyzing timeframes for strategy {strategy_id}, user {user_id}")
        
        # First verify strategy ownership
        strategy = service.get_strategy(strategy_id)
        if strategy.user_id != user_id:
            logger.warning(f"User {user_id} attempted to analyze strategy {strategy_id} owned by {strategy.user_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You can only analyze your own strategies"
            )
        
        # Convert string keys to TimeframeValue enums
        converted_market_data = {}
        for tf_str, data in market_data.items():
            try:
                tf_enum = TimeframeValueEnum(tf_str)
                converted_market_data[tf_enum] = data
            except ValueError:
                logger.error(f"Invalid timeframe: {tf_str}")
                raise ValidationError(f"Invalid timeframe: {tf_str}. Valid values: {[tf.value for tf in TimeframeValueEnum]}")
        
        # Perform timeframe analysis using service
        analysis_result = service.analyze_timeframes(strategy_id, converted_market_data)
        
        logger.info(f"Successfully analyzed timeframes for strategy {strategy_id}. Aligned: {analysis_result.aligned}")
        return analysis_result
        
    except ValueError as e:
        if "not found" in str(e).lower():
            logger.error(f"Strategy {strategy_id} not found: {e}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Strategy with ID {strategy_id} not found"
            )
        else:
            logger.error(f"Validation error analyzing timeframes for strategy {strategy_id}: {e}")
            raise ValidationError(str(e))
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"Error analyzing timeframes for strategy {strategy_id}: {e}")
        raise OperationalError(f"Failed to analyze timeframes: {str(e)}")


@router.post(
    "/market-state/{strategy_id}",
    response_model=MarketStateAnalysis,
    summary="Analyze market state for strategy",
    description="Analyze current market state for strategy execution including trend quality and institutional behavior"
)
async def analyze_strategy_market_state(
    strategy_id: int = Path(..., gt=0, description="Strategy ID"),
    market_data: Dict[str, Dict[str, Any]] = Body(..., description="Market data by timeframe"),
    service: StrategyEngineService = Depends(get_strategy_service),
    user_id: int = Depends(get_current_user_id)
) -> MarketStateAnalysis:
    """
    Analyze current market state for strategy execution.
    
    Implements key requirements including:
    - Railroad vs creeper move detection
    - Detection of institutional behavior
    - Price action vs indicator divergence
    - Break of structure (BOS) identification
    
    Args:
        strategy_id: ID of the strategy
        market_data: Dictionary of market data by timeframe
        service: Strategy service instance
        user_id: Current authenticated user ID
        
    Returns:
        Detailed market state analysis
        
    Raises:
        HTTPException: If strategy not found or access denied
        ValidationError: If market data is invalid
    """
    try:
        logger.info(f"Analyzing market state for strategy {strategy_id}, user {user_id}")
        
        # First verify strategy ownership
        strategy = service.get_strategy(strategy_id)
        if strategy.user_id != user_id:
            logger.warning(f"User {user_id} attempted to analyze strategy {strategy_id} owned by {strategy.user_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You can only analyze your own strategies"
            )
        
        # Convert string keys to TimeframeValue enums
        converted_market_data = {}
        for tf_str, data in market_data.items():
            try:
                tf_enum = TimeframeValueEnum(tf_str)
                converted_market_data[tf_enum] = data
            except ValueError:
                logger.error(f"Invalid timeframe: {tf_str}")
                raise ValidationError(f"Invalid timeframe: {tf_str}. Valid values: {[tf.value for tf in TimeframeValueEnum]}")
        
        # Perform market state analysis using service
        market_state = service.analyze_market_state(strategy_id, converted_market_data)
        
        logger.info(f"Successfully analyzed market state for strategy {strategy_id}. State: {market_state.market_state}")
        return market_state
        
    except ValueError as e:
        if "not found" in str(e).lower():
            logger.error(f"Strategy {strategy_id} not found: {e}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Strategy with ID {strategy_id} not found"
            )
        else:
            logger.error(f"Validation error analyzing market state for strategy {strategy_id}: {e}")
            raise ValidationError(str(e))
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"Error analyzing market state for strategy {strategy_id}: {e}")
        raise OperationalError(f"Failed to analyze market state: {str(e)}")


@router.post(
    "/combined-analysis/{strategy_id}",
    summary="Combined market analysis",
    description="Perform both timeframe and market state analysis in a single request"
)
async def combined_market_analysis(
    strategy_id: int = Path(..., gt=0, description="Strategy ID"),
    market_data: Dict[str, Dict[str, Any]] = Body(..., description="Market data by timeframe"),
    service: StrategyEngineService = Depends(get_strategy_service),
    user_id: int = Depends(get_current_user_id)
) -> Dict[str, Any]:
    """
    Perform both timeframe analysis and market state analysis in a single request.
    
    This is a convenience endpoint that combines both analyses, useful for
    getting a complete picture of market conditions for a strategy.
    
    Args:
        strategy_id: ID of the strategy
        market_data: Dictionary of market data by timeframe
        service: Strategy service instance
        user_id: Current authenticated user ID
        
    Returns:
        Combined analysis results including both timeframe and market state analysis
        
    Raises:
        HTTPException: If strategy not found or access denied
        ValidationError: If market data is invalid
    """
    try:
        logger.info(f"Performing combined analysis for strategy {strategy_id}, user {user_id}")
        
        # First verify strategy ownership
        strategy = service.get_strategy(strategy_id)
        if strategy.user_id != user_id:
            logger.warning(f"User {user_id} attempted to analyze strategy {strategy_id} owned by {strategy.user_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You can only analyze your own strategies"
            )
        
        # Convert string keys to TimeframeValue enums
        converted_market_data = {}
        for tf_str, data in market_data.items():
            try:
                tf_enum = TimeframeValueEnum(tf_str)
                converted_market_data[tf_enum] = data
            except ValueError:
                logger.error(f"Invalid timeframe: {tf_str}")
                raise ValidationError(f"Invalid timeframe: {tf_str}. Valid values: {[tf.value for tf in TimeframeValueEnum]}")
        
        # Perform both analyses
        timeframe_analysis = service.analyze_timeframes(strategy_id, converted_market_data)
        market_state_analysis = service.analyze_market_state(strategy_id, converted_market_data)
        
        # Combine results
        combined_result = {
            "strategy_id": strategy_id,
            "timeframe_analysis": {
                "aligned": timeframe_analysis.aligned,
                "alignment_score": timeframe_analysis.alignment_score,
                "timeframe_results": timeframe_analysis.timeframe_results,
                "primary_direction": timeframe_analysis.primary_direction,
                "require_all_aligned": timeframe_analysis.require_all_aligned,
                "min_alignment_score": timeframe_analysis.min_alignment_score,
                "sufficient_alignment": timeframe_analysis.sufficient_alignment
            },
            "market_state_analysis": {
                "market_state": market_state_analysis.market_state,
                "trend_phase": market_state_analysis.trend_phase,
                "is_railroad_trend": market_state_analysis.is_railroad_trend,
                "is_creeper_move": market_state_analysis.is_creeper_move,
                "has_two_day_trend": market_state_analysis.has_two_day_trend,
                "trend_direction": market_state_analysis.trend_direction,
                "price_indicator_divergence": market_state_analysis.price_indicator_divergence,
                "price_struggling_near_ma": market_state_analysis.price_struggling_near_ma,
                "institutional_fight_in_progress": market_state_analysis.institutional_fight_in_progress,
                "accumulation_detected": market_state_analysis.accumulation_detected,
                "bos_detected": market_state_analysis.bos_detected
            },
            "overall_assessment": {
                "ready_for_trading": (
                    timeframe_analysis.aligned and 
                    not market_state_analysis.institutional_fight_in_progress and
                    not market_state_analysis.is_creeper_move and
                    market_state_analysis.trend_phase == "middle"
                ),
                "confidence_level": min(timeframe_analysis.alignment_score, 0.9) if timeframe_analysis.aligned else 0.0
            }
        }
        
        logger.info(f"Successfully completed combined analysis for strategy {strategy_id}")
        return combined_result
        
    except ValueError as e:
        if "not found" in str(e).lower():
            logger.error(f"Strategy {strategy_id} not found: {e}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Strategy with ID {strategy_id} not found"
            )
        else:
            logger.error(f"Validation error in combined analysis for strategy {strategy_id}: {e}")
            raise ValidationError(str(e))
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"Error in combined analysis for strategy {strategy_id}: {e}")
        raise OperationalError(f"Failed to perform combined analysis: {str(e)}")


# Technical Analysis Helpers

@router.get(
    "/trend-direction",
    summary="Determine trend direction",
    description="Helper endpoint to determine trend direction from market data"
)
async def determine_trend_direction(
    close_prices: str = Query(..., description="Comma-separated close prices"),
    ma_primary: int = Query(21, description="Primary MA period"),
    ma_secondary: int = Query(200, description="Secondary MA period"),
    service: StrategyEngineService = Depends(get_strategy_service)
) -> Dict[str, Any]:
    """
    Helper endpoint to determine trend direction from price and MA data.
    
    Args:
        close_prices: Comma-separated string of close prices
        ma_primary: Primary MA period (default 21)
        ma_secondary: Secondary MA period (default 200)
        service: Strategy service instance
        
    Returns:
        Trend direction analysis
    """
    try:
        # Parse close prices
        prices = [float(p.strip()) for p in close_prices.split(",")]
        
        if len(prices) < max(ma_primary, ma_secondary):
            raise ValidationError(f"Need at least {max(ma_primary, ma_secondary)} price points")
        
        # Calculate moving averages (simplified)
        def simple_ma(data, period):
            if len(data) < period:
                return [None] * len(data)
            mas = []
            for i in range(len(data)):
                if i < period - 1:
                    mas.append(None)
                else:
                    mas.append(sum(data[i-period+1:i+1]) / period)
            return mas
        
        ma_primary_values = simple_ma(prices, ma_primary)
        ma_secondary_values = simple_ma(prices, ma_secondary)
        
        # Create timeframe data
        timeframe_data = {
            "close": prices,
            f"ma{ma_primary}": ma_primary_values,
            f"ma{ma_secondary}": ma_secondary_values
        }
        
        # Use service method to determine trend
        trend_direction = service._determine_trend_direction(timeframe_data, ma_primary, ma_secondary)
        
        return {
            "trend_direction": trend_direction,
            "latest_price": prices[-1],
            "latest_ma_primary": ma_primary_values[-1],
            "latest_ma_secondary": ma_secondary_values[-1],
            "price_above_ma_primary": prices[-1] > ma_primary_values[-1] if ma_primary_values[-1] else None,
            "price_above_ma_secondary": prices[-1] > ma_secondary_values[-1] if ma_secondary_values[-1] else None
        }
        
    except ValueError as e:
        logger.error(f"Validation error determining trend direction: {e}")
        raise ValidationError(str(e))
    except Exception as e:
        logger.error(f"Error determining trend direction: {e}")
        raise OperationalError(f"Failed to determine trend direction: {str(e)}")


@router.get(
    "/market-characterization",
    summary="Characterize market movement",
    description="Helper endpoint to characterize market as railroad vs creeper vs other"
)
async def characterize_market_movement(
    close_prices: str = Query(..., description="Comma-separated close prices"),
    high_prices: str = Query(..., description="Comma-separated high prices"),
    low_prices: str = Query(..., description="Comma-separated low prices"),
    service: StrategyEngineService = Depends(get_strategy_service)
) -> Dict[str, Any]:
    """
    Helper endpoint to characterize market movement type.
    
    Args:
        close_prices: Comma-separated string of close prices
        high_prices: Comma-separated string of high prices  
        low_prices: Comma-separated string of low prices
        service: Strategy service instance
        
    Returns:
        Market movement characterization
    """
    try:
        # Parse prices
        closes = [float(p.strip()) for p in close_prices.split(",")]
        highs = [float(p.strip()) for p in high_prices.split(",")]
        lows = [float(p.strip()) for p in low_prices.split(",")]
        
        if not (len(closes) == len(highs) == len(lows)):
            raise ValidationError("All price arrays must have the same length")
        
        if len(closes) < 10:
            raise ValidationError("Need at least 10 price points for reliable analysis")
        
        # Create timeframe data
        timeframe_data = {
            "close": closes,
            "high": highs,
            "low": lows
        }
        
        # Detect movement patterns
        is_railroad = service._detect_railroad_trend(timeframe_data, threshold=0.8)
        is_creeper = service._detect_creeper_move(timeframe_data)
        
        # Calculate additional metrics
        avg_range = sum((highs[i] - lows[i]) / highs[i] for i in range(len(closes))) / len(closes)
        price_change = (closes[-1] - closes[0]) / closes[0]
        
        return {
            "movement_type": "railroad" if is_railroad else "creeper" if is_creeper else "normal",
            "is_railroad_trend": is_railroad,
            "is_creeper_move": is_creeper,
            "average_daily_range_pct": avg_range * 100,
            "total_price_change_pct": price_change * 100,
            "bars_analyzed": len(closes),
            "recommendation": (
                "Good for trending strategies" if is_railroad else
                "Avoid trending strategies" if is_creeper else
                "Suitable for most strategies"
            )
        }
        
    except ValueError as e:
        logger.error(f"Validation error characterizing market movement: {e}")
        raise ValidationError(str(e))
    except Exception as e:
        logger.error(f"Error characterizing market movement: {e}")
        raise OperationalError(f"Failed to characterize market movement: {str(e)}")


# Health check endpoint for this module
@router.get(
    "/health",
    summary="Market analysis service health",
    description="Check health of market analysis service"
)
async def market_analysis_health() -> Dict[str, str]:
    """
    Health check endpoint for market analysis service.
    
    Returns:
        Health status of the service
    """
    return {
        "status": "healthy",
        "service": "market_analysis",
        "message": "Market analysis service is operational"
    }