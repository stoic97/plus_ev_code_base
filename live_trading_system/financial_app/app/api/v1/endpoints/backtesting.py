"""
Backtesting API Endpoints for Trading Strategies Application

This module provides RESTful endpoints for comprehensive backtesting operations,
integrating with existing infrastructure to provide time travel backtesting capabilities.
Follows existing API patterns and reuses paper trading infrastructure.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, Path, status, BackgroundTasks
from fastapi.responses import JSONResponse

from app.core.database import get_postgres_db
from app.core.error_handling import (
    OperationalError,
    ValidationError,
    DatabaseConnectionError,
    AuthenticationError,
)
from app.services.backtesting_engine import BacktestingEngine
from app.services.backtesting_data_service import BacktestingDataService
from app.services.backtest_performance_calculator import BacktestPerformanceCalculator
from app.services.analytics_service import AnalyticsService
from app.services.strategy_engine import StrategyEngineService

# Import schemas
from app.schemas.backtest import (
    BacktestRunRequest,
    BacktestConfigCreate,
    BacktestConfigBase,
    BacktestResultResponse,
    BacktestDetailedResultResponse,
    BacktestStatusRequest,
    BacktestListRequest,
    BacktestComparisonRequest,
    BacktestComparisonResponse,
    BacktestHealthResponse,
    BacktestErrorResponse,
    BacktestStatusEnum,
    BacktestMetricsBase,
    BacktestTradeResponse,
)

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


# Dependency functions
def get_backtesting_engine(
    db = Depends(get_postgres_db)
) -> BacktestingEngine:
    """
    Dependency to create BacktestingEngine instance.
    
    Args:
        db: Database session from dependency injection
        
    Returns:
        BacktestingEngine instance
    """
    try:
        with db.session() as session:
            # Create services
            data_service = BacktestingDataService()
            strategy_service = StrategyEngineService(session)
            
            # Create engine with dependencies
            return BacktestingEngine(
                data_service=data_service,
                strategy_service=strategy_service
            )
    except Exception as e:
        logger.error(f"Failed to create BacktestingEngine: {e}")
        raise OperationalError(f"Unable to initialize backtesting engine: {str(e)}")


def get_backtest_performance_calculator(
    db = Depends(get_postgres_db)
) -> BacktestPerformanceCalculator:
    """
    Dependency to create BacktestPerformanceCalculator instance.
    
    Args:
        db: Database session from dependency injection
        
    Returns:
        BacktestPerformanceCalculator instance
    """
    try:
        with db.session() as session:
            # Reuse existing analytics service
            initial_capital = 1000000.0  # 10 Lakh INR default
            analytics_service = AnalyticsService(session, initial_capital)
            
            return BacktestPerformanceCalculator(analytics_service)
    except Exception as e:
        logger.error(f"Failed to create BacktestPerformanceCalculator: {e}")
        raise OperationalError(f"Unable to initialize performance calculator: {str(e)}")


def get_strategy_service(
    db = Depends(get_postgres_db)
) -> StrategyEngineService:
    """
    Dependency to create StrategyEngineService instance.
    
    Args:
        db: Database session from dependency injection
        
    Returns:
        StrategyEngineService instance
    """
    try:
        with db.session() as session:
            return StrategyEngineService(session)
    except Exception as e:
        logger.error(f"Failed to create StrategyEngineService: {e}")
        raise DatabaseConnectionError("Unable to connect to database service")


def get_current_user_id() -> int:
    """
    Dependency to get current authenticated user ID.
    
    This is a placeholder - replace with actual auth logic from your AuthMiddleware.
    For now, returns a default user ID.
    
    Returns:
        User ID
    """
    # TODO: Implement actual user extraction from auth middleware
    # This should extract user_id from request.state or JWT token
    return 1  # Placeholder


# Helper function to check strategy ownership
async def check_strategy_ownership(
    strategy_id: int,
    user_id: int,
    service: StrategyEngineService
) -> None:
    """
    Check if user owns the strategy, raise 403 if not.
    
    Args:
        strategy_id: ID of the strategy
        user_id: ID of the user
        service: Strategy service instance
        
    Raises:
        HTTPException: If strategy not found or access denied
    """
    try:
        strategy = service.get_strategy(strategy_id)
        if strategy.user_id != user_id:
            logger.warning(f"User {user_id} attempted to access strategy {strategy_id} owned by {strategy.user_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You can only access your own strategies"
            )
    except ValueError as e:
        logger.error(f"Strategy {strategy_id} not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy with ID {strategy_id} not found"
        )


# Helper function to convert backtest request to config
def backtest_request_to_config(request: BacktestRunRequest):
    """
    Convert BacktestRunRequest to a config object that the engine can use.
    
    Args:
        request: BacktestRunRequest from API
        
    Returns:
        Configuration object for the backtesting engine
    """
    # Create a proper BacktestConfigBase instance
    config = BacktestConfigBase(
        strategy_id=request.strategy_id,
        name=getattr(request, 'name', f"Backtest {request.strategy_id}"),
        description=getattr(request, 'description', None),
        start_date=request.start_date,
        end_date=request.end_date,
        initial_capital=request.initial_capital,
        max_position_size=request.max_position_size,
        data_source=getattr(request, 'data_source', 'csv'),
        timeframe=getattr(request, 'timeframe', '1m'),
        commission_per_trade=request.commission_per_trade,
        slippage_bps=request.slippage_bps,
        warm_up_days=getattr(request, 'warm_up_days', 30),
        benchmark_symbol=getattr(request, 'benchmark_symbol', 'NIFTY50')
    )
    
    return config


# Helper function to create proper config object for response
def create_config_for_response(result) -> BacktestConfigBase:
    """
    Create a proper BacktestConfigBase for the response.
    """
    if hasattr(result.config, 'strategy_id'):
        # If it's already a proper config object, return it
        return result.config
    
    # Otherwise create a proper config object with required fields
    return BacktestConfigBase(
        strategy_id=getattr(result.config, 'strategy_id', result.strategy_id),
        name=getattr(result.config, 'name', f"Backtest {result.strategy_id}"),
        start_date=getattr(result.config, 'start_date', datetime.utcnow() - timedelta(days=90)),
        end_date=getattr(result.config, 'end_date', datetime.utcnow() - timedelta(days=1)),
        initial_capital=getattr(result.config, 'initial_capital', 1000000.0),
        max_position_size=getattr(result.config, 'max_position_size', 0.1),
        commission_per_trade=getattr(result.config, 'commission_per_trade', 20.0),
        slippage_bps=getattr(result.config, 'slippage_bps', 2.0)
    )


# Helper function to convert backtest result to response schema
def create_backtest_result_response(result) -> BacktestResultResponse:
    """
    Convert backtest result to BacktestResultResponse schema.
    
    Args:
        result: Backtest result from engine
        
    Returns:
        BacktestResultResponse instance
    """
    # Create proper config object
    config = create_config_for_response(result)
    
    return BacktestResultResponse(
        backtest_id=result.backtest_id,
        strategy_id=result.strategy_id,
        status=result.status,
        start_time=result.start_time,
        end_time=result.end_time,
        config=config,
        metrics=result.metrics,
        trade_count=result.trade_count,
        error_message=result.error_message,
        warnings=result.warnings if hasattr(result, 'warnings') else [],
        duration_seconds=result.duration_seconds if hasattr(result, 'duration_seconds') else None,
        is_complete=result.is_complete if hasattr(result, 'is_complete') else result.status == BacktestStatusEnum.COMPLETED
    )


def create_detailed_backtest_result_response(result) -> BacktestDetailedResultResponse:
    """
    Convert backtest result to BacktestDetailedResultResponse schema.
    
    Args:
        result: Backtest result from engine
        
    Returns:
        BacktestDetailedResultResponse instance
    """
    # Create proper config object
    config = create_config_for_response(result)
    
    # Convert trades to proper BacktestTradeResponse objects if needed
    trades = []
    if hasattr(result, 'trades') and result.trades:
        for trade in result.trades:
            if hasattr(trade, 'trade_id'):
                # If it's already a proper trade object, convert it to response format
                trades.append(trade)
            else:
                # Skip invalid trade objects for now
                continue
    
    return BacktestDetailedResultResponse(
        backtest_id=result.backtest_id,
        strategy_id=result.strategy_id,
        status=result.status,
        start_time=result.start_time,
        end_time=result.end_time,
        config=config,
        metrics=result.metrics,
        trade_count=result.trade_count,
        error_message=result.error_message,
        warnings=result.warnings if hasattr(result, 'warnings') else [],
        duration_seconds=result.duration_seconds if hasattr(result, 'duration_seconds') else None,
        is_complete=result.is_complete if hasattr(result, 'is_complete') else result.status == BacktestStatusEnum.COMPLETED,
        trades=trades,
        equity_curve=result.equity_curve if hasattr(result, 'equity_curve') else [],
        monthly_returns=result.monthly_returns if hasattr(result, 'monthly_returns') else [],
        drawdown_periods=result.drawdown_periods if hasattr(result, 'drawdown_periods') else [],
        trade_analysis=result.trade_analysis if hasattr(result, 'trade_analysis') else {}
    )


# ==============================
# Core Backtesting Endpoints
# ==============================

@router.post(
    "/backtests/run",
    response_model=BacktestResultResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Run a new backtest",
    description="Execute a comprehensive backtest for a trading strategy using historical data"
)
async def run_backtest(
    backtest_request: BacktestRunRequest,
    background_tasks: BackgroundTasks,
    engine: BacktestingEngine = Depends(get_backtesting_engine),
    strategy_service: StrategyEngineService = Depends(get_strategy_service),
    user_id: int = Depends(get_current_user_id)
) -> BacktestResultResponse:
    """
    Run a comprehensive backtest for a trading strategy.
    
    This endpoint initiates a backtest using the time travel approach, feeding
    historical CSV data to the existing paper trading infrastructure for
    realistic execution simulation.
    """
    try:
        # Validate strategy ownership
        await check_strategy_ownership(
            backtest_request.strategy_id, 
            user_id, 
            strategy_service
        )
        
        # Convert request to config
        config = backtest_request_to_config(backtest_request)
        
        if backtest_request.run_immediately:
            # Execute backtest synchronously
            logger.info(f"Starting immediate backtest for strategy {config.strategy_id}")
            result = engine.run_backtest(config)
            
            logger.info(f"Backtest completed: {result.backtest_id}")
            return create_backtest_result_response(result)
        else:
            # Queue backtest for background execution
            background_tasks.add_task(engine.run_backtest, config)
            
            # Return initial response
            result = engine.create_pending_backtest(config)
            logger.info(f"Backtest queued: {result.backtest_id}")
            return create_backtest_result_response(result)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to run backtest: {e}")
        raise OperationalError(f"Unable to execute backtest: {str(e)}")


@router.get(
    "/backtests/{backtest_id}",
    response_model=BacktestResultResponse,
    summary="Get backtest results",
    description="Retrieve results and status for a specific backtest"
)
async def get_backtest_result(
    backtest_id: str = Path(..., description="Unique backtest identifier"),
    engine: BacktestingEngine = Depends(get_backtesting_engine),
    user_id: int = Depends(get_current_user_id)
) -> BacktestResultResponse:
    """
    Get backtest results and status.
    
    Returns comprehensive backtest results including performance metrics,
    execution status, and any error information.
    """
    try:
        # Get backtest result
        result = engine.get_backtest_result(backtest_id)
        
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Backtest {backtest_id} not found"
            )
        
        # TODO: Add ownership check based on strategy_id
        # For now, allowing access to all results
        
        logger.info(f"Retrieved backtest result: {backtest_id}")
        return create_backtest_result_response(result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get backtest result {backtest_id}: {e}")
        raise OperationalError(f"Unable to retrieve backtest result: {str(e)}")


@router.get(
    "/backtests/{backtest_id}/detailed",
    response_model=BacktestDetailedResultResponse,
    summary="Get detailed backtest results",
    description="Retrieve comprehensive backtest results with trade data and analysis"
)
async def get_detailed_backtest_result(
    backtest_id: str = Path(..., description="Unique backtest identifier"),
    engine: BacktestingEngine = Depends(get_backtesting_engine),
    calculator: BacktestPerformanceCalculator = Depends(get_backtest_performance_calculator),
    user_id: int = Depends(get_current_user_id)
) -> BacktestDetailedResultResponse:
    """
    Get detailed backtest results with comprehensive trade data and analysis.
    
    Returns all backtest information including individual trades, equity curve,
    drawdown periods, and detailed performance analysis.
    """
    try:
        # Get backtest result
        result = engine.get_backtest_result(backtest_id)
        
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Backtest {backtest_id} not found"
            )
        
        # Get additional analysis if backtest completed
        if result.status == BacktestStatusEnum.COMPLETED:
            # Get detailed analysis
            detailed_analysis = calculator.get_detailed_analysis(
                result.trades,
                result.config,
                result.equity_curve
            )
            
            # Merge with result
            result.trade_analysis = detailed_analysis.get('trade_analysis', {})
            result.monthly_returns = detailed_analysis.get('monthly_returns')
            result.drawdown_periods = detailed_analysis.get('drawdown_periods', [])
        
        logger.info(f"Retrieved detailed backtest result: {backtest_id}")
        return create_detailed_backtest_result_response(result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get detailed backtest result {backtest_id}: {e}")
        raise OperationalError(f"Unable to retrieve detailed backtest result: {str(e)}")


@router.get(
    "/backtests",
    response_model=List[BacktestResultResponse],
    summary="List backtests",
    description="List backtests with optional filtering"
)
async def list_backtests(
    strategy_id: Optional[int] = Query(None, description="Filter by strategy ID", gt=0),
    status: Optional[BacktestStatusEnum] = Query(None, description="Filter by status"),
    start_date_from: Optional[datetime] = Query(None, description="Filter by start date (from)"),
    start_date_to: Optional[datetime] = Query(None, description="Filter by start date (to)"),
    limit: int = Query(10, description="Number of results to return", ge=1, le=100),
    offset: int = Query(0, description="Number of results to skip", ge=0),
    engine: BacktestingEngine = Depends(get_backtesting_engine),
    strategy_service: StrategyEngineService = Depends(get_strategy_service),
    user_id: int = Depends(get_current_user_id)
) -> List[BacktestResultResponse]:
    """
    List backtests with optional filtering.
    
    Returns a paginated list of backtests, optionally filtered by strategy,
    status, or date range. Only returns backtests for strategies owned by the user.
    """
    try:
        # Get user's strategy IDs for security
        user_strategies = strategy_service.list_strategies(user_id)
        user_strategy_ids = [s.id for s in user_strategies]
        
        # Apply strategy filter
        if strategy_id:
            if strategy_id not in user_strategy_ids:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied: Strategy not owned by user"
                )
            filter_strategy_ids = [strategy_id]
        else:
            filter_strategy_ids = user_strategy_ids
        
        # Get backtests
        results = engine.list_backtests(
            strategy_ids=filter_strategy_ids,
            status=status,
            start_date_from=start_date_from,
            start_date_to=start_date_to,
            limit=limit,
            offset=offset
        )
        
        logger.info(f"Listed {len(results)} backtests for user {user_id}")
        return [create_backtest_result_response(r) for r in results]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list backtests: {e}")
        raise OperationalError(f"Unable to list backtests: {str(e)}")


# ==============================
# Analysis and Comparison Endpoints
# ==============================

@router.post(
    "/backtests/compare",
    response_model=BacktestComparisonResponse,
    summary="Compare multiple backtests",
    description="Compare performance metrics across multiple backtests"
)
async def compare_backtests(
    comparison_request: BacktestComparisonRequest,
    engine: BacktestingEngine = Depends(get_backtesting_engine),
    calculator: BacktestPerformanceCalculator = Depends(get_backtest_performance_calculator),
    user_id: int = Depends(get_current_user_id)
) -> BacktestComparisonResponse:
    """
    Compare performance metrics across multiple backtests.
    
    Provides side-by-side comparison of key metrics, identifies best performing
    backtests for each metric, and calculates summary statistics.
    """
    try:
        # Get all backtest results
        results = []
        for backtest_id in comparison_request.backtest_ids:
            result = engine.get_backtest_result(backtest_id)
            if result is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Backtest {backtest_id} not found"
                )
            results.append(result)
        
        # TODO: Add ownership validation for all backtests
        
        # Generate comparison
        comparison = calculator.compare_backtests(
            results,
            comparison_request.metrics_to_compare
        )
        
        logger.info(f"Compared {len(results)} backtests")
        return comparison
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to compare backtests: {e}")
        raise OperationalError(f"Unable to compare backtests: {str(e)}")


@router.get(
    "/backtests/{backtest_id}/vs-live",
    response_model=Dict[str, Any],
    summary="Compare backtest vs live performance",
    description="Compare backtest results against live trading performance for the same strategy"
)
async def compare_backtest_vs_live(
    backtest_id: str = Path(..., description="Backtest ID"),
    live_start_date: Optional[datetime] = Query(None, description="Live trading start date"),
    live_end_date: Optional[datetime] = Query(None, description="Live trading end date"),
    engine: BacktestingEngine = Depends(get_backtesting_engine),
    calculator: BacktestPerformanceCalculator = Depends(get_backtest_performance_calculator),
    user_id: int = Depends(get_current_user_id)
) -> Dict[str, Any]:
    """
    Compare backtest results against live trading performance.
    
    Provides insights into how well backtest predictions match actual
    live trading results for the same strategy and time period.
    """
    try:
        # Get backtest result
        backtest_result = engine.get_backtest_result(backtest_id)
        if backtest_result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Backtest {backtest_id} not found"
            )
        
        # Get live performance for comparison
        live_comparison = calculator.compare_with_live_performance(
            backtest_result,
            live_start_date,
            live_end_date
        )
        
        logger.info(f"Compared backtest {backtest_id} vs live performance")
        return live_comparison
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to compare backtest vs live: {e}")
        raise OperationalError(f"Unable to compare backtest vs live performance: {str(e)}")


# ==============================
# Control and Management Endpoints
# ==============================

@router.delete(
    "/backtests/{backtest_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Cancel or delete backtest",
    description="Cancel a running backtest or delete completed backtest results"
)
async def cancel_backtest(
    backtest_id: str = Path(..., description="Backtest ID to cancel"),
    engine: BacktestingEngine = Depends(get_backtesting_engine),
    user_id: int = Depends(get_current_user_id)
) -> None:
    """
    Cancel a running backtest or delete completed backtest results.
    
    If the backtest is running, it will be cancelled. If completed,
    the results will be deleted from storage.
    """
    try:
        # Get backtest result to check ownership
        result = engine.get_backtest_result(backtest_id)
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Backtest {backtest_id} not found"
            )
        
        # TODO: Add ownership validation
        
        # Cancel or delete
        success = engine.cancel_or_delete_backtest(backtest_id)
        if not success:
            raise OperationalError("Failed to cancel/delete backtest")
        
        logger.info(f"Cancelled/deleted backtest: {backtest_id}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel backtest {backtest_id}: {e}")
        raise OperationalError(f"Unable to cancel backtest: {str(e)}")


@router.get(
    "/backtests/health",
    response_model=BacktestHealthResponse,
    summary="Get backtesting service health",
    description="Check the health and status of the backtesting service"
)
async def get_backtest_health(
    engine: BacktestingEngine = Depends(get_backtesting_engine)
) -> BacktestHealthResponse:
    """
    Get backtesting service health and status.
    
    Returns information about service status, database connectivity,
    data file accessibility, and current system load.
    """
    try:
        health_info = engine.get_health_status()
        
        return BacktestHealthResponse(
            status="healthy" if health_info["all_systems_ok"] else "degraded",
            service="backtesting",
            timestamp=datetime.utcnow(),
            version="1.0.0",  # TODO: Get from config
            database_connected=health_info["database_connected"],
            csv_data_accessible=health_info["csv_data_accessible"],
            last_backtest_time=health_info.get("last_backtest_time"),
            running_backtests=health_info["running_backtests"]
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return BacktestHealthResponse(
            status="unhealthy",
            service="backtesting",
            timestamp=datetime.utcnow(),
            version="1.0.0",
            database_connected=False,
            csv_data_accessible=False,
            last_backtest_time=None,
            running_backtests=0
        )


# ==============================
# Utility Endpoints
# ==============================

@router.get(
    "/backtests/data/validate",
    response_model=Dict[str, Any],
    summary="Validate backtesting data",
    description="Validate CSV data file accessibility and format"
)
async def validate_backtest_data(
    engine: BacktestingEngine = Depends(get_backtesting_engine)
) -> Dict[str, Any]:
    """
    Validate backtesting data file accessibility and format.
    
    Checks if the Edata.csv file is accessible, properly formatted,
    and contains the expected columns and data ranges.
    """
    try:
        validation_result = engine.validate_data_source()
        
        logger.info("Data validation completed")
        return validation_result
        
    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        raise OperationalError(f"Unable to validate data source: {str(e)}")


@router.get(
    "/backtests/strategies/{strategy_id}/available-periods",
    response_model=Dict[str, Any],
    summary="Get available backtest periods",
    description="Get available date ranges for backtesting based on data availability"
)
async def get_available_backtest_periods(
    strategy_id: int = Path(..., description="Strategy ID", gt=0),
    engine: BacktestingEngine = Depends(get_backtesting_engine),
    strategy_service: StrategyEngineService = Depends(get_strategy_service),
    user_id: int = Depends(get_current_user_id)
) -> Dict[str, Any]:
    """
    Get available date ranges for backtesting.
    
    Returns the available data ranges based on CSV data availability
    and strategy creation date, helping users select appropriate
    backtest periods.
    """
    try:
        # Validate strategy ownership
        await check_strategy_ownership(strategy_id, user_id, strategy_service)
        
        # Get available periods
        periods = engine.get_available_backtest_periods(strategy_id)
        
        logger.info(f"Retrieved available periods for strategy {strategy_id}")
        return periods
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get available periods for strategy {strategy_id}: {e}")
        raise OperationalError(f"Unable to get available backtest periods: {str(e)}")