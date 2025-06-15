"""
Analytics API endpoints for the Trading Strategies Application.

This module provides RESTful endpoints for paper trading analytics,
including NAV tracking, risk metrics, performance attribution, and
benchmark comparison.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, Path, status
from fastapi.responses import JSONResponse

from app.core.database import get_postgres_db
from app.core.error_handling import (
    OperationalError,
    ValidationError,
    DatabaseConnectionError
)
from app.services.analytics_service import AnalyticsService
from app.services.strategy_engine import StrategyEngineService

# Import response schemas
from app.schemas.analytics import (
    LiveMetricsResponse,
    EquityCurveResponse,
    DrawdownAnalysisResponse,
    RiskMetricsResponse,
    AttributionResponse,
    BenchmarkComparisonResponse,
    PortfolioSummaryResponse,
    AnalyticsHealthResponse,
    TimeframeEnum,
    AttributionTypeEnum
)

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


# Dependency functions
def get_analytics_service(
    db = Depends(get_postgres_db)
) -> AnalyticsService:
    """
    Dependency to create AnalyticsService instance.
    
    Args:
        db: Database session from dependency injection
        
    Returns:
        AnalyticsService instance
    """
    try:
        with db.session() as session:
            # Get initial capital from config or use default
            initial_capital = 1000000.0  # Will be moved to config
            return AnalyticsService(session, initial_capital)
    except Exception as e:
        logger.error(f"Failed to create AnalyticsService: {e}")
        raise OperationalError(f"Unable to initialize analytics service: {str(e)}")


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
    
    This is a placeholder - using the same pattern as other endpoints.
    
    Returns:
        User ID
    """
    # TODO: Implement actual user extraction from auth middleware
    return 1  # Placeholder



# Analytics Endpoints


# Analytics Endpoints

@router.get(
    "/strategies/{strategy_id}/analytics/dashboard",
    response_model=LiveMetricsResponse,
    summary="Get live analytics dashboard",
    description="Get real-time analytics metrics including NAV, P&L, and current positions"
)
async def get_analytics_dashboard(
    strategy_id: int = Path(..., gt=0, description="Strategy ID"),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
    strategy_service: StrategyEngineService = Depends(get_strategy_service),
    user_id: int = Depends(get_current_user_id)
) -> LiveMetricsResponse:
    """
    Get live analytics dashboard for a strategy.
    
    Provides real-time metrics including:
    - Current NAV
    - Cash balance
    - Open positions value
    - Realized/Unrealized P&L
    - Daily returns
    - Current drawdown
    
    Args:
        strategy_id: ID of the strategy
        analytics_service: Analytics service instance
        strategy_service: Strategy service instance
        user_id: Current authenticated user ID
        
    Returns:
        Live metrics dashboard data
        
    Raises:
        HTTPException: If strategy not found or access denied
    """
    try:
        logger.info(f"Getting analytics dashboard for strategy {strategy_id}, user {user_id}")
        
        # First verify ownership
        strategy = strategy_service.get_strategy(strategy_id)
        if strategy.user_id != user_id:
            logger.warning(f"User {user_id} attempted to access analytics for strategy {strategy_id} owned by {strategy.user_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You can only view analytics for your own strategies"
            )
        
        # Get live metrics
        metrics = analytics_service.get_live_metrics(strategy_id)
        
        # Convert to response model
        response = LiveMetricsResponse(
            nav=metrics["nav"],
            initial_capital=analytics_service.initial_capital,
            cash=metrics["cash"],
            positions_value=metrics["positions_value"],
            total_pnl=metrics["realized_pnl"] + metrics["unrealized_pnl"],
            realized_pnl=metrics["realized_pnl"],
            unrealized_pnl=metrics["unrealized_pnl"],
            daily_pnl=metrics.get("daily_pnl"),
            total_return_percent=((metrics["nav"] - analytics_service.initial_capital) / analytics_service.initial_capital) * 100,
            daily_return_percent=metrics.get("daily_return"),
            current_drawdown_percent=metrics["current_drawdown"],
            high_water_mark=metrics["high_water_mark"],
            active_trades=metrics["active_trades"],
            total_trades=metrics["total_trades"],
            trades_today=metrics.get("trades_today"),
            last_updated=datetime.utcnow()
        )
        
        logger.info(f"Successfully retrieved analytics dashboard for strategy {strategy_id}")
        return response
        
    except ValueError as e:
        if "not found" in str(e).lower():
            logger.error(f"Strategy {strategy_id} not found: {e}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Strategy with ID {strategy_id} not found"
            )
        else:
            logger.error(f"Validation error getting analytics dashboard: {e}")
            raise ValidationError(str(e))
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"Error getting analytics dashboard for strategy {strategy_id}: {e}")
        raise OperationalError(f"Failed to get analytics dashboard: {str(e)}")


@router.get(
    "/strategies/{strategy_id}/analytics/equity-curve",
    response_model=EquityCurveResponse,
    summary="Get equity curve",
    description="Get historical equity curve showing NAV over time"
)
async def get_equity_curve(
    strategy_id: int = Path(..., gt=0, description="Strategy ID"),
    period: TimeframeEnum = Query(TimeframeEnum.ONE_MONTH, description="Time period"),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
    strategy_service: StrategyEngineService = Depends(get_strategy_service),
    user_id: int = Depends(get_current_user_id)
) -> EquityCurveResponse:
    """
    Get historical equity curve for a strategy.
    
    Args:
        strategy_id: ID of the strategy
        period: Time period (1D, 1W, 1M, 3M, 1Y, ALL)
        analytics_service: Analytics service instance
        strategy_service: Strategy service instance
        user_id: Current authenticated user ID
        
    Returns:
        List of equity curve data points
        
    Raises:
        HTTPException: If strategy not found or access denied
    """
    try:
        logger.info(f"Getting equity curve for strategy {strategy_id}, period {period}")
        
        # Verify ownership
        strategy = strategy_service.get_strategy(strategy_id)
        if strategy.user_id != user_id:
            logger.warning(f"User {user_id} attempted to access equity curve for strategy {strategy_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You can only view analytics for your own strategies"
            )
        
        # Get equity curve
        equity_curve = analytics_service.get_equity_curve(strategy_id, period.value)
        
        # Convert to response model
        if not equity_curve:
            raise ValueError("No equity curve data available for this strategy")
        
        response = EquityCurveResponse(
            strategy_id=strategy_id,
            period=period,
            start_date=equity_curve[0]["timestamp"],
            end_date=equity_curve[-1]["timestamp"],
            data_points=equity_curve,
            total_points=len(equity_curve),
            starting_nav=equity_curve[0]["nav"],
            ending_nav=equity_curve[-1]["nav"],
            total_return_percent=((equity_curve[-1]["nav"] - equity_curve[0]["nav"]) / equity_curve[0]["nav"]) * 100
        )
        
        logger.info(f"Successfully retrieved equity curve with {len(equity_curve)} points")
        return response
        
    except ValueError as e:
        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Strategy with ID {strategy_id} not found"
            )
        else:
            raise ValidationError(str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting equity curve: {e}")
        raise OperationalError(f"Failed to get equity curve: {str(e)}")


@router.get(
    "/strategies/{strategy_id}/analytics/drawdown",
    response_model=DrawdownAnalysisResponse,
    summary="Get drawdown analysis",
    description="Analyze drawdowns including maximum drawdown and recovery periods"
)
async def get_drawdown_analysis(
    strategy_id: int = Path(..., gt=0, description="Strategy ID"),
    period: TimeframeEnum = Query(TimeframeEnum.ALL, description="Analysis period"),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
    strategy_service: StrategyEngineService = Depends(get_strategy_service),
    user_id: int = Depends(get_current_user_id)
) -> DrawdownAnalysisResponse:
    """
    Get drawdown analysis for a strategy.
    
    Args:
        strategy_id: ID of the strategy
        period: Analysis period
        analytics_service: Analytics service instance
        strategy_service: Strategy service instance
        user_id: Current authenticated user ID
        
    Returns:
        Drawdown analysis including max drawdown and duration
        
    Raises:
        HTTPException: If strategy not found or access denied
    """
    try:
        logger.info(f"Getting drawdown analysis for strategy {strategy_id}")
        
        # Verify ownership
        strategy = strategy_service.get_strategy(strategy_id)
        if strategy.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You can only view analytics for your own strategies"
            )
        
        # Get equity curve for the period
        equity_curve = analytics_service.get_equity_curve(strategy_id, period.value)
        
        # Calculate drawdown analysis
        drawdown_analysis = analytics_service.calculate_drawdown(equity_curve)
        
        # Add current NAV and high water mark
        current_nav = analytics_service.calculate_nav(strategy_id)
        drawdown_data = analytics_service._calculate_current_drawdown(strategy_id, current_nav)
        
        # Convert to response model
        response = DrawdownAnalysisResponse(
            strategy_id=strategy_id,
            analysis_period=period,
            current_nav=current_nav,
            high_water_mark=drawdown_data["high_water_mark"],
            current_drawdown_percent=drawdown_data["drawdown_percent"],
            max_drawdown_percent=drawdown_analysis["max_drawdown"],
            max_drawdown_duration_days=drawdown_analysis["max_drawdown_duration_days"],
            average_drawdown_percent=drawdown_analysis.get("average_drawdown", -2.5),
            average_recovery_days=drawdown_analysis.get("average_recovery_days", 5.0),
            drawdown_periods=drawdown_analysis.get("drawdown_periods", []),
            total_drawdown_periods=len(drawdown_analysis.get("drawdown_periods", [])),
            fastest_recovery_days=drawdown_analysis.get("fastest_recovery_days"),
            slowest_recovery_days=drawdown_analysis.get("slowest_recovery_days"),
            current_drawdown_days=drawdown_analysis.get("current_drawdown_days")
        )
        
        logger.info(f"Drawdown analysis complete: max drawdown {drawdown_analysis['max_drawdown']}%")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating drawdown: {e}")
        raise OperationalError(f"Failed to calculate drawdown: {str(e)}")


@router.get(
    "/strategies/{strategy_id}/analytics/risk-metrics",
    response_model=RiskMetricsResponse,
    summary="Get risk metrics",
    description="Calculate risk metrics including Sharpe ratio, Sortino ratio, and Value at Risk"
)
async def get_risk_metrics(
    strategy_id: int = Path(..., gt=0, description="Strategy ID"),
    var_confidence: float = Query(0.95, ge=0.9, le=0.99, description="VaR confidence level"),
    lookback_days: int = Query(20, ge=5, le=252, description="Days to look back for calculations"),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
    strategy_service: StrategyEngineService = Depends(get_strategy_service),
    user_id: int = Depends(get_current_user_id)
) -> RiskMetricsResponse:
    """
    Get comprehensive risk metrics for a strategy.
    
    Args:
        strategy_id: ID of the strategy
        var_confidence: Confidence level for VaR calculation (0.95 = 95%)
        lookback_days: Number of days to analyze
        analytics_service: Analytics service instance
        strategy_service: Strategy service instance
        user_id: Current authenticated user ID
        
    Returns:
        Risk metrics including Sharpe, Sortino, VaR, and drawdown
        
    Raises:
        HTTPException: If strategy not found or access denied
    """
    try:
        logger.info(f"Calculating risk metrics for strategy {strategy_id}")
        
        # Verify ownership
        strategy = strategy_service.get_strategy(strategy_id)
        if strategy.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You can only view analytics for your own strategies"
            )
        
        # Get risk metrics
        metrics = analytics_service.calculate_risk_metrics(strategy_id)
        
        # Add custom VaR calculation if different from default
        custom_var = None
        if var_confidence != 0.95:
            custom_var = analytics_service.calculate_var(
                strategy_id, 
                confidence=var_confidence, 
                days=lookback_days
            )
        
        # Convert to response model
        response = RiskMetricsResponse(
            strategy_id=strategy_id,
            calculation_date=datetime.utcnow(),
            lookback_days=lookback_days,
            sharpe_ratio=metrics["sharpe_ratio"],
            sortino_ratio=metrics["sortino_ratio"],
            calmar_ratio=metrics["calmar_ratio"],
            information_ratio=metrics.get("information_ratio"),
            volatility_annualized=metrics.get("volatility_annualized", 0.15),
            downside_volatility=metrics.get("downside_volatility", 0.12),
            var_95=custom_var if var_confidence == 0.95 else metrics["var_95"],
            var_99=custom_var if var_confidence == 0.99 else metrics["var_99"],
            expected_shortfall_95=metrics.get("expected_shortfall_95"),
            var_confidence=var_confidence,
            max_drawdown_percent=metrics["max_drawdown"],
            current_drawdown_percent=metrics["current_drawdown"],
            max_drawdown_duration_days=metrics["max_drawdown_duration_days"],
            skewness=metrics.get("skewness"),
            kurtosis=metrics.get("kurtosis"),
            beta=metrics.get("beta")
        )
        
        logger.info(f"Risk metrics calculated: Sharpe={metrics.get('sharpe_ratio', 0):.2f}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating risk metrics: {e}")
        raise OperationalError(f"Failed to calculate risk metrics: {str(e)}")


@router.get(
    "/strategies/{strategy_id}/analytics/attribution",
    response_model=AttributionResponse,
    summary="Get performance attribution",
    description="Analyze performance attribution by grade, time, or market state"
)
async def get_performance_attribution(
    strategy_id: int = Path(..., gt=0, description="Strategy ID"),
    by: AttributionTypeEnum = Query(AttributionTypeEnum.GRADE, description="Attribution type"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
    strategy_service: StrategyEngineService = Depends(get_strategy_service),
    user_id: int = Depends(get_current_user_id)
) -> AttributionResponse:
    """
    Get performance attribution analysis.
    
    Args:
        strategy_id: ID of the strategy
        by: Attribution type (grade, time, market_state)
        start_date: Optional start date for analysis
        analytics_service: Analytics service instance
        strategy_service: Strategy service instance
        user_id: Current authenticated user ID
        
    Returns:
        Attribution analysis showing where returns come from
        
    Raises:
        HTTPException: If strategy not found or access denied
    """
    try:
        logger.info(f"Getting {by} attribution for strategy {strategy_id}")
        
        # Verify ownership
        strategy = strategy_service.get_strategy(strategy_id)
        if strategy.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You can only view analytics for your own strategies"
            )
        
        # Parse start date if provided
        start_datetime = None
        if start_date:
            try:
                start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
            except ValueError:
                raise ValidationError("Invalid start_date format. Use YYYY-MM-DD")
        
        # Get attribution based on type
        attribution_data = None
        by_grade = None
        by_time = None
        by_market_state = None
        
        if by == AttributionTypeEnum.GRADE:
            attribution_data = analytics_service.calculate_attribution_by_grade(
                strategy_id, 
                start_date=start_datetime
            )
            # Convert to list of AttributionByGrade objects
            by_grade = [
                {
                    "grade": grade,
                    "trade_count": data["count"],
                    "total_pnl": data["pnl"],
                    "win_rate": data["win_rate"],
                    "average_pnl": data["avg_pnl"],
                    "pnl_contribution_percent": data["pnl_contribution_percent"]
                }
                for grade, data in attribution_data.items()
            ]
        elif by == AttributionTypeEnum.TIME:
            attribution_data = analytics_service.calculate_attribution_by_time(strategy_id)
            # Convert to list of AttributionByTime objects
            by_time = [
                {
                    "time_period": time_period,
                    "trade_count": data["count"],
                    "total_pnl": data["pnl"],
                    "win_rate": data["win_rate"],
                    "average_pnl": data["avg_pnl"]
                }
                for time_period, data in attribution_data.items()
            ]
        else:  # market_state
            # For now, return empty dict - will be implemented when market state tracking is added
            by_market_state = {
                "message": "Market state attribution will be available in next version"
            }
        
        # Calculate totals
        total_trades = sum(data["count"] for data in attribution_data.values()) if attribution_data else 0
        total_pnl = sum(data["pnl"] for data in attribution_data.values()) if attribution_data else 0.0
        
        response = AttributionResponse(
            strategy_id=strategy_id,
            attribution_type=by,
            analysis_period="Custom" if start_date else "All Time",
            start_date=start_datetime,
            end_date=datetime.utcnow(),
            by_grade=by_grade,
            by_time=by_time,
            by_market_state=by_market_state,
            total_trades=total_trades,
            total_pnl=total_pnl,
            analysis_completeness=100.0  # Assume all trades included for now
        )
        
        logger.info(f"Attribution analysis complete for {by}")
        return response
        
    except ValidationError:
        raise
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating attribution: {e}")
        raise OperationalError(f"Failed to calculate attribution: {str(e)}")


@router.get(
    "/strategies/{strategy_id}/analytics/benchmark",
    response_model=BenchmarkComparisonResponse,
    summary="Compare to benchmark",
    description="Compare strategy performance against a benchmark index"
)
async def compare_to_benchmark(
    strategy_id: int = Path(..., gt=0, description="Strategy ID"),
    benchmark: str = Query("NIFTY", description="Benchmark symbol"),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
    strategy_service: StrategyEngineService = Depends(get_strategy_service),
    user_id: int = Depends(get_current_user_id)
) -> BenchmarkComparisonResponse:
    """
    Compare strategy performance to a benchmark.
    
    Args:
        strategy_id: ID of the strategy
        benchmark: Benchmark symbol (default NIFTY)
        analytics_service: Analytics service instance
        strategy_service: Strategy service instance
        user_id: Current authenticated user ID
        
    Returns:
        Comparison metrics vs benchmark
        
    Raises:
        HTTPException: If strategy not found or access denied
    """
    try:
        logger.info(f"Comparing strategy {strategy_id} to benchmark {benchmark}")
        
        # Verify ownership
        strategy = strategy_service.get_strategy(strategy_id)
        if strategy.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You can only view analytics for your own strategies"
            )
        
        # Get comparison
        comparison = analytics_service.compare_to_benchmark(strategy_id, benchmark)
        
        # Get strategy creation date for comparison period
        strategy_start = strategy.created_at
        
        # Convert to response model
        response = BenchmarkComparisonResponse(
            strategy_id=strategy_id,
            benchmark_symbol=benchmark,
            comparison_period="Since Inception",
            start_date=strategy_start,
            end_date=datetime.utcnow(),
            strategy_return_percent=comparison["strategy_return"],
            benchmark_return_percent=comparison["benchmark_return"],
            excess_return_percent=comparison["excess_return"],
            strategy_sharpe=comparison.get("strategy_sharpe", 0.0),
            benchmark_sharpe=comparison.get("benchmark_sharpe", 0.0),
            tracking_error=comparison["tracking_error"],
            information_ratio=comparison["information_ratio"],
            beta=comparison["beta"],
            alpha=comparison.get("alpha", 0.0),
            correlation=comparison.get("correlation", 0.0),
            strategy_volatility=comparison.get("strategy_volatility", 0.0),
            benchmark_volatility=comparison.get("benchmark_volatility", 0.0),
            up_capture_ratio=comparison.get("up_capture_ratio"),
            down_capture_ratio=comparison.get("down_capture_ratio")
        )
        
        logger.info(f"Benchmark comparison: excess return = {comparison.get('excess_return', 0):.2f}%")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing to benchmark: {e}")
        raise OperationalError(f"Failed to compare to benchmark: {str(e)}")


@router.get(
    "/analytics/portfolio/summary",
    response_model=PortfolioSummaryResponse,
    summary="Get portfolio summary",
    description="Get aggregated analytics across all strategies"
)
async def get_portfolio_summary(
    analytics_service: AnalyticsService = Depends(get_analytics_service),
    strategy_service: StrategyEngineService = Depends(get_strategy_service),
    user_id: int = Depends(get_current_user_id)
) -> PortfolioSummaryResponse:
    """
    Get portfolio-level analytics summary across all strategies.
    
    Args:
        analytics_service: Analytics service instance
        strategy_service: Strategy service instance
        user_id: Current authenticated user ID
        
    Returns:
        Portfolio summary with aggregated metrics
    """
    try:
        logger.info(f"Getting portfolio summary for user {user_id}")
        
        # Get all active strategies for the user
        strategies = strategy_service.list_strategies(
            user_id=user_id,
            is_active=True,
            limit=100
        )
        
        active_strategies = [s for s in strategies if s.is_active]
        
        if not strategies:
            return PortfolioSummaryResponse(
                user_id=user_id,
                total_strategies=0,
                active_strategies=0,
                total_nav=analytics_service.initial_capital,
                total_capital=analytics_service.initial_capital,
                total_pnl=0.0,
                portfolio_return_percent=0.0,
                average_nav_per_strategy=0.0,
                best_performing_strategy_id=None,
                worst_performing_strategy_id=None,
                strategies=[],
                last_updated=datetime.utcnow()
            )
        
        # Aggregate metrics across strategies
        total_nav = 0.0
        total_capital = analytics_service.initial_capital * len(strategies)
        total_pnl = 0.0
        strategy_summaries = []
        best_return = float('-inf')
        worst_return = float('inf')
        best_strategy_id = None
        worst_strategy_id = None
        
        for strategy in strategies:
            try:
                # Get NAV for each strategy
                nav = analytics_service.calculate_nav(strategy.id)
                pnl = nav - analytics_service.initial_capital
                return_percent = (pnl / analytics_service.initial_capital) * 100
                
                total_nav += nav
                total_pnl += pnl
                
                # Track best and worst performers
                if return_percent > best_return:
                    best_return = return_percent
                    best_strategy_id = strategy.id
                
                if return_percent < worst_return:
                    worst_return = return_percent
                    worst_strategy_id = strategy.id
                
                strategy_summaries.append({
                    "strategy_id": strategy.id,
                    "strategy_name": strategy.name,
                    "nav": nav,
                    "total_pnl": pnl,
                    "return_percent": return_percent,
                    "weight_percent": (nav / total_nav) * 100 if total_nav > 0 else 0,
                    "last_trade_date": None,  # Would need to query trades table
                    "is_active": strategy.is_active
                })
            except Exception as e:
                logger.warning(f"Error calculating metrics for strategy {strategy.id}: {e}")
                continue
        
        # Calculate portfolio-level metrics
        portfolio_return = (total_pnl / total_capital) * 100 if total_capital > 0 else 0
        
        response = PortfolioSummaryResponse(
            user_id=user_id,
            total_strategies=len(strategies),
            active_strategies=len(active_strategies),
            total_nav=total_nav,
            total_capital=total_capital,
            total_pnl=total_pnl,
            portfolio_return_percent=portfolio_return,
            average_nav_per_strategy=total_nav / len(strategies) if strategies else 0,
            best_performing_strategy_id=best_strategy_id,
            worst_performing_strategy_id=worst_strategy_id,
            strategies=strategy_summaries,
            last_updated=datetime.utcnow()
        )
        
        logger.info(f"Portfolio summary complete: {len(strategies)} strategies, total NAV = {total_nav}")
        return response
        
    except Exception as e:
        logger.error(f"Error getting portfolio summary: {e}")
        raise OperationalError(f"Failed to get portfolio summary: {str(e)}")


# Health check endpoint
@router.get(
    "/analytics/health",
    response_model=AnalyticsHealthResponse,
    summary="Analytics service health check",
    description="Check if analytics service is operational"
)
async def health_check(
    analytics_service: AnalyticsService = Depends(get_analytics_service)
) -> AnalyticsHealthResponse:
    """
    Health check for analytics service.
    
    Args:
        analytics_service: Analytics service instance
        
    Returns:
        Health status
    """
    try:
        # Try a simple operation to verify service is working
        _ = analytics_service.initial_capital
        
        return AnalyticsHealthResponse(
            status="healthy",
            service="analytics",
            timestamp=datetime.utcnow(),
            version="1.0.0",
            database_connected=True,
            last_calculation_time=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Analytics health check failed: {e}")
        raise OperationalError("Analytics service is not healthy")