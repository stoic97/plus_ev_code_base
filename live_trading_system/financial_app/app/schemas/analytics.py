"""
Analytics API response schemas for the Trading Strategies Application.

This module defines Pydantic response models for all analytics endpoints,
following the same patterns as the existing strategy schemas.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator
from enum import Enum

# Import the base schema from strategy to maintain consistency
from app.schemas.strategy import BaseSchema


# Enums for analytics (following existing enum patterns)
class TimeframeEnum(str, Enum):
    """Time period options for analytics."""
    ONE_DAY = "1D"
    ONE_WEEK = "1W"
    ONE_MONTH = "1M"
    THREE_MONTHS = "3M"
    ONE_YEAR = "1Y"
    ALL = "ALL"


class AttributionTypeEnum(str, Enum):
    """Attribution analysis types."""
    GRADE = "grade"
    TIME = "time"
    MARKET_STATE = "market_state"


# Base analytics schemas
class AnalyticsMetricBase(BaseModel):
    """Base schema for analytics metrics."""
    value: Union[float, int]
    formatted_value: Optional[str] = Field(None, description="Human-readable formatted value")
    change_percent: Optional[float] = Field(None, description="Percentage change from previous period")
    is_positive: Optional[bool] = Field(None, description="Whether the metric indicates positive performance")


class TimestampedMetric(BaseModel):
    """Base schema for timestamped metrics."""
    timestamp: datetime = Field(..., description="Metric timestamp")
    value: float = Field(..., description="Metric value")


# Dashboard response schemas
class LiveMetricsResponse(BaseSchema):
    """Response schema for live analytics dashboard."""
    
    # Core NAV metrics
    nav: float = Field(..., description="Current Net Asset Value")
    initial_capital: float = Field(..., description="Initial capital amount")
    cash: float = Field(..., description="Available cash balance")
    positions_value: float = Field(..., description="Total value of open positions")
    
    # P&L metrics
    total_pnl: float = Field(..., description="Total profit/loss")
    realized_pnl: float = Field(..., description="Realized profit/loss")
    unrealized_pnl: float = Field(..., description="Unrealized profit/loss from open positions")
    daily_pnl: Optional[float] = Field(None, description="Today's profit/loss")
    
    # Returns
    total_return_percent: float = Field(..., description="Total return percentage")
    daily_return_percent: Optional[float] = Field(None, description="Daily return percentage")
    
    # Drawdown metrics
    current_drawdown_percent: float = Field(..., description="Current drawdown percentage")
    high_water_mark: float = Field(..., description="Highest NAV achieved")
    
    # Trading activity
    active_trades: int = Field(..., description="Number of active trades")
    total_trades: int = Field(..., description="Total number of trades executed")
    trades_today: Optional[int] = Field(None, description="Number of trades today")
    
    # Last update
    last_updated: datetime = Field(..., description="Last update timestamp")
    
    @field_validator('total_return_percent', 'daily_return_percent', 'current_drawdown_percent')
    @classmethod
    def validate_percentages(cls, v):
        """Validate percentage values are reasonable."""
        if v is not None and (v < -100 or v > 1000):
            raise ValueError("Percentage values should be between -100% and 1000%")
        return v


# Equity curve schemas
class EquityCurvePoint(BaseModel):
    """Single point in equity curve."""
    timestamp: datetime = Field(..., description="Data point timestamp")
    nav: float = Field(..., description="Net Asset Value at this point")
    cumulative_pnl: float = Field(..., description="Cumulative profit/loss")
    daily_return_percent: float = Field(..., description="Daily return percentage")
    trade_id: Optional[int] = Field(None, description="Trade ID if this point corresponds to a trade")
    volume: Optional[float] = Field(None, description="Trading volume if applicable")


class EquityCurveResponse(BaseSchema):
    """Response schema for equity curve data."""
    strategy_id: int = Field(..., description="Strategy ID")
    period: TimeframeEnum = Field(..., description="Analysis period")
    start_date: datetime = Field(..., description="Curve start date")
    end_date: datetime = Field(..., description="Curve end date")
    data_points: List[EquityCurvePoint] = Field(..., description="Equity curve data points")
    
    # Summary metrics
    total_points: int = Field(..., description="Number of data points")
    starting_nav: float = Field(..., description="Starting NAV")
    ending_nav: float = Field(..., description="Ending NAV")
    total_return_percent: float = Field(..., description="Total return over period")
    
    @field_validator('data_points')
    @classmethod
    def validate_data_points(cls, v):
        """Ensure data points are not empty."""
        if not v:
            raise ValueError("Equity curve must have at least one data point")
        return v


# Drawdown analysis schemas
class DrawdownPeriod(BaseModel):
    """Individual drawdown period."""
    start_date: datetime = Field(..., description="Drawdown start date")
    end_date: Optional[datetime] = Field(None, description="Drawdown end date (None if ongoing)")
    peak_nav: float = Field(..., description="NAV at peak before drawdown")
    trough_nav: float = Field(..., description="Lowest NAV during drawdown")
    recovery_nav: Optional[float] = Field(None, description="NAV at recovery (if recovered)")
    max_drawdown_percent: float = Field(..., description="Maximum drawdown percentage")
    duration_days: int = Field(..., description="Drawdown duration in days")
    recovery_days: Optional[int] = Field(None, description="Days to recover (if recovered)")
    is_recovered: bool = Field(..., description="Whether drawdown has been recovered")


class DrawdownAnalysisResponse(BaseSchema):
    """Response schema for drawdown analysis."""
    strategy_id: int = Field(..., description="Strategy ID")
    analysis_period: TimeframeEnum = Field(..., description="Analysis period")
    
    # Current state
    current_nav: float = Field(..., description="Current NAV")
    high_water_mark: float = Field(..., description="All-time high NAV")
    current_drawdown_percent: float = Field(..., description="Current drawdown percentage")
    
    # Historical drawdown statistics
    max_drawdown_percent: float = Field(..., description="Maximum historical drawdown")
    max_drawdown_duration_days: int = Field(..., description="Longest drawdown duration")
    average_drawdown_percent: float = Field(..., description="Average drawdown percentage")
    average_recovery_days: float = Field(..., description="Average recovery time in days")
    
    # Drawdown periods
    drawdown_periods: List[DrawdownPeriod] = Field(..., description="Individual drawdown periods")
    total_drawdown_periods: int = Field(..., description="Total number of drawdown periods")
    
    # Recovery statistics
    fastest_recovery_days: Optional[int] = Field(None, description="Fastest recovery time")
    slowest_recovery_days: Optional[int] = Field(None, description="Slowest recovery time")
    current_drawdown_days: Optional[int] = Field(None, description="Days in current drawdown")


# Risk metrics schemas
class RiskMetricsResponse(BaseSchema):
    """Response schema for risk metrics."""
    strategy_id: int = Field(..., description="Strategy ID")
    calculation_date: datetime = Field(..., description="When metrics were calculated")
    lookback_days: int = Field(..., description="Days used for calculation")
    
    # Return-based risk metrics
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    sortino_ratio: float = Field(..., description="Sortino ratio")
    calmar_ratio: float = Field(..., description="Calmar ratio")
    information_ratio: Optional[float] = Field(None, description="Information ratio vs benchmark")
    
    # Volatility metrics
    volatility_annualized: float = Field(..., description="Annualized volatility")
    downside_volatility: float = Field(..., description="Downside volatility")
    
    # Value at Risk
    var_95: float = Field(..., description="95% Value at Risk")
    var_99: float = Field(..., description="99% Value at Risk")
    expected_shortfall_95: Optional[float] = Field(None, description="95% Expected Shortfall")
    var_confidence: float = Field(..., description="VaR confidence level used")
    
    # Drawdown metrics
    max_drawdown_percent: float = Field(..., description="Maximum drawdown")
    current_drawdown_percent: float = Field(..., description="Current drawdown")
    max_drawdown_duration_days: int = Field(..., description="Maximum drawdown duration")
    
    # Additional risk measures
    skewness: Optional[float] = Field(None, description="Return distribution skewness")
    kurtosis: Optional[float] = Field(None, description="Return distribution kurtosis")
    beta: Optional[float] = Field(None, description="Beta relative to benchmark")
    
    @field_validator('sharpe_ratio', 'sortino_ratio', 'calmar_ratio')
    @classmethod
    def validate_ratios(cls, v):
        """Validate risk ratios are reasonable."""
        if v < -10 or v > 10:
            raise ValueError("Risk ratios should be between -10 and 10")
        return v


# Attribution analysis schemas
class AttributionByGrade(BaseModel):
    """Performance attribution by setup quality grade."""
    grade: str = Field(..., description="Setup quality grade")
    trade_count: int = Field(..., description="Number of trades")
    total_pnl: float = Field(..., description="Total P&L for this grade")
    win_rate: float = Field(..., description="Win rate for this grade")
    average_pnl: float = Field(..., description="Average P&L per trade")
    pnl_contribution_percent: float = Field(..., description="Percentage contribution to total P&L")
    
    @field_validator('win_rate')
    @classmethod
    def validate_win_rate(cls, v):
        """Ensure win rate is between 0 and 1."""
        if v < 0 or v > 1:
            raise ValueError("Win rate must be between 0 and 1")
        return v


class AttributionByTime(BaseModel):
    """Performance attribution by time period."""
    time_period: str = Field(..., description="Time period (e.g., '09:15-10:00')")
    trade_count: int = Field(..., description="Number of trades")
    total_pnl: float = Field(..., description="Total P&L for this time period")
    win_rate: float = Field(..., description="Win rate for this time period")
    average_pnl: float = Field(..., description="Average P&L per trade")


class AttributionResponse(BaseSchema):
    """Response schema for performance attribution."""
    strategy_id: int = Field(..., description="Strategy ID")
    attribution_type: AttributionTypeEnum = Field(..., description="Type of attribution analysis")
    analysis_period: Optional[str] = Field(None, description="Analysis period")
    start_date: Optional[datetime] = Field(None, description="Analysis start date")
    end_date: Optional[datetime] = Field(None, description="Analysis end date")
    
    # Attribution data (union type to handle different attribution types)
    by_grade: Optional[List[AttributionByGrade]] = Field(None, description="Attribution by grade")
    by_time: Optional[List[AttributionByTime]] = Field(None, description="Attribution by time")
    by_market_state: Optional[Dict[str, Any]] = Field(None, description="Attribution by market state")
    
    # Summary
    total_trades: int = Field(..., description="Total trades analyzed")
    total_pnl: float = Field(..., description="Total P&L")
    analysis_completeness: float = Field(..., description="Percentage of trades included in analysis")


# Benchmark comparison schemas
class BenchmarkComparisonResponse(BaseSchema):
    """Response schema for benchmark comparison."""
    strategy_id: int = Field(..., description="Strategy ID")
    benchmark_symbol: str = Field(..., description="Benchmark symbol")
    comparison_period: str = Field(..., description="Comparison period")
    start_date: datetime = Field(..., description="Comparison start date")
    end_date: datetime = Field(..., description="Comparison end date")
    
    # Return comparison
    strategy_return_percent: float = Field(..., description="Strategy total return")
    benchmark_return_percent: float = Field(..., description="Benchmark total return")
    excess_return_percent: float = Field(..., description="Strategy excess return vs benchmark")
    
    # Risk-adjusted metrics
    strategy_sharpe: float = Field(..., description="Strategy Sharpe ratio")
    benchmark_sharpe: float = Field(..., description="Benchmark Sharpe ratio")
    
    # Tracking metrics
    tracking_error: float = Field(..., description="Tracking error vs benchmark")
    information_ratio: float = Field(..., description="Information ratio")
    beta: float = Field(..., description="Beta relative to benchmark")
    alpha: float = Field(..., description="Alpha relative to benchmark")
    
    # Correlation
    correlation: float = Field(..., description="Correlation with benchmark")
    
    # Additional metrics
    strategy_volatility: float = Field(..., description="Strategy volatility")
    benchmark_volatility: float = Field(..., description="Benchmark volatility")
    up_capture_ratio: Optional[float] = Field(None, description="Up capture ratio")
    down_capture_ratio: Optional[float] = Field(None, description="Down capture ratio")


# Portfolio summary schemas
class StrategyPortfolioSummary(BaseModel):
    """Summary for individual strategy in portfolio."""
    strategy_id: int = Field(..., description="Strategy ID")
    strategy_name: str = Field(..., description="Strategy name")
    nav: float = Field(..., description="Current NAV")
    total_pnl: float = Field(..., description="Total P&L")
    return_percent: float = Field(..., description="Return percentage")
    weight_percent: float = Field(..., description="Portfolio weight percentage")
    last_trade_date: Optional[datetime] = Field(None, description="Date of last trade")
    is_active: bool = Field(..., description="Whether strategy is active")


class PortfolioSummaryResponse(BaseSchema):
    """Response schema for portfolio-level summary."""
    user_id: int = Field(..., description="User ID")
    total_strategies: int = Field(..., description="Total number of strategies")
    active_strategies: int = Field(..., description="Number of active strategies")
    
    # Aggregated financials
    total_nav: float = Field(..., description="Total portfolio NAV")
    total_capital: float = Field(..., description="Total initial capital")
    total_pnl: float = Field(..., description="Total portfolio P&L")
    portfolio_return_percent: float = Field(..., description="Overall portfolio return")
    
    # Portfolio metrics
    average_nav_per_strategy: float = Field(..., description="Average NAV per strategy")
    best_performing_strategy_id: Optional[int] = Field(None, description="Best performing strategy ID")
    worst_performing_strategy_id: Optional[int] = Field(None, description="Worst performing strategy ID")
    
    # Individual strategy summaries
    strategies: List[StrategyPortfolioSummary] = Field(..., description="Individual strategy summaries")
    
    # Last update
    last_updated: datetime = Field(..., description="Last portfolio update timestamp")
    
    @field_validator('total_strategies')
    @classmethod
    def validate_strategy_count(cls, v):
        """Ensure strategy count is non-negative."""
        if v < 0:
            raise ValueError("Strategy count cannot be negative")
        return v


# Health check schema
class AnalyticsHealthResponse(BaseModel):
    """Response schema for analytics health check."""
    status: str = Field(..., description="Health status")
    service: str = Field(..., description="Service name")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: Optional[str] = Field(None, description="Service version")
    database_connected: Optional[bool] = Field(None, description="Database connection status")
    last_calculation_time: Optional[datetime] = Field(None, description="Last successful calculation time")


# Error response schemas
class AnalyticsErrorResponse(BaseModel):
    """Response schema for analytics errors."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    strategy_id: Optional[int] = Field(None, description="Strategy ID if applicable")
    timestamp: datetime = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")


# Utility schemas for complex responses
class PaginatedAnalyticsResponse(BaseModel):
    """Base schema for paginated analytics responses."""
    total_count: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether there is a next page")
    has_previous: bool = Field(..., description="Whether there is a previous page")


class TimeSeriesDataPoint(BaseModel):
    """Generic time series data point."""
    timestamp: datetime = Field(..., description="Data timestamp")
    value: float = Field(..., description="Data value")
    label: Optional[str] = Field(None, description="Data label")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class AnalyticsFilterOptions(BaseModel):
    """Available filter options for analytics endpoints."""
    available_periods: List[TimeframeEnum] = Field(..., description="Available time periods")
    available_attribution_types: List[AttributionTypeEnum] = Field(..., description="Available attribution types")
    available_benchmarks: List[str] = Field(..., description="Available benchmark symbols")
    date_range: Dict[str, datetime] = Field(..., description="Available date range")


# Export all response models for use in endpoints
__all__ = [
    "LiveMetricsResponse",
    "EquityCurveResponse",
    "EquityCurvePoint", 
    "DrawdownAnalysisResponse",
    "DrawdownPeriod",
    "RiskMetricsResponse",
    "AttributionResponse",
    "AttributionByGrade",
    "AttributionByTime",
    "BenchmarkComparisonResponse",
    "PortfolioSummaryResponse",
    "StrategyPortfolioSummary",
    "AnalyticsHealthResponse",
    "AnalyticsErrorResponse",
    "TimeframeEnum",
    "AttributionTypeEnum",
    "PaginatedAnalyticsResponse",
    "TimeSeriesDataPoint",
    "AnalyticsFilterOptions",
    "AnalyticsMetricBase",
    "TimestampedMetric"
]