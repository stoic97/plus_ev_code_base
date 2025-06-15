"""
Backtest API Schemas for Trading Strategies Application

This module defines Pydantic schemas for backtest operations, following the existing
patterns from strategy and analytics schemas. Provides comprehensive validation
and documentation for backtest API endpoints.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from enum import Enum

# Import base schema from strategy to maintain consistency
from app.schemas.strategy import BaseSchema, DirectionEnum


# Enums for backtest validation
class BacktestStatusEnum(str, Enum):
    """Backtest execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BacktestDataSourceEnum(str, Enum):
    """Data source options for backtesting."""
    CSV = "csv"
    DATABASE = "database"
    API = "api"


class TimeframeEnum(str, Enum):
    """Timeframe options for backtesting data."""
    ONE_MIN = "1m"
    FIVE_MIN = "5m"
    FIFTEEN_MIN = "15m"
    THIRTY_MIN = "30m"
    ONE_HOUR = "1h"
    FOUR_HOUR = "4h"
    DAILY = "1d"


# Base schemas
class BacktestBaseSchema(BaseSchema):
    """Base schema with backtest-specific configuration."""
    
    model_config = ConfigDict(
        from_attributes=True,
        arbitrary_types_allowed=True,
        json_encoders={
            datetime: lambda v: v.isoformat() if v else None
        }
    )


# Configuration schemas
class BacktestConfigBase(BacktestBaseSchema):
    """Base schema for backtest configuration."""
    strategy_id: int = Field(..., description="Strategy ID to backtest", gt=0)
    name: str = Field(..., description="Backtest name", min_length=1, max_length=255)
    description: Optional[str] = Field(None, description="Backtest description", max_length=1000)
    
    # Time period
    start_date: datetime = Field(..., description="Backtest start date")
    end_date: datetime = Field(..., description="Backtest end date")
    
    # Financial settings
    initial_capital: float = Field(..., description="Initial capital for backtest", gt=0)
    max_position_size: float = Field(
        0.1, 
        description="Maximum position size as fraction of capital",
        ge=0.001,
        le=1.0
    )
    
    # Data settings
    data_source: BacktestDataSourceEnum = Field(
        BacktestDataSourceEnum.CSV,
        description="Data source for backtesting"
    )
    timeframe: TimeframeEnum = Field(
        TimeframeEnum.ONE_MIN,
        description="Data timeframe for backtesting"
    )
    
    # Cost assumptions
    commission_per_trade: float = Field(
        20.0, 
        description="Commission per trade in INR",
        ge=0
    )
    slippage_bps: float = Field(
        2.0, 
        description="Slippage in basis points",
        ge=0,
        le=100
    )
    
    # Optional parameters
    warm_up_days: int = Field(
        30, 
        description="Warm-up period in days for indicators",
        ge=0,
        le=365
    )
    benchmark_symbol: Optional[str] = Field(
        "NIFTY50",
        description="Benchmark symbol for comparison"
    )
    
    @field_validator('end_date')
    @classmethod
    def validate_date_order(cls, v, info):
        """Ensure end date is after start date."""
        if 'start_date' in info.data and v <= info.data['start_date']:
            raise ValueError("End date must be after start date")
        return v
    
    @field_validator('start_date')
    @classmethod
    def validate_start_date(cls, v):
        """Ensure start date is not in the future."""
        if v > datetime.now():
            raise ValueError("Start date cannot be in the future")
        return v


class BacktestConfigCreate(BacktestConfigBase):
    """Schema for creating a new backtest configuration."""
    pass


class BacktestConfigUpdate(BacktestBaseSchema):
    """Schema for updating backtest configuration."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    max_position_size: Optional[float] = Field(None, ge=0.001, le=1.0)
    commission_per_trade: Optional[float] = Field(None, ge=0)
    slippage_bps: Optional[float] = Field(None, ge=0, le=100)
    benchmark_symbol: Optional[str] = Field(None)


# Trade schemas
class BacktestTradeBase(BacktestBaseSchema):
    """Base schema for backtest trade data."""
    trade_id: str = Field(..., description="Unique trade identifier")
    signal_id: int = Field(..., description="Signal that generated this trade", gt=0)
    strategy_id: int = Field(..., description="Strategy ID", gt=0)
    instrument: str = Field(..., description="Trading instrument", min_length=1)
    direction: DirectionEnum = Field(..., description="Trade direction")
    
    # Entry data
    entry_time: datetime = Field(..., description="Trade entry timestamp")
    entry_price: float = Field(..., description="Entry price", gt=0)
    quantity: int = Field(1, description="Trade quantity", gt=0)
    
    # Exit data (optional for open trades)
    exit_time: Optional[datetime] = Field(None, description="Trade exit timestamp")
    exit_price: Optional[float] = Field(None, description="Exit price", gt=0)
    exit_reason: Optional[str] = Field(None, description="Reason for exit")
    
    # P&L data
    pnl_points: Optional[float] = Field(None, description="P&L in points")
    pnl_inr: Optional[float] = Field(None, description="P&L in INR")
    
    # Costs
    commission: float = Field(0.0, description="Commission paid", ge=0)
    slippage: float = Field(0.0, description="Slippage cost", ge=0)
    
    # Setup quality (from strategy analysis)
    setup_quality: Optional[str] = Field(None, description="Setup quality grade")
    setup_score: Optional[float] = Field(None, description="Setup quality score", ge=0, le=10)
    
    @field_validator('exit_time')
    @classmethod
    def validate_exit_after_entry(cls, v, info):
        """Ensure exit time is after entry time."""
        if v is not None and 'entry_time' in info.data and v <= info.data['entry_time']:
            raise ValueError("Exit time must be after entry time")
        return v


class BacktestTradeResponse(BacktestTradeBase):
    """Schema for backtest trade API responses."""
    is_open: bool = Field(..., description="Whether trade is still open")
    duration_minutes: Optional[int] = Field(None, description="Trade duration in minutes")


# Metrics schemas
class BacktestMetricsBase(BacktestBaseSchema):
    """Base schema for backtest performance metrics."""
    # Basic performance
    total_return_pct: float = Field(..., description="Total return percentage")
    annual_return_pct: float = Field(..., description="Annualized return percentage")
    total_pnl_inr: float = Field(..., description="Total P&L in INR")
    
    # Risk metrics
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    sortino_ratio: float = Field(..., description="Sortino ratio")
    max_drawdown_pct: float = Field(..., description="Maximum drawdown percentage", ge=0)
    max_drawdown_duration_days: int = Field(..., description="Max drawdown duration in days", ge=0)
    volatility_annual_pct: float = Field(..., description="Annualized volatility percentage", ge=0)
    
    # Trade statistics
    total_trades: int = Field(..., description="Total number of trades", ge=0)
    winning_trades: int = Field(..., description="Number of winning trades", ge=0)
    losing_trades: int = Field(..., description="Number of losing trades", ge=0)
    win_rate_pct: float = Field(..., description="Win rate percentage", ge=0, le=100)
    
    # P&L statistics
    avg_win_inr: float = Field(..., description="Average win in INR")
    avg_loss_inr: float = Field(..., description="Average loss in INR")
    largest_win_inr: float = Field(..., description="Largest win in INR")
    largest_loss_inr: float = Field(..., description="Largest loss in INR")
    profit_factor: float = Field(..., description="Profit factor", ge=0)
    
    # Cost analysis
    total_commission_inr: float = Field(..., description="Total commission paid in INR", ge=0)
    total_slippage_inr: float = Field(..., description="Total slippage cost in INR", ge=0)
    total_costs_inr: float = Field(..., description="Total trading costs in INR", ge=0)
    
    # Time analysis
    avg_trade_duration_minutes: float = Field(..., description="Average trade duration in minutes", ge=0)
    trades_per_day: float = Field(..., description="Average trades per day", ge=0)
    
    # Additional metrics
    calmar_ratio: float = Field(..., description="Calmar ratio")
    kelly_criterion: float = Field(..., description="Kelly criterion percentage")
    expectancy_inr: float = Field(..., description="Mathematical expectancy in INR")
    
    @field_validator('win_rate_pct')
    @classmethod
    def validate_win_rate(cls, v):
        """Ensure win rate is between 0 and 100."""
        if not 0 <= v <= 100:
            raise ValueError("Win rate must be between 0 and 100")
        return v


class BacktestMetricsResponse(BacktestMetricsBase):
    """Schema for backtest metrics API responses."""
    calculation_time: datetime = Field(..., description="When metrics were calculated")


# Result schemas
class BacktestResultBase(BacktestBaseSchema):
    """Base schema for backtest results."""
    backtest_id: str = Field(..., description="Unique backtest identifier")
    strategy_id: int = Field(..., description="Strategy ID", gt=0)
    status: BacktestStatusEnum = Field(..., description="Backtest execution status")
    
    # Timing
    start_time: datetime = Field(..., description="Backtest start time")
    end_time: Optional[datetime] = Field(None, description="Backtest completion time")
    
    # Configuration reference
    config: BacktestConfigBase = Field(..., description="Backtest configuration used")
    
    # Results (populated after completion)
    metrics: Optional[BacktestMetricsBase] = Field(None, description="Performance metrics")
    trade_count: Optional[int] = Field(None, description="Total number of trades", ge=0)
    
    # Error information
    error_message: Optional[str] = Field(None, description="Error message if failed")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    
    @field_validator('end_time')
    @classmethod
    def validate_end_after_start(cls, v, info):
        """Ensure end time is after start time."""
        if v is not None and 'start_time' in info.data and v <= info.data['start_time']:
            raise ValueError("End time must be after start time")
        return v


class BacktestResultResponse(BacktestResultBase):
    """Schema for backtest result API responses."""
    duration_seconds: Optional[float] = Field(None, description="Execution duration in seconds")
    is_complete: bool = Field(..., description="Whether backtest completed successfully")


# Detailed result schemas
class BacktestDetailedResultResponse(BacktestResultResponse):
    """Schema for detailed backtest results with trade data."""
    trades: List[BacktestTradeResponse] = Field(..., description="All trades executed")
    equity_curve: List[Dict[str, Any]] = Field(..., description="Equity curve data points")
    monthly_returns: List[Dict[str, Any]] = Field(..., description="Monthly return breakdown")
    drawdown_periods: List[Dict[str, Any]] = Field(..., description="Major drawdown periods")
    trade_analysis: Dict[str, Any] = Field(..., description="Detailed trade analysis")


# Request schemas
class BacktestRunRequest(BacktestConfigCreate):
    """Schema for running a new backtest."""
    run_immediately: bool = Field(True, description="Whether to start backtest immediately")


class BacktestStatusRequest(BacktestBaseSchema):
    """Schema for checking backtest status."""
    backtest_id: str = Field(..., description="Backtest ID to check")


class BacktestListRequest(BacktestBaseSchema):
    """Schema for listing backtests with filters."""
    strategy_id: Optional[int] = Field(None, description="Filter by strategy ID", gt=0)
    status: Optional[BacktestStatusEnum] = Field(None, description="Filter by status")
    start_date_from: Optional[datetime] = Field(None, description="Filter by start date (from)")
    start_date_to: Optional[datetime] = Field(None, description="Filter by start date (to)")
    limit: int = Field(10, description="Number of results to return", ge=1, le=100)
    offset: int = Field(0, description="Number of results to skip", ge=0)


# Comparison schemas
class BacktestComparisonRequest(BacktestBaseSchema):
    """Schema for comparing multiple backtests."""
    backtest_ids: List[str] = Field(..., description="List of backtest IDs to compare", min_length=2, max_length=5)
    metrics_to_compare: List[str] = Field(
        default_factory=lambda: ["total_return_pct", "sharpe_ratio", "max_drawdown_pct", "win_rate_pct"],
        description="List of metrics to compare"
    )


class BacktestComparisonResponse(BacktestBaseSchema):
    """Schema for backtest comparison results."""
    backtests: List[BacktestResultResponse] = Field(..., description="Backtest results being compared")
    comparison_metrics: Dict[str, Dict[str, float]] = Field(..., description="Side-by-side metric comparison")
    best_performing: Dict[str, str] = Field(..., description="Best performing backtest ID for each metric")
    summary_stats: Dict[str, Any] = Field(..., description="Summary statistics across all backtests")


# Health check schema
class BacktestHealthResponse(BacktestBaseSchema):
    """Schema for backtest service health check."""
    status: str = Field(..., description="Health status")
    service: str = Field(..., description="Service name")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: Optional[str] = Field(None, description="Service version")
    database_connected: bool = Field(..., description="Database connection status")
    csv_data_accessible: bool = Field(..., description="CSV data file accessibility")
    last_backtest_time: Optional[datetime] = Field(None, description="Last successful backtest time")
    running_backtests: int = Field(..., description="Number of currently running backtests", ge=0)


# Error response schema
class BacktestErrorResponse(BacktestBaseSchema):
    """Schema for backtest error responses."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    backtest_id: Optional[str] = Field(None, description="Backtest ID if applicable")
    strategy_id: Optional[int] = Field(None, description="Strategy ID if applicable")
    timestamp: datetime = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")


# Export all schemas
__all__ = [
    # Enums
    "BacktestStatusEnum",
    "BacktestDataSourceEnum", 
    "TimeframeEnum",
    
    # Configuration schemas
    "BacktestConfigBase",
    "BacktestConfigCreate",
    "BacktestConfigUpdate",
    
    # Trade schemas
    "BacktestTradeBase",
    "BacktestTradeResponse",
    
    # Metrics schemas
    "BacktestMetricsBase",
    "BacktestMetricsResponse",
    
    # Result schemas
    "BacktestResultBase",
    "BacktestResultResponse",
    "BacktestDetailedResultResponse",
    
    # Request schemas
    "BacktestRunRequest",
    "BacktestStatusRequest",
    "BacktestListRequest",
    
    # Comparison schemas
    "BacktestComparisonRequest",
    "BacktestComparisonResponse",
    
    # Utility schemas
    "BacktestHealthResponse",
    "BacktestErrorResponse",
    "BacktestBaseSchema"
]