"""
Backtest Models for Time Travel Backtesting System

This module defines database models for comprehensive backtesting operations, extending
the existing StrategyBacktest model with additional metadata and result tracking.
Models support both CSV and live data backtesting with detailed analytics.
"""

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, ForeignKey, 
    Enum, Text, JSON, CheckConstraint, Index
)
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func
import enum
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union

from app.core.database import Base
from app.models.base import TimestampMixin



class BacktestStatus(enum.Enum):
    """Status of backtest execution."""
    PENDING = "pending"           # Queued for execution
    INITIALIZING = "initializing" # Setting up data and environment
    RUNNING = "running"           # Currently executing
    COMPLETED = "completed"       # Successfully finished
    FAILED = "failed"            # Failed with errors
    CANCELLED = "cancelled"      # User cancelled
    PAUSED = "paused"            # Temporarily paused


class BacktestDataSource(enum.Enum):
    """Data source for backtest execution."""
    CSV_FILE = "csv_file"        # Historical CSV data (Edata.csv)
    LIVE_API = "live_api"        # Live API data replay
    HYBRID = "hybrid"            # Combination of both


class BacktestType(enum.Enum):
    """Type of backtest being performed."""
    STRATEGY_VALIDATION = "strategy_validation"  # Test strategy logic
    PARAMETER_OPTIMIZATION = "parameter_optimization"  # Optimize parameters
    PERFORMANCE_COMPARISON = "performance_comparison"  # Compare with live trading
    RISK_ASSESSMENT = "risk_assessment"  # Assess risk characteristics



class BacktestRun(Base, TimestampMixin):
    """
    Backtest execution metadata and configuration.
    
    This model extends the existing StrategyBacktest with additional metadata
    for tracking backtest runs, especially for time travel backtesting approach.
    """
    
    __tablename__ = "backtest_runs"
    
    # Primary identification
    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String(50), unique=True, nullable=False, index=True)  # UUID for tracking
    
    # Relationships - reuse existing infrastructure
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False, index=True)
    strategy_backtest_id = Column(Integer, ForeignKey("strategy_backtests.id"), nullable=True, index=True)
    
    # Basic configuration
    name = Column(String(255), nullable=False)
    description = Column(Text)
    backtest_type = Column(Enum(BacktestType), nullable=False, default=BacktestType.STRATEGY_VALIDATION)
    
    # Execution details
    status = Column(Enum(BacktestStatus), nullable=False, default=BacktestStatus.PENDING, index=True)
    data_source = Column(Enum(BacktestDataSource), nullable=False, default=BacktestDataSource.CSV_FILE)
    
    # Time range for backtesting
    start_date = Column(DateTime, nullable=False, index=True)
    end_date = Column(DateTime, nullable=False, index=True)
    
    # Execution timing
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    execution_time_seconds = Column(Float, nullable=True)
    
    # Configuration
    initial_capital = Column(Float, nullable=False, default=1000000.0)  # 10 lakhs INR
    risk_per_trade_percent = Column(Float, default=1.0)
    max_concurrent_trades = Column(Integer, default=3)
    
    # Data source specific configuration
    csv_file_path = Column(String(500), nullable=True)  # Path to CSV file
    data_provider_config = Column(JSON, nullable=True)  # Additional data provider settings
    
    # Strategy configuration for this run
    strategy_config = Column(JSON, nullable=True)  # Strategy parameters for this backtest
    
    # Progress tracking
    progress_percent = Column(Float, default=0.0)
    current_date = Column(DateTime, nullable=True)  # Current simulation date
    processed_bars = Column(Integer, default=0)
    total_bars = Column(Integer, nullable=True)
    
    # Error handling
    error_message = Column(Text, nullable=True)
    warnings = Column(JSON, nullable=True)  # List of warning messages
    
    # Results summary (quick access)
    total_trades = Column(Integer, default=0)
    total_signals = Column(Integer, default=0)
    win_rate = Column(Float, nullable=True)
    total_pnl_inr = Column(Float, nullable=True)
    max_drawdown_percent = Column(Float, nullable=True)
    
    # Relationships
    strategy = relationship("Strategy", back_populates="backtest_runs")
    strategy_backtest = relationship("StrategyBacktest", back_populates="backtest_runs")
    performance_snapshots = relationship("BacktestPerformanceSnapshot", back_populates="backtest_run", cascade="all, delete-orphan")
    execution_logs = relationship("BacktestExecutionLog", back_populates="backtest_run", cascade="all, delete-orphan")
    
    # Table constraints
    __table_args__ = (
        CheckConstraint('start_date < end_date', name='check_date_range'),
        CheckConstraint('initial_capital > 0', name='check_positive_capital'),
        CheckConstraint('risk_per_trade_percent > 0 AND risk_per_trade_percent <= 10', name='check_risk_range'),
        CheckConstraint('progress_percent >= 0 AND progress_percent <= 100', name='check_progress_range'),
        Index('idx_backtest_runs_strategy_status', 'strategy_id', 'status'),
        Index('idx_backtest_runs_date_range', 'start_date', 'end_date'),
    )
    
    @validates('start_date', 'end_date')
    def validate_dates(self, key, value):
        """Validate date ranges."""
        if key == 'end_date' and hasattr(self, 'start_date') and self.start_date:
            if value <= self.start_date:
                raise ValueError("End date must be after start date")
        return value
    
    @validates('risk_per_trade_percent')
    def validate_risk_percent(self, key, value):
        """Validate risk percentage is reasonable."""
        if value <= 0 or value > 10:
            raise ValueError("Risk per trade must be between 0 and 10 percent")
        return value
    
    @property
    def duration_days(self) -> Optional[int]:
        """Get backtest duration in days."""
        if self.start_date and self.end_date:
            return (self.end_date - self.start_date).days
        return None
    
    @property
    def is_running(self) -> bool:
        """Check if backtest is currently running."""
        return self.status in [BacktestStatus.PENDING, BacktestStatus.INITIALIZING, BacktestStatus.RUNNING]
    
    @property
    def is_complete(self) -> bool:
        """Check if backtest completed successfully."""
        return self.status == BacktestStatus.COMPLETED
    
    def update_progress(self, current_date: datetime, processed_bars: int, total_bars: int):
        """Update backtest progress."""
        self.current_date = current_date
        self.processed_bars = processed_bars
        self.total_bars = total_bars
        if total_bars > 0:
            self.progress_percent = min(100.0, (processed_bars / total_bars) * 100)


class BacktestPerformanceSnapshot(Base, TimestampMixin):
    """
    Performance snapshots during backtest execution.
    
    Captures portfolio state at regular intervals for monitoring and analysis.
    Useful for creating equity curves and monitoring progress.
    """
    
    __tablename__ = "backtest_performance_snapshots"
    
    # Primary identification
    id = Column(Integer, primary_key=True, index=True)
    
    # Relationships
    backtest_run_id = Column(Integer, ForeignKey("backtest_runs.id"), nullable=False, index=True)
    
    # Snapshot timing
    snapshot_date = Column(DateTime, nullable=False, index=True)
    simulation_date = Column(DateTime, nullable=False, index=True)  # Date in simulation time
    
    # Portfolio state
    portfolio_value = Column(Float, nullable=False)
    cash_balance = Column(Float, nullable=False)
    position_value = Column(Float, default=0.0)
    unrealized_pnl = Column(Float, default=0.0)
    realized_pnl = Column(Float, default=0.0)
    
    # Performance metrics at this point
    total_return_percent = Column(Float, default=0.0)
    drawdown_percent = Column(Float, default=0.0)
    trades_count = Column(Integer, default=0)
    open_positions_count = Column(Integer, default=0)
    
    # Risk metrics
    daily_var = Column(Float, nullable=True)  # Value at Risk
    beta = Column(Float, nullable=True)       # Portfolio beta
    volatility = Column(Float, nullable=True) # Rolling volatility
    
    # Additional metrics specific to strategy
    setup_quality_score = Column(Float, nullable=True)
    timeframe_alignment_score = Column(Float, nullable=True)
    
    # Relationship
    backtest_run = relationship("BacktestRun", back_populates="performance_snapshots")
    
    # Table constraints
    __table_args__ = (
        Index('idx_performance_snapshots_run_date', 'backtest_run_id', 'simulation_date'),
        CheckConstraint('portfolio_value >= 0', name='check_positive_portfolio_value'),
    )


class BacktestExecutionLog(Base, TimestampMixin):
    """
    Execution logs for backtest runs.
    
    Tracks important events, decisions, and debugging information during backtest execution.
    Essential for debugging and understanding backtest behavior.
    """
    
    __tablename__ = "backtest_execution_logs"
    
    # Primary identification
    id = Column(Integer, primary_key=True, index=True)
    
    # Relationships
    backtest_run_id = Column(Integer, ForeignKey("backtest_runs.id"), nullable=False, index=True)
    
    # Log details
    simulation_date = Column(DateTime, nullable=False, index=True)  # Date in simulation time
    log_level = Column(String(20), nullable=False, default="INFO")  # INFO, WARNING, ERROR, DEBUG
    component = Column(String(50), nullable=False)  # Which component generated the log
    message = Column(Text, nullable=False)
    
    # Contextual data
    context_data = Column(JSON, nullable=True)  # Additional structured data
    
    # Error details (if applicable)
    error_type = Column(String(100), nullable=True)
    stack_trace = Column(Text, nullable=True)
    
    # Relationship
    backtest_run = relationship("BacktestRun", back_populates="execution_logs")
    
    # Table constraints
    __table_args__ = (
        Index('idx_execution_logs_run_level', 'backtest_run_id', 'log_level'),
        Index('idx_execution_logs_simulation_date', 'simulation_date'),
        CheckConstraint("log_level IN ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')", name='check_valid_log_level'),
    )



class BacktestComparison(Base, TimestampMixin):
    """
    Comparison between backtest results and live trading performance.
    
    This model stores side-by-side comparisons to validate strategy effectiveness
    and identify differences between historical and live performance.
    """
    
    __tablename__ = "backtest_comparisons"
    
    # Primary identification
    id = Column(Integer, primary_key=True, index=True)
    comparison_id = Column(String(50), unique=True, nullable=False, index=True)
    
    # Relationships
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False, index=True)
    backtest_run_id = Column(Integer, ForeignKey("backtest_runs.id"), nullable=False, index=True)
    
    # Comparison configuration
    name = Column(String(255), nullable=False)
    description = Column(Text)
    comparison_period_start = Column(DateTime, nullable=False)
    comparison_period_end = Column(DateTime, nullable=False)
    
    # Backtest metrics
    backtest_total_return = Column(Float, nullable=True)
    backtest_win_rate = Column(Float, nullable=True)
    backtest_sharpe_ratio = Column(Float, nullable=True)
    backtest_max_drawdown = Column(Float, nullable=True)
    backtest_total_trades = Column(Integer, nullable=True)
    backtest_avg_trade_duration_hours = Column(Float, nullable=True)
    
    # Live trading metrics (for the same period)
    live_total_return = Column(Float, nullable=True)
    live_win_rate = Column(Float, nullable=True)
    live_sharpe_ratio = Column(Float, nullable=True)
    live_max_drawdown = Column(Float, nullable=True)
    live_total_trades = Column(Integer, nullable=True)
    live_avg_trade_duration_hours = Column(Float, nullable=True)
    
    # Comparison analysis
    return_difference_percent = Column(Float, nullable=True)  # Live - Backtest
    correlation_coefficient = Column(Float, nullable=True)
    divergence_score = Column(Float, nullable=True)  # 0-100, higher = more divergent
    
    # Analysis results
    analysis_summary = Column(JSON, nullable=True)  # Detailed comparison analysis
    recommendations = Column(JSON, nullable=True)   # Suggested improvements
    
    # Status
    comparison_status = Column(String(50), default="pending")  # pending, completed, failed
    
    # Relationships
    strategy = relationship("Strategy")
    backtest_run = relationship("BacktestRun")
    
    # Table constraints
    __table_args__ = (
        CheckConstraint('comparison_period_start < comparison_period_end', name='check_comparison_date_range'),
        Index('idx_backtest_comparisons_strategy', 'strategy_id'),
        Index('idx_backtest_comparisons_period', 'comparison_period_start', 'comparison_period_end'),
    )
    
    @property
    def has_significant_divergence(self) -> bool:
        """Check if there's significant divergence between backtest and live results."""
        if self.divergence_score is None:
            return False
        return self.divergence_score > 25.0  # Threshold for significant divergence
    
    def calculate_divergence_score(self) -> float:
        """Calculate overall divergence score between backtest and live performance."""
        if not all([self.backtest_total_return, self.live_total_return, 
                   self.backtest_win_rate, self.live_win_rate]):
            return 0.0
        
        # Calculate divergence based on key metrics
        return_diff = abs(self.backtest_total_return - self.live_total_return)
        win_rate_diff = abs(self.backtest_win_rate - self.live_win_rate)
        
        # Weighted divergence score (0-100)
        divergence = (return_diff * 0.6 + win_rate_diff * 0.4)
        return min(100.0, divergence)

