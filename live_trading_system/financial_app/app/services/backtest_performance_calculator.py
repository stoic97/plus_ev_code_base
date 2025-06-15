"""
Backtest Performance Calculator Service

Extends the existing AnalyticsService to provide backtest-specific performance calculations.
Calculates comprehensive metrics for historical backtests while reusing existing infrastructure.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from collections import defaultdict

from app.services.analytics_service import AnalyticsService
from app.services.backtesting_engine import BacktestTrade, BacktestConfig, BacktestMetrics
from app.core.error_handling import OperationalError

# Set up logging
logger = logging.getLogger(__name__)


class BacktestPerformanceCalculator:
    """
    Performance calculator for backtests, extending existing analytics infrastructure.
    
    Provides comprehensive backtest metrics while reusing AnalyticsService methods
    for consistent calculations between live and backtest performance.
    """
    
    def __init__(self, analytics_service: AnalyticsService):
        """
        Initialize backtest performance calculator.
        
        Args:
            analytics_service: Existing analytics service instance for method reuse
        """
        self.analytics_service = analytics_service
        self.db = analytics_service.db
        self.initial_capital = analytics_service.initial_capital
    
    # ==============================
    # Primary Performance Calculation
    # ==============================
    
    def calculate_comprehensive_metrics(
        self, 
        trades: List[BacktestTrade], 
        config: BacktestConfig,
        equity_curve: pd.DataFrame
    ) -> BacktestMetrics:
        """
        Calculate comprehensive backtest performance metrics.
        
        Args:
            trades: List of backtest trades
            config: Backtest configuration
            equity_curve: Equity curve DataFrame with columns ['timestamp', 'equity', 'drawdown']
            
        Returns:
            BacktestMetrics object with all calculated performance metrics
        """
        try:
            logger.info(f"Calculating comprehensive metrics for {len(trades)} trades")
            
            # Initialize metrics object
            metrics = BacktestMetrics()
            
            # Basic trade metrics
            self._calculate_basic_metrics(trades, config, metrics)
            
            # Risk-adjusted metrics using existing analytics methods
            self._calculate_risk_metrics(trades, config, equity_curve, metrics)
            
            # Drawdown analysis using existing methods
            self._calculate_drawdown_metrics(equity_curve, metrics)
            
            # Time-based analysis
            self._calculate_time_metrics(trades, config, metrics)
            
            # Cost analysis
            self._calculate_cost_metrics(trades, config, metrics)
            
            # Portfolio metrics
            self._calculate_portfolio_metrics(trades, config, equity_curve, metrics)
            
            logger.info(f"Metrics calculation complete: {metrics.total_trades} trades, "
                       f"{metrics.total_return_pct:.2f}% return")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive metrics: {e}")
            raise OperationalError(f"Failed to calculate backtest metrics: {str(e)}")
    
    # ==============================
    # Basic Metrics Calculation
    # ==============================
    
    def _calculate_basic_metrics(
        self, 
        trades: List[BacktestTrade], 
        config: BacktestConfig,
        metrics: BacktestMetrics
    ) -> None:
        """Calculate basic trade and return metrics."""
        # Filter closed trades
        closed_trades = [t for t in trades if t.exit_time is not None and t.pnl_inr is not None]
        metrics.total_trades = len(closed_trades)
        
        if not closed_trades:
            logger.warning("No closed trades found for metrics calculation")
            return
        
        # P&L calculations
        total_pnl = sum(t.pnl_inr for t in closed_trades)
        metrics.total_pnl_inr = total_pnl
        metrics.total_return_pct = (total_pnl / config.initial_capital) * 100
        
        # Win/Loss analysis
        winning_trades = [t for t in closed_trades if t.pnl_inr > 0]
        losing_trades = [t for t in closed_trades if t.pnl_inr <= 0]
        
        metrics.winning_trades = len(winning_trades)
        metrics.losing_trades = len(losing_trades)
        metrics.win_rate_pct = (len(winning_trades) / len(closed_trades)) * 100
        
        # Average metrics
        if winning_trades:
            metrics.avg_win_inr = sum(t.pnl_inr for t in winning_trades) / len(winning_trades)
            metrics.largest_win_inr = max(t.pnl_inr for t in winning_trades)
        
        if losing_trades:
            metrics.avg_loss_inr = sum(t.pnl_inr for t in losing_trades) / len(losing_trades)
            metrics.largest_loss_inr = min(t.pnl_inr for t in losing_trades)
        
        # Profit factor
        total_wins = sum(t.pnl_inr for t in winning_trades) if winning_trades else 0
        total_losses = abs(sum(t.pnl_inr for t in losing_trades)) if losing_trades else 1
        metrics.profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Expectancy
        metrics.expectancy_inr = total_pnl / len(closed_trades)
        
        logger.debug(f"Basic metrics calculated: {metrics.winning_trades} wins, "
                    f"{metrics.losing_trades} losses, {metrics.win_rate_pct:.1f}% win rate")
    
    # ==============================
    # Risk Metrics Using Existing Analytics
    # ==============================
    
    def _calculate_risk_metrics(
        self, 
        trades: List[BacktestTrade], 
        config: BacktestConfig,
        equity_curve: pd.DataFrame,
        metrics: BacktestMetrics
    ) -> None:
        """Calculate risk-adjusted metrics using existing analytics methods."""
        if len(trades) < 2:
            logger.warning("Not enough trades for risk metrics calculation")
            return
        
        try:
            # Convert equity curve to daily returns
            returns = self._calculate_daily_returns(equity_curve)
            
            if not returns:
                logger.warning("No returns calculated for risk metrics")
                return
            
            # Use existing analytics methods for consistency
            metrics.sharpe_ratio = self.analytics_service.calculate_sharpe_ratio(
                returns, config.risk_free_rate
            )
            
            # Calculate Sortino ratio using existing helper
            metrics.sortino_ratio = self.analytics_service._calculate_sortino_ratio(returns)
            
            # Calculate Calmar ratio (needs max drawdown)
            equity_curve_list = [
                {"nav": row["equity"], "timestamp": row["timestamp"]} 
                for _, row in equity_curve.iterrows()
            ]
            drawdown_data = self.analytics_service.calculate_drawdown(equity_curve_list)
            metrics.calmar_ratio = self.analytics_service._calculate_calmar_ratio(
                returns, drawdown_data["max_drawdown"]
            )
            
            # Calculate additional volatility metrics
            returns_array = np.array(returns)
            metrics.volatility_daily_pct = float(returns_array.std() * 100)
            metrics.volatility_annual_pct = float(returns_array.std() * np.sqrt(252) * 100)
            
            # Calculate VaR using trade-based approach (backtest-specific)
            metrics.var_95_inr = self._calculate_backtest_var(trades, 0.95)
            metrics.var_99_inr = self._calculate_backtest_var(trades, 0.99)
            
            logger.debug(f"Risk metrics calculated: Sharpe {metrics.sharpe_ratio:.2f}, "
                        f"Sortino {metrics.sortino_ratio:.2f}")
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            # Continue with other calculations even if risk metrics fail
    
    def _calculate_daily_returns(self, equity_curve: pd.DataFrame) -> List[float]:
        """Calculate daily returns from equity curve."""
        if len(equity_curve) < 2:
            return []
        
        returns = []
        prev_equity = equity_curve.iloc[0]["equity"]
        
        for _, row in equity_curve.iloc[1:].iterrows():
            curr_equity = row["equity"]
            daily_return = (curr_equity - prev_equity) / prev_equity
            returns.append(daily_return)
            prev_equity = curr_equity
        
        return returns
    
    def _calculate_backtest_var(self, trades: List[BacktestTrade], confidence: float) -> float:
        """Calculate Value at Risk for backtest trades."""
        closed_trades = [t for t in trades if t.pnl_inr is not None]
        
        if not closed_trades:
            return 0.0
        
        pnl_values = [t.pnl_inr for t in closed_trades]
        percentile = (1 - confidence) * 100
        var_value = np.percentile(pnl_values, percentile)
        
        return abs(float(var_value))
    
    # ==============================
    # Drawdown Analysis Using Existing Methods
    # ==============================
    
    def _calculate_drawdown_metrics(
        self, 
        equity_curve: pd.DataFrame,
        metrics: BacktestMetrics
    ) -> None:
        """Calculate drawdown metrics using existing analytics methods."""
        try:
            # Convert to format expected by existing method
            equity_curve_list = [
                {"nav": row["equity"], "timestamp": row["timestamp"]} 
                for _, row in equity_curve.iterrows()
            ]
            
            # Use existing drawdown calculation
            drawdown_data = self.analytics_service.calculate_drawdown(equity_curve_list)
            
            metrics.max_drawdown_pct = drawdown_data["max_drawdown"]
            metrics.max_drawdown_duration_days = drawdown_data["max_drawdown_duration_days"]
            metrics.current_drawdown_pct = drawdown_data["current_drawdown"]
            
            # Additional drawdown analysis
            drawdown_periods = drawdown_data.get("drawdown_periods", [])
            if drawdown_periods:
                durations = [period.get("duration_days", 0) for period in drawdown_periods]
                metrics.avg_drawdown_duration_days = sum(durations) / len(durations)
                metrics.num_drawdown_periods = len(drawdown_periods)
            
            logger.debug(f"Drawdown metrics: Max {metrics.max_drawdown_pct:.2f}%, "
                        f"Duration {metrics.max_drawdown_duration_days} days")
            
        except Exception as e:
            logger.error(f"Error calculating drawdown metrics: {e}")
    
    # ==============================
    # Time and Cost Analysis
    # ==============================
    
    def _calculate_time_metrics(
        self, 
        trades: List[BacktestTrade], 
        config: BacktestConfig,
        metrics: BacktestMetrics
    ) -> None:
        """Calculate time-based metrics."""
        closed_trades = [t for t in trades if t.exit_time is not None]
        
        if not closed_trades:
            return
        
        # Trade duration analysis
        durations = []
        for trade in closed_trades:
            if trade.entry_time and trade.exit_time:
                duration = (trade.exit_time - trade.entry_time).total_seconds() / 60
                durations.append(duration)
        
        if durations:
            metrics.avg_trade_duration_minutes = sum(durations) / len(durations)
            metrics.max_trade_duration_minutes = max(durations)
            metrics.min_trade_duration_minutes = min(durations)
        
        # Trading frequency
        period_days = (config.end_date - config.start_date).days
        metrics.trades_per_day = len(closed_trades) / max(period_days, 1)
        
        # Annual return calculation
        period_years = period_days / 365.0
        if period_years > 0 and metrics.total_return_pct > 0:
            metrics.annual_return_pct = ((1 + metrics.total_return_pct / 100) ** (1 / period_years) - 1) * 100
    
    def _calculate_cost_metrics(
        self, 
        trades: List[BacktestTrade], 
        config: BacktestConfig,
        metrics: BacktestMetrics
    ) -> None:
        """Calculate cost analysis metrics."""
        closed_trades = [t for t in trades if t.exit_time is not None]
        
        # Commission analysis
        metrics.total_commission_inr = sum(getattr(t, 'commission', 0) for t in closed_trades)
        
        # Slippage analysis (if available)
        total_slippage = sum(getattr(t, 'slippage', 0) * config.lot_size for t in closed_trades)
        metrics.total_slippage_inr = total_slippage
        
        # Total costs
        metrics.total_costs_inr = metrics.total_commission_inr + metrics.total_slippage_inr
        
        # Cost as percentage of capital
        if config.initial_capital > 0:
            metrics.costs_pct_of_capital = (metrics.total_costs_inr / config.initial_capital) * 100
    
    def _calculate_portfolio_metrics(
        self, 
        trades: List[BacktestTrade], 
        config: BacktestConfig,
        equity_curve: pd.DataFrame,
        metrics: BacktestMetrics
    ) -> None:
        """Calculate portfolio-level metrics."""
        if equity_curve.empty:
            return
        
        # Final portfolio value
        metrics.final_capital_inr = equity_curve.iloc[-1]["equity"]
        
        # Peak capital
        metrics.peak_capital_inr = equity_curve["equity"].max()
        
        # Capital efficiency (how much of capital was used)
        max_positions = self._calculate_max_concurrent_positions(trades)
        if max_positions > 0:
            estimated_max_exposure = max_positions * config.lot_size * 100  # Rough estimate
            metrics.capital_efficiency_pct = min(100, (estimated_max_exposure / config.initial_capital) * 100)
    
    def _calculate_max_concurrent_positions(self, trades: List[BacktestTrade]) -> int:
        """Calculate maximum number of concurrent positions."""
        events = []
        
        for trade in trades:
            if trade.entry_time:
                events.append((trade.entry_time, 1))  # Position opened
            if trade.exit_time:
                events.append((trade.exit_time, -1))  # Position closed
        
        events.sort(key=lambda x: x[0])  # Sort by time
        
        current_positions = 0
        max_positions = 0
        
        for _, change in events:
            current_positions += change
            max_positions = max(max_positions, current_positions)
        
        return max_positions
    
    # ==============================
    # Performance Comparison Methods
    # ==============================
    
    def compare_to_live_performance(
        self, 
        backtest_metrics: BacktestMetrics,
        strategy_id: int,
        comparison_period_days: int = 90
    ) -> Dict[str, Any]:
        """
        Compare backtest performance to live trading performance.
        
        Args:
            backtest_metrics: Calculated backtest metrics
            strategy_id: Strategy ID for live performance lookup
            comparison_period_days: Number of recent days to analyze for live performance
            
        Returns:
            Comparison analysis dictionary
        """
        try:
            logger.info(f"Comparing backtest to live performance for strategy {strategy_id}")
            
            # Get recent live performance using existing analytics
            cutoff_date = datetime.utcnow() - timedelta(days=comparison_period_days)
            live_metrics = self.analytics_service.calculate_risk_metrics(strategy_id)
            
            # Calculate live returns for the period
            live_equity_curve = self.analytics_service.get_equity_curve(strategy_id, "ALL")
            recent_curve = [
                point for point in live_equity_curve 
                if point["timestamp"] >= cutoff_date
            ]
            
            comparison = {
                "backtest_period": {
                    "sharpe_ratio": backtest_metrics.sharpe_ratio,
                    "sortino_ratio": backtest_metrics.sortino_ratio,
                    "max_drawdown_pct": backtest_metrics.max_drawdown_pct,
                    "win_rate_pct": backtest_metrics.win_rate_pct,
                    "profit_factor": backtest_metrics.profit_factor,
                    "total_return_pct": backtest_metrics.total_return_pct
                },
                "live_period": {
                    "sharpe_ratio": live_metrics.get("sharpe_ratio", 0),
                    "sortino_ratio": live_metrics.get("sortino_ratio", 0),
                    "max_drawdown_pct": live_metrics.get("max_drawdown", 0),
                    "current_drawdown_pct": live_metrics.get("current_drawdown", 0)
                },
                "analysis": {
                    "performance_correlation": "high" if abs(backtest_metrics.sharpe_ratio - live_metrics.get("sharpe_ratio", 0)) < 0.5 else "low",
                    "risk_consistency": "good" if abs(backtest_metrics.max_drawdown_pct - live_metrics.get("max_drawdown", 0)) < 5 else "variable",
                    "comparison_period_days": comparison_period_days,
                    "live_data_points": len(recent_curve)
                }
            }
            
            logger.info(f"Performance comparison complete: "
                       f"Backtest Sharpe {backtest_metrics.sharpe_ratio:.2f} vs "
                       f"Live Sharpe {live_metrics.get('sharpe_ratio', 0):.2f}")
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing performance: {e}")
            return {
                "error": f"Failed to compare performance: {str(e)}",
                "backtest_metrics_available": True,
                "live_metrics_available": False
            }
    
    # ==============================
    # Utility Methods
    # ==============================
    
    def generate_performance_summary(self, metrics: BacktestMetrics) -> Dict[str, Any]:
        """Generate a concise performance summary for dashboards."""
        return {
            "overview": {
                "total_return_pct": round(metrics.total_return_pct, 2),
                "annual_return_pct": round(metrics.annual_return_pct or 0, 2),
                "max_drawdown_pct": round(metrics.max_drawdown_pct, 2),
                "sharpe_ratio": round(metrics.sharpe_ratio or 0, 2),
                "sortino_ratio": round(metrics.sortino_ratio or 0, 2),
                "profit_factor": round(metrics.profit_factor, 2)
            },
            "trading_stats": {
                "total_trades": metrics.total_trades,
                "win_rate_pct": round(metrics.win_rate_pct, 1),
                "avg_win_inr": round(metrics.avg_win_inr, 0),
                "avg_loss_inr": round(metrics.avg_loss_inr, 0),
                "expectancy_inr": round(metrics.expectancy_inr, 0)
            },
            "risk_analysis": {
                "var_95_inr": round(metrics.var_95_inr, 0),
                "volatility_annual_pct": round(metrics.volatility_annual_pct or 0, 2),
                "max_drawdown_duration_days": metrics.max_drawdown_duration_days,
                "current_drawdown_pct": round(metrics.current_drawdown_pct, 2)
            },
            "costs": {
                "total_costs_inr": round(metrics.total_costs_inr, 0),
                "costs_pct_of_capital": round(metrics.costs_pct_of_capital or 0, 3)
            }
        }