"""
Paper Trading Analytics Service

This service provides analytics calculations for paper trading,
leveraging existing Trade and Strategy models.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_
import numpy as np
from collections import defaultdict

from app.models.strategy import (
    Strategy, Trade, Signal, SetupQualityGrade
)
from app.core.database import get_redis_db, get_timescale_db
from app.core.error_handling import OperationalError

# Set up logging
logger = logging.getLogger(__name__)


class AnalyticsService:
    """
    Analytics service for paper trading metrics and calculations.
    
    Provides hedge fund style analytics including NAV, risk metrics,
    and performance attribution using existing infrastructure.
    """
    
    def __init__(self, db: Session, initial_capital: float = 1000000.0):
        """
        Initialize analytics service.
        
        Args:
            db: SQLAlchemy database session
            initial_capital: Starting capital in INR (default 10 lakhs)
        """
        self.db = db
        self.initial_capital = initial_capital
        self.redis = None
        try:
            self.redis = get_redis_db()
        except Exception as e:
            logger.warning(f"Redis not available: {e}. Will work without caching.")
    
    # =====================
    # Live Metrics Methods
    # =====================
    
    def calculate_nav(self, strategy_id: int) -> float:
        """
        Calculate current Net Asset Value including open positions.
        
        Args:
            strategy_id: Strategy ID to calculate NAV for
            
        Returns:
            Current NAV in INR
        """
        # Get initial capital
        nav = self.initial_capital
        
        # Add realized P&L from closed trades
        closed_trades = self.db.query(Trade).filter(
            Trade.strategy_id == strategy_id,
            Trade.exit_price.isnot(None)
        ).all()
        
        for trade in closed_trades:
            # Use the existing profit_loss_inr field
            if trade.profit_loss_inr:
                nav += trade.profit_loss_inr
            # Subtract total costs (commission + taxes)
            if hasattr(trade, 'total_costs') and trade.total_costs:
                nav -= trade.total_costs
        
        # Add unrealized P&L from open positions
        open_trades = self.db.query(Trade).filter(
            Trade.strategy_id == strategy_id,
            Trade.exit_price.is_(None)
        ).all()
        
        for trade in open_trades:
            # For now, assume current price = entry price (no unrealized P&L)
            # In real implementation, you'd fetch current market price
            unrealized_pnl = 0  # Placeholder
            nav += unrealized_pnl
        
        # Cache in Redis if available
        if self.redis:
            try:
                self.redis.set(f"analytics:nav:{strategy_id}", str(nav), ex=60)
            except Exception as e:
                logger.debug(f"Redis cache failed: {e}")
        
        return nav
    
    def get_open_positions_value(self, strategy_id: int) -> float:
        """
        Calculate total value of open positions.
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            Total position value in INR
        """
        open_trades = self.db.query(Trade).filter(
            Trade.strategy_id == strategy_id,
            Trade.exit_price.is_(None)
        ).all()
        
        total_value = 0.0
        for trade in open_trades:
            # Using the lot_size from existing code (50 INR per point)
            lot_size = 50
            position_value = trade.entry_price * trade.position_size * lot_size
            total_value += position_value
        
        return total_value
    
    def get_live_metrics(self, strategy_id: int) -> Dict[str, Any]:
        """
        Get comprehensive live metrics for a strategy.
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            Dictionary of live metrics
        """
        try:
            # Calculate NAV
            nav = self.calculate_nav(strategy_id)
            
            # Get cash balance (NAV - open positions value)
            positions_value = self.get_open_positions_value(strategy_id)
            cash = nav - positions_value
            
            # Calculate P&L components
            realized_pnl = nav - self.initial_capital
            unrealized_pnl = 0.0  # Placeholder for current implementation
            
            # Get trade counts
            total_trades = self.db.query(Trade).filter(
                Trade.strategy_id == strategy_id
            ).count()
            
            active_trades = self.db.query(Trade).filter(
                Trade.strategy_id == strategy_id,
                Trade.exit_price.is_(None)
            ).count()
            
            # Calculate daily metrics (placeholder)
            daily_pnl = 0.0
            daily_return = 0.0
            
            # Calculate drawdown metrics
            high_water_mark = max(nav, self.initial_capital)
            current_drawdown = ((nav - high_water_mark) / high_water_mark) * 100 if high_water_mark > 0 else 0
            
            # Get today's trades count
            today = datetime.utcnow().date()
            trades_today = self.db.query(Trade).filter(
                Trade.strategy_id == strategy_id,
                func.date(Trade.entry_time) == today
            ).count()
            
            return {
                "nav": nav,
                "cash": cash,
                "positions_value": positions_value,
                "realized_pnl": realized_pnl,
                "unrealized_pnl": unrealized_pnl,
                "daily_pnl": daily_pnl,
                "daily_return": daily_return,
                "current_drawdown": current_drawdown,
                "high_water_mark": high_water_mark,
                "active_trades": active_trades,
                "total_trades": total_trades,
                "trades_today": trades_today
            }
            
        except Exception as e:
            logger.error(f"Error calculating live metrics for strategy {strategy_id}: {e}")
            raise OperationalError(f"Failed to calculate live metrics: {str(e)}")
    
    # =====================
    # Equity Curve Methods
    # =====================
    
    def get_equity_curve(self, strategy_id: int, period: str = "1M") -> List[Dict[str, Any]]:
        """
        Get historical equity curve data.
        
        Args:
            strategy_id: Strategy ID
            period: Time period (1D, 1W, 1M, 3M, 1Y, ALL)
            
        Returns:
            List of equity curve points
        """
        try:
            # Determine date range based on period
            end_date = datetime.utcnow()
            if period == "1D":
                start_date = end_date - timedelta(days=1)
            elif period == "1W":
                start_date = end_date - timedelta(weeks=1)
            elif period == "1M":
                start_date = end_date - timedelta(days=30)
            elif period == "3M":
                start_date = end_date - timedelta(days=90)
            elif period == "1Y":
                start_date = end_date - timedelta(days=365)
            else:  # ALL
                start_date = datetime(2020, 1, 1)  # Fallback start date
            
            # Get trades within period
            trades = self.db.query(Trade).filter(
                Trade.strategy_id == strategy_id,
                Trade.entry_time >= start_date,
                Trade.entry_time <= end_date
            ).order_by(Trade.entry_time).all()
            
            # Build equity curve
            equity_curve = []
            running_nav = self.initial_capital
            
            # Add starting point
            equity_curve.append({
                "timestamp": start_date,
                "nav": running_nav,
                "trade_count": 0,
                "realized_pnl": 0.0
            })
            
            # Process each trade
            for trade in trades:
                if trade.profit_loss_inr:
                    running_nav += trade.profit_loss_inr
                
                # Subtract costs if available
                if hasattr(trade, 'total_costs') and trade.total_costs:
                    running_nav -= trade.total_costs
                
                equity_curve.append({
                    "timestamp": trade.entry_time,
                    "nav": running_nav,
                    "trade_count": len([t for t in trades if t.entry_time <= trade.entry_time]),
                    "realized_pnl": running_nav - self.initial_capital
                })
            
            return equity_curve
            
        except Exception as e:
            logger.error(f"Error getting equity curve for strategy {strategy_id}: {e}")
            raise OperationalError(f"Failed to get equity curve: {str(e)}")
    
    # =====================
    # Drawdown Analysis
    # =====================
    
    def calculate_drawdown(self, equity_curve: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate drawdown analysis from equity curve.
        
        Args:
            equity_curve: List of equity curve points
            
        Returns:
            Drawdown analysis dictionary
        """
        if not equity_curve:
            return {
                "max_drawdown": 0.0,
                "max_drawdown_duration_days": 0.0,
                "drawdown_periods": [],
                "average_drawdown": 0.0,
                "average_recovery_days": 0.0
            }
        
        # Calculate running maximum and drawdowns
        navs = [point["nav"] for point in equity_curve]
        timestamps = [point["timestamp"] for point in equity_curve]
        
        running_max = []
        drawdowns = []
        current_max = navs[0]
        
        for nav in navs:
            current_max = max(current_max, nav)
            running_max.append(current_max)
            
            # Calculate drawdown percentage
            if current_max > 0:
                drawdown = ((nav - current_max) / current_max) * 100
            else:
                drawdown = 0
            drawdowns.append(drawdown)
        
        # Find maximum drawdown
        max_drawdown = min(drawdowns) if drawdowns else 0
        
        # Find drawdown periods (simplified)
        drawdown_periods = []
        in_drawdown = False
        start_idx = 0
        
        for i, dd in enumerate(drawdowns):
            if dd < -0.1 and not in_drawdown:  # Start of drawdown (>0.1% loss)
                in_drawdown = True
                start_idx = i
            elif dd >= 0 and in_drawdown:  # End of drawdown
                in_drawdown = False
                if i > start_idx:
                    duration_days = (timestamps[i] - timestamps[start_idx]).days
                    max_dd_in_period = min(drawdowns[start_idx:i+1])
                    drawdown_periods.append({
                        "start_date": timestamps[start_idx],
                        "end_date": timestamps[i],
                        "duration_days": duration_days,
                        "max_drawdown_percent": max_dd_in_period
                    })
        
        # Calculate averages
        if drawdown_periods:
            avg_drawdown = sum(p["max_drawdown_percent"] for p in drawdown_periods) / len(drawdown_periods)
            avg_recovery = sum(p["duration_days"] for p in drawdown_periods) / len(drawdown_periods)
        else:
            avg_drawdown = 0.0
            avg_recovery = 0.0
        
        # Calculate maximum drawdown duration
        max_duration = max([p["duration_days"] for p in drawdown_periods]) if drawdown_periods else 0
        
        return {
            "max_drawdown": max_drawdown,
            "max_drawdown_duration_days": max_duration,
            "drawdown_periods": drawdown_periods,
            "average_drawdown": avg_drawdown,
            "average_recovery_days": avg_recovery,
            "fastest_recovery_days": min([p["duration_days"] for p in drawdown_periods]) if drawdown_periods else None,
            "slowest_recovery_days": max([p["duration_days"] for p in drawdown_periods]) if drawdown_periods else None,
            "current_drawdown_days": None  # Would need current state calculation
        }
    
    def _calculate_current_drawdown(self, strategy_id: int, current_nav: float) -> Dict[str, Any]:
        """
        Calculate current drawdown state.
        
        Args:
            strategy_id: Strategy ID
            current_nav: Current NAV
            
        Returns:
            Current drawdown data
        """
        # Get historical high water mark (simplified)
        high_water_mark = max(current_nav, self.initial_capital)
        
        # Calculate current drawdown
        if high_water_mark > 0:
            drawdown_percent = ((current_nav - high_water_mark) / high_water_mark) * 100
        else:
            drawdown_percent = 0.0
        
        return {
            "high_water_mark": high_water_mark,
            "drawdown_percent": drawdown_percent
        }
    
    # =====================
    # Risk Metrics Methods
    # =====================
    
    def calculate_risk_metrics(self, strategy_id: int, var_confidence: float = 0.95, 
                             lookback_days: int = 20) -> Dict[str, Any]:
        """
        Calculate comprehensive risk metrics.
        
        Args:
            strategy_id: Strategy ID
            var_confidence: VaR confidence level
            lookback_days: Days to look back for calculations
            
        Returns:
            Risk metrics dictionary
        """
        try:
            # Get recent equity curve for calculations
            equity_curve = self.get_equity_curve(strategy_id, "ALL")
            
            if len(equity_curve) < 2:
                return self._default_risk_metrics()
            
            # Calculate daily returns
            daily_returns = []
            for i in range(1, len(equity_curve)):
                prev_nav = equity_curve[i-1]["nav"]
                curr_nav = equity_curve[i]["nav"]
                if prev_nav > 0:
                    daily_return = ((curr_nav - prev_nav) / prev_nav) * 100
                    daily_returns.append(daily_return)
            
            if not daily_returns:
                return self._default_risk_metrics()
            
            # Calculate metrics
            daily_returns_array = np.array(daily_returns)
            
            # Volatility (annualized)
            volatility = np.std(daily_returns_array) * np.sqrt(252)
            
            # Sharpe ratio (simplified, assuming 6% risk-free rate)
            risk_free_rate = 6.0
            avg_return = np.mean(daily_returns_array) * 252  # Annualized
            sharpe_ratio = (avg_return - risk_free_rate) / volatility if volatility > 0 else 0
            
            # Sortino ratio (downside deviation)
            downside_returns = daily_returns_array[daily_returns_array < 0]
            downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = (avg_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
            
            # Value at Risk
            var_value = np.percentile(daily_returns_array, (1 - var_confidence) * 100) if len(daily_returns_array) > 0 else 0
            
            # Maximum daily loss
            max_daily_loss = np.min(daily_returns_array) if len(daily_returns_array) > 0 else 0
            
            # Beta (simplified, assume beta = 1 for now)
            beta = 1.0
            
            return {
                "annualized_volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "var_95": var_value,
                "var_99": np.percentile(daily_returns_array, 1) if len(daily_returns_array) > 0 else 0,
                "max_daily_loss": max_daily_loss,
                "downside_deviation": downside_deviation,
                "beta": beta,
                "positive_days": len([r for r in daily_returns if r > 0]),
                "negative_days": len([r for r in daily_returns if r < 0]),
                "win_rate": len([r for r in daily_returns if r > 0]) / len(daily_returns) * 100 if daily_returns else 0,
                "average_win": np.mean([r for r in daily_returns if r > 0]) if any(r > 0 for r in daily_returns) else 0,
                "average_loss": np.mean([r for r in daily_returns if r < 0]) if any(r < 0 for r in daily_returns) else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return self._default_risk_metrics()
    
    def _default_risk_metrics(self) -> Dict[str, Any]:
        """Return default risk metrics when calculation fails."""
        return {
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "var_95": 0.0,
            "var_99": 0.0,
            "max_daily_loss": 0.0,
            "downside_deviation": 0.0,
            "beta": 1.0,
            "positive_days": 0,
            "negative_days": 0,
            "win_rate": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0
        }
    
    # =====================
    # Attribution Analysis
    # =====================
    
    def calculate_attribution(self, strategy_id: int, attribution_type: str = "grade") -> Dict[str, Any]:
        """
        Calculate performance attribution analysis.
        
        Args:
            strategy_id: Strategy ID
            attribution_type: Type of attribution (grade, time, market_state)
            
        Returns:
            Attribution analysis dictionary
        """
        try:
            # Get all closed trades for the strategy
            trades = self.db.query(Trade).filter(
                Trade.strategy_id == strategy_id,
                Trade.exit_price.isnot(None),
                Trade.profit_loss_inr.isnot(None)
            ).all()
            
            if not trades:
                return {
                    "attribution_by_category": [],
                    "total_pnl": 0.0,
                    "total_trades": 0
                }
            
            if attribution_type == "grade":
                return self._calculate_grade_attribution(trades)
            elif attribution_type == "time":
                return self._calculate_time_attribution(trades)
            else:
                return self._calculate_market_state_attribution(trades)
                
        except Exception as e:
            logger.error(f"Error calculating attribution: {e}")
            raise OperationalError(f"Failed to calculate attribution: {str(e)}")
    
    def _calculate_grade_attribution(self, trades: List[Trade]) -> Dict[str, Any]:
        """Calculate attribution by setup quality grade."""
        grade_stats = defaultdict(lambda: {"pnl": 0.0, "trades": 0, "win_rate": 0.0})
        
        for trade in trades:
            grade = trade.setup_quality or "Unknown"
            grade_stats[grade]["pnl"] += trade.profit_loss_inr or 0
            grade_stats[grade]["trades"] += 1
            
            # Calculate win rate
            if trade.profit_loss_inr and trade.profit_loss_inr > 0:
                grade_stats[grade]["wins"] = grade_stats[grade].get("wins", 0) + 1
        
        # Convert to list format
        attribution_list = []
        total_pnl = sum(stats["pnl"] for stats in grade_stats.values())
        
        for grade, stats in grade_stats.items():
            wins = stats.get("wins", 0)
            win_rate = (wins / stats["trades"]) * 100 if stats["trades"] > 0 else 0
            
            attribution_list.append({
                "category": grade,
                "pnl": stats["pnl"],
                "trade_count": stats["trades"],
                "win_rate": win_rate,
                "contribution_percent": (stats["pnl"] / total_pnl) * 100 if total_pnl != 0 else 0,
                "average_pnl_per_trade": stats["pnl"] / stats["trades"] if stats["trades"] > 0 else 0
            })
        
        return {
            "attribution_by_category": attribution_list,
            "total_pnl": total_pnl,
            "total_trades": len(trades)
        }
    
    def _calculate_time_attribution(self, trades: List[Trade]) -> Dict[str, Any]:
        """Calculate attribution by time periods."""
        time_stats = defaultdict(lambda: {"pnl": 0.0, "trades": 0})
        
        for trade in trades:
            # Group by month for time attribution
            if trade.entry_time:
                time_key = trade.entry_time.strftime("%Y-%m")
                time_stats[time_key]["pnl"] += trade.profit_loss_inr or 0
                time_stats[time_key]["trades"] += 1
        
        # Convert to list format
        attribution_list = []
        total_pnl = sum(stats["pnl"] for stats in time_stats.values())
        
        for time_period, stats in time_stats.items():
            attribution_list.append({
                "category": time_period,
                "pnl": stats["pnl"],
                "trade_count": stats["trades"],
                "contribution_percent": (stats["pnl"] / total_pnl) * 100 if total_pnl != 0 else 0,
                "average_pnl_per_trade": stats["pnl"] / stats["trades"] if stats["trades"] > 0 else 0
            })
        
        return {
            "attribution_by_category": attribution_list,
            "total_pnl": total_pnl,
            "total_trades": len(trades)
        }
    
    def _calculate_market_state_attribution(self, trades: List[Trade]) -> Dict[str, Any]:
        """Calculate attribution by market state (placeholder)."""
        # Simplified market state attribution
        return {
            "attribution_by_category": [
                {
                    "category": "All Market States",
                    "pnl": sum(trade.profit_loss_inr or 0 for trade in trades),
                    "trade_count": len(trades),
                    "contribution_percent": 100.0,
                    "average_pnl_per_trade": sum(trade.profit_loss_inr or 0 for trade in trades) / len(trades) if trades else 0
                }
            ],
            "total_pnl": sum(trade.profit_loss_inr or 0 for trade in trades),
            "total_trades": len(trades)
        }
    
    # =====================
    # Benchmark Comparison
    # =====================
    
    def compare_to_benchmark(self, strategy_id: int, benchmark: str = "NIFTY") -> Dict[str, Any]:
        """
        Compare strategy performance to a benchmark.
        
        Args:
            strategy_id: Strategy ID
            benchmark: Benchmark symbol
            
        Returns:
            Comparison metrics dictionary
        """
        try:
            # Get strategy performance
            equity_curve = self.get_equity_curve(strategy_id, "ALL")
            
            if len(equity_curve) < 2:
                return self._default_benchmark_comparison()
            
            # Calculate strategy return
            initial_nav = equity_curve[0]["nav"]
            final_nav = equity_curve[-1]["nav"]
            strategy_return = ((final_nav - initial_nav) / initial_nav) * 100 if initial_nav > 0 else 0
            
            # Placeholder benchmark return (would fetch real data in production)
            benchmark_return = 12.0  # Assume 12% annual return for NIFTY
            
            # Calculate excess return
            excess_return = strategy_return - benchmark_return
            
            # Calculate tracking error (simplified)
            tracking_error = 5.0  # Placeholder
            
            # Information ratio
            information_ratio = excess_return / tracking_error if tracking_error > 0 else 0
            
            return {
                "strategy_return": strategy_return,
                "benchmark_return": benchmark_return,
                "excess_return": excess_return,
                "tracking_error": tracking_error,
                "information_ratio": information_ratio,
                "beta": 1.0,  # Placeholder
                "alpha": excess_return,  # Simplified alpha calculation
                "correlation": 0.7  # Placeholder correlation
            }
            
        except Exception as e:
            logger.error(f"Error comparing to benchmark: {e}")
            return self._default_benchmark_comparison()
    
    def _default_benchmark_comparison(self) -> Dict[str, Any]:
        """Return default benchmark comparison when calculation fails."""
        return {
            "strategy_return": 0.0,
            "benchmark_return": 0.0,
            "excess_return": 0.0,
            "tracking_error": 0.0,
            "information_ratio": 0.0,
            "beta": 1.0,
            "alpha": 0.0,
            "correlation": 0.0
        }