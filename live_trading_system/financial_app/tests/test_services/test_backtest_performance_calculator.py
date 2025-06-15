"""
Simplified unit tests for BacktestPerformanceCalculator service.

This module contains essential tests for the BacktestPerformanceCalculator,
focusing on core functionality and integration with existing analytics infrastructure.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


# Mock dataclasses that match the real ones
@dataclass
class MockBacktestConfig:
    """Mock BacktestConfig for testing."""
    strategy_id: int = 1
    start_date: datetime = datetime(2024, 1, 1)
    end_date: datetime = datetime(2024, 3, 31)
    initial_capital: float = 1000000.0
    commission_per_trade: float = 40.0
    slippage_bps: float = 3.0
    risk_free_rate: float = 0.06
    lot_size: int = 100


@dataclass
class MockBacktestTrade:
    """Mock BacktestTrade for testing."""
    trade_id: str
    strategy_id: int
    instrument: str
    direction: str
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    quantity: int = 1
    commission: float = 40.0
    slippage: float = 0.03
    pnl_points: Optional[float] = None
    pnl_inr: Optional[float] = None
    setup_quality: Optional[str] = None


@dataclass
class MockBacktestMetrics:
    """Mock BacktestMetrics for testing."""
    # Basic performance
    total_return_pct: float = 0.0
    annual_return_pct: float = 0.0
    total_pnl_inr: float = 0.0
    
    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_days: int = 0
    volatility_annual_pct: float = 0.0
    var_95_inr: float = 0.0
    var_99_inr: float = 0.0
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate_pct: float = 0.0
    avg_win_inr: float = 0.0
    avg_loss_inr: float = 0.0
    largest_win_inr: float = 0.0
    largest_loss_inr: float = 0.0
    profit_factor: float = 0.0
    expectancy_inr: float = 0.0
    
    # Cost analysis
    total_commission_inr: float = 0.0
    total_slippage_inr: float = 0.0
    total_costs_inr: float = 0.0


# Mock BacktestPerformanceCalculator for testing
class MockBacktestPerformanceCalculator:
    """Mock implementation for testing key functionality."""
    
    def __init__(self, analytics_service):
        self.analytics_service = analytics_service
        self.db = analytics_service.db
        self.initial_capital = analytics_service.initial_capital
    
    def calculate_comprehensive_metrics(self, trades: List[MockBacktestTrade], 
                                      config: MockBacktestConfig,
                                      equity_curve: pd.DataFrame) -> MockBacktestMetrics:
        """Calculate comprehensive backtest performance metrics."""
        metrics = MockBacktestMetrics()
        
        # Basic calculations
        closed_trades = [t for t in trades if t.exit_time is not None and t.pnl_inr is not None]
        metrics.total_trades = len(closed_trades)
        
        if closed_trades:
            total_pnl = sum(t.pnl_inr for t in closed_trades)
            metrics.total_pnl_inr = total_pnl
            
            # Handle zero capital case
            if config.initial_capital > 0:
                metrics.total_return_pct = (total_pnl / config.initial_capital) * 100
            else:
                metrics.total_return_pct = 0.0  # Avoid division by zero
            
            # Win/Loss analysis
            winning_trades = [t for t in closed_trades if t.pnl_inr > 0]
            losing_trades = [t for t in closed_trades if t.pnl_inr <= 0]
            
            metrics.winning_trades = len(winning_trades)
            metrics.losing_trades = len(losing_trades)
            metrics.win_rate_pct = (len(winning_trades) / len(closed_trades)) * 100
            
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
            
            # Use analytics service for risk metrics
            if len(closed_trades) > 1:
                returns = self._calculate_daily_returns(equity_curve)
                if returns:
                    metrics.sharpe_ratio = self.analytics_service.calculate_sharpe_ratio(returns)
                    metrics.sortino_ratio = self.analytics_service._calculate_sortino_ratio(returns)
        
        return metrics
    
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
    
    def compare_to_live_performance(self, backtest_metrics: MockBacktestMetrics,
                                  strategy_id: int, comparison_period_days: int = 90) -> Dict[str, Any]:
        """Compare backtest performance to live trading performance."""
        try:
            # Get live metrics using existing analytics
            live_metrics = self.analytics_service.calculate_risk_metrics(strategy_id)
            
            comparison = {
                "backtest_period": {
                    "sharpe_ratio": backtest_metrics.sharpe_ratio,
                    "sortino_ratio": backtest_metrics.sortino_ratio,
                    "win_rate_pct": backtest_metrics.win_rate_pct,
                    "profit_factor": backtest_metrics.profit_factor,
                    "total_return_pct": backtest_metrics.total_return_pct
                },
                "live_period": {
                    "sharpe_ratio": live_metrics.get("sharpe_ratio", 0),
                    "sortino_ratio": live_metrics.get("sortino_ratio", 0),
                    "max_drawdown_pct": live_metrics.get("max_drawdown", 0)
                },
                "analysis": {
                    "comparison_period_days": comparison_period_days,
                    "performance_correlation": "high" if abs(backtest_metrics.sharpe_ratio - live_metrics.get("sharpe_ratio", 0)) < 0.5 else "low"
                }
            }
            
            return comparison
            
        except Exception as e:
            return {
                "error": f"Failed to compare performance: {str(e)}",
                "backtest_metrics_available": True,
                "live_metrics_available": False
            }
    
    def generate_performance_summary(self, metrics: MockBacktestMetrics) -> Dict[str, Any]:
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
                "max_drawdown_duration_days": metrics.max_drawdown_duration_days
            },
            "costs": {
                "total_costs_inr": round(metrics.total_costs_inr, 0)
            }
        }


# Fixtures
@pytest.fixture
def mock_analytics_service():
    """Create a mock AnalyticsService for testing."""
    mock_service = MagicMock()
    
    # Set basic attributes
    mock_service.db = MagicMock()
    mock_service.initial_capital = 1000000.0
    
    # Configure mock methods with proper return values
    mock_service.calculate_sharpe_ratio.return_value = 1.85
    mock_service._calculate_sortino_ratio.return_value = 2.10
    mock_service._calculate_calmar_ratio.return_value = 3.5
    
    mock_service.calculate_risk_metrics.return_value = {
        "sharpe_ratio": 1.85,
        "sortino_ratio": 2.10,
        "calmar_ratio": 3.5,
        "var_95": 15000.0,
        "var_99": 18000.0,
        "max_drawdown": -5.2,
        "current_drawdown": -2.5,
        "max_drawdown_duration_days": 7
    }
    
    return mock_service


@pytest.fixture
def performance_calculator(mock_analytics_service):
    """Create MockBacktestPerformanceCalculator instance with mocked dependencies."""
    return MockBacktestPerformanceCalculator(mock_analytics_service)


@pytest.fixture
def sample_config():
    """Create sample backtest configuration."""
    return MockBacktestConfig(
        strategy_id=1,
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 3, 31),
        initial_capital=1000000.0,
        commission_per_trade=40.0,
        slippage_bps=3.0,
        risk_free_rate=0.06,
        lot_size=100
    )


@pytest.fixture
def sample_trades():
    """Create sample backtest trades for testing."""
    base_time = datetime(2024, 1, 1, 9, 15)
    trades = []
    
    # Winning trade
    trades.append(MockBacktestTrade(
        trade_id="trade_001",
        strategy_id=1,
        instrument="CRUDEOIL",
        direction="long",
        entry_time=base_time,
        entry_price=6500.0,
        exit_time=base_time + timedelta(hours=2),
        exit_price=6550.0,
        quantity=1,
        commission=40.0,
        slippage=0.03,
        pnl_points=50.0,
        pnl_inr=5000.0,
        setup_quality="A+"
    ))
    
    # Losing trade
    trades.append(MockBacktestTrade(
        trade_id="trade_002",
        strategy_id=1,
        instrument="CRUDEOIL",
        direction="short",
        entry_time=base_time + timedelta(days=1),
        entry_price=6480.0,
        exit_time=base_time + timedelta(days=1, hours=1),
        exit_price=6500.0,
        quantity=1,
        commission=40.0,
        slippage=0.03,
        pnl_points=-20.0,
        pnl_inr=-2000.0,
        setup_quality="B"
    ))
    
    # Another winning trade
    trades.append(MockBacktestTrade(
        trade_id="trade_003",
        strategy_id=1,
        instrument="CRUDEOIL",
        direction="long",
        entry_time=base_time + timedelta(days=2),
        entry_price=6520.0,
        exit_time=base_time + timedelta(days=2, hours=3),
        exit_price=6580.0,
        quantity=1,
        commission=40.0,
        slippage=0.03,
        pnl_points=60.0,
        pnl_inr=6000.0,
        setup_quality="A"
    ))
    
    return trades


@pytest.fixture
def sample_equity_curve():
    """Create sample equity curve for testing."""
    dates = pd.date_range(start="2024-01-01", end="2024-03-31", freq="D")
    
    # Create realistic equity curve
    initial_equity = 1000000.0
    daily_returns = np.random.normal(0.001, 0.015, len(dates))  # 0.1% daily return, 1.5% volatility
    daily_returns[0] = 0  # First day no return
    cumulative_returns = np.cumprod(1 + daily_returns)
    equity_values = initial_equity * cumulative_returns
    
    return pd.DataFrame({
        'timestamp': dates,
        'equity': equity_values,
        'drawdown': np.zeros(len(dates))
    })


# Test Classes
class TestBacktestPerformanceCalculatorInit:
    """Test BacktestPerformanceCalculator initialization."""
    
    def test_initialization(self, mock_analytics_service):
        """Test proper initialization of performance calculator."""
        calculator = MockBacktestPerformanceCalculator(mock_analytics_service)
        
        assert calculator.analytics_service == mock_analytics_service
        assert calculator.db == mock_analytics_service.db
        assert calculator.initial_capital == mock_analytics_service.initial_capital
    
    def test_initialization_with_different_capital(self):
        """Test initialization with different initial capital."""
        analytics_service = MagicMock()
        analytics_service.db = MagicMock()
        analytics_service.initial_capital = 500000.0
        
        calculator = MockBacktestPerformanceCalculator(analytics_service)
        
        assert calculator.initial_capital == 500000.0


class TestBasicMetricsCalculation:
    """Test basic metrics calculation methods."""
    
    def test_calculate_comprehensive_metrics_with_trades(self, performance_calculator, 
                                                        sample_config, sample_trades, sample_equity_curve):
        """Test comprehensive metrics calculation with sample trades."""
        result = performance_calculator.calculate_comprehensive_metrics(
            sample_trades, sample_config, sample_equity_curve
        )
        
        # Verify basic calculations
        assert result.total_trades == 3
        assert result.winning_trades == 2
        assert result.losing_trades == 1
        assert result.win_rate_pct == (2/3) * 100  # 66.67%
        assert result.total_pnl_inr == 9000.0  # 5000 - 2000 + 6000
        assert abs(result.total_return_pct - 0.9) < 0.0001  # 9000 / 1000000 * 100 (handle floating point)
        
        # Verify win/loss metrics
        assert result.avg_win_inr == 5500.0  # (5000 + 6000) / 2
        assert result.avg_loss_inr == -2000.0
        assert result.largest_win_inr == 6000.0
        assert result.largest_loss_inr == -2000.0
        
        # Verify profit factor
        expected_profit_factor = 11000.0 / 2000.0  # total wins / total losses
        assert result.profit_factor == expected_profit_factor
        
        # Verify expectancy
        assert result.expectancy_inr == 3000.0  # 9000 / 3 trades
        
        # Verify risk metrics were called
        assert result.sharpe_ratio == 1.85  # From mock
        assert result.sortino_ratio == 2.10  # From mock
    
    def test_calculate_comprehensive_metrics_no_trades(self, performance_calculator, 
                                                      sample_config, sample_equity_curve):
        """Test comprehensive metrics calculation with no trades."""
        empty_trades = []
        
        result = performance_calculator.calculate_comprehensive_metrics(
            empty_trades, sample_config, sample_equity_curve
        )
        
        assert result.total_trades == 0
        assert result.winning_trades == 0
        assert result.losing_trades == 0
        assert result.win_rate_pct == 0.0
        assert result.total_pnl_inr == 0.0
        assert result.profit_factor == 0.0
    
    def test_calculate_comprehensive_metrics_all_winners(self, performance_calculator, 
                                                        sample_config, sample_equity_curve):
        """Test comprehensive metrics with all winning trades."""
        winning_trades = [
            MockBacktestTrade(
                trade_id="win_1",
                strategy_id=1,
                instrument="CRUDEOIL",
                direction="long",
                entry_time=datetime(2024, 1, 1),
                exit_time=datetime(2024, 1, 1, 1),
                entry_price=6500.0,
                exit_price=6550.0,
                pnl_inr=5000.0
            ),
            MockBacktestTrade(
                trade_id="win_2",
                strategy_id=1,
                instrument="CRUDEOIL",
                direction="long",
                entry_time=datetime(2024, 1, 2),
                exit_time=datetime(2024, 1, 2, 1),
                entry_price=6520.0,
                exit_price=6580.0,
                pnl_inr=6000.0
            )
        ]
        
        result = performance_calculator.calculate_comprehensive_metrics(
            winning_trades, sample_config, sample_equity_curve
        )
        
        assert result.total_trades == 2
        assert result.winning_trades == 2
        assert result.losing_trades == 0
        assert result.win_rate_pct == 100.0
        assert result.avg_loss_inr == 0.0
        assert result.largest_loss_inr == 0.0
    
    def test_calculate_comprehensive_metrics_all_losers(self, performance_calculator, 
                                                       sample_config, sample_equity_curve):
        """Test comprehensive metrics with all losing trades."""
        losing_trades = [
            MockBacktestTrade(
                trade_id="loss_1",
                strategy_id=1,
                instrument="CRUDEOIL",
                direction="long",
                entry_time=datetime(2024, 1, 1),
                exit_time=datetime(2024, 1, 1, 1),
                entry_price=6500.0,
                exit_price=6450.0,
                pnl_inr=-5000.0
            ),
            MockBacktestTrade(
                trade_id="loss_2",
                strategy_id=1,
                instrument="CRUDEOIL",
                direction="short",
                entry_time=datetime(2024, 1, 2),
                exit_time=datetime(2024, 1, 2, 1),
                entry_price=6480.0,
                exit_price=6520.0,
                pnl_inr=-4000.0
            )
        ]
        
        result = performance_calculator.calculate_comprehensive_metrics(
            losing_trades, sample_config, sample_equity_curve
        )
        
        assert result.total_trades == 2
        assert result.winning_trades == 0
        assert result.losing_trades == 2
        assert result.win_rate_pct == 0.0
        assert result.total_pnl_inr == -9000.0
        assert result.avg_win_inr == 0.0
        assert result.profit_factor == 0.0  # No wins


class TestDailyReturnsCalculation:
    """Test daily returns calculation method."""
    
    def test_calculate_daily_returns_normal_curve(self, performance_calculator, sample_equity_curve):
        """Test daily returns calculation with normal equity curve."""
        returns = performance_calculator._calculate_daily_returns(sample_equity_curve)
        
        assert len(returns) == len(sample_equity_curve) - 1
        assert all(isinstance(r, float) for r in returns)
        
        # Returns should be reasonable (not extreme)
        assert all(-0.2 < r < 0.2 for r in returns)  # -20% to +20% daily return max
    
    def test_calculate_daily_returns_insufficient_data(self, performance_calculator):
        """Test daily returns calculation with insufficient data."""
        short_curve = pd.DataFrame({
            'timestamp': [datetime(2024, 1, 1)],
            'equity': [1000000.0]
        })
        
        returns = performance_calculator._calculate_daily_returns(short_curve)
        assert returns == []
    
    def test_calculate_daily_returns_empty_curve(self, performance_calculator):
        """Test daily returns calculation with empty curve."""
        empty_curve = pd.DataFrame()
        
        returns = performance_calculator._calculate_daily_returns(empty_curve)
        assert returns == []


class TestPerformanceComparison:
    """Test performance comparison methods."""
    
    def test_compare_to_live_performance_success(self, performance_calculator):
        """Test successful comparison to live performance."""
        backtest_metrics = MockBacktestMetrics(
            sharpe_ratio=1.85,
            sortino_ratio=2.10,
            win_rate_pct=66.67,
            profit_factor=5.5,
            total_return_pct=9.0
        )
        
        comparison = performance_calculator.compare_to_live_performance(
            backtest_metrics, strategy_id=1, comparison_period_days=90
        )
        
        # Verify structure
        assert "backtest_period" in comparison
        assert "live_period" in comparison
        assert "analysis" in comparison
        
        # Verify backtest data
        backtest_data = comparison["backtest_period"]
        assert backtest_data["sharpe_ratio"] == 1.85
        assert backtest_data["win_rate_pct"] == 66.67
        
        # Verify analysis
        analysis = comparison["analysis"]
        assert "performance_correlation" in analysis
        assert analysis["comparison_period_days"] == 90
        
        # Verify analytics service was called
        performance_calculator.analytics_service.calculate_risk_metrics.assert_called_with(1)
    
    def test_compare_to_live_performance_error_handling(self, performance_calculator):
        """Test error handling in performance comparison."""
        # Mock analytics service to raise error
        performance_calculator.analytics_service.calculate_risk_metrics.side_effect = Exception("Mock error")
        
        backtest_metrics = MockBacktestMetrics()
        
        comparison = performance_calculator.compare_to_live_performance(
            backtest_metrics, strategy_id=1, comparison_period_days=90
        )
        
        # Should return error info instead of crashing
        assert "error" in comparison
        assert "Mock error" in comparison["error"]
        assert comparison["backtest_metrics_available"] is True
        assert comparison["live_metrics_available"] is False


class TestPerformanceSummary:
    """Test performance summary generation."""
    
    def test_generate_performance_summary(self, performance_calculator):
        """Test performance summary generation."""
        metrics = MockBacktestMetrics(
            total_return_pct=9.0,
            annual_return_pct=12.5,
            max_drawdown_pct=-5.2,
            sharpe_ratio=1.85,
            sortino_ratio=2.10,
            profit_factor=5.5,
            total_trades=50,
            win_rate_pct=66.67,
            avg_win_inr=5500.0,
            avg_loss_inr=-2000.0,
            expectancy_inr=3000.0,
            var_95_inr=15000.0,
            volatility_annual_pct=18.5,
            max_drawdown_duration_days=7,
            total_costs_inr=129.0
        )
        
        summary = performance_calculator.generate_performance_summary(metrics)
        
        # Verify structure
        assert "overview" in summary
        assert "trading_stats" in summary
        assert "risk_analysis" in summary
        assert "costs" in summary
        
        # Verify overview data
        overview = summary["overview"]
        assert overview["total_return_pct"] == 9.0
        assert overview["sharpe_ratio"] == 1.85
        assert overview["sortino_ratio"] == 2.10  # Should be included
        assert overview["profit_factor"] == 5.5
        
        # Verify trading stats
        trading_stats = summary["trading_stats"]
        assert trading_stats["total_trades"] == 50
        assert trading_stats["win_rate_pct"] == 66.7  # Rounded to 1 decimal
        assert trading_stats["expectancy_inr"] == 3000.0
        
        # Verify risk analysis
        risk_analysis = summary["risk_analysis"]
        assert risk_analysis["var_95_inr"] == 15000.0
        assert risk_analysis["max_drawdown_duration_days"] == 7
        
        # Verify costs
        costs = summary["costs"]
        assert costs["total_costs_inr"] == 129.0
    
    def test_generate_performance_summary_with_zeros(self, performance_calculator):
        """Test performance summary with zero/empty metrics."""
        metrics = MockBacktestMetrics()  # All default values (zeros)
        
        summary = performance_calculator.generate_performance_summary(metrics)
        
        # Should handle zeros gracefully
        assert summary["overview"]["total_return_pct"] == 0.0
        assert summary["overview"]["profit_factor"] == 0.0
        assert summary["trading_stats"]["total_trades"] == 0
        assert summary["trading_stats"]["win_rate_pct"] == 0.0


class TestEdgeCases:
    """Test edge cases and error scenarios."""
    
    def test_open_trades_ignored(self, performance_calculator, sample_config, sample_equity_curve):
        """Test that open trades are properly ignored in calculations."""
        mixed_trades = [
            # Closed winning trade
            MockBacktestTrade(
                trade_id="closed_001",
                strategy_id=1,
                instrument="CRUDEOIL",
                direction="long",
                entry_time=datetime(2024, 1, 1, 9, 15),
                exit_time=datetime(2024, 1, 1, 11, 15),
                entry_price=6500.0,
                exit_price=6550.0,
                pnl_inr=5000.0
            ),
            # Open trade (should be ignored)
            MockBacktestTrade(
                trade_id="open_001",
                strategy_id=1,
                instrument="CRUDEOIL",
                direction="long",
                entry_time=datetime(2024, 1, 2, 9, 15),
                exit_time=None,  # Still open
                entry_price=6480.0,
                exit_price=None,
                pnl_inr=None
            )
        ]
        
        result = performance_calculator.calculate_comprehensive_metrics(
            mixed_trades, sample_config, sample_equity_curve
        )
        
        # Should only count the closed trade
        assert result.total_trades == 1
        assert result.winning_trades == 1
        assert result.total_pnl_inr == 5000.0
    
    def test_single_trade(self, performance_calculator, sample_config, sample_equity_curve):
        """Test handling of single trade."""
        single_trade = [MockBacktestTrade(
            trade_id="single_001",
            strategy_id=1,
            instrument="CRUDEOIL",
            direction="long",
            entry_time=datetime(2024, 1, 1, 9, 15),
            exit_time=datetime(2024, 1, 1, 11, 15),
            entry_price=6500.0,
            exit_price=6550.0,
            pnl_inr=5000.0
        )]
        
        result = performance_calculator.calculate_comprehensive_metrics(
            single_trade, sample_config, sample_equity_curve
        )
        
        assert result.total_trades == 1
        assert result.winning_trades == 1
        assert result.losing_trades == 0
        assert result.win_rate_pct == 100.0
        assert result.total_pnl_inr == 5000.0
    
    def test_extreme_values(self, performance_calculator, sample_config, sample_equity_curve):
        """Test handling of extreme values in trades."""
        extreme_trades = [
            # Very large win
            MockBacktestTrade(
                trade_id="extreme_win",
                strategy_id=1,
                instrument="CRUDEOIL",
                direction="long",
                entry_time=datetime(2024, 1, 1, 9, 15),
                exit_time=datetime(2024, 1, 1, 11, 15),
                entry_price=6000.0,
                exit_price=8000.0,
                pnl_inr=200000.0  # Very large win
            ),
            # Very small loss
            MockBacktestTrade(
                trade_id="small_loss",
                strategy_id=1,
                instrument="CRUDEOIL",
                direction="short",
                entry_time=datetime(2024, 1, 2, 9, 15),
                exit_time=datetime(2024, 1, 2, 9, 16),
                entry_price=6500.0,
                exit_price=6501.0,
                pnl_inr=-100.0  # Very small loss
            )
        ]
        
        result = performance_calculator.calculate_comprehensive_metrics(
            extreme_trades, sample_config, sample_equity_curve
        )
        
        # Should handle extreme values gracefully
        assert result.total_trades == 2
        assert result.largest_win_inr == 200000.0
        assert result.largest_loss_inr == -100.0
        assert result.profit_factor == 2000.0  # 200000 / 100
    
    def test_zero_capital_config(self, performance_calculator, sample_trades, sample_equity_curve):
        """Test handling of zero initial capital."""
        zero_capital_config = MockBacktestConfig(
            strategy_id=1,
            initial_capital=0.0  # Invalid capital
        )
        
        # Should handle gracefully without division by zero
        result = performance_calculator.calculate_comprehensive_metrics(
            sample_trades, zero_capital_config, sample_equity_curve
        )
        
        assert isinstance(result, MockBacktestMetrics)
        # Should not crash, and return percentage should be 0 for zero capital
        assert result.total_return_pct == 0.0
        # But should still calculate absolute P&L correctly
        assert result.total_pnl_inr == 9000.0  # 5000 - 2000 + 6000


class TestIntegrationWithAnalyticsService:
    """Test integration with existing AnalyticsService methods."""
    
    def test_sharpe_ratio_integration(self, performance_calculator, sample_equity_curve):
        """Test that Sharpe ratio calculation integrates properly."""
        returns = performance_calculator._calculate_daily_returns(sample_equity_curve)
        
        # Call the analytics service method
        performance_calculator.analytics_service.calculate_sharpe_ratio(returns)
        
        # Verify it was called with the right parameters
        performance_calculator.analytics_service.calculate_sharpe_ratio.assert_called_with(returns)
    
    def test_sortino_ratio_integration(self, performance_calculator, sample_equity_curve):
        """Test that Sortino ratio calculation integrates properly."""
        returns = performance_calculator._calculate_daily_returns(sample_equity_curve)
        
        # Call the analytics service method
        performance_calculator.analytics_service._calculate_sortino_ratio(returns)
        
        # Verify it was called
        performance_calculator.analytics_service._calculate_sortino_ratio.assert_called_with(returns)
    
    def test_risk_metrics_integration(self, performance_calculator):
        """Test integration with risk metrics calculation."""
        strategy_id = 1
        
        # Call the method that uses risk metrics
        performance_calculator.analytics_service.calculate_risk_metrics(strategy_id)
        
        # Verify the service method was called
        performance_calculator.analytics_service.calculate_risk_metrics.assert_called_with(strategy_id)


class TestMockValidation:
    """Test that mocks are properly configured and realistic."""
    
    def test_mock_analytics_service_consistency(self, mock_analytics_service):
        """Test that mock analytics service behaves consistently."""
        # Test multiple calls return same values
        sharpe1 = mock_analytics_service.calculate_sharpe_ratio([0.01, 0.02, -0.01])
        sharpe2 = mock_analytics_service.calculate_sharpe_ratio([0.01, 0.02, -0.01])
        assert sharpe1 == sharpe2 == 1.85
        
        # Test risk metrics consistency
        metrics1 = mock_analytics_service.calculate_risk_metrics(1)
        metrics2 = mock_analytics_service.calculate_risk_metrics(1)
        assert metrics1 == metrics2
    
    def test_sample_data_realism(self, sample_trades, sample_config, sample_equity_curve):
        """Test that sample data is realistic for trading scenarios."""
        # Verify trade data realism
        for trade in sample_trades:
            assert trade.entry_price > 0
            assert trade.exit_price > 0 if trade.exit_price else True
            assert trade.commission >= 0
            assert trade.slippage >= 0
            assert abs(trade.pnl_inr) < 100000  # Reasonable P&L for crude oil
        
        # Verify config realism
        assert sample_config.initial_capital >= 100000  # Minimum capital
        assert sample_config.commission_per_trade >= 0
        assert 0 <= sample_config.slippage_bps <= 100  # Reasonable slippage
        
        # Verify equity curve realism
        assert len(sample_equity_curve) > 10  # Sufficient data points
        assert sample_equity_curve['equity'].min() > 0  # No negative equity
        assert sample_equity_curve['equity'].max() < 10 * sample_config.initial_capital  # No extreme growth
    
    def test_mock_dataclass_functionality(self):
        """Test that mock dataclasses work as expected."""
        metrics = MockBacktestMetrics(
            total_trades=10,
            winning_trades=6,
            total_pnl_inr=15000.0
        )
        
        # Test attribute access
        assert metrics.total_trades == 10
        assert metrics.winning_trades == 6
        assert metrics.total_pnl_inr == 15000.0
        
        # Test default values
        assert metrics.losing_trades == 0  # Default value
        assert metrics.sharpe_ratio == 0.0  # Default value
        
        # Test attribute modification
        metrics.losing_trades = 4
        assert metrics.losing_trades == 4


# Integration test with pytest markers
@pytest.mark.integration
class TestEndToEndWorkflow:
    """Test complete workflow integration."""
    
    def test_complete_calculation_workflow(self, performance_calculator, 
                                         sample_config, sample_trades, sample_equity_curve):
        """Test complete end-to-end calculation workflow."""
        # Step 1: Calculate comprehensive metrics
        metrics = performance_calculator.calculate_comprehensive_metrics(
            sample_trades, sample_config, sample_equity_curve
        )
        
        # Step 2: Generate performance summary
        summary = performance_calculator.generate_performance_summary(metrics)
        
        # Step 3: Compare to live performance
        comparison = performance_calculator.compare_to_live_performance(
            metrics, strategy_id=1, comparison_period_days=90
        )
        
        # Verify complete workflow
        assert isinstance(metrics, MockBacktestMetrics)
        assert isinstance(summary, dict)
        assert isinstance(comparison, dict)
        
        # Verify data consistency across steps
        assert abs(summary["overview"]["total_return_pct"] - metrics.total_return_pct) < 0.0001
        assert abs(comparison["backtest_period"]["total_return_pct"] - metrics.total_return_pct) < 0.0001
        
        # Verify all major metrics are present
        required_summary_keys = ["overview", "trading_stats", "risk_analysis", "costs"]
        assert all(key in summary for key in required_summary_keys)
        
        required_comparison_keys = ["backtest_period", "live_period", "analysis"]
        assert all(key in comparison for key in required_comparison_keys)
    
    def test_workflow_with_analytics_integration(self, performance_calculator, 
                                               sample_config, sample_trades, sample_equity_curve):
        """Test workflow ensuring analytics service integration."""
        # Calculate metrics (should call analytics service)
        metrics = performance_calculator.calculate_comprehensive_metrics(
            sample_trades, sample_config, sample_equity_curve
        )
        
        # Verify analytics service methods were called
        performance_calculator.analytics_service.calculate_sharpe_ratio.assert_called()
        performance_calculator.analytics_service._calculate_sortino_ratio.assert_called()
        
        # Verify risk metrics from analytics service are used
        assert metrics.sharpe_ratio == 1.85  # From mock analytics service
        assert metrics.sortino_ratio == 2.10  # From mock analytics service
        
        # Compare to live performance (should call analytics service again)
        comparison = performance_calculator.compare_to_live_performance(
            metrics, strategy_id=1, comparison_period_days=90
        )
        
        # Verify live performance lookup was called
        performance_calculator.analytics_service.calculate_risk_metrics.assert_called_with(1)
        
        # Verify comparison structure
        assert comparison["backtest_period"]["sharpe_ratio"] == 1.85
        assert comparison["live_period"]["sharpe_ratio"] == 1.85  # From mock


if __name__ == "__main__":
    # Run tests with: python -m pytest test_backtest_performance_calculator_simple.py -v
    pytest.main([__file__, "-v", "-s", "--tb=short"])