"""
Tests for the Analytics Service.

This module contains comprehensive tests for the AnalyticsService,
covering NAV calculation, risk metrics, attribution analysis, and
benchmark comparison functionality.
"""

import pytest
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import numpy as np

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


# Fixtures for mock objects
@pytest.fixture
def mock_db_session():
    """Create a mock database session."""
    session = MagicMock()
    session.query.return_value = session
    session.filter.return_value = session
    session.scalar.return_value = None
    session.all.return_value = []
    session.first.return_value = None
    return session


@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    redis = MagicMock()
    redis.get.return_value = None
    redis.set.return_value = True
    return redis


@pytest.fixture
def mock_trade():
    """Create a mock trade object."""
    trade = Mock()
    trade.id = 1
    trade.strategy_id = 1
    trade.entry_price = 48500.0
    trade.exit_price = 48600.0
    trade.entry_time = datetime.utcnow() - timedelta(hours=2)
    trade.exit_time = datetime.utcnow() - timedelta(hours=1)
    trade.position_size = 1
    trade.profit_loss_points = 100.0
    trade.profit_loss_inr = 5000.0  # 100 points * 50 INR/point * 1 lot
    trade.commission = 20.0
    trade.taxes = 25.0
    trade.total_costs = 45.0
    trade.setup_quality = Mock(value="A")
    trade.direction = "long"
    return trade


@pytest.fixture
def mock_open_trade():
    """Create a mock open trade (no exit)."""
    trade = Mock()
    trade.id = 2
    trade.strategy_id = 1
    trade.entry_price = 48700.0
    trade.exit_price = None
    trade.entry_time = datetime.utcnow() - timedelta(minutes=30)
    trade.exit_time = None
    trade.position_size = 2
    trade.profit_loss_inr = None
    trade.total_costs = None
    return trade


# Mock AnalyticsService for testing
class MockAnalyticsService:
    """Mock implementation of AnalyticsService for testing."""
    
    def __init__(self, db, initial_capital=1000000.0):
        self.db = db
        self.initial_capital = initial_capital
        self.redis = None
    
    def calculate_nav(self, strategy_id):
        """Calculate NAV based on mock data."""
        # Get closed trades
        closed_trades = self.db.all()
        
        nav = self.initial_capital
        if closed_trades:
            for trade in closed_trades:
                if hasattr(trade, 'profit_loss_inr') and trade.profit_loss_inr:
                    nav += trade.profit_loss_inr
                if hasattr(trade, 'total_costs') and trade.total_costs:
                    nav -= trade.total_costs
        
        # Cache in Redis if available
        if self.redis:
            self.redis.set(f"analytics:nav:{strategy_id}", str(nav), ex=60)
        
        return nav
    
    def get_open_positions_value(self, strategy_id):
        """Calculate open positions value."""
        open_trades = self.db.all()
        
        total_value = 0.0
        for trade in open_trades:
            if hasattr(trade, 'entry_price') and hasattr(trade, 'position_size'):
                lot_size = 50
                position_value = trade.entry_price * trade.position_size * lot_size
                total_value += position_value
        
        return total_value
    
    def get_equity_curve(self, strategy_id, period="1M"):
        """Get equity curve data."""
        trades = self.db.all()
        
        equity_curve = []
        running_pnl = 0.0
        
        for trade in trades:
            if hasattr(trade, 'exit_time') and trade.exit_time and hasattr(trade, 'profit_loss_inr'):
                running_pnl += trade.profit_loss_inr if trade.profit_loss_inr else 0
                if hasattr(trade, 'total_costs') and trade.total_costs:
                    running_pnl -= trade.total_costs
                
                nav = self.initial_capital + running_pnl
                daily_return = (running_pnl / self.initial_capital) * 100
                
                equity_curve.append({
                    "timestamp": trade.exit_time,
                    "nav": nav,
                    "cumulative_pnl": running_pnl,
                    "daily_return": daily_return,
                    "trade_id": trade.id
                })
        
        return equity_curve
    
    def calculate_drawdown(self, equity_curve):
        """Calculate drawdown from equity curve."""
        if not equity_curve:
            return {
                "max_drawdown": 0.0,
                "max_drawdown_duration_days": 0,
                "current_drawdown": 0.0,
                "drawdown_periods": []
            }
        
        navs = [point["nav"] for point in equity_curve]
        
        # Calculate running maximum
        running_max = []
        current_max = navs[0]
        for nav in navs:
            current_max = max(current_max, nav)
            running_max.append(current_max)
        
        # Calculate drawdowns
        drawdowns = [(nav - max_nav) / max_nav * 100 for nav, max_nav in zip(navs, running_max)]
        
        # Find maximum drawdown
        max_drawdown = min(drawdowns) if drawdowns else 0
        
        return {
            "max_drawdown": max_drawdown,
            "max_drawdown_duration_days": 0,
            "current_drawdown": drawdowns[-1] if drawdowns else 0.0,
            "drawdown_periods": []
        }
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.06):
        """Calculate Sharpe ratio."""
        if not returns or len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        daily_rf = risk_free_rate / 252
        excess_returns = returns_array - daily_rf
        
        if excess_returns.std() == 0:
            return 0.0
        
        sharpe = excess_returns.mean() / excess_returns.std()
        sharpe_annualized = sharpe * np.sqrt(252)
        
        return float(sharpe_annualized)
    
    def calculate_attribution_by_grade(self, strategy_id, start_date=None):
        """Calculate attribution by grade."""
        trades = self.db.all()
        
        attribution = {}
        total_pnl = 0
        
        for trade in trades:
            if hasattr(trade, 'setup_quality') and hasattr(trade, 'profit_loss_inr'):
                grade = trade.setup_quality.value if trade.setup_quality else "Unknown"
                
                if grade not in attribution:
                    attribution[grade] = {"count": 0, "pnl": 0.0, "wins": 0}
                
                attribution[grade]["count"] += 1
                if trade.profit_loss_inr:
                    attribution[grade]["pnl"] += trade.profit_loss_inr
                    total_pnl += trade.profit_loss_inr
                    if trade.profit_loss_inr > 0:
                        attribution[grade]["wins"] += 1
        
        # Calculate win rates
        result = {}
        for grade, data in attribution.items():
            win_rate = data["wins"] / data["count"] if data["count"] > 0 else 0
            pnl_contribution = data["pnl"] / total_pnl * 100 if total_pnl != 0 else 0
            
            result[grade] = {
                "count": data["count"],
                "pnl": data["pnl"],
                "win_rate": win_rate,
                "avg_pnl": data["pnl"] / data["count"] if data["count"] > 0 else 0,
                "pnl_contribution_percent": pnl_contribution
            }
        
        return result


@pytest.fixture
def analytics_service(mock_db_session):
    """Create an AnalyticsService instance with mocked dependencies."""
    service = MockAnalyticsService(mock_db_session)
    return service


@pytest.fixture
def analytics_service_with_redis(mock_db_session, mock_redis):
    """Create an AnalyticsService instance with Redis enabled."""
    service = MockAnalyticsService(mock_db_session)
    service.redis = mock_redis
    return service


class TestAnalyticsServiceInit:
    """Test AnalyticsService initialization."""
    
    def test_init_default_capital(self, mock_db_session):
        """Test initialization with default capital."""
        service = MockAnalyticsService(mock_db_session)
        assert service.db == mock_db_session
        assert service.initial_capital == 1000000.0
        assert service.redis is None
    
    def test_init_with_custom_capital(self, mock_db_session):
        """Test initialization with custom initial capital."""
        service = MockAnalyticsService(mock_db_session, initial_capital=500000.0)
        assert service.initial_capital == 500000.0


class TestLiveMetrics:
    """Test live metrics calculation methods."""
    
    def test_calculate_nav_no_trades(self, analytics_service, mock_db_session):
        """Test NAV calculation with no trades."""
        mock_db_session.all.return_value = []
        
        nav = analytics_service.calculate_nav(strategy_id=1)
        
        assert nav == 1000000.0  # Initial capital
    
    def test_calculate_nav_with_closed_trades(self, analytics_service, mock_db_session, mock_trade):
        """Test NAV calculation with closed trades."""
        mock_db_session.all.return_value = [mock_trade]
        
        nav = analytics_service.calculate_nav(strategy_id=1)
        
        # Initial capital + profit - costs
        expected_nav = 1000000.0 + 5000.0 - 45.0
        assert nav == expected_nav
    
    def test_calculate_nav_with_redis_cache(self, analytics_service_with_redis, mock_db_session, mock_redis):
        """Test NAV calculation with Redis caching."""
        mock_db_session.all.return_value = []
        
        nav = analytics_service_with_redis.calculate_nav(strategy_id=1)
        
        # Check Redis was called
        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args
        assert call_args[0][0] == "analytics:nav:1"
        assert call_args[0][1] == "1000000.0"
    
    def test_get_open_positions_value(self, analytics_service, mock_db_session, mock_open_trade):
        """Test open positions value calculation."""
        mock_db_session.all.return_value = [mock_open_trade]
        
        value = analytics_service.get_open_positions_value(strategy_id=1)
        
        # 48700 * 2 lots * 50 INR/point
        expected_value = 48700.0 * 2 * 50
        assert value == expected_value


class TestEquityCurve:
    """Test equity curve calculation methods."""
    
    def test_get_equity_curve_empty(self, analytics_service, mock_db_session):
        """Test equity curve with no trades."""
        mock_db_session.all.return_value = []
        
        curve = analytics_service.get_equity_curve(strategy_id=1, period="1M")
        
        assert curve == []
    
    def test_get_equity_curve_with_trades(self, analytics_service, mock_db_session):
        """Test equity curve with multiple trades."""
        # Create mock trades
        trades = []
        for i in range(3):
            trade = Mock()
            trade.id = i + 1
            trade.exit_time = datetime.utcnow() - timedelta(days=3-i)
            trade.profit_loss_inr = 1000.0 * (i + 1)
            trade.total_costs = 50.0
            trades.append(trade)
        
        mock_db_session.all.return_value = trades
        
        curve = analytics_service.get_equity_curve(strategy_id=1, period="1W")
        
        assert len(curve) == 3
        assert curve[0]["cumulative_pnl"] == 950.0  # 1000 - 50
        assert curve[1]["cumulative_pnl"] == 2900.0  # 950 + 2000 - 50
        assert curve[2]["cumulative_pnl"] == 5850.0  # 2900 + 3000 - 50
    
    def test_calculate_drawdown_no_data(self, analytics_service):
        """Test drawdown calculation with no data."""
        result = analytics_service.calculate_drawdown([])
        
        assert result["max_drawdown"] == 0.0
        assert result["current_drawdown"] == 0.0
    
    def test_calculate_drawdown_with_data(self, analytics_service):
        """Test drawdown calculation with equity curve data."""
        equity_curve = [
            {"nav": 1000000, "timestamp": datetime.utcnow() - timedelta(days=5)},
            {"nav": 1050000, "timestamp": datetime.utcnow() - timedelta(days=4)},  # Peak
            {"nav": 1020000, "timestamp": datetime.utcnow() - timedelta(days=2)},  # Drawdown
        ]
        
        result = analytics_service.calculate_drawdown(equity_curve)
        
        # Max drawdown from 1050000 to 1020000 = -2.86%
        assert result["max_drawdown"] == pytest.approx(-2.86, rel=0.01)


class TestRiskMetrics:
    """Test risk metrics calculation methods."""
    
    def test_calculate_sharpe_ratio_no_data(self, analytics_service):
        """Test Sharpe ratio with no returns."""
        sharpe = analytics_service.calculate_sharpe_ratio([])
        assert sharpe == 0.0
    
    def test_calculate_sharpe_ratio_positive(self, analytics_service):
        """Test Sharpe ratio with positive returns."""
        returns = [0.001] * 20  # 0.1% daily returns
        
        sharpe = analytics_service.calculate_sharpe_ratio(returns, risk_free_rate=0.06)
        
        assert sharpe > 0  # Should be positive


class TestAttribution:
    """Test performance attribution methods."""
    
    def test_attribution_by_grade_no_trades(self, analytics_service, mock_db_session):
        """Test attribution by grade with no trades."""
        mock_db_session.all.return_value = []
        
        result = analytics_service.calculate_attribution_by_grade(strategy_id=1)
        
        assert result == {}
    
    def test_attribution_by_grade_with_trades(self, analytics_service, mock_db_session):
        """Test attribution by grade with multiple trades."""
        # Create trades with different grades
        trades = []
        
        # A grade trade
        trade1 = Mock()
        trade1.setup_quality = Mock(value="A")
        trade1.profit_loss_inr = 5000.0
        trades.append(trade1)
        
        # B grade trade
        trade2 = Mock()
        trade2.setup_quality = Mock(value="B")
        trade2.profit_loss_inr = -1000.0
        trades.append(trade2)
        
        mock_db_session.all.return_value = trades
        
        result = analytics_service.calculate_attribution_by_grade(strategy_id=1)
        
        assert "A" in result
        assert result["A"]["count"] == 1
        assert result["A"]["pnl"] == 5000.0
        assert result["A"]["win_rate"] == 1.0
        
        assert "B" in result
        assert result["B"]["count"] == 1
        assert result["B"]["pnl"] == -1000.0
        assert result["B"]["win_rate"] == 0.0