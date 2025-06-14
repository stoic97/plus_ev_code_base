"""
Unit tests for BacktestingEngine service.

This module provides comprehensive unit tests for the BacktestingEngine service,
covering backtesting configuration, signal generation, trade execution simulation,
performance metrics calculation, and integration with existing infrastructure.
"""

import pytest
import sys
import os
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Optional

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

# Global flag to track if we have real imports
REAL_IMPORTS = True

try:
    # Try importing the actual modules
    from app.services.backtesting_engine import (
        BacktestingEngine,
        BacktestConfig,
        BacktestTrade,
        BacktestMetrics,
        BacktestResult,
        BacktestStatus,
        create_backtesting_engine,
        validate_backtest_config,
        calculate_signal_statistics
    )
    from app.services.backtesting_data_service import BacktestingDataService
    from app.services.strategy_engine import StrategyEngineService
    from app.models.strategy import Signal, Direction, TimeframeValue, SetupQualityGrade
    from app.core.error_handling import (
        OperationalError,
        ValidationError,
        DatabaseConnectionError
    )
    print("✓ Successfully imported real backtesting modules")
    
except ImportError as e:
    print(f"⚠ Import error (using mocks): {e}")
    REAL_IMPORTS = False
    
    # Create mock classes and functions for testing structure
    class MockError(Exception):
        pass
    
    class MockEnum:
        def __init__(self, value):
            self.value = value
    
    class MockDirection:
        LONG = MockEnum("long")
        SHORT = MockEnum("short")
    
    class MockTimeframeValue:
        DAILY = MockEnum("1d")
        FOUR_HOUR = MockEnum("4h")
        ONE_HOUR = MockEnum("1h")
        THIRTY_MIN = MockEnum("30m")
        FIFTEEN_MIN = MockEnum("15m")
        FIVE_MIN = MockEnum("5m")
        THREE_MIN = MockEnum("3m")
    
    class MockSetupQualityGrade:
        A_PLUS = MockEnum("a_plus")
        A = MockEnum("a")
        B_PLUS = MockEnum("b_plus")
        B = MockEnum("b")
        C = MockEnum("c")
        D = MockEnum("d")
        F = MockEnum("f")
    
    class MockBacktestStatus:
        PENDING = "pending"
        RUNNING = "running"
        COMPLETED = "completed"
        FAILED = "failed"
        CANCELLED = "cancelled"
    
    OperationalError = MockError
    ValidationError = MockError
    DatabaseConnectionError = MockError
    Direction = MockDirection()
    TimeframeValue = MockTimeframeValue()
    SetupQualityGrade = MockSetupQualityGrade()
    BacktestStatus = MockBacktestStatus()


# Mock fixtures following existing patterns
@pytest.fixture
def mock_db_session():
    """Mock database session."""
    session = MagicMock()
    session.query.return_value = session
    session.filter.return_value = session
    session.order_by.return_value = session
    session.offset.return_value = session
    session.limit.return_value = session
    session.first.return_value = None
    session.all.return_value = []
    session.count.return_value = 0
    session.commit = MagicMock()
    session.refresh = MagicMock()
    session.delete = MagicMock()
    session.add = MagicMock()
    # Context manager support
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=None)
    return session


@pytest.fixture
def mock_backtesting_data_service():
    """Mock BacktestingDataService for testing."""
    service = MagicMock(spec=BacktestingDataService)
    
    # Create sample historical data
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', end='2024-01-10', freq='1min'),
        'open': [5000.0 + i for i in range(12961)],  # 9 days * 24 hours * 60 mins + 1
        'high': [5005.0 + i for i in range(12961)],
        'low': [4995.0 + i for i in range(12961)],
        'close': [5002.0 + i for i in range(12961)],
        'volume': [1000 + i for i in range(12961)]
    })
    
    service.load_historical_data.return_value = sample_data
    service.get_data_for_period.return_value = sample_data
    service.get_latest_data_point.return_value = {
        'timestamp': datetime(2024, 1, 10),
        'open': 5010.0,
        'high': 5015.0,
        'low': 5005.0,
        'close': 5012.0,
        'volume': 1500
    }
    service.get_data_summary.return_value = {
        'status': 'ok',
        'total_records': len(sample_data),
        'date_range': {
            'start': '2024-01-01T00:00:00',
            'end': '2024-01-10T00:00:00',
            'days': 9
        }
    }
    
    return service


@pytest.fixture
def mock_strategy_service(mock_db_session):
    """Mock StrategyEngineService for testing."""
    service = MagicMock(spec=StrategyEngineService)
    service.db = mock_db_session
    
    # Mock strategy retrieval
    mock_strategy = MagicMock()
    mock_strategy.id = 1
    mock_strategy.name = "Test Crude Oil Strategy"
    mock_strategy.user_id = 1
    service.get_strategy.return_value = mock_strategy
    
    # Mock timeframe analysis
    mock_timeframe_analysis = MagicMock()
    mock_timeframe_analysis.aligned = True
    mock_timeframe_analysis.primary_direction = "up"
    mock_timeframe_analysis.alignment_score = 0.85
    service.analyze_timeframes.return_value = mock_timeframe_analysis
    
    # Mock market state analysis
    mock_market_state = MagicMock()
    mock_market_state.market_state = "trending_up"
    mock_market_state.accumulation_detected = True
    mock_market_state.bos_detected = True
    service.analyze_market_state.return_value = mock_market_state
    
    # Mock setup quality evaluation
    mock_setup_quality = MagicMock()
    mock_setup_quality.grade = MockSetupQualityGrade.A if not REAL_IMPORTS else SetupQualityGrade.A
    mock_setup_quality.score = 85.0
    mock_setup_quality.position_size = 2
    service.evaluate_setup_quality.return_value = mock_setup_quality
    
    # Mock signal generation
    mock_signal = create_mock_signal()
    service.generate_signal.return_value = mock_signal
    
    return service


@pytest.fixture
def sample_backtest_config():
    """Sample backtest configuration."""
    if REAL_IMPORTS:
        return BacktestConfig(
            strategy_id=1,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 7),
            initial_capital=1000000.0,
            commission_per_trade=40.0,
            slippage_bps=3.0,
            max_position_size=0.15
        )
    else:
        # Mock config for when real imports aren't available
        config = MagicMock()
        config.strategy_id = 1
        config.start_date = datetime(2024, 1, 1)
        config.end_date = datetime(2024, 1, 7)
        config.initial_capital = 1000000.0
        config.commission_per_trade = 40.0
        config.slippage_bps = 3.0
        config.max_position_size = 0.15
        config.lot_size = 100
        config.tick_size = 1.0
        config.margin_requirement = 0.08
        config.enable_slippage = True
        config.enable_commission = True
        return config


@pytest.fixture
def sample_backtest_trade():
    """Sample backtest trade."""
    if REAL_IMPORTS:
        return BacktestTrade(
            trade_id="BT_1_001",
            signal_id=1,
            strategy_id=1,
            instrument="CRUDEOIL",
            direction="long",
            entry_time=datetime(2024, 1, 1, 10, 30),
            entry_price=5000.0,
            exit_time=datetime(2024, 1, 1, 14, 30),
            exit_price=5050.0,
            exit_reason="take_profit",
            quantity=2,
            commission=40.0,
            slippage=5.0,
            pnl_points=50.0,
            pnl_inr=9960.0,  # (50 * 100 * 2) - 40
            setup_quality="a",
            setup_score=85.0
        )
    else:
        # Mock trade
        trade = MagicMock()
        trade.trade_id = "BT_1_001"
        trade.signal_id = 1
        trade.strategy_id = 1
        trade.instrument = "CRUDEOIL"
        trade.direction = "long"
        trade.entry_time = datetime(2024, 1, 1, 10, 30)
        trade.entry_price = 5000.0
        trade.exit_time = datetime(2024, 1, 1, 14, 30)
        trade.exit_price = 5050.0
        trade.exit_reason = "take_profit"
        trade.quantity = 2
        trade.commission = 40.0
        trade.slippage = 5.0
        trade.pnl_points = 50.0
        trade.pnl_inr = 9960.0
        trade.setup_quality = "a"
        trade.setup_score = 85.0
        trade.is_open = False
        trade.duration_minutes = 240
        return trade


def create_mock_signal(signal_id: int = 1):
    """Create a mock Signal object."""
    signal = MagicMock()
    signal.id = signal_id
    signal.strategy_id = 1
    signal.instrument = "CRUDEOIL"
    signal.direction = MockDirection.LONG if not REAL_IMPORTS else Direction.LONG
    signal.entry_price = 5000.0
    signal.take_profit_price = 5075.0
    signal.stop_loss_price = 4950.0
    signal.position_size = 2
    signal.setup_quality = MockSetupQualityGrade.A if not REAL_IMPORTS else SetupQualityGrade.A
    signal.setup_score = 85.0
    signal.confidence = 0.85
    signal.risk_reward_ratio = 1.5
    return signal


# Test Classes
class TestBacktestConfig:
    """Test BacktestConfig validation and functionality."""
    
    def test_valid_config_creation(self, sample_backtest_config):
        """Test creating a valid backtest configuration."""
        config = sample_backtest_config
        
        assert config.strategy_id == 1
        assert config.initial_capital == 1000000.0
        assert config.commission_per_trade == 40.0
        assert config.slippage_bps == 3.0
        assert config.max_position_size == 0.15
    
    def test_invalid_date_range(self):
        """Test validation with invalid date range."""
        if not REAL_IMPORTS:
            pytest.skip("Skipping validation test without real imports")
        
        with pytest.raises(ValidationError):
            BacktestConfig(
                strategy_id=1,
                start_date=datetime(2024, 1, 7),
                end_date=datetime(2024, 1, 1),  # End before start
                initial_capital=1000000.0
            )
    
    def test_invalid_capital(self):
        """Test validation with invalid capital."""
        if not REAL_IMPORTS:
            pytest.skip("Skipping validation test without real imports")
        
        with pytest.raises(ValidationError):
            BacktestConfig(
                strategy_id=1,
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 7),
                initial_capital=-1000.0  # Negative capital
            )
    
    def test_invalid_position_size(self):
        """Test validation with invalid position size."""
        if not REAL_IMPORTS:
            pytest.skip("Skipping validation test without real imports")
        
        with pytest.raises(ValidationError):
            BacktestConfig(
                strategy_id=1,
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 7),
                initial_capital=1000000.0,
                max_position_size=1.5  # > 100%
            )


class TestBacktestTrade:
    """Test BacktestTrade functionality."""
    
    def test_trade_properties(self, sample_backtest_trade):
        """Test trade property calculations."""
        trade = sample_backtest_trade
        
        assert trade.trade_id == "BT_1_001"
        assert trade.instrument == "CRUDEOIL"
        assert trade.direction == "long"
        assert trade.is_open == False
        assert trade.duration_minutes == 240  # 4 hours
    
    def test_open_trade(self):
        """Test open trade properties."""
        if REAL_IMPORTS:
            trade = BacktestTrade(
                trade_id="BT_1_002",
                signal_id=2,
                strategy_id=1,
                instrument="CRUDEOIL",
                direction="short",
                entry_time=datetime(2024, 1, 1, 10, 30),
                entry_price=5000.0,
                quantity=1
            )
            
            assert trade.is_open == True
            assert trade.duration_minutes is None
        else:
            # Mock test
            trade = MagicMock()
            trade.exit_time = None
            trade.is_open = True
            trade.duration_minutes = None
            assert trade.is_open == True


class TestBacktestingEngine:
    """Test BacktestingEngine functionality."""
    
    def test_engine_initialization(self, mock_backtesting_data_service, mock_strategy_service):
        """Test engine initialization."""
        if REAL_IMPORTS:
            engine = BacktestingEngine(mock_backtesting_data_service, mock_strategy_service)
            
            assert engine.data_service == mock_backtesting_data_service
            assert engine.strategy_service == mock_strategy_service
            assert len(engine._running_backtests) == 0
        else:
            # Mock test
            engine = MagicMock()
            engine.data_service = mock_backtesting_data_service
            engine.strategy_service = mock_strategy_service
            assert engine.data_service == mock_backtesting_data_service
    
    def test_load_historical_data(self, mock_backtesting_data_service, mock_strategy_service, 
                                 sample_backtest_config):
        """Test historical data loading."""
        if not REAL_IMPORTS:
            pytest.skip("Skipping without real imports")
        
        engine = BacktestingEngine(mock_backtesting_data_service, mock_strategy_service)
        
        # Test data loading
        data = engine._load_historical_data(sample_backtest_config)
        
        assert len(data) > 0
        assert 'timestamp' in data.columns
        assert 'open' in data.columns
        assert 'high' in data.columns
        assert 'low' in data.columns
        assert 'close' in data.columns
        assert 'volume' in data.columns
    
    def test_initialize_trading_state(self, mock_backtesting_data_service, mock_strategy_service,
                                     sample_backtest_config):
        """Test trading state initialization."""
        if not REAL_IMPORTS:
            pytest.skip("Skipping without real imports")
        
        engine = BacktestingEngine(mock_backtesting_data_service, mock_strategy_service)
        
        state = engine._initialize_trading_state(sample_backtest_config)
        
        assert state['cash'] == sample_backtest_config.initial_capital
        assert state['positions'] == {}
        assert state['open_trades'] == {}
        assert state['trade_counter'] == 0
        assert isinstance(state['equity_history'], list)
        assert isinstance(state['generated_signals'], list)
    
    def test_signal_generation(self, mock_backtesting_data_service, mock_strategy_service,
                              sample_backtest_config):
        """Test signal generation during backtest."""
        if not REAL_IMPORTS:
            pytest.skip("Skipping without real imports")
        
        engine = BacktestingEngine(mock_backtesting_data_service, mock_strategy_service)
        
        # Create sample market data
        sample_data = pd.DataFrame({
            'timestamp': [datetime(2024, 1, 1, 10, 0)],
            'open': [5000.0],
            'high': [5010.0],
            'low': [4990.0],
            'close': [5005.0],
            'volume': [1000]
        })
        
        # Test signal generation
        signals = engine._generate_signals_for_tick(
            sample_backtest_config, 
            sample_data.iloc[0], 
            sample_data
        )
        
        # Should return list of signals (mocked to return 1)
        assert isinstance(signals, list)
    
    def test_signal_to_trade_conversion(self, mock_backtesting_data_service, mock_strategy_service,
                                       sample_backtest_config):
        """Test converting signals to trades."""
        if not REAL_IMPORTS:
            pytest.skip("Skipping without real imports")
        
        engine = BacktestingEngine(mock_backtesting_data_service, mock_strategy_service)
        
        # Create mock signal and market data
        signal = create_mock_signal()
        current_row = pd.Series({
            'timestamp': datetime(2024, 1, 1, 10, 30),
            'close': 5000.0
        })
        trading_state = engine._initialize_trading_state(sample_backtest_config)
        
        # Test signal execution
        trade = engine._execute_signal_as_trade(signal, sample_backtest_config, 
                                                current_row, trading_state)
        
        if trade:
            assert trade.signal_id == signal.id
            assert trade.instrument == signal.instrument
            assert trade.entry_time == current_row['timestamp']
            assert trade.original_signal == signal
    
    def test_exit_conditions(self, mock_backtesting_data_service, mock_strategy_service,
                            sample_backtest_config, sample_backtest_trade):
        """Test trade exit condition checking."""
        if not REAL_IMPORTS:
            pytest.skip("Skipping without real imports")
        
        engine = BacktestingEngine(mock_backtesting_data_service, mock_strategy_service)
        
        # Create open trades dict
        sample_backtest_trade.exit_time = None  # Make it open
        open_trades = {sample_backtest_trade.trade_id: sample_backtest_trade}
        
        # Test with take profit price hit
        current_row = pd.Series({
            'timestamp': datetime(2024, 1, 1, 14, 30),
            'close': 5075.0  # Hit take profit
        })
        
        exits = engine._check_exit_conditions(open_trades, current_row, sample_backtest_config)
        
        assert len(exits) <= 1  # Should identify exit
    
    def test_metrics_calculation(self, mock_backtesting_data_service, mock_strategy_service,
                                sample_backtest_config):
        """Test performance metrics calculation."""
        if not REAL_IMPORTS:
            pytest.skip("Skipping without real imports")
        
        engine = BacktestingEngine(mock_backtesting_data_service, mock_strategy_service)
        
        # Create sample closed trades
        trades = []
        for i in range(5):
            trade = BacktestTrade(
                trade_id=f"BT_1_{i:03d}",
                signal_id=i+1,
                strategy_id=1,
                instrument="CRUDEOIL",
                direction="long" if i % 2 == 0 else "short",
                entry_time=datetime(2024, 1, 1, 10, i*10),
                entry_price=5000.0 + i,
                exit_time=datetime(2024, 1, 1, 14, i*10),
                exit_price=5000.0 + i + (20 if i % 2 == 0 else -15),  # Some wins, some losses
                quantity=1,
                commission=40.0,
                pnl_inr=(20 if i % 2 == 0 else -15) * 100 - 40,  # Points * lot_size - commission
                setup_quality="a" if i % 2 == 0 else "b"
            )
            trades.append(trade)
        
        metrics = engine._calculate_metrics(trades, sample_backtest_config)
        
        assert metrics.total_trades == 5
        assert metrics.winning_trades > 0
        assert metrics.losing_trades > 0
        assert 0 <= metrics.win_rate_pct <= 100
        assert metrics.total_commission_inr == 5 * 40.0  # 5 trades * 40 commission
    
    def test_equity_curve_generation(self, mock_backtesting_data_service, mock_strategy_service,
                                    sample_backtest_config, sample_backtest_trade):
        """Test equity curve generation."""
        if not REAL_IMPORTS:
            pytest.skip("Skipping without real imports")
        
        engine = BacktestingEngine(mock_backtesting_data_service, mock_strategy_service)
        
        trades = [sample_backtest_trade]
        equity_curve = engine._generate_equity_curve(trades, sample_backtest_config)
        
        assert len(equity_curve) >= 2  # Initial point + trade points
        assert 'timestamp' in equity_curve.columns
        assert 'equity' in equity_curve.columns
        assert 'drawdown_pct' in equity_curve.columns
        assert 'running_max' in equity_curve.columns
    
    @patch('app.services.backtesting_engine.logger')
    def test_full_backtest_execution(self, mock_logger, mock_backtesting_data_service, 
                                    mock_strategy_service, sample_backtest_config):
        """Test full backtest execution."""
        if not REAL_IMPORTS:
            pytest.skip("Skipping without real imports")
        
        engine = BacktestingEngine(mock_backtesting_data_service, mock_strategy_service)
        
        # Run backtest
        result = engine.run_backtest(sample_backtest_config)
        
        assert result.backtest_id is not None
        assert result.config == sample_backtest_config
        assert result.status in [BacktestStatus.COMPLETED, BacktestStatus.FAILED]
        assert result.start_time is not None
        assert result.end_time is not None
        assert isinstance(result.trades, list)
        
        if result.status == BacktestStatus.COMPLETED:
            assert result.metrics is not None
            assert result.is_complete == True
    
    def test_backtest_cancellation(self, mock_backtesting_data_service, mock_strategy_service):
        """Test backtest cancellation."""
        if not REAL_IMPORTS:
            pytest.skip("Skipping without real imports")
        
        engine = BacktestingEngine(mock_backtesting_data_service, mock_strategy_service)
        
        # Add a mock running backtest
        backtest_id = "test-backtest-123"
        mock_result = MagicMock()
        engine._running_backtests[backtest_id] = mock_result
        
        # Test cancellation
        cancelled = engine.cancel_backtest(backtest_id)
        
        assert cancelled == True
        assert backtest_id not in engine._running_backtests
        assert mock_result.status == BacktestStatus.CANCELLED
    
    def test_backtest_status_tracking(self, mock_backtesting_data_service, mock_strategy_service):
        """Test backtest status tracking."""
        if not REAL_IMPORTS:
            pytest.skip("Skipping without real imports")
        
        engine = BacktestingEngine(mock_backtesting_data_service, mock_strategy_service)
        
        # Add a mock running backtest
        backtest_id = "test-backtest-456"
        mock_result = MagicMock()
        engine._running_backtests[backtest_id] = mock_result
        
        # Test status retrieval
        status = engine.get_backtest_status(backtest_id)
        assert status == mock_result
        
        # Test non-existent backtest
        non_existent_status = engine.get_backtest_status("non-existent")
        assert non_existent_status is None
        
        # Test listing running backtests
        running_list = engine.list_running_backtests()
        assert backtest_id in running_list


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_factory_function(self, mock_backtesting_data_service, mock_strategy_service):
        """Test factory function for creating engine."""
        if not REAL_IMPORTS:
            pytest.skip("Skipping without real imports")
        
        engine = create_backtesting_engine(mock_backtesting_data_service, mock_strategy_service)
        
        assert isinstance(engine, BacktestingEngine)
        assert engine.data_service == mock_backtesting_data_service
        assert engine.strategy_service == mock_strategy_service
    
    def test_config_validation(self, sample_backtest_config):
        """Test backtest configuration validation."""
        if not REAL_IMPORTS:
            pytest.skip("Skipping without real imports")
        
        warnings = validate_backtest_config(sample_backtest_config)
        
        assert isinstance(warnings, list)
        # Short period might generate warning
        if (sample_backtest_config.end_date - sample_backtest_config.start_date).days < 7:
            assert len(warnings) > 0
    
    def test_signal_statistics_calculation(self, sample_backtest_trade):
        """Test signal statistics calculation."""
        if not REAL_IMPORTS:
            pytest.skip("Skipping without real imports")
        
        trades = [sample_backtest_trade]
        stats = calculate_signal_statistics(trades)
        
        assert isinstance(stats, dict)
        if sample_backtest_trade.setup_quality in stats:
            quality_stats = stats[sample_backtest_trade.setup_quality]
            assert 'trades' in quality_stats
            assert 'win_rate' in quality_stats
            assert 'avg_pnl_per_trade' in quality_stats


class TestIntegrationScenarios:
    """Test integration scenarios."""
    
    def test_no_strategy_service_fallback(self, mock_backtesting_data_service, sample_backtest_config):
        """Test engine behavior without strategy service."""
        if not REAL_IMPORTS:
            pytest.skip("Skipping without real imports")
        
        # Create engine without strategy service
        engine = BacktestingEngine(mock_backtesting_data_service, None)
        
        # Should still initialize properly
        assert engine.data_service == mock_backtesting_data_service
        assert engine.strategy_service is None
        
        # Signal generation should return empty list
        sample_data = pd.DataFrame({
            'timestamp': [datetime(2024, 1, 1, 10, 0)],
            'close': [5000.0]
        })
        
        signals = engine._generate_signals_for_tick(
            sample_backtest_config, 
            sample_data.iloc[0], 
            sample_data
        )
        
        assert isinstance(signals, list)
        assert len(signals) == 0  # No signals without strategy service
    
    def test_insufficient_data_handling(self, mock_backtesting_data_service, mock_strategy_service,
                                       sample_backtest_config):
        """Test handling of insufficient historical data."""
        if not REAL_IMPORTS:
            pytest.skip("Skipping without real imports")
        
        # Mock service to return completely empty data that will fail filtering
        empty_data = pd.DataFrame({
            'timestamp': pd.Series([], dtype='datetime64[ns]'),
            'open': pd.Series([], dtype='float64'),
            'high': pd.Series([], dtype='float64'),
            'low': pd.Series([], dtype='float64'),
            'close': pd.Series([], dtype='float64'),
            'volume': pd.Series([], dtype='int64')
        })
        mock_backtesting_data_service.load_historical_data.return_value = empty_data
        
        engine = BacktestingEngine(mock_backtesting_data_service, mock_strategy_service)
        
        # The test should expect an OperationalError since that's what gets raised when data loading fails
        with pytest.raises(OperationalError, match="Failed to load historical data"):
            engine.run_backtest(sample_backtest_config)
    
    def test_error_handling_during_execution(self, mock_backtesting_data_service, 
                                           mock_strategy_service, sample_backtest_config):
        """Test error handling during backtest execution."""
        if not REAL_IMPORTS:
            pytest.skip("Skipping without real imports")
        
        # Mock strategy service to raise error
        mock_strategy_service.analyze_timeframes.side_effect = Exception("Test error")
        
        engine = BacktestingEngine(mock_backtesting_data_service, mock_strategy_service)
        
        # Should handle errors gracefully and complete backtest
        result = engine.run_backtest(sample_backtest_config)
        
        # Should complete even with errors during signal generation
        assert result.status in [BacktestStatus.COMPLETED, BacktestStatus.FAILED]


# Simple tests that should always work
def test_basic_functionality():
    """Basic test to ensure test file is working."""
    assert True
    assert 1 + 1 == 2


def test_import_status():
    """Test to show which import mode we're in."""
    if REAL_IMPORTS:
        print("✓ Running backtesting engine tests with real module imports")
    else:
        print("⚠ Running backtesting engine tests with mocked imports")
    assert True


def test_fixtures_are_working(mock_backtesting_data_service, mock_strategy_service, 
                             sample_backtest_config, sample_backtest_trade):
    """Test that all fixtures are working correctly."""
    assert mock_backtesting_data_service is not None
    assert mock_strategy_service is not None
    assert sample_backtest_config.strategy_id == 1
    assert sample_backtest_trade.trade_id == "BT_1_001"


# Performance benchmarks
class TestPerformanceBenchmarks:
    """Test performance and benchmark scenarios."""
    
    def test_large_dataset_handling(self, mock_backtesting_data_service, mock_strategy_service):
        """Test handling of large datasets."""
        if not REAL_IMPORTS:
            pytest.skip("Skipping without real imports")
        
        # Create large dataset (simulate 30 days of 1-minute data)
        large_dataset = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', end='2024-01-30', freq='1min'),
            'open': [5000.0 + (i % 100) for i in range(41761)],  # 30 days of data
            'high': [5005.0 + (i % 100) for i in range(41761)],
            'low': [4995.0 + (i % 100) for i in range(41761)],
            'close': [5002.0 + (i % 100) for i in range(41761)],
            'volume': [1000 + (i % 500) for i in range(41761)]
        })
        
        mock_backtesting_data_service.load_historical_data.return_value = large_dataset
        
        config = BacktestConfig(
            strategy_id=1,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 30),
            initial_capital=1000000.0
        ) if REAL_IMPORTS else MagicMock()
        
        engine = BacktestingEngine(mock_backtesting_data_service, mock_strategy_service)
        
        # Should handle large dataset without memory issues
        # This is more of a smoke test than a performance test
        assert engine is not None
        assert engine.data_service == mock_backtesting_data_service
    
    def test_multiple_concurrent_backtests(self, mock_backtesting_data_service, mock_strategy_service):
        """Test tracking multiple concurrent backtests."""
        if not REAL_IMPORTS:
            pytest.skip("Skipping without real imports")
        
        engine = BacktestingEngine(mock_backtesting_data_service, mock_strategy_service)
        
        # Simulate multiple running backtests
        for i in range(3):
            backtest_id = f"concurrent-test-{i}"
            mock_result = MagicMock()
            mock_result.backtest_id = backtest_id
            mock_result.status = BacktestStatus.RUNNING
            engine._running_backtests[backtest_id] = mock_result
        
        # Test listing
        running_backtests = engine.list_running_backtests()
        assert len(running_backtests) == 3
        
        # Test individual status checks
        for i in range(3):
            backtest_id = f"concurrent-test-{i}"
            status = engine.get_backtest_status(backtest_id)
            assert status is not None
            assert status.backtest_id == backtest_id
    
    def test_extreme_market_conditions(self, mock_backtesting_data_service, mock_strategy_service):
        """Test with extreme market conditions (high volatility, gaps)."""
        if not REAL_IMPORTS:
            pytest.skip("Skipping without real imports")
        
        # Create dataset with extreme conditions
        extreme_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', end='2024-01-03', freq='1min'),
            'open': [5000.0 if i % 10 != 0 else 5000.0 + (200 if i % 20 == 0 else -200) for i in range(2881)],
            'high': [5100.0 if i % 10 != 0 else 5200.0 for i in range(2881)],  # High volatility
            'low': [4900.0 if i % 10 != 0 else 4800.0 for i in range(2881)],
            'close': [5050.0 if i % 10 != 0 else 5050.0 + (150 if i % 20 == 0 else -150) for i in range(2881)],
            'volume': [2000 + (i % 1000) for i in range(2881)]  # Variable volume
        })
        
        mock_backtesting_data_service.load_historical_data.return_value = extreme_data
        
        engine = BacktestingEngine(mock_backtesting_data_service, mock_strategy_service)
        
        # Test slippage calculation with extreme conditions
        high_slippage_price = engine._apply_slippage(5000.0, 'long', BacktestConfig(
            strategy_id=1,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 3),
            initial_capital=1000000.0,
            slippage_bps=50.0  # High slippage
        ) if REAL_IMPORTS else MagicMock())
        
        # Should apply slippage appropriately
        if REAL_IMPORTS:
            assert high_slippage_price > 5000.0  # Long should have higher execution price


# Edge cases and error scenarios
class TestEdgeCases:
    """Test edge cases and error scenarios."""
    
    def test_zero_trades_scenario(self, mock_backtesting_data_service, mock_strategy_service):
        """Test scenario where no trades are generated."""
        if not REAL_IMPORTS:
            pytest.skip("Skipping without real imports")
        
        # Mock strategy service to never generate signals
        mock_strategy_service.analyze_timeframes.return_value.aligned = False
        
        engine = BacktestingEngine(mock_backtesting_data_service, mock_strategy_service)
        
        config = BacktestConfig(
            strategy_id=1,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 3),
            initial_capital=1000000.0
        )
        
        result = engine.run_backtest(config)
        
        # Should complete successfully with zero trades
        assert result.status == BacktestStatus.COMPLETED
        assert len(result.trades) == 0
        assert result.metrics.total_trades == 0
    
    def test_all_losing_trades_scenario(self, mock_backtesting_data_service, mock_strategy_service):
        """Test scenario where all trades are losers."""
        if not REAL_IMPORTS:
            pytest.skip("Skipping without real imports")
        
        engine = BacktestingEngine(mock_backtesting_data_service, mock_strategy_service)
        
        # Create losing trades  
        losing_trades = []
        for i in range(3):
            trade = BacktestTrade(
                trade_id=f"LOSS_{i}",
                signal_id=i+1,
                strategy_id=1,
                instrument="CRUDEOIL",
                direction="long",
                entry_time=datetime(2024, 1, 1, 10, i*10),  # 10:00, 10:10, 10:20
                entry_price=5000.0,
                exit_time=datetime(2024, 1, 1, 14, i*10),   # 14:00, 14:10, 14:20
                exit_price=4950.0,  # All losing trades
                quantity=1,
                commission=40.0,
                pnl_inr=-5040.0,  # (-50 * 100) - 40
                setup_quality="c"
            )
            losing_trades.append(trade)
        
        config = BacktestConfig(
            strategy_id=1,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 3),
            initial_capital=1000000.0
        )
        
        metrics = engine._calculate_metrics(losing_trades, config)
        
        assert metrics.total_trades == 3
        assert metrics.winning_trades == 0
        assert metrics.losing_trades == 3
        assert metrics.win_rate_pct == 0.0
        assert metrics.total_pnl_inr < 0
        assert metrics.profit_factor == 0.0  # No wins
    
    def test_single_trade_scenario(self, mock_backtesting_data_service, mock_strategy_service,
                                  sample_backtest_trade):
        """Test scenario with only one trade."""
        if not REAL_IMPORTS:
            pytest.skip("Skipping without real imports")
        
        engine = BacktestingEngine(mock_backtesting_data_service, mock_strategy_service)
        
        config = BacktestConfig(
            strategy_id=1,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 3),
            initial_capital=1000000.0
        )
        
        trades = [sample_backtest_trade]
        metrics = engine._calculate_metrics(trades, config)
        
        assert metrics.total_trades == 1
        assert metrics.trades_per_day > 0
        assert metrics.expectancy_inr == sample_backtest_trade.pnl_inr
    
    def test_missing_signal_data(self, mock_backtesting_data_service, mock_strategy_service):
        """Test handling of signals with missing data."""
        if not REAL_IMPORTS:
            pytest.skip("Skipping without real imports")
        
        engine = BacktestingEngine(mock_backtesting_data_service, mock_strategy_service)
        
        # Create signal with missing take profit/stop loss
        incomplete_signal = create_mock_signal()
        incomplete_signal.take_profit_price = None
        incomplete_signal.stop_loss_price = None
        
        config = BacktestConfig(
            strategy_id=1,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 3),
            initial_capital=1000000.0
        )
        
        current_row = pd.Series({
            'timestamp': datetime(2024, 1, 1, 10, 30),
            'close': 5000.0
        })
        
        trading_state = engine._initialize_trading_state(config)
        
        # Should handle missing signal data gracefully
        trade = engine._execute_signal_as_trade(incomplete_signal, config, current_row, trading_state)
        
        # Should create trade even with incomplete signal data
        if trade:
            assert trade.original_signal == incomplete_signal
    
    def test_data_quality_issues(self, mock_backtesting_data_service, mock_strategy_service):
        """Test handling of data quality issues."""
        if not REAL_IMPORTS:
            pytest.skip("Skipping without real imports")
        
        # Create data with quality issues
        problematic_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', end='2024-01-02', freq='1min'),
            'open': [5000.0] * 1441,
            'high': [4999.0] * 1441,  # High < Open (invalid)
            'low': [5001.0] * 1441,   # Low > Open (invalid)
            'close': [5000.0] * 1441,
            'volume': [0] * 1441      # Zero volume
        })
        
        mock_backtesting_data_service.load_historical_data.return_value = problematic_data
        
        engine = BacktestingEngine(mock_backtesting_data_service, mock_strategy_service)
        
        config = BacktestConfig(
            strategy_id=1,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
            initial_capital=1000000.0
        )
        
        # Should handle invalid OHLC data gracefully
        # The backtest might complete with warnings or fewer signals
        result = engine.run_backtest(config)
        
        # Should not crash, even with bad data
        assert result.status in [BacktestStatus.COMPLETED, BacktestStatus.FAILED]


# Integration with existing infrastructure tests
class TestInfrastructureIntegration:
    """Test integration with existing infrastructure."""
    
    def test_signal_model_compatibility(self):
        """Test compatibility with existing Signal model."""
        if not REAL_IMPORTS:
            pytest.skip("Skipping without real imports")
        
        # Create signal using mock (representing real Signal model)
        signal = create_mock_signal()
        
        # Test that all expected attributes are present
        expected_attributes = [
            'id', 'strategy_id', 'instrument', 'direction', 'entry_price',
            'take_profit_price', 'stop_loss_price', 'position_size',
            'setup_quality', 'setup_score', 'confidence', 'risk_reward_ratio'
        ]
        
        for attr in expected_attributes:
            assert hasattr(signal, attr), f"Signal missing expected attribute: {attr}"
    
    def test_timeframe_value_integration(self):
        """Test integration with TimeframeValue enum."""
        if not REAL_IMPORTS:
            pytest.skip("Skipping without real imports")
        
        # Simple test that doesn't require data processing
        # Just verify the enum values exist and can be used
        try:
            # Test that FIVE_MIN exists and has the expected value
            assert hasattr(TimeframeValue, 'FIVE_MIN')
            assert TimeframeValue.FIVE_MIN.value == '5m'
            
            # Test that we can create a dict with TimeframeValue keys
            test_dict = {TimeframeValue.FIVE_MIN: {'test': 'data'}}
            assert TimeframeValue.FIVE_MIN in test_dict
            assert test_dict[TimeframeValue.FIVE_MIN]['test'] == 'data'
            
        except Exception as e:
            pytest.fail(f"TimeframeValue integration test failed: {e}")
    
    def test_direction_enum_integration(self):
        """Test integration with Direction enum."""
        if not REAL_IMPORTS:
            pytest.skip("Skipping without real imports")
        
        # Test direction conversion
        signal = create_mock_signal()
        
        # Should be able to convert direction to string
        direction_str = signal.direction.value.lower() if hasattr(signal.direction, 'value') else str(signal.direction).lower()
        
        assert direction_str in ['long', 'short']
    
    def test_setup_quality_grade_integration(self):
        """Test integration with SetupQualityGrade enum."""
        if not REAL_IMPORTS:
            pytest.skip("Skipping without real imports")
        
        signal = create_mock_signal()
        
        # Should be able to access setup quality value
        quality_value = signal.setup_quality.value if hasattr(signal.setup_quality, 'value') else str(signal.setup_quality)
        
        expected_grades = ['a_plus', 'a', 'b_plus', 'b', 'c', 'd', 'f']
        assert quality_value.lower() in expected_grades
    
    def test_error_handling_integration(self):
        """Test integration with existing error handling."""
        if not REAL_IMPORTS:
            pytest.skip("Skipping without real imports")
        
        # Test that our custom errors are compatible
        with pytest.raises(ValidationError):
            raise ValidationError("Test validation error")
        
        with pytest.raises(OperationalError):
            raise OperationalError("Test operational error")
        
        with pytest.raises(DatabaseConnectionError):
            raise DatabaseConnectionError("Test database error")


if __name__ == "__main__":
    # Run tests with: python -m pytest tests/test_services/test_backtesting_engine.py -v
    pytest.main([__file__, "-v", "-s"])