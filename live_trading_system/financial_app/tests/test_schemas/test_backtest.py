"""
Unit tests for backtest Pydantic schemas.

These tests ensure that validation, serialization, and relationships between
schema fields work correctly according to the backtest business rules.
"""

import pytest
from datetime import datetime, timedelta
from pydantic import ValidationError
from typing import Dict, List, Optional

# Import the schemas to test
from app.schemas.backtest import (
    BacktestStatusEnum, BacktestDataSourceEnum, TimeframeEnum,
    BacktestBaseSchema, BacktestConfigBase, BacktestConfigCreate, BacktestConfigUpdate,
    BacktestTradeBase, BacktestTradeResponse, BacktestMetricsBase, BacktestMetricsResponse,
    BacktestResultBase, BacktestResultResponse, BacktestDetailedResultResponse,
    BacktestRunRequest, BacktestStatusRequest, BacktestListRequest,
    BacktestComparisonRequest, BacktestComparisonResponse,
    BacktestHealthResponse, BacktestErrorResponse
)
from app.schemas.strategy import DirectionEnum


class TestBacktestBaseSchema:
    """Tests for base backtest schema functionality."""
    
    def test_base_schema_configuration(self):
        """Test that BacktestBaseSchema has correct configuration."""
        assert BacktestBaseSchema.model_config["from_attributes"] is True
        assert BacktestBaseSchema.model_config["arbitrary_types_allowed"] is True
        
    def test_datetime_serialization(self):
        """Test that datetime values are serialized to ISO format."""
        test_time = datetime(2023, 6, 15, 12, 30, 45)
        encoder = BacktestBaseSchema.model_config["json_encoders"][datetime]
        assert encoder(test_time) == "2023-06-15T12:30:45"
        
    def test_none_datetime_serialization(self):
        """Test that None datetime values are handled correctly."""
        # Since BacktestBaseSchema inherits from BaseSchema, it should have json_encoders
        encoder = BacktestBaseSchema.model_config["json_encoders"][datetime]
        # Test with None - the encoder should just pass it through
        assert encoder(None) is None


class TestBacktestEnums:
    """Tests for backtest enum values."""
    
    def test_backtest_status_enum_values(self):
        """Test BacktestStatusEnum has all expected values."""
        expected_values = {"pending", "running", "completed", "failed", "cancelled"}
        actual_values = {status.value for status in BacktestStatusEnum}
        assert actual_values == expected_values
        
    def test_data_source_enum_values(self):
        """Test BacktestDataSourceEnum has all expected values."""
        expected_values = {"csv", "database", "api"}
        actual_values = {source.value for source in BacktestDataSourceEnum}
        assert actual_values == expected_values
        
    def test_timeframe_enum_values(self):
        """Test TimeframeEnum has all expected values."""
        expected_values = {"1m", "5m", "15m", "30m", "1h", "4h", "1d"}
        actual_values = {timeframe.value for timeframe in TimeframeEnum}
        assert actual_values == expected_values


class TestBacktestConfigSchemas:
    """Tests for backtest configuration schemas."""
    
    def test_valid_backtest_config_create(self):
        """Test creating a valid backtest configuration."""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 3, 31)
        
        config = BacktestConfigCreate(
            strategy_id=1,
            name="Q1 2023 Backtest",
            description="Testing strategy performance for Q1 2023",
            start_date=start_date,
            end_date=end_date,
            initial_capital=100000.0,
            max_position_size=0.1,
            data_source=BacktestDataSourceEnum.CSV,
            timeframe=TimeframeEnum.ONE_MIN,
            commission_per_trade=20.0,
            slippage_bps=2.0,
            warm_up_days=30,
            benchmark_symbol="NIFTY50"
        )
        
        assert config.strategy_id == 1
        assert config.name == "Q1 2023 Backtest"
        assert config.start_date == start_date
        assert config.end_date == end_date
        assert config.initial_capital == 100000.0
        assert config.max_position_size == 0.1
        assert config.commission_per_trade == 20.0
        assert config.slippage_bps == 2.0
        
    def test_default_values(self):
        """Test default values are set correctly."""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 3, 31)
        
        config = BacktestConfigCreate(
            strategy_id=1,
            name="Basic Test",
            start_date=start_date,
            end_date=end_date,
            initial_capital=100000.0
        )
        
        # Check defaults
        assert config.max_position_size == 0.1
        assert config.data_source == BacktestDataSourceEnum.CSV
        assert config.timeframe == TimeframeEnum.ONE_MIN
        assert config.commission_per_trade == 20.0
        assert config.slippage_bps == 2.0
        assert config.warm_up_days == 30
        assert config.benchmark_symbol == "NIFTY50"
        
    def test_invalid_strategy_id(self):
        """Test validation of strategy ID."""
        with pytest.raises(ValidationError) as exc_info:
            BacktestConfigCreate(
                strategy_id=0,  # Invalid: must be > 0
                name="Test",
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 3, 31),
                initial_capital=100000.0
            )
        assert "greater than 0" in str(exc_info.value)
        
    def test_invalid_capital(self):
        """Test validation of initial capital."""
        with pytest.raises(ValidationError) as exc_info:
            BacktestConfigCreate(
                strategy_id=1,
                name="Test",
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 3, 31),
                initial_capital=-100.0  # Invalid: must be > 0
            )
        assert "greater than 0" in str(exc_info.value)
        
    def test_invalid_position_size_range(self):
        """Test validation of position size range."""
        # Test below minimum
        with pytest.raises(ValidationError):
            BacktestConfigCreate(
                strategy_id=1,
                name="Test",
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 3, 31),
                initial_capital=100000.0,
                max_position_size=0.0005  # Invalid: below 0.001
            )
            
        # Test above maximum
        with pytest.raises(ValidationError):
            BacktestConfigCreate(
                strategy_id=1,
                name="Test",
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 3, 31),
                initial_capital=100000.0,
                max_position_size=1.1  # Invalid: above 1.0
            )
            
    def test_date_validation(self):
        """Test date validation logic."""
        # End date before start date
        with pytest.raises(ValidationError) as exc_info:
            BacktestConfigCreate(
                strategy_id=1,
                name="Test",
                start_date=datetime(2023, 3, 31),
                end_date=datetime(2023, 1, 1),  # Invalid: before start
                initial_capital=100000.0
            )
        assert "End date must be after start date" in str(exc_info.value)
        
        # Start date in future
        future_date = datetime.now() + timedelta(days=30)
        with pytest.raises(ValidationError) as exc_info:
            BacktestConfigCreate(
                strategy_id=1,
                name="Test",
                start_date=future_date,  # Invalid: in future
                end_date=future_date + timedelta(days=30),
                initial_capital=100000.0
            )
        assert "Start date cannot be in the future" in str(exc_info.value)
        
    def test_name_validation(self):
        """Test name field validation."""
        # Empty name
        with pytest.raises(ValidationError):
            BacktestConfigCreate(
                strategy_id=1,
                name="",  # Invalid: empty
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 3, 31),
                initial_capital=100000.0
            )
            
        # Name too long
        long_name = "x" * 256  # Over 255 character limit
        with pytest.raises(ValidationError):
            BacktestConfigCreate(
                strategy_id=1,
                name=long_name,  # Invalid: too long
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 3, 31),
                initial_capital=100000.0
            )
            
    def test_slippage_validation(self):
        """Test slippage validation."""
        # Negative slippage
        with pytest.raises(ValidationError):
            BacktestConfigCreate(
                strategy_id=1,
                name="Test",
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 3, 31),
                initial_capital=100000.0,
                slippage_bps=-1.0  # Invalid: negative
            )
            
        # Excessive slippage
        with pytest.raises(ValidationError):
            BacktestConfigCreate(
                strategy_id=1,
                name="Test",
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 3, 31),
                initial_capital=100000.0,
                slippage_bps=150.0  # Invalid: over 100 bps
            )
            
    def test_backtest_config_update(self):
        """Test BacktestConfigUpdate schema."""
        update = BacktestConfigUpdate(
            name="Updated Name",
            description="Updated description",
            max_position_size=0.05,
            commission_per_trade=15.0,
            slippage_bps=1.5,
            benchmark_symbol="BANKNIFTY"
        )
        
        assert update.name == "Updated Name"
        assert update.max_position_size == 0.05
        assert update.commission_per_trade == 15.0
        assert update.benchmark_symbol == "BANKNIFTY"


class TestBacktestTradeSchemas:
    """Tests for backtest trade schemas."""
    
    def test_valid_backtest_trade(self):
        """Test creating a valid backtest trade."""
        entry_time = datetime(2023, 1, 15, 9, 30)
        exit_time = datetime(2023, 1, 15, 15, 30)
        
        trade = BacktestTradeBase(
            trade_id="trade_123",
            signal_id=456,
            strategy_id=1,
            instrument="BANKNIFTY",
            direction=DirectionEnum.LONG,
            entry_time=entry_time,
            entry_price=40000.0,
            quantity=1,
            exit_time=exit_time,
            exit_price=40200.0,
            exit_reason="profit_target",
            pnl_points=200.0,
            pnl_inr=5000.0,
            commission=20.0,
            slippage=5.0,
            setup_quality="A+",
            setup_score=9.2
        )
        
        assert trade.trade_id == "trade_123"
        assert trade.signal_id == 456
        assert trade.direction == DirectionEnum.LONG
        assert trade.entry_price == 40000.0
        assert trade.exit_price == 40200.0
        assert trade.pnl_inr == 5000.0
        assert trade.setup_quality == "A+"
        
    def test_trade_time_validation(self):
        """Test trade time validation."""
        entry_time = datetime(2023, 1, 15, 15, 30)
        invalid_exit_time = datetime(2023, 1, 15, 9, 30)  # Before entry
        
        with pytest.raises(ValidationError) as exc_info:
            BacktestTradeBase(
                trade_id="trade_123",
                signal_id=456,
                strategy_id=1,
                instrument="BANKNIFTY",
                direction=DirectionEnum.LONG,
                entry_time=entry_time,
                entry_price=40000.0,
                exit_time=invalid_exit_time  # Invalid: before entry
            )
        assert "Exit time must be after entry time" in str(exc_info.value)
        
    def test_trade_price_validation(self):
        """Test trade price validation."""
        # Invalid entry price
        with pytest.raises(ValidationError):
            BacktestTradeBase(
                trade_id="trade_123",
                signal_id=456,
                strategy_id=1,
                instrument="BANKNIFTY",
                direction=DirectionEnum.LONG,
                entry_time=datetime(2023, 1, 15, 9, 30),
                entry_price=-100.0  # Invalid: negative price
            )
            
        # Invalid exit price
        with pytest.raises(ValidationError):
            BacktestTradeBase(
                trade_id="trade_123",
                signal_id=456,
                strategy_id=1,
                instrument="BANKNIFTY",
                direction=DirectionEnum.LONG,
                entry_time=datetime(2023, 1, 15, 9, 30),
                entry_price=40000.0,
                exit_price=0.0  # Invalid: zero price
            )
            
    def test_trade_response_computed_fields(self):
        """Test BacktestTradeResponse computed fields."""
        entry_time = datetime(2023, 1, 15, 9, 30)
        exit_time = datetime(2023, 1, 15, 15, 30)
        
        trade = BacktestTradeResponse(
            trade_id="trade_123",
            signal_id=456,
            strategy_id=1,
            instrument="BANKNIFTY",
            direction=DirectionEnum.LONG,
            entry_time=entry_time,
            entry_price=40000.0,
            exit_time=exit_time,
            exit_price=40200.0,
            is_open=False,
            duration_minutes=360
        )
        
        assert trade.is_open is False
        assert trade.duration_minutes == 360


class TestBacktestMetricsSchemas:
    """Tests for backtest metrics schemas."""
    
    def test_valid_backtest_metrics(self):
        """Test creating valid backtest metrics."""
        metrics = BacktestMetricsBase(
            total_return_pct=15.5,
            annual_return_pct=65.2,
            total_pnl_inr=155000.0,
            sharpe_ratio=1.85,
            sortino_ratio=2.1,
            max_drawdown_pct=8.5,
            max_drawdown_duration_days=12,
            volatility_annual_pct=25.3,
            total_trades=150,
            winning_trades=95,
            losing_trades=55,
            win_rate_pct=63.33,
            avg_win_inr=2500.0,
            avg_loss_inr=-1200.0,
            largest_win_inr=15000.0,
            largest_loss_inr=-8000.0,
            profit_factor=2.1,
            total_commission_inr=3000.0,
            total_slippage_inr=750.0,
            total_costs_inr=3750.0,
            avg_trade_duration_minutes=240.5,
            trades_per_day=1.2,
            calmar_ratio=7.67,
            kelly_criterion=12.5,
            expectancy_inr=1033.33
        )
        
        assert metrics.total_return_pct == 15.5
        assert metrics.win_rate_pct == 63.33
        assert metrics.profit_factor == 2.1
        assert metrics.sharpe_ratio == 1.85
        assert metrics.total_trades == 150
        assert metrics.winning_trades == 95
        assert metrics.losing_trades == 55
        
    def test_win_rate_validation(self):
        """Test win rate validation."""
        # Win rate over 100%
        with pytest.raises(ValidationError) as exc_info:
            BacktestMetricsBase(
                total_return_pct=15.5,
                annual_return_pct=65.2,
                total_pnl_inr=155000.0,
                sharpe_ratio=1.85,
                sortino_ratio=2.1,
                max_drawdown_pct=8.5,
                max_drawdown_duration_days=12,
                volatility_annual_pct=25.3,
                total_trades=150,
                winning_trades=95,
                losing_trades=55,
                win_rate_pct=105.0,  # Invalid: over 100%
                avg_win_inr=2500.0,
                avg_loss_inr=-1200.0,
                largest_win_inr=15000.0,
                largest_loss_inr=-8000.0,
                profit_factor=2.1,
                total_commission_inr=3000.0,
                total_slippage_inr=750.0,
                total_costs_inr=3750.0,
                avg_trade_duration_minutes=240.5,
                trades_per_day=1.2,
                calmar_ratio=7.67,
                kelly_criterion=12.5,
                expectancy_inr=1033.33
            )
        # Pydantic v2 generates different error messages - check for either pattern
        error_str = str(exc_info.value)
        assert ("less than or equal to 100" in error_str or 
                "Win rate must be between 0 and 100" in error_str)
        
    def test_negative_value_validation(self):
        """Test validation of fields that must be non-negative."""
        # Negative total trades
        with pytest.raises(ValidationError):
            BacktestMetricsBase(
                total_return_pct=15.5,
                annual_return_pct=65.2,
                total_pnl_inr=155000.0,
                sharpe_ratio=1.85,
                sortino_ratio=2.1,
                max_drawdown_pct=8.5,
                max_drawdown_duration_days=12,
                volatility_annual_pct=25.3,
                total_trades=-5,  # Invalid: negative
                winning_trades=95,
                losing_trades=55,
                win_rate_pct=63.33,
                avg_win_inr=2500.0,
                avg_loss_inr=-1200.0,
                largest_win_inr=15000.0,
                largest_loss_inr=-8000.0,
                profit_factor=2.1,
                total_commission_inr=3000.0,
                total_slippage_inr=750.0,
                total_costs_inr=3750.0,
                avg_trade_duration_minutes=240.5,
                trades_per_day=1.2,
                calmar_ratio=7.67,
                kelly_criterion=12.5,
                expectancy_inr=1033.33
            )


class TestBacktestResultSchemas:
    """Tests for backtest result schemas."""
    
    def test_valid_backtest_result(self):
        """Test creating a valid backtest result."""
        start_time = datetime(2023, 6, 15, 9, 0)
        end_time = datetime(2023, 6, 15, 9, 30)
        
        config = BacktestConfigBase(
            strategy_id=1,
            name="Test Backtest",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 3, 31),
            initial_capital=100000.0
        )
        
        result = BacktestResultBase(
            backtest_id="bt_12345",
            strategy_id=1,
            status=BacktestStatusEnum.COMPLETED,
            start_time=start_time,
            end_time=end_time,
            config=config,
            trade_count=50,
            warnings=["High slippage detected on some trades"]
        )
        
        assert result.backtest_id == "bt_12345"
        assert result.status == BacktestStatusEnum.COMPLETED
        assert result.trade_count == 50
        assert len(result.warnings) == 1
        
    def test_result_time_validation(self):
        """Test result time validation."""
        start_time = datetime(2023, 6, 15, 9, 30)
        invalid_end_time = datetime(2023, 6, 15, 9, 0)  # Before start
        
        config = BacktestConfigBase(
            strategy_id=1,
            name="Test Backtest",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 3, 31),
            initial_capital=100000.0
        )
        
        with pytest.raises(ValidationError) as exc_info:
            BacktestResultBase(
                backtest_id="bt_12345",
                strategy_id=1,
                status=BacktestStatusEnum.COMPLETED,
                start_time=start_time,
                end_time=invalid_end_time,  # Invalid: before start
                config=config
            )
        assert "End time must be after start time" in str(exc_info.value)


class TestBacktestRequestSchemas:
    """Tests for backtest request schemas."""
    
    def test_backtest_run_request(self):
        """Test BacktestRunRequest schema."""
        request = BacktestRunRequest(
            strategy_id=1,
            name="Test Run",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 3, 31),
            initial_capital=100000.0,
            run_immediately=True
        )
        
        assert request.strategy_id == 1
        assert request.run_immediately is True
        
    def test_backtest_list_request(self):
        """Test BacktestListRequest schema."""
        request = BacktestListRequest(
            strategy_id=1,
            status=BacktestStatusEnum.COMPLETED,
            start_date_from=datetime(2023, 1, 1),
            start_date_to=datetime(2023, 12, 31),
            limit=20,
            offset=10
        )
        
        assert request.strategy_id == 1
        assert request.status == BacktestStatusEnum.COMPLETED
        assert request.limit == 20
        assert request.offset == 10
        
    def test_list_request_validation(self):
        """Test BacktestListRequest validation."""
        # Invalid limit
        with pytest.raises(ValidationError):
            BacktestListRequest(limit=0)  # Below minimum
            
        with pytest.raises(ValidationError):
            BacktestListRequest(limit=200)  # Above maximum
            
        # Invalid offset
        with pytest.raises(ValidationError):
            BacktestListRequest(offset=-1)  # Negative
            
    def test_backtest_comparison_request(self):
        """Test BacktestComparisonRequest schema."""
        request = BacktestComparisonRequest(
            backtest_ids=["bt_1", "bt_2", "bt_3"],
            metrics_to_compare=["total_return_pct", "sharpe_ratio", "max_drawdown_pct"]
        )
        
        assert len(request.backtest_ids) == 3
        assert "sharpe_ratio" in request.metrics_to_compare
        
    def test_comparison_request_validation(self):
        """Test BacktestComparisonRequest validation."""
        # Too few backtests
        with pytest.raises(ValidationError):
            BacktestComparisonRequest(backtest_ids=["bt_1"])  # Below minimum
            
        # Too many backtests
        with pytest.raises(ValidationError):
            BacktestComparisonRequest(
                backtest_ids=["bt_1", "bt_2", "bt_3", "bt_4", "bt_5", "bt_6"]  # Above maximum
            )


class TestBacktestUtilitySchemas:
    """Tests for utility schemas like health check and error responses."""
    
    def test_health_response(self):
        """Test BacktestHealthResponse schema."""
        health = BacktestHealthResponse(
            status="healthy",
            service="BacktestingService",
            timestamp=datetime.now(),
            version="1.0.0",
            database_connected=True,
            csv_data_accessible=True,
            last_backtest_time=datetime.now() - timedelta(hours=1),
            running_backtests=2
        )
        
        assert health.status == "healthy"
        assert health.database_connected is True
        assert health.running_backtests == 2
        
    def test_error_response(self):
        """Test BacktestErrorResponse schema."""
        error = BacktestErrorResponse(
            error="ValidationError",
            message="Invalid strategy configuration",
            backtest_id="bt_12345",
            strategy_id=1,
            timestamp=datetime.now(),
            request_id="req_789"
        )
        
        assert error.error == "ValidationError"
        assert error.backtest_id == "bt_12345"
        assert error.strategy_id == 1


class TestSchemaIntegration:
    """Tests for schema integration and complex scenarios."""
    
    def test_complete_backtest_workflow_schemas(self):
        """Test schemas work together in a complete workflow."""
        # 1. Create backtest request
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 3, 31)
        
        run_request = BacktestRunRequest(
            strategy_id=1,
            name="Integration Test",
            description="Testing complete workflow",
            start_date=start_date,
            end_date=end_date,
            initial_capital=100000.0,
            max_position_size=0.1,
            run_immediately=True
        )
        
        # 2. Create trade data
        trade = BacktestTradeResponse(
            trade_id="trade_001",
            signal_id=123,
            strategy_id=1,
            instrument="BANKNIFTY",
            direction=DirectionEnum.LONG,
            entry_time=datetime(2023, 1, 15, 9, 30),
            entry_price=40000.0,
            exit_time=datetime(2023, 1, 15, 15, 30),
            exit_price=40200.0,
            pnl_inr=5000.0,
            is_open=False,
            duration_minutes=360
        )
        
        # 3. Create metrics
        metrics = BacktestMetricsResponse(
            total_return_pct=15.5,
            annual_return_pct=65.2,
            total_pnl_inr=155000.0,
            sharpe_ratio=1.85,
            sortino_ratio=2.1,
            max_drawdown_pct=8.5,
            max_drawdown_duration_days=12,
            volatility_annual_pct=25.3,
            total_trades=50,
            winning_trades=32,
            losing_trades=18,
            win_rate_pct=64.0,
            avg_win_inr=2500.0,
            avg_loss_inr=-1200.0,
            largest_win_inr=15000.0,
            largest_loss_inr=-8000.0,
            profit_factor=2.1,
            total_commission_inr=1000.0,
            total_slippage_inr=250.0,
            total_costs_inr=1250.0,
            avg_trade_duration_minutes=240.0,
            trades_per_day=1.2,
            calmar_ratio=7.67,
            kelly_criterion=12.5,
            expectancy_inr=3100.0,
            calculation_time=datetime.now()
        )
        
        # 4. Create detailed result
        detailed_result = BacktestDetailedResultResponse(
            backtest_id="bt_integration_test",
            strategy_id=1,
            status=BacktestStatusEnum.COMPLETED,
            start_time=datetime.now() - timedelta(minutes=30),
            end_time=datetime.now(),
            config=run_request,
            metrics=metrics,
            trade_count=50,
            duration_seconds=1800.0,
            is_complete=True,
            trades=[trade],
            equity_curve=[{"timestamp": datetime.now(), "equity": 105000.0}],
            monthly_returns=[{"month": "2023-01", "return_pct": 5.0}],
            drawdown_periods=[{"start": datetime(2023, 2, 1), "end": datetime(2023, 2, 5), "drawdown_pct": 3.2}],
            trade_analysis={"best_setup": "A+", "worst_setup": "C"}
        )
        
        # Verify all schemas work together
        assert detailed_result.backtest_id == "bt_integration_test"
        assert detailed_result.trade_count == 50
        assert detailed_result.metrics.total_trades == 50
        assert len(detailed_result.trades) == 1
        assert detailed_result.trades[0].pnl_inr == 5000.0
        assert detailed_result.is_complete is True


# Performance tests for large datasets
class TestSchemaPerformance:
    """Tests for schema performance with large datasets."""
    
    def test_large_trade_list_validation(self):
        """Test validation performance with many trades."""
        trades = []
        for i in range(100):
            trade = BacktestTradeResponse(
                trade_id=f"trade_{i:03d}",
                signal_id=i + 1,
                strategy_id=1,
                instrument="BANKNIFTY",
                direction=DirectionEnum.LONG if i % 2 == 0 else DirectionEnum.SHORT,
                entry_time=datetime(2023, 1, 1) + timedelta(hours=i),
                entry_price=40000.0 + (i * 10),
                exit_time=datetime(2023, 1, 1) + timedelta(hours=i, minutes=30),
                exit_price=40000.0 + (i * 10) + 100,
                pnl_inr=2500.0 if i % 2 == 0 else -1200.0,
                is_open=False,
                duration_minutes=30
            )
            trades.append(trade)
        
        # Should handle large lists without performance issues
        assert len(trades) == 100
        assert all(isinstance(trade, BacktestTradeResponse) for trade in trades)
        
    def test_complex_nested_schema_validation(self):
        """Test validation of deeply nested schema structures."""
        # Create a complex detailed result with nested data
        equity_curve_data = [
            {"timestamp": datetime(2023, 1, 1) + timedelta(days=i), "equity": 100000 + (i * 1000)}
            for i in range(90)  # 90 days of data
        ]
        
        monthly_returns_data = [
            {"month": f"2023-{i:02d}", "return_pct": 5.0 + (i * 0.5)}
            for i in range(1, 4)  # 3 months
        ]
        
        drawdown_periods_data = [
            {
                "start": datetime(2023, 1, 15) + timedelta(days=i*30),
                "end": datetime(2023, 1, 20) + timedelta(days=i*30),
                "drawdown_pct": 2.0 + i
            }
            for i in range(3)
        ]
        
        # This should validate successfully even with large nested structures
        config = BacktestConfigCreate(
            strategy_id=1,
            name="Performance Test",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 3, 31),
            initial_capital=100000.0
        )
        
        metrics = BacktestMetricsBase(
            total_return_pct=15.5,
            annual_return_pct=65.2,
            total_pnl_inr=155000.0,
            sharpe_ratio=1.85,
            sortino_ratio=2.1,
            max_drawdown_pct=8.5,
            max_drawdown_duration_days=12,
            volatility_annual_pct=25.3,
            total_trades=150,
            winning_trades=95,
            losing_trades=55,
            win_rate_pct=63.33,
            avg_win_inr=2500.0,
            avg_loss_inr=-1200.0,
            largest_win_inr=15000.0,
            largest_loss_inr=-8000.0,
            profit_factor=2.1,
            total_commission_inr=3000.0,
            total_slippage_inr=750.0,
            total_costs_inr=3750.0,
            avg_trade_duration_minutes=240.5,
            trades_per_day=1.2,
            calmar_ratio=7.67,
            kelly_criterion=12.5,
            expectancy_inr=1033.33
        )
        
        detailed_result = BacktestDetailedResultResponse(
            backtest_id="bt_performance_test",
            strategy_id=1,
            status=BacktestStatusEnum.COMPLETED,
            start_time=datetime.now() - timedelta(hours=2),
            end_time=datetime.now(),
            config=config,
            metrics=metrics,
            trade_count=150,
            duration_seconds=7200.0,
            is_complete=True,
            trades=[],  # Empty for performance test
            equity_curve=equity_curve_data,
            monthly_returns=monthly_returns_data,
            drawdown_periods=drawdown_periods_data,
            trade_analysis={"performance_test": True}
        )
        
        assert len(detailed_result.equity_curve) == 90
        assert len(detailed_result.monthly_returns) == 3
        assert len(detailed_result.drawdown_periods) == 3
        assert detailed_result.trade_count == 150


# Edge cases and boundary conditions
class TestSchemaEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_minimal_valid_backtest_config(self):
        """Test minimal valid configuration."""
        config = BacktestConfigCreate(
            strategy_id=1,
            name="X",  # Minimal valid name
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 2),  # Minimal valid period
            initial_capital=1.0  # Minimal valid capital
        )
        
        assert config.strategy_id == 1
        assert config.name == "X"
        assert config.initial_capital == 1.0
        
    def test_boundary_position_sizes(self):
        """Test boundary position size values."""
        # Minimum valid position size
        config_min = BacktestConfigCreate(
            strategy_id=1,
            name="Min Position Size",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 2),
            initial_capital=100000.0,
            max_position_size=0.001  # Minimum valid
        )
        assert config_min.max_position_size == 0.001
        
        # Maximum valid position size
        config_max = BacktestConfigCreate(
            strategy_id=1,
            name="Max Position Size",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 2),
            initial_capital=100000.0,
            max_position_size=1.0  # Maximum valid
        )
        assert config_max.max_position_size == 1.0
        
    def test_boundary_win_rates(self):
        """Test boundary win rate values."""
        # 0% win rate
        metrics_zero = BacktestMetricsBase(
            total_return_pct=-50.0,
            annual_return_pct=-80.0,
            total_pnl_inr=-50000.0,
            sharpe_ratio=-1.0,
            sortino_ratio=-1.5,
            max_drawdown_pct=50.0,
            max_drawdown_duration_days=180,
            volatility_annual_pct=45.0,
            total_trades=100,
            winning_trades=0,
            losing_trades=100,
            win_rate_pct=0.0,  # Boundary: 0%
            avg_win_inr=0.0,
            avg_loss_inr=-500.0,
            largest_win_inr=0.0,
            largest_loss_inr=-5000.0,
            profit_factor=0.0,
            total_commission_inr=2000.0,
            total_slippage_inr=500.0,
            total_costs_inr=2500.0,
            avg_trade_duration_minutes=120.0,
            trades_per_day=2.0,
            calmar_ratio=-1.6,
            kelly_criterion=0.0,
            expectancy_inr=-500.0
        )
        assert metrics_zero.win_rate_pct == 0.0
        
        # 100% win rate
        metrics_perfect = BacktestMetricsBase(
            total_return_pct=100.0,
            annual_return_pct=400.0,
            total_pnl_inr=1000000.0,
            sharpe_ratio=5.0,
            sortino_ratio=8.0,
            max_drawdown_pct=0.0,
            max_drawdown_duration_days=0,
            volatility_annual_pct=15.0,
            total_trades=50,
            winning_trades=50,
            losing_trades=0,
            win_rate_pct=100.0,  # Boundary: 100%
            avg_win_inr=20000.0,
            avg_loss_inr=0.0,
            largest_win_inr=50000.0,
            largest_loss_inr=0.0,
            profit_factor=float('inf'),
            total_commission_inr=1000.0,
            total_slippage_inr=250.0,
            total_costs_inr=1250.0,
            avg_trade_duration_minutes=360.0,
            trades_per_day=0.5,
            calmar_ratio=float('inf'),
            kelly_criterion=25.0,
            expectancy_inr=20000.0
        )
        assert metrics_perfect.win_rate_pct == 100.0
        
    def test_extreme_date_ranges(self):
        """Test extreme but valid date ranges."""
        # Very short period (1 day)
        config_short = BacktestConfigCreate(
            strategy_id=1,
            name="Short Period",
            start_date=datetime(2023, 1, 1, 9, 0),
            end_date=datetime(2023, 1, 1, 15, 30),
            initial_capital=100000.0
        )
        assert (config_short.end_date - config_short.start_date).total_seconds() == 23400  # 6.5 hours
        
        # Very long period (1 year)
        config_long = BacktestConfigCreate(
            strategy_id=1,
            name="Long Period",
            start_date=datetime(2022, 1, 1),
            end_date=datetime(2022, 12, 31),
            initial_capital=100000.0
        )
        assert (config_long.end_date - config_long.start_date).days == 364
        
    def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters in text fields."""
        config = BacktestConfigCreate(
            strategy_id=1,
            name="Test with émojis 🚀📈 and ünïcödé",
            description="Special chars: @#$%^&*()_+-=[]{}|;':\",./<>?",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 3, 31),
            initial_capital=100000.0
        )
        
        assert "🚀📈" in config.name
        assert "ünïcödé" in config.name
        assert "@#$%^&*()" in config.description
        
    def test_zero_and_negative_edge_cases(self):
        """Test handling of zero and near-zero values."""
        # Zero trades scenario
        metrics_no_trades = BacktestMetricsBase(
            total_return_pct=0.0,
            annual_return_pct=0.0,
            total_pnl_inr=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown_pct=0.0,
            max_drawdown_duration_days=0,
            volatility_annual_pct=0.0,
            total_trades=0,  # Zero trades
            winning_trades=0,
            losing_trades=0,
            win_rate_pct=0.0,
            avg_win_inr=0.0,
            avg_loss_inr=0.0,
            largest_win_inr=0.0,
            largest_loss_inr=0.0,
            profit_factor=0.0,
            total_commission_inr=0.0,
            total_slippage_inr=0.0,
            total_costs_inr=0.0,
            avg_trade_duration_minutes=0.0,
            trades_per_day=0.0,
            calmar_ratio=0.0,
            kelly_criterion=0.0,
            expectancy_inr=0.0
        )
        
        assert metrics_no_trades.total_trades == 0
        assert metrics_no_trades.win_rate_pct == 0.0


# Schema serialization and deserialization tests
class TestSchemaSerialization:
    """Tests for schema serialization and deserialization."""
    
    def test_config_json_serialization(self):
        """Test JSON serialization of configuration."""
        config = BacktestConfigCreate(
            strategy_id=1,
            name="Serialization Test",
            description="Testing JSON serialization",
            start_date=datetime(2023, 1, 1, 9, 0),
            end_date=datetime(2023, 3, 31, 15, 30),
            initial_capital=100000.0,
            max_position_size=0.1,
            commission_per_trade=20.0,
            slippage_bps=2.0
        )
        
        # Test model_dump (Pydantic v2)
        config_dict = config.model_dump()
        assert isinstance(config_dict, dict)
        assert config_dict["strategy_id"] == 1
        assert config_dict["name"] == "Serialization Test"
        assert isinstance(config_dict["start_date"], datetime)
        
        # Test JSON serialization with custom encoder
        config_json = config.model_dump(mode='json')
        assert isinstance(config_json, dict)
        # Dates should be serialized as ISO strings in JSON mode
        assert isinstance(config_json["start_date"], str)
        assert "2023-01-01T09:00:00" in config_json["start_date"]
        
    def test_trade_serialization_with_optional_fields(self):
        """Test serialization of trades with optional fields."""
        # Trade without exit (open trade)
        open_trade = BacktestTradeBase(
            trade_id="open_trade_001",
            signal_id=123,
            strategy_id=1,
            instrument="BANKNIFTY",
            direction=DirectionEnum.LONG,
            entry_time=datetime(2023, 1, 15, 9, 30),
            entry_price=40000.0
            # No exit fields - should serialize correctly
        )
        
        trade_dict = open_trade.model_dump()
        assert trade_dict["trade_id"] == "open_trade_001"
        assert trade_dict["exit_time"] is None
        assert trade_dict["exit_price"] is None
        assert trade_dict["pnl_inr"] is None
        
    def test_metrics_json_mode_serialization(self):
        """Test metrics serialization in JSON mode."""
        calculation_time = datetime(2023, 6, 15, 10, 30, 0)
        
        metrics = BacktestMetricsResponse(
            total_return_pct=15.5,
            annual_return_pct=65.2,
            total_pnl_inr=155000.0,
            sharpe_ratio=1.85,
            sortino_ratio=2.1,
            max_drawdown_pct=8.5,
            max_drawdown_duration_days=12,
            volatility_annual_pct=25.3,
            total_trades=150,
            winning_trades=95,
            losing_trades=55,
            win_rate_pct=63.33,
            avg_win_inr=2500.0,
            avg_loss_inr=-1200.0,
            largest_win_inr=15000.0,
            largest_loss_inr=-8000.0,
            profit_factor=2.1,
            total_commission_inr=3000.0,
            total_slippage_inr=750.0,
            total_costs_inr=3750.0,
            avg_trade_duration_minutes=240.5,
            trades_per_day=1.2,
            calmar_ratio=7.67,
            kelly_criterion=12.5,
            expectancy_inr=1033.33,
            calculation_time=calculation_time
        )
        
        # JSON mode should serialize datetime as ISO string
        metrics_json = metrics.model_dump(mode='json')
        assert isinstance(metrics_json["calculation_time"], str)
        assert "2023-06-15T10:30:00" in metrics_json["calculation_time"]
        
        # Regular mode should keep datetime objects
        metrics_dict = metrics.model_dump()
        assert isinstance(metrics_dict["calculation_time"], datetime)
        assert metrics_dict["calculation_time"] == calculation_time


# Run the tests
if __name__ == "__main__":
    # Run with: python -m pytest test_backtest_schemas.py -v
    pytest.main([__file__, "-v", "-s"])