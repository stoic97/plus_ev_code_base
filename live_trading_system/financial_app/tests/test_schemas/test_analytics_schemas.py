"""
Tests for Analytics API response schemas.

This module contains comprehensive tests for all analytics response models,
covering validation, serialization, field constraints, and business logic.
"""

import pytest
import sys
import os
from datetime import datetime, timedelta
from pydantic import ValidationError
from typing import Dict, List, Optional

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import all analytics schemas to test
from app.schemas.analytics import (
    TimeframeEnum, AttributionTypeEnum,
    AnalyticsMetricBase, TimestampedMetric,
    LiveMetricsResponse, EquityCurvePoint, EquityCurveResponse,
    DrawdownPeriod, DrawdownAnalysisResponse,
    RiskMetricsResponse, AttributionByGrade, AttributionByTime,
    AttributionResponse, BenchmarkComparisonResponse,
    StrategyPortfolioSummary, PortfolioSummaryResponse,
    AnalyticsHealthResponse, AnalyticsErrorResponse,
    PaginatedAnalyticsResponse, TimeSeriesDataPoint,
    AnalyticsFilterOptions
)


class TestEnums:
    """Test analytics enums."""
    
    def test_timeframe_enum_values(self):
        """Test TimeframeEnum has correct values."""
        assert TimeframeEnum.ONE_DAY == "1D"
        assert TimeframeEnum.ONE_WEEK == "1W"
        assert TimeframeEnum.ONE_MONTH == "1M"
        assert TimeframeEnum.THREE_MONTHS == "3M"
        assert TimeframeEnum.ONE_YEAR == "1Y"
        assert TimeframeEnum.ALL == "ALL"
    
    def test_attribution_type_enum_values(self):
        """Test AttributionTypeEnum has correct values."""
        assert AttributionTypeEnum.GRADE == "grade"
        assert AttributionTypeEnum.TIME == "time"
        assert AttributionTypeEnum.MARKET_STATE == "market_state"


class TestAnalyticsMetricBase:
    """Test AnalyticsMetricBase schema."""
    
    def test_valid_float_metric(self):
        """Test creating metric with float value."""
        metric = AnalyticsMetricBase(
            value=1250.75,
            formatted_value="$1,250.75",
            change_percent=5.2,
            is_positive=True
        )
        
        assert metric.value == 1250.75
        assert metric.formatted_value == "$1,250.75"
        assert metric.change_percent == 5.2
        assert metric.is_positive is True
    
    def test_valid_int_metric(self):
        """Test creating metric with integer value."""
        metric = AnalyticsMetricBase(
            value=42,
            formatted_value="42 trades",
            change_percent=-2.1,
            is_positive=False
        )
        
        assert metric.value == 42
        assert metric.formatted_value == "42 trades"
        assert metric.change_percent == -2.1
        assert metric.is_positive is False
    
    def test_optional_fields(self):
        """Test metric with only required fields."""
        metric = AnalyticsMetricBase(value=100.0)
        
        assert metric.value == 100.0
        assert metric.formatted_value is None
        assert metric.change_percent is None
        assert metric.is_positive is None


class TestTimestampedMetric:
    """Test TimestampedMetric schema."""
    
    def test_valid_timestamped_metric(self):
        """Test creating valid timestamped metric."""
        timestamp = datetime.utcnow()
        metric = TimestampedMetric(
            timestamp=timestamp,
            value=1050000.0
        )
        
        assert metric.timestamp == timestamp
        assert metric.value == 1050000.0
    
    def test_negative_value_allowed(self):
        """Test that negative values are allowed."""
        metric = TimestampedMetric(
            timestamp=datetime.utcnow(),
            value=-5000.0
        )
        
        assert metric.value == -5000.0


class TestLiveMetricsResponse:
    """Test LiveMetricsResponse schema."""
    
    def test_valid_live_metrics(self):
        """Test creating valid live metrics response."""
        metrics = LiveMetricsResponse(
            nav=1050000.0,
            initial_capital=1000000.0,
            cash=950000.0,
            positions_value=100000.0,
            total_pnl=50000.0,
            realized_pnl=45000.0,
            unrealized_pnl=5000.0,
            total_return_percent=5.0,
            current_drawdown_percent=-2.5,
            high_water_mark=1075000.0,
            active_trades=3,
            total_trades=47,
            last_updated=datetime.utcnow()
        )
        
        assert metrics.nav == 1050000.0
        assert metrics.initial_capital == 1000000.0
        assert metrics.total_pnl == 50000.0
        assert metrics.total_return_percent == 5.0
        assert metrics.active_trades == 3
        assert metrics.total_trades == 47
    
    def test_optional_fields(self):
        """Test live metrics with optional fields."""
        metrics = LiveMetricsResponse(
            nav=1050000.0,
            initial_capital=1000000.0,
            cash=950000.0,
            positions_value=100000.0,
            total_pnl=50000.0,
            realized_pnl=45000.0,
            unrealized_pnl=5000.0,
            total_return_percent=5.0,
            current_drawdown_percent=-2.5,
            high_water_mark=1075000.0,
            active_trades=3,
            total_trades=47,
            last_updated=datetime.utcnow(),
            daily_pnl=2500.0,
            daily_return_percent=0.25,
            trades_today=2
        )
        
        assert metrics.daily_pnl == 2500.0
        assert metrics.daily_return_percent == 0.25
        assert metrics.trades_today == 2
    
    def test_percentage_validation(self):
        """Test percentage field validation."""
        # Valid percentages should pass
        metrics = LiveMetricsResponse(
            nav=1050000.0,
            initial_capital=1000000.0,
            cash=950000.0,
            positions_value=100000.0,
            total_pnl=50000.0,
            realized_pnl=45000.0,
            unrealized_pnl=5000.0,
            total_return_percent=50.0,  # Valid
            current_drawdown_percent=-50.0,  # Valid
            high_water_mark=1075000.0,
            active_trades=3,
            total_trades=47,
            last_updated=datetime.utcnow()
        )
        
        assert metrics.total_return_percent == 50.0
        assert metrics.current_drawdown_percent == -50.0
    
    def test_extreme_percentage_validation(self):
        """Test that extreme percentages are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            LiveMetricsResponse(
                nav=1050000.0,
                initial_capital=1000000.0,
                cash=950000.0,
                positions_value=100000.0,
                total_pnl=50000.0,
                realized_pnl=45000.0,
                unrealized_pnl=5000.0,
                total_return_percent=1500.0,  # Too high
                current_drawdown_percent=-2.5,
                high_water_mark=1075000.0,
                active_trades=3,
                total_trades=47,
                last_updated=datetime.utcnow()
            )
        
        assert "should be between -100% and 1000%" in str(exc_info.value)


class TestEquityCurveResponse:
    """Test EquityCurveResponse schema."""
    
    def test_valid_equity_curve(self):
        """Test creating valid equity curve response."""
        start_date = datetime.utcnow() - timedelta(days=30)
        end_date = datetime.utcnow()
        
        curve_points = [
            EquityCurvePoint(
                timestamp=start_date + timedelta(days=i),
                nav=1000000.0 + (i * 1000),
                cumulative_pnl=i * 1000,
                daily_return_percent=0.1,
                trade_id=i + 1
            )
            for i in range(5)
        ]
        
        curve = EquityCurveResponse(
            strategy_id=1,
            period=TimeframeEnum.ONE_MONTH,
            start_date=start_date,
            end_date=end_date,
            data_points=curve_points,
            total_points=5,
            starting_nav=1000000.0,
            ending_nav=1004000.0,
            total_return_percent=0.4
        )
        
        assert curve.strategy_id == 1
        assert curve.period == TimeframeEnum.ONE_MONTH
        assert len(curve.data_points) == 5
        assert curve.total_points == 5
        assert curve.starting_nav == 1000000.0
        assert curve.ending_nav == 1004000.0
        assert curve.total_return_percent == 0.4
    
    def test_empty_data_points_validation(self):
        """Test that empty data points are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            EquityCurveResponse(
                strategy_id=1,
                period=TimeframeEnum.ONE_MONTH,
                start_date=datetime.utcnow(),
                end_date=datetime.utcnow(),
                data_points=[],  # Empty list
                total_points=0,
                starting_nav=1000000.0,
                ending_nav=1000000.0,
                total_return_percent=0.0
            )
        
        assert "must have at least one data point" in str(exc_info.value)


class TestRiskMetricsResponse:
    """Test RiskMetricsResponse schema."""
    
    def test_valid_risk_metrics(self):
        """Test creating valid risk metrics response."""
        metrics = RiskMetricsResponse(
            strategy_id=1,
            calculation_date=datetime.utcnow(),
            lookback_days=30,
            sharpe_ratio=1.85,
            sortino_ratio=2.10,
            calmar_ratio=3.2,
            information_ratio=1.5,
            volatility_annualized=0.15,
            downside_volatility=0.12,
            var_95=15000.0,
            var_99=22000.0,
            expected_shortfall_95=18000.0,
            var_confidence=0.95,
            max_drawdown_percent=-5.2,
            current_drawdown_percent=-2.1,
            max_drawdown_duration_days=7,
            skewness=0.2,
            kurtosis=3.1,
            beta=0.85
        )
        
        assert metrics.strategy_id == 1
        assert metrics.sharpe_ratio == 1.85
        assert metrics.sortino_ratio == 2.10
        assert metrics.var_95 == 15000.0
        assert metrics.beta == 0.85
    
    def test_optional_fields(self):
        """Test risk metrics with only required fields."""
        metrics = RiskMetricsResponse(
            strategy_id=1,
            calculation_date=datetime.utcnow(),
            lookback_days=30,
            sharpe_ratio=1.85,
            sortino_ratio=2.10,
            calmar_ratio=3.2,
            volatility_annualized=0.15,
            downside_volatility=0.12,
            var_95=15000.0,
            var_99=22000.0,
            var_confidence=0.95,
            max_drawdown_percent=-5.2,
            current_drawdown_percent=-2.1,
            max_drawdown_duration_days=7
        )
        
        assert metrics.information_ratio is None
        assert metrics.expected_shortfall_95 is None
        assert metrics.skewness is None
        assert metrics.kurtosis is None
        assert metrics.beta is None
    
    def test_ratio_validation(self):
        """Test risk ratio validation."""
        # Valid ratios should pass
        metrics = RiskMetricsResponse(
            strategy_id=1,
            calculation_date=datetime.utcnow(),
            lookback_days=30,
            sharpe_ratio=2.5,
            sortino_ratio=3.0,
            calmar_ratio=4.0,
            volatility_annualized=0.15,
            downside_volatility=0.12,
            var_95=15000.0,
            var_99=22000.0,
            var_confidence=0.95,
            max_drawdown_percent=-5.2,
            current_drawdown_percent=-2.1,
            max_drawdown_duration_days=7
        )
        
        assert metrics.sharpe_ratio == 2.5
        assert metrics.sortino_ratio == 3.0
        assert metrics.calmar_ratio == 4.0
    
    def test_extreme_ratio_validation(self):
        """Test that extreme ratios are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            RiskMetricsResponse(
                strategy_id=1,
                calculation_date=datetime.utcnow(),
                lookback_days=30,
                sharpe_ratio=15.0,  # Too high
                sortino_ratio=2.10,
                calmar_ratio=3.2,
                volatility_annualized=0.15,
                downside_volatility=0.12,
                var_95=15000.0,
                var_99=22000.0,
                var_confidence=0.95,
                max_drawdown_percent=-5.2,
                current_drawdown_percent=-2.1,
                max_drawdown_duration_days=7
            )
        
        assert "should be between -10 and 10" in str(exc_info.value)


class TestAttributionSchemas:
    """Test attribution-related schemas."""
    
    def test_attribution_by_grade(self):
        """Test AttributionByGrade schema."""
        attribution = AttributionByGrade(
            grade="A_PLUS",
            trade_count=15,
            total_pnl=30000.0,
            win_rate=0.80,
            average_pnl=2000.0,
            pnl_contribution_percent=60.0
        )
        
        assert attribution.grade == "A_PLUS"
        assert attribution.trade_count == 15
        assert attribution.total_pnl == 30000.0
        assert attribution.win_rate == 0.80
        assert attribution.pnl_contribution_percent == 60.0
    
    def test_attribution_by_time(self):
        """Test AttributionByTime schema."""
        attribution = AttributionByTime(
            time_period="09:15-10:00",
            trade_count=8,
            total_pnl=12000.0,
            win_rate=0.75,
            average_pnl=1500.0
        )
        
        assert attribution.time_period == "09:15-10:00"
        assert attribution.trade_count == 8
        assert attribution.total_pnl == 12000.0
        assert attribution.win_rate == 0.75
        assert attribution.average_pnl == 1500.0
    
    def test_win_rate_validation(self):
        """Test win rate validation in attribution."""
        # Valid win rate
        attribution = AttributionByGrade(
            grade="A",
            trade_count=10,
            total_pnl=15000.0,
            win_rate=0.6,
            average_pnl=1500.0,
            pnl_contribution_percent=30.0
        )
        
        assert attribution.win_rate == 0.6
        
        # Invalid win rate
        with pytest.raises(ValidationError) as exc_info:
            AttributionByGrade(
                grade="A",
                trade_count=10,
                total_pnl=15000.0,
                win_rate=1.5,  # Invalid
                average_pnl=1500.0,
                pnl_contribution_percent=30.0
            )
        
        assert "must be between 0 and 1" in str(exc_info.value)


class TestPortfolioSummaryResponse:
    """Test PortfolioSummaryResponse schema."""
    
    def test_valid_portfolio_summary(self):
        """Test creating valid portfolio summary."""
        strategy_summaries = [
            StrategyPortfolioSummary(
                strategy_id=1,
                strategy_name="Strategy 1",
                nav=1050000.0,
                total_pnl=50000.0,
                return_percent=5.0,
                weight_percent=52.5,
                last_trade_date=datetime.utcnow() - timedelta(hours=2),
                is_active=True
            ),
            StrategyPortfolioSummary(
                strategy_id=2,
                strategy_name="Strategy 2",
                nav=950000.0,
                total_pnl=-50000.0,
                return_percent=-5.0,
                weight_percent=47.5,
                last_trade_date=None,
                is_active=False
            )
        ]
        
        portfolio = PortfolioSummaryResponse(
            user_id=1,
            total_strategies=2,
            active_strategies=1,
            total_nav=2000000.0,
            total_capital=2000000.0,
            total_pnl=0.0,
            portfolio_return_percent=0.0,
            average_nav_per_strategy=1000000.0,
            best_performing_strategy_id=1,
            worst_performing_strategy_id=2,
            strategies=strategy_summaries,
            last_updated=datetime.utcnow()
        )
        
        assert portfolio.user_id == 1
        assert portfolio.total_strategies == 2
        assert portfolio.active_strategies == 1
        assert len(portfolio.strategies) == 2
        assert portfolio.best_performing_strategy_id == 1
        assert portfolio.worst_performing_strategy_id == 2
    
    def test_strategy_count_validation(self):
        """Test strategy count validation."""
        with pytest.raises(ValidationError) as exc_info:
            PortfolioSummaryResponse(
                user_id=1,
                total_strategies=-1,  # Invalid
                active_strategies=0,
                total_nav=1000000.0,
                total_capital=1000000.0,
                total_pnl=0.0,
                portfolio_return_percent=0.0,
                average_nav_per_strategy=1000000.0,
                best_performing_strategy_id=None,
                worst_performing_strategy_id=None,
                strategies=[],
                last_updated=datetime.utcnow()
            )
        
        assert "cannot be negative" in str(exc_info.value)


class TestHealthAndErrorResponses:
    """Test health check and error response schemas."""
    
    def test_analytics_health_response(self):
        """Test AnalyticsHealthResponse schema."""
        health = AnalyticsHealthResponse(
            status="healthy",
            service="analytics",
            timestamp=datetime.utcnow(),
            version="1.0.0",
            database_connected=True,
            last_calculation_time=datetime.utcnow() - timedelta(minutes=5)
        )
        
        assert health.status == "healthy"
        assert health.service == "analytics"
        assert health.version == "1.0.0"
        assert health.database_connected is True
        assert health.last_calculation_time is not None
    
    def test_analytics_error_response(self):
        """Test AnalyticsErrorResponse schema."""
        error = AnalyticsErrorResponse(
            error="ValidationError",
            message="Invalid strategy ID provided",
            strategy_id=999,
            timestamp=datetime.utcnow(),
            request_id="req_12345"
        )
        
        assert error.error == "ValidationError"
        assert error.message == "Invalid strategy ID provided"
        assert error.strategy_id == 999
        assert error.request_id == "req_12345"


class TestUtilitySchemas:
    """Test utility and helper schemas."""
    
    def test_paginated_response(self):
        """Test PaginatedAnalyticsResponse schema."""
        paginated = PaginatedAnalyticsResponse(
            total_count=150,
            page=2,
            page_size=50,
            total_pages=3,
            has_next=True,
            has_previous=True
        )
        
        assert paginated.total_count == 150
        assert paginated.page == 2
        assert paginated.page_size == 50
        assert paginated.total_pages == 3
        assert paginated.has_next is True
        assert paginated.has_previous is True
    
    def test_time_series_data_point(self):
        """Test TimeSeriesDataPoint schema."""
        point = TimeSeriesDataPoint(
            timestamp=datetime.utcnow(),
            value=1250.75,
            label="NAV",
            metadata={"source": "live", "confidence": 0.95}
        )
        
        assert point.value == 1250.75
        assert point.label == "NAV"
        assert point.metadata["source"] == "live"
        assert point.metadata["confidence"] == 0.95


class TestSchemaIntegration:
    """Test schema integration and real-world usage scenarios."""
    
    def test_complete_dashboard_workflow(self):
        """Test creating a complete dashboard response."""
        dashboard = LiveMetricsResponse(
            nav=1125000.0,
            initial_capital=1000000.0,
            cash=925000.0,
            positions_value=200000.0,
            total_pnl=125000.0,
            realized_pnl=100000.0,
            unrealized_pnl=25000.0,
            daily_pnl=5000.0,
            total_return_percent=12.5,
            daily_return_percent=0.45,
            current_drawdown_percent=-1.2,
            high_water_mark=1135000.0,
            active_trades=5,
            total_trades=75,
            trades_today=3,
            last_updated=datetime.utcnow()
        )
        
        # Verify all calculations are consistent
        assert dashboard.total_pnl == dashboard.realized_pnl + dashboard.unrealized_pnl
        assert dashboard.nav == dashboard.cash + dashboard.positions_value
        expected_return = ((dashboard.nav - dashboard.initial_capital) / dashboard.initial_capital) * 100
        assert abs(dashboard.total_return_percent - expected_return) < 0.01
    
    def test_attribution_analysis_totals(self):
        """Test that attribution analysis totals add up correctly."""
        grade_attributions = [
            AttributionByGrade(
                grade="A_PLUS",
                trade_count=12,
                total_pnl=36000.0,
                win_rate=0.83,
                average_pnl=3000.0,
                pnl_contribution_percent=60.0
            ),
            AttributionByGrade(
                grade="A",
                trade_count=18,
                total_pnl=18000.0,
                win_rate=0.67,
                average_pnl=1000.0,
                pnl_contribution_percent=30.0
            ),
            AttributionByGrade(
                grade="B",
                trade_count=10,
                total_pnl=6000.0,
                win_rate=0.60,
                average_pnl=600.0,
                pnl_contribution_percent=10.0
            )
        ]
        
        attribution_response = AttributionResponse(
            strategy_id=1,
            attribution_type=AttributionTypeEnum.GRADE,
            analysis_period="All Time",
            start_date=None,
            end_date=datetime.utcnow(),
            by_grade=grade_attributions,
            by_time=None,
            by_market_state=None,
            total_trades=40,
            total_pnl=60000.0,
            analysis_completeness=100.0
        )
        
        # Verify totals add up
        total_trades_calc = sum(attr.trade_count for attr in grade_attributions)
        total_pnl_calc = sum(attr.total_pnl for attr in grade_attributions)
        total_contribution = sum(attr.pnl_contribution_percent for attr in grade_attributions)
        
        assert total_trades_calc == attribution_response.total_trades
        assert total_pnl_calc == attribution_response.total_pnl
        assert abs(total_contribution - 100.0) < 0.01
    
    def test_risk_metrics_consistency(self):
        """Test that risk metrics are internally consistent."""
        risk_metrics = RiskMetricsResponse(
            strategy_id=1,
            calculation_date=datetime.utcnow(),
            lookback_days=60,
            sharpe_ratio=1.75,
            sortino_ratio=2.25,  # Should be higher than Sharpe for positive strategies
            calmar_ratio=3.0,
            information_ratio=1.2,
            volatility_annualized=0.16,
            downside_volatility=0.11,  # Should be lower than total volatility
            var_95=18000.0,
            var_99=25000.0,  # Should be higher than VaR 95
            expected_shortfall_95=22000.0,  # Should be between VaR 95 and 99
            var_confidence=0.95,
            max_drawdown_percent=-6.2,
            current_drawdown_percent=-2.1,
            max_drawdown_duration_days=12,
            skewness=0.15,
            kurtosis=3.2,
            beta=0.78
        )
        
        # Verify logical relationships
        assert risk_metrics.sortino_ratio > risk_metrics.sharpe_ratio
        assert risk_metrics.downside_volatility < risk_metrics.volatility_annualized
        assert risk_metrics.var_99 > risk_metrics.var_95
        assert risk_metrics.var_95 < risk_metrics.expected_shortfall_95 < risk_metrics.var_99
        assert risk_metrics.current_drawdown_percent > risk_metrics.max_drawdown_percent


class TestEdgeCasesAndBoundaries:
    """Test edge cases and boundary conditions."""
    
    def test_zero_values_allowed(self):
        """Test that zero values are handled correctly."""
        metrics = LiveMetricsResponse(
            nav=1000000.0,
            initial_capital=1000000.0,
            cash=1000000.0,
            positions_value=0.0,
            total_pnl=0.0,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            total_return_percent=0.0,
            current_drawdown_percent=0.0,
            high_water_mark=1000000.0,
            active_trades=0,
            total_trades=0,
            last_updated=datetime.utcnow()
        )
        
        assert metrics.positions_value == 0.0
        assert metrics.total_pnl == 0.0
        assert metrics.active_trades == 0
        assert metrics.total_trades == 0
    
    def test_negative_values_where_appropriate(self):
        """Test that negative values are allowed where appropriate."""
        metrics = LiveMetricsResponse(
            nav=950000.0,
            initial_capital=1000000.0,
            cash=950000.0,
            positions_value=0.0,
            total_pnl=-50000.0,
            realized_pnl=-50000.0,
            unrealized_pnl=0.0,
            total_return_percent=-5.0,
            current_drawdown_percent=-5.0,
            high_water_mark=1000000.0,
            active_trades=0,
            total_trades=10,
            last_updated=datetime.utcnow()
        )
        
        assert metrics.total_pnl == -50000.0
        assert metrics.total_return_percent == -5.0
        assert metrics.current_drawdown_percent == -5.0


class TestSerializationAndDeserialization:
    """Test JSON serialization and deserialization."""
    
    def test_live_metrics_json_serialization(self):
        """Test that LiveMetricsResponse can be serialized to JSON."""
        metrics = LiveMetricsResponse(
            nav=1050000.0,
            initial_capital=1000000.0,
            cash=950000.0,
            positions_value=100000.0,
            total_pnl=50000.0,
            realized_pnl=45000.0,
            unrealized_pnl=5000.0,
            total_return_percent=5.0,
            current_drawdown_percent=-2.5,
            high_water_mark=1075000.0,
            active_trades=3,
            total_trades=47,
            last_updated=datetime.utcnow()
        )
        
        # Test model_dump (Pydantic v2 method)
        json_data = metrics.model_dump()
        
        assert isinstance(json_data, dict)
        assert json_data['nav'] == 1050000.0
        assert json_data['total_pnl'] == 50000.0
        assert 'last_updated' in json_data
    
    def test_datetime_serialization_format(self):
        """Test that datetimes are serialized correctly."""
        test_time = datetime(2024, 1, 15, 14, 30, 45)
        
        point = EquityCurvePoint(
            timestamp=test_time,
            nav=1000000.0,
            cumulative_pnl=0.0,
            daily_return_percent=0.0
        )
        
        json_data = point.model_dump()
        
        assert 'timestamp' in json_data
        assert isinstance(json_data['timestamp'], datetime)


class TestDocumentationAndMetadata:
    """Test that schemas have proper documentation and metadata."""
    
    def test_field_descriptions_present(self):
        """Test that important fields have descriptions."""
        fields = LiveMetricsResponse.model_fields
        
        assert 'nav' in fields
        assert fields['nav'].description == "Current Net Asset Value"
        
        assert 'total_pnl' in fields
        assert fields['total_pnl'].description == "Total profit/loss"
        
        assert 'active_trades' in fields
        assert fields['active_trades'].description == "Number of active trades"
    
    def test_enum_documentation(self):
        """Test that enums have proper values and are documented."""
        timeframe_values = [e.value for e in TimeframeEnum]
        expected_values = ["1D", "1W", "1M", "3M", "1Y", "ALL"]
        
        assert set(timeframe_values) == set(expected_values)
        
        attribution_values = [e.value for e in AttributionTypeEnum]
        expected_attribution = ["grade", "time", "market_state"]
        
        assert set(attribution_values) == set(expected_attribution)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])