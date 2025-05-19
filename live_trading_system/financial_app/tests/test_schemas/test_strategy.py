"""
Unit tests for strategy Pydantic schemas.

These tests ensure that validation, serialization, and relationships between
schema fields work correctly according to the business rules of the trading application.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError
from typing import Dict, List, Optional

# Import the schemas to test
from app.schemas.strategy import (
    TimeframeValueEnum, DirectionEnum, TimeframeImportanceEnum, EntryTechniqueEnum,
    MarketStateRequirementEnum, TrendPhaseEnum, ProfitTargetMethodEnum, SetupQualityGradeEnum,
    SpreadTypeEnum, FeedbackTypeEnum, MovingAverageTypeEnum, TrailingStopMethodEnum,
    StrategyTypeEnum, BaseSchema, TimeframeSettingsBase, InstitutionalBehaviorSettingsBase,
    EntryExitSettingsBase, MarketStateSettingsBase, RiskManagementSettingsBase,
    SetupQualityCriteriaBase, VerticalSpreadSettingsBase, MetaLearningSettingsBase,
    MultiTimeframeConfirmationSettingsBase, StrategyBase, StrategyCreate, StrategyUpdate,
    StrategyResponse, TimeframeAnalysisResult, MarketStateAnalysis, SetupQualityResult,
    SignalBase, SignalCreate, SignalResponse, TradeBase, TradeCreate, TradeResponse,
    FeedbackBase, FeedbackCreate, FeedbackResponse, PerformanceAnalysis
)


class TestBaseSchemas:
    """Tests for base schema functionality."""
    
    def test_base_schema_orm_mode(self):
        """Test that BaseSchema has from_attributes enabled (formerly orm_mode)."""
        assert BaseSchema.model_config["from_attributes"] is True
        assert BaseSchema.model_config["arbitrary_types_allowed"] is True
        
    def test_datetime_serialization(self):
        """Test that datetime values are serialized to ISO format."""
        test_time = datetime(2023, 1, 1, 12, 0, 0)
        encoder = BaseSchema.model_config["json_encoders"][datetime]
        assert encoder(test_time) == "2023-01-01T12:00:00"


class TestTimeframeSettingsSchema:
    """Tests for TimeframeSettings schema."""
    
    def test_valid_timeframe_settings(self):
        """Test creating a valid timeframe settings object."""
        timeframe = TimeframeSettingsBase(
            name="Hourly",
            value=TimeframeValueEnum.ONE_HOUR,
            importance=TimeframeImportanceEnum.PRIMARY,
            order=1,
            ma_type=MovingAverageTypeEnum.SIMPLE,
            ma_period_primary=21,
            ma_period_secondary=200,
            require_alignment=True
        )
        
        assert timeframe.name == "Hourly"
        assert timeframe.value == TimeframeValueEnum.ONE_HOUR
        assert timeframe.importance == TimeframeImportanceEnum.PRIMARY
        assert timeframe.order == 1
        assert timeframe.ma_period_primary == 21
        assert timeframe.ma_period_secondary == 200
        
    def test_default_values(self):
        """Test default values are set correctly."""
        timeframe = TimeframeSettingsBase(
            name="Daily",
            value=TimeframeValueEnum.DAILY,
            importance=TimeframeImportanceEnum.PRIMARY
        )
        
        assert timeframe.ma_type == MovingAverageTypeEnum.SIMPLE
        assert timeframe.ma_period_primary == 21
        assert timeframe.ma_period_secondary == 200
        assert timeframe.require_alignment is True
        
    @pytest.mark.parametrize("ma_period", [3, 0, -5, 1001])
    def test_invalid_ma_periods(self, ma_period):
        """Test validation of MA periods."""
        with pytest.raises(ValidationError):
            TimeframeSettingsBase(
                name="Invalid",
                value=TimeframeValueEnum.DAILY,
                importance=TimeframeImportanceEnum.PRIMARY,
                ma_period_primary=ma_period
            )
            
        with pytest.raises(ValidationError):
            TimeframeSettingsBase(
                name="Invalid",
                value=TimeframeValueEnum.DAILY,
                importance=TimeframeImportanceEnum.PRIMARY,
                ma_period_secondary=ma_period
            )


class TestInstitutionalBehaviorSettingsSchema:
    """Tests for InstitutionalBehaviorSettings schema."""
    
    def test_valid_institutional_settings(self):
        """Test creating valid institutional behavior settings."""
        settings = InstitutionalBehaviorSettingsBase(
            detect_accumulation=True,
            detect_liquidity_grabs=True,
            detect_stop_hunts=True,
            wait_for_institutional_footprints=True,
            wait_for_institutional_fight=False,
            institutional_fight_detection_methods=[
                "high_volume_narrow_range",
                "price_rejection",
                "rapid_reversals"
            ]
        )
        
        assert settings.detect_accumulation is True
        assert len(settings.institutional_fight_detection_methods) == 3
        
    def test_default_values(self):
        """Test default values are set correctly."""
        settings = InstitutionalBehaviorSettingsBase()
        
        assert settings.detect_accumulation is True
        assert settings.wait_for_institutional_fight is False
        assert settings.institutional_fight_detection_methods is None
        
    def test_invalid_detection_methods(self):
        """Test validation of fight detection methods."""
        with pytest.raises(ValidationError):
            InstitutionalBehaviorSettingsBase(
                institutional_fight_detection_methods=[
                    "invalid_method",
                    "high_volume_narrow_range"
                ]
            )


class TestEntryExitSettingsSchema:
    """Tests for EntryExitSettings schema."""
    
    def test_valid_entry_exit_settings(self):
        """Test creating valid entry and exit settings."""
        settings = EntryExitSettingsBase(
            direction=DirectionEnum.BOTH,
            primary_entry_technique=EntryTechniqueEnum.GREEN_BAR_AFTER_PULLBACK,
            require_candle_close_confirmation=True,
            trailing_stop_method=TrailingStopMethodEnum.BAR_BY_BAR,
            profit_target_method=ProfitTargetMethodEnum.FIXED_POINTS,
            profit_target_points=25,
            green_bar_sl_placement="below_bar",
            red_bar_sl_placement="above_bar"
        )
        
        assert settings.direction == DirectionEnum.BOTH
        assert settings.primary_entry_technique == EntryTechniqueEnum.GREEN_BAR_AFTER_PULLBACK
        assert settings.profit_target_points == 25
        
    def test_default_values(self):
        """Test default values are set correctly."""
        settings = EntryExitSettingsBase(
            direction=DirectionEnum.LONG,
            primary_entry_technique=EntryTechniqueEnum.NEAR_MA
        )
        
        assert settings.require_candle_close_confirmation is True
        assert settings.profit_target_method == ProfitTargetMethodEnum.FIXED_POINTS
        assert settings.profit_target_points == 25
        assert settings.green_bar_sl_placement == "below_bar"
        
    def test_invalid_profit_target_points(self):
        """Test validation of profit target points."""
        with pytest.raises(ValidationError):
            EntryExitSettingsBase(
                direction=DirectionEnum.LONG,
                primary_entry_technique=EntryTechniqueEnum.NEAR_MA,
                profit_target_points=2  # Too small
            )
            
        with pytest.raises(ValidationError):
            EntryExitSettingsBase(
                direction=DirectionEnum.LONG,
                primary_entry_technique=EntryTechniqueEnum.NEAR_MA,
                profit_target_points=1500  # Too large
            )
            
    def test_invalid_stop_placement(self):
        """Test validation of stop loss placement."""
        with pytest.raises(ValidationError):
            EntryExitSettingsBase(
                direction=DirectionEnum.LONG,
                primary_entry_technique=EntryTechniqueEnum.NEAR_MA,
                green_bar_sl_placement="invalid_placement"
            )
            
        with pytest.raises(ValidationError):
            EntryExitSettingsBase(
                direction=DirectionEnum.LONG,
                primary_entry_technique=EntryTechniqueEnum.NEAR_MA,
                red_bar_sl_placement="invalid_placement"
            )


class TestMarketStateSettingsSchema:
    """Tests for MarketStateSettings schema."""
    
    def test_valid_market_state_settings(self):
        """Test creating valid market state settings."""
        settings = MarketStateSettingsBase(
            required_market_state=MarketStateRequirementEnum.TRENDING_UP,
            avoid_creeper_moves=True,
            prefer_railroad_trends=True,
            wait_for_15min_alignment=True,
            railroad_momentum_threshold=0.8,
            detect_price_ma_struggle=True,
            ma_struggle_threshold=0.2
        )
        
        assert settings.required_market_state == MarketStateRequirementEnum.TRENDING_UP
        assert settings.avoid_creeper_moves is True
        assert settings.railroad_momentum_threshold == 0.8
        
    def test_default_values(self):
        """Test default values are set correctly."""
        settings = MarketStateSettingsBase()
        
        assert settings.required_market_state is None
        assert settings.avoid_creeper_moves is True
        assert settings.prefer_railroad_trends is True
        assert settings.wait_for_15min_alignment is True
        assert settings.railroad_momentum_threshold == 0.8
        
    @pytest.mark.parametrize("threshold", [-0.1, 1.1, 2.0])
    def test_invalid_thresholds(self, threshold):
        """Test validation of threshold values."""
        with pytest.raises(ValidationError):
            MarketStateSettingsBase(
                railroad_momentum_threshold=threshold
            )
            
        with pytest.raises(ValidationError):
            MarketStateSettingsBase(
                ma_struggle_threshold=threshold
            )


class TestRiskManagementSettingsSchema:
    """Tests for RiskManagementSettings schema."""
    
    def test_valid_risk_settings(self):
        """Test creating valid risk management settings."""
        settings = RiskManagementSettingsBase(
            max_risk_per_trade_percent=1.0,
            max_daily_risk_percent=3.0,
            max_weekly_risk_percent=8.0,
            weekly_drawdown_threshold=8.0,
            daily_drawdown_threshold=4.0,
            target_consistent_points=25,
            show_cost_preview=True,
            max_risk_per_trade_inr=10000,
            position_size_scaling=True
        )
        
        assert settings.max_risk_per_trade_percent == 1.0
        assert settings.max_daily_risk_percent == 3.0
        assert settings.max_weekly_risk_percent == 8.0
        assert settings.max_risk_per_trade_inr == 10000
        
    def test_default_values(self):
        """Test default values are set correctly."""
        settings = RiskManagementSettingsBase()
        
        assert settings.max_risk_per_trade_percent == 1.0
        assert settings.max_daily_risk_percent == 3.0
        assert settings.max_weekly_risk_percent == 8.0
        assert settings.target_consistent_points == 25
        assert settings.show_cost_preview is True
        
    @pytest.mark.parametrize("risk_percents", [
        {"max_risk_per_trade_percent": 0.05},  # Too small
        {"max_risk_per_trade_percent": 25.0},  # Too large
        {"max_daily_risk_percent": 0.05},      # Too small
        {"max_daily_risk_percent": 25.0},      # Too large
        {"max_weekly_risk_percent": 0.05},     # Too small
        {"max_weekly_risk_percent": 25.0},     # Too large
    ])
    def test_invalid_risk_percents(self, risk_percents):
        """Test validation of risk percentages."""
        with pytest.raises(ValidationError):
            RiskManagementSettingsBase(**risk_percents)
            
    @pytest.mark.parametrize("thresholds", [
        {"daily_drawdown_threshold": 0.5},  # Too small
        {"daily_drawdown_threshold": 60.0}, # Too large
        {"weekly_drawdown_threshold": 0.5}, # Too small
        {"weekly_drawdown_threshold": 60.0} # Too large
    ])
    def test_invalid_drawdown_thresholds(self, thresholds):
        """Test validation of drawdown thresholds."""
        with pytest.raises(ValidationError):
            RiskManagementSettingsBase(**thresholds)
            
    def test_risk_hierarchy_validation(self):
        """Test risk hierarchy validation."""
        # Per-trade > daily (invalid)
        with pytest.raises(ValidationError):
            RiskManagementSettingsBase(
                max_risk_per_trade_percent=4.0,
                max_daily_risk_percent=3.0,
                max_weekly_risk_percent=8.0
            )
            
        # Daily > weekly (invalid)
        with pytest.raises(ValidationError):
            RiskManagementSettingsBase(
                max_risk_per_trade_percent=1.0,
                max_daily_risk_percent=10.0,
                max_weekly_risk_percent=8.0
            )


class TestSetupQualityCriteriaSchema:
    """Tests for SetupQualityCriteria schema."""
    
    def test_valid_quality_criteria(self):
        """Test creating valid setup quality criteria."""
        criteria = SetupQualityCriteriaBase(
            a_plus_min_score=90.0,
            a_min_score=80.0,
            b_min_score=70.0,
            c_min_score=60.0,
            d_min_score=50.0,
            a_plus_requires_all_timeframes=True,
            a_plus_requires_entry_near_ma=True,
            a_plus_requires_two_day_trend=True,
            timeframe_alignment_weight=0.3,
            trend_strength_weight=0.2,
            entry_technique_weight=0.15,
            proximity_to_key_level_weight=0.2,
            risk_reward_weight=0.15,
            position_sizing_rules={
                "a_plus": {"lots": 2, "risk_percent": 1.0},
                "a": {"lots": 1, "risk_percent": 0.8},
                "b": {"lots": 0.5, "risk_percent": 0.5}
            },
            auto_trade_a_plus=True,
            auto_trade_a=False
        )
        
        assert criteria.a_plus_min_score == 90.0
        assert criteria.timeframe_alignment_weight == 0.3
        assert criteria.position_sizing_rules["a_plus"]["lots"] == 2
        
    def test_default_values(self):
        """Test default values are set correctly."""
        criteria = SetupQualityCriteriaBase()
        
        assert criteria.a_plus_min_score == 90.0
        assert criteria.a_min_score == 80.0
        assert criteria.b_min_score == 70.0
        assert criteria.a_plus_requires_all_timeframes is True
        assert criteria.auto_trade_a_plus is True
        assert criteria.auto_trade_a is False
        
    @pytest.mark.parametrize("score", [-10.0, 110.0])
    def test_invalid_score_thresholds(self, score):
        """Test validation of score thresholds."""
        with pytest.raises(ValidationError):
            SetupQualityCriteriaBase(a_plus_min_score=score)
            
    @pytest.mark.parametrize("weight", [-0.1, 1.1, 2.0])
    def test_invalid_weights(self, weight):
        """Test validation of weights."""
        with pytest.raises(ValidationError):
            SetupQualityCriteriaBase(timeframe_alignment_weight=weight)
            
    def test_weights_sum_validation(self):
        """Test validation that weights sum to 1.0."""
        with pytest.raises(ValidationError):
            SetupQualityCriteriaBase(
                timeframe_alignment_weight=0.4,  # Total > 1.0
                trend_strength_weight=0.2,
                entry_technique_weight=0.15,
                proximity_to_key_level_weight=0.2,
                risk_reward_weight=0.15
            )
            
    def test_score_order_validation(self):
        """Test validation that scores are in descending order."""
        with pytest.raises(ValidationError):
            SetupQualityCriteriaBase(
                a_plus_min_score=90.0,
                a_min_score=85.0,
                b_min_score=80.0,
                c_min_score=75.0,
                d_min_score=80.0  # Out of order (higher than c)
            )


class TestVerticalSpreadSettingsSchema:
    """Tests for VerticalSpreadSettings schema."""
    
    def test_valid_spread_settings(self):
        """Test creating valid vertical spread settings."""
        settings = VerticalSpreadSettingsBase(
            use_vertical_spreads=True,
            preferred_spread_type=SpreadTypeEnum.BULL_CALL_SPREAD,
            otm_strike_distance=1,
            min_capital_required=500000,
            show_cost_before_execution=True,
            timing_pressure_reduction=True,
            trade_nifty=True,
            trade_banknifty=True,
            use_weekly_options=True
        )
        
        assert settings.use_vertical_spreads is True
        assert settings.preferred_spread_type == SpreadTypeEnum.BULL_CALL_SPREAD
        assert settings.otm_strike_distance == 1
        assert settings.min_capital_required == 500000
        
    def test_default_values(self):
        """Test default values are set correctly."""
        settings = VerticalSpreadSettingsBase()
        
        assert settings.use_vertical_spreads is False
        assert settings.preferred_spread_type is None
        assert settings.otm_strike_distance == 1
        assert settings.min_capital_required == 500000
        
    @pytest.mark.parametrize("strike_distance", [0, -1, 15])
    def test_invalid_strike_distance(self, strike_distance):
        """Test validation of strike distance."""
        with pytest.raises(ValidationError):
            VerticalSpreadSettingsBase(
                use_vertical_spreads=True,
                preferred_spread_type=SpreadTypeEnum.BULL_CALL_SPREAD,
                otm_strike_distance=strike_distance
            )
            
    @pytest.mark.parametrize("capital", [50000, 20000000])
    def test_invalid_min_capital(self, capital):
        """Test validation of minimum capital."""
        with pytest.raises(ValidationError):
            VerticalSpreadSettingsBase(
                use_vertical_spreads=True,
                preferred_spread_type=SpreadTypeEnum.BULL_CALL_SPREAD,
                min_capital_required=capital
            )
            
    def test_spread_configuration_validation(self):
        """Test spread configuration validation."""
        with pytest.raises(ValidationError):
            VerticalSpreadSettingsBase(
                use_vertical_spreads=True,
                preferred_spread_type=None  # Missing required field
            )


class TestMultiTimeframeConfirmationSettingsSchema:
    """Tests for MultiTimeframeConfirmationSettings schema."""
    
    def test_valid_mtf_settings(self):
        """Test creating valid multi-timeframe confirmation settings."""
        settings = MultiTimeframeConfirmationSettingsBase(
            require_all_timeframes_aligned=True,
            primary_timeframe=TimeframeValueEnum.ONE_HOUR,
            confirmation_timeframe=TimeframeValueEnum.FIFTEEN_MIN,
            entry_timeframe=TimeframeValueEnum.FIVE_MIN,
            wait_for_15min_alignment=True,
            use_lower_tf_only_for_entry=True,
            min_alignment_score=0.7,
            min_15min_confirmation_bars=2,
            timeframe_weights={
                "1d": 0.35,
                "4h": 0.25,
                "1h": 0.20,
                "15m": 0.15,
                "5m": 0.05
            }
        )
        
        assert settings.require_all_timeframes_aligned is True
        assert settings.primary_timeframe == TimeframeValueEnum.ONE_HOUR
        assert settings.confirmation_timeframe == TimeframeValueEnum.FIFTEEN_MIN
        assert settings.min_alignment_score == 0.7
        assert settings.timeframe_weights["1d"] == 0.35
        
    def test_default_values(self):
        """Test default values are set correctly."""
        settings = MultiTimeframeConfirmationSettingsBase()
        
        assert settings.require_all_timeframes_aligned is True
        assert settings.primary_timeframe == TimeframeValueEnum.ONE_HOUR
        assert settings.confirmation_timeframe == TimeframeValueEnum.FIFTEEN_MIN
        assert settings.entry_timeframe == TimeframeValueEnum.FIVE_MIN
        assert settings.min_alignment_score == 0.7
        
    @pytest.mark.parametrize("score", [-0.1, 1.1, 2.0])
    def test_invalid_alignment_score(self, score):
        """Test validation of alignment score."""
        with pytest.raises(ValidationError):
            MultiTimeframeConfirmationSettingsBase(min_alignment_score=score)
            
    @pytest.mark.parametrize("bars", [0, -1, 15])
    def test_invalid_confirmation_bars(self, bars):
        """Test validation of confirmation bars."""
        with pytest.raises(ValidationError):
            MultiTimeframeConfirmationSettingsBase(min_15min_confirmation_bars=bars)
            
    def test_timeframe_hierarchy_validation(self):
        """Test timeframe hierarchy validation."""
        # Confirmation higher than primary (invalid)
        with pytest.raises(ValidationError):
            MultiTimeframeConfirmationSettingsBase(
                primary_timeframe=TimeframeValueEnum.ONE_HOUR,
                confirmation_timeframe=TimeframeValueEnum.FOUR_HOUR
            )
            
        # Entry higher than confirmation (invalid)
        with pytest.raises(ValidationError):
            MultiTimeframeConfirmationSettingsBase(
                confirmation_timeframe=TimeframeValueEnum.FIFTEEN_MIN,
                entry_timeframe=TimeframeValueEnum.THIRTY_MIN
            )


class TestStrategySchemas:
    """Tests for Strategy schemas."""
    
    def test_valid_strategy_create(self):
        """Test creating a valid strategy creation schema."""
        timeframe = TimeframeSettingsBase(
            name="Hourly",
            value=TimeframeValueEnum.ONE_HOUR,
            importance=TimeframeImportanceEnum.PRIMARY
        )
        
        entry_exit = EntryExitSettingsBase(
            direction=DirectionEnum.BOTH,
            primary_entry_technique=EntryTechniqueEnum.GREEN_BAR_AFTER_PULLBACK
        )
        
        strategy = StrategyCreate(
            name="Test Strategy",
            description="A test strategy",
            type=StrategyTypeEnum.TREND_FOLLOWING,
            configuration={"indicators": ["ma", "rsi"]},
            parameters={"ma_period": 21},
            timeframes=[timeframe],
            entry_exit_settings=entry_exit
        )
        
        assert strategy.name == "Test Strategy"
        assert strategy.type == StrategyTypeEnum.TREND_FOLLOWING
        assert len(strategy.timeframes) == 1
        assert strategy.timeframes[0].value == TimeframeValueEnum.ONE_HOUR
        assert strategy.entry_exit_settings.direction == DirectionEnum.BOTH
        
    def test_strategy_name_validation(self):
        """Test validation of strategy name."""
        # Too short
        with pytest.raises(ValidationError):
            StrategyCreate(
                name="TS",
                type=StrategyTypeEnum.TREND_FOLLOWING
            )
            
        # Too long
        with pytest.raises(ValidationError):
            StrategyCreate(
                name="T" * 101,
                type=StrategyTypeEnum.TREND_FOLLOWING
            )
    
    def test_vertical_spread_strategy_validation(self):
        """Test validation of vertical spread strategy."""
        with pytest.raises(ValidationError):
            StrategyCreate(
                name="Invalid Spread Strategy",
                type=StrategyTypeEnum.VERTICAL_SPREAD,
                # Missing spread_settings
            )
            
        with pytest.raises(ValidationError):
            StrategyCreate(
                name="Invalid Spread Strategy",
                type=StrategyTypeEnum.VERTICAL_SPREAD,
                spread_settings=VerticalSpreadSettingsBase(
                    use_vertical_spreads=False  # Should be True
                )
            )
            
    def test_strategy_update(self):
        """Test strategy update schema."""
        update = StrategyUpdate(
            name="Updated Strategy",
            description="Updated description",
            parameters={"ma_period": 34}
        )
        
        assert update.name == "Updated Strategy"
        assert update.description == "Updated description"
        assert update.parameters == {"ma_period": 34}
        
        # Check that partial updates are possible
        partial_update = StrategyUpdate(
            parameters={"ma_period": 34}
        )
        
        assert partial_update.name is None
        assert partial_update.description is None
        assert partial_update.parameters == {"ma_period": 34}


class TestAnalysisResultSchemas:
    """Tests for analysis result schemas."""
    
    def test_timeframe_analysis_result(self):
        """Test TimeframeAnalysisResult schema."""
        result = TimeframeAnalysisResult(
            aligned=True,
            alignment_score=0.85,
            timeframe_results={
                "1h": {
                    "analyzed": True,
                    "direction": "up",
                    "ma_trending": True,
                    "price_struggling": False,
                    "price_above_ma": True,
                    "aligned": True
                }
            },
            primary_direction="up",
            require_all_aligned=True,
            min_alignment_score=0.7,
            sufficient_alignment=True,
            fifteen_min_aligned=True
        )
        
        assert result.aligned is True
        assert result.alignment_score == 0.85
        assert result.primary_direction == "up"
        assert result.sufficient_alignment is True
        assert result.fifteen_min_aligned is True
        
    def test_market_state_analysis(self):
        """Test MarketStateAnalysis schema."""
        analysis = MarketStateAnalysis(
            market_state=MarketStateRequirementEnum.TRENDING_UP,
            trend_phase=TrendPhaseEnum.MIDDLE,
            is_railroad_trend=True,
            is_creeper_move=False,
            has_two_day_trend=True,
            trend_direction="up",
            price_indicator_divergence=False,
            price_struggling_near_ma=False,
            institutional_fight_in_progress=False,
            accumulation_detected=True,
            bos_detected=True
        )
        
        assert analysis.market_state == MarketStateRequirementEnum.TRENDING_UP
        assert analysis.trend_phase == TrendPhaseEnum.MIDDLE
        assert analysis.is_railroad_trend is True
        assert analysis.trend_direction == "up"
        
    def test_setup_quality_result(self):
        """Test SetupQualityResult schema."""
        result = SetupQualityResult(
            strategy_id=1,
            grade=SetupQualityGradeEnum.A_PLUS,
            score=92.5,
            factor_scores={
                "timeframe_alignment": 95.0,
                "trend_strength": 90.0,
                "entry_quality": 85.0,
                "key_level_proximity": 100.0,
                "risk_reward": 90.0
            },
            position_size=2,
            risk_percent=1.0,
            can_auto_trade=True,
            analysis_comments=[
                "Excellent timeframe alignment",
                "Strong trend",
                "Clean entry near key level"
            ]
        )
        
        assert result.grade == SetupQualityGradeEnum.A_PLUS
        assert result.score == 92.5
        assert result.position_size == 2
        assert result.can_auto_trade is True
        assert len(result.analysis_comments) == 3


class TestSignalAndTradeSchemas:
    """Tests for Signal and Trade schemas."""
    
    def test_signal_base(self):
        """Test SignalBase schema."""
        signal = SignalBase(
            instrument="NIFTY",
            direction=DirectionEnum.LONG
        )
        
        assert signal.instrument == "NIFTY"
        assert signal.direction == DirectionEnum.LONG
        
    def test_signal_response(self):
        """Test SignalResponse schema."""
        response = SignalResponse(
            id=1,
            strategy_id=1,
            instrument="NIFTY",
            direction=DirectionEnum.LONG,
            signal_type="breakout",
            entry_price=18500.0,
            entry_time=datetime(2023, 1, 1, 10, 30, 0),
            entry_timeframe=TimeframeValueEnum.FIVE_MIN,
            entry_technique=EntryTechniqueEnum.GREEN_BAR_AFTER_PULLBACK,
            take_profit_price=18700.0,
            stop_loss_price=18400.0,
            trailing_stop=True,
            position_size=2,
            risk_reward_ratio=2.0,
            risk_amount=10000.0,
            setup_quality=SetupQualityGradeEnum.A_PLUS,
            setup_score=95.0,
            confidence=0.95,
            market_state=MarketStateRequirementEnum.TRENDING_UP,
            trend_phase=TrendPhaseEnum.MIDDLE,
            is_active=True,
            is_executed=False
        )
        
        assert response.id == 1
        assert response.entry_price == 18500.0
        assert response.take_profit_price == 18700.0
        assert response.stop_loss_price == 18400.0
        assert response.trailing_stop is True
        
    def test_trade_create(self):
        """Test TradeCreate schema."""
        trade = TradeCreate(
            signal_id=1,
            execution_price=18510.0,
            execution_time=datetime(2023, 1, 1, 10, 31, 0),
            exit_price=18650.0,
            exit_reason="target_hit"
        )
        
        assert trade.signal_id == 1
        assert trade.execution_price == 18510.0
        assert trade.exit_price == 18650.0
        assert trade.exit_reason == "target_hit"
        
    def test_invalid_execution_price(self):
        """Test validation of execution price."""
        with pytest.raises(ValidationError):
            TradeCreate(
                signal_id=1,
                execution_price=0.0,  # Invalid (must be positive)
                exit_price=18650.0,
                exit_reason="target_hit"
            )
            
        with pytest.raises(ValidationError):
            TradeCreate(
                signal_id=1,
                execution_price=-100.0,  # Invalid (must be positive)
                exit_price=18650.0,
                exit_reason="target_hit"
            )


class TestFeedbackSchemas:
    """Tests for Feedback schemas."""
    
    def test_feedback_base(self):
        """Test FeedbackBase schema."""
        feedback = FeedbackBase(
            feedback_type=FeedbackTypeEnum.TEXT_NOTE,
            title="Entry Timing Improvement",
            description="Need to wait for better confirmation before entry",
            tags=["entry", "timing", "improvement"],
            improvement_category="entry_timing",
            applies_to_entry=True,
            pre_trade_conviction_level=7.5,
            emotional_state_rating=3,
            lessons_learned="Wait for candle close confirmation",
            action_items="Update entry rules to require candle close"
        )
        
        assert feedback.feedback_type == FeedbackTypeEnum.TEXT_NOTE
        assert feedback.title == "Entry Timing Improvement"
        assert feedback.pre_trade_conviction_level == 7.5
        assert feedback.emotional_state_rating == 3
        
    def test_conviction_level_validation(self):
        """Test validation of conviction level."""
        with pytest.raises(ValidationError):
            FeedbackBase(
                feedback_type=FeedbackTypeEnum.TEXT_NOTE,
                title="Test Feedback",
                applies_to_setup=True,
                pre_trade_conviction_level=12.0  # Invalid (must be between 0-10)
            )
            
    def test_emotional_state_validation(self):
        """Test validation of emotional state rating."""
        with pytest.raises(ValidationError):
            FeedbackBase(
                feedback_type=FeedbackTypeEnum.TEXT_NOTE,
                title="Test Feedback",
                applies_to_setup=True,
                emotional_state_rating=6  # Invalid (must be between 1-5)
            )
            
    def test_application_fields_validation(self):
        """Test validation of application fields."""
        with pytest.raises(ValidationError):
            FeedbackBase(
                feedback_type=FeedbackTypeEnum.TEXT_NOTE,
                title="Test Feedback",
                # All application fields are False (invalid)
                applies_to_setup=False,
                applies_to_entry=False,
                applies_to_exit=False,
                applies_to_risk=False
            )


class TestPerformanceAnalysis:
    """Tests for PerformanceAnalysis schema."""
    
    def test_performance_analysis(self):
        """Test PerformanceAnalysis schema."""
        analysis = PerformanceAnalysis(
            strategy_id=1,
            total_trades=50,
            win_count=35,
            loss_count=15,
            win_rate=0.7,
            total_profit_inr=150000.0,
            avg_win_inr=5000.0,
            avg_loss_inr=-2000.0,
            profit_factor=3.5,
            trades_by_grade={
                "a_plus": {
                    "count": 20,
                    "profit": 100000.0,
                    "win_rate": 0.9
                },
                "a": {
                    "count": 15,
                    "profit": 50000.0,
                    "win_rate": 0.8
                },
                "b": {
                    "count": 10,
                    "profit": 10000.0,
                    "win_rate": 0.6
                },
                "c": {
                    "count": 5,
                    "profit": -10000.0,
                    "win_rate": 0.2
                }
            },
            analysis_period={
                "start": datetime(2023, 1, 1),
                "end": datetime(2023, 3, 31)
            },
            largest_win_inr=15000.0,
            largest_loss_inr=-5000.0,
            consecutive_wins=6,
            consecutive_losses=2,
            avg_holding_period_minutes=240.0
        )
        
        assert analysis.strategy_id == 1
        assert analysis.total_trades == 50
        assert analysis.win_count == 35
        assert analysis.win_rate == 0.7
        assert analysis.total_profit_inr == 150000.0
        assert analysis.profit_factor == 3.5
        assert len(analysis.trades_by_grade) == 4
        
    def test_win_rate_validation(self):
        """Test validation of win rate."""
        with pytest.raises(ValidationError):
            PerformanceAnalysis(
                strategy_id=1,
                total_trades=50,
                win_count=35,
                loss_count=15,
                win_rate=1.2,  # Invalid (must be between 0 and 1)
                total_profit_inr=150000.0,
                avg_win_inr=5000.0,
                avg_loss_inr=-2000.0,
                profit_factor=3.5,
                trades_by_grade={},
                analysis_period={
                    "start": datetime(2023, 1, 1),
                    "end": datetime(2023, 3, 31)
                }
            )
            
    def test_profit_factor_validation(self):
        """Test validation of profit factor."""
        with pytest.raises(ValidationError):
            PerformanceAnalysis(
                strategy_id=1,
                total_trades=50,
                win_count=35,
                loss_count=15,
                win_rate=0.7,
                total_profit_inr=150000.0,
                avg_win_inr=5000.0,
                avg_loss_inr=-2000.0,
                profit_factor=-2.0,  # Invalid (must be positive)
                trades_by_grade={},
                analysis_period={
                    "start": datetime(2023, 1, 1),
                    "end": datetime(2023, 3, 31)
                }
            )