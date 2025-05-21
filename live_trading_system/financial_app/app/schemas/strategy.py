"""
Pydantic schemas for trading strategy models.

These schemas provide validation, serialization, and documentation for the API layer,
with extensive validation rules matching the sophisticated financial trading models.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Set
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

# Enum schemas (matching your model enums)
class TimeframeValueEnum(str, Enum):
    """Trading timeframe options matching the model TimeframeValue enum."""
    DAILY = "1d"
    FOUR_HOUR = "4h"
    ONE_HOUR = "1h"
    THIRTY_MIN = "30m"
    FIFTEEN_MIN = "15m"
    FIVE_MIN = "5m"
    THREE_MIN = "3m"

class DirectionEnum(str, Enum):
    """Trade direction options matching the model Direction enum."""
    LONG = "long"
    SHORT = "short"
    BOTH = "both"  # Strategy works in both directions

class TimeframeImportanceEnum(str, Enum):
    """Timeframe importance levels matching the model TimeframeImportance enum."""
    PRIMARY = "primary"           # Direction-determining timeframe
    CONFIRMATION = "confirmation" # Confirms signals from primary
    ENTRY = "entry"               # Used only for entry timing
    FILTER = "filter"             # Used to filter signals
    CONTEXT = "context"           # Provides broader market context

class EntryTechniqueEnum(str, Enum):
    """Entry techniques matching the model EntryTechnique enum."""
    NEAR_MA = "near_ma"                        # Enter near moving average
    GREEN_BAR_AFTER_PULLBACK = "green_bar_pullback"  # Green bar after pullback (long)
    RED_BAR_AFTER_RALLY = "red_bar_rally"      # Red bar after rally (short)
    BREAKOUT_PULLBACK_LONG = "breakout_pullback_long"  # SPB after upside breakout
    BREAKOUT_PULLBACK_SHORT = "breakout_pullback_short"  # SPB after downside breakout
    DISCOUNT_ZONE_LONG = "discount_zone_long"  # Buy in discount zone
    PREMIUM_ZONE_SHORT = "premium_zone_short"  # Sell in premium zone
    LIQUIDITY_GRAB_REVERSAL_LONG = "liquidity_grab_long"  # Enter after failed breakdown
    LIQUIDITY_GRAB_REVERSAL_SHORT = "liquidity_grab_short"  # Enter after failed breakout
    PDH_BREAK_LONG = "pdh_break_long"          # Previous day high breakout
    PDL_BREAK_SHORT = "pdl_break_short"        # Previous day low breakout
    BOS_ENTRY_LONG = "bos_entry_long"          # Break of structure long
    BOS_ENTRY_SHORT = "bos_entry_short"        # Break of structure short
    MA_BOUNCE_LONG = "ma_bounce_long"          # Bounce off MA in uptrend
    MA_BOUNCE_SHORT = "ma_bounce_short"        # Bounce off MA in downtrend
    TWO_GREEN_DAILY_LONG = "two_green_daily_long"  # Two consecutive green daily bars
    TWO_RED_DAILY_SHORT = "two_red_daily_short"    # Two consecutive red daily bars

class MarketStateRequirementEnum(str, Enum):
    """Market state requirements matching the model MarketStateRequirement enum."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGE_BOUND = "range_bound"
    NARROW_HIGH_VOLUME = "narrow_high_volume"  # Narrow with high volume (breakout potential)
    NARROW_LOW_VOLUME = "narrow_low_volume"    # Narrow with low volume (waiting mode)
    MOMENTUM_MOVE = "momentum_move"            # Strong directional move
    CREEPER_MOVE = "creeper_move"              # Slow grinding price action
    BREAKOUT = "breakout"                      # Fresh breakout from range
    REVERSAL = "reversal"                      # Reversal from trend
    ABSORPTION = "absorption"                  # High volume with little price movement
    PREMIUM_ZONE = "premium_zone"              # Price in premium zone
    DISCOUNT_ZONE = "discount_zone"            # Price in discount zone
    EQUILIBRIUM = "equilibrium"                # Price in equilibrium zone
    ANY = "any"                                # No specific requirement

class TrendPhaseEnum(str, Enum):
    """Trend phase options matching the model TrendPhase enum."""
    EARLY = "early"          # Beginning of trend (avoid)
    MIDDLE = "middle"        # Middle of trend (target)
    LATE = "late"            # End of trend (avoid)
    UNDETERMINED = "undetermined"  # Cannot determine phase

class ProfitTargetMethodEnum(str, Enum):
    """Profit target methods matching the model ProfitTargetMethod enum."""
    FIXED_POINTS = "fixed_points"              # Fixed point target
    ATR_MULTIPLE = "atr_multiple"              # Multiple of ATR
    PREVIOUS_SWING = "previous_swing"          # Previous swing high/low
    FIBONACCI_EXTENSION = "fibonacci_extension"  # Fibonacci extension
    PREMIUM_DISCOUNT_ZONE = "premium_discount_zone"  # Premium/discount zone

class SetupQualityGradeEnum(str, Enum):
    """Setup quality grades matching the model SetupQualityGrade enum."""
    A_PLUS = "a_plus"    # Highest quality setup
    A = "a"              # Excellent setup
    B = "b"              # Good setup
    C = "c"              # Acceptable setup
    D = "d"              # Poor setup
    F = "f"              # Fail/Do not trade

class SpreadTypeEnum(str, Enum):
    """Vertical spread types matching the model SpreadType enum."""
    BULL_CALL_SPREAD = "bull_call_spread"  # Buy ATM call, sell OTM call
    BEAR_PUT_SPREAD = "bear_put_spread"    # Buy ATM put, sell OTM put
    BEAR_CALL_SPREAD = "bear_call_spread"  # Sell ATM call, buy OTM call
    BULL_PUT_SPREAD = "bull_put_spread"    # Sell ATM put, buy OTM put

class FeedbackTypeEnum(str, Enum):
    """Feedback types matching the model FeedbackType enum."""
    TEXT_NOTE = "text_note"          # Text notes on trade
    SCREENSHOT = "screenshot"        # Screenshot of chart
    CHART_ANNOTATION = "chart_annotation"  # Annotated chart
    VIDEO_RECORDING = "video_recording"    # Video recording
    TRADE_REVIEW = "trade_review"    # Full trade review

class MovingAverageTypeEnum(str, Enum):
    """Moving average types matching the model MovingAverageType enum."""
    SIMPLE = "simple"             # Simple Moving Average (SMA)
    EXPONENTIAL = "exponential"   # Exponential Moving Average (EMA)
    WEIGHTED = "weighted"         # Weighted Moving Average
    HULL = "hull"                 # Hull Moving Average

class TrailingStopMethodEnum(str, Enum):
    """Trailing stop methods matching the model TrailingStopMethod enum."""
    BAR_BY_BAR = "bar_by_bar"                         # Trail bar by bar
    PREVIOUS_SWING = "previous_swing"                 # Trail to previous swing
    MOVING_AVERAGE = "moving_average"                 # Trail based on MA
    ATR_MULTIPLE = "atr_multiple"                     # Trail based on ATR
    FIXED_POINTS = "fixed_points"                     # Trail by fixed points
    CLOSE_BELOW_MA = "close_below_ma"                 # Trail to below MA (for longs)
    CLOSE_ABOVE_MA = "close_above_ma"                 # Trail to above MA (for shorts)

class StrategyTypeEnum(str, Enum):
    """Strategy types matching the model StrategyType enum."""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    MOMENTUM = "momentum"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    PATTERN_RECOGNITION = "pattern_recognition"
    MULTI_TIMEFRAME = "multi_timeframe"
    INSTITUTIONAL_FLOW = "institutional_flow"
    VERTICAL_SPREAD = "vertical_spread"
    CUSTOM = "custom"

# Base models
class BaseSchema(BaseModel):
    """Base schema with common configuration."""
    
    model_config = ConfigDict(
        from_attributes=True,  # Replaces orm_mode
        arbitrary_types_allowed=True,
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )

# Strategy Timeframe schemas
class TimeframeSettingsBase(BaseSchema):
    """Timeframe settings for strategy configuration."""
    name: str = Field(..., description="Display name for this timeframe")
    value: TimeframeValueEnum = Field(..., description="Timeframe value")
    importance: TimeframeImportanceEnum = Field(..., description="Importance of this timeframe in the strategy")
    order: Optional[int] = Field(None, description="Order of importance (0 is highest)")
    ma_type: MovingAverageTypeEnum = Field("simple", description="Type of moving average to use")
    ma_period_primary: int = Field(21, description="Primary moving average period (21 MA)")
    ma_period_secondary: int = Field(200, description="Secondary moving average period (200 MA)")
    require_alignment: bool = Field(True, description="Whether this timeframe must align with higher timeframes")
    
    @field_validator('ma_period_primary')
    @classmethod
    def validate_ma_period_primary(cls, v):
        """Ensure primary MA period is reasonable."""
        if v < 5 or v > 500:
            raise ValueError("Primary MA period must be between 5 and 500")
        return v
    
    @field_validator('ma_period_secondary')
    @classmethod
    def validate_ma_period_secondary(cls, v):
        """Ensure secondary MA period is reasonable."""
        if v < 10 or v > 1000:
            raise ValueError("Secondary MA period must be between 10 and 1000")
        return v

# Institutional Behavior settings schemas
class InstitutionalBehaviorSettingsBase(BaseSchema):
    """Settings for institutional behavior detection."""
    detect_accumulation: bool = Field(True, description="Detect accumulation patterns (high volume with little price movement)")
    detect_liquidity_grabs: bool = Field(True, description="Detect liquidity grab patterns (false breakouts)")
    detect_stop_hunts: bool = Field(True, description="Detect stop hunting patterns")
    wait_for_institutional_footprints: bool = Field(True, description="Wait for institutional footprints before entering")
    wait_for_institutional_fight: bool = Field(False, description="Wait for institutional fight to end before entering")
    institutional_fight_detection_methods: Optional[List[str]] = Field(None, description="Methods to detect institutional fights")
    
    @field_validator('institutional_fight_detection_methods')
    @classmethod
    def validate_detection_methods(cls, v):
        """Ensure detection methods are valid."""
        valid_methods = [
            "high_volume_narrow_range", 
            "price_rejection", 
            "rapid_reversals", 
            "failed_breakouts"
        ]
        if v:
            for method in v:
                if method not in valid_methods:
                    raise ValueError(f"Invalid detection method: {method}")
        return v

# Entry/Exit settings schemas
class EntryExitSettingsBase(BaseSchema):
    """Settings for trade entry and exit."""
    direction: DirectionEnum = Field(..., description="Trading direction (long, short, or both)")
    primary_entry_technique: EntryTechniqueEnum = Field(..., description="Primary entry technique")
    require_candle_close_confirmation: bool = Field(True, description="Wait for candle close before entry")
    trailing_stop_method: Optional[TrailingStopMethodEnum] = Field(None, description="Method for trailing stop loss")
    profit_target_method: ProfitTargetMethodEnum = Field(ProfitTargetMethodEnum.FIXED_POINTS, description="Method for setting profit targets")
    profit_target_points: int = Field(25, description="Profit target in points (for fixed point method)")
    green_bar_sl_placement: str = Field("below_bar", description="Stop loss placement for long entries")
    red_bar_sl_placement: str = Field("above_bar", description="Stop loss placement for short entries")
    
    @field_validator('profit_target_points')
    @classmethod
    def validate_profit_target_points(cls, v):
        """Ensure profit target is reasonable."""
        if v < 5 or v > 1000:
            raise ValueError("Profit target points must be between 5 and 1000")
        return v
    
    @model_validator(mode='after')
    def validate_stop_placement(self) -> 'EntryExitSettingsBase':
        """Ensure stop loss placement is valid."""
        green_bar_sl = self.green_bar_sl_placement
        red_bar_sl = self.red_bar_sl_placement
        
        valid_green_placements = ['below_bar', 'below_previous_bar', 'below_low']
        valid_red_placements = ['above_bar', 'above_previous_bar', 'above_high']
        
        if green_bar_sl and green_bar_sl not in valid_green_placements:
            raise ValueError(f"Invalid green bar stop loss placement: {green_bar_sl}")
        
        if red_bar_sl and red_bar_sl not in valid_red_placements:
            raise ValueError(f"Invalid red bar stop loss placement: {red_bar_sl}")
        
        return self

# Market State settings schemas
class MarketStateSettingsBase(BaseSchema):
    """Settings for market state requirements."""
    required_market_state: Optional[MarketStateRequirementEnum] = Field(None, description="Required market state for trading")
    avoid_creeper_moves: bool = Field(True, description="Avoid trading during creeper moves")
    prefer_railroad_trends: bool = Field(True, description="Prefer trading during railroad trends")
    wait_for_15min_alignment: bool = Field(True, description="Wait for 15-minute timeframe alignment")
    railroad_momentum_threshold: float = Field(0.8, description="Threshold for detecting railroad trends")
    detect_price_ma_struggle: bool = Field(True, description="Detect when price is struggling near MA")
    ma_struggle_threshold: float = Field(0.2, description="Threshold for MA struggle detection (as % of price)")
    detect_price_indicator_divergence: bool = Field(True, description="Detect price vs indicator divergence")
    price_action_overrides_indicators: bool = Field(True, description="Allow price action to override indicator signals")
    
    @field_validator('railroad_momentum_threshold')
    @classmethod
    def validate_railroad_threshold(cls, v):
        """Ensure railroad threshold is between 0 and 1."""
        if v < 0 or v > 1:
            raise ValueError("Railroad momentum threshold must be between 0 and 1")
        return v
    
    @field_validator('ma_struggle_threshold')
    @classmethod
    def validate_struggle_threshold(cls, v):
        """Ensure MA struggle threshold is reasonable."""
        if v < 0 or v > 1:
            raise ValueError("MA struggle threshold must be between 0 and 1")
        return v

# Risk Management settings schemas
class RiskManagementSettingsBase(BaseSchema):
    """Settings for risk management."""
    max_risk_per_trade_percent: float = Field(1.0, description="Maximum risk per trade as percentage of account")
    max_daily_risk_percent: float = Field(3.0, description="Maximum daily risk as percentage of account")
    max_weekly_risk_percent: float = Field(8.0, description="Maximum weekly risk as percentage of account")
    weekly_drawdown_threshold: float = Field(8.0, description="Weekly drawdown threshold for risk reduction")
    daily_drawdown_threshold: float = Field(4.0, description="Daily drawdown threshold for risk reduction")
    target_consistent_points: int = Field(25, description="Target consistent points per trade")
    show_cost_preview: bool = Field(True, description="Show cost preview in INR before execution")
    max_risk_per_trade_inr: Optional[int] = Field(None, description="Maximum risk per trade in INR")
    position_size_scaling: bool = Field(True, description="Scale position size based on setup quality")
    
    @field_validator('max_risk_per_trade_percent', 'max_daily_risk_percent', 'max_weekly_risk_percent')
    @classmethod
    def validate_risk_percent(cls, v):
        """Ensure risk percentages are reasonable."""
        if v < 0.1 or v > 20:
            raise ValueError("Risk percentage must be between 0.1% and 20%")
        return v
    
    @field_validator('daily_drawdown_threshold', 'weekly_drawdown_threshold')
    @classmethod
    def validate_drawdown_threshold(cls, v):
        """Ensure drawdown thresholds are reasonable."""
        if v < 1 or v > 50:
            raise ValueError("Drawdown threshold must be between 1% and 50%")
        return v
    
    @model_validator(mode='after')
    def validate_risk_hierarchy(self) -> 'RiskManagementSettingsBase':
        """Ensure risk limits are properly ordered."""
        per_trade = self.max_risk_per_trade_percent
        daily = self.max_daily_risk_percent
        weekly = self.max_weekly_risk_percent
        
        if per_trade > daily:
            raise ValueError("Per-trade risk cannot exceed daily risk")
        if daily > weekly:
            raise ValueError("Daily risk cannot exceed weekly risk")
        
        return self

# Setup Quality Criteria schemas
class SetupQualityCriteriaBase(BaseSchema):
    """Quality criteria for trade setup evaluation."""
    a_plus_min_score: float = Field(90.0, description="Minimum score for A+ grade")
    a_min_score: float = Field(80.0, description="Minimum score for A grade")
    b_min_score: float = Field(70.0, description="Minimum score for B grade")
    c_min_score: float = Field(60.0, description="Minimum score for C grade")
    d_min_score: float = Field(50.0, description="Minimum score for D grade")
    a_plus_requires_all_timeframes: bool = Field(True, description="Require all timeframes to be aligned for A+ grade")
    a_plus_requires_entry_near_ma: bool = Field(True, description="Require entry near MA for A+ grade")
    a_plus_requires_two_day_trend: bool = Field(True, description="Require two-day trend for A+ grade")
    timeframe_alignment_weight: float = Field(0.3, description="Weight for timeframe alignment in quality score")
    trend_strength_weight: float = Field(0.2, description="Weight for trend strength in quality score")
    entry_technique_weight: float = Field(0.15, description="Weight for entry technique in quality score")
    proximity_to_key_level_weight: float = Field(0.2, description="Weight for proximity to key level in quality score")
    risk_reward_weight: float = Field(0.15, description="Weight for risk/reward ratio in quality score")
    position_sizing_rules: Optional[Dict[str, Dict[str, Any]]] = Field(None, description="Position sizing rules by grade")
    auto_trade_a_plus: bool = Field(True, description="Automatically trade A+ setups")
    auto_trade_a: bool = Field(False, description="Automatically trade A setups")
    
    @field_validator('a_plus_min_score', 'a_min_score', 'b_min_score', 'c_min_score', 'd_min_score')
    @classmethod
    def validate_score_thresholds(cls, v):
        """Ensure score thresholds are between 0 and 100."""
        if v < 0 or v > 100:
            raise ValueError("Score threshold must be between 0 and 100")
        return v
    
    @field_validator('timeframe_alignment_weight', 'trend_strength_weight', 'entry_technique_weight', 
              'proximity_to_key_level_weight', 'risk_reward_weight')
    @classmethod
    def validate_weights(cls, v):
        """Ensure weights are between 0 and 1."""
        if v < 0 or v > 1:
            raise ValueError("Weight must be between 0 and 1")
        return v
    
    @model_validator(mode='after')
    def validate_weights_sum(self) -> 'SetupQualityCriteriaBase':
        """Ensure weights sum to 1."""
        weights = [
            self.timeframe_alignment_weight,
            self.trend_strength_weight,
            self.entry_technique_weight,
            self.proximity_to_key_level_weight,
            self.risk_reward_weight
        ]
        
        if abs(sum(weights) - 1.0) > 0.001:
            raise ValueError("Weights must sum to 1.0")
        
        return self
    
    @model_validator(mode='after')
    def validate_score_order(self) -> 'SetupQualityCriteriaBase':
        """Ensure score thresholds are in descending order."""
        if not (self.a_plus_min_score > self.a_min_score > self.b_min_score > 
                self.c_min_score > self.d_min_score):
            raise ValueError("Score thresholds must be in descending order")
        
        return self

# Vertical Spread settings schemas
class VerticalSpreadSettingsBase(BaseSchema):
    """Settings for vertical spread trading."""
    use_vertical_spreads: bool = Field(False, description="Enable vertical spread trading")
    preferred_spread_type: Optional[SpreadTypeEnum] = Field(None, description="Preferred vertical spread type")
    otm_strike_distance: int = Field(1, description="Out-of-the-money strike distance in strikes")
    min_capital_required: int = Field(500000, description="Minimum capital required for spread trading (in INR)")
    show_cost_before_execution: bool = Field(True, description="Show cost preview before execution")
    timing_pressure_reduction: bool = Field(True, description="Benefit from reduced timing pressure")
    trade_nifty: bool = Field(True, description="Trade NIFTY options")
    trade_banknifty: bool = Field(True, description="Trade BANKNIFTY options")
    use_weekly_options: bool = Field(True, description="Use weekly options")
    
    @field_validator('otm_strike_distance')
    @classmethod
    def validate_strike_distance(cls, v):
        """Ensure strike distance is reasonable."""
        if v < 1 or v > 10:
            raise ValueError("OTM strike distance must be between 1 and 10")
        return v
    
    @field_validator('min_capital_required')
    @classmethod
    def validate_min_capital(cls, v):
        """Ensure minimum capital is reasonable."""
        if v < 100000 or v > 10000000:
            raise ValueError("Minimum capital must be between 1 lakh and 1 crore")
        return v
    
    @model_validator(mode='after')
    def validate_spread_configuration(self) -> 'VerticalSpreadSettingsBase':
        """Ensure spread configuration is valid."""
        if self.use_vertical_spreads and not self.preferred_spread_type:
            raise ValueError("Must specify preferred spread type when using vertical spreads")
        
        return self

# Meta Learning settings schemas
class MetaLearningSettingsBase(BaseSchema):
    """Settings for meta-learning and continuous improvement."""
    record_trading_sessions: bool = Field(True, description="Record trading sessions for review")
    record_decision_points: bool = Field(True, description="Record decision points for analysis")
    perform_post_market_analysis: bool = Field(True, description="Perform post-market analysis")
    store_screenshots: bool = Field(True, description="Store screenshots of trades")
    store_trading_notes: bool = Field(True, description="Store trading notes")
    review_frequency: str = Field("daily", description="Frequency of trading review")
    track_market_relationships: bool = Field(True, description="Track relationships between markets")
    detect_regime_changes: bool = Field(True, description="Detect market regime changes")
    
    @field_validator('review_frequency')
    @classmethod
    def validate_review_frequency(cls, v):
        """Ensure review frequency is valid."""
        valid_frequencies = ["daily", "weekly", "monthly"]
        if v not in valid_frequencies:
            raise ValueError(f"Review frequency must be one of: {', '.join(valid_frequencies)}")
        return v

# Multi-Timeframe Confirmation settings schemas
class MultiTimeframeConfirmationSettingsBase(BaseSchema):
    """Settings for multi-timeframe confirmation."""
    require_all_timeframes_aligned: bool = Field(True, description="Require all timeframes to be aligned")
    primary_timeframe: TimeframeValueEnum = Field(TimeframeValueEnum.ONE_HOUR, description="Primary timeframe for trend direction")
    confirmation_timeframe: TimeframeValueEnum = Field(TimeframeValueEnum.FIFTEEN_MIN, description="Confirmation timeframe")
    entry_timeframe: TimeframeValueEnum = Field(TimeframeValueEnum.FIVE_MIN, description="Entry timeframe")
    wait_for_15min_alignment: bool = Field(True, description="Wait for 15-min timeframe to confirm")
    use_lower_tf_only_for_entry: bool = Field(True, description="Use lower timeframes only for entry timing")
    min_alignment_score: float = Field(0.7, description="Minimum alignment score (0-1)")
    min_15min_confirmation_bars: int = Field(2, description="Minimum 15-min bars confirming trend")
    timeframe_weights: Optional[Dict[str, float]] = Field(None, description="Weights for different timeframes")
    
    @field_validator('min_alignment_score')
    @classmethod
    def validate_alignment_score(cls, v):
        """Ensure alignment score is between 0 and 1."""
        if v < 0 or v > 1:
            raise ValueError("Alignment score must be between 0 and 1")
        return v
    
    @field_validator('min_15min_confirmation_bars')
    @classmethod
    def validate_confirmation_bars(cls, v):
        """Ensure confirmation bars is reasonable."""
        if v < 1 or v > 10:
            raise ValueError("Confirmation bars must be between 1 and 10")
        return v
    
    @model_validator(mode='after')
    def validate_timeframe_hierarchy(self) -> 'MultiTimeframeConfirmationSettingsBase':
        """Ensure timeframe hierarchy is valid."""
        # Define timeframe hierarchy
        hierarchy = {
            TimeframeValueEnum.DAILY: 0,
            TimeframeValueEnum.FOUR_HOUR: 1,
            TimeframeValueEnum.ONE_HOUR: 2,
            TimeframeValueEnum.THIRTY_MIN: 3,
            TimeframeValueEnum.FIFTEEN_MIN: 4,
            TimeframeValueEnum.FIVE_MIN: 5,
            TimeframeValueEnum.THREE_MIN: 6
        }
        
        if self.primary_timeframe and self.confirmation_timeframe and hierarchy[self.primary_timeframe] > hierarchy[self.confirmation_timeframe]:
            raise ValueError("Primary timeframe must be higher than confirmation timeframe")
        
        if self.confirmation_timeframe and self.entry_timeframe and hierarchy[self.confirmation_timeframe] > hierarchy[self.entry_timeframe]:
            raise ValueError("Confirmation timeframe must be higher than entry timeframe")
        
        return self

# Strategy schemas
class StrategyBase(BaseSchema):
    """Base schema for trading strategies."""
    name: str = Field(..., min_length=3, max_length=100, description="Strategy name")
    description: Optional[str] = Field(None, description="Strategy description")
    type: StrategyTypeEnum = Field(..., description="Strategy type")

class StrategyCreate(StrategyBase):
    """Schema for creating a new strategy."""
    configuration: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Strategy configuration")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Strategy parameters")
    validation_rules: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Parameter validation rules")
    timeframes: Optional[List[TimeframeSettingsBase]] = Field(None, description="Timeframe settings")
    institutional_settings: Optional[InstitutionalBehaviorSettingsBase] = Field(None, description="Institutional behavior settings")
    entry_exit_settings: Optional[EntryExitSettingsBase] = Field(None, description="Entry and exit settings")
    market_state_settings: Optional[MarketStateSettingsBase] = Field(None, description="Market state settings")
    risk_settings: Optional[RiskManagementSettingsBase] = Field(None, description="Risk management settings")
    quality_criteria: Optional[SetupQualityCriteriaBase] = Field(None, description="Setup quality criteria")
    spread_settings: Optional[VerticalSpreadSettingsBase] = Field(None, description="Vertical spread settings")
    meta_learning: Optional[MetaLearningSettingsBase] = Field(None, description="Meta-learning settings")
    multi_timeframe_settings: Optional[MultiTimeframeConfirmationSettingsBase] = Field(None, description="Multi-timeframe confirmation settings")
    
    @model_validator(mode='after')
    def validate_entry_settings(self) -> 'StrategyCreate':
        """Ensure entry settings are valid for strategy type."""
        if self.type == StrategyTypeEnum.VERTICAL_SPREAD:
            if not self.spread_settings or not self.spread_settings.use_vertical_spreads:
                raise ValueError("Vertical spread strategy must have spread settings with use_vertical_spreads=True")
        
        return self

class StrategyUpdate(BaseSchema):
    """Schema for updating an existing strategy."""
    name: Optional[str] = Field(None, min_length=3, max_length=100, description="Strategy name")
    description: Optional[str] = Field(None, description="Strategy description")
    type: Optional[StrategyTypeEnum] = Field(None, description="Strategy type")
    configuration: Optional[Dict[str, Any]] = Field(None, description="Strategy configuration")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Strategy parameters")
    validation_rules: Optional[Dict[str, Any]] = Field(None, description="Parameter validation rules")
    timeframes: Optional[List[TimeframeSettingsBase]] = Field(None, description="Timeframe settings")
    institutional_settings: Optional[InstitutionalBehaviorSettingsBase] = Field(None, description="Institutional behavior settings")
    entry_exit_settings: Optional[EntryExitSettingsBase] = Field(None, description="Entry and exit settings")
    market_state_settings: Optional[MarketStateSettingsBase] = Field(None, description="Market state settings")
    risk_settings: Optional[RiskManagementSettingsBase] = Field(None, description="Risk management settings")
    quality_criteria: Optional[SetupQualityCriteriaBase] = Field(None, description="Setup quality criteria")
    spread_settings: Optional[VerticalSpreadSettingsBase] = Field(None, description="Vertical spread settings")
    meta_learning: Optional[MetaLearningSettingsBase] = Field(None, description="Meta-learning settings")
    multi_timeframe_settings: Optional[MultiTimeframeConfirmationSettingsBase] = Field(None, description="Multi-timeframe confirmation settings")

class StrategyResponse(StrategyBase):
    """Schema for strategy response from the API."""
    id: int = Field(..., description="Strategy ID")
    user_id: int = Field(..., description="Owner user ID")
    created_by_id: int = Field(..., description="Creator user ID")
    updated_by_id: Optional[int] = Field(None, description="Last updater user ID")
    is_active: bool = Field(False, description="Whether the strategy is active")
    version: int = Field(1, description="Strategy version")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    status: str = Field("draft", description="Strategy status")
    timeframes: Optional[List[TimeframeSettingsBase]] = Field(None, description="Timeframe settings")
    win_rate: Optional[float] = Field(None, description="Strategy win rate")
    profit_factor: Optional[float] = Field(None, description="Strategy profit factor")
    sharpe_ratio: Optional[float] = Field(None, description="Strategy Sharpe ratio")
    sortino_ratio: Optional[float] = Field(None, description="Strategy Sortino ratio")
    max_drawdown: Optional[float] = Field(None, description="Maximum drawdown percentage")
    total_profit_inr: Optional[float] = Field(None, description="Total profit in INR")
    avg_win_inr: Optional[float] = Field(None, description="Average win in INR")
    avg_loss_inr: Optional[float] = Field(None, description="Average loss in INR")

# Analysis result schemas
class TimeframeAnalysisResult(BaseSchema):
    """Result of timeframe analysis."""
    aligned: bool = Field(..., description="Whether timeframes are aligned")
    alignment_score: float = Field(..., description="Timeframe alignment score (0-1)")
    timeframe_results: Dict[str, Any] = Field(..., description="Detailed results by timeframe")
    primary_direction: Optional[str] = Field(None, description="Primary trend direction")
    require_all_aligned: bool = Field(..., description="Whether all timeframes must be aligned")
    min_alignment_score: float = Field(..., description="Minimum required alignment score")
    sufficient_alignment: bool = Field(..., description="Whether alignment score is sufficient")
    fifteen_min_aligned: Optional[bool] = Field(None, description="Whether 15-min timeframe confirms")

class MarketStateAnalysis(BaseSchema):
    """Result of market state analysis."""
    market_state: MarketStateRequirementEnum = Field(..., description="Current market state")
    trend_phase: TrendPhaseEnum = Field(..., description="Current trend phase")
    is_railroad_trend: bool = Field(..., description="Whether this is a railroad trend")
    is_creeper_move: bool = Field(..., description="Whether this is a creeper move")
    has_two_day_trend: bool = Field(..., description="Whether there's a two-day trend")
    trend_direction: str = Field(..., description="Trend direction")
    price_indicator_divergence: bool = Field(..., description="Whether price and indicators diverge")
    price_struggling_near_ma: bool = Field(..., description="Whether price is struggling near MA")
    institutional_fight_in_progress: bool = Field(..., description="Whether institutional fight in progress")
    accumulation_detected: bool = Field(..., description="Whether accumulation is detected")
    bos_detected: bool = Field(..., description="Whether break of structure is detected")

class SetupQualityResult(BaseSchema):
    """Result of setup quality evaluation."""
    strategy_id: int = Field(..., description="Strategy ID")
    grade: SetupQualityGradeEnum = Field(..., description="Setup quality grade")
    score: float = Field(..., description="Setup quality score (0-100)")
    factor_scores: Dict[str, float] = Field(..., description="Scores for individual factors")
    position_size: int = Field(..., description="Recommended position size")
    risk_percent: float = Field(..., description="Recommended risk percentage")
    can_auto_trade: bool = Field(..., description="Whether this setup can be auto-traded")
    analysis_comments: List[str] = Field(..., description="Analysis comments")

# Signal schemas
class SignalBase(BaseSchema):
    """Base schema for trading signals."""
    instrument: str = Field(..., description="Trading instrument")
    direction: DirectionEnum = Field(..., description="Trade direction")

class SignalCreate(SignalBase):
    """Schema for creating a new signal."""
    timeframe_analysis: TimeframeAnalysisResult = Field(..., description="Timeframe analysis result")
    market_state: MarketStateAnalysis = Field(..., description="Market state analysis")
    setup_quality: SetupQualityResult = Field(..., description="Setup quality evaluation")
    market_data: Dict[str, Dict] = Field(..., description="Market data by timeframe")

class SignalResponse(SignalBase):
    """Schema for signal response from the API."""
    id: int = Field(..., description="Signal ID")
    strategy_id: int = Field(..., description="Strategy ID")
    signal_type: str = Field(..., description="Signal type")
    entry_price: float = Field(..., description="Entry price")
    entry_time: datetime = Field(..., description="Entry timestamp")
    entry_timeframe: TimeframeValueEnum = Field(..., description="Entry timeframe")
    entry_technique: EntryTechniqueEnum = Field(..., description="Entry technique")
    take_profit_price: float = Field(..., description="Take profit price")
    stop_loss_price: float = Field(..., description="Stop loss price")
    trailing_stop: bool = Field(..., description="Whether to use trailing stop")
    position_size: int = Field(..., description="Position size in lots")
    risk_reward_ratio: float = Field(..., description="Risk-reward ratio")
    risk_amount: float = Field(..., description="Risk amount in INR")
    setup_quality: SetupQualityGradeEnum = Field(..., description="Setup quality grade")
    setup_score: float = Field(..., description="Setup quality score")
    confidence: float = Field(..., description="Signal confidence (0-1)")
    market_state: MarketStateRequirementEnum = Field(..., description="Market state")
    trend_phase: TrendPhaseEnum = Field(..., description="Trend phase")
    is_active: bool = Field(..., description="Whether signal is active")
    is_executed: bool = Field(..., description="Whether signal has been executed")
    execution_time: Optional[datetime] = Field(None, description="Execution timestamp")
    timeframe_alignment_score: Optional[float] = Field(None, description="Timeframe alignment score")
    primary_timeframe_aligned: Optional[bool] = Field(None, description="Whether primary timeframe is aligned")
    institutional_footprint_detected: Optional[bool] = Field(None, description="Whether institutional footprint detected")
    bos_detected: Optional[bool] = Field(None, description="Whether break of structure detected")
    is_spread_trade: Optional[bool] = Field(False, description="Whether this is a spread trade")
    spread_type: Optional[SpreadTypeEnum] = Field(None, description="Spread type if applicable")

# Trade schemas
class TradeBase(BaseSchema):
    """Base schema for trades."""
    exit_price: float = Field(..., description="Exit price")
    exit_reason: str = Field("manual", description="Reason for exit")

class TradeCreate(TradeBase):
    """Schema for creating a new trade."""
    signal_id: int = Field(..., description="Signal ID")
    execution_price: float = Field(..., description="Execution price")
    execution_time: Optional[datetime] = Field(None, description="Execution timestamp")
    
    @field_validator('execution_price')
    @classmethod
    def validate_execution_price(cls, v):
        """Ensure execution price is positive."""
        if v <= 0:
            raise ValueError("Execution price must be positive")
        return v

class TradeResponse(BaseSchema):
    """Schema for trade response from the API."""
    id: int = Field(..., description="Trade ID")
    strategy_id: int = Field(..., description="Strategy ID")
    signal_id: int = Field(..., description="Signal ID")
    instrument: str = Field(..., description="Trading instrument")
    direction: DirectionEnum = Field(..., description="Trade direction")
    entry_price: float = Field(..., description="Entry price")
    entry_time: datetime = Field(..., description="Entry timestamp")
    exit_price: Optional[float] = Field(None, description="Exit price")
    exit_time: Optional[datetime] = Field(None, description="Exit timestamp")
    exit_reason: Optional[str] = Field(None, description="Reason for exit")
    position_size: int = Field(..., description="Position size in lots")
    commission: float = Field(..., description="Commission in INR")
    taxes: float = Field(..., description="Taxes in INR")
    slippage: float = Field(..., description="Slippage in points")
    profit_loss_points: Optional[float] = Field(None, description="Profit/loss in points")
    profit_loss_inr: Optional[float] = Field(None, description="Profit/loss in INR")
    initial_risk_points: float = Field(..., description="Initial risk in points")
    initial_risk_inr: float = Field(..., description="Initial risk in INR")
    initial_risk_percent: float = Field(..., description="Initial risk as percentage")
    risk_reward_planned: float = Field(..., description="Planned risk-reward ratio")
    actual_risk_reward: Optional[float] = Field(None, description="Actual risk-reward ratio")
    setup_quality: SetupQualityGradeEnum = Field(..., description="Setup quality grade")
    setup_score: float = Field(..., description="Setup quality score")
    holding_period_minutes: Optional[int] = Field(None, description="Holding period in minutes")
    total_costs: Optional[float] = Field(None, description="Total trading costs")
    is_spread_trade: Optional[bool] = Field(False, description="Whether this is a spread trade")
    spread_type: Optional[SpreadTypeEnum] = Field(None, description="Spread type if applicable")

# Feedback schemas
class FeedbackBase(BaseSchema):
    """Base schema for trade feedback."""
    feedback_type: FeedbackTypeEnum = Field(..., description="Feedback type")
    title: str = Field(..., description="Feedback title")
    description: Optional[str] = Field(None, description="Feedback description")
    file_path: Optional[str] = Field(None, description="Path to attached file")
    file_type: Optional[str] = Field(None, description="Type of attached file")
    tags: Optional[List[str]] = Field(None, description="Tags for categorizing feedback")
    improvement_category: Optional[str] = Field(None, description="Category for improvement")
    applies_to_setup: bool = Field(False, description="Whether feedback applies to setup")
    applies_to_entry: bool = Field(False, description="Whether feedback applies to entry")
    applies_to_exit: bool = Field(False, description="Whether feedback applies to exit")
    applies_to_risk: bool = Field(False, description="Whether feedback applies to risk management")
    pre_trade_conviction_level: Optional[float] = Field(None, description="Pre-trade conviction level (0-10)")
    emotional_state_rating: Optional[int] = Field(None, description="Emotional state rating (1-5)")
    lessons_learned: Optional[str] = Field(None, description="Lessons learned")
    action_items: Optional[str] = Field(None, description="Action items for improvement")
    
    @field_validator('pre_trade_conviction_level')
    @classmethod
    def validate_conviction_level(cls, v):
        """Ensure conviction level is between 0 and 10."""
        if v is not None and (v < 0 or v > 10):
            raise ValueError("Conviction level must be between 0 and 10")
        return v
    
    @field_validator('emotional_state_rating')
    @classmethod
    def validate_emotional_state(cls, v):
        """Ensure emotional state rating is between 1 and 5."""
        if v is not None and (v < 1 or v > 5):
            raise ValueError("Emotional state rating must be between 1 and 5")
        return v
    
    @model_validator(mode='after')
    def validate_application_fields(self) -> 'FeedbackBase':
        """Ensure at least one application field is True."""
        if not any([self.applies_to_setup, self.applies_to_entry, 
                   self.applies_to_exit, self.applies_to_risk]):
            raise ValueError("Feedback must apply to at least one aspect (setup, entry, exit, or risk)")
        
        return self

class FeedbackCreate(FeedbackBase):
    """Schema for creating new feedback."""
    pass

class FeedbackResponse(FeedbackBase):
    """Schema for feedback response from the API."""
    id: int = Field(..., description="Feedback ID")
    strategy_id: int = Field(..., description="Strategy ID")
    trade_id: Optional[int] = Field(None, description="Trade ID if applicable")
    created_at: datetime = Field(..., description="Creation timestamp")
    has_been_applied: bool = Field(False, description="Whether feedback has been applied")
    applied_date: Optional[datetime] = Field(None, description="Date feedback was applied")
    applied_to_version_id: Optional[int] = Field(None, description="Strategy version that applied this feedback")

# Performance analysis schema
class PerformanceAnalysis(BaseSchema):
    """Schema for strategy performance analysis."""
    strategy_id: int = Field(..., description="Strategy ID")
    total_trades: int = Field(..., description="Total number of trades")
    win_count: int = Field(..., description="Number of winning trades")
    loss_count: int = Field(..., description="Number of losing trades")
    win_rate: float = Field(..., description="Win rate (0-1)")
    total_profit_inr: float = Field(..., description="Total profit in INR")
    avg_win_inr: float = Field(..., description="Average win in INR")
    avg_loss_inr: float = Field(..., description="Average loss in INR")
    profit_factor: float = Field(..., description="Profit factor")
    trades_by_grade: Dict[str, Dict[str, Any]] = Field(..., description="Performance by setup quality grade")
    analysis_period: Dict[str, datetime] = Field(..., description="Analysis time period")
    largest_win_inr: Optional[float] = Field(None, description="Largest win in INR")
    largest_loss_inr: Optional[float] = Field(None, description="Largest loss in INR")
    consecutive_wins: Optional[int] = Field(None, description="Maximum consecutive wins")
    consecutive_losses: Optional[int] = Field(None, description="Maximum consecutive losses")
    avg_holding_period_minutes: Optional[float] = Field(None, description="Average holding period in minutes")
    
    @field_validator('win_rate')
    @classmethod
    def validate_win_rate(cls, v):
        """Ensure win rate is between 0 and 1."""
        if v < 0 or v > 1:
            raise ValueError("Win rate must be between 0 and 1")
        return v
    
    @field_validator('profit_factor')
    @classmethod
    def validate_profit_factor(cls, v):
        """Ensure profit factor is positive."""
        if v < 0:
            raise ValueError("Profit factor must be positive")
        return v