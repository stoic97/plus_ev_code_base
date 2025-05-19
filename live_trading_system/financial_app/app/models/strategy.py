"""
Comprehensive Strategy Models Implementing Rikk's Trading Principles

This model layer meticulously captures all aspects of Rikk's sophisticated trading approach including:
- Hierarchical timeframe structure with strict alignment requirements (Daily > 4H > 1H > 15m > 5m)
- Bidirectional application of all techniques (symmetrical long/short implementation)
- Institutional behavior recognition and precise waiting periods
- Specific entry/exit techniques with exact stop placement rules
- Quality-based trade grading system (A+ through F) determining position sizing
- Market phase recognition targeting only the "middle" of trends
- Railroad vs creeper move distinction with clear quality thresholds
- MA relationship analysis (21 MA and 200 MA focus)
- Trading box creation from previous day references
- Price action overriding indicator signals when divergent
- Profit targets in points (20-25 points) rather than percentages
- INR-based cost previews for all trading activities
- Meta-learning system for continuous improvement
- Vertical spread configuration with reduced timing pressure
"""

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, ForeignKey, Table, JSON, Enum, Text, SmallInteger, 
    CheckConstraint, UniqueConstraint, Index
)
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func
import enum
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, time

# Import only the specific classes we need - avoid circular references
from app.core.database import Base
from app.models.base import (
    TimestampMixin, UserRelationMixin, AuditMixin,
    SoftDeleteMixin, VersionedMixin, StatusMixin
)

# ---- ENHANCED ENUMERATIONS SECTION ----

class StrategyType(enum.Enum):
    """Enumeration of strategy types"""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    MOMENTUM = "momentum"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    PATTERN_RECOGNITION = "pattern_recognition"
    MULTI_TIMEFRAME = "multi_timeframe"  # Added specifically for Rikk's approach
    INSTITUTIONAL_FLOW = "institutional_flow"  # Added for institutional behavior focus
    VERTICAL_SPREAD = "vertical_spread"  # Added for options spread strategy
    CUSTOM = "custom"


class StrategyStatus(enum.Enum):
    """Enumeration of strategy statuses"""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"
    BACKTEST = "backtest"


class Direction(enum.Enum):
    """Trade direction - essential for bidirectional technique application"""
    LONG = "long"
    SHORT = "short"
    BOTH = "both"  # Strategy works in both directions


class EntryTechnique(enum.Enum):
    """Entry techniques based on Rikk's principles - fully bidirectional"""
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


class TimeframeImportance(enum.Enum):
    """Importance levels for timeframes in strategy - Rikk's hierarchy"""
    PRIMARY = "primary"           # Direction-determining timeframe
    CONFIRMATION = "confirmation" # Confirms signals from primary
    ENTRY = "entry"               # Used only for entry timing
    FILTER = "filter"             # Used to filter signals
    CONTEXT = "context"           # Provides broader market context


class TimeframeValue(enum.Enum):
    """Standard timeframe values"""
    DAILY = "1d"
    FOUR_HOUR = "4h"
    ONE_HOUR = "1h"
    THIRTY_MIN = "30m"
    FIFTEEN_MIN = "15m"
    FIVE_MIN = "5m"
    THREE_MIN = "3m"


class MarketStateRequirement(enum.Enum):
    """Market state requirements - expanded from Rikk's descriptions"""
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


class VolumePattern(enum.Enum):
    """Volume patterns for institutional behavior detection"""
    HIGH_VOLUME_NARROW_RANGE = "high_volume_narrow_range"  # Accumulation/distribution
    CLIMAX = "climax"                                      # Exhaustion move
    ABSORPTION = "absorption"                             # High volume, little price movement
    DECLINING_VOLUME_TREND = "declining_volume_trend"      # Weakening trend
    INCREASING_VOLUME_TREND = "increasing_volume_trend"    # Strengthening trend
    VOLUME_DIVERGENCE = "volume_divergence"               # Price/volume divergence


class PriceActionPattern(enum.Enum):
    """Price action patterns from Rikk's descriptions"""
    RAILROAD_TREND = "railroad_trend"          # One-sided strong trend
    CREEPER_MOVE = "creeper_move"              # Slow grinding price action
    FAIR_VALUE_GAP = "fair_value_gap"          # Gap in price action
    ORDER_BLOCK = "order_block"                # Institutional order block
    LIQUIDITY_GRAB = "liquidity_grab"          # False breakout for liquidity
    STOP_HUNT = "stop_hunt"                    # Hunting for stops before actual move
    V_REVERSAL = "v_reversal"                  # Sharp V-shaped reversal
    TORPEDO = "torpedo"                        # Strong momentum candle


class MovingAverageType(enum.Enum):
    """Types of moving averages for strategy"""
    SIMPLE = "simple"             # Simple Moving Average (SMA)
    EXPONENTIAL = "exponential"   # Exponential Moving Average (EMA)
    WEIGHTED = "weighted"         # Weighted Moving Average
    HULL = "hull"                 # Hull Moving Average


class FollowThroughType(enum.Enum):
    """Types of follow-through confirmation for breakouts"""
    IMMEDIATE_CONTINUATION = "immediate_continuation"  # Immediate follow-through
    CLOSE_ABOVE_BREAKOUT = "close_above_breakout"     # Close above breakout level
    CLOSE_BELOW_BREAKOUT = "close_below_breakout"     # Close below breakout level
    NEXT_BAR_CONFIRMATION = "next_bar_confirmation"   # Next bar confirms


class TrailingStopMethod(enum.Enum):
    """Methods for trailing stops based on Rikk's approach"""
    BAR_BY_BAR = "bar_by_bar"                         # Trail bar by bar
    PREVIOUS_SWING = "previous_swing"                 # Trail to previous swing
    MOVING_AVERAGE = "moving_average"                 # Trail based on MA
    ATR_MULTIPLE = "atr_multiple"                     # Trail based on ATR
    FIXED_POINTS = "fixed_points"                     # Trail by fixed points
    CLOSE_BELOW_MA = "close_below_ma"                 # Trail to below MA (for longs)
    CLOSE_ABOVE_MA = "close_above_ma"                 # Trail to above MA (for shorts)


class ProfitTargetMethod(enum.Enum):
    """Methods for setting profit targets"""
    FIXED_POINTS = "fixed_points"              # Fixed point target (Rikk's 20-25 points)
    ATR_MULTIPLE = "atr_multiple"              # Multiple of ATR
    PREVIOUS_SWING = "previous_swing"          # Previous swing high/low
    FIBONACCI_EXTENSION = "fibonacci_extension"  # Fibonacci extension
    PREMIUM_DISCOUNT_ZONE = "premium_discount_zone"  # Premium/discount zone


class SetupQualityGrade(enum.Enum):
    """Trade setup quality grades"""
    A_PLUS = "a_plus"    # Highest quality setup
    A = "a"              # Excellent setup
    B = "b"              # Good setup
    C = "c"              # Acceptable setup
    D = "d"              # Poor setup
    F = "f"              # Fail/Do not trade


class SpreadType(enum.Enum):
    """Types of vertical spreads"""
    BULL_CALL_SPREAD = "bull_call_spread"  # Buy ATM call, sell OTM call
    BEAR_PUT_SPREAD = "bear_put_spread"    # Buy ATM put, sell OTM put
    BEAR_CALL_SPREAD = "bear_call_spread"  # Sell ATM call, buy OTM call
    BULL_PUT_SPREAD = "bull_put_spread"    # Sell ATM put, buy OTM put


class TrendPhase(enum.Enum):
    """Phases of a trend - for targeting specific portions"""
    EARLY = "early"          # Beginning of trend (avoid)
    MIDDLE = "middle"        # Middle of trend (target - Rikk's focus)
    LATE = "late"            # End of trend (avoid)
    UNDETERMINED = "undetermined"  # Cannot determine phase


class BOSType(enum.Enum):
    """Types of Break of Structure patterns"""
    SWING_HIGH_BREAK = "swing_high_break"  # Break of swing high
    SWING_LOW_BREAK = "swing_low_break"    # Break of swing low
    RANGE_HIGH_BREAK = "range_high_break"  # Break of range high
    RANGE_LOW_BREAK = "range_low_break"    # Break of range low
    MA_CROSS = "ma_cross"                  # Moving average cross
    TREND_LINE_BREAK = "trend_line_break"  # Break of trend line


class FeedbackType(enum.Enum):
    """Types of feedback for meta-learning"""
    TEXT_NOTE = "text_note"          # Text notes on trade
    SCREENSHOT = "screenshot"        # Screenshot of chart
    CHART_ANNOTATION = "chart_annotation"  # Annotated chart
    VIDEO_RECORDING = "video_recording"    # Video recording
    TRADE_REVIEW = "trade_review"    # Full trade review


# Define many-to-many relationship for strategy categories
strategy_category_association = Table(
    'strategy_category_association',
    Base.metadata,
    Column('strategy_id', Integer, ForeignKey('strategies.id'), primary_key=True),
    Column('category_id', Integer, ForeignKey('strategy_categories.id'), primary_key=True)
)


# Create a base model class specifically for this file to avoid MRO issues
class StrategyBaseModel(Base):
    """Base model for strategy models to avoid MRO conflicts."""
    __abstract__ = True
    
    id = Column(Integer, primary_key=True, index=True)
    
    @classmethod
    def get_by_id(cls, session, id):
        """Get a record by its primary key."""
        return session.query(cls).filter(cls.id == id).first()
    
    @classmethod
    def list_all(cls, session, limit=100, offset=0, order_by=None):
        """Get all records with pagination."""
        query = session.query(cls)
        
        if order_by:
            if isinstance(order_by, (list, tuple)):
                for order_field in order_by:
                    query = query.order_by(order_field)
            else:
                query = query.order_by(order_by)
        
        return query.limit(limit).offset(offset).all()
    
    def save(self, session):
        """Save the current model to the database."""
        session.add(self)
        session.flush()  # Flush to get the ID
        return self


# Model for timeframe definitions
class StrategyTimeframe(StrategyBaseModel):
    """
    Strategy timeframe model defines multi-timeframe alignment requirements.
    
    This model explicitly defines the hierarchical timeframes used in a strategy and their 
    relationships, following Rikk's strict principle of higher timeframe alignment.
    """
    __tablename__ = "strategy_timeframes"
    
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False)
    name = Column(String(50), nullable=False)  # e.g., "Daily", "Hourly", "15min"
    value = Column(Enum(TimeframeValue), nullable=False)
    importance = Column(Enum(TimeframeImportance), nullable=False, 
                       default=TimeframeImportance.CONFIRMATION)
    order = Column(SmallInteger, nullable=False, default=0)  # Order of importance (0 is highest)
    
    # Moving average settings for this timeframe
    # Using 21 and 200 MAs as specifically emphasized by Rikk
    ma_type = Column(Enum(MovingAverageType), default=MovingAverageType.SIMPLE)  # SMA by default
    ma_period_primary = Column(Integer, default=21)  # 21 MA (not 20) as emphasized by Rikk
    ma_period_secondary = Column(Integer, default=200)  # 200 MA as a key reference level
    
    # Requirements for this timeframe (bidirectional)
    require_alignment = Column(Boolean, default=True)  # Must align with higher timeframes
    require_price_above_ma = Column(Boolean)  # Price must be above MA (for longs)
    require_price_below_ma = Column(Boolean)  # Price must be below MA (for shorts)
    require_ma_slope_up = Column(Boolean)     # MA must be sloping up (for longs)
    require_ma_slope_down = Column(Boolean)   # MA must be sloping down (for shorts)
    
    # Minimum bars for confirmation (bidirectional)
    min_confirmation_bars = Column(Integer, default=1)  # Bars needed to confirm trend
    
    # MA struggle detection - "prices are struggling to move away from EMA"
    detect_ma_struggle = Column(Boolean, default=True)
    ma_struggle_threshold = Column(Float, default=0.2)  # % threshold for struggle detection
    
    # MA slope requirements - "hourly should begin to trend"
    min_ma_slope_for_trend = Column(Float, default=0.0005)  # Minimum slope for valid trend
    
    # Two-day trend relationship - "2 green daily bars = 12 hourly green bars"
    if_daily_tf_check_consecutive_days = Column(Integer, default=2)  # Check for 2 consecutive days
    equivalent_hourly_bars = Column(Integer, default=12)  # Equivalent to 2 daily bars
    
    # 15-min confirmation - "Wait for 15-min to confirm"
    if_15m_tf_wait_for_confirmation = Column(Boolean, default=True) 
    min_15m_confirmation_bars = Column(Integer, default=2)  # At least 2 bars confirming
    
    # Relationship
    strategy = relationship("Strategy", back_populates="timeframes")
    
    __table_args__ = (
        UniqueConstraint('strategy_id', 'value', name='uix_strategy_timeframe_value'),
    )


# Previous day reference model
class PreviousDayReference(StrategyBaseModel):
    """
    Previous day reference data model.
    
    Contains reference points from the previous trading day for context at market open,
    implementing Rikk's concept of the "trading box" and the importance of previous
    day data for current day's context.
    """
    __tablename__ = "previous_day_references"
    
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False, unique=True)
    
    # Previous day key levels
    use_previous_day_data = Column(Boolean, default=True)
    track_previous_close = Column(Boolean, default=True)
    track_previous_high = Column(Boolean, default=True)  # PDH - key breakout level
    track_previous_low = Column(Boolean, default=True)   # PDL - key breakout level
    track_previous_open = Column(Boolean, default=True)
    
    # MA positions from previous day (21 MA and 200 MA)
    track_ma21_position = Column(Boolean, default=True)  # Track 21 MA position
    track_ma200_position = Column(Boolean, default=True)  # Track 200 MA position
    
    # Last 45 minutes of previous day - Rikk's specific emphasis
    track_last_45min_activity = Column(Boolean, default=True)
    last_45min_analysis_fields = Column(JSON, default=lambda: [
        "price_movement",
        "volume_profile",
        "closing_momentum",
        "ma_relationship"
    ])
    
    # Trading box structure - Rikk's contextual frame for the current day
    create_trading_box = Column(Boolean, default=True)
    trading_box_fields = Column(JSON, default=lambda: {
        "upper_boundary": "previous_high",
        "lower_boundary": "previous_low",
        "reference_level": "previous_close",
        "ma21_position": True,
        "ma200_position": True
    })
    
    # 15-min alignment check for market open
    check_15min_alignment_at_open = Column(Boolean, default=True)
    wait_for_15min_confirmation = Column(Boolean, default=True)
    min_15min_bars_for_trend_confirmation = Column(Integer, default=3)
    
    # Open assessment - "I place lot of emphasis on open"
    analyze_open_strength = Column(Boolean, default=True)
    open_strength_criteria = Column(JSON, default=lambda: {
        "gap_direction": True,
        "first_bar_size": True, 
        "first_bar_close_vs_open": True,
        "volume_profile": True,
        "ma_relationship": True
    })
    
    # Daily bars importance - 2 green/red daily bars check
    check_two_day_trend = Column(Boolean, default=True)  # "2 green daily bars..."
    daily_trend_threshold = Column(Integer, default=2)  # Min consecutive daily bars
    
    # Relationship
    strategy = relationship("Strategy", back_populates="previous_day_reference")


# Institutional behavior detection settings
class InstitutionalBehaviorSettings(StrategyBaseModel):
    """
    Settings for detecting institutional trading behavior patterns.
    
    Captures Rikk's emphasis on institutional behavior detection including
    accumulation, stop hunts, liquidity grabs, and waiting for the "fight to get over".
    """
    __tablename__ = "institutional_behavior_settings"
    
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False, unique=True)
    
    # Accumulation/distribution detection
    detect_accumulation = Column(Boolean, default=True)  # "They are accumulating..."
    accumulation_volume_threshold = Column(Float, default=1.5)  # Relative volume threshold
    accumulation_price_threshold = Column(Float, default=0.002)  # Max price movement %
    accumulation_min_time = Column(Integer, default=20)  # "It will take days for setup to take place"
    
    # Liquidity grab detection (bidirectional)
    detect_liquidity_grabs = Column(Boolean, default=True)  # "Liquidity trap kind of move"
    liquidity_grab_reversal_threshold = Column(Float, default=0.5)  # % of breakout move
    
    # Stop hunt detection (bidirectional)
    detect_stop_hunts = Column(Boolean, default=True)  # "It took out the SL of longs, enticed shorts..."
    stop_hunt_threshold = Column(Float, default=0.3)  # % of move to qualify as stop hunt
    
    # Volume pattern detection
    detect_volume_patterns = Column(Boolean, default=True)
    volume_pattern_types = Column(JSON, default=lambda: [p.value for p in VolumePattern])
    
    # Price action pattern detection
    detect_price_patterns = Column(Boolean, default=True)
    price_pattern_types = Column(JSON, default=lambda: [p.value for p in PriceActionPattern])
    
    # CHOCH detection (Change of Character)
    detect_choch = Column(Boolean, default=True)  # Rikk mentions this specifically
    choch_lookback_bars = Column(Integer, default=10)
    
    # Trading with institutions settings - key Rikk principle
    require_institutional_alignment = Column(Boolean, default=True)  # "Trade with institutions, not against them"
    wait_for_institutional_footprints = Column(Boolean, default=True)  # "Let footprint of big traders be seen"
    
    # Multi-leg move tracking - Rikk tracks the "legs" of moves
    track_move_legs = Column(Boolean, default=True)  # "2 legs of down moves"
    max_legs_to_track = Column(Integer, default=3)
    
    # BOS (Break of Structure) detection
    detect_bos = Column(Boolean, default=True)  # "Once u see BOS happening..."
    bos_confirmation_bars = Column(Integer, default=1)
    bos_detection_timeframes = Column(JSON, default=lambda: ["1d", "4h", "1h", "15m"])
    
    # BOS types to detect (bidirectional)
    bos_types = Column(JSON, default=lambda: [t.value for t in BOSType])
    
    # BOS qualification criteria
    bos_volume_confirmation = Column(Boolean, default=True)  # Require volume confirmation
    bos_minimum_size = Column(Float)  # Minimum size of the break in points
    
    # Wait for institutional fight to end - keystone principle
    wait_for_institutional_fight = Column(Boolean, default=True)  # "Let the institutions fight get over"
    institutional_fight_detection_methods = Column(JSON, default=lambda: [
        "high_volume_narrow_range",
        "price_rejection",
        "rapid_reversals",
        "failed_breakouts"
    ])
    
    # Market microstructure analysis
    analyze_order_flow = Column(Boolean, default=True)
    track_limit_order_book_changes = Column(Boolean, default=False)  # Requires advanced data feed
    track_trade_volume_delta = Column(Boolean, default=True)
    order_flow_metrics = Column(JSON, default=lambda: [
        "buy_sell_imbalance", "large_orders", "iceberg_detection",
        "aggressive_orders", "passive_orders", "time_of_order_placement"
    ])
    volume_profile_analysis = Column(Boolean, default=True)
    volume_profile_zones = Column(JSON, default=lambda: [
        "high_volume_node", "low_volume_node", "point_of_control",
        "value_area_high", "value_area_low", "volume_gaps"
    ])
    microstructure_timeframes = Column(JSON, default=lambda: ["5m", "15m", "1h"])
    delta_divergence_tracking = Column(Boolean, default=True)  # Track price/delta divergences
    
    # Relationship
    strategy = relationship("Strategy", back_populates="institutional_settings")


# Entry/exit technique settings
class EntryExitSettings(StrategyBaseModel):
    """
    Detailed entry and exit technique settings with full bidirectional implementation.
    
    Captures Rikk's specific entry and exit methods including green/red bar after 
    pullback/rally, precise stop placement, and trailing stop approaches.
    """
    __tablename__ = "entry_exit_settings"
    
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False, unique=True)
    
    # Direction
    direction = Column(Enum(Direction), nullable=False, default=Direction.BOTH)
    
    # Entry techniques (with bidirectional options)
    primary_entry_technique = Column(Enum(EntryTechnique), nullable=False)
    secondary_entry_technique = Column(Enum(EntryTechnique))
    
    # Entry conditions - universal
    require_candle_close_confirmation = Column(Boolean, default=True)  # "Wait for candle close confirmation"
    max_distance_from_ma_points = Column(Integer)  # Max distance from MA in points
    min_pullback_points = Column(Integer)  # Minimum pullback size in points
    max_pullback_points = Column(Integer)  # Maximum pullback size in points
    
    # MA alignment for entry
    require_ma_alignment = Column(Boolean, default=True)  # MA must be in correct direction
    require_price_ma_alignment = Column(Boolean, default=True)  # Price must be on correct side of MA
    
    # Follow-through requirements (bidirectional)
    require_follow_through = Column(Boolean, default=True)  # "A breakout candle should have some follow-through"
    follow_through_type = Column(Enum(FollowThroughType), 
                                default=FollowThroughType.NEXT_BAR_CONFIRMATION)
    follow_through_min_points = Column(Integer, default=5)  # Minimum points for follow-through
    
    # Green bar after pullback specific settings (LONG) - Rikk's exact technique
    green_bar_min_body_percent = Column(Float, default=0.5)  # Min body size relative to range
    green_bar_min_points = Column(Integer)  # Minimum points size for green bar
    green_bar_sl_placement = Column(String(50), default="below_bar")  # "Enter above it with SL below first green bar"
    
    # Red bar after rally specific settings (SHORT) - Bidirectional implementation
    red_bar_min_body_percent = Column(Float, default=0.5)  # Min body size relative to range
    red_bar_min_points = Column(Integer)  # Minimum points size for red bar
    red_bar_sl_placement = Column(String(50), default="above_bar")  # Place SL above the bar (mirror image of long)
    
    # Exit techniques
    trailing_stop_method = Column(Enum(TrailingStopMethod), default=TrailingStopMethod.BAR_BY_BAR)
    trailing_stop_atr_multiple = Column(Float, default=1.0)  # If using ATR trailing
    trailing_stop_points = Column(Integer)  # Points for fixed point trailing
    
    # Profit targets - Rikk's specific point targets
    profit_target_method = Column(Enum(ProfitTargetMethod), default=ProfitTargetMethod.FIXED_POINTS)
    profit_target_points = Column(Integer, default=25)  # Rikk's 20-25 point target
    profit_target_atr_multiple = Column(Float, default=2.0)  # If using ATR multiple
    
    # Scaling out settings - Using points instead of percentages
    use_scaling_out = Column(Boolean, default=False)  # "You may close 6-7 and let 3-4 mini continue"
    scaling_out_points = Column(JSON, default=[])  # List of points to scale out at
    scaling_out_percentages = Column(JSON, default=[])  # List of position percentages to scale out
    
    # Breakout confirmation
    confirm_breakouts_with_15min = Column(Boolean, default=True)  # Rikk's emphasis on 15-min confirmation
    wait_for_breakout_retest = Column(Boolean, default=False)
    
    # Trend phase targeting - Rikk's specific focus
    target_trend_phase = Column(Enum(TrendPhase), default=TrendPhase.MIDDLE)  # "We want the part in the middle"
    
    # Anti-chasing mechanism
    prevent_chasing = Column(Boolean, default=True)  # "Avoid chasing price that's already moved far"
    max_distance_from_value_area = Column(Float)  # Maximum distance from value area
    
    # Regular trading cost preview (like spreads)
    show_cost_preview = Column(Boolean, default=True)  # Show cost preview in INR
    include_brokerage_in_preview = Column(Boolean, default=True) 
    include_taxes_in_preview = Column(Boolean, default=True)
    brokerage_per_lot = Column(Float, default=20.0)
    exchange_transaction_tax = Column(Float, default=0.0005)  # 0.05% typical value
    
    # Automated trading for regular trades
    auto_trade_regular = Column(Boolean, default=False)
    require_confirmation_regular = Column(Boolean, default=True)  # Show preview before executing
    auto_trade_only_a_plus_regular = Column(Boolean, default=True)  # Only auto-trade A+ setups
    
    # Special instructions from Rikk
    close_if_against_higher_tf = Column(Boolean, default=True)  # "If trade goes against you, close and stay away"
    stay_away_after_loss = Column(Boolean, default=True)  # Stay away after closing losing trade
    
    # Relationship
    strategy = relationship("Strategy", back_populates="entry_exit_settings")


# Market state requirements
class MarketStateSettings(StrategyBaseModel):
    """
    Market state requirements for strategy activation.
    
    Captures Rikk's emphasis on trading in the right market conditions,
    avoiding creeper moves, and focusing on trending markets.
    """
    __tablename__ = "market_state_settings"
    
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False, unique=True)
    
    # Market state requirements
    required_market_state = Column(Enum(MarketStateRequirement), 
                                  default=MarketStateRequirement.ANY)
    avoid_creeper_moves = Column(Boolean, default=True)  # "One of the slowest upmoves in BN..."
    
    # Trend requirements
    require_two_day_trend = Column(Boolean, default=False)  # "2 green daily bars = 12 hourly green bars"
    min_trend_strength = Column(Float)  # Minimum trend strength score
    
    # Volume requirements
    min_relative_volume = Column(Float)  # Minimum relative volume for entry
    
    # Volatility settings
    max_volatility_percentile = Column(Float)  # Max volatility to enter (avoid excessive volatility)
    min_volatility_percentile = Column(Float)  # Min volatility to enter (avoid too quiet markets)
    
    # Previous day levels
    use_pdh_pdl_levels = Column(Boolean, default=True)  # Previous day high/low levels as key points
    
    # Break of structure settings
    detect_break_of_structure = Column(Boolean, default=True)  # "Once u see BOS happening"
    bos_confirmation_bars = Column(Integer, default=1)  # Bars to confirm BOS
    
    # Price vs Indicator divergence - key Rikk principle
    detect_price_indicator_divergence = Column(Boolean, default=True)  # "Though 50 EMA showed a trend, price action did show weakness"
    price_action_overrides_indicators = Column(Boolean, default=True)  # Price action overrides indicators
    
    # Range-to-trend transition detection
    detect_range_breakouts = Column(Boolean, default=True)  # "A break of 15 min TF range will ensure a new trend start"
    range_identification_bars = Column(Integer, default=20)  # Bars to identify range
    
    # MA struggle detection
    detect_price_ma_struggle = Column(Boolean, default=True)  # "Currently the EMA is going flat and prices are struggling to move away from EMA"
    ma_struggle_threshold = Column(Float, default=0.2)  # % threshold for struggle detection
    
    # Railroad vs creeper move distinction - important quality distinction
    prefer_railroad_trends = Column(Boolean, default=True)  # "Railroad trend" preference
    railroad_momentum_threshold = Column(Float, default=0.8)  # Threshold to qualify as railroad
    
    # Wait for 15 minutes confirmation - fundamental principle
    wait_for_15min_alignment = Column(Boolean, default=True)  # "15 mins for entry which should show u probable entry areas"
    min_15min_confirmation_bars = Column(Integer, default=2)
    
    # Relationship
    strategy = relationship("Strategy", back_populates="market_state_settings")


# Risk management settings
class RiskManagementSettings(StrategyBaseModel):
    """
    Comprehensive risk management settings with both percentage and absolute INR values.
    
    Implements Rikk's specific risk control thresholds, position sizing based on 
    setup quality, and consistency-first approach.
    """
    __tablename__ = "risk_management_settings"
    
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False, unique=True)
    
    # Per-trade risk limits
    max_risk_per_trade_percent = Column(Float, default=1.0)  # Max % risk per trade
    max_risk_per_trade_inr = Column(Integer)  # Max INR risk per trade
    
    # Daily risk limits
    max_daily_risk_percent = Column(Float, default=3.0)  # Max % risk per day
    max_daily_risk_inr = Column(Integer)  # Max INR risk per day
    
    # Weekly risk limits - Rikk's specific thresholds
    max_weekly_risk_percent = Column(Float, default=8.0)  # 8% weekly threshold explicitly mentioned
    max_weekly_risk_inr = Column(Integer)  # Max INR risk per week
    
    # Weekly drawdown controls (explicit thresholds)
    weekly_drawdown_threshold = Column(Float, default=8.0)  # 8% weekly threshold
    weekly_drawdown_actions = Column(JSON, default=lambda: [
        "suspend_all_trading",
        "conduct_full_audit",
        "require_committee_approval"
    ])
    
    # Daily drawdown controls
    daily_drawdown_threshold = Column(Float, default=4.0)  # 4% daily threshold
    daily_drawdown_actions = Column(JSON, default=lambda: [
        "reduce_exposure_by_half",
        "notify_risk_committee",
        "perform_strategy_review"
    ])
    
    # Drawdown controls and intelligent recovery
    max_drawdown_percent = Column(Float, default=15.0)  # Max drawdown %
    max_drawdown_inr = Column(Integer)  # Max drawdown in INR
    drawdown_tiers = Column(JSON, default=lambda: {
        "tier1": {"threshold": 5.0, "size_reduction": 0.2, "trade_frequency_reduction": 0.1},
        "tier2": {"threshold": 8.0, "size_reduction": 0.4, "trade_frequency_reduction": 0.3},
        "tier3": {"threshold": 12.0, "size_reduction": 0.6, "trade_frequency_reduction": 0.5}
    })
    
    # Intelligent recovery system
    use_progressive_recovery = Column(Boolean, default=True)
    recovery_win_streak_threshold = Column(Integer, default=3)  # Trades needed to start recovery
    progressive_size_increase = Column(Float, default=0.1)  # Increase by 10% after each win during recovery
    max_recovery_increase_steps = Column(Integer, default=5)  # Maximum steps in recovery
    only_count_a_plus_setups_for_recovery = Column(Boolean, default=True)
    
    # Risk response actions
    suspend_trading_on_max_daily = Column(Boolean, default=True)
    reduce_size_after_loss = Column(Boolean, default=True)  # "Reduce size after loss"
    reduction_factor = Column(Float, default=0.5)  # Reduce position size by this factor after loss
    wait_for_next_a_plus_setup_after_max_daily = Column(Boolean, default=True)
    
    # Position sizing
    base_position_size = Column(Integer, default=1)  # "Start with 1 lot"
    position_size_scaling = Column(Boolean, default=True)  # Scale position size based on setup quality
    
    # Grade-based position scaling
    a_plus_grade_multiplier = Column(Float, default=2.0)  # Multiply position size for A+ setups
    a_grade_multiplier = Column(Float, default=1.5)  # Multiply position size for A setups
    b_grade_multiplier = Column(Float, default=1.0)  # Standard position size for B setups
    c_grade_multiplier = Column(Float, default=0.5)  # Reduce position size for C setups
    
    # Trade frequency controls
    max_trades_per_day = Column(Integer, default=3)
    max_trades_per_week = Column(Integer, default=10)
    
    # Consistency first approach - Rikk's specific guidance
    target_consistent_points = Column(Integer, default=25)  # "See if u can consistently make 20-25 points with 1 lot"
    track_points_consistency = Column(Boolean, default=True)  # Track consistency of hitting point targets
    
    # Risk control for account growth
    scale_size_with_account_growth = Column(Boolean, default=True)  # "Slowly trading account will grow and u can trade with 2 lots"
    account_growth_threshold_percent = Column(Float, default=20.0)  # % growth needed before increasing lot size
    
    # Cost preview in INR - ensure trades costs are clearly understood
    show_cost_preview = Column(Boolean, default=True)  # Show INR cost before execution
    include_taxes_in_preview = Column(Boolean, default=True)
    include_brokerage_in_preview = Column(Boolean, default=True)

    # Correlation management
    track_asset_correlations = Column(Boolean, default=True)
    max_correlation_exposure = Column(Float, default=2.0)  # Maximum exposure to correlated assets
    correlation_lookback_period = Column(Integer, default=60)  # Days to calculate correlation
    high_correlation_threshold = Column(Float, default=0.7)  # Correlation coefficient threshold
    adjust_position_sizing_for_correlations = Column(Boolean, default=True)
    correlation_adjustment_method = Column(String(50), default="proportional_reduction")
    correlation_groups = Column(JSON, default=lambda: {
        "equity_indices": ["NIFTY", "BANKNIFTY", "FINNIFTY"],
        "currencies": ["USDINR", "EURINR", "GBPINR"],
        "commodities": ["CRUDEOIL", "GOLD", "SILVER"]
    })
    
    # Relationship
    strategy = relationship("Strategy", back_populates="risk_settings")


# Setup quality criteria
class SetupQualityCriteria(StrategyBaseModel):
    """
    Setup quality scoring criteria for trade grade determination.
    
    Implements Rikk's grading system (A+ to F) for determining
    which setups deserve larger position sizes and auto-execution.
    """
    __tablename__ = "setup_quality_criteria"
    
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False, unique=True)
    
    # Core quality factors with weights
    timeframe_alignment_weight = Column(Float, default=0.3)  # Weight for timeframe alignment
    proximity_to_key_level_weight = Column(Float, default=0.2)  # Weight for proximity to key level
    trend_strength_weight = Column(Float, default=0.2)  # Weight for trend strength
    entry_technique_weight = Column(Float, default=0.15)  # Weight for entry technique quality
    risk_reward_weight = Column(Float, default=0.15)  # Weight for risk/reward ratio
    
    # Grade thresholds (0-100 score)
    a_plus_min_score = Column(Float, default=90.0)  # Minimum score for A+ grade
    a_min_score = Column(Float, default=80.0)  # Minimum score for A grade
    b_min_score = Column(Float, default=70.0)  # Minimum score for B grade
    c_min_score = Column(Float, default=60.0)  # Minimum score for C grade
    d_min_score = Column(Float, default=50.0)  # Minimum score for D grade
    
    # A+ setup requirements - captures Rikk's specific criteria
    a_plus_requires_all_timeframes = Column(Boolean, default=True)  # All timeframes must align
    a_plus_requires_entry_near_ma = Column(Boolean, default=True)  # Entry must be near MA
    a_plus_requires_two_day_trend = Column(Boolean, default=True)  # 2-day trend required
    
    # Detailed A+ criteria
    a_plus_criteria = Column(JSON, default=lambda: {
        "all_timeframes_aligned": True,
        "price_near_ma": True,
        "clear_trend_direction": True,
        "proper_risk_reward": 3.0,  # Minimum R:R ratio
        "minimal_price_struggle": True,
        "clean_volume_profile": True,
        "institutional_behavior_aligned": True,
        "clear_bos_on_higher_tf": True
    })
    
    # Position sizing rules based on setup quality
    position_sizing_rules = Column(JSON, default=lambda: {
        "a_plus": {"lots": 2, "risk_percent": 1.0},
        "a": {"lots": 1, "risk_percent": 0.8},
        "b": {"lots": 0.5, "risk_percent": 0.5},
        "c": {"lots": 0.25, "risk_percent": 0.3},
        "d_and_below": {"lots": 0, "risk_percent": 0}
    })
    
    # Auto trade settings
    auto_trade_a_plus = Column(Boolean, default=True)  # Automatically trade A+ setups
    auto_trade_a = Column(Boolean, default=True)  # Automatically trade A setups
    auto_trade_b = Column(Boolean, default=False)  # Don't auto-trade B and below
    
    # Relationship
    strategy = relationship("Strategy", back_populates="quality_criteria")


# Vertical spread settings
class VerticalSpreadSettings(StrategyBaseModel):
    """
    Settings for vertical spread trading.
    
    Captures Rikk's approach to vertical spread trading as an alternative
    to directional trading, with less timing pressure.
    """
    __tablename__ = "vertical_spread_settings"
    
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False, unique=True)
    
    # Spread configuration
    use_vertical_spreads = Column(Boolean, default=False)
    preferred_spread_type = Column(Enum(SpreadType))
    otm_strike_distance = Column(Integer, default=1)  # Strikes away from ATM
    hold_till_expiry = Column(Boolean, default=False)
    
    # Capital requirements
    min_capital_required = Column(Integer, default=500000)  # 5 lakhs
    
    # Instruments to trade
    trade_nifty = Column(Boolean, default=True)
    trade_banknifty = Column(Boolean, default=True)
    trade_finnifty = Column(Boolean, default=False)
    trade_top10_stocks = Column(Boolean, default=False)
    trade_crude = Column(Boolean, default=True)
    
    # Weekly vs monthly options
    use_weekly_options = Column(Boolean, default=True)
    
    # Trend requirement for spreads
    require_trend_for_directional_spread = Column(Boolean, default=True)
    trend_confirmation_timeframes = Column(JSON, default=lambda: ["1d", "4h", "1h"])
    
    # Strategy specific spread settings - "Buy ATM, sell OTM" configuration
    bull_call_settings = Column(JSON, default=lambda: {
        "buy_strike": "ATM",
        "sell_strike": "ATM+1",
        "profit_target_percent": 30,
        "max_loss_percent": 70
    })
    
    bear_put_settings = Column(JSON, default=lambda: {
        "buy_strike": "ATM",
        "sell_strike": "ATM-1",
        "profit_target_percent": 30,
        "max_loss_percent": 70
    })
    
    # "Sell ATM, buy OTM" configuration
    bear_call_settings = Column(JSON, default=lambda: {
        "sell_strike": "ATM",
        "buy_strike": "ATM+1",
        "profit_target_percent": 80,  # Premium collection
        "max_loss_percent": 20
    })
    
    bull_put_settings = Column(JSON, default=lambda: {
        "sell_strike": "ATM",
        "buy_strike": "ATM-1",
        "profit_target_percent": 80,  # Premium collection
        "max_loss_percent": 20
    })
    
    # Timing settings for spread trades
    days_to_expiry_min = Column(Integer, default=3)
    days_to_expiry_max = Column(Integer, default=15)
    
    # Auto-trading for spreads - with INR cost preview
    auto_trade_spreads = Column(Boolean, default=False)
    require_confirmation = Column(Boolean, default=True)  # Require confirmation before execution
    show_cost_before_execution = Column(Boolean, default=True)  # Show INR cost before execution
    
    # Cost preview settings - all costs shown in INR
    include_brokerage_in_preview = Column(Boolean, default=True)
    include_taxes_in_preview = Column(Boolean, default=True)
    brokerage_per_lot = Column(Float, default=20.0)
    
    # Less timing pressure advantage - key benefit highlighted by Rikk
    timing_pressure_reduction = Column(Boolean, default=True)  # "No need to time market"
    
    # Relationship
    strategy = relationship("Strategy", back_populates="spread_settings")


# Meta-learning and continuous improvement
class MetaLearningSettings(StrategyBaseModel):
    """
    Settings for meta-learning and continuous improvement.
    
    Supports recording, review, and deliberate practice of trading decisions,
    implementing Rikk's emphasis on post-market analysis and learning.
    """
    __tablename__ = "meta_learning_settings"
    
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False, unique=True)
    
    # Recording settings
    record_trading_sessions = Column(Boolean, default=True)
    record_decision_points = Column(Boolean, default=True)
    
    # Review settings
    review_frequency = Column(String(20), default="daily")  # daily, weekly, monthly
    
    # Forward testing
    track_market_relationships = Column(Boolean, default=True)
    # Market regime detection and adaptation
    detect_regime_changes = Column(Boolean, default=True)
    adaptive_regime_parameters = Column(Boolean, default=True)
    regime_types = Column(JSON, default=lambda: [
        "high_volatility", "low_volatility", "trending", "ranging", 
        "high_liquidity", "low_liquidity", "news_driven"
    ])
    parameter_adjustments_by_regime = Column(JSON, default=lambda: {
        "high_volatility": {"position_size_multiplier": 0.7, "risk_percentage": 0.8},
        "low_volatility": {"position_size_multiplier": 1.2, "risk_percentage": 1.1},
        "trending": {"trailing_stop_multiplier": 1.3, "entry_filter_strictness": 0.8},
        "ranging": {"profit_target_multiplier": 0.8, "entry_filter_strictness": 1.2}
    })
    min_samples_for_regime_adaptation = Column(Integer, default=30)
    
    # Learning metrics and factor attribution
    track_improvement_metrics = Column(Boolean, default=True)
    improvement_categories = Column(JSON, default=lambda: [
        "entry_timing",
        "exit_execution", 
        "trend_identification",
        "risk_management",
        "psychological_factors"
    ])
    track_success_factors = Column(Boolean, default=True)
    success_factor_categories = Column(JSON, default=lambda: [
        "timeframe_alignment", "ma_relationship", "price_action_pattern", 
        "institutional_behavior", "volume_profile", "volatility_regime",
        "trend_quality", "entry_technique", "multi_timeframe_confirmation",
        "stop_placement", "time_of_day"
    ])
    factor_contribution_tracking = Column(Boolean, default=True)
    contribution_measurement_method = Column(String(50), default="weighted_ranking")  # "weighted_ranking", "regression", "decision_tree"
    min_trades_for_factor_analysis = Column(Integer, default=20)
    
    # Media feedback capabilities
    store_screenshots = Column(Boolean, default=True)
    screenshot_storage_path = Column(String(255), default="screenshots")
    
    # Text feedback and notes
    store_trading_notes = Column(Boolean, default=True)
    notes_format = Column(String(50), default="markdown")  # markdown, plain, html
    
    # Recording formats
    recording_formats = Column(JSON, default=lambda: ["text", "screenshot", "video"])
    
    # Learning feedback loop
    apply_feedback_to_strategy = Column(Boolean, default=True)
    feedback_review_frequency = Column(String(20), default="weekly")
    
    # Chart annotations
    store_chart_annotations = Column(Boolean, default=True)
    annotation_types = Column(JSON, default=lambda: ["entry", "exit", "missed_opportunity", "mistake"])
    
    # Post-market analysis - specific Rikk emphasis
    perform_post_market_analysis = Column(Boolean, default=True)  # "Do your analysis outside of market hours"
    
    # Synthetic market generation
    use_synthetic_markets = Column(Boolean, default=False)
    synthetic_generation_methods = Column(JSON, default=lambda: [
        "historical_remix", "statistical_simulation", "scenario_based", "monte_carlo"
    ])
    synthetic_scenarios_to_generate = Column(JSON, default=lambda: [
        "trending_periods", "choppy_markets", "volatility_expansion", "liquidity_crisis",
        "false_breakouts", "support_resistance_tests"
    ])
    synthetic_market_training_frequency = Column(String(20), default="monthly")
    base_scenarios_on_real_markets = Column(Boolean, default=True)

    # Time-of-day analysis
    track_time_of_day_performance = Column(Boolean, default=True)
    time_slots = Column(JSON, default=lambda: [
        "pre_market", "opening_hour", "mid_morning", "lunch_hour",
        "early_afternoon", "closing_hour", "post_market"
    ])
    market_session_tracking = Column(JSON, default=lambda: {
        "asia": True,
        "europe": True,
        "us": True,
        "overlap_periods": True
    })
    optimize_trading_hours = Column(Boolean, default=True)
    min_trades_per_time_slot = Column(Integer, default=15)  # Minimum trades for statistically valid analysis
    day_of_week_tracking = Column(Boolean, default=True)  # Track performance by day of week

    # Relationship
    strategy = relationship("Strategy", back_populates="meta_learning")


# Multi-timeframe confirmation settings
class MultiTimeframeConfirmationSettings(StrategyBaseModel):
    """
    Settings for multi-timeframe alignment and confirmation.
    
    Implements Rikk's hierarchical timeframe approach with explicit
    waiting periods for alignment between timeframes.
    """
    __tablename__ = "multi_timeframe_confirmation_settings"
    
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False, unique=True)
    
    # Alignment requirements
    require_all_timeframes_aligned = Column(Boolean, default=True)
    min_alignment_score = Column(Float, default=0.7)  # 0-1 score
    
    # Hierarchy definition
    primary_timeframe = Column(Enum(TimeframeValue), default=TimeframeValue.ONE_HOUR)
    confirmation_timeframe = Column(Enum(TimeframeValue), default=TimeframeValue.FIFTEEN_MIN)
    entry_timeframe = Column(Enum(TimeframeValue), default=TimeframeValue.FIVE_MIN)
    
    # Explicit 15-min wait requirement
    wait_for_15min_alignment = Column(Boolean, default=True)  # "15 mins for entry which should show u probable entry areas"
    min_15min_confirmation_bars = Column(Integer, default=2)
    
    # Two-day trend requirement
    require_two_day_trend = Column(Boolean, default=False)  # "2 green daily bars" requirement
    daily_trend_lookback = Column(Integer, default=2)
    
    # MA trend requirements
    require_ma_trend_alignment = Column(Boolean, default=True)
    ma_trend_threshold = Column(Float, default=0.0005)  # Minimum slope
    
    # BOS requirements
    require_bos_on_primary_tf = Column(Boolean, default=True)  # "Once u see BOS happening"
    wait_for_bos_confirmation = Column(Boolean, default=True)
    
    # Alignment weights for different timeframes
    timeframe_weights = Column(JSON, default=lambda: {
        "1d": 0.35,
        "4h": 0.25,
        "1h": 0.20,
        "15m": 0.15,
        "5m": 0.05
    })
    
    # Lower timeframe usage restriction - fundamental principle
    use_lower_tf_only_for_entry = Column(Boolean, default=True)  # "And 5 mins when price touches those areas"
    avoid_3min_noise = Column(Boolean, default=True)  # "3 mins is just noise"
    
    # Relationship
    strategy = relationship("Strategy", back_populates="multi_timeframe_settings")


# Feedback log for meta-learning
class TradeFeedback(StrategyBaseModel, TimestampMixin):
    """
    Trading feedback record for meta-learning.
    
    Stores various types of feedback for improving the strategy through
    deliberate practice and review.
    """
    __tablename__ = "trade_feedback"
    
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False)
    trade_id = Column(Integer, ForeignKey("trades.id"), nullable=True)  # Optional link to specific trade
    
    # Basic feedback information
    feedback_type = Column(Enum(FeedbackType), nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text)
    
    # File attachments
    file_path = Column(String(255))  # Path to screenshot, video, or chart
    file_type = Column(String(50))  # mime type or format
    
    # Tagging and categorization
    tags = Column(JSON, default=list)
    improvement_category = Column(String(100))  # Which aspect this feedback addresses
    
    # Feedback relevance
    applies_to_setup = Column(Boolean, default=True)  # Feedback applies to setup identification
    applies_to_entry = Column(Boolean, default=True)  # Feedback applies to entry execution
    applies_to_exit = Column(Boolean, default=True)  # Feedback applies to exit strategy
    applies_to_risk = Column(Boolean, default=True)  # Feedback applies to risk management
    
    # Lessons learned
    # Quantitative decision metrics
    pre_trade_conviction_level = Column(Float)  # 0-10 scale
    emotional_state_rating = Column(Integer)  # 1-5 scale (1=fear, 3=neutral, 5=overconfident)
    stress_level = Column(Integer)  # 1-5 scale
    trade_expectation = Column(JSON)  # Expected outcome
    reality_vs_expectation_gap = Column(Float)  # How much reality differed from expectation
    key_decision_factors = Column(JSON, default=list)  # List of factors that influenced decision
    decision_time_taken = Column(Integer)  # Seconds taken to make decision
    distraction_level = Column(Integer)  # 1-5 scale
    focus_quality = Column(Integer)  # 1-5 scale
    
    # Lessons learned
    lessons_learned = Column(Text)
    action_items = Column(Text)
    objectivity_rating = Column(Integer)  # Self-assessment of objectivity
    impulsivity_factor = Column(Integer)  # Was the trade impulsive? (1-5)
    psychological_pattern_identified = Column(String(100))  # Pattern in decision making
    
    # Application status
    has_been_applied = Column(Boolean, default=False)  # Whether feedback has been incorporated
    applied_date = Column(DateTime)
    applied_to_version_id = Column(Integer)  # Which strategy version incorporated this feedback
    
    # Relationships
    strategy = relationship("Strategy", back_populates="feedback")
    trade = relationship("Trade", back_populates="feedback")


# Core strategy model
class Strategy(StrategyBaseModel, UserRelationMixin, 
               AuditMixin, SoftDeleteMixin, VersionedMixin, StatusMixin):
    """
    Comprehensive trading strategy model incorporating all of Rikk's principles.
    
    This model serves as the central entity for trading strategies, connecting
    to specialized setting models for different aspects of Rikk's approach.
    """
    __tablename__ = "strategies"
    
    # Basic strategy information
    name = Column(String(255), nullable=False)
    description = Column(Text)
    type = Column(Enum(StrategyType), nullable=False, default=StrategyType.CUSTOM)
    
    # Strategy configuration
    configuration = Column(JSON, nullable=False, default={})
    parameters = Column(JSON, nullable=False, default={})
    validation_rules = Column(JSON, default={})
    
    # Execution settings
    is_active = Column(Boolean, default=False)
    execution_schedule = Column(String(100))  # Cron-like schedule
    
    # Security and access control
    is_public = Column(Boolean, default=False)
    access_level = Column(String(50), default="private")
    
    # Performance metrics (summary)
    win_rate = Column(Float)
    profit_factor = Column(Float)
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    max_drawdown = Column(Float)
    
    # Performance metrics in INR - for clear monetary understanding
    total_profit_inr = Column(Float)
    avg_win_inr = Column(Float)
    avg_loss_inr = Column(Float)
    largest_win_inr = Column(Float)
    largest_loss_inr = Column(Float)
    
    # Versioning
    parent_version_id = Column(Integer, ForeignKey('strategies.id'))
    
    # Multi-timeframe alignment requirements - fundamental principle
    require_timeframe_alignment = Column(Boolean, default=True)  # "Always align with higher timeframes"
    minimum_alignment_score = Column(Float, default=0.7)  # 0-1 score for alignment strength
    
    # Notes field for additional strategy context
    notes = Column(Text)  # For any additional context or observations
    
    # Relationships
    categories = relationship("StrategyCategory", secondary=strategy_category_association, 
                              back_populates="strategies")
    signals = relationship("Signal", back_populates="strategy")
    trades = relationship("Trade", back_populates="strategy")  
    parent_version = relationship("Strategy", remote_side=[id], 
                                 backref="child_versions")
    timeframes = relationship("StrategyTimeframe", back_populates="strategy",
                             cascade="all, delete-orphan")
    institutional_settings = relationship("InstitutionalBehaviorSettings", uselist=False,
                                         back_populates="strategy", cascade="all, delete-orphan")
    entry_exit_settings = relationship("EntryExitSettings", uselist=False,
                                      back_populates="strategy", cascade="all, delete-orphan")
    market_state_settings = relationship("MarketStateSettings", uselist=False,
                                        back_populates="strategy", cascade="all, delete-orphan")
    previous_day_reference = relationship("PreviousDayReference", uselist=False,
                                         back_populates="strategy", cascade="all, delete-orphan")
    risk_settings = relationship("RiskManagementSettings", uselist=False,
                                back_populates="strategy", cascade="all, delete-orphan")
    quality_criteria = relationship("SetupQualityCriteria", uselist=False,
                                   back_populates="strategy", cascade="all, delete-orphan")
    spread_settings = relationship("VerticalSpreadSettings", uselist=False,
                                  back_populates="strategy", cascade="all, delete-orphan")
    meta_learning = relationship("MetaLearningSettings", uselist=False,
                                back_populates="strategy", cascade="all, delete-orphan")
    multi_timeframe_settings = relationship("MultiTimeframeConfirmationSettings", uselist=False,
                                           back_populates="strategy", cascade="all, delete-orphan")
    feedback = relationship("TradeFeedback", back_populates="strategy")
    backtests = relationship("StrategyBacktest", back_populates="strategy")
    
    # Indexes for common queries
    __table_args__ = (
        Index('ix_strategies_type', 'type'),
        Index('ix_strategies_active', 'is_active'),
    )
    
    # We use composition instead of inheritance for serialization
    def to_dict(self, include_relationships=False, exclude=None):
        """
        Convert strategy to dictionary for API responses.
        
        Args:
            include_relationships: Whether to include related entities
            exclude: List of fields to exclude
            
        Returns:
            Dictionary representation of the strategy
        """
        exclude = exclude or []
        result = {}
        
        # Convert model attributes to dict (similar to SerializableMixin's to_dict)
        for column in self.__table__.columns:
            if column.name in exclude:
                continue
                
            value = getattr(self, column.name)
            
            # Handle special types
            if isinstance(value, (datetime, datetime.date)):
                value = value.isoformat()
            elif isinstance(value, enum.Enum):
                value = value.value
                
            result[column.name] = value
            
        # Include relationship data if requested
        if include_relationships:
            # Include categories
            result['categories'] = [category.name for category in self.categories]
            
            # Include timeframes
            result['timeframes'] = [
                {
                    'name': tf.name,
                    'value': tf.value.value,
                    'importance': tf.importance.value,
                    'order': tf.order,
                    'ma_type': tf.ma_type.value,
                    'ma_period_primary': tf.ma_period_primary,  # 21 MA not 20
                    'ma_period_secondary': tf.ma_period_secondary  # 200 MA explicit
                } for tf in sorted(self.timeframes, key=lambda x: x.order)
            ] if self.timeframes else []
            
            # Include institutional settings
            if self.institutional_settings:
                result['institutional_settings'] = {
                    'detect_accumulation': self.institutional_settings.detect_accumulation,
                    'detect_liquidity_grabs': self.institutional_settings.detect_liquidity_grabs,
                    'detect_stop_hunts': self.institutional_settings.detect_stop_hunts,
                    'detect_volume_patterns': self.institutional_settings.detect_volume_patterns,
                    'detect_price_patterns': self.institutional_settings.detect_price_patterns,
                    'detect_choch': self.institutional_settings.detect_choch,
                    'detect_bos': self.institutional_settings.detect_bos,
                    'wait_for_institutional_footprints': self.institutional_settings.wait_for_institutional_footprints,
                    'wait_for_institutional_fight': self.institutional_settings.wait_for_institutional_fight
                }
                
            # Include entry/exit settings
            if self.entry_exit_settings:
                result['entry_exit_settings'] = {
                    'direction': self.entry_exit_settings.direction.value,
                    'primary_entry_technique': self.entry_exit_settings.primary_entry_technique.value,
                    'require_candle_close_confirmation': self.entry_exit_settings.require_candle_close_confirmation,
                    'trailing_stop_method': self.entry_exit_settings.trailing_stop_method.value,
                    'profit_target_method': self.entry_exit_settings.profit_target_method.value,
                    'profit_target_points': self.entry_exit_settings.profit_target_points,  # 20-25 points
                    'target_trend_phase': self.entry_exit_settings.target_trend_phase.value,  # Middle phase
                    'close_if_against_higher_tf': self.entry_exit_settings.close_if_against_higher_tf,
                    'green_bar_sl_placement': self.entry_exit_settings.green_bar_sl_placement,  # Below bar
                    'red_bar_sl_placement': self.entry_exit_settings.red_bar_sl_placement,  # Above bar
                    'auto_trade_regular': self.entry_exit_settings.auto_trade_regular,
                    'show_cost_preview': self.entry_exit_settings.show_cost_preview  # INR preview
                }
                
            # Include market state settings
            if self.market_state_settings:
                result['market_state_settings'] = {
                    'required_market_state': self.market_state_settings.required_market_state.value,
                    'avoid_creeper_moves': self.market_state_settings.avoid_creeper_moves,
                    'require_two_day_trend': self.market_state_settings.require_two_day_trend,
                    'detect_price_indicator_divergence': self.market_state_settings.detect_price_indicator_divergence,
                    'price_action_overrides_indicators': self.market_state_settings.price_action_overrides_indicators,
                    'prefer_railroad_trends': self.market_state_settings.prefer_railroad_trends,
                    'detect_break_of_structure': self.market_state_settings.detect_break_of_structure,
                    'detect_price_ma_struggle': self.market_state_settings.detect_price_ma_struggle
                }
                
            # Include multi-timeframe settings
            if self.multi_timeframe_settings:
                result['multi_timeframe_settings'] = {
                    'require_all_timeframes_aligned': self.multi_timeframe_settings.require_all_timeframes_aligned,
                    'primary_timeframe': self.multi_timeframe_settings.primary_timeframe.value,
                    'confirmation_timeframe': self.multi_timeframe_settings.confirmation_timeframe.value,
                    'entry_timeframe': self.multi_timeframe_settings.entry_timeframe.value,
                    'wait_for_15min_alignment': self.multi_timeframe_settings.wait_for_15min_alignment,
                    'require_two_day_trend': self.multi_timeframe_settings.require_two_day_trend,
                    'use_lower_tf_only_for_entry': self.multi_timeframe_settings.use_lower_tf_only_for_entry,
                    'avoid_3min_noise': self.multi_timeframe_settings.avoid_3min_noise
                }
                
            # Include quality criteria
            if self.quality_criteria:
                result['quality_criteria'] = {
                    'a_plus_min_score': self.quality_criteria.a_plus_min_score,
                    'a_plus_criteria': self.quality_criteria.a_plus_criteria,
                    'position_sizing_rules': self.quality_criteria.position_sizing_rules,
                    'auto_trade_a_plus': self.quality_criteria.auto_trade_a_plus
                }
                
            # Include risk settings
            if self.risk_settings:
                result['risk_settings'] = {
                    'max_risk_per_trade_percent': self.risk_settings.max_risk_per_trade_percent,
                    'max_risk_per_trade_inr': self.risk_settings.max_risk_per_trade_inr,
                    'daily_drawdown_threshold': self.risk_settings.daily_drawdown_threshold,  # 4%
                    'weekly_drawdown_threshold': self.risk_settings.weekly_drawdown_threshold,  # 8%
                    'position_size_scaling': self.risk_settings.position_size_scaling,
                    'target_consistent_points': self.risk_settings.target_consistent_points,  # 20-25 points
                    'show_cost_preview': self.risk_settings.show_cost_preview  # INR preview
                }
                
            # Include spread settings
            if self.spread_settings:
                result['spread_settings'] = {
                    'use_vertical_spreads': self.spread_settings.use_vertical_spreads,
                    'preferred_spread_type': self.spread_settings.preferred_spread_type.value if self.spread_settings.preferred_spread_type else None,
                    'min_capital_required': self.spread_settings.min_capital_required,
                    'auto_trade_spreads': self.spread_settings.auto_trade_spreads,
                    'show_cost_before_execution': self.spread_settings.show_cost_before_execution,  # INR preview
                    'timing_pressure_reduction': self.spread_settings.timing_pressure_reduction  # Key benefit
                }
                
            # Include meta-learning settings
            if self.meta_learning:
                result['meta_learning'] = {
                    'record_trading_sessions': self.meta_learning.record_trading_sessions,
                    'track_market_relationships': self.meta_learning.track_market_relationships,
                    'improvement_categories': self.meta_learning.improvement_categories,
                    'perform_post_market_analysis': self.meta_learning.perform_post_market_analysis,  # Post-market analysis
                    'store_screenshots': self.meta_learning.store_screenshots,
                    'store_trading_notes': self.meta_learning.store_trading_notes
                }
            
            # Include signal count
            result['signal_count'] = len(self.signals) if self.signals else 0
            
            # Include version information
            result['has_parent'] = self.parent_version_id is not None
            result['has_children'] = len(self.child_versions) > 0 if hasattr(self, 'child_versions') else False
            
            # Include performance metrics in INR
            result['performance_inr'] = {
                'total_profit_inr': self.total_profit_inr,
                'avg_win_inr': self.avg_win_inr,
                'avg_loss_inr': self.avg_loss_inr,
                'largest_win_inr': self.largest_win_inr,
                'largest_loss_inr': self.largest_loss_inr
            }
        
        return result
    
    def to_json(self, include_relationships=False, exclude=None):
        """Convert model to JSON string."""
        return json.dumps(self.to_dict(include_relationships, exclude))
    
    @validates('configuration', 'parameters', 'validation_rules')
    def validate_json(self, key, value):
        """Validate that JSON fields contain valid JSON data."""
        if isinstance(value, str):
            try:
                # If it's a string, try to parse it
                return json.loads(value)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON in {key}")
        # If it's already a dict, return as is
        return value
    
    def get_parameter(self, param_name: str, default: Any = None) -> Any:
        """
        Get a strategy parameter value.
        
        Args:
            param_name: Name of the parameter to retrieve
            default: Default value if parameter doesn't exist
            
        Returns:
            Parameter value or default
        """
        if not self.parameters:
            return default
        return self.parameters.get(param_name, default)
    
    def update_parameter(self, param_name: str, value: Any) -> None:
        """
        Update a single strategy parameter.
        
        Args:
            param_name: Name of the parameter to update
            value: New parameter value
        """
        if not self.parameters:
            self.parameters = {}
        self.parameters[param_name] = value
        self.version += 1  # Increment version on parameter changes
    
    def validate_parameters(self) -> List[str]:
        """
        Validate all parameters against validation rules.
        
        Returns:
            List of validation error messages (empty if all valid)
        """
        errors = []
        if not self.validation_rules:
            return errors
            
        for param_name, rules in self.validation_rules.items():
            if param_name not in self.parameters:
                if rules.get("required", False):
                    errors.append(f"Required parameter '{param_name}' is missing")
                continue
                
            value = self.parameters[param_name]
            
            # Type validation
            expected_type = rules.get("type")
            if expected_type:
                if expected_type == "number" and not isinstance(value, (int, float)):
                    errors.append(f"Parameter '{param_name}' must be a number")
                elif expected_type == "string" and not isinstance(value, str):
                    errors.append(f"Parameter '{param_name}' must be a string")
                elif expected_type == "boolean" and not isinstance(value, bool):
                    errors.append(f"Parameter '{param_name}' must be a boolean")
            
            # Range validation for numeric parameters
            if isinstance(value, (int, float)):
                min_value = rules.get("min")
                max_value = rules.get("max")
                
                if min_value is not None and value < min_value:
                    errors.append(f"Parameter '{param_name}' cannot be less than {min_value}")
                if max_value is not None and value > max_value:
                    errors.append(f"Parameter '{param_name}' cannot be greater than {max_value}")
            
            # Length validation for string parameters
            if isinstance(value, str):
                min_length = rules.get("minLength")
                max_length = rules.get("maxLength")
                
                if min_length is not None and len(value) < min_length:
                    errors.append(f"Parameter '{param_name}' must be at least {min_length} characters")
                if max_length is not None and len(value) > max_length:
                    errors.append(f"Parameter '{param_name}' cannot exceed {max_length} characters")
        
        return errors
    
    def create_new_version(self) -> 'Strategy':
        """
        Create a new version of this strategy.
        
        Returns:
            New Strategy instance as the next version
        """
        # Create a copy of the current strategy as a new version
        new_version = Strategy(
            name=self.name,
            description=self.description,
            type=self.type,
            configuration=self.configuration.copy() if self.configuration else {},
            parameters=self.parameters.copy() if self.parameters else {},
            validation_rules=self.validation_rules.copy() if self.validation_rules else {},
            require_timeframe_alignment=self.require_timeframe_alignment,
            minimum_alignment_score=self.minimum_alignment_score,
            user_id=self.user_id,
            created_by_id=self.created_by_id,
            parent_version_id=self.id,
            version=self.version + 1
        )
        
        return new_version
    
    def update_performance_metrics(self, win_rate: float = None, profit_factor: float = None,
                                 sharpe_ratio: float = None, sortino_ratio: float = None,
                                 max_drawdown: float = None, total_profit_inr: float = None,
                                 avg_win_inr: float = None, avg_loss_inr: float = None,
                                 largest_win_inr: float = None, largest_loss_inr: float = None) -> None:
        """
        Update strategy performance metrics, including INR amounts.
        
        Args:
            win_rate: Percentage of winning trades
            profit_factor: Ratio of gross profits to gross losses
            sharpe_ratio: Risk-adjusted return metric
            sortino_ratio: Downside risk-adjusted return metric
            max_drawdown: Maximum peak-to-trough decline
            total_profit_inr: Total profit in INR
            avg_win_inr: Average winning trade in INR
            avg_loss_inr: Average losing trade in INR
            largest_win_inr: Largest winning trade in INR
            largest_loss_inr: Largest losing trade in INR
        """
        if win_rate is not None:
            self.win_rate = win_rate
        if profit_factor is not None:
            self.profit_factor = profit_factor
        if sharpe_ratio is not None:
            self.sharpe_ratio = sharpe_ratio
        if sortino_ratio is not None:
            self.sortino_ratio = sortino_ratio
        if max_drawdown is not None:
            self.max_drawdown = max_drawdown
        if total_profit_inr is not None:
            self.total_profit_inr = total_profit_inr
        if avg_win_inr is not None:
            self.avg_win_inr = avg_win_inr
        if avg_loss_inr is not None:
            self.avg_loss_inr = avg_loss_inr
        if largest_win_inr is not None:
            self.largest_win_inr = largest_win_inr
        if largest_loss_inr is not None:
            self.largest_loss_inr = largest_loss_inr
    
    def check_timeframe_alignment(self) -> float:
        """
        Check alignment between timeframes based on Rikk's principles.
        
        Returns:
            Alignment score between 0 and 1
        """
        if not self.multi_timeframe_settings or not self.timeframes:
            return 0.0
            
        # Placeholder for real implementation
        # This would use the weights from multi_timeframe_settings
        # and check alignments of MAs, price action, etc.
        return 0.8  # Placeholder value
    
    def evaluate_setup_quality(self, market_data: Dict) -> Dict:
        """
        Evaluate the quality of a potential trading setup.
        
        Implements Rikk's A+ to F grading system based on
        timeframe alignment, trend quality, etc.
        
        Args:
            market_data: Dictionary with current market data
            
        Returns:
            Dict with grade and score
        """
        if not self.quality_criteria:
            return {"grade": "F", "score": 0, "position_size": 0}
            
        # Placeholder for real implementation
        # In real code, this would:
        # 1. Check timeframe alignment
        # 2. Validate entry technique quality
        # 3. Check distance from MA
        # 4. Validate trend strength
        # 5. Calculate risk/reward ratio
        
        # Placeholder score
        score = 85
        
        # Determine grade based on score
        grade = "F"
        if score >= self.quality_criteria.a_plus_min_score:
            grade = "A_PLUS"
        elif score >= self.quality_criteria.a_min_score:
            grade = "A"
        elif score >= self.quality_criteria.b_min_score:
            grade = "B"
        elif score >= self.quality_criteria.c_min_score:
            grade = "C"
        elif score >= self.quality_criteria.d_min_score:
            grade = "D"
            
        # Determine position size based on grade
        position_size = 0
        if self.quality_criteria.position_sizing_rules:
            if grade == "A_PLUS" and "a_plus" in self.quality_criteria.position_sizing_rules:
                position_size = self.quality_criteria.position_sizing_rules["a_plus"]["lots"]
            elif grade == "A" and "a" in self.quality_criteria.position_sizing_rules:
                position_size = self.quality_criteria.position_sizing_rules["a"]["lots"]
            elif grade == "B" and "b" in self.quality_criteria.position_sizing_rules:
                position_size = self.quality_criteria.position_sizing_rules["b"]["lots"]
            elif grade == "C" and "c" in self.quality_criteria.position_sizing_rules:
                position_size = self.quality_criteria.position_sizing_rules["c"]["lots"]
                
        return {
            "grade": grade,
            "score": score,
            "position_size": position_size
        }
    
    def check_market_state(self, market_data: Dict) -> bool:
        """
        Check if current market state meets strategy requirements.
        
        Implements Rikk's focus on trading in the right market conditions,
        avoiding creeper moves, etc.
        
        Args:
            market_data: Dictionary with current market data
            
        Returns:
            True if market state is acceptable for this strategy
        """
        if not self.market_state_settings:
            return True
            
        # Skip creeper moves if configured
        if (self.market_state_settings.avoid_creeper_moves and
            market_data.get("is_creeper_move", False)):
            return False
            
        # Check required market state
        if (self.market_state_settings.required_market_state != MarketStateRequirement.ANY and
            market_data.get("market_state") != self.market_state_settings.required_market_state.value):
            return False
            
        # Check two-day trend if required
        if (self.market_state_settings.require_two_day_trend and
            not market_data.get("has_two_day_trend", False)):
            return False
            
        # Prefer railroad trends if configured
        if (self.market_state_settings.prefer_railroad_trends and
            not market_data.get("is_railroad_trend", False) and
            market_data.get("is_creeper_move", False)):
            return False
            
        # Check for price struggling near MA
        if (self.market_state_settings.detect_price_ma_struggle and
            market_data.get("price_struggling_near_ma", False)):
            return False
            
        # Check for price vs indicator divergence (price action override)
        if (self.market_state_settings.price_action_overrides_indicators and
            market_data.get("price_indicator_divergence", False)):
            # Use price action signal instead of indicator signal
            # For now, just logged as a consideration
            pass
            
        # Wait for institutional fight to end
        if (hasattr(self.institutional_settings, 'wait_for_institutional_fight') and
            self.institutional_settings.wait_for_institutional_fight and
            market_data.get("institutional_fight_in_progress", False)):
            return False
            
        return True
    
    def check_institutional_footprints(self, market_data: Dict) -> bool:
        """
        Check for institutional footprints in the market.
        
        Implements Rikk's principle: "Let footprint of big traders be seen"
        
        Args:
            market_data: Dictionary with current market data
            
        Returns:
            True if institutional footprints are detected in the expected direction
        """
        if not self.institutional_settings:
            return True
            
        # Skip if not configured to wait for institutional footprints
        if not self.institutional_settings.wait_for_institutional_footprints:
            return True
            
        # Check for accumulation in expected direction
        if (self.institutional_settings.detect_accumulation and
            not market_data.get("accumulation_detected", False)):
            return False
            
        # Check for institutional BOS pattern
        if (self.institutional_settings.detect_bos and
            not market_data.get("bos_detected", False)):
            return False
            
        return True
    
    def calculate_trade_cost_preview(self, instrument: str, quantity: int, price: float) -> Dict:
        """
        Calculate trade cost preview in INR.
        
        Implements Rikk's requirement to see trade costs in INR before execution.
        
        Args:
            instrument: Trading instrument (e.g., "NIFTY", "BANKNIFTY")
            quantity: Number of lots/contracts
            price: Current price per unit
            
        Returns:
            Dictionary with cost breakdown
        """
        # Get settings based on whether this is a spread or regular trade
        if self.spread_settings and self.spread_settings.use_vertical_spreads:
            settings = self.spread_settings
        else:
            settings = self.entry_exit_settings
            
        # Skip if cost preview is disabled
        if not hasattr(settings, 'show_cost_preview') or not settings.show_cost_preview:
            return {}
            
        # Calculate trade value
        trade_value = price * quantity
        
        # Initialize cost components
        brokerage = 0.0
        taxes = 0.0
        
        # Add brokerage if configured
        if hasattr(settings, 'include_brokerage_in_preview') and settings.include_brokerage_in_preview:
            brokerage = settings.brokerage_per_lot * quantity
            
        # Add taxes if configured
        if hasattr(settings, 'include_taxes_in_preview') and settings.include_taxes_in_preview:
            # Example tax calculation (would vary by market/country)
            if hasattr(settings, 'exchange_transaction_tax'):
                tax_rate = settings.exchange_transaction_tax
            else:
                tax_rate = 0.0005  # Default: 0.05%
                
            taxes = trade_value * tax_rate
            
        # Calculate total cost
        total_cost = brokerage + taxes
        
        # Return cost breakdown
        return {
            "instrument": instrument,
            "quantity": quantity,
            "price": price,
            "trade_value": trade_value,
            "brokerage": brokerage,
            "taxes": taxes,
            "total_cost": total_cost,
            "currency": "INR"
        }


class StrategyCategory(StrategyBaseModel, TimestampMixin):
    """
    Strategy category model for organizing strategies by type.
    
    Categories allow grouping strategies by their approach or methodology
    to facilitate discovery and management.
    """
    __tablename__ = "strategy_categories"
    
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text)
    
    # Relationships
    strategies = relationship("Strategy", secondary=strategy_category_association, 
                             back_populates="categories")


class StrategyBacktest(StrategyBaseModel, TimestampMixin, UserRelationMixin):
    """
    Strategy backtest results model.
    
    Stores the results of strategy backtests, including performance metrics,
    trade history, and configuration settings used for the test.
    """
    __tablename__ = "strategy_backtests"
    
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Backtest settings
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    initial_capital = Column(Float, nullable=False)
    parameters = Column(JSON)
    
    # Performance metrics
    total_return = Column(Float)
    annualized_return = Column(Float)
    win_rate = Column(Float)
    profit_factor = Column(Float)
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    max_drawdown = Column(Float)
    max_drawdown_duration = Column(Integer)  # Days
    total_trades = Column(Integer)
    
    # Setup quality metrics
    a_plus_setups = Column(Integer, default=0)  # Count of A+ setups identified
    a_setups = Column(Integer, default=0)  # Count of A setups identified
    b_setups = Column(Integer, default=0)  # Count of B setups identified
    c_setups = Column(Integer, default=0)  # Count of C setups identified
    lower_grade_setups = Column(Integer, default=0)  # Count of D and F setups
    
    # INR metrics - for clear monetary understanding
    total_profit_inr = Column(Float)
    avg_win_inr = Column(Float)
    avg_loss_inr = Column(Float)
    largest_win_inr = Column(Float)
    largest_loss_inr = Column(Float)
    
    # Detailed results
    equity_curve = Column(JSON)  # Time series of portfolio values
    trade_history = Column(JSON)  # List of all trades
    monthly_returns = Column(JSON)  # Monthly return percentages
    
    # Timeframe alignment metrics
    timeframe_alignment_score = Column(Float)  # Average alignment score
    
    # Trend quality metrics
    railroad_trend_count = Column(Integer)  # Count of railroad trends identified
    creeper_move_count = Column(Integer)  # Count of creeper moves identified
    
    # Institutional behavior detection
    institutional_alignment_score = Column(Float)  # Success rate of institutional behavior detection
    
    # Relationship
    strategy = relationship("Strategy", back_populates="backtests")
    
    def calculate_metrics(self, trades: List[Dict], equity_curve: List[Dict]) -> None:
        """
        Calculate performance metrics from trade history and equity curve.
        
        Args:
            trades: List of trade dictionaries
            equity_curve: List of equity values over time
        """
        if not trades or not equity_curve:
            return
            
        # Count total trades
        self.total_trades = len(trades)
        
        # Calculate win rate
        winning_trades = sum(1 for trade in trades if trade.get('profit', 0) > 0)
        self.win_rate = winning_trades / self.total_trades if self.total_trades > 0 else 0
        
        # Calculate profit factor
        gross_profit = sum(trade.get('profit', 0) for trade in trades if trade.get('profit', 0) > 0)
        gross_loss = abs(sum(trade.get('profit', 0) for trade in trades if trade.get('profit', 0) < 0))
        self.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate returns
        initial_equity = equity_curve[0]['equity'] if equity_curve else self.initial_capital
        final_equity = equity_curve[-1]['equity'] if equity_curve else self.initial_capital
        
        self.total_return = (final_equity - initial_equity) / initial_equity
        
        # Calculate INR metrics
        self.total_profit_inr = final_equity - initial_equity
        
        # Calculate average win/loss in INR
        winning_amounts = [trade.get('profit', 0) for trade in trades if trade.get('profit', 0) > 0]
        losing_amounts = [trade.get('profit', 0) for trade in trades if trade.get('profit', 0) < 0]
        
        self.avg_win_inr = sum(winning_amounts) / len(winning_amounts) if winning_amounts else 0
        self.avg_loss_inr = sum(losing_amounts) / len(losing_amounts) if losing_amounts else 0
        
        # Calculate largest win/loss in INR
        self.largest_win_inr = max(winning_amounts) if winning_amounts else 0
        self.largest_loss_inr = min(losing_amounts) if losing_amounts else 0
        
        # Store the complete equity curve and trade history
        self.equity_curve = equity_curve
        self.trade_history = trades


class Signal(StrategyBaseModel, TimestampMixin):
    """
    Trading signal model.
    
    Represents a trading signal generated by a strategy, including
    entry, exit, and risk management parameters.
    """
    __tablename__ = "signals"
    
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False)
    instrument = Column(String(50), nullable=False)  # Trading instrument
    
    # Signal properties (bidirectional support)
    direction = Column(Enum(Direction), nullable=False)
    signal_type = Column(String(50))  # e.g., "breakout", "reversal", "pullback"
    
    # Entry parameters
    entry_price = Column(Float)
    entry_time = Column(DateTime)
    entry_timeframe = Column(Enum(TimeframeValue))
    entry_technique = Column(Enum(EntryTechnique))
    
    # Exit parameters
    take_profit_price = Column(Float)
    stop_loss_price = Column(Float)
    trailing_stop = Column(Boolean, default=False)
    
    # Risk parameters
    position_size = Column(Integer)  # Number of lots/contracts
    risk_reward_ratio = Column(Float)
    risk_amount = Column(Float)  # Amount at risk in INR
    
    # Signal quality - based on Rikk's grading system
    setup_quality = Column(Enum(SetupQualityGrade))
    setup_score = Column(Float)  # 0-100 score
    confidence = Column(Float)  # 0-1 confidence score
    
    # Market context
    market_state = Column(Enum(MarketStateRequirement))
    trend_phase = Column(Enum(TrendPhase))
    
    # Signal status
    is_active = Column(Boolean, default=True)
    is_executed = Column(Boolean, default=False)
    execution_time = Column(DateTime)
    
    # Timeframe alignment
    timeframe_alignment_score = Column(Float)
    primary_timeframe_aligned = Column(Boolean)
    
    # Institutional behavior
    institutional_footprint_detected = Column(Boolean)
    bos_detected = Column(Boolean)
    
    # Price action patterns
    price_action_pattern = Column(String(50))
    
    # Trade result (after execution)
    trade_result = Column(Float)  # Profit/loss in points
    trade_result_inr = Column(Float)  # Profit/loss in INR
    
    # Spread trade specifics
    is_spread_trade = Column(Boolean, default=False)
    spread_type = Column(Enum(SpreadType))
    
    # Cost preview - in INR
    brokerage_cost = Column(Float)
    tax_cost = Column(Float)
    total_cost = Column(Float)
    
    # Relationships
    strategy = relationship("Strategy", back_populates="signals")
    trades = relationship("Trade", back_populates="signal")
    
    def calculate_cost_preview(self) -> Dict:
        """
        Calculate cost preview for this signal.
        
        Returns:
            Dictionary with cost breakdown in INR
        """
        if self.strategy:
            return self.strategy.calculate_trade_cost_preview(
                self.instrument, 
                self.position_size, 
                self.entry_price
            )
        return {}


class Trade(StrategyBaseModel, TimestampMixin, UserRelationMixin):
    """
    Trade execution model.
    
    Represents an executed trade from a signal, including entry, exit,
    and performance details with full INR cost tracking.
    """
    __tablename__ = "trades"
    
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False)
    signal_id = Column(Integer, ForeignKey("signals.id"))
    
    # Trade identification
    instrument = Column(String(50), nullable=False)
    direction = Column(Enum(Direction), nullable=False)
    
    # Entry details
    entry_price = Column(Float, nullable=False)
    entry_time = Column(DateTime, nullable=False)
    position_size = Column(Integer, nullable=False)
    
    # Exit details
    exit_price = Column(Float)
    exit_time = Column(DateTime)
    exit_reason = Column(String(50))  # e.g., "target_reached", "stop_loss", "trailing_stop"
    
    # Trade result
    profit_loss_points = Column(Float)  # Profit/loss in points
    profit_loss_inr = Column(Float)  # Profit/loss in INR
    
    # Trading costs - all in INR
    commission = Column(Float)
    taxes = Column(Float)
    slippage = Column(Float)
    total_costs = Column(Float)
    
    # Initial risk assessment
    initial_risk_points = Column(Float)
    initial_risk_inr = Column(Float)
    initial_risk_percent = Column(Float)
    risk_reward_planned = Column(Float)
    
    # Actual performance
    actual_risk_reward = Column(Float)
    holding_period_minutes = Column(Integer)
    
    # Setup quality
    setup_quality = Column(Enum(SetupQualityGrade))
    setup_score = Column(Float)
    
    # Spread trade details
    is_spread_trade = Column(Boolean, default=False)
    spread_type = Column(Enum(SpreadType))
    
    # Relationships
    strategy = relationship("Strategy", back_populates="trades")
    signal = relationship("Signal", back_populates="trades")
    feedback = relationship("TradeFeedback", back_populates="trade")
    
    def calculate_performance(self) -> None:
        """
        Calculate performance metrics for this trade.
        """
        if not self.exit_price or not self.exit_time:
            return
            
        # Calculate profit/loss in points (bidirectional)
        point_multiplier = -1 if self.direction == Direction.SHORT else 1
        self.profit_loss_points = point_multiplier * (self.exit_price - self.entry_price)
        
        # Calculate profit/loss in INR (simplified, would depend on contract specifications)
        # This is a placeholder - real implementation would use proper lot size and tick value
        lot_size = 50  # Example lot size
        self.profit_loss_inr = self.profit_loss_points * lot_size * self.position_size
        
        # Calculate holding period
        if self.entry_time and self.exit_time:
            delta = self.exit_time - self.entry_time
            self.holding_period_minutes = delta.total_seconds() / 60
            
        # Calculate actual risk-reward ratio
        if self.initial_risk_points and self.initial_risk_points > 0:
            self.actual_risk_reward = abs(self.profit_loss_points / self.initial_risk_points)
            
        # Calculate total costs
        self.total_costs = (self.commission or 0) + (self.taxes or 0) + (self.slippage or 0)


# Set up table structure and models for one-to-many and many-to-many relationships
def initialize_models():
    """Set up database tables and relationships."""
    # This function would be called during application initialization
    # to ensure all tables are created with proper relationships
    pass