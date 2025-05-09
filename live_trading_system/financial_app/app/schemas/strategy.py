"""
Schema definitions for trading strategy components.
"""
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field

class TimeframeAnalysisResult(BaseModel):
    """Result of multi-timeframe analysis."""
    aligned: bool
    alignment_score: float
    timeframe_results: Dict[str, Any]
    primary_direction: str
    require_all_aligned: bool
    min_alignment_score: float
    sufficient_alignment: bool

class MarketStateAnalysis(BaseModel):
    """Analysis of market state for strategy execution."""
    market_state: str
    trend_phase: str
    is_railroad_trend: bool
    is_creeper_move: bool
    has_two_day_trend: bool
    trend_direction: str
    price_indicator_divergence: bool
    price_struggling_near_ma: bool
    institutional_fight_in_progress: bool
    accumulation_detected: bool
    bos_detected: bool

class SetupQualityResult(BaseModel):
    """Trading setup quality evaluation result."""
    strategy_id: int
    grade: str
    score: float
    factor_scores: Dict[str, float]
    position_size: int
    risk_percent: float
    can_auto_trade: bool
    analysis_comments: List[str]

class StrategyCreate(BaseModel):
    """Schema for creating a new strategy."""
    name: str
    description: str
    type: str
    configuration: Optional[Dict[str, Any]] = {}
    parameters: Optional[Dict[str, Any]] = {}
    validation_rules: Optional[Dict[str, Any]] = {}
    timeframes: Optional[List[Dict[str, Any]]] = []
    institutional_settings: Optional[Dict[str, Any]] = None
    entry_exit_settings: Optional[Dict[str, Any]] = None
    market_state_settings: Optional[Dict[str, Any]] = None
    risk_settings: Optional[Dict[str, Any]] = None
    quality_criteria: Optional[Dict[str, Any]] = None
    multi_timeframe_settings: Optional[Dict[str, Any]] = None
    spread_settings: Optional[Dict[str, Any]] = None
    meta_learning: Optional[Dict[str, Any]] = None

class StrategyUpdate(BaseModel):
    """Schema for updating an existing strategy."""
    name: Optional[str] = None
    description: Optional[str] = None
    type: Optional[str] = None
    configuration: Optional[Dict[str, Any]] = None
    parameters: Optional[Dict[str, Any]] = None
    validation_rules: Optional[Dict[str, Any]] = None
    timeframes: Optional[List[Dict[str, Any]]] = None
    institutional_settings: Optional[Dict[str, Any]] = None
    entry_exit_settings: Optional[Dict[str, Any]] = None
    market_state_settings: Optional[Dict[str, Any]] = None
    risk_settings: Optional[Dict[str, Any]] = None
    quality_criteria: Optional[Dict[str, Any]] = None
    multi_timeframe_settings: Optional[Dict[str, Any]] = None
    spread_settings: Optional[Dict[str, Any]] = None
    meta_learning: Optional[Dict[str, Any]] = None

    def dict(self, exclude_unset=False):
        """Override to support exclude_unset."""
        return super().dict(exclude_unset=exclude_unset)

class FeedbackCreate(BaseModel):
    """Schema for creating strategy feedback."""
    feedback_type: str
    title: str
    description: str
    file_path: Optional[str] = None
    file_type: Optional[str] = None
    tags: Optional[List[str]] = []
    improvement_category: Optional[str] = None
    applies_to_setup: bool = False
    applies_to_entry: bool = False
    applies_to_exit: bool = False
    applies_to_risk: bool = False
    pre_trade_conviction_level: Optional[int] = None
    emotional_state_rating: Optional[int] = None
    lessons_learned: Optional[str] = None
    action_items: Optional[List[str]] = []