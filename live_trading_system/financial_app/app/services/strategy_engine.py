"""
StrategyEngineService: Core service for trading strategy processing

This service implements a comprehensive engine for trading strategy management, 
execution, and analysis according to sophisticated financial models.
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from sqlalchemy.orm import Session

from app.models.base import TimestampMixin, UserRelationMixin, AuditMixin
from app.models.strategy import (
    Strategy, StrategyTimeframe, InstitutionalBehaviorSettings, 
    EntryExitSettings, MarketStateSettings, RiskManagementSettings,
    SetupQualityCriteria, VerticalSpreadSettings, MetaLearningSettings,
    MultiTimeframeConfirmationSettings, TradeFeedback, Signal, Trade,
    TimeframeValue, Direction, EntryTechnique, TimeframeImportance,
    MarketStateRequirement, TrendPhase, ProfitTargetMethod, SetupQualityGrade,
    SpreadType
)
from app.schemas.strategy import (
    StrategyCreate, StrategyUpdate, FeedbackCreate, 
    TimeframeAnalysisResult, MarketStateAnalysis, SetupQualityResult
)


class StrategyEngineService:
    """
    Main service class for strategy processing, signal generation, and execution.
    
    This service implements the business logic for trading strategies, including:
    - Strategy CRUD operations
    - Multi-timeframe analysis with strict alignment requirements
    - Market state classification and trading setups identification
    - Signal generation with quality-based position sizing
    - Trade execution with proper risk management
    - Meta-learning and continuous improvement
    """
    
    def __init__(self, db: Session):
        """Initialize with a database session."""
        self.db = db
    #
    # Strategy Management Methods
    #
    
    def create_strategy(self, strategy_data: StrategyCreate, user_id: int) -> Strategy:
        """
        Create a new trading strategy.
        
        Args:
            strategy_data: Strategy creation data
            user_id: ID of the user creating the strategy
            
        Returns:
            Newly created Strategy instance
        """
        # Create the base strategy
        strategy = Strategy(
            name=strategy_data.name,
            description=strategy_data.description,
            type=strategy_data.type,
            configuration=strategy_data.configuration or {},
            parameters=strategy_data.parameters or {},
            validation_rules=strategy_data.validation_rules or {},
            user_id=user_id,
            created_by_id=user_id,
            status="draft"
        )
        
        # Create related settings
        self._create_strategy_settings(strategy, strategy_data)
        
        # Save to database
        self.db.add(strategy)
        self.db.commit()
        self.db.refresh(strategy)
        
        return strategy
        
    def _create_strategy_settings(self, strategy: Strategy, strategy_data: StrategyCreate) -> None:
        """
        Create all related settings objects for a strategy.
        
        Args:
            strategy: Strategy instance
            strategy_data: Strategy creation data
        """
        # Create timeframe settings
        if strategy_data.timeframes:
            for i, tf_data in enumerate(strategy_data.timeframes):
                timeframe = StrategyTimeframe(
                    strategy=strategy,
                    name=tf_data.name,
                    value=tf_data.value,
                    importance=tf_data.importance,
                    order=tf_data.order or i,
                    ma_type=tf_data.ma_type,
                    ma_period_primary=tf_data.ma_period_primary or 21,  # Default to 21
                    ma_period_secondary=tf_data.ma_period_secondary or 200  # Default to 200
                )
                self.db.add(timeframe)
        
        # Create institutional behavior settings
        if strategy_data.institutional_settings:
            inst_settings = InstitutionalBehaviorSettings(
                strategy=strategy,
                detect_accumulation=strategy_data.institutional_settings.detect_accumulation,
                detect_liquidity_grabs=strategy_data.institutional_settings.detect_liquidity_grabs,
                detect_stop_hunts=strategy_data.institutional_settings.detect_stop_hunts,
                wait_for_institutional_footprints=strategy_data.institutional_settings.wait_for_institutional_footprints,
                wait_for_institutional_fight=strategy_data.institutional_settings.wait_for_institutional_fight
            )
            self.db.add(inst_settings)
        else:
            # Create with defaults
            inst_settings = InstitutionalBehaviorSettings(strategy=strategy)
            self.db.add(inst_settings)
            
        # Create entry/exit settings
        if strategy_data.entry_exit_settings:
            entry_exit = EntryExitSettings(
                strategy=strategy,
                direction=strategy_data.entry_exit_settings.direction,
                primary_entry_technique=strategy_data.entry_exit_settings.primary_entry_technique,
                require_candle_close_confirmation=strategy_data.entry_exit_settings.require_candle_close_confirmation,
                trailing_stop_method=strategy_data.entry_exit_settings.trailing_stop_method,
                profit_target_method=strategy_data.entry_exit_settings.profit_target_method,
                profit_target_points=strategy_data.entry_exit_settings.profit_target_points or 25,  # Default to 25 points
                green_bar_sl_placement=strategy_data.entry_exit_settings.green_bar_sl_placement or "below_bar",
                red_bar_sl_placement=strategy_data.entry_exit_settings.red_bar_sl_placement or "above_bar"
            )
            self.db.add(entry_exit)
        else:
            # Create with defaults
            entry_exit = EntryExitSettings(
                strategy=strategy,
                direction=Direction.BOTH,
                primary_entry_technique=EntryTechnique.NEAR_MA,
                profit_target_points=25  # Default to 25 points target
            )
            self.db.add(entry_exit)
        # Create market state settings
        if strategy_data.market_state_settings:
            market_state = MarketStateSettings(
                strategy=strategy,
                required_market_state=strategy_data.market_state_settings.required_market_state,
                avoid_creeper_moves=strategy_data.market_state_settings.avoid_creeper_moves,
                prefer_railroad_trends=strategy_data.market_state_settings.prefer_railroad_trends,
                wait_for_15min_alignment=strategy_data.market_state_settings.wait_for_15min_alignment
            )
            self.db.add(market_state)
        else:
            # Create with defaults
            market_state = MarketStateSettings(
                strategy=strategy,
                avoid_creeper_moves=True,
                prefer_railroad_trends=True,
                wait_for_15min_alignment=True
            )
            self.db.add(market_state)
            
        # Create risk management settings
        if strategy_data.risk_settings:
            risk_settings = RiskManagementSettings(
                strategy=strategy,
                max_risk_per_trade_percent=strategy_data.risk_settings.max_risk_per_trade_percent,
                max_daily_risk_percent=strategy_data.risk_settings.max_daily_risk_percent,
                max_weekly_risk_percent=strategy_data.risk_settings.max_weekly_risk_percent,
                weekly_drawdown_threshold=strategy_data.risk_settings.weekly_drawdown_threshold or 8.0,
                daily_drawdown_threshold=strategy_data.risk_settings.daily_drawdown_threshold or 4.0,
                target_consistent_points=strategy_data.risk_settings.target_consistent_points or 25,
                show_cost_preview=strategy_data.risk_settings.show_cost_preview
            )
            self.db.add(risk_settings)
        else:
            # Create with defaults
            risk_settings = RiskManagementSettings(
                strategy=strategy,
                max_risk_per_trade_percent=1.0,
                max_daily_risk_percent=3.0,
                max_weekly_risk_percent=8.0,
                weekly_drawdown_threshold=8.0,
                daily_drawdown_threshold=4.0,
                target_consistent_points=25,
                show_cost_preview=True
            )
            self.db.add(risk_settings)
        
        # Create setup quality criteria
        if strategy_data.quality_criteria:
            quality = SetupQualityCriteria(
                strategy=strategy,
                a_plus_min_score=strategy_data.quality_criteria.a_plus_min_score,
                a_plus_requires_all_timeframes=strategy_data.quality_criteria.a_plus_requires_all_timeframes,
                a_plus_requires_entry_near_ma=strategy_data.quality_criteria.a_plus_requires_entry_near_ma,
                a_plus_requires_two_day_trend=strategy_data.quality_criteria.a_plus_requires_two_day_trend
            )
            self.db.add(quality)
        else:
            # Create with defaults
            quality = SetupQualityCriteria(
                strategy=strategy,
                a_plus_min_score=90.0,
                a_plus_requires_all_timeframes=True,
                a_plus_requires_entry_near_ma=True,
                a_plus_requires_two_day_trend=True
            )
            self.db.add(quality)
        # Create multi-timeframe settings
        if strategy_data.multi_timeframe_settings:
            mtf = MultiTimeframeConfirmationSettings(
                strategy=strategy,
                require_all_timeframes_aligned=strategy_data.multi_timeframe_settings.require_all_timeframes_aligned,
                primary_timeframe=strategy_data.multi_timeframe_settings.primary_timeframe or TimeframeValue.ONE_HOUR,
                confirmation_timeframe=strategy_data.multi_timeframe_settings.confirmation_timeframe or TimeframeValue.FIFTEEN_MIN,
                entry_timeframe=strategy_data.multi_timeframe_settings.entry_timeframe or TimeframeValue.FIVE_MIN,
                wait_for_15min_alignment=strategy_data.multi_timeframe_settings.wait_for_15min_alignment,
                use_lower_tf_only_for_entry=strategy_data.multi_timeframe_settings.use_lower_tf_only_for_entry
            )
            self.db.add(mtf)
        else:
            # Create with defaults
            mtf = MultiTimeframeConfirmationSettings(
                strategy=strategy,
                require_all_timeframes_aligned=True,
                primary_timeframe=TimeframeValue.ONE_HOUR,
                confirmation_timeframe=TimeframeValue.FIFTEEN_MIN,
                entry_timeframe=TimeframeValue.FIVE_MIN,
                wait_for_15min_alignment=True,
                use_lower_tf_only_for_entry=True
            )
            self.db.add(mtf)
        
        # Create vertical spread settings if enabled
        if strategy_data.spread_settings and strategy_data.spread_settings.use_vertical_spreads:
            spread = VerticalSpreadSettings(
                strategy=strategy,
                use_vertical_spreads=True,
                preferred_spread_type=strategy_data.spread_settings.preferred_spread_type,
                otm_strike_distance=strategy_data.spread_settings.otm_strike_distance or 1,
                min_capital_required=strategy_data.spread_settings.min_capital_required or 500000,
                show_cost_before_execution=strategy_data.spread_settings.show_cost_before_execution
            )
            self.db.add(spread)
            
        # Create meta-learning settings
        if strategy_data.meta_learning:
            meta = MetaLearningSettings(
                strategy=strategy,
                record_trading_sessions=strategy_data.meta_learning.record_trading_sessions,
                record_decision_points=strategy_data.meta_learning.record_decision_points,
                perform_post_market_analysis=strategy_data.meta_learning.perform_post_market_analysis,
                store_screenshots=strategy_data.meta_learning.store_screenshots,
                store_trading_notes=strategy_data.meta_learning.store_trading_notes
            )
            self.db.add(meta)
        else:
            # Create with defaults
            meta = MetaLearningSettings(
                strategy=strategy,
                record_trading_sessions=True,
                record_decision_points=True,
                perform_post_market_analysis=True,
                store_screenshots=True,
                store_trading_notes=True
            )
            self.db.add(meta)
    def update_strategy(self, strategy_id: int, strategy_data: StrategyUpdate, user_id: int) -> Strategy:
        """
        Update an existing strategy.
        
        Args:
            strategy_id: ID of the strategy to update
            strategy_data: Strategy update data
            user_id: ID of the user updating the strategy
            
        Returns:
            Updated Strategy instance
            
        Raises:
            ValueError: If strategy not found
        """
        strategy = self.db.query(Strategy).filter(Strategy.id == strategy_id).first()
        if not strategy:
            raise ValueError(f"Strategy with ID {strategy_id} not found")
        
        # Create a new version for tracking
        new_version = strategy.create_new_version()
        new_version.updated_by_id = user_id
        self.db.add(new_version)
        
        # Update basic strategy attributes
        for key, value in strategy_data.dict(exclude_unset=True).items():
            if key not in ["timeframes", "institutional_settings", "entry_exit_settings", 
                         "market_state_settings", "risk_settings", "quality_criteria", 
                         "multi_timeframe_settings", "spread_settings", "meta_learning"]:
                setattr(strategy, key, value)
        
        # Update related settings
        self._update_strategy_settings(strategy, strategy_data)
        
        # Update timestamps and user info
        strategy.updated_at = datetime.utcnow()
        strategy.updated_by_id = user_id
        strategy.version += 1
        
        # Save to database
        self.db.add(strategy)
        self.db.commit()
        self.db.refresh(strategy)
        
        return strategy
    
    def _update_strategy_settings(self, strategy: Strategy, strategy_data: StrategyUpdate) -> None:
        """
        Update related settings for a strategy.
        
        Args:
            strategy: Strategy instance
            strategy_data: Strategy update data
        """
        # Only implement for fields that are provided in the update
        if strategy_data.timeframes:
            # Delete existing timeframes
            self.db.query(StrategyTimeframe).filter(
                StrategyTimeframe.strategy_id == strategy.id
            ).delete()
            
            # Create new timeframes
            for i, tf_data in enumerate(strategy_data.timeframes):
                timeframe = StrategyTimeframe(
                    strategy=strategy,
                    name=tf_data.name,
                    value=tf_data.value,
                    importance=tf_data.importance,
                    order=tf_data.order or i,
                    ma_type=tf_data.ma_type,
                    ma_period_primary=tf_data.ma_period_primary or 21,
                    ma_period_secondary=tf_data.ma_period_secondary or 200
                )
                self.db.add(timeframe)
        
        # Update institutional behavior settings
        if strategy_data.institutional_settings and strategy.institutional_settings:
            inst_settings = strategy.institutional_settings
            for key, value in strategy_data.institutional_settings.dict(exclude_unset=True).items():
                setattr(inst_settings, key, value)
            self.db.add(inst_settings)
        # Update entry/exit settings
        if strategy_data.entry_exit_settings and strategy.entry_exit_settings:
            entry_exit = strategy.entry_exit_settings
            for key, value in strategy_data.entry_exit_settings.dict(exclude_unset=True).items():
                setattr(entry_exit, key, value)
            self.db.add(entry_exit)
        
        # Update market state settings
        if strategy_data.market_state_settings and strategy.market_state_settings:
            market_state = strategy.market_state_settings
            for key, value in strategy_data.market_state_settings.dict(exclude_unset=True).items():
                setattr(market_state, key, value)
            self.db.add(market_state)
        
        # Update risk management settings
        if strategy_data.risk_settings and strategy.risk_settings:
            risk_settings = strategy.risk_settings
            for key, value in strategy_data.risk_settings.dict(exclude_unset=True).items():
                setattr(risk_settings, key, value)
            self.db.add(risk_settings)
        
        # Update setup quality criteria
        if strategy_data.quality_criteria and strategy.quality_criteria:
            quality = strategy.quality_criteria
            for key, value in strategy_data.quality_criteria.dict(exclude_unset=True).items():
                setattr(quality, key, value)
            self.db.add(quality)
            
        # Update multi-timeframe settings
        if strategy_data.multi_timeframe_settings and strategy.multi_timeframe_settings:
            mtf = strategy.multi_timeframe_settings
            for key, value in strategy_data.multi_timeframe_settings.dict(exclude_unset=True).items():
                setattr(mtf, key, value)
            self.db.add(mtf)
        
        # Update vertical spread settings
        if strategy_data.spread_settings and strategy.spread_settings:
            spread = strategy.spread_settings
            for key, value in strategy_data.spread_settings.dict(exclude_unset=True).items():
                setattr(spread, key, value)
            self.db.add(spread)
        elif strategy_data.spread_settings and strategy_data.spread_settings.use_vertical_spreads and not strategy.spread_settings:
            # Create new spread settings if enabled
            spread = VerticalSpreadSettings(
                strategy=strategy,
                use_vertical_spreads=True,
                preferred_spread_type=strategy_data.spread_settings.preferred_spread_type,
                otm_strike_distance=strategy_data.spread_settings.otm_strike_distance or 1,
                min_capital_required=strategy_data.spread_settings.min_capital_required or 500000,
                show_cost_before_execution=strategy_data.spread_settings.show_cost_before_execution
            )
            self.db.add(spread)
        
        # Update meta-learning settings
        if strategy_data.meta_learning and strategy.meta_learning:
            meta = strategy.meta_learning
            for key, value in strategy_data.meta_learning.dict(exclude_unset=True).items():
                setattr(meta, key, value)
            self.db.add(meta)
    def get_strategy(self, strategy_id: int) -> Strategy:
        """
        Get a strategy by ID.
        
        Args:
            strategy_id: ID of the strategy
            
        Returns:
            Strategy instance
            
        Raises:
            ValueError: If strategy not found
        """
        strategy = self.db.query(Strategy).filter(Strategy.id == strategy_id).first()
        if not strategy:
            raise ValueError(f"Strategy with ID {strategy_id} not found")
        return strategy
    
    def list_strategies(self, user_id: Optional[int] = None, 
                       offset: int = 0, limit: int = 100,
                       include_inactive: bool = False) -> List[Strategy]:
        """
        List strategies with optional filtering.
        
        Args:
            user_id: Optional user ID to filter by owner
            offset: Pagination offset
            limit: Maximum number of results
            include_inactive: Whether to include inactive strategies
            
        Returns:
            List of Strategy instances
        """
        query = self.db.query(Strategy)
        
        # Apply filters
        if user_id:
            query = query.filter(Strategy.user_id == user_id)
        
        if not include_inactive:
            query = query.filter(Strategy.is_active == True)
        
        # Exclude soft-deleted
        query = query.filter(Strategy.deleted_at.is_(None))
        
        # Apply pagination
        query = query.offset(offset).limit(limit)
        
        return query.all()
        
    def delete_strategy(self, strategy_id: int, user_id: int, hard_delete: bool = False) -> bool:
        """
        Delete a strategy.
        
        Args:
            strategy_id: ID of the strategy to delete
            user_id: ID of the user deleting the strategy
            hard_delete: Whether to permanently delete (True) or soft delete (False)
            
        Returns:
            True if successful
            
        Raises:
            ValueError: If strategy not found
        """
        strategy = self.db.query(Strategy).filter(Strategy.id == strategy_id).first()
        if not strategy:
            raise ValueError(f"Strategy with ID {strategy_id} not found")
        
        if hard_delete:
            self.db.delete(strategy)
        else:
            strategy.soft_delete(user_id)
            self.db.add(strategy)
        
        self.db.commit()
        return True
    
    def activate_strategy(self, strategy_id: int, user_id: int) -> Strategy:
        """
        Activate a strategy.
        
        Args:
            strategy_id: ID of the strategy to activate
            user_id: ID of the user activating the strategy
            
        Returns:
            Activated Strategy instance
            
        Raises:
            ValueError: If strategy not found
        """
        strategy = self.db.query(Strategy).filter(Strategy.id == strategy_id).first()
        if not strategy:
            raise ValueError(f"Strategy with ID {strategy_id} not found")
        
        # Validate strategy parameters before activation
        validation_errors = strategy.validate_parameters()
        if validation_errors:
            raise ValueError(f"Strategy parameters validation failed: {', '.join(validation_errors)}")
        
        strategy.is_active = True
        strategy.updated_by_id = user_id
        strategy.updated_at = datetime.utcnow()
        strategy.update_status("active")
        
        self.db.add(strategy)
        self.db.commit()
        self.db.refresh(strategy)
        
        return strategy
        
    def deactivate_strategy(self, strategy_id: int, user_id: int) -> Strategy:
        """
        Deactivate a strategy.
        
        Args:
            strategy_id: ID of the strategy to deactivate
            user_id: ID of the user deactivating the strategy
            
        Returns:
            Deactivated Strategy instance
            
        Raises:
            ValueError: If strategy not found
        """
        strategy = self.db.query(Strategy).filter(Strategy.id == strategy_id).first()
        if not strategy:
            raise ValueError(f"Strategy with ID {strategy_id} not found")
        
        strategy.is_active = False
        strategy.updated_by_id = user_id
        strategy.updated_at = datetime.utcnow()
        strategy.update_status("paused")
        
        self.db.add(strategy)
        self.db.commit()
        self.db.refresh(strategy)
        
        return strategy
    #
    # Multi-Timeframe Analysis Methods
    #
    
    def analyze_timeframes(self, strategy_id: int, market_data: Dict[TimeframeValue, Dict]) -> TimeframeAnalysisResult:
        """
        Analyze market data across multiple timeframes.
        
        Implements the hierarchical timeframe structure with strict alignment requirements.
        
        Args:
            strategy_id: ID of the strategy to analyze
            market_data: Dictionary of market data by timeframe
            
        Returns:
            TimeframeAnalysisResult with analysis results
            
        Raises:
            ValueError: If strategy not found
        """
        strategy = self.get_strategy(strategy_id)
        if not strategy.multi_timeframe_settings:
            raise ValueError(f"Strategy {strategy_id} doesn't have multi-timeframe settings")
        
        # Get timeframes in order of importance
        timeframes = sorted(
            strategy.timeframes, 
            key=lambda tf: tf.order
        )
        
        # Analyze each timeframe
        results = {}
        aligned = True
        expected_direction = None
        
        for tf in timeframes:
            # Skip if we don't have data for this timeframe
            if tf.value not in market_data:
                results[tf.value.value] = {
                    "analyzed": False,
                    "reason": "No data available"
                }
                continue
            
            # Analyze this timeframe
            tf_data = market_data[tf.value]
            
            # Determine trend direction on this timeframe
            direction = self._determine_trend_direction(
                tf_data, 
                tf.ma_period_primary, 
                tf.ma_period_secondary
            )
            
            # Check if MA is trending
            ma_trending = self._check_ma_trending(
                tf_data, 
                tf.ma_period_primary,
                tf.min_ma_slope_for_trend if hasattr(tf, 'min_ma_slope_for_trend') else 0.0005
            )
            
            # Check for price struggling near MA
            price_struggling = self._check_price_ma_struggle(
                tf_data,
                tf.ma_period_primary,
                tf.ma_struggle_threshold if hasattr(tf, 'ma_struggle_threshold') else 0.2
            )
            
            # Check if price is above/below MA
            price_above_ma = tf_data.get("close", [])[-1] > tf_data.get(f"ma{tf.ma_period_primary}", [])[-1]
            
            # Store results for this timeframe
            results[tf.value.value] = {
                "analyzed": True,
                "direction": direction,
                "ma_trending": ma_trending,
                "price_struggling": price_struggling,
                "price_above_ma": price_above_ma
            }
            
            # Check alignment with expected direction (from higher timeframes)
            if tf.importance == TimeframeImportance.PRIMARY:
                # This sets the expected direction for lower timeframes
                expected_direction = direction
            elif tf.require_alignment and expected_direction is not None and direction != expected_direction:
                aligned = False
                results[tf.value.value]["aligned"] = False
            else:
                results[tf.value.value]["aligned"] = True
            # Calculate overall alignment score
        weights = {}
        for tf in timeframes:
            weights[tf.value.value] = 0.0
            
        # Get weights from settings if available
        if hasattr(strategy.multi_timeframe_settings, 'timeframe_weights') and strategy.multi_timeframe_settings.timeframe_weights:
            weights.update(strategy.multi_timeframe_settings.timeframe_weights)
        else:
            # Default weights prioritize higher timeframes
            total_tfs = len(timeframes)
            for i, tf in enumerate(timeframes):
                weight = 1.0 - (i / total_tfs * 0.7)  # Higher timeframes get higher weights
                weights[tf.value.value] = weight
        
        # Calculate weighted score
        weighted_sum = 0.0
        total_weight = 0.0
        
        for tf in timeframes:
            tf_key = tf.value.value
            if tf_key in results and results[tf_key].get("analyzed", False):
                is_aligned = results[tf_key].get("aligned", True)
                if is_aligned:
                    weighted_sum += weights[tf_key]
                total_weight += weights[tf_key]
        
        # Get alignment score (0-1)
        alignment_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # Determine if we have the minimum alignment score
        min_score = strategy.multi_timeframe_settings.min_alignment_score
        sufficient_alignment = alignment_score >= min_score
        
        # Determine if all required timeframes are aligned
        all_aligned = aligned and all(
            results[tf.value.value].get("aligned", True) 
            for tf in timeframes 
            if tf.value.value in results and results[tf.value.value].get("analyzed", False)
        )
        
        # Determine if we need all timeframes aligned
        require_all = strategy.multi_timeframe_settings.require_all_timeframes_aligned
        
        # Final alignment decision
        is_aligned = (not require_all and sufficient_alignment) or (require_all and all_aligned)
        
        # Special case: check 15-min alignment if required
        if (strategy.multi_timeframe_settings.wait_for_15min_alignment and
            TimeframeValue.FIFTEEN_MIN in market_data):
            
            fifteen_min_data = market_data[TimeframeValue.FIFTEEN_MIN]
            fifteen_min_aligned = self._check_15min_alignment(
                fifteen_min_data,
                expected_direction,
                strategy.multi_timeframe_settings.min_15min_confirmation_bars
                if hasattr(strategy.multi_timeframe_settings, 'min_15min_confirmation_bars') else 2
            )
            
            # Update alignment status based on 15-min
            is_aligned = is_aligned and fifteen_min_aligned
            results["fifteen_min_aligned"] = fifteen_min_aligned
        
        return TimeframeAnalysisResult(
            aligned=is_aligned,
            alignment_score=alignment_score,
            timeframe_results=results,
            primary_direction=expected_direction,
            require_all_aligned=require_all,
            min_alignment_score=min_score,
            sufficient_alignment=sufficient_alignment
        )
    def _determine_trend_direction(self, timeframe_data: Dict, primary_ma: int, secondary_ma: int) -> str:
        """
        Determine trend direction based on price and MA relationship.
        
        Args:
            timeframe_data: Market data for a timeframe
            primary_ma: Primary MA period (typically 21)
            secondary_ma: Secondary MA period (typically 200)
            
        Returns:
            "up", "down", or "sideways"
        """
        # Get latest values
        close = timeframe_data.get("close", [])[-1] if timeframe_data.get("close") else None
        primary_ma_value = timeframe_data.get(f"ma{primary_ma}", [])[-1] if timeframe_data.get(f"ma{primary_ma}") else None
        secondary_ma_value = timeframe_data.get(f"ma{secondary_ma}", [])[-1] if timeframe_data.get(f"ma{secondary_ma}") else None
        
        # Get previous values (5 bars ago) for slope calculation
        primary_ma_prev = timeframe_data.get(f"ma{primary_ma}", [])[-6] if len(timeframe_data.get(f"ma{primary_ma}", [])) > 5 else None
        
        # Check if we have enough data
        if close is None or primary_ma_value is None:
            return "unknown"
        
        # Calculate primary MA slope
        ma_slope = 0
        if primary_ma_prev is not None:
            ma_slope = (primary_ma_value - primary_ma_prev) / primary_ma_prev
        
        # Determine trend based on price vs MA and MA slope
        if close > primary_ma_value and ma_slope > 0.0001:
            return "up"
        elif close < primary_ma_value and ma_slope < -0.0001:
            return "down"
        else:
            # Check secondary MA relationship if available
            if secondary_ma_value is not None:
                if primary_ma_value > secondary_ma_value:
                    return "up"
                elif primary_ma_value < secondary_ma_value:
                    return "down"
            
            return "sideways"

    def _check_ma_trending(self, timeframe_data: Dict, ma_period: int, min_slope: float) -> bool:
        """
        Check if MA is trending (has significant slope).
        
        Args:
            timeframe_data: Market data for a timeframe
            ma_period: Moving average period
            min_slope: Minimum slope to consider trending
            
        Returns:
            True if MA is trending, False otherwise
        """
        # Get MA values
        ma_values = timeframe_data.get(f"ma{ma_period}", [])
        if len(ma_values) < 10:  # Need at least 10 values for reliable slope
            return False
        
        # Calculate slope over last 5 bars
        ma_current = ma_values[-1]
        ma_prev = ma_values[-6]  # 5 bars ago
        
        if ma_current == 0 or ma_prev == 0:
            return False
        
        # Calculate relative slope
        ma_slope = (ma_current - ma_prev) / ma_prev
        
        # Check if slope exceeds minimum
        return abs(ma_slope) > min_slope

    def _check_price_ma_struggle(self, timeframe_data: Dict, ma_period: int, threshold: float) -> bool:
        """
        Check if price is struggling to move away from MA.
        
        Args:
            timeframe_data: Market data for a timeframe
            ma_period: Moving average period
            threshold: Maximum distance (as % of price) to consider "struggling"
            
        Returns:
            True if price is struggling near MA, False otherwise
        """
        # Get latest values
        close_values = timeframe_data.get("close", [])
        ma_values = timeframe_data.get(f"ma{ma_period}", [])
        
        if len(close_values) < 5 or len(ma_values) < 5:
            return False
        
        # Check last 5 bars
        for i in range(1, 6):
            close = close_values[-i]
            ma = ma_values[-i]
            
            # Calculate distance as percentage of price
            if ma == 0:
                continue
                
            distance_pct = abs(close - ma) / ma
            
            # If any bar exceeds threshold, not struggling
            if distance_pct > threshold:
                return False
        
        # All bars are within threshold of MA
        return True

    def _check_15min_alignment(self, fifteen_min_data: Dict, expected_direction: str, min_bars: int) -> bool:
        """
        Check if 15-min timeframe confirms the expected direction.
        
        Args:
            fifteen_min_data: Market data for 15-min timeframe
            expected_direction: Expected trend direction from higher timeframes
            min_bars: Minimum number of bars for confirmation
            
        Returns:
            True if 15-min confirms, False otherwise
        """
        # Get close values
        close_values = fifteen_min_data.get("close", [])
        if len(close_values) < min_bars + 1:  # Need min_bars + 1 to check min_bars bars
            return False
        
        # Get the last min_bars bars
        recent_closes = close_values[-(min_bars+1):]
        
        # Check if bars confirm direction
        if expected_direction == "up":
            # Count bullish bars (close > open)
            bullish_bars = sum(1 for i in range(min_bars) if recent_closes[i+1] > recent_closes[i])
            return bullish_bars >= min_bars * 0.7  # At least 70% of bars should be bullish
        elif expected_direction == "down":
            # Count bearish bars (close < open)
            bearish_bars = sum(1 for i in range(min_bars) if recent_closes[i+1] < recent_closes[i])
            return bearish_bars >= min_bars * 0.7  # At least 70% of bars should be bearish
        
        return False  # Unknown direction
    #
    # Market State Analysis Methods
    #
    
    def analyze_market_state(self, strategy_id: int, market_data: Dict[TimeframeValue, Dict]) -> MarketStateAnalysis:
        """
        Analyze current market state for strategy execution.
        
        Implements key requirements including:
        - Railroad vs creeper move detection
        - Detection of institutional behavior
        - Price action vs indicator divergence
        
        Args:
            strategy_id: ID of the strategy
            market_data: Dictionary of market data by timeframe
            
        Returns:
            MarketStateAnalysis with detailed market state assessment
            
        Raises:
            ValueError: If strategy not found
        """
        strategy = self.get_strategy(strategy_id)
        if not strategy.market_state_settings:
            raise ValueError(f"Strategy {strategy_id} doesn't have market state settings")
        
        # Get primary data from 1H timeframe (default)
        primary_tf = TimeframeValue.ONE_HOUR
        if strategy.multi_timeframe_settings and strategy.multi_timeframe_settings.primary_timeframe:
            primary_tf = strategy.multi_timeframe_settings.primary_timeframe
        
        # Ensure we have data for primary timeframe
        if primary_tf not in market_data:
            raise ValueError(f"No data available for primary timeframe {primary_tf.value}")
        
        primary_data = market_data[primary_tf]
        
        # Check for railroad vs creeper trend type
        is_railroad = self._detect_railroad_trend(
            primary_data,
            threshold=strategy.market_state_settings.railroad_momentum_threshold
            if hasattr(strategy.market_state_settings, 'railroad_momentum_threshold') else 0.8
        )
        
        is_creeper = not is_railroad and self._detect_creeper_move(primary_data)
        
        # Check for two-day trend if daily data is available
        has_two_day_trend = False
        trend_direction = "unknown"
        if TimeframeValue.DAILY in market_data:
            daily_data = market_data[TimeframeValue.DAILY]
            has_two_day_trend, trend_direction = self._check_two_day_trend(daily_data)
        
        # Check for price vs indicator divergence
        price_indicator_divergence = False
        if strategy.market_state_settings.detect_price_indicator_divergence:
            price_indicator_divergence = self._detect_price_indicator_divergence(primary_data)
        
        # Check for price struggling with MA
        price_struggling_near_ma = False
        if strategy.market_state_settings.detect_price_ma_struggle:
            price_struggling_near_ma = self._check_price_ma_struggle(
                primary_data,
                21,  # Default primary MA period
                strategy.market_state_settings.ma_struggle_threshold
                if hasattr(strategy.market_state_settings, 'ma_struggle_threshold') else 0.2
            )
        # Check for institutional behavior
        institutional_fight_in_progress = False
        accumulation_detected = False
        bos_detected = False
        
        if strategy.institutional_settings:
            # Detect institutional fight
            if strategy.institutional_settings.wait_for_institutional_fight:
                institutional_fight_in_progress = self._detect_institutional_fight(
                    primary_data,
                    strategy.institutional_settings.institutional_fight_detection_methods
                    if hasattr(strategy.institutional_settings, 'institutional_fight_detection_methods') 
                    else ["high_volume_narrow_range", "price_rejection"]
                )
            
            # Detect accumulation
            if strategy.institutional_settings.detect_accumulation:
                accumulation_detected = self._detect_accumulation(
                    primary_data,
                    volume_threshold=strategy.institutional_settings.accumulation_volume_threshold
                    if hasattr(strategy.institutional_settings, 'accumulation_volume_threshold') else 1.5,
                    price_threshold=strategy.institutional_settings.accumulation_price_threshold
                    if hasattr(strategy.institutional_settings, 'accumulation_price_threshold') else 0.002
                )
            
            # Detect break of structure
            if strategy.institutional_settings.detect_bos:
                bos_detected = self._detect_bos(
                    primary_data,
                    strategy.institutional_settings.bos_confirmation_bars
                    if hasattr(strategy.institutional_settings, 'bos_confirmation_bars') else 1
                )
        
        # Determine overall market state
        market_state = self._determine_market_state(
            primary_data,
            is_railroad,
            is_creeper,
            trend_direction
        )
        
        # Determine trend phase
        trend_phase = self._determine_trend_phase(primary_data)
        
        return MarketStateAnalysis(
            market_state=market_state,
            trend_phase=trend_phase,
            is_railroad_trend=is_railroad,
            is_creeper_move=is_creeper,
            has_two_day_trend=has_two_day_trend,
            trend_direction=trend_direction,
            price_indicator_divergence=price_indicator_divergence,
            price_struggling_near_ma=price_struggling_near_ma,
            institutional_fight_in_progress=institutional_fight_in_progress,
            accumulation_detected=accumulation_detected,
            bos_detected=bos_detected
        )
    def _detect_railroad_trend(self, timeframe_data: Dict, threshold: float) -> bool:
        """
        Detect a railroad trend (strong one-sided trend).
        
        Args:
            timeframe_data: Market data for a timeframe
            threshold: Momentum threshold
            
        Returns:
            True if railroad trend detected, False otherwise
        """
        # Get close and open values
        close_values = timeframe_data.get("close", [])
        open_values = timeframe_data.get("open", [])
        
        if len(close_values) < 10 or len(open_values) < 10:  # Need at least 10 bars
            return False
        
        # Check last 5 bars
        bullish_count = 0
        bearish_count = 0
        strong_bars = 0
        
        for i in range(1, 6):
            if i >= len(close_values) or i >= len(open_values):
                break
                
            close = close_values[-i]
            open = open_values[-i]
            
            # Count bullish and bearish bars
            if close > open:
                bullish_count += 1
            elif close < open:
                bearish_count += 1
            
            # Check if bar size is significant
            if close != 0 and open != 0:
                bar_size = abs(close - open) / ((close + open) / 2)
                if bar_size > 0.003:  # More than 0.3% body
                    strong_bars += 1
        
        # Calculate consistency (one-sidedness)
        consistency = max(bullish_count, bearish_count) / 5.0
        
        # Return true if trend is consistent and strong
        return consistency > threshold and strong_bars >= 3

    def _detect_creeper_move(self, timeframe_data: Dict) -> bool:
        """
        Detect a creeper move (slow grinding price action).
        
        Args:
            timeframe_data: Market data for a timeframe
            
        Returns:
            True if creeper move detected, False otherwise
        """
        # Get close values
        close_values = timeframe_data.get("close", [])
        
        if len(close_values) < 10:  # Need at least 10 bars
            return False
        
        # Calculate recent average daily range
        recent_ranges = []
        for i in range(1, 8):  # Last 7 bars
            if i >= len(close_values):
                break
                
            high = timeframe_data.get("high", [])[-i] if timeframe_data.get("high") else close_values[-i]
            low = timeframe_data.get("low", [])[-i] if timeframe_data.get("low") else close_values[-i]
            
            if high and low and high != 0:
                daily_range = (high - low) / high
                recent_ranges.append(daily_range)
        
        if not recent_ranges:
            return False
        
        # Calculate average range
        avg_range = sum(recent_ranges) / len(recent_ranges)
        
        # Creeper move has small ranges
        return avg_range < 0.005  # Less than 0.5% average range

    def _check_two_day_trend(self, daily_data: Dict) -> Tuple[bool, str]:
        """
        Check if there's a consistent two-day trend.
        
        Args:
            daily_data: Market data for daily timeframe
            
        Returns:
            Tuple of (has_two_day_trend, direction)
        """
        # Get close values
        close_values = daily_data.get("close", [])
        open_values = daily_data.get("open", [])
        
        if len(close_values) < 3 or len(open_values) < 3:  # Need at least 3 days
            return False, "unknown"
        
        # Check last 2 completed days
        day1_bullish = close_values[-2] > open_values[-2]
        day2_bullish = close_values[-1] > open_values[-1]
        
        # Both days should have same direction
        if day1_bullish and day2_bullish:
            return True, "up"
        elif not day1_bullish and not day2_bullish:
            return True, "down"
        
        return False, "mixed"
    def _detect_price_indicator_divergence(self, timeframe_data: Dict) -> bool:
        """
        Detect divergence between price action and indicators.
        
        Args:
            timeframe_data: Market data for a timeframe
            
        Returns:
            True if divergence detected, False otherwise
        """
        # Get close values and 21 MA
        close_values = timeframe_data.get("close", [])
        ma21_values = timeframe_data.get("ma21", [])
        
        if len(close_values) < 10 or len(ma21_values) < 10:
            return False
        
        # Get trend from MA
        ma_slope = (ma21_values[-1] - ma21_values[-5]) / ma21_values[-5] if ma21_values[-5] != 0 else 0
        ma_trend = "up" if ma_slope > 0 else "down" if ma_slope < 0 else "sideways"
        
        # Get trend from price action (last 5 bars)
        price_slope = (close_values[-1] - close_values[-5]) / close_values[-5] if close_values[-5] != 0 else 0
        price_trend = "up" if price_slope > 0 else "down" if price_slope < 0 else "sideways"
        
        # Check for divergence (opposite directions)
        return (ma_trend == "up" and price_trend == "down") or (ma_trend == "down" and price_trend == "up")

    def _detect_institutional_fight(self, timeframe_data: Dict, detection_methods: List[str]) -> bool:
        """
        Detect if institutional traders are fighting over direction.
        
        Args:
            timeframe_data: Market data for a timeframe
            detection_methods: List of methods to detect institutional fight
            
        Returns:
            True if institutional fight detected, False otherwise
        """
        # Track which methods detected a fight
        detected = []
        
        # Check for high volume with narrow range
        if "high_volume_narrow_range" in detection_methods:
            # Get volume and range data
            volumes = timeframe_data.get("volume", [])
            high_values = timeframe_data.get("high", [])
            low_values = timeframe_data.get("low", [])
            
            if len(volumes) > 10 and len(high_values) > 10 and len(low_values) > 10:
                # Calculate average volume and range
                avg_volume = sum(volumes[-11:-1]) / 10  # Previous 10 bars
                
                # Check last 3 bars
                for i in range(1, 4):
                    if i >= len(volumes) or i >= len(high_values) or i >= len(low_values):
                        break
                        
                    volume = volumes[-i]
                    high = high_values[-i]
                    low = low_values[-i]
                    
                    # Calculate range as percentage
                    if high != 0:
                        price_range = (high - low) / high
                        
                        # High volume (>150% of average) with narrow range (<0.5%)
                        if volume > avg_volume * 1.5 and price_range < 0.005:
                            detected.append("high_volume_narrow_range")
                            break
                    # Check for price rejection
        if "price_rejection" in detection_methods:
            # Get recent bar data
            close_values = timeframe_data.get("close", [])
            open_values = timeframe_data.get("open", [])
            high_values = timeframe_data.get("high", [])
            low_values = timeframe_data.get("low", [])
            
            if (len(close_values) > 5 and len(open_values) > 5 and 
                len(high_values) > 5 and len(low_values) > 5):
                
                # Check last 3 bars for rejection patterns
                for i in range(1, 4):
                    if (i >= len(close_values) or i >= len(open_values) or 
                        i >= len(high_values) or i >= len(low_values)):
                        break
                        
                    close = close_values[-i]
                    open = open_values[-i]
                    high = high_values[-i]
                    low = low_values[-i]
                    
                    # Skip if any value is 0
                    if close == 0 or open == 0 or high == 0 or low == 0:
                        continue
                    
                    body_size = abs(close - open)
                    total_range = high - low
                    
                    # Check for rejection (small body, long wicks)
                    if total_range > 0 and body_size / total_range < 0.3:
                        detected.append("price_rejection")
                        break
        
        # Check for rapid reversals
        if "rapid_reversals" in detection_methods:
            # Get close values
            close_values = timeframe_data.get("close", [])
            
            if len(close_values) > 10:
                # Check for direction changes in last 5 bars
                direction_changes = 0
                for i in range(1, 5):
                    if close_values[-i] > close_values[-i-1] and close_values[-i-1] < close_values[-i-2]:
                        direction_changes += 1
                    elif close_values[-i] < close_values[-i-1] and close_values[-i-1] > close_values[-i-2]:
                        direction_changes += 1
                
                # Multiple direction changes indicate fight
                if direction_changes >= 2:
                    detected.append("rapid_reversals")
        
        # Check for failed breakouts
        if "failed_breakouts" in detection_methods:
            # Get high, low, and close values
            high_values = timeframe_data.get("high", [])
            low_values = timeframe_data.get("low", [])
            close_values = timeframe_data.get("close", [])
            
            if len(high_values) > 10 and len(low_values) > 10 and len(close_values) > 10:
                # Find recent high and low
                recent_high = max(high_values[-10:-1])
                recent_low = min(low_values[-10:-1])
                
                # Check if most recent bar broke out and failed
                latest_high = high_values[-1]
                latest_low = low_values[-1]
                latest_close = close_values[-1]
                
                # Failed upside breakout
                if latest_high > recent_high and latest_close < recent_high:
                    detected.append("failed_breakouts")
                
                # Failed downside breakout
                if latest_low < recent_low and latest_close > recent_low:
                    detected.append("failed_breakouts")
        
        # Institutional fight detected if any method detected it
        return len(detected) > 0
    def _detect_accumulation(self, timeframe_data: Dict, volume_threshold: float, price_threshold: float) -> bool:
        """
        Detect accumulation pattern (high volume with little price movement).
        
        Args:
            timeframe_data: Market data for a timeframe
            volume_threshold: Relative volume threshold for accumulation
            price_threshold: Maximum price movement for accumulation
            
        Returns:
            True if accumulation detected, False otherwise
        """
        # Get volume and price data
        volumes = timeframe_data.get("volume", [])
        close_values = timeframe_data.get("close", [])
        
        if len(volumes) < 20 or len(close_values) < 20:  # Need adequate history
            return False
        
        # Calculate average volume (excluding most recent bars)
        avg_volume = sum(volumes[-20:-5]) / 15  # Previous 15 bars
        
        # Check for high volume with little price movement in last 3-5 bars
        high_volume_bars = 0
        price_stable = True
        
        for i in range(1, 6):  # Check last 5 bars
            if i >= len(volumes) or i >= len(close_values):
                break
                
            volume = volumes[-i]
            
            # Check if volume exceeds threshold
            if volume > avg_volume * volume_threshold:
                high_volume_bars += 1
            
            # Check price stability if we have previous close
            if i < len(close_values) - 1:
                curr_close = close_values[-i]
                prev_close = close_values[-i-1]
                
                if curr_close != 0 and prev_close != 0:
                    price_change = abs(curr_close - prev_close) / prev_close
                    
                    # Price change exceeds threshold
                    if price_change > price_threshold:
                        price_stable = False
        
        # Accumulation detected if multiple high volume bars with stable price
        return high_volume_bars >= 2 and price_stable

    def _detect_bos(self, timeframe_data: Dict, confirmation_bars: int) -> bool:
        """
        Detect break of structure pattern.
        
        Args:
            timeframe_data: Market data for a timeframe
            confirmation_bars: Bars needed to confirm BOS
            
        Returns:
            True if BOS detected, False otherwise
        """
        # Get high, low, and close values
        high_values = timeframe_data.get("high", [])
        low_values = timeframe_data.get("low", [])
        close_values = timeframe_data.get("close", [])
        
        if len(high_values) < 20 or len(low_values) < 20 or len(close_values) < 20:
            return False
        
        # Find recent swing high and low (last 10-20 bars)
        swing_high = max(high_values[-20:-5])
        swing_low = min(low_values[-20:-5])
        
        # Check for break of swing high or low
        for i in range(confirmation_bars + 1):
            if i >= len(close_values):
                break
                
            close = close_values[-i-1]
            
            # Check for break of swing high (bullish BOS)
            if close > swing_high:
                # Confirm with additional bars
                confirmed = True
                for j in range(1, confirmation_bars + 1):
                    if (i + j) >= len(close_values):
                        confirmed = False
                        break
                    
                    confirm_close = close_values[-i-j-1]
                    if confirm_close <= swing_high:
                        confirmed = False
                        break
                
                if confirmed:
                    return True
            
            # Check for break of swing low (bearish BOS)
            if close < swing_low:
                # Confirm with additional bars
                confirmed = True
                for j in range(1, confirmation_bars + 1):
                    if (i + j) >= len(close_values):
                        confirmed = False
                        break
                    
                    confirm_close = close_values[-i-j-1]
                    if confirm_close >= swing_low:
                        confirmed = False
                        break
                
                if confirmed:
                    return True
        
        return False
    def _determine_market_state(self, timeframe_data: Dict, is_railroad: bool, is_creeper: bool, trend_direction: str) -> MarketStateRequirement:
        """
        Determine overall market state.
        
        Args:
            timeframe_data: Market data for a timeframe
            is_railroad: Whether a railroad trend is detected
            is_creeper: Whether a creeper move is detected
            trend_direction: Trend direction
            
        Returns:
            MarketStateRequirement enum value
        """
        # Get close values for volatility assessment
        close_values = timeframe_data.get("close", [])
        
        if len(close_values) < 20:
            return MarketStateRequirement.ANY
        
        # Calculate recent volatility
        volatility = self._calculate_volatility(close_values[-20:])
        
        # Determine based on trend and volatility
        if is_railroad and trend_direction == "up":
            return MarketStateRequirement.TRENDING_UP
        elif is_railroad and trend_direction == "down":
            return MarketStateRequirement.TRENDING_DOWN
        elif is_creeper:
            return MarketStateRequirement.CREEPER_MOVE
        elif volatility > 0.01:  # High volatility (>1%)
            return MarketStateRequirement.MOMENTUM_MOVE
        elif volatility < 0.003:  # Low volatility (<0.3%)
            return MarketStateRequirement.NARROW_LOW_VOLUME
        else:
            return MarketStateRequirement.RANGE_BOUND

    def _calculate_volatility(self, close_values: List[float]) -> float:
        """
        Calculate volatility from close values.
        
        Args:
            close_values: List of close prices
            
        Returns:
            Volatility as average true range percentage
        """
        if not close_values or len(close_values) < 2:
            return 0.0
        
        # Calculate daily changes
        changes = []
        for i in range(1, len(close_values)):
            if close_values[i-1] != 0:
                change = abs(close_values[i] - close_values[i-1]) / close_values[i-1]
                changes.append(change)
        
        # Return average change
        return sum(changes) / len(changes) if changes else 0.0

    def _determine_trend_phase(self, timeframe_data: Dict) -> TrendPhase:
        """
        Determine the current phase of the trend.
        
        Args:
            timeframe_data: Market data for a timeframe
            
        Returns:
            TrendPhase enum value
        """
        # Get price and MA data
        close_values = timeframe_data.get("close", [])
        ma21_values = timeframe_data.get("ma21", [])
        
        if len(close_values) < 50 or len(ma21_values) < 50:  # Need adequate history
            return TrendPhase.UNDETERMINED
        
        # Determine trend direction
        ma_slope = (ma21_values[-1] - ma21_values[-10]) / ma21_values[-10] if ma21_values[-10] != 0 else 0
        trend_up = ma_slope > 0.001
        trend_down = ma_slope < -0.001
        
        if not trend_up and not trend_down:
            return TrendPhase.UNDETERMINED
        
        # Calculate distance from trend start
        trend_start_idx = 0
        trend_length = 0
        
        # Find where trend started (cross of MA)
        for i in range(10, min(len(close_values), 50)):
            idx = len(close_values) - i
            
            if trend_up and close_values[idx] < ma21_values[idx] and close_values[idx+1] > ma21_values[idx+1]:
                trend_start_idx = idx
                trend_length = i
                break
            elif trend_down and close_values[idx] > ma21_values[idx] and close_values[idx+1] < ma21_values[idx+1]:
                trend_start_idx = idx
                trend_length = i
                break
        
        # Determine phase based on trend length
        if trend_length == 0:
            return TrendPhase.UNDETERMINED
        
        # Early phase (first 30% of typical trend duration)
        if trend_length < 15:
            return TrendPhase.EARLY
        # Late phase (last 30% of typical trend duration)
        elif trend_length > 30:
            return TrendPhase.LATE
        # Middle phase (middle 40% of typical trend duration)
        else:
            return TrendPhase.MIDDLE
    #
    # Setup Quality Evaluation Methods
    #
    
    def evaluate_setup_quality(self, strategy_id: int, 
                              timeframe_analysis: TimeframeAnalysisResult,
                              market_state: MarketStateAnalysis,
                              entry_data: Dict) -> SetupQualityResult:
        """
        Evaluate setup quality based on Rikk's A+ to F grading system.
        
        Args:
            strategy_id: ID of the strategy
            timeframe_analysis: Timeframe analysis results
            market_state: Market state analysis
            entry_data: Entry point data
            
        Returns:
            SetupQualityResult with grade and position sizing recommendation
            
        Raises:
            ValueError: If strategy not found
        """
        strategy = self.get_strategy(strategy_id)
        if not strategy.quality_criteria:
            raise ValueError(f"Strategy {strategy_id} doesn't have quality criteria settings")
        
        # Start with perfect score
        score = 100.0
        
        # Factors with weights
        factor_scores = {}
        
        # 1. Timeframe alignment (30%)
        alignment_weight = strategy.quality_criteria.timeframe_alignment_weight or 0.3
        alignment_score = timeframe_analysis.alignment_score * 100
        factor_scores["timeframe_alignment"] = alignment_score
        
        # 2. Trend strength (20%)
        trend_weight = strategy.quality_criteria.trend_strength_weight or 0.2
        trend_score = 100
        
        # Reduce score for adverse conditions
        if market_state.is_creeper_move:
            trend_score -= 50  # Major penalty for creeper moves
        
        if market_state.price_struggling_near_ma:
            trend_score -= 30  # Penalty for price struggling with MA
        
        if not market_state.has_two_day_trend and strategy.quality_criteria.a_plus_requires_two_day_trend:
            trend_score -= 30  # Penalty for missing two-day trend
        
        if market_state.trend_phase != TrendPhase.MIDDLE:
            trend_score -= 25  # Penalty for not being in middle phase
            
        if market_state.is_railroad_trend:
            trend_score += 15  # Bonus for railroad trend
        
        factor_scores["trend_strength"] = max(0, trend_score)
        
        # 3. Entry technique quality (15%)
        entry_weight = strategy.quality_criteria.entry_technique_weight or 0.15
        entry_score = 100
        
        # Proximity to key level or MA
        if entry_data.get("near_key_level", False):
            entry_score += 10
        elif not entry_data.get("near_ma", False) and strategy.quality_criteria.a_plus_requires_entry_near_ma:
            entry_score -= 40  # Major penalty for not being near MA
        
        if entry_data.get("clean_entry", False):
            entry_score += 10
        
        factor_scores["entry_quality"] = min(100, max(0, entry_score))
        
        # 4. Proximity to key level (20%)
        level_weight = strategy.quality_criteria.proximity_to_key_level_weight or 0.2
        level_score = 100
        
        if not entry_data.get("near_key_level", False) and not entry_data.get("near_ma", False):
            level_score -= 50  # Major penalty for not being near any key level
        
        factor_scores["key_level_proximity"] = max(0, level_score)
        
        # 5. Risk/reward ratio (15%)
        rr_weight = strategy.quality_criteria.risk_reward_weight or 0.15
        rr_score = 100
        
        # Get risk/reward ratio
        risk_reward = entry_data.get("risk_reward", 1.0)
        
        # Score based on R:R ratio
        if risk_reward < 1.0:
            rr_score = 0
        elif risk_reward < 1.5:
            rr_score = 40
        elif risk_reward < 2.0:
            rr_score = 70
        elif risk_reward < 3.0:
            rr_score = 90
        
        factor_scores["risk_reward"] = rr_score
        # Calculate weighted score
        weighted_score = (
            alignment_score * alignment_weight +
            trend_score * trend_weight +
            entry_score * entry_weight +
            level_score * level_weight +
            rr_score * rr_weight
        )
        
        # Additional penalties for critical issues
        
        # 1. Institutional fight in progress
        if market_state.institutional_fight_in_progress:
            weighted_score *= 0.7  # 30% penalty
        
        # 2. A+ criteria not met
        if strategy.quality_criteria.a_plus_requires_all_timeframes and not timeframe_analysis.aligned:
            weighted_score = min(weighted_score, strategy.quality_criteria.a_min_score - 1)
            
        if (strategy.quality_criteria.a_plus_requires_entry_near_ma and 
            not entry_data.get("near_ma", False)):
            weighted_score = min(weighted_score, strategy.quality_criteria.a_min_score - 1)
            
        if (strategy.quality_criteria.a_plus_requires_two_day_trend and 
            not market_state.has_two_day_trend):
            weighted_score = min(weighted_score, strategy.quality_criteria.a_min_score - 1)
        
        # Determine grade based on score
        grade = SetupQualityGrade.F
        if weighted_score >= strategy.quality_criteria.a_plus_min_score:
            grade = SetupQualityGrade.A_PLUS
        elif weighted_score >= strategy.quality_criteria.a_min_score:
            grade = SetupQualityGrade.A
        elif weighted_score >= strategy.quality_criteria.b_min_score:
            grade = SetupQualityGrade.B
        elif weighted_score >= strategy.quality_criteria.c_min_score:
            grade = SetupQualityGrade.C
        elif weighted_score >= strategy.quality_criteria.d_min_score:
            grade = SetupQualityGrade.D
        
        # Determine position size based on grade
        position_size = 0
        risk_percent = 0.0
        
        if strategy.quality_criteria.position_sizing_rules:
            if grade == SetupQualityGrade.A_PLUS and "a_plus" in strategy.quality_criteria.position_sizing_rules:
                sizing_rule = strategy.quality_criteria.position_sizing_rules["a_plus"]
                position_size = sizing_rule.get("lots", 0)
                risk_percent = sizing_rule.get("risk_percent", 0.0)
            elif grade == SetupQualityGrade.A and "a" in strategy.quality_criteria.position_sizing_rules:
                sizing_rule = strategy.quality_criteria.position_sizing_rules["a"]
                position_size = sizing_rule.get("lots", 0)
                risk_percent = sizing_rule.get("risk_percent", 0.0)
            elif grade == SetupQualityGrade.B and "b" in strategy.quality_criteria.position_sizing_rules:
                sizing_rule = strategy.quality_criteria.position_sizing_rules["b"]
                position_size = sizing_rule.get("lots", 0)
                risk_percent = sizing_rule.get("risk_percent", 0.0)
            elif grade == SetupQualityGrade.C and "c" in strategy.quality_criteria.position_sizing_rules:
                sizing_rule = strategy.quality_criteria.position_sizing_rules["c"]
                position_size = sizing_rule.get("lots", 0)
                risk_percent = sizing_rule.get("risk_percent", 0.0)
        
        # Check if auto-trading is enabled for this grade
        can_auto_trade = False
        if grade == SetupQualityGrade.A_PLUS and strategy.quality_criteria.auto_trade_a_plus:
            can_auto_trade = True
        elif grade == SetupQualityGrade.A and strategy.quality_criteria.auto_trade_a:
            can_auto_trade = True
        elif grade == SetupQualityGrade.B and hasattr(strategy.quality_criteria, 'auto_trade_b') and strategy.quality_criteria.auto_trade_b:
            can_auto_trade = True
        
        return SetupQualityResult(
            strategy_id=strategy_id,
            grade=grade,
            score=weighted_score,
            factor_scores=factor_scores,
            position_size=position_size,
            risk_percent=risk_percent,
            can_auto_trade=can_auto_trade,
            analysis_comments=[
                f"Timeframe alignment score: {alignment_score:.1f}%",
                f"Trend strength score: {trend_score:.1f}%",
                f"Entry quality score: {entry_score:.1f}%",
                f"Key level proximity score: {level_score:.1f}%",
                f"Risk/reward score: {rr_score:.1f}%",
                f"Final setup grade: {grade.value}"
            ]
        )
    #
    # Signal Generation Methods
    #
    
    def generate_signal(self, strategy_id: int, timeframe_analysis: TimeframeAnalysisResult,
                      market_state: MarketStateAnalysis, setup_quality: SetupQualityResult,
                      market_data: Dict[TimeframeValue, Dict], instrument: str,
                      direction: Direction) -> Signal:
        """
        Generate a trading signal based on strategy criteria.
        
        Args:
            strategy_id: ID of the strategy
            timeframe_analysis: Timeframe analysis results
            market_state: Market state analysis
            setup_quality: Setup quality evaluation results
            market_data: Dictionary of market data by timeframe
            instrument: Trading instrument
            direction: Signal direction
            
        Returns:
            Generated Signal instance
            
        Raises:
            ValueError: If strategy not found or signal cannot be generated
        """
        strategy = self.get_strategy(strategy_id)
        if not strategy.entry_exit_settings:
            raise ValueError(f"Strategy {strategy_id} doesn't have entry/exit settings")
        
        # Skip if we don't have required data
        primary_tf = TimeframeValue.ONE_HOUR
        if strategy.multi_timeframe_settings and strategy.multi_timeframe_settings.primary_timeframe:
            primary_tf = strategy.multi_timeframe_settings.primary_timeframe
            
        entry_tf = TimeframeValue.FIVE_MIN
        if strategy.multi_timeframe_settings and strategy.multi_timeframe_settings.entry_timeframe:
            entry_tf = strategy.multi_timeframe_settings.entry_timeframe
        
        # Verify we have data for both timeframes
        if primary_tf not in market_data or entry_tf not in market_data:
            raise ValueError(f"Missing required market data for timeframes: {primary_tf.value} or {entry_tf.value}")
        
        # Get current price
        primary_data = market_data[primary_tf]
        entry_data = market_data[entry_tf]
        
        current_price = entry_data.get("close", [])[-1] if entry_data.get("close") else None
        if current_price is None:
            raise ValueError("Cannot determine current price from market data")
        
        # Calculate entry parameters
        entry_price = current_price
        
        # Calculate stop loss based on entry technique
        stop_loss_price = self._calculate_stop_loss(
            strategy, entry_data, direction, entry_price,
            strategy.entry_exit_settings.primary_entry_technique
        )
        
        # Calculate take profit
        take_profit_price = self._calculate_take_profit(
            strategy, entry_price, stop_loss_price, direction
        )
        
        # Calculate position size - either from setup quality or risk limits
        position_size = setup_quality.position_size
        
        # Calculate risk amount in INR
        risk_points = abs(entry_price - stop_loss_price)
        
        # Simplified INR calculation - in real implementation this would use proper contract specs
        lot_size = 50  # Example: 50 INR per point per lot
        risk_amount = risk_points * lot_size * position_size
        
        # Calculate risk/reward ratio
        reward_points = abs(take_profit_price - entry_price)
        risk_reward_ratio = reward_points / risk_points if risk_points > 0 else 0
        
        # Calculate signal confidence based on setup quality
        confidence = setup_quality.score / 100.0
        # Create the signal
        signal = Signal(
            strategy_id=strategy_id,
            instrument=instrument,
            direction=direction,
            signal_type=self._determine_signal_type(strategy, market_state),
            entry_price=entry_price,
            entry_time=datetime.utcnow(),
            entry_timeframe=entry_tf,
            entry_technique=strategy.entry_exit_settings.primary_entry_technique,
            take_profit_price=take_profit_price,
            stop_loss_price=stop_loss_price,
            trailing_stop=(strategy.entry_exit_settings.trailing_stop_method != None),
            position_size=position_size,
            risk_reward_ratio=risk_reward_ratio,
            risk_amount=risk_amount,
            setup_quality=setup_quality.grade,
            setup_score=setup_quality.score,
            confidence=confidence,
            market_state=market_state.market_state,
            trend_phase=market_state.trend_phase,
            is_active=True,
            is_executed=False,
            timeframe_alignment_score=timeframe_analysis.alignment_score,
            primary_timeframe_aligned=timeframe_analysis.aligned,
            institutional_footprint_detected=market_state.accumulation_detected,
            bos_detected=market_state.bos_detected
        )
        
        # Save to database (optional, can also let caller handle this)
        self.db.add(signal)
        self.db.commit()
        
        return signal
    
    def _calculate_stop_loss(self, strategy: Strategy, market_data: Dict, 
                           direction: Direction, entry_price: float,
                           entry_technique: EntryTechnique) -> float:
        """
        Calculate stop loss based on entry technique and market conditions.
        
        Args:
            strategy: Strategy instance
            market_data: Market data for entry timeframe
            direction: Trade direction
            entry_price: Entry price
            entry_technique: Entry technique being used
            
        Returns:
            Stop loss price
        """
        # Default stop distance (in case no specific rule applies)
        default_stop_distance = 20  # points
        
        # Get recent price data
        high_values = market_data.get("high", [])
        low_values = market_data.get("low", [])
        
        if not high_values or not low_values or len(high_values) < 5 or len(low_values) < 5:
            # Not enough data, use default distance
            if direction == Direction.LONG:
                return entry_price - default_stop_distance
            else:
                return entry_price + default_stop_distance
        # Get stop placement based on entry technique
        if direction == Direction.LONG:
            if entry_technique == EntryTechnique.GREEN_BAR_AFTER_PULLBACK:
                # Place stop below the green bar
                if strategy.entry_exit_settings.green_bar_sl_placement == "below_bar":
                    return low_values[-1]  # Below current bar
                else:
                    return low_values[-2]  # Below previous bar
            
            elif entry_technique == EntryTechnique.BREAKOUT_PULLBACK_LONG:
                # Place stop below the pullback low
                recent_low = min(low_values[-3:])
                return recent_low
            
            elif entry_technique == EntryTechnique.MA_BOUNCE_LONG:
                # Place stop below the MA
                ma_value = market_data.get("ma21", [])[-1] if market_data.get("ma21") else None
                if ma_value:
                    return ma_value - 5  # Just below MA
                
            elif entry_technique == EntryTechnique.BOS_ENTRY_LONG:
                # Place stop below the recent structure low
                recent_low = min(low_values[-5:])
                return recent_low
            
            elif entry_technique == EntryTechnique.DISCOUNT_ZONE_LONG:
                # Place stop below the discount zone
                return entry_price * 0.99  # 1% below entry
        
        elif direction == Direction.SHORT:
            if entry_technique == EntryTechnique.RED_BAR_AFTER_RALLY:
                # Place stop above the red bar
                if strategy.entry_exit_settings.red_bar_sl_placement == "above_bar":
                    return high_values[-1]  # Above current bar
                else:
                    return high_values[-2]  # Above previous bar
            
            elif entry_technique == EntryTechnique.BREAKOUT_PULLBACK_SHORT:
                # Place stop above the pullback high
                recent_high = max(high_values[-3:])
                return recent_high
            
            elif entry_technique == EntryTechnique.MA_BOUNCE_SHORT:
                # Place stop above the MA
                ma_value = market_data.get("ma21", [])[-1] if market_data.get("ma21") else None
                if ma_value:
                    return ma_value + 5  # Just above MA
                
            elif entry_technique == EntryTechnique.BOS_ENTRY_SHORT:
                # Place stop above the recent structure high
                recent_high = max(high_values[-5:])
                return recent_high
            
            elif entry_technique == EntryTechnique.PREMIUM_ZONE_SHORT:
                # Place stop above the premium zone
                return entry_price * 1.01  # 1% above entry
        
        # Default fallback - use recent highs/lows
        if direction == Direction.LONG:
            recent_low = min(low_values[-3:])
            return recent_low
        else:
            recent_high = max(high_values[-3:])
            return recent_high
    def _calculate_take_profit(self, strategy: Strategy, entry_price: float, 
                             stop_loss_price: float, direction: Direction) -> float:
        """
        Calculate take profit level based on strategy settings and entry/stop prices.
        
        Args:
            strategy: Strategy instance
            entry_price: Entry price
            stop_loss_price: Stop loss price
            direction: Trade direction
            
        Returns:
            Take profit price
        """
        if not strategy.entry_exit_settings:
            # Default to 2:1 reward:risk ratio
            risk_distance = abs(entry_price - stop_loss_price)
            if direction == Direction.LONG:
                return entry_price + (risk_distance * 2)
            else:
                return entry_price - (risk_distance * 2)
        
        # Calculate based on profit target method
        profit_target_method = strategy.entry_exit_settings.profit_target_method
        
        if profit_target_method == ProfitTargetMethod.FIXED_POINTS:
            # Fixed point target (Rikk's specific approach)
            profit_points = strategy.entry_exit_settings.profit_target_points or 25
            
            if direction == Direction.LONG:
                return entry_price + profit_points
            else:
                return entry_price - profit_points
        
        elif profit_target_method == ProfitTargetMethod.ATR_MULTIPLE:
            # ATR multiple - would need ATR data in real implementation
            atr_multiple = strategy.entry_exit_settings.profit_target_atr_multiple or 2.0
            atr_value = 10  # Placeholder, would get from market data
            
            if direction == Direction.LONG:
                return entry_price + (atr_value * atr_multiple)
            else:
                return entry_price - (atr_value * atr_multiple)
        
        else:
            # Default to fixed risk:reward
            risk_distance = abs(entry_price - stop_loss_price)
            reward_distance = risk_distance * 2  # 2:1 reward:risk ratio
            
            if direction == Direction.LONG:
                return entry_price + reward_distance
            else:
                return entry_price - reward_distance
    
    def _determine_signal_type(self, strategy: Strategy, market_state: MarketStateAnalysis) -> str:
        """
        Determine signal type based on market state.
        
        Args:
            strategy: Strategy instance
            market_state: Market state analysis
            
        Returns:
            Signal type string
        """
        # Check for BOS
        if market_state.bos_detected:
            return "breakout"
        
        # Check trend phase
        if market_state.trend_phase == TrendPhase.MIDDLE:
            return "trend_continuation"
        elif market_state.trend_phase == TrendPhase.EARLY:
            return "trend_start"
        elif market_state.trend_phase == TrendPhase.LATE:
            return "reversal"
        
        # Check market state
        if market_state.market_state == MarketStateRequirement.TRENDING_UP:
            return "uptrend"
        elif market_state.market_state == MarketStateRequirement.TRENDING_DOWN:
            return "downtrend"
        elif market_state.market_state == MarketStateRequirement.RANGE_BOUND:
            return "range_play"
        
        # Default
        return "unclassified"
    #
    # Trade Management Methods
    #
    
    def execute_signal(self, signal_id: int, execution_price: float, 
                     execution_time: Optional[datetime] = None,
                     user_id: Optional[int] = None) -> Trade:
        """
        Execute a trading signal.
        
        Args:
            signal_id: ID of the signal to execute
            execution_price: Actual execution price
            execution_time: Execution timestamp (defaults to now)
            user_id: ID of the user executing the trade
            
        Returns:
            Trade instance
            
        Raises:
            ValueError: If signal not found or already executed
        """
        # Get signal
        signal = self.db.query(Signal).filter(Signal.id == signal_id).first()
        if not signal:
            raise ValueError(f"Signal with ID {signal_id} not found")
        
        if signal.is_executed:
            raise ValueError(f"Signal with ID {signal_id} already executed")
        
        # Get strategy for additional settings
        strategy = self.get_strategy(signal.strategy_id)
        
        # Calculate execution details
        slippage = abs(execution_price - signal.entry_price)
        slippage_percent = slippage / signal.entry_price if signal.entry_price != 0 else 0
        
        # Calculate trade risk in points and INR
        risk_points = abs(execution_price - signal.stop_loss_price)
        
        # Simplified calculation - in real implementation would use contract specs
        lot_size = 50  # Example: 50 INR per point per lot
        risk_inr = risk_points * lot_size * signal.position_size
        account_size = 100000  # Example account size in INR
        risk_percent = (risk_inr / account_size) * 100
        
        # Calculate risk/reward ratio
        reward_points = abs(signal.take_profit_price - execution_price)
        risk_reward = reward_points / risk_points if risk_points > 0 else 0
        
        # Create trade record
        trade = Trade(
            strategy_id=signal.strategy_id,
            signal_id=signal.id,
            instrument=signal.instrument,
            direction=signal.direction,
            entry_price=execution_price,
            entry_time=execution_time or datetime.utcnow(),
            position_size=signal.position_size,
            commission=self._calculate_commission(signal.position_size, execution_price),
            taxes=self._calculate_taxes(signal.position_size, execution_price),
            slippage=slippage,
            initial_risk_points=risk_points,
            initial_risk_inr=risk_inr,
            initial_risk_percent=risk_percent,
            risk_reward_planned=risk_reward,
            setup_quality=signal.setup_quality,
            setup_score=signal.setup_score,
            is_spread_trade=signal.is_spread_trade,
            spread_type=signal.spread_type,
            user_id=user_id or strategy.user_id
        )
        
        # Update signal as executed
        signal.is_executed = True
        signal.execution_time = trade.entry_time
        
        # Save to database
        self.db.add(trade)
        self.db.add(signal)
        self.db.commit()
        self.db.refresh(trade)
        
        return trade
    def _calculate_commission(self, position_size: int, execution_price: float) -> float:
        """
        Calculate commission for a trade.
        
        Args:
            position_size: Number of lots/contracts
            execution_price: Execution price
            
        Returns:
            Commission amount in INR
        """
        # Simplified calculation - in real implementation would use broker's commission structure
        return position_size * 20.0  # 20 INR per lot
    
    def _calculate_taxes(self, position_size: int, execution_price: float) -> float:
        """
        Calculate taxes for a trade.
        
        Args:
            position_size: Number of lots/contracts
            execution_price: Execution price
            
        Returns:
            Tax amount in INR
        """
        # Simplified calculation - in real implementation would use proper tax rates
        contract_value = position_size * execution_price * 50  # Assuming 50 INR per point per lot
        return contract_value * 0.0005  # 0.05% transaction tax
    
    def close_trade(self, trade_id: int, exit_price: float, 
                  exit_time: Optional[datetime] = None,
                  exit_reason: str = "manual") -> Trade:
        """
        Close a trade.
        
        Args:
            trade_id: ID of the trade to close
            exit_price: Exit price
            exit_time: Exit timestamp (defaults to now)
            exit_reason: Reason for closing the trade
            
        Returns:
            Updated Trade instance
            
        Raises:
            ValueError: If trade not found or already closed
        """
        # Get trade
        trade = self.db.query(Trade).filter(Trade.id == trade_id).first()
        if not trade:
            raise ValueError(f"Trade with ID {trade_id} not found")
        
        if trade.exit_price is not None and trade.exit_time is not None:
            raise ValueError(f"Trade with ID {trade_id} already closed")
        
        # Update trade with exit details
        trade.exit_price = exit_price
        trade.exit_time = exit_time or datetime.utcnow()
        trade.exit_reason = exit_reason
        
        # Calculate profit/loss
        point_multiplier = -1 if trade.direction == Direction.SHORT else 1
        trade.profit_loss_points = point_multiplier * (exit_price - trade.entry_price)
        
        # Simplified calculation - in real implementation would use contract specs
        lot_size = 50  # Example: 50 INR per point per lot
        trade.profit_loss_inr = trade.profit_loss_points * lot_size * trade.position_size
        
        # Calculate total costs
        exit_commission = self._calculate_commission(trade.position_size, exit_price)
        exit_taxes = self._calculate_taxes(trade.position_size, exit_price)
        
        trade.commission = (trade.commission or 0) + exit_commission
        trade.taxes = (trade.taxes or 0) + exit_taxes
        trade.total_costs = (trade.commission or 0) + (trade.taxes or 0) + (trade.slippage or 0)
        
        # Calculate holding period
        if trade.entry_time:
            delta = trade.exit_time - trade.entry_time
            trade.holding_period_minutes = int(delta.total_seconds() / 60)
        
        # Calculate actual risk/reward
        if trade.initial_risk_points and trade.initial_risk_points > 0:
            trade.actual_risk_reward = abs(trade.profit_loss_points / trade.initial_risk_points)
        
        # Save to database
        self.db.add(trade)
        self.db.commit()
        self.db.refresh(trade)
        
        return trade
    #
    # Feedback and Continuous Improvement Methods
    #
    
    def record_feedback(self, strategy_id: int, feedback_data: FeedbackCreate, 
                      trade_id: Optional[int] = None, user_id: int = None) -> TradeFeedback:
        """
        Record feedback for strategy improvement.
        
        Args:
            strategy_id: ID of the strategy
            feedback_data: Feedback data
            trade_id: Optional ID of related trade
            user_id: ID of the user providing feedback
            
        Returns:
            Created TradeFeedback instance
        """
        # Create feedback record
        feedback = TradeFeedback(
            strategy_id=strategy_id,
            trade_id=trade_id,
            feedback_type=feedback_data.feedback_type,
            title=feedback_data.title,
            description=feedback_data.description,
            file_path=feedback_data.file_path,
            file_type=feedback_data.file_type,
            tags=feedback_data.tags,
            improvement_category=feedback_data.improvement_category,
            applies_to_setup=feedback_data.applies_to_setup,
            applies_to_entry=feedback_data.applies_to_entry,
            applies_to_exit=feedback_data.applies_to_exit,
            applies_to_risk=feedback_data.applies_to_risk,
            pre_trade_conviction_level=feedback_data.pre_trade_conviction_level,
            emotional_state_rating=feedback_data.emotional_state_rating,
            lessons_learned=feedback_data.lessons_learned,
            action_items=feedback_data.action_items
        )
        
        # Save to database
        self.db.add(feedback)
        self.db.commit()
        self.db.refresh(feedback)
        
        return feedback
        
    def list_feedback(self, strategy_id: int, limit: int = 50, offset: int = 0) -> List[TradeFeedback]:
        """
        List feedback for a strategy.
        
        Args:
            strategy_id: ID of the strategy
            limit: Maximum number of results
            offset: Pagination offset
            
        Returns:
            List of TradeFeedback instances
        """
        query = self.db.query(TradeFeedback).filter(
            TradeFeedback.strategy_id == strategy_id
        ).order_by(TradeFeedback.created_at.desc())
        
        return query.offset(offset).limit(limit).all()
    
    def apply_feedback(self, feedback_id: int, strategy_id: int, user_id: int) -> Strategy:
        """
        Apply feedback to strategy by creating a new version with improvements.
        
        Args:
            feedback_id: ID of the feedback to apply
            strategy_id: ID of the strategy
            user_id: ID of the user applying the feedback
            
        Returns:
            Updated Strategy instance
            
        Raises:
            ValueError: If feedback or strategy not found
        """
        # Get feedback and strategy
        feedback = self.db.query(TradeFeedback).filter(TradeFeedback.id == feedback_id).first()
        if not feedback:
            raise ValueError(f"Feedback with ID {feedback_id} not found")
        
        strategy = self.get_strategy(strategy_id)
        
        # Mark feedback as applied
        feedback.has_been_applied = True
        feedback.applied_date = datetime.utcnow()
        feedback.applied_to_version_id = strategy.version + 1
        
        # Create a new strategy version
        new_version = strategy.create_new_version()
        new_version.updated_by_id = user_id
        new_version.notes = f"Applied feedback: {feedback.title}"
        
        # Save changes
        self.db.add(feedback)
        self.db.add(new_version)
        self.db.commit()
        
        return new_version
    def analyze_performance(self, strategy_id: int, start_date: Optional[datetime] = None, 
                          end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Analyze strategy performance.
        
        Args:
            strategy_id: ID of the strategy
            start_date: Optional start date for analysis
            end_date: Optional end date for analysis
            
        Returns:
            Dictionary with performance metrics
        """
        # Get strategy
        strategy = self.get_strategy(strategy_id)
        
        # Build query
        query = self.db.query(Trade).filter(Trade.strategy_id == strategy_id)
        
        # Apply date filters
        if start_date:
            query = query.filter(Trade.entry_time >= start_date)
        if end_date:
            query = query.filter(Trade.entry_time <= end_date)
        
        # Execute query
        trades = query.all()
        
        # Calculate performance metrics
        total_trades = len(trades)
        if total_trades == 0:
            return {
                "strategy_id": strategy_id,
                "total_trades": 0,
                "message": "No trades found for this strategy"
            }
        
        # Count winning and losing trades
        winning_trades = [t for t in trades if t.profit_loss_inr and t.profit_loss_inr > 0]
        losing_trades = [t for t in trades if t.profit_loss_inr and t.profit_loss_inr < 0]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        
        # Calculate win rate
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        # Calculate profit metrics
        total_profit = sum(t.profit_loss_inr for t in winning_trades) if winning_trades else 0
        total_loss = sum(t.profit_loss_inr for t in losing_trades) if losing_trades else 0
        net_profit = total_profit + total_loss
        
        # Calculate average metrics
        avg_win = total_profit / win_count if win_count > 0 else 0
        avg_loss = total_loss / loss_count if loss_count > 0 else 0
        
        # Calculate profit factor
        profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
        
        # Calculate setup quality metrics
        trades_by_grade = {}
        for grade in SetupQualityGrade:
            grade_trades = [t for t in trades if t.setup_quality == grade]
            if grade_trades:
                grade_profit = sum(t.profit_loss_inr for t in grade_trades)
                grade_win_rate = len([t for t in grade_trades if t.profit_loss_inr and t.profit_loss_inr > 0]) / len(grade_trades)
                
                trades_by_grade[grade.value] = {
                    "count": len(grade_trades),
                    "profit": grade_profit,
                    "win_rate": grade_win_rate
                }
                
        return {
            "strategy_id": strategy_id,
            "total_trades": total_trades,
            "win_count": win_count,
            "loss_count": loss_count,
            "win_rate": win_rate,
            "total_profit_inr": net_profit,
            "avg_win_inr": avg_win,
            "avg_loss_inr": avg_loss,
            "profit_factor": profit_factor,
            "trades_by_grade": trades_by_grade,
            "analysis_period": {
                "start": start_date or min(t.entry_time for t in trades if t.entry_time),
                "end": end_date or max(t.entry_time for t in trades if t.entry_time)
            }
        }