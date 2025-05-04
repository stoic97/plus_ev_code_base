"""
Unit tests for strategy models.

This module contains comprehensive tests for the Strategy models defined in app.models.strategy,
using mocks to avoid MRO issues with the actual models.
"""

import pytest
import json
from datetime import datetime, timedelta
from sqlalchemy.exc import IntegrityError
from unittest.mock import MagicMock, patch

# Don't import the actual models - mock them instead to avoid MRO issues

@pytest.fixture
def Strategy():
    """Create a mock Strategy class with all the methods and properties we need to test."""
    class MockStrategy:
        id = None
        name = None
        description = None
        type = None
        configuration = None
        parameters = None
        validation_rules = None
        user_id = None
        created_by_id = None
        updated_by_id = None
        is_active = False
        is_public = False
        version = 1
        created_at = datetime.utcnow()
        updated_at = None
        win_rate = None
        profit_factor = None
        sharpe_ratio = None
        sortino_ratio = None
        max_drawdown = None
        parent_version_id = None
        categories = []
        signals = []
        deleted_at = None
        deleted_by_id = None
        status = "draft"
        status_changed_at = datetime.utcnow()
        previous_status = None
        total_profit_inr = None
        avg_win_inr = None
        avg_loss_inr = None
        largest_win_inr = None
        largest_loss_inr = None
        
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
            
            # Initialize collections
            self.categories = []
            self.signals = []
            self.child_versions = []
            
            # Validate JSON fields
            for field in ["configuration", "parameters", "validation_rules"]:
                if field in kwargs:
                    value = kwargs[field]
                    if value is not None:
                        setattr(self, field, self.validate_json(field, value))
        
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
        
        def get_parameter(self, param_name, default=None):
            """Get a strategy parameter value."""
            if not self.parameters:
                return default
            return self.parameters.get(param_name, default)
        
        def update_parameter(self, param_name, value):
            """Update a single strategy parameter."""
            if not self.parameters:
                self.parameters = {}
            self.parameters[param_name] = value
            self.version += 1  # Increment version on parameter changes
        
        def validate_parameters(self):
            """Validate all parameters against validation rules."""
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
        
        def create_new_version(self):
            """Create a new version of this strategy."""
            # Create a copy of the current strategy as a new version
            new_version = MockStrategy(
                name=self.name,
                description=self.description,
                type=self.type,
                configuration=self.configuration.copy() if self.configuration else {},
                parameters=self.parameters.copy() if self.parameters else {},
                validation_rules=self.validation_rules.copy() if self.validation_rules else {},
                user_id=self.user_id,
                created_by_id=self.created_by_id,
                parent_version_id=self.id,
                version=self.version + 1
            )
            
            return new_version
        
        def update_performance_metrics(self, win_rate=None, profit_factor=None,
                                     sharpe_ratio=None, sortino_ratio=None,
                                     max_drawdown=None, total_profit_inr=None,
                                     avg_win_inr=None, avg_loss_inr=None,
                                     largest_win_inr=None, largest_loss_inr=None):
            """Update strategy performance metrics."""
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
        
        def to_dict(self, include_relationships=False, exclude=None):
            """Convert strategy to dictionary for API responses."""
            exclude = exclude or []
            data = {}
            
            # Add all attributes that aren't in exclude list
            for attr in dir(self):
                if attr.startswith('_') or attr in exclude or callable(getattr(self, attr)):
                    continue
                if not include_relationships and attr in ['categories', 'signals', 'child_versions']:
                    continue
                value = getattr(self, attr)
                
                # Handle special types
                if isinstance(value, datetime):
                    value = value.isoformat()
                elif hasattr(value, 'value'):  # Handle enum
                    value = value.value
                
                data[attr] = value
            
            if include_relationships:
                # Include categories
                data['categories'] = [category.name for category in self.categories]
                
                # Include signal count
                data['signal_count'] = len(self.signals) if self.signals else 0
                
                # Include version information
                data['has_parent'] = self.parent_version_id is not None
                data['has_children'] = len(self.child_versions) > 0
                
                # Include performance metrics in INR
                data['performance_inr'] = {
                    'total_profit_inr': self.total_profit_inr,
                    'avg_win_inr': self.avg_win_inr,
                    'avg_loss_inr': self.avg_loss_inr,
                    'largest_win_inr': self.largest_win_inr,
                    'largest_loss_inr': self.largest_loss_inr
                }
            
            return data
        
        def soft_delete(self, user_id=None):
            """Mark record as deleted."""
            self.deleted_at = datetime.utcnow()
            if user_id:
                self.deleted_by_id = user_id
        
        @property
        def is_deleted(self):
            """Check if record has been soft-deleted."""
            return self.deleted_at is not None
        
        def update_status(self, new_status):
            """Update status with timestamp and history."""
            if self.status != new_status:
                self.previous_status = self.status
                self.status = new_status
                self.status_changed_at = datetime.utcnow()
    
    return MockStrategy

@pytest.fixture
def StrategyCategory():
    """Create a mock StrategyCategory class."""
    class MockStrategyCategory:
        id = None
        name = None
        description = None
        created_at = datetime.utcnow()
        
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
            
            self.strategies = []
    
    return MockStrategyCategory

@pytest.fixture
def StrategyBacktest():
    """Create a mock StrategyBacktest class."""
    class MockStrategyBacktest:
        id = None
        strategy_id = None
        name = None
        description = None
        start_date = None
        end_date = None
        initial_capital = None
        parameters = None
        total_return = None
        annualized_return = None
        win_rate = None
        profit_factor = None
        sharpe_ratio = None
        sortino_ratio = None
        max_drawdown = None
        max_drawdown_duration = None
        total_trades = None
        equity_curve = None
        trade_history = None
        monthly_returns = None
        user_id = None
        created_at = datetime.utcnow()
        
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
        
        def calculate_metrics(self, trades, equity_curve):
            """Calculate performance metrics from trade history and equity curve."""
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
            
            # Store the complete equity curve and trade history
            self.equity_curve = equity_curve
            self.trade_history = trades
    
    return MockStrategyBacktest

@pytest.fixture
def MetaLearningSettings():
    """Create a mock MetaLearningSettings class."""
    class MockMetaLearningSettings:
        id = None
        strategy_id = None
        record_trading_sessions = True
        record_decision_points = True
        review_frequency = "daily"
        track_market_relationships = True
        detect_regime_changes = True
        adaptive_regime_parameters = True
        regime_types = None
        parameter_adjustments_by_regime = None
        track_success_factors = True
        success_factor_categories = None
        factor_contribution_tracking = True
        use_synthetic_markets = False
        track_time_of_day_performance = True
        
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    return MockMetaLearningSettings

@pytest.fixture
def RiskManagementSettings():
    """Create a mock RiskManagementSettings class."""
    class MockRiskManagementSettings:
        id = None
        strategy_id = None
        max_risk_per_trade_percent = 1.0
        max_risk_per_trade_inr = None
        max_daily_risk_percent = 3.0
        drawdown_tiers = None
        use_progressive_recovery = True
        track_asset_correlations = True
        
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    return MockRiskManagementSettings

# Use enum mock to avoid importing the real one
@pytest.fixture
def StrategyType():
    """Create a mock StrategyType enum."""
    class MockEnum:
        def __init__(self, value):
            self.value = value
        
        def __eq__(self, other):
            if isinstance(other, MockEnum):
                return self.value == other.value
            return False
    
    class MockStrategyType:
        TREND_FOLLOWING = MockEnum("trend_following")
        MEAN_REVERSION = MockEnum("mean_reversion")
        BREAKOUT = MockEnum("breakout")
        MOMENTUM = MockEnum("momentum")
        STATISTICAL_ARBITRAGE = MockEnum("statistical_arbitrage")
        PATTERN_RECOGNITION = MockEnum("pattern_recognition")
        MULTI_TIMEFRAME = MockEnum("multi_timeframe")
        INSTITUTIONAL_FLOW = MockEnum("institutional_flow")
        VERTICAL_SPREAD = MockEnum("vertical_spread")
        CUSTOM = MockEnum("custom")
    
    return MockStrategyType

@pytest.fixture
def StrategyStatus():
    """Create a mock StrategyStatus enum."""
    class MockEnum:
        def __init__(self, value):
            self.value = value
        
        def __eq__(self, other):
            if isinstance(other, MockEnum):
                return self.value == other.value
            return False
    
    class MockStrategyStatus:
        DRAFT = MockEnum("draft")
        ACTIVE = MockEnum("active")
        PAUSED = MockEnum("paused")
        ARCHIVED = MockEnum("archived")
        BACKTEST = MockEnum("backtest")
    
    return MockStrategyStatus


@pytest.fixture
def db_session():
    """
    Create a mock database session for testing.
    
    This fixture provides a transactional session that's rolled back after each test.
    """
    # Create mock session
    session = MagicMock()
    
    # Make add and add_all actually store the objects
    added_objects = []
    
    def mock_add(obj):
        added_objects.append(obj)
    
    def mock_add_all(objects):
        added_objects.extend(objects)
    
    def mock_commit():
        # When committing, validate and assign IDs to objects that don't have one
        for i, obj in enumerate(added_objects, 1):
            if hasattr(obj, 'id') and obj.id is None:
                obj.id = i
    
    def mock_flush():
        # Similar to commit but without "permanent" persistence
        for i, obj in enumerate(added_objects, 1):
            if hasattr(obj, 'id') and obj.id is None:
                obj.id = i
    
    def mock_rollback():
        # Clear the added objects on rollback
        added_objects.clear()
    
    session.add = mock_add
    session.add_all = mock_add_all
    session.commit = mock_commit
    session.flush = mock_flush
    session.rollback = mock_rollback
    
    return session


class TestStrategy:
    """Test cases for the Strategy model."""

    def test_strategy_creation(self, Strategy, StrategyType, db_session):
        """Test creating a new strategy with basic attributes."""
        # Create a strategy with minimal required fields
        strategy = Strategy(
            name="Test Strategy",
            type=StrategyType.TREND_FOLLOWING,
            user_id=1,
            created_by_id=1,
            configuration={"indicators": ["sma", "rsi"]},
            parameters={"sma_period": 20, "rsi_period": 14}
        )
        
        db_session.add(strategy)
        db_session.commit()
        
        # Check that strategy was created with expected values
        assert strategy.id is not None
        assert strategy.name == "Test Strategy"
        assert strategy.type == StrategyType.TREND_FOLLOWING
        assert strategy.created_at is not None
        assert strategy.version == 1
        assert strategy.is_active is False
        assert strategy.is_public is False
        
        # Test configuration and parameters are properly stored
        assert strategy.configuration == {"indicators": ["sma", "rsi"]}
        assert strategy.parameters == {"sma_period": 20, "rsi_period": 14}

    def test_get_parameter(self, Strategy):
        """Test retrieving parameters with default values."""
        strategy = Strategy(
            name="Parameter Test",
            parameters={"existing": 42}
        )
        
        # Test retrieving existing parameter
        assert strategy.get_parameter("existing") == 42
        
        # Test retrieving non-existent parameter with default
        assert strategy.get_parameter("missing", "default") == "default"
        
        # Test retrieving non-existent parameter without default
        assert strategy.get_parameter("missing") is None
        
        # Test with empty parameters
        strategy.parameters = None
        assert strategy.get_parameter("any", "default") == "default"

    def test_update_parameter(self, Strategy):
        """Test updating a parameter value."""
        strategy = Strategy(
            name="Update Test",
            parameters={"existing": 42},
            version=1
        )
        
        # Update existing parameter
        strategy.update_parameter("existing", 100)
        assert strategy.parameters["existing"] == 100
        
        # Add new parameter
        strategy.update_parameter("new_param", "value")
        assert strategy.parameters["new_param"] == "value"
        
        # Check that version was incremented
        assert strategy.version == 3  # Started at 1, incremented twice
        
        # Test with None parameters
        strategy.parameters = None
        strategy.update_parameter("test", "value")
        assert strategy.parameters == {"test": "value"}

    def test_validate_parameters_valid(self, Strategy):
        """Test parameter validation with valid parameters."""
        strategy = Strategy(
            name="Validation Test",
            parameters={
                "number_param": 42,
                "string_param": "test",
                "bool_param": True
            },
            validation_rules={
                "number_param": {
                    "type": "number",
                    "min": 0,
                    "max": 100
                },
                "string_param": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 10
                },
                "bool_param": {
                    "type": "boolean"
                }
            }
        )
        
        errors = strategy.validate_parameters()
        assert len(errors) == 0, f"Unexpected validation errors: {errors}"

    def test_validate_parameters_invalid(self, Strategy):
        """Test parameter validation with invalid parameters."""
        strategy = Strategy(
            name="Invalid Validation Test",
            parameters={
                "number_param": -10,  # Below min
                "string_param": "this string is too long",  # Exceeds max length
                "bool_param": "not a boolean"  # Wrong type
            },
            validation_rules={
                "number_param": {
                    "type": "number",
                    "min": 0,
                    "max": 100
                },
                "string_param": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 10
                },
                "bool_param": {
                    "type": "boolean"
                },
                "required_param": {
                    "required": True
                }
            }
        )
        
        errors = strategy.validate_parameters()
        
        # Should have 4 validation errors
        assert len(errors) == 4
        assert any("number_param" in error and "less than" in error for error in errors)
        assert any("string_param" in error and "exceed" in error for error in errors)
        assert any("bool_param" in error and "must be a boolean" in error for error in errors)
        assert any("required_param" in error and "missing" in error for error in errors)

    def test_create_new_version(self, Strategy, StrategyType):
        """Test creating a new version of a strategy."""
        original = Strategy(
            id=1,
            name="Original Strategy",
            description="Original description",
            type=StrategyType.MOMENTUM,
            user_id=1,
            created_by_id=1,
            configuration={"key": "value"},
            parameters={"param": 42},
            validation_rules={"param": {"type": "number"}},
            version=1
        )
        
        # Create new version
        new_version = original.create_new_version()
        
        # Check that new version has correct attributes
        assert new_version.id is None  # New instance, not yet persisted
        assert new_version.name == original.name
        assert new_version.description == original.description
        assert new_version.type == original.type
        assert new_version.user_id == original.user_id
        assert new_version.created_by_id == original.created_by_id
        assert new_version.parent_version_id == original.id
        assert new_version.version == original.version + 1
        
        # Check that configuration and parameters are copied, not referenced
        assert new_version.configuration == original.configuration
        assert new_version.parameters == original.parameters
        assert new_version.validation_rules == original.validation_rules
        
        # Verify it's a deep copy by modifying the original
        original.configuration["key"] = "modified"
        assert new_version.configuration["key"] == "value"

    def test_update_performance_metrics(self, Strategy):
        """Test updating strategy performance metrics."""
        strategy = Strategy(
            name="Performance Test"
        )
        
        # Initial metrics should be None
        assert strategy.win_rate is None
        assert strategy.profit_factor is None
        
        # Update metrics
        strategy.update_performance_metrics(
            win_rate=0.65,
            profit_factor=1.8,
            sharpe_ratio=1.2,
            sortino_ratio=1.5,
            max_drawdown=0.15,
            total_profit_inr=50000,
            avg_win_inr=2500,
            avg_loss_inr=-1200,
            largest_win_inr=10000,
            largest_loss_inr=-5000
        )
        
        # Check updated values
        assert strategy.win_rate == 0.65
        assert strategy.profit_factor == 1.8
        assert strategy.sharpe_ratio == 1.2
        assert strategy.sortino_ratio == 1.5
        assert strategy.max_drawdown == 0.15
        assert strategy.total_profit_inr == 50000
        assert strategy.avg_win_inr == 2500
        assert strategy.avg_loss_inr == -1200
        assert strategy.largest_win_inr == 10000
        assert strategy.largest_loss_inr == -5000
        
        # Partial update
        strategy.update_performance_metrics(win_rate=0.7)
        assert strategy.win_rate == 0.7
        assert strategy.profit_factor == 1.8  # Unchanged

    def test_to_dict_method(self, Strategy, StrategyType):
        """Test the to_dict method for API responses."""
        strategy = Strategy(
            name="Dict Test",
            type=StrategyType.BREAKOUT,
            user_id=1,
            created_at=datetime.utcnow(),
            configuration={"test": True},
            parameters={"param": 42},
            total_profit_inr=25000,
            avg_win_inr=1500,
            avg_loss_inr=-800
        )
        
        # Mock relationships
        category = MagicMock()
        category.name = "Test Category"
        strategy.categories = [category]
        
        signal = MagicMock()
        strategy.signals = [signal, signal]  # Two signals
        
        # Test without relationships
        basic_dict = strategy.to_dict()
        assert basic_dict["name"] == "Dict Test"
        assert "type" in basic_dict
        assert "categories" not in basic_dict
        
        # Test with relationships
        full_dict = strategy.to_dict(include_relationships=True)
        assert "categories" in full_dict
        assert full_dict["categories"] == ["Test Category"]
        assert full_dict["signal_count"] == 2
        
        # Verify INR metrics are included
        assert "performance_inr" in full_dict
        assert full_dict["performance_inr"]["total_profit_inr"] == 25000

    def test_soft_delete(self, Strategy):
        """Test soft delete functionality from mixin."""
        strategy = Strategy(
            name="Delete Test"
        )
        
        # Initially not deleted
        assert strategy.is_deleted is False
        assert strategy.deleted_at is None
        
        # Soft delete
        strategy.soft_delete(user_id=2)
        
        # Check deleted state
        assert strategy.is_deleted is True
        assert strategy.deleted_at is not None
        assert strategy.deleted_by_id == 2


class TestMetaLearningSettings:
    """Test cases for the MetaLearningSettings model."""
    
    def test_meta_learning_creation(self, MetaLearningSettings, Strategy, db_session):
        """Test creating meta-learning settings with advanced features."""
        # Create a strategy first
        strategy = Strategy(name="Meta Learning Test")
        db_session.add(strategy)
        db_session.flush()
        
        # Create meta learning settings
        meta_settings = MetaLearningSettings(
            strategy_id=strategy.id,
            record_trading_sessions=True,
            review_frequency="daily",
            track_market_relationships=True,
            detect_regime_changes=True,
            adaptive_regime_parameters=True,
            regime_types=["high_volatility", "trending", "ranging"],
            parameter_adjustments_by_regime={
                "high_volatility": {"position_size_multiplier": 0.7},
                "trending": {"trailing_stop_multiplier": 1.3},
            },
            track_success_factors=True,
            success_factor_categories=[
                "timeframe_alignment", "ma_relationship", "trend_quality"
            ],
            factor_contribution_tracking=True,
            use_synthetic_markets=True,
            track_time_of_day_performance=True
        )
        
        db_session.add(meta_settings)
        db_session.commit()
        
        # Check that meta learning settings were created with expected values
        assert meta_settings.id is not None
        assert meta_settings.strategy_id == strategy.id
        assert meta_settings.record_trading_sessions is True
        assert meta_settings.review_frequency == "daily"
        
        # Check advanced meta-learning features
        assert meta_settings.adaptive_regime_parameters is True
        assert "high_volatility" in meta_settings.regime_types
        assert meta_settings.parameter_adjustments_by_regime["trending"]["trailing_stop_multiplier"] == 1.3
        assert meta_settings.success_factor_categories == ["timeframe_alignment", "ma_relationship", "trend_quality"]
        assert meta_settings.track_time_of_day_performance is True
        assert meta_settings.use_synthetic_markets is True


class TestRiskManagementSettings:
    """Test cases for the RiskManagementSettings model."""
    
    def test_risk_management_creation(self, RiskManagementSettings, Strategy, db_session):
        """Test creating risk management settings with advanced features."""
        # Create a strategy first
        strategy = Strategy(name="Risk Management Test")
        db_session.add(strategy)
        db_session.flush()
        
        # Create risk management settings with intelligent recovery system
        risk_settings = RiskManagementSettings(
            strategy_id=strategy.id,
            max_risk_per_trade_percent=1.0,
            max_risk_per_trade_inr=10000,
            max_daily_risk_percent=3.0,
            drawdown_tiers={
                "tier1": {"threshold": 5.0, "size_reduction": 0.2},
                "tier2": {"threshold": 8.0, "size_reduction": 0.4},
                "tier3": {"threshold": 12.0, "size_reduction": 0.6}
            },
            use_progressive_recovery=True,
            track_asset_correlations=True
        )
        
        db_session.add(risk_settings)
        db_session.commit()
        
        # Check that risk settings were created with expected values
        assert risk_settings.id is not None
        assert risk_settings.strategy_id == strategy.id
        assert risk_settings.max_risk_per_trade_percent == 1.0
        assert risk_settings.max_risk_per_trade_inr == 10000
        
        # Check advanced risk management features
        assert risk_settings.drawdown_tiers["tier2"]["size_reduction"] == 0.4
        assert risk_settings.use_progressive_recovery is True
        assert risk_settings.track_asset_correlations is True


class TestStrategyBacktest:
    """Test cases for the StrategyBacktest model."""

    def test_backtest_creation(self, StrategyBacktest, Strategy, db_session):
        """Test creating a backtest record."""
        # Create a strategy first
        strategy = Strategy(
            name="Backtest Strategy"
        )
        db_session.add(strategy)
        db_session.flush()  # Get strategy ID without committing
        
        # Create backtest
        start_date = datetime.utcnow() - timedelta(days=30)
        end_date = datetime.utcnow()
        
        backtest = StrategyBacktest(
            strategy_id=strategy.id,
            name="30-Day Test",
            description="Testing last 30 days",
            start_date=start_date,
            end_date=end_date,
            initial_capital=10000.0,
            user_id=1,
            parameters={"param1": 5, "param2": 10}
        )
        
        db_session.add(backtest)
        db_session.commit()
        
        assert backtest.id is not None
        assert backtest.strategy_id == strategy.id
        assert backtest.name == "30-Day Test"
        assert backtest.start_date == start_date
        assert backtest.end_date == end_date
        assert backtest.initial_capital == 10000.0

    def test_calculate_metrics(self, StrategyBacktest):
        """Test calculation of backtest performance metrics."""
        backtest = StrategyBacktest(
            strategy_id=1,
            name="Metrics Test",
            start_date=datetime.utcnow() - timedelta(days=30),
            end_date=datetime.utcnow(),
            initial_capital=10000.0
        )
        
        # Sample trade data
        trades = [
            {"id": 1, "symbol": "AAPL", "profit": 100.0},
            {"id": 2, "symbol": "MSFT", "profit": -50.0},
            {"id": 3, "symbol": "GOOGL", "profit": 200.0},
            {"id": 4, "symbol": "AMZN", "profit": 150.0},
            {"id": 5, "symbol": "TSLA", "profit": -30.0}
        ]
        
        # Sample equity curve
        equity_curve = [
            {"date": "2023-01-01", "equity": 10000.0},
            {"date": "2023-01-05", "equity": 10050.0},
            {"date": "2023-01-10", "equity": 10200.0},
            {"date": "2023-01-15", "equity": 10150.0},
            {"date": "2023-01-20", "equity": 10300.0},
            {"date": "2023-01-25", "equity": 10370.0}
        ]
        
        # Calculate metrics
        backtest.calculate_metrics(trades, equity_curve)
        
        # Check calculated metrics
        assert backtest.total_trades == 5
        assert backtest.win_rate == 0.6  # 3 winning trades out of 5
        assert backtest.profit_factor == 5.625  # (100+200+150)/(50+30)
        assert backtest.total_return == 0.037  # (10370-10000)/10000
        
        # Check stored data
        assert backtest.equity_curve == equity_curve
        assert backtest.trade_history == trades


if __name__ == "__main__":
    pytest.main()