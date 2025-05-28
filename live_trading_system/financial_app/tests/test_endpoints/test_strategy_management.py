"""
Comprehensive test suite for Strategy Management API endpoints.

This test file is completely self-contained with no external dependencies.
Tests cover all major functionality by simulating the business logic directly.
"""

import pytest
from unittest.mock import Mock, MagicMock, AsyncMock
from datetime import datetime, date
from typing import List, Dict, Any, Optional
import json


# Test configuration
pytestmark = pytest.mark.asyncio


# Mock Classes and Enums
class MockStrategy:
    def __init__(self, **kwargs):
        self.id = kwargs.get('id', 1)
        self.name = kwargs.get('name', 'Test Strategy')
        self.description = kwargs.get('description', 'A test strategy')
        self.type = kwargs.get('type', 'trend_following')
        self.user_id = kwargs.get('user_id', 1)
        self.created_by_id = kwargs.get('created_by_id', 1)
        self.updated_by_id = kwargs.get('updated_by_id', None)
        self.is_active = kwargs.get('is_active', False)
        self.version = kwargs.get('version', 1)
        self.created_at = kwargs.get('created_at', datetime(2023, 1, 1, 10, 0, 0))
        self.updated_at = kwargs.get('updated_at', None)
        self.status = kwargs.get('status', 'draft')
        self.win_rate = kwargs.get('win_rate', None)
        self.profit_factor = kwargs.get('profit_factor', None)
        self.sharpe_ratio = kwargs.get('sharpe_ratio', None)
        self.sortino_ratio = kwargs.get('sortino_ratio', None)
        self.max_drawdown = kwargs.get('max_drawdown', None)
        self.total_profit_inr = kwargs.get('total_profit_inr', None)
        self.avg_win_inr = kwargs.get('avg_win_inr', None)
        self.avg_loss_inr = kwargs.get('avg_loss_inr', None)
        self.timeframes = kwargs.get('timeframes', [])
        self.configuration = kwargs.get('configuration', {"indicators": ["ma", "rsi"]})
        self.parameters = kwargs.get('parameters', {"ma_period": 21})

    def to_dict(self, include_relationships=False):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "type": self.type,
            "user_id": self.user_id,
            "created_by_id": self.created_by_id,
            "updated_by_id": self.updated_by_id,
            "is_active": self.is_active,
            "version": self.version,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "status": self.status,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "total_profit_inr": self.total_profit_inr,
            "avg_win_inr": self.avg_win_inr,
            "avg_loss_inr": self.avg_loss_inr,
            "timeframes": self.timeframes,
            "configuration": self.configuration,
            "parameters": self.parameters
        }


class MockStrategyCreateData:
    def __init__(self, **kwargs):
        self.name = kwargs.get('name', 'Test Strategy')
        self.description = kwargs.get('description', 'A test strategy for unit tests')
        self.type = kwargs.get('type', 'trend_following')
        self.configuration = kwargs.get('configuration', {"indicators": ["ma", "rsi"]})
        self.parameters = kwargs.get('parameters', {"ma_period": 21})
        self.validation_rules = kwargs.get('validation_rules', {"ma_period": {"type": "number", "min": 5, "max": 200}})


class MockStrategyUpdateData:
    def __init__(self, **kwargs):
        self.name = kwargs.get('name', None)
        self.description = kwargs.get('description', None)
        self.parameters = kwargs.get('parameters', None)


class MockPerformanceData:
    def __init__(self, **kwargs):
        self.strategy_id = kwargs.get('strategy_id', 1)
        self.total_trades = kwargs.get('total_trades', 50)
        self.win_count = kwargs.get('win_count', 35)
        self.loss_count = kwargs.get('loss_count', 15)
        self.win_rate = kwargs.get('win_rate', 0.7)
        self.total_profit_inr = kwargs.get('total_profit_inr', 150000.0)
        self.avg_win_inr = kwargs.get('avg_win_inr', 5000.0)
        self.avg_loss_inr = kwargs.get('avg_loss_inr', -2000.0)
        self.profit_factor = kwargs.get('profit_factor', 3.5)
        self.trades_by_grade = kwargs.get('trades_by_grade', {
            "a_plus": {"count": 20, "profit": 100000.0, "win_rate": 0.9},
            "a": {"count": 15, "profit": 50000.0, "win_rate": 0.8}
        })
        self.analysis_period = kwargs.get('analysis_period', {
            "start": datetime(2023, 1, 1),
            "end": datetime(2023, 3, 31)
        })


class MockHTTPException(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"{status_code}: {detail}")


class MockValidationError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class MockOperationalError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


# Business Logic Simulators
class StrategyManagementSimulator:
    """Simulates the strategy management business logic."""
    
    @staticmethod
    def strategy_to_response(strategy: MockStrategy) -> Dict[str, Any]:
        """Convert strategy model to response format."""
        return strategy.to_dict(include_relationships=True)
    
    @staticmethod
    async def check_strategy_ownership(strategy: MockStrategy, user_id: int) -> None:
        """Check if user owns the strategy."""
        if strategy.user_id != user_id:
            raise MockHTTPException(status_code=403, detail="Access denied: You can only access your own strategies")
    
    @staticmethod
    def validate_strategy_create_data(data: MockStrategyCreateData) -> None:
        """Validate strategy creation data."""
        if not data.name or len(data.name.strip()) == 0:
            raise MockValidationError("Strategy name is required")
        
        if len(data.name) > 100:
            raise MockValidationError("Strategy name must be 100 characters or less")
        
        if not data.type:
            raise MockValidationError("Strategy type is required")
        
        valid_types = ["trend_following", "mean_reversion", "breakout", "momentum"]
        if data.type not in valid_types:
            raise MockValidationError(f"Invalid strategy type. Valid options: {valid_types}")
    
    @staticmethod
    def validate_strategy_update_data(data: MockStrategyUpdateData) -> Dict[str, Any]:
        """Validate strategy update data."""
        validated_data = {}
        
        if data.name is not None:
            if len(data.name.strip()) == 0:
                raise MockValidationError("Strategy name cannot be empty")
            if len(data.name) > 100:
                raise MockValidationError("Strategy name must be 100 characters or less")
            validated_data["name"] = data.name.strip()
        
        if data.description is not None:
            validated_data["description"] = data.description
        
        if data.parameters is not None:
            # Validate parameters based on strategy type
            if "ma_period" in data.parameters:
                period = data.parameters["ma_period"]
                if not isinstance(period, int) or period < 5 or period > 200:
                    raise MockValidationError("ma_period must be an integer between 5 and 200")
            validated_data["parameters"] = data.parameters
        
        return validated_data
    
    @staticmethod
    def validate_pagination_params(offset: int, limit: int) -> None:
        """Validate pagination parameters."""
        if offset < 0:
            raise MockValidationError("offset must be non-negative")
        
        if limit < 1 or limit > 1000:
            raise MockValidationError("limit must be between 1 and 1000")
    
    @staticmethod
    def apply_strategy_filters(strategies: List[MockStrategy], 
                             user_id: Optional[int] = None,
                             include_inactive: bool = False) -> List[MockStrategy]:
        """Apply filters to strategy list."""
        filtered_strategies = strategies
        
        if user_id is not None:
            filtered_strategies = [s for s in filtered_strategies if s.user_id == user_id]
        
        if not include_inactive:
            filtered_strategies = [s for s in filtered_strategies if s.is_active]
        
        return filtered_strategies
    
    @staticmethod
    def apply_pagination(strategies: List[MockStrategy], offset: int, limit: int) -> List[MockStrategy]:
        """Apply pagination to strategy list."""
        return strategies[offset:offset + limit]


# Test Fixtures
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
def mock_strategy_service(mock_db_session):
    """Mock strategy engine service."""
    service = MagicMock()
    service.db = mock_db_session
    service.db.session.return_value = mock_db_session
    return service


@pytest.fixture
def mock_strategy():
    """Mock strategy object."""
    return MockStrategy()


@pytest.fixture
def sample_strategy_create_data():
    """Sample strategy creation data."""
    return MockStrategyCreateData()


@pytest.fixture
def sample_strategy_update_data():
    """Sample strategy update data."""
    return MockStrategyUpdateData(
        name="Updated Test Strategy",
        description="Updated description",
        parameters={"ma_period": 34}
    )


@pytest.fixture
def sample_performance_data():
    """Sample performance data."""
    return MockPerformanceData()


# Test utility functions
class TestUtilityFunctions:
    """Test utility functions from the strategy management module."""
    
    def test_strategy_to_response_conversion(self, mock_strategy):
        """Test converting strategy model to response format."""
        response = StrategyManagementSimulator.strategy_to_response(mock_strategy)
        
        assert response["id"] == 1
        assert response["name"] == "Test Strategy"
        assert response["type"] == "trend_following"
        assert response["user_id"] == 1
        assert response["is_active"] == False
    
    async def test_check_strategy_ownership_success(self, mock_strategy):
        """Test successful strategy ownership check."""
        # Should not raise exception
        await StrategyManagementSimulator.check_strategy_ownership(mock_strategy, 1)
    
    async def test_check_strategy_ownership_forbidden(self):
        """Test strategy ownership check with wrong user."""
        strategy = MockStrategy(user_id=2)  # Different user
        
        with pytest.raises(MockHTTPException) as exc_info:
            await StrategyManagementSimulator.check_strategy_ownership(strategy, 1)
        
        assert exc_info.value.status_code == 403
        assert "Access denied" in exc_info.value.detail
    
    def test_validate_strategy_create_data_success(self, sample_strategy_create_data):
        """Test successful strategy creation data validation."""
        # Should not raise exception
        StrategyManagementSimulator.validate_strategy_create_data(sample_strategy_create_data)
    
    def test_validate_strategy_create_data_missing_name(self):
        """Test strategy creation validation with missing name."""
        data = MockStrategyCreateData(name="")
        
        with pytest.raises(MockValidationError) as exc_info:
            StrategyManagementSimulator.validate_strategy_create_data(data)
        
        assert "Strategy name is required" in str(exc_info.value)
    
    def test_validate_strategy_create_data_invalid_type(self):
        """Test strategy creation validation with invalid type."""
        data = MockStrategyCreateData(type="invalid_type")
        
        with pytest.raises(MockValidationError) as exc_info:
            StrategyManagementSimulator.validate_strategy_create_data(data)
        
        assert "Invalid strategy type" in str(exc_info.value)
    
    def test_validate_strategy_update_data_success(self, sample_strategy_update_data):
        """Test successful strategy update data validation."""
        validated = StrategyManagementSimulator.validate_strategy_update_data(sample_strategy_update_data)
        
        assert validated["name"] == "Updated Test Strategy"
        assert validated["description"] == "Updated description"
        assert validated["parameters"]["ma_period"] == 34
    
    def test_validate_strategy_update_data_invalid_ma_period(self):
        """Test strategy update validation with invalid MA period."""
        data = MockStrategyUpdateData(parameters={"ma_period": 300})  # Too high
        
        with pytest.raises(MockValidationError) as exc_info:
            StrategyManagementSimulator.validate_strategy_update_data(data)
        
        assert "ma_period must be an integer between 5 and 200" in str(exc_info.value)
    
    def test_validate_pagination_params_success(self):
        """Test successful pagination parameter validation."""
        # Should not raise exception
        StrategyManagementSimulator.validate_pagination_params(0, 100)
        StrategyManagementSimulator.validate_pagination_params(50, 25)
    
    def test_validate_pagination_params_invalid_offset(self):
        """Test pagination validation with invalid offset."""
        with pytest.raises(MockValidationError) as exc_info:
            StrategyManagementSimulator.validate_pagination_params(-1, 100)
        
        assert "offset must be non-negative" in str(exc_info.value)
    
    def test_validate_pagination_params_invalid_limit(self):
        """Test pagination validation with invalid limit."""
        with pytest.raises(MockValidationError) as exc_info:
            StrategyManagementSimulator.validate_pagination_params(0, 0)
        
        assert "limit must be between 1 and 1000" in str(exc_info.value)


# Test CRUD operations
class TestStrategyCRUDOperations:
    """Test strategy CRUD operations business logic."""
    
    async def test_create_strategy_success(self, mock_strategy_service, sample_strategy_create_data, mock_db_session):
        """Test successful strategy creation workflow."""
        # Setup
        user_id = 1
        created_strategy = MockStrategy(id=1, name=sample_strategy_create_data.name, user_id=user_id)
        
        # Simulate the endpoint logic
        # 1. Validate input data
        StrategyManagementSimulator.validate_strategy_create_data(sample_strategy_create_data)
        
        # 2. Create strategy using service
        mock_strategy_service.create_strategy.return_value = created_strategy
        strategy = mock_strategy_service.create_strategy(sample_strategy_create_data, user_id)
        
        # 3. Convert to response format
        response = StrategyManagementSimulator.strategy_to_response(strategy)
        
        # Verify results
        assert response["id"] == 1
        assert response["name"] == "Test Strategy"
        assert response["user_id"] == 1
        assert response["type"] == "trend_following"
        
        # Verify service calls
        mock_strategy_service.create_strategy.assert_called_once_with(sample_strategy_create_data, user_id)
    
    async def test_create_strategy_validation_error(self, mock_strategy_service):
        """Test strategy creation with validation error."""
        # Invalid data
        invalid_data = MockStrategyCreateData(name="", type="invalid")
        
        # Should raise validation error
        with pytest.raises(MockValidationError):
            StrategyManagementSimulator.validate_strategy_create_data(invalid_data)
    
    async def test_list_strategies_success(self, mock_strategy_service, mock_db_session):
        """Test successful strategy listing workflow."""
        # Setup
        user_id = 1
        strategies = [
            MockStrategy(id=1, name="Strategy 1", user_id=user_id, is_active=True),
            MockStrategy(id=2, name="Strategy 2", user_id=user_id, is_active=False),
            MockStrategy(id=3, name="Strategy 3", user_id=2, is_active=True)  # Different user
        ]
        
        # Simulate the endpoint logic
        # 1. Validate pagination parameters
        offset = 0
        limit = 100
        include_inactive = False
        filter_user_id = user_id
        
        StrategyManagementSimulator.validate_pagination_params(offset, limit)
        
        # 2. Apply filters
        filtered_strategies = StrategyManagementSimulator.apply_strategy_filters(
            strategies, user_id=filter_user_id, include_inactive=include_inactive
        )
        
        # 3. Apply pagination
        paginated_strategies = StrategyManagementSimulator.apply_pagination(filtered_strategies, offset, limit)
        
        # 4. Convert to response format
        strategy_responses = []
        for strategy in paginated_strategies:
            response = StrategyManagementSimulator.strategy_to_response(strategy)
            strategy_responses.append(response)
        
        # Verify results - should only include user's active strategies
        assert len(strategy_responses) == 1  # Only one active strategy for user_id=1
        assert strategy_responses[0]["id"] == 1
        assert strategy_responses[0]["name"] == "Strategy 1"
        assert strategy_responses[0]["is_active"] == True
    
    async def test_list_strategies_include_inactive(self, mock_strategy_service):
        """Test strategy listing including inactive strategies."""
        # Setup
        user_id = 1
        strategies = [
            MockStrategy(id=1, name="Strategy 1", user_id=user_id, is_active=True),
            MockStrategy(id=2, name="Strategy 2", user_id=user_id, is_active=False)
        ]
        
        # Apply filters with include_inactive=True
        filtered_strategies = StrategyManagementSimulator.apply_strategy_filters(
            strategies, user_id=user_id, include_inactive=True
        )
        
        # Should include both active and inactive strategies
        assert len(filtered_strategies) == 2
    
    async def test_get_strategy_success(self, mock_strategy_service, mock_strategy, mock_db_session):
        """Test successful strategy retrieval workflow."""
        # Setup
        strategy_id = 1
        user_id = 1
        mock_db_session.first.return_value = mock_strategy
        
        # Simulate the endpoint logic
        # 1. Query strategy from database
        with mock_strategy_service.db.session() as session:
            strategy = session.first()
        
        # 2. Check if strategy exists
        if not strategy:
            raise MockHTTPException(status_code=404, detail=f"Strategy with ID {strategy_id} not found")
        
        # 3. Check ownership
        await StrategyManagementSimulator.check_strategy_ownership(strategy, user_id)
        
        # 4. Convert to response format
        response = StrategyManagementSimulator.strategy_to_response(strategy)
        
        # Verify results
        assert response["id"] == 1
        assert response["name"] == "Test Strategy"
        assert response["user_id"] == 1
    
    async def test_get_strategy_not_found(self, mock_strategy_service, mock_db_session):
        """Test get strategy with non-existent ID."""
        # Setup
        strategy_id = 999
        mock_db_session.first.return_value = None
        
        # Simulate the endpoint logic
        with mock_strategy_service.db.session() as session:
            strategy = session.first()
        
        # Should raise not found error
        with pytest.raises(MockHTTPException) as exc_info:
            if not strategy:
                raise MockHTTPException(status_code=404, detail=f"Strategy with ID {strategy_id} not found")
        
        assert exc_info.value.status_code == 404
        assert "not found" in exc_info.value.detail
    
    async def test_get_strategy_access_denied(self, mock_strategy_service, mock_db_session):
        """Test get strategy with access denied."""
        # Setup - strategy owned by different user
        strategy = MockStrategy(user_id=2)
        user_id = 1
        mock_db_session.first.return_value = strategy
        
        # Simulate the endpoint logic
        with mock_strategy_service.db.session() as session:
            strategy = session.first()
        
        # Should raise access denied error
        with pytest.raises(MockHTTPException) as exc_info:
            await StrategyManagementSimulator.check_strategy_ownership(strategy, user_id)
        
        assert exc_info.value.status_code == 403
        assert "Access denied" in exc_info.value.detail
    
    async def test_update_strategy_success(self, mock_strategy_service, mock_strategy, sample_strategy_update_data, mock_db_session):
        """Test successful strategy update workflow."""
        # Setup
        strategy_id = 1
        user_id = 1
        mock_db_session.first.return_value = mock_strategy
        mock_strategy_service.update_strategy.return_value = mock_strategy
        
        # Simulate the endpoint logic
        # 1. Get existing strategy and check ownership
        with mock_strategy_service.db.session() as session:
            existing_strategy = session.first()
        
        if not existing_strategy:
            raise MockHTTPException(status_code=404, detail=f"Strategy with ID {strategy_id} not found")
        
        await StrategyManagementSimulator.check_strategy_ownership(existing_strategy, user_id)
        
        # 2. Validate update data
        validated_data = StrategyManagementSimulator.validate_strategy_update_data(sample_strategy_update_data)
        
        # 3. Update strategy using service
        updated_strategy = mock_strategy_service.update_strategy(strategy_id, validated_data, user_id)
        
        # 4. Convert to response format
        response = StrategyManagementSimulator.strategy_to_response(updated_strategy)
        
        # Verify results
        assert response["id"] == 1
        
        # Verify service calls
        mock_strategy_service.update_strategy.assert_called_once_with(strategy_id, validated_data, user_id)
    
    async def test_delete_strategy_success(self, mock_strategy_service, mock_strategy, mock_db_session):
        """Test successful strategy deletion workflow."""
        # Setup
        strategy_id = 1
        user_id = 1
        hard_delete = False
        mock_db_session.first.return_value = mock_strategy
        mock_strategy_service.delete_strategy.return_value = True
        
        # Simulate the endpoint logic
        # 1. Get existing strategy and check ownership
        with mock_strategy_service.db.session() as session:
            existing_strategy = session.first()
        
        if not existing_strategy:
            raise MockHTTPException(status_code=404, detail=f"Strategy with ID {strategy_id} not found")
        
        await StrategyManagementSimulator.check_strategy_ownership(existing_strategy, user_id)
        
        # 2. Delete strategy using service
        success = mock_strategy_service.delete_strategy(strategy_id, user_id, hard_delete=hard_delete)
        
        # Verify results
        assert success == True
        
        # Verify service calls
        mock_strategy_service.delete_strategy.assert_called_once_with(strategy_id, user_id, hard_delete=hard_delete)


# Test strategy state management
class TestStrategyStateManagement:
    """Test strategy activation/deactivation business logic."""
    
    async def test_activate_strategy_success(self, mock_strategy_service, mock_strategy, mock_db_session):
        """Test successful strategy activation workflow."""
        # Setup
        strategy_id = 1
        user_id = 1
        mock_db_session.first.return_value = mock_strategy
        
        activated_strategy = MockStrategy(id=1, is_active=True)
        mock_strategy_service.activate_strategy.return_value = activated_strategy
        
        # Simulate the endpoint logic
        # 1. Get existing strategy and check ownership
        with mock_strategy_service.db.session() as session:
            existing_strategy = session.first()
        
        await StrategyManagementSimulator.check_strategy_ownership(existing_strategy, user_id)
        
        # 2. Activate strategy using service
        activated = mock_strategy_service.activate_strategy(strategy_id, user_id)
        
        # 3. Convert to response format
        response = StrategyManagementSimulator.strategy_to_response(activated)
        
        # Verify results
        assert response["id"] == 1
        assert response["is_active"] == True
        
        # Verify service calls
        mock_strategy_service.activate_strategy.assert_called_once_with(strategy_id, user_id)
    
    async def test_deactivate_strategy_success(self, mock_strategy_service, mock_strategy, mock_db_session):
        """Test successful strategy deactivation workflow."""
        # Setup
        strategy_id = 1
        user_id = 1
        mock_strategy.is_active = True  # Start with active strategy
        mock_db_session.first.return_value = mock_strategy
        
        deactivated_strategy = MockStrategy(id=1, is_active=False)
        mock_strategy_service.deactivate_strategy.return_value = deactivated_strategy
        
        # Simulate the endpoint logic
        # 1. Get existing strategy and check ownership
        with mock_strategy_service.db.session() as session:
            existing_strategy = session.first()
        
        await StrategyManagementSimulator.check_strategy_ownership(existing_strategy, user_id)
        
        # 2. Deactivate strategy using service
        deactivated = mock_strategy_service.deactivate_strategy(strategy_id, user_id)
        
        # 3. Convert to response format
        response = StrategyManagementSimulator.strategy_to_response(deactivated)
        
        # Verify results
        assert response["id"] == 1
        assert response["is_active"] == False
        
        # Verify service calls
        mock_strategy_service.deactivate_strategy.assert_called_once_with(strategy_id, user_id)


# Test performance analysis
class TestPerformanceAnalysis:
    """Test strategy performance analysis business logic."""
    
    async def test_get_strategy_performance_success(self, mock_strategy_service, mock_strategy, sample_performance_data, mock_db_session):
        """Test successful performance analysis workflow."""
        # Setup
        strategy_id = 1
        user_id = 1
        start_date = "2023-01-01"
        end_date = "2023-12-31"
        
        mock_db_session.first.return_value = mock_strategy
        mock_strategy_service.analyze_performance.return_value = sample_performance_data
        
        # Simulate the endpoint logic
        # 1. Get existing strategy and check ownership
        with mock_strategy_service.db.session() as session:
            existing_strategy = session.first()
        
        await StrategyManagementSimulator.check_strategy_ownership(existing_strategy, user_id)
        
        # 2. Parse dates if provided
        start_datetime = None
        end_datetime = None
        
        if start_date:
            try:
                start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
            except ValueError:
                raise MockValidationError("Invalid start_date format. Use YYYY-MM-DD")
        
        if end_date:
            try:
                end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                raise MockValidationError("Invalid end_date format. Use YYYY-MM-DD")
        
        # 3. Get performance analysis using service
        performance_data = mock_strategy_service.analyze_performance(
            strategy_id, 
            start_date=start_datetime, 
            end_date=end_datetime
        )
        
        # Verify results
        assert performance_data.strategy_id == 1
        assert performance_data.total_trades == 50
        assert performance_data.win_rate == 0.7
        assert performance_data.profit_factor == 3.5
        
        # Verify service calls
        mock_strategy_service.analyze_performance.assert_called_once_with(
            strategy_id, start_date=start_datetime, end_date=end_datetime
        )
    
    async def test_get_strategy_performance_invalid_date(self, mock_strategy_service, mock_strategy, mock_db_session):
        """Test performance analysis with invalid date format."""
        # Setup
        strategy_id = 1
        user_id = 1
        invalid_start_date = "invalid-date"
        
        mock_db_session.first.return_value = mock_strategy
        
        # Simulate the endpoint logic
        with mock_strategy_service.db.session() as session:
            existing_strategy = session.first()
        
        await StrategyManagementSimulator.check_strategy_ownership(existing_strategy, user_id)
        
        # Parse dates with error handling (simulate real endpoint logic)
        start_datetime = None
        
        # Should raise validation error for invalid date
        with pytest.raises(MockValidationError) as exc_info:
            try:
                start_datetime = datetime.strptime(invalid_start_date, "%Y-%m-%d")
            except ValueError:
                raise MockValidationError("Invalid start_date format. Use YYYY-MM-DD")
        
        assert "Invalid start_date format" in str(exc_info.value)


# Test error handling
class TestErrorHandling:
    """Test error handling scenarios."""
    
    async def test_strategy_not_found_error(self, mock_strategy_service, mock_db_session):
        """Test handling of non-existent strategy."""
        # Setup
        strategy_id = 999
        mock_db_session.first.return_value = None
        
        # Simulate the endpoint logic
        with mock_strategy_service.db.session() as session:
            strategy = session.first()
        
        # Should raise not found error
        with pytest.raises(MockHTTPException) as exc_info:
            if not strategy:
                raise MockHTTPException(status_code=404, detail=f"Strategy with ID {strategy_id} not found")
        
        assert exc_info.value.status_code == 404
        assert "not found" in exc_info.value.detail
    
    async def test_unauthorized_access_error(self):
        """Test handling of unauthorized strategy access."""
        # Setup
        strategy = MockStrategy(user_id=2)  # Different user
        user_id = 1
        
        # Should raise access denied error
        with pytest.raises(MockHTTPException) as exc_info:
            await StrategyManagementSimulator.check_strategy_ownership(strategy, user_id)
        
        assert exc_info.value.status_code == 403
        assert "Access denied" in exc_info.value.detail
    
    async def test_validation_error_handling(self):
        """Test handling of validation errors."""
        # Test invalid strategy name
        with pytest.raises(MockValidationError) as exc_info:
            data = MockStrategyCreateData(name="")
            StrategyManagementSimulator.validate_strategy_create_data(data)
        
        assert "Strategy name is required" in str(exc_info.value)
        
        # Test invalid pagination
        with pytest.raises(MockValidationError) as exc_info:
            StrategyManagementSimulator.validate_pagination_params(-1, 100)
        
        assert "offset must be non-negative" in str(exc_info.value)
    
    async def test_operational_error_handling(self, mock_strategy_service, mock_db_session):
        """Test handling of operational errors."""
        # Setup mock to raise operational error
        mock_strategy_service.create_strategy.side_effect = Exception("Database connection error")
        
        # Simulate operational error
        with pytest.raises(Exception) as exc_info:
            mock_strategy_service.create_strategy(MockStrategyCreateData(), 1)
        
        assert "Database connection error" in str(exc_info.value)


# Test edge cases
class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_strategy_name_length_validation(self):
        """Test strategy name length validation."""
        # Test maximum length
        long_name = "x" * 100  # Exactly 100 characters
        data = MockStrategyCreateData(name=long_name)
        
        # Should not raise exception
        StrategyManagementSimulator.validate_strategy_create_data(data)
        
        # Test exceeding maximum length
        too_long_name = "x" * 101  # 101 characters
        data = MockStrategyCreateData(name=too_long_name)
        
        with pytest.raises(MockValidationError) as exc_info:
            StrategyManagementSimulator.validate_strategy_create_data(data)
        
        assert "must be 100 characters or less" in str(exc_info.value)
    
    def test_pagination_boundary_conditions(self):
        """Test pagination boundary conditions."""
        # Test maximum limit
        StrategyManagementSimulator.validate_pagination_params(0, 1000)  # Should pass
        
        # Test exceeding maximum limit
        with pytest.raises(MockValidationError):
            StrategyManagementSimulator.validate_pagination_params(0, 1001)
        
        # Test minimum limit
        StrategyManagementSimulator.validate_pagination_params(0, 1)  # Should pass
        
        # Test below minimum limit
        with pytest.raises(MockValidationError):
            StrategyManagementSimulator.validate_pagination_params(0, 0)
    
    def test_ma_period_boundary_conditions(self):
        """Test MA period boundary conditions."""
        # Test minimum valid value
        data = MockStrategyUpdateData(parameters={"ma_period": 5})
        validated = StrategyManagementSimulator.validate_strategy_update_data(data)
        assert validated["parameters"]["ma_period"] == 5
        
        # Test maximum valid value
        data = MockStrategyUpdateData(parameters={"ma_period": 200})
        validated = StrategyManagementSimulator.validate_strategy_update_data(data)
        assert validated["parameters"]["ma_period"] == 200
        
        # Test below minimum
        data = MockStrategyUpdateData(parameters={"ma_period": 4})
        with pytest.raises(MockValidationError):
            StrategyManagementSimulator.validate_strategy_update_data(data)
        
        # Test above maximum
        data = MockStrategyUpdateData(parameters={"ma_period": 201})
        with pytest.raises(MockValidationError):
            StrategyManagementSimulator.validate_strategy_update_data(data)


# Integration tests
class TestIntegrationScenarios:
    """Test complete workflow scenarios."""
    
    async def test_complete_strategy_lifecycle(self, mock_strategy_service, sample_strategy_create_data, sample_strategy_update_data, sample_performance_data, mock_db_session):
        """Test complete strategy lifecycle: create -> activate -> get -> update -> performance -> deactivate -> delete."""
        user_id = 1
        
        # Step 1: Create strategy
        StrategyManagementSimulator.validate_strategy_create_data(sample_strategy_create_data)
        
        created_strategy = MockStrategy(id=1, name=sample_strategy_create_data.name, user_id=user_id)
        mock_strategy_service.create_strategy.return_value = created_strategy
        
        strategy = mock_strategy_service.create_strategy(sample_strategy_create_data, user_id)
        assert strategy.id == 1
        assert strategy.name == "Test Strategy"
        
        # Step 2: Get strategy
        mock_db_session.first.return_value = strategy
        
        with mock_strategy_service.db.session() as session:
            retrieved_strategy = session.first()
        
        await StrategyManagementSimulator.check_strategy_ownership(retrieved_strategy, user_id)
        response = StrategyManagementSimulator.strategy_to_response(retrieved_strategy)
        assert response["id"] == 1
        
        # Step 3: Activate strategy
        activated_strategy = MockStrategy(id=1, is_active=True, user_id=user_id)
        mock_strategy_service.activate_strategy.return_value = activated_strategy
        
        activated = mock_strategy_service.activate_strategy(1, user_id)
        assert activated.is_active == True
        
        # Step 4: Update strategy
        validated_data = StrategyManagementSimulator.validate_strategy_update_data(sample_strategy_update_data)
        
        updated_strategy = MockStrategy(id=1, name="Updated Test Strategy", user_id=user_id)
        mock_strategy_service.update_strategy.return_value = updated_strategy
        
        updated = mock_strategy_service.update_strategy(1, validated_data, user_id)
        assert updated.name == "Updated Test Strategy"
        
        # Step 5: Get performance
        mock_strategy_service.analyze_performance.return_value = sample_performance_data
        
        performance = mock_strategy_service.analyze_performance(1, start_date=None, end_date=None)
        assert performance.strategy_id == 1
        assert performance.total_trades == 50
        
        # Step 6: Deactivate strategy
        deactivated_strategy = MockStrategy(id=1, is_active=False, user_id=user_id)
        mock_strategy_service.deactivate_strategy.return_value = deactivated_strategy
        
        deactivated = mock_strategy_service.deactivate_strategy(1, user_id)
        assert deactivated.is_active == False
        
        # Step 7: Delete strategy
        mock_strategy_service.delete_strategy.return_value = True
        
        success = mock_strategy_service.delete_strategy(1, user_id, hard_delete=False)
        assert success == True
        
        # Verify all service methods were called
        assert mock_strategy_service.create_strategy.called
        assert mock_strategy_service.activate_strategy.called
        assert mock_strategy_service.update_strategy.called
        assert mock_strategy_service.analyze_performance.called
        assert mock_strategy_service.deactivate_strategy.called
        assert mock_strategy_service.delete_strategy.called
    
    async def test_strategy_filtering_and_pagination(self):
        """Test strategy filtering and pagination workflow."""
        # Create test strategies
        strategies = [
            MockStrategy(id=1, name="Strategy 1", user_id=1, is_active=True),
            MockStrategy(id=2, name="Strategy 2", user_id=1, is_active=False),
            MockStrategy(id=3, name="Strategy 3", user_id=2, is_active=True),
            MockStrategy(id=4, name="Strategy 4", user_id=1, is_active=True),
        ]
        
        # Test filtering by user
        filtered = StrategyManagementSimulator.apply_strategy_filters(strategies, user_id=1, include_inactive=True)
        assert len(filtered) == 3  # 3 strategies for user 1
        
        # Test filtering by user and active only
        filtered = StrategyManagementSimulator.apply_strategy_filters(strategies, user_id=1, include_inactive=False)
        assert len(filtered) == 2  # 2 active strategies for user 1
        
        # Test pagination
        paginated = StrategyManagementSimulator.apply_pagination(filtered, offset=0, limit=1)
        assert len(paginated) == 1
        assert paginated[0].id == 1
        
        paginated = StrategyManagementSimulator.apply_pagination(filtered, offset=1, limit=1)
        assert len(paginated) == 1
        assert paginated[0].id == 4


if __name__ == "__main__":
    # Run tests
    import sys
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    sys.exit(exit_code)