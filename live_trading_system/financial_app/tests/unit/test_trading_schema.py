"""
Unit tests for trading schemas in the application.

These tests ensure that the Pydantic schema models properly validate
input data, handle edge cases correctly, and maintain consistency
with the SQLAlchemy models they represent.

Tests include:
- Basic validation of required and optional fields
- Validation of numeric constraints (positive values, ranges)
- Validation of field interdependencies (e.g., certain fields required based on others)
- Conversion between model and schema objects
- Proper enum handling
- Edge cases for date/time and financial values
"""

import json
import unittest
from datetime import datetime, timedelta
from decimal import Decimal
from uuid import uuid4, UUID

import pytest
from pydantic import ValidationError

from app.models.trading import (
    OrderStatus, OrderType, OrderSide, TimeInForce, 
    PositionDirection, OrderEventType, Order, Execution, 
    Position, OrderEvent, Trade, BracketOrder
)

from app.schemas.trading import (
    # Base schemas
    BaseSchema, PaginationParams, TimeRangeParams, StatusMessage,
    
    # Order schemas
    OrderFilter, OrderBase, OrderCreate, OrderUpdate, OrderStatusUpdate,
    ExecutionCreate, ExecutionResponse, OrderEventResponse, OrderResponse,
    OrderListResponse,
    
    # Position schemas
    PositionFilter, PositionBase, PositionCreate, PositionUpdate,
    PositionRiskUpdate, PositionResponse, PositionListResponse,
    
    # Trade schemas
    TradeFilter, TradeBase, TradeCreate, TradeResponse, TradeListResponse,
    
    # Bracket order schemas
    BracketOrderCreate, BracketOrderResponse, BracketOrderListResponse,
    
    # OCO order schemas
    OCOOrderCreate,
    
    # Conversion functions
    order_model_to_schema, execution_model_to_schema, 
    order_event_model_to_schema, position_model_to_schema,
    trade_model_to_schema, bracket_order_model_to_schema
)


#################################################
# Test Fixtures
#################################################

@pytest.fixture
def valid_order_data():
    """Fixture for valid order data."""
    return {
        "symbol": "AAPL",
        "side": "buy",
        "order_type": "limit",
        "quantity": "100.0",
        "price": "150.50",
        "time_in_force": "day",
        "strategy_id": "mean_reversion_v1",
        "tags": ["tech", "momentum"]
    }


@pytest.fixture
def valid_execution_data():
    """Fixture for valid execution data."""
    return {
        "quantity": "50.5",
        "price": "150.25",
        "execution_id": "ex-12345",
        "fees": "0.75",
        "executed_at": datetime.utcnow().isoformat(),
        "liquidity": "taker"
    }


@pytest.fixture
def valid_position_data():
    """Fixture for valid position data."""
    return {
        "symbol": "AAPL",
        "direction": "long",
        "quantity": "100.5",
        "average_entry_price": "150.25",
        "current_price": "152.50",
        "strategy_id": "mean_reversion_v1"
    }


@pytest.fixture
def valid_trade_data():
    """Fixture for valid trade data."""
    return {
        "symbol": "AAPL",
        "side": "buy",
        "quantity": "100.5",
        "price": "150.25",
        "fees": "0.75",
        "order_id": str(uuid4()),
        "execution_id": "ex-67890",
        "strategy_id": "mean_reversion_v1",
        "executed_at": datetime.utcnow().isoformat()
    }


@pytest.fixture
def valid_bracket_order_data():
    """Fixture for valid bracket order data."""
    return {
        "account_id": 123,
        "symbol": "AAPL",
        "side": "buy",
        "quantity": "100.5",
        "entry_type": "limit",
        "entry_price": "150.00",
        "stop_loss_price": "145.00",
        "take_profit_price": "160.00",
        "time_in_force": "gtc",
        "strategy_id": "breakout_v1"
    }


@pytest.fixture
def valid_oco_order_data():
    """Fixture for valid OCO order data."""
    return {
        "account_id": 123,
        "symbol": "AAPL",
        "side": "buy",
        "quantity": "100.5",
        "price_1": "145.00",
        "type_1": "limit",
        "price_2": "155.00",
        "type_2": "stop",
        "time_in_force": "day",
        "strategy_id": "breakout_v1"
    }


@pytest.fixture
def mock_order():
    """Create a mock Order instance with realistic data."""
    order = Order()
    order.id = 1
    order.order_id = str(uuid4())
    order.account_id = 123
    order.symbol = "AAPL"
    order.side = OrderSide.BUY
    order.order_type = OrderType.LIMIT
    order.quantity = Decimal("100")
    order.price = Decimal("150.50")
    order.status = OrderStatus.SUBMITTED
    order.time_in_force = TimeInForce.DAY
    order.filled_quantity = Decimal("0")
    order.remaining_quantity = Decimal("100")
    order.strategy_id = "mean_reversion_v1"
    order.risk_check_passed = True
    order.created_at = datetime.utcnow()
    order.submitted_at = datetime.utcnow() + timedelta(minutes=1)
    order.tags = json.dumps(["tech", "momentum"])
    order.notes = "Test order"
    order.broker = "test_broker"
    order.venue = "test_venue"
    
    # Add properties that would normally be computed
    order.is_active = True
    order.fill_percent = Decimal("0")
    order.value = Decimal("15050.00")
    order.executions = []
    order.order_events = []
    
    return order


@pytest.fixture
def mock_execution():
    """Create a mock Execution instance."""
    execution = Execution()
    execution.id = 456
    execution.order_id = str(uuid4())
    execution.execution_id = "ex-12345"
    execution.quantity = Decimal("50")
    execution.price = Decimal("150.25")
    execution.fees = Decimal("0.75")
    execution.venue = "NYSE"
    execution.liquidity = "taker"
    execution.route = "smart"
    execution.executed_at = datetime.utcnow()
    execution.recorded_at = datetime.utcnow() + timedelta(seconds=1)
    return execution


@pytest.fixture
def mock_position():
    """Create a mock Position instance."""
    position = Position()
    position.id = 42
    position.account_id = 123
    position.symbol = "AAPL"
    position.direction = PositionDirection.LONG
    position.quantity = Decimal("100")
    position.average_entry_price = Decimal("150.25")
    position.current_price = Decimal("152.50")
    position.realized_pnl = Decimal("0")
    position.unrealized_pnl = Decimal("225.00")
    position.stop_loss_price = Decimal("145.00")
    position.take_profit_price = Decimal("165.00")
    position.strategy_id = "mean_reversion_v1"
    position.opened_at = datetime.utcnow() - timedelta(days=5)
    position.last_trade_at = datetime.utcnow() - timedelta(days=5)
    position.last_pnl_update = datetime.utcnow()
    
    # Properties that would normally be computed
    position.total_pnl = Decimal("225.00")
    position.pnl_percentage = 1.5
    position.market_value = Decimal("15250.00")
    position.cost_basis = Decimal("15025.00")
    
    return position


@pytest.fixture
def mock_order_event():
    """Create a mock OrderEvent instance."""
    event = OrderEvent()
    event.id = 789
    event.order_id = str(uuid4())
    event.event_type = OrderEventType.SUBMITTED
    event.description = "Order submitted to broker"
    event.event_data = json.dumps({"broker_confirmation": "conf-56789"})
    event.created_at = datetime.utcnow()
    
    # Properties that would normally be computed
    event.data = {"broker_confirmation": "conf-56789"}
    
    return event


@pytest.fixture
def mock_trade():
    """Create a mock Trade instance."""
    trade = Trade()
    trade.id = 789
    trade.trade_id = str(uuid4())
    trade.account_id = 123
    trade.symbol = "AAPL"
    trade.side = OrderSide.BUY
    trade.quantity = Decimal("100")
    trade.price = Decimal("150.25")
    trade.fees = Decimal("0.75")
    trade.value = Decimal("15025.00")
    trade.total_cost = Decimal("15025.75")
    trade.realized_pnl = None
    trade.order_id = str(uuid4())
    trade.execution_id = "ex-67890"
    trade.strategy_id = "mean_reversion_v1"
    trade.notes = None
    trade.tax_lot_id = None
    trade.wash_sale = False
    trade.executed_at = datetime.utcnow() - timedelta(hours=1)
    trade.recorded_at = datetime.utcnow() - timedelta(hours=1) + timedelta(seconds=1)
    
    return trade


@pytest.fixture
def mock_bracket_order():
    """Create a mock BracketOrder instance."""
    bracket = BracketOrder()
    bracket.id = 123
    bracket.account_id = 123
    bracket.symbol = "AAPL"
    bracket.status = "active"
    bracket.entry_order_id = str(uuid4())
    bracket.stop_loss_order_id = str(uuid4())
    bracket.take_profit_order_id = str(uuid4())
    bracket.strategy_id = "breakout_v1"
    bracket.created_at = datetime.utcnow() - timedelta(hours=1)
    bracket.updated_at = datetime.utcnow() - timedelta(minutes=30)
    
    # Mock relationships
    bracket.entry_order = None
    bracket.stop_loss_order = None
    bracket.take_profit_order = None
    
    return bracket


#################################################
# Base Schema Tests
#################################################

class TestBaseSchemas:
    """Tests for base schemas used throughout the application."""
    
    def test_pagination_params_validation(self):
        """Test validation of pagination parameters."""
        # Valid data
        valid_data = {"page": 1, "limit": 100}
        params = PaginationParams(**valid_data)
        assert params.page == 1
        assert params.limit == 100
        
        # Default values
        params = PaginationParams()
        assert params.page == 1
        assert params.limit == 100
        
        # Invalid page (must be >= 1)
        with pytest.raises(ValidationError) as exc_info:
            PaginationParams(page=0)
        assert "less than or equal" in str(exc_info.value)
        
        # Invalid limit (must be between 1 and 1000)
        with pytest.raises(ValidationError) as exc_info:
            PaginationParams(limit=0)
        assert "less than or equal" in str(exc_info.value)
        
        with pytest.raises(ValidationError) as exc_info:
            PaginationParams(limit=1001)
        assert "greater than or equal" in str(exc_info.value)
    
    def test_time_range_params_validation(self):
        """Test validation of time range parameters."""
        # Valid data with both times
        now = datetime.utcnow()
        valid_data = {
            "start_time": now - timedelta(days=1),
            "end_time": now
        }
        params = TimeRangeParams(**valid_data)
        assert params.start_time == valid_data["start_time"]
        assert params.end_time == valid_data["end_time"]
        
        # Valid data with only start_time
        params = TimeRangeParams(start_time=now)
        assert params.start_time == now
        assert params.end_time is None
        
        # Valid data with only end_time
        params = TimeRangeParams(end_time=now)
        assert params.start_time is None
        assert params.end_time == now
        
        # Invalid: end_time before start_time
        with pytest.raises(ValidationError) as exc_info:
            TimeRangeParams(
                start_time=now,
                end_time=now - timedelta(days=1)
            )
        assert "end_time must be after start_time" in str(exc_info.value)
        
        # Both times as strings (test automatic conversion)
        valid_str_data = {
            "start_time": (now - timedelta(days=1)).isoformat(),
            "end_time": now.isoformat()
        }
        params = TimeRangeParams(**valid_str_data)
        assert isinstance(params.start_time, datetime)
        assert isinstance(params.end_time, datetime)
    
    def test_status_message(self):
        """Test StatusMessage validation."""
        # Valid data
        valid_data = {
            "success": True,
            "message": "Operation successful"
        }
        message = StatusMessage(**valid_data)
        assert message.success is True
        assert message.message == "Operation successful"
        
        # Missing required field
        with pytest.raises(ValidationError):
            StatusMessage(success=True)
        
        with pytest.raises(ValidationError):
            StatusMessage(message="Operation failed")


#################################################
# Order Schema Tests
#################################################

class TestOrderSchemas:
    """Tests for order-related schemas."""
    
    def test_order_base_validation(self, valid_order_data):
        """Test validation of OrderBase schema."""
        # Valid data
        order = OrderBase(**valid_order_data)
        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.LIMIT
        assert order.quantity == Decimal("100.0")
        assert order.price == Decimal("150.50")
        assert order.time_in_force == TimeInForce.DAY
        assert order.strategy_id == "mean_reversion_v1"
        assert order.tags == ["tech", "momentum"]
        
        # Test Decimal conversion
        assert isinstance(order.quantity, Decimal)
        assert isinstance(order.price, Decimal)
        
        # Missing required field
        invalid_data = valid_order_data.copy()
        del invalid_data["symbol"]
        with pytest.raises(ValidationError) as exc_info:
            OrderBase(**invalid_data)
        assert "field required" in str(exc_info.value)
        
        # Invalid symbol (too long)
        invalid_data = valid_order_data.copy()
        invalid_data["symbol"] = "A" * 30
        with pytest.raises(ValidationError) as exc_info:
            OrderBase(**invalid_data)
        assert "ensure this value has at most 20 characters" in str(exc_info.value)
        
        # Invalid quantity (negative)
        invalid_data = valid_order_data.copy()
        invalid_data["quantity"] = "-100"
        with pytest.raises(ValidationError) as exc_info:
            OrderBase(**invalid_data)
        assert "greater than" in str(exc_info.value)
        
        # Invalid price (negative)
        invalid_data = valid_order_data.copy()
        invalid_data["price"] = "-150.50"
        with pytest.raises(ValidationError) as exc_info:
            OrderBase(**invalid_data)
        assert "greater than" in str(exc_info.value)
    
    def test_order_type_validations(self, valid_order_data):
        """Test order type-specific validations."""
        # Limit order requires price
        limit_data = valid_order_data.copy()
        limit_data["order_type"] = "limit"
        del limit_data["price"]
        with pytest.raises(ValidationError) as exc_info:
            OrderBase(**limit_data)
        assert "Price is required for limit orders" in str(exc_info.value)
        
        # Stop order requires stop_price
        stop_data = valid_order_data.copy()
        stop_data["order_type"] = "stop"
        with pytest.raises(ValidationError) as exc_info:
            OrderBase(**stop_data)
        assert "Stop price is required for stop orders" in str(exc_info.value)
        
        # Stop limit order requires both price and stop_price
        stop_limit_data = valid_order_data.copy()
        stop_limit_data["order_type"] = "stop_limit"
        stop_limit_data["stop_price"] = "155.00"
        order = OrderBase(**stop_limit_data)
        assert order.stop_price == Decimal("155.00")
        
        # Trailing stop requires trailing parameters
        trailing_stop_data = valid_order_data.copy()
        trailing_stop_data["order_type"] = "trailing_stop"
        del trailing_stop_data["price"]
        
        with pytest.raises(ValidationError) as exc_info:
            OrderBase(**trailing_stop_data)
        assert "trailing_amount or trailing_percent is required" in str(exc_info.value)
        
        # Trailing stop with amount
        trailing_stop_data["trailing_amount"] = "5.00"
        order = OrderBase(**trailing_stop_data)
        assert order.trailing_amount == Decimal("5.00")
        
        # Trailing stop with percent
        trailing_stop_data = valid_order_data.copy()
        trailing_stop_data["order_type"] = "trailing_stop"
        del trailing_stop_data["price"]
        trailing_stop_data["trailing_percent"] = "2.5"
        order = OrderBase(**trailing_stop_data)
        assert order.trailing_percent == Decimal("2.5")
    
    def test_time_in_force_validations(self, valid_order_data):
        """Test time in force validations."""
        # GTD requires expire_at
        gtd_data = valid_order_data.copy()
        gtd_data["time_in_force"] = "gtd"
        with pytest.raises(ValidationError) as exc_info:
            OrderBase(**gtd_data)
        assert "Expiration time is required for GTD orders" in str(exc_info.value)
        
        # GTD with expire_at
        gtd_data["expire_at"] = datetime.utcnow() + timedelta(days=1)
        order = OrderBase(**gtd_data)
        assert order.expire_at is not None
        
        # Other time in force values don't require expire_at
        for tif in ["day", "gtc", "ioc", "fok"]:
            data = valid_order_data.copy()
            data["time_in_force"] = tif
            order = OrderBase(**data)
            assert order.time_in_force == tif
    
    def test_order_create_schema(self, valid_order_data):
        """Test OrderCreate schema validation."""
        # Valid data
        create_data = valid_order_data.copy()
        create_data["account_id"] = 123
        order = OrderCreate(**create_data)
        assert order.account_id == 123
        
        # Missing account_id
        with pytest.raises(ValidationError) as exc_info:
            OrderCreate(**valid_order_data)
        assert "field required" in str(exc_info.value)
        
        # Invalid account_id (negative)
        invalid_data = create_data.copy()
        invalid_data["account_id"] = -1
        with pytest.raises(ValidationError) as exc_info:
            OrderCreate(**invalid_data)
        assert "greater than" in str(exc_info.value)
        
        # With parent_order_id
        create_data["parent_order_id"] = str(uuid4())
        order = OrderCreate(**create_data)
        assert order.parent_order_id is not None
    
    def test_order_update_schema(self):
        """Test OrderUpdate schema validation."""
        # Valid data with all fields
        valid_data = {
            "price": "152.75",
            "quantity": "150",
            "time_in_force": "gtc",
            "notes": "Updated notes"
        }
        update = OrderUpdate(**valid_data)
        assert update.price == Decimal("152.75")
        assert update.quantity == Decimal("150")
        assert update.time_in_force == TimeInForce.GTC
        assert update.notes == "Updated notes"
        
        # Empty update is valid (no fields are required)
        empty_update = OrderUpdate()
        assert empty_update.price is None
        assert empty_update.quantity is None
        
        # Invalid price (negative)
        with pytest.raises(ValidationError) as exc_info:
            OrderUpdate(price="-10")
        assert "greater than" in str(exc_info.value)
    
    def test_order_status_update_schema(self):
        """Test OrderStatusUpdate schema validation."""
        # Valid data
        valid_data = {
            "status": "canceled",
            "reason": "Manual cancellation"
        }
        update = OrderStatusUpdate(**valid_data)
        assert update.status == OrderStatus.CANCELED
        assert update.reason == "Manual cancellation"
        
        # Status is required
        with pytest.raises(ValidationError) as exc_info:
            OrderStatusUpdate(reason="Manual cancellation")
        assert "field required" in str(exc_info.value)
        
        # Reason is optional
        update = OrderStatusUpdate(status="filled")
        assert update.status == OrderStatus.FILLED
        assert update.reason is None
        
        # Invalid status
        with pytest.raises(ValidationError) as exc_info:
            OrderStatusUpdate(status="not_a_valid_status")
        assert "not a valid enumeration member" in str(exc_info.value)
    
    def test_execution_create_schema(self, valid_execution_data):
        """Test ExecutionCreate schema validation."""
        # Valid data
        execution = ExecutionCreate(**valid_execution_data)
        assert execution.quantity == Decimal("50.5")
        assert execution.price == Decimal("150.25")
        assert execution.fees == Decimal("0.75")
        assert execution.liquidity == "taker"
        
        # Required fields
        with pytest.raises(ValidationError) as exc_info:
            # Missing quantity
            invalid_data = valid_execution_data.copy()
            del invalid_data["quantity"]
            ExecutionCreate(**invalid_data)
        assert "field required" in str(exc_info.value)
        
        # Invalid quantity (negative)
        with pytest.raises(ValidationError) as exc_info:
            invalid_data = valid_execution_data.copy()
            invalid_data["quantity"] = "-50"
            ExecutionCreate(**invalid_data)
        assert "Execution quantity must be positive" in str(exc_info.value)
        
        # Optional fields can be omitted
        min_data = {
            "quantity": "50",
            "price": "150.25"
        }
        execution = ExecutionCreate(**min_data)
        assert execution.execution_id is None
        assert execution.fees is None
    
    def test_order_response_schema(self, mock_order):
        """Test OrderResponse schema validation and conversion."""
        # Convert model to schema
        order_response = order_model_to_schema(mock_order)
        
        # Verify fields
        assert order_response.id == mock_order.id
        assert order_response.order_id == mock_order.order_id
        assert order_response.symbol == mock_order.symbol
        assert order_response.side == mock_order.side
        assert order_response.order_type == mock_order.order_type
        assert order_response.quantity == mock_order.quantity
        assert order_response.price == mock_order.price
        assert order_response.status == mock_order.status
        assert order_response.tags == ["tech", "momentum"]
        
        # Computed fields
        assert order_response.is_active == mock_order.is_active
        assert order_response.fill_percent == float(mock_order.fill_percent)
        assert order_response.value == float(mock_order.value)
        
        # Create from dict - make sure all fields validate
        order_dict = order_response.model_dump()
        order_from_dict = OrderResponse(**order_dict)
        assert order_from_dict.id == mock_order.id
        assert order_from_dict.status == mock_order.status
    
    def test_order_list_response(self, mock_order):
        """Test OrderListResponse schema."""
        # Create response with single order
        order_response = order_model_to_schema(mock_order)
        list_response = OrderListResponse(
            items=[order_response],
            total=42,
            page=1,
            limit=10,
            pages=5
        )
        
        # Verify fields
        assert len(list_response.items) == 1
        assert list_response.items[0].id == mock_order.id
        assert list_response.total == 42
        assert list_response.page == 1
        assert list_response.limit == 10
        assert list_response.pages == 5
        
        # Create from dict
        response_dict = list_response.model_dump()
        response_from_dict = OrderListResponse(**response_dict)
        assert len(response_from_dict.items) == 1
        assert response_from_dict.total == 42
    
    def test_order_filter_schema(self):
        """Test OrderFilter schema validation."""
        # Valid data with all fields
        valid_data = {
            "account_id": 123,
            "symbol": "AAPL",
            "status": ["submitted", "partially_filled"],
            "side": "buy",
            "order_type": "limit",
            "strategy_id": "mean_reversion_v1",
            "is_active": True,
            "page": 2,
            "limit": 50,
            "start_time": datetime.utcnow() - timedelta(days=1),
            "end_time": datetime.utcnow()
        }
        
        filter_params = OrderFilter(**valid_data)
        assert filter_params.account_id == 123
        assert filter_params.symbol == "AAPL"
        assert filter_params.status == [OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]
        assert filter_params.side == OrderSide.BUY
        assert filter_params.order_type == OrderType.LIMIT
        assert filter_params.is_active is True
        assert filter_params.page == 2
        assert filter_params.limit == 50
        
        # Minimal filter (everything optional)
        min_filter = OrderFilter()
        assert min_filter.account_id is None
        assert min_filter.symbol is None
        assert min_filter.status is None
        assert min_filter.page == 1  # Default value from PaginationParams
        
        # Status can be a single value or list
        single_status = OrderFilter(status="filled")
        assert single_status.status == [OrderStatus.FILLED]
        
        multi_status = OrderFilter(status=["filled", "canceled"])
        assert multi_status.status == [OrderStatus.FILLED, OrderStatus.CANCELED]


#################################################
# Position Schema Tests
#################################################

class TestPositionSchemas:
    """Tests for position-related schemas."""
    
    def test_position_base_validation(self, valid_position_data):
        """Test validation of PositionBase schema."""
        # Valid data
        position = PositionBase(**valid_position_data)
        assert position.symbol == "AAPL"
        assert position.direction == PositionDirection.LONG
        assert position.quantity == Decimal("100.5")
        assert position.average_entry_price == Decimal("150.25")
        assert position.current_price == Decimal("152.50")
        
        # Missing required field
        invalid_data = valid_position_data.copy()
        del invalid_data["symbol"]
        with pytest.raises(ValidationError) as exc_info:
            PositionBase(**invalid_data)
        assert "field required" in str(exc_info.value)
        
        # Invalid quantity (negative)
        invalid_data = valid_position_data.copy()
        invalid_data["quantity"] = "-100"
        with pytest.raises(ValidationError) as exc_info:
            PositionBase(**invalid_data)
        assert "greater than" in str(exc_info.value)
        
        # Invalid direction
        invalid_data = valid_position_data.copy()
        invalid_data["direction"] = "invalid"
        with pytest.raises(ValidationError) as exc_info:
            PositionBase(**invalid_data)
        assert "not a valid enumeration member" in str(exc_info.value)
    
    def test_position_create_schema(self, valid_position_data):
        """Test PositionCreate schema validation."""
        # Valid data
        create_data = valid_position_data.copy()
        create_data["account_id"] = 123
        position = PositionCreate(**create_data)
        assert position.account_id == 123
        assert position.realized_pnl == Decimal("0")
        assert position.unrealized_pnl is None
        
        # With unrealized PNL
        create_data["unrealized_pnl"] = "225.00"
        position = PositionCreate(**create_data)
        assert position.unrealized_pnl == Decimal("225.00")
        
        # Missing account_id
        with pytest.raises(ValidationError) as exc_info:
            PositionCreate(**valid_position_data)
        assert "field required" in str(exc_info.value)
        
        # Invalid account_id (non-positive)
        invalid_data = create_data.copy()
        invalid_data["account_id"] = 0
        with pytest.raises(ValidationError) as exc_info:
            PositionCreate(**invalid_data)
        assert "greater than" in str(exc_info.value)
    
    def test_position_update_schema(self):
        """Test PositionUpdate schema validation."""
        # Valid data with all fields
        valid_data = {
            "current_price": "155.75",
            "quantity": "120",
            "average_entry_price": "151.25",
            "realized_pnl": "300.00",
            "unrealized_pnl": "550.00"
        }
        
        update = PositionUpdate(**valid_data)
        assert update.current_price == Decimal("155.75")
        assert update.quantity == Decimal("120")
        assert update.realized_pnl == Decimal("300.00")
        assert update.unrealized_pnl == Decimal("550.00")
        
        # Empty update is valid (no fields are required)
        empty_update = PositionUpdate()
        assert empty_update.current_price is None
        assert empty_update.quantity is None
        
        # Invalid values
        with pytest.raises(ValidationError) as exc_info:
            PositionUpdate(current_price="-10")
        assert "greater than" in str(exc_info.value)
        
        with pytest.raises(ValidationError) as exc_info:
            PositionUpdate(quantity="-50")
        assert "greater than" in str(exc_info.value)
    
    def test_position_risk_update_schema(self):
        """Test PositionRiskUpdate schema validation."""
        # Valid data with all fields
        valid_data = {
            "stop_loss_price": "145.00",
            "take_profit_price": "165.00",
            "trailing_stop_percent": "5",
            "trailing_stop_activation_price": "160.00"
        }
        
        update = PositionRiskUpdate(**valid_data)
        assert update.stop_loss_price == Decimal("145.00")
        assert update.take_profit_price == Decimal("165.00")
        assert update.trailing_stop_percent == Decimal("5")
        assert update.trailing_stop_activation_price == Decimal("160.00")
        
        # With trailing_stop_distance instead of percent
        distance_data = {
            "stop_loss_price": "145.00",
            "trailing_stop_distance": "5.00",
            "trailing_stop_activation_price": "160.00"
        }
        
        update = PositionRiskUpdate(**distance_data)
        assert update.trailing_stop_distance == Decimal("5.00")
        
        # Trailing activation price without distance or percent
        invalid_data = {
            "trailing_stop_activation_price": "160.00"
        }
        
        with pytest.raises(ValidationError) as exc_info:
            PositionRiskUpdate(**invalid_data)
        assert "trailing_stop_distance or trailing_stop_percent must be provided" in str(exc_info.value)
        
        # Invalid values
        with pytest.raises(ValidationError) as exc_info:
            PositionRiskUpdate(trailing_stop_distance="-5.00")
        assert "greater than" in str(exc_info.value)
        
        with pytest.raises(ValidationError) as exc_info:
            PositionRiskUpdate(trailing_stop_percent="101")
        assert "less than or equal to" in str(exc_info.value)
    
    def test_position_response_schema(self, mock_position):
        """Test PositionResponse schema validation and conversion."""
        # Convert model to schema
        position_response = position_model_to_schema(mock_position)
        
        # Verify fields
        assert position_response.id == mock_position.id
        assert position_response.account_id == mock_position.account_id
        assert position_response.symbol == mock_position.symbol
        assert position_response.direction == mock_position.direction
        assert position_response.quantity == mock_position.quantity
        assert position_response.average_entry_price == mock_position.average_entry_price
        assert position_response.current_price == mock_position.current_price
        assert position_response.realized_pnl == mock_position.realized_pnl
        assert position_response.unrealized_pnl == mock_position.unrealized_pnl
        assert position_response.total_pnl == mock_position.total_pnl
        assert position_response.pnl_percentage == mock_position.pnl_percentage
        assert position_response.market_value == mock_position.market_value
        assert position_response.cost_basis == mock_position.cost_basis
        assert position_response.stop_loss_price == mock_position.stop_loss_price
        assert position_response.take_profit_price == mock_position.take_profit_price
        
        # Create from dict - make sure all fields validate
        position_dict = position_response.model_dump()
        position_from_dict = PositionResponse(**position_dict)
        assert position_from_dict.id == mock_position.id
        assert position_from_dict.symbol == mock_position.symbol
    
    def test_position_list_response(self, mock_position):
        """Test PositionListResponse schema."""
        # Create response with single position
        position_response = position_model_to_schema(mock_position)
        list_response = PositionListResponse(
            items=[position_response],
            total=5
        )
        
        # Verify fields
        assert len(list_response.items) == 1
        assert list_response.items[0].id == mock_position.id
        assert list_response.total == 5
        
        # Create from dict
        response_dict = list_response.model_dump()
        response_from_dict = PositionListResponse(**response_dict)
        assert len(response_from_dict.items) == 1
        assert response_from_dict.total == 5
    
    def test_position_filter_schema(self):
        """Test PositionFilter schema validation."""
        # Valid data with all fields
        valid_data = {
            "account_id": 123,
            "symbol": "AAPL",
            "direction": "long",
            "strategy_id": "mean_reversion_v1",
            "min_quantity": "10"
        }
        
        filter_params = PositionFilter(**valid_data)
        assert filter_params.account_id == 123
        assert filter_params.symbol == "AAPL"
        assert filter_params.direction == PositionDirection.LONG
        assert filter_params.strategy_id == "mean_reversion_v1"
        assert filter_params.min_quantity == Decimal("10")
        
        # Minimal filter (everything optional)
        min_filter = PositionFilter()
        assert min_filter.account_id is None
        assert min_filter.symbol is None
        assert min_filter.direction is None
        assert min_filter.min_quantity is None
        
        # Invalid min_quantity
        with pytest.raises(ValidationError) as exc_info:
            PositionFilter(min_quantity="-10")
        assert "greater than" in str(exc_info.value)


#################################################
# Trade Schema Tests
#################################################

class TestTradeSchemas:
    """Tests for trade-related schemas."""
    
    def test_trade_base_validation(self, valid_trade_data):
        """Test validation of TradeBase schema."""
        # Valid data
        trade = TradeBase(**valid_trade_data)
        assert trade.symbol == "AAPL"
        assert trade.side == OrderSide.BUY
        assert trade.quantity == Decimal("100.5")
        assert trade.price == Decimal("150.25")
        assert trade.fees == Decimal("0.75")
        assert trade.order_id == valid_trade_data["order_id"]
        assert trade.execution_id == "ex-67890"
        
        # Test default values
        trade_no_fees = valid_trade_data.copy()
        del trade_no_fees["fees"]
        trade = TradeBase(**trade_no_fees)
        assert trade.fees == Decimal("0")
        
        # Missing required fields
        for field in ["symbol", "side", "quantity", "price", "order_id", "executed_at"]:
            invalid_data = valid_trade_data.copy()
            del invalid_data[field]
            with pytest.raises(ValidationError) as exc_info:
                TradeBase(**invalid_data)
            assert "field required" in str(exc_info.value)
        
        # Invalid quantity (non-positive)
        invalid_data = valid_trade_data.copy()
        invalid_data["quantity"] = "0"
        with pytest.raises(ValidationError) as exc_info:
            TradeBase(**invalid_data)
        assert "greater than" in str(exc_info.value)
        
        # Invalid price (non-positive)
        invalid_data = valid_trade_data.copy()
        invalid_data["price"] = "0"
        with pytest.raises(ValidationError) as exc_info:
            TradeBase(**invalid_data)
        assert "greater than" in str(exc_info.value)
        
        # Invalid fees (negative)
        invalid_data = valid_trade_data.copy()
        invalid_data["fees"] = "-0.5"
        with pytest.raises(ValidationError) as exc_info:
            TradeBase(**invalid_data)
        assert "greater than or equal to" in str(exc_info.value)
    
    def test_trade_create_schema(self, valid_trade_data):
        """Test TradeCreate schema validation."""
        # Valid data
        create_data = valid_trade_data.copy()
        create_data["account_id"] = 123
        trade = TradeCreate(**create_data)
        assert trade.account_id == 123
        
        # Verify auto-calculated values
        assert trade.value == Decimal("100.5") * Decimal("150.25")
        assert trade.total_cost == (Decimal("100.5") * Decimal("150.25")) + Decimal("0.75")
        
        # Explicitly provided values override calculations
        create_data["value"] = "16000.00"
        create_data["total_cost"] = "16001.50"
        trade = TradeCreate(**create_data)
        assert trade.value == Decimal("16000.00")
        assert trade.total_cost == Decimal("16001.50")
        
        # Missing account_id
        with pytest.raises(ValidationError) as exc_info:
            TradeCreate(**valid_trade_data)
        assert "field required" in str(exc_info.value)
        
        # Invalid account_id (non-positive)
        invalid_data = create_data.copy()
        invalid_data["account_id"] = 0
        with pytest.raises(ValidationError) as exc_info:
            TradeCreate(**invalid_data)
        assert "greater than" in str(exc_info.value)
        
        # Optional fields
        min_data = {
            "account_id": 123,
            "symbol": "AAPL",
            "side": "buy",
            "quantity": "100",
            "price": "150.25",
            "order_id": str(uuid4()),
            "executed_at": datetime.utcnow().isoformat()
        }
        trade = TradeCreate(**min_data)
        assert trade.realized_pnl is None
        assert trade.tax_lot_id is None
        assert trade.wash_sale is False  # Default
    
    def test_trade_response_schema(self, mock_trade):
        """Test TradeResponse schema validation and conversion."""
        # Convert model to schema
        trade_response = trade_model_to_schema(mock_trade)
        
        # Verify fields
        assert trade_response.id == mock_trade.id
        assert trade_response.trade_id == mock_trade.trade_id
        assert trade_response.account_id == mock_trade.account_id
        assert trade_response.symbol == mock_trade.symbol
        assert trade_response.side == mock_trade.side
        assert trade_response.quantity == mock_trade.quantity
        assert trade_response.price == mock_trade.price
        assert trade_response.fees == mock_trade.fees
        assert trade_response.value == mock_trade.value
        assert trade_response.total_cost == mock_trade.total_cost
        assert trade_response.realized_pnl == mock_trade.realized_pnl
        assert trade_response.order_id == mock_trade.order_id
        assert trade_response.execution_id == mock_trade.execution_id
        assert trade_response.wash_sale == mock_trade.wash_sale
        assert trade_response.executed_at == mock_trade.executed_at
        
        # Create from dict - make sure all fields validate
        trade_dict = trade_response.model_dump()
        trade_from_dict = TradeResponse(**trade_dict)
        assert trade_from_dict.id == mock_trade.id
        assert trade_from_dict.symbol == mock_trade.symbol
    
    def test_trade_list_response(self, mock_trade):
        """Test TradeListResponse schema."""
        # Create response with single trade
        trade_response = trade_model_to_schema(mock_trade)
        list_response = TradeListResponse(
            items=[trade_response],
            total=25,
            page=1,
            limit=10,
            pages=3
        )
        
        # Verify fields
        assert len(list_response.items) == 1
        assert list_response.items[0].id == mock_trade.id
        assert list_response.total == 25
        assert list_response.page == 1
        assert list_response.limit == 10
        assert list_response.pages == 3
        
        # Create from dict
        response_dict = list_response.model_dump()
        response_from_dict = TradeListResponse(**response_dict)
        assert len(response_from_dict.items) == 1
        assert response_from_dict.total == 25
    
    def test_trade_filter_schema(self):
        """Test TradeFilter schema validation."""
        # Valid data with all fields
        start_time = datetime.utcnow() - timedelta(days=30)
        end_time = datetime.utcnow()
        
        valid_data = {
            "account_id": 123,
            "symbol": "AAPL",
            "side": "buy",
            "min_value": "1000",
            "strategy_id": "mean_reversion_v1",
            "start_time": start_time,
            "end_time": end_time,
            "page": 2,
            "limit": 50
        }
        
        filter_params = TradeFilter(**valid_data)
        assert filter_params.account_id == 123
        assert filter_params.symbol == "AAPL"
        assert filter_params.side == OrderSide.BUY
        assert filter_params.min_value == Decimal("1000")
        assert filter_params.strategy_id == "mean_reversion_v1"
        assert filter_params.start_time == start_time
        assert filter_params.end_time == end_time
        assert filter_params.page == 2
        assert filter_params.limit == 50
        
        # Minimal filter (everything optional)
        min_filter = TradeFilter()
        assert min_filter.account_id is None
        assert min_filter.symbol is None
        assert min_filter.side is None
        assert min_filter.min_value is None
        assert min_filter.page == 1  # Default from PaginationParams
        
        # Invalid min_value
        with pytest.raises(ValidationError) as exc_info:
            TradeFilter(min_value="-100")
        assert "greater than" in str(exc_info.value)
        
        # Invalid time range
        with pytest.raises(ValidationError) as exc_info:
            TradeFilter(
                start_time=end_time,
                end_time=start_time
            )
        assert "end_time must be after start_time" in str(exc_info.value)


#################################################
# Bracket Order Schema Tests
#################################################

class TestBracketOrderSchemas:
    """Tests for bracket order-related schemas."""
    
    def test_bracket_order_create_schema(self, valid_bracket_order_data):
        """Test BracketOrderCreate schema validation."""
        # Valid data
        bracket = BracketOrderCreate(**valid_bracket_order_data)
        assert bracket.account_id == 123
        assert bracket.symbol == "AAPL"
        assert bracket.side == OrderSide.BUY
        assert bracket.quantity == Decimal("100.5")
        assert bracket.entry_type == OrderType.LIMIT
        assert bracket.entry_price == Decimal("150.00")
        assert bracket.stop_loss_price == Decimal("145.00")
        assert bracket.take_profit_price == Decimal("160.00")
        assert bracket.time_in_force == TimeInForce.GTC
        
        # Missing required fields
        for field in ["account_id", "symbol", "side", "quantity", "entry_type"]:
            invalid_data = valid_bracket_order_data.copy()
            del invalid_data[field]
            with pytest.raises(ValidationError) as exc_info:
                BracketOrderCreate(**invalid_data)
            assert "field required" in str(exc_info.value)
        
        # Entry price required for limit orders
        limit_data = valid_bracket_order_data.copy()
        limit_data["entry_type"] = "limit"
        del limit_data["entry_price"]
        with pytest.raises(ValidationError) as exc_info:
            BracketOrderCreate(**limit_data)
        assert "Entry price is required for limit orders" in str(exc_info.value)
        
        # Market order doesn't require entry price
        market_data = valid_bracket_order_data.copy()
        market_data["entry_type"] = "market"
        del market_data["entry_price"]
        bracket = BracketOrderCreate(**market_data)
        assert bracket.entry_price is None
        
        # Stop loss and take profit validation for buy orders
        buy_invalid_sl = valid_bracket_order_data.copy()
        buy_invalid_sl["side"] = "buy"
        buy_invalid_sl["stop_loss_price"] = "151.00"  # Should be below entry
        with pytest.raises(ValidationError) as exc_info:
            BracketOrderCreate(**buy_invalid_sl)
        assert "Stop loss price must be below entry price for buy orders" in str(exc_info.value)
        
        buy_invalid_tp = valid_bracket_order_data.copy()
        buy_invalid_tp["side"] = "buy"
        buy_invalid_tp["take_profit_price"] = "149.00"  # Should be above entry
        with pytest.raises(ValidationError) as exc_info:
            BracketOrderCreate(**buy_invalid_tp)
        assert "Take profit price must be above entry price for buy orders" in str(exc_info.value)
        
        # Stop loss and take profit validation for sell orders
        sell_valid = valid_bracket_order_data.copy()
        sell_valid["side"] = "sell"
        sell_valid["stop_loss_price"] = "155.00"  # Above entry for sell
        sell_valid["take_profit_price"] = "145.00"  # Below entry for sell
        bracket = BracketOrderCreate(**sell_valid)
        assert bracket.stop_loss_price == Decimal("155.00")
        assert bracket.take_profit_price == Decimal("145.00")
        
        sell_invalid_sl = valid_bracket_order_data.copy()
        sell_invalid_sl["side"] = "sell"
        sell_invalid_sl["stop_loss_price"] = "149.00"  # Should be above entry
        with pytest.raises(ValidationError) as exc_info:
            BracketOrderCreate(**sell_invalid_sl)
        assert "Stop loss price must be above entry price for sell orders" in str(exc_info.value)
        
        sell_invalid_tp = valid_bracket_order_data.copy()
        sell_invalid_tp["side"] = "sell"
        sell_invalid_tp["take_profit_price"] = "151.00"  # Should be below entry
        with pytest.raises(ValidationError) as exc_info:
            BracketOrderCreate(**sell_invalid_tp)
        assert "Take profit price must be below entry price for sell orders" in str(exc_info.value)
    
    def test_bracket_order_response_schema(self, mock_bracket_order):
        """Test BracketOrderResponse schema validation and conversion."""
        # Convert model to schema
        bracket_response = bracket_order_model_to_schema(mock_bracket_order)
        
        # Verify fields
        assert bracket_response.id == mock_bracket_order.id
        assert bracket_response.account_id == mock_bracket_order.account_id
        assert bracket_response.symbol == mock_bracket_order.symbol
        assert bracket_response.status == mock_bracket_order.status
        assert bracket_response.entry_order_id == mock_bracket_order.entry_order_id
        assert bracket_response.stop_loss_order_id == mock_bracket_order.stop_loss_order_id
        assert bracket_response.take_profit_order_id == mock_bracket_order.take_profit_order_id
        assert bracket_response.strategy_id == mock_bracket_order.strategy_id
        assert bracket_response.created_at == mock_bracket_order.created_at
        assert bracket_response.updated_at == mock_bracket_order.updated_at
        
        # Default None values for orders
        assert bracket_response.entry_order is None
        assert bracket_response.stop_loss_order is None
        assert bracket_response.take_profit_order is None
        
        # Create from dict - make sure all fields validate
        bracket_dict = bracket_response.model_dump()
        bracket_from_dict = BracketOrderResponse(**bracket_dict)
        assert bracket_from_dict.id == mock_bracket_order.id
        assert bracket_from_dict.symbol == mock_bracket_order.symbol
    
    def test_bracket_order_list_response(self, mock_bracket_order):
        """Test BracketOrderListResponse schema."""
        # Create response with single bracket order
        bracket_response = bracket_order_model_to_schema(mock_bracket_order)
        list_response = BracketOrderListResponse(
            items=[bracket_response],
            total=5
        )
        
        # Verify fields
        assert len(list_response.items) == 1
        assert list_response.items[0].id == mock_bracket_order.id
        assert list_response.total == 5
        
        # Create from dict
        response_dict = list_response.model_dump()
        response_from_dict = BracketOrderListResponse(**response_dict)
        assert len(response_from_dict.items) == 1
        assert response_from_dict.total == 5


#################################################
# OCO Order Schema Tests
#################################################

class TestOCOOrderSchemas:
    """Tests for OCO (One-Cancels-Other) order schemas."""
    
    def test_oco_order_create_schema(self, valid_oco_order_data):
        """Test OCOOrderCreate schema validation."""
        # Valid data
        oco = OCOOrderCreate(**valid_oco_order_data)
        assert oco.account_id == 123
        assert oco.symbol == "AAPL"
        assert oco.side == OrderSide.BUY
        assert oco.quantity == Decimal("100.5")
        assert oco.price_1 == Decimal("145.00")
        assert oco.type_1 == OrderType.LIMIT
        assert oco.price_2 == Decimal("155.00")
        assert oco.type_2 == OrderType.STOP
        assert oco.time_in_force == TimeInForce.DAY
        
        # Missing required fields
        for field in ["account_id", "symbol", "side", "quantity", "price_1", "type_1", "price_2", "type_2"]:
            invalid_data = valid_oco_order_data.copy()
            del invalid_data[field]
            with pytest.raises(ValidationError) as exc_info:
                OCOOrderCreate(**invalid_data)
            assert "field required" in str(exc_info.value)
        
        # Invalid order types
        invalid_type = valid_oco_order_data.copy()
        invalid_type["type_1"] = "market"  # Market orders not valid for OCO
        with pytest.raises(ValidationError) as exc_info:
            OCOOrderCreate(**invalid_type)
        assert "First order type must be one of:" in str(exc_info.value)
        
        invalid_type = valid_oco_order_data.copy()
        invalid_type["type_2"] = "market"  # Market orders not valid for OCO
        with pytest.raises(ValidationError) as exc_info:
            OCOOrderCreate(**invalid_type)
        assert "Second order type must be one of:" in str(exc_info.value)
        
        # Same order types
        same_types = valid_oco_order_data.copy()
        same_types["type_1"] = "limit"
        same_types["type_2"] = "limit"
        with pytest.raises(ValidationError) as exc_info:
            OCOOrderCreate(**same_types)
        assert "OCO orders should have different order types" in str(exc_info.value)
        
        # Price validation for buy orders
        buy_invalid = valid_oco_order_data.copy()
        buy_invalid["side"] = "buy"
        buy_invalid["type_1"] = "limit"
        buy_invalid["type_2"] = "stop"
        buy_invalid["price_1"] = "160.00"  # Limit buy should be below stop buy
        buy_invalid["price_2"] = "155.00"
        with pytest.raises(ValidationError) as exc_info:
            OCOOrderCreate(**buy_invalid)
        assert "For buy OCO, limit price should be below stop price" in str(exc_info.value)
        
        # Price validation for sell orders
        sell_valid = valid_oco_order_data.copy()
        sell_valid["side"] = "sell"
        sell_valid["type_1"] = "limit"
        sell_valid["type_2"] = "stop"
        sell_valid["price_1"] = "155.00"  # Limit sell above stop sell
        sell_valid["price_2"] = "145.00"
        oco = OCOOrderCreate(**sell_valid)
        assert oco.price_1 == Decimal("155.00")
        assert oco.price_2 == Decimal("145.00")
        
        sell_invalid = valid_oco_order_data.copy()
        sell_invalid["side"] = "sell"
        sell_invalid["type_1"] = "limit"
        sell_invalid["type_2"] = "stop"
        sell_invalid["price_1"] = "145.00"  # Limit sell should be above stop sell
        sell_invalid["price_2"] = "155.00"
        with pytest.raises(ValidationError) as exc_info:
            OCOOrderCreate(**sell_invalid)
        assert "For sell OCO, limit price should be above stop price" in str(exc_info.value)


#################################################
# Model-to-Schema Conversion Tests
#################################################

class TestConversionFunctions:
    """Tests for model-to-schema conversion functions."""
    
    def test_order_model_to_schema(self, mock_order, mock_execution, mock_order_event):
        """Test conversion of Order model to OrderResponse schema."""
        # Basic conversion
        response = order_model_to_schema(mock_order)
        assert response.id == mock_order.id
        assert response.order_id == mock_order.order_id
        assert response.symbol == mock_order.symbol
        assert response.tags == ["tech", "momentum"]
        
        # With executions
        mock_order.executions = [mock_execution]
        response = order_model_to_schema(mock_order, include_executions=True)
        assert len(response.executions) == 1
        assert response.executions[0].id == mock_execution.id
        
        # With events
        mock_order.order_events = [mock_order_event]
        response = order_model_to_schema(mock_order, include_events=True)
        assert len(response.events) == 1
        assert response.events[0].id == mock_order_event.id
    
    def test_execution_model_to_schema(self, mock_execution):
        """Test conversion of Execution model to ExecutionResponse schema."""
        response = execution_model_to_schema(mock_execution)
        assert response.id == mock_execution.id
        assert response.order_id == mock_execution.order_id
        assert response.quantity == mock_execution.quantity
        assert response.price == mock_execution.price
        assert response.fees == mock_execution.fees
        assert response.executed_at == mock_execution.executed_at
    
    def test_order_event_model_to_schema(self, mock_order_event):
        """Test conversion of OrderEvent model to OrderEventResponse schema."""
        response = order_event_model_to_schema(mock_order_event)
        assert response.id == mock_order_event.id
        assert response.order_id == mock_order_event.order_id
        assert response.event_type == mock_order_event.event_type
        assert response.description == mock_order_event.description
        assert response.event_data == mock_order_event.data
        assert response.created_at == mock_order_event.created_at
    
    def test_position_model_to_schema(self, mock_position):
        """Test conversion of Position model to PositionResponse schema."""
        response = position_model_to_schema(mock_position)
        assert response.id == mock_position.id
        assert response.account_id == mock_position.account_id
        assert response.symbol == mock_position.symbol
        assert response.direction == mock_position.direction
        assert response.quantity == mock_position.quantity
        assert response.average_entry_price == mock_position.average_entry_price
        assert response.current_price == mock_position.current_price
        assert response.realized_pnl == mock_position.realized_pnl
        assert response.unrealized_pnl == mock_position.unrealized_pnl
        assert response.total_pnl == mock_position.total_pnl
        assert response.pnl_percentage == mock_position.pnl_percentage
        assert response.market_value == mock_position.market_value
        assert response.cost_basis == mock_position.cost_basis
    
    def test_trade_model_to_schema(self, mock_trade):
        """Test conversion of Trade model to TradeResponse schema."""
        response = trade_model_to_schema(mock_trade)
        assert response.id == mock_trade.id
        assert response.trade_id == mock_trade.trade_id
        assert response.account_id == mock_trade.account_id
        assert response.symbol == mock_trade.symbol
        assert response.side == mock_trade.side
        assert response.quantity == mock_trade.quantity
        assert response.price == mock_trade.price
        assert response.fees == mock_trade.fees
        assert response.value == mock_trade.value
        assert response.total_cost == mock_trade.total_cost
        assert response.realized_pnl == mock_trade.realized_pnl
        assert response.executed_at == mock_trade.executed_at
    
    def test_bracket_order_model_to_schema(self, mock_bracket_order, mock_order):
        """Test conversion of BracketOrder model to BracketOrderResponse schema."""
        # Basic conversion
        response = bracket_order_model_to_schema(mock_bracket_order)
        assert response.id == mock_bracket_order.id
        assert response.account_id == mock_bracket_order.account_id
        assert response.symbol == mock_bracket_order.symbol
        assert response.status == mock_bracket_order.status
        assert response.entry_order_id == mock_bracket_order.entry_order_id
        assert response.stop_loss_order_id == mock_bracket_order.stop_loss_order_id
        assert response.take_profit_order_id == mock_bracket_order.take_profit_order_id
        
        # Without included orders
        assert response.entry_order is None
        assert response.stop_loss_order is None
        assert response.take_profit_order is None
        
        # With included orders
        mock_bracket_order.entry_order = mock_order
        response = bracket_order_model_to_schema(mock_bracket_order, include_orders=True)
        assert response.entry_order is not None
        assert response.entry_order.id == mock_order.id
        assert response.entry_order.symbol == mock_order.symbol


#################################################
# Edge Case and Comprehensive Tests
#################################################

class TestEdgeCases:
    """Tests for edge cases and complex validations."""
    
    def test_decimal_precision_handling(self, valid_order_data):
        """Test handling of decimal precision."""
        # Very large decimal values
        large_value_data = valid_order_data.copy()
        large_value_data["quantity"] = "9999999999999999.99999999"
        large_value_data["price"] = "9999999999999999.99999999"
        
        order = OrderBase(**large_value_data)
        assert order.quantity == Decimal("9999999999999999.99999999")
        assert order.price == Decimal("9999999999999999.99999999")
        
        # Very small decimal values
        small_value_data = valid_order_data.copy()
        small_value_data["quantity"] = "0.00000001"
        small_value_data["price"] = "0.00000001"
        
        order = OrderBase(**small_value_data)
        assert order.quantity == Decimal("0.00000001")
        assert order.price == Decimal("0.00000001")
        
        # Non-standard decimal notation
        notation_data = valid_order_data.copy()
        notation_data["quantity"] = "1.5e2"  # Scientific notation
        
        order = OrderBase(**notation_data)
        assert order.quantity == Decimal("150")
        
        # String decimal with trailing zeros
        trailing_zeros_data = valid_order_data.copy()
        trailing_zeros_data["quantity"] = "100.500000"
        
        order = OrderBase(**trailing_zeros_data)
        assert order.quantity == Decimal("100.500000")
        assert str(order.quantity) == "100.500000"  # Preserves trailing zeros
    
    def test_enum_values_and_serialization(self):
        """Test handling of enum values and serialization."""
        # Test all possible enum values for order status
        for status in OrderStatus:
            update = OrderStatusUpdate(status=status.value)
            assert update.status == status.value
        
        # Test serialization and deserialization of enums
        update = OrderStatusUpdate(status="submitted")
        serialized = update.model_dump()
        assert serialized["status"] == "submitted"
        
        deserialized = OrderStatusUpdate(**serialized)
        assert deserialized.status == OrderStatus.SUBMITTED
        
        # Test case sensitivity in enum validation
        with pytest.raises(ValidationError):
            OrderStatusUpdate(status="SUBMITTED")  # All caps not valid
        
        with pytest.raises(ValidationError):
            OrderStatusUpdate(status="Submitted")  # Title case not valid
    
    def test_date_time_formats(self):
        """Test handling of various date/time formats."""
        # ISO format string
        params = TimeRangeParams(
            start_time="2023-01-01T00:00:00Z",
            end_time="2023-12-31T23:59:59Z"
        )
        assert isinstance(params.start_time, datetime)
        assert isinstance(params.end_time, datetime)
        assert params.start_time.year == 2023
        assert params.start_time.month == 1
        assert params.start_time.day == 1
        assert params.end_time.year == 2023
        assert params.end_time.month == 12
        assert params.end_time.day == 31
        
        # Different ISO formats
        formats = [
            "2023-01-01T12:30:45",  # No timezone
            "2023-01-01T12:30:45Z",  # UTC
            "2023-01-01T12:30:45+00:00",  # UTC with offset
            "2023-01-01T12:30:45.123456Z",  # With microseconds
            "2023-01-01 12:30:45Z",  # Space instead of T
        ]
        
        for dt_format in formats:
            params = TimeRangeParams(start_time=dt_format)
            assert isinstance(params.start_time, datetime)
            assert params.start_time.hour == 12
            assert params.start_time.minute == 30
            assert params.start_time.second == 45
        
        # Invalid formats
        invalid_formats = [
            "2023/01/01",  # Wrong format
            "January 1, 2023",  # Text format
            "01-01-2023",  # MM-DD-YYYY format
            "12:30:45",  # Time only
        ]
        
        for inv_format in invalid_formats:
            with pytest.raises(ValidationError):
                TimeRangeParams(start_time=inv_format)
    
    def test_validation_with_multiple_conditions(self, valid_bracket_order_data):
        """Test validation rules that depend on multiple conditions."""
        # Test bracket order validation with market entry
        market_entry = valid_bracket_order_data.copy()
        market_entry["entry_type"] = "market"
        del market_entry["entry_price"]
        
        # Should validate even without price validations for stops & takes
        bracket = BracketOrderCreate(**market_entry)
        assert bracket.entry_type == OrderType.MARKET
        assert bracket.entry_price is None
        
        # But with limit entry, validations should work
        limit_entry = valid_bracket_order_data.copy()
        limit_entry["entry_type"] = "limit"
        limit_entry["entry_price"] = "150.00"
        limit_entry["side"] = "buy"
        limit_entry["stop_loss_price"] = "155.00"  # Invalid: SL above entry for buy
        
        with pytest.raises(ValidationError) as exc_info:
            BracketOrderCreate(**limit_entry)
        assert "Stop loss price must be below entry price for buy orders" in str(exc_info.value)
        
        # Test OCO order with different type combinations
        valid_combinations = [
            {"type_1": "limit", "type_2": "stop"},
            {"type_1": "limit", "type_2": "stop_limit"},
            {"type_1": "stop", "type_2": "limit"},
            {"type_1": "stop_limit", "type_2": "limit"}
        ]
        
        for combo in valid_combinations:
            data = {
                "account_id": 123,
                "symbol": "AAPL",
                "side": "buy",
                "quantity": "100",
                "price_1": "145.00",
                "price_2": "155.00",
                "time_in_force": "day"
            }
            data.update(combo)
            
            # Should validate
            oco = OCOOrderCreate(**data)
            assert oco.type_1 == combo["type_1"]
            assert oco.type_2 == combo["type_2"]
    
    def test_json_serialization_and_deserialization(self, mock_order):
        """Test full JSON serialization and deserialization cycle."""
        # Convert model to schema
        order_schema = order_model_to_schema(mock_order)
        
        # Convert schema to JSON
        order_json = order_schema.model_dump_json()
        
        # Verify JSON is valid
        json_dict = json.loads(order_json)
        assert json_dict["id"] == mock_order.id
        assert json_dict["symbol"] == mock_order.symbol
        
        # Convert JSON back to schema
        order_from_json = OrderResponse.model_validate_json(order_json)
        
        # Verify reconstituted schema
        assert order_from_json.id == mock_order.id
        assert order_from_json.symbol == mock_order.symbol
        assert order_from_json.quantity == mock_order.quantity
        assert order_from_json.price == mock_order.price
        
        # Nested schemas
        mock_order.executions = [mock_execution := Execution()]
        mock_execution.id = 456
        mock_execution.order_id = mock_order.order_id
        mock_execution.quantity = Decimal("50")
        mock_execution.price = Decimal("150.25")
        mock_execution.executed_at = datetime.utcnow()
        mock_execution.recorded_at = datetime.utcnow()
        
        # Convert with nested objects
        order_schema = order_model_to_schema(mock_order, include_executions=True)
        order_json = order_schema.model_dump_json()
        
        # Verify JSON has nested objects
        json_dict = json.loads(order_json)
        assert "executions" in json_dict
        assert len(json_dict["executions"]) == 1
        assert json_dict["executions"][0]["id"] == 456
        
        # Convert back to schema
        order_from_json = OrderResponse.model_validate_json(order_json)
        assert len(order_from_json.executions) == 1
        assert order_from_json.executions[0].id == 456
    
    def test_all_enum_values(self):
        """Test all possible enumeration values."""
        # OrderStatus
        for status in OrderStatus:
            assert isinstance(status.value, str)
            assert OrderStatusUpdate(status=status.value).status == status.value
        
        # OrderType
        for order_type in OrderType:
            assert isinstance(order_type.value, str)
            # Skip validation for types requiring additional parameters
            if order_type in [OrderType.LIMIT, OrderType.STOP, OrderType.STOP_LIMIT, OrderType.TRAILING_STOP]:
                continue
            
            data = {
                "symbol": "AAPL",
                "side": "buy",
                "order_type": order_type.value,
                "quantity": "100"
            }
            
            if order_type == OrderType.MARKET:
                # Market orders should validate
                order = OrderBase(**data)
                assert order.order_type == order_type.value
        
        # OrderSide
        for side in OrderSide:
            assert isinstance(side.value, str)
            data = {
                "symbol": "AAPL",
                "side": side.value,
                "order_type": "market",
                "quantity": "100"
            }
            order = OrderBase(**data)
            assert order.side == side.value
        
        # TimeInForce
        for tif in TimeInForce:
            assert isinstance(tif.value, str)
            data = {
                "symbol": "AAPL",
                "side": "buy",
                "order_type": "market",
                "quantity": "100",
                "time_in_force": tif.value
            }
            
            # GTD requires expire_at
            if tif == TimeInForce.GTD:
                data["expire_at"] = datetime.utcnow() + timedelta(days=1)
            
            order = OrderBase(**data)
            assert order.time_in_force == tif.value
        
        # PositionDirection
        for direction in PositionDirection:
            assert isinstance(direction.value, str)
            data = {
                "symbol": "AAPL",
                "direction": direction.value,
                "quantity": "100",
                "average_entry_price": "150.00",
                "current_price": "155.00"
            }
            position = PositionBase(**data)
            assert position.direction == direction.value
    
    def test_comprehensive_validation_chain(self):
        """Test comprehensive validation chain with multiple dependent validations."""
        # Create a complex order with trailing stop
        complex_order_data = {
            "account_id": 123,
            "symbol": "AAPL",
            "side": "buy",
            "order_type": "trailing_stop",
            "quantity": "100",
            "trailing_percent": "2.5",
            "time_in_force": "gtd",
            "expire_at": (datetime.utcnow() + timedelta(days=7)).isoformat(),
            "strategy_id": "complex_strategy",
            "tags": ["tech", "momentum", "trailing"]
        }
        
        # Should validate
        order = OrderCreate(**complex_order_data)
        assert order.order_type == OrderType.TRAILING_STOP
        assert order.trailing_percent == Decimal("2.5")
        assert order.time_in_force == TimeInForce.GTD
        assert isinstance(order.expire_at, datetime)
        assert order.tags == ["tech", "momentum", "trailing"]
        
        # Break validation chain one by one
        invalid_data = complex_order_data.copy()
        invalid_data["trailing_percent"] = None
        with pytest.raises(ValidationError) as exc_info:
            OrderCreate(**invalid_data)
        assert "trailing_amount or trailing_percent is required" in str(exc_info.value)
        
        invalid_data = complex_order_data.copy()
        invalid_data["expire_at"] = None
        with pytest.raises(ValidationError) as exc_info:
            OrderCreate(**invalid_data)
        assert "Expiration time is required for GTD orders" in str(exc_info.value)
    
    def test_schema_to_model_compatibility(self, valid_order_data):
        """Test that schema structures can be used to update models."""
        # Create a schema
        create_data = valid_order_data.copy()
        create_data["account_id"] = 123
        order_schema = OrderCreate(**create_data)
        
        # Convert to dict to update a model
        model_attrs = order_schema.model_dump(exclude_unset=True)
        
        # Verify all fields are present and correctly typed
        assert isinstance(model_attrs, dict)
        assert model_attrs["symbol"] == "AAPL"
        assert model_attrs["side"] == "buy"
        assert isinstance(model_attrs["quantity"], Decimal)
        assert model_attrs["quantity"] == Decimal("100.0")
        
        # Create a model mock and update it
        model = Order()
        for key, value in model_attrs.items():
            setattr(model, key, value)
        
        # Verify model was updated correctly
        assert model.symbol == "AAPL"
        assert model.side == "buy"
        assert model.quantity == Decimal("100.0")
        assert model.price == Decimal("150.50")
    
    def test_large_order_list(self, mock_order):
        """Test handling of large response lists."""
        # Create a large list of orders
        order_response = order_model_to_schema(mock_order)
        orders = [order_response.model_copy() for _ in range(100)]
        
        # Should handle large lists efficiently
        list_response = OrderListResponse(
            items=orders,
            total=1000,
            page=1,
            limit=100,
            pages=10
        )
        
        assert len(list_response.items) == 100
        assert list_response.total == 1000
        
        # JSON serialization should work efficiently
        json_data = list_response.model_dump_json()
        assert len(json_data) > 1000  # Just a basic check that it serialized something substantial
        
        # Deserialization should also work
        parsed_list = OrderListResponse.model_validate_json(json_data)
        assert len(parsed_list.items) == 100
        assert parsed_list.total == 1000
        
        # Check first and last items to ensure they're intact
        assert parsed_list.items[0].id == mock_order.id
        assert parsed_list.items[-1].id == mock_order.id