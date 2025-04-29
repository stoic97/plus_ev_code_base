"""
Unit tests for the order models in app/models/orders.py.

This module tests the functionality of order models, including:
- Order creation and validation
- Order status transitions
- Order execution tracking
- Multi-leg orders
- Conditional orders
- Order batches
"""

import pytest
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

from sqlalchemy.orm import Session

# Import the models to test
from app.models.orders import (
    Order, OrderLeg, Execution, OrderEvent, OrderAmendment, ConditionalOrder,
    OrderBatch, OrderBatchItem, 
    OrderType, OrderSide, OrderStatus, TimeInForce, ExecutionAlgorithm,
    OptionType, OrderTriggerType, OrderSource, OrderPriorityLevel,
    create_order, create_option_spread_order, create_conditional_order, create_order_batch
)


@pytest.fixture
def mock_session():
    """Create a mock session for testing."""
    session = MagicMock(spec=Session)
    
    # Mock add method to simulate SQLAlchemy's session.add
    session.add = MagicMock()
    
    # Mock flush method to simulate SQLAlchemy's session.flush
    session.flush = MagicMock()
    
    return session


@pytest.fixture
def sample_order():
    """Create a sample order for testing."""
    return Order(
        id=uuid.uuid4(),
        client_order_id="TEST123456",
        account_id=1,
        order_type=OrderType.LIMIT,
        side=OrderSide.BUY,
        symbol="AAPL",
        instrument_type="stock",
        quantity=Decimal("100"),
        price=Decimal("150.50"),
        status=OrderStatus.CREATED,
        time_in_force=TimeInForce.DAY,
        is_active=True,
        remaining_quantity=Decimal("100")
    )


class TestOrderModel:
    """Test the Order model functionality."""
    
    def test_order_creation(self, sample_order):
        """Test basic order creation with valid parameters."""
        # Assert that the order was created with correct attributes
        assert sample_order.client_order_id == "TEST123456"
        assert sample_order.order_type == OrderType.LIMIT
        assert sample_order.side == OrderSide.BUY
        assert sample_order.symbol == "AAPL"
        assert sample_order.quantity == Decimal("100")
        assert sample_order.price == Decimal("150.50")
        assert sample_order.status == OrderStatus.CREATED
        assert sample_order.is_active is True
        assert sample_order.remaining_quantity == Decimal("100")
        assert sample_order.filled_quantity == Decimal("0")
    
    def test_order_hybrid_properties(self, sample_order):
        """Test the hybrid properties of the Order model."""
        # Test is_filled property
        assert sample_order.is_filled is False
        sample_order.status = OrderStatus.FILLED
        assert sample_order.is_filled is True
        
        # Test is_cancelled property
        assert sample_order.is_cancelled is False
        sample_order.status = OrderStatus.CANCELLED
        assert sample_order.is_cancelled is True
        
        # Test can_be_amended property
        sample_order.status = OrderStatus.CREATED
        assert sample_order.can_be_amended is True
        sample_order.status = OrderStatus.FILLED
        assert sample_order.can_be_amended is False
        
        # Test can_be_cancelled property
        sample_order.status = OrderStatus.PENDING
        sample_order.is_active = True
        assert sample_order.can_be_cancelled is True
        sample_order.status = OrderStatus.FILLED
        assert sample_order.can_be_cancelled is False
        
        # Test execution_progress property
        sample_order.status = OrderStatus.PARTIALLY_FILLED
        sample_order.quantity = Decimal("100")
        sample_order.filled_quantity = Decimal("25")
        assert sample_order.execution_progress == 25.0
        
        # Test age_seconds property
        sample_order.created_at = datetime.utcnow() - timedelta(seconds=60)
        # Give some flexibility in the assertion due to timing
        assert 55 <= sample_order.age_seconds <= 65
    
    def test_update_status(self, sample_order):
        """Test the update_status method."""
        # Test updating to PENDING status
        sample_order.update_status(OrderStatus.PENDING)
        assert sample_order.status == OrderStatus.PENDING
        assert sample_order.submitted_at is not None
        assert sample_order.is_active is True
        
        # Test updating to FILLED status
        sample_order.update_status(OrderStatus.FILLED)
        assert sample_order.status == OrderStatus.FILLED
        assert sample_order.executed_at is not None
        assert sample_order.is_active is False
        
        # Test updating to CANCELLED status
        sample_order.status = OrderStatus.ACCEPTED  # Reset
        sample_order.is_active = True
        sample_order.update_status(OrderStatus.CANCELLED)
        assert sample_order.status == OrderStatus.CANCELLED
        assert sample_order.cancelled_at is not None
        assert sample_order.is_active is False
        
        # Test updating to REJECTED status with error message
        sample_order.status = OrderStatus.CREATED  # Reset
        sample_order.is_active = True
        sample_order.update_status(OrderStatus.REJECTED, error_message="Invalid price")
        assert sample_order.status == OrderStatus.REJECTED
        assert sample_order.error_message == "Invalid price"
        assert sample_order.is_active is False
    
    def test_update_fill(self, sample_order):
        """Test the update_fill method for partial and complete fills."""
        # Test partial fill
        sample_order.update_fill(Decimal("40"), Decimal("151.00"))
        assert sample_order.filled_quantity == Decimal("40")
        assert sample_order.remaining_quantity == Decimal("60")
        assert sample_order.avg_execution_price == Decimal("151.00")
        assert sample_order.status == OrderStatus.PARTIALLY_FILLED
        
        # Test another partial fill with different price
        sample_order.update_fill(Decimal("30"), Decimal("152.00"))
        assert sample_order.filled_quantity == Decimal("70")
        assert sample_order.remaining_quantity == Decimal("30")
        # Verify weighted average price calculation
        expected_avg_price = (Decimal("40") * Decimal("151.00") + Decimal("30") * Decimal("152.00")) / Decimal("70")
        assert sample_order.avg_execution_price == expected_avg_price
        
        # Test complete fill
        sample_order.update_fill(Decimal("30"), Decimal("153.00"))
        assert sample_order.filled_quantity == Decimal("100")
        assert sample_order.remaining_quantity == Decimal("0")
        # Verify final weighted average price
        expected_final_avg = (Decimal("40") * Decimal("151.00") + Decimal("30") * Decimal("152.00") + Decimal("30") * Decimal("153.00")) / Decimal("100")
        assert sample_order.avg_execution_price == expected_final_avg
        assert sample_order.status == OrderStatus.FILLED
    
    def test_cancel(self, sample_order):
        """Test the cancel method."""
        # Test successful cancellation
        assert sample_order.cancel("Market closed") is True
        assert sample_order.status == OrderStatus.CANCELLED
        assert sample_order.is_active is False
        assert "Cancelled: Market closed" in sample_order.notes
        
        # Test cancellation of already filled order
        sample_order.status = OrderStatus.FILLED
        assert sample_order.cancel("Test cancellation") is False
        assert sample_order.status == OrderStatus.FILLED  # Should remain filled
    
    def test_clone(self, sample_order):
        """Test the clone method creates a proper copy."""
        # Set some additional attributes
        sample_order.metadata = {"test": "value"}
        sample_order.strategy_id = uuid.uuid4()
        sample_order.filled_quantity = Decimal("50")
        sample_order.remaining_quantity = Decimal("50")
        sample_order.avg_execution_price = Decimal("151.00")
        sample_order.status = OrderStatus.PARTIALLY_FILLED
        
        # Clone the order
        cloned_order = sample_order.clone()
        
        # Verify cloned attributes
        assert cloned_order.id != sample_order.id  # New ID
        assert cloned_order.account_id == sample_order.account_id
        assert cloned_order.symbol == sample_order.symbol
        assert cloned_order.order_type == sample_order.order_type
        assert cloned_order.side == sample_order.side
        assert cloned_order.quantity == sample_order.quantity
        assert cloned_order.price == sample_order.price
        assert cloned_order.strategy_id == sample_order.strategy_id
        assert cloned_order.metadata == sample_order.metadata
        
        # Verify execution state is reset
        assert cloned_order.status == OrderStatus.CREATED
        assert cloned_order.filled_quantity == Decimal("0")
        assert cloned_order.remaining_quantity == sample_order.quantity
        assert cloned_order.avg_execution_price is None
        assert cloned_order.is_active is True
        assert "Cloned from order" in cloned_order.notes


class TestOrderLegModel:
    """Test the OrderLeg model functionality."""
    
    def test_order_leg_creation(self):
        """Test basic order leg creation with valid parameters."""
        parent_id = uuid.uuid4()
        leg = OrderLeg(
            parent_order_id=parent_id,
            leg_number=1,
            side=OrderSide.BUY,
            symbol="AAPL",
            quantity=Decimal("10"),
            price=Decimal("150.00"),
            ratio=1,
            is_option=True,
            option_type=OptionType.CALL,
            strike_price=Decimal("155.00"),
            expiration_date=datetime.utcnow() + timedelta(days=30),
            must_execute_first=True
        )
        
        # Assert that the leg was created with correct attributes
        assert leg.parent_order_id == parent_id
        assert leg.leg_number == 1
        assert leg.side == OrderSide.BUY
        assert leg.symbol == "AAPL"
        assert leg.quantity == Decimal("10")
        assert leg.price == Decimal("150.00")
        assert leg.ratio == 1
        assert leg.is_option is True
        assert leg.option_type == OptionType.CALL
        assert leg.strike_price == Decimal("155.00")
        assert leg.must_execute_first is True


class TestExecutionModel:
    """Test the Execution model functionality."""
    
    def test_execution_creation(self):
        """Test basic execution creation with valid parameters."""
        order_id = uuid.uuid4()
        execution = Execution(
            order_id=order_id,
            execution_id="EXE12345",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("50"),
            price=Decimal("151.25"),
            timestamp=datetime.utcnow(),
            venue="NASDAQ",
            fees=Decimal("1.50"),
            fee_currency="USD",
            commission=Decimal("2.50")
        )
        
        # Assert that the execution was created with correct attributes
        assert execution.order_id == order_id
        assert execution.execution_id == "EXE12345"
        assert execution.symbol == "AAPL"
        assert execution.side == OrderSide.BUY
        assert execution.quantity == Decimal("50")
        assert execution.price == Decimal("151.25")
        assert execution.venue == "NASDAQ"
        assert execution.fees == Decimal("1.50")
        assert execution.fee_currency == "USD"
        assert execution.commission == Decimal("2.50")
    
    def test_execution_hybrid_properties(self):
        """Test the hybrid properties of the Execution model."""
        # Create execution for buy side
        buy_execution = Execution(
            order_id=uuid.uuid4(),
            execution_id="EXE12345",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("50"),
            price=Decimal("150.00"),
            timestamp=datetime.utcnow(),
            venue="NASDAQ",
            fees=Decimal("1.50"),
            commission=Decimal("2.50")
        )
        
        # Test value property
        assert buy_execution.value == Decimal("7500.00")  # 50 * 150.00
        
        # Test total_cost property for buy side
        assert buy_execution.total_cost == Decimal("7504.00")  # 7500 + 1.50 + 2.50
        
        # Create execution for sell side
        sell_execution = Execution(
            order_id=uuid.uuid4(),
            execution_id="EXE67890",
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=Decimal("50"),
            price=Decimal("155.00"),
            timestamp=datetime.utcnow(),
            venue="NASDAQ",
            fees=Decimal("1.50"),
            commission=Decimal("2.50")
        )
        
        # Test total_cost property for sell side
        assert sell_execution.value == Decimal("7750.00")  # 50 * 155.00
        assert sell_execution.total_cost == Decimal("7746.00")  # 7750 - 1.50 - 2.50


class TestConditionalOrderModel:
    """Test the ConditionalOrder model functionality."""
    
    def test_conditional_order_creation(self):
        """Test conditional order creation with valid parameters."""
        order_id = uuid.uuid4()
        
        # Create a price-based conditional order
        price_condition = ConditionalOrder(
            order_id=order_id,
            trigger_type=OrderTriggerType.PRICE,
            symbol="AAPL",
            price_target=Decimal("160.00"),
            price_comparator=">="
        )
        
        assert price_condition.order_id == order_id
        assert price_condition.trigger_type == OrderTriggerType.PRICE
        assert price_condition.symbol == "AAPL"
        assert price_condition.price_target == Decimal("160.00")
        assert price_condition.price_comparator == ">="
        assert price_condition.is_triggered is False
        
        # Create a time-based conditional order
        trigger_time = datetime.utcnow() + timedelta(hours=1)
        time_condition = ConditionalOrder(
            order_id=uuid.uuid4(),
            trigger_type=OrderTriggerType.TIME,
            trigger_time=trigger_time
        )
        
        assert time_condition.trigger_type == OrderTriggerType.TIME
        assert time_condition.trigger_time == trigger_time
    
    def test_is_expired_property(self):
        """Test the is_expired property."""
        # Create condition that expires in the future
        future_expiry = datetime.utcnow() + timedelta(hours=1)
        condition = ConditionalOrder(
            order_id=uuid.uuid4(),
            trigger_type=OrderTriggerType.PRICE,
            symbol="AAPL",
            price_target=Decimal("160.00"),
            price_comparator=">=",
            expires_at=future_expiry
        )
        
        assert condition.is_expired is False
        
        # Create condition that has already expired
        past_expiry = datetime.utcnow() - timedelta(hours=1)
        expired_condition = ConditionalOrder(
            order_id=uuid.uuid4(),
            trigger_type=OrderTriggerType.PRICE,
            symbol="AAPL",
            price_target=Decimal("160.00"),
            price_comparator=">=",
            expires_at=past_expiry
        )
        
        assert expired_condition.is_expired is True
        
        # Create condition with no expiry
        no_expiry_condition = ConditionalOrder(
            order_id=uuid.uuid4(),
            trigger_type=OrderTriggerType.PRICE,
            symbol="AAPL",
            price_target=Decimal("160.00"),
            price_comparator=">="
        )
        
        assert no_expiry_condition.is_expired is False
    
    def test_mark_triggered(self):
        """Test the mark_triggered method."""
        condition = ConditionalOrder(
            order_id=uuid.uuid4(),
            trigger_type=OrderTriggerType.PRICE,
            symbol="AAPL",
            price_target=Decimal("160.00"),
            price_comparator=">="
        )
        
        # Initial state
        assert condition.is_triggered is False
        assert condition.triggered_at is None
        assert condition.trigger_attempts == 0
        
        # Mark as triggered
        condition.mark_triggered()
        
        # Verify state change
        assert condition.is_triggered is True
        assert condition.triggered_at is not None
        assert condition.last_check_time is not None
        assert condition.trigger_attempts == 1
    
    def test_update_check_time(self):
        """Test the update_check_time method."""
        condition = ConditionalOrder(
            order_id=uuid.uuid4(),
            trigger_type=OrderTriggerType.PRICE,
            symbol="AAPL",
            price_target=Decimal("160.00"),
            price_comparator=">="
        )
        
        # Initial state
        assert condition.last_check_time is None
        assert condition.trigger_attempts == 0
        
        # Update check time
        condition.update_check_time()
        
        # Verify state change
        assert condition.last_check_time is not None
        assert condition.trigger_attempts == 1


class TestOrderBatchModel:
    """Test the OrderBatch model functionality."""
    
    @pytest.fixture
    def sample_batch(self):
        """Create a sample order batch for testing."""
        return OrderBatch(
            id=uuid.uuid4(),
            batch_name="Test Batch",
            batch_type="basket",
            account_id=1,
            is_active=True,
            status="created",
            requires_all_filled=True,
            cancel_on_partial_reject=False
        )
    
    @pytest.fixture
    def sample_batch_with_orders(self, sample_batch):
        """Create a sample batch with mock orders."""
        # Create orders with different statuses
        order1 = Order(id=uuid.uuid4(), status=OrderStatus.FILLED, is_active=False)
        order2 = Order(id=uuid.uuid4(), status=OrderStatus.PARTIALLY_FILLED, is_active=True)
        order3 = Order(id=uuid.uuid4(), status=OrderStatus.PENDING, is_active=True)
        
        # Add orders to batch
        sample_batch.orders = [order1, order2, order3]
        
        return sample_batch
    
    def test_batch_creation(self, sample_batch):
        """Test basic batch creation with valid parameters."""
        assert sample_batch.batch_name == "Test Batch"
        assert sample_batch.batch_type == "basket"
        assert sample_batch.account_id == 1
        assert sample_batch.is_active is True
        assert sample_batch.status == "created"
        assert sample_batch.requires_all_filled is True
        assert sample_batch.cancel_on_partial_reject is False
    
    def test_batch_hybrid_properties(self, sample_batch_with_orders):
        """Test the hybrid properties of the OrderBatch model."""
        batch = sample_batch_with_orders
        
        # Test orders_count property
        assert batch.orders_count == 3
        
        # Test filled_orders_count property
        assert batch.filled_orders_count == 1
        
        # Test cancelled_orders_count property
        # Add a cancelled order
        cancelled_order = Order(id=uuid.uuid4(), status=OrderStatus.CANCELLED, is_active=False)
        batch.orders.append(cancelled_order)
        assert batch.cancelled_orders_count == 1
        
        # Test fill_percentage property
        assert batch.fill_percentage == 25.0  # 1 out of 4 orders filled
    
    def test_cancel_all_orders(self, sample_batch_with_orders):
        """Test the cancel_all_orders method."""
        batch = sample_batch_with_orders
        
        # Initial state - 3 orders, 2 of which can be cancelled
        assert len(batch.orders) == 3
        assert sum(1 for order in batch.orders if order.can_be_cancelled) == 2
        
        # Cancel all orders
        cancelled_count = batch.cancel_all_orders("Test cancellation")
        
        # Verify 2 orders were cancelled
        assert cancelled_count == 2
        assert batch.status == "cancelled"
        assert batch.is_active is False
        
        # Verify all orders now show as cancelled or completed
        for order in batch.orders:
            if order.status != OrderStatus.FILLED:
                assert order.status == OrderStatus.CANCELLED
                assert order.is_active is False
    
    def test_update_status(self, sample_batch_with_orders):
        """Test the update_status method."""
        batch = sample_batch_with_orders
        
        # Test with mixed order statuses
        batch.update_status()
        assert batch.status == "partially_filled"
        assert batch.is_active is True
        
        # Test with all orders filled
        for order in batch.orders:
            order.status = OrderStatus.FILLED
            order.is_active = False
        
        batch.update_status()
        assert batch.status == "completed"
        assert batch.is_active is False
        assert batch.completed_at is not None
        
        # Test with all orders cancelled
        batch.status = "created"  # Reset
        batch.is_active = True
        for order in batch.orders:
            order.status = OrderStatus.CANCELLED
            order.is_active = False
        
        batch.update_status()
        assert batch.status == "cancelled"
        assert batch.is_active is False


class TestCreateOrderFunction:
    """Test the create_order utility function."""
    
    def test_create_basic_order(self, mock_session):
        """Test creating a basic order."""
        # Call the function
        order = create_order(
            session=mock_session,
            account_id=1,
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            order_type=OrderType.LIMIT,
            price=Decimal("150.00"),
            instrument_type="stock",
            time_in_force=TimeInForce.DAY
        )
        
        # Verify the order was created with correct attributes
        assert order.account_id == 1
        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.quantity == Decimal("100")
        assert order.order_type == OrderType.LIMIT
        assert order.price == Decimal("150.00")
        assert order.instrument_type == "stock"
        assert order.time_in_force == TimeInForce.DAY
        assert order.remaining_quantity == Decimal("100")
        assert order.status == OrderStatus.CREATED
        
        # Verify client_order_id was generated if not provided
        assert order.client_order_id is not None
        assert len(order.client_order_id) >= 8
        
        # Verify session.add was called twice (once for order, once for event)
        assert mock_session.add.call_count == 2
    
    def test_create_order_with_client_id(self, mock_session):
        """Test creating an order with provided client_order_id."""
        order = create_order(
            session=mock_session,
            account_id=1,
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            order_type=OrderType.MARKET,
            instrument_type="stock",
            client_order_id="CLIENT123456"
        )
        
        assert order.client_order_id == "CLIENT123456"
    
    def test_create_order_validation(self, mock_session):
        """Test validation in create_order function."""
        # Test validation for limit orders without price
        with pytest.raises(ValueError, match="Price is required for limit orders"):
            create_order(
                session=mock_session,
                account_id=1,
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=Decimal("100"),
                order_type=OrderType.LIMIT,
                instrument_type="stock"
                # Missing price
            )
        
        # Test validation for option orders without required fields
        with pytest.raises(ValueError, match="Option orders require option_type, strike_price, and expiration_date"):
            create_order(
                session=mock_session,
                account_id=1,
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=Decimal("100"),
                order_type=OrderType.LIMIT,
                price=Decimal("150.00"),
                instrument_type="option",
                is_option_order=True
                # Missing option_type, strike_price, expiration_date
            )


class TestCreateOptionSpreadOrderFunction:
    """Test the create_option_spread_order utility function."""
    
    def test_create_option_spread(self, mock_session):
        """Test creating an option spread order."""
        expiration_date = datetime.utcnow() + timedelta(days=30)
        
        # Define option legs
        legs = [
            {
                "option_type": OptionType.CALL,
                "strike_price": Decimal("155.00"),
                "side": OrderSide.BUY,
                "quantity": Decimal("1"),
                "ratio": 1
            },
            {
                "option_type": OptionType.CALL,
                "strike_price": Decimal("160.00"),
                "side": OrderSide.SELL,
                "quantity": Decimal("1"),
                "ratio": 1
            }
        ]
        
        # Call the function
        order = create_option_spread_order(
            session=mock_session,
            account_id=1,
            spread_type="vertical",
            underlying_symbol="AAPL",
            expiration_date=expiration_date,
            legs=legs
        )
        
        # Verify parent order was created correctly
        assert order.account_id == 1
        assert order.symbol == "AAPL"
        assert order.order_type == OrderType.LIMIT
        assert order.is_option_order is True
        assert order.is_multi_leg is True
        assert order.instrument_type == "option_spread"
        assert order.quantity == Decimal("2")  # Sum of leg quantities * ratios
        assert "vertical spread" in order.notes.lower()
        assert order.metadata["spread_type"] == "vertical"
        assert order.metadata["leg_count"] == 2
        
        # Verify session.add was called for parent order and each leg
        # Plus one for the order event = 4 calls total
        assert mock_session.add.call_count == 4
    
    def test_create_option_spread_validation(self, mock_session):
        """Test validation in create_option_spread_order function."""
        expiration_date = datetime.utcnow() + timedelta(days=30)
        
        # Test validation for insufficient legs
        with pytest.raises(ValueError, match="Option spread order must have at least 2 legs"):
            create_option_spread_order(
                session=mock_session,
                account_id=1,
                spread_type="vertical",
                underlying_symbol="AAPL",
                expiration_date=expiration_date,
                legs=[]  # Empty legs list
            )


class TestCreateConditionalOrderFunction:
    """Test the create_conditional_order utility function."""
    
    def test_create_price_conditional_order(self, mock_session, sample_order):
        """Test creating a price-based conditional order."""
        # Define trigger parameters
        trigger_params = {
            "symbol": "AAPL",
            "price_target": Decimal("155.00"),
            "price_comparator": ">="
        }
        
        # Call the function
        conditional = create_conditional_order(
            session=mock_session,
            order=sample_order,
            trigger_type=OrderTriggerType.PRICE,
            trigger_params=trigger_params
        )
        
        # Verify conditional order was created correctly
        assert conditional.order_id == sample_order.id
        assert conditional.trigger_type == OrderTriggerType.PRICE
        assert conditional.symbol == "AAPL"
        assert conditional.price_target == Decimal("155.00")
        assert conditional.price_comparator == ">="
        assert conditional.is_triggered is False
        
        # Verify order was updated
        assert sample_order.is_conditional is True
        assert sample_order.trigger_type == OrderTriggerType.PRICE
        assert sample_order.trigger_details == trigger_params
        
        # Verify session.add was called twice (once for conditional, once for event)
        assert mock_session.add.call_count == 2
    
    def test_create_time_conditional_order(self, mock_session, sample_order):
        """Test creating a time-based conditional order."""
        trigger_time = datetime.utcnow() + timedelta(hours=1)
        
        # Define trigger parameters
        trigger_params = {
            "trigger_time": trigger_time
        }
        
        # Call the function
        conditional = create_conditional_order(
            session=mock_session,
            order=sample_order,
            trigger_type=OrderTriggerType.TIME,
            trigger_params=trigger_params
        )
        
        # Verify conditional order was created correctly
        assert conditional.order_id == sample_order.id
        assert conditional.trigger_type == OrderTriggerType.TIME
        assert conditional.trigger_time == trigger_time
    
    def test_create_conditional_order_validation(self, mock_session, sample_order):
        """Test validation in create_conditional_order function."""
        # Test validation for missing parameters
        with pytest.raises(ValueError, match="Missing required parameters for price trigger"):
            create_conditional_order(
                session=mock_session,
                order=sample_order,
                trigger_type=OrderTriggerType.PRICE,
                trigger_params={
                    "symbol": "AAPL"
                    # Missing price_target and price_comparator
                }
            )


class TestCreateOrderBatchFunction:
    """Test the create_order_batch utility function."""
    
    def test_create_order_batch(self, mock_session):
        """Test creating an order batch."""
        # Define orders for the batch
        orders = [
            {
                "symbol": "AAPL",
                "side": OrderSide.BUY,
                "quantity": Decimal("100"),
                "order_type": OrderType.LIMIT,
                "price": Decimal("150.00"),
                "instrument_type": "stock",
                "sequence_number": 1
            },
            {
                "symbol": "GOOGL",
                "side": OrderSide.BUY,
                "quantity": Decimal("50"),
                "order_type": OrderType.LIMIT,
                "price": Decimal("2500.00"),
                "instrument_type": "stock",
                "sequence_number": 2,
                "wait_for_order_idx": 0  # Wait for the first order
            }
        ]
        
        # Mock the create_order function to return dummy orders
        with patch("app.models.orders.create_order") as mock_create_order:
            # Configure mock to return a new dummy order for each call
            mock_create_order.side_effect = lambda **kwargs: Order(
                id=uuid.uuid4(),
                client_order_id=f"MOCK_{uuid.uuid4().hex[:8]}",
                account_id=kwargs.get("account_id"),
                symbol=kwargs.get("symbol"),
                side=kwargs.get("side"),
                quantity=kwargs.get("quantity"),
                order_type=kwargs.get("order_type"),
                price=kwargs.get("price"),
                instrument_type=kwargs.get("instrument_type"),
                remaining_quantity=kwargs.get("quantity")
            )
            
            # Call the function
            batch = create_order_batch(
                session=mock_session,
                account_id=1,
                batch_name="Test Basket",
                orders=orders,
                batch_type="basket",
                requires_all_filled=True,
                cancel_on_partial_reject=False,
                notes="Test batch order"
            )
            
            # Verify batch was created correctly
            assert batch.account_id == 1
            assert batch.batch_name == "Test Basket"
            assert batch.batch_type == "basket"
            assert batch.requires_all_filled is True
            assert batch.cancel_on_partial_reject is False
            assert batch.notes == "Test batch order"
            
            # Verify create_order was called for each order in the batch
            assert mock_create_order.call_count == 2
            
            # Verify session.add was called for batch and batch items
            # The batch itself + 2 batch items
            assert mock_session.add.call_count >= 3
            
            # Verify session.flush was called to generate batch ID
            assert mock_session.flush.call_count == 1
    
    def test_create_order_batch_with_dependencies(self, mock_session):
        """Test creating an order batch with dependencies."""
        # Define orders with dependencies
        orders = [
            {
                "symbol": "AAPL",
                "side": OrderSide.BUY,
                "quantity": Decimal("100"),
                "order_type": OrderType.LIMIT,
                "price": Decimal("150.00"),
                "instrument_type": "stock",
                "can_execute_alone": True
            },
            {
                "symbol": "GOOGL",
                "side": OrderSide.BUY,
                "quantity": Decimal("50"),
                "order_type": OrderType.LIMIT,
                "price": Decimal("2500.00"),
                "instrument_type": "stock",
                "wait_for_order_idx": 0,  # Wait for the first order
                "can_execute_alone": False,
                "execution_conditions": {"min_fill_percentage": 80}
            }
        ]
        
        # Mock order creation to create dummy orders
        with patch("app.models.orders.create_order") as mock_create_order:
            mock_create_order.side_effect = lambda **kwargs: Order(
                id=uuid.uuid4(),
                client_order_id=f"MOCK_{uuid.uuid4().hex[:8]}",
                account_id=kwargs.get("account_id"),
                symbol=kwargs.get("symbol"),
                side=kwargs.get("side"),
                quantity=kwargs.get("quantity"),
                order_type=kwargs.get("order_type"),
                price=kwargs.get("price"),
                instrument_type=kwargs.get("instrument_type"),
                remaining_quantity=kwargs.get("quantity")
            )
            
            # Call the function
            batch = create_order_batch(
                session=mock_session,
                account_id=1,
                batch_name="Test Dependent Basket",
                orders=orders,
                batch_type="dependent",
                strategy_id=uuid.uuid4(),
                user_id=uuid.uuid4()
            )
            
            # Verify batch items have correct dependency settings
            assert len(batch.batch_items) == 2
            
            # First order can execute alone
            first_batch_item = batch.batch_items[0]
            assert first_batch_item.can_execute_alone is True
            assert first_batch_item.wait_for_order_id is None
            
            # Second order is dependent on first order
            second_batch_item = batch.batch_items[1]
            assert second_batch_item.can_execute_alone is False
            assert second_batch_item.wait_for_order_id is not None
            assert second_batch_item.execution_conditions == {"min_fill_percentage": 80}


class TestOrderEventModel:
    """Test the OrderEvent model functionality."""
    
    def test_order_event_creation(self):
        """Test basic order event creation with valid parameters."""
        order_id = uuid.uuid4()
        user_id = uuid.uuid4()
        
        event = OrderEvent(
            order_id=order_id,
            event_type="submitted",
            user_id=user_id,
            previous_status=OrderStatus.CREATED,
            new_status=OrderStatus.PENDING,
            details={"exchange": "NASDAQ", "client_ref": "REF123456"}
        )
        
        # Assert that the event was created with correct attributes
        assert event.order_id == order_id
        assert event.event_type == "submitted"
        assert event.user_id == user_id
        assert event.previous_status == OrderStatus.CREATED
        assert event.new_status == OrderStatus.PENDING
        assert event.details == {"exchange": "NASDAQ", "client_ref": "REF123456"}
        assert event.timestamp is not None


class TestOrderAmendmentModel:
    """Test the OrderAmendment model functionality."""
    
    def test_order_amendment_creation(self):
        """Test basic order amendment creation with valid parameters."""
        order_id = uuid.uuid4()
        user_id = uuid.uuid4()
        
        amendment = OrderAmendment(
            order_id=order_id,
            user_id=user_id,
            field_name="price",
            old_value="150.00",
            new_value="155.00",
            was_successful=True
        )
        
        # Assert that the amendment was created with correct attributes
        assert amendment.order_id == order_id
        assert amendment.user_id == user_id
        assert amendment.field_name == "price"
        assert amendment.old_value == "150.00"
        assert amendment.new_value == "155.00"
        assert amendment.was_successful is True
        assert amendment.error_message is None
        assert amendment.timestamp is not None
        
        # Test failed amendment
        failed_amendment = OrderAmendment(
            order_id=order_id,
            user_id=user_id,
            field_name="quantity",
            old_value="100",
            new_value="200",
            was_successful=False,
            error_message="Insufficient buying power"
        )
        
        assert failed_amendment.was_successful is False
        assert failed_amendment.error_message == "Insufficient buying power"


class TestOrderBatchItemModel:
    """Test the OrderBatchItem model functionality."""
    
    def test_order_batch_item_creation(self):
        """Test basic order batch item creation with valid parameters."""
        batch_id = uuid.uuid4()
        order_id = uuid.uuid4()
        dependent_order_id = uuid.uuid4()
        
        batch_item = OrderBatchItem(
            batch_id=batch_id,
            order_id=order_id,
            sequence_number=2,
            wait_for_order_id=dependent_order_id,
            can_execute_alone=False,
            execution_conditions={"min_fill_percentage": 90}
        )
        
        # Assert that the batch item was created with correct attributes
        assert batch_item.batch_id == batch_id
        assert batch_item.order_id == order_id
        assert batch_item.sequence_number == 2
        assert batch_item.wait_for_order_id == dependent_order_id
        assert batch_item.can_execute_alone is False
        assert batch_item.execution_conditions == {"min_fill_percentage": 90}


class TestValidations:
    """Test validations across various models."""
    
    def test_order_quantity_validation(self):
        """Test validation of order quantity."""
        # Test valid quantity
        order = Order(
            client_order_id="TEST12345",
            account_id=1,
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            symbol="AAPL",
            instrument_type="stock",
            quantity=Decimal("100"),
            remaining_quantity=Decimal("100")
        )
        assert order.quantity == Decimal("100")
        
        # Test invalid quantity (negative)
        with pytest.raises(ValueError, match="quantity must be positive"):
            Order(
                client_order_id="TEST12345",
                account_id=1,
                order_type=OrderType.MARKET,
                side=OrderSide.BUY,
                symbol="AAPL",
                instrument_type="stock",
                quantity=Decimal("-100"),
                remaining_quantity=Decimal("-100")
            )
        
        # Test invalid quantity (zero)
        with pytest.raises(ValueError, match="quantity must be positive"):
            Order(
                client_order_id="TEST12345",
                account_id=1,
                order_type=OrderType.MARKET,
                side=OrderSide.BUY,
                symbol="AAPL",
                instrument_type="stock",
                quantity=Decimal("0"),
                remaining_quantity=Decimal("0")
            )
    
    def test_conditional_order_comparator_validation(self):
        """Test validation of conditional order comparators."""
        order_id = uuid.uuid4()
        
        # Test valid comparator
        condition = ConditionalOrder(
            order_id=order_id,
            trigger_type=OrderTriggerType.PRICE,
            symbol="AAPL",
            price_target=Decimal("150.00"),
            price_comparator=">="
        )
        assert condition.price_comparator == ">="
        
        # Test invalid comparator
        with pytest.raises(ValueError, match="must be one of"):
            ConditionalOrder(
                order_id=order_id,
                trigger_type=OrderTriggerType.PRICE,
                symbol="AAPL",
                price_target=Decimal("150.00"),
                price_comparator="!="  # Invalid comparator
            )
    
    def test_execution_numeric_validation(self):
        """Test validation of execution numeric fields."""
        order_id = uuid.uuid4()
        
        # Test valid values
        execution = Execution(
            order_id=order_id,
            execution_id="EXE12345",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("50"),
            price=Decimal("150.00"),
            timestamp=datetime.utcnow(),
            venue="NASDAQ"
        )
        assert execution.quantity == Decimal("50")
        assert execution.price == Decimal("150.00")
        
        # Test invalid quantity (negative)
        with pytest.raises(ValueError, match="quantity must be positive"):
            Execution(
                order_id=order_id,
                execution_id="EXE12345",
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=Decimal("-50"),  # Negative quantity
                price=Decimal("150.00"),
                timestamp=datetime.utcnow(),
                venue="NASDAQ"
            )
        
        # Test invalid price (zero)
        with pytest.raises(ValueError, match="price must be positive"):
            Execution(
                order_id=order_id,
                execution_id="EXE12345",
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=Decimal("50"),
                price=Decimal("0"),  # Zero price
                timestamp=datetime.utcnow(),
                venue="NASDAQ"
            )


# This section tests integration between models

class TestModelIntegration:
    """Test integration between different models."""
    
    def test_order_execution_integration(self, mock_session):
        """Test integration between Order and Execution models."""
        # Create an order
        order = Order(
            id=uuid.uuid4(),
            client_order_id="TEST123456",
            account_id=1,
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            symbol="AAPL",
            instrument_type="stock",
            quantity=Decimal("100"),
            price=Decimal("150.00"),
            status=OrderStatus.ACCEPTED,
            time_in_force=TimeInForce.DAY,
            is_active=True,
            remaining_quantity=Decimal("100"),
            filled_quantity=Decimal("0")
        )
        
        # Mock the executions relationship
        order.executions = []
        
        # Create an execution
        execution = Execution(
            order_id=order.id,
            execution_id="EXE12345",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("50"),
            price=Decimal("149.50"),
            timestamp=datetime.utcnow(),
            venue="NASDAQ"
        )
        
        # Add execution to order
        order.executions.append(execution)
        
        # Update order based on execution
        order.update_fill(execution.quantity, execution.price)
        
        # Verify order state was updated correctly
        assert order.filled_quantity == Decimal("50")
        assert order.remaining_quantity == Decimal("50")
        assert order.avg_execution_price == Decimal("149.50")
        assert order.status == OrderStatus.PARTIALLY_FILLED
        
        # Create another execution
        execution2 = Execution(
            order_id=order.id,
            execution_id="EXE67890",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("50"),
            price=Decimal("150.50"),
            timestamp=datetime.utcnow(),
            venue="NASDAQ"
        )
        
        # Add second execution to order
        order.executions.append(execution2)
        
        # Update order based on second execution
        order.update_fill(execution2.quantity, execution2.price)
        
        # Verify order state was updated correctly
        assert order.filled_quantity == Decimal("100")
        assert order.remaining_quantity == Decimal("0")
        assert order.status == OrderStatus.FILLED
        
        # Verify average price calculation
        expected_avg_price = (Decimal("50") * Decimal("149.50") + Decimal("50") * Decimal("150.50")) / Decimal("100")
        assert order.avg_execution_price == expected_avg_price
    
    def test_order_conditional_integration(self, mock_session):
        """Test integration between Order and ConditionalOrder models."""
        # Create an order
        order = Order(
            id=uuid.uuid4(),
            client_order_id="TEST123456",
            account_id=1,
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            symbol="AAPL",
            instrument_type="stock",
            quantity=Decimal("100"),
            price=Decimal("150.00"),
            status=OrderStatus.CREATED,
            time_in_force=TimeInForce.DAY,
            is_active=True,
            remaining_quantity=Decimal("100")
        )
        
        # Create conditional order parameters
        trigger_params = {
            "symbol": "AAPL",
            "price_target": Decimal("148.00"),
            "price_comparator": "<="
        }
        
        # Call the create_conditional_order function
        with patch("app.models.orders.OrderEvent") as mock_order_event:
            # Configure mock to return a dummy order event
            mock_order_event.return_value = OrderEvent(
                order_id=order.id,
                event_type="made_conditional",
                new_status=order.status
            )
            
            conditional = create_conditional_order(
                session=mock_session,
                order=order,
                trigger_type=OrderTriggerType.PRICE,
                trigger_params=trigger_params
            )
        
        # Verify order was updated
        assert order.is_conditional is True
        assert order.trigger_type == OrderTriggerType.PRICE
        assert order.trigger_details == trigger_params
        
        # Verify conditional order was created correctly
        assert conditional.order_id == order.id
        assert conditional.trigger_type == OrderTriggerType.PRICE
        assert conditional.symbol == "AAPL"
        assert conditional.price_target == Decimal("148.00")
        assert conditional.price_comparator == "<="
        
        # Simulate market conditions meeting the trigger
        conditional.mark_triggered()
        
        # Verify conditional order state
        assert conditional.is_triggered is True
        assert conditional.triggered_at is not None


if __name__ == "__main__":
    pytest.main(["-v", "test_models_orders.py"])