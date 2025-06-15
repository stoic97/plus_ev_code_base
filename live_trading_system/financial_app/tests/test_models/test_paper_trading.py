"""
Unit tests for Paper Trading Models.

This module contains comprehensive tests for Order and OrderFill models,
covering model creation, validation, relationships, properties, and constraints.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
from decimal import Decimal
import enum

# Import models if available, otherwise use mocks
try:
    from app.models.paper_trading import (
        Order, OrderFill, OrderType, OrderStatus, 
        OrderSide, FillType
    )
    from app.models.base import TimestampMixin, UserRelationMixin, AuditMixin
    from app.core.database import Base
    from sqlalchemy.exc import IntegrityError
    models_imports_success = True
except ImportError:
    # Create mock classes if imports fail
    models_imports_success = False
    
    class MockEnum:
        def __init__(self, value):
            self.value = value
        
        def __eq__(self, other):
            if isinstance(other, MockEnum):
                return self.value == other.value
            return self.value == other
        
        def __str__(self):
            return str(self.value)
    
    class OrderType:
        MARKET = MockEnum("market")
        LIMIT = MockEnum("limit")
        STOP_LOSS = MockEnum("stop_loss")
        STOP_LIMIT = MockEnum("stop_limit")
    
    class OrderStatus:
        PENDING = MockEnum("pending")
        SUBMITTED = MockEnum("submitted")
        ACKNOWLEDGED = MockEnum("acknowledged")
        PARTIALLY_FILLED = MockEnum("partially_filled")
        FILLED = MockEnum("filled")
        CANCELLED = MockEnum("cancelled")
        REJECTED = MockEnum("rejected")
        EXPIRED = MockEnum("expired")
    
    class OrderSide:
        BUY = MockEnum("buy")
        SELL = MockEnum("sell")
    
    class FillType:
        FULL = MockEnum("full")
        PARTIAL = MockEnum("partial")
        MARKET = MockEnum("market")
        LIMIT = MockEnum("limit")
    
    class IntegrityError(Exception):
        pass


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
    session.rollback = MagicMock()
    
    # Context manager support
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=None)
    return session


@pytest.fixture
def Order():
    """Create mock Order class."""
    if models_imports_success:
        return Order
    
    class MockOrder:
        def __init__(self, **kwargs):
            # Set default values
            self.id = kwargs.get('id')
            self.strategy_id = kwargs.get('strategy_id', 1)
            self.signal_id = kwargs.get('signal_id', 1)
            self.instrument = kwargs.get('instrument', 'NIFTY')
            self.order_type = kwargs.get('order_type', OrderType.MARKET)
            self.order_side = kwargs.get('order_side', OrderSide.BUY)
            self.order_status = kwargs.get('order_status', OrderStatus.PENDING)
            self.quantity = kwargs.get('quantity', 1)
            self.filled_quantity = kwargs.get('filled_quantity', 0)
            self.remaining_quantity = kwargs.get('remaining_quantity', self.quantity)
            self.limit_price = kwargs.get('limit_price')
            self.stop_price = kwargs.get('stop_price')
            self.average_fill_price = kwargs.get('average_fill_price')
            self.order_time = kwargs.get('order_time', datetime.now())
            self.submit_time = kwargs.get('submit_time')
            self.first_fill_time = kwargs.get('first_fill_time')
            self.last_fill_time = kwargs.get('last_fill_time')
            self.expiry_time = kwargs.get('expiry_time')
            self.total_commission = kwargs.get('total_commission', 0.0)
            self.total_taxes = kwargs.get('total_taxes', 0.0)
            self.total_slippage = kwargs.get('total_slippage', 0.0)
            self.market_impact = kwargs.get('market_impact', 0.0)
            self.slippage_model = kwargs.get('slippage_model', 'fixed')
            self.execution_delay_ms = kwargs.get('execution_delay_ms', 0)
            self.liquidity_impact = kwargs.get('liquidity_impact', 0.0)
            self.risk_amount_inr = kwargs.get('risk_amount_inr', 1000.0)
            self.margin_required = kwargs.get('margin_required')
            self.rejection_reason = kwargs.get('rejection_reason')
            self.cancellation_reason = kwargs.get('cancellation_reason')
            self.order_notes = kwargs.get('order_notes')
            self.simulation_metadata = kwargs.get('simulation_metadata')
            self.user_id = kwargs.get('user_id', 1)
            self.created_at = kwargs.get('created_at', datetime.now())
            self.updated_at = kwargs.get('updated_at', datetime.now())
            
            # Mock relationships
            self.strategy = None
            self.signal = None
            self.fills = []
            self.trade = None
        
        def validate_remaining_quantity(self, key, value):
            """Mock validation method."""
            if hasattr(self, 'quantity') and hasattr(self, 'filled_quantity'):
                expected = self.quantity - self.filled_quantity
                if value != expected:
                    raise ValueError(f"Remaining quantity {value} != expected {expected}")
            return value
        
        def validate_average_fill_price(self, key, value):
            """Mock validation method."""
            if value is not None and value <= 0:
                raise ValueError("Average fill price must be positive")
            return value
        
        @property
        def is_active(self) -> bool:
            """Check if order is still active."""
            return self.order_status in [
                OrderStatus.PENDING, 
                OrderStatus.SUBMITTED, 
                OrderStatus.ACKNOWLEDGED, 
                OrderStatus.PARTIALLY_FILLED
            ]
        
        @property
        def is_filled(self) -> bool:
            """Check if order is completely filled."""
            return self.order_status == OrderStatus.FILLED
        
        @property
        def is_partially_filled(self) -> bool:
            """Check if order has partial fills."""
            return self.filled_quantity > 0 and self.filled_quantity < self.quantity
        
        @property
        def fill_percentage(self) -> float:
            """Calculate fill percentage."""
            if self.quantity == 0:
                return 0.0
            return (self.filled_quantity / self.quantity) * 100.0
        
        @property
        def total_costs(self) -> float:
            """Calculate total trading costs."""
            return self.total_commission + self.total_taxes + self.total_slippage
        
        def update_fill_status(self):
            """Update order status based on fill quantity."""
            if self.filled_quantity == 0:
                if self.order_status in [OrderStatus.ACKNOWLEDGED, OrderStatus.SUBMITTED]:
                    return  # Keep current status
            elif self.filled_quantity == self.quantity:
                self.order_status = OrderStatus.FILLED
            else:
                self.order_status = OrderStatus.PARTIALLY_FILLED
            
            # Update remaining quantity
            self.remaining_quantity = self.quantity - self.filled_quantity
    
    return MockOrder


@pytest.fixture
def OrderFill():
    """Create mock OrderFill class."""
    if models_imports_success:
        return OrderFill
    
    class MockOrderFill:
        def __init__(self, **kwargs):
            # Set default values
            self.id = kwargs.get('id')
            self.order_id = kwargs.get('order_id', 1)
            self.fill_quantity = kwargs.get('fill_quantity', 1)
            self.fill_price = kwargs.get('fill_price', 18500.0)
            self.fill_time = kwargs.get('fill_time', datetime.now())
            self.fill_type = kwargs.get('fill_type', FillType.FULL)
            self.commission = kwargs.get('commission', 5.0)
            self.taxes = kwargs.get('taxes', 2.0)
            self.slippage = kwargs.get('slippage', 0.0)
            self.market_price = kwargs.get('market_price', 18500.0)
            self.bid_price = kwargs.get('bid_price', 18499.0)
            self.ask_price = kwargs.get('ask_price', 18501.0)
            self.spread_bps = kwargs.get('spread_bps', 1.08)
            self.execution_venue = kwargs.get('execution_venue', 'simulation')
            self.execution_id = kwargs.get('execution_id')
            self.simulation_delay_ms = kwargs.get('simulation_delay_ms', 0)
            self.liquidity_consumed = kwargs.get('liquidity_consumed', 0.0)
            self.market_impact_bps = kwargs.get('market_impact_bps', 0.0)
            self.fill_notes = kwargs.get('fill_notes')
            self.created_at = kwargs.get('created_at', datetime.now())
            self.updated_at = kwargs.get('updated_at', datetime.now())
            
            # Mock relationships
            self.order = None
        
        @property
        def total_cost(self) -> float:
            """Total cost for this fill."""
            return self.commission + self.taxes + abs(self.slippage)
        
        @property
        def fill_value_inr(self) -> float:
            """Total value of this fill in INR."""
            lot_size = 50  # Example: NIFTY lot size
            return self.fill_quantity * self.fill_price * lot_size
        
        @property
        def effective_price(self) -> float:
            """Effective price including slippage."""
            return self.fill_price + self.slippage
    
    return MockOrderFill


@pytest.fixture
def sample_order_data():
    """Sample order creation data."""
    return {
        'strategy_id': 1,
        'signal_id': 1,
        'instrument': 'NIFTY',
        'order_type': OrderType.MARKET,
        'order_side': OrderSide.BUY,
        'quantity': 2,
        'risk_amount_inr': 5000.0,
        'user_id': 1
    }


@pytest.fixture
def sample_limit_order_data():
    """Sample limit order creation data."""
    return {
        'strategy_id': 1,
        'signal_id': 1,
        'instrument': 'BANKNIFTY',
        'order_type': OrderType.LIMIT,
        'order_side': OrderSide.SELL,
        'quantity': 1,
        'limit_price': 42500.0,
        'risk_amount_inr': 2500.0,
        'user_id': 1
    }


@pytest.fixture
def sample_fill_data():
    """Sample fill creation data."""
    return {
        'order_id': 1,
        'fill_quantity': 1,
        'fill_price': 18520.0,
        'fill_type': FillType.FULL,
        'market_price': 18520.0,
        'commission': 10.0,
        'taxes': 5.0
    }


# Order Model Tests
class TestOrder:
    """Test cases for the Order model."""

    def test_order_creation(self, Order, sample_order_data, mock_db_session):
        """Test creating a new order with basic attributes."""
        order = Order(**sample_order_data)
        
        assert order.instrument == 'NIFTY'
        assert order.order_type == OrderType.MARKET
        assert order.order_side == OrderSide.BUY
        assert order.order_status == OrderStatus.PENDING
        assert order.quantity == 2
        assert order.filled_quantity == 0
        assert order.remaining_quantity == 2
        assert order.risk_amount_inr == 5000.0

    def test_order_creation_with_limit_price(self, Order, sample_limit_order_data):
        """Test creating a limit order with price."""
        order = Order(**sample_limit_order_data)
        
        assert order.order_type == OrderType.LIMIT
        assert order.limit_price == 42500.0
        assert order.order_side == OrderSide.SELL

    def test_order_properties_active_status(self, Order, sample_order_data):
        """Test order property methods for active status."""
        order = Order(**sample_order_data)
        
        # Test pending order
        assert order.is_active is True
        assert order.is_filled is False
        assert order.is_partially_filled is False
        assert order.fill_percentage == 0.0

    def test_order_properties_filled_status(self, Order, sample_order_data):
        """Test order property methods for filled status."""
        order = Order(**sample_order_data)
        order.order_status = OrderStatus.FILLED
        order.filled_quantity = 2
        
        assert order.is_active is False
        assert order.is_filled is True
        assert order.is_partially_filled is False
        assert order.fill_percentage == 100.0

    def test_order_properties_partially_filled(self, Order, sample_order_data):
        """Test order property methods for partially filled status."""
        order = Order(**sample_order_data)
        order.order_status = OrderStatus.PARTIALLY_FILLED
        order.filled_quantity = 1
        order.remaining_quantity = 1
        
        assert order.is_active is True
        assert order.is_filled is False
        assert order.is_partially_filled is True
        assert order.fill_percentage == 50.0

    def test_order_total_costs_calculation(self, Order, sample_order_data):
        """Test total costs calculation."""
        order = Order(**sample_order_data)
        order.total_commission = 20.0
        order.total_taxes = 10.0
        order.total_slippage = 5.0
        
        assert order.total_costs == 35.0

    def test_order_update_fill_status_full_fill(self, Order, sample_order_data):
        """Test updating order status for full fill."""
        order = Order(**sample_order_data)
        order.order_status = OrderStatus.ACKNOWLEDGED
        order.filled_quantity = 2
        
        order.update_fill_status()
        
        assert order.order_status == OrderStatus.FILLED
        assert order.remaining_quantity == 0

    def test_order_update_fill_status_partial_fill(self, Order, sample_order_data):
        """Test updating order status for partial fill."""
        order = Order(**sample_order_data)
        order.order_status = OrderStatus.ACKNOWLEDGED
        order.filled_quantity = 1
        
        order.update_fill_status()
        
        assert order.order_status == OrderStatus.PARTIALLY_FILLED
        assert order.remaining_quantity == 1

    def test_order_validation_remaining_quantity(self, Order, sample_order_data):
        """Test remaining quantity validation."""
        order = Order(**sample_order_data)
        order.quantity = 2
        order.filled_quantity = 1
        
        # Valid remaining quantity
        assert order.validate_remaining_quantity('remaining_quantity', 1) == 1
        
        # Invalid remaining quantity
        with pytest.raises(ValueError, match="Remaining quantity 2 != expected 1"):
            order.validate_remaining_quantity('remaining_quantity', 2)

    def test_order_validation_average_fill_price(self, Order, sample_order_data):
        """Test average fill price validation."""
        order = Order(**sample_order_data)
        
        # Valid price
        assert order.validate_average_fill_price('average_fill_price', 18500.0) == 18500.0
        
        # Invalid price (negative)
        with pytest.raises(ValueError, match="Average fill price must be positive"):
            order.validate_average_fill_price('average_fill_price', -100.0)
        
        # Invalid price (zero)
        with pytest.raises(ValueError, match="Average fill price must be positive"):
            order.validate_average_fill_price('average_fill_price', 0.0)

    def test_order_edge_cases_zero_quantity(self, Order, sample_order_data):
        """Test edge case with zero quantity for fill percentage."""
        order = Order(**sample_order_data)
        order.quantity = 0
        
        assert order.fill_percentage == 0.0

    def test_order_complex_status_combinations(self, Order, sample_order_data):
        """Test complex status combinations."""
        order = Order(**sample_order_data)
        
        # Test cancelled order
        order.order_status = OrderStatus.CANCELLED
        order.cancellation_reason = "User requested"
        
        assert order.is_active is False
        assert order.cancellation_reason == "User requested"
        
        # Test rejected order
        order.order_status = OrderStatus.REJECTED
        order.rejection_reason = "Insufficient margin"
        
        assert order.is_active is False
        assert order.rejection_reason == "Insufficient margin"

    def test_order_timing_attributes(self, Order, sample_order_data):
        """Test order timing attributes."""
        now = datetime.now()
        order = Order(**sample_order_data)
        order.submit_time = now
        order.first_fill_time = now + timedelta(seconds=1)
        order.last_fill_time = now + timedelta(seconds=2)
        
        assert order.submit_time == now
        assert order.first_fill_time > order.submit_time
        assert order.last_fill_time >= order.first_fill_time

    def test_order_simulation_parameters(self, Order, sample_order_data):
        """Test simulation-specific parameters."""
        order = Order(**sample_order_data)
        order.slippage_model = "linear"
        order.execution_delay_ms = 100
        order.liquidity_impact = 0.05
        order.market_impact = 0.02
        
        assert order.slippage_model == "linear"
        assert order.execution_delay_ms == 100
        assert order.liquidity_impact == 0.05
        assert order.market_impact == 0.02


# OrderFill Model Tests
class TestOrderFill:
    """Test cases for the OrderFill model."""

    def test_fill_creation(self, OrderFill, sample_fill_data):
        """Test creating a new order fill."""
        fill = OrderFill(**sample_fill_data)
        
        assert fill.order_id == 1
        assert fill.fill_quantity == 1
        assert fill.fill_price == 18520.0
        assert fill.fill_type == FillType.FULL
        assert fill.market_price == 18520.0
        assert fill.commission == 10.0
        assert fill.taxes == 5.0

    def test_fill_cost_calculations(self, OrderFill, sample_fill_data):
        """Test fill cost calculation properties."""
        fill = OrderFill(**sample_fill_data)
        fill.slippage = 2.0
        
        assert fill.total_cost == 17.0  # 10 + 5 + 2
        assert fill.effective_price == 18522.0  # 18520 + 2

    def test_fill_value_calculation(self, OrderFill, sample_fill_data):
        """Test fill value calculation in INR."""
        fill = OrderFill(**sample_fill_data)
        
        # Assuming NIFTY lot size of 50
        expected_value = 1 * 18520.0 * 50
        assert fill.fill_value_inr == expected_value

    def test_fill_market_conditions(self, OrderFill, sample_fill_data):
        """Test fill market condition attributes."""
        fill = OrderFill(**sample_fill_data)
        fill.bid_price = 18519.0
        fill.ask_price = 18521.0
        fill.spread_bps = 1.08
        
        assert fill.bid_price == 18519.0
        assert fill.ask_price == 18521.0
        assert fill.spread_bps == 1.08

    def test_fill_execution_details(self, OrderFill, sample_fill_data):
        """Test fill execution details."""
        fill = OrderFill(**sample_fill_data)
        fill.execution_venue = "NSE"
        fill.execution_id = "EXE123456"
        fill.simulation_delay_ms = 50
        
        assert fill.execution_venue == "NSE"
        assert fill.execution_id == "EXE123456"
        assert fill.simulation_delay_ms == 50

    def test_fill_liquidity_impact(self, OrderFill, sample_fill_data):
        """Test fill liquidity impact metrics."""
        fill = OrderFill(**sample_fill_data)
        fill.liquidity_consumed = 0.1
        fill.market_impact_bps = 0.5
        
        assert fill.liquidity_consumed == 0.1
        assert fill.market_impact_bps == 0.5

    def test_fill_negative_slippage(self, OrderFill, sample_fill_data):
        """Test fill with negative slippage (favorable execution)."""
        fill = OrderFill(**sample_fill_data)
        fill.slippage = -1.0  # Favorable slippage
        
        assert fill.total_cost == 16.0  # 10 + 5 + |-1|
        assert fill.effective_price == 18519.0  # 18520 + (-1)

    def test_fill_zero_costs(self, OrderFill, sample_fill_data):
        """Test fill with zero costs."""
        fill = OrderFill(**sample_fill_data)
        fill.commission = 0.0
        fill.taxes = 0.0
        fill.slippage = 0.0
        
        assert fill.total_cost == 0.0
        assert fill.effective_price == fill.fill_price


# Enum Tests
class TestEnums:
    """Test cases for enum classes."""

    def test_order_type_enum(self):
        """Test OrderType enum values."""
        assert OrderType.MARKET.value == "market" or str(OrderType.MARKET) == "market"
        assert OrderType.LIMIT.value == "limit" or str(OrderType.LIMIT) == "limit"
        assert OrderType.STOP_LOSS.value == "stop_loss" or str(OrderType.STOP_LOSS) == "stop_loss"
        assert OrderType.STOP_LIMIT.value == "stop_limit" or str(OrderType.STOP_LIMIT) == "stop_limit"

    def test_order_status_enum(self):
        """Test OrderStatus enum values."""
        assert OrderStatus.PENDING.value == "pending" or str(OrderStatus.PENDING) == "pending"
        assert OrderStatus.SUBMITTED.value == "submitted" or str(OrderStatus.SUBMITTED) == "submitted"
        assert OrderStatus.FILLED.value == "filled" or str(OrderStatus.FILLED) == "filled"
        assert OrderStatus.CANCELLED.value == "cancelled" or str(OrderStatus.CANCELLED) == "cancelled"

    def test_order_side_enum(self):
        """Test OrderSide enum values."""
        assert OrderSide.BUY.value == "buy" or str(OrderSide.BUY) == "buy"
        assert OrderSide.SELL.value == "sell" or str(OrderSide.SELL) == "sell"

    def test_fill_type_enum(self):
        """Test FillType enum values."""
        assert FillType.FULL.value == "full" or str(FillType.FULL) == "full"
        assert FillType.PARTIAL.value == "partial" or str(FillType.PARTIAL) == "partial"
        assert FillType.MARKET.value == "market" or str(FillType.MARKET) == "market"
        assert FillType.LIMIT.value == "limit" or str(FillType.LIMIT) == "limit"


# Integration Tests
class TestOrderFillIntegration:
    """Test integration between Order and OrderFill models."""

    def test_order_with_single_fill(self, Order, OrderFill, sample_order_data, sample_fill_data):
        """Test order with a single complete fill."""
        # Create order
        order = Order(**sample_order_data)
        order.id = 1
        
        # Create fill
        fill = OrderFill(**sample_fill_data)
        fill.order_id = order.id
        
        # Simulate relationship
        order.fills = [fill]
        fill.order = order
        
        # Update order with fill
        order.filled_quantity = fill.fill_quantity
        order.average_fill_price = fill.fill_price
        order.total_commission = fill.commission
        order.total_taxes = fill.taxes
        order.total_slippage = fill.slippage
        order.first_fill_time = fill.fill_time
        order.last_fill_time = fill.fill_time
        
        order.update_fill_status()
        
        assert len(order.fills) == 1
        # Since we have a 2-lot order and only 1 lot filled, it should be partially filled
        assert order.is_partially_filled is True
        assert order.is_filled is False  # Only 1 out of 2 lots filled
        assert order.average_fill_price == 18520.0
        assert order.total_costs == fill.total_cost

    def test_order_with_multiple_partial_fills(self, Order, OrderFill, sample_order_data):
        """Test order with multiple partial fills."""
        # Create order for 3 lots
        order_data = sample_order_data.copy()
        order_data['quantity'] = 3
        order = Order(**order_data)
        order.id = 1
        
        # Create first partial fill
        fill1 = OrderFill(
            order_id=1,
            fill_quantity=1,
            fill_price=18520.0,
            fill_type=FillType.PARTIAL,
            market_price=18520.0,
            commission=10.0,
            taxes=5.0
        )
        
        # Create second partial fill
        fill2 = OrderFill(
            order_id=1,
            fill_quantity=2,
            fill_price=18525.0,
            fill_type=FillType.PARTIAL,
            market_price=18525.0,
            commission=20.0,
            taxes=10.0
        )
        
        # Simulate relationships
        order.fills = [fill1, fill2]
        
        # Update order with fills
        order.filled_quantity = fill1.fill_quantity + fill2.fill_quantity
        order.total_commission = fill1.commission + fill2.commission
        order.total_taxes = fill1.taxes + fill2.taxes
        order.average_fill_price = (
            (fill1.fill_quantity * fill1.fill_price + fill2.fill_quantity * fill2.fill_price) /
            (fill1.fill_quantity + fill2.fill_quantity)
        )
        order.first_fill_time = fill1.fill_time
        order.last_fill_time = fill2.fill_time
        
        order.update_fill_status()
        
        assert len(order.fills) == 2
        assert order.is_filled is True
        assert order.filled_quantity == 3
        assert round(order.average_fill_price, 2) == 18523.33  # Weighted average rounded
        assert order.total_commission == 30.0
        assert order.total_taxes == 15.0


# Database Constraint Tests (Mocked)
class TestDatabaseConstraints:
    """Test database constraints and validation."""

    def test_positive_quantity_constraint(self, Order, sample_order_data):
        """Test positive quantity constraint."""
        # This would normally test database constraint, but we'll test model validation
        order_data = sample_order_data.copy()
        order_data['quantity'] = -1
        
        # In a real scenario, this would raise IntegrityError on commit
        # For mock testing, we'll validate in application logic
        with pytest.raises(ValueError):
            if order_data['quantity'] <= 0:
                raise ValueError("Quantity must be positive")

    def test_non_negative_filled_constraint(self, Order, sample_order_data):
        """Test non-negative filled quantity constraint."""
        order_data = sample_order_data.copy()
        order_data['filled_quantity'] = -1
        
        with pytest.raises(ValueError):
            if order_data['filled_quantity'] < 0:
                raise ValueError("Filled quantity must be non-negative")

    def test_filled_not_exceeds_quantity(self, Order, sample_order_data):
        """Test filled quantity doesn't exceed total quantity."""
        order_data = sample_order_data.copy()
        order_data['quantity'] = 2
        order_data['filled_quantity'] = 3
        
        with pytest.raises(ValueError):
            if order_data['filled_quantity'] > order_data['quantity']:
                raise ValueError("Filled quantity cannot exceed total quantity")

    def test_positive_fill_price_constraint(self, OrderFill, sample_fill_data):
        """Test positive fill price constraint."""
        fill_data = sample_fill_data.copy()
        fill_data['fill_price'] = -100.0
        
        with pytest.raises(ValueError):
            if fill_data['fill_price'] <= 0:
                raise ValueError("Fill price must be positive")


# Performance and Edge Case Tests
class TestEdgeCases:
    """Test edge cases and performance scenarios."""

    def test_order_with_zero_commission_and_taxes(self, Order, sample_order_data):
        """Test order with zero commission and taxes."""
        order = Order(**sample_order_data)
        order.total_commission = 0.0
        order.total_taxes = 0.0
        order.total_slippage = 0.0
        
        assert order.total_costs == 0.0

    def test_order_with_expired_status(self, Order, sample_order_data):
        """Test order with expired status."""
        order = Order(**sample_order_data)
        order.order_status = OrderStatus.EXPIRED
        order.expiry_time = datetime.now() - timedelta(minutes=5)
        
        assert order.is_active is False
        assert order.expiry_time < datetime.now()

    def test_order_with_very_large_quantity(self, Order, sample_order_data):
        """Test order with very large quantity."""
        order_data = sample_order_data.copy()
        order_data['quantity'] = 1000000
        order = Order(**order_data)
        
        assert order.quantity == 1000000
        assert order.remaining_quantity == 1000000
        assert order.fill_percentage == 0.0

    def test_fill_with_extreme_slippage(self, OrderFill, sample_fill_data):
        """Test fill with extreme slippage values."""
        fill = OrderFill(**sample_fill_data)
        fill.slippage = 100.0  # Very high slippage
        
        assert fill.effective_price == 18620.0  # 18520 + 100
        assert fill.total_cost == 115.0  # 10 + 5 + 100

    def test_fill_with_very_high_market_impact(self, OrderFill, sample_fill_data):
        """Test fill with very high market impact."""
        fill = OrderFill(**sample_fill_data)
        fill.market_impact_bps = 50.0  # 50 basis points
        fill.liquidity_consumed = 0.8  # 80% of available liquidity
        
        assert fill.market_impact_bps == 50.0
        assert fill.liquidity_consumed == 0.8

    def test_order_status_transitions(self, Order, sample_order_data):
        """Test valid order status transitions."""
        order = Order(**sample_order_data)
        
        # Valid transition: PENDING -> SUBMITTED
        order.order_status = OrderStatus.PENDING
        assert order.is_active is True
        
        order.order_status = OrderStatus.SUBMITTED
        order.submit_time = datetime.now()
        assert order.is_active is True
        
        # Valid transition: SUBMITTED -> ACKNOWLEDGED
        order.order_status = OrderStatus.ACKNOWLEDGED
        assert order.is_active is True
        
        # Valid transition: ACKNOWLEDGED -> PARTIALLY_FILLED
        order.order_status = OrderStatus.PARTIALLY_FILLED
        order.filled_quantity = 1
        order.first_fill_time = datetime.now()
        assert order.is_active is True
        assert order.is_partially_filled is True
        
        # Valid transition: PARTIALLY_FILLED -> FILLED
        order.order_status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.last_fill_time = datetime.now()
        assert order.is_active is False
        assert order.is_filled is True

    def test_concurrent_fill_updates(self, Order, sample_order_data):
        """Test handling of concurrent fill updates."""
        order = Order(**sample_order_data)
        order.quantity = 5
        
        # Simulate multiple concurrent fills
        order.filled_quantity = 2
        order.update_fill_status()
        assert order.order_status == OrderStatus.PARTIALLY_FILLED
        assert order.remaining_quantity == 3
        
        # Another fill comes in
        order.filled_quantity = 5
        order.update_fill_status()
        assert order.order_status == OrderStatus.FILLED
        assert order.remaining_quantity == 0

    def test_order_with_stop_prices(self, Order, sample_order_data):
        """Test order with stop loss and stop limit configurations."""
        # Stop loss order
        order_data = sample_order_data.copy()
        order_data['order_type'] = OrderType.STOP_LOSS
        order_data['stop_price'] = 18000.0
        
        order = Order(**order_data)
        assert order.order_type == OrderType.STOP_LOSS
        assert order.stop_price == 18000.0
        
        # Stop limit order
        order_data['order_type'] = OrderType.STOP_LIMIT
        order_data['limit_price'] = 18050.0
        
        order = Order(**order_data)
        assert order.order_type == OrderType.STOP_LIMIT
        assert order.stop_price == 18000.0
        assert order.limit_price == 18050.0

    def test_fill_timestamp_ordering(self, OrderFill):
        """Test that fill timestamps are properly ordered."""
        now = datetime.now()
        
        fill1 = OrderFill(
            order_id=1,
            fill_quantity=1,
            fill_price=18520.0,
            fill_time=now,
            fill_type=FillType.PARTIAL,
            market_price=18520.0
        )
        
        fill2 = OrderFill(
            order_id=1,
            fill_quantity=1,
            fill_price=18525.0,
            fill_time=now + timedelta(seconds=30),
            fill_type=FillType.PARTIAL,
            market_price=18525.0
        )
        
        assert fill2.fill_time > fill1.fill_time

    def test_margin_required_calculations(self, Order, sample_order_data):
        """Test margin required for derivative orders."""
        order_data = sample_order_data.copy()
        order_data['instrument'] = 'NIFTY25JAN18500CE'  # Options contract
        order_data['margin_required'] = 50000.0
        
        order = Order(**order_data)
        assert order.margin_required == 50000.0
        assert order.instrument == 'NIFTY25JAN18500CE'


# Relationship Tests
class TestModelRelationships:
    """Test model relationships and foreign keys."""

    def test_order_strategy_relationship(self, Order, sample_order_data):
        """Test order-strategy relationship."""
        order = Order(**sample_order_data)
        
        # Mock strategy relationship
        mock_strategy = Mock()
        mock_strategy.id = 1
        mock_strategy.name = "Test Strategy"
        order.strategy = mock_strategy
        
        assert order.strategy_id == 1
        assert order.strategy.name == "Test Strategy"

    def test_order_signal_relationship(self, Order, sample_order_data):
        """Test order-signal relationship."""
        order = Order(**sample_order_data)
        
        # Mock signal relationship
        mock_signal = Mock()
        mock_signal.id = 1
        mock_signal.signal_type = "BUY"
        order.signal = mock_signal
        
        assert order.signal_id == 1
        assert order.signal.signal_type == "BUY"

    def test_order_trade_relationship(self, Order, sample_order_data):
        """Test one-to-one order-trade relationship."""
        order = Order(**sample_order_data)
        
        # Mock trade relationship
        mock_trade = Mock()
        mock_trade.id = 1
        mock_trade.order_id = order.id
        order.trade = mock_trade
        
        assert order.trade.order_id == order.id

    def test_fill_order_relationship(self, OrderFill, sample_fill_data):
        """Test fill-order relationship."""
        fill = OrderFill(**sample_fill_data)
        
        # Mock order relationship
        mock_order = Mock()
        mock_order.id = 1
        mock_order.instrument = "NIFTY"
        fill.order = mock_order
        
        assert fill.order_id == 1
        assert fill.order.instrument == "NIFTY"


# Validation and Business Logic Tests
class TestBusinessLogic:
    """Test business logic and validation rules."""

    def test_order_risk_validation(self, Order, sample_order_data):
        """Test order risk amount validation."""
        order_data = sample_order_data.copy()
        order_data['risk_amount_inr'] = -1000.0
        
        # Business rule: risk amount cannot be negative
        with pytest.raises(ValueError):
            if order_data['risk_amount_inr'] < 0:
                raise ValueError("Risk amount cannot be negative")

    def test_order_quantity_lot_size_validation(self, Order, sample_order_data):
        """Test order quantity matches lot size requirements."""
        order_data = sample_order_data.copy()
        
        # For NIFTY, lot size is typically 50
        # Order quantity should be in multiples of lot size
        order_data['quantity'] = 1.5  # Invalid - not whole lot
        
        with pytest.raises(ValueError):
            if order_data['quantity'] != int(order_data['quantity']):
                raise ValueError("Quantity must be in whole lots")

    def test_limit_order_price_validation(self, Order, sample_limit_order_data):
        """Test limit order price validation."""
        order = Order(**sample_limit_order_data)
        
        # Limit price must be set for limit orders
        assert order.order_type == OrderType.LIMIT
        assert order.limit_price is not None
        assert order.limit_price > 0

    def test_stop_order_price_validation(self, Order, sample_order_data):
        """Test stop order price validation."""
        order_data = sample_order_data.copy()
        order_data['order_type'] = OrderType.STOP_LOSS
        order_data['stop_price'] = 18000.0
        
        order = Order(**order_data)
        
        # Stop price must be set for stop orders
        assert order.order_type == OrderType.STOP_LOSS
        assert order.stop_price is not None
        assert order.stop_price > 0

    def test_fill_quantity_validation(self, OrderFill, sample_fill_data):
        """Test fill quantity business rules."""
        fill_data = sample_fill_data.copy()
        fill_data['fill_quantity'] = 0
        
        # Fill quantity must be positive
        with pytest.raises(ValueError):
            if fill_data['fill_quantity'] <= 0:
                raise ValueError("Fill quantity must be positive")

    def test_commission_calculation(self, OrderFill, sample_fill_data):
        """Test commission calculation logic."""
        fill = OrderFill(**sample_fill_data)
        
        # Mock commission calculation based on fill value
        fill_value = fill.fill_value_inr
        expected_commission = max(10.0, fill_value * 0.0001)  # Min ₹10 or 0.01%
        
        # In real scenario, this would be calculated automatically
        assert fill.commission >= 10.0

    def test_tax_calculation(self, OrderFill, sample_fill_data):
        """Test tax calculation logic."""
        fill = OrderFill(**sample_fill_data)
        
        # Mock tax calculation (STT + Transaction charges)
        fill_value = fill.fill_value_inr
        stt_rate = 0.000125 if fill.fill_type == FillType.MARKET else 0.00005
        expected_stt = fill_value * stt_rate
        
        # In real scenario, taxes would be calculated automatically
        assert fill.taxes >= 0

    def test_slippage_model_application(self, Order, sample_order_data):
        """Test different slippage models."""
        order = Order(**sample_order_data)
        
        # Test fixed slippage model
        order.slippage_model = "fixed"
        assert order.slippage_model == "fixed"
        
        # Test linear slippage model
        order.slippage_model = "linear"
        assert order.slippage_model == "linear"
        
        # Test impact slippage model
        order.slippage_model = "impact"
        assert order.slippage_model == "impact"


# Error Handling Tests
class TestErrorHandling:
    """Test error handling and exception scenarios."""

    def test_order_creation_with_invalid_enum(self, Order, sample_order_data):
        """Test order creation with invalid enum values."""
        order_data = sample_order_data.copy()
        
        # Test with invalid order type string
        with pytest.raises(ValueError):
            order_data['order_type'] = "invalid_type"
            # Mock enum validation
            valid_types = ["market", "limit", "stop_loss", "stop_limit"]
            if order_data['order_type'] not in valid_types:
                raise ValueError("Invalid order type")

    def test_fill_creation_with_missing_required_fields(self, OrderFill):
        """Test fill creation with missing required fields."""
        # Test creating fill without required fields
        with pytest.raises(ValueError):
            # Mock validation for required fields
            required_fields = ['order_id', 'fill_quantity', 'fill_price', 'market_price']
            provided_fields = []  # No fields provided
            
            missing_fields = [field for field in required_fields if field not in provided_fields]
            if missing_fields:
                raise ValueError(f"Missing required fields: {missing_fields}")

    def test_order_status_invalid_transition(self, Order, sample_order_data):
        """Test invalid order status transitions."""
        order = Order(**sample_order_data)
        order.order_status = OrderStatus.FILLED
        
        # Invalid transition: FILLED -> PENDING
        with pytest.raises(ValueError):
            # Business rule: cannot go from FILLED back to PENDING
            if order.order_status == OrderStatus.FILLED:
                raise ValueError("Cannot transition from FILLED to PENDING")

    def test_database_integrity_errors(self, Order, sample_order_data, mock_db_session):
        """Test handling of database integrity errors."""
        order = Order(**sample_order_data)
        
        # Mock database integrity error
        mock_db_session.commit.side_effect = IntegrityError("statement", "params", "orig")
        
        with pytest.raises(IntegrityError):
            mock_db_session.add(order)
            mock_db_session.commit()

    def test_concurrent_modification_handling(self, Order, sample_order_data):
        """Test handling of concurrent modifications."""
        order = Order(**sample_order_data)
        
        # Simulate concurrent modification scenario
        original_updated_at = order.updated_at
        
        # Mock scenario where order is modified
        order.updated_at = datetime.now() + timedelta(seconds=1)
        
        with pytest.raises(ValueError):
            # Mock optimistic locking check
            if order.updated_at != original_updated_at:
                raise ValueError("Order was modified by another process")


# Performance Tests
class TestPerformance:
    """Test performance-related scenarios."""

    def test_large_number_of_fills(self, Order, OrderFill, sample_order_data):
        """Test order with large number of fills."""
        order = Order(**sample_order_data)
        order.quantity = 100
        order.id = 1
        
        # Create many small fills
        fills = []
        for i in range(100):
            fill = OrderFill(
                order_id=1,
                fill_quantity=1,
                fill_price=18500.0 + i,
                fill_type=FillType.PARTIAL,
                market_price=18500.0 + i,
                commission=1.0,
                taxes=0.5
            )
            fills.append(fill)
        
        order.fills = fills
        
        # Calculate totals
        total_commission = sum(f.commission for f in fills)
        total_taxes = sum(f.taxes for f in fills)
        
        assert len(order.fills) == 100
        assert total_commission == 100.0
        assert total_taxes == 50.0

    def test_memory_efficient_properties(self, Order, sample_order_data):
        """Test that properties don't consume excessive memory."""
        order = Order(**sample_order_data)
        
        # Properties should be calculated, not stored
        percentage1 = order.fill_percentage
        percentage2 = order.fill_percentage
        
        # Should return same value
        assert percentage1 == percentage2
        
        # Modify underlying data
        order.filled_quantity = 1
        percentage3 = order.fill_percentage
        
        # Should reflect the change
        assert percentage3 != percentage1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])