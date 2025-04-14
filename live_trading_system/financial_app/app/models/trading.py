"""
Trading models for the financial application.

This module defines SQLAlchemy models related to orders, executions, positions,
and trades for the trading platform. These models handle the core trading operations,
including order management, execution tracking, and position calculations.

Key features:
- Complete order lifecycle management with state transitions
- Execution and fill tracking with market impact analysis
- Position calculation with real-time P&L tracking
- Trade history for reporting and analysis
- Integration with risk management systems
- Comprehensive audit trails for compliance

Usage:
    These models provide the foundation for the order management system (OMS) and
    execution management system (EMS) components of the trading platform.
"""

from datetime import datetime, date, timedelta
from decimal import Decimal
from enum import Enum, auto
from typing import List, Optional, Dict, Any, Union, Tuple
from sqlalchemy import (
    Column, Integer, String, Float, Numeric, DateTime, Date, 
    ForeignKey, Text, Boolean, Index, CheckConstraint, 
    UniqueConstraint, Table, func, select, and_, or_, text, event, Enum as SQLAEnum
)
from sqlalchemy.orm import relationship, validates, column_property
from sqlalchemy.ext.hybrid import hybrid_property, hybrid_method
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.sql.expression import cast
from uuid import uuid4
import json
import logging

from app.core.database import Base
from app.models.account import Account, Balance

logger = logging.getLogger(__name__)

# Order status enum to ensure consistent state management
class OrderStatus(str, Enum):
    CREATED = "created"           # Initial state, not yet sent to broker
    PENDING_SUBMIT = "pending_submit"  # In process of being submitted
    SUBMITTED = "submitted"       # Sent to broker/exchange
    PARTIALLY_FILLED = "partially_filled"  # Some quantity executed
    FILLED = "filled"             # Fully executed
    PENDING_CANCEL = "pending_cancel"  # Cancel requested, not confirmed
    CANCELED = "canceled"         # Canceled before full execution
    REJECTED = "rejected"         # Rejected by broker/exchange
    EXPIRED = "expired"           # Order expired based on time-in-force
    ERROR = "error"               # System error occurred

# Order types enum
class OrderType(str, Enum):
    MARKET = "market"             # Execute immediately at market price
    LIMIT = "limit"               # Execute at specified price or better
    STOP = "stop"                 # Market order when price hits stop
    STOP_LIMIT = "stop_limit"     # Limit order when price hits stop
    TRAILING_STOP = "trailing_stop"  # Stop that moves with market

# Order sides enum
class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

# Time in force enum
class TimeInForce(str, Enum):
    DAY = "day"                   # Valid for the trading day
    GTC = "gtc"                   # Good till canceled
    IOC = "ioc"                   # Immediate or cancel
    FOK = "fok"                   # Fill or kill
    GTD = "gtd"                   # Good till date

# Position direction enum
class PositionDirection(str, Enum):
    LONG = "long"
    SHORT = "short"

# Order-related events for the audit trail
class OrderEventType(str, Enum):
    CREATED = "created"
    SUBMITTED = "submitted"
    UPDATED = "updated"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    ERROR = "error"


class Order(Base):
    """
    Trading order model.
    
    Represents an order to buy or sell a financial instrument. Orders have 
    a lifecycle from creation through submission to execution or cancellation.
    """
    __tablename__ = "orders"
    
    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(String(36), default=lambda: str(uuid4()), unique=True, nullable=False)
    account_id = Column(Integer, ForeignKey("accounts.id", ondelete="CASCADE"), nullable=False)
    
    # Order details
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(SQLAEnum(OrderSide), nullable=False)
    order_type = Column(SQLAEnum(OrderType), nullable=False)
    quantity = Column(Numeric(precision=18, scale=8), nullable=False)
    price = Column(Numeric(precision=18, scale=8))  # For limit orders
    stop_price = Column(Numeric(precision=18, scale=8))  # For stop orders
    trailing_amount = Column(Numeric(precision=18, scale=8))  # For trailing stops
    trailing_percent = Column(Numeric(precision=10, scale=2))  # For percent-based trailing stops
    
    # Order status and lifecycle
    status = Column(SQLAEnum(OrderStatus), nullable=False, default=OrderStatus.CREATED, index=True)
    time_in_force = Column(SQLAEnum(TimeInForce), nullable=False, default=TimeInForce.DAY)
    expire_at = Column(DateTime)  # For GTD orders
    
    # Execution tracking
    filled_quantity = Column(Numeric(precision=18, scale=8), default=0, nullable=False)
    average_fill_price = Column(Numeric(precision=18, scale=8))
    remaining_quantity = Column(Numeric(precision=18, scale=8))
    
    # Order identifiers
    client_order_id = Column(String(50))  # Client-side order identifier
    broker_order_id = Column(String(50))  # Broker-side order identifier
    
    # Strategy and metadata
    strategy_id = Column(String(50))  # Strategy that generated this order
    tags = Column(Text)  # JSON array of tags for filtering/categorization
    notes = Column(Text)  # Any additional notes
    parent_order_id = Column(String(36), ForeignKey("orders.order_id"), nullable=True)  # For child orders
    
    # Risk parameters
    max_slippage_percent = Column(Numeric(precision=5, scale=2))  # Maximum acceptable slippage
    risk_check_passed = Column(Boolean, default=False)  # Indicates if order passed risk checks
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    submitted_at = Column(DateTime(timezone=True))  # When the order was sent to broker
    filled_at = Column(DateTime(timezone=True))  # When the order was completely filled
    canceled_at = Column(DateTime(timezone=True))  # When the order was canceled
    rejected_at = Column(DateTime(timezone=True))  # When the order was rejected
    
    # Broker and venue information
    broker = Column(String(50))  # Broker or exchange name
    venue = Column(String(50))  # Execution venue if different from broker
    
    # Relationships
    account = relationship("Account", back_populates="orders")
    executions = relationship("Execution", back_populates="order", cascade="all, delete-orphan")
    order_events = relationship("OrderEvent", back_populates="order", cascade="all, delete-orphan")
    child_orders = relationship("Order", backref=relationship.backref("parent_order", remote_side=[order_id]))
    
    # Indices and constraints
    __table_args__ = (
        Index("ix_orders_account_id_symbol", account_id, symbol),
        Index("ix_orders_account_id_status", account_id, status),
        Index("ix_orders_created_at", created_at),
        CheckConstraint("(order_type != 'limit' AND order_type != 'stop_limit') OR price IS NOT NULL", 
                        name="ck_limit_orders_require_price"),
        CheckConstraint("(order_type != 'stop' AND order_type != 'stop_limit') OR stop_price IS NOT NULL", 
                        name="ck_stop_orders_require_stop_price"),
        CheckConstraint("(order_type != 'trailing_stop') OR (trailing_amount IS NOT NULL OR trailing_percent IS NOT NULL)", 
                        name="ck_trailing_stop_requires_parameters"),
        CheckConstraint("time_in_force != 'gtd' OR expire_at IS NOT NULL", 
                        name="ck_gtd_orders_require_expire_at"),
        CheckConstraint("quantity > 0", 
                        name="ck_order_quantity_positive"),
    )
    
    @validates('status')
    def validate_status_transition(self, key, new_status):
        """Validates that the order status transition is valid."""
        if hasattr(self, 'status') and self.status:
            current_status = self.status
            
            # Define valid status transitions
            valid_transitions = {
                OrderStatus.CREATED: [OrderStatus.PENDING_SUBMIT, OrderStatus.REJECTED, OrderStatus.ERROR],
                OrderStatus.PENDING_SUBMIT: [OrderStatus.SUBMITTED, OrderStatus.REJECTED, OrderStatus.ERROR],
                OrderStatus.SUBMITTED: [OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED, 
                                      OrderStatus.PENDING_CANCEL, OrderStatus.CANCELED, 
                                      OrderStatus.REJECTED, OrderStatus.EXPIRED, OrderStatus.ERROR],
                OrderStatus.PARTIALLY_FILLED: [OrderStatus.FILLED, OrderStatus.PENDING_CANCEL, 
                                             OrderStatus.CANCELED, OrderStatus.EXPIRED, OrderStatus.ERROR],
                OrderStatus.PENDING_CANCEL: [OrderStatus.CANCELED, OrderStatus.PARTIALLY_FILLED, 
                                           OrderStatus.FILLED, OrderStatus.ERROR],
                # Terminal states should not transition
                OrderStatus.FILLED: [OrderStatus.ERROR],
                OrderStatus.CANCELED: [OrderStatus.ERROR],
                OrderStatus.REJECTED: [OrderStatus.ERROR],
                OrderStatus.EXPIRED: [OrderStatus.ERROR],
                OrderStatus.ERROR: []
            }
            
            if new_status not in valid_transitions[current_status]:
                raise ValueError(f"Invalid status transition from {current_status} to {new_status}")
                
            # Record status change in event log
            if new_status != current_status:
                self.add_event(OrderEventType.UPDATED, 
                              f"Status changed from {current_status} to {new_status}")
                
                # Update timestamps based on status
                now = datetime.utcnow()
                if new_status == OrderStatus.SUBMITTED:
                    self.submitted_at = now
                elif new_status == OrderStatus.FILLED:
                    self.filled_at = now
                elif new_status == OrderStatus.CANCELED:
                    self.canceled_at = now
                elif new_status == OrderStatus.REJECTED:
                    self.rejected_at = now
        
        return new_status
    
    @validates('price', 'stop_price', 'quantity', 'filled_quantity')
    def validate_numeric_values(self, key, value):
        """Ensures numeric values are Decimal and positive when required."""
        if value is not None:
            value = Decimal(str(value))
            
            # Quantity and filled_quantity must be positive
            if key in ['quantity', 'filled_quantity'] and value < 0:
                raise ValueError(f"{key} must be positive")
        
        return value
    
    @hybrid_property
    def is_active(self):
        """Returns True if the order is still active (not in a terminal state)."""
        active_statuses = [
            OrderStatus.CREATED, 
            OrderStatus.PENDING_SUBMIT,
            OrderStatus.SUBMITTED, 
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.PENDING_CANCEL
        ]
        return self.status in active_statuses
    
    @hybrid_property
    def tags_list(self):
        """Returns the tags as a list."""
        if not self.tags:
            return []
        return json.loads(self.tags)
    
    @tags_list.setter
    def tags_list(self, tag_list):
        """Sets the tags from a list."""
        if tag_list:
            self.tags = json.dumps(tag_list)
        else:
            self.tags = None
    
    @hybrid_property
    def fill_percent(self):
        """Calculate the percentage of the order that has been filled."""
        if not self.quantity or self.quantity == 0:
            return Decimal('0')
        return (self.filled_quantity / self.quantity) * 100
    
    @hybrid_property
    def value(self):
        """Calculate the total value of the order based on price and quantity."""
        if self.order_type in [OrderType.MARKET, OrderType.STOP]:
            # For market orders, use average fill price if available
            price_to_use = self.average_fill_price
            if not price_to_use and self.is_active:
                # For active orders without fills, we can't determine exact value
                return None
        else:
            # For limit orders, use the limit price
            price_to_use = self.price
            
        if not price_to_use:
            return None
            
        return price_to_use * self.quantity
    
    def add_event(self, event_type, description=None, data=None):
        """
        Adds an event to the order's audit trail.
        
        Args:
            event_type: Type of event
            description: Optional description
            data: Optional JSON-serializable data
        """
        event = OrderEvent(
            order_id=self.order_id,
            event_type=event_type,
            description=description,
            event_data=json.dumps(data) if data else None
        )
        self.order_events.append(event)
        return event
    
    def submit(self):
        """
        Mark the order as being submitted to the broker.
        
        This updates the status and triggers any necessary validations.
        """
        if not self.risk_check_passed:
            raise ValueError("Cannot submit order that has not passed risk checks")
            
        self.status = OrderStatus.PENDING_SUBMIT
        self.add_event(OrderEventType.SUBMITTED, "Order submitted to broker")
    
    def cancel(self):
        """
        Request cancellation of the order.
        
        This updates the status if cancellation is possible.
        """
        if self.status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.ERROR]:
            raise ValueError(f"Cannot cancel order in {self.status} status")
            
        self.status = OrderStatus.PENDING_CANCEL
        self.add_event(OrderEventType.UPDATED, "Cancellation requested")
    
    def add_execution(self, quantity, price, timestamp=None, execution_id=None, fees=None):
        """
        Adds an execution (fill) to the order and updates fill statistics.
        
        Args:
            quantity: Quantity executed
            price: Execution price
            timestamp: Execution timestamp (default: now)
            execution_id: External execution ID
            fees: Execution fees
            
        Returns:
            The created Execution object
        """
        if not timestamp:
            timestamp = datetime.utcnow()
            
        # Validate the execution
        if quantity <= 0:
            raise ValueError("Execution quantity must be positive")
            
        if self.filled_quantity + quantity > self.quantity:
            raise ValueError("Execution would exceed order quantity")
        
        # Create the execution record
        execution = Execution(
            order_id=self.order_id,
            quantity=quantity,
            price=price,
            execution_id=execution_id,
            fees=fees,
            executed_at=timestamp
        )
        self.executions.append(execution)
        
        # Update order fill statistics
        old_filled = self.filled_quantity or Decimal('0')
        new_filled = old_filled + quantity
        
        # Calculate new average price
        if old_filled > 0:
            # Weighted average of old and new fills
            self.average_fill_price = (
                (old_filled * self.average_fill_price) + (quantity * price)
            ) / new_filled
        else:
            # First fill, just use the execution price
            self.average_fill_price = price
        
        self.filled_quantity = new_filled
        self.remaining_quantity = self.quantity - new_filled
        
        # Update order status
        old_status = self.status
        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
        elif old_status not in [OrderStatus.PARTIALLY_FILLED, OrderStatus.PENDING_CANCEL]:
            self.status = OrderStatus.PARTIALLY_FILLED
        
        # Add event
        self.add_event(
            OrderEventType.PARTIALLY_FILLED if self.status == OrderStatus.PARTIALLY_FILLED else OrderEventType.FILLED,
            f"Executed {quantity} at {price}",
            {"execution_id": execution_id, "fees": str(fees) if fees else None}
        )
        
        return execution
    
    def create_child_order(self, **kwargs):
        """
        Creates a child order linked to this parent order.
        
        Args:
            **kwargs: Order parameters
            
        Returns:
            The created Order object
        """
        # Inherit certain properties from parent if not specified
        for field in ['symbol', 'account_id', 'broker', 'venue', 'strategy_id']:
            if field not in kwargs and hasattr(self, field) and getattr(self, field) is not None:
                kwargs[field] = getattr(self, field)
        
        # Create the child order
        child_order = Order(parent_order_id=self.order_id, **kwargs)
        return child_order
    
    def __repr__(self):
        return (f"<Order(id={self.id}, order_id={self.order_id}, symbol={self.symbol}, "
                f"side={self.side}, type={self.order_type}, status={self.status})>")


class Execution(Base):
    """
    Order execution/fill model.
    
    Represents a single execution (full or partial fill) of an order.
    """
    __tablename__ = "executions"
    
    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(String(36), ForeignKey("orders.order_id", ondelete="CASCADE"), nullable=False)
    execution_id = Column(String(50))  # External execution ID from broker
    
    # Execution details
    quantity = Column(Numeric(precision=18, scale=8), nullable=False)
    price = Column(Numeric(precision=18, scale=8), nullable=False)
    fees = Column(Numeric(precision=18, scale=8), default=0)
    
    # Execution metadata
    venue = Column(String(50))  # Execution venue
    liquidity = Column(String(10))  # Maker/taker liquidity designation
    route = Column(String(50))  # Routing information
    
    # Timestamps
    executed_at = Column(DateTime(timezone=True), nullable=False)
    recorded_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    order = relationship("Order", back_populates="executions")
    
    # Indices
    __table_args__ = (
        Index("ix_executions_order_id", order_id),
        Index("ix_executions_executed_at", executed_at),
        CheckConstraint("quantity > 0", name="ck_execution_quantity_positive"),
    )
    
    @hybrid_property
    def value(self):
        """Calculate the total value of this execution."""
        return self.price * self.quantity
    
    @hybrid_property
    def net_value(self):
        """Calculate the total value including fees."""
        return self.value - (self.fees or 0)
    
    def __repr__(self):
        return (f"<Execution(id={self.id}, order_id={self.order_id}, "
                f"quantity={self.quantity}, price={self.price})>")


class ActivePosition(Base):
    """
    Trading position model.
    
    Represents a current position (holding) in a financial instrument.
    """
    __tablename__ = "ActivePositions"
    
    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey("accounts.id", ondelete="CASCADE"), nullable=False)
    symbol = Column(String(20), nullable=False, index=True)
    
    # Position details
    direction = Column(SQLAEnum(PositionDirection), nullable=False)
    quantity = Column(Numeric(precision=18, scale=8), nullable=False)
    average_entry_price = Column(Numeric(precision=18, scale=8), nullable=False)
    current_price = Column(Numeric(precision=18, scale=8), nullable=False)
    
    # P&L tracking
    realized_pnl = Column(Numeric(precision=18, scale=8), default=0, nullable=False)
    unrealized_pnl = Column(Numeric(precision=18, scale=8), default=0, nullable=False)
    last_pnl_update = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    
    # Risk management
    stop_loss_price = Column(Numeric(precision=18, scale=8))
    take_profit_price = Column(Numeric(precision=18, scale=8))
    trailing_stop_price = Column(Numeric(precision=18, scale=8))
    trailing_stop_distance = Column(Numeric(precision=18, scale=8))  # Absolute price distance
    trailing_stop_percent = Column(Numeric(precision=10, scale=2))  # Percentage distance
    trailing_stop_activation_price = Column(Numeric(precision=18, scale=8))
    
    # Position metadata
    opened_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_trade_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    strategy_id = Column(String(50))  # Strategy that created/manages this position
    
    # Hedging and grouping
    is_hedged = Column(Boolean, default=False)
    hedge_id = Column(String(50))  # Links related hedged positions
    
    # Relationships
    account = relationship("Account", back_populates="positions")
    
    # Indices and constraints
    __table_args__ = (
        UniqueConstraint("account_id", "symbol", name="uq_position_account_symbol"),
        Index("ix_positions_account_id_symbol", account_id, symbol),
        Index("ix_positions_strategy_id", strategy_id),
        CheckConstraint("quantity > 0", name="ck_position_quantity_positive"),
    )
    
    @validates('direction')
    def validate_direction(self, key, direction):
        """Validate position direction."""
        if direction not in [PositionDirection.LONG, PositionDirection.SHORT]:
            raise ValueError(f"Direction must be one of: long, short")
        return direction
    
    @validates('quantity')
    def validate_quantity(self, key, quantity):
        """Validate position quantity."""
        quantity = Decimal(str(quantity))
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        return quantity
    
    @hybrid_property
    def market_value(self):
        """
        Calculate the current market value of the position.
        
        For long positions: quantity * price
        For short positions: quantity * price (negative)
        """
        value = self.quantity * self.current_price
        return value if self.direction == PositionDirection.LONG else -value
    
    @hybrid_property
    def cost_basis(self):
        """Calculate the original cost of the position."""
        value = self.quantity * self.average_entry_price
        return value if self.direction == PositionDirection.LONG else -value
    
    @hybrid_property
    def pnl_percentage(self):
        """
        Calculate percentage profit/loss of position.
        
        Takes direction into account.
        """
        if self.average_entry_price == 0:
            return 0
            
        if self.direction == PositionDirection.LONG:
            return float((self.current_price - self.average_entry_price) / self.average_entry_price * 100)
        else:  # Short position
            return float((self.average_entry_price - self.current_price) / self.average_entry_price * 100)
    
    @hybrid_property
    def total_pnl(self):
        """Get the combined realized and unrealized P&L."""
        return self.realized_pnl + self.unrealized_pnl
    
    def update_price(self, new_price):
        """
        Update the current price and recalculate P&L.
        
        Args:
            new_price: New market price
        """
        old_price = self.current_price
        self.current_price = Decimal(str(new_price))
        
        # Update unrealized P&L
        if self.direction == PositionDirection.LONG:
            self.unrealized_pnl = (self.current_price - self.average_entry_price) * self.quantity
        else:  # Short position
            self.unrealized_pnl = (self.average_entry_price - self.current_price) * self.quantity
            
        # Update the last P&L update timestamp
        self.last_pnl_update = datetime.utcnow()
        
        # Handle trailing stop updates
        self._update_trailing_stop(old_price, new_price)
        
        return self.unrealized_pnl
    
    def _update_trailing_stop(self, old_price, new_price):
        """
        Update trailing stop price if conditions are met.
        
        Args:
            old_price: Previous price
            new_price: Current price
        """
        # Only update if trailing stop is configured
        if not (self.trailing_stop_distance or self.trailing_stop_percent):
            return
            
        # Check if trailing stop is activated
        if self.trailing_stop_activation_price:
            if (self.direction == PositionDirection.LONG and 
                new_price >= self.trailing_stop_activation_price):
                # Calculate new stop loss for long position
                if self.trailing_stop_distance:
                    new_stop = new_price - self.trailing_stop_distance
                else:
                    new_stop = new_price * (1 - self.trailing_stop_percent / 100)
                
                # Only update if the new stop is higher than the current one
                if not self.trailing_stop_price or new_stop > self.trailing_stop_price:
                    self.trailing_stop_price = new_stop
            
            elif (self.direction == PositionDirection.SHORT and 
                  new_price <= self.trailing_stop_activation_price):
                # Calculate new stop loss for short position
                if self.trailing_stop_distance:
                    new_stop = new_price + self.trailing_stop_distance
                else:
                    new_stop = new_price * (1 + self.trailing_stop_percent / 100)
                
                # Only update if the new stop is lower than the current one
                if not self.trailing_stop_price or new_stop < self.trailing_stop_price:
                    self.trailing_stop_price = new_stop
    
    def would_hit_stop_loss(self, price):
        """
        Check if a given price would trigger the stop loss.
        
        Args:
            price: Price to check
            
        Returns:
            True if the price would trigger stop loss, False otherwise
        """
        if not self.stop_loss_price:
            return False
            
        if self.direction == PositionDirection.LONG:
            return price <= self.stop_loss_price
        else:  # Short position
            return price >= self.stop_loss_price
            
    def would_hit_take_profit(self, price):
        """
        Check if a given price would trigger the take profit.
        
        Args:
            price: Price to check
            
        Returns:
            True if the price would trigger take profit, False otherwise
        """
        if not self.take_profit_price:
            return False
            
        if self.direction == PositionDirection.LONG:
            return price >= self.take_profit_price
        else:  # Short position
            return price <= self.take_profit_price
    
    def would_hit_trailing_stop(self, price):
        """
        Check if a given price would trigger the trailing stop.
        
        Args:
            price: Price to check
            
        Returns:
            True if the price would trigger the trailing stop, False otherwise
        """
        if not self.trailing_stop_price:
            return False
            
        if self.direction == PositionDirection.LONG:
            return price <= self.trailing_stop_price
        else:  # Short position
            return price >= self.trailing_stop_price
    
    def add_quantity(self, quantity, price):
        """
        Add to position quantity and recalculate average entry price.
        
        Args:
            quantity: Quantity to add
            price: Price of the new quantity
        """
        if quantity <= 0:
            raise ValueError("Quantity to add must be positive")
            
        # Calculate new average entry price
        old_quantity = self.quantity
        old_price = self.average_entry_price
        new_quantity = old_quantity + quantity
        
        # Weighted average calculation
        self.average_entry_price = (old_quantity * old_price + quantity * price) / new_quantity
        self.quantity = new_quantity
        self.last_trade_at = datetime.utcnow()
        
        # Update P&L
        self.update_price(self.current_price)
        
        return self.quantity
    
    def reduce_quantity(self, quantity, price):
        """
        Reduce position quantity and calculate realized P&L.
        
        Args:
            quantity: Quantity to reduce
            price: Price at which reduction occurs
            
        Returns:
            Realized P&L from this reduction
        """
        if quantity <= 0:
            raise ValueError("Quantity to reduce must be positive")
            
        if quantity > self.quantity:
            raise ValueError(f"Cannot reduce by more than current quantity ({self.quantity})")
            
        # Calculate realized P&L
        if self.direction == PositionDirection.LONG:
            realized_pnl = (price - self.average_entry_price) * quantity
        else:  # Short position
            realized_pnl = (self.average_entry_price - price) * quantity
            
        # Update position
        self.quantity -= quantity
        self.realized_pnl += realized_pnl
        self.last_trade_at = datetime.utcnow()
        
        # If position is fully closed, mark for deletion
        if self.quantity == 0:
            # This position will typically be deleted by the caller
            pass
        else:
            # Update P&L (average_entry_price remains unchanged)
            self.update_price(self.current_price)
        
        return realized_pnl
    
    def set_stop_loss(self, price=None, percent=None):
        """
        Set a stop loss for this position.
        
        Args:
            price: Specific price level for stop loss
            percent: Percentage below/above entry price for stop loss
        """
        if price is not None:
            self.stop_loss_price = Decimal(str(price))
        elif percent is not None:
            percent = Decimal(str(percent))
            if self.direction == PositionDirection.LONG:
                self.stop_loss_price = self.average_entry_price * (1 - percent / 100)
            else:  # Short position
                self.stop_loss_price = self.average_entry_price * (1 + percent / 100)
        else:
            raise ValueError("Either price or percent must be specified")
    
    def set_take_profit(self, price=None, percent=None):
        """
        Set a take profit for this position.
        
        Args:
            price: Specific price level for take profit
            percent: Percentage above/below entry price for take profit
        """
        if price is not None:
            self.take_profit_price = Decimal(str(price))
        elif percent is not None:
            percent = Decimal(str(percent))
            if self.direction == PositionDirection.LONG:
                self.take_profit_price = self.average_entry_price * (1 + percent / 100)
            else:  # Short position
                self.take_profit_price = self.average_entry_price * (1 - percent / 100)
        else:
            raise ValueError("Either price or percent must be specified")
    
    def set_trailing_stop(self, distance=None, percent=None, activation_percent=None):
        """
        Set a trailing stop for this position.
        
        Args:
            distance: Absolute price distance for trailing stop
            percent: Percentage distance for trailing stop
            activation_percent: Percentage profit at which to activate the trailing stop
        """
        if not (distance or percent):
            raise ValueError("Either distance or percent must be specified")
            
        if distance:
            self.trailing_stop_distance = Decimal(str(distance))
            self.trailing_stop_percent = None
        else:
            self.trailing_stop_percent = Decimal(str(percent))
            self.trailing_stop_distance = None
        
        # Calculate activation price if provided
        if activation_percent:
            activation_percent = Decimal(str(activation_percent))
            if self.direction == PositionDirection.LONG:
                self.trailing_stop_activation_price = self.average_entry_price * (1 + activation_percent / 100)
            else:  # Short position
                self.trailing_stop_activation_price = self.average_entry_price * (1 - activation_percent / 100)
                
            # Initialize trailing stop price (will be updated once activated)
            if self.direction == PositionDirection.LONG:
                if self.trailing_stop_distance:
                    self.trailing_stop_price = self.trailing_stop_activation_price - self.trailing_stop_distance
                else:
                    self.trailing_stop_price = self.trailing_stop_activation_price * (1 - self.trailing_stop_percent / 100)
            else:  # Short position
                if self.trailing_stop_distance:
                    self.trailing_stop_price = self.trailing_stop_activation_price + self.trailing_stop_distance
                else:
                    self.trailing_stop_price = self.trailing_stop_activation_price * (1 + self.trailing_stop_percent / 100)
        else:
            # Activate immediately based on current price
            self.trailing_stop_activation_price = self.current_price
            
            if self.direction == PositionDirection.LONG:
                if self.trailing_stop_distance:
                    self.trailing_stop_price = self.current_price - self.trailing_stop_distance
                else:
                    self.trailing_stop_price = self.current_price * (1 - self.trailing_stop_percent / 100)
            else:  # Short position
                if self.trailing_stop_distance:
                    self.trailing_stop_price = self.current_price + self.trailing_stop_distance
                else:
                    self.trailing_stop_price = self.current_price * (1 + self.trailing_stop_percent / 100)
    
    def __repr__(self):
        return (f"<Position(id={self.id}, account_id={self.account_id}, "
                f"symbol={self.symbol}, direction={self.direction}, "
                f"quantity={self.quantity}, entry={self.average_entry_price}, "
                f"current={self.current_price})>")


class OrderEvent(Base):
    """
    Order event model for audit trail.
    
    Tracks every significant event in an order's lifecycle for
    compliance, debugging, and analytics.
    """
    __tablename__ = "order_events"
    
    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(String(36), ForeignKey("orders.order_id", ondelete="CASCADE"), nullable=False)
    event_type = Column(SQLAEnum(OrderEventType), nullable=False)
    description = Column(Text)
    event_data = Column(Text)  # JSON data
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    order = relationship("Order", back_populates="order_events")
    
    # Indices
    __table_args__ = (
        Index("ix_order_events_order_id", order_id),
        Index("ix_order_events_created_at", created_at),
        Index("ix_order_events_type", event_type),
    )
    
    @hybrid_property
    def data(self):
        """Parse the event_data JSON."""
        if not self.event_data:
            return None
        return json.loads(self.event_data)
    
    def __repr__(self):
        return (f"<OrderEvent(id={self.id}, order_id={self.order_id}, "
                f"type={self.event_type}, created_at={self.created_at})>")


class Trade(Base):
    """
    Completed trade model.
    
    Represents a completed trade for accounting/reporting purposes.
    Trades are derived from order executions and provide a historical
    record of all trading activity.
    """
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True, index=True)
    trade_id = Column(String(36), default=lambda: str(uuid4()), unique=True, nullable=False)
    account_id = Column(Integer, ForeignKey("accounts.id", ondelete="CASCADE"), nullable=False)
    
    # Trade details
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(SQLAEnum(OrderSide), nullable=False)
    quantity = Column(Numeric(precision=18, scale=8), nullable=False)
    price = Column(Numeric(precision=18, scale=8), nullable=False)
    value = Column(Numeric(precision=18, scale=8), nullable=False)  # price * quantity
    
    # Fees and costs
    fees = Column(Numeric(precision=18, scale=8), default=0)
    total_cost = Column(Numeric(precision=18, scale=8), nullable=False)  # value + fees
    
    # Trade results
    realized_pnl = Column(Numeric(precision=18, scale=8))
    
    # Trade metadata
    order_id = Column(String(36), ForeignKey("orders.order_id"), nullable=False)
    execution_id = Column(String(50))  # Reference to execution
    strategy_id = Column(String(50))
    notes = Column(Text)
    
    # Tax and regulatory information
    tax_lot_id = Column(String(50))  # For tax lot accounting
    wash_sale = Column(Boolean, default=False)  # Flag for wash sales
    
    # Timestamps
    executed_at = Column(DateTime(timezone=True), nullable=False)
    recorded_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    account = relationship("Account", back_populates="trades")
    
    # Indices
    __table_args__ = (
        Index("ix_trades_account_id", account_id),
        Index("ix_trades_symbol", symbol),
        Index("ix_trades_executed_at", executed_at),
        Index("ix_trades_strategy_id", strategy_id),
        CheckConstraint("quantity > 0", name="ck_trade_quantity_positive"),
    )
    
    def __repr__(self):
        return (f"<Trade(id={self.id}, trade_id={self.trade_id}, "
                f"symbol={self.symbol}, side={self.side}, "
                f"quantity={self.quantity}, price={self.price})>")


class BracketOrder(Base):
    """
    Bracket order model.
    
    Represents a bracket order consisting of an entry order,
    stop loss order, and take profit order.
    """
    __tablename__ = "bracket_orders"
    
    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey("accounts.id", ondelete="CASCADE"), nullable=False)
    
    # Component orders
    entry_order_id = Column(String(36), ForeignKey("orders.order_id", ondelete="CASCADE"), nullable=False)
    stop_loss_order_id = Column(String(36), ForeignKey("orders.order_id", ondelete="SET NULL"))
    take_profit_order_id = Column(String(36), ForeignKey("orders.order_id", ondelete="SET NULL"))
    
    # Bracket metadata
    symbol = Column(String(20), nullable=False, index=True)
    status = Column(String(20), nullable=False, default="active")  # active, completed, canceled
    strategy_id = Column(String(50))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    account = relationship("Account")
    entry_order = relationship("Order", foreign_keys=[entry_order_id])
    stop_loss_order = relationship("Order", foreign_keys=[stop_loss_order_id])
    take_profit_order = relationship("Order", foreign_keys=[take_profit_order_id])
    
    # Indices
    __table_args__ = (
        Index("ix_bracket_orders_account_id", account_id),
        Index("ix_bracket_orders_symbol", symbol),
        Index("ix_bracket_orders_status", status),
    )
    
    def cancel(self):
        """Cancel all active orders in the bracket."""
        # Track which orders were canceled
        canceled = []
        
        # Try to cancel each order if active
        for order_attr, order_obj in [
            ('entry_order', self.entry_order),
            ('stop_loss_order', self.stop_loss_order),
            ('take_profit_order', self.take_profit_order)
        ]:
            if order_obj and order_obj.is_active:
                try:
                    order_obj.cancel()
                    canceled.append(order_attr)
                except ValueError as e:
                    logger.warning(f"Failed to cancel {order_attr} for bracket {self.id}: {e}")
        
        # If any orders were canceled, update the bracket status
        if canceled:
            if self.entry_order.status in [OrderStatus.CANCELED, OrderStatus.REJECTED]:
                # If entry order is canceled/rejected, the bracket is effectively canceled
                self.status = "canceled"
            elif self.entry_order.status == OrderStatus.FILLED:
                # If entry is filled but SL/TP were canceled, bracket is incomplete
                self.status = "incomplete"
        
        return canceled
    
    def __repr__(self):
        return (f"<BracketOrder(id={self.id}, symbol={self.symbol}, "
                f"status={self.status})>")


# Add relationship to Account model
Account.orders = relationship("Order", back_populates="account", cascade="all, delete-orphan")
Account.trades = relationship("Trade", back_populates="account", cascade="all, delete-orphan")

# Event listeners and hooks
@event.listens_for(Position, 'before_update')
def position_before_update(mapper, connection, target):
    """
    Check for stop loss or take profit triggers before position updates.
    
    This allows for immediate detection of price levels that would trigger
    stop loss or take profit orders.
    """
    if hasattr(target, '_sa_instance_state') and not target._sa_instance_state.modified:
        return
        
    # Check if current_price has been updated
    if 'current_price' in target._sa_instance_state.committed_state:
        old_price = target._sa_instance_state.committed_state['current_price']
        new_price = target.current_price
        
        # Check for stop loss hit
        if target.stop_loss_price and target.would_hit_stop_loss(new_price) and not target.would_hit_stop_loss(old_price):
            # Log the stop loss hit
            logger.info(f"Stop loss hit for position {target.id} ({target.symbol}) "
                       f"at price {new_price}")
            
        # Check for take profit hit
        if target.take_profit_price and target.would_hit_take_profit(new_price) and not target.would_hit_take_profit(old_price):
            # Log the take profit hit
            logger.info(f"Take profit hit for position {target.id} ({target.symbol}) "
                       f"at price {new_price}")
            
        # Check for trailing stop hit
        if target.trailing_stop_price and target.would_hit_trailing_stop(new_price) and not target.would_hit_trailing_stop(old_price):
            # Log the trailing stop hit
            logger.info(f"Trailing stop hit for position {target.id} ({target.symbol}) "
                       f"at price {new_price}")


# Utility functions for order and trade operations

def create_market_order(session, account_id, symbol, side, quantity, **kwargs):
    """
    Create a market order.
    
    Args:
        session: SQLAlchemy session
        account_id: Account ID
        symbol: Trading symbol
        side: Order side (buy/sell)
        quantity: Order quantity
        **kwargs: Additional order parameters
        
    Returns:
        The created Order object
    """
    order = Order(
        account_id=account_id,
        symbol=symbol,
        side=side,
        order_type=OrderType.MARKET,
        quantity=quantity,
        **kwargs
    )
    
    # Add creation event
    order.add_event(OrderEventType.CREATED, f"Market order created for {quantity} {symbol}")
    
    session.add(order)
    return order


def create_limit_order(session, account_id, symbol, side, quantity, price, **kwargs):
    """
    Create a limit order.
    
    Args:
        session: SQLAlchemy session
        account_id: Account ID
        symbol: Trading symbol
        side: Order side (buy/sell)
        quantity: Order quantity
        price: Limit price
        **kwargs: Additional order parameters
        
    Returns:
        The created Order object
    """
    order = Order(
        account_id=account_id,
        symbol=symbol,
        side=side,
        order_type=OrderType.LIMIT,
        quantity=quantity,
        price=price,
        **kwargs
    )
    
    # Add creation event
    order.add_event(OrderEventType.CREATED, f"Limit order created for {quantity} {symbol} at {price}")
    
    session.add(order)
    return order


def create_stop_order(session, account_id, symbol, side, quantity, stop_price, **kwargs):
    """
    Create a stop order.
    
    Args:
        session: SQLAlchemy session
        account_id: Account ID
        symbol: Trading symbol
        side: Order side (buy/sell)
        quantity: Order quantity
        stop_price: Stop price
        **kwargs: Additional order parameters
        
    Returns:
        The created Order object
    """
    order = Order(
        account_id=account_id,
        symbol=symbol,
        side=side,
        order_type=OrderType.STOP,
        quantity=quantity,
        stop_price=stop_price,
        **kwargs
    )
    
    # Add creation event
    order.add_event(OrderEventType.CREATED, f"Stop order created for {quantity} {symbol} at {stop_price}")
    
    session.add(order)
    return order


def create_stop_limit_order(session, account_id, symbol, side, quantity, stop_price, limit_price, **kwargs):
    """
    Create a stop-limit order.
    
    Args:
        session: SQLAlchemy session
        account_id: Account ID
        symbol: Trading symbol
        side: Order side (buy/sell)
        quantity: Order quantity
        stop_price: Stop price
        limit_price: Limit price
        **kwargs: Additional order parameters
        
    Returns:
        The created Order object
    """
    order = Order(
        account_id=account_id,
        symbol=symbol,
        side=side,
        order_type=OrderType.STOP_LIMIT,
        quantity=quantity,
        stop_price=stop_price,
        price=limit_price,
        **kwargs
    )
    
    # Add creation event
    order.add_event(
        OrderEventType.CREATED, 
        f"Stop-limit order created for {quantity} {symbol} at stop {stop_price}, limit {limit_price}"
    )
    
    session.add(order)
    return order


def create_bracket_order(
    session, account_id, symbol, side, quantity, 
    entry_price=None, entry_type=OrderType.MARKET,
    stop_loss_price=None, take_profit_price=None, 
    **kwargs
):
    """
    Create a bracket order consisting of an entry order with optional
    stop loss and take profit orders.
    
    Args:
        session: SQLAlchemy session
        account_id: Account ID
        symbol: Trading symbol
        side: Order side (buy/sell)
        quantity: Order quantity
        entry_price: Price for limit entry (None for market)
        entry_type: Type of entry order (MARKET or LIMIT)
        stop_loss_price: Stop loss price
        take_profit_price: Take profit price
        **kwargs: Additional parameters for all orders
        
    Returns:
        BracketOrder object containing the three component orders
    """
    # Validate parameters
    if entry_type == OrderType.LIMIT and entry_price is None:
        raise ValueError("Limit entry order requires a price")
        
    # Create the entry order
    if entry_type == OrderType.MARKET:
        entry_order = create_market_order(
            session, account_id, symbol, side, quantity, **kwargs
        )
    else:
        entry_order = create_limit_order(
            session, account_id, symbol, side, quantity, entry_price, **kwargs
        )
    
    # Create child orders (these will be activated when entry fills)
    stop_loss_order = None
    take_profit_order = None
    
    # Determine opposite side for exit orders
    exit_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY
    
    # Create stop loss order if price provided
    if stop_loss_price is not None:
        stop_loss_order = create_stop_order(
            session, account_id, symbol, exit_side, quantity, stop_loss_price,
            parent_order_id=entry_order.order_id,
            status=OrderStatus.CREATED,  # Will be submitted when entry fills
            **kwargs
        )
    
    # Create take profit order if price provided
    if take_profit_price is not None:
        take_profit_order = create_limit_order(
            session, account_id, symbol, exit_side, quantity, take_profit_price,
            parent_order_id=entry_order.order_id,
            status=OrderStatus.CREATED,  # Will be submitted when entry fills
            **kwargs
        )
    
    # Create the bracket order to link them
    bracket = BracketOrder(
        account_id=account_id,
        entry_order_id=entry_order.order_id,
        stop_loss_order_id=stop_loss_order.order_id if stop_loss_order else None,
        take_profit_order_id=take_profit_order.order_id if take_profit_order else None,
        symbol=symbol,
        strategy_id=kwargs.get('strategy_id')
    )
    session.add(bracket)
    
    return bracket


def update_position_from_execution(session, account_id, symbol, execution, direction=None):
    """
    Update a position based on an execution/fill.
    
    Args:
        session: SQLAlchemy session
        account_id: Account ID
        symbol: Trading symbol
        execution: Execution object
        direction: Override direction (for opening positions)
        
    Returns:
        Updated or created Position object
    """
    # Get the linked order to determine direction
    order = execution.order
    
    # Get current position if exists
    position = session.query(Position).filter(
        Position.account_id == account_id,
        Position.symbol == symbol
    ).first()
    
    # Determine direction for new positions
    if not position:
        # New position - determine direction
        if direction:
            pos_direction = direction
        else:
            # Default mapping of order side to position direction
            pos_direction = (PositionDirection.LONG if order.side == OrderSide.BUY 
                             else PositionDirection.SHORT)
            
        # Create new position
        position = Position(
            account_id=account_id,
            symbol=symbol,
            direction=pos_direction,
            quantity=execution.quantity,
            average_entry_price=execution.price,
            current_price=execution.price,
            strategy_id=order.strategy_id
        )
        session.add(position)
        
    else:
        # Existing position - handle based on order side and position direction
        if ((order.side == OrderSide.BUY and position.direction == PositionDirection.LONG) or
            (order.side == OrderSide.SELL and position.direction == PositionDirection.SHORT)):
            # Adding to position
            position.add_quantity(execution.quantity, execution.price)
            
        elif ((order.side == OrderSide.SELL and position.direction == PositionDirection.LONG) or
              (order.side == OrderSide.BUY and position.direction == PositionDirection.SHORT)):
            # Reducing/closing position
            if execution.quantity > position.quantity:
                raise ValueError(f"Execution quantity ({execution.quantity}) exceeds position quantity ({position.quantity})")
                
            position.reduce_quantity(execution.quantity, execution.price)
            
            # If position is closed (quantity=0), mark for removal
            if position.quantity == 0:
                session.delete(position)
                position = None
                
    # Create trade record
    trade = Trade(
        account_id=account_id,
        trade_id=str(uuid4()),
        symbol=symbol,
        side=order.side,
        quantity=execution.quantity,
        price=execution.price,
        value=execution.quantity * execution.price,
        fees=execution.fees or 0,
        total_cost=(execution.quantity * execution.price) + (execution.fees or 0),
        order_id=order.order_id,
        execution_id=execution.execution_id,
        strategy_id=order.strategy_id,
        executed_at=execution.executed_at
    )
    session.add(trade)
    
    return position, trade