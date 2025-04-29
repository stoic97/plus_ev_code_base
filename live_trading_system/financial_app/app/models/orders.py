"""
Order models for the trading application.

This module defines SQLAlchemy models for order management, execution tracking,
and trade lifecycle handling for both option strategies and AI-based prediction trading.

Key features:
- Comprehensive order state tracking
- Support for multi-leg option orders
- Integration with AI prediction metadata
- Full audit trail of order lifecycle events
- Flexible execution algorithm configuration
- Advanced order types including conditional orders

Usage:
    These models provide the foundation for order submission, tracking, and execution
    in the intraday trading environment, supporting both automated strategy execution
    and manual order placement.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum, auto
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID, uuid4

from sqlalchemy import (
    Column, Integer, String, Float, Numeric, DateTime, 
    ForeignKey, Enum as SQLAEnum, Text, Boolean, Index, 
    CheckConstraint, UniqueConstraint, Table, JSON,
    PrimaryKeyConstraint, ForeignKeyConstraint, func
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import relationship, validates

from app.core.database import Base


# Enumerations for order-related fields
class OrderType(str, Enum):
    """Types of orders that can be placed."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    ICEBERG = "iceberg"


class OrderSide(str, Enum):
    """Order direction."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    """States in the order lifecycle."""
    CREATED = "created"          # Initial state
    VALIDATED = "validated"      # Passed pre-trade checks
    REJECTED = "rejected"        # Failed validation or pre-trade checks
    PENDING = "pending"          # Sent to broker/exchange but not confirmed
    ACCEPTED = "accepted"        # Accepted by broker/exchange
    PARTIALLY_FILLED = "partially_filled"  # Some quantity executed
    FILLED = "filled"            # Fully executed
    CANCELLED = "cancelled"      # Explicitly cancelled
    EXPIRED = "expired"          # Timed out without execution
    AMENDED = "amended"          # Successfully modified
    ERROR = "error"              # System error during processing


class TimeInForce(str, Enum):
    """How long an order remains active."""
    DAY = "day"                  # Valid for the trading day
    GTC = "gtc"                  # Good Till Cancelled
    IOC = "ioc"                  # Immediate Or Cancel
    FOK = "fok"                  # Fill Or Kill
    GTD = "gtd"                  # Good Till Date


class ExecutionAlgorithm(str, Enum):
    """Algorithms for order execution."""
    DIRECT = "direct"            # Direct market access
    TWAP = "twap"                # Time-Weighted Average Price
    VWAP = "vwap"                # Volume-Weighted Average Price
    POV = "pov"                  # Percentage of Volume
    IS = "implementation_shortfall"  # Implementation Shortfall
    DARK_POOL = "dark_pool"      # Dark Pool access
    SNIPER = "sniper"            # Aggressive liquidity-seeking
    PASSIVE = "passive"          # Passive/patient execution


class OptionType(str, Enum):
    """Option contract types."""
    CALL = "call"
    PUT = "put"


class OrderTriggerType(str, Enum):
    """Types of conditions that can trigger an order."""
    PRICE = "price"              # Based on asset price
    TIME = "time"                # Based on specific time
    SIGNAL = "signal"            # Based on strategy signal
    VOLUME = "volume"            # Based on trading volume
    VOLATILITY = "volatility"    # Based on volatility change
    MANUAL = "manual"            # Manually triggered


class OrderSource(str, Enum):
    """Source of the order."""
    STRATEGY = "strategy"        # From automated strategy
    SIGNAL = "signal"            # From trading signal
    AI_MODEL = "ai_model"        # From AI prediction model
    USER = "user"                # From user interface
    API = "api"                  # From external API
    SYSTEM = "system"            # System-generated (e.g., liquidation)


class OrderPriorityLevel(int, Enum):
    """Priority level for order execution."""
    HIGHEST = 1  # Critical orders (e.g., risk reduction)
    HIGH = 2     # High priority signals
    NORMAL = 3   # Standard priority
    LOW = 4      # Low priority/opportunistic orders
    LOWEST = 5   # Lowest priority


# Association table for order tags
order_tags = Table(
    'order_tags',
    Base.metadata,
    Column('order_id', PGUUID(as_uuid=True), ForeignKey('orders.id', ondelete="CASCADE"), primary_key=True),
    Column('tag_id', Integer, ForeignKey('tags.id', ondelete="CASCADE"), primary_key=True),
    Index('ix_order_tags_tag_id', 'tag_id')
)


class Order(Base):
    """
    Main order model for all trading activities.
    
    Represents a single order with all its parameters, state, and execution details.
    Parent orders can have child orders for complex order types.
    """
    __tablename__ = "orders"
    
    # Basic identification
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    client_order_id = Column(String(50), unique=True, nullable=False, index=True)
    parent_order_id = Column(PGUUID(as_uuid=True), ForeignKey('orders.id', ondelete="SET NULL"), nullable=True)
    
    # Order relationships
    account_id = Column(Integer, ForeignKey("accounts.id", ondelete="CASCADE"), nullable=False)
    strategy_id = Column(PGUUID(as_uuid=True), ForeignKey("strategies.id", ondelete="SET NULL"), nullable=True)
    user_id = Column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    position_id = Column(Integer, ForeignKey("positions.id", ondelete="SET NULL"), nullable=True)
    
    # Order details
    order_type = Column(SQLAEnum(OrderType), nullable=False)
    side = Column(SQLAEnum(OrderSide), nullable=False)
    symbol = Column(String(20), nullable=False, index=True)
    instrument_type = Column(String(20), nullable=False)
    quantity = Column(Numeric(18, 8), nullable=False)
    price = Column(Numeric(18, 8), nullable=True)  # Null for market orders
    stop_price = Column(Numeric(18, 8), nullable=True)  # For stop and stop-limit orders
    avg_execution_price = Column(Numeric(18, 8), nullable=True)  # Average fill price
    filled_quantity = Column(Numeric(18, 8), nullable=False, default=0)
    remaining_quantity = Column(Numeric(18, 8), nullable=False)  # Initially equals quantity
    
    # Order state
    status = Column(SQLAEnum(OrderStatus), nullable=False, default=OrderStatus.CREATED)
    time_in_force = Column(SQLAEnum(TimeInForce), nullable=False, default=TimeInForce.DAY)
    is_active = Column(Boolean, nullable=False, default=True)  # Quick flag for active orders
    
    # Execution parameters
    execution_algorithm = Column(SQLAEnum(ExecutionAlgorithm), nullable=True)
    execution_params = Column(JSON, nullable=True)  # Algorithm-specific parameters
    routing_destination = Column(String(50), nullable=True)  # Exchange or venue
    
    # Option-specific fields
    is_option_order = Column(Boolean, nullable=False, default=False)
    option_type = Column(SQLAEnum(OptionType), nullable=True)
    strike_price = Column(Numeric(18, 8), nullable=True)
    expiration_date = Column(DateTime, nullable=True)
    
    # AI model specific fields
    prediction_id = Column(String(50), nullable=True)  # Reference to AI prediction
    confidence_score = Column(Float, nullable=True)  # AI model confidence (0-1)
    signal_strength = Column(Float, nullable=True)  # Signal strength indicator
    
    # Timing and lifecycle
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    submitted_at = Column(DateTime(timezone=True), nullable=True)
    executed_at = Column(DateTime(timezone=True), nullable=True)
    cancelled_at = Column(DateTime(timezone=True), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)  # For GTD orders
    
    # Advanced order features
    is_algo_order = Column(Boolean, nullable=False, default=False)
    is_multi_leg = Column(Boolean, nullable=False, default=False)
    is_conditional = Column(Boolean, nullable=False, default=False)
    trigger_type = Column(SQLAEnum(OrderTriggerType), nullable=True)
    trigger_details = Column(JSON, nullable=True)  # Details of trigger condition
    
    # Metadata
    source = Column(SQLAEnum(OrderSource), nullable=False, default=OrderSource.USER)
    priority = Column(SQLAEnum(OrderPriorityLevel), nullable=False, default=OrderPriorityLevel.NORMAL)
    notes = Column(Text, nullable=True)
    metadata = Column(JSON, nullable=True)  # Additional flexible metadata
    
    # Error tracking
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, nullable=False, default=0)
    
    # Relationships
    account = relationship("Account", back_populates="orders")
    strategy = relationship("Strategy", back_populates="orders")
    user = relationship("User", back_populates="orders")
    position = relationship("Position", back_populates="orders")
    
    parent_order = relationship("Order", remote_side=[id], backref="child_orders")
    
    order_legs = relationship("OrderLeg", back_populates="parent_order", cascade="all, delete-orphan")
    executions = relationship("Execution", back_populates="order", cascade="all, delete-orphan")
    order_events = relationship("OrderEvent", back_populates="order", cascade="all, delete-orphan")
    order_amendments = relationship("OrderAmendment", back_populates="order", cascade="all, delete-orphan")
    tags = relationship("Tag", secondary=order_tags, backref="orders")
    
    # Constraints and indexes
    __table_args__ = (
        # Indexes for common queries
        Index("ix_orders_account_id_status", account_id, status),
        Index("ix_orders_symbol_side", symbol, side),
        Index("ix_orders_created_at", created_at),
        Index("ix_orders_strategy_id", strategy_id),
        
        # Ensure remaining quantity calculation is correct
        CheckConstraint(
            "remaining_quantity = quantity - filled_quantity",
            name="ck_orders_remaining_quantity"
        ),
        
        # Ensure positive quantity values
        CheckConstraint(
            "quantity > 0 AND filled_quantity >= 0 AND remaining_quantity >= 0",
            name="ck_orders_positive_quantities"
        ),
        
        # Ensure price is not null for limit orders
        CheckConstraint(
            "(order_type != 'limit' AND order_type != 'stop_limit') OR price IS NOT NULL",
            name="ck_orders_limit_price_not_null"
        ),
        
        # Ensure stop price is not null for stop orders
        CheckConstraint(
            "(order_type != 'stop' AND order_type != 'stop_limit' AND order_type != 'trailing_stop') OR stop_price IS NOT NULL",
            name="ck_orders_stop_price_not_null"
        ),
        
        # Ensure option fields are set for option orders
        CheckConstraint(
            "NOT is_option_order OR (option_type IS NOT NULL AND strike_price IS NOT NULL AND expiration_date IS NOT NULL)",
            name="ck_orders_option_fields"
        ),
    )
    
    @validates('client_order_id')
    def validate_client_order_id(self, key, value):
        """Ensure client_order_id meets required format."""
        if not value or len(value) < 8:
            raise ValueError("client_order_id must be at least 8 characters")
        return value
    
    @validates('quantity', 'price', 'stop_price', 'strike_price')
    def validate_numeric_values(self, key, value):
        """Ensure numeric values are positive where required."""
        if value is not None:
            if key == 'quantity' and value <= 0:
                raise ValueError("quantity must be positive")
            elif key in ('price', 'stop_price', 'strike_price') and value <= 0:
                raise ValueError(f"{key} must be positive")
        return value
    
    @hybrid_property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == OrderStatus.FILLED
    
    @hybrid_property
    def is_cancelled(self) -> bool:
        """Check if order is cancelled."""
        return self.status == OrderStatus.CANCELLED
    
    @hybrid_property
    def can_be_amended(self) -> bool:
        """Check if order can be amended."""
        amendable_statuses = [
            OrderStatus.CREATED,
            OrderStatus.VALIDATED,
            OrderStatus.ACCEPTED,
            OrderStatus.PENDING,
            OrderStatus.PARTIALLY_FILLED
        ]
        return self.status in amendable_statuses and self.is_active
    
    @hybrid_property
    def can_be_cancelled(self) -> bool:
        """Check if order can be cancelled."""
        cancellable_statuses = [
            OrderStatus.CREATED,
            OrderStatus.VALIDATED,
            OrderStatus.ACCEPTED,
            OrderStatus.PENDING,
            OrderStatus.PARTIALLY_FILLED
        ]
        return self.status in cancellable_statuses and self.is_active
    
    @hybrid_property
    def execution_progress(self) -> float:
        """Calculate execution progress as percentage."""
        if not self.quantity or self.quantity == 0:
            return 0.0
        return float(self.filled_quantity / self.quantity * 100)
    
    @hybrid_property
    def age_seconds(self) -> int:
        """Calculate age of order in seconds."""
        if not self.created_at:
            return 0
        
        now = datetime.utcnow()
        return int((now - self.created_at).total_seconds())
    
    def update_status(self, new_status: OrderStatus, error_message: Optional[str] = None) -> None:
        """
        Update order status and related fields.
        
        Args:
            new_status: New status to set
            error_message: Optional error message for REJECTED or ERROR statuses
        """
        self.status = new_status
        
        # Update timestamps based on status
        now = datetime.utcnow()
        
        if new_status == OrderStatus.PENDING or new_status == OrderStatus.ACCEPTED:
            if not self.submitted_at:
                self.submitted_at = now
        
        elif new_status == OrderStatus.FILLED:
            self.executed_at = now
            self.is_active = False
        
        elif new_status == OrderStatus.CANCELLED:
            self.cancelled_at = now
            self.is_active = False
        
        elif new_status == OrderStatus.EXPIRED:
            self.is_active = False
        
        elif new_status == OrderStatus.REJECTED or new_status == OrderStatus.ERROR:
            self.is_active = False
            if error_message:
                self.error_message = error_message
    
    def update_fill(self, fill_quantity: Decimal, fill_price: Decimal) -> None:
        """
        Update order with a new fill.
        
        Args:
            fill_quantity: Quantity that was filled
            fill_price: Price at which the fill occurred
        """
        # Calculate current and new values
        current_fill_value = self.filled_quantity * (self.avg_execution_price or Decimal('0'))
        new_fill_value = fill_quantity * fill_price
        total_filled = self.filled_quantity + fill_quantity
        
        # Update average execution price
        if total_filled > 0:
            self.avg_execution_price = (current_fill_value + new_fill_value) / total_filled
        
        # Update quantities
        self.filled_quantity = total_filled
        self.remaining_quantity = self.quantity - total_filled
        
        # Update status
        if self.filled_quantity >= self.quantity:
            self.update_status(OrderStatus.FILLED)
        elif self.filled_quantity > 0:
            self.update_status(OrderStatus.PARTIALLY_FILLED)
    
    def cancel(self, reason: Optional[str] = None) -> bool:
        """
        Cancel the order if possible.
        
        Args:
            reason: Optional reason for cancellation
        
        Returns:
            True if order was cancelled, False if order cannot be cancelled
        """
        if not self.can_be_cancelled:
            return False
        
        self.update_status(OrderStatus.CANCELLED)
        if reason:
            self.notes = (self.notes or "") + f"\nCancelled: {reason}"
        
        return True
    
    def clone(self) -> 'Order':
        """
        Create a clone of this order with a new ID and reset execution state.
        Useful for order resubmission or creating similar orders.
        
        Returns:
            New Order object based on this order
        """
        # Create new order with same parameters but new ID
        new_order = Order(
            client_order_id=f"{self.client_order_id}_clone_{uuid4().hex[:8]}",
            account_id=self.account_id,
            strategy_id=self.strategy_id,
            user_id=self.user_id,
            position_id=self.position_id,
            order_type=self.order_type,
            side=self.side,
            symbol=self.symbol,
            instrument_type=self.instrument_type,
            quantity=self.quantity,
            price=self.price,
            stop_price=self.stop_price,
            time_in_force=self.time_in_force,
            execution_algorithm=self.execution_algorithm,
            execution_params=self.execution_params,
            routing_destination=self.routing_destination,
            is_option_order=self.is_option_order,
            option_type=self.option_type,
            strike_price=self.strike_price,
            expiration_date=self.expiration_date,
            prediction_id=self.prediction_id,
            confidence_score=self.confidence_score,
            signal_strength=self.signal_strength,
            is_algo_order=self.is_algo_order,
            is_multi_leg=self.is_multi_leg,
            is_conditional=self.is_conditional,
            trigger_type=self.trigger_type,
            trigger_details=self.trigger_details,
            source=self.source,
            priority=self.priority,
            notes=f"Cloned from order {self.id}",
            metadata=self.metadata
        )
        
        # Reset execution state
        new_order.status = OrderStatus.CREATED
        new_order.filled_quantity = Decimal('0')
        new_order.remaining_quantity = new_order.quantity
        new_order.avg_execution_price = None
        new_order.is_active = True
        
        return new_order
    
    def __repr__(self):
        return (
            f"<Order {self.id} {self.side.value} {self.quantity} {self.symbol} "
            f"type={self.order_type.value} status={self.status.value}>"
        )


class OrderLeg(Base):
    """
    Component of a multi-leg order such as option spreads.
    
    Each leg represents a separate instrument within a complex order strategy.
    """
    __tablename__ = "order_legs"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    parent_order_id = Column(PGUUID(as_uuid=True), ForeignKey('orders.id', ondelete="CASCADE"), nullable=False)
    
    # Leg details
    leg_number = Column(Integer, nullable=False)  # Order of execution
    side = Column(SQLAEnum(OrderSide), nullable=False)
    symbol = Column(String(20), nullable=False)
    quantity = Column(Numeric(18, 8), nullable=False)
    price = Column(Numeric(18, 8), nullable=True)  # Limit price for this leg
    ratio = Column(Integer, nullable=False, default=1)  # For ratio spreads
    
    # Option-specific fields
    is_option = Column(Boolean, nullable=False, default=False)
    option_type = Column(SQLAEnum(OptionType), nullable=True)
    strike_price = Column(Numeric(18, 8), nullable=True)
    expiration_date = Column(DateTime, nullable=True)
    
    # Execution tracking
    individual_order_id = Column(PGUUID(as_uuid=True), ForeignKey('orders.id', ondelete="SET NULL"), nullable=True)
    status = Column(SQLAEnum(OrderStatus), nullable=False, default=OrderStatus.CREATED)
    filled_quantity = Column(Numeric(18, 8), nullable=False, default=0)
    avg_execution_price = Column(Numeric(18, 8), nullable=True)
    
    # Execution sequence
    must_execute_first = Column(Boolean, nullable=False, default=False)  # This leg must execute first
    simultaneous_execution = Column(Boolean, nullable=False, default=False)  # Execute with other legs
    
    # Relationships
    parent_order = relationship("Order", foreign_keys=[parent_order_id], back_populates="order_legs")
    individual_order = relationship("Order", foreign_keys=[individual_order_id])
    
    # Constraints and indexes
    __table_args__ = (
        # Ensure unique leg numbers within a parent order
        UniqueConstraint("parent_order_id", "leg_number", name="uc_order_legs_order_leg"),
        
        # Index for efficient lookup by parent order
        Index("ix_order_legs_parent_order_id", parent_order_id),
        
        # Ensure option fields are set for option legs
        CheckConstraint(
            "NOT is_option OR (option_type IS NOT NULL AND strike_price IS NOT NULL AND expiration_date IS NOT NULL)",
            name="ck_order_legs_option_fields"
        ),
        
        # Ensure positive quantity
        CheckConstraint(
            "quantity > 0",
            name="ck_order_legs_positive_quantity"
        ),
        
        # Ensure positive ratio
        CheckConstraint(
            "ratio > 0",
            name="ck_order_legs_positive_ratio"
        ),
    )
    
    @validates('quantity', 'price', 'strike_price')
    def validate_numeric_values(self, key, value):
        """Ensure numeric values are positive where required."""
        if value is not None:
            if key == 'quantity' and value <= 0:
                raise ValueError("quantity must be positive")
            elif key in ('price', 'strike_price') and value <= 0:
                raise ValueError(f"{key} must be positive")
        return value


class Execution(Base):
    """
    Record of an execution (fill) for an order.
    
    Each execution represents a trade that occurred as a result of an order.
    """
    __tablename__ = "executions"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    order_id = Column(PGUUID(as_uuid=True), ForeignKey('orders.id', ondelete="CASCADE"), nullable=False)
    
    # Execution details
    execution_id = Column(String(50), nullable=False, unique=True)  # Exchange/broker execution ID
    symbol = Column(String(20), nullable=False)
    side = Column(SQLAEnum(OrderSide), nullable=False)
    quantity = Column(Numeric(18, 8), nullable=False)
    price = Column(Numeric(18, 8), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    venue = Column(String(50), nullable=False)  # Exchange or trading venue
    
    # Fees and costs
    fees = Column(Numeric(18, 8), nullable=False, default=0)
    fee_currency = Column(String(10), nullable=False, default="USD")
    commission = Column(Numeric(18, 8), nullable=False, default=0)
    
    # Additional data
    is_option_execution = Column(Boolean, nullable=False, default=False)
    metadata = Column(JSON, nullable=True)  # Additional execution details
    
    # Relationships
    order = relationship("Order", back_populates="executions")
    
    # Constraints and indexes
    __table_args__ = (
        # Index for efficient lookup by order
        Index("ix_executions_order_id", order_id),
        
        # Index by timestamp for time-based queries
        Index("ix_executions_timestamp", timestamp),
        
        # Ensure positive quantity and price
        CheckConstraint(
            "quantity > 0 AND price > 0",
            name="ck_executions_positive_values"
        ),
    )
    
    @validates('quantity', 'price')
    def validate_numeric_values(self, key, value):
        """Ensure numeric values are positive."""
        if value <= 0:
            raise ValueError(f"{key} must be positive")
        return value
    
    @hybrid_property
    def value(self) -> Decimal:
        """Calculate the execution value."""
        return self.quantity * self.price
    
    @hybrid_property
    def total_cost(self) -> Decimal:
        """Calculate total cost including fees and commission."""
        # For buy orders, add fees/commission; for sell orders, subtract
        if self.side == OrderSide.BUY:
            return self.value + self.fees + self.commission
        else:  # SELL
            return self.value - self.fees - self.commission


class OrderEvent(Base):
    """
    Record of events in an order's lifecycle.
    
    Provides a complete audit trail of all actions taken on an order.
    """
    __tablename__ = "order_events"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    order_id = Column(PGUUID(as_uuid=True), ForeignKey('orders.id', ondelete="CASCADE"), nullable=False)
    
    # Event details
    event_type = Column(String(50), nullable=False)  # Created, Submitted, Amended, Cancelled, etc.
    timestamp = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    user_id = Column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    
    # State information
    previous_status = Column(SQLAEnum(OrderStatus), nullable=True)
    new_status = Column(SQLAEnum(OrderStatus), nullable=True)
    details = Column(JSON, nullable=True)  # Specific details of the event
    
    # Relationships
    order = relationship("Order", back_populates="order_events")
    user = relationship("User")
    
    # Constraints and indexes
    __table_args__ = (
        # Index for efficient lookup by order
        Index("ix_order_events_order_id", order_id),
        
        # Index by timestamp for time-based queries
        Index("ix_order_events_timestamp", timestamp),
        
        # Index for user auditing
        Index("ix_order_events_user_id", user_id),
    )


class OrderAmendment(Base):
    """
    Record of amendments made to an order.
    
    Tracks changes to order parameters for audit and history purposes.
    """
    __tablename__ = "order_amendments"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    order_id = Column(PGUUID(as_uuid=True), ForeignKey('orders.id', ondelete="CASCADE"), nullable=False)
    
    # Amendment details
    timestamp = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    user_id = Column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    
    # Changes made
    field_name = Column(String(50), nullable=False)  # Name of field that was changed
    old_value = Column(Text, nullable=True)  # Previous value (as string)
    new_value = Column(Text, nullable=True)  # New value (as string)
    
    # Success tracking
    was_successful = Column(Boolean, nullable=False, default=True)
    error_message = Column(Text, nullable=True)
    
    # Relationships
    order = relationship("Order", back_populates="order_amendments")
    user = relationship("User")
    
    # Constraints and indexes
    __table_args__ = (
        # Index for efficient lookup by order
        Index("ix_order_amendments_order_id", order_id),
        
        # Index by timestamp for time-based queries
        Index("ix_order_amendments_timestamp", timestamp),
    )


class ConditionalOrder(Base):
    """
    Conditional order configuration.
    
    Defines conditions under which an order should be triggered.
    """
    __tablename__ = "conditional_orders"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    order_id = Column(PGUUID(as_uuid=True), ForeignKey('orders.id', ondelete="CASCADE"), nullable=False, unique=True)
    
    # Condition type
    trigger_type = Column(SQLAEnum(OrderTriggerType), nullable=False)
    
    # Price-based triggers
    symbol = Column(String(20), nullable=True)  # Symbol to monitor (can be different from order)
    price_target = Column(Numeric(18, 8), nullable=True)
    price_comparator = Column(String(2), nullable=True)  # >, <, >=, <=, ==
    
    # Time-based triggers
    trigger_time = Column(DateTime, nullable=True)
    
    # Signal-based triggers
    signal_id = Column(String(50), nullable=True)
    signal_threshold = Column(Float, nullable=True)
    
    # Volume-based triggers
    volume_threshold = Column(Numeric(18, 8), nullable=True)
    volume_period_seconds = Column(Integer, nullable=True)
    
    # Volatility-based triggers
    volatility_target = Column(Float, nullable=True)  # Target volatility level
    volatility_comparator = Column(String(2), nullable=True)  # >, <, >=,
    # Volatility-based triggers
    volatility_target = Column(Float, nullable=True)  # Target volatility level
    volatility_comparator = Column(String(2), nullable=True)  # >, <, >=, <=, ==
    volatility_period_seconds = Column(Integer, nullable=True)  # Period to measure volatility over
    
    # Combined condition logic
    require_all_conditions = Column(Boolean, nullable=False, default=False)  # AND vs OR for multiple conditions
    
    # Trigger status
    is_triggered = Column(Boolean, nullable=False, default=False)
    triggered_at = Column(DateTime(timezone=True), nullable=True)
    last_check_time = Column(DateTime(timezone=True), nullable=True)
    trigger_attempts = Column(Integer, nullable=False, default=0)
    
    # Expiration of condition
    expires_at = Column(DateTime(timezone=True), nullable=True)  # When the condition expires
    
    # Relationships
    order = relationship("Order", back_populates="conditional_trigger")
    
    # Constraints and indexes
    __table_args__ = (
        # Index for quick lookup when checking conditions
        Index("ix_conditional_orders_symbol_not_triggered", symbol, is_triggered),
        Index("ix_conditional_orders_signal_id", signal_id),
        
        # Ensure at least one trigger condition is set based on trigger_type
        CheckConstraint(
            "(trigger_type = 'price' AND symbol IS NOT NULL AND price_target IS NOT NULL AND price_comparator IS NOT NULL) OR "
            "(trigger_type = 'time' AND trigger_time IS NOT NULL) OR "
            "(trigger_type = 'signal' AND signal_id IS NOT NULL) OR "
            "(trigger_type = 'volume' AND symbol IS NOT NULL AND volume_threshold IS NOT NULL AND volume_period_seconds IS NOT NULL) OR "
            "(trigger_type = 'volatility' AND symbol IS NOT NULL AND volatility_target IS NOT NULL AND volatility_comparator IS NOT NULL AND volatility_period_seconds IS NOT NULL) OR "
            "(trigger_type = 'manual')",
            name="ck_conditional_orders_valid_condition"
        ),
        
        # Ensure valid comparator values
        CheckConstraint(
            "price_comparator IS NULL OR price_comparator IN ('>', '<', '>=', '<=', '==')",
            name="ck_conditional_orders_price_comparator"
        ),
        CheckConstraint(
            "volatility_comparator IS NULL OR volatility_comparator IN ('>', '<', '>=', '<=', '==')",
            name="ck_conditional_orders_volatility_comparator"
        )
    )
    
    @validates('price_target', 'volume_threshold')
    def validate_numeric_values(self, key, value):
        """Ensure numeric values are positive where required."""
        if value is not None and value <= 0:
            raise ValueError(f"{key} must be positive")
        return value
    
    @validates('price_comparator', 'volatility_comparator')
    def validate_comparator(self, key, value):
        """Ensure comparator is valid."""
        if value is not None and value not in ('>', '<', '>=', '<=', '=='):
            raise ValueError(f"{key} must be one of: >, <, >=, <=, ==")
        return value
    
    @validates('trigger_type')
    def validate_trigger_type(self, key, value):
        """Set trigger requirements based on type."""
        return value
    
    @hybrid_property
    def is_expired(self) -> bool:
        """Check if the conditional order is expired."""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at
    
    def mark_triggered(self) -> None:
        """Mark the conditional order as triggered."""
        now = datetime.utcnow()
        self.is_triggered = True
        self.triggered_at = now
        self.last_check_time = now
        self.trigger_attempts += 1
    
    def update_check_time(self) -> None:
        """Update the last check time for the conditional order."""
        self.last_check_time = datetime.utcnow()
        self.trigger_attempts += 1
    
    def __repr__(self):
        return f"<ConditionalOrder {self.id} type={self.trigger_type} triggered={self.is_triggered}>"


# Add the missing relationship in Order class
Order.conditional_trigger = relationship("ConditionalOrder", back_populates="order", uselist=False, cascade="all, delete-orphan")


class OrderBatch(Base):
    """
    Batch of related orders that are managed together.
    
    Useful for basket trades, pairs trading, or multi-asset strategies.
    """
    __tablename__ = "order_batches"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Batch identification
    batch_name = Column(String(100), nullable=False)
    batch_type = Column(String(50), nullable=False)  # basket, pairs, multi-leg, etc.
    
    # Creator info
    strategy_id = Column(PGUUID(as_uuid=True), ForeignKey("strategies.id", ondelete="SET NULL"), nullable=True)
    user_id = Column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    account_id = Column(Integer, ForeignKey("accounts.id", ondelete="CASCADE"), nullable=False)
    
    # Timing and lifecycle
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Batch status
    is_active = Column(Boolean, nullable=False, default=True)
    status = Column(String(50), nullable=False, default="created")  # created, in_progress, completed, cancelled, etc.
    
    # Execution requirements
    requires_all_filled = Column(Boolean, nullable=False, default=True)  # All orders must fill for success
    cancel_on_partial_reject = Column(Boolean, nullable=False, default=False)  # Cancel all if any rejected
    execution_instructions = Column(JSON, nullable=True)  # Additional execution instructions
    
    # Metadata
    notes = Column(Text, nullable=True)
    metadata = Column(JSON, nullable=True)
    
    # Relationships
    strategy = relationship("Strategy")
    user = relationship("User")
    account = relationship("Account")
    orders = relationship("Order", secondary="order_batch_items", back_populates="batches")
    batch_items = relationship("OrderBatchItem", back_populates="batch", cascade="all, delete-orphan")
    
    # Constraints and indexes
    __table_args__ = (
        # Indexes for common queries
        Index("ix_order_batches_account_id", account_id),
        Index("ix_order_batches_strategy_id", strategy_id),
        Index("ix_order_batches_status", status),
    )
    
    @hybrid_property
    def orders_count(self) -> int:
        """Get the number of orders in this batch."""
        return len(self.orders)
    
    @hybrid_property
    def filled_orders_count(self) -> int:
        """Get the number of filled orders in this batch."""
        return sum(1 for order in self.orders if order.is_filled)
    
    @hybrid_property
    def cancelled_orders_count(self) -> int:
        """Get the number of cancelled orders in this batch."""
        return sum(1 for order in self.orders if order.is_cancelled)
    
    @hybrid_property
    def fill_percentage(self) -> float:
        """Calculate the percentage of orders that have been filled."""
        if not self.orders:
            return 0.0
        return float(self.filled_orders_count / self.orders_count * 100)
    
    def cancel_all_orders(self, reason: str = "Batch cancelled") -> int:
        """
        Cancel all active orders in the batch.
        
        Args:
            reason: Reason for cancellation
            
        Returns:
            Number of orders cancelled
        """
        cancelled_count = 0
        
        for order in self.orders:
            if order.can_be_cancelled and order.cancel(reason=reason):
                cancelled_count += 1
        
        if cancelled_count > 0:
            self.status = "cancelled"
            self.is_active = False
        
        return cancelled_count
    
    def update_status(self) -> None:
        """Update batch status based on order states."""
        if not self.orders:
            return
            
        # Check if all orders are filled
        if all(order.is_filled for order in self.orders):
            self.status = "completed"
            self.is_active = False
            self.completed_at = datetime.utcnow()
            return
            
        # Check if any orders are active
        if any(order.is_active for order in self.orders):
            if any(order.status == OrderStatus.PARTIALLY_FILLED for order in self.orders):
                self.status = "partially_filled"
            elif any(order.status in (OrderStatus.ACCEPTED, OrderStatus.PENDING) for order in self.orders):
                self.status = "in_progress"
            else:
                self.status = "created"
            return
            
        # If we get here, no orders are active, but not all are filled
        if any(order.is_cancelled for order in self.orders):
            self.status = "cancelled"
        else:
            self.status = "incomplete"
            
        self.is_active = False
    
    def __repr__(self):
        return f"<OrderBatch {self.id} '{self.batch_name}' status={self.status}>"


# Association table for order batches
class OrderBatchItem(Base):
    """
    Links orders to a batch with additional metadata.
    
    Tracks order sequence and dependencies within a batch.
    """
    __tablename__ = "order_batch_items"
    
    batch_id = Column(PGUUID(as_uuid=True), ForeignKey('order_batches.id', ondelete="CASCADE"), primary_key=True)
    order_id = Column(PGUUID(as_uuid=True), ForeignKey('orders.id', ondelete="CASCADE"), primary_key=True)
    
    # Order sequence and dependencies
    sequence_number = Column(Integer, nullable=False)  # Execution sequence
    wait_for_order_id = Column(PGUUID(as_uuid=True), ForeignKey('orders.id', ondelete="SET NULL"), nullable=True)
    
    # Item-specific settings
    can_execute_alone = Column(Boolean, nullable=False, default=True)  # Can execute if other orders in batch fail
    execution_conditions = Column(JSON, nullable=True)  # Additional conditions for this order
    
    # Relationships
    batch = relationship("OrderBatch", back_populates="batch_items")
    order = relationship("Order", foreign_keys=[order_id])
    dependent_on = relationship("Order", foreign_keys=[wait_for_order_id])
    
    # Indexes
    __table_args__ = (
        # Index for sequence ordering
        Index("ix_order_batch_items_batch_sequence", batch_id, sequence_number),
    )


# Add the missing relationship in Order class
Order.batches = relationship("OrderBatch", secondary="order_batch_items", back_populates="orders")


# Utility functions for common order operations

def create_order(
    session,
    account_id: int,
    symbol: str,
    side: OrderSide,
    quantity: Decimal,
    order_type: OrderType,
    price: Optional[Decimal] = None,
    instrument_type: str = "stock",
    client_order_id: Optional[str] = None,
    user_id: Optional[UUID] = None,
    strategy_id: Optional[UUID] = None,
    position_id: Optional[int] = None,
    time_in_force: TimeInForce = TimeInForce.DAY,
    execution_algorithm: Optional[ExecutionAlgorithm] = None,
    execution_params: Optional[Dict[str, Any]] = None,
    routing_destination: Optional[str] = None,
    is_option_order: bool = False,
    option_type: Optional[OptionType] = None,
    strike_price: Optional[Decimal] = None,
    expiration_date: Optional[datetime] = None,
    source: OrderSource = OrderSource.USER,
    priority: OrderPriorityLevel = OrderPriorityLevel.NORMAL,
    notes: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Order:
    """
    Create a new order with validation.
    
    Args:
        session: SQLAlchemy session
        account_id: Account ID
        symbol: Trading symbol
        side: Order side (buy/sell)
        quantity: Order quantity
        order_type: Type of order (market, limit, etc.)
        price: Limit price (required for limit orders)
        instrument_type: Type of instrument being traded
        client_order_id: Client-provided order ID (generated if not provided)
        user_id: User who created the order
        strategy_id: Strategy that generated the order
        position_id: Related position
        time_in_force: How long the order remains active
        execution_algorithm: Algorithm to use for execution
        execution_params: Algorithm-specific parameters
        routing_destination: Exchange or venue to route to
        is_option_order: Whether this is an option order
        option_type: Call or put (for options)
        strike_price: Strike price (for options)
        expiration_date: Expiration date (for options)
        source: Source of the order
        priority: Order priority level
        notes: Order notes
        metadata: Additional metadata
        
    Returns:
        Created Order object
    """
    # Generate client_order_id if not provided
    if not client_order_id:
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        random_suffix = uuid4().hex[:8]
        client_order_id = f"ORD_{timestamp}_{random_suffix}"
    
    # Validate order parameters
    if order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT) and price is None:
        raise ValueError(f"Price is required for {order_type.value} orders")
    
    if is_option_order and (option_type is None or strike_price is None or expiration_date is None):
        raise ValueError("Option orders require option_type, strike_price, and expiration_date")
    
    # Create order object
    order = Order(
        client_order_id=client_order_id,
        account_id=account_id,
        strategy_id=strategy_id,
        user_id=user_id,
        position_id=position_id,
        order_type=order_type,
        side=side,
        symbol=symbol,
        instrument_type=instrument_type,
        quantity=quantity,
        price=price,
        time_in_force=time_in_force,
        execution_algorithm=execution_algorithm,
        execution_params=execution_params,
        routing_destination=routing_destination,
        is_option_order=is_option_order,
        option_type=option_type,
        strike_price=strike_price,
        expiration_date=expiration_date,
        source=source,
        priority=priority,
        notes=notes,
        metadata=metadata,
        remaining_quantity=quantity  # Initially, remaining = total
    )
    
    # Add to session
    session.add(order)
    
    # Create initial order event
    order_event = OrderEvent(
        order_id=order.id,
        event_type="created",
        user_id=user_id,
        new_status=OrderStatus.CREATED,
        details={
            "source": source.value if source else None,
            "client_order_id": client_order_id
        }
    )
    
    session.add(order_event)
    
    # If this is a GTD order, set expiration time
    if time_in_force == TimeInForce.GTD and "expire_time" in (metadata or {}):
        order.expires_at = metadata["expire_time"]
    
    return order


def create_option_spread_order(
    session,
    account_id: int,
    spread_type: str,
    underlying_symbol: str,
    expiration_date: datetime,
    legs: List[Dict[str, Any]],
    client_order_id: Optional[str] = None,
    user_id: Optional[UUID] = None,
    strategy_id: Optional[UUID] = None,
    time_in_force: TimeInForce = TimeInForce.DAY,
    execution_algorithm: Optional[ExecutionAlgorithm] = None,
    routing_destination: Optional[str] = None,
    source: OrderSource = OrderSource.USER,
    priority: OrderPriorityLevel = OrderPriorityLevel.NORMAL,
    notes: Optional[str] = None
) -> Order:
    """
    Create a multi-leg option spread order.
    
    Args:
        session: SQLAlchemy session
        account_id: Account ID
        spread_type: Type of spread (vertical, iron condor, butterfly, etc.)
        underlying_symbol: The underlying security symbol
        expiration_date: Expiration date for all legs
        legs: List of leg specifications, each containing:
            - option_type: Call or put
            - strike_price: Strike price
            - side: Buy or sell
            - quantity: Number of contracts
            - ratio: Leg ratio (default 1)
        client_order_id: Client-provided order ID (generated if not provided)
        user_id: User who created the order
        strategy_id: Strategy that generated the order
        time_in_force: How long the order remains active
        execution_algorithm: Algorithm to use for execution
        routing_destination: Exchange or venue to route to
        source: Source of the order
        priority: Order priority level
        notes: Order notes
        
    Returns:
        Created parent Order object
    """
    # Generate client_order_id if not provided
    if not client_order_id:
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        random_suffix = uuid4().hex[:8]
        client_order_id = f"SPREAD_{timestamp}_{random_suffix}"
    
    # Validate legs
    if not legs or len(legs) < 2:
        raise ValueError("Option spread order must have at least 2 legs")
    
    total_quantity = sum(leg["quantity"] * leg.get("ratio", 1) for leg in legs)
    
    # Create parent order
    parent_order = Order(
        client_order_id=client_order_id,
        account_id=account_id,
        strategy_id=strategy_id,
        user_id=user_id,
        order_type=OrderType.LIMIT,  # Spread orders are typically limit orders
        side=OrderSide.BUY,  # Side will be determined from leg composition
        symbol=underlying_symbol,
        instrument_type="option_spread",
        quantity=total_quantity,
        price=None,  # Will be calculated from legs
        time_in_force=time_in_force,
        execution_algorithm=execution_algorithm,
        routing_destination=routing_destination,
        is_option_order=True,
        is_multi_leg=True,
        source=source,
        priority=priority,
        notes=notes or f"{spread_type.capitalize()} spread on {underlying_symbol}",
        metadata={
            "spread_type": spread_type,
            "leg_count": len(legs)
        },
        remaining_quantity=total_quantity  # Initially, remaining = total
    )
    
    session.add(parent_order)
    
    # Create order legs
    for i, leg in enumerate(legs):
        option_symbol = f"{underlying_symbol}{leg['option_type'][0]}{leg['strike_price']}"
        leg_obj = OrderLeg(
            parent_order_id=parent_order.id,
            leg_number=i + 1,
            side=leg["side"],
            symbol=option_symbol,
            quantity=leg["quantity"],
            price=leg.get("price"),
            ratio=leg.get("ratio", 1),
            is_option=True,
            option_type=leg["option_type"],
            strike_price=leg["strike_price"],
            expiration_date=expiration_date,
            must_execute_first=leg.get("must_execute_first", False),
            simultaneous_execution=leg.get("simultaneous_execution", True)
        )
        
        session.add(leg_obj)
    
    # Create initial order event
    order_event = OrderEvent(
        order_id=parent_order.id,
        event_type="created",
        user_id=user_id,
        new_status=OrderStatus.CREATED,
        details={
            "source": source.value if source else None,
            "spread_type": spread_type,
            "leg_count": len(legs)
        }
    )
    
    session.add(order_event)
    
    return parent_order


def create_conditional_order(
    session,
    order: Order,
    trigger_type: OrderTriggerType,
    trigger_params: Dict[str, Any],
    expires_at: Optional[datetime] = None,
    require_all_conditions: bool = False
) -> ConditionalOrder:
    """
    Create a conditional order trigger for an existing order.
    
    Args:
        session: SQLAlchemy session
        order: Existing order to make conditional
        trigger_type: Type of trigger condition
        trigger_params: Parameters for the trigger condition
        expires_at: When the condition expires
        require_all_conditions: Whether all conditions must be met
        
    Returns:
        Created ConditionalOrder object
    """
    # Validate trigger parameters
    if trigger_type == OrderTriggerType.PRICE:
        required_params = {"symbol", "price_target", "price_comparator"}
    elif trigger_type == OrderTriggerType.TIME:
        required_params = {"trigger_time"}
    elif trigger_type == OrderTriggerType.SIGNAL:
        required_params = {"signal_id", "signal_threshold"}
    elif trigger_type == OrderTriggerType.VOLUME:
        required_params = {"symbol", "volume_threshold", "volume_period_seconds"}
    elif trigger_type == OrderTriggerType.VOLATILITY:
        required_params = {"symbol", "volatility_target", "volatility_comparator", "volatility_period_seconds"}
    elif trigger_type == OrderTriggerType.MANUAL:
        required_params = set()
    else:
        raise ValueError(f"Invalid trigger type: {trigger_type}")
    
    # Check for missing parameters
    missing_params = required_params - set(trigger_params.keys())
    if missing_params:
        raise ValueError(f"Missing required parameters for {trigger_type} trigger: {', '.join(missing_params)}")
    
    # Update order to be conditional
    order.is_conditional = True
    order.trigger_type = trigger_type
    order.trigger_details = trigger_params
    
    # Create conditional order object with base fields
    conditional = ConditionalOrder(
        order_id=order.id,
        trigger_type=trigger_type,
        require_all_conditions=require_all_conditions,
        expires_at=expires_at
    )
    
    # Set specific fields based on trigger type
    if trigger_type == OrderTriggerType.PRICE:
        conditional.symbol = trigger_params["symbol"]
        conditional.price_target = trigger_params["price_target"]
        conditional.price_comparator = trigger_params["price_comparator"]
    
    elif trigger_type == OrderTriggerType.TIME:
        conditional.trigger_time = trigger_params["trigger_time"]
    
    elif trigger_type == OrderTriggerType.SIGNAL:
        conditional.signal_id = trigger_params["signal_id"]
        conditional.signal_threshold = trigger_params["signal_threshold"]
    
    elif trigger_type == OrderTriggerType.VOLUME:
        conditional.symbol = trigger_params["symbol"]
        conditional.volume_threshold = trigger_params["volume_threshold"]
        conditional.volume_period_seconds = trigger_params["volume_period_seconds"]
    
    elif trigger_type == OrderTriggerType.VOLATILITY:
        conditional.symbol = trigger_params["symbol"]
        conditional.volatility_target = trigger_params["volatility_target"]
        conditional.volatility_comparator = trigger_params["volatility_comparator"]
        conditional.volatility_period_seconds = trigger_params["volatility_period_seconds"]
    
    session.add(conditional)
    
    # Create order event
    order_event = OrderEvent(
        order_id=order.id,
        event_type="made_conditional",
        user_id=order.user_id,
        previous_status=order.status,
        new_status=order.status,
        details={
            "trigger_type": trigger_type.value,
            "trigger_params": trigger_params,
            "expires_at": expires_at.isoformat() if expires_at else None
        }
    )
    
    session.add(order_event)
    
    return conditional


def create_order_batch(
    session,
    account_id: int,
    batch_name: str,
    orders: List[Dict[str, Any]],
    batch_type: str = "basket",
    strategy_id: Optional[UUID] = None,
    user_id: Optional[UUID] = None,
    requires_all_filled: bool = True,
    cancel_on_partial_reject: bool = False,
    execution_instructions: Optional[Dict[str, Any]] = None,
    notes: Optional[str] = None
) -> OrderBatch:
    """
    Create a batch of related orders.
    
    Args:
        session: SQLAlchemy session
        account_id: Account ID
        batch_name: Name of the batch
        orders: List of order specifications
        batch_type: Type of batch (basket, pairs, etc.)
        strategy_id: Strategy that generated the batch
        user_id: User who created the batch
        requires_all_filled: Whether all orders must be filled for success
        cancel_on_partial_reject: Whether to cancel all orders if any are rejected
        execution_instructions: Additional execution instructions
        notes: Batch notes
        
    Returns:
        Created OrderBatch object
    """
    # Create batch object
    batch = OrderBatch(
        batch_name=batch_name,
        batch_type=batch_type,
        account_id=account_id,
        strategy_id=strategy_id,
        user_id=user_id,
        requires_all_filled=requires_all_filled,
        cancel_on_partial_reject=cancel_on_partial_reject,
        execution_instructions=execution_instructions,
        notes=notes
    )
    
    session.add(batch)
    session.flush()  # Ensure batch.id is available
    
    # Create orders
    created_orders = []
    for i, order_spec in enumerate(orders):
        # Extract relationship fields
        wait_for_order_idx = order_spec.pop("wait_for_order_idx", None)
        sequence_number = order_spec.pop("sequence_number", i + 1)
        can_execute_alone = order_spec.pop("can_execute_alone", True)
        execution_conditions = order_spec.pop("execution_conditions", None)
        
        # Set account_id from batch
        order_spec["account_id"] = account_id
        
        # Set user_id and strategy_id if not specified in order
        if "user_id" not in order_spec and user_id:
            order_spec["user_id"] = user_id
        if "strategy_id" not in order_spec and strategy_id:
            order_spec["strategy_id"] = strategy_id
        
        # Create the order
        order = create_order(session, **order_spec)
        created_orders.append(order)
        
        # Create batch item
        batch_item = OrderBatchItem(
            batch_id=batch.id,
            order_id=order.id,
            sequence_number=sequence_number,
            can_execute_alone=can_execute_alone,
            execution_conditions=execution_conditions
        )
        
        # Set dependency if specified
        if wait_for_order_idx is not None and 0 <= wait_for_order_idx < len(created_orders):
            batch_item.wait_for_order_id = created_orders[wait_for_order_idx].id
        
        session.add(batch_item)
    
    return batch