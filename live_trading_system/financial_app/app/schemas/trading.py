"""
Schema module for trading components in the application.

This module provides Pydantic models for request/response validation
and serialization related to trading operations. It works with the
SQLAlchemy models defined in app.models.trading for API validations.

Key features:
- Type validation for all trading operations (orders, positions, trades)
- Consistent response formats for all trading operations
- Support for partial updates and status transitions
- Detailed error messages for validation failures
- Nested schema support for complex objects
"""

from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import List, Dict, Optional, Any, Union, Set
from uuid import UUID

from pydantic import BaseModel, Field, validator, root_validator, ConfigDict, field_validator
from pydantic.types import PositiveFloat, confloat, constr, conint

# Import enums from models to maintain consistency
from app.models.trading import (
    OrderStatus, OrderType, OrderSide, TimeInForce, 
    PositionDirection, OrderEventType
)


#################################################
# Base Schema Models
#################################################

class BaseSchema(BaseModel):
    """Base schema model with common configuration."""
    
    model_config = ConfigDict(
        populate_by_name=True,  
        validate_assignment=True,
        use_enum_values=True,
        arbitrary_types_allowed=True,
        json_encoders={
            Decimal: lambda v: float(v)
        },
        json_schema_extra={
            "example": {}  # To be overridden in subclasses
        }
    )


class PaginationParams(BaseSchema):
    """Parameters for paginated requests."""
    
    page: int = Field(1, ge=1, description="Page number (1-indexed)")
    limit: int = Field(100, ge=1, le=1000, description="Number of items per page")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "page": 1,
                "limit": 100
            }
        }
    )


class TimeRangeParams(BaseSchema):
    """Time range filtering parameters."""
    
    start_time: Optional[datetime] = Field(None, description="Start of time range")
    end_time: Optional[datetime] = Field(None, description="End of time range")
    
    @field_validator('end_time')
    def end_time_after_start_time(cls, v, values):
        """Validate that end_time is after start_time if both are provided."""
        if v and 'start_time' in values.data and values.data['start_time'] and v < values.data['start_time']:
            raise ValueError("end_time must be after start_time")
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "start_time": "2023-01-01T00:00:00Z",
                "end_time": "2023-01-31T23:59:59Z"
            }
        }
    )


class StatusMessage(BaseSchema):
    """Standard status message response."""
    
    success: bool = Field(..., description="Whether the operation succeeded")
    message: str = Field(..., description="Status message")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "message": "Operation completed successfully"
            }
        }
    )


#################################################
# Order Schemas
#################################################

class OrderFilter(PaginationParams, TimeRangeParams):
    """Filter parameters for order queries."""
    
    account_id: Optional[int] = Field(None, description="Filter by account ID")
    symbol: Optional[str] = Field(None, description="Filter by symbol")
    status: Optional[List[OrderStatus]] = Field(None, description="Filter by status (multiple allowed)")
    side: Optional[OrderSide] = Field(None, description="Filter by side")
    order_type: Optional[OrderType] = Field(None, description="Filter by order type")
    strategy_id: Optional[str] = Field(None, description="Filter by strategy ID")
    is_active: Optional[bool] = Field(None, description="Filter active/inactive orders")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "account_id": 123,
                "symbol": "AAPL",
                "status": ["submitted", "partially_filled"],
                "side": "buy",
                "page": 1,
                "limit": 50
            }
        }
    )


class OrderBase(BaseSchema):
    """Base schema for order operations."""
    
    symbol: str = Field(..., min_length=1, max_length=20, description="Trading symbol")
    side: OrderSide = Field(..., description="Order side (buy/sell)")
    order_type: OrderType = Field(..., description="Order type")
    quantity: Decimal = Field(..., gt=0, description="Order quantity")
    price: Optional[Decimal] = Field(None, gt=0, description="Limit price")
    stop_price: Optional[Decimal] = Field(None, gt=0, description="Stop price")
    trailing_amount: Optional[Decimal] = Field(None, gt=0, description="Trailing stop amount")
    trailing_percent: Optional[Decimal] = Field(None, gt=0, le=100, description="Trailing stop percent")
    time_in_force: TimeInForce = Field(TimeInForce.DAY, description="Time in force")
    expire_at: Optional[datetime] = Field(None, description="Expiration time for GTD orders")
    client_order_id: Optional[str] = Field(None, max_length=50, description="Client-assigned order ID")
    strategy_id: Optional[str] = Field(None, max_length=50, description="Strategy identifier")
    tags: Optional[List[str]] = Field(None, description="Order tags for categorization")
    notes: Optional[str] = Field(None, description="Additional notes")
    max_slippage_percent: Optional[Decimal] = Field(None, ge=0, le=100, description="Maximum acceptable slippage")
    broker: Optional[str] = Field(None, max_length=50, description="Broker or exchange name")
    venue: Optional[str] = Field(None, max_length=50, description="Execution venue")
    
    @field_validator("price")
    def validate_price_for_limit_orders(cls, v, values):
        """Validate that limit orders have a price."""
        if 'order_type' in values.data and values.data["order_type"] in [OrderType.LIMIT, OrderType.STOP_LIMIT] and v is None:
            raise ValueError(f"Price is required for {values.data['order_type']} orders")
        return v
    
    @field_validator("stop_price")
    def validate_stop_price_for_stop_orders(cls, v, values):
        """Validate that stop orders have a stop price."""
        if 'order_type' in values.data and values.data["order_type"] in [OrderType.STOP, OrderType.STOP_LIMIT] and v is None:
            raise ValueError(f"Stop price is required for {values.data['order_type']} orders")
        return v
    
    @field_validator("trailing_amount", "trailing_percent")
    def validate_trailing_stop_parameters(cls, v, values):
        """Validate trailing stop parameters."""
        if 'order_type' in values.data and values.data["order_type"] == OrderType.TRAILING_STOP:
            # Check if either trailing_amount or trailing_percent is provided
            has_trailing_amount = 'trailing_amount' in values.data and values.data["trailing_amount"] is not None
            has_trailing_percent = 'trailing_percent' in values.data and values.data["trailing_percent"] is not None
            
            if not (has_trailing_amount or has_trailing_percent):
                raise ValueError("Either trailing_amount or trailing_percent is required for trailing stop orders")
        return v

    @field_validator("expire_at")
    def validate_expire_at_for_gtd_orders(cls, v, values):
        """Validate that GTD orders have an expiration time."""
        if 'time_in_force' in values.data and values.data["time_in_force"] == TimeInForce.GTD and v is None:
            raise ValueError("Expiration time is required for GTD orders")
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "symbol": "AAPL",
                "side": "buy",
                "order_type": "limit",
                "quantity": 100,
                "price": 150.50,
                "time_in_force": "day",
                "strategy_id": "mean_reversion_v1",
                "tags": ["tech", "momentum"]
            }
        }
    )


class OrderCreate(OrderBase):
    """Schema for creating a new order."""
    
    account_id: int = Field(..., gt=0, description="Account ID")
    parent_order_id: Optional[str] = Field(None, description="Parent order ID for child orders")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "account_id": 123,
                "symbol": "AAPL",
                "side": "buy",
                "order_type": "limit",
                "quantity": 100,
                "price": 150.50,
                "time_in_force": "day",
                "strategy_id": "mean_reversion_v1"
            }
        }
    )


class OrderUpdate(BaseSchema):
    """Schema for updating an existing order."""
    
    price: Optional[Decimal] = Field(None, gt=0, description="Updated limit price")
    stop_price: Optional[Decimal] = Field(None, gt=0, description="Updated stop price")
    quantity: Optional[Decimal] = Field(None, gt=0, description="Updated quantity")
    time_in_force: Optional[TimeInForce] = Field(None, description="Updated time in force")
    expire_at: Optional[datetime] = Field(None, description="Updated expiration time for GTD orders")
    tags: Optional[List[str]] = Field(None, description="Updated order tags")
    notes: Optional[str] = Field(None, description="Updated notes")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "price": 152.75,
                "quantity": 150,
                "notes": "Increased position size based on new signal"
            }
        }
    )


class OrderStatusUpdate(BaseSchema):
    """Schema for updating an order's status."""
    
    status: OrderStatus = Field(..., description="New order status")
    reason: Optional[str] = Field(None, description="Reason for status change")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "canceled",
                "reason": "Manual cancellation by user"
            }
        }
    )


class ExecutionCreate(BaseSchema):
    """Schema for creating a new execution (fill)."""
    
    quantity: Decimal = Field(..., gt=0, description="Executed quantity")
    price: Decimal = Field(..., gt=0, description="Execution price")
    execution_id: Optional[str] = Field(None, max_length=50, description="External execution ID")
    fees: Optional[Decimal] = Field(None, ge=0, description="Execution fees")
    executed_at: Optional[datetime] = Field(None, description="Execution timestamp")
    venue: Optional[str] = Field(None, max_length=50, description="Execution venue")
    liquidity: Optional[str] = Field(None, max_length=10, description="Maker/taker liquidity")
    route: Optional[str] = Field(None, max_length=50, description="Routing information")
    
    @field_validator("quantity")
    def validate_quantity(cls, v):
        """Validate execution quantity."""
        if v <= 0:
            raise ValueError("Execution quantity must be positive")
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "quantity": 50,
                "price": 150.25,
                "execution_id": "ex-12345",
                "fees": 0.75,
                "executed_at": "2023-01-15T14:30:00Z",
                "liquidity": "taker"
            }
        }
    )


class ExecutionResponse(ExecutionCreate):
    """Schema for execution response."""
    
    id: int = Field(..., description="Execution ID")
    order_id: str = Field(..., description="Order ID")
    recorded_at: datetime = Field(..., description="Time execution was recorded")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": 456,
                "order_id": "ord-12345",
                "quantity": 50,
                "price": 150.25,
                "execution_id": "ex-12345",
                "fees": 0.75,
                "executed_at": "2023-01-15T14:30:00Z",
                "recorded_at": "2023-01-15T14:30:01Z",
                "liquidity": "taker"
            }
        }
    )


class OrderEventResponse(BaseSchema):
    """Schema for order event response."""
    
    id: int = Field(..., description="Event ID")
    order_id: str = Field(..., description="Order ID")
    event_type: OrderEventType = Field(..., description="Event type")
    description: Optional[str] = Field(None, description="Event description")
    event_data: Optional[Dict[str, Any]] = Field(None, description="Event data")
    created_at: datetime = Field(..., description="Event timestamp")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": 789,
                "order_id": "ord-12345",
                "event_type": "submitted",
                "description": "Order submitted to broker",
                "event_data": {"broker_confirmation": "conf-56789"},
                "created_at": "2023-01-15T14:25:00Z"
            }
        }
    )


class OrderResponse(OrderBase):
    """Schema for order response."""
    
    id: int = Field(..., description="Order ID")
    order_id: str = Field(..., description="Order UUID")
    account_id: int = Field(..., description="Account ID")
    status: OrderStatus = Field(..., description="Order status")
    filled_quantity: Decimal = Field(..., ge=0, description="Filled quantity")
    average_fill_price: Optional[Decimal] = Field(None, description="Average fill price")
    remaining_quantity: Optional[Decimal] = Field(None, ge=0, description="Remaining quantity to fill")
    broker_order_id: Optional[str] = Field(None, description="Broker's order ID")
    parent_order_id: Optional[str] = Field(None, description="Parent order ID")
    risk_check_passed: bool = Field(..., description="Whether order passed risk checks")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    submitted_at: Optional[datetime] = Field(None, description="Submission timestamp")
    filled_at: Optional[datetime] = Field(None, description="Filled timestamp")
    canceled_at: Optional[datetime] = Field(None, description="Canceled timestamp")
    rejected_at: Optional[datetime] = Field(None, description="Rejected timestamp")
    is_active: bool = Field(..., description="Whether order is still active")
    fill_percent: float = Field(..., description="Percentage filled")
    value: Optional[float] = Field(None, description="Total order value")
    executions: Optional[List[ExecutionResponse]] = Field(None, description="Order executions")
    events: Optional[List[OrderEventResponse]] = Field(None, description="Order events")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": 1,
                "order_id": "ord-12345",
                "account_id": 123,
                "symbol": "AAPL",
                "side": "buy",
                "order_type": "limit",
                "quantity": 100,
                "price": 150.50,
                "status": "submitted",
                "time_in_force": "day",
                "filled_quantity": 0,
                "remaining_quantity": 100,
                "strategy_id": "mean_reversion_v1",
                "risk_check_passed": True,
                "created_at": "2023-01-15T14:20:00Z",
                "submitted_at": "2023-01-15T14:25:00Z",
                "is_active": True,
                "fill_percent": 0,
                "value": 15050.00
            }
        }
    )


class OrderListResponse(BaseSchema):
    """Schema for paginated order list response."""
    
    items: List[OrderResponse] = Field(..., description="List of orders")
    total: int = Field(..., description="Total number of matching orders")
    page: int = Field(..., description="Current page number")
    limit: int = Field(..., description="Items per page")
    pages: int = Field(..., description="Total number of pages")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "items": [
                    {
                        "id": 1,
                        "order_id": "ord-12345",
                        "account_id": 123,
                        "symbol": "AAPL",
                        "side": "buy",
                        "order_type": "limit",
                        "quantity": 100,
                        "price": 150.50,
                        "status": "submitted",
                        "time_in_force": "day",
                        "filled_quantity": 0,
                        "remaining_quantity": 100,
                        "created_at": "2023-01-15T14:20:00Z",
                        "is_active": True,
                        "fill_percent": 0
                    }
                ],
                "total": 42,
                "page": 1,
                "limit": 10,
                "pages": 5
            }
        }
    )


#################################################
# Position Schemas
#################################################

class PositionFilter(BaseSchema):
    """Filter parameters for position queries."""
    
    account_id: Optional[int] = Field(None, description="Filter by account ID")
    symbol: Optional[str] = Field(None, description="Filter by symbol")
    direction: Optional[PositionDirection] = Field(None, description="Filter by direction")
    strategy_id: Optional[str] = Field(None, description="Filter by strategy ID")
    min_quantity: Optional[Decimal] = Field(None, gt=0, description="Minimum quantity")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "account_id": 123,
                "direction": "long",
                "min_quantity": 10
            }
        }
    )


class PositionBase(BaseSchema):
    """Base schema for position operations."""
    
    symbol: str = Field(..., min_length=1, max_length=20, description="Trading symbol")
    direction: PositionDirection = Field(..., description="Position direction")
    quantity: Decimal = Field(..., gt=0, description="Position size")
    average_entry_price: Decimal = Field(..., gt=0, description="Average entry price")
    current_price: Decimal = Field(..., gt=0, description="Current market price")
    strategy_id: Optional[str] = Field(None, max_length=50, description="Strategy identifier")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "symbol": "AAPL",
                "direction": "long",
                "quantity": 100,
                "average_entry_price": 150.25,
                "current_price": 152.50,
                "strategy_id": "mean_reversion_v1"
            }
        }
    )


class PositionCreate(PositionBase):
    """Schema for creating a position."""
    
    account_id: int = Field(..., gt=0, description="Account ID")
    realized_pnl: Decimal = Field(0, description="Realized P&L")
    unrealized_pnl: Optional[Decimal] = Field(None, description="Unrealized P&L")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "account_id": 123,
                "symbol": "AAPL",
                "direction": "long",
                "quantity": 100,
                "average_entry_price": 150.25,
                "current_price": 152.50,
                "strategy_id": "mean_reversion_v1",
                "realized_pnl": 0
            }
        }
    )


class PositionUpdate(BaseSchema):
    """Schema for updating a position."""
    
    current_price: Optional[Decimal] = Field(None, gt=0, description="Updated market price")
    quantity: Optional[Decimal] = Field(None, gt=0, description="Updated position size")
    average_entry_price: Optional[Decimal] = Field(None, gt=0, description="Updated entry price")
    realized_pnl: Optional[Decimal] = Field(None, description="Updated realized P&L")
    unrealized_pnl: Optional[Decimal] = Field(None, description="Updated unrealized P&L")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "current_price": 155.75,
                "unrealized_pnl": 550.00
            }
        }
    )


class PositionRiskUpdate(BaseSchema):
    """Schema for updating position risk parameters."""
    
    stop_loss_price: Optional[Decimal] = Field(None, description="Stop loss price")
    take_profit_price: Optional[Decimal] = Field(None, description="Take profit price")
    trailing_stop_distance: Optional[Decimal] = Field(None, gt=0, description="Trailing stop distance")
    trailing_stop_percent: Optional[Decimal] = Field(None, gt=0, le=100, description="Trailing stop percent")
    trailing_stop_activation_price: Optional[Decimal] = Field(None, gt=0, description="Trailing stop activation price")
    
    @root_validator(skip_on_failure=True)
    def check_trailing_stop_params(cls, values):
        """Validate trailing stop parameters."""
        # If setting up trailing stop, ensure either distance or percent is provided
        if (values.get('trailing_stop_activation_price') is not None and 
            values.get('trailing_stop_distance') is None and 
            values.get('trailing_stop_percent') is None):
            raise ValueError("Either trailing_stop_distance or trailing_stop_percent must be provided")
        return values
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "stop_loss_price": 145.00,
                "take_profit_price": 165.00,
                "trailing_stop_percent": 5,
                "trailing_stop_activation_price": 160.00
            }
        }
    )


class PositionResponse(PositionBase):
    """Schema for position response."""
    
    id: int = Field(..., description="Position ID")
    account_id: int = Field(..., description="Account ID")
    realized_pnl: Decimal = Field(..., description="Realized P&L")
    unrealized_pnl: Decimal = Field(..., description="Unrealized P&L")
    total_pnl: Decimal = Field(..., description="Total P&L")
    pnl_percentage: float = Field(..., description="P&L as percentage")
    market_value: Decimal = Field(..., description="Current market value")
    cost_basis: Decimal = Field(..., description="Original cost basis")
    stop_loss_price: Optional[Decimal] = Field(None, description="Stop loss price")
    take_profit_price: Optional[Decimal] = Field(None, description="Take profit price")
    trailing_stop_price: Optional[Decimal] = Field(None, description="Current trailing stop price")
    trailing_stop_distance: Optional[Decimal] = Field(None, description="Trailing stop distance")
    trailing_stop_percent: Optional[Decimal] = Field(None, description="Trailing stop percentage")
    trailing_stop_activation_price: Optional[Decimal] = Field(None, description="Trailing stop activation price")
    opened_at: datetime = Field(..., description="When position was opened")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    last_trade_at: datetime = Field(..., description="Last trade timestamp")
    last_pnl_update: datetime = Field(..., description="Last P&L update timestamp")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": 42,
                "account_id": 123,
                "symbol": "AAPL",
                "direction": "long",
                "quantity": 100,
                "average_entry_price": 150.25,
                "current_price": 152.50,
                "realized_pnl": 0,
                "unrealized_pnl": 225.00,
                "total_pnl": 225.00,
                "pnl_percentage": 1.5,
                "market_value": 15250.00,
                "cost_basis": 15025.00,
                "stop_loss_price": 145.00,
                "take_profit_price": 165.00,
                "opened_at": "2023-01-10T09:30:00Z",
                "last_trade_at": "2023-01-10T09:30:00Z",
                "last_pnl_update": "2023-01-15T16:00:00Z"
            }
        }
    )


class PositionListResponse(BaseSchema):
    """Schema for position list response."""
    
    items: List[PositionResponse] = Field(..., description="List of positions")
    total: int = Field(..., description="Total number of matching positions")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "items": [
                    {
                        "id": 42,
                        "account_id": 123,
                        "symbol": "AAPL",
                        "direction": "long",
                        "quantity": 100,
                        "average_entry_price": 150.25,
                        "current_price": 152.50,
                        "realized_pnl": 0,
                        "unrealized_pnl": 225.00,
                        "total_pnl": 225.00,
                        "pnl_percentage": 1.5,
                        "market_value": 15250.00,
                        "cost_basis": 15025.00
                    }
                ],
                "total": 5
            }
        }
    )


#################################################
# Trade Schemas
#################################################

class TradeFilter(PaginationParams, TimeRangeParams):
    """Filter parameters for trade queries."""
    
    account_id: Optional[int] = Field(None, description="Filter by account ID")
    symbol: Optional[str] = Field(None, description="Filter by symbol")
    side: Optional[OrderSide] = Field(None, description="Filter by side")
    min_value: Optional[Decimal] = Field(None, gt=0, description="Minimum trade value")
    strategy_id: Optional[str] = Field(None, description="Filter by strategy ID")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "account_id": 123,
                "symbol": "AAPL",
                "start_time": "2023-01-01T00:00:00Z",
                "end_time": "2023-01-31T23:59:59Z",
                "page": 1,
                "limit": 50
            }
        }
    )


class TradeBase(BaseSchema):
    """Base schema for trade operations."""
    
    symbol: str = Field(..., min_length=1, max_length=20, description="Trading symbol")
    side: OrderSide = Field(..., description="Trade side")
    quantity: Decimal = Field(..., gt=0, description="Trade quantity")
    price: Decimal = Field(..., gt=0, description="Execution price")
    fees: Decimal = Field(0, ge=0, description="Trade fees")
    order_id: str = Field(..., description="Source order ID")
    execution_id: Optional[str] = Field(None, description="Execution ID")
    strategy_id: Optional[str] = Field(None, description="Strategy ID")
    notes: Optional[str] = Field(None, description="Trade notes")
    executed_at: datetime = Field(..., description="Execution timestamp")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "symbol": "AAPL",
                "side": "buy",
                "quantity": 100,
                "price": 150.25,
                "fees": 0.75,
                "order_id": "ord-12345",
                "execution_id": "ex-67890",
                "strategy_id": "mean_reversion_v1",
                "executed_at": "2023-01-15T14:30:00Z"
            }
        }
    )


class TradeCreate(TradeBase):
    """Schema for creating a trade record."""
    
    account_id: int = Field(..., gt=0, description="Account ID")
    value: Optional[Decimal] = Field(None, description="Trade value (price * quantity)")
    total_cost: Optional[Decimal] = Field(None, description="Total cost including fees")
    realized_pnl: Optional[Decimal] = Field(None, description="Realized P&L")
    tax_lot_id: Optional[str] = Field(None, description="Tax lot identifier")
    wash_sale: Optional[bool] = Field(False, description="Wash sale flag")
    
    @field_validator("value", "total_cost")
    def calculate_values(cls, v, values):
        """Calculate value and total_cost if not provided."""
        if v is None and values.data.get("quantity") is not None and values.data.get("price") is not None:
            if values.field_name == "value":
                return values.data["quantity"] * values.data["price"]
            elif values.field_name == "total_cost":
                value = values.data["quantity"] * values.data["price"]
                fees = values.data.get("fees", 0)
                return value + fees
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "account_id": 123,
                "symbol": "AAPL",
                "side": "buy",
                "quantity": 100,
                "price": 150.25,
                "fees": 0.75,
                "order_id": "ord-12345",
                "execution_id": "ex-67890",
                "strategy_id": "mean_reversion_v1",
                "executed_at": "2023-01-15T14:30:00Z"
            }
        }
    )


class TradeResponse(TradeBase):
    """Schema for trade response."""
    
    id: int = Field(..., description="Trade ID")
    trade_id: str = Field(..., description="Trade UUID")
    account_id: int = Field(..., description="Account ID")
    value: Decimal = Field(..., description="Trade value (price * quantity)")
    total_cost: Decimal = Field(..., description="Total cost including fees")
    realized_pnl: Optional[Decimal] = Field(None, description="Realized P&L")
    tax_lot_id: Optional[str] = Field(None, description="Tax lot identifier")
    wash_sale: bool = Field(..., description="Wash sale flag")
    recorded_at: datetime = Field(..., description="When trade was recorded")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": 789,
                "trade_id": "trade-12345",
                "account_id": 123,
                "symbol": "AAPL",
                "side": "buy",
                "quantity": 100,
                "price": 150.25,
                "fees": 0.75,
                "value": 15025.00,
                "total_cost": 15025.75,
                "order_id": "ord-12345",
                "execution_id": "ex-67890",
                "strategy_id": "mean_reversion_v1",
                "executed_at": "2023-01-15T14:30:00Z",
                "recorded_at": "2023-01-15T14:30:01Z"
            }
        }
    )


class TradeListResponse(BaseSchema):
    """Schema for paginated trade list response."""
    
    items: List[TradeResponse] = Field(..., description="List of trades")
    total: int = Field(..., description="Total number of matching trades")
    page: int = Field(..., description="Current page number")
    limit: int = Field(..., description="Items per page")
    pages: int = Field(..., description="Total number of pages")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "items": [
                    {
                        "id": 789,
                        "trade_id": "trade-12345",
                        "account_id": 123,
                        "symbol": "AAPL",
                        "side": "buy",
                        "quantity": 100,
                        "price": 150.25,
                        "fees": 0.75,
                        "value": 15025.00,
                        "total_cost": 15025.75,
                        "order_id": "ord-12345",
                        "executed_at": "2023-01-15T14:30:00Z",
                        "recorded_at": "2023-01-15T14:30:01Z"
                    }
                ],
                "total": 25,
                "page": 1,
                "limit": 10,
                "pages": 3
            }
        }
    )


#################################################
# Bracket Order Schemas
#################################################

class BracketOrderCreate(BaseSchema):
    """Schema for creating a bracket order."""
    
    account_id: int = Field(..., gt=0, description="Account ID")
    symbol: str = Field(..., min_length=1, max_length=20, description="Trading symbol")
    side: OrderSide = Field(..., description="Entry order side (buy/sell)")
    quantity: Decimal = Field(..., gt=0, description="Order quantity")
    
    # Entry order parameters
    entry_type: OrderType = Field(..., description="Entry order type")
    entry_price: Optional[Decimal] = Field(None, gt=0, description="Entry limit price")
    
    # Stop loss parameters
    stop_loss_price: Optional[Decimal] = Field(None, gt=0, description="Stop loss price")
    
    # Take profit parameters
    take_profit_price: Optional[Decimal] = Field(None, gt=0, description="Take profit price")
    
    # Common parameters
    time_in_force: TimeInForce = Field(TimeInForce.DAY, description="Time in force")
    expire_at: Optional[datetime] = Field(None, description="Expiration time for GTD orders")
    client_order_id: Optional[str] = Field(None, max_length=50, description="Client-assigned order ID")
    strategy_id: Optional[str] = Field(None, max_length=50, description="Strategy identifier")
    tags: Optional[List[str]] = Field(None, description="Order tags")
    notes: Optional[str] = Field(None, description="Additional notes")
    
    @field_validator("entry_price")
    def validate_entry_price(cls, v, values):
        """Validate entry price for limit orders."""
        if 'entry_type' in values.data and values.data["entry_type"] == OrderType.LIMIT and v is None:
            raise ValueError("Entry price is required for limit orders")
        return v
    
    @root_validator(skip_on_failure=True)
    def validate_stop_take_prices(cls, values):
        """Validate that stop loss and take profit prices make sense for the order side."""
        side = values.get("side")
        stop_price = values.get("stop_loss_price")
        take_price = values.get("take_profit_price")
        entry_price = values.get("entry_price")
        
        # For market orders, we can't validate against entry price
        if values.get("entry_type") == OrderType.MARKET:
            return values
            
        if side and stop_price and take_price and entry_price:
            if side == OrderSide.BUY:
                # For buy orders: stop loss < entry < take profit
                if stop_price >= entry_price:
                    raise ValueError("Stop loss price must be below entry price for buy orders")
                if take_price <= entry_price:
                    raise ValueError("Take profit price must be above entry price for buy orders")
            else:  # SELL
                # For sell orders: stop loss > entry > take profit
                if stop_price <= entry_price:
                    raise ValueError("Stop loss price must be above entry price for sell orders")
                if take_price >= entry_price:
                    raise ValueError("Take profit price must be below entry price for sell orders")
        
        return values
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "account_id": 123,
                "symbol": "AAPL",
                "side": "buy",
                "quantity": 100,
                "entry_type": "limit",
                "entry_price": 150.00,
                "stop_loss_price": 145.00,
                "take_profit_price": 160.00,
                "time_in_force": "gtc",
                "strategy_id": "breakout_v1"
            }
        }
    )


class BracketOrderResponse(BaseSchema):
    """Schema for bracket order response."""
    
    id: int = Field(..., description="Bracket order ID")
    account_id: int = Field(..., description="Account ID")
    symbol: str = Field(..., description="Trading symbol")
    status: str = Field(..., description="Bracket order status")
    entry_order_id: str = Field(..., description="Entry order ID")
    stop_loss_order_id: Optional[str] = Field(None, description="Stop loss order ID")
    take_profit_order_id: Optional[str] = Field(None, description="Take profit order ID")
    strategy_id: Optional[str] = Field(None, description="Strategy ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    
    # Include the detailed orders
    entry_order: Optional[OrderResponse] = Field(None, description="Entry order details")
    stop_loss_order: Optional[OrderResponse] = Field(None, description="Stop loss order details")
    take_profit_order: Optional[OrderResponse] = Field(None, description="Take profit order details")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": 123,
                "account_id": 123,
                "symbol": "AAPL",
                "status": "active",
                "entry_order_id": "ord-12345",
                "stop_loss_order_id": "ord-12346",
                "take_profit_order_id": "ord-12347",
                "strategy_id": "breakout_v1",
                "created_at": "2023-01-15T14:20:00Z",
                "updated_at": "2023-01-15T14:25:00Z"
            }
        }
    )


class BracketOrderListResponse(BaseSchema):
    """Schema for bracket order list response."""
    
    items: List[BracketOrderResponse] = Field(..., description="List of bracket orders")
    total: int = Field(..., description="Total number of matching bracket orders")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "items": [
                    {
                        "id": 123,
                        "account_id": 123,
                        "symbol": "AAPL",
                        "status": "active",
                        "entry_order_id": "ord-12345",
                        "stop_loss_order_id": "ord-12346",
                        "take_profit_order_id": "ord-12347",
                        "strategy_id": "breakout_v1",
                        "created_at": "2023-01-15T14:20:00Z"
                    }
                ],
                "total": 5
            }
        }
    )


#################################################
# OCO Order Schemas
#################################################

class OCOOrderCreate(BaseSchema):
    """Schema for creating a One-Cancels-Other (OCO) order pair."""
    
    account_id: int = Field(..., gt=0, description="Account ID")
    symbol: str = Field(..., min_length=1, max_length=20, description="Trading symbol")
    side: OrderSide = Field(..., description="Order side (buy/sell)")
    quantity: Decimal = Field(..., gt=0, description="Order quantity")
    
    # First order (typically limit order)
    price_1: Decimal = Field(..., gt=0, description="First order price")
    type_1: OrderType = Field(..., description="First order type")
    
    # Second order (typically stop order)
    price_2: Decimal = Field(..., gt=0, description="Second order price")
    type_2: OrderType = Field(..., description="Second order type")
    
    # Common parameters
    time_in_force: TimeInForce = Field(TimeInForce.DAY, description="Time in force")
    expire_at: Optional[datetime] = Field(None, description="Expiration time for GTD orders")
    client_order_id: Optional[str] = Field(None, max_length=50, description="Client-assigned order ID prefix")
    strategy_id: Optional[str] = Field(None, max_length=50, description="Strategy identifier")
    
    @root_validator(skip_on_failure=True)
    def validate_order_types(cls, values):
        """Validate that the order types are valid for OCO orders."""
        valid_types = {OrderType.LIMIT, OrderType.STOP, OrderType.STOP_LIMIT}
        
        type_1 = values.get("type_1")
        type_2 = values.get("type_2")
        
        if type_1 not in valid_types:
            raise ValueError(f"First order type must be one of: {valid_types}")
            
        if type_2 not in valid_types:
            raise ValueError(f"Second order type must be one of: {valid_types}")
            
        # Typically one is a limit and one is a stop
        if type_1 == type_2:
            raise ValueError("OCO orders should have different order types")
            
        # Price validation depends on side
        side = values.get("side")
        price_1 = values.get("price_1")
        price_2 = values.get("price_2")
        
        if side and price_1 and price_2:
            if side == OrderSide.BUY:
                # For buy OCO, typically we have a lower limit buy and higher stop buy
                if type_1 == OrderType.LIMIT and type_2 in {OrderType.STOP, OrderType.STOP_LIMIT}:
                    if price_1 >= price_2:
                        raise ValueError("For buy OCO, limit price should be below stop price")
                elif type_2 == OrderType.LIMIT and type_1 in {OrderType.STOP, OrderType.STOP_LIMIT}:
                    if price_2 >= price_1:
                        raise ValueError("For buy OCO, limit price should be below stop price")
            else:  # SELL
                # For sell OCO, typically we have a higher limit sell and lower stop sell
                if type_1 == OrderType.LIMIT and type_2 in {OrderType.STOP, OrderType.STOP_LIMIT}:
                    if price_1 <= price_2:
                        raise ValueError("For sell OCO, limit price should be above stop price")
                elif type_2 == OrderType.LIMIT and type_1 in {OrderType.STOP, OrderType.STOP_LIMIT}:
                    if price_2 <= price_1:
                        raise ValueError("For sell OCO, limit price should be above stop price")
        
        return values
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "account_id": 123,
                "symbol": "AAPL",
                "side": "buy",
                "quantity": 100,
                "price_1": 145.00,  # Limit buy below current price
                "type_1": "limit",
                "price_2": 155.00,  # Stop buy above current price
                "type_2": "stop",
                "time_in_force": "day",
                "strategy_id": "breakout_v1"
            }
        }
    )


#################################################
# Order Conversion Functions
#################################################

def order_model_to_schema(order, include_executions: bool = False, include_events: bool = False) -> OrderResponse:
    """
    Convert a SQLAlchemy Order model to a Pydantic OrderResponse schema.
    
    Args:
        order: SQLAlchemy Order model instance
        include_executions: Include execution details
        include_events: Include order events
        
    Returns:
        OrderResponse schema instance
    """
    data = {
        "id": order.id,
        "order_id": order.order_id,
        "account_id": order.account_id,
        "symbol": order.symbol,
        "side": order.side,
        "order_type": order.order_type,
        "quantity": order.quantity,
        "price": order.price,
        "stop_price": order.stop_price,
        "trailing_amount": order.trailing_amount,
        "trailing_percent": order.trailing_percent,
        "time_in_force": order.time_in_force,
        "expire_at": order.expire_at,
        "status": order.status,
        "filled_quantity": order.filled_quantity,
        "average_fill_price": order.average_fill_price,
        "remaining_quantity": order.remaining_quantity,
        "client_order_id": order.client_order_id,
        "broker_order_id": order.broker_order_id,
        "strategy_id": order.strategy_id,
        "tags": order.tags_list if hasattr(order, "tags_list") else None,
        "notes": order.notes,
        "parent_order_id": order.parent_order_id,
        "max_slippage_percent": order.max_slippage_percent,
        "risk_check_passed": order.risk_check_passed,
        "created_at": order.created_at,
        "updated_at": order.updated_at,
        "submitted_at": order.submitted_at,
        "filled_at": order.filled_at,
        "canceled_at": order.canceled_at,
        "rejected_at": order.rejected_at,
        "broker": order.broker,
        "venue": order.venue,
        "is_active": order.is_active if hasattr(order, "is_active") else False,
        "fill_percent": float(order.fill_percent) if hasattr(order, "fill_percent") else 0,
        "value": float(order.value) if hasattr(order, "value") and order.value else None
    }
    
    # Include executions if requested
    if include_executions and hasattr(order, "executions") and order.executions:
        data["executions"] = [execution_model_to_schema(exec) for exec in order.executions]
        
    # Include events if requested
    if include_events and hasattr(order, "order_events") and order.order_events:
        data["events"] = [order_event_model_to_schema(event) for event in order.order_events]
    
    return OrderResponse(**data)


def execution_model_to_schema(execution) -> ExecutionResponse:
    """
    Convert a SQLAlchemy Execution model to a Pydantic ExecutionResponse schema.
    
    Args:
        execution: SQLAlchemy Execution model instance
        
    Returns:
        ExecutionResponse schema instance
    """
    return ExecutionResponse(
        id=execution.id,
        order_id=execution.order_id,
        execution_id=execution.execution_id,
        quantity=execution.quantity,
        price=execution.price,
        fees=execution.fees,
        venue=execution.venue,
        liquidity=execution.liquidity,
        route=execution.route,
        executed_at=execution.executed_at,
        recorded_at=execution.recorded_at
    )


def order_event_model_to_schema(event) -> OrderEventResponse:
    """
    Convert a SQLAlchemy OrderEvent model to a Pydantic OrderEventResponse schema.
    
    Args:
        event: SQLAlchemy OrderEvent model instance
        
    Returns:
        OrderEventResponse schema instance
    """
    return OrderEventResponse(
        id=event.id,
        order_id=event.order_id,
        event_type=event.event_type,
        description=event.description,
        event_data=event.data if hasattr(event, "data") else None,
        created_at=event.created_at
    )


def position_model_to_schema(position) -> PositionResponse:
    """
    Convert a SQLAlchemy Position model to a Pydantic PositionResponse schema.
    
    Args:
        position: SQLAlchemy Position model instance
        
    Returns:
        PositionResponse schema instance
    """
    return PositionResponse(
        id=position.id,
        account_id=position.account_id,
        symbol=position.symbol,
        direction=position.direction,
        quantity=position.quantity,
        average_entry_price=position.average_entry_price,
        current_price=position.current_price,
        realized_pnl=position.realized_pnl,
        unrealized_pnl=position.unrealized_pnl,
        total_pnl=position.total_pnl if hasattr(position, "total_pnl") else (position.realized_pnl + position.unrealized_pnl),
        pnl_percentage=float(position.pnl_percentage) if hasattr(position, "pnl_percentage") else 0,
        market_value=position.market_value if hasattr(position, "market_value") else None,
        cost_basis=position.cost_basis if hasattr(position, "cost_basis") else None,
        stop_loss_price=position.stop_loss_price,
        take_profit_price=position.take_profit_price,
        trailing_stop_price=position.trailing_stop_price,
        trailing_stop_distance=position.trailing_stop_distance,
        trailing_stop_percent=position.trailing_stop_percent,
        trailing_stop_activation_price=position.trailing_stop_activation_price,
        strategy_id=position.strategy_id,
        opened_at=position.opened_at,
        updated_at=position.updated_at,
        last_trade_at=position.last_trade_at,
        last_pnl_update=position.last_pnl_update
    )


def trade_model_to_schema(trade) -> TradeResponse:
    """
    Convert a SQLAlchemy Trade model to a Pydantic TradeResponse schema.
    
    Args:
        trade: SQLAlchemy Trade model instance
        
    Returns:
        TradeResponse schema instance
    """
    return TradeResponse(
        id=trade.id,
        trade_id=trade.trade_id,
        account_id=trade.account_id,
        symbol=trade.symbol,
        side=trade.side,
        quantity=trade.quantity,
        price=trade.price,
        fees=trade.fees,
        value=trade.value,
        total_cost=trade.total_cost,
        realized_pnl=trade.realized_pnl,
        order_id=trade.order_id,
        execution_id=trade.execution_id,
        strategy_id=trade.strategy_id,
        notes=trade.notes,
        tax_lot_id=trade.tax_lot_id,
        wash_sale=trade.wash_sale,
        executed_at=trade.executed_at,
        recorded_at=trade.recorded_at
    )


def bracket_order_model_to_schema(bracket, include_orders: bool = False) -> BracketOrderResponse:
    """
    Convert a SQLAlchemy BracketOrder model to a Pydantic BracketOrderResponse schema.
    
    Args:
        bracket: SQLAlchemy BracketOrder model instance
        include_orders: Include detailed order information
        
    Returns:
        BracketOrderResponse schema instance
    """
    data = {
        "id": bracket.id,
        "account_id": bracket.account_id,
        "symbol": bracket.symbol,
        "status": bracket.status,
        "entry_order_id": bracket.entry_order_id,
        "stop_loss_order_id": bracket.stop_loss_order_id,
        "take_profit_order_id": bracket.take_profit_order_id,
        "strategy_id": bracket.strategy_id,
        "created_at": bracket.created_at,
        "updated_at": bracket.updated_at
    }
    
    # Include detailed orders if requested
    if include_orders:
        if hasattr(bracket, "entry_order") and bracket.entry_order:
            data["entry_order"] = order_model_to_schema(bracket.entry_order)
            
        if hasattr(bracket, "stop_loss_order") and bracket.stop_loss_order:
            data["stop_loss_order"] = order_model_to_schema(bracket.stop_loss_order)
            
        if hasattr(bracket, "take_profit_order") and bracket.take_profit_order:
            data["take_profit_order"] = order_model_to_schema(bracket.take_profit_order)
    
    return BracketOrderResponse(**data)