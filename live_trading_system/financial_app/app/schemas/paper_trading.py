"""
Order Schemas for Simulation-Based Trading System

This module defines Pydantic schemas for order management in our simulation environment.
These schemas handle validation, serialization, and API communication for orders.
"""

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# Enums for schema validation
class OrderTypeEnum(str, Enum):
    """Order types for API validation."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LIMIT = "stop_limit"


class OrderStatusEnum(str, Enum):
    """Order status for API validation."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderSideEnum(str, Enum):
    """Order side for API validation."""
    BUY = "buy"
    SELL = "sell"


class FillTypeEnum(str, Enum):
    """Fill types for API validation."""
    FULL = "full"
    PARTIAL = "partial"
    MARKET = "market"
    LIMIT = "limit"


class SlippageModelEnum(str, Enum):
    """Slippage models for simulation."""
    FIXED = "fixed"              # Fixed slippage amount
    PERCENTAGE = "percentage"     # Percentage-based slippage
    VOLUME_BASED = "volume_based" # Based on order size vs market volume
    SPREAD_BASED = "spread_based" # Based on bid-ask spread
    MARKET_IMPACT = "market_impact" # Based on market impact model


# Base schemas
class BaseOrderSchema(BaseModel):
    """Base schema with common configuration."""
    
    model_config = ConfigDict(
        from_attributes=True,
        arbitrary_types_allowed=True,
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )


# Order creation and management schemas
class OrderCreate(BaseOrderSchema):
    """Schema for creating a new order from a signal."""
    signal_id: int = Field(..., gt=0, description="Signal ID to execute")
    order_type: OrderTypeEnum = Field(OrderTypeEnum.MARKET, description="Type of order")
    quantity: int = Field(..., gt=0, description="Order quantity in lots")
    limit_price: Optional[float] = Field(None, gt=0, description="Limit price (for limit orders)")
    stop_price: Optional[float] = Field(None, gt=0, description="Stop price (for stop orders)")
    expiry_time: Optional[datetime] = Field(None, description="Order expiry time (optional)")
    order_notes: Optional[str] = Field(None, max_length=500, description="Order notes")
    
    # Simulation parameters
    slippage_model: SlippageModelEnum = Field(SlippageModelEnum.FIXED, description="Slippage model to use")
    max_slippage_bps: Optional[float] = Field(50, ge=0, le=500, description="Maximum allowed slippage in basis points")
    execution_delay_ms: int = Field(0, ge=0, le=5000, description="Simulated execution delay in milliseconds")
    
    @model_validator(mode='after')
    def validate_price_requirements(self) -> 'OrderCreate':
        """Validate price requirements based on order type."""
        if self.order_type in [OrderTypeEnum.LIMIT, OrderTypeEnum.STOP_LIMIT] and not self.limit_price:
            raise ValueError("Limit price required for limit orders")
        
        if self.order_type in [OrderTypeEnum.STOP_LOSS, OrderTypeEnum.STOP_LIMIT] and not self.stop_price:
            raise ValueError("Stop price required for stop orders")
        
        if self.order_type == OrderTypeEnum.MARKET and (self.limit_price or self.stop_price):
            raise ValueError("Market orders cannot have limit or stop prices")
        
        return self


class OrderUpdate(BaseOrderSchema):
    """Schema for updating order parameters."""
    quantity: Optional[int] = Field(None, gt=0, description="New order quantity")
    limit_price: Optional[float] = Field(None, gt=0, description="New limit price")
    stop_price: Optional[float] = Field(None, gt=0, description="New stop price")
    expiry_time: Optional[datetime] = Field(None, description="New expiry time")
    order_notes: Optional[str] = Field(None, max_length=500, description="Updated notes")


class OrderCancel(BaseOrderSchema):
    """Schema for cancelling an order."""
    cancellation_reason: str = Field(..., max_length=200, description="Reason for cancellation")


# Fill schemas
class OrderFillCreate(BaseOrderSchema):
    """Schema for creating an order fill (internal use)."""
    order_id: int = Field(..., gt=0, description="Order ID")
    fill_quantity: int = Field(..., gt=0, description="Quantity filled")
    fill_price: float = Field(..., gt=0, description="Fill price")
    fill_type: FillTypeEnum = Field(..., description="Type of fill")
    market_price: float = Field(..., gt=0, description="Market price at fill time")
    bid_price: Optional[float] = Field(None, gt=0, description="Best bid at fill")
    ask_price: Optional[float] = Field(None, gt=0, description="Best ask at fill")
    commission: float = Field(0, ge=0, description="Commission for this fill")
    taxes: float = Field(0, ge=0, description="Taxes for this fill")
    slippage: float = Field(0, description="Slippage for this fill")
    execution_delay_ms: int = Field(0, ge=0, description="Execution delay")
    
    @field_validator('fill_price', 'market_price', 'bid_price', 'ask_price')
    @classmethod
    def validate_positive_prices(cls, v):
        """Ensure all prices are positive."""
        if v is not None and v <= 0:
            raise ValueError("Price must be positive")
        return v


class OrderFillResponse(BaseOrderSchema):
    """Schema for order fill response."""
    id: int = Field(..., description="Fill ID")
    order_id: int = Field(..., description="Order ID")
    fill_quantity: int = Field(..., description="Quantity filled")
    fill_price: float = Field(..., description="Fill price")
    fill_time: datetime = Field(..., description="Fill timestamp")
    fill_type: FillTypeEnum = Field(..., description="Type of fill")
    commission: float = Field(..., description="Commission for this fill")
    taxes: float = Field(..., description="Taxes for this fill")
    slippage: float = Field(..., description="Slippage for this fill")
    market_price: float = Field(..., description="Market price at fill")
    bid_price: Optional[float] = Field(None, description="Best bid at fill")
    ask_price: Optional[float] = Field(None, description="Best ask at fill")
    spread_bps: Optional[float] = Field(None, description="Bid-ask spread in basis points")
    total_cost: float = Field(..., description="Total cost for this fill")
    fill_value_inr: float = Field(..., description="Fill value in INR")
    effective_price: float = Field(..., description="Effective price including slippage")


# Order response schemas
class OrderResponse(BaseOrderSchema):
    """Schema for order response from API."""
    id: int = Field(..., description="Order ID")
    strategy_id: int = Field(..., description="Strategy ID")
    signal_id: int = Field(..., description="Signal ID")
    instrument: str = Field(..., description="Trading instrument")
    order_type: OrderTypeEnum = Field(..., description="Order type")
    order_side: OrderSideEnum = Field(..., description="Order side (buy/sell)")
    order_status: OrderStatusEnum = Field(..., description="Current order status")
    
    # Quantities
    quantity: int = Field(..., description="Total order quantity")
    filled_quantity: int = Field(..., description="Quantity filled")
    remaining_quantity: int = Field(..., description="Remaining quantity")
    
    # Pricing
    limit_price: Optional[float] = Field(None, description="Limit price")
    stop_price: Optional[float] = Field(None, description="Stop price")
    average_fill_price: Optional[float] = Field(None, description="Average fill price")
    
    # Timing
    order_time: datetime = Field(..., description="Order creation time")
    submit_time: Optional[datetime] = Field(None, description="Submission time")
    first_fill_time: Optional[datetime] = Field(None, description="First fill time")
    last_fill_time: Optional[datetime] = Field(None, description="Last fill time")
    expiry_time: Optional[datetime] = Field(None, description="Order expiry time")
    
    # Costs and execution
    total_commission: float = Field(..., description="Total commission")
    total_taxes: float = Field(..., description="Total taxes")
    total_slippage: float = Field(..., description="Total slippage")
    total_costs: float = Field(..., description="Total trading costs")
    risk_amount_inr: float = Field(..., description="Risk amount in INR")
    
    # Status information
    rejection_reason: Optional[str] = Field(None, description="Rejection reason")
    cancellation_reason: Optional[str] = Field(None, description="Cancellation reason")
    
    # Computed properties
    is_active: bool = Field(..., description="Whether order is still active")
    is_filled: bool = Field(..., description="Whether order is completely filled")
    is_partially_filled: bool = Field(..., description="Whether order has partial fills")
    fill_percentage: float = Field(..., description="Fill percentage")
    
    # Fills
    fills: List[OrderFillResponse] = Field(default_factory=list, description="Order fills")


class OrderSummary(BaseOrderSchema):
    """Summary schema for order lists."""
    id: int = Field(..., description="Order ID")
    instrument: str = Field(..., description="Trading instrument")
    order_type: OrderTypeEnum = Field(..., description="Order type")
    order_side: OrderSideEnum = Field(..., description="Order side")
    order_status: OrderStatusEnum = Field(..., description="Order status")
    quantity: int = Field(..., description="Order quantity")
    filled_quantity: int = Field(..., description="Filled quantity")
    average_fill_price: Optional[float] = Field(None, description="Average fill price")
    order_time: datetime = Field(..., description="Order time")
    total_costs: float = Field(..., description="Total costs")
    fill_percentage: float = Field(..., description="Fill percentage")


# Batch operations
class BatchOrderCreate(BaseOrderSchema):
    """Schema for creating multiple orders in batch."""
    orders: List[OrderCreate] = Field(..., min_length=1, max_length=50, description="List of orders to create")
    execution_mode: str = Field("sequential", description="Execution mode (sequential/parallel)")
    stop_on_error: bool = Field(True, description="Stop processing on first error")


class BatchOrderResponse(BaseOrderSchema):
    """Schema for batch order creation response."""
    successful_orders: List[OrderResponse] = Field(..., description="Successfully created orders")
    failed_orders: List[Dict[str, Any]] = Field(..., description="Failed order creations with errors")
    total_submitted: int = Field(..., description="Total orders submitted")
    successful_count: int = Field(..., description="Number of successful orders")
    failed_count: int = Field(..., description="Number of failed orders")


# Risk and validation schemas
class OrderRiskCheck(BaseOrderSchema):
    """Schema for order risk validation."""
    signal_id: int = Field(..., gt=0, description="Signal ID")
    quantity: int = Field(..., gt=0, description="Proposed quantity")
    order_type: OrderTypeEnum = Field(..., description="Order type")
    max_risk_percent: float = Field(2.0, ge=0.1, le=10.0, description="Maximum risk percentage")
    check_correlation: bool = Field(True, description="Check position correlation")
    check_daily_limits: bool = Field(True, description="Check daily trading limits")


class OrderRiskResult(BaseOrderSchema):
    """Schema for order risk check result."""
    is_approved: bool = Field(..., description="Whether order passes risk checks")
    risk_amount_inr: float = Field(..., description="Calculated risk amount")
    risk_percentage: float = Field(..., description="Risk as percentage of account")
    warnings: List[str] = Field(default_factory=list, description="Risk warnings")
    blocking_issues: List[str] = Field(default_factory=list, description="Issues that block order")
    recommended_quantity: Optional[int] = Field(None, description="Recommended quantity if different")
    margin_required: Optional[float] = Field(None, description="Margin required")


# Market simulation schemas
class MarketConditions(BaseOrderSchema):
    """Schema for current market conditions."""
    instrument: str = Field(..., description="Trading instrument")
    current_price: float = Field(..., gt=0, description="Current market price")
    bid_price: float = Field(..., gt=0, description="Best bid price")
    ask_price: float = Field(..., gt=0, description="Best ask price")
    volume: int = Field(..., ge=0, description="Current volume")
    volatility: float = Field(..., ge=0, description="Current volatility")
    liquidity_score: float = Field(..., ge=0, le=1, description="Liquidity score (0-1)")
    timestamp: datetime = Field(..., description="Market data timestamp")


class ExecutionSimulationConfig(BaseOrderSchema):
    """Configuration for execution simulation."""
    slippage_model: SlippageModelEnum = Field(SlippageModelEnum.FIXED, description="Slippage model")
    base_slippage_bps: float = Field(10, ge=0, le=100, description="Base slippage in basis points")
    volume_impact_factor: float = Field(0.1, ge=0, le=1, description="Volume impact factor")
    latency_ms: int = Field(100, ge=0, le=2000, description="Execution latency in milliseconds")
    commission_per_lot: float = Field(20, ge=0, description="Commission per lot in INR")
    tax_rate: float = Field(0.1, ge=0, le=1, description="Tax rate as percentage")
    market_hours_only: bool = Field(True, description="Execute only during market hours")
    weekend_execution: bool = Field(False, description="Allow weekend execution")


# Analytics and reporting schemas
class OrderAnalytics(BaseOrderSchema):
    """Schema for order execution analytics."""
    total_orders: int = Field(..., description="Total number of orders")
    filled_orders: int = Field(..., description="Number of filled orders")
    cancelled_orders: int = Field(..., description="Number of cancelled orders")
    rejected_orders: int = Field(..., description="Number of rejected orders")
    average_fill_time_seconds: float = Field(..., description="Average time to fill")
    average_slippage_bps: float = Field(..., description="Average slippage in basis points")
    total_commission_inr: float = Field(..., description="Total commission paid")
    total_taxes_inr: float = Field(..., description="Total taxes paid")
    execution_success_rate: float = Field(..., description="Execution success rate percentage")
    partial_fill_rate: float = Field(..., description="Partial fill rate percentage")


class ExecutionQualityMetrics(BaseOrderSchema):
    """Schema for execution quality metrics."""
    instrument: str = Field(..., description="Trading instrument")
    time_period: str = Field(..., description="Analysis time period")
    total_volume: int = Field(..., description="Total volume executed")
    vwap: float = Field(..., description="Volume weighted average price")
    implementation_shortfall: float = Field(..., description="Implementation shortfall in basis points")
    market_impact: float = Field(..., description="Market impact in basis points")
    timing_cost: float = Field(..., description="Timing cost in basis points")
    slippage_distribution: Dict[str, float] = Field(..., description="Slippage distribution statistics")
    fill_rate_by_time: Dict[str, float] = Field(..., description="Fill rate by time of day")