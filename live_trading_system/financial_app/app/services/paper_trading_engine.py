"""
Order Execution Service for Simulation-Based Trading System

This service orchestrates the complete order execution flow in our simulation environment:
1. Takes signals from StrategyEngineService
2. Performs comprehensive pre-trade risk checks
3. Creates and manages orders through their lifecycle
4. Simulates realistic market execution with slippage, delays, and costs
5. Creates trade records when orders are filled
6. Provides execution analytics and quality metrics

The service bridges the gap between signal generation and trade creation,
allowing for realistic testing of trading strategies without capital risk.
"""

import logging
import asyncio
import random
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from decimal import Decimal, ROUND_HALF_UP
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func

from app.models.order import Order, OrderFill, OrderType, OrderStatus, OrderSide, FillType
from app.models.strategy import Signal, Trade, Strategy, Direction
from app.schemas.order import (
    OrderCreate, OrderResponse, OrderFillCreate, OrderFillResponse,
    OrderRiskCheck, OrderRiskResult, MarketConditions, ExecutionSimulationConfig,
    SlippageModelEnum, OrderTypeEnum, OrderStatusEnum
)
from app.core.error_handling import ValidationError, OperationalError


# Set up logging
logger = logging.getLogger(__name__)


class OrderExecutionService:
    """
    Main service for order execution and lifecycle management in simulation environment.
    
    This service handles the complete order flow from signal reception to trade creation,
    including risk management, execution simulation, and performance tracking.
    """
    
    def __init__(self, db: Session, execution_config: Optional[ExecutionSimulationConfig] = None):
        """
        Initialize the Order Execution Service.
        
        Args:
            db: Database session
            execution_config: Configuration for execution simulation parameters
        """
        self.db = db
        self.execution_config = execution_config or ExecutionSimulationConfig()
        
        # Initialize simulation components
        self.risk_manager = RiskManager(db)
        self.market_simulator = MarketSimulator(self.execution_config)
        self.execution_engine = ExecutionEngine(db, self.market_simulator)
        
        # Tracking metrics
        self.execution_metrics = {
            "orders_created": 0,
            "orders_filled": 0,
            "orders_rejected": 0,
            "total_slippage": 0.0,
            "total_commission": 0.0
        }
    
    #
    # Core Order Execution Methods
    #
    
    async def execute_signal(
        self, 
        signal_id: int, 
        user_id: int,
        order_params: Optional[OrderCreate] = None
    ) -> OrderResponse:
        """
        Execute a trading signal by creating and processing an order.
        
        This is the main entry point for converting signals into orders.
        
        Args:
            signal_id: ID of the signal to execute
            user_id: User ID for authorization and risk management
            order_params: Optional order parameters (defaults to signal parameters)
            
        Returns:
            OrderResponse with execution details
            
        Raises:
            ValidationError: If signal validation fails
            OperationalError: If execution fails
        """
        try:
            logger.info(f"Executing signal {signal_id} for user {user_id}")
            
            # 1. Validate and retrieve signal
            signal = await self._validate_signal(signal_id, user_id)
            
            # 2. Create order parameters if not provided
            if not order_params:
                order_params = self._create_order_from_signal(signal)
            
            # 3. Perform pre-trade risk checks
            risk_result = await self.check_order_risk(
                signal_id=signal_id,
                quantity=order_params.quantity,
                order_type=order_params.order_type,
                user_id=user_id
            )
            
            if not risk_result.is_approved:
                raise ValidationError(f"Order rejected by risk management: {', '.join(risk_result.blocking_issues)}")
            
            # 4. Create order record
            order = await self._create_order(signal, order_params, user_id, risk_result)
            
            # 5. Submit order for execution
            await self._submit_order(order)
            
            # 6. Update metrics
            self.execution_metrics["orders_created"] += 1
            
            logger.info(f"Successfully created order {order.id} for signal {signal_id}")
            return self._order_to_response(order)
            
        except Exception as e:
            logger.error(f"Error executing signal {signal_id}: {e}")
            self.execution_metrics["orders_rejected"] += 1
            raise
    
    async def _validate_signal(self, signal_id: int, user_id: int) -> Signal:
        """Validate signal exists and is executable."""
        signal = self.db.query(Signal).filter(
            Signal.id == signal_id,
            Signal.user_id == user_id
        ).first()
        
        if not signal:
            raise ValidationError(f"Signal {signal_id} not found")
        
        if signal.is_executed:
            raise ValidationError(f"Signal {signal_id} has already been executed")
        
        if not signal.is_active:
            raise ValidationError(f"Signal {signal_id} is not active")
        
        return signal
    
    def _create_order_from_signal(self, signal: Signal) -> OrderCreate:
        """Create default order parameters from signal."""
        return OrderCreate(
            signal_id=signal.id,
            order_type=OrderTypeEnum.MARKET,  # Default to market orders
            quantity=signal.position_size,
            slippage_model=self.execution_config.slippage_model,
            max_slippage_bps=self.execution_config.base_slippage_bps,
            execution_delay_ms=self.execution_config.latency_ms
        )
    
    async def _create_order(
        self, 
        signal: Signal, 
        order_params: OrderCreate, 
        user_id: int,
        risk_result: OrderRiskResult
    ) -> Order:
        """Create order record in database."""
        
        # Determine order side from signal direction
        order_side = OrderSide.BUY if signal.direction == Direction.LONG else OrderSide.SELL
        
        order = Order(
            strategy_id=signal.strategy_id,
            signal_id=signal.id,
            instrument=signal.instrument,
            order_type=OrderType(order_params.order_type.value),
            order_side=order_side,
            order_status=OrderStatus.PENDING,
            quantity=order_params.quantity,
            filled_quantity=0,
            remaining_quantity=order_params.quantity,
            limit_price=order_params.limit_price,
            stop_price=order_params.stop_price,
            order_time=datetime.utcnow(),
            expiry_time=order_params.expiry_time,
            risk_amount_inr=risk_result.risk_amount_inr,
            margin_required=risk_result.margin_required,
            slippage_model=order_params.slippage_model.value,
            execution_delay_ms=order_params.execution_delay_ms,
            order_notes=order_params.order_notes,
            user_id=user_id
        )
        
        self.db.add(order)
        self.db.commit()
        self.db.refresh(order)
        
        return order
    
    async def _submit_order(self, order: Order):
        """Submit order to execution engine."""
        try:
            # Update order status
            order.order_status = OrderStatus.SUBMITTED
            order.submit_time = datetime.utcnow()
            
            # For market orders, execute immediately
            if order.order_type == OrderType.MARKET:
                await self.execution_engine.execute_market_order(order)
            else:
                # For limit/stop orders, add to order book simulation
                await self.execution_engine.add_to_order_book(order)
            
            self.db.commit()
            
        except Exception as e:
            order.order_status = OrderStatus.REJECTED
            order.rejection_reason = str(e)
            self.db.commit()
            raise
    
    #
    # Risk Management Methods
    #
    
    async def check_order_risk(
        self, 
        signal_id: int, 
        quantity: int, 
        order_type: OrderTypeEnum,
        user_id: int
    ) -> OrderRiskResult:
        """
        Perform comprehensive pre-trade risk checks.
        
        Args:
            signal_id: Signal ID
            quantity: Proposed order quantity
            order_type: Order type
            user_id: User ID
            
        Returns:
            OrderRiskResult with approval status and details
        """
        return await self.risk_manager.check_order_risk(
            signal_id, quantity, order_type, user_id
        )
    
    #
    # Order Management Methods
    #
    
    async def cancel_order(self, order_id: int, user_id: int, reason: str) -> OrderResponse:
        """Cancel an active order."""
        order = self.db.query(Order).filter(
            Order.id == order_id,
            Order.user_id == user_id
        ).first()
        
        if not order:
            raise ValidationError(f"Order {order_id} not found")
        
        if not order.is_active:
            raise ValidationError(f"Order {order_id} is not active and cannot be cancelled")
        
        order.order_status = OrderStatus.CANCELLED
        order.cancellation_reason = reason
        
        # Remove from order book if it's there
        await self.execution_engine.remove_from_order_book(order_id)
        
        self.db.commit()
        
        logger.info(f"Cancelled order {order_id}: {reason}")
        return self._order_to_response(order)
    
    async def modify_order(
        self, 
        order_id: int, 
        user_id: int, 
        modifications: Dict[str, Any]
    ) -> OrderResponse:
        """Modify an active order."""
        order = self.db.query(Order).filter(
            Order.id == order_id,
            Order.user_id == user_id
        ).first()
        
        if not order:
            raise ValidationError(f"Order {order_id} not found")
        
        if not order.is_active:
            raise ValidationError(f"Order {order_id} is not active and cannot be modified")
        
        # Apply modifications
        for field, value in modifications.items():
            if hasattr(order, field):
                setattr(order, field, value)
        
        # Re-validate order
        await self.execution_engine.revalidate_order(order)
        
        self.db.commit()
        
        logger.info(f"Modified order {order_id}")
        return self._order_to_response(order)
    
    def get_order(self, order_id: int, user_id: int) -> OrderResponse:
        """Get order details by ID."""
        order = self.db.query(Order).filter(
            Order.id == order_id,
            Order.user_id == user_id
        ).first()
        
        if not order:
            raise ValidationError(f"Order {order_id} not found")
        
        return self._order_to_response(order)
    
    def list_orders(
        self, 
        user_id: int,
        status_filter: Optional[List[OrderStatusEnum]] = None,
        instrument_filter: Optional[str] = None,
        strategy_filter: Optional[int] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[OrderResponse]:
        """List orders with optional filtering."""
        query = self.db.query(Order).filter(Order.user_id == user_id)
        
        if status_filter:
            status_values = [OrderStatus(s.value) for s in status_filter]
            query = query.filter(Order.order_status.in_(status_values))
        
        if instrument_filter:
            query = query.filter(Order.instrument == instrument_filter)
        
        if strategy_filter:
            query = query.filter(Order.strategy_id == strategy_filter)
        
        orders = query.order_by(Order.order_time.desc()).offset(offset).limit(limit).all()
        
        return [self._order_to_response(order) for order in orders]
    
    #
    # Execution Monitoring and Analytics
    #
    
    async def process_pending_orders(self):
        """Process all pending orders (background task)."""
        pending_orders = self.db.query(Order).filter(
            Order.order_status.in_([OrderStatus.SUBMITTED, OrderStatus.ACKNOWLEDGED])
        ).all()
        
        for order in pending_orders:
            try:
                await self.execution_engine.process_order(order)
            except Exception as e:
                logger.error(f"Error processing order {order.id}: {e}")
                order.order_status = OrderStatus.REJECTED
                order.rejection_reason = str(e)
        
        if pending_orders:
            self.db.commit()
    
    def get_execution_analytics(self, user_id: int, days: int = 30) -> Dict[str, Any]:
        """Get execution analytics for the user."""
        start_date = datetime.utcnow() - timedelta(days=days)
        
        orders = self.db.query(Order).filter(
            Order.user_id == user_id,
            Order.order_time >= start_date
        ).all()
        
        if not orders:
            return {"message": "No orders found in the specified period"}
        
        total_orders = len(orders)
        filled_orders = sum(1 for o in orders if o.order_status == OrderStatus.FILLED)
        cancelled_orders = sum(1 for o in orders if o.order_status == OrderStatus.CANCELLED)
        rejected_orders = sum(1 for o in orders if o.order_status == OrderStatus.REJECTED)
        
        # Calculate average metrics
        filled_order_objects = [o for o in orders if o.order_status == OrderStatus.FILLED]
        
        avg_fill_time = 0
        avg_slippage = 0
        total_commission = sum(o.total_commission for o in orders)
        total_taxes = sum(o.total_taxes for o in orders)
        
        if filled_order_objects:
            fill_times = []
            slippages = []
            
            for order in filled_order_objects:
                if order.first_fill_time and order.submit_time:
                    fill_time = (order.first_fill_time - order.submit_time).total_seconds()
                    fill_times.append(fill_time)
                
                slippages.append(order.total_slippage)
            
            avg_fill_time = sum(fill_times) / len(fill_times) if fill_times else 0
            avg_slippage = sum(slippages) / len(slippages) if slippages else 0
        
        return {
            "period_days": days,
            "total_orders": total_orders,
            "filled_orders": filled_orders,
            "cancelled_orders": cancelled_orders,
            "rejected_orders": rejected_orders,
            "fill_rate_percent": (filled_orders / total_orders) * 100 if total_orders > 0 else 0,
            "average_fill_time_seconds": avg_fill_time,
            "average_slippage_points": avg_slippage,
            "total_commission_inr": total_commission,
            "total_taxes_inr": total_taxes,
            "total_costs_inr": total_commission + total_taxes
        }
    
    #
    # Helper Methods
    #
    
    def _order_to_response(self, order: Order) -> OrderResponse:
        """Convert Order model to OrderResponse schema."""
        fills = [self._fill_to_response(fill) for fill in order.fills]
        
        return OrderResponse(
            id=order.id,
            strategy_id=order.strategy_id,
            signal_id=order.signal_id,
            instrument=order.instrument,
            order_type=OrderTypeEnum(order.order_type.value),
            order_side=order.order_side.value,
            order_status=OrderStatusEnum(order.order_status.value),
            quantity=order.quantity,
            filled_quantity=order.filled_quantity,
            remaining_quantity=order.remaining_quantity,
            limit_price=order.limit_price,
            stop_price=order.stop_price,
            average_fill_price=order.average_fill_price,
            order_time=order.order_time,
            submit_time=order.submit_time,
            first_fill_time=order.first_fill_time,
            last_fill_time=order.last_fill_time,
            expiry_time=order.expiry_time,
            total_commission=order.total_commission,
            total_taxes=order.total_taxes,
            total_slippage=order.total_slippage,
            total_costs=order.total_costs,
            risk_amount_inr=order.risk_amount_inr,
            rejection_reason=order.rejection_reason,
            cancellation_reason=order.cancellation_reason,
            is_active=order.is_active,
            is_filled=order.is_filled,
            is_partially_filled=order.is_partially_filled,
            fill_percentage=order.fill_percentage,
            fills=fills
        )
    
    def _fill_to_response(self, fill: OrderFill) -> OrderFillResponse:
        """Convert OrderFill model to OrderFillResponse schema."""
        return OrderFillResponse(
            id=fill.id,
            order_id=fill.order_id,
            fill_quantity=fill.fill_quantity,
            fill_price=fill.fill_price,
            fill_time=fill.fill_time,
            fill_type=FillType(fill.fill_type.value),
            commission=fill.commission,
            taxes=fill.taxes,
            slippage=fill.slippage,
            market_price=fill.market_price,
            bid_price=fill.bid_price,
            ask_price=fill.ask_price,
            spread_bps=fill.spread_bps,
            total_cost=fill.total_cost,
            fill_value_inr=fill.fill_value_inr,
            effective_price=fill.effective_price
        )


class RiskManager:
    """Risk management component for pre-trade validation."""
    
    def __init__(self, db: Session):
        self.db = db
    
    async def check_order_risk(
        self, 
        signal_id: int, 
        quantity: int, 
        order_type: OrderTypeEnum,
        user_id: int
    ) -> OrderRiskResult:
        """Perform comprehensive risk checks."""
        
        warnings = []
        blocking_issues = []
        
        # Get signal details
        signal = self.db.query(Signal).filter(Signal.id == signal_id).first()
        if not signal:
            blocking_issues.append("Signal not found")
            return OrderRiskResult(
                is_approved=False,
                risk_amount_inr=0,
                risk_percentage=0,
                warnings=warnings,
                blocking_issues=blocking_issues,
                recommended_quantity=None,
                margin_required=None
            )
        
        # Calculate risk amount
        risk_points = abs(signal.entry_price - signal.stop_loss_price)
        lot_size = 50  # Example lot size
        risk_amount_inr = risk_points * lot_size * quantity
        
        # Check account balance (example)
        account_balance = 1000000  # 10 lakh INR
        risk_percentage = (risk_amount_inr / account_balance) * 100
        
        # Risk checks
        if risk_percentage > 5.0:  # Max 5% risk per trade
            blocking_issues.append(f"Risk {risk_percentage:.1f}% exceeds maximum 5%")
        
        if risk_percentage > 3.0:
            warnings.append(f"High risk trade: {risk_percentage:.1f}%")
        
        # Position size checks
        max_position_size = 5  # Max 5 lots per trade
        if quantity > max_position_size:
            blocking_issues.append(f"Quantity {quantity} exceeds maximum {max_position_size}")
        
        # Check daily limits
        today_orders = self.db.query(Order).filter(
            Order.user_id == user_id,
            func.date(Order.order_time) == datetime.utcnow().date()
        ).count()
        
        if today_orders >= 10:  # Max 10 orders per day
            blocking_issues.append("Daily order limit reached")
        
        # Calculate margin (for derivatives)
        margin_required = risk_amount_inr * 0.1  # 10% margin requirement
        
        is_approved = len(blocking_issues) == 0
        
        return OrderRiskResult(
            is_approved=is_approved,
            risk_amount_inr=risk_amount_inr,
            risk_percentage=risk_percentage,
            warnings=warnings,
            blocking_issues=blocking_issues,
            recommended_quantity=max_position_size if quantity > max_position_size else None,
            margin_required=margin_required
        )


class MarketSimulator:
    """Market conditions simulator for realistic execution."""
    
    def __init__(self, config: ExecutionSimulationConfig):
        self.config = config
    
    def get_current_market_conditions(self, instrument: str) -> MarketConditions:
        """Simulate current market conditions."""
        
        # Base price (would come from real market data)
        base_price = 18500  # Example NIFTY price
        
        # Simulate bid-ask spread
        spread_bps = random.uniform(5, 20)  # 5-20 basis points spread
        spread_amount = base_price * (spread_bps / 10000)
        
        bid_price = base_price - (spread_amount / 2)
        ask_price = base_price + (spread_amount / 2)
        
        # Simulate volume and volatility
        volume = random.randint(1000, 10000)
        volatility = random.uniform(0.01, 0.05)  # 1-5% volatility
        
        # Liquidity score based on volume and spread
        liquidity_score = min(1.0, volume / 5000) * (1 - spread_bps / 100)
        
        return MarketConditions(
            instrument=instrument,
            current_price=base_price,
            bid_price=bid_price,
            ask_price=ask_price,
            volume=volume,
            volatility=volatility,
            liquidity_score=liquidity_score,
            timestamp=datetime.utcnow()
        )
    
    def calculate_execution_price(
        self, 
        order: Order, 
        market_conditions: MarketConditions
    ) -> Tuple[float, float]:
        """Calculate execution price and slippage."""
        
        base_price = market_conditions.current_price
        
        if order.order_side == OrderSide.BUY:
            # Buying at ask price + slippage
            execution_price = market_conditions.ask_price
        else:
            # Selling at bid price - slippage
            execution_price = market_conditions.bid_price
        
        # Calculate slippage based on model
        slippage = self._calculate_slippage(order, market_conditions)
        
        if order.order_side == OrderSide.BUY:
            execution_price += slippage
        else:
            execution_price -= slippage
        
        return execution_price, slippage
    
    def _calculate_slippage(self, order: Order, market_conditions: MarketConditions) -> float:
        """Calculate slippage based on configured model."""
        
        if order.slippage_model == "fixed":
            return self.config.base_slippage_bps / 100  # Convert bps to points
        
        elif order.slippage_model == "percentage":
            return market_conditions.current_price * (self.config.base_slippage_bps / 10000)
        
        elif order.slippage_model == "volume_based":
            # Higher slippage for larger orders relative to market volume
            volume_ratio = order.quantity / max(market_conditions.volume, 1)
            volume_impact = volume_ratio * self.config.volume_impact_factor
            base_slippage = market_conditions.current_price * (self.config.base_slippage_bps / 10000)
            return base_slippage * (1 + volume_impact)
        
        elif order.slippage_model == "spread_based":
            # Slippage based on bid-ask spread
            spread = market_conditions.ask_price - market_conditions.bid_price
            return spread * 0.5  # Half the spread as slippage
        
        else:
            return self.config.base_slippage_bps / 100


class ExecutionEngine:
    """Core execution engine for processing orders."""
    
    def __init__(self, db: Session, market_simulator: MarketSimulator):
        self.db = db
        self.market_simulator = market_simulator
        self.order_book = {}  # In-memory order book for limit orders
    
    async def execute_market_order(self, order: Order):
        """Execute a market order immediately."""
        
        # Simulate execution delay
        if order.execution_delay_ms > 0:
            await asyncio.sleep(order.execution_delay_ms / 1000)
        
        # Get market conditions
        market_conditions = self.market_simulator.get_current_market_conditions(order.instrument)
        
        # Calculate execution price
        execution_price, slippage = self.market_simulator.calculate_execution_price(
            order, market_conditions
        )
        
        # Create fill record
        fill = self._create_fill(order, order.quantity, execution_price, slippage, market_conditions)
        
        # Update order
        order.filled_quantity = order.quantity
        order.remaining_quantity = 0
        order.average_fill_price = execution_price
        order.total_slippage = slippage
        order.first_fill_time = fill.fill_time
        order.last_fill_time = fill.fill_time
        order.update_fill_status()
        
        # Create trade record
        await self._create_trade_from_order(order)
        
        logger.info(f"Executed market order {order.id} at price {execution_price}")
    
    def _create_fill(
        self, 
        order: Order, 
        quantity: int, 
        price: float, 
        slippage: float,
        market_conditions: MarketConditions
    ) -> OrderFill:
        """Create a fill record."""
        
        # Calculate costs
        commission = quantity * 20.0  # 20 INR per lot
        taxes = commission * 0.18    # 18% GST
        
        fill = OrderFill(
            order_id=order.id,
            fill_quantity=quantity,
            fill_price=price,
            fill_time=datetime.utcnow(),
            fill_type=FillType.MARKET,
            commission=commission,
            taxes=taxes,
            slippage=slippage,
            market_price=market_conditions.current_price,
            bid_price=market_conditions.bid_price,
            ask_price=market_conditions.ask_price,
            spread_bps=(market_conditions.ask_price - market_conditions.bid_price) / market_conditions.current_price * 10000,
            execution_venue="simulation",
            simulation_delay_ms=order.execution_delay_ms
        )
        
        # Update order totals
        order.total_commission += commission
        order.total_taxes += taxes
        
        self.db.add(fill)
        return fill
    
    async def _create_trade_from_order(self, order: Order):
        """Create trade record when order is filled."""
        
        # Get the signal for trade details
        signal = self.db.query(Signal).filter(Signal.id == order.signal_id).first()
        
        trade = Trade(
            strategy_id=order.strategy_id,
            signal_id=order.signal_id,
            order_id=order.id,
            instrument=order.instrument,
            direction=signal.direction,
            entry_price=order.average_fill_price,
            entry_time=order.first_fill_time or datetime.utcnow(),
            position_size=order.filled_quantity,
            commission=order.total_commission,
            taxes=order.total_taxes,
            slippage=order.total_slippage,
            initial_risk_points=abs(order.average_fill_price - signal.stop_loss_price),
            initial_risk_inr=order.risk_amount_inr,
            initial_risk_percent=(order.risk_amount_inr / 1000000) * 100,  # Assuming 10L account
            risk_reward_planned=signal.risk_reward_ratio,
            setup_quality=signal.setup_quality,
            setup_score=signal.setup_score,
            is_spread_trade=signal.is_spread_trade,
            spread_type=signal.spread_type,
            user_id=order.user_id
        )
        
        # Mark signal as executed
        signal.is_executed = True
        signal.execution_time = trade.entry_time
        
        self.db.add(trade)
        
        logger.info(f"Created trade {trade.id} from order {order.id}")
    
    async def add_to_order_book(self, order: Order):
        """Add limit/stop order to order book simulation."""
        # This would implement order book logic for limit orders
        # For now, we'll acknowledge the order
        order.order_status = OrderStatus.ACKNOWLEDGED
        logger.info(f"Added order {order.id} to order book")
    
    async def remove_from_order_book(self, order_id: int):
        """Remove order from order book."""
        if order_id in self.order_book:
            del self.order_book[order_id]
    
    async def process_order(self, order: Order):
        """Process an order (for background processing)."""
        if order.order_type == OrderType.MARKET:
            await self.execute_market_order(order)
        # Add logic for limit/stop orders
    
    async def revalidate_order(self, order: Order):
        """Revalidate order after modification."""
        # Add validation logic
        pass