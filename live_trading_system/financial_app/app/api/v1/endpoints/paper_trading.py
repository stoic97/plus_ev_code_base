"""
Order Execution API endpoints for Simulation-Based Trading System.

This module provides RESTful endpoints for:
- Signal-to-order execution
- Order lifecycle management (create, modify, cancel)
- Risk validation and pre-trade checks
- Order status monitoring and analytics
- Execution quality metrics and reporting
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, Path, status, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.core.database import get_db as get_postgres_db
from app.core.error_handling import (
    DatabaseConnectionError,
    OperationalError,
    ValidationError,
    AuthenticationError,
)
from app.services.strategy_engine import StrategyEngineService
from app.schemas.strategy import (
    TradeCreate, TradeResponse, SignalResponse, TradeBase,
    FeedbackCreate, FeedbackResponse
)

def get_strategy_service() -> StrategyEngineService:
    # Use the existing StrategyEngineService instead
    """
    Dependency to create OrderExecutionService instance.
    
    Args:
        db: Database session from dependency injection
        
    Returns:
        OrderExecutionService instance
    """
    try:
        # Default execution configuration
        config = ExecutionSimulationConfig()
        return OrderExecutionService(db, config)
    except Exception as e:
        logger.error(f"Failed to create OrderExecutionService: {e}")
        raise DatabaseConnectionError("Unable to connect to order execution service")


def get_strategy_service(
    db: Session = Depends(get_postgres_db)
) -> StrategyEngineService:
    """Dependency to create StrategyEngineService instance."""
    try:
        return StrategyEngineService(db)
    except Exception as e:
        logger.error(f"Failed to create StrategyEngineService: {e}")
        raise DatabaseConnectionError("Unable to connect to strategy service")


def get_current_user_id() -> int:
    """
    Dependency to get current authenticated user ID.
    
    TODO: Implement actual user extraction from auth middleware
    """
    return 1  # Placeholder


#
# Signal Execution Endpoints
#

@router.post(
    "/signals/{signal_id}/execute",
    response_model=OrderResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Execute trading signal",
    description="Convert a trading signal into an executable order with risk validation"
)
async def execute_signal(
    signal_id: int = Path(..., gt=0, description="Signal ID to execute"),
    order_params: Optional[OrderCreate] = None,
    execution_service: OrderExecutionService = Depends(get_order_execution_service),
    user_id: int = Depends(get_current_user_id)
) -> OrderResponse:
    """
    Execute a trading signal by creating and processing an order.
    
    This endpoint bridges signal generation and trade execution by:
    1. Validating the signal exists and is executable
    2. Performing comprehensive pre-trade risk checks
    3. Creating an order with proper execution parameters
    4. Submitting the order to the simulation execution engine
    5. Returning detailed execution status and tracking information
    
    Args:
        signal_id: ID of the signal to execute
        order_params: Optional order parameters (defaults to signal settings)
        execution_service: Order execution service instance
        user_id: Current authenticated user ID
        
    Returns:
        OrderResponse with execution details and tracking info
        
    Raises:
        HTTPException: If signal validation or execution fails
    """
    try:
        logger.info(f"Executing signal {signal_id} for user {user_id}")
        
        order_response = await execution_service.execute_signal(
            signal_id=signal_id,
            user_id=user_id,
            order_params=order_params
        )
        
        logger.info(f"Successfully executed signal {signal_id}, created order {order_response.id}")
        return order_response
        
    except ValidationError as e:
        logger.error(f"Validation error executing signal {signal_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error executing signal {signal_id}: {e}")
        raise OperationalError(f"Failed to execute signal: {str(e)}")


#
# Risk Management Endpoints
#

@router.post(
    "/risk/check",
    response_model=OrderRiskResult,
    summary="Check order risk",
    description="Perform pre-trade risk validation without creating an order"
)
async def check_order_risk(
    risk_check: OrderRiskCheck,
    execution_service: OrderExecutionService = Depends(get_order_execution_service),
    user_id: int = Depends(get_current_user_id)
) -> OrderRiskResult:
    """
    Perform comprehensive pre-trade risk validation.
    
    This endpoint allows testing risk parameters before order creation:
    - Position size validation against account limits
    - Risk percentage calculation and validation
    - Correlation checks with existing positions
    - Daily trading limit validation
    - Margin requirement calculation
    
    Args:
        risk_check: Risk check parameters
        execution_service: Order execution service instance
        user_id: Current authenticated user ID
        
    Returns:
        OrderRiskResult with approval status and detailed feedback
    """
    try:
        logger.info(f"Checking order risk for signal {risk_check.signal_id}, user {user_id}")
        
        risk_result = await execution_service.check_order_risk(
            signal_id=risk_check.signal_id,
            quantity=risk_check.quantity,
            order_type=risk_check.order_type,
            user_id=user_id
        )
        
        logger.info(f"Risk check complete: approved={risk_result.is_approved}")
        return risk_result
        
    except ValidationError as e:
        logger.error(f"Validation error in risk check: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error in risk check: {e}")
        raise OperationalError(f"Failed to check order risk: {str(e)}")


#
# Order Management Endpoints
#

@router.get(
    "/orders/",
    response_model=List[OrderSummary],
    summary="List orders",
    description="List orders with optional filtering and pagination"
)
async def list_orders(
    status_filter: Optional[List[OrderStatusEnum]] = Query(None, description="Filter by order status"),
    instrument_filter: Optional[str] = Query(None, description="Filter by trading instrument"),
    strategy_filter: Optional[int] = Query(None, gt=0, description="Filter by strategy ID"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    execution_service: OrderExecutionService = Depends(get_order_execution_service),
    user_id: int = Depends(get_current_user_id)
) -> List[OrderResponse]:
    """
    List orders with comprehensive filtering options.
    
    Args:
        status_filter: Filter by order status (active, filled, cancelled, etc.)
        instrument_filter: Filter by trading instrument (NIFTY, BANKNIFTY, etc.)
        strategy_filter: Filter by strategy ID
        limit: Maximum number of results
        offset: Pagination offset
        execution_service: Order execution service instance
        user_id: Current authenticated user ID
        
    Returns:
        List of orders matching the specified filters
    """
    try:
        logger.info(f"Listing orders for user {user_id} with filters")
        
        orders = execution_service.list_orders(
            user_id=user_id,
            status_filter=status_filter,
            instrument_filter=instrument_filter,
            strategy_filter=strategy_filter,
            limit=limit,
            offset=offset
        )
        
        logger.info(f"Retrieved {len(orders)} orders for user {user_id}")
        return orders
        
    except Exception as e:
        logger.error(f"Error listing orders: {e}")
        raise OperationalError(f"Failed to list orders: {str(e)}")


@router.get(
    "/orders/{order_id}",
    response_model=OrderResponse,
    summary="Get order details",
    description="Get detailed information about a specific order including fills"
)
async def get_order(
    order_id: int = Path(..., gt=0, description="Order ID"),
    execution_service: OrderExecutionService = Depends(get_order_execution_service),
    user_id: int = Depends(get_current_user_id)
) -> OrderResponse:
    """
    Get comprehensive details for a specific order.
    
    Args:
        order_id: ID of the order to retrieve
        execution_service: Order execution service instance
        user_id: Current authenticated user ID
        
    Returns:
        OrderResponse with complete order details and fill history
        
    Raises:
        HTTPException: If order not found or access denied
    """
    try:
        logger.info(f"Getting order {order_id} for user {user_id}")
        
        order = execution_service.get_order(order_id, user_id)
        
        logger.info(f"Retrieved order {order_id}")
        return order
        
    except ValidationError as e:
        if "not found" in str(e).lower():
            logger.error(f"Order {order_id} not found for user {user_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Order {order_id} not found"
            )
        else:
            logger.error(f"Validation error getting order {order_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
    except Exception as e:
        logger.error(f"Error getting order {order_id}: {e}")
        raise OperationalError(f"Failed to get order: {str(e)}")


@router.put(
    "/orders/{order_id}",
    response_model=OrderResponse,
    summary="Modify order",
    description="Modify parameters of an active order"
)
async def modify_order(
    order_id: int = Path(..., gt=0, description="Order ID"),
    modifications: OrderUpdate = ...,
    execution_service: OrderExecutionService = Depends(get_order_execution_service),
    user_id: int = Depends(get_current_user_id)
) -> OrderResponse:
    """
    Modify an active order's parameters.
    
    Only active orders (pending, submitted, acknowledged, partially filled) can be modified.
    Modifications include quantity, limit price, stop price, and expiry time.
    
    Args:
        order_id: ID of the order to modify
        modifications: Order modification parameters
        execution_service: Order execution service instance
        user_id: Current authenticated user ID
        
    Returns:
        OrderResponse with updated order details
        
    Raises:
        HTTPException: If order not found, not modifiable, or validation fails
    """
    try:
        logger.info(f"Modifying order {order_id} for user {user_id}")
        
        # Convert Pydantic model to dict for modifications
        modification_dict = {
            k: v for k, v in modifications.model_dump().items() 
            if v is not None
        }
        
        order = await execution_service.modify_order(
            order_id=order_id,
            user_id=user_id,
            modifications=modification_dict
        )
        
        logger.info(f"Successfully modified order {order_id}")
        return order
        
    except ValidationError as e:
        logger.error(f"Validation error modifying order {order_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error modifying order {order_id}: {e}")
        raise OperationalError(f"Failed to modify order: {str(e)}")


@router.delete(
    "/orders/{order_id}",
    response_model=OrderResponse,
    summary="Cancel order",
    description="Cancel an active order"
)
async def cancel_order(
    order_id: int = Path(..., gt=0, description="Order ID"),
    cancellation: OrderCancel = ...,
    execution_service: OrderExecutionService = Depends(get_order_execution_service),
    user_id: int = Depends(get_current_user_id)
) -> OrderResponse:
    """
    Cancel an active order.
    
    Only active orders can be cancelled. Once cancelled, orders cannot be reactivated.
    
    Args:
        order_id: ID of the order to cancel
        cancellation: Cancellation details including reason
        execution_service: Order execution service instance
        user_id: Current authenticated user ID
        
    Returns:
        OrderResponse with cancelled order details
        
    Raises:
        HTTPException: If order not found or not cancellable
    """
    try:
        logger.info(f"Cancelling order {order_id} for user {user_id}")
        
        order = await execution_service.cancel_order(
            order_id=order_id,
            user_id=user_id,
            reason=cancellation.cancellation_reason
        )
        
        logger.info(f"Successfully cancelled order {order_id}")
        return order
        
    except ValidationError as e:
        logger.error(f"Validation error cancelling order {order_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error cancelling order {order_id}: {e}")
        raise OperationalError(f"Failed to cancel order: {str(e)}")


#
# Batch Operations
#

@router.post(
    "/orders/batch",
    response_model=BatchOrderResponse,
    summary="Create multiple orders",
    description="Create multiple orders in a single batch operation"
)
async def create_batch_orders(
    batch_request: BatchOrderCreate,
    execution_service: OrderExecutionService = Depends(get_order_execution_service),
    user_id: int = Depends(get_current_user_id)
) -> BatchOrderResponse:
    """
    Create multiple orders in a batch operation.
    
    Useful for executing multiple signals simultaneously or creating complex
    multi-leg strategies.
    
    Args:
        batch_request: Batch order creation request
        execution_service: Order execution service instance
        user_id: Current authenticated user ID
        
    Returns:
        BatchOrderResponse with results for each order
    """
    try:
        logger.info(f"Creating batch of {len(batch_request.orders)} orders for user {user_id}")
        
        successful_orders = []
        failed_orders = []
        
        for i, order_create in enumerate(batch_request.orders):
            try:
                order_response = await execution_service.execute_signal(
                    signal_id=order_create.signal_id,
                    user_id=user_id,
                    order_params=order_create
                )
                successful_orders.append(order_response)
                
            except Exception as e:
                failed_orders.append({
                    "index": i,
                    "signal_id": order_create.signal_id,
                    "error": str(e)
                })
                
                if batch_request.stop_on_error:
                    break
        
        response = BatchOrderResponse(
            successful_orders=successful_orders,
            failed_orders=failed_orders,
            total_submitted=len(batch_request.orders),
            successful_count=len(successful_orders),
            failed_count=len(failed_orders)
        )
        
        logger.info(f"Batch processing complete: {len(successful_orders)} successful, {len(failed_orders)} failed")
        return response
        
    except Exception as e:
        logger.error(f"Error in batch order creation: {e}")
        raise OperationalError(f"Failed to create batch orders: {str(e)}")


#
# Analytics and Monitoring
#

@router.get(
    "/analytics/execution",
    response_model=Dict[str, Any],
    summary="Get execution analytics",
    description="Get comprehensive execution performance analytics"
)
async def get_execution_analytics(
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    execution_service: OrderExecutionService = Depends(get_order_execution_service),
    user_id: int = Depends(get_current_user_id)
) -> Dict[str, Any]:
    """
    Get execution performance analytics and metrics.
    
    Provides insights into:
    - Order fill rates and execution quality
    - Average slippage and transaction costs
    - Execution timing and latency metrics
    - Success rates by order type and instrument
    
    Args:
        days: Number of days to include in analysis
        execution_service: Order execution service instance
        user_id: Current authenticated user ID
        
    Returns:
        Dictionary with comprehensive execution analytics
    """
    try:
        logger.info(f"Getting execution analytics for user {user_id}, {days} days")
        
        analytics = execution_service.get_execution_analytics(user_id, days)
        
        logger.info(f"Generated execution analytics for user {user_id}")
        return analytics
        
    except Exception as e:
        logger.error(f"Error getting execution analytics: {e}")
        raise OperationalError(f"Failed to get execution analytics: {str(e)}")


@router.get(
    "/orders/active",
    response_model=List[OrderResponse],
    summary="Get active orders",
    description="Get all currently active orders (pending, submitted, acknowledged, partially filled)"
)
async def get_active_orders(
    execution_service: OrderExecutionService = Depends(get_order_execution_service),
    user_id: int = Depends(get_current_user_id)
) -> List[OrderResponse]:
    """
    Get all currently active orders for monitoring and management.
    
    Args:
        execution_service: Order execution service instance
        user_id: Current authenticated user ID
        
    Returns:
        List of active orders
    """
    try:
        logger.info(f"Getting active orders for user {user_id}")
        
        active_statuses = [
            OrderStatusEnum.PENDING,
            OrderStatusEnum.SUBMITTED,
            OrderStatusEnum.ACKNOWLEDGED,
            OrderStatusEnum.PARTIALLY_FILLED
        ]
        
        orders = execution_service.list_orders(
            user_id=user_id,
            status_filter=active_statuses,
            limit=1000
        )
        
        logger.info(f"Retrieved {len(orders)} active orders for user {user_id}")
        return orders
        
    except Exception as e:
        logger.error(f"Error getting active orders: {e}")
        raise OperationalError(f"Failed to get active orders: {str(e)}")


#
# Background Tasks and System Management
#

@router.post(
    "/system/process-orders",
    summary="Process pending orders",
    description="Trigger background processing of pending orders (admin/system use)"
)
async def process_pending_orders(
    background_tasks: BackgroundTasks,
    execution_service: OrderExecutionService = Depends(get_order_execution_service)
):
    """
    Trigger background processing of all pending orders.
    
    This endpoint is typically called by system schedulers or for manual
    order processing in simulation environments.
    
    Args:
        background_tasks: FastAPI background tasks
        execution_service: Order execution service instance
        
    Returns:
        Success message
    """
    try:
        logger.info("Triggering background order processing")
        
        background_tasks.add_task(execution_service.process_pending_orders)
        
        return {"message": "Order processing triggered successfully"}
        
    except Exception as e:
        logger.error(f"Error triggering order processing: {e}")
        raise OperationalError(f"Failed to trigger order processing: {str(e)}")


#
# Configuration and Simulation Management
#

@router.put(
    "/simulation/config",
    response_model=Dict[str, Any],
    summary="Update simulation configuration",
    description="Update execution simulation parameters"
)
async def update_simulation_config(
    config: ExecutionSimulationConfig,
    execution_service: OrderExecutionService = Depends(get_order_execution_service)
) -> Dict[str, Any]:
    """
    Update simulation configuration parameters.
    
    Allows real-time adjustment of:
    - Slippage models and parameters
    - Execution delays and latency
    - Commission and tax rates
    - Market impact factors
    
    Args:
        config: New simulation configuration
        execution_service: Order execution service instance
        
    Returns:
        Confirmation of updated configuration
    """
    try:
        logger.info("Updating simulation configuration")
        
        # Update the service configuration
        execution_service.execution_config = config
        execution_service.market_simulator.config = config
        
        logger.info("Simulation configuration updated successfully")
        return {
            "message": "Simulation configuration updated successfully",
            "config": config.model_dump()
        }
        
    except Exception as e:
        logger.error(f"Error updating simulation config: {e}")
        raise OperationalError(f"Failed to update simulation config: {str(e)}")


@router.get(
    "/simulation/config",
    response_model=ExecutionSimulationConfig,
    summary="Get simulation configuration",
    description="Get current execution simulation parameters"
)
async def get_simulation_config(
    execution_service: OrderExecutionService = Depends(get_order_execution_service)
) -> ExecutionSimulationConfig:
    """
    Get current simulation configuration.
    
    Args:
        execution_service: Order execution service instance
        
    Returns:
        Current simulation configuration
    """
    try:
        logger.info("Getting simulation configuration")
        return execution_service.execution_config
        
    except Exception as e:
        logger.error(f"Error getting simulation config: {e}")
        raise OperationalError(f"Failed to get simulation config: {str(e)}")