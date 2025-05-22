"""
Trade Execution API endpoints for the Trading Strategies Application.

This module provides RESTful endpoints for trade execution, position management, 
trade lifecycle operations, performance analytics, batch operations, risk management,
and trader note-taking functionality.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query, Path, status
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
    TradeCreate,
    TradeResponse,
    SignalResponse,
    TradeBase,
    FeedbackCreate,
    FeedbackResponse
)

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


# Dependency functions
def get_strategy_service(
    db: Session = Depends(get_postgres_db)
) -> StrategyEngineService:
    """
    Dependency to create StrategyEngineService instance.
    
    Args:
        db: Database session from dependency injection
        
    Returns:
        StrategyEngineService instance
    """
    try:
        return StrategyEngineService(db.session())
    except Exception as e:
        logger.error(f"Failed to create StrategyEngineService: {e}")
        raise DatabaseConnectionError("Unable to connect to database service")


def get_current_user_id() -> int:
    """
    Dependency to get current authenticated user ID.
    
    This is a placeholder - replace with actual auth logic from your AuthMiddleware.
    For now, returns a default user ID.
    
    Returns:
        User ID
    """
    # TODO: Implement actual user extraction from auth middleware
    # This should extract user_id from request.state or JWT token
    return 1  # Placeholder


# Signal Execution Endpoints

@router.post(
    "/signals/{signal_id}/execute",
    response_model=TradeResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Execute a trading signal",
    description="Execute a trading signal by converting it to an active trade"
)
async def execute_signal(
    signal_id: int = Path(..., gt=0, description="Signal ID to execute"),
    execution_price: float = Query(..., gt=0, description="Actual execution price"),
    execution_time: Optional[datetime] = Query(None, description="Execution timestamp (defaults to now)"),
    service: StrategyEngineService = Depends(get_strategy_service),
    user_id: int = Depends(get_current_user_id)
) -> TradeResponse:
    """
    Execute a trading signal by converting it to an active trade.
    
    Args:
        signal_id: ID of the signal to execute
        execution_price: Actual price at which the trade was executed
        execution_time: Optional execution timestamp (defaults to current time)
        service: Strategy service instance
        user_id: Current authenticated user ID
        
    Returns:
        Created trade details with all execution information
        
    Raises:
        HTTPException: If signal not found, already executed, or access denied
        ValidationError: If execution parameters are invalid
    """
    try:
        logger.info(f"Executing signal {signal_id} at price {execution_price} for user {user_id}")
        
        # Execute signal using service
        trade = service.execute_signal(
            signal_id=signal_id,
            execution_price=execution_price,
            execution_time=execution_time,
            user_id=user_id
        )
        
        # Convert to response model
        trade_dict = {
            "id": trade.id,
            "strategy_id": trade.strategy_id,
            "signal_id": trade.signal_id,
            "instrument": trade.instrument,
            "direction": trade.direction,
            "entry_price": trade.entry_price,
            "entry_time": trade.entry_time,
            "exit_price": trade.exit_price,
            "exit_time": trade.exit_time,
            "exit_reason": trade.exit_reason,
            "position_size": trade.position_size,
            "commission": trade.commission,
            "taxes": trade.taxes,
            "slippage": trade.slippage,
            "profit_loss_points": trade.profit_loss_points,
            "profit_loss_inr": trade.profit_loss_inr,
            "initial_risk_points": trade.initial_risk_points,
            "initial_risk_inr": trade.initial_risk_inr,
            "initial_risk_percent": trade.initial_risk_percent,
            "risk_reward_planned": trade.risk_reward_planned,
            "actual_risk_reward": trade.actual_risk_reward,
            "setup_quality": trade.setup_quality,
            "setup_score": trade.setup_score,
            "holding_period_minutes": trade.holding_period_minutes,
            "total_costs": trade.total_costs,
            "is_spread_trade": trade.is_spread_trade,
            "spread_type": trade.spread_type
        }
        
        logger.info(f"Successfully executed signal {signal_id}, created trade {trade.id}")
        return TradeResponse(**trade_dict)
        
    except ValueError as e:
        if "not found" in str(e).lower():
            logger.error(f"Signal {signal_id} not found: {e}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Signal with ID {signal_id} not found"
            )
        elif "already executed" in str(e).lower():
            logger.error(f"Signal {signal_id} already executed: {e}")
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Signal with ID {signal_id} has already been executed"
            )
        else:
            logger.error(f"Validation error executing signal {signal_id}: {e}")
            raise ValidationError(str(e))
    except Exception as e:
        logger.error(f"Error executing signal {signal_id}: {e}")
        raise OperationalError(f"Failed to execute signal: {str(e)}")


# Trade Management Endpoints

@router.get(
    "/trades/",
    response_model=List[TradeResponse],
    summary="List trades",
    description="List all trades with optional filtering"
)
async def list_trades(
    strategy_id: Optional[int] = Query(None, gt=0, description="Filter by strategy ID"),
    instrument: Optional[str] = Query(None, description="Filter by trading instrument"),
    direction: Optional[str] = Query(None, description="Filter by trade direction (long/short)"),
    is_open: Optional[bool] = Query(None, description="Filter by open/closed status"),
    start_date: Optional[str] = Query(None, description="Start date for filtering (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date for filtering (YYYY-MM-DD)"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(100, ge=1, le=1000, description="Pagination limit"),
    service: StrategyEngineService = Depends(get_strategy_service),
    user_id: int = Depends(get_current_user_id)
) -> List[TradeResponse]:
    """
    List trades with optional filtering.
    
    Args:
        strategy_id: Optional filter by strategy ID
        instrument: Optional filter by trading instrument
        direction: Optional filter by trade direction
        is_open: Optional filter by open/closed status
        start_date: Optional start date filter (YYYY-MM-DD format)
        end_date: Optional end date filter (YYYY-MM-DD format)
        offset: Pagination offset
        limit: Maximum number of results
        service: Strategy service instance
        user_id: Current authenticated user ID
        
    Returns:
        List of trades matching filters
    """
    try:
        logger.info(f"Listing trades for user {user_id}, filters: strategy_id={strategy_id}, instrument={instrument}")
        
        # Parse dates if provided
        start_datetime = None
        end_datetime = None
        
        if start_date:
            try:
                start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
            except ValueError:
                raise ValidationError("Invalid start_date format. Use YYYY-MM-DD")
        
        if end_date:
            try:
                end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                raise ValidationError("Invalid end_date format. Use YYYY-MM-DD")
        
        # Build filter criteria
        filters = {
            "user_id": user_id,  # Always filter by current user
            "strategy_id": strategy_id,
            "instrument": instrument,
            "direction": direction,
            "is_open": is_open,
            "start_date": start_datetime,
            "end_date": end_datetime,
            "offset": offset,
            "limit": limit
        }
        
        # For now, use a simplified approach - in a real implementation,
        # you'd have a dedicated service method for filtered trade listing
        trades = service.list_trades(**{k: v for k, v in filters.items() if v is not None})
        
        # Convert to response models
        trade_responses = []
        for trade in trades:
            trade_dict = {
                "id": trade.id,
                "strategy_id": trade.strategy_id,
                "signal_id": trade.signal_id,
                "instrument": trade.instrument,
                "direction": trade.direction,
                "entry_price": trade.entry_price,
                "entry_time": trade.entry_time,
                "exit_price": trade.exit_price,
                "exit_time": trade.exit_time,
                "exit_reason": trade.exit_reason,
                "position_size": trade.position_size,
                "commission": trade.commission,
                "taxes": trade.taxes,
                "slippage": trade.slippage,
                "profit_loss_points": trade.profit_loss_points,
                "profit_loss_inr": trade.profit_loss_inr,
                "initial_risk_points": trade.initial_risk_points,
                "initial_risk_inr": trade.initial_risk_inr,
                "initial_risk_percent": trade.initial_risk_percent,
                "risk_reward_planned": trade.risk_reward_planned,
                "actual_risk_reward": trade.actual_risk_reward,
                "setup_quality": trade.setup_quality,
                "setup_score": trade.setup_score,
                "holding_period_minutes": trade.holding_period_minutes,
                "total_costs": trade.total_costs,
                "is_spread_trade": trade.is_spread_trade,
                "spread_type": trade.spread_type
            }
            trade_responses.append(TradeResponse(**trade_dict))
        
        logger.info(f"Found {len(trade_responses)} trades")
        return trade_responses
        
    except ValidationError:
        raise  # Re-raise validation errors as-is
    except Exception as e:
        logger.error(f"Error listing trades: {e}")
        raise OperationalError(f"Failed to list trades: {str(e)}")


@router.get(
    "/trades/{trade_id}",
    response_model=TradeResponse,
    summary="Get trade details",
    description="Get detailed information about a specific trade"
)
async def get_trade(
    trade_id: int = Path(..., gt=0, description="Trade ID"),
    service: StrategyEngineService = Depends(get_strategy_service),
    user_id: int = Depends(get_current_user_id)
) -> TradeResponse:
    """
    Get detailed information about a specific trade.
    
    Args:
        trade_id: ID of the trade to retrieve
        service: Strategy service instance
        user_id: Current authenticated user ID
        
    Returns:
        Trade details with all information
        
    Raises:
        HTTPException: If trade not found or access denied
    """
    try:
        logger.info(f"Getting trade {trade_id} for user {user_id}")
        
        trade = service.get_trade(trade_id)
        
        # Check access permissions (user can only access their own trades)
        if trade.user_id != user_id:
            logger.warning(f"User {user_id} attempted to access trade {trade_id} owned by {trade.user_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You can only access your own trades"
            )
        
        # Convert to response model
        trade_dict = {
            "id": trade.id,
            "strategy_id": trade.strategy_id,
            "signal_id": trade.signal_id,
            "instrument": trade.instrument,
            "direction": trade.direction,
            "entry_price": trade.entry_price,
            "entry_time": trade.entry_time,
            "exit_price": trade.exit_price,
            "exit_time": trade.exit_time,
            "exit_reason": trade.exit_reason,
            "position_size": trade.position_size,
            "commission": trade.commission,
            "taxes": trade.taxes,
            "slippage": trade.slippage,
            "profit_loss_points": trade.profit_loss_points,
            "profit_loss_inr": trade.profit_loss_inr,
            "initial_risk_points": trade.initial_risk_points,
            "initial_risk_inr": trade.initial_risk_inr,
            "initial_risk_percent": trade.initial_risk_percent,
            "risk_reward_planned": trade.risk_reward_planned,
            "actual_risk_reward": trade.actual_risk_reward,
            "setup_quality": trade.setup_quality,
            "setup_score": trade.setup_score,
            "holding_period_minutes": trade.holding_period_minutes,
            "total_costs": trade.total_costs,
            "is_spread_trade": trade.is_spread_trade,
            "spread_type": trade.spread_type
        }
        
        logger.info(f"Successfully retrieved trade {trade_id}")
        return TradeResponse(**trade_dict)
        
    except ValueError as e:
        logger.error(f"Trade {trade_id} not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trade with ID {trade_id} not found"
        )
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"Error getting trade {trade_id}: {e}")
        raise OperationalError(f"Failed to get trade: {str(e)}")


@router.put(
    "/trades/{trade_id}/close",
    response_model=TradeResponse,
    summary="Close trade",
    description="Close an open trade at the specified exit price"
)
async def close_trade(
    trade_id: int = Path(..., gt=0, description="Trade ID to close"),
    exit_price: float = Query(..., gt=0, description="Exit price for the trade"),
    exit_time: Optional[datetime] = Query(None, description="Exit timestamp (defaults to now)"),
    exit_reason: str = Query("manual", description="Reason for closing the trade"),
    service: StrategyEngineService = Depends(get_strategy_service),
    user_id: int = Depends(get_current_user_id)
) -> TradeResponse:
    """
    Close an open trade at the specified exit price.
    
    Args:
        trade_id: ID of the trade to close
        exit_price: Price at which the trade is being closed
        exit_time: Optional exit timestamp (defaults to current time)
        exit_reason: Reason for closing the trade
        service: Strategy service instance
        user_id: Current authenticated user ID
        
    Returns:
        Updated trade details with exit information
        
    Raises:
        HTTPException: If trade not found, already closed, or access denied
        ValidationError: If exit parameters are invalid
    """
    try:
        logger.info(f"Closing trade {trade_id} at price {exit_price} for user {user_id}")
        
        # First verify ownership
        existing_trade = service.get_trade(trade_id)
        if existing_trade.user_id != user_id:
            logger.warning(f"User {user_id} attempted to close trade {trade_id} owned by {existing_trade.user_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You can only close your own trades"
            )
        
        # Close trade using service
        closed_trade = service.close_trade(
            trade_id=trade_id,
            exit_price=exit_price,
            exit_time=exit_time,
            exit_reason=exit_reason
        )
        
        # Convert to response model
        trade_dict = {
            "id": closed_trade.id,
            "strategy_id": closed_trade.strategy_id,
            "signal_id": closed_trade.signal_id,
            "instrument": closed_trade.instrument,
            "direction": closed_trade.direction,
            "entry_price": closed_trade.entry_price,
            "entry_time": closed_trade.entry_time,
            "exit_price": closed_trade.exit_price,
            "exit_time": closed_trade.exit_time,
            "exit_reason": closed_trade.exit_reason,
            "position_size": closed_trade.position_size,
            "commission": closed_trade.commission,
            "taxes": closed_trade.taxes,
            "slippage": closed_trade.slippage,
            "profit_loss_points": closed_trade.profit_loss_points,
            "profit_loss_inr": closed_trade.profit_loss_inr,
            "initial_risk_points": closed_trade.initial_risk_points,
            "initial_risk_inr": closed_trade.initial_risk_inr,
            "initial_risk_percent": closed_trade.initial_risk_percent,
            "risk_reward_planned": closed_trade.risk_reward_planned,
            "actual_risk_reward": closed_trade.actual_risk_reward,
            "setup_quality": closed_trade.setup_quality,
            "setup_score": closed_trade.setup_score,
            "holding_period_minutes": closed_trade.holding_period_minutes,
            "total_costs": closed_trade.total_costs,
            "is_spread_trade": closed_trade.is_spread_trade,
            "spread_type": closed_trade.spread_type
        }
        
        logger.info(f"Successfully closed trade {trade_id}")
        return TradeResponse(**trade_dict)
        
    except ValueError as e:
        if "not found" in str(e).lower():
            logger.error(f"Trade {trade_id} not found: {e}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Trade with ID {trade_id} not found"
            )
        elif "already closed" in str(e).lower():
            logger.error(f"Trade {trade_id} already closed: {e}")
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Trade with ID {trade_id} is already closed"
            )
        else:
            logger.error(f"Validation error closing trade {trade_id}: {e}")
            raise ValidationError(str(e))
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"Error closing trade {trade_id}: {e}")
        raise OperationalError(f"Failed to close trade: {str(e)}")


# Position Management Endpoints

@router.get(
    "/positions/",
    response_model=List[TradeResponse],
    summary="Get open positions",
    description="Get all currently open positions (trades without exit)"
)
async def get_open_positions(
    strategy_id: Optional[int] = Query(None, gt=0, description="Filter by strategy ID"),
    instrument: Optional[str] = Query(None, description="Filter by trading instrument"),
    service: StrategyEngineService = Depends(get_strategy_service),
    user_id: int = Depends(get_current_user_id)
) -> List[TradeResponse]:
    """
    Get all currently open positions for the user.
    
    Args:
        strategy_id: Optional filter by strategy ID
        instrument: Optional filter by trading instrument
        service: Strategy service instance
        user_id: Current authenticated user ID
        
    Returns:
        List of open trades (positions)
    """
    try:
        logger.info(f"Getting open positions for user {user_id}")
        
        # Get open positions using the list_trades endpoint with is_open=True
        filters = {
            "is_open": True,
            "user_id": user_id,
            "limit": 1000  # Get all open positions
        }
        
        if strategy_id:
            filters["strategy_id"] = strategy_id
        if instrument:
            filters["instrument"] = instrument
        
        # For now, use a simplified approach - in a real implementation,
        # you'd have a dedicated service method for open positions
        open_trades = service.list_open_positions(**{k: v for k, v in filters.items() if v is not None})
        
        # Convert to response models
        position_responses = []
        for trade in open_trades:
            trade_dict = {
                "id": trade.id,
                "strategy_id": trade.strategy_id,
                "signal_id": trade.signal_id,
                "instrument": trade.instrument,
                "direction": trade.direction,
                "entry_price": trade.entry_price,
                "entry_time": trade.entry_time,
                "exit_price": trade.exit_price,
                "exit_time": trade.exit_time,
                "exit_reason": trade.exit_reason,
                "position_size": trade.position_size,
                "commission": trade.commission,
                "taxes": trade.taxes,
                "slippage": trade.slippage,
                "profit_loss_points": trade.profit_loss_points,
                "profit_loss_inr": trade.profit_loss_inr,
                "initial_risk_points": trade.initial_risk_points,
                "initial_risk_inr": trade.initial_risk_inr,
                "initial_risk_percent": trade.initial_risk_percent,
                "risk_reward_planned": trade.risk_reward_planned,
                "actual_risk_reward": trade.actual_risk_reward,
                "setup_quality": trade.setup_quality,
                "setup_score": trade.setup_score,
                "holding_period_minutes": trade.holding_period_minutes,
                "total_costs": trade.total_costs,
                "is_spread_trade": trade.is_spread_trade,
                "spread_type": trade.spread_type
            }
            position_responses.append(TradeResponse(**trade_dict))
        
        logger.info(f"Found {len(position_responses)} open positions")
        return position_responses
        
    except Exception as e:
        logger.error(f"Error getting open positions: {e}")
        raise OperationalError(f"Failed to get open positions: {str(e)}")


@router.get(
    "/positions/summary",
    summary="Get position summary",
    description="Get a summary of all open positions including totals and P&L"
)
async def get_position_summary(
    service: StrategyEngineService = Depends(get_strategy_service),
    user_id: int = Depends(get_current_user_id)
):
    """
    Get a summary of all open positions including totals and P&L.
    
    Args:
        service: Strategy service instance
        user_id: Current authenticated user ID
        
    Returns:
        Position summary with aggregated metrics
    """
    try:
        logger.info(f"Getting position summary for user {user_id}")
        
        # Get all open positions
        open_trades = service.list_open_positions(user_id=user_id)
        
        # Calculate summary metrics
        total_positions = len(open_trades)
        total_risk_inr = sum(trade.initial_risk_inr or 0 for trade in open_trades)
        total_unrealized_pnl = sum(trade.profit_loss_inr or 0 for trade in open_trades)
        
        # Count positions by direction
        long_positions = sum(1 for trade in open_trades if trade.direction == "long")
        short_positions = sum(1 for trade in open_trades if trade.direction == "short")
        
        # Group by instrument
        instruments = {}
        for trade in open_trades:
            if trade.instrument not in instruments:
                instruments[trade.instrument] = {
                    "count": 0,
                    "total_size": 0,
                    "unrealized_pnl": 0
                }
            instruments[trade.instrument]["count"] += 1
            instruments[trade.instrument]["total_size"] += trade.position_size
            instruments[trade.instrument]["unrealized_pnl"] += trade.profit_loss_inr or 0
        
        # Group by strategy
        strategies = {}
        for trade in open_trades:
            if trade.strategy_id not in strategies:
                strategies[trade.strategy_id] = {
                    "count": 0,
                    "unrealized_pnl": 0
                }
            strategies[trade.strategy_id]["count"] += 1
            strategies[trade.strategy_id]["unrealized_pnl"] += trade.profit_loss_inr or 0
        
        summary = {
            "total_positions": total_positions,
            "long_positions": long_positions,
            "short_positions": short_positions,
            "total_risk_inr": total_risk_inr,
            "total_unrealized_pnl_inr": total_unrealized_pnl,
            "positions_by_instrument": instruments,
            "positions_by_strategy": strategies,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(f"Generated position summary with {total_positions} positions")
        return summary
        
    except Exception as e:
        logger.error(f"Error getting position summary: {e}")
        raise OperationalError(f"Failed to get position summary: {str(e)}")


# Trade History Endpoints

@router.get(
    "/trades/{trade_id}/history",
    summary="Get trade execution history",
    description="Get detailed execution history and timeline for a trade"
)
async def get_trade_history(
    trade_id: int = Path(..., gt=0, description="Trade ID"),
    service: StrategyEngineService = Depends(get_strategy_service),
    user_id: int = Depends(get_current_user_id)
):
    """
    Get detailed execution history and timeline for a trade.
    
    Args:
        trade_id: ID of the trade
        service: Strategy service instance
        user_id: Current authenticated user ID
        
    Returns:
        Trade execution history and timeline
        
    Raises:
        HTTPException: If trade not found or access denied
    """
    try:
        logger.info(f"Getting history for trade {trade_id}, user {user_id}")
        
        # First verify ownership
        trade = service.get_trade(trade_id)
        if trade.user_id != user_id:
            logger.warning(f"User {user_id} attempted to access history for trade {trade_id} owned by {trade.user_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You can only access history of your own trades"
            )
        
        # Build execution timeline
        timeline = []
        
        # Entry event
        timeline.append({
            "event": "trade_opened",
            "timestamp": trade.entry_time.isoformat() if trade.entry_time else None,
            "price": trade.entry_price,
            "details": {
                "position_size": trade.position_size,
                "direction": trade.direction,
                "signal_id": trade.signal_id,
                "setup_quality": trade.setup_quality
            }
        })
        
        # Exit event (if trade is closed)
        if trade.exit_time and trade.exit_price:
            timeline.append({
                "event": "trade_closed",
                "timestamp": trade.exit_time.isoformat(),
                "price": trade.exit_price,
                "details": {
                    "exit_reason": trade.exit_reason,
                    "profit_loss_points": trade.profit_loss_points,
                    "profit_loss_inr": trade.profit_loss_inr,
                    "holding_period_minutes": trade.holding_period_minutes
                }
            })
        
        # Calculate duration if trade is open
        duration_minutes = None
        if trade.entry_time:
            if trade.exit_time:
                duration_minutes = (trade.exit_time - trade.entry_time).total_seconds() / 60
            else:
                duration_minutes = (datetime.utcnow() - trade.entry_time).total_seconds() / 60
        
        history = {
            "trade_id": trade_id,
            "timeline": timeline,
            "status": "closed" if trade.exit_time else "open",
            "duration_minutes": duration_minutes,
            "total_events": len(timeline),
            "performance": {
                "profit_loss_points": trade.profit_loss_points,
                "profit_loss_inr": trade.profit_loss_inr,
                "risk_reward_planned": trade.risk_reward_planned,
                "risk_reward_actual": trade.actual_risk_reward,
                "total_costs": trade.total_costs
            }
        }
        
        logger.info(f"Successfully retrieved history for trade {trade_id}")
        return history
        
    except ValueError as e:
        logger.error(f"Trade {trade_id} not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trade with ID {trade_id} not found"
        )
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"Error getting trade history for {trade_id}: {e}")
        raise OperationalError(f"Failed to get trade history: {str(e)}")


# Trade Analytics Endpoints

@router.get(
    "/trades/analytics",
    summary="Get trade analytics",
    description="Get performance analytics for trades within a specific period"
)
async def get_trade_analytics(
    strategy_id: Optional[int] = Query(None, gt=0, description="Filter by strategy ID"),
    start_date: Optional[str] = Query(None, description="Start date for analysis (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date for analysis (YYYY-MM-DD)"),
    group_by: Optional[str] = Query("day", description="Group results by (day, week, month)"),
    service: StrategyEngineService = Depends(get_strategy_service),
    user_id: int = Depends(get_current_user_id)
):
    """
    Get performance analytics for trades within a specific period.
    
    Args:
        strategy_id: Optional filter by strategy ID
        start_date: Optional start date filter (YYYY-MM-DD format)
        end_date: Optional end date filter (YYYY-MM-DD format)
        group_by: Group results by time period (day, week, month)
        service: Strategy service instance
        user_id: Current authenticated user ID
        
    Returns:
        Trade performance analytics grouped by specified time period
    """
    try:
        logger.info(f"Getting trade analytics for user {user_id}, strategy_id={strategy_id}")
        
        # Parse dates if provided
        start_datetime = None
        end_datetime = None
        
        if start_date:
            try:
                start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
            except ValueError:
                raise ValidationError("Invalid start_date format. Use YYYY-MM-DD")
        
        if end_date:
            try:
                end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                raise ValidationError("Invalid end_date format. Use YYYY-MM-DD")
        
        # Default dates if not provided
        if not start_datetime:
            # Default to last 30 days
            start_datetime = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            start_datetime = start_datetime.replace(day=start_datetime.day - 30)
        
        if not end_datetime:
            end_datetime = datetime.utcnow().replace(hour=23, minute=59, second=59, microsecond=999999)
        
        # Validate group_by parameter
        if group_by not in ["day", "week", "month"]:
            raise ValidationError("group_by must be one of: day, week, month")
        
        # Call service method
        analytics = service.analyze_trades(
            user_id=user_id,
            strategy_id=strategy_id,
            start_date=start_datetime,
            end_date=end_datetime,
            group_by=group_by
        )
        
        logger.info(f"Successfully generated trade analytics for user {user_id}")
        return analytics
        
    except ValidationError:
        raise  # Re-raise validation errors as-is
    except Exception as e:
        logger.error(f"Error getting trade analytics: {e}")
        raise OperationalError(f"Failed to get trade analytics: {str(e)}")


@router.get(
    "/trades/metrics",
    summary="Get trading metrics",
    description="Get comprehensive trading metrics and performance indicators"
)
async def get_trading_metrics(
    strategy_id: Optional[int] = Query(None, gt=0, description="Filter by strategy ID"),
    period: str = Query("all", description="Time period (all, daily, weekly, monthly, yearly)"),
    service: StrategyEngineService = Depends(get_strategy_service),
    user_id: int = Depends(get_current_user_id)
):
    """
    Get comprehensive trading metrics and performance indicators.
    
    Args:
        strategy_id: Optional filter by strategy ID
        period: Time period for metrics calculation
        service: Strategy service instance
        user_id: Current authenticated user ID
        
    Returns:
        Comprehensive trading metrics and performance indicators
    """
    try:
        logger.info(f"Getting trading metrics for user {user_id}, period={period}")
        
        # Validate period parameter
        valid_periods = ["all", "daily", "weekly", "monthly", "yearly"]
        if period not in valid_periods:
            raise ValidationError(f"period must be one of: {', '.join(valid_periods)}")
        
        # Call service method
        metrics = service.calculate_trading_metrics(
            user_id=user_id,
            strategy_id=strategy_id,
            period=period
        )
        
        logger.info(f"Successfully generated trading metrics for user {user_id}")
        return metrics
        
    except ValidationError:
        raise  # Re-raise validation errors as-is
    except Exception as e:
        logger.error(f"Error getting trading metrics: {e}")
        raise OperationalError(f"Failed to get trading metrics: {str(e)}")


# Batch Operations Endpoints

@router.post(
    "/trades/batch-close",
    response_model=List[TradeResponse],
    summary="Close multiple trades",
    description="Close multiple open trades in a single batch operation"
)
async def batch_close_trades(
    trade_ids: List[int] = Query(..., description="List of trade IDs to close"),
    exit_price_map: Dict[str, float] = Query(..., description="Mapping of instrument to exit price"),
    exit_reason: str = Query("batch_close", description="Reason for closing the trades"),
    service: StrategyEngineService = Depends(get_strategy_service),
    user_id: int = Depends(get_current_user_id)
) -> List[TradeResponse]:
    """
    Close multiple open trades in a single batch operation.
    
    Args:
        trade_ids: List of trade IDs to close
        exit_price_map: Mapping of instrument to exit price
        exit_reason: Reason for closing the trades
        service: Strategy service instance
        user_id: Current authenticated user ID
        
    Returns:
        List of updated closed trades
        
    Raises:
        HTTPException: If any trades are not found, already closed, or access denied
        ValidationError: If exit parameters are invalid
    """
    try:
        logger.info(f"Batch closing {len(trade_ids)} trades for user {user_id}")
        
        # Verify ownership of all trades
        for trade_id in trade_ids:
            try:
                existing_trade = service.get_trade(trade_id)
                if existing_trade.user_id != user_id:
                    logger.warning(f"User {user_id} attempted to close trade {trade_id} owned by {existing_trade.user_id}")
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Access denied: Trade {trade_id} belongs to another user"
                    )
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Trade with ID {trade_id} not found"
                )
        
        # Call service method for batch closure
        closed_trades = service.batch_close_trades(
            trade_ids=trade_ids,
            exit_price_map=exit_price_map,
            exit_reason=exit_reason,
            exit_time=datetime.utcnow()
        )
        
        # Convert to response models
        trade_responses = []
        for trade in closed_trades:
            trade_dict = {
                "id": trade.id,
                "strategy_id": trade.strategy_id,
                "signal_id": trade.signal_id,
                "instrument": trade.instrument,
                "direction": trade.direction,
                "entry_price": trade.entry_price,
                "entry_time": trade.entry_time,
                "exit_price": trade.exit_price,
                "exit_time": trade.exit_time,
                "exit_reason": trade.exit_reason,
                "position_size": trade.position_size,
                "profit_loss_points": trade.profit_loss_points,
                "profit_loss_inr": trade.profit_loss_inr,
                "setup_quality": trade.setup_quality,
                "setup_score": trade.setup_score,
                # Include other fields...
            }
            trade_responses.append(TradeResponse(**trade_dict))
        
        logger.info(f"Successfully closed {len(closed_trades)} trades in batch")
        return trade_responses
        
    except ValidationError:
        raise  # Re-raise validation errors as-is
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"Error in batch closing trades: {e}")
        raise OperationalError(f"Failed to batch close trades: {str(e)}")


# Risk Management Endpoints

@router.get(
    "/risk/exposure",
    summary="Get risk exposure",
    description="Get current risk exposure across all open positions"
)
async def get_risk_exposure(
    service: StrategyEngineService = Depends(get_strategy_service),
    user_id: int = Depends(get_current_user_id)
):
    """
    Get current risk exposure across all open positions.
    
    Args:
        service: Strategy service instance
        user_id: Current authenticated user ID
        
    Returns:
        Risk exposure metrics including total risk, risk by instrument, etc.
    """
    try:
        logger.info(f"Getting risk exposure for user {user_id}")
        
        # Get all open positions
        open_trades = service.list_open_positions(user_id=user_id)
        
        # Calculate risk metrics
        total_risk_inr = sum(trade.initial_risk_inr or 0 for trade in open_trades)
        max_drawdown_inr = sum(abs(trade.profit_loss_inr) if (trade.profit_loss_inr or 0) < 0 else 0 for trade in open_trades)
        
        # Calculate risk by instrument
        risk_by_instrument = {}
        for trade in open_trades:
            if trade.instrument not in risk_by_instrument:
                risk_by_instrument[trade.instrument] = 0
            risk_by_instrument[trade.instrument] += trade.initial_risk_inr or 0
        
        # Calculate risk by strategy
        risk_by_strategy = {}
        for trade in open_trades:
            strategy_id = str(trade.strategy_id)
            if strategy_id not in risk_by_strategy:
                risk_by_strategy[strategy_id] = 0
            risk_by_strategy[strategy_id] += trade.initial_risk_inr or 0
        
        # Calculate risk by direction
        risk_long = sum(trade.initial_risk_inr or 0 for trade in open_trades if trade.direction == "long")
        risk_short = sum(trade.initial_risk_inr or 0 for trade in open_trades if trade.direction == "short")
        
        # Get account info (placeholder - would come from account service)
        account_balance = 1000000  # Example: 10 lakh INR
        
        # Calculate risk percentages
        risk_percentage = (total_risk_inr / account_balance) * 100 if account_balance else 0
        
        risk_exposure = {
            "total_risk_inr": total_risk_inr,
            "max_drawdown_inr": max_drawdown_inr,
            "risk_percentage": risk_percentage,
            "risk_by_instrument": risk_by_instrument,
            "risk_by_strategy": risk_by_strategy,
            "risk_by_direction": {
                "long": risk_long,
                "short": risk_short
            },
            "account_balance": account_balance,
            "risk_within_limits": risk_percentage <= 5.0,  # Example 5% max risk threshold
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(f"Generated risk exposure report for user {user_id}")
        return risk_exposure
        
    except Exception as e:
        logger.error(f"Error getting risk exposure: {e}")
        raise OperationalError(f"Failed to get risk exposure: {str(e)}")


@router.get(
    "/risk/limits",
    summary="Get risk limits",
    description="Get configured risk limits for trading"
)
async def get_risk_limits(
    service: StrategyEngineService = Depends(get_strategy_service),
    user_id: int = Depends(get_current_user_id)
):
    """
    Get configured risk limits for trading.
    
    Args:
        service: Strategy service instance
        user_id: Current authenticated user ID
        
    Returns:
        Risk limits configuration
    """
    try:
        logger.info(f"Getting risk limits for user {user_id}")
        
        # This would typically come from a user or account settings service
        # Here we're using a placeholder as an example
        risk_limits = {
            "max_risk_per_trade_percent": 1.0,
            "max_daily_risk_percent": 3.0,
            "max_weekly_risk_percent": 8.0,
            "weekly_drawdown_threshold": 8.0,
            "daily_drawdown_threshold": 4.0,
            "max_positions": 10,
            "max_instrument_exposure_percent": 15.0,
            "max_correlated_exposure_percent": 20.0,
            "position_size_scaling": {
                "a_plus_grade": 2.0,
                "a_grade": 1.5,
                "b_grade": 1.0,
                "c_grade": 0.5,
                "d_grade": 0.0
            }
        }
        
        logger.info(f"Retrieved risk limits for user {user_id}")
        return risk_limits
        
    except Exception as e:
        logger.error(f"Error getting risk limits: {e}")
        raise OperationalError(f"Failed to get risk limits: {str(e)}")


# Trade Feedback Endpoints

@router.post(
    "/trades/{trade_id}/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add trade feedback",
    description="Add feedback and notes to a trade for learning purposes"
)
async def add_trade_feedback(
    trade_id: int = Path(..., gt=0, description="Trade ID"),
    feedback_data: FeedbackCreate = None,
    service: StrategyEngineService = Depends(get_strategy_service),
    user_id: int = Depends(get_current_user_id)
) -> FeedbackResponse:
    """
    Add feedback and notes to a trade for learning purposes.
    
    Args:
        trade_id: ID of the trade
        feedback_data: Feedback data to add
        service: Strategy service instance
        user_id: Current authenticated user ID
        
    Returns:
        Created feedback record
        
    Raises:
        HTTPException: If trade not found or access denied
        ValidationError: If feedback data is invalid
    """
    try:
        logger.info(f"Adding feedback to trade {trade_id} for user {user_id}")
        
        # First verify ownership of the trade
        trade = service.get_trade(trade_id)
        if trade.user_id != user_id:
            logger.warning(f"User {user_id} attempted to add feedback to trade {trade_id} owned by {trade.user_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You can only add feedback to your own trades"
            )
        
        # Record feedback using service
        strategy_id = trade.strategy_id
        feedback = service.record_feedback(
            strategy_id=strategy_id,
            feedback_data=feedback_data,
            trade_id=trade_id,
            user_id=user_id
        )
        
        # Convert to response model
        feedback_dict = {
            "id": feedback.id,
            "strategy_id": feedback.strategy_id,
            "trade_id": feedback.trade_id,
            "feedback_type": feedback.feedback_type,
            "title": feedback.title,
            "description": feedback.description,
            "file_path": feedback.file_path,
            "file_type": feedback.file_type,
            "tags": feedback.tags,
            "improvement_category": feedback.improvement_category,
            "applies_to_setup": feedback.applies_to_setup,
            "applies_to_entry": feedback.applies_to_entry,
            "applies_to_exit": feedback.applies_to_exit,
            "applies_to_risk": feedback.applies_to_risk,
            "pre_trade_conviction_level": feedback.pre_trade_conviction_level,
            "emotional_state_rating": feedback.emotional_state_rating,
            "lessons_learned": feedback.lessons_learned,
            "action_items": feedback.action_items,
            "created_at": feedback.created_at
        }
        
        logger.info(f"Successfully added feedback to trade {trade_id}")
        return FeedbackResponse(**feedback_dict)
        
    except ValueError as e:
        if "not found" in str(e).lower():
            logger.error(f"Trade {trade_id} not found: {e}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Trade with ID {trade_id} not found"
            )
        else:
            logger.error(f"Validation error adding feedback: {e}")
            raise ValidationError(str(e))
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"Error adding feedback to trade {trade_id}: {e}")
        raise OperationalError(f"Failed to add feedback: {str(e)}")


@router.get(
    "/trades/{trade_id}/feedback",
    response_model=List[FeedbackResponse],
    summary="Get trade feedback",
    description="Get all feedback records for a specific trade"
)
async def get_trade_feedback(
    trade_id: int = Path(..., gt=0, description="Trade ID"),
    service: StrategyEngineService = Depends(get_strategy_service),
    user_id: int = Depends(get_current_user_id)
) -> List[FeedbackResponse]:
    """
    Get all feedback records for a specific trade.
    
    Args:
        trade_id: ID of the trade
        service: Strategy service instance
        user_id: Current authenticated user ID
        
    Returns:
        List of feedback records for this trade
        
    Raises:
        HTTPException: If trade not found or access denied
    """
    try:
        logger.info(f"Getting feedback for trade {trade_id}, user {user_id}")
        
        # First verify ownership of the trade
        trade = service.get_trade(trade_id)
        if trade.user_id != user_id:
            logger.warning(f"User {user_id} attempted to get feedback for trade {trade_id} owned by {trade.user_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You can only view feedback for your own trades"
            )
        
        # Get feedback records using service
        feedback_records = service.get_trade_feedback(trade_id)
        
        # Convert to response models
        feedback_responses = []
        for feedback in feedback_records:
            feedback_dict = {
                "id": feedback.id,
                "strategy_id": feedback.strategy_id,
                "trade_id": feedback.trade_id,
                "feedback_type": feedback.feedback_type,
                "title": feedback.title,
                "description": feedback.description,
                "file_path": feedback.file_path,
                "file_type": feedback.file_type,
                "tags": feedback.tags,
                "improvement_category": feedback.improvement_category,
                "applies_to_setup": feedback.applies_to_setup,
                "applies_to_entry": feedback.applies_to_entry,
                "applies_to_exit": feedback.applies_to_exit,
                "applies_to_risk": feedback.applies_to_risk,
                "pre_trade_conviction_level": feedback.pre_trade_conviction_level,
                "emotional_state_rating": feedback.emotional_state_rating,
                "lessons_learned": feedback.lessons_learned,
                "action_items": feedback.action_items,
                "created_at": feedback.created_at
            }
            feedback_responses.append(FeedbackResponse(**feedback_dict))
        
        logger.info(f"Found {len(feedback_responses)} feedback records for trade {trade_id}")
        return feedback_responses
        
    except ValueError as e:
        logger.error(f"Trade {trade_id} not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trade with ID {trade_id} not found"
        )
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"Error getting feedback for trade {trade_id}: {e}")
        raise OperationalError(f"Failed to get trade feedback: {str(e)}")