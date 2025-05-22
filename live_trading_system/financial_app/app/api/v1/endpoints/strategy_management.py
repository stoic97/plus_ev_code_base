"""
Strategy Management API endpoints for the Trading Strategies Application.

This module provides RESTful endpoints for strategy CRUD operations,
versioning, activation/deactivation, and basic management functions.
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Path, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.core.database import get_postgres_db
from app.core.error_handling import (
    DatabaseConnectionError,
    OperationalError,
    ValidationError,
    AuthenticationError,
)
from app.services.strategy_engine import StrategyEngineService
from app.schemas.strategy import (
    StrategyCreate,
    StrategyUpdate,
    StrategyResponse,
    PerformanceAnalysis,
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


# Strategy CRUD Endpoints

@router.post(
    "/",
    response_model=StrategyResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new trading strategy",
    description="Create a new trading strategy with all configuration settings"
)
async def create_strategy(
    strategy_data: StrategyCreate,
    service: StrategyEngineService = Depends(get_strategy_service),
    user_id: int = Depends(get_current_user_id)
) -> StrategyResponse:
    """
    Create a new trading strategy.
    
    Args:
        strategy_data: Strategy creation data with all settings
        service: Strategy service instance
        user_id: Current authenticated user ID
        
    Returns:
        Created strategy with all details
        
    Raises:
        ValidationError: If strategy data is invalid
        OperationalError: If creation fails
    """
    try:
        logger.info(f"Creating new strategy: {strategy_data.name} for user {user_id}")
        
        # Create strategy using service
        strategy = service.create_strategy(strategy_data, user_id)
        
        # Convert to response model
        strategy_dict = strategy.to_dict(include_relationships=True)
        
        logger.info(f"Successfully created strategy ID: {strategy.id}")
        return StrategyResponse(**strategy_dict)
        
    except ValueError as e:
        logger.error(f"Validation error creating strategy: {e}")
        raise ValidationError(str(e))
    except Exception as e:
        logger.error(f"Error creating strategy: {e}")
        raise OperationalError(f"Failed to create strategy: {str(e)}")


@router.get(
    "/",
    response_model=List[StrategyResponse],
    summary="List trading strategies",
    description="List all trading strategies with optional filtering"
)
async def list_strategies(
    user_id: Optional[int] = Query(None, description="Filter by owner user ID"),
    include_inactive: bool = Query(False, description="Include inactive strategies"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(100, ge=1, le=1000, description="Pagination limit"),
    service: StrategyEngineService = Depends(get_strategy_service),
    current_user_id: int = Depends(get_current_user_id)
) -> List[StrategyResponse]:
    """
    List trading strategies with optional filtering.
    
    Args:
        user_id: Optional filter by owner user ID
        include_inactive: Whether to include inactive strategies
        offset: Pagination offset
        limit: Maximum number of results
        service: Strategy service instance
        current_user_id: Current authenticated user ID
        
    Returns:
        List of strategies matching filters
    """
    try:
        logger.info(f"Listing strategies for user {current_user_id}, filters: user_id={user_id}, include_inactive={include_inactive}")
        
        # Use user_id filter if provided, otherwise show only current user's strategies
        filter_user_id = user_id if user_id is not None else current_user_id
        
        strategies = service.list_strategies(
            user_id=filter_user_id,
            offset=offset,
            limit=limit,
            include_inactive=include_inactive
        )
        
        # Convert to response models
        strategy_responses = []
        for strategy in strategies:
            strategy_dict = strategy.to_dict(include_relationships=True)
            strategy_responses.append(StrategyResponse(**strategy_dict))
        
        logger.info(f"Found {len(strategy_responses)} strategies")
        return strategy_responses
        
    except Exception as e:
        logger.error(f"Error listing strategies: {e}")
        raise OperationalError(f"Failed to list strategies: {str(e)}")


@router.get(
    "/{strategy_id}",
    response_model=StrategyResponse,
    summary="Get strategy details",
    description="Get detailed information about a specific strategy"
)
async def get_strategy(
    strategy_id: int = Path(..., gt=0, description="Strategy ID"),
    service: StrategyEngineService = Depends(get_strategy_service),
    user_id: int = Depends(get_current_user_id)
) -> StrategyResponse:
    """
    Get detailed information about a specific strategy.
    
    Args:
        strategy_id: ID of the strategy to retrieve
        service: Strategy service instance
        user_id: Current authenticated user ID
        
    Returns:
        Strategy details with all relationships
        
    Raises:
        HTTPException: If strategy not found or access denied
    """
    try:
        logger.info(f"Getting strategy {strategy_id} for user {user_id}")
        
        strategy = service.get_strategy(strategy_id)
        
        # Check access permissions (user can only access their own strategies)
        if strategy.user_id != user_id:
            logger.warning(f"User {user_id} attempted to access strategy {strategy_id} owned by {strategy.user_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You can only access your own strategies"
            )
        
        # Convert to response model
        strategy_dict = strategy.to_dict(include_relationships=True)
        
        logger.info(f"Successfully retrieved strategy {strategy_id}")
        return StrategyResponse(**strategy_dict)
        
    except ValueError as e:
        logger.error(f"Strategy {strategy_id} not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy with ID {strategy_id} not found"
        )
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"Error getting strategy {strategy_id}: {e}")
        raise OperationalError(f"Failed to get strategy: {str(e)}")


@router.put(
    "/{strategy_id}",
    response_model=StrategyResponse,
    summary="Update strategy",
    description="Update an existing strategy's configuration"
)
async def update_strategy(
    strategy_id: int = Path(..., gt=0, description="Strategy ID"),
    strategy_data: StrategyUpdate = None,
    service: StrategyEngineService = Depends(get_strategy_service),
    user_id: int = Depends(get_current_user_id)
) -> StrategyResponse:
    """
    Update an existing strategy.
    
    Args:
        strategy_id: ID of the strategy to update
        strategy_data: Strategy update data
        service: Strategy service instance
        user_id: Current authenticated user ID
        
    Returns:
        Updated strategy details
        
    Raises:
        HTTPException: If strategy not found or access denied
        ValidationError: If update data is invalid
    """
    try:
        logger.info(f"Updating strategy {strategy_id} for user {user_id}")
        
        # First verify ownership
        existing_strategy = service.get_strategy(strategy_id)
        if existing_strategy.user_id != user_id:
            logger.warning(f"User {user_id} attempted to update strategy {strategy_id} owned by {existing_strategy.user_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You can only update your own strategies"
            )
        
        # Update strategy using service
        updated_strategy = service.update_strategy(strategy_id, strategy_data, user_id)
        
        # Convert to response model
        strategy_dict = updated_strategy.to_dict(include_relationships=True)
        
        logger.info(f"Successfully updated strategy {strategy_id}")
        return StrategyResponse(**strategy_dict)
        
    except ValueError as e:
        if "not found" in str(e).lower():
            logger.error(f"Strategy {strategy_id} not found: {e}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Strategy with ID {strategy_id} not found"
            )
        else:
            logger.error(f"Validation error updating strategy {strategy_id}: {e}")
            raise ValidationError(str(e))
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"Error updating strategy {strategy_id}: {e}")
        raise OperationalError(f"Failed to update strategy: {str(e)}")


@router.delete(
    "/{strategy_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete strategy",
    description="Delete a strategy (soft delete by default)"
)
async def delete_strategy(
    strategy_id: int = Path(..., gt=0, description="Strategy ID"),
    hard_delete: bool = Query(False, description="Perform hard delete (permanent)"),
    service: StrategyEngineService = Depends(get_strategy_service),
    user_id: int = Depends(get_current_user_id)
):
    """
    Delete a strategy.
    
    Args:
        strategy_id: ID of the strategy to delete
        hard_delete: Whether to perform hard delete (permanent) or soft delete
        service: Strategy service instance
        user_id: Current authenticated user ID
        
    Raises:
        HTTPException: If strategy not found or access denied
    """
    try:
        logger.info(f"Deleting strategy {strategy_id} for user {user_id}, hard_delete={hard_delete}")
        
        # First verify ownership
        existing_strategy = service.get_strategy(strategy_id)
        if existing_strategy.user_id != user_id:
            logger.warning(f"User {user_id} attempted to delete strategy {strategy_id} owned by {existing_strategy.user_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You can only delete your own strategies"
            )
        
        # Delete strategy using service
        success = service.delete_strategy(strategy_id, user_id, hard_delete=hard_delete)
        
        if success:
            delete_type = "hard deleted" if hard_delete else "soft deleted"
            logger.info(f"Successfully {delete_type} strategy {strategy_id}")
        else:
            logger.error(f"Failed to delete strategy {strategy_id}")
            raise OperationalError("Failed to delete strategy")
            
    except ValueError as e:
        logger.error(f"Strategy {strategy_id} not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy with ID {strategy_id} not found"
        )
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"Error deleting strategy {strategy_id}: {e}")
        raise OperationalError(f"Failed to delete strategy: {str(e)}")


# Strategy State Management Endpoints

@router.post(
    "/{strategy_id}/activate",
    response_model=StrategyResponse,
    summary="Activate strategy",
    description="Activate a strategy for live trading"
)
async def activate_strategy(
    strategy_id: int = Path(..., gt=0, description="Strategy ID"),
    service: StrategyEngineService = Depends(get_strategy_service),
    user_id: int = Depends(get_current_user_id)
) -> StrategyResponse:
    """
    Activate a strategy for live trading.
    
    Args:
        strategy_id: ID of the strategy to activate
        service: Strategy service instance
        user_id: Current authenticated user ID
        
    Returns:
        Activated strategy details
        
    Raises:
        HTTPException: If strategy not found or access denied
        ValidationError: If strategy parameters are invalid
    """
    try:
        logger.info(f"Activating strategy {strategy_id} for user {user_id}")
        
        # First verify ownership
        existing_strategy = service.get_strategy(strategy_id)
        if existing_strategy.user_id != user_id:
            logger.warning(f"User {user_id} attempted to activate strategy {strategy_id} owned by {existing_strategy.user_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You can only activate your own strategies"
            )
        
        # Activate strategy using service
        activated_strategy = service.activate_strategy(strategy_id, user_id)
        
        # Convert to response model
        strategy_dict = activated_strategy.to_dict(include_relationships=True)
        
        logger.info(f"Successfully activated strategy {strategy_id}")
        return StrategyResponse(**strategy_dict)
        
    except ValueError as e:
        if "not found" in str(e).lower():
            logger.error(f"Strategy {strategy_id} not found: {e}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Strategy with ID {strategy_id} not found"
            )
        elif "validation failed" in str(e).lower():
            logger.error(f"Validation error activating strategy {strategy_id}: {e}")
            raise ValidationError(str(e))
        else:
            logger.error(f"Error activating strategy {strategy_id}: {e}")
            raise ValidationError(str(e))
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"Error activating strategy {strategy_id}: {e}")
        raise OperationalError(f"Failed to activate strategy: {str(e)}")


@router.post(
    "/{strategy_id}/deactivate",
    response_model=StrategyResponse,
    summary="Deactivate strategy",
    description="Deactivate a strategy to stop live trading"
)
async def deactivate_strategy(
    strategy_id: int = Path(..., gt=0, description="Strategy ID"),
    service: StrategyEngineService = Depends(get_strategy_service),
    user_id: int = Depends(get_current_user_id)
) -> StrategyResponse:
    """
    Deactivate a strategy to stop live trading.
    
    Args:
        strategy_id: ID of the strategy to deactivate
        service: Strategy service instance
        user_id: Current authenticated user ID
        
    Returns:
        Deactivated strategy details
        
    Raises:
        HTTPException: If strategy not found or access denied
    """
    try:
        logger.info(f"Deactivating strategy {strategy_id} for user {user_id}")
        
        # First verify ownership
        existing_strategy = service.get_strategy(strategy_id)
        if existing_strategy.user_id != user_id:
            logger.warning(f"User {user_id} attempted to deactivate strategy {strategy_id} owned by {existing_strategy.user_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You can only deactivate your own strategies"
            )
        
        # Deactivate strategy using service
        deactivated_strategy = service.deactivate_strategy(strategy_id, user_id)
        
        # Convert to response model
        strategy_dict = deactivated_strategy.to_dict(include_relationships=True)
        
        logger.info(f"Successfully deactivated strategy {strategy_id}")
        return StrategyResponse(**strategy_dict)
        
    except ValueError as e:
        logger.error(f"Strategy {strategy_id} not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy with ID {strategy_id} not found"
        )
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"Error deactivating strategy {strategy_id}: {e}")
        raise OperationalError(f"Failed to deactivate strategy: {str(e)}")


# Performance Analysis Endpoint

@router.get(
    "/{strategy_id}/performance",
    response_model=PerformanceAnalysis,
    summary="Get strategy performance",
    description="Analyze strategy performance with detailed metrics"
)
async def get_strategy_performance(
    strategy_id: int = Path(..., gt=0, description="Strategy ID"),
    start_date: Optional[str] = Query(None, description="Start date for analysis (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date for analysis (YYYY-MM-DD)"),
    service: StrategyEngineService = Depends(get_strategy_service),
    user_id: int = Depends(get_current_user_id)
) -> PerformanceAnalysis:
    """
    Get comprehensive performance analysis for a strategy.
    
    Args:
        strategy_id: ID of the strategy to analyze
        start_date: Optional start date for analysis (YYYY-MM-DD format)
        end_date: Optional end date for analysis (YYYY-MM-DD format)
        service: Strategy service instance
        user_id: Current authenticated user ID
        
    Returns:
        Detailed performance analysis including metrics by setup grade
        
    Raises:
        HTTPException: If strategy not found or access denied
        ValidationError: If date format is invalid
    """
    try:
        logger.info(f"Getting performance for strategy {strategy_id}, user {user_id}")
        
        # First verify ownership
        existing_strategy = service.get_strategy(strategy_id)
        if existing_strategy.user_id != user_id:
            logger.warning(f"User {user_id} attempted to access performance for strategy {strategy_id} owned by {existing_strategy.user_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You can only view performance of your own strategies"
            )
        
        # Parse dates if provided
        start_datetime = None
        end_datetime = None
        
        if start_date:
            try:
                from datetime import datetime
                start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
            except ValueError:
                raise ValidationError("Invalid start_date format. Use YYYY-MM-DD")
        
        if end_date:
            try:
                from datetime import datetime
                end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                raise ValidationError("Invalid end_date format. Use YYYY-MM-DD")
        
        # Get performance analysis using service
        performance_data = service.analyze_performance(
            strategy_id, 
            start_date=start_datetime, 
            end_date=end_datetime
        )
        
        logger.info(f"Successfully retrieved performance for strategy {strategy_id}")
        return PerformanceAnalysis(**performance_data)
        
    except ValueError as e:
        if "not found" in str(e).lower():
            logger.error(f"Strategy {strategy_id} not found: {e}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Strategy with ID {strategy_id} not found"
            )
        else:
            logger.error(f"Validation error getting performance for strategy {strategy_id}: {e}")
            raise ValidationError(str(e))
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"Error getting performance for strategy {strategy_id}: {e}")
        raise OperationalError(f"Failed to get strategy performance: {str(e)}")