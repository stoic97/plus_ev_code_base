"""
Health check endpoints for system monitoring and status reporting.

Provides API endpoints for checking the health of all system components,
specific components, or categories of components.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.core.security import has_role, Roles
from app.monitoring.health_checks import (
    HealthStatus,
    check_health,
    get_system_health,
    check_critical_components,
    check_database_components
)

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["health"])


#################################################
# Response Models
#################################################

class HealthDetail(BaseModel):
    """Detailed health information for a component."""
    
    component: str = Field(..., description="Component name")
    status: str = Field(..., description="Health status")
    timestamp: str = Field(..., description="Timestamp of health check")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional details")
    error: Optional[str] = Field(None, description="Error message if applicable")


class SystemHealthResponse(BaseModel):
    """System health check response model."""
    
    status: str = Field(..., description="Overall system health status")
    timestamp: str = Field(..., description="Timestamp of health check")
    components: Dict[str, HealthDetail] = Field(
        ..., description="Health details for all components"
    )


class CategoryHealthResponse(BaseModel):
    """Health check response for a category of components."""
    
    category: str = Field(..., description="Category name")
    status: str = Field(..., description="Overall category health status")
    timestamp: str = Field(..., description="Timestamp of health check")
    components: Dict[str, HealthDetail] = Field(
        ..., description="Health details for components in the category"
    )


#################################################
# Endpoints
#################################################

@router.get(
    "/",
    response_model=SystemHealthResponse,
    summary="Get overall system health",
    description="Returns health status for all system components."
)
async def get_health():
    """
    Get overall system health status.
    
    Returns:
        Overall system health status and component details.
    """
    try:
        health_data = get_system_health()
        return health_data
    except Exception as e:
        logger.error(f"Error checking system health: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error checking system health: {str(e)}"
        )


@router.get(
    "/critical",
    response_model=SystemHealthResponse,
    summary="Get critical components health",
    description="Returns health status for critical system components only."
)
async def get_critical_health():
    """
    Get health status of critical components.
    
    Returns:
        Health status for critical system components.
    """
    try:
        health_data = check_critical_components()
        
        # Format to match SystemHealthResponse
        return {
            "status": health_data["status"],
            "timestamp": health_data["timestamp"],
            "components": health_data["components"]
        }
    except Exception as e:
        logger.error(f"Error checking critical components health: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error checking critical components health: {str(e)}"
        )


@router.get(
    "/databases",
    response_model=Dict[str, HealthDetail],
    summary="Get database components health",
    description="Returns health status for all database components.",
    dependencies=[Depends(has_role([Roles.ADMIN, Roles.ANALYST, Roles.RISK_MANAGER]))]
)
async def get_database_health():
    """
    Get health status of all database components.
    
    Returns:
        Health status for database components.
    """
    try:
        results = check_database_components()
        
        # Convert to dictionary of HealthDetail objects
        return {
            component: result.to_dict()
            for component, result in results.items()
        }
    except Exception as e:
        logger.error(f"Error checking database health: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error checking database health: {str(e)}"
        )


@router.get(
    "/component/{component_name}",
    response_model=HealthDetail,
    summary="Get specific component health",
    description="Returns health status for a specific component."
)
async def get_component_health(component_name: str):
    """
    Get health status of a specific component.
    
    Args:
        component_name: Name of the component to check
        
    Returns:
        Health status for the specified component.
    """
    try:
        result = check_health(component=component_name)
        return result.to_dict()
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Component not found: {component_name}"
        )
    except Exception as e:
        logger.error(f"Error checking component health: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error checking component health: {str(e)}"
        )


@router.get(
    "/category/{category_name}",
    response_model=CategoryHealthResponse,
    summary="Get category health",
    description="Returns health status for all components in a category.",
    dependencies=[Depends(has_role([Roles.ADMIN, Roles.RISK_MANAGER]))]
)
async def get_category_health(category_name: str):
    """
    Get health status of all components in a category.
    
    Args:
        category_name: Name of the category to check
        
    Returns:
        Health status for the specified category.
    """
    try:
        results = check_health(category=category_name)
        
        if not results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Category not found: {category_name}"
            )
        
        # Determine overall status
        # If any component is unhealthy, category is unhealthy
        # If any component is degraded, category is degraded
        # Otherwise category is healthy
        if any(result.status == HealthStatus.UNHEALTHY for result in results.values()):
            overall_status = HealthStatus.UNHEALTHY
        elif any(result.status == HealthStatus.DEGRADED for result in results.values()):
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
        
        # Convert to CategoryHealthResponse format
        return {
            "category": category_name,
            "status": overall_status,
            "timestamp": next(iter(results.values())).timestamp.isoformat(),
            "components": {
                component: result.to_dict()
                for component, result in results.items()
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking category health: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error checking category health: {str(e)}"
        )


@router.get(
    "/liveness",
    response_model=Dict[str, str],
    summary="Liveness probe",
    description="Simple liveness probe for kubernetes health checks."
)
async def liveness_probe():
    """
    Simple liveness probe for kubernetes health checks.
    
    Returns:
        Status confirmation message.
    """
    return {"status": "alive"}


@router.get(
    "/readiness",
    response_model=Dict[str, str],
    summary="Readiness probe",
    description="Readiness probe that checks critical dependencies."
)
async def readiness_probe():
    """
    Readiness probe that checks if the application is ready to handle traffic.
    Verifies that critical dependencies are available.
    
    Returns:
        Status confirmation message.
    """
    # Check critical components
    try:
        health_data = check_critical_components()
        
        # If any critical component is unhealthy, the application is not ready
        if health_data["status"] == HealthStatus.UNHEALTHY:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Application not ready: critical dependencies unavailable"
            )
        
        return {"status": "ready"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking readiness: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error checking readiness: {str(e)}"
        )