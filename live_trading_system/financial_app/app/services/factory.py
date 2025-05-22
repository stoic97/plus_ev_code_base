"""
Service factory for dependency injection and service instantiation.
"""
from sqlalchemy.orm import Session
from fastapi import Depends

from app.core.database import get_db
from app.services.strategy_engine import StrategyEngineService

def get_strategy_engine_service(db: Session = Depends(get_db)) -> StrategyEngineService:
    """
    Factory function to create a StrategyEngineService instance.
    
    Args:
        db: SQLAlchemy database session
        
    Returns:
        StrategyEngineService: Configured service instance
    """
    return StrategyEngineService(db)