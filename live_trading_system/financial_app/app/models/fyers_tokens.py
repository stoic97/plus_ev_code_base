"""
Fyers Token Model

This module defines the SQLAlchemy ORM model for Fyers authentication tokens,
providing persistent storage of access and refresh tokens.
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any
from contextlib import contextmanager

from sqlalchemy import (
    Column, Integer, String, DateTime, Index, 
    func, Boolean, Text, select, delete, update, insert, text
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import validates, Session
from sqlalchemy.exc import SQLAlchemyError

from app.core.database import Base

# Set up logging
logger = logging.getLogger(__name__)


class FyersToken(Base):
    """
    Database model for Fyers authentication tokens.
    
    This model stores Fyers API tokens with their expiry information,
    allowing for persistent token management across application restarts.
    """
    __tablename__ = "fyers_tokens"
    
    # Primary key
    id = Column(Integer, primary_key=True)
    
    # Fyers app identifier (must be unique)
    app_id = Column(String(50), nullable=False, unique=True, index=True, 
                   info={"description": "Fyers app ID in the format APP_ID-100"})
    
    # Authentication tokens
    access_token = Column(Text, nullable=False,
                         info={"description": "Fyers API access token (JWT)"})
    refresh_token = Column(Text, nullable=True,
                          info={"description": "Refresh token for token renewal"})
    
    # Expiration and metadata
    expiry = Column(DateTime(timezone=True), nullable=False, index=True,
                   info={"description": "Token expiry datetime"})
    is_active = Column(Boolean, default=True, nullable=False,
                      info={"description": "Whether this token is active"})
    
    # Additional token metadata
    token_type = Column(String(20), default="access_token", nullable=False,
                       info={"description": "Type of token"})
    
    # Tracking fields
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), 
                       onupdate=func.now(), nullable=False)
    
    # JSON field for additional metadata
    token_metadata = Column(JSONB, nullable=True, 
                           info={"description": "Additional token metadata"})
    
    # Add indexes for efficient queries
    __table_args__ = (
        # Index for quickly finding valid tokens
        Index('ix_fyers_tokens_app_id_expiry', 'app_id', 'expiry'),
        
        # Index for cleanup of expired tokens
        Index('ix_fyers_tokens_expiry', 'expiry'),
    )
    
    @validates('app_id')
    def validate_app_id(self, key, value):
        """Validate that app_id is in the correct format."""
        if not value or '-' not in value:
            raise ValueError("app_id must be in the format APP_ID-100")
        return value
    
    @validates('access_token')
    def validate_access_token(self, key, value):
        """Validate that access_token is not empty."""
        if not value:
            raise ValueError("access_token cannot be empty")
        return value
    
    @classmethod
    def get_by_app_id(cls, session: Session, app_id: str) -> Optional["FyersToken"]:
        """
        Get token by app_id.
        
        Args:
            session: SQLAlchemy session
            app_id: Fyers app ID
            
        Returns:
            Token record or None if not found
        """
        try:
            stmt = select(cls).where(cls.app_id == app_id)
            result = session.execute(stmt)
            return result.scalars().first()
        except Exception as e:
            logger.error(f"Failed to get token by app_id: {e}")
            return None
    
    @classmethod
    def save_token(cls, session: Session, app_id: str, access_token: str, 
                  refresh_token: Optional[str], expiry: datetime) -> bool:
        """
        Save token to database.
        
        Args:
            session: SQLAlchemy session
            app_id: Fyers app ID
            access_token: Access token
            refresh_token: Refresh token (optional)
            expiry: Token expiry datetime
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if token exists
            token = cls.get_by_app_id(session, app_id)
            
            if token:
                # Update existing token
                token.access_token = access_token
                token.refresh_token = refresh_token
                token.expiry = expiry
                token.is_active = True
                token.updated_at = func.now()
            else:
                # Create new token
                token = cls(
                    app_id=app_id,
                    access_token=access_token,
                    refresh_token=refresh_token,
                    expiry=expiry,
                    is_active=True
                )
                session.add(token)
            
            session.commit()
            return True
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save token to database: {e}")
            return False
    
    @classmethod
    def delete_token(cls, session: Session, app_id: str) -> bool:
        """
        Delete token from database.
        
        Args:
            session: SQLAlchemy session
            app_id: Fyers app ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            stmt = delete(cls).where(cls.app_id == app_id)
            session.execute(stmt)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to delete token from database: {e}")
            return False
    
    @classmethod
    def cleanup_expired_tokens(cls, session: Session) -> int:
        """
        Clean up expired tokens.
        
        Args:
            session: SQLAlchemy session
            
        Returns:
            Number of tokens deleted
        """
        try:
            # First query to get count
            stmt_count = select(func.count()).select_from(cls).where(cls.expiry < func.now())
            result = session.execute(stmt_count)
            count = result.scalar() or 0
            
            # Then delete
            stmt_delete = delete(cls).where(cls.expiry < func.now())
            session.execute(stmt_delete)
            session.commit()
            
            if count > 0:
                logger.info(f"Cleaned up {count} expired tokens")
            return count
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to clean up expired tokens: {e}")
            return 0
    
    def __repr__(self):
        """String representation."""
        return f"<FyersToken(app_id='{self.app_id}', expiry='{self.expiry}')>"