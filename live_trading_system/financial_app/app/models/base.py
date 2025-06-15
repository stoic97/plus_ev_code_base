"""
Base module for SQLAlchemy models.

This module provides common imports, mixins, and utilities for all models
to ensure consistency and reduce code duplication across the financial application.
"""

# SQLAlchemy imports
from sqlalchemy import (
    Boolean, Column, DateTime, Date, Time, ForeignKey, 
    Integer, String, Float, Enum, Text, DECIMAL, JSON,
    UniqueConstraint, Index, event, Table, MetaData, func
)
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import relationship, backref, validates
from sqlalchemy.sql import expression
from sqlalchemy.dialects.postgresql import UUID, JSONB

# Type imports
import enum
from datetime import datetime, date, time, timedelta
from typing import List, Dict, Optional, Union, Any, Type, TypeVar, Generic, cast
from uuid import uuid4
import json
# Import User model to make it available for relationships
try:
    from app.models.user import User
except ImportError:
    # User model not available - relationships will be string-based
    pass
# Import the SQLAlchemy Base
from app.core.database import Base

# Utilities
def generate_uuid():
    """Generate a UUID string for use as a unique identifier."""
    return str(uuid4())

def model_to_dict(model, exclude=None):
    """Convert a model instance to a dictionary."""
    exclude = exclude or []
    return {c.name: getattr(model, c.name) for c in model.__table__.columns if c.name not in exclude}

def validate_non_negative(value):
    """Validate that a value is non-negative."""
    if value is not None and value < 0:
        raise ValueError("Value cannot be negative")
    return value

# Mixins
class TimestampMixin:
    """
    Add creation and update timestamps to a model.
    
    Attributes:
        created_at: Timestamp when the record was created
        updated_at: Timestamp when the record was last updated
    """
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class UserRelationMixin:
    """
    Add a relationship to the User model.
    
    This mixin automatically creates a foreign key to the users table
    and establishes a relationship to the User model.
    """
    @declared_attr
    def user_id(cls):
        from sqlalchemy.dialects.postgresql import UUID
        return Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
        
        
    @declared_attr
    def user(cls):
        return relationship("User", foreign_keys=[cls.user_id])

class PositiveValueMixin:
    """
    Mixin that adds validation for positive numeric fields.
    
    Automatically validates amount, price, and quantity fields
    to ensure they are non-negative.
    """
    @validates('amount', 'price', 'quantity', 'balance', 'volume')
    def validate_positive(self, key, value):
        if value is not None:  # Allow None values
            return validate_non_negative(value)
        return value

class AuditMixin(TimestampMixin):
    """
    Add audit trail fields to a model.
    
    Extends the TimestampMixin by adding fields to track which users
    created and updated the record.
    
    Attributes:
        created_by_id: ID of user who created the record
        updated_by_id: ID of user who last updated the record
        created_at: Timestamp when created (from TimestampMixin)
        updated_at: Timestamp when updated (from TimestampMixin)
    """
    from sqlalchemy.dialects.postgresql import UUID
    created_by_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), index=True)
    updated_by_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    
    
    @declared_attr
    def created_by(cls):
        return relationship("User", foreign_keys=[cls.created_by_id])
        
    @declared_attr
    def updated_by(cls):
        return relationship("User", foreign_keys=[cls.updated_by_id])

class SoftDeleteMixin:
    """
    Add soft delete capabilities to a model.
    
    Instead of actually deleting records, this marks them as deleted
    by setting a deletion timestamp and tracking who deleted them.
    
    Attributes:
        deleted_at: Timestamp when the record was "deleted"
        deleted_by_id: ID of user who deleted the record
    """
    deleted_at = Column(DateTime(timezone=True))
    from sqlalchemy.dialects.postgresql import UUID
    deleted_by_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    
    @property
    def is_deleted(self):
        """Check if record has been soft-deleted."""
        return self.deleted_at is not None
    
    def soft_delete(self, user_id=None):
        """Mark record as deleted."""
        self.deleted_at = datetime.utcnow()
        if user_id:
            self.deleted_by_id = user_id
    
    @declared_attr
    def deleted_by(cls):
        return relationship("User", foreign_keys=[cls.deleted_by_id])

class SerializableMixin:
    """
    Add JSON serialization support to a model.
    
    Provides methods to convert models to dictionaries and handle
    nested relationships and special data types.
    """
    def to_dict(self, include_relationships=False, exclude=None):
        """
        Convert model to a dictionary.
        
        Args:
            include_relationships: Whether to include related models
            exclude: List of fields to exclude
            
        Returns:
            Dict representation of the model
        """
        exclude = exclude or []
        result = {}
        
        for column in self.__table__.columns:
            if column.name in exclude:
                continue
                
            value = getattr(self, column.name)
            
            # Handle special types
            if isinstance(value, (datetime, date)):
                value = value.isoformat()
            elif isinstance(value, enum.Enum):
                value = value.value
            elif isinstance(value, uuid4):
                value = str(value)
                
            result[column.name] = value
            
        # Include relationship data if requested
        if include_relationships:
            for relationship in self.__mapper__.relationships:
                rel_name = relationship.key
                
                if rel_name in exclude:
                    continue
                    
                rel_value = getattr(self, rel_name)
                
                # Handle collections
                if hasattr(rel_value, '__iter__') and not isinstance(rel_value, (str, bytes, dict)):
                    result[rel_name] = [
                        item.to_dict() if hasattr(item, 'to_dict') else str(item)
                        for item in rel_value
                    ]
                # Handle single objects
                elif hasattr(rel_value, 'to_dict'):
                    result[rel_name] = rel_value.to_dict()
                    
        return result
    
    def to_json(self, include_relationships=False, exclude=None):
        """Convert model to JSON string."""
        return json.dumps(self.to_dict(include_relationships, exclude))

class VersionedMixin:
    """
    Track version history of a model.
    
    Keeps a version number that increments with each update
    and can be used for optimistic concurrency control.
    """
    version = Column(Integer, nullable=False, default=1)
    
    @validates('version')
    def validate_version(self, key, value):
        """Ensure version only increments by one."""
        current_version = getattr(self, 'version', 0)
        if value != current_version + 1:
            raise ValueError(f"Version can only increment by 1. Current: {current_version}, Attempted: {value}")
        return value

class StatusMixin:
    """
    Add status tracking to a model.
    
    Provides a standard way to track status changes with timestamps.
    """
    status = Column(String(50), nullable=False, index=True)
    status_changed_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    previous_status = Column(String(50))
    
    def update_status(self, new_status):
        """Update status with timestamp and history."""
        if self.status != new_status:
            self.previous_status = self.status
            self.status = new_status
            self.status_changed_at = datetime.utcnow()

class BaseModel(Base, SerializableMixin):
    """
    Base class for all models providing common functionality.
    
    This abstract base class provides standard fields and methods
    that should be available on all models.
    
    Attributes:
        id: Primary key
    """
    __abstract__ = True
    
    id = Column(Integer, primary_key=True, index=True)
    
    @classmethod
    def get_by_id(cls, session, id):
        """Get a record by its primary key."""
        return session.query(cls).filter(cls.id == id).first()
    
    @classmethod
    def list_all(cls, session, limit=100, offset=0, order_by=None):
        """Get all records with pagination."""
        query = session.query(cls)
        
        if order_by:
            if isinstance(order_by, (list, tuple)):
                for order_field in order_by:
                    query = query.order_by(order_field)
            else:
                query = query.order_by(order_by)
        
        return query.limit(limit).offset(offset).all()
    
    def save(self, session):
        """Save the current model to the database."""
        session.add(self)
        session.flush()  # Flush to get the ID
        return self

class UUIDBaseModel(BaseModel):
    """
    Base model using UUIDs as primary keys instead of integers.
    
    Useful for distributed systems or when you want to hide sequential IDs.
    
    Attributes:
        id: UUID primary key
    """
    __abstract__ = True
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4, index=True)