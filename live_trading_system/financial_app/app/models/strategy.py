"""
Strategy models for trading platform with fixed MRO.

This module defines the database models for strategy management, including:
- Trading strategies
- Strategy parameters
- Strategy versions
- Strategy categories
- Strategy performance metrics

These models form the core data structure for the trading strategy service.
"""

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, ForeignKey, Table, JSON, Enum, Text
)
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func
import enum
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

# Import only the specific classes we need - avoid circular references
from app.core.database import Base
from app.models.base import (
    TimestampMixin, UserRelationMixin, AuditMixin, 
    SoftDeleteMixin, SerializableMixin, VersionedMixin, StatusMixin
)

# Define many-to-many relationship for strategy categories
strategy_category_association = Table(
    'strategy_category_association',
    Base.metadata,
    Column('strategy_id', Integer, ForeignKey('strategies.id'), primary_key=True),
    Column('category_id', Integer, ForeignKey('strategy_categories.id'), primary_key=True)
)

class StrategyType(enum.Enum):
    """Enumeration of strategy types"""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    MOMENTUM = "momentum"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    PATTERN_RECOGNITION = "pattern_recognition"
    CUSTOM = "custom"


class StrategyStatus(enum.Enum):
    """Enumeration of strategy statuses"""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"
    BACKTEST = "backtest"


# Create a base model class specifically for this file to avoid MRO issues
class StrategyBaseModel(Base):
    """Base model for strategy models to avoid MRO conflicts."""
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


# Fix MRO issue by using a simplified inheritance chain 
# and applying composition where needed
class Strategy(StrategyBaseModel, TimestampMixin, UserRelationMixin, 
               AuditMixin, SoftDeleteMixin, VersionedMixin, StatusMixin):
    """
    Trading strategy model representing a complete trading strategy.
    
    This model stores all persistent data related to a trading strategy, including
    its configuration, parameters, statistics, and relationships to other entities.
    """
    __tablename__ = "strategies"
    
    # Basic strategy information
    name = Column(String(255), nullable=False)
    description = Column(Text)
    type = Column(Enum(StrategyType), nullable=False, default=StrategyType.CUSTOM)
    
    # Strategy configuration
    configuration = Column(JSON, nullable=False, default={})
    parameters = Column(JSON, nullable=False, default={})
    validation_rules = Column(JSON, default={})
    
    # Execution settings
    is_active = Column(Boolean, default=False)
    execution_schedule = Column(String(100))  # Cron-like schedule
    
    # Security and access control
    is_public = Column(Boolean, default=False)
    access_level = Column(String(50), default="private")
    
    # Performance metrics (summary)
    win_rate = Column(Float)
    profit_factor = Column(Float)
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    max_drawdown = Column(Float)
    
    # Versioning
    parent_version_id = Column(Integer, ForeignKey('strategies.id'))
    
    # Relationships
    categories = relationship("StrategyCategory", secondary=strategy_category_association, 
                             back_populates="strategies")
    signals = relationship("Signal", back_populates="strategy")
    parent_version = relationship("Strategy", remote_side=[id], 
                                 backref="child_versions")
    
    # We use composition instead of inheritance for serialization
    def to_dict(self, include_relationships=False, exclude=None):
        """
        Convert strategy to dictionary for API responses.
        
        Args:
            include_relationships: Whether to include related entities
            exclude: List of fields to exclude
            
        Returns:
            Dictionary representation of the strategy
        """
        exclude = exclude or []
        result = {}
        
        # Convert model attributes to dict (similar to SerializableMixin's to_dict)
        for column in self.__table__.columns:
            if column.name in exclude:
                continue
                
            value = getattr(self, column.name)
            
            # Handle special types
            if isinstance(value, (datetime, datetime.date)):
                value = value.isoformat()
            elif isinstance(value, enum.Enum):
                value = value.value
                
            result[column.name] = value
            
        # Include relationship data if requested
        if include_relationships:
            # Include categories
            result['categories'] = [category.name for category in self.categories]
            
            # Include signal count
            result['signal_count'] = len(self.signals) if self.signals else 0
            
            # Include version information
            result['has_parent'] = self.parent_version_id is not None
            result['has_children'] = len(self.child_versions) > 0 if hasattr(self, 'child_versions') else False
        
        return result
    
    def to_json(self, include_relationships=False, exclude=None):
        """Convert model to JSON string."""
        return json.dumps(self.to_dict(include_relationships, exclude))
    
    @validates('configuration', 'parameters', 'validation_rules')
    def validate_json(self, key, value):
        """Validate that JSON fields contain valid JSON data."""
        if isinstance(value, str):
            try:
                # If it's a string, try to parse it
                return json.loads(value)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON in {key}")
        # If it's already a dict, return as is
        return value
    
    def get_parameter(self, param_name: str, default: Any = None) -> Any:
        """
        Get a strategy parameter value.
        
        Args:
            param_name: Name of the parameter to retrieve
            default: Default value if parameter doesn't exist
            
        Returns:
            Parameter value or default
        """
        if not self.parameters:
            return default
        return self.parameters.get(param_name, default)
    
    def update_parameter(self, param_name: str, value: Any) -> None:
        """
        Update a single strategy parameter.
        
        Args:
            param_name: Name of the parameter to update
            value: New parameter value
        """
        if not self.parameters:
            self.parameters = {}
        self.parameters[param_name] = value
        self.version += 1  # Increment version on parameter changes
    
    def validate_parameters(self) -> List[str]:
        """
        Validate all parameters against validation rules.
        
        Returns:
            List of validation error messages (empty if all valid)
        """
        errors = []
        if not self.validation_rules:
            return errors
            
        for param_name, rules in self.validation_rules.items():
            if param_name not in self.parameters:
                if rules.get("required", False):
                    errors.append(f"Required parameter '{param_name}' is missing")
                continue
                
            value = self.parameters[param_name]
            
            # Type validation
            expected_type = rules.get("type")
            if expected_type:
                if expected_type == "number" and not isinstance(value, (int, float)):
                    errors.append(f"Parameter '{param_name}' must be a number")
                elif expected_type == "string" and not isinstance(value, str):
                    errors.append(f"Parameter '{param_name}' must be a string")
                elif expected_type == "boolean" and not isinstance(value, bool):
                    errors.append(f"Parameter '{param_name}' must be a boolean")
            
            # Range validation for numeric parameters
            if isinstance(value, (int, float)):
                min_value = rules.get("min")
                max_value = rules.get("max")
                
                if min_value is not None and value < min_value:
                    errors.append(f"Parameter '{param_name}' cannot be less than {min_value}")
                if max_value is not None and value > max_value:
                    errors.append(f"Parameter '{param_name}' cannot be greater than {max_value}")
            
            # Length validation for string parameters
            if isinstance(value, str):
                min_length = rules.get("minLength")
                max_length = rules.get("maxLength")
                pattern = rules.get("pattern")
                
                if min_length is not None and len(value) < min_length:
                    errors.append(f"Parameter '{param_name}' must be at least {min_length} characters")
                if max_length is not None and len(value) > max_length:
                    errors.append(f"Parameter '{param_name}' cannot exceed {max_length} characters")
        
        return errors
    
    def create_new_version(self) -> 'Strategy':
        """
        Create a new version of this strategy.
        
        Returns:
            New Strategy instance as the next version
        """
        # Create a copy of the current strategy as a new version
        new_version = Strategy(
            name=self.name,
            description=self.description,
            type=self.type,
            configuration=self.configuration.copy() if self.configuration else {},
            parameters=self.parameters.copy() if self.parameters else {},
            validation_rules=self.validation_rules.copy() if self.validation_rules else {},
            user_id=self.user_id,
            created_by_id=self.created_by_id,
            parent_version_id=self.id,
            version=self.version + 1
        )
        
        return new_version
    
    def update_performance_metrics(self, win_rate: float = None, profit_factor: float = None,
                                 sharpe_ratio: float = None, sortino_ratio: float = None,
                                 max_drawdown: float = None) -> None:
        """
        Update strategy performance metrics.
        
        Args:
            win_rate: Percentage of winning trades
            profit_factor: Ratio of gross profits to gross losses
            sharpe_ratio: Risk-adjusted return metric
            sortino_ratio: Downside risk-adjusted return metric
            max_drawdown: Maximum peak-to-trough decline
        """
        if win_rate is not None:
            self.win_rate = win_rate
        if profit_factor is not None:
            self.profit_factor = profit_factor
        if sharpe_ratio is not None:
            self.sharpe_ratio = sharpe_ratio
        if sortino_ratio is not None:
            self.sortino_ratio = sortino_ratio
        if max_drawdown is not None:
            self.max_drawdown = max_drawdown


class StrategyCategory(StrategyBaseModel, TimestampMixin):
    """
    Strategy category model for organizing strategies by type.
    
    Categories allow grouping strategies by their approach or methodology
    to facilitate discovery and management.
    """
    __tablename__ = "strategy_categories"
    
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text)
    
    # Relationships
    strategies = relationship("Strategy", secondary=strategy_category_association, 
                             back_populates="categories")


class StrategyBacktest(StrategyBaseModel, TimestampMixin, UserRelationMixin):
    """
    Strategy backtest results model.
    
    Stores the results of strategy backtests, including performance metrics,
    trade history, and configuration settings used for the test.
    """
    __tablename__ = "strategy_backtests"
    
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Backtest settings
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    initial_capital = Column(Float, nullable=False)
    parameters = Column(JSON)
    
    # Performance metrics
    total_return = Column(Float)
    annualized_return = Column(Float)
    win_rate = Column(Float)
    profit_factor = Column(Float)
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    max_drawdown = Column(Float)
    max_drawdown_duration = Column(Integer)  # Days
    total_trades = Column(Integer)
    
    # Detailed results
    equity_curve = Column(JSON)  # Time series of portfolio values
    trade_history = Column(JSON)  # List of all trades
    monthly_returns = Column(JSON)  # Monthly return percentages
    
    # Relationships
    strategy = relationship("Strategy", backref="backtests")
    
    def calculate_metrics(self, trades: List[Dict], equity_curve: List[Dict]) -> None:
        """
        Calculate performance metrics from trade history and equity curve.
        
        Args:
            trades: List of trade dictionaries
            equity_curve: List of equity values over time
        """
        if not trades or not equity_curve:
            return
            
        # Count total trades
        self.total_trades = len(trades)
        
        # Calculate win rate
        winning_trades = sum(1 for trade in trades if trade.get('profit', 0) > 0)
        self.win_rate = winning_trades / self.total_trades if self.total_trades > 0 else 0
        
        # Calculate profit factor
        gross_profit = sum(trade.get('profit', 0) for trade in trades if trade.get('profit', 0) > 0)
        gross_loss = abs(sum(trade.get('profit', 0) for trade in trades if trade.get('profit', 0) < 0))
        self.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate returns
        initial_equity = equity_curve[0]['equity'] if equity_curve else self.initial_capital
        final_equity = equity_curve[-1]['equity'] if equity_curve else self.initial_capital
        
        self.total_return = (final_equity - initial_equity) / initial_equity
        
        # Store the complete equity curve and trade history
        self.equity_curve = equity_curve
        self.trade_history = trades