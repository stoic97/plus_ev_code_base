"""
Account models for the trading application.

This module defines SQLAlchemy models related to user accounts, balances,
positions, transactions, and risk management for the trading platform.

Key features:
- Efficient indexing for high-performance queries during live trading
- Precise numeric types to avoid floating-point errors in financial calculations
- Risk management parameters integrated directly into the data model
- Hybrid properties for derived calculations to simplify business logic
- Comprehensive audit trail for regulatory compliance

Usage:
    These models provide the foundation for position tracking, risk management,
    and trading strategy constraints in the live trading environment.
"""

from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import List, Optional, Dict, Any
from sqlalchemy import (
    Column, Integer, String, Float, Numeric, DateTime, Date, 
    ForeignKey, Enum, Text, Boolean, Index, CheckConstraint, 
    UniqueConstraint, Table, func, select, and_, or_, text, event
)
from sqlalchemy.orm import relationship, validates, column_property
from sqlalchemy.ext.hybrid import hybrid_property, hybrid_method
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.sql.expression import cast
from typing import Tuple, Optional

from app.core.database import Base


# Many-to-many association table for position tags
position_tags = Table(
    'position_tags',
    Base.metadata,
    Column('position_id', Integer, ForeignKey('positions.id', ondelete="CASCADE"), primary_key=True),
    Column('tag_id', Integer, ForeignKey('tags.id', ondelete="CASCADE"), primary_key=True),
    Index('ix_position_tags_tag_id', 'tag_id')
)


class Account(Base):
    """
    User trading account information.
    
    Each user can have multiple accounts (e.g., main, demo, retirement).
    This is the central entity that ties together balances, positions, and transactions.
    """
    __tablename__ = "accounts"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    account_number = Column(String(50), unique=True, index=True, nullable=False)
    account_type = Column(
        String(20), 
        nullable=False, 
        comment="Type of account: main, demo, retirement, margin, etc."
    )
    name = Column(String(100), nullable=False)
    description = Column(Text)
    base_currency = Column(String(10), nullable=False, default="USD")
    status = Column(
        String(20), 
        nullable=False, 
        default="active",
        comment="Account status: active, suspended, closed, etc."
    )
    is_margin_enabled = Column(Boolean, default=False, nullable=False)
    margin_level = Column(
        Numeric(precision=10, scale=2), 
        default=0,
        comment="Current margin level as percentage (used margin / account equity)"
    )
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_activity_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="accounts")
    balances = relationship(
        "Balance", 
        back_populates="account", 
        cascade="all, delete-orphan", 
        lazy="selectin"  # Optimize for eager loading of balances with account
    )
    positions = relationship(
        "Position", 
        back_populates="account", 
        cascade="all, delete-orphan",
        lazy="selectin"  # Optimize for eager loading of positions with account
    )
    transactions = relationship(
        "Transaction", 
        back_populates="account",
        lazy="dynamic"  # Use dynamic loading for transactions as there could be many
    )
    risk_profile = relationship(
        "RiskProfile", 
        back_populates="account", 
        uselist=False, 
        cascade="all, delete-orphan",
        lazy="joined"  # Always load risk profile with account
    )
    
    # Indexes and constraints
    __table_args__ = (
        # Optimize queries that filter accounts by user and type
        Index("ix_accounts_user_id_account_type", user_id, account_type),
        # Ensure status is one of allowed values
        CheckConstraint(
            "status IN ('active', 'suspended', 'closed', 'liquidation', 'pending')",
            name="ck_accounts_status"
        ),
        # Ensure account_type is one of allowed values
        CheckConstraint(
            "account_type IN ('main', 'demo', 'retirement', 'margin', 'institutional')",
            name="ck_accounts_type"
        ),
    )
    
    @validates('status')
    def validate_status(self, key, status):
        """Validate that account status is one of the allowed values."""
        allowed_statuses = ['active', 'suspended', 'closed', 'liquidation', 'pending']
        if status not in allowed_statuses:
            raise ValueError(f"Status must be one of: {', '.join(allowed_statuses)}")
        return status
    
    @hybrid_property
    def total_equity(self) -> Decimal:
        """
        Calculate total account equity (cash + positions value).
        
        This is a critical value for risk management and position sizing.
        """
        # Sum all cash balances converted to base currency
        # Note: This is simplified and should use actual FX rates in production
        cash_total = sum(balance.amount for balance in self.balances if balance.asset_type == "currency")
        
        # Sum all positions market value in base currency
        positions_value = sum(position.market_value for position in self.positions)
        
        return cash_total + positions_value
    
    @hybrid_property
    def available_margin(self) -> Decimal:
        """
        Calculate available margin for new positions.
        
        For margin accounts, this represents buying power.
        """
        if not self.is_margin_enabled:
            # For cash accounts, available margin is just available cash
            return sum(balance.available for balance in self.balances if balance.asset_type == "currency")
        
        # For margin accounts, apply margin multiplier (simplified)
        # In production, this would use asset-specific margin requirements
        if self.risk_profile:
            margin_multiplier = self.risk_profile.max_leverage
        else:
            margin_multiplier = Decimal('1.0')  # Default to cash account if no risk profile
            
        return self.total_equity * margin_multiplier - sum(p.margin_used for p in self.positions)
    
    @hybrid_property
    def unrealized_pnl(self) -> Decimal:
        """Calculate total unrealized profit/loss across all positions."""
        return sum(position.unrealized_pnl for position in self.positions)
    
    @hybrid_property
    def daily_pnl(self) -> Decimal:
        """Calculate profit/loss for the current trading day."""
        today = date.today()
        
        # Get today's realized P&L from transactions
        realized_pnl = self.transactions.filter(
            Transaction.transaction_type.in_(["trade", "fee"]),
            func.date(Transaction.created_at) == today
        ).with_entities(func.sum(Transaction.amount)).scalar() or Decimal('0')
        
        # Add unrealized P&L
        return realized_pnl + self.unrealized_pnl
    
    def update_activity_timestamp(self):
        """Update the last activity timestamp."""
        self.last_activity_at = datetime.utcnow()
    
    def get_position_by_symbol(self, symbol: str) -> Optional["Position"]:
        """
        Get a position by symbol.
        
        Optimized method to find a position without hitting the database again
        if positions are already loaded.
        
        Args:
            symbol: Trading symbol to look up
            
        Returns:
            Position object or None if not found
        """
        for position in self.positions:
            if position.symbol == symbol:
                return position
        return None
    
    def get_balance_by_asset(self, asset: str) -> Optional["Balance"]:
        """
        Get a balance by asset symbol.
        
        Optimized method to find a balance without hitting the database again
        if balances are already loaded.
        
        Args:
            asset: Asset symbol to look up
            
        Returns:
            Balance object or None if not found
        """
        for balance in self.balances:
            if balance.asset == asset:
                return balance
        return None


class Balance(Base):
    """
    Account balance for a specific asset or currency.
    
    Each account can have multiple balances (USD, EUR, BTC, etc.).
    Balances track both total amounts and what portion is available vs. reserved.
    """
    __tablename__ = "balances"
    
    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey("accounts.id", ondelete="CASCADE"), nullable=False)
    asset = Column(String(20), nullable=False, comment="Asset code (USD, EUR, BTC, etc.)")
    asset_type = Column(
        String(20), 
        nullable=False, 
        comment="Type of asset: currency, crypto, stock, etc."
    )
    amount = Column(
        Numeric(precision=18, scale=8), 
        nullable=False, 
        default=0,
        comment="Total balance amount"
    )
    available = Column(
        Numeric(precision=18, scale=8), 
        nullable=False, 
        default=0,
        comment="Amount available for trading (not in orders)"
    )
    reserved = Column(
        Numeric(precision=18, scale=8), 
        nullable=False, 
        default=0,
        comment="Amount reserved in open orders"
    )
    updated_at = Column(
        DateTime(timezone=True), 
        nullable=False, 
        server_default=func.now(), 
        onupdate=func.now()
    )
    
    # Relationships
    account = relationship("Account", back_populates="balances")
    
    # Constraints and indexes
    __table_args__ = (
        # Each account can have only one balance per asset
        UniqueConstraint("account_id", "asset", name="uc_balances_account_asset"),
        # Optimize lookup by account_id and asset
        Index("ix_balances_account_id_asset", account_id, asset),
        # Ensure amount = available + reserved
        CheckConstraint(
            "amount = available + reserved",
            name="ck_balances_amount_equals_available_plus_reserved"
        ),
        # Ensure all amounts are non-negative
        CheckConstraint(
            "amount >= 0 AND available >= 0 AND reserved >= 0",
            name="ck_balances_non_negative"
        ),
        # Ensure asset_type is one of allowed values
        CheckConstraint(
            "asset_type IN ('currency', 'crypto', 'stock', 'commodity', 'bond')",
            name="ck_balances_asset_type"
        ),
    )
    
    @validates('available', 'reserved')
    def validate_amounts(self, key, value):
        """Validate balance components and update total amount."""
        value = Decimal(str(value))
        if key == 'available':
            self.amount = value + (self.reserved or Decimal('0'))
        elif key == 'reserved':
            self.amount = (self.available or Decimal('0')) + value
        
        # Ensure non-negative values
        if value < 0:
            raise ValueError(f"{key} cannot be negative")
        
        return value
    
    @hybrid_property
    def locked_percentage(self) -> float:
        """Calculate percentage of balance that is locked in orders."""
        if self.amount == 0:
            return 0
        return float(self.reserved / self.amount * 100)
    
    def reserve(self, amount: Decimal) -> bool:
        """
        Reserve an amount from available balance.
        
        Args:
            amount: Amount to reserve
            
        Returns:
            True if successful, False if insufficient available balance
        """
        if amount > self.available:
            return False
        
        self.available -= amount
        self.reserved += amount
        return True
    
    def release(self, amount: Decimal) -> bool:
        """
        Release an amount from reserved back to available.
        
        Args:
            amount: Amount to release
            
        Returns:
            True if successful, False if insufficient reserved balance
        """
        if amount > self.reserved:
            return False
        
        self.reserved -= amount
        self.available += amount
        return True
    
    def add(self, amount: Decimal) -> None:
        """
        Add to available balance.
        
        Args:
            amount: Amount to add (can be negative for subtracting)
        """
        if self.available + amount < 0:
            raise ValueError("Cannot reduce available balance below zero")
        
        self.available += amount
        self.amount = self.available + self.reserved


class Position(Base):
    """
    Open trading position in the portfolio.
    
    Represents holdings of a specific instrument with detailed tracking
    of profit/loss, risk parameters, and exposure.
    """
    __tablename__ = "positions"
    
    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey("accounts.id", ondelete="CASCADE"), nullable=False)
    symbol = Column(String(20), nullable=False, comment="Trading symbol (AAPL, BTCUSD, etc.)")
    instrument_type = Column(
        String(20), 
        nullable=False, 
        comment="Type of instrument: stock, crypto, forex, futures, etc."
    )
    direction = Column(
        String(5), 
        nullable=False, 
        default="long",
        comment="Position direction: long or short"
    )
    quantity = Column(
        Numeric(precision=18, scale=8), 
        nullable=False,
        comment="Position size in units"
    )
    average_entry_price = Column(
        Numeric(precision=18, scale=8), 
        nullable=False,
        comment="Average price at which the position was opened"
    )
    current_price = Column(
        Numeric(precision=18, scale=8), 
        nullable=False,
        comment="Latest market price of the instrument"
    )
    opened_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    
    # P&L tracking
    unrealized_pnl = Column(
        Numeric(precision=18, scale=8), 
        nullable=False, 
        default=0,
        comment="Current unrealized profit/loss"
    )
    realized_pnl = Column(
        Numeric(precision=18, scale=8), 
        nullable=False, 
        default=0,
        comment="Cumulative realized profit/loss from partial closes"
    )
    
    # Risk management fields
    stop_loss = Column(
        Numeric(precision=18, scale=8),
        comment="Stop loss price level"
    )
    take_profit = Column(
        Numeric(precision=18, scale=8),
        comment="Take profit price level"
    )
    trailing_stop = Column(
        Numeric(precision=18, scale=8),
        comment="Trailing stop distance in price units"
    )
    trailing_stop_activation = Column(
        Numeric(precision=18, scale=8),
        comment="Price at which trailing stop activates"
    )
    liquidation_price = Column(
        Numeric(precision=18, scale=8),
        comment="Price at which position would be liquidated (for margin)"
    )
    
    # Position metadata
    strategy_id = Column(
        String(50),
        comment="ID of the strategy that opened this position"
    )
    notes = Column(Text)
    is_hedged = Column(
        Boolean, 
        default=False,
        comment="Whether this position is part of a hedge"
    )
    hedge_id = Column(
        String(50),
        comment="ID that links related hedged positions"
    )
    
    # For margin trading
    leverage = Column(
        Numeric(precision=10, scale=2), 
        default=1.0,
        comment="Leverage used for this position"
    )
    margin_used = Column(
        Numeric(precision=18, scale=8), 
        default=0,
        comment="Amount of margin allocated to this position"
    )
    
    # Relationships
    account = relationship("Account", back_populates="positions")
    tags = relationship(
        "Tag", 
        secondary=position_tags, 
        backref="positions",
        lazy="selectin"
    )
    
    # Constraints and indexes
    __table_args__ = (
        # Each account can have only one position per symbol
        UniqueConstraint("account_id", "symbol", name="uc_positions_account_symbol"),
        # Optimize lookup by account_id and symbol
        Index("ix_positions_account_id_symbol", account_id, symbol),
        # Additional indexes for filtered queries
        Index("ix_positions_instrument_type", instrument_type),
        Index("ix_positions_strategy_id", strategy_id),
        # Ensure direction is either 'long' or 'short'
        CheckConstraint(
            "direction IN ('long', 'short')",
            name="ck_positions_direction"
        ),
        # Ensure quantity is positive
        CheckConstraint(
            "quantity > 0",
            name="ck_positions_quantity_positive"
        ),
        # Ensure leverage is positive
        CheckConstraint(
            "leverage > 0",
            name="ck_positions_leverage_positive"
        ),
    )
    
    @validates('direction')
    def validate_direction(self, key, direction):
        """Validate position direction."""
        allowed_directions = ['long', 'short']
        if direction not in allowed_directions:
            raise ValueError(f"Direction must be one of: {', '.join(allowed_directions)}")
        return direction
    
    @hybrid_property
    def market_value(self) -> Decimal:
        """
        Calculate current market value of position.
        
        For long positions: quantity * price
        For short positions: quantity * price (negative)
        """
        value = self.quantity * self.current_price
        return value if self.direction == 'long' else -value
    
    @hybrid_property
    def cost_basis(self) -> Decimal:
        """Calculate the original cost of the position."""
        value = self.quantity * self.average_entry_price
        return value if self.direction == 'long' else -value
    
    @hybrid_property
    def pnl_percentage(self) -> float:
        """
        Calculate percentage profit/loss of position.
        
        Takes direction into account.
        """
        if self.average_entry_price == 0:
            return 0
            
        if self.direction == 'long':
            return float((self.current_price - self.average_entry_price) / self.average_entry_price * 100)
        else:
            return float((self.average_entry_price - self.current_price) / self.average_entry_price * 100)
    
    @hybrid_method
    def would_hit_stop_loss(self, price: Decimal) -> bool:
        """
        Check if a given price would trigger the stop loss.
        
        Args:
            price: Price to check
            
        Returns:
            True if the price would trigger stop loss, False otherwise
        """
        if not self.stop_loss:
            return False
            
        if self.direction == 'long':
            return price <= self.stop_loss
        else:
            return price >= self.stop_loss
            
    @hybrid_method
    def would_hit_take_profit(self, price: Decimal) -> bool:
        """
        Check if a given price would trigger the take profit.
        
        Args:
            price: Price to check
            
        Returns:
            True if the price would trigger take profit, False otherwise
        """
        if not self.take_profit:
            return False
            
        if self.direction == 'long':
            return price >= self.take_profit
        else:
            return price <= self.take_profit
    
    def update_price(self, new_price: Decimal) -> None:
        """
        Update the current price and recalculate P&L.
        
        Args:
            new_price: New market price
        """
        self.current_price = new_price
        
        # Update unrealized P&L
        if self.direction == 'long':
            self.unrealized_pnl = (self.current_price - self.average_entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.average_entry_price - self.current_price) * self.quantity
            
        # Update trailing stop if activated
        if self.trailing_stop and self.trailing_stop_activation:
            if self.direction == 'long' and new_price >= self.trailing_stop_activation:
                # Calculate new stop loss based on trailing distance
                new_stop = new_price - self.trailing_stop
                if not self.stop_loss or new_stop > self.stop_loss:
                    self.stop_loss = new_stop
            elif self.direction == 'short' and new_price <= self.trailing_stop_activation:
                # Calculate new stop loss based on trailing distance
                new_stop = new_price + self.trailing_stop
                if not self.stop_loss or new_stop < self.stop_loss:
                    self.stop_loss = new_stop
    
    def add_quantity(self, quantity: Decimal, price: Decimal) -> None:
        """
        Add to position quantity and recalculate average entry price.
        
        Args:
            quantity: Quantity to add
            price: Price of the new quantity
        """
        if quantity <= 0:
            raise ValueError("Quantity to add must be positive")
            
        # Calculate new average entry price
        total_cost = self.average_entry_price * self.quantity
        additional_cost = price * quantity
        new_quantity = self.quantity + quantity
        
        self.average_entry_price = (total_cost + additional_cost) / new_quantity
        self.quantity = new_quantity
        
        # Update P&L
        self.update_price(self.current_price)
    
    def reduce_quantity(self, quantity: Decimal, price: Decimal) -> Decimal:
        """
        Reduce position quantity and calculate realized P&L.
        
        Args:
            quantity: Quantity to reduce
            price: Price at which reduction occurs
            
        Returns:
            Realized P&L from this reduction
        """
        if quantity <= 0:
            raise ValueError("Quantity to reduce must be positive")
            
        if quantity > self.quantity:
            raise ValueError("Cannot reduce by more than current quantity")
            
        # Calculate realized P&L
        if self.direction == 'long':
            realized_pnl = (price - self.average_entry_price) * quantity
        else:
            realized_pnl = (self.average_entry_price - price) * quantity
            
        # Update position
        self.quantity -= quantity
        self.realized_pnl += realized_pnl
        
        # Update P&L (average_entry_price remains unchanged)
        self.update_price(self.current_price)
        
        return realized_pnl


class Transaction(Base):
    """
    Account transaction history.
    
    Records all activities affecting the account (deposits, withdrawals, trades, fees).
    This provides a complete audit trail for compliance and reconciliation.
    """
    __tablename__ = "transactions"
    
    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey("accounts.id", ondelete="CASCADE"), nullable=False)
    transaction_type = Column(
        String(20), 
        nullable=False,
        comment="Type of transaction: deposit, withdrawal, trade, fee, dividend, etc."
    )
    asset = Column(String(20), nullable=False, comment="Asset involved (USD, EUR, BTC, AAPL, etc.)")
    amount = Column(
        Numeric(precision=18, scale=8), 
        nullable=False,
        comment="Transaction amount (positive for credits, negative for debits)"
    )
    fee = Column(
        Numeric(precision=18, scale=8), 
        nullable=False, 
        default=0,
        comment="Fee amount (always positive)"
    )
    price = Column(
        Numeric(precision=18, scale=8),
        comment="Price per unit (for trades)"
    )
    quantity = Column(
        Numeric(precision=18, scale=8),
        comment="Quantity of asset (for trades)"
    )
    symbol = Column(
        String(20),
        comment="Trading symbol (for trades)"
    )
    direction = Column(
        String(5),
        comment="Trade direction: buy, sell (for trades)"
    )
    reference_id = Column(
        String(100),
        comment="Order ID, payment reference, etc."
    )
    counterparty = Column(
        String(100),
        comment="Exchange, payment processor, etc."
    )
    status = Column(
        String(20), 
        nullable=False, 
        default="completed",
        comment="Transaction status: pending, completed, failed, canceled"
    )
    description = Column(Text)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # For trade settlement
    settlement_date = Column(
        Date,
        comment="Date when trade settles (T+2 for stocks, etc.)"
    )
    is_settled = Column(
        Boolean, 
        default=False,
        comment="Whether the transaction has settled"
    )
    
    # Strategy tracking
    strategy_id = Column(
        String(50),
        comment="ID of the strategy that generated this transaction"
    )
    
    # Relationships
    account = relationship("Account", back_populates="transactions")
    
    # Indexes for efficient querying
    __table_args__ = (
        # Optimize lookups by account and time (common for reports)
        Index("ix_transactions_account_id_created_at", account_id, created_at),
        # Optimize lookups by reference ID (for order matching)
        Index("ix_transactions_reference_id", reference_id),
        # Optimize lookups by type (for filtering)
        Index("ix_transactions_type", transaction_type),
        # Optimize lookups by status (for pending transactions)
        Index("ix_transactions_status", status),
        # Optimize lookups by strategy (for performance analysis)
        Index("ix_transactions_strategy_id", strategy_id),
        # Ensure status is one of allowed values
        CheckConstraint(
            "status IN ('pending', 'completed', 'failed', 'canceled')",
            name="ck_transactions_status"
        ),
        # Ensure transaction_type is one of allowed values
        CheckConstraint(
            "transaction_type IN ('deposit', 'withdrawal', 'trade', 'fee', 'dividend', 'interest', 'adjustment')",
            name="ck_transactions_type"
        ),
        # Ensure fee is non-negative
        CheckConstraint(
            "fee >= 0",
            name="ck_transactions_fee_non_negative"
        ),
        # Ensure trade direction is valid when present
        CheckConstraint(
            "direction IS NULL OR direction IN ('buy', 'sell')",
            name="ck_transactions_direction"
        )
    )
    
    @validates('status')
    def validate_status(self, key, status):
        """Validate transaction status."""
        allowed_statuses = ['pending', 'completed', 'failed', 'canceled']
        if status not in allowed_statuses:
            raise ValueError(f"Status must be one of: {', '.join(allowed_statuses)}")
        return status
    
    @validates('transaction_type')
    def validate_type(self, key, transaction_type):
        """Validate transaction type."""
        allowed_types = ['deposit', 'withdrawal', 'trade', 'fee', 'dividend', 'interest', 'adjustment']
        if transaction_type not in allowed_types:
            raise ValueError(f"Transaction type must be one of: {', '.join(allowed_types)}")
        return transaction_type
    
    @validates('direction')
    def validate_direction(self, key, direction):
        """Validate trade direction."""
        if direction is not None:
            allowed_directions = ['buy', 'sell']
            if direction not in allowed_directions:
                raise ValueError(f"Direction must be one of: {', '.join(allowed_directions)}")
        return direction


class Tag(Base):
    """
    Tags for categorizing positions.
    
    Used for grouping positions by strategy, sector, risk level, etc.
    """
    __tablename__ = "tags"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), nullable=False, unique=True)
    description = Column(Text)
    color = Column(String(7))  # Hex color code


class RiskProfile(Base):
    """
    Risk management parameters for an account.
    
    Defines limits and thresholds for risk control.
    These parameters are used by trading strategies to constrain 
    position sizes and enforce risk limits.
    """
    __tablename__ = "risk_profiles"
    
    id = Column(Integer, primary_key=True)
    account_id = Column(Integer, ForeignKey("accounts.id", ondelete="CASCADE"), unique=True, nullable=False)
    
    # Position size limits
    max_position_size_percentage = Column(
        Numeric(precision=5, scale=2), 
        nullable=False, 
        default=5.0,
        comment="Maximum position size as percentage of portfolio"
    )
    max_position_size_absolute = Column(
        Numeric(precision=18, scale=8),
        comment="Absolute maximum position size in base currency"
    )
    
    # Concentration limits
    max_sector_exposure_percentage = Column(
        Numeric(precision=5, scale=2), 
        nullable=False, 
        default=20.0,
        comment="Maximum exposure to a single sector as percentage of portfolio"
    )
    max_asset_class_exposure_percentage = Column(
        Numeric(precision=5, scale=2), 
        nullable=False, 
        default=40.0,
        comment="Maximum exposure to a single asset class as percentage of portfolio"
    )
    
    # Loss limits
    max_daily_loss_percentage = Column(
        Numeric(precision=5, scale=2), 
        nullable=False, 
        default=3.0,
        comment="Maximum allowed daily loss as percentage of portfolio"
    )
    max_total_loss_percentage = Column(
        Numeric(precision=5, scale=2), 
        nullable=False, 
        default=15.0,
        comment="Maximum allowed total drawdown as percentage of portfolio"
    )
    trailing_stop_activation = Column(
        Numeric(precision=5, scale=2), 
        default=10.0,
        comment="Profit percentage at which to activate trailing stop"
    )
    
    # Leverage limits
    max_leverage = Column(
        Numeric(precision=5, scale=2), 
        nullable=False, 
        default=1.0,
        comment="Maximum allowed leverage (1.0 = no leverage)"
    )
    margin_call_level = Column(
        Numeric(precision=5, scale=2), 
        nullable=False, 
        default=80.0,
        comment="Margin level percentage at which to issue margin call"
    )
    liquidation_level = Column(
        Numeric(precision=5, scale=2), 
        nullable=False, 
        default=95.0,
        comment="Margin level percentage at which to liquidate positions"
    )
    
    # Strategy constraints
    max_trades_per_day = Column(
        Integer, 
        default=20,
        comment="Maximum number of trades allowed per day"
    )
    min_trade_interval_seconds = Column(
        Integer, 
        default=60,
        comment="Minimum time between trades in seconds"
    )
    max_drawdown_pause_percentage = Column(
        Numeric(precision=5, scale=2),
        default=10.0,
        comment="Drawdown percentage that triggers trading pause"
    )
    
    # Risk status flags
    is_active = Column(
        Boolean, 
        nullable=False, 
        default=True,
        comment="Whether risk management is active"
    )
    is_in_cooldown = Column(
        Boolean, 
        nullable=False, 
        default=False,
        comment="Whether account is in post-loss cooldown period"
    )
    cooldown_until = Column(
        DateTime(timezone=True),
        comment="Timestamp until which trading is paused after loss limit hit"
    )
    
    # Timestamps
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    account = relationship("Account", back_populates="risk_profile")
    
    # Constraints
    __table_args__ = (
        # Ensure percentages are in valid range
        CheckConstraint(
            "max_position_size_percentage > 0 AND max_position_size_percentage <= 100",
            name="ck_risk_profiles_position_size_percentage"
        ),
        CheckConstraint(
            "max_sector_exposure_percentage > 0 AND max_sector_exposure_percentage <= 100",
            name="ck_risk_profiles_sector_exposure_percentage"
        ),
        CheckConstraint(
            "max_asset_class_exposure_percentage > 0 AND max_asset_class_exposure_percentage <= 100",
            name="ck_risk_profiles_asset_class_exposure_percentage"
        ),
        CheckConstraint(
            "max_daily_loss_percentage > 0 AND max_daily_loss_percentage <= 100",
            name="ck_risk_profiles_daily_loss_percentage"
        ),
        CheckConstraint(
            "max_total_loss_percentage > 0 AND max_total_loss_percentage <= 100",
            name="ck_risk_profiles_total_loss_percentage"
        ),
        CheckConstraint(
            "max_leverage >= 1.0",
            name="ck_risk_profiles_leverage"
        ),
        CheckConstraint(
            "margin_call_level < liquidation_level",
            name="ck_risk_profiles_margin_levels"
        ),
    )
    
    def is_trade_allowed(self, days_lookback: int = 1) -> bool:
        """
        Check if trading is allowed based on cooldown status.
        
        Args:
            days_lookback: Number of days to look back for trade count
            
        Returns:
            True if trading is allowed, False otherwise
        """
        if not self.is_active:
            return False
            
        if self.is_in_cooldown and self.cooldown_until:
            if datetime.utcnow() < self.cooldown_until:
                return False
                
        return True
    
    def has_reached_daily_trade_limit(self, session) -> bool:
        """
        Check if account has reached daily trade limit.
        
        Args:
            session: SQLAlchemy session
            
        Returns:
            True if limit reached, False otherwise
        """
        if not self.max_trades_per_day:
            return False
            
        # Count trades for today
        today = date.today()
        trade_count = session.query(func.count(Transaction.id)).filter(
            Transaction.account_id == self.account_id,
            Transaction.transaction_type == 'trade',
            func.date(Transaction.created_at) == today
        ).scalar() or 0
        
        return trade_count >= self.max_trades_per_day
    
    def check_position_size_limit(self, symbol: str, quantity: Decimal, price: Decimal, instrument_type: str, session) -> bool:
        """
        Check if a new position would exceed size limits.
        
        Args:
            symbol: Trading symbol
            quantity: Position quantity
            price: Current price
            instrument_type: Type of instrument
            session: SQLAlchemy session
            
        Returns:
            True if within limits, False if exceeds limits
        """
        # Get the account
        account = session.query(Account).filter(Account.id == self.account_id).one()
        
        # Calculate position value
        position_value = quantity * price
        
        # Check against percentage limit
        if self.max_position_size_percentage:
            max_value = account.total_equity * (self.max_position_size_percentage / 100)
            if position_value > max_value:
                return False
                
        # Check against absolute limit
        if self.max_position_size_absolute and position_value > self.max_position_size_absolute:
            return False
            
        # Check for existing positions of the same instrument type
        existing_exposure = session.query(func.sum(Position.market_value)).filter(
            Position.account_id == self.account_id,
            Position.instrument_type == instrument_type
        ).scalar() or 0
        
        # Calculate new total exposure for this asset class
        new_total_exposure = existing_exposure + position_value
        
        # Check against asset class limit
        if self.max_asset_class_exposure_percentage:
            max_asset_class_value = account.total_equity * (self.max_asset_class_exposure_percentage / 100)
            if new_total_exposure > max_asset_class_value:
                return False
                
        return True


# Event listeners and hooks for account models

@event.listens_for(RiskProfile, 'after_insert')
def risk_profile_after_insert(mapper, connection, target):
    """Add log entry when a risk profile is created."""
    # This could be expanded to notify risk management systems
    pass


@event.listens_for(Position, 'before_update')
def position_before_update(mapper, connection, target):
    """
    Check for stop loss or take profit triggers before position updates.
    
    This allows for immediate detection of price levels that would trigger
    stop loss or take profit orders.
    """
    if hasattr(target, '_sa_instance_state') and not target._sa_instance_state.modified:
        return
        
    # Check if current_price has been updated
    if 'current_price' in target._sa_instance_state.committed_state:
        old_price = target._sa_instance_state.committed_state['current_price']
        new_price = target.current_price
        
        # Check for stop loss hit
        if target.stop_loss and target.would_hit_stop_loss(new_price) and not target.would_hit_stop_loss(old_price):
            # Log potential stop loss hit - in a real system, this would trigger an order
            pass
            
        # Check for take profit hit
        if target.take_profit and target.would_hit_take_profit(new_price) and not target.would_hit_take_profit(old_price):
            # Log potential take profit hit - in a real system, this would trigger an order
            pass


@event.listens_for(Account, 'before_update')
def account_before_update(mapper, connection, target):
    """
    Check for margin call conditions before account updates.
    
    This allows for automatic detection of margin call situations.
    """
    if not target.is_margin_enabled or not hasattr(target, '_sa_instance_state'):
        return
        
    # Check if margin_level has been updated
    if 'margin_level' in target._sa_instance_state.committed_state:
        old_level = target._sa_instance_state.committed_state['margin_level']
        new_level = target.margin_level
        
        # Get risk profile
        if target.risk_profile:
            # Check for margin call condition
            if new_level >= target.risk_profile.margin_call_level and old_level < target.risk_profile.margin_call_level:
                # Log margin call - in a real system, this would trigger notifications
                pass
                
            # Check for liquidation condition
            if new_level >= target.risk_profile.liquidation_level and old_level < target.risk_profile.liquidation_level:
                # Log liquidation event - in a real system, this would trigger forced position closure
                pass


# Utility functions for common account operations

def create_transaction(
    session,
    account_id: int, 
    transaction_type: str, 
    asset: str, 
    amount: Decimal,
    price: Optional[Decimal] = None,
    quantity: Optional[Decimal] = None,
    symbol: Optional[str] = None,
    direction: Optional[str] = None,
    fee: Decimal = Decimal('0'),
    reference_id: Optional[str] = None,
    description: Optional[str] = None,
    strategy_id: Optional[str] = None,
    counterparty: Optional[str] = None
) -> Transaction:
    """
    Create a new transaction and update relevant balances.
    
    This function handles both the transaction record and the associated
    balance updates in a single atomic operation.
    
    Args:
        session: SQLAlchemy session
        account_id: Account ID
        transaction_type: Type of transaction (deposit, withdrawal, trade, etc.)
        asset: Asset code
        amount: Transaction amount
        price: Price per unit (for trades)
        quantity: Quantity (for trades)
        symbol: Trading symbol (for trades)
        direction: Trade direction (for trades)
        fee: Fee amount
        reference_id: External reference ID
        description: Transaction description
        strategy_id: Strategy that generated the transaction
        counterparty: External party involved
        
    Returns:
        Created Transaction object
    """
    # Create transaction record
    transaction = Transaction(
        account_id=account_id,
        transaction_type=transaction_type,
        asset=asset,
        amount=amount,
        price=price,
        quantity=quantity,
        symbol=symbol,
        direction=direction,
        fee=fee,
        reference_id=reference_id,
        description=description,
        strategy_id=strategy_id,
        counterparty=counterparty
    )
    
    # Add to session
    session.add(transaction)
    
    # Update balance if not a trade (trades update positions)
    if transaction_type in ['deposit', 'withdrawal', 'dividend', 'interest', 'adjustment']:
        # Get or create balance for this asset
        balance = session.query(Balance).filter(
            Balance.account_id == account_id,
            Balance.asset == asset
        ).first()
        
        if not balance:
            # Determine asset type (simplified)
            asset_type = "currency"  # Default assumption
            
            balance = Balance(
                account_id=account_id,
                asset=asset,
                asset_type=asset_type,
                amount=Decimal('0'),
                available=Decimal('0'),
                reserved=Decimal('0')
            )
            session.add(balance)
            
        # Update balance
        balance.add(amount)
        
    # For trades, handle position updates separately using trade_execution function
    
    # Update account last activity timestamp
    account = session.query(Account).filter(Account.id == account_id).one()
    account.update_activity_timestamp()
    
    return transaction


def trade_execution(
    session,
    account_id: int,
    symbol: str,
    instrument_type: str,
    direction: str,  # 'buy' or 'sell'
    quantity: Decimal,
    price: Decimal,
    fee: Decimal = Decimal('0'),
    strategy_id: Optional[str] = None,
    stop_loss: Optional[Decimal] = None,
    take_profit: Optional[Decimal] = None,
    reference_id: Optional[str] = None,
    counterparty: Optional[str] = None
) -> Tuple[Transaction, Optional[Position]]:
    """
    Execute a trade including all related records and checks.
    
    This function handles:
    1. Risk checks
    2. Balance updates
    3. Position updates
    4. Transaction creation
    
    Args:
        session: SQLAlchemy session
        account_id: Account ID
        symbol: Trading symbol
        instrument_type: Type of instrument
        direction: 'buy' or 'sell'
        quantity: Trade quantity
        price: Trade price
        fee: Trade fee
        strategy_id: Strategy that generated the trade
        stop_loss: Stop loss price
        take_profit: Take profit price
        reference_id: External reference ID
        counterparty: Exchange or counterparty
        
    Returns:
        Tuple of (Transaction, Position)
    """
    # Get account
    account = session.query(Account).filter(Account.id == account_id).one()
    
    # Check risk limits if this is a new position or increasing position
    if direction == 'buy' and account.risk_profile:
        if not account.risk_profile.is_trade_allowed():
            raise ValueError("Trading not allowed due to risk profile constraints")
            
        if account.risk_profile.has_reached_daily_trade_limit(session):
            raise ValueError("Daily trade limit reached")
            
        # Skip position size check for sells
        if not account.risk_profile.check_position_size_limit(
            symbol, quantity, price, instrument_type, session
        ):
            raise ValueError("Trade exceeds position size limits")
    
    # Calculate trade value
    trade_value = quantity * price
    
    # Get or create position
    position = account.get_position_by_symbol(symbol)
    
    # Update or create position
    if position is None:
        # For sells, can't sell what you don't have
        if direction == 'sell':
            raise ValueError(f"Cannot sell {symbol}: no position exists")
            
        # Create new long position
        position = Position(
            account_id=account_id,
            symbol=symbol,
            instrument_type=instrument_type,
            direction='long',
            quantity=quantity,
            average_entry_price=price,
            current_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy_id=strategy_id
        )
        session.add(position)
    else:
        # Position exists - handle differently based on direction
        if direction == 'buy':
            if position.direction == 'long':
                # Adding to long position
                position.add_quantity(quantity, price)
                # Update stop/take if provided
                if stop_loss:
                    position.stop_loss = stop_loss
                if take_profit:
                    position.take_profit = take_profit
            else:
                # Reducing short position (closing)
                if quantity > position.quantity:
                    raise ValueError(f"Cannot close more than existing position: {position.quantity}")
                position.reduce_quantity(quantity, price)
                
                # If fully closed, remove the position
                if position.quantity == 0:
                    session.delete(position)
                    position = None
        else:  # sell
            if position.direction == 'long':
                # Reducing long position (closing)
                if quantity > position.quantity:
                    raise ValueError(f"Cannot close more than existing position: {position.quantity}")
                position.reduce_quantity(quantity, price)
                
                # If fully closed, remove the position
                if position.quantity == 0:
                    session.delete(position)
                    position = None
            else:
                # Adding to short position
                position.add_quantity(quantity, price)
                # Update stop/take if provided
                if stop_loss:
                    position.stop_loss = stop_loss
                if take_profit:
                    position.take_profit = take_profit
    
    # Create transaction record
    transaction = create_transaction(
        session=session,
        account_id=account_id,
        transaction_type='trade',
        asset=symbol,
        amount=-trade_value if direction == 'buy' else trade_value,
        price=price,
        quantity=quantity,
        symbol=symbol,
        direction=direction,
        fee=fee,
        reference_id=reference_id,
        strategy_id=strategy_id,
        counterparty=counterparty,
        description=f"{direction.capitalize()} {quantity} {symbol} @ {price}"
    )
    
    # Update account last activity
    account.update_activity_timestamp()
    
    return transaction, position


def get_account_risk_summary(session, account_id: int) -> Dict[str, Any]:
    """
    Get comprehensive risk metrics for an account.
    
    This function calculates key risk indicators for real-time monitoring
    and decision making in the trading system.
    
    Args:
        session: SQLAlchemy session
        account_id: Account ID
        
    Returns:
        Dictionary of risk metrics
    """
    # Get account with related data
    account = session.query(Account).filter(Account.id == account_id).one()
    
    # Calculate key risk metrics
    total_equity = float(account.total_equity)
    positions_value = sum(float(p.market_value) for p in account.positions)
    exposure_percentage = (positions_value / total_equity * 100) if total_equity else 0
    
    # Group by asset class for concentration analysis
    asset_class_exposure = {}
    for position in account.positions:
        asset_class = position.instrument_type
        if asset_class not in asset_class_exposure:
            asset_class_exposure[asset_class] = 0
        asset_class_exposure[asset_class] += float(position.market_value)
    
    # Convert to percentages
    asset_class_percentages = {
        k: (v / total_equity * 100) if total_equity else 0 
        for k, v in asset_class_exposure.items()
    }
    
    # Calculate daily P&L
    daily_pnl = session.query(func.sum(Transaction.amount)).filter(
        Transaction.account_id == account_id,
        Transaction.transaction_type.in_(["trade", "fee"]),
        func.date(Transaction.created_at) == date.today()
    ).scalar() or 0
    
    daily_pnl_percentage = (float(daily_pnl) / total_equity * 100) if total_equity else 0
    
    # Get largest position
    largest_position_value = max((float(p.market_value) for p in account.positions), default=0)
    largest_position_percentage = (largest_position_value / total_equity * 100) if total_equity else 0
    
    # Calculate risk profile status
    risk_status = "inactive"
    if account.risk_profile:
        if account.risk_profile.is_in_cooldown:
            risk_status = "cooldown"
        elif account.risk_profile.is_active:
            risk_status = "active"
    
    # Check if daily loss limit is approaching
    daily_loss_warning = False
    if account.risk_profile and account.risk_profile.max_daily_loss_percentage:
        if abs(daily_pnl_percentage) > account.risk_profile.max_daily_loss_percentage * 0.7:
            daily_loss_warning = True
    
    return {
        "account_id": account_id,
        "account_name": account.name,
        "total_equity": total_equity,
        "cash_balance": sum(float(b.available) for b in account.balances if b.asset_type == "currency"),
        "exposure_percentage": exposure_percentage,
        "leverage_used": exposure_percentage / 100,  # Simplified calculation
        "asset_class_exposure": asset_class_percentages,
        "daily_pnl": float(daily_pnl),
        "daily_pnl_percentage": daily_pnl_percentage,
        "risk_status": risk_status,
        "positions_count": len(account.positions),
        "largest_position_percentage": largest_position_percentage,
        "daily_loss_warning": daily_loss_warning,
        "margin_call_warning": account.is_margin_enabled and account.margin_level > 70.0,
        "updated_at": datetime.utcnow().isoformat()
    }