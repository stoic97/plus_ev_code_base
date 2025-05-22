"""
Market Data ORM Module

This module defines SQLAlchemy ORM models for market data storage.
It includes models for different types of market data with appropriate
indexes and validations.

Models:
- MarketDataBase: Abstract base class for market data models
- OHLCV: Open-High-Low-Close-Volume data for interval-based pricing
- Tick: Individual trade data with price and volume
- OrderBookSnapshot: Order book state at a point in time
- Instrument: Financial instrument metadata
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
import uuid
import json

from sqlalchemy import (
    Column, Integer, Numeric, String, DateTime, 
    Boolean, ForeignKey, Index, Text, func,
    UniqueConstraint, Float, JSON
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import relationship, validates

from app.core.database import Base


class AssetClass(str, Enum):
    """Enum for asset classes"""
    EQUITY = "equity"
    FOREX = "forex"
    CRYPTO = "crypto"
    COMMODITY = "commodity"
    FIXED_INCOME = "fixed_income"
    OPTION = "option"
    FUTURE = "future"
    ETF = "etf"
    INDEX = "index"
    OTHER = "other"


class DataSource(str, Enum):
    """Enum for data sources"""
    EXCHANGE = "exchange"
    BROKER = "broker"
    VENDOR = "vendor"
    CALCULATED = "calculated"
    SIMULATED = "simulated"
    OTHER = "other"


class MarketDataBase:
    """Base class for all market data models with common fields and behaviors"""
    
    @declared_attr
    def __tablename__(cls):
        """Generate tablename from class name"""
        return cls.__name__.lower()
    
    # All market data has a related instrument
    @declared_attr
    def instrument_id(cls):
        """Reference to the instrument"""
        return Column(UUID(as_uuid=True), ForeignKey('instrument.id'), nullable=False)
    
    @declared_attr
    def instrument(cls):
        """Relationship to the instrument"""
        return relationship("Instrument")
    
    # Primary timestamp field for all market data
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    
    # Data provenance information
    source = Column(String(50), nullable=False, 
                   info={"description": "Source of the data"})
    
    source_timestamp = Column(DateTime(timezone=True), nullable=True, 
                             info={"description": "Timestamp from data source"})
    
    # Data quality fields
    is_anomaly = Column(Boolean, default=False, 
                        info={"description": "Flag for anomalous data points"})
    
    anomaly_reason = Column(String(255), nullable=True, 
                           info={"description": "Reason if data point is flagged as anomaly"})
    
    # System metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    modified_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Additional metadata as JSON
    metadata = Column(JSONB, nullable=True, 
                     info={"description": "Additional metadata for the data point"})

    @validates('source')
    def validate_source(self, key, value):
        """Validate that source is a valid enum value"""
        try:
            return DataSource(value).value
        except ValueError:
            raise ValueError(f"Invalid data source: {value}")


class Instrument(Base):
    """Financial instrument metadata"""
    __tablename__ = 'instrument'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Basic instrument identifiers
    symbol = Column(String(32), nullable=False, index=True)
    name = Column(String(255), nullable=True)
    
    # Standard identifiers
    isin = Column(String(12), nullable=True, unique=True, 
                 info={"description": "International Securities Identification Number"})
    
    figi = Column(String(12), nullable=True, unique=True, 
                 info={"description": "Financial Instrument Global Identifier"})
    
    # Classification
    asset_class = Column(String(20), nullable=False)
    
    # Exchange information
    exchange = Column(String(50), nullable=True, 
                     info={"description": "Primary exchange where instrument is traded"})
    
    currency = Column(String(3), nullable=False, 
                     info={"description": "Currency of the instrument price"})
    
    # Contract specifications for derivatives
    expiry_date = Column(DateTime, nullable=True, 
                        info={"description": "Expiration date for derivatives"})
    
    contract_size = Column(Numeric(18, 8), nullable=True, 
                          info={"description": "Contract size for derivatives"})
    
    # Corporate action adjustment factors
    price_adj_factor = Column(Numeric(18, 8), default=1.0, nullable=False, 
                             info={"description": "Price adjustment factor for corporate actions"})
    
    # Additional specifications as JSON
    specifications = Column(JSONB, nullable=True, 
                          info={"description": "Additional instrument specifications"})
    
    # Metadata
    active = Column(Boolean, default=True, nullable=False, 
                   info={"description": "Whether the instrument is actively traded"})
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    modified_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Unique constraint on symbol and exchange
    __table_args__ = (
        UniqueConstraint('symbol', 'exchange', name='uix_symbol_exchange'),
    )
    
    @validates('asset_class')
    def validate_asset_class(self, key, value):
        """Validate that asset_class is a valid enum value"""
        try:
            return AssetClass(value).value
        except ValueError:
            raise ValueError(f"Invalid asset class: {value}")
    
    @validates('currency')
    def validate_currency(self, key, value):
        """Validate currency is a 3-letter ISO code"""
        if not value or len(value) != 3:
            raise ValueError("Currency must be a 3-letter ISO code")
        return value.upper()
    
    def __repr__(self):
        """String representation"""
        return f"<Instrument(symbol='{self.symbol}', exchange='{self.exchange}')>"


class OHLCV(Base, MarketDataBase):
    """
    Open-High-Low-Close-Volume data for interval-based pricing.
    Used for candlestick charts and most trading strategies.
    """
    __tablename__ = 'ohlcv'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # OHLCV price data - using Numeric for precise decimal representation
    open = Column(Numeric(18, 8), nullable=False, 
                 info={"description": "Opening price for the interval"})
    
    high = Column(Numeric(18, 8), nullable=False, 
                 info={"description": "Highest price for the interval"})
    
    low = Column(Numeric(18, 8), nullable=False, 
                info={"description": "Lowest price for the interval"})
    
    close = Column(Numeric(18, 8), nullable=False, 
                  info={"description": "Closing price for the interval"})
    
    volume = Column(Numeric(18, 8), nullable=False, 
                   info={"description": "Trading volume for the interval"})
    
    # Interval information
    interval = Column(String(20), nullable=False, 
                     info={"description": "Time interval (e.g., '1m', '1h', '1d')"})
    
    # Additional derived metrics
    vwap = Column(Numeric(18, 8), nullable=True, 
                 info={"description": "Volume-weighted average price"})
    
    trades_count = Column(Integer, nullable=True, 
                         info={"description": "Number of trades in the interval"})
    
    # For some assets: additional metrics
    open_interest = Column(Numeric(18, 8), nullable=True, 
                          info={"description": "Open interest for derivatives"})
    
    adjusted_close = Column(Numeric(18, 8), nullable=True, 
                           info={"description": "Close price adjusted for corporate actions"})

    # Indexes for efficient querying
    __table_args__ = (
        # Composite index for symbol+interval queries with time range
        Index('ix_ohlcv_instrument_interval_timestamp', 
              'instrument_id', 'interval', 'timestamp'),
        
        # Index for latest data retrieval
        Index('ix_ohlcv_timestamp_desc', 'timestamp', postgresql_using='brin'),
    )

    @validates('interval')
    def validate_interval(self, key, value):
        """Validate interval format"""
        valid_intervals = {'1s', '5s', '15s', '30s', '1m', '3m', '5m', '15m', '30m', 
                          '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1mo'}
        if value not in valid_intervals:
            raise ValueError(f"Invalid interval: {value}. Must be one of {valid_intervals}")
        return value

    @validates('open', 'high', 'low', 'close')
    def validate_prices(self, key, value):
        """Validate price values are positive"""
        if value < 0:
            raise ValueError(f"{key} price cannot be negative")
        return value

    def __repr__(self):
        """String representation"""
        return (f"<OHLCV(symbol='{self.instrument.symbol if self.instrument else None}', "
                f"interval='{self.interval}', timestamp='{self.timestamp}')>")


class Tick(Base, MarketDataBase):
    """
    Individual trade data with price and volume.
    Represents the most granular level of market data.
    """
    __tablename__ = 'tick'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Trade information
    price = Column(Numeric(18, 8), nullable=False, 
                   info={"description": "Trade price"})
    
    volume = Column(Numeric(18, 8), nullable=False, 
                    info={"description": "Trade volume"})
    
    # Trade identifiers
    trade_id = Column(String(64), nullable=True, 
                      info={"description": "Exchange-specific trade ID"})
    
    # Trade direction
    side = Column(String(4), nullable=True, 
                  info={"description": "Trade side (buy/sell) if available"})
    
    # Additional trade data as JSON
    trade_data = Column(JSONB, nullable=True, 
                        info={"description": "Additional trade-specific data"})
    
    # Indexes for efficient querying
    __table_args__ = (
        # Composite index for symbol with time range
        Index('ix_tick_instrument_timestamp', 'instrument_id', 'timestamp'),
        
        # Time-based index
        Index('ix_tick_timestamp', 'timestamp', postgresql_using='brin'),
        
        # Trade ID index
        Index('ix_tick_trade_id', 'trade_id'),
    )

    @validates('side')
    def validate_side(self, key, value):
        """Validate trade side"""
        if value and value not in ('buy', 'sell'):
            raise ValueError("Trade side must be 'buy' or 'sell'")
        return value

    def __repr__(self):
        """String representation"""
        return (f"<Tick(symbol='{self.instrument.symbol if self.instrument else None}', "
                f"price={self.price}, volume={self.volume}, timestamp='{self.timestamp}')>")


class OrderBookSnapshot(Base, MarketDataBase):
    """
    Order book state at a point in time.
    Represents the market depth for an instrument.
    """
    __tablename__ = 'order_book_snapshot'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Order book depth level
    depth = Column(Integer, nullable=False, default=10, 
                  info={"description": "Number of price levels in the order book"})
    
    # Order book data
    bids = Column(JSONB, nullable=False, 
                 info={"description": "Bid side of the order book as [[price, volume], ...]"})
    
    asks = Column(JSONB, nullable=False, 
                 info={"description": "Ask side of the order book as [[price, volume], ...]"})
    
    # Order book metrics
    spread = Column(Numeric(18, 8), nullable=True, 
                   info={"description": "Bid-ask spread"})
    
    weighted_mid_price = Column(Numeric(18, 8), nullable=True, 
                               info={"description": "Volume-weighted mid price"})
    
    imbalance = Column(Float, nullable=True, 
                      info={"description": "Order book imbalance from -1 (sell pressure) to 1 (buy pressure)"})
    
    # Indexes for efficient querying
    __table_args__ = (
        # Composite index for symbol with time range
        Index('ix_orderbook_instrument_timestamp', 'instrument_id', 'timestamp'),
        
        # Time-based index
        Index('ix_orderbook_timestamp', 'timestamp', postgresql_using='brin'),
    )

    @validates('bids', 'asks')
    def validate_book_data(self, key, value):
        """Validate order book data structure"""
        if isinstance(value, str):
            # If passed as string, parse JSON
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON for {key}")
        
        # Check structure - should be list of [price, volume] pairs
        if not isinstance(value, list):
            raise ValueError(f"{key} must be a list of [price, volume] pairs")
        
        for level in value:
            if not isinstance(level, list) or len(level) != 2:
                raise ValueError(f"Each {key} level must be a [price, volume] pair")
        
        return value

    @validates('imbalance')
    def validate_imbalance(self, key, value):
        """Validate imbalance is between -1 and 1"""
        if value is not None and (value < -1 or value > 1):
            raise ValueError("Imbalance must be between -1 and 1")
        return value

    def __repr__(self):
        """String representation"""
        bid_count = len(self.bids) if isinstance(self.bids, list) else '?'
        ask_count = len(self.asks) if isinstance(self.asks, list) else '?'
        return (f"<OrderBookSnapshot(symbol='{self.instrument.symbol if self.instrument else None}', "
                f"bids={bid_count}, asks={ask_count}, timestamp='{self.timestamp}')>")