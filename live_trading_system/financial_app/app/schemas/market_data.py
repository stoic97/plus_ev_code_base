"""
Market Data Schema Module

This module defines Pydantic models for market data validation and serialization.
These schemas correspond to the SQLAlchemy ORM models in app/models/market_data.py
and are used for API request/response validation and documentation.

Schemas:
- InstrumentBase/Create/Update/Response: Financial instrument schemas
- OHLCVBase/Create/Update/Response: Open-High-Low-Close-Volume data schemas
- TickBase/Create/Update/Response: Individual trade data schemas
- OrderBookSnapshotBase/Create/Update/Response: Order book snapshot schemas
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Union, Literal, Annotated
from uuid import UUID, uuid4
from decimal import Decimal
from enum import Enum

from pydantic import (
    BaseModel, Field, ConfigDict, model_validator, 
    field_validator, AfterValidator
)


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


class TimeInterval(str, Enum):
    """Valid time intervals for OHLCV data"""
    SECOND_1 = "1s"
    SECOND_5 = "5s"
    SECOND_15 = "15s"
    SECOND_30 = "30s"
    MINUTE_1 = "1m"
    MINUTE_3 = "3m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_2 = "2h"
    HOUR_4 = "4h"
    HOUR_6 = "6h"
    HOUR_8 = "8h"
    HOUR_12 = "12h"
    DAY_1 = "1d"
    DAY_3 = "3d"
    WEEK_1 = "1w"
    MONTH_1 = "1mo"


class TradeSide(str, Enum):
    """Trade direction enumeration"""
    BUY = "buy"
    SELL = "sell"


# Base schema for common market data fields
class MarketDataBase(BaseModel):
    """Base schema for all market data types with common fields"""
    instrument_id: UUID
    timestamp: datetime
    source: DataSource
    source_timestamp: Optional[datetime] = None
    is_anomaly: bool = False
    anomaly_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(
        extra="forbid"  # Prevent extra fields
    )


# === Instrument Schemas ===

class InstrumentBase(BaseModel):
    """Base schema for instrument data"""
    symbol: str = Field(..., min_length=1, max_length=32)
    name: Optional[str] = Field(None, max_length=255)
    asset_class: AssetClass
    exchange: Optional[str] = Field(None, max_length=50)
    currency: str = Field(..., min_length=3, max_length=3)
    expiry_date: Optional[datetime] = None
    contract_size: Optional[Decimal] = None
    price_adj_factor: Decimal = Field(1.0, ge=0)
    specifications: Optional[Dict[str, Any]] = None
    active: bool = True
    
    @field_validator('currency')
    @classmethod
    def currency_must_be_uppercase(cls, v):
        """Ensure currency code is uppercase"""
        return v.upper()
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "asset_class": "equity",
                "exchange": "NASDAQ",
                "currency": "USD",
                "price_adj_factor": 1.0,
                "active": True
            }
        }
    )


class InstrumentCreate(InstrumentBase):
    """Schema for creating a new instrument"""
    isin: Optional[str] = Field(None, min_length=12, max_length=12)
    figi: Optional[str] = Field(None, min_length=12, max_length=12)


class InstrumentUpdate(BaseModel):
    """Schema for updating an instrument"""
    name: Optional[str] = Field(None, max_length=255)
    asset_class: Optional[AssetClass] = None
    exchange: Optional[str] = Field(None, max_length=50)
    currency: Optional[str] = Field(None, min_length=3, max_length=3)
    expiry_date: Optional[datetime] = None
    contract_size: Optional[Decimal] = None
    price_adj_factor: Optional[Decimal] = Field(None, ge=0)
    specifications: Optional[Dict[str, Any]] = None
    active: Optional[bool] = None
    isin: Optional[str] = Field(None, min_length=12, max_length=12)
    figi: Optional[str] = Field(None, min_length=12, max_length=12)
    
    @field_validator('currency')
    @classmethod
    def currency_must_be_uppercase(cls, v):
        """Ensure currency code is uppercase if provided"""
        if v is not None:
            return v.upper()
        return v
    
    @model_validator(mode='after')
    def check_at_least_one_field(self):
        """Ensure at least one field is being updated"""
        if all(v is None for v in self.model_dump().values()):
            raise ValueError("at least one field must be provided for update")
        return self


class InstrumentResponse(InstrumentBase):
    """Schema for instrument response"""
    id: UUID
    isin: Optional[str] = None
    figi: Optional[str] = None
    created_at: datetime
    modified_at: Optional[datetime] = None
    
    model_config = ConfigDict(
        from_attributes=True
    )


# === OHLCV Schemas ===

class OHLCVBase(MarketDataBase):
    """Base schema for OHLCV data"""
    open: Decimal = Field(..., ge=0)
    high: Decimal = Field(..., ge=0)
    low: Decimal = Field(..., ge=0)
    close: Decimal = Field(..., ge=0)
    volume: Decimal = Field(..., ge=0)
    interval: TimeInterval
    vwap: Optional[Decimal] = Field(None, ge=0)
    trades_count: Optional[int] = Field(None, ge=0)
    open_interest: Optional[Decimal] = Field(None, ge=0)
    adjusted_close: Optional[Decimal] = Field(None, ge=0)
    
    @model_validator(mode='after')
    def validate_price_relationships(self):
        """Validate that high >= open, close, low and low <= open, close, high"""
        values = self.model_dump()
        high = values.get('high')
        low = values.get('low')
        open_price = values.get('open')
        close = values.get('close')
        
        if all(x is not None for x in [high, low, open_price, close]):
            if high < low:
                raise ValueError("high cannot be less than low")
            if high < open_price:
                raise ValueError("high cannot be less than open")
            if high < close:
                raise ValueError("high cannot be less than close")
            # Change the order of these checks - check close first
            if low > close:
                raise ValueError("low cannot be greater than close")
            if low > open_price:
                raise ValueError("low cannot be greater than open")
        
        return self
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "instrument_id": "123e4567-e89b-12d3-a456-426614174000",
                "timestamp": "2023-01-01T12:00:00Z",
                "source": "exchange",
                "open": "150.25",
                "high": "152.50",
                "low": "149.75",
                "close": "151.00",
                "volume": "1000000.00",
                "interval": "1h",
                "vwap": "151.25",
                "trades_count": 2500
            }
        }
    )


class OHLCVCreate(OHLCVBase):
    """Schema for creating OHLCV data"""
    pass


class OHLCVUpdate(BaseModel):
    """Schema for updating OHLCV data"""
    open: Optional[Decimal] = Field(None, ge=0)
    high: Optional[Decimal] = Field(None, ge=0)
    low: Optional[Decimal] = Field(None, ge=0)
    close: Optional[Decimal] = Field(None, ge=0)
    volume: Optional[Decimal] = Field(None, ge=0)
    vwap: Optional[Decimal] = Field(None, ge=0)
    trades_count: Optional[int] = Field(None, ge=0)
    open_interest: Optional[Decimal] = Field(None, ge=0)
    adjusted_close: Optional[Decimal] = Field(None, ge=0)
    is_anomaly: Optional[bool] = None
    anomaly_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @model_validator(mode='after')
    def validate_price_relationships(self):
        """Validate that high >= open, close, low and low <= open, close, high if all provided"""
        values = self.model_dump()
        high = values.get('high')
        low = values.get('low') 
        open_price = values.get('open')
        close = values.get('close')
        
        # Only validate if all relevant fields are provided
        if all(x is not None for x in [high, low, open_price, close]):
            if high < low:
                raise ValueError("high cannot be less than low")
            if high < open_price:
                raise ValueError("high cannot be less than open")
            if high < close:
                raise ValueError("high cannot be less than close")
            if low > close:
                raise ValueError("low cannot be greater than close")
            if low > open_price:
                raise ValueError("low cannot be greater than open")
            
        
        return self
    
    @model_validator(mode='after')
    def check_at_least_one_field(self):
        """Ensure at least one field is being updated"""
        if all(v is None for v in self.model_dump().values()):
            raise ValueError("At least one field must be provided for update")
        return self


class OHLCVResponse(OHLCVBase):
    """Schema for OHLCV data response"""
    id: UUID
    created_at: datetime
    modified_at: Optional[datetime] = None
    
    model_config = ConfigDict(
        from_attributes=True
    )


# === Tick Schemas ===

class TickBase(MarketDataBase):
    """Base schema for tick data"""
    price: Decimal = Field(..., ge=0)
    volume: Decimal = Field(..., ge=0)
    trade_id: Optional[str] = Field(None, max_length=64)
    side: Optional[TradeSide] = None
    trade_data: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "instrument_id": "123e4567-e89b-12d3-a456-426614174000",
                "timestamp": "2023-01-01T12:00:00.123456Z",
                "source": "exchange",
                "price": "150.25",
                "volume": "100.00",
                "trade_id": "T12345678",
                "side": "buy"
            }
        }
    )


class TickCreate(TickBase):
    """Schema for creating tick data"""
    pass


class TickUpdate(BaseModel):
    """Schema for updating tick data"""
    price: Optional[Decimal] = Field(None, ge=0)
    volume: Optional[Decimal] = Field(None, ge=0)
    trade_id: Optional[str] = Field(None, max_length=64)
    side: Optional[TradeSide] = None
    trade_data: Optional[Dict[str, Any]] = None
    is_anomaly: Optional[bool] = None
    anomaly_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @model_validator(mode='after')
    def check_at_least_one_field(self):
        """Ensure at least one field is being updated"""
        if all(v is None for v in self.model_dump().values()):
            raise ValueError("at least one field must be provided for update")
        return self


class TickResponse(TickBase):
    """Schema for tick data response"""
    id: UUID
    created_at: datetime
    modified_at: Optional[datetime] = None
    
    model_config = ConfigDict(
        from_attributes=True
    )


# === OrderBookSnapshot Schemas ===

class PriceLevel(BaseModel):
    """Price level in order book"""
    price: Decimal
    volume: Decimal
    
    @field_validator('price', 'volume')
    @classmethod
    def validate_positive(cls, v):
        """Ensure price and volume are positive"""
        if v < 0:
            raise ValueError("Value must be positive")
        return v


class OrderBookSnapshotBase(MarketDataBase):
    """Base schema for order book snapshot data"""
    depth: int = Field(..., ge=1, le=100)
    bids: List[List[Decimal]] = Field(..., min_items=1)
    asks: List[List[Decimal]] = Field(..., min_items=1)
    spread: Optional[Decimal] = Field(None, ge=0)
    weighted_mid_price: Optional[Decimal] = Field(None, ge=0)
    imbalance: Optional[float] = Field(None, ge=-1, le=1)
    
    @field_validator('bids', 'asks')
    @classmethod
    def validate_price_levels(cls, v):
        """Validate price level structure"""
        for level in v:
            if not isinstance(level, list) or len(level) != 2:
                raise ValueError("Each price level must be a list of [price, volume]")
            if level[0] < 0 or level[1] < 0:
                raise ValueError("Price and volume must be positive")
        return v
    
    @model_validator(mode='after')
    def validate_books_and_metrics(self):
        """Calculate and validate spread and imbalance if not provided"""
        values = self.model_dump()
        bids = values.get('bids')
        asks = values.get('asks')
        
        if bids and asks:
            # Bids are sorted highest to lowest
            best_bid = max(bids, key=lambda x: x[0])[0] if bids else Decimal(0)
            # Asks are sorted lowest to highest
            best_ask = min(asks, key=lambda x: x[0])[0] if asks else Decimal('inf')
            
            # Calculate spread if not provided
            if self.spread is None and best_ask != Decimal('inf') and best_bid > 0:
                self.spread = best_ask - best_bid
            
            # Calculate weighted mid price if not provided
            if self.weighted_mid_price is None and best_ask != Decimal('inf') and best_bid > 0:
                # Simple midpoint - could be enhanced with volume weighting
                self.weighted_mid_price = (best_bid + best_ask) / 2
            
            # Calculate imbalance if not provided
            if self.imbalance is None:
                total_bid_volume = sum(level[1] for level in bids)
                total_ask_volume = sum(level[1] for level in asks)
                
                if total_bid_volume + total_ask_volume > 0:
                    # Imbalance from -1 (all asks) to 1 (all bids)
                    imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
                    self.imbalance = float(imbalance)
        
        return self
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "instrument_id": "123e4567-e89b-12d3-a456-426614174000",
                "timestamp": "2023-01-01T12:00:00.123456Z",
                "source": "exchange",
                "depth": 5,
                "bids": [
                    [150.00, 100.00],
                    [149.95, 200.00],
                    [149.90, 300.00],
                    [149.85, 150.00],
                    [149.80, 250.00]
                ],
                "asks": [
                    [150.05, 150.00],
                    [150.10, 200.00],
                    [150.15, 100.00],
                    [150.20, 300.00],
                    [150.25, 200.00]
                ],
                "spread": 0.05,
                "weighted_mid_price": 150.025,
                "imbalance": -0.05
            }
        }
    )


class OrderBookSnapshotCreate(OrderBookSnapshotBase):
    """Schema for creating order book snapshot data"""
    pass


class OrderBookSnapshotUpdate(BaseModel):
    """Schema for updating order book snapshot data"""
    depth: Optional[int] = Field(None, ge=1, le=100)
    bids: Optional[List[List[Decimal]]] = None
    asks: Optional[List[List[Decimal]]] = None
    spread: Optional[Decimal] = Field(None, ge=0)
    weighted_mid_price: Optional[Decimal] = Field(None, ge=0)
    imbalance: Optional[float] = Field(None, ge=-1, le=1)
    is_anomaly: Optional[bool] = None
    anomaly_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @field_validator('bids', 'asks')
    @classmethod
    def validate_price_levels(cls, v):
        """Validate price level structure if provided"""
        if v is not None:
            for level in v:
                if not isinstance(level, list) or len(level) != 2:
                    raise ValueError("Each price level must be a list of [price, volume]")
                if level[0] < 0 or level[1] < 0:
                    raise ValueError("Price and volume must be positive")
        return v
    
    @model_validator(mode='after')
    def check_at_least_one_field(self):
        """Ensure at least one field is being updated"""
        if all(v is None for v in self.model_dump().values()):
            raise ValueError("at least one field must be provided for update")
        return self


class OrderBookSnapshotResponse(OrderBookSnapshotBase):
    """Schema for order book snapshot data response"""
    id: UUID
    created_at: datetime
    modified_at: Optional[datetime] = None
    
    model_config = ConfigDict(
        from_attributes=True
    )


# === Query Models for Filtering ===

class MarketDataFilter(BaseModel):
    """Base filter model for market data queries"""
    instrument_id: Optional[UUID] = None
    symbol: Optional[str] = None
    start_timestamp: Optional[datetime] = None
    end_timestamp: Optional[datetime] = None
    source: Optional[DataSource] = None
    limit: Optional[int] = Field(100, ge=1, le=1000)
    offset: Optional[int] = Field(0, ge=0)


class OHLCVFilter(MarketDataFilter):
    """Filter model for OHLCV queries"""
    interval: Optional[TimeInterval] = None
    min_volume: Optional[Decimal] = Field(None, ge=0)
    exclude_anomalies: bool = True


class TickFilter(MarketDataFilter):
    """Filter model for Tick queries"""
    trade_id: Optional[str] = None
    side: Optional[TradeSide] = None
    min_price: Optional[Decimal] = Field(None, ge=0)
    max_price: Optional[Decimal] = Field(None, ge=0)
    min_volume: Optional[Decimal] = Field(None, ge=0)


class OrderBookFilter(MarketDataFilter):
    """Filter model for OrderBook queries"""
    min_depth: Optional[int] = Field(None, ge=1)
    max_spread: Optional[Decimal] = Field(None, ge=0)


# === Specialized Response Models ===

class MarketDataStats(BaseModel):
    """Statistics about market data"""
    instrument_id: UUID
    symbol: str
    first_timestamp: datetime
    last_timestamp: datetime
    record_count: int
    data_sources: List[DataSource]
    anomaly_percentage: float
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "instrument_id": "123e4567-e89b-12d3-a456-426614174000",
                "symbol": "AAPL",
                "first_timestamp": "2023-01-01T00:00:00Z",
                "last_timestamp": "2023-01-31T23:59:59Z",
                "record_count": 44640,
                "data_sources": ["exchange", "vendor"],
                "anomaly_percentage": 0.02
            }
        }
    )


class PaginatedResponse(BaseModel):
    """Generic paginated response model"""
    total: int
    offset: int
    limit: int
    items: List[Any]

# Add alias for backward compatibility with tests
OrderBookUpdate = OrderBookSnapshotUpdate