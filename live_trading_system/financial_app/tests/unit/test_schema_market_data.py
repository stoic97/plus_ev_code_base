"""
Tests for market_data schema validation models.

Tests validate that:
1. Valid data passes schema validation
2. Invalid data raises appropriate validation errors
3. Field transformations work correctly
4. Complex validators for relationships between fields work as expected
5. Default values are correctly applied
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from uuid import UUID, uuid4
import json
from pydantic import ValidationError

from app.schemas.market_data import (
    AssetClass, DataSource, TimeInterval, TradeSide,
    InstrumentBase, InstrumentCreate, InstrumentUpdate, InstrumentResponse,
    OHLCVBase, OHLCVCreate, OHLCVUpdate, OHLCVResponse,
    TickBase, TickCreate, TickUpdate, TickResponse,
    OrderBookSnapshotBase, OrderBookSnapshotCreate, OrderBookSnapshotUpdate, OrderBookSnapshotResponse,
    MarketDataFilter, OHLCVFilter, TickFilter, OrderBookFilter,
    MarketDataStats, PaginatedResponse
)


# Fixture for test UUID
@pytest.fixture
def test_uuid():
    """Return a fixed UUID for testing"""
    return UUID('123e4567-e89b-12d3-a456-426614174000')


# Fixture for timestamp
@pytest.fixture
def test_timestamp():
    """Return a fixed timestamp for testing"""
    return datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


# Fixture for valid instrument data
@pytest.fixture
def valid_instrument_data():
    """Return valid instrument data"""
    return {
        "symbol": "AAPL",
        "name": "Apple Inc.",
        "asset_class": "equity",
        "exchange": "NASDAQ",
        "currency": "usd",  # lowercase to test normalization
        "price_adj_factor": Decimal("1.0"),
        "active": True
    }


# Fixture for valid OHLCV data
@pytest.fixture
def valid_ohlcv_data(test_uuid, test_timestamp):
    """Return valid OHLCV data"""
    return {
        "instrument_id": test_uuid,
        "timestamp": test_timestamp,
        "source": "exchange",
        "open": Decimal("150.25"),
        "high": Decimal("152.50"),
        "low": Decimal("149.75"),
        "close": Decimal("151.00"),
        "volume": Decimal("1000000.00"),
        "interval": "1h",
        "vwap": Decimal("151.25"),
        "trades_count": 2500
    }


# Fixture for valid Tick data
@pytest.fixture
def valid_tick_data(test_uuid, test_timestamp):
    """Return valid Tick data"""
    return {
        "instrument_id": test_uuid,
        "timestamp": test_timestamp,
        "source": "exchange",
        "price": Decimal("150.25"),
        "volume": Decimal("100.00"),
        "trade_id": "T12345678",
        "side": "buy"
    }


# Fixture for valid OrderBookSnapshot data
@pytest.fixture
def valid_orderbook_data(test_uuid, test_timestamp):
    """Return valid OrderBookSnapshot data"""
    return {
        "instrument_id": test_uuid,
        "timestamp": test_timestamp,
        "source": "exchange",
        "depth": 5,
        "bids": [
            [Decimal("150.00"), Decimal("100.00")],
            [Decimal("149.95"), Decimal("200.00")],
            [Decimal("149.90"), Decimal("300.00")],
            [Decimal("149.85"), Decimal("150.00")],
            [Decimal("149.80"), Decimal("250.00")]
        ],
        "asks": [
            [Decimal("150.05"), Decimal("150.00")],
            [Decimal("150.10"), Decimal("200.00")],
            [Decimal("150.15"), Decimal("100.00")],
            [Decimal("150.20"), Decimal("300.00")],
            [Decimal("150.25"), Decimal("200.00")]
        ]
    }


class TestInstrumentSchemas:
    """Tests for Instrument schemas"""

    def test_instrument_base_valid(self, valid_instrument_data):
        """Test that valid data passes validation"""
        model = InstrumentBase(**valid_instrument_data)
        assert model.symbol == "AAPL"
        assert model.currency == "USD"  # Should be normalized to uppercase
        assert model.asset_class == AssetClass.EQUITY

    def test_instrument_base_invalid(self):
        """Test that invalid data raises appropriate errors"""
        # Missing required fields
        with pytest.raises(ValidationError) as excinfo:
            InstrumentBase(symbol="AAPL")
        assert "asset_class" in str(excinfo.value)
        assert "currency" in str(excinfo.value)
        
        # Invalid asset class
        with pytest.raises(ValidationError) as excinfo:
            InstrumentBase(symbol="AAPL", asset_class="invalid", currency="USD")
        assert "asset_class" in str(excinfo.value)
        
        # Invalid currency length
        with pytest.raises(ValidationError) as excinfo:
            InstrumentBase(symbol="AAPL", asset_class="equity", currency="US")
        assert "currency" in str(excinfo.value)

    def test_instrument_create(self, valid_instrument_data):
        """Test instrument creation schema"""
        # With standard identifiers
        data = valid_instrument_data.copy()
        data["isin"] = "US0378331005"
        data["figi"] = "BBG000B9XRY4"
        
        model = InstrumentCreate(**data)
        assert model.isin == "US0378331005"
        
        # With invalid identifiers
        data["isin"] = "SHORT"
        with pytest.raises(ValidationError) as excinfo:
            InstrumentCreate(**data)
        assert "isin" in str(excinfo.value)

    def test_instrument_update(self, valid_instrument_data):
        """Test instrument update schema"""
        # Valid partial update
        model = InstrumentUpdate(name="Apple Inc Updated")
        assert model.name == "Apple Inc Updated"
        
        # Empty update should fail
        with pytest.raises(ValidationError) as excinfo:
            InstrumentUpdate()
        assert "at least one field" in str(excinfo.value)
        
        # Currency normalization in update
        model = InstrumentUpdate(currency="eur")
        assert model.currency == "EUR"

    def test_instrument_response(self, valid_instrument_data, test_uuid, test_timestamp):
        """Test instrument response schema"""
        data = valid_instrument_data.copy()
        data["id"] = test_uuid
        data["created_at"] = test_timestamp
        
        model = InstrumentResponse(**data)
        assert model.id == test_uuid
        assert model.created_at == test_timestamp
        assert model.currency == "USD"  # Should be normalized


class TestOHLCVSchemas:
    """Tests for OHLCV schemas"""

    def test_ohlcv_base_valid(self, valid_ohlcv_data):
        """Test that valid OHLCV data passes validation"""
        model = OHLCVBase(**valid_ohlcv_data)
        assert model.open == Decimal("150.25")
        assert model.high == Decimal("152.50")
        assert model.interval == TimeInterval.HOUR_1

    def test_ohlcv_base_invalid_interval(self, valid_ohlcv_data):
        """Test validation of interval field"""
        data = valid_ohlcv_data.copy()
        data["interval"] = "invalid"
        
        with pytest.raises(ValidationError) as excinfo:
            OHLCVBase(**data)
        assert "interval" in str(excinfo.value)

    def test_ohlcv_price_relationships(self, valid_ohlcv_data):
        """Test validation of price relationships (high, low, open, close)"""
        data = valid_ohlcv_data.copy()
        
        # Test high < low (invalid)
        data["high"] = Decimal("149.00")
        with pytest.raises(ValidationError) as excinfo:
            OHLCVBase(**data)
        assert "high cannot be less than low" in str(excinfo.value)
        
        # Reset and test high < open (invalid)
        data = valid_ohlcv_data.copy()
        data["high"] = Decimal("150.00")
        with pytest.raises(ValidationError) as excinfo:
            OHLCVBase(**data)
        assert "high cannot be less than open" in str(excinfo.value)
        
        # Reset and test low > close (invalid)
        data = valid_ohlcv_data.copy()
        data["low"] = Decimal("152.00")
        with pytest.raises(ValidationError) as excinfo:
            OHLCVBase(**data)
        assert "low cannot be greater than close" in str(excinfo.value)

    def test_ohlcv_negative_values(self, valid_ohlcv_data):
        """Test validation of negative values"""
        data = valid_ohlcv_data.copy()
        data["open"] = Decimal("-1.0")
        
        with pytest.raises(ValidationError) as excinfo:
            OHLCVBase(**data)
        assert "open" in str(excinfo.value)
        assert "greater than or equal to 0" in str(excinfo.value)

    def test_ohlcv_update_partial(self, valid_ohlcv_data):
        """Test partial updates with OHLCV update schema"""
        # Update just the close price
        model = OHLCVUpdate(close=Decimal("155.00"))
        assert model.close == Decimal("155.00")
        assert model.open is None
        
        # Update with invalid relationship
        with pytest.raises(ValidationError) as excinfo:
            OHLCVUpdate(high=Decimal("150.00"), low=Decimal("151.00"), 
                       open=Decimal("149.00"), close=Decimal("152.00"))
        assert "high cannot be less than low" in str(excinfo.value)

    def test_ohlcv_create(self, valid_ohlcv_data):
        """Test OHLCV create schema"""
        model = OHLCVCreate(**valid_ohlcv_data)
        assert model.open == Decimal("150.25")
        assert model.interval == TimeInterval.HOUR_1

    def test_ohlcv_response(self, valid_ohlcv_data, test_uuid, test_timestamp):
        """Test OHLCV response schema"""
        data = valid_ohlcv_data.copy()
        data["id"] = test_uuid
        data["created_at"] = test_timestamp
        
        model = OHLCVResponse(**data)
        assert model.id == test_uuid
        assert model.created_at == test_timestamp


class TestTickSchemas:
    """Tests for Tick schemas"""

    def test_tick_base_valid(self, valid_tick_data):
        """Test that valid Tick data passes validation"""
        model = TickBase(**valid_tick_data)
        assert model.price == Decimal("150.25")
        assert model.volume == Decimal("100.00")
        assert model.side == TradeSide.BUY

    def test_tick_base_invalid_side(self, valid_tick_data):
        """Test validation of side field"""
        data = valid_tick_data.copy()
        data["side"] = "invalid"
        
        with pytest.raises(ValidationError) as excinfo:
            TickBase(**data)
        assert "side" in str(excinfo.value)

    def test_tick_negative_values(self, valid_tick_data):
        """Test validation of negative values"""
        data = valid_tick_data.copy()
        data["price"] = Decimal("-1.0")
        
        with pytest.raises(ValidationError) as excinfo:
            TickBase(**data)
        assert "price" in str(excinfo.value)
        assert "greater than or equal to 0" in str(excinfo.value)

    def test_tick_update(self, valid_tick_data):
        """Test Tick update schema"""
        # Valid partial update
        model = TickUpdate(price=Decimal("151.00"))
        assert model.price == Decimal("151.00")
        assert model.volume is None
        
        # Empty update should fail
        with pytest.raises(ValidationError) as excinfo:
            TickUpdate()
        assert "at least one field" in str(excinfo.value)

    def test_tick_create(self, valid_tick_data):
        """Test Tick create schema"""
        model = TickCreate(**valid_tick_data)
        assert model.price == Decimal("150.25")
        assert model.side == TradeSide.BUY

    def test_tick_response(self, valid_tick_data, test_uuid, test_timestamp):
        """Test Tick response schema"""
        data = valid_tick_data.copy()
        data["id"] = test_uuid
        data["created_at"] = test_timestamp
        
        model = TickResponse(**data)
        assert model.id == test_uuid
        assert model.created_at == test_timestamp


class TestOrderBookSchemas:
    """Tests for OrderBook schemas"""

    def test_orderbook_base_valid(self, valid_orderbook_data):
        """Test that valid OrderBookSnapshot data passes validation"""
        model = OrderBookSnapshotBase(**valid_orderbook_data)
        assert model.depth == 5
        assert len(model.bids) == 5
        assert len(model.asks) == 5
        
        # Check that spread was calculated
        assert model.spread is not None
        assert model.spread > 0

    def test_orderbook_invalid_structure(self, valid_orderbook_data):
        """Test validation of order book structure"""
        data = valid_orderbook_data.copy()
        
        # Invalid bids structure
        data["bids"] = [[Decimal("150.00")]]  # Missing volume
        with pytest.raises(ValidationError) as excinfo:
            OrderBookSnapshotBase(**data)
        assert "bids" in str(excinfo.value)
        
        # Invalid asks with negative value
        data = valid_orderbook_data.copy()
        data["asks"] = [[Decimal("150.05"), Decimal("-10.0")]]  # Negative volume
        with pytest.raises(ValidationError) as excinfo:
            OrderBookSnapshotBase(**data)
        assert "asks" in str(excinfo.value)

    def test_orderbook_imbalance_validation(self, valid_orderbook_data):
        """Test validation of imbalance field"""
        data = valid_orderbook_data.copy()
        data["imbalance"] = 1.5  # Invalid: > 1
        
        with pytest.raises(ValidationError) as excinfo:
            OrderBookSnapshotBase(**data)
        assert "imbalance" in str(excinfo.value)
        
        # Valid edge cases
        data["imbalance"] = 1.0
        model = OrderBookSnapshotBase(**data)
        assert model.imbalance == 1.0
        
        data["imbalance"] = -1.0
        model = OrderBookSnapshotBase(**data)
        assert model.imbalance == -1.0

    def test_orderbook_auto_calculations(self, valid_orderbook_data):
        """Test automatic calculation of metrics when not provided"""
        # Remove calculated fields
        data = valid_orderbook_data.copy()
        data.pop("spread", None)
        data.pop("weighted_mid_price", None)
        data.pop("imbalance", None)
        
        model = OrderBookSnapshotBase(**data)
        
        # Check that metrics were calculated
        assert model.spread is not None
        assert model.weighted_mid_price is not None
        assert model.imbalance is not None
        
        # Verify spread calculation (best_ask - best_bid)
        best_bid = max([bid[0] for bid in data["bids"]])
        best_ask = min([ask[0] for ask in data["asks"]])
        expected_spread = best_ask - best_bid
        assert model.spread == expected_spread

    def test_orderbook_update(self, valid_orderbook_data):
        """Test OrderBookSnapshot update schema"""
        # Valid partial update
        model = OrderBookSnapshotUpdate(depth=10)
        assert model.depth == 10
        assert model.bids is None
        
        # Update bids only
        new_bids = [[Decimal("151.00"), Decimal("200.00")]]
        model = OrderBookSnapshotUpdate(bids=new_bids)
        assert model.bids == new_bids
        
        # Empty update should fail
        with pytest.raises(ValidationError) as excinfo:
            OrderBookSnapshotUpdate()
        assert "at least one field" in str(excinfo.value)

    def test_orderbook_create(self, valid_orderbook_data):
        """Test OrderBookSnapshot create schema"""
        model = OrderBookSnapshotCreate(**valid_orderbook_data)
        assert model.depth == 5
        assert len(model.bids) == 5
        assert len(model.asks) == 5

    def test_orderbook_response(self, valid_orderbook_data, test_uuid, test_timestamp):
        """Test OrderBookSnapshot response schema"""
        data = valid_orderbook_data.copy()
        data["id"] = test_uuid
        data["created_at"] = test_timestamp
        data["spread"] = Decimal("0.05")  # Explicitly provide calculated values
        data["weighted_mid_price"] = Decimal("150.025")
        data["imbalance"] = -0.05
        
        model = OrderBookSnapshotResponse(**data)
        assert model.id == test_uuid
        assert model.created_at == test_timestamp
        assert model.spread == Decimal("0.05")


class TestFilterSchemas:
    """Tests for filter schemas"""

    def test_market_data_filter(self, test_uuid):
        """Test base market data filter"""
        model = MarketDataFilter(
            instrument_id=test_uuid,
            symbol="AAPL",
            start_timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
            end_timestamp=datetime(2023, 1, 2, tzinfo=timezone.utc),
            limit=50
        )
        assert model.instrument_id == test_uuid
        assert model.symbol == "AAPL"
        assert model.limit == 50
        
        # Test default values
        model = MarketDataFilter()
        assert model.limit == 100
        assert model.offset == 0

    def test_ohlcv_filter(self, test_uuid):
        """Test OHLCV filter"""
        model = OHLCVFilter(
            instrument_id=test_uuid,
            interval="1h",
            min_volume=Decimal("1000.0"),
            exclude_anomalies=True
        )
        assert model.instrument_id == test_uuid
        assert model.interval == TimeInterval.HOUR_1
        assert model.min_volume == Decimal("1000.0")
        assert model.exclude_anomalies is True
        
        # Test default values
        model = OHLCVFilter()
        assert model.exclude_anomalies is True

    def test_tick_filter(self, test_uuid):
        """Test Tick filter"""
        model = TickFilter(
            instrument_id=test_uuid,
            side="buy",
            min_price=Decimal("100.0"),
            max_price=Decimal("200.0")
        )
        assert model.instrument_id == test_uuid
        assert model.side == TradeSide.BUY
        assert model.min_price == Decimal("100.0")
        assert model.max_price == Decimal("200.0")
        
        # Invalid side value
        with pytest.raises(ValidationError) as excinfo:
            TickFilter(side="invalid")
        assert "side" in str(excinfo.value)

    def test_orderbook_filter(self, test_uuid):
        """Test OrderBook filter"""
        model = OrderBookFilter(
            instrument_id=test_uuid,
            min_depth=5,
            max_spread=Decimal("0.1")
        )
        assert model.instrument_id == test_uuid
        assert model.min_depth == 5
        assert model.max_spread == Decimal("0.1")


class TestResponseModels:
    """Tests for additional response models"""

    def test_market_data_stats(self, test_uuid):
        """Test MarketDataStats model"""
        model = MarketDataStats(
            instrument_id=test_uuid,
            symbol="AAPL",
            first_timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
            last_timestamp=datetime(2023, 1, 31, tzinfo=timezone.utc),
            record_count=44640,
            data_sources=["exchange", "vendor"],
            anomaly_percentage=0.02
        )
        assert model.instrument_id == test_uuid
        assert model.symbol == "AAPL"
        assert model.record_count == 44640
        assert "exchange" in model.data_sources
        assert model.anomaly_percentage == 0.02

    def test_paginated_response(self, valid_ohlcv_data, test_uuid, test_timestamp):
        """Test PaginatedResponse model with different item types"""
        # Create a few sample items
        data = valid_ohlcv_data.copy()
        data["id"] = test_uuid
        data["created_at"] = test_timestamp
        
        ohlcv_item = OHLCVResponse(**data)
        
        # Create paginated response with OHLCV items
        model = PaginatedResponse(
            total=100,
            offset=0,
            limit=10,
            items=[ohlcv_item]
        )
        assert model.total == 100
        assert model.limit == 10
        assert len(model.items) == 1
        assert isinstance(model.items[0], OHLCVResponse)