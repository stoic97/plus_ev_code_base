"""
Unit tests for market_data.py models module.

This test suite validates the ORM models and validations for market data 
storage in the trading system.
"""

import pytest
import uuid
import json
import os
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import patch, MagicMock

from sqlalchemy import create_engine, String, Text, event, MetaData, inspect, Index
from sqlalchemy.orm import sessionmaker, clear_mappers
from sqlalchemy.exc import IntegrityError
from sqlalchemy.types import TypeDecorator, CHAR, JSON
from sqlalchemy.dialects.postgresql import JSONB

# Before importing models, register the type adapters
# SQLite-compatible UUID type
class GUID(TypeDecorator):
    """Platform-independent GUID type.
    Uses CHAR(36), storing as stringified hex values.
    """
    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        if not isinstance(value, uuid.UUID):
            try:
                return str(uuid.UUID(value))
            except (TypeError, ValueError):
                return str(value)
        else:
            return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        try:
            return uuid.UUID(value)
        except (TypeError, ValueError):
            return value


# SQLite-compatible JSON type (for JSONB columns)
class JsonType(TypeDecorator):
    """Platform-independent JSON type.
    Uses SQLite's JSON type, storing as stringified JSON.
    """
    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return json.dumps(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if isinstance(value, str):
            return json.loads(value)
        return value


# Now we can safely import models
from app.core.database import Base
from app.models.market_data import (
    Instrument, OHLCV, Tick, OrderBookSnapshot,
    AssetClass, DataSource
)


@pytest.fixture(scope="session", autouse=True)
def setup_models():
    """Set up model configuration before any tests run"""
    # Replace PostgreSQL-specific types with SQLite-compatible ones
    for model in [Instrument, OHLCV, Tick, OrderBookSnapshot]:
        # Handle type conversion first
        for column in model.__table__.columns:
            # Check if column type is UUID
            if hasattr(column.type, "as_uuid") or str(column.type).startswith('UUID'):
                column.type = GUID()
            
            # Check if column type is JSONB
            elif isinstance(column.type, JSONB) or str(column.type).startswith('JSONB'):
                column.type = JsonType()
        
        # Handle Tick table's PostgreSQL-specific index
        if model.__tablename__ == 'tick':
            table_args = list(model.__table_args__)
            # Find and modify the timestamp index to remove PostgreSQL-specific options
            for i, index in enumerate(table_args):
                if isinstance(index, Index) and index.name == 'ix_tick_timestamp':
                    # Create a new index without postgresql_using
                    new_index = Index('ix_tick_timestamp', 'timestamp')
                    table_args[i] = new_index
            
            # Apply modified table args
            model.__table_args__ = tuple(table_args)
    
    yield


@pytest.fixture(scope="session")
def db_engine():
    """Create a database engine with the problematic index removed"""
    # First, check if the tick table and the problematic index exist in metadata
    metadata = Base.metadata
    if 'tick' in metadata.tables:
        tick_table = metadata.tables['tick']
        indexes_to_remove = []
        
        # Find the problematic index
        for idx in list(tick_table.indexes):
            if idx.name == 'ix_tick_timestamp':
                indexes_to_remove.append(idx)
        
        # Remove it
        for idx in indexes_to_remove:
            tick_table.indexes.remove(idx)
            print(f"Removed index: {idx.name}")
    
    # Create the engine and tables
    engine = create_engine('sqlite:///:memory:', echo=False)
    Base.metadata.create_all(engine)
    
    yield engine
    
    # Cleanup
    Base.metadata.drop_all(engine)


from sqlalchemy import text

@pytest.fixture
def test_db(db_engine):
    """Create a test session with clean tables for each test"""
    # Create session
    Session = sessionmaker(bind=db_engine)
    session = Session()
    
    # Clear all existing data from tables in reverse foreign key order
    # Use text() to properly wrap SQL statements
    session.execute(text("DELETE FROM order_book_snapshot"))
    session.execute(text("DELETE FROM tick"))
    session.execute(text("DELETE FROM ohlcv"))
    session.execute(text("DELETE FROM instrument"))
    session.commit()
    
    yield session
    
    # Clean up after test
    session.rollback()
    session.close()



@pytest.fixture
def sample_instrument(test_db):
    """Create a sample instrument for testing"""
    instrument = Instrument(
        id=uuid.uuid4(),
        symbol="AAPL",
        name="Apple Inc.",
        asset_class=AssetClass.EQUITY.value,
        exchange="NASDAQ",
        currency="USD",
        price_adj_factor=Decimal("1.0"),
        active=True,
        specifications={
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "listing_date": "1980-12-12"
        }
    )
    test_db.add(instrument)
    test_db.commit()
    test_db.refresh(instrument)  # Refresh to get the committed values
    return instrument


@pytest.fixture
def sample_ohlcv(test_db, sample_instrument):
    """Create a sample OHLCV record for testing"""
    ohlcv = OHLCV(
        id=uuid.uuid4(),
        instrument_id=sample_instrument.id,
        timestamp=datetime.utcnow(),
        source=DataSource.EXCHANGE.value,
        source_timestamp=datetime.utcnow(),
        open=Decimal("150.25"),
        high=Decimal("152.75"),
        low=Decimal("149.50"),
        close=Decimal("151.80"),
        volume=Decimal("10000000"),
        interval="1h",
        vwap=Decimal("151.25"),
        trades_count=5000,
        metadata={
            "session": "regular",
            "adjusted": True
        }
    )
    test_db.add(ohlcv)
    test_db.commit()
    test_db.refresh(ohlcv)
    return ohlcv


@pytest.fixture
def sample_tick(test_db, sample_instrument):
    """Create a sample tick record for testing"""
    tick = Tick(
        id=uuid.uuid4(),
        instrument_id=sample_instrument.id,
        timestamp=datetime.utcnow(),
        source=DataSource.EXCHANGE.value,
        source_timestamp=datetime.utcnow(),
        price=Decimal("151.80"),
        volume=Decimal("100"),
        trade_id="T12345678",
        side="buy",
        trade_data={
            "venue": "NASDAQ",
            "condition": "regular"
        },
        metadata={
            "session": "regular"
        }
    )
    test_db.add(tick)
    test_db.commit()
    test_db.refresh(tick)
    return tick


@pytest.fixture
def sample_orderbook(test_db, sample_instrument):
    """Create a sample order book snapshot for testing"""
    orderbook = OrderBookSnapshot(
        id=uuid.uuid4(),
        instrument_id=sample_instrument.id,
        timestamp=datetime.utcnow(),
        source=DataSource.EXCHANGE.value,
        source_timestamp=datetime.utcnow(),
        depth=5,
        bids=[[151.75, 100], [151.70, 200], [151.65, 300], [151.60, 400], [151.55, 500]],
        asks=[[151.85, 150], [151.90, 250], [151.95, 350], [152.00, 450], [152.05, 550]],
        spread=Decimal("0.10"),
        weighted_mid_price=Decimal("151.80"),
        imbalance=0.25,
        metadata={
            "venue": "NASDAQ",
            "market_state": "open"
        }
    )
    test_db.add(orderbook)
    test_db.commit()
    test_db.refresh(orderbook)
    return orderbook


class TestInstrument:
    """Test suite for Instrument model"""

    def test_create_instrument(self, test_db):
        """Test basic instrument creation"""
        instrument = Instrument(
            id=uuid.uuid4(),
            symbol="MSFT",
            name="Microsoft Corporation",
            asset_class=AssetClass.EQUITY.value,
            exchange="NASDAQ",
            currency="USD",
            price_adj_factor=Decimal("1.0"),
            active=True,
            specifications={
                "sector": "Technology",
                "industry": "Software"
            }
        )
        
        test_db.add(instrument)
        test_db.commit()
        
        # Retrieve and verify
        saved = test_db.query(Instrument).filter_by(symbol="MSFT").first()
        assert saved is not None
        assert saved.symbol == "MSFT"
        assert saved.name == "Microsoft Corporation"
        assert saved.asset_class == AssetClass.EQUITY.value
        assert saved.currency == "USD"
        assert saved.specifications["sector"] == "Technology"
        assert saved.specifications["industry"] == "Software"

    def test_instrument_unique_constraint(self, test_db, sample_instrument):
        """Test unique constraint on symbol and exchange"""
        # Try to create another instrument with same symbol and exchange
        duplicate = Instrument(
            id=uuid.uuid4(),
            symbol=sample_instrument.symbol,
            name="Duplicate Inc.",
            asset_class=AssetClass.EQUITY.value,
            exchange=sample_instrument.exchange,
            currency="USD",
            price_adj_factor=Decimal("1.0"),
            active=True
        )
        
        test_db.add(duplicate)
        
        # Should raise an IntegrityError due to unique constraint
        with pytest.raises(IntegrityError):
            test_db.commit()
        
        # Rollback for cleanup
        test_db.rollback()

    def test_invalid_asset_class(self, test_db):
        """Test validation of asset class"""
        with pytest.raises(ValueError, match="Invalid asset class"):
            instrument = Instrument(
                id=uuid.uuid4(),
                symbol="INVALID",
                name="Invalid Asset Class",
                asset_class="not_a_valid_asset_class",  # Invalid value
                exchange="TEST",
                currency="USD"
            )
            test_db.add(instrument)
            test_db.commit()
        
        # Rollback for cleanup
        test_db.rollback()

    def test_invalid_currency(self, test_db):
        """Test validation of currency format"""
        with pytest.raises(ValueError, match="Currency must be a 3-letter ISO code"):
            instrument = Instrument(
                id=uuid.uuid4(),
                symbol="INVALID",
                name="Invalid Currency",
                asset_class=AssetClass.EQUITY.value,
                exchange="TEST",
                currency="USDD"  # Invalid length
            )
            test_db.add(instrument)
            test_db.commit()
        
        # Rollback for cleanup
        test_db.rollback()

    def test_currency_uppercase(self, test_db):
        """Test that currency is stored as uppercase"""
        instrument = Instrument(
            id=uuid.uuid4(),
            symbol="CURR",
            name="Currency Test",
            asset_class=AssetClass.EQUITY.value,
            exchange="TEST",
            currency="usd"  # Lowercase
        )
        
        test_db.add(instrument)
        test_db.commit()
        
        # Verify currency is uppercase
        saved = test_db.query(Instrument).filter_by(symbol="CURR").first()
        assert saved.currency == "USD"

    def test_json_specifications(self, test_db):
        """Test JSON specifications field"""
        specs = {
            "sector": "Finance",
            "industry": "Banking",
            "market_cap": 1000000000,
            "employees": 50000,
            "founded": 1950
        }
        
        instrument = Instrument(
            id=uuid.uuid4(),
            symbol="BANK",
            name="Banking Corp",
            asset_class=AssetClass.EQUITY.value,
            exchange="NYSE",
            currency="USD",
            specifications=specs
        )
        
        test_db.add(instrument)
        test_db.commit()
        
        # Retrieve and verify
        saved = test_db.query(Instrument).filter_by(symbol="BANK").first()
        assert saved.specifications == specs
        assert saved.specifications["sector"] == "Finance"
        assert saved.specifications["market_cap"] == 1000000000


class TestOHLCV:
    """Test suite for OHLCV model"""

    def test_create_ohlcv(self, test_db, sample_instrument):
        """Test basic OHLCV creation"""
        timestamp = datetime.utcnow()
        ohlcv = OHLCV(
            id=uuid.uuid4(),
            instrument_id=sample_instrument.id,
            timestamp=timestamp,
            source=DataSource.EXCHANGE.value,
            open=Decimal("150.25"),
            high=Decimal("152.75"),
            low=Decimal("149.50"),
            close=Decimal("151.80"),
            volume=Decimal("10000000"),
            interval="1d",
            metadata={
                "session": "regular",
                "has_gaps": False
            }
        )
        
        test_db.add(ohlcv)
        test_db.commit()
        
        # Retrieve and verify
        saved = test_db.query(OHLCV).filter_by(instrument_id=sample_instrument.id).order_by(OHLCV.timestamp.desc()).first()
        assert saved is not None
        assert saved.open == Decimal("150.25")
        assert saved.high == Decimal("152.75")
        assert saved.low == Decimal("149.50")
        assert saved.close == Decimal("151.80")
        assert saved.interval == "1d"
        assert saved.metadata["session"] == "regular"
        
        # Test relationship to instrument
        assert saved.instrument_id == sample_instrument.id
        assert saved.instrument.symbol == "AAPL"

    def test_invalid_interval(self, test_db, sample_instrument):
        """Test validation of interval format"""
        with pytest.raises(ValueError, match="Invalid interval"):
            ohlcv = OHLCV(
                id=uuid.uuid4(),
                instrument_id=sample_instrument.id,
                timestamp=datetime.utcnow(),
                source=DataSource.EXCHANGE.value,
                open=Decimal("150.25"),
                high=Decimal("152.75"),
                low=Decimal("149.50"),
                close=Decimal("151.80"),
                volume=Decimal("10000000"),
                interval="invalid"  # Invalid interval
            )
            test_db.add(ohlcv)
            test_db.commit()
        
        # Rollback for cleanup
        test_db.rollback()

    def test_negative_price_validation(self, test_db, sample_instrument):
        """Test validation of price values (must be positive)"""
        with pytest.raises(ValueError, match="price cannot be negative"):
            ohlcv = OHLCV(
                id=uuid.uuid4(),
                instrument_id=sample_instrument.id,
                timestamp=datetime.utcnow(),
                source=DataSource.EXCHANGE.value,
                open=Decimal("150.25"),
                high=Decimal("152.75"),
                low=Decimal("-1.50"),  # Negative price
                close=Decimal("151.80"),
                volume=Decimal("10000000"),
                interval="1d"
            )
            test_db.add(ohlcv)
            test_db.commit()
        
        # Rollback for cleanup
        test_db.rollback()

    def test_data_source_validation(self, test_db, sample_instrument):
        """Test validation of data source enum"""
        with pytest.raises(ValueError, match="Invalid data source"):
            ohlcv = OHLCV(
                id=uuid.uuid4(),
                instrument_id=sample_instrument.id,
                timestamp=datetime.utcnow(),
                source="invalid_source",  # Invalid source
                open=Decimal("150.25"),
                high=Decimal("152.75"),
                low=Decimal("149.50"),
                close=Decimal("151.80"),
                volume=Decimal("10000000"),
                interval="1d"
            )
            test_db.add(ohlcv)
            test_db.commit()
        
        # Rollback for cleanup
        test_db.rollback()


class TestTick:
    """Test suite for Tick model"""

    def test_create_tick(self, test_db, sample_instrument):
        """Test basic tick creation"""
        timestamp = datetime.utcnow()
        tick = Tick(
            id=uuid.uuid4(),
            instrument_id=sample_instrument.id,
            timestamp=timestamp,
            source=DataSource.EXCHANGE.value,
            price=Decimal("151.80"),
            volume=Decimal("100"),
            trade_id="T12345678",
            side="buy"
        )
        
        test_db.add(tick)
        test_db.commit()
        
        # Retrieve and verify
        saved = test_db.query(Tick).filter_by(trade_id="T12345678").first()
        assert saved is not None
        assert saved.price == Decimal("151.80")
        assert saved.volume == Decimal("100")
        assert saved.side == "buy"
        
        # Test relationship to instrument
        assert saved.instrument_id == sample_instrument.id
        assert saved.instrument.symbol == "AAPL"

    def test_invalid_side(self, test_db, sample_instrument):
        """Test validation of side field"""
        with pytest.raises(ValueError, match="Trade side must be 'buy' or 'sell'"):
            tick = Tick(
                id=uuid.uuid4(),
                instrument_id=sample_instrument.id,
                timestamp=datetime.utcnow(),
                source=DataSource.EXCHANGE.value,
                price=Decimal("151.80"),
                volume=Decimal("100"),
                trade_id="T12345678",
                side="invalid_side"  # Invalid side
            )
            test_db.add(tick)
            test_db.commit()
        
        # Rollback for cleanup
        test_db.rollback()

    def test_tick_with_trade_data(self, test_db, sample_instrument):
        """Test tick with additional trade data as JSON"""
        trade_data = {
            "maker": True,
            "fee": 0.1,
            "fee_currency": "USD",
            "order_id": "O12345678"
        }
        
        tick = Tick(
            id=uuid.uuid4(),
            instrument_id=sample_instrument.id,
            timestamp=datetime.utcnow(),
            source=DataSource.EXCHANGE.value,
            price=Decimal("151.80"),
            volume=Decimal("100"),
            trade_id="T12345679",
            side="sell",
            trade_data=trade_data
        )
        
        test_db.add(tick)
        test_db.commit()
        
        # Retrieve and verify
        saved = test_db.query(Tick).filter_by(trade_id="T12345679").first()
        assert saved is not None
        assert saved.trade_data["maker"] is True
        assert saved.trade_data["fee"] == 0.1
        assert saved.trade_data["order_id"] == "O12345678"


class TestOrderBookSnapshot:
    """Test suite for OrderBookSnapshot model"""

    def test_create_orderbook(self, test_db, sample_instrument):
        """Test basic order book snapshot creation"""
        timestamp = datetime.utcnow()
        orderbook = OrderBookSnapshot(
            id=uuid.uuid4(),
            instrument_id=sample_instrument.id,
            timestamp=timestamp,
            source=DataSource.EXCHANGE.value,
            depth=3,
            bids=[[151.75, 100], [151.70, 200], [151.65, 300]],
            asks=[[151.85, 150], [151.90, 250], [151.95, 350]],
            spread=Decimal("0.10"),
            weighted_mid_price=Decimal("151.80"),
            imbalance=0.25
        )
        
        test_db.add(orderbook)
        test_db.commit()
        
        # Retrieve and verify
        saved = test_db.query(OrderBookSnapshot).filter_by(instrument_id=sample_instrument.id).first()
        assert saved is not None
        assert saved.depth == 3
        assert len(saved.bids) == 3
        assert len(saved.asks) == 3
        assert saved.bids[0][0] == 151.75
        assert saved.asks[0][0] == 151.85
        assert saved.spread == Decimal("0.10")
        assert saved.imbalance == 0.25
        
        # Test relationship to instrument
        assert saved.instrument_id == sample_instrument.id
        assert saved.instrument.symbol == "AAPL"

    def test_invalid_book_structure(self, test_db, sample_instrument):
        """Test validation of book data structure"""
        with pytest.raises(ValueError, match="must be a list of"):
            orderbook = OrderBookSnapshot(
                id=uuid.uuid4(),
                instrument_id=sample_instrument.id,
                timestamp=datetime.utcnow(),
                source=DataSource.EXCHANGE.value,
                depth=3,
                # Invalid structure - not a list of lists
                bids={"price": 151.75, "volume": 100},
                asks=[[151.85, 150], [151.90, 250], [151.95, 350]],
                spread=Decimal("0.10")
            )
            test_db.add(orderbook)
            test_db.commit()
        
        # Rollback for cleanup
        test_db.rollback()
        
        with pytest.raises(ValueError, match="must be a"):
            orderbook = OrderBookSnapshot(
                id=uuid.uuid4(),
                instrument_id=sample_instrument.id,
                timestamp=datetime.utcnow(),
                source=DataSource.EXCHANGE.value,
                depth=3,
                bids=[[151.75, 100], [151.70, 200], [151.65, 300]],
                # Invalid structure - each level should be [price, volume]
                asks=[[151.85, 150, "extra"], [151.90], [151.95, 350]],
                spread=Decimal("0.10")
            )
            test_db.add(orderbook)
            test_db.commit()
        
        # Rollback for cleanup
        test_db.rollback()

    def test_invalid_imbalance(self, test_db, sample_instrument):
        """Test validation of imbalance range (-1 to 1)"""
        with pytest.raises(ValueError, match="Imbalance must be between -1 and 1"):
            orderbook = OrderBookSnapshot(
                id=uuid.uuid4(),
                instrument_id=sample_instrument.id,
                timestamp=datetime.utcnow(),
                source=DataSource.EXCHANGE.value,
                depth=3,
                bids=[[151.75, 100], [151.70, 200], [151.65, 300]],
                asks=[[151.85, 150], [151.90, 250], [151.95, 350]],
                spread=Decimal("0.10"),
                imbalance=1.5  # Invalid imbalance - must be between -1 and 1
            )
            test_db.add(orderbook)
            test_db.commit()
        
        # Rollback for cleanup
        test_db.rollback()

    def test_json_book_data(self, test_db, sample_instrument):
        """Test order book with JSON string input"""
        bids_json = json.dumps([[151.75, 100], [151.70, 200], [151.65, 300]])
        asks_json = json.dumps([[151.85, 150], [151.90, 250], [151.95, 350]])
        
        orderbook = OrderBookSnapshot(
            id=uuid.uuid4(),
            instrument_id=sample_instrument.id,
            timestamp=datetime.utcnow(),
            source=DataSource.EXCHANGE.value,
            depth=3,
            bids=bids_json,  # JSON string
            asks=asks_json,  # JSON string
            spread=Decimal("0.10")
        )
        
        test_db.add(orderbook)
        test_db.commit()
        
        # Retrieve and verify - should be parsed to proper structure
        saved = test_db.query(OrderBookSnapshot).order_by(OrderBookSnapshot.timestamp.desc()).first()
        assert saved is not None
        assert len(saved.bids) == 3
        assert len(saved.asks) == 3
        assert saved.bids[0][0] == 151.75
        assert saved.asks[0][0] == 151.85


class TestMarketDataQueries:
    """Test suite for common market data queries"""

    def test_latest_ohlcv(self, test_db, sample_instrument):
        """Test retrieving latest OHLCV data"""
        # Create multiple OHLCV records with different timestamps
        now = datetime.utcnow()
        
        ohlcv1 = OHLCV(
            id=uuid.uuid4(),
            instrument_id=sample_instrument.id,
            timestamp=now - timedelta(hours=2),
            source=DataSource.EXCHANGE.value,
            open=Decimal("150.00"),
            high=Decimal("152.00"),
            low=Decimal("149.00"),
            close=Decimal("151.00"),
            volume=Decimal("10000"),
            interval="1h"
        )
        
        ohlcv2 = OHLCV(
            id=uuid.uuid4(),
            instrument_id=sample_instrument.id,
            timestamp=now - timedelta(hours=1),
            source=DataSource.EXCHANGE.value,
            open=Decimal("151.00"),
            high=Decimal("153.00"),
            low=Decimal("150.00"),
            close=Decimal("152.00"),
            volume=Decimal("11000"),
            interval="1h"
        )
        
        ohlcv3 = OHLCV(
            id=uuid.uuid4(),
            instrument_id=sample_instrument.id,
            timestamp=now,
            source=DataSource.EXCHANGE.value,
            open=Decimal("152.00"),
            high=Decimal("154.00"),
            low=Decimal("151.00"),
            close=Decimal("153.00"),
            volume=Decimal("12000"),
            interval="1h"
        )
        
        test_db.add_all([ohlcv1, ohlcv2, ohlcv3])
        test_db.commit()
        
        # Query the latest OHLCV
        latest = test_db.query(OHLCV).filter_by(
            instrument_id=sample_instrument.id, 
            interval="1h"
        ).order_by(OHLCV.timestamp.desc()).first()
        
        # Verify it's the most recent one
        assert latest is not None
        assert latest.close == Decimal("153.00")
        assert latest.timestamp.replace(microsecond=0) == now.replace(microsecond=0)

    def test_ohlcv_time_range(self, test_db, sample_instrument):
        """Test retrieving OHLCV data within a time range"""
        # Create multiple OHLCV records with different timestamps
        now = datetime.utcnow()
        
        # Create 5 hourly candles over the last 5 hours
        candles = []
        for i in range(5):
            candles.append(
                OHLCV(
                    id=uuid.uuid4(),
                    instrument_id=sample_instrument.id,
                    timestamp=now - timedelta(hours=4-i),
                    source=DataSource.EXCHANGE.value,
                    open=Decimal(f"15{i}.00"),
                    high=Decimal(f"15{i+1}.00"),
                    low=Decimal(f"14{i+8}.00"),
                    close=Decimal(f"15{i}.50"),
                    volume=Decimal(f"1000{i}"),
                    interval="1h"
                )
            )
        
        test_db.add_all(candles)
        test_db.commit()
        
        # Query for a 3-hour range in the middle
        start_time = now - timedelta(hours=3)
        end_time = now - timedelta(hours=1)
        
        results = test_db.query(OHLCV).filter(
            OHLCV.instrument_id == sample_instrument.id,
            OHLCV.interval == "1h",
            OHLCV.timestamp >= start_time,
            OHLCV.timestamp <= end_time
        ).order_by(OHLCV.timestamp).all()
        
        # Should get 3 candles
        assert len(results) == 3
        assert results[0].close == Decimal("151.50")
        assert results[1].close == Decimal("152.50")
        assert results[2].close == Decimal("153.50")

    def test_order_book_depth(self, test_db, sample_instrument):
        """Test retrieving order book with specific depth"""
        # Create an order book with deep depth
        deep_book = OrderBookSnapshot(
            id=uuid.uuid4(),
            instrument_id=sample_instrument.id,
            timestamp=datetime.utcnow(),
            source=DataSource.EXCHANGE.value,
            depth=10,
            bids=[[p, 100*i] for i, p in enumerate([151.75, 151.70, 151.65, 151.60, 151.55, 
                                                 151.50, 151.45, 151.40, 151.35, 151.30], 1)],
            asks=[[p, 150*i] for i, p in enumerate([151.85, 151.90, 151.95, 152.00, 152.05, 
                                                 152.10, 152.15, 152.20, 152.25, 152.30], 1)],
            spread=Decimal("0.10")
        )
        
        test_db.add(deep_book)
        test_db.commit()
        
        # Retrieve the order book
        saved_book = test_db.query(OrderBookSnapshot).filter_by(
            instrument_id=sample_instrument.id
        ).order_by(OrderBookSnapshot.timestamp.desc()).first()
        
        # Verify full depth
        assert saved_book is not None
        assert len(saved_book.bids) == 10
        assert len(saved_book.asks) == 10
        
        # Get first 3 levels only
        top_bids = saved_book.bids[:3]
        top_asks = saved_book.asks[:3]
        
        assert len(top_bids) == 3
        assert len(top_asks) == 3
        assert top_bids[0][0] == 151.75  # Best bid price
        assert top_asks[0][0] == 151.85  # Best ask price