"""
Unit tests for the Market Data Service.

These tests verify the business logic for retrieving and processing market data,
with properly mocked database dependencies.
"""
import pytest
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

from fastapi import HTTPException

from app.core.database import DatabaseType
from app.services.market_data import MarketDataService
from app.schemas.market_data import (
    AssetClass, DataSource, TimeInterval, 
    InstrumentResponse, OHLCVResponse, MarketDataStats,
    OHLCVFilter
)

from app.services import market_data
from app.services.market_data import MarketDataService

@pytest.fixture
def market_data_service(monkeypatch, mock_db_connections):
    """Create a MarketDataService instance with mocked dependencies."""
    # Override the get_db_instance function to return our mocks before initializing the service
    monkeypatch.setattr('app.services.market_data.get_db_instance', mock_db_connections.side_effect)
    
    # Now create the service with the mocked dependencies
    service = market_data.MarketDataService()
    
    # Ensure our mocked databases are correctly assigned
    service.timescale_db = mock_db_connections.mock_timescale
    service.redis_db = mock_db_connections.mock_redis
    
    return service

@pytest.fixture
def db_session_mock():
    """Mock for database session context manager."""
    session_mock = MagicMock()
    context_manager_mock = MagicMock()
    context_manager_mock.__enter__.return_value = session_mock
    return context_manager_mock, session_mock

@pytest.fixture
def sample_instrument_id():
    """Sample instrument ID for testing."""
    return uuid.uuid4()

@pytest.fixture
def sample_instrument(sample_instrument_id):
    """Sample instrument response for testing."""
    return InstrumentResponse(
        id=sample_instrument_id,
        symbol="AAPL",
        name="Apple Inc.",
        asset_class=AssetClass.EQUITY,
        exchange="NASDAQ",
        currency="USD",
        price_adj_factor=Decimal("1.0"),
        active=True,
        created_at=datetime.utcnow()
    )

@pytest.fixture
def sample_ohlcv(sample_instrument_id):
    """Sample OHLCV data for testing."""
    return OHLCVResponse(
        id=uuid.uuid4(),
        instrument_id=sample_instrument_id,
        timestamp=datetime.utcnow(),
        source=DataSource.EXCHANGE,
        open=Decimal("150.25"),
        high=Decimal("155.50"),
        low=Decimal("149.75"),
        close=Decimal("152.30"),
        volume=Decimal("1000000"),
        interval=TimeInterval.DAY_1,
        vwap=Decimal("151.50"),
        trades_count=5000,
        created_at=datetime.utcnow()
    )

class TestMarketDataService:
    """Tests for the MarketDataService class."""
    
    def test_get_instrument_by_id(self, market_data_service, sample_instrument, db_session_mock):
        """Test retrieving an instrument by ID."""
        context_mock, session_mock = db_session_mock
        
        # Setup mocks
        market_data_service.redis_db.get_json.return_value = None  # Cache miss
        market_data_service.timescale_db.session.return_value = context_mock
        
        # Mock the database result
        result_mock = MagicMock()
        for key, value in sample_instrument.model_dump().items():
            setattr(result_mock, key, value)
        session_mock.execute().fetchone.return_value = result_mock
        
        # Call the method
        result = market_data_service.get_instrument_by_id(sample_instrument.id)
        
        # Verify the result
        assert result is not None
        assert result.id == sample_instrument.id
        assert result.symbol == sample_instrument.symbol
        assert result.asset_class == sample_instrument.asset_class
        
        # Verify the database was queried
        session_mock.execute.assert_called_once()
        
        # Verify the cache was used
        market_data_service.redis_db.get_json.assert_called_once()
        market_data_service.redis_db.set_json.assert_called_once()
    
    def test_get_instrument_by_symbol(self, market_data_service, sample_instrument, db_session_mock):
        """Test retrieving an instrument by symbol."""
        context_mock, session_mock = db_session_mock
        
        # Setup mocks
        market_data_service.redis_db.get_json.return_value = None  # Cache miss
        market_data_service.timescale_db.session.return_value = context_mock
        
        # Mock the database result
        result_mock = MagicMock()
        for key, value in sample_instrument.model_dump().items():
            setattr(result_mock, key, value)
        session_mock.execute().fetchone.return_value = result_mock
        
        # Call the method
        result = market_data_service.get_instrument_by_symbol(sample_instrument.symbol)
        
        # Verify the result
        assert result is not None
        assert result.id == sample_instrument.id
        assert result.symbol == sample_instrument.symbol
        assert result.asset_class == sample_instrument.asset_class
    
    def test_resolve_instrument_id_with_id(self, market_data_service, sample_instrument_id):
        """Test resolving an instrument ID when the ID is already provided."""
        # Call the method
        result = market_data_service.resolve_instrument_id(instrument_id=sample_instrument_id)
        
        # Verify the result
        assert result == sample_instrument_id
    
    def test_resolve_instrument_id_with_symbol(self, market_data_service, sample_instrument):
        """Test resolving an instrument ID from a symbol."""
        # Setup mock
        with patch.object(
            market_data_service, 'get_instrument_by_symbol', return_value=sample_instrument
        ) as mock_get_instrument:
            
            # Call the method
            result = market_data_service.resolve_instrument_id(symbol=sample_instrument.symbol)
            
            # Verify the result
            assert result == sample_instrument.id
            
            # Verify the correct method was called
            mock_get_instrument.assert_called_once_with(sample_instrument.symbol)
    
    def test_resolve_instrument_id_with_invalid_symbol(self, market_data_service):
        """Test resolving an instrument ID with an invalid symbol."""
        # Setup mock
        with patch.object(
            market_data_service, 'get_instrument_by_symbol', return_value=None
        ) as mock_get_instrument:
            
            # Call the method
            result = market_data_service.resolve_instrument_id(symbol="INVALID")
            
            # Verify the result
            assert result is None
            
            # Verify the correct method was called
            mock_get_instrument.assert_called_once_with("INVALID")
    
    def test_resolve_instrument_id_with_no_params(self, market_data_service):
        """Test resolving an instrument ID with no parameters."""
        # Call the method and expect an exception
        with pytest.raises(ValueError, match="Either instrument_id or symbol must be provided"):
            market_data_service.resolve_instrument_id()
    
    def test_get_latest_ohlcv(self, market_data_service, sample_instrument, sample_ohlcv, db_session_mock):
        """Test retrieving the latest OHLCV data."""
        context_mock, session_mock = db_session_mock
        
        # Setup mocks
        with patch.object(
            market_data_service, 'resolve_instrument_id', return_value=sample_instrument.id
        ) as mock_resolve:
            market_data_service.redis_db.get_json.return_value = None  # Cache miss
            market_data_service.timescale_db.session.return_value = context_mock
            
            # Mock the database result
            result_mock = MagicMock()
            for key, value in sample_ohlcv.model_dump().items():
                setattr(result_mock, key, value)
            session_mock.execute().fetchone.return_value = result_mock
            
            # Call the method
            result = market_data_service.get_latest_ohlcv(
                symbol=sample_instrument.symbol,
                interval=TimeInterval.DAY_1
            )
            
            # Verify the result
            assert result is not None
            assert result.instrument_id == sample_ohlcv.instrument_id
            assert result.open == sample_ohlcv.open
            assert result.close == sample_ohlcv.close
            assert result.interval == sample_ohlcv.interval
            
            # Verify the correct methods were called
            mock_resolve.assert_called_once_with(None, sample_instrument.symbol)
            market_data_service.redis_db.get_json.assert_called_once()
            market_data_service.redis_db.set_json.assert_called_once()
    
    def test_get_latest_ohlcv_instrument_not_found(self, market_data_service):
        """Test retrieving the latest OHLCV data when instrument is not found."""
        # Setup mocks
        with patch.object(
            market_data_service, 'resolve_instrument_id', return_value=None
        ) as mock_resolve:
            # Call the method and expect an exception
            with pytest.raises(HTTPException) as excinfo:
                market_data_service.get_latest_ohlcv(symbol="INVALID")
            
            # Verify the exception
            assert excinfo.value.status_code == 404
            assert "Instrument not found" in excinfo.value.detail
            
            # Verify the correct method was called
            mock_resolve.assert_called_once_with(None, "INVALID")
    
    def test_get_ohlcv_data(self, market_data_service, sample_instrument, sample_ohlcv, db_session_mock):
        """Test retrieving OHLCV data with filters."""
        context_mock, session_mock = db_session_mock
        
        # Setup mocks
        with patch.object(
            market_data_service, 'resolve_instrument_id', return_value=sample_instrument.id
        ) as mock_resolve:
            market_data_service.timescale_db.session.return_value = context_mock
            
            # Mock the database results
            result_mock = MagicMock()
            for key, value in sample_ohlcv.model_dump().items():
                setattr(result_mock, key, value)
            session_mock.execute().fetchall.return_value = [result_mock]
            
            # Create filter parameters
            filter_params = OHLCVFilter(
                symbol=sample_instrument.symbol,
                interval=TimeInterval.DAY_1,
                start_timestamp=datetime.utcnow() - timedelta(days=30),
                end_timestamp=datetime.utcnow(),
                limit=100
            )
            
            # Call the method
            results = market_data_service.get_ohlcv_data(filter_params)
            
            # Verify the results
            assert len(results) == 1
            assert results[0].instrument_id == sample_ohlcv.instrument_id
            assert results[0].open == sample_ohlcv.open
            assert results[0].close == sample_ohlcv.close
            assert results[0].interval == sample_ohlcv.interval
            
            # Verify the correct methods were called
            mock_resolve.assert_called_once_with(filter_params.instrument_id, filter_params.symbol)
            session_mock.execute.assert_called_once()
    
    def test_get_aggregated_ohlcv(self, market_data_service, sample_instrument, db_session_mock):
        """Test retrieving aggregated OHLCV data."""
        context_mock, session_mock = db_session_mock
        
        # Setup mocks
        with patch.object(
            market_data_service, 'resolve_instrument_id', return_value=sample_instrument.id
        ) as mock_resolve:
            market_data_service.redis_db.get_json.return_value = None  # Cache miss
            market_data_service.timescale_db.session.return_value = context_mock
            
            # Sample aggregated data
            now = datetime.utcnow()
            bucket1 = now - timedelta(days=7)
            bucket2 = now
            
            # Mock database results
            row1 = MagicMock()
            row1.bucket = bucket1
            row1.instrument_id = sample_instrument.id
            row1.open = Decimal("150.25")
            row1.high = Decimal("155.50")
            row1.low = Decimal("149.75")
            row1.close = Decimal("152.30")
            row1.volume = Decimal("1000000")
            row1.vwap = Decimal("151.50")
            row1.trades_count = 5000
            row1.open_interest = None
            row1.adjusted_close = None
            row1.interval = "1w"
            row1.data_points = 5
            
            row2 = MagicMock()
            row2.bucket = bucket2
            row2.instrument_id = sample_instrument.id
            row2.open = Decimal("152.30")
            row2.high = Decimal("157.50")
            row2.low = Decimal("151.75")
            row2.close = Decimal("155.30")
            row2.volume = Decimal("1200000")
            row2.vwap = Decimal("154.50")
            row2.trades_count = 6000
            row2.open_interest = None
            row2.adjusted_close = None
            row2.interval = "1w"
            row2.data_points = 5
            
            session_mock.execute().fetchall.return_value = [row1, row2]
            
            # Call the method
            results = market_data_service.get_aggregated_ohlcv(
                symbol=sample_instrument.symbol,
                target_interval=TimeInterval.WEEK_1
            )
            
            # Verify the results
            assert len(results) == 2
            
            # Check first bucket
            assert results[0]["timestamp"] == bucket1
            assert results[0]["open"] == Decimal("150.25")
            assert results[0]["close"] == Decimal("152.30")
            assert results[0]["interval"] == TimeInterval.WEEK_1
            assert results[0]["source"] == DataSource.CALCULATED
            assert "metadata" in results[0]
            assert results[0]["metadata"]["aggregated"] is True
            assert results[0]["metadata"]["data_points"] == 5
            
            # Check second bucket
            assert results[1]["timestamp"] == bucket2
            assert results[1]["open"] == Decimal("152.30")
            assert results[1]["close"] == Decimal("155.30")
            
            # Verify the correct methods were called
            mock_resolve.assert_called_once_with(None, sample_instrument.symbol)
            market_data_service.redis_db.get_json.assert_called_once()
            market_data_service.redis_db.set_json.assert_called_once()
    
    def test_get_market_data_statistics(self, market_data_service, sample_instrument, db_session_mock):
        """Test retrieving market data statistics."""
        context_mock, session_mock = db_session_mock
        
        # Setup mocks
        with patch.object(
            market_data_service, 'resolve_instrument_id', return_value=sample_instrument.id
        ) as mock_resolve:
            with patch.object(
                market_data_service, 'get_instrument_by_id', return_value=sample_instrument
            ) as mock_get_instrument:
                market_data_service.redis_db.get_json.return_value = None  # Cache miss
                market_data_service.timescale_db.session.return_value = context_mock
                
                # Sample statistics data
                stats = MagicMock()
                stats.first_timestamp = datetime.utcnow() - timedelta(days=30)
                stats.last_timestamp = datetime.utcnow()
                stats.record_count = 100
                stats.data_sources = ["exchange", "vendor"]
                stats.anomaly_percentage = 0.02
                
                session_mock.execute().fetchone.return_value = stats
                
                # Call the method
                result = market_data_service.get_market_data_statistics(
                    symbol=sample_instrument.symbol
                )
                
                # Verify the result
                assert result.instrument_id == sample_instrument.id
                assert result.symbol == sample_instrument.symbol
                assert result.record_count == 100
                assert len(result.data_sources) == 2
                assert DataSource.EXCHANGE in result.data_sources
                assert DataSource.VENDOR in result.data_sources
                assert result.anomaly_percentage == 0.02
                
                # Verify the correct methods were called
                mock_resolve.assert_called_once_with(None, sample_instrument.symbol)
                mock_get_instrument.assert_called_once_with(sample_instrument.id)
    
    def test_calculate_price_statistics(self, market_data_service, sample_instrument, db_session_mock):
        """Test calculating price statistics."""
        context_mock, session_mock = db_session_mock
        
        # Setup mocks
        with patch.object(
            market_data_service, 'resolve_instrument_id', return_value=sample_instrument.id
        ) as mock_resolve:
            market_data_service.timescale_db.session.return_value = context_mock
            
            # Sample statistics data
            stats = MagicMock()
            stats.avg_price = 150.50
            stats.min_price = 145.00
            stats.max_price = 155.00
            stats.std_dev_price = 2.50
            stats.median_price = 150.00
            stats.avg_volume = 1000000
            stats.max_volume = 2000000
            stats.total_volume = 30000000
            stats.price_range_percent = 0.0689  # (155-145)/145
            stats.data_points = 30
            
            # For price change calculation
            price_data = MagicMock()
            price_data.first_price = 148.00
            price_data.last_price = 152.00
            
            # Set up multiple return values for session_mock.execute().fetchone()
            session_mock.execute().fetchone.side_effect = [stats, price_data]
            
            # Call the method
            result = market_data_service.calculate_price_statistics(
                symbol=sample_instrument.symbol,
                interval=TimeInterval.DAY_1
            )
            
            # Verify the result
            assert result["avg_price"] == 150.50
            assert result["min_price"] == 145.00
            assert result["max_price"] == 155.00
            assert result["std_dev_price"] == 2.50
            assert result["first_price"] == 148.00
            assert result["last_price"] == 152.00
            assert result["price_change"] == 4.00
            assert result["percent_change"] == pytest.approx(2.70, 0.1)  # (152-148)/148 * 100
            
            # Verify the correct method was called
            mock_resolve.assert_called_once_with(None, sample_instrument.symbol)
            assert session_mock.execute.call_count == 2
    
    def test_get_multi_symbol_ohlcv(self, market_data_service):
        """Test retrieving OHLCV data for multiple symbols."""
        # Setup mock for get_ohlcv_data
        with patch.object(market_data_service, 'get_ohlcv_data') as mock_get_ohlcv:
            # Sample OHLCV responses
            aapl_data = [MagicMock()]
            msft_data = [MagicMock()]
            
            # Configure mock to return different data for different symbols
            def get_ohlcv_side_effect(filter_params):
                if filter_params.symbol == "AAPL":
                    return aapl_data
                elif filter_params.symbol == "MSFT":
                    return msft_data
                else:
                    raise HTTPException(status_code=404, detail="Instrument not found")
            
            mock_get_ohlcv.side_effect = get_ohlcv_side_effect
            
            # Mock model_dump to return a dictionary
            aapl_data[0].model_dump.return_value = {"symbol": "AAPL", "close": 152.30}
            msft_data[0].model_dump.return_value = {"symbol": "MSFT", "close": 252.30}
            
            # Call the method
            results = market_data_service.get_multi_symbol_ohlcv(
                symbols=["AAPL", "MSFT", "INVALID"],
                interval=TimeInterval.DAY_1
            )
            
            # Verify the results
            assert "AAPL" in results
            assert "MSFT" in results
            assert "INVALID" in results
            
            assert isinstance(results["AAPL"], list)
            assert isinstance(results["MSFT"], list)
            assert isinstance(results["INVALID"], dict)
            
            assert results["AAPL"][0]["symbol"] == "AAPL"
            assert results["MSFT"][0]["symbol"] == "MSFT"
            assert "error" in results["INVALID"]
            
            # Verify correct calls to the underlying method
            assert mock_get_ohlcv.call_count == 3
    
    def test_compare_instruments(self, market_data_service):
        """Test comparing multiple instruments."""
        with patch.object(market_data_service, 'calculate_price_statistics') as mock_stats:
            with patch.object(market_data_service, 'get_ohlcv_data') as mock_get_ohlcv:
                # Configure mock for calculate_price_statistics
                def stats_side_effect(symbol, **kwargs):
                    if symbol == "AAPL":
                        return {
                            "avg_price": 150.50,
                            "percent_change": 2.70
                        }
                    elif symbol == "MSFT":
                        return {
                            "avg_price": 250.50,
                            "percent_change": 5.40
                        }
                    else:
                        raise HTTPException(status_code=404, detail="Instrument not found")
                
                mock_stats.side_effect = stats_side_effect
                
                # Configure mock for get_ohlcv_data to return price data
                aapl_ohlcv1 = MagicMock()
                aapl_ohlcv1.timestamp = datetime(2023, 1, 1)
                aapl_ohlcv1.close = Decimal("150.00")
                
                aapl_ohlcv2 = MagicMock()
                aapl_ohlcv2.timestamp = datetime(2023, 1, 2)
                aapl_ohlcv2.close = Decimal("152.00")
                
                msft_ohlcv1 = MagicMock()
                msft_ohlcv1.timestamp = datetime(2023, 1, 1)
                msft_ohlcv1.close = Decimal("250.00")
                
                msft_ohlcv2 = MagicMock()
                msft_ohlcv2.timestamp = datetime(2023, 1, 2)
                msft_ohlcv2.close = Decimal("255.00")
                
                def get_ohlcv_side_effect(filter_params):
                    if filter_params.symbol == "AAPL":
                        return [aapl_ohlcv1, aapl_ohlcv2]
                    elif filter_params.symbol == "MSFT":
                        return [msft_ohlcv1, msft_ohlcv2]
                    else:
                        return []
                
                mock_get_ohlcv.side_effect = get_ohlcv_side_effect
                
                # Call the method
                result = market_data_service.compare_instruments(
                    symbols=["AAPL", "MSFT"],
                    interval=TimeInterval.DAY_1
                )
                
                # Verify the result
                assert "symbol_stats" in result
                assert "correlation_matrix" in result
                assert "relative_performance" in result
                assert "best_performer" in result
                assert "worst_performer" in result
                assert "time_period" in result
                
                # Check symbol stats
                assert "AAPL" in result["symbol_stats"]
                assert "MSFT" in result["symbol_stats"]
                assert result["symbol_stats"]["AAPL"]["percent_change"] == 2.70
                assert result["symbol_stats"]["MSFT"]["percent_change"] == 5.40
                
                # Check best/worst performers
                assert result["best_performer"][0] == "MSFT"
                assert result["best_performer"][1] == 5.40
                assert result["worst_performer"][0] == "AAPL"
                assert result["worst_performer"][1] == 2.70
                
                # Verify the correlation matrix
                assert "AAPL" in result["correlation_matrix"]
                assert "MSFT" in result["correlation_matrix"]["AAPL"]
                
                # Verify the correct methods were called
                assert mock_stats.call_count == 2
                assert mock_get_ohlcv.call_count == 2
    
    def test_detect_data_gaps(self, market_data_service, sample_instrument, db_session_mock):
        """Test detecting gaps in time-series data."""
        context_mock, session_mock = db_session_mock
        
        # Setup mocks
        with patch.object(
            market_data_service, 'resolve_instrument_id', return_value=sample_instrument.id
        ) as mock_resolve:
            market_data_service.timescale_db.session.return_value = context_mock
            
            # Sample timestamps with gaps
            now = datetime.utcnow()
            ts1 = now - timedelta(days=30)
            ts2 = now - timedelta(days=25)
            ts3 = now - timedelta(days=20)
            # Gap between day 20 and day 15
            ts4 = now - timedelta(days=15)
            ts5 = now - timedelta(days=10)
            # Gap between day 10 and day 5
            ts6 = now - timedelta(days=5)
            ts7 = now
            
            # Create mock rows
            rows = []
            for ts in [ts1, ts2, ts3, ts4, ts5, ts6, ts7]:
                row = MagicMock()
                row.timestamp = ts
                rows.append(row)
            
            session_mock.execute().fetchall.return_value = rows
            
            # Call the method
            results = market_data_service.detect_data_gaps(
                symbol=sample_instrument.symbol,
                interval=TimeInterval.DAY_1
            )
            
            # Verify the results
            assert len(results) == 2  # Should detect 2 gaps
            
            # First gap should be between day 20 and day 15
            assert results[0]["start"] == ts3.isoformat()
            assert results[0]["end"] == ts4.isoformat()
            assert results[0]["gap_hours"] == 120.0  # 5 days = 120 hours
            
            # Second gap should be between day 10 and day 5
            assert results[1]["start"] == ts5.isoformat()
            assert results[1]["end"] == ts6.isoformat()
            assert results[1]["gap_hours"] == 120.0  # 5 days = 120 hours
            
            # Verify the correct method was called
            mock_resolve.assert_called_once_with(None, sample_instrument.symbol)
    

    def test_correlation_calculation(self):
        """Test the correlation calculation with various inputs."""
        # Create a service instance
        service = MarketDataService()
        service.timescale_db = MagicMock()
        
        # Test with perfectly correlated series
        series1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        series2 = [2.0, 4.0, 6.0, 8.0, 10.0]
        correlation = service._calculate_correlation(series1, series2)
        assert correlation == pytest.approx(1.0)
        
        # Test with perfectly anti-correlated series
        series3 = [5.0, 4.0, 3.0, 2.0, 1.0]
        correlation = service._calculate_correlation(series1, series3)
        assert correlation == pytest.approx(-1.0)
        
        # Test with relatively uncorrelated series
        # Note: this actually has a correlation around 0.4, not "strictly" uncorrelated
        series4 = [1.0, 3.0, 2.0, 5.0, 4.0]
        correlation = service._calculate_correlation(series1, series4)
        assert 0.3 < correlation < 0.5
        
        # Test with empty series
        empty_series = []
        correlation = service._calculate_correlation(empty_series, series1)
        assert correlation == 0
        
        # Test with series containing only one element
        one_element = [1.0]
        correlation = service._calculate_correlation(one_element, series1)
        assert correlation == 0   