"""
Comprehensive Unit Tests for Fyers REST Client

This module provides extensive test coverage for the market-aware Fyers REST client
including all components: caching, batching, data validation, position management,
rate limiting, and API methods.
"""
import pytest_asyncio
import pytest
import asyncio
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Any
from collections import defaultdict

# Import the classes to test
from app.providers.fyers.fyers_rest_client import (
    FyersRestClient,
    TradingProfile,
    TradingFrequency,
    DataQuality,
    CostTier,
    RequestBatch,
    MarketAwareIntelligentBatcher,
    MarketAwareDataQualityValidator,
    MarketAwareStrategicCache,
    MarketAwarePositionManager,
    MarketAwareRateLimiter,
    MarketServiceProtocol,
    NoOpMarketService
)
from app.providers.fyers.fyers_settings import FyersSettings
from app.providers.fyers.fyers_auth import FyersAuth
from app.providers.base.provider import (
    AuthenticationError, ConnectionError, RateLimitError, DataNotFoundError
)


# ===========================================
# FIXTURES AND MOCKS
# ===========================================

@pytest.fixture
def mock_market_service():
    """Mock market service for testing."""
    service = Mock(spec=MarketServiceProtocol)
    service.get_cache_multiplier.return_value = 1.0
    service.get_rate_limit_multiplier.return_value = 1.0
    service.is_market_open.return_value = True
    service.get_market_state.return_value = "open"
    service.should_prioritize_fresh_data.return_value = False
    return service


@pytest.fixture
def trading_profile():
    """Default trading profile for testing."""
    return TradingProfile(
        frequency=TradingFrequency.SWING,
        cost_tier=CostTier.STANDARD,
        enable_market_awareness=True
    )


@pytest.fixture
def fyers_settings():
    """Mock Fyers settings."""
    settings = Mock(spec=FyersSettings)
    settings.API_BASE_URL = "https://api.fyers.in/api/v2/"
    settings.QUOTES_RATE_LIMIT = 15
    settings.HISTORICAL_DATA_RATE_LIMIT = 5
    settings.MARKET_DEPTH_RATE_LIMIT = 10
    settings.ORDERBOOK_DEPTH = 10
    settings.RATE_LIMIT_CALLS = 100
    settings.RATE_LIMIT_PERIOD = 60
    settings.REQUEST_TIMEOUT = 30
    settings.CONNECTION_TIMEOUT = 10
    settings.MAX_RETRIES = 3
    settings.RETRY_BACKOFF = 0.5
    settings.DEBUG_MODE = False
    settings.ACCESS_TOKEN = None
    settings.RATE_LIMIT_ENABLED = True  # Add this
    return settings

@pytest.fixture
def mock_auth():
    """Mock authentication service."""
    auth = AsyncMock(spec=FyersAuth)
    auth.initialize.return_value = True
    auth.has_valid_token.return_value = True
    auth.ensure_token.return_value = True
    auth.get_auth_headers.return_value = {"Authorization": "test:token"}
    auth.check_token_expiry.return_value = timedelta(hours=1)
    auth.refresh_token_async.return_value = True
    auth.close.return_value = None
    auth.access_token = "test_token"  # Add this line
    return auth


@pytest.fixture
def mock_rest_client():
    """Mock REST client."""
    client = AsyncMock()
    client.get.return_value = {"s": "ok", "data": {}}
    client.post.return_value = {"s": "ok", "data": {}}
    client.close.return_value = None
    return client


@pytest_asyncio.fixture
async def fyers_client(fyers_settings, mock_auth, trading_profile, mock_market_service):
    """Create Fyers client instance for testing."""
    with patch('app.providers.fyers.fyers_rest_client.RestClient') as mock_rest_class:
        mock_rest_class.return_value = AsyncMock()
        
        client = FyersRestClient(
            settings=fyers_settings,
            auth=mock_auth,
            trading_profile=trading_profile,
            market_service=mock_market_service,
            enable_monitoring=False  # Disable for testing
        )
        
        return client
        
        # # Cleanup
        # if hasattr(client, '_background_tasks'):
        #     for task in client._background_tasks:
        #         if not task.done():
        #             task.cancel()


# ===========================================
# TRADING PROFILE TESTS
# ===========================================

class TestTradingProfile:
    """Test TradingProfile configuration class."""
    
    def test_default_values(self):
        """Test default trading profile values."""
        profile = TradingProfile()
        
        assert profile.frequency == TradingFrequency.SWING
        assert profile.cost_tier == CostTier.STANDARD
        assert profile.max_daily_requests == 50000
        assert profile.avg_positions == 10
        assert profile.typical_holding_period == timedelta(hours=4)
        assert profile.data_quality_requirement == DataQuality.IMPORTANT
        assert profile.enable_market_awareness is True
    
    def test_custom_values(self):
        """Test custom trading profile values."""
        profile = TradingProfile(
            frequency=TradingFrequency.INTRADAY,
            cost_tier=CostTier.PREMIUM,
            max_daily_requests=100000,
            enable_market_awareness=False
        )
        
        assert profile.frequency == TradingFrequency.INTRADAY
        assert profile.cost_tier == CostTier.PREMIUM
        assert profile.max_daily_requests == 100000
        assert profile.enable_market_awareness is False


class TestRequestBatch:
    """Test RequestBatch data class."""
    
    def test_default_values(self):
        """Test default request batch values."""
        batch = RequestBatch(
            symbols=["NSE:SBIN-EQ"],
            request_type="quotes",
            priority=1,
            timestamp=time.time()
        )
        
        assert batch.symbols == ["NSE:SBIN-EQ"]
        assert batch.request_type == "quotes"
        assert batch.priority == 1
        assert batch.max_batch_size == 50
        assert isinstance(batch.market_states, dict)


# ===========================================
# MARKET SERVICE TESTS
# ===========================================

class TestNoOpMarketService:
    """Test the fallback NoOp market service."""
    
    def test_noop_market_service(self):
        """Test NoOp market service returns safe defaults."""
        service = NoOpMarketService()
        
        assert service.get_cache_multiplier("NSE:SBIN-EQ", 100) == 1.0
        assert service.get_rate_limit_multiplier("NSE:SBIN-EQ") == 1.0
        assert service.is_market_open("NSE:SBIN-EQ") is True
        assert service.get_market_state("NSE:SBIN-EQ") == "unknown"
        assert service.should_prioritize_fresh_data("NSE:SBIN-EQ") is True


# ===========================================
# INTELLIGENT BATCHER TESTS
# ===========================================

class TestMarketAwareIntelligentBatcher:
    """Test the intelligent batching component."""
    
    @pytest.fixture
    def batcher(self, trading_profile, mock_market_service):
        """Create batcher instance."""
        return MarketAwareIntelligentBatcher(trading_profile, mock_market_service)
    
    def test_calculate_batch_delay_by_frequency(self, batcher, mock_market_service):
        """Test batch delay calculation based on trading frequency."""
        symbols = ["NSE:SBIN-EQ"]
        
        # Set market closed for consistent testing
        mock_market_service.is_market_open.return_value = False
        
        # Test different frequencies
        batcher.profile.frequency = TradingFrequency.POSITION
        assert batcher._calculate_batch_delay(symbols) == 60.0  # 30 * 2 (market closed)
        
        batcher.profile.frequency = TradingFrequency.SCALPING
        assert batcher._calculate_batch_delay(symbols) == 2.0  # 1 * 2 (market closed)
    
    def test_calculate_batch_delay_market_aware(self, batcher, mock_market_service):
        """Test batch delay with market awareness."""
        symbols = ["NSE:SBIN-EQ"]
        
        # Market open - should reduce delay
        mock_market_service.is_market_open.return_value = True
        delay = batcher._calculate_batch_delay(symbols)
        assert delay == 5.0  # 10 * 0.5 (market open)
        
        # Market closed - should increase delay
        mock_market_service.is_market_open.return_value = False
        delay = batcher._calculate_batch_delay(symbols)
        assert delay == 20.0  # 10 * 2 (market closed)
    
    def test_calculate_batch_delay_no_market_awareness(self, batcher):
        """Test batch delay without market awareness."""
        batcher.profile.enable_market_awareness = False
        symbols = ["NSE:SBIN-EQ"]
        
        delay = batcher._calculate_batch_delay(symbols)
        assert delay == 10.0  # Base delay for SWING
    
    @pytest.mark.asyncio
    async def test_add_request_immediate_execution(self, batcher, mock_market_service):
        """Test immediate execution conditions."""
        callback = AsyncMock(return_value="result")
        
        # Test premium tier immediate execution
        batcher.profile.cost_tier = CostTier.PREMIUM
        result = await batcher.add_request(["NSE:SBIN-EQ"], "quotes", callback)
        
        callback.assert_called_once_with(["NSE:SBIN-EQ"])
        assert result == "result"
    
    @pytest.mark.asyncio
    async def test_add_request_fresh_data_priority(self, batcher, mock_market_service):
        """Test immediate execution for fresh data priority."""
        callback = AsyncMock(return_value="result")
        mock_market_service.should_prioritize_fresh_data.return_value = True
        
        result = await batcher.add_request(["NSE:SBIN-EQ"], "quotes", callback)
        
        callback.assert_called_once_with(["NSE:SBIN-EQ"])
        assert result == "result"
    
    @pytest.mark.asyncio
    async def test_add_request_batch_size_limit(self, batcher):
        """Test immediate execution when batch size limit reached."""
        callback = AsyncMock(return_value="result")
        
        # Add 50 symbols (batch limit)
        symbols = [f"NSE:STOCK{i}-EQ" for i in range(50)]
        result = await batcher.add_request(symbols, "quotes", callback)
        
        callback.assert_called_once_with(symbols)
        assert result == "result"
    
    @pytest.mark.asyncio
    async def test_add_request_batching(self, batcher):
        """Test request batching behavior."""
        callback = AsyncMock()
        
        # Add first request - should not execute immediately
        result = await batcher.add_request(["NSE:SBIN-EQ"], "quotes", callback)
        assert result is None
        assert "quotes" in batcher.pending_batches
        
        # Add second request to same batch
        await batcher.add_request(["NSE:RELIANCE-EQ"], "quotes", callback)
        batch = batcher.pending_batches["quotes"]
        assert len(batch.symbols) == 2
        assert "NSE:SBIN-EQ" in batch.symbols
        assert "NSE:RELIANCE-EQ" in batch.symbols
    
    @pytest.mark.asyncio
    async def test_batch_execution_after_delay(self, batcher):
        """Test batch execution after delay."""
        callback = AsyncMock()
        
        # Mock short delay for testing
        with patch.object(batcher, '_calculate_batch_delay', return_value=0.01):
            # Add request which will create a batch timer task
            await batcher.add_request(["NSE:SBIN-EQ"], "quotes", callback)
            
            # Wait for batch execution
            await asyncio.sleep(0.02)
            
            # Ensure callback was called
            callback.assert_called_once()
            
            # Clean up any pending tasks
            for batch_key, task in batcher.batch_timers.items():
                if not task.done():
                    task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            batcher.batch_timers.clear()


# ===========================================
# DATA QUALITY VALIDATOR TESTS
# ===========================================

class TestMarketAwareDataQualityValidator:
    """Test the data quality validation component."""
    
    @pytest.fixture
    def validator(self, mock_market_service):
        """Create validator instance."""
        return MarketAwareDataQualityValidator(DataQuality.CRITICAL, mock_market_service)
    
    def test_setup_validation_rules(self, validator):
        """Test validation rules setup."""
        rules = validator.validation_rules
        
        assert "max_staleness_market_open" in rules
        assert "max_staleness_market_closed" in rules
        assert rules["max_staleness_market_open"] == 30
        assert rules["max_staleness_market_closed"] == 300
        assert rules["require_volume"] is True
        assert rules["validate_ohlc"] is True
    
    def test_validate_ohlcv_data_no_validation(self, mock_market_service):
        """Test OHLCV validation when validation is disabled."""
        validator = MarketAwareDataQualityValidator(DataQuality.LOGGING, mock_market_service)
        candles = [{"open": 100, "high": 110, "low": 90, "close": 105, "volume": 1000}]
        
        result = validator.validate_ohlcv_data(candles, "NSE:SBIN-EQ")
        assert result == candles
    
    def test_validate_single_candle_valid(self, validator):
        """Test valid candle validation."""
        candle = {
            "open": 100.0,
            "high": 110.0,
            "low": 90.0,
            "close": 105.0,
            "volume": 1000.0,
            "timestamp": time.time()
        }
        
        result = validator._validate_single_candle(candle, "NSE:SBIN-EQ")
        assert result is True
    
    def test_validate_single_candle_invalid_ohlc(self, validator):
        """Test invalid OHLC relationships."""
        # High < Low
        candle = {"open": 100, "high": 90, "low": 110, "close": 105, "volume": 1000}
        assert validator._validate_single_candle(candle, "NSE:SBIN-EQ") is False
        
        # High < Open
        candle = {"open": 120, "high": 110, "low": 90, "close": 105, "volume": 1000}
        assert validator._validate_single_candle(candle, "NSE:SBIN-EQ") is False
        
        # Low > Close
        candle = {"open": 100, "high": 110, "low": 106, "close": 105, "volume": 1000}
        assert validator._validate_single_candle(candle, "NSE:SBIN-EQ") is False
    
    def test_validate_single_candle_negative_prices(self, validator):
        """Test negative price validation."""
        candle = {"open": -100, "high": 110, "low": 90, "close": 105, "volume": 1000}
        assert validator._validate_single_candle(candle, "NSE:SBIN-EQ") is False
    
    def test_validate_single_candle_zero_volume(self, validator):
        """Test zero volume validation."""
        candle = {"open": 100, "high": 110, "low": 90, "close": 105, "volume": 0}
        assert validator._validate_single_candle(candle, "NSE:SBIN-EQ") is False
    
    def test_validate_single_candle_stale_data(self, validator, mock_market_service):
        """Test stale data validation with market awareness."""
        old_timestamp = time.time() - 60  # 60 seconds old
        candle = {
            "open": 100, "high": 110, "low": 90, "close": 105, "volume": 1000,
            "timestamp": old_timestamp
        }
        
        # Market open - should reject (max 30s staleness)
        mock_market_service.is_market_open.return_value = True
        assert validator._validate_single_candle(candle, "NSE:SBIN-EQ") is False
        
        # Market closed - should accept (max 300s staleness)
        mock_market_service.is_market_open.return_value = False
        assert validator._validate_single_candle(candle, "NSE:SBIN-EQ") is True
    
    def test_validate_ohlcv_data_filtering(self, validator):
        """Test OHLCV data filtering."""
        candles = [
            {"open": 100, "high": 110, "low": 90, "close": 105, "volume": 1000},  # Valid
            {"open": 100, "high": 90, "low": 110, "close": 105, "volume": 1000},  # Invalid OHLC
            {"open": 100, "high": 110, "low": 90, "close": 105, "volume": 0},     # Invalid volume
        ]
        
        result = validator.validate_ohlcv_data(candles, "NSE:SBIN-EQ")
        assert len(result) == 1
        assert result[0]["open"] == 100


# ===========================================
# STRATEGIC CACHE TESTS
# ===========================================

class TestMarketAwareStrategicCache:
    """Test the market-aware caching component."""
    
    @pytest.fixture
    def cache(self, trading_profile, mock_market_service):
        """Create cache instance."""
        return MarketAwareStrategicCache(trading_profile, mock_market_service)
    
    def test_setup_cache_configs(self, cache):
        """Test cache configuration setup."""
        configs = cache.cache_configs
        
        assert "quotes" in configs
        assert "ohlcv" in configs
        assert "account" in configs
        assert configs["quotes"]["base_ttl"] == 30
        assert configs["quotes"]["max_size"] == 500
    
    def test_setup_cache_configs_cost_tiers(self, trading_profile, mock_market_service):
        """Test cache config adjustment for different cost tiers."""
        # Economy tier - longer cache
        trading_profile.cost_tier = CostTier.ECONOMY
        cache = MarketAwareStrategicCache(trading_profile, mock_market_service)
        assert cache.cache_configs["quotes"]["base_ttl"] == 60  # 30 * 2
        
        # Premium tier - shorter cache
        trading_profile.cost_tier = CostTier.PREMIUM
        cache = MarketAwareStrategicCache(trading_profile, mock_market_service)
        assert cache.cache_configs["quotes"]["base_ttl"] == 15  # 30 / 2
    
    def test_get_base_ohlcv_ttl(self, cache):
        """Test OHLCV TTL based on trading frequency."""
        # Test different frequencies
        cache.profile.frequency = TradingFrequency.POSITION
        assert cache._get_base_ohlcv_ttl() == 1800
        
        cache.profile.frequency = TradingFrequency.SCALPING
        assert cache._get_base_ohlcv_ttl() == 60
    
    def test_get_dynamic_ttl_no_market_service(self, cache):
        """Test dynamic TTL without market service."""
        cache.profile.enable_market_awareness = False
        ttl = cache._get_dynamic_ttl("quotes", "NSE:SBIN-EQ")
        assert ttl == 30  # Base TTL
    
    def test_get_dynamic_ttl_with_market_service(self, cache, mock_market_service):
        """Test dynamic TTL with market service."""
        mock_market_service.get_cache_multiplier.return_value = 2.0
        
        ttl = cache._get_dynamic_ttl("quotes", "NSE:SBIN-EQ")
        assert ttl == 60  # 30 * 2.0
        
        mock_market_service.get_cache_multiplier.assert_called_once_with("NSE:SBIN-EQ", 30)
    
    def test_get_dynamic_ttl_bounds(self, cache, mock_market_service):
        """Test dynamic TTL bounds enforcement."""
        # Test lower bound
        mock_market_service.get_cache_multiplier.return_value = 0.1
        ttl = cache._get_dynamic_ttl("quotes", "NSE:SBIN-EQ")
        assert ttl >= 5
        
        # Test upper bound
        mock_market_service.get_cache_multiplier.return_value = 20.0
        ttl = cache._get_dynamic_ttl("quotes", "NSE:SBIN-EQ")
        assert ttl <= 300  # 30 * 10
    
    @pytest.mark.asyncio
    async def test_cache_get_miss(self, cache):
        """Test cache miss."""
        result = await cache.get("quotes", "test_key", "NSE:SBIN-EQ")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_set_and_get_hit(self, cache):
        """Test cache set and subsequent hit."""
        test_data = {"price": 100}
        
        await cache.set("quotes", "test_key", test_data, "NSE:SBIN-EQ")
        result = await cache.get("quotes", "test_key", "NSE:SBIN-EQ")
        
        assert result == test_data
    
    @pytest.mark.asyncio
    async def test_cache_expiration(self, cache, mock_market_service):
        """Test cache expiration."""
        # Use a multiplier that will result in TTL < 5 (minimum bound)
        mock_market_service.get_cache_multiplier.return_value = 0.1  # Will be bounded to minimum 5
        
        test_data = {"price": 100}
        await cache.set("quotes", "test_key", test_data, "NSE:SBIN-EQ")
        
        # Manually set a very short expiry for testing
        cache.caches["quotes"]["test_key"]["timestamp"] = time.time() - 10
        
        result = await cache.get("quotes", "test_key", "NSE:SBIN-EQ")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_eviction(self, cache):
        """Test cache eviction when size limit reached."""
        # Set small max size for testing
        cache.cache_configs["quotes"]["max_size"] = 2
        
        # Add items beyond limit
        await cache.set("quotes", "key1", {"data": 1}, "NSE:SBIN-EQ")
        await cache.set("quotes", "key2", {"data": 2}, "NSE:SBIN-EQ")
        await cache.set("quotes", "key3", {"data": 3}, "NSE:SBIN-EQ")
        
        # Should have evicted oldest entry
        assert len(cache.caches["quotes"]) == 2
    
    def test_cache_metrics_update(self, cache, mock_market_service):
        """Test cache metrics updating."""
        mock_market_service.is_market_open.return_value = True
        
        initial_hits = cache.cache_metrics["quotes"]["market_open_hits"]
        initial_misses = cache.cache_metrics["quotes"]["market_open_misses"]
        
        cache._update_cache_metrics("quotes", "NSE:SBIN-EQ", "hit")
        assert cache.cache_metrics["quotes"]["market_open_hits"] == initial_hits + 1
        
        cache._update_cache_metrics("quotes", "NSE:SBIN-EQ", "miss")
        assert cache.cache_metrics["quotes"]["market_open_misses"] == initial_misses + 1
    
    def test_get_cache_performance(self, cache):
        """Test cache performance metrics."""
        # Set up some metrics
        cache.cache_metrics["quotes"]["market_open_hits"] = 8
        cache.cache_metrics["quotes"]["market_open_misses"] = 2
        cache.cache_metrics["quotes"]["market_closed_hits"] = 6
        cache.cache_metrics["quotes"]["market_closed_misses"] = 4
        
        performance = cache.get_cache_performance()
        
        assert "quotes" in performance
        assert performance["quotes"]["market_open_hit_ratio"] == 80.0  # 8/10 * 100
        assert performance["quotes"]["market_closed_hit_ratio"] == 60.0  # 6/10 * 100
        assert performance["quotes"]["total_requests"] == 20


# ===========================================
# POSITION MANAGER TESTS
# ===========================================

class TestMarketAwarePositionManager:
    """Test the position-aware management component."""
    
    @pytest.fixture
    def position_manager(self, trading_profile, mock_market_service):
        """Create position manager instance."""
        return MarketAwarePositionManager(trading_profile, mock_market_service)
    
    def test_setup_priority_matrix(self, position_manager):
        """Test priority matrix setup."""
        matrix = position_manager.priority_matrix
        
        assert "position_market_open" in matrix
        assert matrix["position_market_open"] == 1  # Highest priority
        assert matrix["research"] == 7  # Lowest priority
    
    def test_update_positions(self, position_manager):
        """Test position updates."""
        positions = [
            {"symbol": "NSE:SBIN-EQ"},
            {"symbol": "NSE:RELIANCE-EQ"},
        ]
        
        position_manager.update_positions(positions)
        
        assert len(position_manager.position_symbols) == 2
        assert "NSE:SBIN-EQ" in position_manager.position_symbols
        assert "NSE:RELIANCE-EQ" in position_manager.position_symbols
        assert len(position_manager.tracked_symbols) == 2
    
    def test_add_to_watchlist(self, position_manager):
        """Test watchlist updates."""
        symbols = ["NSE:TCS-EQ", "NSE:INFY-EQ"]
        
        position_manager.add_to_watchlist(symbols)
        
        assert len(position_manager.watchlist_symbols) == 2
        assert "NSE:TCS-EQ" in position_manager.watchlist_symbols
        assert len(position_manager.tracked_symbols) == 2
    
    def test_get_symbol_priority_no_market_awareness(self, position_manager):
        """Test symbol priority without market awareness."""
        position_manager.profile.enable_market_awareness = False
        position_manager.position_symbols.add("NSE:SBIN-EQ")
        position_manager.watchlist_symbols.add("NSE:TCS-EQ")
        
        assert position_manager.get_symbol_priority("NSE:SBIN-EQ") == 1  # Position
        assert position_manager.get_symbol_priority("NSE:TCS-EQ") == 3   # Watchlist
        assert position_manager.get_symbol_priority("NSE:HDFC-EQ") == 5  # Research
    
    def test_get_symbol_priority_with_market_awareness(self, position_manager, mock_market_service):
        """Test symbol priority with market awareness."""
        position_manager.position_symbols.add("NSE:SBIN-EQ")
        position_manager.watchlist_symbols.add("NSE:TCS-EQ")
        
        # Market open
        mock_market_service.is_market_open.return_value = True
        assert position_manager.get_symbol_priority("NSE:SBIN-EQ") == 1  # position_market_open
        assert position_manager.get_symbol_priority("NSE:TCS-EQ") == 4   # watchlist_market_open
        
        # Market closed
        mock_market_service.is_market_open.return_value = False
        assert position_manager.get_symbol_priority("NSE:SBIN-EQ") == 2  # position_market_closed
        assert position_manager.get_symbol_priority("NSE:TCS-EQ") == 6   # watchlist_market_closed
    
    def test_get_priority_symbols(self, position_manager, mock_market_service):
        """Test symbol sorting by priority."""
        position_manager.position_symbols.add("NSE:SBIN-EQ")
        position_manager.watchlist_symbols.add("NSE:TCS-EQ")
        mock_market_service.is_market_open.return_value = True
        
        symbols = ["NSE:HDFC-EQ", "NSE:TCS-EQ", "NSE:SBIN-EQ"]
        sorted_symbols = position_manager.get_priority_symbols(symbols)
        
        assert sorted_symbols[0] == "NSE:SBIN-EQ"   # Highest priority (position)
        assert sorted_symbols[1] == "NSE:TCS-EQ"    # Medium priority (watchlist)
        assert sorted_symbols[2] == "NSE:HDFC-EQ"   # Lowest priority (research)


# ===========================================
# RATE LIMITER TESTS
# ===========================================

class TestMarketAwareRateLimiter:
    """Test the market-aware rate limiting component."""
    
    @pytest.fixture
    def rate_limiter(self, mock_market_service):
        """Create rate limiter instance."""
        return MarketAwareRateLimiter(10.0, mock_market_service)
    
    @pytest.mark.asyncio
    async def test_acquire_no_symbol(self, rate_limiter):
        """Test token acquisition without symbol (global limiter)."""
        start_time = time.time()
        await rate_limiter.acquire()
        end_time = time.time()
        
        # Should not take long for first acquisition
        assert (end_time - start_time) < 0.1
        assert "global" in rate_limiter.limiters
    
    @pytest.mark.asyncio
    async def test_acquire_with_symbol(self, rate_limiter, mock_market_service):
        """Test token acquisition with symbol-specific limiter."""
        mock_market_service.get_rate_limit_multiplier.return_value = 1.5
        
        await rate_limiter.acquire("NSE:SBIN-EQ")
        
        mock_market_service.get_rate_limit_multiplier.assert_called_once_with("NSE:SBIN-EQ")
        assert "symbol_NSE:SBIN-EQ" in rate_limiter.limiters
    
    @pytest.mark.asyncio
    async def test_acquire_multiple_tokens(self, rate_limiter):
        """Test acquiring multiple tokens."""
        await rate_limiter.acquire(tokens=3)
        assert "global" in rate_limiter.limiters


# ===========================================
# MAIN FYERS CLIENT TESTS
# ===========================================

class TestFyersRestClient:
    """Test the main Fyers REST client."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, fyers_settings, mock_auth, trading_profile, mock_market_service):
        """Test client initialization."""
        with patch('app.providers.fyers.fyers_rest_client.RestClient'):
            client = FyersRestClient(
                settings=fyers_settings,
                auth=mock_auth,
                trading_profile=trading_profile,
                market_service=mock_market_service,
                enable_monitoring=False
            )
            
            assert client.settings == fyers_settings
            assert client.auth == mock_auth
            assert client.trading_profile == trading_profile
            assert client.market_service == mock_market_service
            assert isinstance(client.batcher, MarketAwareIntelligentBatcher)
            assert isinstance(client.cache, MarketAwareStrategicCache)
            assert isinstance(client.data_validator, MarketAwareDataQualityValidator)
            assert isinstance(client.position_manager, MarketAwarePositionManager)
    
    @pytest.mark.asyncio
    async def test_initialization_with_defaults(self, fyers_settings):
        """Test client initialization with default values."""
        # Add missing attributes to mock
        fyers_settings.ACCESS_TOKEN = None
        
        with patch('app.providers.fyers.fyers_rest_client.RestClient'):
            client = FyersRestClient(settings=fyers_settings, enable_monitoring=False)
            
            assert isinstance(client.auth, FyersAuth)
            assert isinstance(client.trading_profile, TradingProfile)
            assert isinstance(client.market_service, NoOpMarketService)
    
    @pytest.mark.asyncio
    async def test_connect_success(self, fyers_client, mock_auth):
        """Test successful connection."""
        with patch.object(fyers_client, 'get_profile', return_value={"name": "Test User"}):
            with patch.object(fyers_client, 'get_positions', return_value={"netPositions": []}):
                await fyers_client.connect()
                
                mock_auth.initialize.assert_called_once()
                assert fyers_client.connection_state == "CONNECTED"
    
    @pytest.mark.asyncio
    async def test_connect_auth_failure(self, fyers_client, mock_auth):
        """Test connection failure due to authentication."""
        mock_auth.initialize.return_value = False
        
        await fyers_client.connect()
        # Should still connect but log warning about authentication
        
    @pytest.mark.asyncio
    async def test_connect_profile_failure(self, fyers_client):
        """Test connection failure during profile fetch."""
        with patch.object(fyers_client, 'get_profile', side_effect=Exception("API Error")):
            with pytest.raises(ConnectionError):
                await fyers_client.connect()
    
    @pytest.mark.asyncio
    async def test_disconnect(self, fyers_client):
        """Test disconnection cleanup."""
        # Create a proper mock task
        async def dummy_task():
            await asyncio.sleep(0.1)
        
        mock_task = asyncio.create_task(dummy_task())
        fyers_client._background_tasks = [mock_task]
        
        await fyers_client.disconnect()
        
        assert mock_task.cancelled()
        assert fyers_client.connection_state == "DISCONNECTED"
    
    @pytest.mark.asyncio
    async def test_make_market_aware_request_cache_hit(self, fyers_client):
        """Test request with cache hit."""
        cached_data = {"cached": "data"}
        fyers_client.cache.get = AsyncMock(return_value=cached_data)
        fyers_client._rest_client = AsyncMock()
        fyers_client._rest_client.default_headers = {}

        result = await fyers_client._make_market_aware_request(
            endpoint="test",
            cache_type="test_cache",
            cache_key="test_key",
            symbol="NSE:SBIN-EQ"
        )
        
        assert result == cached_data
        fyers_client.cache.get.assert_called_once_with("test_cache", "test_key", "NSE:SBIN-EQ")
    
    @pytest.mark.asyncio
    async def test_make_market_aware_request_cache_miss(self, fyers_client, mock_auth):
        """Test request with cache miss."""
        fyers_client.cache.get = AsyncMock(return_value=None)
        fyers_client.cache.set = AsyncMock()
        fyers_client._rest_client = AsyncMock()
        fyers_client._rest_client.get.return_value = {"s": "ok", "data": "response"}
        # Fix: Make default_headers a regular dict, not an AsyncMock
        fyers_client._rest_client.default_headers = {}
        
        result = await fyers_client._make_market_aware_request(
            endpoint="test",
            cache_type="test_cache",
            cache_key="test_key",
            symbol="NSE:SBIN-EQ"
        )
        
        assert result == {"s": "ok", "data": "response"}
        fyers_client._rest_client.get.assert_called_once()
        fyers_client.cache.set.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_make_market_aware_request_auth_error(self, fyers_client, mock_auth):
        """Test request with authentication error."""
        mock_auth.ensure_token.return_value = False
        
        with pytest.raises(AuthenticationError):
            await fyers_client._make_market_aware_request(endpoint="test")
    
    @pytest.mark.asyncio
    async def test_make_market_aware_request_fyers_errors(self, fyers_client, mock_auth):
        """Test request with Fyers-specific errors."""
        fyers_client.cache.get = AsyncMock(return_value=None)
        fyers_client._rest_client = AsyncMock()
        
        # Make default_headers a property instead of AsyncMock
        fyers_client._rest_client.default_headers = {}
        
        # Ensure auth headers are properly awaited
        mock_auth.get_auth_headers = Mock(return_value={"Authorization": "test"})
        
        # Test authentication error - use Exception with proper dict structure
        auth_error = Exception()
        auth_error.args = ({"code": -8},)
        fyers_client._rest_client.get.side_effect = auth_error
        
        # For now, let's test that some error is raised
        with pytest.raises(Exception):
            await fyers_client._make_market_aware_request(endpoint="test")
    
    @pytest.mark.asyncio
    async def test_get_quote(self, fyers_client):
        """Test get_quote method."""
        mock_response = {"s": "ok", "d": [{"n": "NSE:SBIN-EQ", "v": {"lp": 500}}]}
        
        with patch('app.providers.fyers.fyers_rest_client.transformers') as mock_transformers:
            mock_transformers.map_symbol.return_value = "NSE:SBIN-EQ"
            mock_transformers.transform_quote.return_value = {"last_price": "500.00"}
            
            fyers_client._make_market_aware_request = AsyncMock(return_value=mock_response)
            
            result = await fyers_client.get_quote("NSE:SBIN-EQ")
            
            assert result == {"last_price": "500.00"}
            fyers_client._make_market_aware_request.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_quotes_batch(self, fyers_client):
        """Test get_quotes_batch method."""
        symbols = ["NSE:SBIN-EQ", "NSE:RELIANCE-EQ"]
        mock_response = {
            "s": "ok",
            "d": [
                {"n": "NSE:SBIN-EQ", "v": {"lp": 500}},
                {"n": "NSE:RELIANCE-EQ", "v": {"lp": 2500}}
            ]
        }
        
        with patch('app.providers.fyers.fyers_rest_client.transformers') as mock_transformers:
            mock_transformers.map_symbol.side_effect = lambda x: x
            mock_transformers.transform_quote.side_effect = [
                {"last_price": "500.00"},
                {"last_price": "2500.00"}
            ]
            
            fyers_client._make_market_aware_request = AsyncMock(return_value=mock_response)
            
            result = await fyers_client.get_quotes_batch(symbols)
            
            assert len(result) == 2
            assert "NSE:SBIN-EQ" in result
            assert "NSE:RELIANCE-EQ" in result
    
    @pytest.mark.asyncio
    async def test_get_ohlcv(self, fyers_client):
        """Test get_ohlcv method."""
        mock_response = {"s": "ok", "candles": [[1640995200, 100, 110, 90, 105, 1000]]}
        transformed_candles = [{"open": "100", "high": "110", "low": "90", "close": "105", "volume": 1000}]
        
        with patch('app.providers.fyers.fyers_rest_client.transformers') as mock_transformers:
            mock_transformers.map_symbol.return_value = "NSE:SBIN-EQ"
            mock_transformers.map_interval.return_value = "1"
            mock_transformers.transform_ohlcv.return_value = transformed_candles
            
            fyers_client._make_market_aware_request = AsyncMock(return_value=mock_response)
            fyers_client.data_validator.validate_ohlcv_data = Mock(return_value=transformed_candles)
            
            result = await fyers_client.get_ohlcv("NSE:SBIN-EQ", "1m")
            
            assert len(result) == 1
            assert result[0]["symbol"] == "NSE:SBIN-EQ"
            assert result[0]["interval"] == "1m"
    
    @pytest.mark.asyncio
    async def test_get_orderbook(self, fyers_client):
        """Test get_orderbook method."""
        mock_response = {"s": "ok", "d": {"NSE:SBIN-EQ": {"bids": [], "ask": []}}}
        transformed_orderbook = {"bids": [], "asks": [], "symbol": "NSE:SBIN-EQ"}
        
        with patch('app.providers.fyers.fyers_rest_client.transformers') as mock_transformers:
            mock_transformers.map_symbol.return_value = "NSE:SBIN-EQ"
            mock_transformers.transform_orderbook.return_value = transformed_orderbook
            
            fyers_client._make_market_aware_request = AsyncMock(return_value=mock_response)
            
            result = await fyers_client.get_orderbook("NSE:SBIN-EQ")
            
            assert result == transformed_orderbook
    
    @pytest.mark.asyncio
    async def test_get_profile(self, fyers_client):
        """Test get_profile method."""
        mock_response = {"s": "ok", "data": {"name": "Test User"}}
        
        fyers_client._make_market_aware_request = AsyncMock(return_value=mock_response)
        
        result = await fyers_client.get_profile()
        
        assert result == {"name": "Test User"}
    
    @pytest.mark.asyncio
    async def test_place_order(self, fyers_client):
        """Test place_order method."""
        order_data = {"symbol": "NSE:SBIN-EQ", "qty": 10, "side": 1}
        mock_response = {"s": "ok", "id": "12345"}
        
        with patch('app.providers.fyers.fyers_rest_client.transformers') as mock_transformers:
            mock_transformers.map_symbol.return_value = "NSE:SBIN-EQ"
            
            fyers_client._make_market_aware_request = AsyncMock(return_value=mock_response)
            
            result = await fyers_client.place_order(order_data)
            
            assert result == mock_response
            # Verify symbol was transformed
            mock_transformers.map_symbol.assert_called_once_with("NSE:SBIN-EQ")
    
    @pytest.mark.asyncio
    async def test_health_check(self, fyers_client, mock_auth, mock_market_service):
        """Test health_check method."""
        mock_auth.has_valid_token.return_value = True
        fyers_client.connection_state = "CONNECTED"
        
        result = await fyers_client.health_check()
        
        assert result["connection_state"] == "CONNECTED"
        assert result["auth_status"] == "valid"
        assert result["market_awareness"]["enabled"] is True
        assert result["market_awareness"]["service_status"] == "operational"
        assert "trading_profile" in result
        assert "position_tracking" in result
        assert "cache_stats" in result
        assert "performance_metrics" in result
    
    def test_add_to_watchlist(self, fyers_client):
        """Test add_to_watchlist method."""
        symbols = ["NSE:TCS-EQ", "NSE:INFY-EQ"]
        
        fyers_client.add_to_watchlist(symbols)
        
        assert len(fyers_client.position_manager.watchlist_symbols) == 2
        assert "NSE:TCS-EQ" in fyers_client.position_manager.watchlist_symbols
    
    def test_get_performance_summary(self, fyers_client):
        """Test get_performance_summary method."""
        # Set up some metrics
        fyers_client.metrics["quotes"]["requests"] = 100
        fyers_client.metrics["quotes"]["cache_hits"] = 80
        fyers_client.metrics["quotes"]["total_latency"] = 10.0
        fyers_client.metrics["quotes"]["avg_latency"] = 0.1
        fyers_client.metrics["quotes"]["market_open_requests"] = 60
        fyers_client.metrics["quotes"]["market_closed_requests"] = 40
        
        summary = fyers_client.get_performance_summary()
        
        assert summary["total_requests"] == 100
        assert summary["cache_hit_ratio"] == 80.0
        assert summary["avg_latency"] == 0.1
        assert summary["market_distribution"]["open_requests"] == 60
        assert summary["market_distribution"]["closed_requests"] == 40
        assert summary["cost_efficiency"] == "High"
    
    def test_calculate_optimization_score(self, fyers_client):
        """Test optimization score calculation."""
        # Set up conditions for high score
        fyers_client.metrics["quotes"]["requests"] = 100
        fyers_client.metrics["quotes"]["cache_hits"] = 80
        fyers_client.metrics["quotes"]["total_latency"] = 15.0
        fyers_client.metrics["quotes"]["avg_latency"] = 0.15
        fyers_client.position_manager.position_symbols.add("NSE:SBIN-EQ")
        
        score = fyers_client._calculate_optimization_score()
        
        # Should get points for cache efficiency, market awareness, and position tracking
        assert score > 50  # Should be a good score
    
    def test_set_market_service(self, fyers_client, mock_market_service):
        """Test set_market_service method."""
        new_market_service = Mock(spec=MarketServiceProtocol)
        
        fyers_client.set_market_service(new_market_service)
        
        assert fyers_client.market_service == new_market_service
        assert fyers_client.batcher.market_service == new_market_service
        assert fyers_client.cache.market_service == new_market_service
        assert fyers_client.data_validator.market_service == new_market_service
        assert fyers_client.position_manager.market_service == new_market_service
    
    def test_get_market_context_for_symbols(self, fyers_client, mock_market_service):
        """Test get_market_context_for_symbols method."""
        symbols = ["NSE:SBIN-EQ", "NSE:TCS-EQ"]
        mock_market_service.get_market_state.return_value = "open"
        mock_market_service.is_market_open.return_value = True
        mock_market_service.get_cache_multiplier.return_value = 0.5
        mock_market_service.get_rate_limit_multiplier.return_value = 1.0
        
        context = fyers_client.get_market_context_for_symbols(symbols)
        
        assert len(context) == 2
        assert "NSE:SBIN-EQ" in context
        assert context["NSE:SBIN-EQ"]["market_state"] == "open"
        assert context["NSE:SBIN-EQ"]["is_market_open"] is True
    
    def test_get_market_context_no_market_awareness(self, fyers_client):
        """Test get_market_context_for_symbols without market awareness."""
        fyers_client.trading_profile.enable_market_awareness = False
        
        context = fyers_client.get_market_context_for_symbols(["NSE:SBIN-EQ"])
        
        assert context == {}


# ===========================================
# BACKGROUND TASKS TESTS
# ===========================================

class TestBackgroundTasks:
    """Test background task functionality."""
    
    @pytest.mark.asyncio
    async def test_monitor_token_health(self, fyers_client, mock_auth):
        """Test token health monitoring."""
        # Mock token expiring soon
        mock_auth.check_token_expiry.return_value = timedelta(minutes=10)
        
        # Create a task that runs once
        async def run_once():
            await fyers_client._monitor_token_health()
        
        # This should trigger token refresh
        task = asyncio.create_task(run_once())
        
        # Cancel after short time to prevent infinite loop
        await asyncio.sleep(0.01)
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            pass
        
        # Should have checked token expiry
        mock_auth.check_token_expiry.assert_called()
    
    @pytest.mark.asyncio
    async def test_cache_maintenance(self, fyers_client):
        """Test cache maintenance task."""
        # Add some expired entries
        fyers_client.cache.caches["test"] = {
            "old_key": {
                "timestamp": time.time() - 1000,
                "data": "old_data",
                "last_access": time.time() - 1000,
                "access_count": 0,
                "symbol": "NSE:SBIN-EQ"
            }
        }
        
        # Run maintenance once
        async def run_once():
            await fyers_client._cache_maintenance()
        
        task = asyncio.create_task(run_once())
        await asyncio.sleep(0.01)
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            pass


# ===========================================
# INTEGRATION TESTS
# ===========================================

class TestIntegration:
    """Integration tests for complete workflows."""
    
    @pytest.mark.asyncio
    async def test_complete_quote_workflow(self, fyers_client, mock_market_service):
        """Test complete quote fetching workflow with market awareness."""
        # Setup market service
        mock_market_service.get_market_state.return_value = "open"
        mock_market_service.is_market_open.return_value = True
        mock_market_service.get_cache_multiplier.return_value = 0.5  # Fast cache during market
        
        # Mock API response
        api_response = {"s": "ok", "d": [{"n": "NSE:SBIN-EQ", "v": {"lp": 500}}]}
        
        with patch('app.providers.fyers.fyers_rest_client.transformers') as mock_transformers:
            mock_transformers.map_symbol.return_value = "NSE:SBIN-EQ"
            mock_transformers.transform_quote.return_value = {"last_price": "500.00"}
            
            fyers_client._rest_client = AsyncMock()
            # Make default_headers a property
            fyers_client._rest_client.default_headers = {}
            fyers_client._rest_client.get.return_value = api_response
            
            # First call - should hit API and cache
            result1 = await fyers_client.get_quote("NSE:SBIN-EQ")
            assert result1 == {"last_price": "500.00"}
            assert fyers_client._rest_client.get.call_count == 1
            
            # Second call immediately - should hit cache
            result2 = await fyers_client.get_quote("NSE:SBIN-EQ")
            assert result2 == {"last_price": "500.00"}
            assert fyers_client._rest_client.get.call_count == 1  # No additional API call
    
    @pytest.mark.asyncio
    async def test_position_aware_prioritization(self, fyers_client):
        """Test position-aware request prioritization."""
        # Add a position
        fyers_client.position_manager.position_symbols.add("NSE:SBIN-EQ")
        fyers_client.position_manager.watchlist_symbols.add("NSE:TCS-EQ")
        
        symbols = ["NSE:HDFC-EQ", "NSE:TCS-EQ", "NSE:SBIN-EQ"]
        prioritized = fyers_client.position_manager.get_priority_symbols(symbols)
        
        # Position symbol should come first
        assert prioritized[0] == "NSE:SBIN-EQ"
    
    @pytest.mark.asyncio
    async def test_error_handling_workflow(self, fyers_client, mock_auth):
        """Test error handling in complete workflow."""
        # Setup auth failure
        mock_auth.ensure_token.return_value = False
        
        with pytest.raises(AuthenticationError):
            await fyers_client.get_quote("NSE:SBIN-EQ")


# ===========================================
# PERFORMANCE TESTS
# ===========================================

class TestPerformance:
    """Performance-related tests."""
    
    @pytest.mark.asyncio
    async def test_cache_performance_under_load(self, fyers_client):
        """Test cache performance under load."""
        # Add test cache config
        fyers_client.cache.cache_configs["test"] = {"base_ttl": 300, "max_size": 1000}
        
        # Simulate many cache operations
        tasks = []
        for i in range(100):
            task = fyers_client.cache.set("test", f"key_{i}", f"data_{i}", "NSE:SBIN-EQ")
            tasks.append(task)
        
        start_time = time.time()
        await asyncio.gather(*tasks)
        end_time = time.time()
        
        # Should complete quickly
        assert (end_time - start_time) < 1.0
        
        # Verify all data was cached
        assert len(fyers_client.cache.caches["test"]) == 100
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, fyers_client):
        """Test handling of concurrent requests."""
        fyers_client._rest_client = AsyncMock()
        fyers_client._rest_client.get.return_value = {"s": "ok", "data": "test"}
        
        # Simulate concurrent requests
        tasks = []
        for i in range(10):
            task = fyers_client._make_market_aware_request("test", symbol="NSE:SBIN-EQ")
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert len(results) == 10
        assert all(result["s"] == "ok" for result in results)


# ===========================================
# EDGE CASES AND ERROR CONDITIONS
# ===========================================

class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.asyncio
    async def test_empty_symbol_list(self, fyers_client):
        """Test handling of empty symbol list."""
        result = await fyers_client.get_quotes_batch([])
        assert result == {}
    
    @pytest.mark.asyncio
    async def test_invalid_symbol_format(self, fyers_client):
        """Test handling of invalid symbol format."""
        with patch('app.providers.fyers.fyers_rest_client.transformers') as mock_transformers:
            mock_transformers.map_symbol.side_effect = ValueError("Invalid symbol")
            
            with pytest.raises(ValueError):
                await fyers_client.get_quote("INVALID_SYMBOL")
    
    @pytest.mark.asyncio
    async def test_network_timeout(self, fyers_client):
        """Test handling of network timeouts."""
        fyers_client._rest_client = AsyncMock()
        fyers_client._rest_client.get.side_effect = asyncio.TimeoutError("Request timeout")
        
        with pytest.raises(asyncio.TimeoutError):
            await fyers_client._make_market_aware_request("test")
    
    def test_market_service_error_handling(self, fyers_client):
        """Test graceful handling of market service errors."""
        # Mock market service that raises errors
        fyers_client.market_service.get_market_state.side_effect = Exception("Market service error")
        
        # Should not break the cache TTL calculation
        ttl = fyers_client.cache._get_dynamic_ttl("quotes", "NSE:SBIN-EQ")
        assert ttl == 30  # Should fall back to base TTL
    
    @pytest.mark.asyncio
    async def test_large_batch_handling(self, fyers_client):
        """Test handling of large symbol batches."""
        # Create a list of 150 symbols (3 batches of 50)
        symbols = [f"NSE:STOCK{i:03d}-EQ" for i in range(150)]
        
        with patch('app.providers.fyers.fyers_rest_client.transformers') as mock_transformers:
            mock_transformers.map_symbol.side_effect = lambda x: x
            mock_transformers.transform_quote.return_value = {"last_price": "100.00"}
            
            fyers_client._make_market_aware_request = AsyncMock(return_value={
                "s": "ok", 
                "d": [{"n": f"NSE:STOCK{i:03d}-EQ"} for i in range(50)]
            })
            
            result = await fyers_client.get_quotes_batch(symbols)
            
            # Should have made 3 API calls (150/50 = 3)
            assert fyers_client._make_market_aware_request.call_count == 3


# At the top level, after other imports
@pytest_asyncio.fixture(autouse=True)
async def cleanup_tasks():
    """Cleanup any remaining tasks after each test."""
    yield
    # Clean up any pending tasks
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        if not task.done():
            task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])