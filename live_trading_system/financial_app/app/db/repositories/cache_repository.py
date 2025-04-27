"""
Cache repository for market data.

This module provides a repository layer for caching market data,
reducing database load and improving performance for frequently
accessed data.
"""

import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

from app.core.database import RedisDB, get_redis_db

# Set up logging
logger = logging.getLogger(__name__)


class CacheRepository:
    """
    Repository for caching market data.
    
    Provides methods for storing and retrieving market data from cache,
    with automatic JSON serialization/deserialization and TTL management.
    """
    
    def __init__(self, redis_db: Optional[RedisDB] = None):
        """
        Initialize a new cache repository.
        
        Args:
            redis_db: Redis database instance (optional)
        """
        self.redis = redis_db or get_redis_db()
        
        # Default TTLs
        self.default_ttl = 3600  # 1 hour
        self.ohlcv_ttl = 300     # 5 minutes
        self.tick_ttl = 60       # 1 minute
        self.orderbook_ttl = 10  # 10 seconds
    
    def _get_key(self, key_type: str, *parts: str) -> str:
        """
        Generate a cache key with namespace.
        
        Args:
            key_type: Type of data (ohlcv, tick, orderbook)
            *parts: Additional parts to include in the key
            
        Returns:
            Formatted cache key
        """
        return f"market:{key_type}:{':'.join(parts)}"
    
    def cache_latest_ohlcv(self, symbol: str, interval: str, data: Dict[str, Any]) -> bool:
        """
        Cache the latest OHLCV data for a symbol and interval.
        
        Args:
            symbol: Instrument symbol
            interval: Time interval
            data: OHLCV data to cache
            
        Returns:
            True if successful, False otherwise
        """
        key = self._get_key("ohlcv", "latest", symbol, interval)
        
        try:
            return self.redis.set_json(key, data, expiration=self.ohlcv_ttl)
        except Exception as e:
            logger.warning(f"Failed to cache latest OHLCV for {symbol}/{interval}: {e}")
            return False
    
    def get_latest_ohlcv(self, symbol: str, interval: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest cached OHLCV data for a symbol and interval.
        
        Args:
            symbol: Instrument symbol
            interval: Time interval
            
        Returns:
            Cached OHLCV data or None if not found
        """
        key = self._get_key("ohlcv", "latest", symbol, interval)
        
        try:
            return self.redis.get_json(key)
        except Exception as e:
            logger.warning(f"Failed to get latest OHLCV for {symbol}/{interval} from cache: {e}")
            return None
    
    def cache_latest_orderbook(self, symbol: str, data: Dict[str, Any]) -> bool:
        """
        Cache the latest order book data for a symbol.
        
        Args:
            symbol: Instrument symbol
            data: Order book data to cache
            
        Returns:
            True if successful, False otherwise
        """
        key = self._get_key("orderbook", "latest", symbol)
        
        try:
            return self.redis.set_json(key, data, expiration=self.orderbook_ttl)
        except Exception as e:
            logger.warning(f"Failed to cache latest orderbook for {symbol}: {e}")
            return False
    
    def get_latest_orderbook(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest cached order book data for a symbol.
        
        Args:
            symbol: Instrument symbol
            
        Returns:
            Cached order book data or None if not found
        """
        key = self._get_key("orderbook", "latest", symbol)
        
        try:
            return self.redis.get_json(key)
        except Exception as e:
            logger.warning(f"Failed to get latest orderbook for {symbol} from cache: {e}")
            return None
    
    def cache_ticks(self, symbol: str, start_time: datetime, end_time: datetime, data: List[Dict[str, Any]]) -> bool:
        """
        Cache tick data for a symbol and time range.
        
        Args:
            symbol: Instrument symbol
            start_time: Start of time range
            end_time: End of time range
            data: Tick data to cache
            
        Returns:
            True if successful, False otherwise
        """
        # Format timestamps for key
        start_str = start_time.strftime("%Y%m%d%H%M%S")
        end_str = end_time.strftime("%Y%m%d%H%M%S")
        
        key = self._get_key("ticks", symbol, start_str, end_str)
        
        try:
            return self.redis.set_json(key, data, expiration=self.tick_ttl)
        except Exception as e:
            logger.warning(f"Failed to cache ticks for {symbol}: {e}")
            return False
    
    def get_ticks(self, symbol: str, start_time: datetime, end_time: datetime) -> Optional[List[Dict[str, Any]]]:
        """
        Get cached tick data for a symbol and time range.
        
        Args:
            symbol: Instrument symbol
            start_time: Start of time range
            end_time: End of time range
            
        Returns:
            Cached tick data or None if not found
        """
        # Format timestamps for key
        start_str = start_time.strftime("%Y%m%d%H%M%S")
        end_str = end_time.strftime("%Y%m%d%H%M%S")
        
        key = self._get_key("ticks", symbol, start_str, end_str)
        
        try:
            return self.redis.get_json(key)
        except Exception as e:
            logger.warning(f"Failed to get ticks for {symbol} from cache: {e}")
            return None
    
    def invalidate_symbol_cache(self, symbol: str) -> None:
        """
        Invalidate all cached data for a symbol.
        
        Args:
            symbol: Instrument symbol
        """
        pattern = f"market:*:{symbol}:*"
        
        try:
            # Get all keys matching the pattern
            cursor = 0
            while True:
                cursor, keys = self.redis.client.scan(cursor, match=pattern, count=100)
                
                # Delete all found keys
                if keys:
                    self.redis.client.delete(*keys)
                
                # Break if no more keys
                if cursor == 0:
                    break
                    
            logger.debug(f"Invalidated cache for symbol {symbol}")
            
        except Exception as e:
            logger.warning(f"Failed to invalidate cache for symbol {symbol}: {e}")
    
    def cache_instrument(self, symbol: str, data: Dict[str, Any]) -> bool:
        """
        Cache instrument data.
        
        Args:
            symbol: Instrument symbol
            data: Instrument data to cache
            
        Returns:
            True if successful, False otherwise
        """
        key = self._get_key("instrument", symbol)
        
        try:
            # Longer TTL for instrument data which changes less frequently
            return self.redis.set_json(key, data, expiration=86400)  # 24 hours
        except Exception as e:
            logger.warning(f"Failed to cache instrument for {symbol}: {e}")
            return False
    
    def get_instrument(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get cached instrument data.
        
        Args:
            symbol: Instrument symbol
            
        Returns:
            Cached instrument data or None if not found
        """
        key = self._get_key("instrument", symbol)
        
        try:
            return self.redis.get_json(key)
        except Exception as e:
            logger.warning(f"Failed to get instrument for {symbol} from cache: {e}")
            return None