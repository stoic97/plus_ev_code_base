"""
Market-aware strategic cache implementation.

This module provides a caching mechanism that is aware of market conditions
and can adjust its behavior accordingly.
"""

import logging
from typing import Any, Dict, Optional, List
from datetime import datetime, timedelta

from .provider import MarketServiceProtocol

logger = logging.getLogger(__name__)

class MarketAwareStrategicCache:
    """
    A market-aware strategic cache implementation.
    
    This cache adjusts its behavior based on market conditions and data freshness requirements.
    """
    
    def __init__(self, market_service: Optional[MarketServiceProtocol] = None):
        """
        Initialize the cache.
        
        Args:
            market_service: Optional market service for market awareness
        """
        self.market_service = market_service
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.metrics = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_requests": 0
        }
        
        # Configure cache settings
        self._setup_cache_configs()
    
    def _setup_cache_configs(self) -> None:
        """Set up cache configuration."""
        # Base TTL values in seconds
        self.ttl_configs = {
            "quotes": 1,  # 1 second for quotes
            "trades": 5,  # 5 seconds for trades
            "orderbook": 2,  # 2 seconds for orderbook
            "ohlcv": {
                "1m": 60,  # 1 minute for 1m candles
                "5m": 300,  # 5 minutes for 5m candles
                "15m": 900,  # 15 minutes for 15m candles
                "1h": 3600,  # 1 hour for 1h candles
                "1d": 86400  # 1 day for daily candles
            }
        }
        
        # Cost tiers for different data types
        self.cost_tiers = {
            "quotes": 1,
            "trades": 2,
            "orderbook": 3,
            "ohlcv": 4
        }
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        self.metrics["total_requests"] += 1
        
        if key not in self.cache:
            self.metrics["misses"] += 1
            return None
        
        entry = self.cache[key]
        if self._is_entry_expired(entry):
            self.metrics["misses"] += 1
            del self.cache[key]
            return None
        
        self.metrics["hits"] += 1
        return entry["value"]
    
    def set(self, key: str, value: Any, data_type: str, interval: Optional[str] = None) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            data_type: Type of data (quotes, trades, orderbook, ohlcv)
            interval: Optional interval for OHLCV data
        """
        ttl = self._get_dynamic_ttl(data_type, interval)
        
        self.cache[key] = {
            "value": value,
            "timestamp": datetime.now(),
            "ttl": ttl,
            "data_type": data_type,
            "interval": interval
        }
        
        # Perform cache maintenance
        self._maintain_cache()
    
    def _get_dynamic_ttl(self, data_type: str, interval: Optional[str] = None) -> float:
        """
        Get dynamic TTL based on market conditions.
        
        Args:
            data_type: Type of data
            interval: Optional interval for OHLCV data
            
        Returns:
            TTL in seconds
        """
        # Get base TTL
        if data_type == "ohlcv" and interval:
            base_ttl = self.ttl_configs["ohlcv"].get(interval, 60)
        else:
            base_ttl = self.ttl_configs.get(data_type, 60)
        
        # Adjust TTL based on market conditions
        if self.market_service:
            if not self.market_service.is_market_open():
                # Extend TTL when market is closed
                return base_ttl * 5
            
            market_state = self.market_service.get_market_state()
            if market_state in ["pre_market", "post_market"]:
                # Reduce TTL during pre/post market
                return max(1, base_ttl * 0.5)
        
        return base_ttl
    
    def _is_entry_expired(self, entry: Dict[str, Any]) -> bool:
        """
        Check if a cache entry is expired.
        
        Args:
            entry: Cache entry to check
            
        Returns:
            True if expired, False otherwise
        """
        age = (datetime.now() - entry["timestamp"]).total_seconds()
        return age > entry["ttl"]
    
    def _maintain_cache(self) -> None:
        """Perform cache maintenance."""
        # Remove expired entries
        expired = [k for k, v in self.cache.items() if self._is_entry_expired(v)]
        for key in expired:
            del self.cache[key]
            self.metrics["evictions"] += 1
    
    def get_cache_performance(self) -> Dict[str, float]:
        """
        Get cache performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        total_requests = self.metrics["total_requests"]
        if total_requests == 0:
            return {
                "hit_ratio": 0.0,
                "eviction_rate": 0.0,
                "total_entries": len(self.cache)
            }
        
        return {
            "hit_ratio": self.metrics["hits"] / total_requests,
            "eviction_rate": self.metrics["evictions"] / total_requests,
            "total_entries": len(self.cache)
        } 