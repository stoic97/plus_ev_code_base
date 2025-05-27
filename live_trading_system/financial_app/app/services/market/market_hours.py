"""
Market hours configuration and management.

This module handles the configuration of market hours for different exchanges and market types,
including regular hours, pre/post market hours, and holidays.
"""

from datetime import date, time
from typing import Dict, List, Optional, Set
from zoneinfo import ZoneInfo
from enum import Enum, auto

class MarketType(str, Enum):
    """Types of markets available."""
    EQUITY = "equity"
    COMMODITY = "commodity"
    FOREX = "forex"
    CRYPTO = "crypto"
    DERIVATIVES = "derivatives"

class MarketHours:
    """
    Configuration for market hours of an exchange.
    
    This class manages:
    - Regular market hours for different market types
    - Pre-market hours
    - Post-market hours
    - Holidays and special trading days
    """
    
    def __init__(
        self,
        timezone: ZoneInfo,
        market_types: Dict[MarketType, Dict[str, Dict[str, time]]],
        holidays: Optional[Dict[MarketType, List[date]]] = None
    ):
        """
        Initialize market hours configuration.
        
        Args:
            timezone: The timezone for this exchange
            market_types: Dict mapping MarketType to their hours configuration
                        Each market type has a dict with 'regular_market', 'pre_market', 'post_market'
                        Each of these has 'start' and 'end' times
            holidays: Optional dict mapping MarketType to their holiday lists
        """
        self.timezone = timezone
        self.market_types = market_types
        self._holidays: Dict[MarketType, Set[date]] = {
            market_type: set(holidays.get(market_type, []) if holidays else [])
            for market_type in MarketType
        }
    
    def add_holiday(self, holiday_date: date, market_type: Optional[MarketType] = None) -> None:
        """
        Add a holiday to the calendar.
        
        Args:
            holiday_date: The date to add as a holiday
            market_type: Optional market type. If None, adds to all market types
        """
        if market_type:
            self._holidays[market_type].add(holiday_date)
        else:
            for holidays in self._holidays.values():
                holidays.add(holiday_date)
    
    def remove_holiday(self, holiday_date: date, market_type: Optional[MarketType] = None) -> None:
        """
        Remove a holiday from the calendar.
        
        Args:
            holiday_date: The date to remove from holidays
            market_type: Optional market type. If None, removes from all market types
        """
        if market_type:
            self._holidays[market_type].discard(holiday_date)
        else:
            for holidays in self._holidays.values():
                holidays.discard(holiday_date)
    
    def is_holiday(self, check_date: date, market_type: MarketType) -> bool:
        """
        Check if a date is a holiday for a specific market type.
        
        Args:
            check_date: The date to check
            market_type: The market type to check
            
        Returns:
            bool: True if the date is a holiday, False otherwise
        """
        return check_date in self._holidays[market_type]
    
    def get_market_hours(self, market_type: MarketType) -> Dict[str, Dict[str, time]]:
        """
        Get market hours configuration for a specific market type.
        
        Args:
            market_type: The market type to get hours for
            
        Returns:
            Dict with regular_market, pre_market, and post_market configurations
        """
        return self.market_types.get(market_type, {})
    
    def update_market_hours(
        self,
        market_type: MarketType,
        regular_market: Optional[Dict[str, time]] = None,
        pre_market: Optional[Dict[str, time]] = None,
        post_market: Optional[Dict[str, time]] = None
    ) -> None:
        """
        Update market hours for a specific market type.
        
        Args:
            market_type: The market type to update
            regular_market: Optional new regular market hours
            pre_market: Optional new pre-market hours
            post_market: Optional new post-market hours
        """
        if market_type not in self.market_types:
            self.market_types[market_type] = {}
        
        if regular_market:
            self.market_types[market_type]['regular_market'] = regular_market
        if pre_market:
            self.market_types[market_type]['pre_market'] = pre_market
        if post_market:
            self.market_types[market_type]['post_market'] = post_market 