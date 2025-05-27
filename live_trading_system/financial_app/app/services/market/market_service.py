"""
Market service implementation.

This module provides the implementation of market-related services,
including market hours tracking and market state management.
"""

from datetime import datetime, time
from typing import Dict, Optional, Tuple
from zoneinfo import ZoneInfo

from .protocols import MarketServiceProtocol
from .market_hours import MarketHours, MarketType

class MarketService(MarketServiceProtocol):
    """
    Service for tracking market hours and states.
    
    This service handles:
    - Market hours for different exchanges and market types
    - Holiday schedules
    - Special trading hours
    - Market state transitions
    """
    
    def __init__(self):
        """Initialize the market service."""
        self._market_hours: Dict[str, MarketHours] = {}
        self._initialize_market_hours()
    
    def _initialize_market_hours(self):
        """Initialize market hours for different exchanges."""
        # NSE (Indian National Stock Exchange)
        self._market_hours['NSE'] = MarketHours(
            timezone=ZoneInfo('Asia/Kolkata'),
            market_types={
                MarketType.EQUITY: {
                    'regular_market': {
                        'start': time(9, 15),   # 9:15 AM IST
                        'end': time(15, 30)     # 3:30 PM IST
                    },
                    'pre_market': {
                        'start': time(9, 0),    # 9:00 AM IST
                        'end': time(9, 15)      # 9:15 AM IST
                    },
                    'post_market': {
                        'start': time(15, 30),  # 3:30 PM IST
                        'end': time(15, 50)     # 3:50 PM IST
                    }
                },
                MarketType.COMMODITY: {
                    'regular_market': {
                        'start': time(9, 0),    # 9:00 AM IST
                        'end': time(23, 30)     # 11:30 PM IST
                    }
                }
            }
        )
        
        # MCX (Multi Commodity Exchange)
        self._market_hours['MCX'] = MarketHours(
            timezone=ZoneInfo('Asia/Kolkata'),
            market_types={
                MarketType.COMMODITY: {
                    'regular_market': {
                        'start': time(9, 0),    # 9:00 AM IST
                        'end': time(23, 30)     # 11:30 PM IST
                    }
                }
            }
        )
    
    def _get_exchange_and_market_type(self, symbol: str) -> Tuple[Optional[str], Optional[MarketType]]:
        """
        Get the exchange and market type for a given symbol.
        
        Args:
            symbol: The trading symbol
            
        Returns:
            Tuple of (exchange code, market type) or (None, None) if not found
        """
        if symbol.endswith('.NS'):
            return 'NSE', MarketType.EQUITY
        elif symbol.endswith('.NFO'):
            return 'NSE', MarketType.DERIVATIVES
        elif symbol.endswith('.MCX'):
            return 'MCX', MarketType.COMMODITY
        elif 'CRUDE' in symbol or 'GOLD' in symbol or 'SILVER' in symbol:
            return 'MCX', MarketType.COMMODITY
        
        # Add more mappings as needed
        return None, None
    
    def is_market_open(self, symbol: str) -> bool:
        """
        Check if the market is open for a given symbol.
        
        Args:
            symbol: The trading symbol to check
            
        Returns:
            bool: True if the market is open, False otherwise
        """
        exchange, market_type = self._get_exchange_and_market_type(symbol)
        if not exchange or not market_type or exchange not in self._market_hours:
            return False
        
        market_hours = self._market_hours[exchange]
        current_time = datetime.now(market_hours.timezone)
        
        # Check if it's a holiday for this market type
        if market_hours.is_holiday(current_time.date(), market_type):
            return False
        
        # Get hours for this market type
        hours = market_hours.get_market_hours(market_type)
        if not hours or 'regular_market' not in hours:
            return False
        
        # Check if within regular market hours
        current_time_only = current_time.time()
        regular_hours = hours['regular_market']
        return regular_hours['start'] <= current_time_only < regular_hours['end']
    
    def get_market_state(self, symbol: str) -> str:
        """
        Get the current market state for a given symbol.
        
        Args:
            symbol: The trading symbol to check
            
        Returns:
            str: Current market state (e.g., "pre-market", "open", "closed", "post-market")
        """
        exchange, market_type = self._get_exchange_and_market_type(symbol)
        if not exchange or not market_type or exchange not in self._market_hours:
            return "unknown"
        
        market_hours = self._market_hours[exchange]
        current_time = datetime.now(market_hours.timezone)
        
        # Check if it's a holiday for this market type
        if market_hours.is_holiday(current_time.date(), market_type):
            return "holiday"
        
        # Get hours for this market type
        hours = market_hours.get_market_hours(market_type)
        if not hours or 'regular_market' not in hours:
            return "unknown"
        
        current_time_only = current_time.time()
        
        # Check pre-market
        if ('pre_market' in hours and 
            hours['pre_market']['start'] <= current_time_only < hours['pre_market']['end']):
            return "pre-market"
        
        # Check regular market hours
        if hours['regular_market']['start'] <= current_time_only < hours['regular_market']['end']:
            return "open"
        
        # Check post-market
        if ('post_market' in hours and 
            hours['post_market']['start'] <= current_time_only < hours['post_market']['end']):
            return "post-market"
        
        return "closed" 