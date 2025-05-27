"""
Market service protocols and interfaces.

This module defines the protocols that market-related services must implement.
"""

from typing import Protocol

class MarketServiceProtocol(Protocol):
    """Protocol defining the interface for market service implementations."""
    
    def is_market_open(self, symbol: str) -> bool:
        """
        Check if the market is open for a given symbol.
        
        Args:
            symbol: The trading symbol to check
            
        Returns:
            bool: True if the market is open, False otherwise
        """
        ...
    
    def get_market_state(self, symbol: str) -> str:
        """
        Get the current market state for a given symbol.
        
        Args:
            symbol: The trading symbol to check
            
        Returns:
            str: Current market state (e.g., "pre-market", "open", "closed", "post-market")
        """
        ... 