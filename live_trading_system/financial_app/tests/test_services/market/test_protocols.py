"""
Unit tests for market service protocols.

This module tests the protocol definitions and ensures proper interface
implementation by concrete classes.
"""

import pytest
from typing import Protocol

from app.services.market.protocols import MarketServiceProtocol
from app.services.market.market_service import MarketService


class TestMarketServiceProtocol:
    """Test suite for MarketServiceProtocol."""
    
    def test_protocol_definition(self):
        """Test that MarketServiceProtocol is properly defined."""
        assert issubclass(MarketServiceProtocol, Protocol)
        
        # Check required methods
        assert hasattr(MarketServiceProtocol, 'is_market_open')
        assert hasattr(MarketServiceProtocol, 'get_market_state')
    
    def test_concrete_implementation(self):
        """Test that concrete implementation satisfies protocol."""
        # Check that MarketService implements the protocol
        assert issubclass(MarketService, MarketServiceProtocol)
        
        # Create instance and verify runtime compatibility
        service = MarketService()
        assert isinstance(service, MarketServiceProtocol)
    
    def test_method_signatures(self):
        """Test method signatures in protocol."""
        import inspect
        
        # Check is_market_open signature
        is_market_open = MarketServiceProtocol.is_market_open
        sig = inspect.signature(is_market_open)
        params = list(sig.parameters.items())
        
        # Check parameters (excluding self)
        assert len(params) == 2  # self and symbol
        assert params[1][0] == 'symbol'
        assert params[1][1].annotation == str
        assert sig.return_annotation == bool
        
        # Check get_market_state signature
        get_market_state = MarketServiceProtocol.get_market_state
        sig = inspect.signature(get_market_state)
        params = list(sig.parameters.items())
        
        # Check parameters (excluding self)
        assert len(params) == 2  # self and symbol
        assert params[1][0] == 'symbol'
        assert params[1][1].annotation == str
        assert sig.return_annotation == str


class MockMarketService:
    """Mock implementation of MarketServiceProtocol for testing."""
    
    def is_market_open(self, symbol: str) -> bool:
        return True
    
    def get_market_state(self, symbol: str) -> str:
        return "open"


def test_mock_implementation():
    """Test that mock implementation satisfies protocol."""
    mock_service = MockMarketService()
    assert isinstance(mock_service, MarketServiceProtocol)
    
    # Test method calls
    assert mock_service.is_market_open("TEST") is True
    assert mock_service.get_market_state("TEST") == "open"


class InvalidMarketService:
    """Invalid implementation missing required methods."""
    pass


def test_invalid_implementation():
    """Test that invalid implementation fails protocol checks."""
    invalid_service = InvalidMarketService()
    assert not isinstance(invalid_service, MarketServiceProtocol) 