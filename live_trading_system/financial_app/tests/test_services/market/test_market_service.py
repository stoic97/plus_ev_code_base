"""
Unit tests for market service implementation.

This module tests the market service functionality, including market hours tracking,
market state management, and symbol mapping.
"""

import pytest
from datetime import datetime, date, time
from unittest.mock import patch
from zoneinfo import ZoneInfo

from app.services.market.market_service import MarketService
from app.services.market.market_hours import MarketType

@pytest.fixture
def market_service():
    """Fixture for MarketService instance."""
    return MarketService()

class TestMarketService:
    """Test suite for MarketService class."""
    
    def test_initialization(self, market_service):
        """Test initialization of MarketService."""
        assert hasattr(market_service, '_market_hours')
        assert 'NSE' in market_service._market_hours
        assert 'MCX' in market_service._market_hours
    
    def test_exchange_market_type_mapping(self, market_service):
        """Test symbol to exchange and market type mapping."""
        # Test NSE equity symbols
        exchange, market_type = market_service._get_exchange_and_market_type('RELIANCE.NS')
        assert exchange == 'NSE'
        assert market_type == MarketType.EQUITY
        
        # Test NSE derivatives symbols
        exchange, market_type = market_service._get_exchange_and_market_type('NIFTY24FEB18000CE.NFO')
        assert exchange == 'NSE'
        assert market_type == MarketType.DERIVATIVES
        
        # Test MCX commodity symbols
        exchange, market_type = market_service._get_exchange_and_market_type('CRUDEOIL.MCX')
        assert exchange == 'MCX'
        assert market_type == MarketType.COMMODITY
        
        # Test commodity symbols without explicit exchange
        exchange, market_type = market_service._get_exchange_and_market_type('GOLD24FEBFUT')
        assert exchange == 'MCX'
        assert market_type == MarketType.COMMODITY
        
        # Test unknown symbol
        exchange, market_type = market_service._get_exchange_and_market_type('UNKNOWN')
        assert exchange is None
        assert market_type is None
    
    @pytest.mark.parametrize('current_time,expected_state', [
        # Regular market hours
        (time(9, 30), 'open'),
        (time(15, 0), 'open'),
        
        # Pre-market hours
        (time(9, 5), 'pre-market'),
        
        # Post-market hours
        (time(15, 35), 'post-market'),
        
        # Closed hours
        (time(8, 0), 'closed'),
        (time(16, 0), 'closed'),
    ])
    def test_market_state(self, market_service, current_time, expected_state):
        """Test market state determination."""
        test_date = date(2024, 2, 1)  # Non-holiday
        test_datetime = datetime.combine(test_date, current_time)
        
        with patch('app.services.market.market_service.datetime') as mock_datetime:
            mock_datetime.now.return_value = test_datetime.replace(
                tzinfo=ZoneInfo('Asia/Kolkata')
            )
            
            state = market_service.get_market_state('RELIANCE.NS')
            assert state == expected_state
    
    @pytest.mark.parametrize('current_time,expected_open', [
        # Regular market hours
        (time(9, 30), True),
        (time(15, 0), True),
        
        # Pre-market hours
        (time(9, 5), False),
        
        # Post-market hours
        (time(15, 35), False),
        
        # Closed hours
        (time(8, 0), False),
        (time(16, 0), False),
    ])
    def test_is_market_open(self, market_service, current_time, expected_open):
        """Test market open status determination."""
        test_date = date(2024, 2, 1)  # Non-holiday
        test_datetime = datetime.combine(test_date, current_time)
        
        with patch('app.services.market.market_service.datetime') as mock_datetime:
            mock_datetime.now.return_value = test_datetime.replace(
                tzinfo=ZoneInfo('Asia/Kolkata')
            )
            
            is_open = market_service.is_market_open('RELIANCE.NS')
            assert is_open == expected_open
    
    def test_holiday_handling(self, market_service):
        """Test market state and open status on holidays."""
        # Add a holiday for NSE Equity
        nse_hours = market_service._market_hours['NSE']
        holiday_date = date(2024, 1, 26)  # Republic Day
        nse_hours.add_holiday(holiday_date, MarketType.EQUITY)
        
        test_datetime = datetime.combine(
            holiday_date,
            time(9, 30)  # During regular market hours
        )
        
        with patch('app.services.market.market_service.datetime') as mock_datetime:
            mock_datetime.now.return_value = test_datetime.replace(
                tzinfo=ZoneInfo('Asia/Kolkata')
            )
            
            # Check market state and open status
            assert market_service.get_market_state('RELIANCE.NS') == 'holiday'
            assert not market_service.is_market_open('RELIANCE.NS')
    
    def test_commodity_market_hours(self, market_service):
        """Test commodity market hours handling."""
        test_date = date(2024, 2, 1)  # Non-holiday
        
        # Test during commodity trading hours
        test_datetime = datetime.combine(test_date, time(22, 0))
        
        with patch('app.services.market.market_service.datetime') as mock_datetime:
            mock_datetime.now.return_value = test_datetime.replace(
                tzinfo=ZoneInfo('Asia/Kolkata')
            )
            
            # Check MCX commodity
            assert market_service.is_market_open('CRUDEOIL.MCX')
            assert market_service.get_market_state('CRUDEOIL.MCX') == 'open'
            
            # Check commodity without explicit exchange
            assert market_service.is_market_open('GOLD24FEBFUT')
            assert market_service.get_market_state('GOLD24FEBFUT') == 'open'
    
    def test_unknown_symbol_handling(self, market_service):
        """Test handling of unknown symbols."""
        assert not market_service.is_market_open('UNKNOWN')
        assert market_service.get_market_state('UNKNOWN') == 'unknown'
    
    def test_protocol_conformance(self, market_service):
        """Test that MarketService properly implements MarketServiceProtocol."""
        from app.services.market.protocols import MarketServiceProtocol
        
        # Check if MarketService implements all protocol methods
        assert isinstance(market_service, MarketServiceProtocol)
        assert hasattr(market_service, 'is_market_open')
        assert hasattr(market_service, 'get_market_state')
        
        # Test method signatures
        import inspect
        
        is_market_open_sig = inspect.signature(market_service.is_market_open)
        assert list(is_market_open_sig.parameters.keys()) == ['symbol']
        assert is_market_open_sig.return_annotation == bool
        
        get_market_state_sig = inspect.signature(market_service.get_market_state)
        assert list(get_market_state_sig.parameters.keys()) == ['symbol']
        assert get_market_state_sig.return_annotation == str 