"""
Unit tests for market hours functionality.

This module tests the market hours configuration and management,
including regular hours, pre/post market hours, and holidays.
"""

import pytest
from datetime import date, time
from zoneinfo import ZoneInfo

from app.services.market.market_hours import MarketHours, MarketType

@pytest.fixture
def ist_timezone():
    """Fixture for IST timezone."""
    return ZoneInfo('Asia/Kolkata')

@pytest.fixture
def nse_market_hours(ist_timezone):
    """Fixture for NSE market hours configuration."""
    return {
        MarketType.EQUITY: {
            'regular_market': {
                'start': time(9, 15),
                'end': time(15, 30)
            },
            'pre_market': {
                'start': time(9, 0),
                'end': time(9, 15)
            },
            'post_market': {
                'start': time(15, 30),
                'end': time(15, 50)
            }
        }
    }

@pytest.fixture
def nse_holidays():
    """Fixture for NSE holidays."""
    return {
        MarketType.EQUITY: [
            date(2024, 1, 26),  # Republic Day
            date(2024, 8, 15),  # Independence Day
            date(2024, 10, 2)   # Gandhi Jayanti
        ]
    }

@pytest.fixture
def market_hours(ist_timezone, nse_market_hours, nse_holidays):
    """Fixture for MarketHours instance."""
    return MarketHours(
        timezone=ist_timezone,
        market_types=nse_market_hours,
        holidays=nse_holidays
    )

class TestMarketHours:
    """Test suite for MarketHours class."""
    
    def test_initialization(self, market_hours, ist_timezone):
        """Test initialization of MarketHours."""
        assert market_hours.timezone == ist_timezone
        assert MarketType.EQUITY in market_hours.market_types
        
        # Check if holidays are properly initialized
        assert market_hours.is_holiday(date(2024, 1, 26), MarketType.EQUITY)
        assert not market_hours.is_holiday(date(2024, 1, 27), MarketType.EQUITY)
    
    def test_add_holiday(self, market_hours):
        """Test adding holidays."""
        new_holiday = date(2024, 12, 25)
        
        # Add holiday for specific market type
        market_hours.add_holiday(new_holiday, MarketType.EQUITY)
        assert market_hours.is_holiday(new_holiday, MarketType.EQUITY)
        assert not market_hours.is_holiday(new_holiday, MarketType.COMMODITY)
        
        # Add holiday for all market types
        all_markets_holiday = date(2024, 12, 31)
        market_hours.add_holiday(all_markets_holiday)
        for market_type in MarketType:
            assert market_hours.is_holiday(all_markets_holiday, market_type)
    
    def test_remove_holiday(self, market_hours):
        """Test removing holidays."""
        existing_holiday = date(2024, 1, 26)
        
        # Remove from specific market type
        market_hours.remove_holiday(existing_holiday, MarketType.EQUITY)
        assert not market_hours.is_holiday(existing_holiday, MarketType.EQUITY)
        
        # Add and remove from all market types
        new_holiday = date(2024, 12, 25)
        market_hours.add_holiday(new_holiday)
        market_hours.remove_holiday(new_holiday)
        for market_type in MarketType:
            assert not market_hours.is_holiday(new_holiday, market_type)
    
    def test_get_market_hours(self, market_hours):
        """Test getting market hours configuration."""
        equity_hours = market_hours.get_market_hours(MarketType.EQUITY)
        
        assert 'regular_market' in equity_hours
        assert equity_hours['regular_market']['start'] == time(9, 15)
        assert equity_hours['regular_market']['end'] == time(15, 30)
        
        assert 'pre_market' in equity_hours
        assert equity_hours['pre_market']['start'] == time(9, 0)
        assert equity_hours['pre_market']['end'] == time(9, 15)
        
        # Test non-existent market type
        assert market_hours.get_market_hours(MarketType.FOREX) == {}
    
    def test_update_market_hours(self, market_hours):
        """Test updating market hours configuration."""
        # Update regular market hours
        new_regular_hours = {
            'start': time(9, 0),
            'end': time(16, 0)
        }
        market_hours.update_market_hours(
            MarketType.EQUITY,
            regular_market=new_regular_hours
        )
        
        updated_hours = market_hours.get_market_hours(MarketType.EQUITY)
        assert updated_hours['regular_market'] == new_regular_hours
        
        # Update pre-market hours
        new_pre_market = {
            'start': time(8, 45),
            'end': time(9, 0)
        }
        market_hours.update_market_hours(
            MarketType.EQUITY,
            pre_market=new_pre_market
        )
        
        updated_hours = market_hours.get_market_hours(MarketType.EQUITY)
        assert updated_hours['pre_market'] == new_pre_market
        
        # Add hours for new market type
        forex_hours = {
            'start': time(0, 0),
            'end': time(23, 59)
        }
        market_hours.update_market_hours(
            MarketType.FOREX,
            regular_market=forex_hours
        )
        
        assert market_hours.get_market_hours(MarketType.FOREX)['regular_market'] == forex_hours
    
    def test_holiday_management(self, market_hours):
        """Test comprehensive holiday management."""
        # Test adding multiple holidays
        holidays = [
            date(2024, 1, 1),
            date(2024, 12, 25),
            date(2024, 12, 31)
        ]
        
        for holiday in holidays:
            market_hours.add_holiday(holiday, MarketType.EQUITY)
            assert market_hours.is_holiday(holiday, MarketType.EQUITY)
        
        # Test removing multiple holidays
        for holiday in holidays:
            market_hours.remove_holiday(holiday, MarketType.EQUITY)
            assert not market_hours.is_holiday(holiday, MarketType.EQUITY)
        
        # Test adding holiday for all market types
        common_holiday = date(2024, 1, 1)
        market_hours.add_holiday(common_holiday)
        for market_type in MarketType:
            assert market_hours.is_holiday(common_holiday, market_type) 