"""
Unit tests for Fyers data transformation utilities.

This module tests the transformation functions in app/providers/fyers/transformers.py
to ensure accurate conversion between Fyers API formats and internal data formats.
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone
import json
from unittest.mock import patch

from app.providers.fyers.transformers import (
    map_symbol,
    map_interval,
    normalize_symbol,
    to_decimal_string,
    to_unix_timestamp,
    transform_ohlcv,
    transform_orderbook,
    transform_quote,
    transform_websocket_message
)


class TestSymbolMapping:
    """Tests for symbol mapping functions."""
    
    def test_map_symbol_with_valid_inputs(self):
        """Test map_symbol with various valid input formats."""
        # Test cases: (input, expected output)
        test_cases = [
            # Already in Fyers format
            ("NSE:SBIN-EQ", "NSE:SBIN-EQ"),
            ("BSE:RELIANCE-EQ", "BSE:RELIANCE-EQ"),
            ("MCX:GOLD-FUT", "MCX:GOLD-FUT"),
            
            # Needs conversion
            ("NSE:SBIN", "NSE:SBIN-EQ"),
            ("NSE:NIFTY", "NSE:NIFTY-INDEX"),
            ("NSE:BANKNIFTY", "NSE:BANKNIFTY-INDEX"),
            ("NSE:SENSEX", "NSE:SENSEX-INDEX"),
            
            # Without exchange (default to NSE)
            ("SBIN", "NSE:SBIN-EQ"),
            ("NIFTY50", "NSE:NIFTY50-INDEX"),
            
            # Special cases
            ("NSE:BANKNIFTY24JUNFUT", "NSE:BANKNIFTY24JUNFUT"),  # Already has FUT
            ("NSE:RELIANCE24JUN2100CE", "NSE:RELIANCE24JUN2100CE"),  # Option symbol
        ]
        
        for input_symbol, expected_output in test_cases:
            assert map_symbol(input_symbol) == expected_output
    
    def test_map_symbol_with_invalid_inputs(self):
        """Test map_symbol with invalid inputs."""
        with pytest.raises(ValueError):
            map_symbol(None)
        
        with pytest.raises(ValueError):
            map_symbol("")
        
        with pytest.raises(ValueError):
            map_symbol(123)  # Non-string input
    
    def test_normalize_symbol(self):
        """Test normalize_symbol function."""
        # Test cases: (input, expected output)
        test_cases = [
            # Equity symbols
            ("NSE:SBIN-EQ", "NSE:SBIN"),
            ("BSE:RELIANCE-EQ", "BSE:RELIANCE"),
            
            # Non-equity symbols should keep type
            ("NSE:NIFTY-INDEX", "NSE:NIFTY-index"),
            ("MCX:GOLD-FUT", "MCX:GOLD-fut"),
            ("NSE:BANKNIFTY24JUN18000CE", "NSE:BANKNIFTY24JUN18000CE"),
            
            # No type, return as is
            ("NSE:USDINR", "NSE:USDINR"),
            
            # No exchange, return as is
            ("SBIN", "SBIN"),
        ]
        
        for input_symbol, expected_output in test_cases:
            assert normalize_symbol(input_symbol) == expected_output


class TestIntervalMapping:
    """Tests for interval mapping functions."""
    
    def test_map_interval_with_valid_inputs(self):
        """Test map_interval with valid input formats."""
        # Test cases: (input, expected output)
        test_cases = [
            # Seconds
            ("5s", "5S"), ("10s", "10S"), ("15s", "15S"), ("30s", "30S"),
            
            # Minutes
            ("1m", "1"), ("5m", "5"), ("15m", "15"), ("30m", "30"),
            
            # Hours
            ("1h", "60"), ("2h", "120"), ("4h", "240"),
            
            # Days and weeks
            ("1d", "D"), ("D", "D"), ("d", "D"),
            ("1w", "W"), ("W", "W"),
            
            # Months
            ("1mo", "M"), ("M", "M"), ("mo", "M"),
            
            # With spaces
            (" 5m ", "5"),
            (" 1d ", "D"),
        ]
        
        for input_interval, expected_output in test_cases:
            assert map_interval(input_interval) == expected_output
    
    def test_map_interval_with_invalid_inputs(self):
        """Test map_interval with invalid inputs."""
        # Invalid intervals
        invalid_intervals = [None, "", "invalid", "0m", "6min", "24h", "2d"]
        
        for interval in invalid_intervals:
            with pytest.raises(ValueError):
                map_interval(interval)


class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_to_decimal_string(self):
        """Test to_decimal_string utility function."""
        # Test cases: (input, expected output)
        test_cases = [
            (100, "100"),
            (100.5, "100.5"),
            (100.505, "100.505"),
            (100.5050, "100.505"),  # Trimming trailing zeros
            ("100.5", "100.5"),
            (Decimal("100.5"), "100.5"),
            (Decimal("100.50"), "100.5"),  # Trimming trailing zeros
            (Decimal("100.5000000"), "100.5"),  # Trimming trailing zeros
            (0, "0"),
            (0.0, "0"),
            (None, "0.00"),
            ("invalid", "0.00"),  # Invalid input
        ]
        
        for input_value, expected_output in test_cases:
            assert to_decimal_string(input_value) == expected_output
    
    def test_to_unix_timestamp(self):
        """Test to_unix_timestamp function."""
        # Test with datetime object
        dt = datetime(2022, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        expected_timestamp = 1641038400  # 2022-01-01 12:00:00 UTC in Unix timestamp
        assert to_unix_timestamp(dt) == expected_timestamp
        
        # Test with integer
        assert to_unix_timestamp(1641038400) == 1641038400


class TestOHLCVTransformation:
    """Tests for OHLCV data transformation."""
    
    def test_transform_ohlcv_with_valid_data(self):
        """Test transform_ohlcv with valid data."""
        # Sample Fyers OHLCV response
        fyers_response = {
            "s": "ok",
            "candles": [
                # timestamp, open, high, low, close, volume
                [1641038400, 100.5, 101.2, 100.1, 100.8, 1000000],
                [1641124800, 100.8, 102.0, 100.5, 101.5, 1200000],
            ]
        }
        
        result = transform_ohlcv(fyers_response)
        
        # Verify length
        assert len(result) == 2
        
        # Verify first candle
        first_candle = result[0]
        assert first_candle["timestamp"] == 1641038400
        assert first_candle["open"] == "100.5"
        assert first_candle["high"] == "101.2"
        assert first_candle["low"] == "100.1"
        assert first_candle["close"] == "100.8"
        assert first_candle["volume"] == 1000000
        assert first_candle["source"] == "fyers"
        assert "datetime" in first_candle  # ISO timestamp should be there
    
    def test_transform_ohlcv_with_invalid_data(self):
        """Test transform_ohlcv with invalid data."""
        # Invalid response
        with pytest.raises(ValueError):
            transform_ohlcv({"s": "error", "message": "Some error"})
        
        # Empty candles
        assert transform_ohlcv({"s": "ok", "candles": []}) == []
        
        # Invalid candle data
        response_with_invalid_candle = {
            "s": "ok",
            "candles": [
                # Invalid OHLC relationship (high < low)
                [1641038400, 100.5, 99.0, 100.1, 100.8, 1000000],
            ]
        }
        assert transform_ohlcv(response_with_invalid_candle) == []


class TestOrderbookTransformation:
    """Tests for orderbook data transformation."""
    
    def test_transform_orderbook_with_valid_data(self):
        """Test transform_orderbook with valid data."""
        # Sample Fyers orderbook response
        fyers_response = {
            "s": "ok",
            "d": {
                "NSE:SBIN-EQ": {
                    "totalbuyqty": 5000,
                    "totalsellqty": 6000,
                    "bids": [
                        {"price": 100.5, "volume": 1000, "ord": 5},
                        {"price": 100.4, "volume": 2000, "ord": 8}
                    ],
                    "ask": [
                        {"price": 100.6, "volume": 1500, "ord": 6},
                        {"price": 100.7, "volume": 2500, "ord": 10}
                    ],
                    "ltp": 100.55,
                    "ltq": 100,
                    "ltt": 1641038400,
                    "v": 1500000,
                    "atp": 100.52
                }
            }
        }
        
        # Mock datetime.now for consistent test results
        with patch('app.providers.fyers.transformers.datetime') as mock_dt:
            mock_now = datetime(2022, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
            mock_dt.now.return_value = mock_now
            mock_dt.fromtimestamp.return_value = mock_now
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            result = transform_orderbook(fyers_response, "NSE:SBIN")
        
        # Verify basic fields
        assert result["symbol"] == "NSE:SBIN"
        assert result["best_bid"] == "100.5"
        assert result["best_ask"] == "100.6"
        assert result["spread"] == "0.1"
        assert result["mid_price"] == "100.55"
        assert result["total_bid_volume"] == 5000
        assert result["total_ask_volume"] == 6000
        assert result["last_price"] == "100.55"
        
        # Verify bids and asks arrays
        assert len(result["bids"]) == 2
        assert result["bids"][0]["price"] == "100.5"
        assert result["bids"][0]["volume"] == 1000
        assert result["bids"][0]["orders"] == 5
        
        assert len(result["asks"]) == 2
        assert result["asks"][0]["price"] == "100.6"
        assert result["asks"][0]["volume"] == 1500
        assert result["asks"][0]["orders"] == 6
    
    def test_transform_orderbook_with_depth_limit(self):
        """Test transform_orderbook with depth limit."""
        # Sample Fyers orderbook response with multiple levels
        fyers_response = {
            "s": "ok",
            "d": {
                "NSE:SBIN-EQ": {
                    "totalbuyqty": 10000,
                    "totalsellqty": 12000,
                    "bids": [
                        {"price": 100.5, "volume": 1000, "ord": 5},
                        {"price": 100.4, "volume": 2000, "ord": 8},
                        {"price": 100.3, "volume": 3000, "ord": 12}
                    ],
                    "ask": [
                        {"price": 100.6, "volume": 1500, "ord": 6},
                        {"price": 100.7, "volume": 2500, "ord": 10},
                        {"price": 100.8, "volume": 3500, "ord": 14}
                    ],
                    "ltp": 100.55
                }
            }
        }
        
        # Test with depth = 2
        result = transform_orderbook(fyers_response, "NSE:SBIN", depth=2)
        
        # Verify depth limitation
        assert len(result["bids"]) == 2
        assert len(result["asks"]) == 2


class TestQuoteTransformation:
    """Tests for quote data transformation."""
    
    def test_transform_quote_with_valid_data(self):
        """Test transform_quote with valid data."""
        # Sample Fyers quote response
        fyers_response = {
            "s": "ok",
            "d": [
                {
                    "n": "NSE:SBIN-EQ",
                    "s": "ok",
                    "v": {
                        "lp": 100.5,
                        "bid": 100.45,
                        "ask": 100.55,
                        "spread": 0.1,
                        "open_price": 100.0,
                        "high_price": 101.0,
                        "low_price": 99.8,
                        "prev_close_price": 99.5,
                        "ch": 1.0,
                        "chp": 1.01,
                        "volume": 1500000,
                        "atp": 100.25,
                        "upper_ckt": 110.0,
                        "lower_ckt": 90.0,
                        "exchange": "NSE",
                        "fyToken": "12345678"
                    }
                }
            ]
        }
        
        # Mock datetime.now for consistent test results
        with patch('app.providers.fyers.transformers.datetime') as mock_dt:
            mock_now = datetime(2022, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            result = transform_quote(fyers_response, "NSE:SBIN")
        
        # Verify basic fields
        assert result["symbol"] == "NSE:SBIN"
        assert result["last_price"] == "100.5"
        assert result["bid_price"] == "100.45"
        assert result["ask_price"] == "100.55"
        assert result["spread"] == "0.1"
        assert result["open_price"] == "100"
        assert result["high_price"] == "101"
        assert result["low_price"] == "99.8"
        assert result["prev_close"] == "99.5"
        assert result["change"] == "1"
        assert result["change_percent"] == 1.01
        assert result["volume"] == 1500000
        assert result["avg_price"] == "100.25"
        
        # Verify circuit limits
        assert result["upper_circuit"] == "110"
        assert result["lower_circuit"] == "90"
        
        # Verify exchange metadata
        assert result["exchange"] == "NSE"
        assert result["fyers_token"] == "12345678"


class TestWebSocketTransformation:
    """Tests for WebSocket message transformation."""
    
    def test_transform_websocket_message(self):
        """Test transform_websocket_message with symbol feed."""
        # Sample Fyers WebSocket message
        ws_message = {
            "type": "sf",
            "symbol": "NSE:SBIN-EQ",
            "ltp": 100.5,
            "bid_price": 100.45,
            "ask_price": 100.55,
            "vol_traded_today": 1500000,
            "ch": 1.0,
            "chp": 1.01
        }
        
        # Mock datetime.now for consistent test results
        with patch('app.providers.fyers.transformers.datetime') as mock_dt:
            mock_now = datetime(2022, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            result = transform_websocket_message(ws_message)
        
        # Verify basic fields
        assert result["symbol"] == "NSE:SBIN"
        assert result["message_type"] == "sf"
        assert result["last_price"] == "100.5"
        assert result["bid_price"] == "100.45"
        assert result["ask_price"] == "100.55"
        assert result["volume"] == 1500000
        assert result["change"] == "1"
        assert result["change_percent"] == 1.01
        assert result["source"] == "fyers_ws"


if __name__ == "__main__":
    pytest.main()