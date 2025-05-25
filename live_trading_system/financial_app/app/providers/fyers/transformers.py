"""
Fyers Data Transformation Utilities

This module provides functions to convert between internal data formats
and Fyers API formats, optimized for algorithmic trading operations.

Internal Conventions:
- Symbols: EXCHANGE:SYMBOL[-TYPE] format
- Timestamps: Unix timestamps (UTC) with optional ISO strings
- Prices: Decimal as strings for precision
- Volumes: Integers for share quantities 
- Field names: snake_case with consistent prefixes
"""

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from decimal import Decimal, InvalidOperation

# Set up logging
logger = logging.getLogger(__name__)

# Symbol type mappings
SYMBOL_TYPE_MAPPING = {
    'EQ': 'equity',
    'FUT': 'futures',
    'CE': 'call_option',
    'PE': 'put_option',
    'INDEX': 'index',
    'ETF': 'etf'
}

# Interval mappings: internal format -> Fyers format
INTERVAL_MAPPING = {
    # Seconds (for high-frequency trading)
    '5s': '5S', '10s': '10S', '15s': '15S', '30s': '30S', '45s': '45S',
    
    # Minutes (most common for algo trading)
    '1m': '1', '2m': '2', '3m': '3', '5m': '5', '10m': '10', 
    '15m': '15', '20m': '20', '30m': '30', '45m': '45',
    
    # Hours (for swing trading)
    '1h': '60', '2h': '120', '3h': '180', '4h': '240', 
    '6h': '360', '8h': '480', '12h': '720',
    
    # Days (for position trading)
    '1d': 'D', 'D': 'D', 'd': 'D',
    
    # Weekly/Monthly (for long-term analysis)  
    '1w': 'W', 'W': 'W', 'w': 'W',
    '1mo': 'M', 'M': 'M', 'mo': 'M'
}

# Exchange standardization
EXCHANGE_MAPPING = {
    'NSE': 'NSE',
    'BSE': 'BSE', 
    'MCX': 'MCX',
    'NCDEX': 'NCDEX'
}


def map_symbol(symbol: str) -> str:
    """
    Convert internal symbol format to Fyers format.
    
    Internal Format: EXCHANGE:SYMBOL[-TYPE]
    Fyers Format: EXCHANGE:SYMBOL-SEGMENT
    
    Args:
        symbol: Symbol in internal format
        
    Returns:
        Symbol in Fyers format
        
    Examples:
        >>> map_symbol("NSE:SBIN")
        "NSE:SBIN-EQ"
        >>> map_symbol("NSE:BANKNIFTY-INDEX")  
        "NSE:BANKNIFTY-INDEX"
    """
    if not symbol or not isinstance(symbol, str):
        raise ValueError(f"Invalid symbol: {symbol}")
    
    symbol = symbol.strip().upper()
    
    # Already in Fyers format
    if re.match(r'^[A-Z]+:[A-Z0-9]+-[A-Z]+$', symbol):
        return symbol
    
    # Parse internal format
    if ':' not in symbol:
        # Assume NSE if no exchange specified
        symbol = f"NSE:{symbol}"
    
    exchange, rest = symbol.split(':', 1)
    
    # Add default segment if not specified
    if '-' not in rest:
        # Determine default segment based on symbol patterns
        if rest in ['NIFTY', 'NIFTY50', 'BANKNIFTY', 'SENSEX']:
            rest = f"{rest}-INDEX"
        elif rest.endswith('FUT') or 'FUT' in rest:
            # Already has FUT in name
            pass
        elif rest.endswith(('CE', 'PE')):
            # Already has option type
            pass
        else:
            # Default to equity
            rest = f"{rest}-EQ"
    
    return f"{exchange}:{rest}"


def map_interval(interval: str) -> str:
    """
    Convert internal interval format to Fyers format.
    
    Args:
        interval: Time interval in internal format
        
    Returns:
        Interval in Fyers format
        
    Examples:
        >>> map_interval("1m")
        "1"
        >>> map_interval("1d")
        "D"
    """
    if not interval:
        raise ValueError("Interval cannot be empty")
    
    interval_norm = interval.lower().strip()

    if interval.upper() == "M":
        return "M"  # For monthly interval
    if interval.upper() == "W":
        return "W"  # For weekly interval
    
    if interval_norm in INTERVAL_MAPPING:
        return INTERVAL_MAPPING[interval_norm]
    
    raise ValueError(f"Unsupported interval: {interval}")


def normalize_symbol(fyers_symbol: str) -> str:
    """
    Convert Fyers symbol to standardized internal format.
    
    Args:
        fyers_symbol: Symbol in Fyers format (e.g., "NSE:SBIN-EQ")
        
    Returns:
        Symbol in internal format (e.g., "NSE:SBIN")
    """
    if ':' not in fyers_symbol:
        return fyers_symbol
    
    exchange, rest = fyers_symbol.split(':', 1)
    
    if '-' in rest:
        symbol, segment = rest.split('-', 1)
        # For equity, use simplified format
        if segment == 'EQ':
            return f"{exchange}:{symbol}"
        # For others, keep the type
        else:
            return f"{exchange}:{symbol}-{segment.lower()}"
    
    return fyers_symbol


def to_decimal_string(value: Any) -> str:
    """Convert numeric value to decimal string for precision."""
    if value is None:
        return "0.00"
    try:
        # Convert to Decimal for precision, then to string
        decimal_val = Decimal(str(value))
        # Format to reasonable precision (8 decimal places)
        return f"{decimal_val:.8f}".rstrip('0').rstrip('.')
    except (TypeError, ValueError, InvalidOperation):
        # Added decimal.InvalidOperation to catch invalid string inputs
        return "0.00"


def to_unix_timestamp(dt: datetime) -> int:
    """Convert datetime to Unix timestamp."""
    if isinstance(dt, datetime):
        return int(dt.timestamp())
    return int(dt)


def transform_ohlcv(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Transform Fyers OHLCV response to internal format optimized for trading.
    
    Internal Format (per candle):
    {
        "symbol": "NSE:SBIN",
        "timestamp": 1640995200,
        "datetime": "2022-01-01T00:00:00Z",
        "open": "100.50",
        "high": "101.20", 
        "low": "100.10",
        "close": "100.80",
        "volume": 1000000,
        "interval": "1m",
        "source": "fyers"
    }
    """
    if not isinstance(response, dict) or response.get("s") != "ok":
        error_msg = response.get("message", "Invalid response")
        raise ValueError(f"Fyers API error: {error_msg}")
    
    candles = response.get("candles", [])
    if not candles:
        return []
    
    result = []
    
    for candle in candles:
        if not isinstance(candle, list) or len(candle) < 6:
            continue
            
        try:
            timestamp = int(candle[0])
            dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            
            # Validate OHLC relationships
            open_price = float(candle[1])
            high_price = float(candle[2])
            low_price = float(candle[3])
            close_price = float(candle[4])
            volume = int(candle[5])
            
            # Skip invalid candles
            if high_price < low_price or high_price < max(open_price, close_price) or \
               low_price > min(open_price, close_price):
                logger.warning(f"Invalid OHLC data: {candle}")
                continue
            
            candle_data = {
                "timestamp": timestamp,
                "datetime": dt.isoformat(),
                "open": to_decimal_string(open_price),
                "high": to_decimal_string(high_price),
                "low": to_decimal_string(low_price),
                "close": to_decimal_string(close_price),
                "volume": volume,
                "source": "fyers"
            }
            
            result.append(candle_data)
            
        except (ValueError, TypeError, IndexError) as e:
            logger.error(f"Error processing candle: {e}")
            continue
    
    logger.info(f"Transformed {len(result)} OHLCV candles")
    return result


def transform_orderbook(
    response: Dict[str, Any], 
    symbol: str, 
    depth: Optional[int] = None
) -> Dict[str, Any]:
    """
    Transform Fyers orderbook response to internal format optimized for trading.
    
    Internal Format:
    {
        "symbol": "NSE:SBIN",
        "timestamp": 1640995200,
        "datetime": "2022-01-01T10:00:00Z",
        "bids": [
            {"price": "100.50", "volume": 1000, "orders": 5},
            {"price": "100.45", "volume": 800, "orders": 3}
        ],
        "asks": [
            {"price": "100.55", "volume": 1200, "orders": 4},
            {"price": "100.60", "volume": 900, "orders": 2}
        ],
        "best_bid": "100.50",
        "best_ask": "100.55", 
        "spread": "0.05",
        "mid_price": "100.525",
        "total_bid_volume": 1800,
        "total_ask_volume": 2100,
        "last_price": "100.52",
        "last_volume": 100,
        "daily_volume": 1500000,
        "source": "fyers"
    }
    """
    if not isinstance(response, dict) or response.get("s") != "ok":
        error_msg = response.get("message", "Invalid response")
        raise ValueError(f"Fyers API error: {error_msg}")
    
    # Extract orderbook data
    data = response.get("d", {})
    fyers_symbol = map_symbol(symbol)
    
    if fyers_symbol not in data:
        raise ValueError(f"No orderbook data for symbol: {symbol}")
    
    book_data = data[fyers_symbol]
    
    # Transform bids (sorted by price descending)
    bids = []
    for bid in book_data.get("bids", [])[:depth] if depth else book_data.get("bids", []):
        bids.append({
            "price": to_decimal_string(bid.get("price")),
            "volume": int(bid.get("volume", 0)),
            "orders": int(bid.get("ord", 0))
        })
    
    # Transform asks (sorted by price ascending)  
    asks = []
    for ask in book_data.get("ask", [])[:depth] if depth else book_data.get("ask", []):
        asks.append({
            "price": to_decimal_string(ask.get("price")),
            "volume": int(ask.get("volume", 0)),
            "orders": int(ask.get("ord", 0))
        })
    
    # Calculate derived values
    best_bid = bids[0]["price"] if bids else "0.00"
    best_ask = asks[0]["price"] if asks else "0.00"
    
    spread = "0.00"
    mid_price = "0.00"
    if best_bid != "0.00" and best_ask != "0.00":
        bid_decimal = Decimal(best_bid)
        ask_decimal = Decimal(best_ask)
        spread = to_decimal_string(ask_decimal - bid_decimal)
        mid_price = to_decimal_string((bid_decimal + ask_decimal) / 2)
    
    # Current timestamp
    now = datetime.now(timezone.utc)
    
    result = {
        "symbol": normalize_symbol(fyers_symbol),
        "timestamp": int(now.timestamp()),
        "datetime": now.isoformat(),
        "bids": bids,
        "asks": asks,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": spread,
        "mid_price": mid_price,
        "total_bid_volume": int(book_data.get("totalbuyqty", 0)),
        "total_ask_volume": int(book_data.get("totalsellqty", 0)),
        "last_price": to_decimal_string(book_data.get("ltp", 0)),
        "last_volume": int(book_data.get("ltq", 0)),
        "daily_volume": int(book_data.get("v", 0)),
        "avg_price": to_decimal_string(book_data.get("atp", 0)),
        "source": "fyers"
    }
    
    # Add optional fields
    if "ltt" in book_data:
        result["last_trade_timestamp"] = int(book_data["ltt"])
    
    logger.info(f"Transformed orderbook for {symbol}: {len(bids)} bids, {len(asks)} asks")
    return result


def transform_quote(response: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    """
    Transform Fyers quote response to internal format optimized for trading.
    
    Internal Format:
    {
        "symbol": "NSE:SBIN",
        "timestamp": 1640995200,
        "datetime": "2022-01-01T10:00:00Z",
        "last_price": "100.50",
        "bid_price": "100.45",
        "ask_price": "100.55",
        "spread": "0.10",
        "open_price": "100.20",
        "high_price": "100.80",
        "low_price": "100.10", 
        "prev_close": "100.00",
        "change": "0.50",
        "change_percent": 0.50,
        "volume": 1500000,
        "avg_price": "100.35",
        "upper_circuit": "110.00",
        "lower_circuit": "90.00",
        "source": "fyers"
    }
    """
    if not isinstance(response, dict) or response.get("s") != "ok":
        error_msg = response.get("message", "Invalid response")
        raise ValueError(f"Fyers API error: {error_msg}")
    
    # Extract quote data
    quotes = response.get("d", [])
    if not quotes:
        raise ValueError("No quote data in response")
    
    # Find matching quote
    quote_data = None
    fyers_symbol = map_symbol(symbol)
    
    for quote in quotes:
        if quote.get("n") == fyers_symbol:
            quote_data = quote
            break
    
    if not quote_data or quote_data.get("s") != "ok":
        raise ValueError(f"No valid quote data for symbol: {symbol}")
    
    values = quote_data.get("v", {})
    
    # Current timestamp
    now = datetime.now(timezone.utc)
    
    result = {
        "symbol": normalize_symbol(fyers_symbol),
        "timestamp": int(now.timestamp()),
        "datetime": now.isoformat(),
        "last_price": to_decimal_string(values.get("lp", 0)),
        "bid_price": to_decimal_string(values.get("bid", 0)),
        "ask_price": to_decimal_string(values.get("ask", 0)),
        "spread": to_decimal_string(values.get("spread", 0)),
        "open_price": to_decimal_string(values.get("open_price", 0)),
        "high_price": to_decimal_string(values.get("high_price", 0)),
        "low_price": to_decimal_string(values.get("low_price", 0)),
        "prev_close": to_decimal_string(values.get("prev_close_price", 0)),
        "change": to_decimal_string(values.get("ch", 0)),
        "change_percent": float(values.get("chp", 0)),
        "volume": int(values.get("volume", 0)),
        "avg_price": to_decimal_string(values.get("atp", 0)),
        "source": "fyers"
    }
    
    # Add circuit limits if available
    if "upper_ckt" in values:
        result["upper_circuit"] = to_decimal_string(values["upper_ckt"])
    if "lower_ckt" in values:
        result["lower_circuit"] = to_decimal_string(values["lower_ckt"])
    
    # Add exchange metadata
    if "exchange" in values:
        result["exchange"] = str(values["exchange"])
    if "fyToken" in values:
        result["fyers_token"] = str(values["fyToken"])
    
    logger.info(f"Transformed quote for {symbol}: LTP={result['last_price']}")
    return result


def transform_websocket_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform WebSocket message to internal format for real-time data.
    
    Args:
        message: Raw WebSocket message from Fyers
        
    Returns:
        Standardized real-time data format
    """
    msg_type = message.get("type", "")
    symbol = message.get("symbol", "")
    
    # Current timestamp
    now = datetime.now(timezone.utc)
    
    base_data = {
        "symbol": normalize_symbol(symbol) if symbol else "",
        "timestamp": int(now.timestamp()),
        "datetime": now.isoformat(),
        "message_type": msg_type,
        "source": "fyers_ws"
    }
    
    if msg_type == "sf":  # Symbol feed
        base_data.update({
            "last_price": to_decimal_string(message.get("ltp", 0)),
            "bid_price": to_decimal_string(message.get("bid_price", 0)),
            "ask_price": to_decimal_string(message.get("ask_price", 0)),
            "volume": int(message.get("vol_traded_today", 0)),
            "change": to_decimal_string(message.get("ch", 0)),
            "change_percent": float(message.get("chp", 0))
        })
    
    return base_data


# Export main functions
__all__ = [
    'map_symbol',
    'map_interval', 
    'normalize_symbol',
    'transform_ohlcv',
    'transform_orderbook',
    'transform_quote',
    'transform_websocket_message',
    'to_decimal_string'
]