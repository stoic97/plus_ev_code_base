"""
WebSocket Protocol Definitions for Fyers API

This module defines the message types, data structures, and protocols for
the Fyers WebSocket API. It provides type hints and data validation for
WebSocket communication.
"""
from enum import Enum, auto
from typing import Dict, List, Optional, Union, Any, Callable, TypeVar
from dataclasses import dataclass
from datetime import datetime


class WebSocketMessageType(str, Enum):
    """WebSocket message types."""
    SYMBOL_FEED = "sf"
    DEPTH_FEED = "df"
    ORDER_UPDATE = "ou"
    TRADE_UPDATE = "tu"
    POSITION_UPDATE = "pu"
    ERROR = "error"


class WebSocketConnectionState(Enum):
    """WebSocket connection states."""
    DISCONNECTED = auto()
    CONNECTING = auto()
    AUTHENTICATING = auto()
    CONNECTED = auto()
    RECONNECTING = auto()
    ERROR = auto()


@dataclass
class WebSocketConfig:
    """WebSocket configuration settings."""
    base_url: str = "wss://api.fyers.in/socket/v2/"
    max_retries: int = 50
    retry_delay: float = 1.0  # Initial delay in seconds
    max_retry_delay: float = 30.0  # Maximum delay between retries
    heartbeat_interval: float = 30.0  # Seconds between heartbeats
    connection_timeout: float = 10.0  # Connection timeout in seconds
    max_symbols_per_connection: int = 5000  # Fyers limit


@dataclass
class SubscriptionRequest:
    """Subscription request message."""
    symbols: List[str]
    data_type: WebSocketMessageType = WebSocketMessageType.SYMBOL_FEED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for WebSocket message."""
        return {
            "type": self.data_type.value,
            "symbols": self.symbols
        }


@dataclass
class MarketDataUpdate:
    """Market data update message."""
    symbol: str
    last_price: float
    volume: int
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    change: float
    change_percent: float
    
    @classmethod
    def from_message(cls, message: Dict[str, Any]) -> "MarketDataUpdate":
        """
        Create from WebSocket message.
        
        Handles both old and new message formats:
        Old: {"symbol": "...", "ltp": 100.0, "volume": 1000, "timestamp": 1234567890000}
        New: {"symbol": "...", "last_price": 100.0, "volume": 1000, "timestamp": 1234567890}
        """
        # Handle timestamp (old format uses milliseconds)
        ts = message.get("timestamp", 0)
        if ts > 1e12:  # If timestamp is in milliseconds
            ts = ts / 1000
        
        return cls(
            symbol=message.get("symbol", ""),
            last_price=float(message.get("last_price") or message.get("ltp", 0)),
            volume=int(message.get("volume", 0)),
            timestamp=datetime.fromtimestamp(ts),
            open=float(message.get("open") or message.get("open_price", 0)),
            high=float(message.get("high") or message.get("high_price", 0)),
            low=float(message.get("low") or message.get("low_price", 0)),
            close=float(message.get("close") or message.get("close_price", 0)),
            change=float(message.get("change", 0)),
            change_percent=float(message.get("change_percent") or message.get("change_percentage", 0))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "last_price": self.last_price,
            "volume": self.volume,
            "timestamp": self.timestamp.timestamp(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "change": self.change,
            "change_percent": self.change_percent
        }


@dataclass
class MarketDepthUpdate:
    """Market depth (order book) update message."""
    symbol: str
    timestamp: datetime
    bids: List[Dict[str, float]]  # List of {price, quantity} dicts
    asks: List[Dict[str, float]]  # List of {price, quantity} dicts

    @classmethod
    def from_message(cls, message: Dict[str, Any]) -> 'MarketDepthUpdate':
        """
        Create instance from WebSocket message.
        
        Args:
            message: WebSocket message containing market depth data
            
        Returns:
            MarketDepthUpdate instance
            
        Raises:
            TypeError: If bids or asks have invalid format
            ValueError: If price or quantity values are invalid
        """
        # Validate bids and asks format
        bids = message.get("bids", [])
        asks = message.get("asks", [])
        
        if not isinstance(bids, list) or not isinstance(asks, list):
            raise TypeError("Bids and asks must be lists")
        
        # Convert bids and asks to proper format with validation
        try:
            formatted_bids = []
            for bid in bids:
                if not isinstance(bid, (list, tuple)) or len(bid) < 2:
                    raise TypeError("Each bid must be a list/tuple with price and quantity")
                formatted_bids.append({
                    "price": float(bid[0]),
                    "quantity": float(bid[1])
                })
            
            formatted_asks = []
            for ask in asks:
                if not isinstance(ask, (list, tuple)) or len(ask) < 2:
                    raise TypeError("Each ask must be a list/tuple with price and quantity")
                formatted_asks.append({
                    "price": float(ask[0]),
                    "quantity": float(ask[1])
                })
            
            # Create instance with validated data
            return cls(
                symbol=message.get("symbol", ""),
                timestamp=datetime.fromtimestamp(message.get("timestamp", 0) / 1000),
                bids=formatted_bids,
                asks=formatted_asks
            )
            
        except (IndexError, ValueError, TypeError) as e:
            raise TypeError(f"Invalid market depth data format: {str(e)}")


class WebSocketError(Exception):
    """WebSocket error with code and message."""
    
    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(f"WebSocket error {code}: {message}")


# Type definitions
T = TypeVar('T')
WebSocketMessageHandler = Callable[[Dict[str, Any]], None]

# Type alias for WebSocket messages
WebSocketMessage = Union[MarketDataUpdate, MarketDepthUpdate, WebSocketError] 