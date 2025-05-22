"""
Market data repository for database operations.

This module provides a repository layer for market data operations,
including saving and retrieving market data.
"""

import logging
import json
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from datetime import datetime, timedelta
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import func, desc

from app.core.database import db_session, DatabaseType
from app.models.market_data import Instrument, OHLCV, Tick, OrderBookSnapshot
from app.consumers.utils.circuit_breaker import circuit_breaker

# Set up logging
logger = logging.getLogger(__name__)


class MarketDataRepository:
    """
    Repository for market data operations.
    
    Provides methods for storing and retrieving different types of market data,
    with efficient batch operations and error handling.
    """
    
    def __init__(self):
        """Initialize a new market data repository."""
        pass
    
    @circuit_breaker("db_operations")
    def get_or_create_instrument(self, symbol: str, exchange: Optional[str] = None) -> Instrument:
        """
        Get an instrument by symbol, creating it if it doesn't exist.
        
        Args:
            symbol: Instrument symbol
            exchange: Exchange where the instrument is traded
            
        Returns:
            Instrument model
            
        Raises:
            Exception: If getting or creating the instrument fails
        """
        try:
            with db_session(DatabaseType.TIMESCALEDB) as session:
                # Try to find existing instrument
                query = session.query(Instrument).filter(Instrument.symbol == symbol)
                if exchange:
                    query = query.filter(Instrument.exchange == exchange)
                    
                instrument = query.first()
                
                if instrument:
                    return instrument
                
                # Create new instrument if not found
                instrument = Instrument(
                    symbol=symbol,
                    exchange=exchange or "unknown",
                    name=symbol,  # Default name to symbol
                    asset_class="unknown",  # Default asset class
                    currency="USD",  # Default currency
                    active=True
                )
                
                session.add(instrument)
                session.commit()
                
                logger.debug(f"Created new instrument: {symbol}")
                
                return instrument
                
        except SQLAlchemyError as e:
            logger.error(f"Database error in get_or_create_instrument: {e}")
            raise
        except Exception as e:
            logger.error(f"Error in get_or_create_instrument: {e}")
            raise
    
    @circuit_breaker("db_operations")
    def save_ohlcv_batch(self, ohlcv_data: List[OHLCV]) -> None:
        """
        Save a batch of OHLCV data to the database.
        
        Args:
            ohlcv_data: List of OHLCV models to save
            
        Raises:
            Exception: If saving the data fails
        """
        if not ohlcv_data:
            return
            
        try:
            with db_session(DatabaseType.TIMESCALEDB) as session:
                # Add all models to session
                session.add_all(ohlcv_data)
                session.commit()
                
                logger.debug(f"Saved {len(ohlcv_data)} OHLCV records")
                
        except SQLAlchemyError as e:
            logger.error(f"Database error in save_ohlcv_batch: {e}")
            raise
        except Exception as e:
            logger.error(f"Error in save_ohlcv_batch: {e}")
            raise
    
    @circuit_breaker("db_operations")
    def save_tick_batch(self, tick_data: List[Tick]) -> None:
        """
        Save a batch of tick data to the database.
        
        Args:
            tick_data: List of Tick models to save
            
        Raises:
            Exception: If saving the data fails
        """
        if not tick_data:
            return
            
        try:
            with db_session(DatabaseType.TIMESCALEDB) as session:
                # Add all models to session
                session.add_all(tick_data)
                session.commit()
                
                logger.debug(f"Saved {len(tick_data)} Tick records")
                
        except SQLAlchemyError as e:
            logger.error(f"Database error in save_tick_batch: {e}")
            raise
        except Exception as e:
            logger.error(f"Error in save_tick_batch: {e}")
            raise
    
    @circuit_breaker("db_operations")
    def save_orderbook_batch(self, orderbook_data: List[OrderBookSnapshot]) -> None:
        """
        Save a batch of order book data to the database.
        
        Args:
            orderbook_data: List of OrderBookSnapshot models to save
            
        Raises:
            Exception: If saving the data fails
        """
        if not orderbook_data:
            return
            
        try:
            with db_session(DatabaseType.TIMESCALEDB) as session:
                # Add all models to session
                session.add_all(orderbook_data)
                session.commit()
                
                logger.debug(f"Saved {len(orderbook_data)} OrderBookSnapshot records")
                
        except SQLAlchemyError as e:
            logger.error(f"Database error in save_orderbook_batch: {e}")
            raise
        except Exception as e:
            logger.error(f"Error in save_orderbook_batch: {e}")
            raise
    
    @circuit_breaker("db_operations")
    def get_latest_ohlcv(self, symbol: str, interval: str) -> Optional[OHLCV]:
        """
        Get the latest OHLCV data for a symbol and interval.
        
        Args:
            symbol: Instrument symbol
            interval: Time interval (e.g., '1m', '1h', '1d')
            
        Returns:
            Latest OHLCV record or None if not found
            
        Raises:
            Exception: If retrieving the data fails
        """
        try:
            with db_session(DatabaseType.TIMESCALEDB) as session:
                # Join with Instrument to filter by symbol
                query = (
                    session.query(OHLCV)
                    .join(Instrument, OHLCV.instrument_id == Instrument.id)
                    .filter(Instrument.symbol == symbol)
                    .filter(OHLCV.interval == interval)
                    .order_by(OHLCV.timestamp.desc())
                    .limit(1)
                )
                
                return query.first()
                
        except SQLAlchemyError as e:
            logger.error(f"Database error in get_latest_ohlcv: {e}")
            raise
        except Exception as e:
            logger.error(f"Error in get_latest_ohlcv: {e}")
            raise
    
    @circuit_breaker("db_operations")
    def get_latest_orderbook(self, symbol: str) -> Optional[OrderBookSnapshot]:
        """
        Get the latest order book snapshot for a symbol.
        
        Args:
            symbol: Instrument symbol
            
        Returns:
            Latest OrderBookSnapshot record or None if not found
            
        Raises:
            Exception: If retrieving the data fails
        """
        try:
            with db_session(DatabaseType.TIMESCALEDB) as session:
                # Join with Instrument to filter by symbol
                query = (
                    session.query(OrderBookSnapshot)
                    .join(Instrument, OrderBookSnapshot.instrument_id == Instrument.id)
                    .filter(Instrument.symbol == symbol)
                    .order_by(OrderBookSnapshot.timestamp.desc())
                    .limit(1)
                )
                
                return query.first()
                
        except SQLAlchemyError as e:
            logger.error(f"Database error in get_latest_orderbook: {e}")
            raise
        except Exception as e:
            logger.error(f"Error in get_latest_orderbook: {e}")
            raise
    
    @circuit_breaker("db_operations")
    def get_ohlcv_range(
        self, 
        symbol: str, 
        interval: str, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[OHLCV]:
        """
        Get OHLCV data for a symbol and interval within a time range.
        
        Args:
            symbol: Instrument symbol
            interval: Time interval (e.g., '1m', '1h', '1d')
            start_time: Start of time range
            end_time: End of time range
            
        Returns:
            List of OHLCV records
            
        Raises:
            Exception: If retrieving the data fails
        """
        try:
            with db_session(DatabaseType.TIMESCALEDB) as session:
                # Join with Instrument to filter by symbol
                query = (
                    session.query(OHLCV)
                    .join(Instrument, OHLCV.instrument_id == Instrument.id)
                    .filter(Instrument.symbol == symbol)
                    .filter(OHLCV.interval == interval)
                    .filter(OHLCV.timestamp >= start_time)
                    .filter(OHLCV.timestamp <= end_time)
                    .order_by(OHLCV.timestamp)
                )
                
                return query.all()
                
        except SQLAlchemyError as e:
            logger.error(f"Database error in get_ohlcv_range: {e}")
            raise
        except Exception as e:
            logger.error(f"Error in get_ohlcv_range: {e}")
            raise
    
    @circuit_breaker("db_operations")
    def get_ticks_range(
        self, 
        symbol: str, 
        start_time: datetime, 
        end_time: datetime,
        limit: Optional[int] = None
    ) -> List[Tick]:
        """
        Get tick data for a symbol within a time range.
        
        Args:
            symbol: Instrument symbol
            start_time: Start of time range
            end_time: End of time range
            limit: Maximum number of records to return
            
        Returns:
            List of Tick records
            
        Raises:
            Exception: If retrieving the data fails
        """
        try:
            with db_session(DatabaseType.TIMESCALEDB) as session:
                # Join with Instrument to filter by symbol
                query = (
                    session.query(Tick)
                    .join(Instrument, Tick.instrument_id == Instrument.id)
                    .filter(Instrument.symbol == symbol)
                    .filter(Tick.timestamp >= start_time)
                    .filter(Tick.timestamp <= end_time)
                    .order_by(Tick.timestamp)
                )
                
                if limit:
                    query = query.limit(limit)
                
                return query.all()
                
        except SQLAlchemyError as e:
            logger.error(f"Database error in get_ticks_range: {e}")
            raise
        except Exception as e:
            logger.error(f"Error in get_ticks_range: {e}")
            raise
    
    @circuit_breaker("db_operations")
    def get_instrument_by_symbol(self, symbol: str) -> Optional[Instrument]:
        """
        Get an instrument by symbol.
        
        Args:
            symbol: Instrument symbol
            
        Returns:
            Instrument or None if not found
            
        Raises:
            Exception: If retrieving the instrument fails
        """
        try:
            with db_session(DatabaseType.TIMESCALEDB) as session:
                return session.query(Instrument).filter(Instrument.symbol == symbol).first()
                
        except SQLAlchemyError as e:
            logger.error(f"Database error in get_instrument_by_symbol: {e}")
            raise
        except Exception as e:
            logger.error(f"Error in get_instrument_by_symbol: {e}")
            raise
    
    # Methods needed for the tests
    @circuit_breaker("db_operations")
    def get_ohlcv_by_symbol(
        self,
        symbol: str,
        interval: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[OHLCV]:
        """
        Get OHLCV data for a specific symbol.
        
        Args:
            symbol: Instrument symbol
            interval: OHLCV interval (e.g., '1m', '1h', '1d')
            start_date: Start date for data range
            end_date: End date for data range
            limit: Maximum number of records to return
            offset: Number of records to skip
            
        Returns:
            List of OHLCV records
        """
        try:
            with db_session(DatabaseType.TIMESCALEDB) as session:
                # Join with Instrument to filter by symbol
                query = (
                    session.query(OHLCV)
                    .join(Instrument, OHLCV.instrument_id == Instrument.id)
                    .filter(Instrument.symbol == symbol)
                )
                
                # Apply filters
                if interval:
                    query = query.filter(OHLCV.interval == interval)
                
                if start_date:
                    query = query.filter(OHLCV.timestamp >= start_date)
                
                if end_date:
                    query = query.filter(OHLCV.timestamp <= end_date)
                
                # Apply sorting and pagination
                query = query.order_by(desc(OHLCV.timestamp)).limit(limit).offset(offset)
                
                return query.all()
                
        except SQLAlchemyError as e:
            logger.error(f"Database error in get_ohlcv_by_symbol: {e}")
            raise
        except Exception as e:
            logger.error(f"Error in get_ohlcv_by_symbol: {e}")
            raise
    
    @circuit_breaker("db_operations")
    def get_trades_by_symbol(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Tick]:
        """
        Get trade data for a specific symbol.
        
        Args:
            symbol: Instrument symbol
            start_date: Start date for data range
            end_date: End date for data range
            limit: Maximum number of records to return
            offset: Number of records to skip
            
        Returns:
            List of Tick records
        """
        try:
            with db_session(DatabaseType.TIMESCALEDB) as session:
                # Join with Instrument to filter by symbol
                query = (
                    session.query(Tick)
                    .join(Instrument, Tick.instrument_id == Instrument.id)
                    .filter(Instrument.symbol == symbol)
                )
                
                # Apply filters
                if start_date:
                    query = query.filter(Tick.timestamp >= start_date)
                
                if end_date:
                    query = query.filter(Tick.timestamp <= end_date)
                
                # Apply sorting and pagination
                query = query.order_by(desc(Tick.timestamp)).limit(limit).offset(offset)
                
                return query.all()
                
        except SQLAlchemyError as e:
            logger.error(f"Database error in get_trades_by_symbol: {e}")
            raise
        except Exception as e:
            logger.error(f"Error in get_trades_by_symbol: {e}")
            raise
    
    @circuit_breaker("db_operations")
    def get_orderbook_by_symbol(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1,
        offset: int = 0
    ) -> List[OrderBookSnapshot]:
        """
        Get order book data for a specific symbol.
        
        Args:
            symbol: Instrument symbol
            start_date: Start date for data range
            end_date: End date for data range
            limit: Maximum number of records to return
            offset: Number of records to skip
            
        Returns:
            List of OrderBookSnapshot records
        """
        try:
            with db_session(DatabaseType.TIMESCALEDB) as session:
                # Join with Instrument to filter by symbol
                query = (
                    session.query(OrderBookSnapshot)
                    .join(Instrument, OrderBookSnapshot.instrument_id == Instrument.id)
                    .filter(Instrument.symbol == symbol)
                )
                
                # Apply filters
                if start_date:
                    query = query.filter(OrderBookSnapshot.timestamp >= start_date)
                
                if end_date:
                    query = query.filter(OrderBookSnapshot.timestamp <= end_date)
                
                # Apply sorting and pagination
                query = query.order_by(desc(OrderBookSnapshot.timestamp)).limit(limit).offset(offset)
                
                return query.all()
                
        except SQLAlchemyError as e:
            logger.error(f"Database error in get_orderbook_by_symbol: {e}")
            raise
        except Exception as e:
            logger.error(f"Error in get_orderbook_by_symbol: {e}")
            raise