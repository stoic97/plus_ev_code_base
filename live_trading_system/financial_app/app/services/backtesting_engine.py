"""
Backtesting Engine for Trading Strategies Application.

This service orchestrates complete strategy backtesting by replaying strategies
against historical market data. It integrates with existing paper trading
infrastructure to provide realistic execution simulation and comprehensive
performance analysis.

Key Features:
- Strategy replay using historical OHLCV data from MCX Crude Oil
- Integration with existing StrategyEngineService for signal generation
- Uses existing Signal model - no new signal creation
- Realistic execution simulation with slippage and commission
- Comprehensive performance metrics calculation
- Trade-by-trade analysis and reporting
- Risk metrics and drawdown analysis
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
import uuid

from app.core.error_handling import (
    OperationalError,
    ValidationError,
    DatabaseConnectionError
)
from app.services.backtesting_data_service import BacktestingDataService
from app.services.strategy_engine import StrategyEngineService
from app.models.strategy import Signal, Direction, TimeframeValue, SetupQualityGrade

# Set up logging
logger = logging.getLogger(__name__)


class BacktestStatus(Enum):
    """Backtest execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BacktestConfig:
    """Configuration for backtest execution."""
    strategy_id: int
    start_date: datetime
    end_date: datetime
    initial_capital: float = 1000000.0  # 10 Lakh INR default
    commission_per_trade: float = 40.0  # MCX Crude Oil commission
    slippage_bps: float = 3.0  # Crude oil has higher slippage than equity
    max_position_size: float = 0.15  # 15% max per position for commodities
    risk_free_rate: float = 0.06  # 6% annual risk-free rate
    
    # Crude Oil specific configuration
    lot_size: int = 100  # MCX Crude Oil lot size (100 barrels)
    tick_size: float = 1.0  # Minimum price movement (1 INR)
    margin_requirement: float = 0.08  # 8% margin requirement for crude oil
    
    # Advanced configuration
    enable_slippage: bool = True
    enable_commission: bool = True
    market_impact_model: str = "linear"
    execution_delay_ms: int = 150  # Higher for commodities
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.start_date >= self.end_date:
            raise ValidationError("Start date must be before end date")
        if self.initial_capital <= 0:
            raise ValidationError("Initial capital must be positive")
        if not 0 <= self.max_position_size <= 1:
            raise ValidationError("Max position size must be between 0 and 1")


@dataclass
class BacktestTrade:
    """Individual trade record during backtesting - mirrors actual trade execution."""
    trade_id: str
    signal_id: int
    strategy_id: int
    instrument: str
    direction: str  # "long" or "short"
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    quantity: int = 1
    commission: float = 0.0
    slippage: float = 0.0
    pnl_points: Optional[float] = None
    pnl_inr: Optional[float] = None
    setup_quality: Optional[str] = None
    setup_score: Optional[float] = None
    original_signal: Optional[Signal] = None  # Reference to actual signal
    
    @property
    def is_open(self) -> bool:
        """Check if trade is still open."""
        return self.exit_time is None
    
    @property
    def duration_minutes(self) -> Optional[int]:
        """Get trade duration in minutes."""
        if self.exit_time is None:
            return None
        return int((self.exit_time - self.entry_time).total_seconds() / 60)


@dataclass
class BacktestMetrics:
    """Comprehensive backtest performance metrics."""
    # Basic performance
    total_return_pct: float = 0.0
    annual_return_pct: float = 0.0
    total_pnl_inr: float = 0.0
    
    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_days: int = 0
    volatility_annual_pct: float = 0.0
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate_pct: float = 0.0
    avg_win_inr: float = 0.0
    avg_loss_inr: float = 0.0
    largest_win_inr: float = 0.0
    largest_loss_inr: float = 0.0
    profit_factor: float = 0.0
    
    # Cost analysis
    total_commission_inr: float = 0.0
    total_slippage_inr: float = 0.0
    total_costs_inr: float = 0.0
    
    # Time analysis
    avg_trade_duration_minutes: float = 0.0
    trades_per_day: float = 0.0
    
    # Additional metrics
    calmar_ratio: float = 0.0
    kelly_criterion: float = 0.0
    expectancy_inr: float = 0.0


@dataclass
class BacktestResult:
    """Complete backtest result containing all analysis."""
    backtest_id: str
    config: BacktestConfig
    status: BacktestStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Core results
    trades: List[BacktestTrade] = field(default_factory=list)
    metrics: Optional[BacktestMetrics] = None
    equity_curve: Optional[pd.DataFrame] = None
    
    # Analysis data
    monthly_returns: Optional[pd.DataFrame] = None
    drawdown_periods: Optional[List[Dict[str, Any]]] = None
    trade_analysis: Optional[Dict[str, Any]] = None
    
    # Error information
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Get backtest execution duration."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def is_complete(self) -> bool:
        """Check if backtest completed successfully."""
        return self.status == BacktestStatus.COMPLETED


class BacktestingEngine:
    """
    Main backtesting engine that orchestrates strategy replay and analysis.
    
    This engine integrates multiple services to provide comprehensive backtesting:
    - Uses BacktestingDataService for historical data
    - Leverages existing StrategyEngineService for signal generation
    - Uses existing Signal model - no custom signal creation
    - Calculates comprehensive performance metrics
    """
    
    def __init__(self, 
                 data_service: Optional[BacktestingDataService] = None,
                 strategy_service: Optional[StrategyEngineService] = None):
        """
        Initialize the backtesting engine.
        
        Args:
            data_service: Historical data service
            strategy_service: Strategy engine service for signal generation
        """
        self.data_service = data_service or BacktestingDataService()
        self.strategy_service = strategy_service
        
        # Track running backtests
        self._running_backtests: Dict[str, BacktestResult] = {}
        
        logger.info("BacktestingEngine initialized")
    
    def run_backtest(self, config: BacktestConfig) -> BacktestResult:
        """
        Execute a complete backtest for the given configuration.
        
        Args:
            config: Backtest configuration parameters
            
        Returns:
            BacktestResult with complete analysis
        """
        backtest_id = str(uuid.uuid4())
        logger.info(f"Starting backtest {backtest_id} for strategy {config.strategy_id}")
        
        result = BacktestResult(
            backtest_id=backtest_id,
            config=config,
            status=BacktestStatus.PENDING,
            start_time=datetime.now()
        )
        
        try:
            self._running_backtests[backtest_id] = result
            result.status = BacktestStatus.RUNNING
            
            # Step 1: Load historical data
            logger.info(f"Loading historical data for period {config.start_date} to {config.end_date}")
            historical_data = self._load_historical_data(config)
            
            if len(historical_data) == 0:
                raise ValidationError(f"No historical data found for period")
            
            logger.info(f"Loaded {len(historical_data)} data points for backtesting")
            
            # Step 2: Initialize trading state
            trading_state = self._initialize_trading_state(config)
            
            # Step 3: Execute strategy replay
            logger.info("Starting strategy replay...")
            trades = self._replay_strategy(config, historical_data, trading_state)
            result.trades = trades
            
            # Step 4: Calculate metrics
            logger.info("Calculating performance metrics...")
            result.metrics = self._calculate_metrics(trades, config)
            
            # Step 5: Generate analysis
            result.equity_curve = self._generate_equity_curve(trades, config)
            result.monthly_returns = self._calculate_monthly_returns(result.equity_curve)
            result.drawdown_periods = self._analyze_drawdown_periods(result.equity_curve)
            result.trade_analysis = self._analyze_trades(trades)
            
            # Mark as completed
            result.status = BacktestStatus.COMPLETED
            result.end_time = datetime.now()
            
            logger.info(f"Backtest {backtest_id} completed successfully")
            logger.info(f"Results: {len(trades)} trades, {result.metrics.total_return_pct:.2f}% return")
            
            return result
            
        except Exception as e:
            logger.error(f"Backtest {backtest_id} failed: {e}")
            result.status = BacktestStatus.FAILED
            result.end_time = datetime.now()
            result.error_message = str(e)
            
            if isinstance(e, (ValidationError, OperationalError)):
                raise
            else:
                raise OperationalError(f"Backtest execution failed: {str(e)}")
                
        finally:
            if backtest_id in self._running_backtests:
                del self._running_backtests[backtest_id]
    
    def _load_historical_data(self, config: BacktestConfig) -> pd.DataFrame:
        """Load historical data for the backtest period."""
        try:
            # Add buffer for strategy initialization
            buffer_days = 30
            buffered_start = config.start_date - timedelta(days=buffer_days)
            
            # Load all data and filter by date
            all_data = self.data_service.load_historical_data()
            data = all_data[
                (all_data['timestamp'] >= buffered_start) & 
                (all_data['timestamp'] <= config.end_date)
            ].copy()
            
            if len(data) == 0:
                raise ValidationError("No historical data available")
            
            # Validate required columns
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = set(required_columns) - set(data.columns)
            if missing_columns:
                raise ValidationError(f"Missing columns: {missing_columns}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            raise OperationalError(f"Failed to load historical data: {str(e)}")
    
    def _initialize_trading_state(self, config: BacktestConfig) -> Dict[str, Any]:
        """Initialize the trading state for backtest execution."""
        return {
            'cash': config.initial_capital,
            'positions': {},
            'open_trades': {},
            'trade_counter': 0,
            'equity_history': [],
            'generated_signals': []  # Track signals created during backtest
        }
    
    def _replay_strategy(self, config: BacktestConfig, data: pd.DataFrame, 
                        trading_state: Dict[str, Any]) -> List[BacktestTrade]:
        """
        Replay strategy through historical data using existing StrategyEngineService.
        
        This method uses the actual StrategyEngineService to generate real signals,
        then simulates execution of those signals.
        """
        trades = []
        
        # Filter data to actual backtest period
        backtest_data = data[data['timestamp'] >= config.start_date].copy()
        logger.info(f"Replaying strategy through {len(backtest_data)} data points")
        
        for idx, row in backtest_data.iterrows():
            current_time = row['timestamp']
            current_price = row['close']
            
            try:
                # Update equity tracking
                self._update_equity_tracking(trading_state, current_time, current_price, config)
                
                # Generate signals using existing StrategyEngineService
                signals = self._generate_signals_for_tick(config, row, data.loc[:idx])
                
                # Process signals and create trades
                for signal in signals:
                    trade = self._execute_signal_as_trade(signal, config, row, trading_state)
                    if trade:
                        trades.append(trade)
                
                # Check exit conditions for open trades
                exits = self._check_exit_conditions(trading_state['open_trades'], row, config)
                for exit_trade in exits:
                    self._close_trade(exit_trade, row, config, trading_state)
                
                # Risk management
                self._apply_risk_management(trading_state, config, current_price)
                
            except Exception as e:
                logger.warning(f"Error processing tick at {current_time}: {e}")
                continue
        
        # Close remaining trades
        for trade in list(trading_state['open_trades'].values()):
            final_row = backtest_data.iloc[-1]
            self._close_trade(trade, final_row, config, trading_state, exit_reason="backtest_end")
            
        logger.info(f"Strategy replay completed. Generated {len(trades)} trades")
        return trades
    
    def _generate_signals_for_tick(self, config: BacktestConfig, current_row: pd.Series, 
                                  historical_data: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals for the current tick using existing StrategyEngineService.
        
        Returns actual Signal objects from the existing model.
        """
        signals = []
        
        try:
            # Skip if we don't have the strategy service
            if not self.strategy_service:
                return []
            
            # Need sufficient data for analysis
            if len(historical_data) < 200:
                return []
            
            # Prepare market data for strategy service
            market_data = self._prepare_market_data_for_strategy(historical_data)
            if not market_data:
                return []
            
            # Use StrategyEngineService methods for comprehensive analysis
            try:
                # Step 1: Analyze timeframes
                timeframe_analysis = self.strategy_service.analyze_timeframes(
                    config.strategy_id, market_data
                )
                
                if not timeframe_analysis.aligned:
                    return []
                
                # Step 2: Analyze market state
                market_state = self.strategy_service.analyze_market_state(
                    config.strategy_id, market_data
                )
                
                # Step 3: Evaluate setup quality
                entry_data = {
                    'near_ma': True,
                    'risk_reward': 3.0,
                    'price_action_confirmation': True,
                    'close': [current_row['close']]
                }
                
                setup_quality = self.strategy_service.evaluate_setup_quality(
                    config.strategy_id, timeframe_analysis, market_state, entry_data
                )
                
                # Only trade high-quality setups (B+ and above)
                if setup_quality.grade.value in ['a_plus', 'a', 'b_plus', 'b']:
                    # Determine direction based on analysis
                    direction = Direction.LONG if timeframe_analysis.primary_direction == "up" else Direction.SHORT
                    
                    # Generate actual signal using StrategyEngineService
                    signal = self.strategy_service.generate_signal(
                        config.strategy_id,
                        timeframe_analysis,
                        market_state,
                        setup_quality,
                        market_data,
                        'CRUDEOIL',  # MCX Crude Oil
                        direction
                    )
                    
                    signals.append(signal)
                    
            except Exception as e:
                logger.warning(f"Error using strategy service: {e}")
                
        except Exception as e:
            logger.warning(f"Error generating signals: {e}")
            
        return signals
    
    def _prepare_market_data_for_strategy(self, historical_data: pd.DataFrame) -> Dict[TimeframeValue, Dict[str, Any]]:
        """
        Prepare market data in the format expected by StrategyEngineService.
        
        Returns data with proper TimeframeValue enum keys.
        """
        try:
            recent_data = historical_data.tail(200).copy()
            
            market_data = {
                TimeframeValue.ONE_MIN: {
                    'close': recent_data['close'].tolist(),
                    'high': recent_data['high'].tolist(),
                    'low': recent_data['low'].tolist(),
                    'open': recent_data['open'].tolist(),
                    'volume': recent_data['volume'].tolist()
                }
            }
            
            # Add moving averages if we have enough data
            if len(recent_data) >= 21:
                ma21 = recent_data['close'].rolling(window=21).mean().fillna(method='bfill')
                market_data[TimeframeValue.ONE_MIN]['ma21'] = ma21.tolist()
            
            return market_data
            
        except Exception as e:
            logger.warning(f"Error preparing market data: {e}")
            return {}
    
    def _execute_signal_as_trade(self, signal: Signal, config: BacktestConfig, 
                                current_row: pd.Series, trading_state: Dict[str, Any]) -> Optional[BacktestTrade]:
        """
        Execute a trading signal as a backtest trade.
        
        Converts the actual Signal object into a BacktestTrade.
        """
        try:
            # Check capacity
            if len(trading_state['open_trades']) >= 3:
                return None
            
            # Calculate position size
            position_size = self._calculate_position_size(signal, config, trading_state)
            if position_size <= 0:
                return None
            
            # Apply slippage and commission
            direction_str = signal.direction.value.lower()
            execution_price = self._apply_slippage(signal.entry_price, direction_str, config)
            commission = config.commission_per_trade if config.enable_commission else 0.0
            
            # Create trade from signal
            trade_id = f"BT_{config.strategy_id}_{trading_state['trade_counter']}"
            trading_state['trade_counter'] += 1
            
            trade = BacktestTrade(
                trade_id=trade_id,
                signal_id=signal.id,
                strategy_id=config.strategy_id,
                instrument=signal.instrument,
                direction=direction_str,
                entry_time=current_row['timestamp'],
                entry_price=execution_price,
                quantity=position_size,
                commission=commission,
                slippage=abs(execution_price - signal.entry_price),
                setup_quality=signal.setup_quality.value if signal.setup_quality else 'c',
                setup_score=signal.setup_score or 50.0,
                original_signal=signal
            )
            
            # Update state
            trading_state['open_trades'][trade_id] = trade
            trading_state['cash'] -= commission
            trading_state['generated_signals'].append(signal)
            
            logger.debug(f"Executed {direction_str} trade {trade_id} from signal {signal.id}")
            return trade
            
        except Exception as e:
            logger.error(f"Error executing signal as trade: {e}")
            return None
    
    def _check_exit_conditions(self, open_trades: Dict[str, BacktestTrade], 
                              current_row: pd.Series, config: BacktestConfig) -> List[BacktestTrade]:
        """Check exit conditions for open trades using signal parameters."""
        exits = []
        current_price = current_row['close']
        current_time = current_row['timestamp']
        
        for trade in open_trades.values():
            exit_reason = None
            
            # Use exit conditions from the original signal if available
            if trade.original_signal:
                signal = trade.original_signal
                
                # Check take profit
                if signal.take_profit_price:
                    if trade.direction == 'long' and current_price >= signal.take_profit_price:
                        exit_reason = "take_profit"
                    elif trade.direction == 'short' and current_price <= signal.take_profit_price:
                        exit_reason = "take_profit"
                
                # Check stop loss
                if not exit_reason and signal.stop_loss_price:
                    if trade.direction == 'long' and current_price <= signal.stop_loss_price:
                        exit_reason = "stop_loss"
                    elif trade.direction == 'short' and current_price >= signal.stop_loss_price:
                        exit_reason = "stop_loss"
            
            # Fallback to default crude oil exit conditions
            if not exit_reason:
                # Calculate P&L for default exits
                if trade.direction == 'long':
                    pnl_per_barrel = current_price - trade.entry_price
                else:
                    pnl_per_barrel = trade.entry_price - current_price
                
                # Default exit conditions for crude oil
                if pnl_per_barrel >= 50.0:  # +50 INR profit target
                    exit_reason = "take_profit"
                elif pnl_per_barrel <= -30.0:  # -30 INR stop loss
                    exit_reason = "stop_loss"
                elif (current_time - trade.entry_time).total_seconds() >= 8 * 3600:  # 8 hours
                    exit_reason = "time_exit"
            
            if exit_reason:
                trade.exit_time = current_time
                trade.exit_price = current_price
                trade.exit_reason = exit_reason
                exits.append(trade)
        
        return exits
    
    def _close_trade(self, trade: BacktestTrade, current_row: pd.Series, 
                    config: BacktestConfig, trading_state: Dict[str, Any], 
                    exit_reason: Optional[str] = None):
        """Close a trade and update state."""
        if exit_reason:
            trade.exit_reason = exit_reason
            trade.exit_time = current_row['timestamp']
            trade.exit_price = current_row['close']
        
        # Calculate P&L
        if trade.direction == 'long':
            trade.pnl_points = trade.exit_price - trade.entry_price
        else:
            trade.pnl_points = trade.entry_price - trade.exit_price
        
        # Convert to INR for crude oil
        trade.pnl_inr = (trade.pnl_points * config.lot_size * trade.quantity) - trade.commission
        
        # Update cash
        trading_state['cash'] += trade.pnl_inr
        
        # Remove from open trades
        if trade.trade_id in trading_state['open_trades']:
            del trading_state['open_trades'][trade.trade_id]
        
        logger.debug(f"Closed trade {trade.trade_id}: {trade.pnl_inr:.0f} INR")
    
    def _apply_slippage(self, price: float, direction: str, config: BacktestConfig) -> float:
        """Apply slippage to execution price."""
        if not config.enable_slippage:
            return price
        
        slippage_amount = price * (config.slippage_bps / 10000)
        
        if direction == 'long':
            return price + slippage_amount
        else:
            return price - slippage_amount
    
    def _calculate_position_size(self, signal: Signal, config: BacktestConfig, 
                                trading_state: Dict[str, Any]) -> int:
        """Calculate position size for crude oil."""
        # Use signal's position size if available
        if signal.position_size and signal.position_size > 0:
            return min(signal.position_size, 5)  # Cap at 5 lots
        
        # Fallback calculation
        available_capital = trading_state['cash']
        max_position_value = available_capital * config.max_position_size
        
        # Crude oil position sizing
        contract_value = signal.entry_price * config.lot_size
        margin_per_lot = contract_value * config.margin_requirement
        
        max_lots = int(max_position_value / margin_per_lot) if margin_per_lot > 0 else 1
        return max(1, min(max_lots, 5))  # 1-5 lots
    
    def _apply_risk_management(self, trading_state: Dict[str, Any], 
                              config: BacktestConfig, current_price: float):
        """Apply risk management rules."""
        total_exposure = 0.0
        for trade in trading_state['open_trades'].values():
            exposure = trade.entry_price * config.lot_size * trade.quantity
            total_exposure += exposure
        
        max_exposure = config.initial_capital * 0.5
        if total_exposure > max_exposure:
            logger.warning(f"Total exposure {total_exposure:.0f} exceeds limit")
    
    def _update_equity_tracking(self, trading_state: Dict[str, Any], 
                               current_time: datetime, current_price: float, config: BacktestConfig):
        """Update equity curve tracking."""
        current_equity = trading_state['cash']
        
        # Add unrealized P&L from open trades
        for trade in trading_state['open_trades'].values():
            if trade.direction == 'long':
                unrealized_pnl = current_price - trade.entry_price
            else:
                unrealized_pnl = trade.entry_price - current_price
            
            unrealized_pnl_inr = unrealized_pnl * config.lot_size * trade.quantity
            current_equity += unrealized_pnl_inr
        
        trading_state['equity_history'].append({
            'timestamp': current_time,
            'equity': current_equity,
            'cash': trading_state['cash'],
            'unrealized_pnl': current_equity - trading_state['cash']
        })
    
    def _calculate_metrics(self, trades: List[BacktestTrade], config: BacktestConfig) -> BacktestMetrics:
        """Calculate comprehensive performance metrics."""
        metrics = BacktestMetrics()
        
        if not trades:
            return metrics
        
        closed_trades = [t for t in trades if not t.is_open]
        metrics.total_trades = len(closed_trades)
        
        if metrics.total_trades == 0:
            return metrics
        
        # P&L analysis
        total_pnl = sum(t.pnl_inr for t in closed_trades if t.pnl_inr is not None)
        metrics.total_pnl_inr = total_pnl
        metrics.total_return_pct = (total_pnl / config.initial_capital) * 100
        
        # Win/Loss analysis
        winning_trades = [t for t in closed_trades if t.pnl_inr and t.pnl_inr > 0]
        losing_trades = [t for t in closed_trades if t.pnl_inr and t.pnl_inr <= 0]
        
        metrics.winning_trades = len(winning_trades)
        metrics.losing_trades = len(losing_trades)
        metrics.win_rate_pct = (metrics.winning_trades / metrics.total_trades) * 100
        
        if winning_trades:
            metrics.avg_win_inr = sum(t.pnl_inr for t in winning_trades) / len(winning_trades)
            metrics.largest_win_inr = max(t.pnl_inr for t in winning_trades)
        
        if losing_trades:
            metrics.avg_loss_inr = sum(t.pnl_inr for t in losing_trades) / len(losing_trades)
            metrics.largest_loss_inr = min(t.pnl_inr for t in losing_trades)
        
        # Profit factor
        total_wins = sum(t.pnl_inr for t in winning_trades) if winning_trades else 0
        total_losses = abs(sum(t.pnl_inr for t in losing_trades)) if losing_trades else 1
        metrics.profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Cost analysis
        metrics.total_commission_inr = sum(t.commission for t in closed_trades)
        metrics.total_slippage_inr = sum(t.slippage * config.lot_size for t in closed_trades)
        metrics.total_costs_inr = metrics.total_commission_inr + metrics.total_slippage_inr
        
        # Time analysis
        durations = [t.duration_minutes for t in closed_trades if t.duration_minutes is not None]
        if durations:
            metrics.avg_trade_duration_minutes = sum(durations) / len(durations)
        
        period_days = (config.end_date - config.start_date).days
        metrics.trades_per_day = metrics.total_trades / max(period_days, 1)
        
        # Expectancy
        metrics.expectancy_inr = total_pnl / metrics.total_trades if metrics.total_trades > 0 else 0
        
        # Risk metrics (simplified)
        if len(closed_trades) > 1:
            returns = [t.pnl_inr / config.initial_capital for t in closed_trades]
            avg_return = sum(returns) / len(returns)
            return_std = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
            
            if return_std > 0:
                annual_return = avg_return * 252
                annual_std = return_std * (252 ** 0.5)
                metrics.sharpe_ratio = (annual_return - config.risk_free_rate) / annual_std
        
        # Annual return calculation
        period_years = (config.end_date - config.start_date).days / 365.0
        if period_years > 0:
            metrics.annual_return_pct = ((1 + metrics.total_return_pct / 100) ** (1 / period_years) - 1) * 100
        
        return metrics
    
    def _generate_equity_curve(self, trades: List[BacktestTrade], config: BacktestConfig) -> pd.DataFrame:
        """Generate equity curve from trades."""
        equity_data = []
        current_equity = config.initial_capital
        
        # Add initial point
        start_time = config.start_date
        equity_data.append({
            'timestamp': start_time,
            'equity': current_equity,
            'drawdown_pct': 0.0,
            'running_max': current_equity
        })
        
        # Track running maximum for drawdown calculation
        running_max = current_equity
        
        # Process closed trades in chronological order
        closed_trades = [t for t in trades if not t.is_open]
        closed_trades.sort(key=lambda x: x.exit_time if x.exit_time else x.entry_time)
        
        for trade in closed_trades:
            if trade.exit_time and trade.pnl_inr is not None:
                current_equity += trade.pnl_inr
                running_max = max(running_max, current_equity)
                
                drawdown_pct = ((running_max - current_equity) / running_max) * 100 if running_max > 0 else 0
                
                equity_data.append({
                    'timestamp': trade.exit_time,
                    'equity': current_equity,
                    'drawdown_pct': drawdown_pct,
                    'running_max': running_max
                })
        
        return pd.DataFrame(equity_data)
    
    def _calculate_monthly_returns(self, equity_curve: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Calculate monthly returns from equity curve."""
        if equity_curve is None or len(equity_curve) < 2:
            return None
        
        try:
            # Resample to monthly data
            monthly_data = equity_curve.set_index('timestamp').resample('M').last()
            monthly_data['monthly_return'] = monthly_data['equity'].pct_change() * 100
            monthly_data['month'] = monthly_data.index.strftime('%Y-%m')
            
            return monthly_data[['equity', 'monthly_return', 'month']].reset_index()
        except Exception as e:
            logger.error(f"Error calculating monthly returns: {e}")
            return None
    
    def _analyze_drawdown_periods(self, equity_curve: Optional[pd.DataFrame]) -> Optional[List[Dict[str, Any]]]:
        """Analyze drawdown periods from equity curve."""
        if equity_curve is None or len(equity_curve) < 2:
            return None
        
        try:
            drawdown_periods = []
            in_drawdown = False
            drawdown_start = None
            max_drawdown = 0.0
            
            for _, row in equity_curve.iterrows():
                if row['drawdown_pct'] > 0:
                    if not in_drawdown:
                        # Start of new drawdown
                        in_drawdown = True
                        drawdown_start = row['timestamp']
                        max_drawdown = row['drawdown_pct']
                    else:
                        # Continue drawdown
                        max_drawdown = max(max_drawdown, row['drawdown_pct'])
                else:
                    if in_drawdown:
                        # End of drawdown
                        duration_days = (row['timestamp'] - drawdown_start).days
                        drawdown_periods.append({
                            'start_date': drawdown_start,
                            'end_date': row['timestamp'],
                            'duration_days': duration_days,
                            'max_drawdown_pct': max_drawdown
                        })
                        in_drawdown = False
            
            return drawdown_periods
        except Exception as e:
            logger.error(f"Error analyzing drawdown periods: {e}")
            return None
    
    def _analyze_trades(self, trades: List[BacktestTrade]) -> Optional[Dict[str, Any]]:
        """Analyze trade patterns and statistics."""
        if not trades:
            return None
        
        closed_trades = [t for t in trades if not t.is_open]
        if not closed_trades:
            return None
        
        try:
            analysis = {
                'total_trades': len(closed_trades),
                'trades_by_direction': {
                    'long': len([t for t in closed_trades if t.direction == 'long']),
                    'short': len([t for t in closed_trades if t.direction == 'short'])
                },
                'trades_by_setup_quality': {},
                'avg_trade_duration_hours': 0.0,
                'best_trade': None,
                'worst_trade': None,
                'consecutive_wins': 0,
                'consecutive_losses': 0
            }
            
            # Group by setup quality
            quality_groups = {}
            for trade in closed_trades:
                quality = trade.setup_quality or 'unknown'
                if quality not in quality_groups:
                    quality_groups[quality] = []
                quality_groups[quality].append(trade)
            
            for quality, quality_trades in quality_groups.items():
                winning = [t for t in quality_trades if t.pnl_inr and t.pnl_inr > 0]
                analysis['trades_by_setup_quality'][quality] = {
                    'total': len(quality_trades),
                    'winning': len(winning),
                    'win_rate': len(winning) / len(quality_trades) * 100 if quality_trades else 0,
                    'avg_pnl': sum(t.pnl_inr for t in quality_trades if t.pnl_inr) / len(quality_trades)
                }
            
            # Duration analysis
            durations = [t.duration_minutes for t in closed_trades if t.duration_minutes is not None]
            if durations:
                analysis['avg_trade_duration_hours'] = sum(durations) / len(durations) / 60
            
            # Best and worst trades
            pnl_trades = [t for t in closed_trades if t.pnl_inr is not None]
            if pnl_trades:
                best_trade = max(pnl_trades, key=lambda x: x.pnl_inr)
                worst_trade = min(pnl_trades, key=lambda x: x.pnl_inr)
                
                analysis['best_trade'] = {
                    'trade_id': best_trade.trade_id,
                    'pnl_inr': best_trade.pnl_inr,
                    'direction': best_trade.direction,
                    'setup_quality': best_trade.setup_quality
                }
                
                analysis['worst_trade'] = {
                    'trade_id': worst_trade.trade_id,
                    'pnl_inr': worst_trade.pnl_inr,
                    'direction': worst_trade.direction,
                    'setup_quality': worst_trade.setup_quality
                }
            
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing trades: {e}")
            return None
    
    def get_backtest_status(self, backtest_id: str) -> Optional[BacktestResult]:
        """Get status of a running backtest."""
        return self._running_backtests.get(backtest_id)
    
    def cancel_backtest(self, backtest_id: str) -> bool:
        """Cancel a running backtest."""
        if backtest_id in self._running_backtests:
            result = self._running_backtests[backtest_id]
            result.status = BacktestStatus.CANCELLED
            result.end_time = datetime.now()
            del self._running_backtests[backtest_id]
            logger.info(f"Backtest {backtest_id} cancelled")
            return True
        return False
    
    def list_running_backtests(self) -> List[str]:
        """List all currently running backtest IDs."""
        return list(self._running_backtests.keys())


# Utility functions for backtesting operations
def create_backtesting_engine(data_service: Optional[BacktestingDataService] = None,
                             strategy_service: Optional[StrategyEngineService] = None) -> BacktestingEngine:
    """
    Factory function to create a BacktestingEngine instance.
    
    Args:
        data_service: Optional data service instance
        strategy_service: Optional strategy service instance
        
    Returns:
        Configured BacktestingEngine instance
    """
    return BacktestingEngine(data_service, strategy_service)


def validate_backtest_config(config: BacktestConfig) -> List[str]:
    """
    Validate backtest configuration and return any warnings.
    
    Args:
        config: Backtest configuration to validate
        
    Returns:
        List of warning messages
    """
    warnings = []
    
    # Date range validation
    if (config.end_date - config.start_date).days < 7:
        warnings.append("Backtest period is less than 7 days - results may not be statistically significant")
    
    if (config.end_date - config.start_date).days > 365:
        warnings.append("Backtest period is longer than 1 year - consider shorter periods for faster execution")
    
    # Capital validation
    if config.initial_capital < 100000:  # 1 lakh
        warnings.append("Initial capital is quite low for crude oil trading")
    
    # Risk parameters
    if config.max_position_size > 0.2:
        warnings.append("Max position size above 20% may be risky for commodities")
    
    if config.slippage_bps > 10:
        warnings.append("Slippage of more than 10 bps seems high for crude oil")
    
    return warnings


def calculate_signal_statistics(trades: List[BacktestTrade]) -> Dict[str, Any]:
    """
    Calculate statistics about signal quality and performance.
    
    Args:
        trades: List of backtest trades
        
    Returns:
        Dictionary with signal performance statistics
    """
    if not trades:
        return {}
    
    # Group trades by signal quality
    quality_stats = {}
    
    for trade in trades:
        if not trade.is_open and trade.setup_quality:
            quality = trade.setup_quality
            if quality not in quality_stats:
                quality_stats[quality] = {
                    'trades': [],
                    'total_pnl': 0.0,
                    'win_count': 0,
                    'avg_score': 0.0
                }
            
            quality_stats[quality]['trades'].append(trade)
            if trade.pnl_inr:
                quality_stats[quality]['total_pnl'] += trade.pnl_inr
                if trade.pnl_inr > 0:
                    quality_stats[quality]['win_count'] += 1
            
            if trade.setup_score:
                quality_stats[quality]['avg_score'] += trade.setup_score
    
    # Calculate averages
    for quality, stats in quality_stats.items():
        trade_count = len(stats['trades'])
        if trade_count > 0:
            stats['win_rate'] = stats['win_count'] / trade_count * 100
            stats['avg_pnl_per_trade'] = stats['total_pnl'] / trade_count
            stats['avg_score'] = stats['avg_score'] / trade_count
            stats['trade_count'] = trade_count
    
    return quality_stats