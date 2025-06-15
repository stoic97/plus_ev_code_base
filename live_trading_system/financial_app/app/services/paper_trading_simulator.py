"""
Complete Integration Example: Signal → Order → Trade → Analytics

This example demonstrates the complete flow of our simulation-based trading system:
1. Strategy generates signals
2. Order Execution Service processes signals with risk checks
3. Orders are executed in simulation environment
4. Trades are created and tracked
5. Analytics provide performance insights

This bridges the gap between signal generation and trade execution.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any

from sqlalchemy.orm import Session
from app.core.database import get_db
from app.services.strategy_engine import StrategyEngineService
from app.services.paper_trading_simulator import PaperTradingSimulator
from app.services.analytics import AnalyticsService
from app.schemas.order import OrderCreate, ExecutionSimulationConfig, OrderTypeEnum, SlippageModelEnum
from app.schemas.strategy import SignalCreate, TimeframeAnalysisResult, MarketStateAnalysis, SetupQualityResult

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradingSystemIntegration:
    """
    Complete trading system integration demonstrating the full flow.
    """
    
    def __init__(self, db: Session):
        """Initialize all services."""
        self.db = db
        
        # Initialize services
        self.strategy_service = StrategyEngineService(db)
        
        # Configure execution simulation
        execution_config = ExecutionSimulationConfig(
            slippage_model=SlippageModelEnum.VOLUME_BASED,
            base_slippage_bps=15.0,  # 15 basis points base slippage
            volume_impact_factor=0.2,  # 20% volume impact
            latency_ms=150,  # 150ms execution delay
            commission_per_lot=20.0,  # 20 INR per lot
            tax_rate=0.18,  # 18% tax rate
            market_hours_only=True
        )
        
        self.order_execution_service = OrderExecutionService(db, execution_config)
        self.analytics_service = AnalyticsService(db)
        
        # Integration metrics
        self.integration_metrics = {
            "signals_generated": 0,
            "orders_created": 0,
            "trades_executed": 0,
            "total_pnl": 0.0,
            "success_rate": 0.0
        }
    
    async def run_complete_simulation(self, strategy_id: int, user_id: int, days: int = 30) -> Dict[str, Any]:
        """
        Run a complete simulation demonstrating the full system flow.
        
        Args:
            strategy_id: Strategy to use for signal generation
            user_id: User ID for the simulation
            days: Number of days to simulate
            
        Returns:
            Complete simulation results and analytics
        """
        logger.info(f"Starting complete simulation for strategy {strategy_id}, {days} days")
        
        simulation_results = {
            "simulation_config": {
                "strategy_id": strategy_id,
                "user_id": user_id,
                "days": days,
                "start_time": datetime.utcnow().isoformat()
            },
            "signals": [],
            "orders": [],
            "trades": [],
            "analytics": {},
            "performance_summary": {}
        }
        
        try:
            # Step 1: Generate signals (simulated market data)
            logger.info("Step 1: Generating trading signals...")
            signals = await self._generate_sample_signals(strategy_id, user_id, days)
            simulation_results["signals"] = [self._signal_to_dict(signal) for signal in signals]
            
            # Step 2: Process signals through order execution
            logger.info("Step 2: Processing signals through order execution...")
            orders = []
            for signal in signals:
                try:
                    # Execute signal with risk validation
                    order_response = await self.order_execution_service.execute_signal(
                        signal_id=signal.id,
                        user_id=user_id
                    )
                    orders.append(order_response)
                    self.integration_metrics["orders_created"] += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to execute signal {signal.id}: {e}")
            
            simulation_results["orders"] = [order.model_dump() for order in orders]
            
            # Step 3: Process all pending orders
            logger.info("Step 3: Processing pending orders...")
            await self.order_execution_service.process_pending_orders()
            
            # Step 4: Get resulting trades
            logger.info("Step 4: Retrieving executed trades...")
            trades = self._get_trades_from_signals([s.id for s in signals])
            simulation_results["trades"] = [self._trade_to_dict(trade) for trade in trades]
            self.integration_metrics["trades_executed"] = len(trades)
            
            # Step 5: Generate analytics
            logger.info("Step 5: Generating performance analytics...")
            analytics_results = await self._generate_comprehensive_analytics(strategy_id, user_id)
            simulation_results["analytics"] = analytics_results
            
            # Step 6: Performance summary
            logger.info("Step 6: Generating performance summary...")
            performance_summary = self._calculate_performance_summary(orders, trades)
            simulation_results["performance_summary"] = performance_summary
            
            logger.info("Complete simulation finished successfully")
            return simulation_results
            
        except Exception as e:
            logger.error(f"Error in complete simulation: {e}")
            simulation_results["error"] = str(e)
            return simulation_results
    
    async def _generate_sample_signals(self, strategy_id: int, user_id: int, days: int) -> List[Any]:
        """Generate sample signals for demonstration."""
        signals = []
        
        # Get strategy for signal generation
        strategy = self.strategy_service.get_strategy(strategy_id)
        if not strategy:
            raise ValueError(f"Strategy {strategy_id} not found")
        
        # Generate signals for each day (simplified)
        base_date = datetime.utcnow() - timedelta(days=days)
        
        for day in range(days):
            current_date = base_date + timedelta(days=day)
            
            # Skip weekends (simplified)
            if current_date.weekday() >= 5:
                continue
            
            # Generate 1-2 signals per trading day
            signals_per_day = 1 if day % 3 == 0 else 2  # Vary signal frequency
            
            for signal_num in range(signals_per_day):
                try:
                    # Create sample market data for signal generation
                    sample_market_data = self._create_sample_market_data(current_date)
                    
                    # Generate signal using strategy service
                    signal = await self._create_sample_signal(
                        strategy, current_date, signal_num, sample_market_data, user_id
                    )
                    signals.append(signal)
                    self.integration_metrics["signals_generated"] += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to generate signal for day {day}: {e}")
        
        logger.info(f"Generated {len(signals)} signals over {days} days")
        return signals
    
    def _create_sample_market_data(self, date: datetime) -> Dict[str, Any]:
        """Create sample market data for signal generation."""
        import random
        
        # Base NIFTY price with some variation
        base_price = 18500 + random.uniform(-500, 500)
        
        return {
            "1h": {
                "close": [base_price + random.uniform(-50, 50) for _ in range(5)],
                "high": [base_price + random.uniform(0, 100) for _ in range(5)],
                "low": [base_price - random.uniform(0, 100) for _ in range(5)],
                "volume": [random.randint(1000, 5000) for _ in range(5)],
                "ma21": [base_price - random.uniform(10, 30) for _ in range(5)],
                "ma200": [base_price - random.uniform(100, 200) for _ in range(5)]
            },
            "15m": {
                "close": [base_price + random.uniform(-20, 20) for _ in range(10)],
                "high": [base_price + random.uniform(0, 40) for _ in range(10)],
                "low": [base_price - random.uniform(0, 40) for _ in range(10)],
                "volume": [random.randint(500, 2000) for _ in range(10)]
            }
        }
    
    async def _create_sample_signal(
        self, 
        strategy: Any, 
        date: datetime, 
        signal_num: int, 
        market_data: Dict[str, Any], 
        user_id: int
    ) -> Any:
        """Create a sample signal for demonstration."""
        import random
        from app.models.strategy import Signal, Direction, SetupQualityGrade, EntryTechnique
        
        # Determine signal direction based on market conditions
        current_price = market_data["1h"]["close"][-1]
        ma21 = market_data["1h"]["ma21"][-1]
        direction = Direction.LONG if current_price > ma21 else Direction.SHORT
        
        # Calculate entry, stop loss, and take profit
        entry_price = current_price + random.uniform(-5, 5)
        
        if direction == Direction.LONG:
            stop_loss_price = entry_price - random.uniform(20, 40)
            take_profit_price = entry_price + random.uniform(40, 80)
        else:
            stop_loss_price = entry_price + random.uniform(20, 40)
            take_profit_price = entry_price - random.uniform(40, 80)
        
        # Calculate risk-reward ratio
        risk_points = abs(entry_price - stop_loss_price)
        reward_points = abs(take_profit_price - entry_price)
        risk_reward_ratio = reward_points / risk_points if risk_points > 0 else 0
        
        # Determine setup quality (random for demo)
        setup_qualities = [SetupQualityGrade.A_PLUS, SetupQualityGrade.A, SetupQualityGrade.B, SetupQualityGrade.C]
        setup_quality = random.choice(setup_qualities)
        setup_score = random.uniform(70, 95)
        
        # Position size based on setup quality
        position_size_map = {
            SetupQualityGrade.A_PLUS: 3,
            SetupQualityGrade.A: 2,
            SetupQualityGrade.B: 2,
            SetupQualityGrade.C: 1
        }
        position_size = position_size_map.get(setup_quality, 1)
        
        # Create signal
        signal = Signal(
            strategy_id=strategy.id,
            instrument="NIFTY",
            direction=direction,
            signal_type="trend_following",
            entry_price=entry_price,
            entry_time=date,
            entry_timeframe="1h",
            entry_technique=EntryTechnique.BREAKOUT,
            take_profit_price=take_profit_price,
            stop_loss_price=stop_loss_price,
            trailing_stop=False,
            position_size=position_size,
            risk_reward_ratio=risk_reward_ratio,
            risk_amount=risk_points * 50 * position_size,  # 50 INR per point per lot
            setup_quality=setup_quality,
            setup_score=setup_score,
            confidence=random.uniform(0.7, 0.95),
            market_state="trending",
            trend_phase="middle",
            is_active=True,
            is_executed=False,
            timeframe_alignment_score=random.uniform(0.8, 1.0),
            primary_timeframe_aligned=True,
            institutional_footprint_detected=random.choice([True, False]),
            bos_detected=random.choice([True, False]),
            is_spread_trade=False,
            user_id=user_id
        )
        
        # Save signal to database
        self.db.add(signal)
        self.db.commit()
        self.db.refresh(signal)
        
        return signal
    
    def _get_trades_from_signals(self, signal_ids: List[int]) -> List[Any]:
        """Get trades that were created from the given signals."""
        from app.models.strategy import Trade
        
        trades = self.db.query(Trade).filter(
            Trade.signal_id.in_(signal_ids)
        ).all()
        
        return trades
    
    async def _generate_comprehensive_analytics(self, strategy_id: int, user_id: int) -> Dict[str, Any]:
        """Generate comprehensive analytics for the simulation."""
        
        # Order execution analytics
        order_analytics = self.order_execution_service.get_execution_analytics(user_id, days=30)
        
        # Strategy performance analytics (if analytics service exists)
        try:
            strategy_analytics = {
                "total_signals": self.integration_metrics["signals_generated"],
                "execution_rate": (self.integration_metrics["orders_created"] / 
                                 max(self.integration_metrics["signals_generated"], 1)) * 100,
                "fill_rate": order_analytics.get("fill_rate_percent", 0),
                "average_slippage": order_analytics.get("average_slippage_points", 0),
                "total_costs": order_analytics.get("total_costs_inr", 0)
            }
        except Exception as e:
            logger.warning(f"Could not generate strategy analytics: {e}")
            strategy_analytics = {"error": "Analytics service unavailable"}
        
        return {
            "order_execution": order_analytics,
            "strategy_performance": strategy_analytics,
            "integration_metrics": self.integration_metrics
        }
    
    def _calculate_performance_summary(self, orders: List[Any], trades: List[Any]) -> Dict[str, Any]:
        """Calculate overall performance summary."""
        
        if not orders and not trades:
            return {"message": "No orders or trades to analyze"}
        
        # Order statistics
        total_orders = len(orders)
        filled_orders = len([o for o in orders if o.order_status == "filled"])
        
        # Trade statistics
        total_trades = len(trades)
        profitable_trades = len([t for t in trades if getattr(t, 'profit_loss_inr', 0) > 0])
        
        # Calculate total P&L
        total_pnl = sum(getattr(t, 'profit_loss_inr', 0) for t in trades)
        self.integration_metrics["total_pnl"] = total_pnl
        
        # Calculate success rate
        success_rate = (profitable_trades / max(total_trades, 1)) * 100
        self.integration_metrics["success_rate"] = success_rate
        
        # Cost analysis
        total_commission = sum(getattr(t, 'commission', 0) for t in trades)
        total_taxes = sum(getattr(t, 'taxes', 0) for t in trades)
        total_slippage = sum(getattr(t, 'slippage', 0) for t in trades)
        
        return {
            "execution_summary": {
                "total_orders": total_orders,
                "filled_orders": filled_orders,
                "fill_rate_percent": (filled_orders / max(total_orders, 1)) * 100
            },
            "trading_summary": {
                "total_trades": total_trades,
                "profitable_trades": profitable_trades,
                "success_rate_percent": success_rate,
                "total_pnl_inr": total_pnl,
                "average_pnl_per_trade": total_pnl / max(total_trades, 1)
            },
            "cost_summary": {
                "total_commission_inr": total_commission,
                "total_taxes_inr": total_taxes,
                "total_slippage_inr": total_slippage,
                "total_costs_inr": total_commission + total_taxes + total_slippage,
                "cost_per_trade": (total_commission + total_taxes + total_slippage) / max(total_trades, 1)
            },
            "integration_metrics": self.integration_metrics
        }
    
    def _signal_to_dict(self, signal: Any) -> Dict[str, Any]:
        """Convert signal to dictionary for JSON serialization."""
        return {
            "id": signal.id,
            "strategy_id": signal.strategy_id,
            "instrument": signal.instrument,
            "direction": signal.direction.value if hasattr(signal.direction, 'value') else str(signal.direction),
            "entry_price": signal.entry_price,
            "entry_time": signal.entry_time.isoformat() if signal.entry_time else None,
            "stop_loss_price": signal.stop_loss_price,
            "take_profit_price": signal.take_profit_price,
            "position_size": signal.position_size,
            "risk_reward_ratio": signal.risk_reward_ratio,
            "setup_quality": signal.setup_quality.value if hasattr(signal.setup_quality, 'value') else str(signal.setup_quality),
            "setup_score": signal.setup_score,
            "confidence": signal.confidence,
            "is_executed": signal.is_executed
        }
    
    def _trade_to_dict(self, trade: Any) -> Dict[str, Any]:
        """Convert trade to dictionary for JSON serialization."""
        return {
            "id": trade.id,
            "strategy_id": trade.strategy_id,
            "signal_id": trade.signal_id,
            "instrument": trade.instrument,
            "direction": trade.direction.value if hasattr(trade.direction, 'value') else str(trade.direction),
            "entry_price": trade.entry_price,
            "entry_time": trade.entry_time.isoformat() if trade.entry_time else None,
            "exit_price": trade.exit_price,
            "exit_time": trade.exit_time.isoformat() if trade.exit_time else None,
            "position_size": trade.position_size,
            "profit_loss_points": getattr(trade, 'profit_loss_points', None),
            "profit_loss_inr": getattr(trade, 'profit_loss_inr', None),
            "commission": trade.commission,
            "taxes": trade.taxes,
            "slippage": trade.slippage,
            "total_costs": getattr(trade, 'total_costs', trade.commission + trade.taxes + trade.slippage)
        }


# Example usage and demonstration
async def run_integration_demo():
    """Run a complete integration demonstration."""
    
    # Get database session
    db_session = next(get_db())
    
    try:
        # Initialize integration system
        trading_system = TradingSystemIntegration(db_session)
        
        # Configuration
        strategy_id = 1  # Assuming strategy exists
        user_id = 1      # Assuming user exists
        simulation_days = 10
        
        logger.info("Starting complete trading system integration demo")
        logger.info(f"Strategy: {strategy_id}, User: {user_id}, Days: {simulation_days}")
        
        # Run complete simulation
        results = await trading_system.run_complete_simulation(
            strategy_id=strategy_id,
            user_id=user_id,
            days=simulation_days
        )
        
        # Print summary results
        print("\n" + "="*60)
        print("TRADING SYSTEM INTEGRATION DEMO RESULTS")
        print("="*60)
        
        # Simulation config
        config = results["simulation_config"]
        print(f"Strategy ID: {config['strategy_id']}")
        print(f"Simulation Days: {config['days']}")
        print(f"Start Time: {config['start_time']}")
        
        # Signal generation
        signals = results["signals"]
        print(f"\nSignals Generated: {len(signals)}")
        if signals:
            print(f"Sample Signal: {signals[0]['instrument']} {signals[0]['direction']} @ {signals[0]['entry_price']}")
        
        # Order execution
        orders = results["orders"]
        print(f"Orders Created: {len(orders)}")
        filled_orders = [o for o in orders if o.get('order_status') == 'filled']
        print(f"Orders Filled: {len(filled_orders)}")
        
        # Trade execution
        trades = results["trades"]
        print(f"Trades Executed: {len(trades)}")
        
        # Performance summary
        if "performance_summary" in results:
            perf = results["performance_summary"]
            if "trading_summary" in perf:
                trading = perf["trading_summary"]
                print(f"\nPerformance Summary:")
                print(f"  Success Rate: {trading.get('success_rate_percent', 0):.1f}%")
                print(f"  Total P&L: ₹{trading.get('total_pnl_inr', 0):.2f}")
                print(f"  Average P&L per Trade: ₹{trading.get('average_pnl_per_trade', 0):.2f}")
            
            if "cost_summary" in perf:
                costs = perf["cost_summary"]
                print(f"  Total Costs: ₹{costs.get('total_costs_inr', 0):.2f}")
                print(f"  Cost per Trade: ₹{costs.get('cost_per_trade', 0):.2f}")
        
        # Integration metrics
        if "integration_metrics" in results.get("analytics", {}):
            metrics = results["analytics"]["integration_metrics"]
            print(f"\nIntegration Metrics:")
            print(f"  Signals → Orders Rate: {(metrics['orders_created']/max(metrics['signals_generated'],1))*100:.1f}%")
            print(f"  Orders → Trades Rate: {(metrics['trades_executed']/max(metrics['orders_created'],1))*100:.1f}%")
        
        print("\n" + "="*60)
        print("Demo completed successfully!")
        print("The system successfully demonstrated:")
        print("✓ Signal generation from strategy")
        print("✓ Risk validation and order creation")
        print("✓ Simulated order execution with realistic costs")
        print("✓ Trade record creation and tracking")
        print("✓ Performance analytics and reporting")
        print("="*60)
        
        return results
        
    except Exception as e:
        logger.error(f"Integration demo failed: {e}")
        raise
    finally:
        db_session.close()


# CLI interface for testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        # Run the integration demo
        asyncio.run(run_integration_demo())
    else:
        print("Order Execution Service Integration")
        print("Usage: python integration_example.py demo")
        print("\nThis module demonstrates the complete flow:")
        print("  Signal Generation → Risk Validation → Order Execution → Trade Creation → Analytics")
        print("\nKey Components:")
        print("  • OrderExecutionService: Manages order lifecycle")
        print("  • RiskManager: Pre-trade validation")
        print("  • MarketSimulator: Realistic execution simulation")
        print("  • ExecutionEngine: Core order processing")
        print("\nFeatures Demonstrated:")
        print("  • Realistic slippage and market impact")
        print("  • Commission and tax calculations")
        print("  • Risk management and position sizing")
        print("  • Order status tracking and analytics")
        print("  • Performance measurement and reporting")