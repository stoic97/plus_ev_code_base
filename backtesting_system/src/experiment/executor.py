elif signal_method == 'rule':
            # Rule-based signals
            rule_config = signal_config.get('rule_config', {})
            prediction_col = signal_config.get('prediction_col', 'prediction')
            actual_col = signal_config.get('actual_col', 'actual')
            
            # Get predictions
            if 'model' in predictions.columns:
                # Use best model
                best_model = signal_config.get('best_model')
                
                if best_model:
                    model_preds = predictions[predictions['model'] == best_model]
                    signals['prediction'] = model_preds[prediction_col]
                else:
                    # Group by index and take mean
                    preds_group = predictions.groupby(predictions.index)[prediction_col].mean()
                    signals['prediction'] = preds_group
            else:
                signals['prediction'] = predictions[prediction_col]
                
            # Apply rule
            rule_type = rule_config.get('type', 'momentum')
            
            if rule_type == 'momentum':
                # Momentum rule
                window = rule_config.get('window', 5)
                signals['momentum'] = signals['prediction'].rolling(window=window).mean()
                
                # Generate signals
                signals['signal'] = 'hold'
                signals.loc[signals['momentum'] > 0, 'signal'] = 'buy'
                signals.loc[signals['momentum'] < 0, 'signal'] = 'sell'
                
            elif rule_type == 'crossover':
                # Moving average crossover
                fast_window = rule_config.get('fast_window', 5)
                slow_window = rule_config.get('slow_window', 20)
                
                signals['fast_ma'] = signals['prediction'].rolling(window=fast_window).mean()
                signals['slow_ma'] = signals['prediction'].rolling(window=slow_window).mean()
                
                # Generate signals
                signals['signal'] = 'hold'
                signals.loc[signals['fast_ma'] > signals['slow_ma'], 'signal'] = 'buy'
                signals.loc[signals['fast_ma'] < signals['slow_ma'], 'signal'] = 'sell'
                
            elif rule_type == 'reversal':
                # Mean reversal
                window = rule_config.get('window', 20)
                std_multiple = rule_config.get('std_multiple', 2.0)
                
                signals['mean'] = signals['prediction'].rolling(window=window).mean()
                signals['std'] = signals['prediction'].rolling(window=window).std()
                
                # Generate signals
                signals['signal'] = 'hold'
                signals.loc[signals['prediction'] < signals['mean'] - std_multiple * signals['std'], 'signal'] = 'buy'
                signals.loc[signals['prediction'] > signals['mean'] + std_multiple * signals['std'], 'signal'] = 'sell'
                
            else:
                raise ValueError(f"Unsupported rule type: {rule_type}")
                
        else:
            raise ValueError(f"Unsupported signal method: {signal_method}")
            
        # Clean up signals
        signals = signals.dropna(subset=['signal'])
        
        self.logger.info(f"Generated {len(signals)} signals")
        
        return signals
        
    def _calculate_position_sizes(
        self,
        signals: pd.DataFrame,
        data: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate position sizes for signals.
        
        Args:
            signals: Signals dataframe
            data: Data dataframe
            
        Returns:
            Series with position sizes
        """
        position_config = self.config.get('position_config', {})
        sizing_method = position_config.get('method', 'fixed')
        
        self.logger.info(f"Calculating position sizes using method: {sizing_method}")
        
        # Create position sizer
        position_sizer = PositionSizer(
            sizing_method,
            position_config.get('params', {})
        )
        
        # Get price and volatility series
        price_col = position_config.get('price_col', 'close')
        volatility_col = position_config.get('volatility_col')
        confidence_col = position_config.get('confidence_col')
        
        prices = data[price_col] if price_col in data.columns else None
        volatility = data[volatility_col] if volatility_col and volatility_col in data.columns else None
        confidence = data[confidence_col] if confidence_col and confidence_col in data.columns else None
        
        # Calculate position sizes
        position_sizes = position_sizer.calculate_position_sizes(
            signals['signal'],
            prices,
            volatility,
            confidence
        )
        
        self.logger.info(f"Calculated {len(position_sizes)} position sizes")
        
        return position_sizes
        
    def _run_backtest(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        position_sizes: pd.Series
    ) -> Dict[str, Any]:
        """
        Run backtest with signals and position sizes.
        
        Args:
            data: Data dataframe
            signals: Signals dataframe
            position_sizes: Position sizes series
            
        Returns:
            Backtest results
        """
        backtest_config = self.config.get('backtest_config', {})
        
        # Set up parameters
        initial_capital = backtest_config.get('initial_capital', 1000000)
        transaction_cost = backtest_config.get('transaction_cost', 0.001)
        enable_fractional = backtest_config.get('enable_fractional', True)
        execution_delay = backtest_config.get('execution_delay', 0)
        price_col = backtest_config.get('price_col', 'close')
        
        self.logger.info("Running backtest")
        
        # Create backtest engine
        backtest_engine = BacktestEngine(
            initial_capital,
            transaction_cost,
            enable_fractional,
            execution_delay
        )
        
        # Run backtest
        results = backtest_engine.run_backtest(
            data,
            signals,
            position_sizes,
            price_col
        )
        
        self.logger.info("Completed backtest")
        
        return results
        
    def _analyze_results(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze backtest results.
        
        Args:
            backtest_results: Backtest results
            
        Returns:
            Analysis results
        """
        analysis_config = self.config.get('analysis_config', {})
        
        # Set up parameters
        benchmark_index = analysis_config.get('benchmark_index')
        run_monte_carlo = analysis_config.get('run_monte_carlo', False)
        detect_regimes = analysis_config.get('detect_regimes', False)
        
        self.logger.info("Analyzing results")
        
        # Basic analysis
        portfolio = backtest_results['portfolio']
        metrics = backtest_results['metrics']
        
        analysis = {
            'metrics': metrics,
            'trade_count': len(portfolio[portfolio['trade']]),
            'win_rate': (portfolio[portfolio['trade'] & (portfolio['returns'] > 0)].shape[0] / 
                        max(1, portfolio[portfolio['trade']].shape[0])),
            'average_trade_return': portfolio[portfolio['trade']]['returns'].mean(),
            'average_trade_duration': None,  # Need to calculate based on trade entry/exit
            'sharpe_ratio': metrics.get('sharpe_ratio'),
            'sortino_ratio': None,  # Need to calculate based on downside deviation
            'max_drawdown': metrics.get('max_drawdown'),
            'calmar_ratio': metrics.get('calmar_ratio'),
            'monthly_returns': portfolio['returns'].resample('M').apply(lambda x: (1 + x).prod() - 1).to_dict(),
            'annual_returns': portfolio['returns'].resample('Y').apply(lambda x: (1 + x).prod() - 1).to_dict()
        }
        
        # Calculate Sortino ratio
        downside_returns = portfolio['returns'][portfolio['returns'] < 0]
        if not downside_returns.empty:
            downside_deviation = downside_returns.std() * np.sqrt(252)
            if downside_deviation > 0:
                analysis['sortino_ratio'] = (portfolio['returns'].mean() * 252) / downside_deviation
                
        # Calculate average trade duration
        trade_starts = portfolio.index[portfolio['trade'] & (portfolio['position'] > 0)]
        trade_ends = portfolio.index[portfolio['trade'] & (portfolio['position'] < 0)]
        
        if len(trade_starts) > 0 and len(trade_ends) > 0:
            trade_durations = []
            
            for start in trade_starts:
                # Find next end after this start
                end_idx = np.searchsorted(trade_ends, start)
                if end_idx < len(trade_ends):
                    duration = (trade_ends[end_idx] - start).total_seconds() / 86400  # in days
                    trade_durations.append(duration)
                    
            if trade_durations:
                analysis['average_trade_duration'] = np.mean(trade_durations)
                
        # Regime detection
        if detect_regimes:
            try:
                self.logger.info("Detecting market regimes")
                
                # Set up regime detector
                regime_detector = MarketRegimeDetector(
                    portfolio,
                    price_col='equity',
                    returns_col='returns',
                    n_regimes=analysis_config.get('n_regimes', 2)
                )
                
                # Fit and predict regimes
                regime_detector.fit()
                regimes = regime_detector.predict_regimes()
                
                # Analyze regimes
                regime_analysis = regime_detector.analyze_regimes(regimes)
                
                # Save regime plot
                plot_path = os.path.join(self.experiment_dirs['visualizations'], "regimes.png")
                regime_analysis['plot'].savefig(plot_path)
                
                analysis['regime_analysis'] = regime_analysis
                analysis['regimes'] = regimes.to_dict()
                
            except Exception as e:
                self.logger.warning(f"Regime detection failed: {str(e)}")
                
        # Monte Carlo simulation
        if run_monte_carlo:
            try:
                self.logger.info("Running Monte Carlo simulation")
                
                # Strategy function for simulation
                def strategy_func(data):
                    return signals['signal']
                    
                # Set up simulator
                simulator = MonteCarloSimulator(
                    strategy_func,
                    data,
                    price_col=price_col
                )
                
                # Run simulation
                simulation_results = simulator.run_simulation(
                    n_simulations=analysis_config.get('n_simulations', 100),
                    initial_capital=backtest_config.get('initial_capital', 1000000),
                    transaction_cost=backtest_config.get('transaction_cost', 0.001)
                )
                
                # Save simulation plot
                if 'plot' in simulation_results:
                    plot_path = os.path.join(self.experiment_dirs['visualizations'], "monte_carlo.png")
                    simulation_results['plot'].savefig(plot_path)
                    
                analysis['monte_carlo'] = simulation_results
                
            except Exception as e:
                self.logger.warning(f"Monte Carlo simulation failed: {str(e)}")
                
        self.logger.info("Completed analysis")
        
        return analysis
        
    def _generate_visualizations(
        self,
        backtest_results: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Generate visualizations of backtest results.
        
        Args:
            backtest_results: Backtest results
            
        Returns:
            Dictionary of visualization paths
        """
        visualization_config = self.config.get('visualization_config', {})
        portfolio = backtest_results['portfolio']
        
        self.logger.info("Generating visualizations")
        
        # Create visualization manager
        visualization_manager = VisualizationManager(
            self.experiment_dirs['visualizations'],
            interactive=visualization_config.get('interactive', True)
        )
        
        # Generate visualizations
        visualizations = {}
        
        # Equity curve
        equity_fig = visualization_manager.create_equity_curve(
            portfolio,
            title=f"Equity Curve - {self.experiment_id}",
            include_drawdown=True
        )
        visualizations['equity_curve'] = os.path.join(
            self.experiment_dirs['visualizations'],
            "equity_curve.png"
        )
        
        # Returns analysis
        returns_fig = visualization_manager.create_returns_analysis(
            portfolio,
            title=f"Returns Analysis - {self.experiment_id}"
        )
        visualizations['returns_analysis'] = os.path.join(
            self.experiment_dirs['visualizations'],
            "returns_analysis.png"
        )
        
        # Trade analysis
        trade_fig = visualization_manager.create_trade_analysis(
            portfolio,
            title=f"Trade Analysis - {self.experiment_id}"
        )
        visualizations['trade_analysis'] = os.path.join(
            self.experiment_dirs['visualizations'],
            "trade_analysis.png"
        )
        
        # Interactive equity curve
        if visualization_config.get('interactive', True):
            visualization_manager.create_interactive_equity_curve(
                portfolio,
                title=f"Interactive Equity Curve - {self.experiment_id}",
                include_drawdown=True
            )
            visualizations['interactive_equity_curve'] = os.path.join(
                self.experiment_dirs['visualizations'],
                "interactive_equity_curve.html"
            )
            
        self.logger.info(f"Generated {len(visualizations)} visualizations")
        
        return visualizations
        
    def _generate_report(
        self,
        backtest_results: Dict[str, Any],
        analysis_results: Dict[str, Any],
        visualization_paths: Dict[str, str]
    ) -> str:
        """
        Generate report of experiment results.
        
        Args:
            backtest_results: Backtest results
            analysis_results: Analysis results
            visualization_paths: Visualization paths
            
        Returns:
            Path to the generated report
        """
        report_config = self.config.get('report_config', {})
        report_format = report_config.get('format', 'html')
        
        self.logger.info(f"Generating {report_format} report")
        
        # Create report generator
        report_generator = ReportGenerator(
            self.experiment_dirs['results'],
            template_dir=report_config.get('template_dir')
        )
        
        # Prepare report data
        report_data = {
            'backtest': backtest_results,
            'analysis': analysis_results,
            'visualizations': visualization_paths
        }
        
        # Generate report
        if report_format == 'html':
            report_path = report_generator.generate_html_report(
                self.experiment_id,
                report_data,
                self.config,
                {
                    'start_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'duration': 'N/A',  # Would need to calculate from experiment metadata
                    'status': 'Completed'
                },
                template_name=report_config.get('template_name')
            )
        elif report_format == 'pdf':
            report_path = report_generator.generate_pdf_report(
                self.experiment_id,
                report_data,
                self.config,
                {
                    'start_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'duration': 'N/A',
                    'status': 'Completed'
                },
                template_name=report_config.get('template_name')
            )
        elif report_format == 'markdown':
            report_path = report_generator.generate_markdown_report(
                self.experiment_id,
                report_data,
                self.config,
                {
                    'start_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'duration': 'N/A',
                    'status': 'Completed'
                }
            )
        else:
            raise ValueError(f"Unsupported report format: {report_format}")
            
        self.logger.info(f"Generated report: {report_path}")
        
        return report_path
        
    def _log_to_wandb(self, results: Dict[str, Any]) -> None:
        """
        Log experiment results to Weights & Biases.
        
        Args:
            results: Experiment results
        """
        wandb_config = self.config.get('wandb_config', {})
        project = wandb_config.get('project', 'trading_experiments')
        
        self.logger.info(f"Logging to W&B project: {project}")
        
        # Create W&B logger
        wandb_logger = WandbLogger(
            project=project,
            entity=wandb_config.get('entity'),
            experiment_name=self.experiment_id,
            config=self.config,
            tags=wandb_config.get('tags'),
            group=wandb_config.get('group'),
            notes=wandb_config.get('notes')
        )
        
        # Start logger
        wandb_logger.start()
        
        # Log metrics
        if 'metrics' in results and 'overall' in results['metrics']:
            wandb_logger.log_metrics(results['metrics']['overall'])
            
        # Log analysis results
        if 'analysis_results' in results:
            wandb_logger.log_metrics({
                'analysis': {
                    k: v for k, v in results['analysis_results'].items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                }
            })
            
        # Log portfolio dataframe
        if 'backtest_results' in results and 'portfolio' in results['backtest_results']:
            wandb_logger.log_dataframe(
                results['backtest_results']['portfolio'],
                'portfolio'
            )
            
        # Log visualizations
        if 'visualization_paths' in results:
            for name, path in results['visualization_paths'].items():
                if os.path.exists(path):
                    if path.endswith('.png'):
                        img = plt.imread(path)
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.imshow(img)
                        ax.axis('off')
                        wandb_logger.log_plot(fig, name)
                        plt.close(fig)
                        
        # Log report
        if 'report_path' in results:
            wandb_logger.log_artifact(
                results['report_path'],
                'experiment_report',
                'report'
            )
            
        # Finish logging
        wandb_logger.finish()
        
        self.logger.info("Completed W&B logging")


def run_experiment(config_path: str) -> Dict[str, Any]:
    """
    Run an experiment from a configuration file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Experiment results
    """
    # Create and run executor
    executor = ExperimentExecutor(config_path)
    results = executor.run()
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run a trading experiment")
    parser.add_argument("config", help="Path to configuration file")
    args = parser.parse_args()
    
    results = run_experiment(args.config)
    print(f"Experiment completed: {results['experiment_id']}")
            ax.set_xticks(np.arange(len(returns_pivot.columns)))
            ax.set_yticks(np.arange(len(returns_pivot.index)))
            
            # Set tick labels
            ax.set_xticklabels(returns_pivot.columns)
            ax.set_yticklabels(returns_pivot.index)
            
            # Add text annotations
            for i in range(len(returns_pivot.index)):
                for j in range(len(returns_pivot.columns)):
                    value = returns_pivot.iloc[i, j]
                    if not np.isnan(value):
                        text_color = 'white' if abs(value) > 0.05 else 'black'
                        ax.text(j, i, f"{value:.1%}",
                                ha="center", va="center", color=text_color)
                                
            # Add colorbar
            fig.colorbar(im, ax=ax)
            
        fig.suptitle(title, fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save plot
        plot_path = os.path.join(self.output_dir, "returns_analysis.png")
        fig.savefig(plot_path)
        
        return fig
        
    def create_trade_analysis(
        self,
        portfolio: pd.DataFrame,
        title: str = "Trade Analysis"
    ) -> plt.Figure:
        """
        Create trade analysis plots.
        
        Args:
            portfolio: Portfolio dataframe
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        if 'trade' not in portfolio.columns:
            raise ValueError("Portfolio dataframe must have 'trade' column")
            
        # Extract trades
        trades = portfolio[portfolio['trade']].copy()
        
        if len(trades) == 0:
            self.logger.warning("No trades found in portfolio")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "No trades found", ha='center', va='center')
            return fig
            
        # Calculate trade statistics
        trades['trade_return'] = trades['returns']
        trades['winning'] = trades['trade_return'] > 0
        
        # Create plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot trade returns
        ax = axes[0, 0]
        trades['trade_return'].hist(bins=50, ax=ax)
        ax.axvline(0, color='k', linestyle='--')
        ax.set_title("Trade Returns Distribution")
        ax.set_xlabel("Return")
        ax.set_ylabel("Frequency")
        ax.grid(True)
        
        # Plot trade returns by month
        ax = axes[0, 1]
        monthly_trades = trades.groupby(trades.index.to_period('M'))['trade_return'].mean()
        monthly_trades.plot(kind='bar', ax=ax)
        ax.set_title("Average Trade Return by Month")
        ax.set_xlabel("Month")
        ax.set_ylabel("Average Return")
        ax.grid(True)
        
        # Plot win rate over time
        ax = axes[1, 0]
        win_rate = trades['winning'].rolling(window=20).mean()
        win_rate.plot(ax=ax)
        ax.axhline(win_rate.mean(), color='r', linestyle='--', label=f"Average: {win_rate.mean():.2f}")
        ax.set_title("Win Rate (20-trade rolling window)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Win Rate")
        ax.legend()
        ax.grid(True)
        
        # Plot trade frequency
        ax = axes[1, 1]
        trade_counts = trades.resample('M').size()
        trade_counts.plot(kind='bar', ax=ax)
        ax.set_title("Monthly Trade Frequency")
        ax.set_xlabel("Month")
        ax.set_ylabel("Number of Trades")
        ax.grid(True)
        
        fig.suptitle(title, fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save plot
        plot_path = os.path.join(self.output_dir, "trade_analysis.png")
        fig.savefig(plot_path)
        
        return fig
        
    def create_interactive_equity_curve(
        self,
        portfolio: pd.DataFrame,
        title: str = "Interactive Equity Curve",
        include_drawdown: bool = True,
        benchmark: Optional[pd.Series] = None
    ) -> Any:
        """
        Create interactive equity curve plot.
        
        Args:
            portfolio: Portfolio dataframe
            title: Plot title
            include_drawdown: Whether to include drawdown
            benchmark: Benchmark series
            
        Returns:
            Interactive plot
        """
        if not self.interactive:
            self.logger.warning("Interactive plots are disabled")
            return self.create_equity_curve(portfolio, title, include_drawdown, benchmark)
            
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            if 'equity' not in portfolio.columns:
                raise ValueError("Portfolio dataframe must have 'equity' column")
                
            if include_drawdown and 'drawdown' not in portfolio.columns:
                # Calculate drawdown
                portfolio['drawdown'] = 1 - (portfolio['equity'] / portfolio['equity'].cummax())
                
            # Create figure
            if include_drawdown:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                   vertical_spacing=0.1, row_heights=[0.7, 0.3])
            else:
                fig = go.Figure()
                
            # Add equity curve
            if include_drawdown:
                fig.add_trace(
                    go.Scatter(
                        x=portfolio.index,
                        y=portfolio['equity'],
                        name="Strategy"
                    ),
                    row=1, col=1
                )
                
                # Add benchmark if provided
                if benchmark is not None:
                    # Normalize benchmark to same starting value
                    norm_benchmark = benchmark / benchmark.iloc[0] * portfolio['equity'].iloc[0]
                    fig.add_trace(
                        go.Scatter(
                            x=norm_benchmark.index,
                            y=norm_benchmark,
                            name="Benchmark",
                            opacity=0.7
                        ),
                        row=1, col=1
                    )
                    
                # Add drawdown
                fig.add_trace(
                    go.Scatter(
                        x=portfolio.index,
                        y=portfolio['drawdown'] * 100,
                        name="Drawdown",
                        fill='tozeroy',
                        fillcolor='rgba(255, 0, 0, 0.3)',
                        line=dict(color='red')
                    ),
                    row=2, col=1
                )
                
                # Update layout
                fig.update_layout(
                    title=title,
                    xaxis_title="Date",
                    yaxis_title="Equity",
                    yaxis2_title="Drawdown (%)",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
            else:
                fig.add_trace(
                    go.Scatter(
                        x=portfolio.index,
                        y=portfolio['equity'],
                        name="Strategy"
                    )
                )
                
                # Add benchmark if provided
                if benchmark is not None:
                    # Normalize benchmark to same starting value
                    norm_benchmark = benchmark / benchmark.iloc[0] * portfolio['equity'].iloc[0]
                    fig.add_trace(
                        go.Scatter(
                            x=norm_benchmark.index,
                            y=norm_benchmark,
                            name="Benchmark",
                            opacity=0.7
                        )
                    )
                    
                # Update layout
                fig.update_layout(
                    title=title,
                    xaxis_title="Date",
                    yaxis_title="Equity",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
            # Save plot
            plot_path = os.path.join(self.output_dir, "interactive_equity_curve.html")
            fig.write_html(plot_path)
            
            return fig
            
        except ImportError:
            self.logger.warning("plotly not installed. Using static plot instead.")
            return self.create_equity_curve(portfolio, title, include_drawdown, benchmark)


# ================================
# Main Experiment Executor Class
# ================================

class ExperimentExecutor:
    """
    Main class for executing trading experiments.
    
    This class provides a comprehensive framework for executing trading experiments,
    with support for walk-forward analysis, multiple models, and extensive reporting.
    """
    
    def __init__(
        self,
        config: Union[Dict[str, Any], str, ExperimentConfig],
        base_dir: Optional[str] = None,
        schema_path: Optional[str] = None,
        custom_validators: Optional[List[Callable]] = None
    ):
        """
        Initialize experiment executor.
        
        Args:
            config: Experiment configuration
            base_dir: Base directory for experiment outputs
            schema_path: Path to configuration schema
            custom_validators: Custom configuration validators
        """
        # Set up logging
        self.logger = logger
        
        # Set up base directory
        if base_dir is None:
            base_dir = os.path.join(os.getcwd(), "experiments")
            
        os.makedirs(base_dir, exist_ok=True)
        
        # Set up configuration
        self.config_manager = self._setup_config_manager(config, schema_path, custom_validators)
        
        # Set up experiment manager
        self.experiment_manager = ExperimentManager(base_dir, self.config_manager)
        
        # Set up GPU manager
        self.gpu_manager = GPUManager()
        
        # Set up metrics tracker
        metrics_dir = os.path.join(base_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        self.metrics_tracker = MetricsTracker(metrics_dir)
        
        # Set up model factory
        models_dir = os.path.join(os.getcwd(), "models")
        os.makedirs(models_dir, exist_ok=True)
        self.model_factory = ModelFactory(models_dir)
        
        # Set up resource monitor
        self.resource_monitor = ResourceMonitor()
        
        # Store config
        if isinstance(config, dict):
            self.config = config
        elif isinstance(config, str):
            self.config = self.config_manager.load_and_validate(config)[0].to_dict()
        elif isinstance(config, ExperimentConfig):
            self.config = config.to_dict()
        else:
            raise ValueError(f"Unsupported config type: {type(config)}")
            
        # Initialize experiment
        self.experiment_id = None
        self.experiment_dirs = None
        
    def _setup_config_manager(
        self,
        config: Union[Dict[str, Any], str, ExperimentConfig],
        schema_path: Optional[str],
        custom_validators: Optional[List[Callable]]
    ) -> ConfigManager:
        """
        Set up configuration manager.
        
        Args:
            config: Experiment configuration
            schema_path: Path to configuration schema
            custom_validators: Custom configuration validators
            
        Returns:
            Configuration manager
        """
        # Determine default config path
        if isinstance(config, ExperimentConfig):
            # Create a temporary default config
            default_config_path = os.path.join(os.getcwd(), "_temp_default_config.json")
            config.save(default_config_path, ConfigFormat.JSON)
        elif isinstance(config, dict):
            # Create a temporary default config
            default_config_path = os.path.join(os.getcwd(), "_temp_default_config.json")
            with open(default_config_path, 'w') as f:
                json.dump(config, f, indent=2)
        else:
            # Use provided config path
            default_config_path = config
            
        # Create config manager
        config_manager = ConfigManager(
            default_config_path,
            schema_path,
            custom_validators
        )
        
        return config_manager
        
    def initialize(self) -> None:
        """Initialize experiment."""
        # Generate experiment ID
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = self.config.get('experiment_name', 'experiment')
        self.experiment_id = f"{experiment_name}_{timestamp}"
        
        # Initialize experiment
        metadata = self.experiment_manager.initialize_experiment(
            ExperimentConfig.from_dict(self.config),
            self.experiment_id
        )
        
        # Store experiment directories
        self.experiment_dirs = {
            'root': metadata.output_dir,
            'checkpoints': metadata.checkpoint_dir,
            'logs': os.path.join(metadata.output_dir, 'logs'),
            'models': os.path.join(metadata.output_dir, 'models'),
            'results': os.path.join(metadata.output_dir, 'results'),
            'visualizations': os.path.join(metadata.output_dir, 'visualizations'),
        }
        
        # Initialize GPU if available
        if self.config.get('use_gpu', True):
            self.gpu_manager.initialize_gpu_context()
            
        # Start resource monitoring
        self.resource_monitor.start_monitoring()
        
        # Mark experiment as started
        self.experiment_manager.start_experiment()
        
        self.logger.info(f"Initialized experiment: {self.experiment_id}")
        
    def run(self) -> Dict[str, Any]:
        """
        Run the experiment.
        
        Returns:
            Results dictionary
        """
        # Initialize if not already initialized
        if self.experiment_id is None:
            self.initialize()
            
        try:
            # Execute experiment phases
            self.logger.info("Starting experiment execution")
            
            # Load data
            data = self._load_data()
            
            # Create feature set
            data_with_features = self._compute_features(data)
            
            # Set up models
            models = self._setup_models()
            
            # Perform walk-forward analysis
            walk_forward_results = self._execute_walk_forward(data_with_features, models)
            
            # Generate trading signals
            signals = self._generate_signals(walk_forward_results)
            
            # Calculate position sizes
            position_sizes = self._calculate_position_sizes(signals, data_with_features)
            
            # Run backtest
            backtest_results = self._run_backtest(data_with_features, signals, position_sizes)
            
            # Analyze results
            analysis_results = self._analyze_results(backtest_results)
            
            # Generate visualizations
            visualization_paths = self._generate_visualizations(backtest_results)
            
            # Generate report
            report_path = self._generate_report(
                backtest_results,
                analysis_results,
                visualization_paths
            )
            
            # Compile final results
            results = {
                'experiment_id': self.experiment_id,
                'predictions': walk_forward_results.get('predictions'),
                'signals': signals,
                'backtest_results': backtest_results,
                'analysis_results': analysis_results,
                'visualization_paths': visualization_paths,
                'report_path': report_path,
                'metrics': backtest_results.get('metrics'),
                'config': self.config
            }
            
            # Log to Weights & Biases if enabled
            if self.config.get('use_wandb', False):
                self._log_to_wandb(results)
                
            # Mark experiment as completed
            self.experiment_manager.complete_experiment(success=True)
            
            # Stop resource monitoring
            self.resource_monitor.stop_monitoring()
            
            # Save resource log
            resource_log_path = os.path.join(self.experiment_dirs['logs'], "resource_log.json")
            self.resource_monitor.save_resource_log(resource_log_path)
            
            # Save metrics
            self.metrics_tracker.save_metrics(
                {
                    'backtest': backtest_results.get('metrics', {}),
                    'analysis': analysis_results,
                    'resource_usage': self.resource_monitor.get_resource_summary()
                },
                self.experiment_id
            )
            
            self.logger.info(f"Experiment completed: {self.experiment_id}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {str(e)}")
            traceback.print_exc()
            
            # Mark experiment as failed
            self.experiment_manager.complete_experiment(success=False)
            
            # Stop resource monitoring
            self.resource_monitor.stop_monitoring()
            
            # Save resource log
            resource_log_path = os.path.join(self.experiment_dirs['logs'], "resource_log.json")
            self.resource_monitor.save_resource_log(resource_log_path)
            
            raise
            
    def _load_data(self) -> pd.DataFrame:
        """
        Load data for the experiment.
        
        Returns:
            Dataframe with loaded data
        """
        data_config = self.config.get('data_config', {})
        data_path = data_config.get('data_path')
        
        if not data_path:
            raise ValueError("Data path not specified in configuration")
            
        self.logger.info(f"Loading data from {data_path}")
        
        # Load data based on file extension
        file_ext = os.path.splitext(data_path)[1].lower()
        
        if file_ext == '.csv':
            # Load CSV
            df = pd.read_csv(
                data_path,
                parse_dates=data_config.get('parse_dates', True),
                index_col=data_config.get('index_col', 0)
            )
        elif file_ext in ['.xls', '.xlsx']:
            # Load Excel
            df = pd.read_excel(
                data_path,
                parse_dates=data_config.get('parse_dates', True),
                index_col=data_config.get('index_col', 0)
            )
        elif file_ext == '.parquet':
            # Load Parquet
            df = pd.read_parquet(data_path)
        elif file_ext == '.pickle' or file_ext == '.pkl':
            # Load Pickle
            df = pd.read_pickle(data_path)
        elif file_ext in ['.h5', '.hdf5']:
            # Load HDF5
            key = data_config.get('hdf_key', 'data')
            df = pd.read_hdf(data_path, key=key)
        else:
            raise ValueError(f"Unsupported file extension: {file_ext}")
            
        # Apply date range filter if specified
        start_date = data_config.get('start_date')
        end_date = data_config.get('end_date')
        
        if start_date or end_date:
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]
                
        # Apply column filter if specified
        columns = data_config.get('columns')
        if columns:
            df = df[columns]
            
        self.logger.info(f"Loaded data: {len(df)} rows, {len(df.columns)} columns")
        
        # Validate data
        if data_config.get('validate_data', True):
            validation_config = data_config.get('validation_config', {})
            data_validator = DataValidator()
            validation_results = data_validator.validate_dataframe(df, validation_config)
            
            # Log validation results
            if validation_results['validation_summary']['status'] != 'ok':
                self.logger.warning("Data validation issues found:")
                for issue in validation_results['validation_summary']['issues']:
                    self.logger.warning(f"  - {issue}")
                    
        return df
        
    def _compute_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute features for the experiment.
        
        Args:
            data: Input dataframe
            
        Returns:
            Dataframe with computed features
        """
        feature_config = self.config.get('feature_config', {})
        compute_features = feature_config.get('compute_features', True)
        
        if not compute_features:
            self.logger.info("Feature computation disabled")
            return data
            
        features_list = feature_config.get('features', [])
        
        if not features_list:
            self.logger.info("No features specified")
            return data
            
        self.logger.info(f"Computing {len(features_list)} features")
        
        # Set up feature computer
        cache_dir = os.path.join(os.getcwd(), "feature_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        feature_computer = FeatureComputer(cache_dir, self.gpu_manager)
        
        # Calculate dataset hash
        dataset_id = self.config.get('experiment_name', 'experiment')
        dataset_hash = hashlib.sha256(str(data.shape).encode()).hexdigest()
        
        # Compute features
        data_with_features = feature_computer.compute_feature_set(
            data,
            dataset_id,
            dataset_hash,
            features_list,
            use_cache=feature_config.get('use_cache', True),
            force_recompute=feature_config.get('force_recompute', False),
            parallel=feature_config.get('parallel', True),
            n_jobs=feature_config.get('n_jobs')
        )
        
        self.logger.info(f"Computed features: {len(data_with_features.columns)} total columns")
        
        return data_with_features
        
    def _setup_models(self) -> Dict[str, Any]:
        """
        Set up models for the experiment.
        
        Returns:
            Dictionary of models
        """
        model_config = self.config.get('model_config', {})
        models_list = model_config.get('models', [])
        
        if not models_list:
            self.logger.warning("No models specified")
            return {}
            
        self.logger.info(f"Setting up {len(models_list)} models")
        
        models = {}
        
        for model_spec in models_list:
            model_name = model_spec.get('name')
            
            if not model_name:
                raise ValueError("Model name not specified")
                
            try:
                # Create model
                model = self.model_factory.create_model(model_spec, self.gpu_manager)
                models[model_name] = model
                self.logger.info(f"Created model: {model_name}")
            except Exception as e:
                self.logger.error(f"Failed to create model {model_name}: {str(e)}")
                
        return models
        
    def _execute_walk_forward(self, data: pd.DataFrame, models: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute walk-forward analysis.
        
        Args:
            data: Input dataframe
            models: Dictionary of models
            
        Returns:
            Walk-forward results
        """
        walk_forward_config = self.config.get('walk_forward_config', {})
        
        # Set up parameters
        window_size = walk_forward_config.get('window_size', 252)  # Default to 1 year
        step_size = walk_forward_config.get('step_size', 21)  # Default to 1 month
        min_train_size = walk_forward_config.get('min_train_size', window_size)
        forecast_horizon = walk_forward_config.get('forecast_horizon', 1)
        validation_ratio = walk_forward_config.get('validation_ratio', 0.3)
        gap_size = walk_forward_config.get('gap_size', 0)
        
        # Target and feature columns
        target_col = walk_forward_config.get('target_col')
        if not target_col:
            raise ValueError("Target column not specified")
            
        feature_cols = walk_forward_config.get('feature_cols', [])
        if not feature_cols:
            raise ValueError("Feature columns not specified")
            
        # Checkpoint settings
        checkpoint_interval = walk_forward_config.get('checkpoint_interval', 10)
        save_models = walk_forward_config.get('save_models', False)
        
        self.logger.info("Starting walk-forward analysis")
        
        # Create walk-forward manager
        walk_forward_manager = WalkForwardManager(
            window_size,
            step_size,
            min_train_size,
            forecast_horizon,
            validation_ratio,
            gap_size
        )
        
        # Create models directory
        models_dir = os.path.join(self.experiment_dirs['models'])
        os.makedirs(models_dir, exist_ok=True)
        
        # Execute walk-forward for each model
        results = {}
        
        for model_name, model in models.items():
            self.logger.info(f"Executing walk-forward for model: {model_name}")
            
            # Get model config
            model_config = next(
                (m for m in self.config.get('model_config', {}).get('models', []) if m.get('name') == model_name),
                {}
            )
            
            # Execute walk-forward
            model_results = walk_forward_manager.execute_walk_forward(
                data,
                self.model_factory,
                model_config,
                target_col,
                feature_cols,
                gpu_manager=self.gpu_manager,
                checkpoint_interval=checkpoint_interval,
                save_models=save_models,
                models_dir=models_dir,
                experiment_id=self.experiment_id
            )
            
            results[model_name] = model_results
            
        self.logger.info("Completed walk-forward analysis")
        
        # Combine results
        combined_predictions = None
        combined_metrics = []
        
        for model_name, model_results in results.items():
            if combined_predictions is None:
                combined_predictions = model_results['predictions'].copy()
                combined_predictions['model'] = model_name
            else:
                model_preds = model_results['predictions'].copy()
                model_preds['model'] = model_name
                combined_predictions = pd.concat([combined_predictions, model_preds])
                
            # Add model name to metrics
            for window_metrics in model_results['metrics']['windows']:
                window_metrics['model'] = model_name
                combined_metrics.append(window_metrics)
                
        # Create combined metrics
        combined_results = {
            'predictions': combined_predictions,
            'metrics': {
                'windows': combined_metrics,
                'by_model': {
                    model_name: model_results['metrics']['overall']
                    for model_name, model_results in results.items()
                }
            },
            'windows': results[next(iter(results))]['windows'],
            'models': list(results.keys())
        }
        
        return combined_results
        
    def _generate_signals(self, walk_forward_results: Dict[str, Any]) -> pd.DataFrame:
        """
        Generate trading signals from predictions.
        
        Args:
            walk_forward_results: Walk-forward results
            
        Returns:
            Dataframe with trading signals
        """
        signal_config = self.config.get('signal_config', {})
        signal_method = signal_config.get('method', 'threshold')
        
        predictions = walk_forward_results['predictions']
        
        if predictions is None or predictions.empty:
            raise ValueError("No predictions available")
            
        self.logger.info(f"Generating signals using method: {signal_method}")
        
        # Create signals dataframe
        signals = pd.DataFrame(index=predictions.index.unique())
        
        if signal_method == 'threshold':
            # Threshold-based signals
            threshold = signal_config.get('threshold', 0.0)
            prediction_col = signal_config.get('prediction_col', 'prediction')
            actual_col = signal_config.get('actual_col', 'actual')
            
            # Group by index (in case of multiple models)
            if 'model' in predictions.columns:
                # Use predictions from best model at each point
                best_model = signal_config.get('best_model')
                
                if best_model:
                    # Use specified model
                    model_preds = predictions[predictions['model'] == best_model]
                    signals['prediction'] = model_preds[prediction_col]
                else:
                    # Use model with highest average accuracy
                    model_metrics = walk_forward_results['metrics']['by_model']
                    
                    if model_metrics:
                        # Find best model by R^2
                        best_model = max(
                            model_metrics.items(),
                            key=lambda x: x[1].get('r2', 0)
                        )[0]
                        
                        model_preds = predictions[predictions['model'] == best_model]
                        signals['prediction'] = model_preds[prediction_col]
                    else:
                        # Group by index and take mean
                        preds_group = predictions.groupby(predictions.index)[prediction_col].mean()
                        signals['prediction'] = preds_group
            else:
                signals['prediction'] = predictions[prediction_col]
                
            # Generate signals based on threshold
            signals['signal'] = 'hold'
            signals.loc[signals['prediction'] > threshold, 'signal'] = 'buy'
            signals.loc[signals['prediction'] < -threshold, 'signal'] = 'sell'
            
        elif signal_method == 'ml':
            # Machine learning-based signals
            ml_model_path = signal_config.get('ml_model_path')
            
            if not ml_model_path:
                raise ValueError("ML model path not specified")
                
            # Load ML model
            with open(ml_model_path, 'rb') as f:
                ml_model = pickle.load(f)
                
            # Prepare features
            feature_cols = signal_config.get('feature_cols', [])
            
            if not feature_cols:
                # Use all available columns except signal-related ones
                exclude_cols = ['signal', 'actual', 'prediction', 'window_id', 'model']
                feature_cols = [col for col in predictions.columns if col not in exclude_cols]
                
            # Generate predictions
            X = predictions[feature_cols]
            y_pred = ml_model.predict(X)
            
            # Map predictions to signals
            signal_map = signal_config.get('signal_map', {0: 'hold', 1: 'buy', -1: 'sell'})
            signals['signal'] = [signal_map.get(p, 'hold') for p in y_pred]
            
        elif signal_method == 'rule':
            # Rule        self.price_data = price_data
        self.price_col = price_col
        self.returns_col = returns_col
        self.n_regimes = n_regimes
        self.model_type = model_type
        self.logger = logger
        
        # Calculate returns if not provided
        if returns_col is None:
            self.returns_col = 'returns'
            self.price_data[self.returns_col] = self.price_data[price_col].pct_change()
            
        # Initialize model
        self.model = None
        
    def fit(
        self,
        feature_columns: Optional[List[str]] = None,
        window_size: Optional[int] = None
    ) -> None:
        """
        Fit the regime detection model.
        
        Args:
            feature_columns: Columns to use for regime detection
            window_size: Window size for rolling features
        """
        # Prepare features
        if feature_columns is None:
            # Use returns and some basic features
            features = self._calculate_features(window_size)
        else:
            features = self.price_data[feature_columns].copy()
            
        # Drop NaN values
        features = features.dropna()
        
        if len(features) == 0:
            raise ValueError("No valid data after feature calculation")
            
        # Fit model
        if self.model_type == 'hmm':
            self._fit_hmm(features)
        elif self.model_type == 'kmeans':
            self._fit_kmeans(features)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
        self.logger.info(f"Fitted {self.model_type} model with {self.n_regimes} regimes")
        
    def _calculate_features(self, window_size: Optional[int] = None) -> pd.DataFrame:
        """
        Calculate features for regime detection.
        
        Args:
            window_size: Window size for rolling features
            
        Returns:
            Dataframe with features
        """
        if window_size is None:
            window_size = 20
            
        # Create features dataframe
        features = pd.DataFrame(index=self.price_data.index)
        
        # Add returns
        features['returns'] = self.price_data[self.returns_col]
        
        # Add moving averages
        features['ma20'] = self.price_data[self.price_col].rolling(window=window_size).mean()
        features['ma50'] = self.price_data[self.price_col].rolling(window=50).mean()
        
        # Add volatility
        features['volatility'] = self.price_data[self.returns_col].rolling(window=window_size).std()
        
        # Add RSI
        delta = self.price_data[self.price_col].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=window_size).mean()
        avg_loss = loss.rolling(window=window_size).mean()
        rs = avg_gain / avg_loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # Add trend indicators
        features['trend'] = (features['ma20'] / features['ma50']) - 1
        
        # Drop NaN values
        features = features.dropna()
        
        return features
        
    def _fit_hmm(self, features: pd.DataFrame) -> None:
        """
        Fit Hidden Markov Model for regime detection.
        
        Args:
            features: Features dataframe
        """
        try:
            from sklearn.preprocessing import StandardScaler
            from hmmlearn import hmm
            
            # Standardize features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            # Fit HMM
            model = hmm.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="full",
                n_iter=1000,
                random_state=42
            )
            model.fit(scaled_features)
            
            # Store model and scaler
            self.model = {
                'type': 'hmm',
                'model': model,
                'scaler': scaler,
                'features': features.columns.tolist()
            }
            
        except ImportError:
            self.logger.error("hmmlearn not installed. Please install with: pip install hmmlearn")
            raise
            
    def _fit_kmeans(self, features: pd.DataFrame) -> None:
        """
        Fit K-means for regime detection.
        
        Args:
            features: Features dataframe
        """
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import KMeans
            
            # Standardize features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            # Fit K-means
            model = KMeans(
                n_clusters=self.n_regimes,
                random_state=42,
                n_init=10
            )
            model.fit(scaled_features)
            
            # Store model and scaler
            self.model = {
                'type': 'kmeans',
                'model': model,
                'scaler': scaler,
                'features': features.columns.tolist()
            }
            
        except ImportError:
            self.logger.error("scikit-learn not installed. Please install with: pip install scikit-learn")
            raise
            
    def predict_regimes(self) -> pd.Series:
        """
        Predict regimes for the data.
        
        Returns:
            Series with regime labels
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        # Prepare features
        features = self._calculate_features()
        
        # Filter to model features
        model_features = self.model['features']
        features = features[model_features]
        
        # Standardize features
        scaled_features = self.model['scaler'].transform(features)
        
        # Predict regimes
        if self.model['type'] == 'hmm':
            regimes = self.model['model'].predict(scaled_features)
        else:
            regimes = self.model['model'].predict(scaled_features)
            
        # Create series
        regime_series = pd.Series(regimes, index=features.index, name='regime')
        
        return regime_series
        
    def analyze_regimes(self, regime_series: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Analyze the properties of different regimes.
        
        Args:
            regime_series: Series with regime labels (predicted if None)
            
        Returns:
            Dictionary with regime analysis
        """
        if regime_series is None:
            regime_series = self.predict_regimes()
            
        # Merge regimes with price data
        data = self.price_data.join(regime_series)
        
        # Calculate regime properties
        regime_stats = {}
        
        for regime in range(self.n_regimes):
            regime_data = data[data['regime'] == regime]
            
            if len(regime_data) == 0:
                continue
                
            # Calculate statistics
            returns = regime_data[self.returns_col].dropna()
            
            regime_stats[regime] = {
                'count': len(regime_data),
                'percent': len(regime_data) / len(data) * 100,
                'mean_return': returns.mean(),
                'std_return': returns.std(),
                'sharpe': returns.mean() / returns.std() if returns.std() > 0 else 0,
                'min_return': returns.min(),
                'max_return': returns.max(),
                'positive_days': (returns > 0).mean() * 100,
                'start_dates': regime_data.index[0].strftime('%Y-%m-%d'),
                'end_dates': regime_data.index[-1].strftime('%Y-%m-%d'),
            }
            
        # Count regime transitions
        transitions = {}
        prev_regime = None
        
        for regime in regime_series:
            if prev_regime is not None:
                transition = (prev_regime, regime)
                transitions[transition] = transitions.get(transition, 0) + 1
                
            prev_regime = regime
            
        # Calculate transition probabilities
        transition_probs = {}
        
        for (from_regime, to_regime), count in transitions.items():
            total = sum(count for (f, t), count in transitions.items() if f == from_regime)
            transition_probs[(from_regime, to_regime)] = count / total
            
        # Plot regime distributions
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot regime over time
        ax = axes[0]
        for regime in range(self.n_regimes):
            regime_data = data[data['regime'] == regime]
            ax.scatter(
                regime_data.index,
                regime_data[self.price_col],
                label=f"Regime {regime}",
                s=10
            )
            
        ax.set_title("Price by Regime")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True)
        
        # Plot return distributions by regime
        ax = axes[1]
        returns_by_regime = {
            regime: data[data['regime'] == regime][self.returns_col].dropna()
            for regime in range(self.n_regimes)
        }
        
        for regime, returns in returns_by_regime.items():
            if len(returns) > 0:
                returns.hist(
                    ax=ax,
                    bins=50,
                    alpha=0.5,
                    label=f"Regime {regime}"
                )
                
        ax.set_title("Return Distributions by Regime")
        ax.set_xlabel("Return")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True)
        
        fig.tight_layout()
        
        return {
            'regime_stats': regime_stats,
            'transitions': transitions,
            'transition_probs': transition_probs,
            'plot': fig
        }


# ================================
# Visualization and Reporting Classes
# ================================

class ReportGenerator:
    """
    Generates reports from experiment results.
    
    This class provides utilities for generating comprehensive reports
    from experiment results, with support for various formats.
    """
    
    def __init__(
        self,
        output_dir: str,
        template_dir: Optional[str] = None
    ):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory for output reports
            template_dir: Directory for report templates
        """
        self.output_dir = output_dir
        self.template_dir = template_dir
        self.logger = logger
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_html_report(
        self,
        experiment_id: str,
        results: Dict[str, Any],
        config: Dict[str, Any],
        metadata: Dict[str, Any],
        template_name: Optional[str] = None
    ) -> str:
        """
        Generate HTML report.
        
        Args:
            experiment_id: Experiment ID
            results: Experiment results
            config: Experiment configuration
            metadata: Experiment metadata
            template_name: Name of template to use
            
        Returns:
            Path to the generated report
        """
        try:
            import jinja2
            
            # Load template
            if template_name and self.template_dir:
                template_path = os.path.join(self.template_dir, template_name)
                
                with open(template_path, 'r') as f:
                    template_str = f.read()
            else:
                # Use default template
                template_str = self._get_default_html_template()
                
            # Create Jinja environment
            env = jinja2.Environment()
            template = env.from_string(template_str)
            
            # Prepare context
            context = {
                'experiment_id': experiment_id,
                'results': results,
                'config': config,
                'metadata': metadata,
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'plots': self._generate_plots(results)
            }
            
            # Render template
            html = template.render(**context)
            
            # Save report
            report_path = os.path.join(self.output_dir, f"{experiment_id}_report.html")
            with open(report_path, 'w') as f:
                f.write(html)
                
            self.logger.info(f"Generated HTML report: {report_path}")
            
            return report_path
            
        except ImportError:
            self.logger.error("jinja2 not installed. Please install with: pip install jinja2")
            raise
            
    def _generate_plots(self, results: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate plots for the report.
        
        Args:
            results: Experiment results
            
        Returns:
            Dictionary of plot paths
        """
        plots = {}
        
        try:
            # Generate equity curve
            if 'portfolio' in results:
                fig, ax = plt.subplots(figsize=(10, 6))
                portfolio = results['portfolio']
                
                ax.plot(portfolio.index, portfolio['equity'])
                ax.set_title("Equity Curve")
                ax.set_xlabel("Date")
                ax.set_ylabel("Equity")
                ax.grid(True)
                
                # Save plot
                plot_path = os.path.join(self.output_dir, "equity_curve.png")
                fig.savefig(plot_path)
                plots['equity_curve'] = plot_path
                
            # Generate drawdown chart
            if 'portfolio' in results and 'drawdown' in results['portfolio'].columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                portfolio = results['portfolio']
                
                ax.fill_between(portfolio.index, 0, portfolio['drawdown'] * 100)
                ax.set_title("Drawdown")
                ax.set_xlabel("Date")
                ax.set_ylabel("Drawdown (%)")
                ax.grid(True)
                
                # Save plot
                plot_path = os.path.join(self.output_dir, "drawdown.png")
                fig.savefig(plot_path)
                plots['drawdown'] = plot_path
                
            # Generate monthly returns heatmap
            if 'portfolio' in results and 'returns' in results['portfolio'].columns:
                from matplotlib.colors import LinearSegmentedColormap
                
                portfolio = results['portfolio']
                
                # Calculate monthly returns
                monthly_returns = portfolio['returns'].resample('M').apply(
                    lambda x: (1 + x).prod() - 1
                )
                
                # Reshape into year x month
                monthly_returns.index = monthly_returns.index.to_period('M')
                monthly_returns_table = monthly_returns.unstack(level=0)
                
                # Create heatmap
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Create colormap
                cmap = LinearSegmentedColormap.from_list(
                    'returns_colormap',
                    ['#d60000', '#FFFFFF', '#4CAF50'],
                    N=256
                )
                
                ax.pcolor(monthly_returns_table, cmap=cmap, vmin=-0.05, vmax=0.05)
                
                # Set labels
                ax.set_title("Monthly Returns Heatmap")
                ax.set_xlabel("Year")
                ax.set_ylabel("Month")
                
                # Set x and y ticks
                ax.set_xticks(np.arange(monthly_returns_table.shape[1]) + 0.5)
                ax.set_yticks(np.arange(monthly_returns_table.shape[0]) + 0.5)
                
                # Set tick labels
                ax.set_xticklabels(monthly_returns_table.columns.astype(str))
                ax.set_yticklabels([f"{i+1}" for i in range(monthly_returns_table.shape[0])])
                
                # Add text annotations
                for i in range(monthly_returns_table.shape[0]):
                    for j in range(monthly_returns_table.shape[1]):
                        value = monthly_returns_table.iloc[i, j]
                        if not np.isnan(value):
                            text_color = 'white' if abs(value) > 0.03 else 'black'
                            ax.text(
                                j + 0.5, i + 0.5, f"{value:.1%}",
                                ha="center", va="center", color=text_color
                            )
                            
                # Save plot
                plot_path = os.path.join(self.output_dir, "monthly_returns.png")
                fig.savefig(plot_path)
                plots['monthly_returns'] = plot_path
                
        except Exception as e:
            self.logger.warning(f"Failed to generate plots: {str(e)}")
            
        return plots
        
    def _get_default_html_template(self) -> str:
        """
        Get default HTML template.
        
        Returns:
            Template string
        """
        return '''<!DOCTYPE html>
<html>
<head>
    <title>Experiment Report: {{ experiment_id }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            color: #333;
        }
        h1, h2, h3 {
            color: #0066cc;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .section {
            margin-bottom: 30px;
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 5px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f2f2f2;
        }
        .metric {
            font-weight: bold;
        }
        .plot {
            margin-top: 20px;
            text-align: center;
        }
        .plot img {
            max-width: 100%;
            height: auto;
        }
        .footer {
            margin-top: 30px;
            text-align: center;
            font-size: 12px;
            color: #999;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Experiment Report: {{ experiment_id }}</h1>
        <p>Generated on {{ timestamp }}</p>
        
        <div class="section">
            <h2>Experiment Information</h2>
            <table>
                <tr>
                    <th>Property</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Experiment ID</td>
                    <td>{{ experiment_id }}</td>
                </tr>
                {% for key, value in metadata.items() %}
                <tr>
                    <td>{{ key }}</td>
                    <td>{{ value }}</td>
                </tr>
                {% endfor %}
            </table>
        </div>
        
        <div class="section">
            <h2>Performance Metrics</h2>
            {% if results.metrics %}
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                {% for key, value in results.metrics.overall.items() %}
                <tr>
                    <td>{{ key }}</td>
                    <td>{{ value }}</td>
                </tr>
                {% endfor %}
            </table>
            {% else %}
            <p>No metrics available.</p>
            {% endif %}
        </div>
        
        <div class="section">
            <h2>Plots</h2>
            {% if plots %}
                {% for name, path in plots.items() %}
                <div class="plot">
                    <h3>{{ name }}</h3>
                    <img src="{{ path }}" alt="{{ name }}">
                </div>
                {% endfor %}
            {% else %}
            <p>No plots available.</p>
            {% endif %}
        </div>
        
        <div class="section">
            <h2>Configuration</h2>
            <pre>{{ config }}</pre>
        </div>
        
        <div class="footer">
            Generated by Experiment Executor
        </div>
    </div>
</body>
</html>
'''
    
    def generate_pdf_report(
        self,
        experiment_id: str,
        results: Dict[str, Any],
        config: Dict[str, Any],
        metadata: Dict[str, Any],
        template_name: Optional[str] = None
    ) -> str:
        """
        Generate PDF report.
        
        Args:
            experiment_id: Experiment ID
            results: Experiment results
            config: Experiment configuration
            metadata: Experiment metadata
            template_name: Name of template to use
            
        Returns:
            Path to the generated report
        """
        try:
            # First generate HTML report
            html_path = self.generate_html_report(
                experiment_id,
                results,
                config,
                metadata,
                template_name
            )
            
            # Convert HTML to PDF
            pdf_path = os.path.join(self.output_dir, f"{experiment_id}_report.pdf")
            
            try:
                import weasyprint
                
                # Convert HTML to PDF
                weasyprint.HTML(filename=html_path).write_pdf(pdf_path)
                
                self.logger.info(f"Generated PDF report: {pdf_path}")
                
                return pdf_path
                
            except ImportError:
                self.logger.warning("weasyprint not installed. Using alternative PDF conversion.")
                
                try:
                    import pdfkit
                    
                    # Convert HTML to PDF
                    pdfkit.from_file(html_path, pdf_path)
                    
                    self.logger.info(f"Generated PDF report: {pdf_path}")
                    
                    return pdf_path
                    
                except ImportError:
                    self.logger.error("No PDF conversion library available. Install weasyprint or pdfkit.")
                    raise
                    
        except Exception as e:
            self.logger.error(f"Failed to generate PDF report: {str(e)}")
            raise
            
    def generate_markdown_report(
        self,
        experiment_id: str,
        results: Dict[str, Any],
        config: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> str:
        """
        Generate Markdown report.
        
        Args:
            experiment_id: Experiment ID
            results: Experiment results
            config: Experiment configuration
            metadata: Experiment metadata
            
        Returns:
            Path to the generated report
        """
        # Generate plots
        plots = self._generate_plots(results)
        
        # Create markdown
        markdown = f"# Experiment Report: {experiment_id}\n\n"
        markdown += f"Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Experiment information
        markdown += "## Experiment Information\n\n"
        markdown += "| Property | Value |\n"
        markdown += "|----------|-------|\n"
        markdown += f"| Experiment ID | {experiment_id} |\n"
        
        for key, value in metadata.items():
            markdown += f"| {key} | {value} |\n"
            
        markdown += "\n"
        
        # Performance metrics
        markdown += "## Performance Metrics\n\n"
        
        if 'metrics' in results and 'overall' in results['metrics']:
            markdown += "| Metric | Value |\n"
            markdown += "|--------|-------|\n"
            
            for key, value in results['metrics']['overall'].items():
                markdown += f"| {key} | {value} |\n"
                
        else:
            markdown += "No metrics available.\n"
            
        markdown += "\n"
        
        # Plots
        markdown += "## Plots\n\n"
        
        if plots:
            for name, path in plots.items():
                markdown += f"### {name}\n\n"
                markdown += f"![{name}]({path})\n\n"
        else:
            markdown += "No plots available.\n\n"
            
        # Configuration
        markdown += "## Configuration\n\n"
        markdown += "```json\n"
        markdown += json.dumps(config, indent=2)
        markdown += "\n```\n"
        
        # Save report
        report_path = os.path.join(self.output_dir, f"{experiment_id}_report.md")
        with open(report_path, 'w') as f:
            f.write(markdown)
            
        self.logger.info(f"Generated Markdown report: {report_path}")
        
        return report_path


class VisualizationManager:
    """
    Manages visualization of experiment results.
    
    This class provides utilities for creating various visualizations
    of experiment results, with support for interactive plots.
    """
    
    def __init__(
        self,
        output_dir: str,
        interactive: bool = True
    ):
        """
        Initialize visualization manager.
        
        Args:
            output_dir: Directory for output visualizations
            interactive: Whether to create interactive visualizations
        """
        self.output_dir = output_dir
        self.interactive = interactive
        self.logger = logger
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def create_equity_curve(
        self,
        portfolio: pd.DataFrame,
        title: str = "Equity Curve",
        include_drawdown: bool = True,
        benchmark: Optional[pd.Series] = None
    ) -> plt.Figure:
        """
        Create equity curve plot.
        
        Args:
            portfolio: Portfolio dataframe
            title: Plot title
            include_drawdown: Whether to include drawdown
            benchmark: Benchmark series
            
        Returns:
            Matplotlib figure
        """
        if 'equity' not in portfolio.columns:
            raise ValueError("Portfolio dataframe must have 'equity' column")
            
        if include_drawdown and 'drawdown' not in portfolio.columns:
            # Calculate drawdown
            portfolio['drawdown'] = 1 - (portfolio['equity'] / portfolio['equity'].cummax())
            
        # Create plot
        if include_drawdown:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        else:
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
        # Plot equity curve
        ax1.plot(portfolio.index, portfolio['equity'], label="Strategy")
        
        # Plot benchmark if provided
        if benchmark is not None:
            # Normalize benchmark to same starting value
            norm_benchmark = benchmark / benchmark.iloc[0] * portfolio['equity'].iloc[0]
            ax1.plot(norm_benchmark.index, norm_benchmark, label="Benchmark", alpha=0.7)
            
        ax1.set_title(title)
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Equity")
        ax1.legend()
        ax1.grid(True)
        
        # Plot drawdown
        if include_drawdown:
            ax2.fill_between(portfolio.index, 0, portfolio['drawdown'] * 100)
            ax2.set_title("Drawdown")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Drawdown (%)")
            ax2.grid(True)
            
        fig.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, "equity_curve.png")
        fig.savefig(plot_path)
        
        return fig
        
    def create_returns_analysis(
        self,
        portfolio: pd.DataFrame,
        title: str = "Returns Analysis"
    ) -> plt.Figure:
        """
        Create returns analysis plots.
        
        Args:
            portfolio: Portfolio dataframe
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        if 'returns' not in portfolio.columns:
            raise ValueError("Portfolio dataframe must have 'returns' column")
            
        # Calculate statistics
        daily_returns = portfolio['returns'].resample('D').sum()
        monthly_returns = portfolio['returns'].resample('M').apply(
            lambda x: (1 + x).prod() - 1
        )
        
        # Create plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot returns histogram
        ax = axes[0, 0]
        daily_returns.hist(bins=50, ax=ax)
        ax.set_title("Daily Returns Distribution")
        ax.set_xlabel("Return")
        ax.set_ylabel("Frequency")
        ax.grid(True)
        
        # Plot returns QQ plot
        ax = axes[0, 1]
        from scipy import stats
        stats.probplot(daily_returns.dropna(), dist="norm", plot=ax)
        ax.set_title("Daily Returns Q-Q Plot")
        ax.grid(True)
        
        # Plot autocorrelation
        ax = axes[1, 0]
        from pandas.plotting import autocorrelation_plot
        autocorrelation_plot(daily_returns.dropna(), ax=ax)
        ax.set_title("Returns Autocorrelation")
        ax.grid(True)
        
        # Plot monthly returns heatmap
        ax = axes[1, 1]
        
        # Prepare data for heatmap
        returns_pivot = pd.DataFrame()
        
        if not monthly_returns.empty:
            returns_pivot = monthly_returns.copy()
            returns_pivot.index = returns_pivot.index.to_period('M')
            
            # Extract month and year
            month = returns_pivot.index.month
            year = returns_pivot.index.year
            
            # Create pivot table
            returns_pivot = pd.DataFrame({
                'month': month,
                'year': year,
                'return': returns_pivot.values
            })
            
            returns_pivot = returns_pivot.pivot(index='month', columns='year', values='return')
            
            # Create heatmap
            im = ax.imshow(returns_pivot, cmap='RdYlGn', vmin=-0.1, vmax=0.1)
            
            # Set labels
            ax.set_title("Monthly Returns Heatmap")
            ax.set_xlabel("Year")
            ax.set_ylabel("Month")
            
            # Set ticks
            ax.set_xticks(np.arange(len(returns_pivot.columns)))
            ax.set_yticks(np.arange(len(returns                group_value = metrics[group_by]
                if group_value not in grouped_metrics:
                    grouped_metrics[group_value] = []
                grouped_metrics[group_value].append(metrics)
                
            # Aggregate within each group
            result = {}
            for group_value, group_metrics in grouped_metrics.items():
                result[group_value] = self._aggregate_metrics_list(group_metrics)
                
            return result
            
        # Aggregate all metrics
        return self._aggregate_metrics_list(metrics_list)
        
    def _aggregate_metrics_list(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate a list of metrics dictionaries.
        
        Args:
            metrics_list: List of metrics dictionaries
            
        Returns:
            Aggregated metrics dictionary
        """
        if not metrics_list:
            return {}
            
        # Find numeric metrics to aggregate
        first_metrics = metrics_list[0]
        numeric_keys = [
            k for k, v in first_metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        ]
        
        # Aggregate numeric metrics
        aggregated = {}
        for key in numeric_keys:
            values = [m[key] for m in metrics_list if key in m]
            if not values:
                continue
                
            aggregated[key + '_mean'] = np.mean(values)
            aggregated[key + '_median'] = np.median(values)
            aggregated[key + '_min'] = min(values)
            aggregated[key + '_max'] = max(values)
            aggregated[key + '_std'] = np.std(values) if len(values) > 1 else 0
            
        # Count total
        aggregated['count'] = len(metrics_list)
        
        return aggregated
    
    def compare_experiments(
        self,
        experiment_ids: List[str],
        metrics_keys: List[str]
    ) -> pd.DataFrame:
        """
        Compare metrics across experiments.
        
        Args:
            experiment_ids: List of experiment IDs
            metrics_keys: List of metrics keys to compare
            
        Returns:
            Dataframe with comparison
        """
        # Load metrics for each experiment
        experiment_metrics = []
        
        for exp_id in experiment_ids:
            # Find metrics files for this experiment
            pattern = os.path.join(self.metrics_dir, f"{exp_id}_*_metrics.json")
            metrics_files = glob.glob(pattern)
            
            if not metrics_files:
                self.logger.warning(f"No metrics found for experiment {exp_id}")
                continue
                
            # Load latest metrics file
            latest_file = sorted(metrics_files)[-1]
            metrics = self.load_metrics(latest_file)
            metrics['experiment_id'] = exp_id
            experiment_metrics.append(metrics)
            
        if not experiment_metrics:
            return pd.DataFrame()
            
        # Create comparison dataframe
        comparison = []
        
        for metrics in experiment_metrics:
            row = {'experiment_id': metrics['experiment_id']}
            
            for key in metrics_keys:
                if key in metrics:
                    row[key] = metrics[key]
                else:
                    nested_keys = key.split('.')
                    value = metrics
                    for nk in nested_keys:
                        if isinstance(value, dict) and nk in value:
                            value = value[nk]
                        else:
                            value = None
                            break
                    row[key] = value
                    
            comparison.append(row)
            
        return pd.DataFrame(comparison)


class WandbLogger:
    """
    Logs experiment metrics and artifacts to Weights & Biases.
    
    This class provides utilities for logging experiment data to Weights & Biases,
    a popular tool for experiment tracking and visualization.
    """
    
    def __init__(
        self,
        project: str,
        entity: Optional[str] = None,
        experiment_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        group: Optional[str] = None,
        notes: Optional[str] = None,
        resume: bool = False
    ):
        """
        Initialize W&B logger.
        
        Args:
            project: W&B project name
            entity: W&B entity
            experiment_name: Name for the experiment
            config: Configuration dictionary
            tags: Tags for the experiment
            group: Group for the experiment
            notes: Notes for the experiment
            resume: Whether to resume a previous run
        """
        self.project = project
        self.entity = entity
        self.experiment_name = experiment_name
        self.config = config
        self.tags = tags
        self.group = group
        self.notes = notes
        self.resume = resume
        self.logger = logger
        self.run = None
        
    def start(self) -> None:
        """Start W&B logging."""
        try:
            self.run = wandb.init(
                project=self.project,
                entity=self.entity,
                name=self.experiment_name,
                config=self.config,
                tags=self.tags,
                group=self.group,
                notes=self.notes,
                resume=self.resume
            )
            self.logger.info(f"Started W&B logging: {self.run.name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize W&B: {str(e)}")
            self.run = None
            
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics to W&B.
        
        Args:
            metrics: Metrics dictionary
            step: Step for the metrics
        """
        if not self.run:
            return
            
        try:
            # Flatten nested dictionaries
            flat_metrics = {}
            
            def flatten_dict(d, prefix=''):
                for k, v in d.items():
                    if isinstance(v, dict):
                        flatten_dict(v, prefix + k + '/')
                    elif isinstance(v, (int, float)) and not isinstance(v, bool):
                        flat_metrics[prefix + k] = v
                        
            flatten_dict(metrics)
            
            # Log metrics
            self.run.log(flat_metrics, step=step)
            
        except Exception as e:
            self.logger.warning(f"Failed to log metrics to W&B: {str(e)}")
            
    def log_artifact(
        self,
        artifact_path: str,
        name: str,
        artifact_type: str,
        description: Optional[str] = None
    ) -> None:
        """
        Log an artifact to W&B.
        
        Args:
            artifact_path: Path to the artifact
            name: Name for the artifact
            artifact_type: Type of the artifact
            description: Description of the artifact
        """
        if not self.run:
            return
            
        try:
            artifact = wandb.Artifact(
                name=name,
                type=artifact_type,
                description=description
            )
            artifact.add_file(artifact_path)
            self.run.log_artifact(artifact)
            
        except Exception as e:
            self.logger.warning(f"Failed to log artifact to W&B: {str(e)}")
            
    def log_dataframe(
        self,
        df: pd.DataFrame,
        name: str,
        description: Optional[str] = None
    ) -> None:
        """
        Log a dataframe as a table to W&B.
        
        Args:
            df: Dataframe to log
            name: Name for the table
            description: Description of the table
        """
        if not self.run:
            return
            
        try:
            table = wandb.Table(dataframe=df)
            self.run.log({name: table})
            
        except Exception as e:
            self.logger.warning(f"Failed to log dataframe to W&B: {str(e)}")
            
    def log_plot(
        self,
        figure: plt.Figure,
        name: str
    ) -> None:
        """
        Log a matplotlib figure to W&B.
        
        Args:
            figure: Matplotlib figure
            name: Name for the plot
        """
        if not self.run:
            return
            
        try:
            self.run.log({name: wandb.Image(figure)})
            
        except Exception as e:
            self.logger.warning(f"Failed to log plot to W&B: {str(e)}")
            
    def finish(self) -> None:
        """Finish W&B logging."""
        if not self.run:
            return
            
        try:
            self.run.finish()
            self.logger.info("Finished W&B logging")
        except Exception as e:
            self.logger.warning(f"Failed to finish W&B run: {str(e)}")


# ================================
# Advanced Analysis Classes
# ================================

class SensitivityAnalyzer:
    """
    Analyzes sensitivity of results to parameter changes.
    
    This class provides utilities for analyzing how changes in parameters
    affect the performance of a model or strategy.
    """
    
    def __init__(
        self,
        executor_cls: Type,
        base_config: Dict[str, Any],
        output_dir: str
    ):
        """
        Initialize sensitivity analyzer.
        
        Args:
            executor_cls: Experiment executor class
            base_config: Base configuration
            output_dir: Output directory
        """
        self.executor_cls = executor_cls
        self.base_config = base_config
        self.output_dir = output_dir
        self.logger = logger
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def _update_config(
        self,
        config: Dict[str, Any],
        param_path: str,
        value: Any
    ) -> Dict[str, Any]:
        """
        Update a nested configuration value.
        
        Args:
            config: Configuration dictionary
            param_path: Path to parameter (dot-separated)
            value: New value
            
        Returns:
            Updated configuration
        """
        new_config = copy.deepcopy(config)
        
        # Navigate to parameter
        parts = param_path.split('.')
        current = new_config
        
        for i, part in enumerate(parts[:-1]):
            if part not in current:
                current[part] = {}
            current = current[part]
            
        # Update value
        current[parts[-1]] = value
        
        return new_config
        
    def analyze_parameter(
        self,
        param_path: str,
        values: List[Any],
        metric_key: str,
        higher_is_better: bool = True,
        n_jobs: int = 1
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Analyze sensitivity to a single parameter.
        
        Args:
            param_path: Path to parameter (dot-separated)
            values: List of values to test
            metric_key: Key for the metric to track
            higher_is_better: Whether higher metric values are better
            n_jobs: Number of parallel jobs
            
        Returns:
            Tuple of (results_df, best_config)
        """
        self.logger.info(f"Analyzing sensitivity to parameter: {param_path}")
        
        # Prepare configurations
        configs = []
        for value in values:
            config = self._update_config(self.base_config, param_path, value)
            config['experiment_name'] = f"sensitivity_{param_path.replace('.', '_')}_{value}"
            configs.append((config, value))
            
        # Run experiments
        results = []
        
        if n_jobs > 1:
            # Run in parallel
            with multiprocessing.Pool(n_jobs) as pool:
                results = pool.map(self._run_experiment, configs)
        else:
            # Run sequentially
            for config_value in configs:
                results.append(self._run_experiment(config_value))
                
        # Compile results
        results_df = pd.DataFrame(results)
        
        # Find best configuration
        if not results_df.empty:
            if higher_is_better:
                best_idx = results_df[metric_key].idxmax()
            else:
                best_idx = results_df[metric_key].idxmin()
                
            best_value = results_df.loc[best_idx, 'value']
            best_config = self._update_config(self.base_config, param_path, best_value)
            best_config['experiment_name'] = f"best_{param_path.replace('.', '_')}"
            
            self.logger.info(f"Best value for {param_path}: {best_value} ({metric_key}: {results_df.loc[best_idx, metric_key]})")
            
        else:
            best_config = copy.deepcopy(self.base_config)
            
        # Plot results
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(results_df['value'], results_df[metric_key], 'o-')
        ax.set_xlabel(param_path)
        ax.set_ylabel(metric_key)
        ax.set_title(f"Sensitivity to {param_path}")
        ax.grid(True)
        
        if 'value' in results_df.columns and isinstance(results_df['value'].iloc[0], (int, float, bool, str)):
            for i, row in results_df.iterrows():
                ax.annotate(f"{row[metric_key]:.4f}", 
                           (row['value'], row[metric_key]),
                           textcoords="offset points",
                           xytext=(0, 10),
                           ha='center')
                
        fig.tight_layout()
        
        # Save plot
        plot_file = os.path.join(self.output_dir, f"sensitivity_{param_path.replace('.', '_')}.png")
        fig.savefig(plot_file)
        
        # Save results
        results_file = os.path.join(self.output_dir, f"sensitivity_{param_path.replace('.', '_')}.csv")
        results_df.to_csv(results_file, index=False)
        
        return results_df, best_config
        
    def _run_experiment(self, config_value: Tuple[Dict[str, Any], Any]) -> Dict[str, Any]:
        """
        Run a single experiment.
        
        Args:
            config_value: Tuple of (config, parameter_value)
            
        Returns:
            Results dictionary
        """
        config, value = config_value
        
        try:
            # Create and run executor
            executor = self.executor_cls(config)
            metrics = executor.run()
            
            # Extract relevant metric
            result = {
                'value': value,
                'status': 'success'
            }
            
            # Extract nested metrics
            metric_path = config.get('sensitivity_metric', 'metrics.overall.total_return')
            metric_parts = metric_path.split('.')
            
            metric_value = metrics
            for part in metric_parts:
                if isinstance(metric_value, dict) and part in metric_value:
                    metric_value = metric_value[part]
                else:
                    metric_value = None
                    break
                    
            result[metric_path] = metric_value
            
            return result
            
        except Exception as e:
            self.logger.error(f"Experiment failed for value {value}: {str(e)}")
            return {
                'value': value,
                'status': 'failed',
                'error': str(e)
            }
            
    def analyze_multiple_parameters(
        self,
        params: Dict[str, List[Any]],
        metric_key: str,
        higher_is_better: bool = True,
        n_jobs: int = 1
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Analyze sensitivity to multiple parameters.
        
        Args:
            params: Dictionary of parameter paths and values
            metric_key: Key for the metric to track
            higher_is_better: Whether higher metric values are better
            n_jobs: Number of parallel jobs
            
        Returns:
            Tuple of (results_df, best_config)
        """
        self.logger.info(f"Analyzing sensitivity to multiple parameters: {', '.join(params.keys())}")
        
        # Generate parameter combinations
        param_names = list(params.keys())
        param_values = list(params.values())
        combinations = list(itertools.product(*param_values))
        
        # Prepare configurations
        configs = []
        for combo in combinations:
            config = copy.deepcopy(self.base_config)
            combo_name = []
            
            for name, value in zip(param_names, combo):
                config = self._update_config(config, name, value)
                combo_name.append(f"{name.split('.')[-1]}_{value}")
                
            config['experiment_name'] = f"sensitivity_{'_'.join(combo_name)}"
            configs.append((config, dict(zip(param_names, combo))))
            
        # Run experiments
        results = []
        
        if n_jobs > 1:
            # Run in parallel
            with multiprocessing.Pool(n_jobs) as pool:
                results = pool.map(self._run_multi_experiment, configs)
        else:
            # Run sequentially
            for config_params in configs:
                results.append(self._run_multi_experiment(config_params))
                
        # Compile results
        results_df = pd.DataFrame(results)
        
        # Find best configuration
        if not results_df.empty and metric_key in results_df.columns:
            if higher_is_better:
                best_idx = results_df[metric_key].idxmax()
            else:
                best_idx = results_df[metric_key].idxmin()
                
            best_params = eval(results_df.loc[best_idx, 'params'])
            best_config = copy.deepcopy(self.base_config)
            
            for param_path, value in best_params.items():
                best_config = self._update_config(best_config, param_path, value)
                
            best_config['experiment_name'] = "best_combined_params"
            
            self.logger.info(f"Best parameters: {best_params} ({metric_key}: {results_df.loc[best_idx, metric_key]})")
            
        else:
            best_config = copy.deepcopy(self.base_config)
            
        # Save results
        results_file = os.path.join(self.output_dir, "sensitivity_multi_params.csv")
        results_df.to_csv(results_file, index=False)
        
        return results_df, best_config
        
    def _run_multi_experiment(self, config_params: Tuple[Dict[str, Any], Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run a single multi-parameter experiment.
        
        Args:
            config_params: Tuple of (config, parameter_dict)
            
        Returns:
            Results dictionary
        """
        config, params = config_params
        
        try:
            # Create and run executor
            executor = self.executor_cls(config)
            metrics = executor.run()
            
            # Extract relevant metric
            result = {
                'params': str(params),
            }
            
            # Add individual parameters
            for param_path, value in params.items():
                param_name = param_path.replace('.', '_')
                result[param_name] = value
                
            # Extract nested metrics
            metric_path = config.get('sensitivity_metric', 'metrics.overall.total_return')
            metric_parts = metric_path.split('.')
            
            metric_value = metrics
            for part in metric_parts:
                if isinstance(metric_value, dict) and part in metric_value:
                    metric_value = metric_value[part]
                else:
                    metric_value = None
                    break
                    
            result[metric_path] = metric_value
            result['status'] = 'success'
            
            return result
            
        except Exception as e:
            self.logger.error(f"Experiment failed for params {params}: {str(e)}")
            return {
                'params': str(params),
                'status': 'failed',
                'error': str(e)
            }


class MonteCarloSimulator:
    """
    Performs Monte Carlo simulations of trading strategies.
    
    This class provides utilities for assessing the robustness of a trading
    strategy through randomized trials.
    """
    
    def __init__(
        self,
        strategy_func: Callable,
        price_data: pd.DataFrame,
        price_col: str = 'close',
        random_seed: Optional[int] = None
    ):
        """
        Initialize Monte Carlo simulator.
        
        Args:
            strategy_func: Function that generates trading signals
            price_data: Price data dataframe
            price_col: Column name for prices
            random_seed: Random seed for reproducibility
        """
        self.strategy_func = strategy_func
        self.price_data = price_data
        self.price_col = price_col
        self.random_seed = random_seed
        self.logger = logger
        
        if random_seed is not None:
            np.random.seed(random_seed)
            
    def run_simulation(
        self,
        n_simulations: int = 100,
        window_size: Optional[int] = None,
        initial_capital: float = 1e6,
        transaction_cost: float = 0.0003,
        bootstrap_method: str = 'block',
        block_size: int = 20,
        parallel: bool = True,
        n_jobs: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulations.
        
        Args:
            n_simulations: Number of simulations
            window_size: Window size for simulations
            initial_capital: Initial capital
            transaction_cost: Transaction cost
            bootstrap_method: Bootstrap method ('block' or 'stationary')
            block_size: Block size for block bootstrap
            parallel: Whether to run simulations in parallel
            n_jobs: Number of parallel jobs
            
        Returns:
            Dictionary with simulation results
        """
        self.logger.info(f"Running {n_simulations} Monte Carlo simulations")
        
        # Set up backtest engine
        backtest_engine = BacktestEngine(
            initial_capital=initial_capital,
            transaction_cost=transaction_cost
        )
        
        # Set up parallel execution
        if parallel:
            n_jobs = n_jobs or max(1, multiprocessing.cpu_count() - 1)
            
        # Generate simulation seeds
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            
        seeds = np.random.randint(0, 2**32 - 1, size=n_simulations)
        
        # Run simulations
        if parallel and n_jobs > 1:
            with multiprocessing.Pool(n_jobs) as pool:
                simulation_results = list(
                    tqdm(
                        pool.imap(
                            lambda seed: self._run_single_simulation(
                                seed, window_size, bootstrap_method, block_size, backtest_engine
                            ),
                            seeds
                        ),
                        total=n_simulations,
                        desc="Running simulations"
                    )
                )
        else:
            simulation_results = []
            for seed in tqdm(seeds, desc="Running simulations"):
                result = self._run_single_simulation(
                    seed, window_size, bootstrap_method, block_size, backtest_engine
                )
                simulation_results.append(result)
                
        # Aggregate results
        return self._aggregate_simulation_results(simulation_results)
        
    def _run_single_simulation(
        self,
        seed: int,
        window_size: Optional[int],
        bootstrap_method: str,
        block_size: int,
        backtest_engine: BacktestEngine
    ) -> Dict[str, Any]:
        """
        Run a single simulation.
        
        Args:
            seed: Random seed
            window_size: Window size
            bootstrap_method: Bootstrap method
            block_size: Block size
            backtest_engine: Backtest engine
            
        Returns:
            Simulation results
        """
        np.random.seed(seed)
        
        try:
            # Generate resampled data
            if bootstrap_method == 'block':
                data = self._block_bootstrap(window_size)
            elif bootstrap_method == 'stationary':
                data = self._stationary_bootstrap(window_size, block_size)
            else:
                raise ValueError(f"Unsupported bootstrap method: {bootstrap_method}")
                
            # Generate signals
            signals = self.strategy_func(data)
            
            # Run backtest
            result = backtest_engine.run_backtest(
                data, signals, price_col=self.price_col, verbose=False
            )
            
            # Add simulation metadata
            result['simulation_id'] = seed
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Simulation {seed} failed: {str(e)}")
            return {
                'simulation_id': seed,
                'status': 'failed',
                'error': str(e)
            }
            
    def _block_bootstrap(self, window_size: Optional[int]) -> pd.DataFrame:
        """
        Perform block bootstrap resampling.
        
        Args:
            window_size: Window size
            
        Returns:
            Resampled dataframe
        """
        data = self.price_data.copy()
        
        if window_size is None:
            return data
            
        # Determine number of blocks
        n_samples = len(data)
        n_blocks = int(np.ceil(n_samples / window_size))
        
        # Sample blocks with replacement
        block_indices = np.random.choice(n_samples - window_size + 1, size=n_blocks, replace=True)
        
        # Concatenate blocks
        blocks = []
        for start_idx in block_indices:
            block = data.iloc[start_idx:start_idx + window_size].copy()
            blocks.append(block)
            
        # Combine blocks and trim to original length
        resampled = pd.concat(blocks).iloc[:n_samples]
        resampled.index = data.index
        
        return resampled
        
    def _stationary_bootstrap(self, window_size: Optional[int], block_size: int) -> pd.DataFrame:
        """
        Perform stationary bootstrap resampling.
        
        Args:
            window_size: Window size
            block_size: Block size
            
        Returns:
            Resampled dataframe
        """
        data = self.price_data.copy()
        
        if window_size is None:
            return data
            
        # Determine number of samples
        n_samples = len(data)
        
        # Initialize resampled data
        resampled_indices = []
        current_idx = np.random.choice(n_samples)
        
        while len(resampled_indices) < n_samples:
            # Add current index
            resampled_indices.append(current_idx)
            
            # Determine block length
            p = 1 / block_size
            geom_rv = np.random.geometric(p)
            block_length = min(geom_rv, n_samples - len(resampled_indices))
            
            # Add indices for the block
            for j in range(1, block_length):
                next_idx = (current_idx + j) % n_samples
                resampled_indices.append(next_idx)
                
            # Randomly select next starting point
            current_idx = np.random.choice(n_samples)
            
        # Trim to original length
        resampled_indices = resampled_indices[:n_samples]
        resampled = data.iloc[resampled_indices].copy()
        resampled.index = data.index
        
        return resampled
        
    def _aggregate_simulation_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate simulation results.
        
        Args:
            results: List of simulation results
            
        Returns:
            Aggregated results
        """
        # Extract metrics
        metrics = []
        for result in results:
            if 'metrics' in result:
                metrics.append(result['metrics'])
                
        if not metrics:
            return {'status': 'failed', 'error': 'No successful simulations'}
            
        # Create dataframe of metrics
        metrics_df = pd.DataFrame(metrics)
        
        # Calculate statistics
        stats = {}
        
        for col in metrics_df.columns:
            if pd.api.types.is_numeric_dtype(metrics_df[col]):
                stats[col] = {
                    'mean': metrics_df[col].mean(),
                    'median': metrics_df[col].median(),
                    'std': metrics_df[col].std(),
                    'min': metrics_df[col].min(),
                    'max': metrics_df[col].max(),
                    'p5': metrics_df[col].quantile(0.05),
                    'p25': metrics_df[col].quantile(0.25),
                    'p75': metrics_df[col].quantile(0.75),
                    'p95': metrics_df[col].quantile(0.95)
                }
                
        # Plot key metrics
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        metrics_to_plot = [
            'total_return_pct',
            'annualized_return_pct',
            'sharpe_ratio',
            'max_drawdown_pct'
        ]
        
        for i, metric in enumerate(metrics_to_plot):
            if metric in metrics_df.columns:
                ax = axes[i // 2, i % 2]
                metrics_df[metric].hist(bins=20, ax=ax)
                ax.axvline(metrics_df[metric].mean(), color='r', linestyle='--')
                ax.set_title(f"{metric}\nMean: {metrics_df[metric].mean():.2f}")
                ax.grid(True)
                
        fig.tight_layout()
        
        return {
            'status': 'success',
            'n_simulations': len(results),
            'successful_simulations': len(metrics),
            'metrics_df': metrics_df,
            'statistics': stats,
            'plot': fig
        }


class MarketRegimeDetector:
    """
    Detects market regimes from price data.
    
    This class provides utilities for identifying different market regimes,
    such as trending or mean-reverting periods.
    """
    
    def __init__(
        self,
        price_data: pd.DataFrame,
        price_col: str = 'close',
        returns_col: Optional[str] = None,
        n_regimes: int = 2,
        model_type: str = 'hmm'
    ):
        """
        Initialize market regime detector.
        
        Args:
            price_data: Price data dataframe
            price_col: Column name for prices
            returns_col: Column name for returns (calculated if None)
            n_regimes: Number of regimes to detect
            model_type: Model type ('hmm' or 'kmeans')
        """
        self.price_data = price_data
        self.price_col = price_col
        self.returns_col = returns_"""
Experiment Executor Module

This module provides a comprehensive experiment execution framework for trading strategy development.
It orchestrates the entire process from configuration loading to results reporting, with support for
hardware abstraction, GPU optimization, experiment tracking, and reproducibility.

Features:
- Dynamic configuration management with validation
- Experiment lifecycle management with checkpointing
- Resource optimization for CPU/GPU
- Integrated data pipeline orchestration
- Model management with versioning
- Backtesting engine integration
- Performance tracking and visualization
- Advanced analysis capabilities
"""

import os
import sys
import json
import yaml
import time
import uuid
import logging
import shutil
import pickle
import hashlib
import datetime
import traceback
import multiprocessing
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Type, Set
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
import jsonschema
import git
from git import Repo
import psutil
import cupy as cp
from tqdm.auto import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)

logger = logging.getLogger("ExperimentExecutor")

# ================================
# Configuration Management Classes
# ================================

class ConfigFormat(Enum):
    """Enum for supported configuration formats."""
    JSON = "json"
    YAML = "yaml"
    PYTHON = "py"


@dataclass
class ExperimentConfig:
    """
    Dataclass representing experiment configuration with versioning.
    
    Attributes:
        name: Name of the experiment
        version: Version of the configuration
        description: Description of the experiment
        tags: Tags associated with the experiment
        environment: Target environment (dev/prod)
        data_config: Configuration for data sources and preprocessing
        model_config: Configuration for model and hyperparameters
        backtesting_config: Configuration for backtesting engine
        hardware_config: Configuration for hardware resources
        metrics_config: Configuration for performance metrics
        visualization_config: Configuration for visualizations
        advanced_analysis_config: Configuration for advanced analyses
        metadata: Additional metadata
    """
    name: str
    version: str = "0.1.0"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    environment: str = "dev"
    data_config: Dict[str, Any] = field(default_factory=dict)
    model_config: Dict[str, Any] = field(default_factory=dict)
    backtesting_config: Dict[str, Any] = field(default_factory=dict)
    hardware_config: Dict[str, Any] = field(default_factory=dict)
    metrics_config: Dict[str, Any] = field(default_factory=dict)
    visualization_config: Dict[str, Any] = field(default_factory=dict)
    advanced_analysis_config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert config to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def to_yaml(self) -> str:
        """Convert config to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False)
    
    def save(self, path: str, format: ConfigFormat = ConfigFormat.JSON) -> None:
        """Save configuration to file."""
        with open(path, 'w') as f:
            if format == ConfigFormat.JSON:
                json.dump(self.to_dict(), f, indent=2)
            elif format == ConfigFormat.YAML:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ExperimentConfig':
        """Create config from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    @classmethod
    def from_yaml(cls, yaml_str: str) -> 'ExperimentConfig':
        """Create config from YAML string."""
        return cls.from_dict(yaml.safe_load(yaml_str))
    
    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        """Load configuration from file."""
        file_ext = os.path.splitext(path)[1].lower()
        
        with open(path, 'r') as f:
            if file_ext == '.json':
                return cls.from_dict(json.load(f))
            elif file_ext in ['.yaml', '.yml']:
                return cls.from_dict(yaml.safe_load(f))
            elif file_ext == '.py':
                # Load Python module dynamically
                spec = importlib.util.spec_from_file_location("config_module", path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Look for CONFIG dictionary or get_config function
                if hasattr(module, 'CONFIG'):
                    return cls.from_dict(module.CONFIG)
                elif hasattr(module, 'get_config'):
                    return cls.from_dict(module.get_config())
                else:
                    raise ValueError(f"Python config file must contain CONFIG dict or get_config function: {path}")
            else:
                raise ValueError(f"Unsupported file extension: {file_ext}")


class ConfigValidator:
    """
    Validates experiment configurations against schema.
    
    This class ensures that experiment configurations meet the required structure
    and contain valid parameter values before experiments are executed.
    """
    
    def __init__(self, schema_path: str = None):
        """
        Initialize validator with schema.
        
        Args:
            schema_path: Path to JSON schema file
        """
        self.schema = None
        if schema_path:
            with open(schema_path, 'r') as f:
                self.schema = json.load(f)
    
    def validate(self, config: Union[Dict[str, Any], ExperimentConfig]) -> Tuple[bool, List[str]]:
        """
        Validate configuration against schema.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        if isinstance(config, ExperimentConfig):
            config_dict = config.to_dict()
        else:
            config_dict = config
            
        if not self.schema:
            logger.warning("No schema provided for validation. Skipping validation.")
            return True, []
            
        validator = jsonschema.Draft7Validator(self.schema)
        errors = list(validator.iter_errors(config_dict))
        
        if errors:
            error_messages = [
                f"{error.path}: {error.message}" for error in errors
            ]
            return False, error_messages
        
        return True, []
    
    def validate_with_custom_rules(
        self, 
        config: Union[Dict[str, Any], ExperimentConfig],
        custom_rules: List[Callable[[Dict[str, Any]], Tuple[bool, str]]]
    ) -> Tuple[bool, List[str]]:
        """
        Validate configuration with both schema and custom rules.
        
        Args:
            config: Configuration to validate
            custom_rules: List of functions that validate config and return (is_valid, error_message)
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        # First validate with schema
        is_valid, errors = self.validate(config)
        
        if isinstance(config, ExperimentConfig):
            config_dict = config.to_dict()
        else:
            config_dict = config
            
        # Then apply custom rules
        for rule_func in custom_rules:
            rule_valid, error_msg = rule_func(config_dict)
            if not rule_valid:
                is_valid = False
                errors.append(error_msg)
                
        return is_valid, errors


class ConfigManager:
    """
    Manages configuration loading, validation, and versioning.
    
    This class handles the loading of configurations from various sources,
    applies validation, merges with defaults, and tracks configuration versions.
    """
    
    def __init__(
        self, 
        default_config_path: str,
        config_schema_path: Optional[str] = None,
        custom_validators: Optional[List[Callable]] = None
    ):
        """
        Initialize config manager.
        
        Args:
            default_config_path: Path to default configuration file
            config_schema_path: Path to JSON schema for validation
            custom_validators: List of custom validation functions
        """
        self.default_config_path = default_config_path
        self.validator = ConfigValidator(config_schema_path)
        self.custom_validators = custom_validators or []
        self.config_history: List[ExperimentConfig] = []
        
        # Load default configuration
        self.default_config = self._load_config(default_config_path)
        
    def _load_config(self, config_path: str) -> ExperimentConfig:
        """Load configuration from file."""
        try:
            return ExperimentConfig.load(config_path)
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {str(e)}")
            raise
            
    def load_and_validate(
        self, 
        config_path: Optional[str] = None,
        override_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[ExperimentConfig, bool]:
        """
        Load, validate and merge configuration.
        
        Args:
            config_path: Path to experiment-specific configuration
            override_params: Parameters to override in the configuration
            
        Returns:
            Tuple of (merged_config, is_valid)
        """
        # Start with default config
        merged_config = self.default_config
        
        # Load experiment-specific config if provided
        if config_path:
            try:
                exp_config = self._load_config(config_path)
                # Merge with default config
                merged_dict = {**merged_config.to_dict(), **exp_config.to_dict()}
                merged_config = ExperimentConfig.from_dict(merged_dict)
            except Exception as e:
                logger.error(f"Error loading experiment config from {config_path}: {str(e)}")
                return merged_config, False
                
        # Apply override parameters if provided
        if override_params:
            merged_dict = merged_config.to_dict()
            self._deep_update(merged_dict, override_params)
            merged_config = ExperimentConfig.from_dict(merged_dict)
            
        # Validate the merged configuration
        is_valid, errors = self.validator.validate_with_custom_rules(
            merged_config, self.custom_validators
        )
        
        if not is_valid:
            logger.error("Configuration validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
                
        # Add to history if valid
        if is_valid:
            self.config_history.append(merged_config)
            
        return merged_config, is_valid
    
    def _deep_update(self, d: Dict[str, Any], u: Dict[str, Any]) -> None:
        """
        Recursively update a dictionary with another dictionary.
        
        Args:
            d: Dictionary to update
            u: Dictionary with updates
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._deep_update(d[k], v)
            else:
                d[k] = v
    
    def get_config_hash(self, config: ExperimentConfig) -> str:
        """
        Generate a unique hash for a configuration.
        
        Args:
            config: Configuration to hash
            
        Returns:
            SHA-256 hash of the configuration
        """
        config_str = json.dumps(config.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def save_config_snapshot(
        self, 
        config: ExperimentConfig,
        output_dir: str,
        format: ConfigFormat = ConfigFormat.JSON
    ) -> str:
        """
        Save a snapshot of the configuration.
        
        Args:
            config: Configuration to save
            output_dir: Directory to save the snapshot
            format: Format to save the configuration in
            
        Returns:
            Path to the saved snapshot
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config_hash = self.get_config_hash(config)
        
        filename = f"config_{timestamp}_{config_hash[:8]}"
        
        if format == ConfigFormat.JSON:
            filepath = os.path.join(output_dir, f"{filename}.json")
        elif format == ConfigFormat.YAML:
            filepath = os.path.join(output_dir, f"{filename}.yaml")
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        config.save(filepath, format)
        logger.info(f"Saved configuration snapshot to {filepath}")
        
        return filepath


# ================================
# Experiment Lifecycle Management
# ================================

class ExperimentStatus(Enum):
    """Enum for experiment status."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@dataclass
class ExperimentMetadata:
    """
    Dataclass for experiment metadata.
    
    Attributes:
        id: Unique identifier for the experiment
        name: Name of the experiment
        status: Current status of the experiment
        start_time: Start time of the experiment
        end_time: End time of the experiment
        config_hash: Hash of the configuration
        config_path: Path to the configuration file
        output_dir: Directory for experiment outputs
        log_file: Path to the log file
        checkpoint_dir: Directory for checkpoints
        git_commit: Git commit hash
        tags: Tags associated with the experiment
        notes: Additional notes about the experiment
    """
    id: str
    name: str
    status: ExperimentStatus = ExperimentStatus.INITIALIZING
    start_time: Optional[datetime.datetime] = None
    end_time: Optional[datetime.datetime] = None
    config_hash: Optional[str] = None
    config_path: Optional[str] = None
    output_dir: Optional[str] = None
    log_file: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    git_commit: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        result = asdict(self)
        # Convert enum to string
        result['status'] = self.status.value
        # Convert datetimes to strings
        if self.start_time:
            result['start_time'] = self.start_time.isoformat()
        if self.end_time:
            result['end_time'] = self.end_time.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentMetadata':
        """Create metadata from dictionary."""
        # Handle enum
        if 'status' in data:
            data['status'] = ExperimentStatus(data['status'])
        # Handle datetimes
        if 'start_time' in data and data['start_time']:
            data['start_time'] = datetime.datetime.fromisoformat(data['start_time'])
        if 'end_time' in data and data['end_time']:
            data['end_time'] = datetime.datetime.fromisoformat(data['end_time'])
        return cls(**data)


class ExperimentManager:
    """
    Manages experiment lifecycle, checkpointing, and versioning.
    
    This class handles the initialization, execution, and finalization of experiments,
    as well as checkpointing for resumable experiments and experiment versioning.
    """
    
    def __init__(
        self, 
        base_dir: str,
        config_manager: ConfigManager,
        enable_git_tracking: bool = True
    ):
        """
        Initialize experiment manager.
        
        Args:
            base_dir: Base directory for experiment outputs
            config_manager: Configuration manager instance
            enable_git_tracking: Whether to track git information
        """
        self.base_dir = base_dir
        self.config_manager = config_manager
        self.enable_git_tracking = enable_git_tracking
        self.current_experiment: Optional[ExperimentMetadata] = None
        self.logger = logger
        
        # Create base directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)
        
        # Initialize git repository info if tracking is enabled
        self.git_repo = None
        self.git_root = None
        if enable_git_tracking:
            try:
                self.git_repo = git.Repo(search_parent_directories=True)
                self.git_root = self.git_repo.git.rev_parse("--show-toplevel")
            except git.InvalidGitRepositoryError:
                self.logger.warning("No git repository found. Git tracking disabled.")
                self.enable_git_tracking = False
                
    def _setup_experiment_directories(self, experiment_id: str, config: ExperimentConfig) -> Dict[str, str]:
        """
        Set up directories for an experiment.
        
        Args:
            experiment_id: Unique identifier for the experiment
            config: Experiment configuration
            
        Returns:
            Dictionary of directory paths
        """
        # Create main experiment directory
        exp_dir = os.path.join(self.base_dir, experiment_id)
        os.makedirs(exp_dir, exist_ok=True)
        
        # Create subdirectories
        dirs = {
            'root': exp_dir,
            'checkpoints': os.path.join(exp_dir, 'checkpoints'),
            'logs': os.path.join(exp_dir, 'logs'),
            'models': os.path.join(exp_dir, 'models'),
            'results': os.path.join(exp_dir, 'results'),
            'visualizations': os.path.join(exp_dir, 'visualizations'),
            'configs': os.path.join(exp_dir, 'configs'),
            'metrics': os.path.join(exp_dir, 'metrics'),
            'reports': os.path.join(exp_dir, 'reports'),
        }
        
        # Create all directories
        for dir_path in dirs.values():
            os.makedirs(dir_path, exist_ok=True)
            
        return dirs
        
    def _setup_experiment_logging(self, experiment_id: str, log_dir: str) -> str:
        """
        Set up logging for an experiment.
        
        Args:
            experiment_id: Unique identifier for the experiment
            log_dir: Directory for log files
            
        Returns:
            Path to the log file
        """
        log_file = os.path.join(log_dir, f"{experiment_id}.log")
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
        
        return log_file
        
    def _get_git_info(self) -> Dict[str, str]:
        """
        Get current git repository information.
        
        Returns:
            Dictionary with git information
        """
        if not self.enable_git_tracking or not self.git_repo:
            return {}
            
        try:
            return {
                'commit': self.git_repo.head.commit.hexsha,
                'branch': self.git_repo.active_branch.name,
                'author': self.git_repo.head.commit.author.name,
                'authored_date': datetime.datetime.fromtimestamp(
                    self.git_repo.head.commit.authored_date
                ).isoformat(),
                'is_dirty': self.git_repo.is_dirty(),
                'modified_files': [item.a_path for item in self.git_repo.index.diff(None)],
                'untracked_files': self.git_repo.untracked_files
            }
        except Exception as e:
            self.logger.warning(f"Failed to get git info: {str(e)}")
            return {}
            
    def initialize_experiment(
        self, 
        config: ExperimentConfig,
        experiment_id: Optional[str] = None,
        resume_from: Optional[str] = None
    ) -> ExperimentMetadata:
        """
        Initialize a new experiment or resume an existing one.
        
        Args:
            config: Experiment configuration
            experiment_id: Unique identifier for the experiment (generated if None)
            resume_from: ID of experiment to resume from
            
        Returns:
            Experiment metadata
        """
        # Generate experiment ID if not provided
        if not experiment_id:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_id = f"{config.name}_{timestamp}"
            
        # Create experiment metadata
        exp_metadata = ExperimentMetadata(
            id=experiment_id,
            name=config.name,
            tags=config.tags.copy(),
            status=ExperimentStatus.INITIALIZING,
            start_time=datetime.datetime.now(),
            config_hash=self.config_manager.get_config_hash(config),
        )
        
        # Set up experiment directories
        dirs = self._setup_experiment_directories(experiment_id, config)
        exp_metadata.output_dir = dirs['root']
        exp_metadata.checkpoint_dir = dirs['checkpoints']
        
        # Set up logging
        exp_metadata.log_file = self._setup_experiment_logging(experiment_id, dirs['logs'])
        
        # Save configuration
        config_path = self.config_manager.save_config_snapshot(
            config,
            dirs['configs'],
            ConfigFormat.JSON
        )
        exp_metadata.config_path = config_path
        
        # Get git information
        if self.enable_git_tracking:
            git_info = self._get_git_info()
            if git_info:
                exp_metadata.git_commit = git_info['commit']
                # Add git info to metadata notes
                git_info_str = '\n'.join([f"{k}: {v}" for k, v in git_info.items()])
                exp_metadata.notes += f"\nGit Information:\n{git_info_str}"
                
        # Save metadata
        self._save_metadata(exp_metadata, dirs['root'])
        
        # If resuming from another experiment, copy checkpoints
        if resume_from:
            self._copy_checkpoint_files(resume_from, experiment_id)
            exp_metadata.notes += f"\nResumed from experiment: {resume_from}"
            self._save_metadata(exp_metadata, dirs['root'])
            
        self.current_experiment = exp_metadata
        self.logger.info(f"Initialized experiment {experiment_id}")
        
        return exp_metadata
    
    def _save_metadata(self, metadata: ExperimentMetadata, output_dir: str) -> None:
        """
        Save experiment metadata.
        
        Args:
            metadata: Experiment metadata
            output_dir: Directory to save metadata
        """
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
            
    def _load_metadata(self, experiment_id: str) -> ExperimentMetadata:
        """
        Load experiment metadata.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            Experiment metadata
        """
        exp_dir = os.path.join(self.base_dir, experiment_id)
        metadata_path = os.path.join(exp_dir, "metadata.json")
        
        try:
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
            return ExperimentMetadata.from_dict(metadata_dict)
        except Exception as e:
            self.logger.error(f"Failed to load metadata for experiment {experiment_id}: {str(e)}")
            raise
            
    def _copy_checkpoint_files(self, source_id: str, target_id: str) -> None:
        """
        Copy checkpoint files from one experiment to another.
        
        Args:
            source_id: ID of source experiment
            target_id: ID of target experiment
        """
        source_dir = os.path.join(self.base_dir, source_id, 'checkpoints')
        target_dir = os.path.join(self.base_dir, target_id, 'checkpoints')
        
        if not os.path.exists(source_dir):
            self.logger.warning(f"Source checkpoint directory does not exist: {source_dir}")
            return
            
        try:
            # Get all checkpoint files
            checkpoint_files = [f for f in os.listdir(source_dir) if f.endswith('.pkl')]
            
            if not checkpoint_files:
                self.logger.warning(f"No checkpoint files found in {source_dir}")
                return
                
            # Copy the latest checkpoint
            latest_checkpoint = sorted(checkpoint_files)[-1]
            shutil.copy2(
                os.path.join(source_dir, latest_checkpoint),
                os.path.join(target_dir, latest_checkpoint)
            )
            self.logger.info(f"Copied checkpoint {latest_checkpoint} from {source_id} to {target_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to copy checkpoints: {str(e)}")
    
    def save_checkpoint(
        self, 
        state: Dict[str, Any],
        checkpoint_name: Optional[str] = None
    ) -> str:
        """
        Save experiment state checkpoint.
        
        Args:
            state: Experiment state to save
            checkpoint_name: Custom name for checkpoint
            
        Returns:
            Path to the saved checkpoint
        """
        if not self.current_experiment:
            raise ValueError("No active experiment. Initialize experiment first.")
            
        checkpoint_dir = self.current_experiment.checkpoint_dir
        
        if not checkpoint_name:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"checkpoint_{timestamp}.pkl"
        elif not checkpoint_name.endswith('.pkl'):
            checkpoint_name += '.pkl'
            
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(state, f)
            self.logger.info(f"Saved checkpoint to {checkpoint_path}")
            return checkpoint_path
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {str(e)}")
            raise
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load experiment state from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Loaded experiment state
        """
        if not self.current_experiment:
            raise ValueError("No active experiment. Initialize experiment first.")
            
        checkpoint_dir = self.current_experiment.checkpoint_dir
        
        # If no specific checkpoint is provided, find the latest one
        if not checkpoint_path:
            checkpoint_files = [
                f for f in os.listdir(checkpoint_dir) if f.endswith('.pkl')
            ]
            
            if not checkpoint_files:
                self.logger.warning("No checkpoints found.")
                return {}
                
            checkpoint_path = os.path.join(
                checkpoint_dir, sorted(checkpoint_files)[-1]
            )
            
        try:
            with open(checkpoint_path, 'rb') as f:
                state = pickle.load(f)
            self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
            return state
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {str(e)}")
            raise
    
    def start_experiment(self) -> None:
        """Mark the current experiment as running."""
        if not self.current_experiment:
            raise ValueError("No active experiment. Initialize experiment first.")
            
        self.current_experiment.status = ExperimentStatus.RUNNING
        self._save_metadata(self.current_experiment, self.current_experiment.output_dir)
        self.logger.info(f"Started experiment {self.current_experiment.id}")
    
    def complete_experiment(self, success: bool = True) -> None:
        """
        Mark the current experiment as completed or failed.
        
        Args:
            success: Whether the experiment completed successfully
        """
        if not self.current_experiment:
            raise ValueError("No active experiment. Initialize experiment first.")
            
        self.current_experiment.end_time = datetime.datetime.now()
        
        if success:
            self.current_experiment.status = ExperimentStatus.COMPLETED
            self.logger.info(f"Completed experiment {self.current_experiment.id}")
        else:
            self.current_experiment.status = ExperimentStatus.FAILED
            self.logger.warning(f"Experiment {self.current_experiment.id} failed")
            
        self._save_metadata(self.current_experiment, self.current_experiment.output_dir)
        
    def interrupt_experiment(self) -> None:
        """Mark the current experiment as interrupted."""
        if not self.current_experiment:
            raise ValueError("No active experiment. Initialize experiment first.")
            
        self.current_experiment.end_time = datetime.datetime.now()
        self.current_experiment.status = ExperimentStatus.INTERRUPTED
        self._save_metadata(self.current_experiment, self.current_experiment.output_dir)
        self.logger.warning(f"Interrupted experiment {self.current_experiment.id}")
    
    def add_experiment_tags(self, tags: List[str]) -> None:
        """
        Add tags to the current experiment.
        
        Args:
            tags: Tags to add
        """
        if not self.current_experiment:
            raise ValueError("No active experiment. Initialize experiment first.")
            
        for tag in tags:
            if tag not in self.current_experiment.tags:
                self.current_experiment.tags.append(tag)
                
        self._save_metadata(self.current_experiment, self.current_experiment.output_dir)
        self.logger.debug(f"Added tags to experiment {self.current_experiment.id}: {tags}")
    
    def add_experiment_note(self, note: str) -> None:
        """
        Add a note to the current experiment.
        
        Args:
            note: Note to add
        """
        if not self.current_experiment:
            raise ValueError("No active experiment. Initialize experiment first.")
            
        self.current_experiment.notes += f"\n{note}"
        self._save_metadata(self.current_experiment, self.current_experiment.output_dir)
        self.logger.debug(f"Added note to experiment {self.current_experiment.id}")

    def list_experiments(self, filter_tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        List all experiments, optionally filtered by tags.
        
        Args:
            filter_tags: Filter experiments by these tags
            
        Returns:
            List of experiment metadata dictionaries
        """
        experiments = []
        
        for item in os.listdir(self.base_dir):
            item_path = os.path.join(self.base_dir, item)
            if not os.path.isdir(item_path):
                continue
                
            metadata_path = os.path.join(item_path, "metadata.json")
            if not os.path.exists(metadata_path):
                continue
                
            try:
                with open(metadata_path, 'r') as f:
                    metadata_dict = json.load(f)
                    
                if filter_tags:
                    # Only include if all filter tags are present
                    metadata_tags = set(metadata_dict.get('tags', []))
                    if not set(filter_tags).issubset(metadata_tags):
                        continue
                        
                experiments.append(metadata_dict)
            except Exception as e:
                self.logger.warning(f"Failed to load metadata from {metadata_path}: {str(e)}")
                
        return sorted(experiments, key=lambda x: x.get('start_time', ''), reverse=True)


# ================================
# Resource Management Classes
# ================================

class HardwareInfo:
    """
    Collects and monitors hardware resource information.
    
    This class provides utilities for querying and monitoring CPU, GPU, and memory usage.
    """
    
    @staticmethod
    def get_cpu_info() -> Dict[str, Any]:
        """
        Get CPU information.
        
        Returns:
            Dictionary with CPU information
        """
        cpu_info = {
            'count_logical': psutil.cpu_count(logical=True),
            'count_physical': psutil.cpu_count(logical=False),
            'percent_used': psutil.cpu_percent(interval=0.1),
            'load_avg': psutil.getloadavg(),
        }
        
        # Get per-core usage
        per_cpu = psutil.cpu_percent(interval=0.1, percpu=True)
        for i, usage in enumerate(per_cpu):
            cpu_info[f'core_{i}_percent'] = usage
            
        return cpu_info
    
    @staticmethod
    def get_memory_info() -> Dict[str, Any]:
        """
        Get memory information.
        
        Returns:
            Dictionary with memory information
        """
        memory = psutil.virtual_memory()
        return {
            'total': memory.total,
            'available': memory.available,
            'used': memory.used,
            'percent_used': memory.percent,
            'free': memory.free,
        }
    
    @staticmethod
    def get_disk_info(path: str = '/') -> Dict[str, Any]:
        """
        Get disk information.
        
        Args:
            path: Path to get disk usage for
            
        Returns:
            Dictionary with disk information
        """
        disk = psutil.disk_usage(path)
        return {
            'total': disk.total,
            'used': disk.used,
            'free': disk.free,
            'percent_used': disk.percent,
        }
    
    @staticmethod
    def get_gpu_info() -> List[Dict[str, Any]]:
        """
        Get GPU information using CuPy.
        
        Returns:
            List of dictionaries with GPU information
        """
        try:
            num_gpus = cp.cuda.runtime.getDeviceCount()
            gpu_info = []
            
            for i in range(num_gpus):
                with cp.cuda.Device(i):
                    props = cp.cuda.runtime.getDeviceProperties(i)
                    free, total = cp.cuda.runtime.memGetInfo()
                    
                    gpu_info.append({
                        'id': i,
                        'name': props['name'].decode('utf-8'),
                        'total_memory': total,
                        'free_memory': free,
                        'used_memory': total - free,
                        'memory_used_percent': (total - free) / total * 100,
                        'compute_capability': f"{props['major']}.{props['minor']}",
                        'clock_rate': props['clockRate'],
                        'multi_processor_count': props['multiProcessorCount'],
                    })
                    
            return gpu_info
        except Exception as e:
            logger.warning(f"Failed to get GPU information: {str(e)}")
            return []
    
    @classmethod
    def get_all_info(cls) -> Dict[str, Any]:
        """
        Get all hardware information.
        
        Returns:
            Dictionary with all hardware information
        """
        return {
            'cpu': cls.get_cpu_info(),
            'memory': cls.get_memory_info(),
            'disk': cls.get_disk_info(),
            'gpu': cls.get_gpu_info(),
            'timestamp': datetime.datetime.now().isoformat()
        }


class ResourceMonitor:
    """
    Monitors system resources during experiment execution.
    
    This class provides real-time monitoring of CPU, GPU, and memory usage,
    with support for periodic logging and resource constraint enforcement.
    """
    
    def __init__(
        self, 
        log_interval_seconds: float = 5.0,
        enforce_limits: bool = True,
        memory_limit_percent: float = 90.0,
        gpu_memory_limit_percent: float = 90.0
    ):
        """
        Initialize resource monitor.
        
        Args:
            log_interval_seconds: Interval between resource logs
            enforce_limits: Whether to enforce resource limits
            memory_limit_percent: Memory usage limit as percentage
            gpu_memory_limit_percent: GPU memory usage limit as percentage
        """
        self.log_interval = log_interval_seconds
        self.enforce_limits = enforce_limits
        self.memory_limit = memory_limit_percent
        self.gpu_memory_limit = gpu_memory_limit_percent
        self.logger = logger
        
        self.monitoring = False
        self.monitor_thread = None
        self.resource_log = []
        self.start_time = None
    
    def start_monitoring(self) -> None:
        """Start resource monitoring in a background thread."""
        if self.monitoring:
            self.logger.warning("Resource monitoring is already active.")
            return
            
        self.monitoring = True
        self.start_time = time.time()
        self.resource_log = []
        
        def _monitor_loop():
            last_log_time = time.time()
            
            while self.monitoring:
                current_time = time.time()
                
                # Get resource info
                resource_info = HardwareInfo.get_all_info()
                
                # Log if interval has passed
                if current_time - last_log_time >= self.log_interval:
                    self.resource_log.append(resource_info)
                    last_log_time = current_time
                    
                    # Log summary
                    cpu_percent = resource_info['cpu']['percent_used']
                    memory_percent = resource_info['memory']['percent_used']
                    
                    gpu_summary = ""
                    if resource_info['gpu']:
                        gpu_percents = [g['memory_used_percent'] for g in resource_info['gpu']]
                        gpu_summary = f", GPU Memory: {gpu_percents}%"
                        
                    elapsed = current_time - self.start_time
                    self.logger.debug(
                        f"Resource usage [{elapsed:.1f}s] - CPU: {cpu_percent}%, "
                        f"Memory: {memory_percent}%{gpu_summary}"
                    )
                    
                # Check resource limits if enabled
                if self.enforce_limits:
                    self._check_resource_limits(resource_info)
                    
                # Sleep briefly to avoid high CPU usage from the monitoring itself
                time.sleep(0.2)
                
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=_monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Resource monitoring started")
    
    def _check_resource_limits(self, resource_info: Dict[str, Any]) -> None:
        """
        Check if resource usage exceeds limits.
        
        Args:
            resource_info: Current resource information
        """
        # Check memory usage
        memory_percent = resource_info['memory']['percent_used']
        if memory_percent > self.memory_limit:
            self.logger.warning(
                f"Memory usage ({memory_percent}%) exceeds limit ({self.memory_limit}%). "
                "Consider freeing memory."
            )
            
        # Check GPU memory usage
        for gpu in resource_info['gpu']:
            gpu_percent = gpu['memory_used_percent']
            if gpu_percent > self.gpu_memory_limit:
                self.logger.warning(
                    f"GPU {gpu['id']} memory usage ({gpu_percent}%) exceeds limit "
                    f"({self.gpu_memory_limit}%). Consider freeing GPU memory."
                )
    
    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        if not self.monitoring:
            return
            
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
            
        elapsed = time.time() - self.start_time if self.start_time else 0
        self.logger.info(f"Resource monitoring stopped after {elapsed:.1f}s")
    
    def save_resource_log(self, output_path: str) -> None:
        """
        Save resource monitoring log to file.
        
        Args:
            output_path: Path to output file
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(self.resource_log, f, indent=2)
            self.logger.info(f"Resource log saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save resource log: {str(e)}")
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """
        Get summary of resource usage.
        
        Returns:
            Dictionary with resource usage statistics
        """
        if not self.resource_log:
            return {}
            
        # Helper function to extract a time series
        def extract_series(key_path):
            result = []
            for entry in self.resource_log:
                value = entry
                for key in key_path:
                    if key in value:
                        value = value[key]
                    else:
                        value = None
                        break
                if value is not None:
                    result.append(value)
            return result
        
        # Extract time series
        cpu_usage = extract_series(['cpu', 'percent_used'])
        memory_usage = extract_series(['memory', 'percent_used'])
        
        # GPU usage by device
        gpu_usage = {}
        if self.resource_log and 'gpu' in self.resource_log[0]:
            for i, gpu in enumerate(self.resource_log[0]['gpu']):
                gpu_id = gpu['id']
                gpu_usage[f'gpu_{gpu_id}'] = [
                    log['gpu'][i]['memory_used_percent'] 
                    for log in self.resource_log 
                    if 'gpu' in log and i < len(log['gpu'])
                ]
        
        # Calculate statistics
        def calculate_stats(series):
            if not series:
                return {}
            return {
                'min': min(series),
                'max': max(series),
                'avg': sum(series) / len(series),
                'median': sorted(series)[len(series) // 2],
                'p95': sorted(series)[int(len(series) * 0.95)],
                'p99': sorted(series)[int(len(series) * 0.99)],
            }
        
        return {
            'duration_seconds': time.time() - self.start_time if self.start_time else 0,
            'n_samples': len(self.resource_log),
            'cpu_stats': calculate_stats(cpu_usage),
            'memory_stats': calculate_stats(memory_usage),
            'gpu_stats': {k: calculate_stats(v) for k, v in gpu_usage.items()},
        }


class GPUManager:
    """
    Manages GPU resources and memory allocation.
    
    This class provides utilities for GPU device selection, memory optimization,
    and batch processing to avoid out-of-memory errors.
    """
    
    def __init__(
        self, 
        memory_fraction: float = 0.9,
        prefer_fastest: bool = True
    ):
        """
        Initialize GPU manager.
        
        Args:
            memory_fraction: Fraction of GPU memory to allocate
            prefer_fastest: Whether to prefer the fastest GPU
        """
        self.memory_fraction = memory_fraction
        self.prefer_fastest = prefer_fastest
        self.logger = logger
        self.selected_device = None
        
        # Check if CuPy is available
        self.has_gpu = self._check_gpu_availability()
        
    def _check_gpu_availability(self) -> bool:
        """
        Check if GPUs are available through CuPy.
        
        Returns:
            Whether GPUs are available
        """
        try:
            num_gpus = cp.cuda.runtime.getDeviceCount()
            return num_gpus > 0
        except Exception as e:
            self.logger.warning(f"GPU check failed: {str(e)}")
            return False
            
    def select_best_gpu(self) -> int:
        """
        Select the best available GPU.
        
        Returns:
            ID of the selected GPU
        """
        if not self.has_gpu:
            self.logger.warning("No GPUs available")
            return -1
            
        try:
            num_gpus = cp.cuda.runtime.getDeviceCount()
            if num_gpus == 0:
                self.logger.warning("No GPUs available")
                return -1
                
            # Get info for all GPUs
            gpu_info = []
            for i in range(num_gpus):
                with cp.cuda.Device(i):
                    props = cp.cuda.runtime.getDeviceProperties(i)
                    free, total = cp.cuda.runtime.memGetInfo()
                    
                    gpu_info.append({
                        'id': i,
                        'name': props['name'].decode('utf-8'),
                        'total_memory': total,
                        'free_memory': free,
                        'compute_capability': f"{props['major']}.{props['minor']}",
                        'clock_rate': props['clockRate'],
                        'multi_processor_count': props['multiProcessorCount'],
                    })
            
            # Select GPU based on criteria
            if self.prefer_fastest:
                # Sort by compute capability, clock rate, and free memory
                gpu_info.sort(key=lambda x: (
                    float(x['compute_capability']),
                    x['clock_rate'],
                    x['free_memory']
                ), reverse=True)
            else:
                # Sort by free memory
                gpu_info.sort(key=lambda x: x['free_memory'], reverse=True)
                
            # Select the best GPU
            selected_gpu = gpu_info[0]['id']
            self.selected_device = selected_gpu
            
            self.logger.info(
                f"Selected GPU {selected_gpu}: {gpu_info[0]['name']}, "
                f"{gpu_info[0]['free_memory']/1e9:.2f} GB free"
            )
            
            return selected_gpu
        
        except Exception as e:
            self.logger.error(f"Failed to select GPU: {str(e)}")
            return -1
            
    def initialize_gpu_context(self) -> bool:
        """
        Initialize GPU context for the selected GPU.
        
        Returns:
            Whether initialization was successful
        """
        if not self.has_gpu:
            self.logger.warning("No GPUs available")
            return False
            
        if self.selected_device is None:
            self.select_best_gpu()
            
        if self.selected_device < 0:
            self.logger.warning("No suitable GPU found")
            return False
            
        try:
            # Set device
            cp.cuda.Device(self.selected_device).use()
            
            # Set memory limit
            with cp.cuda.Device(self.selected_device):
                _, total = cp.cuda.runtime.memGetInfo()
                pool = cp.get_default_memory_pool()
                pool.set_limit(size=int(total * self.memory_fraction))
                
            self.logger.info(
                f"Initialized GPU {self.selected_device} with "
                f"memory limit {self.memory_fraction*100:.1f}%"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize GPU context: {str(e)}")
            return False
            
    def cleanup_gpu_memory(self) -> None:
        """Free all allocated memory pools in the GPU."""
        if not self.has_gpu:
            return
            
        try:
            # Get memory pools
            mem_pool = cp.get_default_memory_pool()
            pinned_pool = cp.get_default_pinned_memory_pool()
            
            # Free all blocks
            mem_pool.free_all_blocks()
            pinned_pool.free_all_blocks()
            
            self.logger.debug("GPU memory pools cleared")
            
        except Exception as e:
            self.logger.warning(f"Failed to clean up GPU memory: {str(e)}")
            
    def get_optimal_batch_size(
        self, 
        sample_data_size_bytes: int,
        target_memory_usage: float = 0.7
    ) -> int:
        """
        Calculate optimal batch size based on available GPU memory.
        
        Args:
            sample_data_size_bytes: Size of a single data sample in bytes
            target_memory_usage: Target GPU memory usage fraction
            
        Returns:
            Optimal batch size
        """
        if not self.has_gpu or self.selected_device is None:
            return 128  # Default batch size for CPU
            
        try:
            with cp.cuda.Device(self.selected_device):
                free, total = cp.cuda.runtime.memGetInfo()
                
                # Reserve memory for other data
                available_memory = free * target_memory_usage
                
                # Calculate batch size
                batch_size = max(1, int(available_memory / sample_data_size_bytes))
                
                self.logger.debug(
                    f"Calculated optimal batch size: {batch_size} "
                    f"(sample size: {sample_data_size_bytes/1e6:.2f} MB)"
                )
                return batch_size
                
        except Exception as e:
            self.logger.warning(f"Failed to calculate batch size: {str(e)}")
            return 64  # Fallback to conservative batch size
            
    def process_in_batches(
        self, 
        data: np.ndarray,
        process_func: Callable[[cp.ndarray], cp.ndarray],
        batch_size: Optional[int] = None,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Process data in batches to optimize GPU memory usage.
        
        Args:
            data: Input data
            process_func: Function to process each batch
            batch_size: Batch size (calculated if None)
            show_progress: Whether to show progress bar
            
        Returns:
            Processed data
        """
        if not self.has_gpu:
            raise ValueError("GPU processing requested but no GPU available")
            
        # Calculate batch size if not provided
        if batch_size is None:
            # Estimate sample size
            sample_size = data.itemsize * np.prod(data.shape[1:]) if data.ndim > 1 else data.itemsize
            batch_size = self.get_optimal_batch_size(sample_size)
            
        # Prepare output array
        output_shape = list(data.shape)
        n_samples = output_shape[0]
        
        # Process in batches
        results = []
        iterator = range(0, n_samples, batch_size)
        
        if show_progress:
            iterator = tqdm(iterator, desc="Processing batches", unit="batch")
            
        for i in iterator:
            batch_end = min(i + batch_size, n_samples)
            batch = data[i:batch_end]
            
            # Transfer to GPU, process, and transfer back
            try:
                batch_gpu = cp.asarray(batch)
                result_gpu = process_func(batch_gpu)
                result = cp.asnumpy(result_gpu)
                results.append(result)
                
                # Explicitly delete GPU arrays to free memory
                del batch_gpu, result_gpu
                self.cleanup_gpu_memory()
                
            except Exception as e:
                self.logger.error(f"Batch processing failed at index {i}: {str(e)}")
                raise
                
        # Combine results
        return np.concatenate(results, axis=0)
    
    def memory_status(self) -> Dict[str, Any]:
        """
        Get current GPU memory status.
        
        Returns:
            Dictionary with memory status
        """
        if not self.has_gpu or self.selected_device is None:
            return {"available": False}
            
        try:
            with cp.cuda.Device(self.selected_device):
                free, total = cp.cuda.runtime.memGetInfo()
                
                # Get memory pool info
                pool = cp.get_default_memory_pool()
                used_bytes = pool.used_bytes()
                
                return {
                    "available": True,
                    "device_id": self.selected_device,
                    "total_bytes": total,
                    "free_bytes": free,
                    "used_bytes": total - free,
                    "pool_used_bytes": used_bytes,
                    "usage_percent": (total - free) / total * 100,
                }
        except Exception as e:
            self.logger.warning(f"Failed to get GPU memory status: {str(e)}")
            return {"available": False, "error": str(e)}


class ParallelExecutor:
    """
    Manages parallel execution of tasks across CPUs and GPUs.
    
    This class provides utilities for distributing work across available
    computational resources, with support for both CPU and GPU parallelism.
    """
    
    def __init__(
        self, 
        n_jobs: Optional[int] = None,
        gpu_manager: Optional[GPUManager] = None
    ):
        """
        Initialize parallel executor.
        
        Args:
            n_jobs: Number of CPU jobs (None for auto)
            gpu_manager: GPU manager instance
        """
        self.n_jobs = n_jobs if n_jobs is not None else max(1, multiprocessing.cpu_count() - 1)
        self.gpu_manager = gpu_manager
        self.logger = logger
        
    def parallel_cpu(
        self, 
        func: Callable,
        items: List[Any],
        *args, 
        **kwargs
    ) -> List[Any]:
        """
        Execute function in parallel using CPU.
        
        Args:
            func: Function to execute
            items: Items to process
            *args: Additional positional arguments for func
            **kwargs: Additional keyword arguments for func
            
        Returns:
            List of results
        """
        self.logger.debug(f"Running {len(items)} tasks in parallel with {self.n_jobs} CPU jobs")
        
        with multiprocessing.Pool(self.n_jobs) as pool:
            # Create partial function with fixed args and kwargs
            if args or kwargs:
                from functools import partial
                partial_func = partial(func, *args, **kwargs)
                results = list(tqdm(pool.imap(partial_func, items), total=len(items)))
            else:
                results = list(tqdm(pool.imap(func, items), total=len(items)))
                
        return results
    
    def parallel_gpu(
        self, 
        func: Callable,
        items: List[Any],
        batch_size: Optional[int] = None,
        *args, 
        **kwargs
    ) -> List[Any]:
        """
        Execute function in parallel using GPU.
        
        Args:
            func: Function to execute
            items: Items to process
            batch_size: Batch size (calculated if None)
            *args: Additional positional arguments for func
            **kwargs: Additional keyword arguments for func
            
        Returns:
            List of results
        """
        if not self.gpu_manager or not self.gpu_manager.has_gpu:
            self.logger.warning("GPU execution requested but no GPU available, using CPU instead")
            return self.parallel_cpu(func, items, *args, **kwargs)
            
        # Determine batch size
        if batch_size is None:
            # Crude estimate - in real code, this would need to be calibrated
            batch_size = max(1, min(1000, len(items) // 10))
            
        # Process in batches
        results = []
        for i in tqdm(range(0, len(items), batch_size), desc="GPU batch processing"):
            batch_items = items[i:i+batch_size]
            
            try:
                # Process batch
                batch_results = [func(item, *args, **kwargs) for item in batch_items]
                results.extend(batch_results)
                
                # Clean up GPU memory
                self.gpu_manager.cleanup_gpu_memory()
                
            except Exception as e:
                self.logger.error(f"GPU batch processing failed: {str(e)}")
                self.gpu_manager.cleanup_gpu_memory()
                raise
                
        return results


# ================================
# Data Pipeline Classes
# ================================

class DataValidator:
    """
    Validates data quality and integrity.
    
    This class provides utilities for checking data quality issues like missing values,
    outliers, and inconsistencies before running experiments.
    """
    
    @staticmethod
    def check_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check for missing values in a dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary with missing value statistics
        """
        missing_counts = df.isna().sum()
        missing_percent = df.isna().mean() * 100
        
        return {
            'total_missing': int(missing_counts.sum()),
            'percent_missing': float(missing_percent.mean()),
            'missing_by_column': {
                col: {
                    'count': int(missing_counts[col]),
                    'percent': float(missing_percent[col])
                }
                for col in df.columns
            }
        }
    
    @staticmethod
    def check_outliers(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = 'zscore',
        threshold: float = 3.0
    ) -> Dict[str, Any]:
        """
        Check for outliers in numeric columns.
        
        Args:
            df: Input dataframe
            columns: Columns to check (all numeric if None)
            method: Outlier detection method ('zscore' or 'iqr')
            threshold: Threshold for outlier detection
            
        Returns:
            Dictionary with outlier statistics
        """
        if columns is None:
            columns = df.select_dtypes(include=np.number).columns.tolist()
            
        outliers_info = {}
        
        for col in columns:
            if col not in df.columns or not np.issubdtype(df[col].dtype, np.number):
                continue
                
            # Get non-missing values
            values = df[col].dropna().values
            
            if method == 'zscore':
                # Z-score method
                z_scores = (values - np.mean(values)) / np.std(values)
                outliers = np.abs(z_scores) > threshold
            elif method == 'iqr':
                # IQR method
                q1, q3 = np.percentile(values, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                outliers = (values < lower_bound) | (values > upper_bound)
            else:
                raise ValueError(f"Unsupported outlier detection method: {method}")
                
            outliers_info[col] = {
                'total_outliers': int(outliers.sum()),
                'percent_outliers': float(outliers.mean() * 100),
                'min_value': float(values.min()),
                'max_value': float(values.max()),
                'mean': float(values.mean()),
                'std': float(values.std()),
            }
            
        return outliers_info
    
    @staticmethod
    def check_consistency(
        df: pd.DataFrame,
        rules: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Check data consistency based on predefined rules.
        
        Args:
            df: Input dataframe
            rules: List of consistency rules
            
        Returns:
            Dictionary with consistency check results
        """
        results = {}
        
        for i, rule in enumerate(rules):
            rule_type = rule.get('type')
            rule_id = rule.get('id', f"rule_{i}")
            
            if rule_type == 'value_range':
                # Value range check
                column = rule.get('column')
                min_value = rule.get('min')
                max_value = rule.get('max')
                
                if column not in df.columns:
                    results[rule_id] = {
                    'status': 'ok' if duplicates == 0 else 'violation',
                    'violations': int(duplicates),
                    'percent_violations': float(duplicates / len(df) * 100),
                }
                
            elif rule_type == 'relationship':
                # Relationship check
                column1 = rule.get('column1')
                column2 = rule.get('column2')
                operator = rule.get('operator', '>')
                
                if column1 not in df.columns or column2 not in df.columns:
                    results[rule_id] = {
                        'status': 'error',
                        'message': f"Column not found: {column1 if column1 not in df.columns else column2}"
                    }
                    continue
                    
                if operator == '>':
                    violations = (df[column1] <= df[column2]).sum()
                elif operator == '>=':
                    violations = (df[column1] < df[column2]).sum()
                elif operator == '<':
                    violations = (df[column1] >= df[column2]).sum()
                elif operator == '<=':
                    violations = (df[column1] > df[column2]).sum()
                elif operator == '==':
                    violations = (df[column1] != df[column2]).sum()
                elif operator == '!=':
                    violations = (df[column1] == df[column2]).sum()
                else:
                    results[rule_id] = {'status': 'error', 'message': f"Unsupported operator: {operator}"}
                    continue
                    
                results[rule_id] = {
                    'status': 'ok' if violations == 0 else 'violation',
                    'violations': int(violations),
                    'percent_violations': float(violations / len(df) * 100),
                }
                
            else:
                results[rule_id] = {'status': 'error', 'message': f"Unsupported rule type: {rule_type}"}
                
        return results
    
    @staticmethod
    def check_date_continuity(
        df: pd.DataFrame,
        date_column: str,
        freq: str = 'D',
        allow_gaps: bool = False
    ) -> Dict[str, Any]:
        """
        Check for continuity in date/time series.
        
        Args:
            df: Input dataframe
            date_column: Name of date column
            freq: Expected frequency ('D' for daily, etc.)
            allow_gaps: Whether to allow gaps in the series
            
        Returns:
            Dictionary with date continuity check results
        """
        if date_column not in df.columns:
            return {'status': 'error', 'message': f"Column not found: {date_column}"}
            
        # Ensure date column is datetime
        try:
            dates = pd.to_datetime(df[date_column])
        except Exception as e:
            return {'status': 'error', 'message': f"Failed to convert to datetime: {str(e)}"}
            
        # Sort dates
        dates = dates.sort_values().reset_index(drop=True)
        
        # Check if sorted
        if not (dates == df[date_column].sort_values().reset_index(drop=True)).all():
            return {'status': 'warning', 'message': "Date column is not sorted in the dataframe"}
            
        # Create expected date range
        try:
            expected_dates = pd.date_range(start=dates.min(), end=dates.max(), freq=freq)
        except Exception as e:
            return {'status': 'error', 'message': f"Failed to create expected date range: {str(e)}"}
            
        # Check for missing dates
        missing_dates = set(expected_dates) - set(dates)
        
        # Check for duplicate dates
        duplicate_dates = dates[dates.duplicated()].unique()
        
        # Compute gaps
        if not allow_gaps and len(missing_dates) > 0:
            status = 'violation'
        elif len(duplicate_dates) > 0:
            status = 'violation'
        else:
            status = 'ok'
            
        return {
            'status': status,
            'missing_dates': [d.strftime('%Y-%m-%d') for d in sorted(missing_dates)],
            'duplicate_dates': [d.strftime('%Y-%m-%d') for d in sorted(duplicate_dates)],
            'total_missing': len(missing_dates),
            'total_duplicates': len(duplicate_dates),
            'percent_missing': len(missing_dates) / len(expected_dates) * 100,
        }
    
    @classmethod
    def validate_dataframe(
        cls, 
        df: pd.DataFrame,
        validation_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform comprehensive validation of a dataframe.
        
        Args:
            df: Input dataframe
            validation_config: Configuration for validation checks
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'data_shape': {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': df.columns.tolist(),
                'dtypes': {col: str(df[col].dtype) for col in df.columns}
            }
        }
        
        # Check missing values
        if validation_config.get('check_missing', True):
            results['missing_values'] = cls.check_missing_values(df)
            
        # Check outliers
        if validation_config.get('check_outliers', True):
            outlier_columns = validation_config.get('outlier_columns')
            outlier_method = validation_config.get('outlier_method', 'zscore')
            outlier_threshold = validation_config.get('outlier_threshold', 3.0)
            
            results['outliers'] = cls.check_outliers(
                df,
                columns=outlier_columns,
                method=outlier_method,
                threshold=outlier_threshold
            )
            
        # Check consistency rules
        if 'consistency_rules' in validation_config:
            results['consistency'] = cls.check_consistency(
                df,
                validation_config['consistency_rules']
            )
            
        # Check date continuity
        if 'date_column' in validation_config:
            results['date_continuity'] = cls.check_date_continuity(
                df,
                validation_config['date_column'],
                freq=validation_config.get('date_freq', 'D'),
                allow_gaps=validation_config.get('allow_date_gaps', False)
            )
            
        # Overall validation status
        status = 'ok'
        issues = []
        
        # Check missing values
        if 'missing_values' in results and results['missing_values']['percent_missing'] > validation_config.get('max_missing_percent', 5):
            status = 'warning'
            issues.append(f"High percentage of missing values: {results['missing_values']['percent_missing']:.2f}%")
            
        # Check outliers
        if 'outliers' in results:
            for col, info in results['outliers'].items():
                if info['percent_outliers'] > validation_config.get('max_outlier_percent', 5):
                    status = 'warning'
                    issues.append(f"High percentage of outliers in {col}: {info['percent_outliers']:.2f}%")
                    
        # Check consistency
        if 'consistency' in results:
            for rule_id, result in results['consistency'].items():
                if result.get('status') == 'violation':
                    status = 'warning'
                    issues.append(f"Consistency rule violation in {rule_id}: {result.get('percent_violations', 0):.2f}%")
                    
        # Check date continuity
        if 'date_continuity' in results and results['date_continuity']['status'] == 'violation':
            status = 'warning'
            issues.append(f"Date continuity issues: {results['date_continuity']['total_missing']} missing dates")
            
        results['validation_summary'] = {
            'status': status,
            'issues': issues,
            'issue_count': len(issues)
        }
        
        return results


class DatasetVersionManager:
    """
    Manages dataset versions and tracking.
    
    This class provides utilities for tracking dataset versions, generating
    dataset hashes, and maintaining a registry of available datasets.
    """
    
    def __init__(self, data_registry_path: str):
        """
        Initialize dataset version manager.
        
        Args:
            data_registry_path: Path to dataset registry file
        """
        self.registry_path = data_registry_path
        self.logger = logger
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(data_registry_path), exist_ok=True)
        
        # Initialize registry if it doesn't exist
        if not os.path.exists(data_registry_path):
            self._initialize_registry()
            
    def _initialize_registry(self) -> None:
        """Initialize empty dataset registry."""
        registry = {
            'datasets': {},
            'last_updated': datetime.datetime.now().isoformat()
        }
        
        with open(self.registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
            
    def _load_registry(self) -> Dict[str, Any]:
        """
        Load dataset registry.
        
        Returns:
            Dataset registry dictionary
        """
        try:
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load dataset registry: {str(e)}")
            self._initialize_registry()
            return {'datasets': {}, 'last_updated': datetime.datetime.now().isoformat()}
            
    def _save_registry(self, registry: Dict[str, Any]) -> None:
        """
        Save dataset registry.
        
        Args:
            registry: Dataset registry dictionary
        """
        registry['last_updated'] = datetime.datetime.now().isoformat()
        
        try:
            with open(self.registry_path, 'w') as f:
                json.dump(registry, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save dataset registry: {str(e)}")
            
    def calculate_dataset_hash(self, df: pd.DataFrame) -> str:
        """
        Calculate hash for a dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            SHA-256 hash of the dataframe
        """
        # Convert relevant properties to string for hashing
        hash_components = [
            str(df.shape),
            str(df.columns.tolist()),
            str([str(df[col].dtype) for col in df.columns]),
            # Add sample of data values (first and last 5 rows)
            str(df.head().values.tolist() if len(df) >= 5 else df.values.tolist()),
            str(df.tail().values.tolist() if len(df) >= 5 else []),
            # Add some statistics
            str([df[col].mean() if np.issubdtype(df[col].dtype, np.number) else None for col in df.columns]),
            str([df[col].sum() if np.issubdtype(df[col].dtype, np.number) else None for col in df.columns]),
        ]
        
        # Join components and compute hash
        hash_str = '|'.join(hash_components)
        return hashlib.sha256(hash_str.encode()).hexdigest()
        
    def register_dataset(
        self,
        dataset_id: str,
        df: pd.DataFrame,
        source_info: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Register a dataset in the registry.
        
        Args:
            dataset_id: Identifier for the dataset
            df: Dataset dataframe
            source_info: Information about the data source
            metadata: Additional metadata
            
        Returns:
            Dataset version hash
        """
        # Load registry
        registry = self._load_registry()
        
        # Calculate dataset hash
        dataset_hash = self.calculate_dataset_hash(df)
        
        # Create dataset entry if it doesn't exist
        if dataset_id not in registry['datasets']:
            registry['datasets'][dataset_id] = {
                'id': dataset_id,
                'versions': [],
                'current_version': None
            }
            
        # Add version
        timestamp = datetime.datetime.now().isoformat()
        version = {
            'hash': dataset_hash,
            'timestamp': timestamp,
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': df.columns.tolist(),
            'source_info': source_info,
            'metadata': metadata or {}
        }
        
        # Check if version already exists
        existing_versions = [v for v in registry['datasets'][dataset_id]['versions'] if v['hash'] == dataset_hash]
        
        if existing_versions:
            # Update existing version
            existing_versions[0].update(version)
            self.logger.info(f"Updated existing dataset version: {dataset_id} ({dataset_hash})")
        else:
            # Add new version
            registry['datasets'][dataset_id]['versions'].append(version)
            self.logger.info(f"Registered new dataset version: {dataset_id} ({dataset_hash})")
            
        # Set as current version
        registry['datasets'][dataset_id]['current_version'] = dataset_hash
        
        # Save registry
        self._save_registry(registry)
        
        return dataset_hash
        
    def get_dataset_info(self, dataset_id: str, version_hash: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a dataset.
        
        Args:
            dataset_id: Identifier for the dataset
            version_hash: Version hash (current version if None)
            
        Returns:
            Dataset information
        """
        # Load registry
        registry = self._load_registry()
        
        if dataset_id not in registry['datasets']:
            return {}
            
        dataset_info = registry['datasets'][dataset_id]
        
        # Get version
        if version_hash is None:
            version_hash = dataset_info['current_version']
            
        version_info = next(
            (v for v in dataset_info['versions'] if v['hash'] == version_hash),
            None
        )
        
        if not version_info:
            return {}
            
        return {
            'dataset_id': dataset_id,
            'version_hash': version_hash,
            'version_info': version_info,
            'is_current': version_hash == dataset_info['current_version'],
            'version_count': len(dataset_info['versions'])
        }
        
    def list_datasets(self) -> List[Dict[str, Any]]:
        """
        List all datasets in the registry.
        
        Returns:
            List of dataset information dictionaries
        """
        # Load registry
        registry = self._load_registry()
        
        datasets = []
        for dataset_id, dataset_info in registry['datasets'].items():
            # Find current version
            current_version = next(
                (v for v in dataset_info['versions'] if v['hash'] == dataset_info['current_version']),
                None
            )
            
            if current_version:
                datasets.append({
                    'id': dataset_id,
                    'current_version': dataset_info['current_version'],
                    'version_count': len(dataset_info['versions']),
                    'rows': current_version.get('rows'),
                    'columns': current_version.get('columns'),
                    'last_updated': current_version.get('timestamp')
                })
                
        return sorted(datasets, key=lambda x: x.get('last_updated', ''), reverse=True)


class FeatureComputer:
    """
    Computes features from raw data.
    
    This class provides utilities for feature computation, with support for
    caching and GPU acceleration where applicable.
    """
    
    def __init__(
        self, 
        cache_dir: str,
        gpu_manager: Optional[GPUManager] = None
    ):
        """
        Initialize feature computer.
        
        Args:
            cache_dir: Directory for feature cache
            gpu_manager: GPU manager instance
        """
        self.cache_dir = cache_dir
        self.gpu_manager = gpu_manager
        self.logger = logger
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
    def _get_cache_path(
        self,
        dataset_id: str,
        feature_name: str,
        dataset_hash: str,
        feature_params_hash: str
    ) -> str:
        """
        Get path for cached feature.
        
        Args:
            dataset_id: Identifier for the dataset
            feature_name: Name of the feature
            dataset_hash: Hash of the dataset
            feature_params_hash: Hash of feature parameters
            
        Returns:
            Path to the cached feature
        """
        filename = f"{dataset_id}_{feature_name}_{dataset_hash[:8]}_{feature_params_hash[:8]}.pkl"
        return os.path.join(self.cache_dir, filename)
        
    def _compute_feature_params_hash(self, params: Dict[str, Any]) -> str:
        """
        Compute hash for feature parameters.
        
        Args:
            params: Feature parameters
            
        Returns:
            SHA-256 hash of the parameters
        """
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.sha256(param_str.encode()).hexdigest()
        
    def compute_feature(
        self,
        df: pd.DataFrame,
        dataset_id: str,
        dataset_hash: str,
        feature_name: str,
        feature_func: Callable,
        feature_params: Dict[str, Any],
        use_cache: bool = True,
        force_recompute: bool = False
    ) -> pd.DataFrame:
        """
        Compute a feature, with caching.
        
        Args:
            df: Input dataframe
            dataset_id: Identifier for the dataset
            dataset_hash: Hash of the dataset
            feature_name: Name of the feature
            feature_func: Function to compute the feature
            feature_params: Parameters for the feature function
            use_cache: Whether to use cache
            force_recompute: Whether to force recomputation
            
        Returns:
            Dataframe with computed feature
        """
        # Compute parameter hash
        feature_params_hash = self._compute_feature_params_hash(feature_params)
        
        # Get cache path
        cache_path = self._get_cache_path(
            dataset_id, feature_name, dataset_hash, feature_params_hash
        )
        
        # Check cache
        if use_cache and not force_recompute and os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    feature_df = pickle.load(f)
                self.logger.debug(f"Loaded cached feature: {feature_name}")
                return feature_df
            except Exception as e:
                self.logger.warning(f"Failed to load cached feature: {str(e)}")
                
        # Compute feature
        self.logger.debug(f"Computing feature: {feature_name}")
        
        try:
            # Use GPU if available
            if self.gpu_manager and self.gpu_manager.has_gpu:
                # Get numpy arrays for GPU processing
                arrays = {col: df[col].values for col in df.columns}
                
                # Transfer arrays to GPU
                with cp.cuda.Device(self.gpu_manager.selected_device):
                    gpu_arrays = {k: cp.asarray(v) for k, v in arrays.items()}
                    
                    # Compute feature
                    start_time = time.time()
                    feature_result = feature_func(gpu_arrays, **feature_params)
                    elapsed = time.time() - start_time
                    
                    # Transfer back to CPU
                    if isinstance(feature_result, dict):
                        feature_result = {k: cp.asnumpy(v) for k, v in feature_result.items()}
                    elif isinstance(feature_result, cp.ndarray):
                        feature_result = cp.asnumpy(feature_result)
                        
                    # Clean up GPU memory
                    self.gpu_manager.cleanup_gpu_memory()
            else:
                # Compute feature on CPU
                start_time = time.time()
                feature_result = feature_func(df, **feature_params)
                elapsed = time.time() - start_time
                
            # Convert to dataframe if result is a numpy array
            if isinstance(feature_result, np.ndarray):
                if feature_result.ndim == 1:
                    feature_df = pd.DataFrame({feature_name: feature_result}, index=df.index)
                else:
                    features = {}
                    for i in range(feature_result.shape[1]):
                        features[f"{feature_name}_{i}"] = feature_result[:, i]
                    feature_df = pd.DataFrame(features, index=df.index)
            elif isinstance(feature_result, dict):
                feature_df = pd.DataFrame(feature_result, index=df.index)
            else:
                feature_df = feature_result
                
            self.logger.info(f"Computed feature {feature_name} in {elapsed:.2f}s")
            
            # Cache feature
            if use_cache:
                with open(cache_path, 'wb') as f:
                    pickle.dump(feature_df, f)
                self.logger.debug(f"Cached feature: {feature_name}")
                
            return feature_df
            
        except Exception as e:
            self.logger.error(f"Failed to compute feature {feature_name}: {str(e)}")
            traceback.print_exc()
            return pd.DataFrame()
            
    def compute_feature_set(
        self,
        df: pd.DataFrame,
        dataset_id: str,
        dataset_hash: str,
        features_config: List[Dict[str, Any]],
        use_cache: bool = True,
        force_recompute: bool = False,
        parallel: bool = True,
        n_jobs: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Compute a set of features.
        
        Args:
            df: Input dataframe
            dataset_id: Identifier for the dataset
            dataset_hash: Hash of the dataset
            features_config: Configuration for features
            use_cache: Whether to use cache
            force_recompute: Whether to force recomputation
            parallel: Whether to compute features in parallel
            n_jobs: Number of parallel jobs
            
        Returns:
            Dataframe with computed features
        """
        if not features_config:
            return df.copy()
            
        feature_dfs = []
        
        # Define function to compute a single feature
        def compute_single_feature(feature_config):
            feature_name = feature_config['name']
            feature_func = self._get_feature_function(feature_config['function'])
            feature_params = feature_config.get('params', {})
            
            return self.compute_feature(
                df,
                dataset_id,
                dataset_hash,
                feature_name,
                feature_func,
                feature_params,
                use_cache,
                force_recompute
            )
            
        if parallel and len(features_config) > 1:
            # Compute features in parallel
            n_jobs = n_jobs or max(1, multiprocessing.cpu_count() - 1)
            self.logger.info(f"Computing {len(features_config)} features in parallel with {n_jobs} jobs")
            
            with multiprocessing.Pool(n_jobs) as pool:
                feature_dfs = list(
                    tqdm(
                        pool.imap(compute_single_feature, features_config),
                        total=len(features_config),
                        desc="Computing features"
                    )
                )
        else:
            # Compute features sequentially
            self.logger.info(f"Computing {len(features_config)} features sequentially")
            
            for config in tqdm(features_config, desc="Computing features"):
                feature_df = compute_single_feature(config)
                feature_dfs.append(feature_df)
                
        # Combine all feature dataframes
        result_df = df.copy()
        
        for feature_df in feature_dfs:
            if not feature_df.empty:
                # Get new columns
                new_columns = [col for col in feature_df.columns if col not in result_df.columns]
                
                if new_columns:
                    result_df = result_df.join(feature_df[new_columns])
                    
        return result_df
        
    def _get_feature_function(self, function_path: str) -> Callable:
        """
        Get feature function by path.
        
        Args:
            function_path: Path to function (module.submodule.function)
            
        Returns:
            Feature function
        """
        try:
            module_path, function_name = function_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            return getattr(module, function_name)
        except Exception as e:
            self.logger.error(f"Failed to load feature function {function_path}: {str(e)}")
            raise


# ================================
# Model Management Classes
# ================================

class ModelFactory:
    """
    Creates model instances based on configuration.
    
    This class provides utilities for dynamically creating model instances
    based on configuration, with support for different model types.
    """
    
    def __init__(self, models_dir: str):
        """
        Initialize model factory.
        
        Args:
            models_dir: Directory for model modules
        """
        self.models_dir = models_dir
        self.logger = logger
        
        # Add models directory to path if it exists
        if os.path.exists(models_dir) and os.path.isdir(models_dir):
            sys.path.append(os.path.abspath(models_dir))
            
    def create_model(
        self,
        model_config: Dict[str, Any],
        gpu_manager: Optional[GPUManager] = None
    ) -> Any:
        """
        Create a model instance based on configuration.
        
        Args:
            model_config: Model configuration
            gpu_manager: GPU manager instance
            
        Returns:
            Model instance
        """
        model_type = model_config.get('type')
        model_class = model_config.get('class')
        model_params = model_config.get('params', {})
        
        if not model_type or not model_class:
            raise ValueError("Model configuration must include 'type' and 'class'")
            
        try:
            # Dynamically import model class
            module_path = f"models.{model_type}.{model_class}"
            
            try:
                # Try direct import first
                module = importlib.import_module(module_path)
                model_cls = getattr(module, model_class)
            except ImportError:
                # If that fails, try importing from models directory
                if not os.path.exists(self.models_dir):
                    raise ImportError(f"Models directory not found: {self.models_dir}")
                    
                # Find model module file
                model_file = os.path.join(self.models_dir, model_type, f"{model_class.lower()}.py")
                
                if not os.path.exists(model_file):
                    raise ImportError(f"Model module not found: {model_file}")
                    
                # Load module from file
                spec = importlib.util.spec_from_file_location(model_class, model_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Get model class
                model_cls = getattr(module, model_class)
                
            # Add GPU manager to params if available
            if gpu_manager:
                model_params['gpu_manager'] = gpu_manager
                
            # Create model instance
            model_instance = model_cls(**model_params)
            
            self.logger.info(f"Created model: {model_type}.{model_class}")
            return model_instance
            
        except Exception as e:
            self.logger.error(f"Failed to create model {model_type}.{model_class}: {str(e)}")
            traceback.print_exc()
            raise
            
    def save_model(
        self,
        model: Any,
        model_dir: str,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a model to disk.
        
        Args:
            model: Model instance
            model_dir: Directory to save the model
            model_name: Name for the model
            metadata: Additional metadata
            
        Returns:
            Path to the saved model
        """
        os.makedirs(model_dir, exist_ok=True)
        
        # Create model filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_name}_{timestamp}.pkl"
        model_path = os.path.join(model_dir, model_filename)
        
        # Save model
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
                
            self.logger.info(f"Saved model to {model_path}")
            
            # Save metadata if provided
            if metadata:
                metadata_path = os.path.join(model_dir, f"{model_name}_{timestamp}_metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                    
            return model_path
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")
            raise
            
    def load_model(
        self,
        model_path: str
    ) -> Any:
        """
        Load a model from disk.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Loaded model instance
        """
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                
            self.logger.info(f"Loaded model from {model_path}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model from {model_path}: {str(e)}")
            raise


class WalkForwardManager:
    """
    Manages walk-forward analysis for models.
    
    This class provides utilities for performing walk-forward analysis,
    which involves training models on rolling windows of data.
    """
    
    def __init__(
        self,
        window_size: int,
        step_size: int,
        min_train_size: Optional[int] = None,
        forecast_horizon: int = 1,
        validation_ratio: float = 0.3,
        gap_size: int = 0
    ):
        """
        Initialize walk-forward manager.
        
        Args:
            window_size: Size of the rolling window
            step_size: Size of the step between windows
            min_train_size: Minimum training size (window_size if None)
            forecast_horizon: Horizon for forecasting
            validation_ratio: Ratio of window to use for validation
            gap_size: Gap between training and testing periods
        """
        self.window_size = window_size
        self.step_size = step_size
        self.min_train_size = min_train_size or window_size
        self.forecast_horizon = forecast_horizon
        self.validation_ratio = validation_ratio
        self.gap_size = gap_size
        self.logger = logger
        
    def generate_windows(
        self,
        data_length: int,
        start_index: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate walk-forward windows.
        
        Args:
            data_length: Length of the data
            start_index: Starting index (min_train_size if None)
            
        Returns:
            List of window dictionaries
        """
        if start_index is None:
            start_index = self.min_train_size
            
        windows = []
        test_index = start_index
        
        while test_index + self.forecast_horizon <= data_length:
            train_start = max(0, test_index - self.window_size)
            train_end = test_index - self.gap_size
            
            # Skip if training window is too small
            if train_end - train_start < self.min_train_size:
                test_index += self.step_size
                continue
                
            # Add validation window if needed
            if self.validation_ratio > 0:
                val_size = int((train_end - train_start) * self.validation_ratio)
                val_start = train_end - val_size
                
                windows.append({
                    'train_start': train_start,
                    'train_end': val_start,
                    'val_start': val_start,
                    'val_end': train_end,
                    'test_start': test_index,
                    'test_end': test_index + self.forecast_horizon,
                    'window_id': len(windows)
                })
            else:
                windows.append({
                    'train_start': train_start,
                    'train_end': train_end,
                    'val_start': None,
                    'val_end': None,
                    'test_start': test_index,
                    'test_end': test_index + self.forecast_horizon,
                    'window_id': len(windows)
                })
                
            test_index += self.step_size
            
        return windows
        
    def execute_walk_forward(
        self,
        df: pd.DataFrame,
        model_factory: ModelFactory,
        model_config: Dict[str, Any],
        target_col: str,
        feature_cols: List[str],
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
        gpu_manager: Optional[GPUManager] = None,
        checkpoint_interval: int = 10,
        save_models: bool = False,
        models_dir: Optional[str] = None,
        experiment_id: Optional[str] = None,
        callbacks: Optional[List[Callable]] = None
    ) -> Dict[str, Any]:
        """
        Execute walk-forward analysis.
        
        Args:
            df: Input dataframe
            model_factory: Model factory instance
            model_config: Model configuration
            target_col: Target column name
            feature_cols: Feature column names
            start_index: Starting index
            end_index: Ending index
            gpu_manager: GPU manager instance
            checkpoint_interval: Interval for checkpointing
            save_models: Whether to save trained models
            models_dir: Directory to save models
            experiment_id: Experiment ID
            callbacks: Callback functions
            
        Returns:
            Results dictionary
        """
        # Validate inputs
        if target_col not in df.columns:
            raise ValueError(f"Target column not found: {target_col}")
            
        for col in feature_cols:
            if col not in df.columns:
                raise ValueError(f"Feature column not found: {col}")
                
        # Prepare data
        data_length = len(df)
        if end_index is None:
            end_index = data_length
            
        # Generate windows
        windows = self.generate_windows(end_index, start_index)
        
        if not windows:
            self.logger.warning("No valid windows generated")
            return {'predictions': pd.DataFrame(), 'metrics': {}, 'windows': []}
            
        self.logger.info(f"Generated {len(windows)} walk-forward windows")
        
        # Initialize results
        all_predictions = []
        all_metrics = []
        saved_models = []
        
        # Create results dataframe
        results_df = pd.DataFrame(index=df.index)
        results_df['actual'] = df[target_col]
        results_df['prediction'] = np.nan
        results_df['window_id'] = np.nan
        
        # Execute walk-forward analysis
        for i, window in enumerate(tqdm(windows, desc="Walk-forward progress")):
            window_id = window['window_id']
            
            # Get window data
            train_df = df.iloc[window['train_start']:window['train_end']]
            
            if window['val_start'] is not None:
                val_df = df.iloc[window['val_start']:window['val_end']]
            else:
                val_df = None
                
            test_df = df.iloc[window['test_start']:window['test_end']]
            
            # Prepare training data
            X_train = train_df[feature_cols]
            y_train = train_df[target_col]
            
            if val_df is not None:
                X_val = val_df[feature_cols]
                y_val = val_df[target_col]
            else:
                X_val, y_val = None, None
                
            X_test = test_df[feature_cols]
            y_test = test_df[target_col]
            
            # Create and train model
            try:
                model = model_factory.create_model(model_config, gpu_manager)
                
                # Train model
                train_start_time = time.time()
                
                if val_df is not None:
                    model.fit(X_train, y_train, X_val, y_val)
                else:
                    model.fit(X_train, y_train)
                    
                train_time = time.time() - train_start_time
                
                # Make predictions
                pred_start_time = time.time()
                predictions = model.predict(X_test)
                pred_time = time.time() - pred_start_time
                
                # Calculate metrics
                metrics = self._calculate_metrics(y_test, predictions)
                metrics.update({
                    'window_id': window_id,
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    'train_time': train_time,
                    'pred_time': pred_time,
                })
                
                # Store results
                test_indices = test_df.index
                for j, idx in enumerate(test_indices):
                    results_df.loc[idx, 'prediction'] = predictions[j]
                    results_df.loc[idx, 'window_id'] = window_id
                    
                all_metrics.append(metrics)
                
                # Save model if requested
                if save_models and models_dir:
                    model_path = model_factory.save_model(
                        model,
                        models_dir,
                        f"{experiment_id}_window_{window_id}",
                        {
                            'window': window,
                            'metrics': metrics,
                            'feature_cols': feature_cols,
                            'target_col': target_col,
                            'model_config': model_config
                        }
                    )
                    saved_models.append(model_path)
                    
                # Call callbacks if provided
                if callbacks:
                    for callback in callbacks:
                        callback(window, model, metrics, predictions)
                        
                # Clean up GPU memory
                if gpu_manager:
                    gpu_manager.cleanup_gpu_memory()
                    
                # Checkpoint if needed
                if checkpoint_interval > 0 and (i + 1) % checkpoint_interval == 0:
                    self.logger.info(f"Checkpoint at window {i + 1}/{len(windows)}")
                    checkpoint_data = {
                        'results_df': results_df,
                        'metrics': all_metrics,
                        'saved_models': saved_models,
                        'completed_windows': i + 1
                    }
                    yield checkpoint_data
                    
            except Exception as e:
                self.logger.error(f"Error in window {window_id}: {str(e)}")
                traceback.print_exc()
                
                # Add failed window to metrics
                all_metrics.append({
                    'window_id': window_id,
                    'error': str(e),
                    'status': 'failed'
                })
                
                # Clean up GPU memory
                if gpu_manager:
                    gpu_manager.cleanup_gpu_memory()
                    
        # Compile final results
        metrics_df = pd.DataFrame(all_metrics)
        
        # Calculate overall metrics
        overall_metrics = {}
        try:
            valid_predictions = results_df.dropna(subset=['prediction'])
            if len(valid_predictions) > 0:
                overall_metrics = self._calculate_metrics(
                    valid_predictions['actual'],
                    valid_predictions['prediction']
                )
                overall_metrics['total_windows'] = len(windows)
                overall_metrics['successful_windows'] = len([m for m in all_metrics if 'error' not in m])
        except Exception as e:
            self.logger.error(f"Error calculating overall metrics: {str(e)}")
            
        results = {
            'predictions': results_df,
            'metrics': {
                'windows': all_metrics,
                'overall': overall_metrics
            },
            'windows': windows,
            'saved_models': saved_models
        }
        
        return results
        
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        # Skip metrics if data is empty
        if len(y_true) == 0 or len(y_pred) == 0:
            return {}
            
        # Calculate basic metrics
        metrics = {}
        
        try:
            # Mean squared error
            metrics['mse'] = np.mean((y_true - y_pred) ** 2)
            
            # Root mean squared error
            metrics['rmse'] = np.sqrt(metrics['mse'])
            
            # Mean absolute error
            metrics['mae'] = np.mean(np.abs(y_true - y_pred))
            
            # Mean absolute percentage error
            with np.errstate(divide='ignore', invalid='ignore'):
                mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                metrics['mape'] = float(mape) if not np.isnan(mape) else None
                
            # R-squared
            if np.var(y_true) > 0:
                metrics['r2'] = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
            else:
                metrics['r2'] = 0
                
            # Direction accuracy
            if len(y_true) > 1:
                true_dir = np.sign(np.diff(y_true))
                pred_dir = np.sign(np.diff(y_pred))
                metrics['dir_acc'] = np.mean(true_dir == pred_dir)
                
        except Exception as e:
            self.logger.warning(f"Error calculating metrics: {str(e)}")
            
        return metrics


# ================================
# Backtesting Engine Classes
# ================================

class BacktestEngine:
    """
    Manages backtesting of trading strategies.
    
    This class provides utilities for backtesting trading strategies,
    with support for position sizing, transaction costs, and performance metrics.
    """
    
    def __init__(
        self,
        initial_capital: float = 1e6,
        transaction_cost: float = 0.0003,
        enable_fractional: bool = True,
        execution_delay: int = 0
    ):
        """
        Initialize backtest engine.
        
        Args:
            initial_capital: Initial capital
            transaction_cost: Transaction cost as a fraction
            enable_fractional: Whether to allow fractional positions
            execution_delay: Delay between signal and execution (in periods)
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.enable_fractional = enable_fractional
        self.execution_delay = execution_delay
        self.logger = logger
        
    def run_backtest(
        self,
        price_data: pd.DataFrame,
        signals: pd.DataFrame,
        position_sizes: Optional[pd.Series] = None,
        price_col: str = 'close',
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run a backtest.
        
        Args:
            price_data: Price data dataframe
            signals: Trading signals dataframe
            position_sizes: Position sizes series
            price_col: Column name for prices
            verbose: Whether to log progress
            
        Returns:
            Dictionary with backtest results
        """
        # Validate inputs
        if price_col not in price_data.columns:
            raise ValueError(f"Price column not found: {price_col}")
            
        # Convert inputs to same index
        prices = price_data[price_col].reindex(signals.index)
        signals = signals.reindex(prices.index)
        
        # Initialize position sizing if not provided
        if position_sizes is None:
            position_sizes = pd.Series(1.0, index=signals.index)
        else:
            position_sizes = position_sizes.reindex(signals.index).fillna(0)
            
        # Apply execution delay if needed
        if self.execution_delay > 0:
            signals = signals.shift(self.execution_delay)
            position_sizes = position_sizes.shift(self.execution_delay)
            
        # Initialize portfolio
        portfolio = pd.DataFrame(index=prices.index)
        portfolio['price'] = prices
        portfolio['signal'] = signals
        portfolio['position_size'] = position_sizes
        portfolio['position'] = 0
        portfolio['cash'] = self.initial_capital
        portfolio['holdings'] = 0
        portfolio['equity'] = self.initial_capital
        portfolio['returns'] = 0
        portfolio['trade'] = False
        portfolio['cost'] = 0
        
        # Execute backtest
        current_position = 0
        buy_price = 0
        
        for i, idx in enumerate(tqdm(portfolio.index, desc="Backtesting", disable=not verbose)):
            price = portfolio.loc[idx, 'price']
            signal = portfolio.loc[idx, 'signal']
            position_size = portfolio.loc[idx, 'position_size']
            
            # Skip if price is missing
            if pd.isna(price):
                continue
                
            # Get previous portfolio state
            if i > 0:
                prev_idx = portfolio.index[i-1]
                prev_cash = portfolio.loc[prev_idx, 'cash']
                prev_holdings = portfolio.loc[prev_idx, 'holdings']
                prev_position = portfolio.loc[prev_idx, 'position']
            else:
                prev_cash = self.initial_capital
                prev_holdings = 0
                prev_position = 0
                
            # Initialize with previous state
            cash = prev_cash
            holdings = prev_holdings
            position = prev_position
            trade = False
            cost = 0
            
            # Execute trading logic
            if signal == 'buy' and position <= 0:
                # Calculate position size
                trade_value = prev_cash * position_size
                shares = trade_value / price
                
                if not self.enable_fractional:
                    shares = np.floor(shares)
                    
                # Calculate cost
                cost = shares * price * self.transaction_cost
                
                # Update portfolio
                cash = prev_cash - (shares * price) - cost
                holdings = shares
                position = 1
                trade = True
                buy_price = price
                
            elif signal == 'sell' and position >= 0:
                # Calculate cost
                cost = prev_holdings * price * self.transaction_cost
                
                # Update portfolio
                cash = prev_cash + (prev_holdings * price) - cost
                holdings = 0
                position = -1
                trade = True
                
            # Update portfolio values
            portfolio.loc[idx, 'position'] = position
            portfolio.loc[idx, 'cash'] = cash
            portfolio.loc[idx, 'holdings'] = holdings
            portfolio.loc[idx, 'equity'] = cash + (holdings * price)
            portfolio.loc[idx, 'trade'] = trade
            portfolio.loc[idx, 'cost'] = cost
            
            # Calculate returns
            if i > 0:
                prev_equity = portfolio.loc[prev_idx, 'equity']
                portfolio.loc[idx, 'returns'] = (portfolio.loc[idx, 'equity'] / prev_equity) - 1
                
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(portfolio)
        
        return {
            'portfolio': portfolio,
            'metrics': metrics,
            'config': {
                'initial_capital': self.initial_capital,
                'transaction_cost': self.transaction_cost,
                'enable_fractional': self.enable_fractional,
                'execution_delay': self.execution_delay
            }
        }
        
    def _calculate_performance_metrics(self, portfolio: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate performance metrics for a backtest.
        
        Args:
            portfolio: Portfolio dataframe
            
        Returns:
            Dictionary with performance metrics
        """
        metrics = {}
        
        try:
            # Total return
            initial_equity = portfolio['equity'].iloc[0]
            final_equity = portfolio['equity'].iloc[-1]
            metrics['total_return'] = (final_equity / initial_equity) - 1
            metrics['total_return_pct'] = metrics['total_return'] * 100
            
            # Annualized return
            days = (portfolio.index[-1] - portfolio.index[0]).days
            if days > 0:
                years = days / 365
                metrics['annualized_return'] = ((1 + metrics['total_return']) ** (1 / years)) - 1
                metrics['annualized_return_pct'] = metrics['annualized_return'] * 100
                
            # Daily returns
            daily_returns = portfolio['returns'].resample('D').sum()
            metrics['daily_return_mean'] = daily_returns.mean()
            metrics['daily_return_std'] = daily_returns.std()
            
            # Sharpe ratio
            if metrics['daily_return_std'] > 0:
                metrics['sharpe_ratio'] = (metrics['daily_return_mean'] / metrics['daily_return_std']) * np.sqrt(252)
                
            # Drawdown analysis
            portfolio['drawdown'] = 1 - (portfolio['equity'] / portfolio['equity'].cummax())
            metrics['max_drawdown'] = portfolio['drawdown'].max()
            metrics['max_drawdown_pct'] = metrics['max_drawdown'] * 100
            
            # Find max drawdown period
            max_dd_end = portfolio['drawdown'].idxmax()
            max_dd_start = portfolio.loc[:max_dd_end, 'equity'].idxmax()
            metrics['max_drawdown_start'] = max_dd_start
            metrics['max_drawdown_end'] = max_dd_end
            metrics['max_drawdown_days'] = (max_dd_end - max_dd_start).days
            
            # Trade analysis
            trades = portfolio[portfolio['trade']]
            metrics['trade_count'] = len(trades)
            metrics['transaction_costs'] = portfolio['cost'].sum()
            
            # Win rate (requires more detailed tracking of individual trades)
            # This is a simplification
            if metrics['trade_count'] > 0:
                wins = 0
                ongoing_trade = False
                entry_price = 0
                entry_idx = None
                
                for i, (idx, row) in enumerate(portfolio.iterrows()):
                    if row['trade']:
                        if row['position'] == 1:  # Buy signal
                            ongoing_trade = True
                            entry_price = row['price']
                            entry_idx = idx
                        elif row['position'] == -1 and ongoing_trade:  # Sell signal after a buy
                            ongoing_trade = False
                            if row['price'] > entry_price:
                                wins += 1
                                
                metrics['win_count'] = wins
                metrics['win_rate'] = wins / metrics['trade_count']
                
            # Risk-adjusted metrics
            if metrics.get('annualized_return') and metrics.get('max_drawdown'):
                metrics['calmar_ratio'] = metrics['annualized_return'] / max(0.01, metrics['max_drawdown'])
                
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {str(e)}")
            
        return metrics


class PositionSizer:
    """
    Determines position sizes for trading signals.
    
    This class provides utilities for calculating position sizes based on
    various strategies, such as Kelly criterion and risk-based sizing.
    """
    
    def __init__(
        self,
        strategy: str = 'fixed',
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize position sizer.
        
        Args:
            strategy: Position sizing strategy
            params: Parameters for the strategy
        """
        self.strategy = strategy
        self.params = params or {}
        self.logger = logger
        
    def calculate_position_sizes(
        self,
        signals: pd.DataFrame,
        prices: pd.Series,
        volatility: Optional[pd.Series] = None,
        confidence: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Calculate position sizes for trading signals.
        
        Args:
            signals: Trading signals dataframe
            prices: Price series
            volatility: Volatility series
            confidence: Confidence series
            
        Returns:
            Position sizes series
        """
        if self.strategy == 'fixed':
            return self._fixed_sizing(signals)
        elif self.strategy == 'kelly':
            return self._kelly_sizing(signals, prices, volatility, confidence)
        elif self.strategy == 'risk_parity':
            return self._risk_parity_sizing(signals, prices, volatility)
        elif self.strategy == 'volatility_adjusted':
            return self._volatility_adjusted_sizing(signals, prices, volatility)
        else:
            self.logger.warning(f"Unknown position sizing strategy: {self.strategy}, using fixed sizing")
            return self._fixed_sizing(signals)
            
    def _fixed_sizing(self, signals: pd.DataFrame) -> pd.Series:
        """
        Fixed position sizing strategy.
        
        Args:
            signals: Trading signals dataframe
            
        Returns:
            Position sizes series
        """
        default_size = self.params.get('size', 1.0)
        return pd.Series(default_size, index=signals.index)
        
    def _kelly_sizing(
        self,
        signals: pd.DataFrame,
        prices: pd.Series,
        volatility: Optional[pd.Series] = None,
        confidence: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Kelly criterion position sizing strategy.
        
        Args:
            signals: Trading signals dataframe
            prices: Price series
            volatility: Volatility series
            confidence: Confidence series
            
        Returns:
            Position sizes series
        """
        # Parameters
        win_rate = self.params.get('win_rate', 0.5)
        payoff_ratio = self.params.get('payoff_ratio', 2.0)
        kelly_fraction = self.params.get('kelly_fraction', 0.5)
        max_size = self.params.get('max_size', 1.0)
        
        # Create position sizes series
        position_sizes = pd.Series(0.0, index=signals.index)
        
        # Adjust win rate with confidence if available
        if confidence is not None:
            win_rate = confidence.clip(0.0, 1.0)
            
        # Calculate Kelly criterion
        kelly = (win_rate * payoff_ratio - (1 - win_rate)) / payoff_ratio
        kelly = kelly * kelly_fraction
        kelly = np.clip(kelly, 0.0, max_size)
        
        # Apply to signals
        position_sizes[signals == 'buy'] = kelly
        
        # Adjust for volatility if available
        if volatility is not None:
            # Normalize volatility
            vol_factor = self.params.get('vol_target', 0.01) / volatility
            vol_factor = vol_factor.clip(0.1, 10.0)
            
            # Apply volatility adjustment
            position_sizes = position_sizes * vol_factor
            
        return position_sizes
        
    def _risk_parity_sizing(
        self,
        signals: pd.DataFrame,
        prices: pd.Series,
        volatility: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Risk parity position sizing strategy.
        
        Args:
            signals: Trading signals dataframe
            prices: Price series
            volatility: Volatility series
            
        Returns:
            Position sizes series
        """
        # Parameters
        risk_target = self.params.get('risk_target', 0.01)
        max_size = self.params.get('max_size', 1.0)
        
        # Create position sizes series
        position_sizes = pd.Series(0.0, index=signals.index)
        
        # Simple implementation if no volatility is provided
        if volatility is None:
            position_sizes[signals == 'buy'] = max_size
            return position_sizes
            
        # Calculate position sizes based on risk parity
        position_sizes[signals == 'buy'] = risk_target / volatility
        position_sizes = position_sizes.clip(0.0, max_size)
        
        return position_sizes
        
    def _volatility_adjusted_sizing(
        self,
        signals: pd.DataFrame,
        prices: pd.Series,
        volatility: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Volatility-adjusted position sizing strategy.
        
        Args:
            signals: Trading signals dataframe
            prices: Price series
            volatility: Volatility series
            
        Returns:
            Position sizes series
        """
        # Parameters
        base_size = self.params.get('base_size', 0.5)
        vol_target = self.params.get('vol_target', 0.01)
        max_size = self.params.get('max_size', 1.0)
        
        # Create position sizes series
        position_sizes = pd.Series(base_size, index=signals.index)
        
        # Apply only to buy signals
        position_sizes[signals != 'buy'] = 0.0
        
        # Adjust for volatility if available
        if volatility is not None:
            # Normalize volatility
            vol_factor = vol_target / volatility
            vol_factor = vol_factor.clip(0.1, 10.0)
            
            # Apply volatility adjustment
            position_sizes = position_sizes * vol_factor
            
        # Clip to maximum size
        position_sizes = position_sizes.clip(0.0, max_size)
        
        return position_sizes


# ================================
# Performance Tracking Classes
# ================================

class MetricsTracker:
    """
    Tracks and aggregates performance metrics.
    
    This class provides utilities for tracking, aggregating, and analyzing
    performance metrics from experiments.
    """
    
    def __init__(self, metrics_dir: str):
        """
        Initialize metrics tracker.
        
        Args:
            metrics_dir: Directory for metrics storage
        """
        self.metrics_dir = metrics_dir
        self.logger = logger
        
        # Create directory if it doesn't exist
        os.makedirs(metrics_dir, exist_ok=True)
        
    def save_metrics(
        self,
        metrics: Dict[str, Any],
        experiment_id: str,
        timestamp: Optional[str] = None
    ) -> str:
        """
        Save metrics to disk.
        
        Args:
            metrics: Metrics dictionary
            experiment_id: Experiment ID
            timestamp: Timestamp string
            
        Returns:
            Path to the saved metrics file
        """
        if timestamp is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
        metrics_file = os.path.join(self.metrics_dir, f"{experiment_id}_{timestamp}_metrics.json")
        
        try:
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
                
            self.logger.info(f"Saved metrics to {metrics_file}")
            return metrics_file
            
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {str(e)}")
            raise
            
    def load_metrics(self, metrics_file: str) -> Dict[str, Any]:
        """
        Load metrics from disk.
        
        Args:
            metrics_file: Path to metrics file
            
        Returns:
            Metrics dictionary
        """
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                
            self.logger.debug(f"Loaded metrics from {metrics_file}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to load metrics from {metrics_file}: {str(e)}")
            raise
            
    def aggregate_metrics(
        self,
        metrics_list: List[Dict[str, Any]],
        group_by: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Aggregate metrics from multiple sources.
        
        Args:
            metrics_list: List of metrics dictionaries
            group_by: Field to group by
            
        Returns:
            Aggregated metrics dictionary
        """
        if not metrics_list:
            return {}
            
        # Group metrics if needed
        if group_by:
            grouped_metrics = {}
            
            for metrics in metrics_list:
                if group_by not in metrics:
                    continue
                    
                group_value = metrics[group_by]
                if group_value not in grouped_metrics:
                    grouped_metrics[] = {'status': 'error', 'message': f"Column not found: {column}"}
                    continue
                    
                violations = 0
                if min_value is not None:
                    violations += (df[column] < min_value).sum()
                if max_value is not None:
                    violations += (df[column] > max_value).sum()
                    
                results[rule_id] = {
                    'status': 'ok' if violations == 0 else 'violation',
                    'violations': int(violations),
                    'percent_violations': float(violations / len(df) * 100),
                }
                
            elif rule_type == 'uniqueness':
                # Uniqueness check
                columns = rule.get('columns', [])
                
                if not all(col in df.columns for col in columns):
                    results[rule_id] = {
                        'status': 'error', 
                        'message': f"One or more columns not found: {columns}"
                    }
                    continue
                    
                duplicates = df.duplicated(subset=columns).sum()
                results[rule_id
                        