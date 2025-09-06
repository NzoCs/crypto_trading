#!/usr/bin/env python3
"""
Backtesting Script for IFCOB Strategies

This script provides a command-line interface to run backtests using strategies 
from the backtesting.strategies module. It supports running single or multiple strategies,
custom data sources, and saving results.

Usage:
    python scripts/run_backtest.py --strategy TFCumulativeReturnStrategy --data-index 1
    ifcob-backtest --strategy Mateo2StartStrategy --data-index 2 --save-trades
"""

import argparse
import sys
import os
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Type
import json
import pandas as pd
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backtesting.backtest import Backtester, BacktestResult, BacktestConfig
from backtesting.base_strategy import Strategy
from backtesting.backtest_types import Coin, FeesGraph
from backtesting.dataloader import OrderBookDataFromDf


class BacktestRunner:
    """Class to manage and run backtests with different strategies."""
    
    def __init__(self):
        self.available_strategies = self._discover_strategies()
        
    def _discover_strategies(self) -> Dict[str, Type[Strategy]]:
        """Discover all available strategy classes."""
        strategies = {}
        
        # Import all strategy modules
        strategy_modules = [
            "backtesting.strategies.trend_following",
            "backtesting.strategies.momentum_strategy", 
            "backtesting.strategies.linear_regression",
            "backtesting.strategies.mateo_2_start",
            "backtesting.strategies.rf_pred_all_signed_strat_mateo",
        ]
        
        for module_name in strategy_modules:
            try:
                module = importlib.import_module(module_name)
                
                # Find all Strategy subclasses in the module
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, Strategy) and 
                        obj != Strategy):
                        strategies[name] = obj
                        
            except ImportError as e:
                print(f"Warning: Could not import {module_name}: {e}")
                
        return strategies
    
    def list_strategies(self):
        """Print all available strategies."""
        print("Available strategies:")
        for name in sorted(self.available_strategies.keys()):
            print(f"  - {name}")
    
    def get_data_path(self, data_index: int, coin: str) -> str:
        """Get the path to preprocessed data."""
        base_path = Path("data/preprocessed")
        data_folder = f"DATA_{data_index}"
        filename = f"{coin}_EUR.parquet"
        
        full_path = base_path / data_folder / filename
        
        if not full_path.exists():
            raise FileNotFoundError(f"Data file not found: {full_path}")
            
        return str(full_path)
    
    def run_backtest(
        self,
        strategy_name: str,
        data_index: int,
        window_size: int = 5,
        initial_capital: float = 100000.0,
        data_sources: Optional[Dict[str, str]] = None,
        verbose: bool = True
    ) -> BacktestResult:
        """
        Run a backtest with the specified strategy.
        
        Args:
            strategy_name: Name of the strategy class to use
            data_index: Data folder index (0, 1, 2)
            window_size: Window size parameter for strategies
            initial_capital: Starting capital
            data_sources: Custom data file paths {coin: path}
            verbose: Print progress information
            
        Returns:
            BacktestResult object with results
        """
        if strategy_name not in self.available_strategies:
            raise ValueError(f"Strategy '{strategy_name}' not found. Available: {list(self.available_strategies.keys())}")
        
        strategy_class = self.available_strategies[strategy_name]
        
        # Determine data sources
        if data_sources is None:
            data_sources = {
                "XBT": self.get_data_path(data_index, "XBT"),
                "ETH": self.get_data_path(data_index, "ETH")
            }
        
        if verbose:
            print(f"Running backtest with {strategy_name}")
            print(f"Data sources:")
            for coin, path in data_sources.items():
                print(f"  {coin}: {path}")
            print(f"Initial capital: ${initial_capital:,.2f}")
            print(f"Window size: {window_size}")
            print()
        
        # Create data sources for dataloader
        sources = [(coin, path) for coin, path in data_sources.items()]
        
        # Create dataloader
        dataloader = OrderBookDataFromDf(sources)
        
        # Create simple fees graph (you can customize this)
        fees_graph = FeesGraph(
            maker_fee=0.001,  # 0.1% maker fee
            taker_fee=0.002   # 0.2% taker fee
        )
        
        # Create backtest config
        config = BacktestConfig(
            initial_capital=initial_capital,
            fees_graph=fees_graph,
            symbols=list(data_sources.keys()),
            window_size=window_size
        )
        
        # Create strategy instance
        strategy = strategy_class(window_size=window_size)
        
        # Create backtest instance
        backtester = Backtester(dataloader, config)
        
        # Run the backtest
        results = backtester.backtest([strategy])
        
        # Extract result for our single strategy
        strategy_key = list(results.keys())[0]
        result = results[strategy_key][0]  # Use in-sample result
        
        if verbose:
            self._print_results(result, strategy_name)
        
        return result
    
    def _print_results(self, result: BacktestResult, strategy_name: str):
        """Print formatted backtest results."""
        print(f"\n{'='*60}")
        print(f"BACKTEST RESULTS: {strategy_name}")
        print(f"{'='*60}")
        print(f"Final Portfolio Value: ${result.final_portfolio_value:,.2f}")
        print(f"Total Return: {result.total_return:.2%}")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.4f}")
        print(f"Max Drawdown: {result.max_drawdown:.2%}")
        print(f"Win Rate: {result.win_rate:.2%}")
        print(f"Total Transaction Costs: ${result.transaction_costs:,.2f}")
        print(f"Number of Trades: {len(result.trades)}")
        print(f"{'='*60}")
    
    def save_results(
        self, 
        result: BacktestResult, 
        strategy_name: str, 
        output_dir: str,
        save_trades: bool = False
    ):
        """Save backtest results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{strategy_name}_{timestamp}"
        
        # Save summary results
        summary = {
            "strategy": strategy_name,
            "timestamp": timestamp,
            "final_portfolio_value": result.final_portfolio_value,
            "total_return": result.total_return,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "transaction_costs": result.transaction_costs,
            "num_trades": len(result.trades)
        }
        
        summary_file = output_path / f"{base_filename}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save portfolio values time series
        portfolio_df = pd.DataFrame({
            'timestamp': result.timestamps,
            'portfolio_value': result.portfolio_values
        })
        portfolio_file = output_path / f"{base_filename}_portfolio.csv"
        portfolio_df.to_csv(portfolio_file, index=False)
        
        # Save trades if requested
        if save_trades and result.trades:
            trades_df = pd.DataFrame(result.trades)
            trades_file = output_path / f"{base_filename}_trades.csv"
            trades_df.to_csv(trades_file, index=False)
        
        print(f"Results saved to {output_path}/")
        print(f"  Summary: {summary_file.name}")
        print(f"  Portfolio: {portfolio_file.name}")
        if save_trades:
            print(f"  Trades: {trades_file.name}")


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run backtests using IFCOB strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_backtest.py --strategy TFCumulativeReturnStrategy --data-index 1
  python scripts/run_backtest.py --strategy Mateo2StartStrategy --data-index 2 --save-trades
  python scripts/run_backtest.py --list-strategies
        """
    )
    
    parser.add_argument(
        "--strategy", 
        type=str,
        action="append",
        help="Strategy class name to run (can be used multiple times)"
    )
    
    parser.add_argument(
        "--data-index",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Data folder index (0, 1, or 2)"
    )
    
    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="Window size parameter for strategies (default: 5)"
    )
    
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=100000.0,
        help="Initial capital for backtesting (default: 100000)"
    )
    
    parser.add_argument(
        "--data-sources",
        type=str,
        nargs="+",
        help="Custom data sources in format COIN:path (e.g., XBT:data/custom/XBT.parquet)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="backtest_results",
        help="Output directory for results (default: backtest_results)"
    )
    
    parser.add_argument(
        "--save-trades",
        action="store_true",
        help="Save detailed trade logs"
    )
    
    parser.add_argument(
        "--list-strategies",
        action="store_true",
        help="List all available strategies and exit"
    )
    
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling for performance analysis"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    return parser


def parse_data_sources(data_sources_list: List[str]) -> Dict[str, str]:
    """Parse data sources from command line format."""
    data_sources = {}
    for item in data_sources_list:
        if ":" not in item:
            raise ValueError(f"Invalid data source format: {item}. Expected COIN:path")
        
        coin, path = item.split(":", 1)
        data_sources[coin.upper()] = path
    
    return data_sources


def main():
    """Main entry point for the script."""
    parser = create_parser()
    args = parser.parse_args()
    
    runner = BacktestRunner()
    
    if args.list_strategies:
        runner.list_strategies()
        return
    
    if not args.strategy:
        print("Error: No strategy specified. Use --strategy or --list-strategies")
        parser.print_help()
        return
    
    # Parse custom data sources if provided
    data_sources = None
    if args.data_sources:
        try:
            data_sources = parse_data_sources(args.data_sources)
        except ValueError as e:
            print(f"Error: {e}")
            return
    
    verbose = not args.quiet
    
    # Run backtests for each strategy
    for strategy_name in args.strategy:
        try:
            if args.profile:
                import cProfile
                import pstats
                
                profiler = cProfile.Profile()
                profiler.enable()
            
            result = runner.run_backtest(
                strategy_name=strategy_name,
                data_index=args.data_index,
                window_size=args.window_size,
                initial_capital=args.initial_capital,
                data_sources=data_sources,
                verbose=verbose
            )
            
            if args.profile:
                profiler.disable()
                stats = pstats.Stats(profiler)
                stats.sort_stats('cumulative')
                print(f"\nProfiling results for {strategy_name}:")
                stats.print_stats(20)  # Top 20 functions
            
            # Save results if output directory specified
            if args.output_dir:
                runner.save_results(
                    result=result,
                    strategy_name=strategy_name,
                    output_dir=args.output_dir,
                    save_trades=args.save_trades
                )
            
        except Exception as e:
            print(f"Error running backtest for {strategy_name}: {e}")
            if verbose:
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()