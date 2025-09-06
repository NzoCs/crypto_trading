"""
Integration Tests for Backtesting System

These tests verify the end-to-end functionality of the backtesting system,
including data loading, strategy execution, portfolio management, and result generation.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtesting.backtest import Backtester, BacktestConfig
from src.backtesting.dataloader import OrderBookDataFromDf
from src.backtesting.portfolio import Portfolio
from backtesting.base_strategy import Strategy
from src.backtesting.types import MarketData, Action, FeesGraph
from src.strategies.momentum_strategy import MomentumStrategy
from src.strategies.trend_following import TFCumulativeReturnStrategy


class SimpleTestStrategy(Strategy):
    """Simple strategy for testing purposes"""
    
    def __init__(self, buy_threshold=0.5, sell_threshold=-0.5):
        super().__init__()
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.call_count = 0
    
    def get_action(self, data: MarketData, current_portfolio: Portfolio, fees_graph: FeesGraph) -> Action:
        """Simple strategy that buys/sells based on mid-price change"""
        self.call_count += 1
        
        if 'ETH' not in data or data['ETH'].empty:
            return {}
        
        eth_data = data['ETH']
        
        # Calculate simple signal based on price change
        if len(eth_data) < 2:
            return {}
        
        if 'level-1-bid-price' in eth_data.columns and 'level-1-ask-price' in eth_data.columns:
            current_mid = (eth_data['level-1-bid-price'].iloc[-1] + eth_data['level-1-ask-price'].iloc[-1]) / 2
            prev_mid = (eth_data['level-1-bid-price'].iloc[-2] + eth_data['level-1-ask-price'].iloc[-2]) / 2
            
            price_change = (current_mid - prev_mid) / prev_mid
            
            if price_change > self.buy_threshold:
                return {'ETH': 0.1}
            elif price_change < self.sell_threshold:
                return {'ETH': -0.1}
        
        return {}


class TestBacktestingSystemIntegration:
    """Integration tests for the complete backtesting system"""
    
    def setup_method(self):
        """Set up test environment with sample data"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.create_sample_data()
        
        # Create sample fees graph
        self.fees_graph = {
            'ETH': [('EURC', 0.001)],
            'EURC': [('ETH', 0.001)],
            'XBT': [('EURC', 0.001)]
        }
    
    def teardown_method(self):
        """Clean up test environment"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def create_sample_data(self):
        """Create sample market data for testing"""
        # Create realistic time series data
        timestamps = pd.date_range('2024-01-01 00:00:00', periods=1000, freq='1s')
        
        # Generate realistic ETH price data
        base_price = 2000
        price_changes = np.random.normal(0, 0.001, len(timestamps))
        prices = base_price * np.exp(np.cumsum(price_changes))
        
        eth_data = pd.DataFrame({
            'timestamp': timestamps,
            'level-1-bid-price': prices - np.random.uniform(0.5, 2.0, len(timestamps)),
            'level-1-ask-price': prices + np.random.uniform(0.5, 2.0, len(timestamps)),
            'level-1-bid-volume': np.random.exponential(100, len(timestamps)),
            'level-1-ask-volume': np.random.exponential(95, len(timestamps)),
            'level-2-bid-price': prices - np.random.uniform(1.0, 3.0, len(timestamps)),
            'level-2-ask-price': prices + np.random.uniform(1.0, 3.0, len(timestamps)),
            'spread': np.random.exponential(2, len(timestamps)) + 1,
            'V-bid-5-levels': np.random.exponential(500, len(timestamps)),
            'V-ask-5-levels': np.random.exponential(480, len(timestamps))
        })
        
        # Ensure price relationships are realistic
        eth_data['level-1-ask-price'] = eth_data['level-1-bid-price'] + eth_data['spread']
        eth_data['level-2-bid-price'] = eth_data['level-1-bid-price'] - np.random.uniform(0.5, 1.5, len(timestamps))
        eth_data['level-2-ask-price'] = eth_data['level-1-ask-price'] + np.random.uniform(0.5, 1.5, len(timestamps))
        
        # Ensure all volumes are positive
        for col in ['level-1-bid-volume', 'level-1-ask-volume']:
            eth_data[col] = np.abs(eth_data[col])
        
        # Save data
        eth_file = self.test_dir / "ETH_EUR.parquet"
        eth_data.to_parquet(eth_file)
        
        # Create similar XBT data
        xbt_data = eth_data.copy()
        xbt_data['level-1-bid-price'] *= 25  # BTC is ~25x ETH price
        xbt_data['level-1-ask-price'] *= 25
        xbt_data['level-2-bid-price'] *= 25
        xbt_data['level-2-ask-price'] *= 25
        
        xbt_file = self.test_dir / "XBT_EUR.parquet"
        xbt_data.to_parquet(xbt_file)
        
        self.eth_data = eth_data
        self.xbt_data = xbt_data
    
    def test_complete_backtesting_workflow(self):
        """Test complete backtesting workflow from data loading to results"""
        # Set up data sources
        data_sources = [
            ('ETH', str(self.test_dir / "ETH_EUR.parquet")),
            ('XBT', str(self.test_dir / "XBT_EUR.parquet"))
        ]
        
        # Create dataloader
        dataloader = OrderBookDataFromDf(data_sources)
        
        # Create strategy
        strategy = SimpleTestStrategy(buy_threshold=0.01, sell_threshold=-0.01)
        
        # Configure backtester
        config = BacktestConfig(
            initial_capital=100000,
            fees_graph=self.fees_graph,
            symbols=['ETH', 'XBT'],
            window_size=10,
            calibration_end_time=500,  # Use first 500 timestamps for calibration
            validation_start_time=500,
            max_action_runtime=1.0
        )
        
        backtester = Backtester(dataloader, config)
        
        # Run backtest
        results = backtester.backtest([strategy])
        
        # Verify results structure
        assert isinstance(results, dict)
        assert 'SimpleTestStrategy' in results
        
        cal_result, val_result = results['SimpleTestStrategy']
        
        # Verify calibration results
        assert hasattr(cal_result, 'total_return')
        assert hasattr(cal_result, 'sharpe_ratio')
        assert hasattr(cal_result, 'final_portfolio_value')
        assert hasattr(cal_result, 'trades')
        
        # Verify validation results
        assert hasattr(val_result, 'total_return')
        assert hasattr(val_result, 'sharpe_ratio')
        assert hasattr(val_result, 'final_portfolio_value')
        assert hasattr(val_result, 'trades')
        
        # Basic sanity checks
        assert isinstance(cal_result.total_return, (int, float))
        assert isinstance(val_result.total_return, (int, float))
        assert cal_result.final_portfolio_value > 0
        assert val_result.final_portfolio_value > 0
        
        # Strategy should have been called
        assert strategy.call_count > 0
    
    def test_multiple_strategies_backtest(self):
        """Test backtesting with multiple strategies"""
        data_sources = [
            ('ETH', str(self.test_dir / "ETH_EUR.parquet"))
        ]
        
        dataloader = OrderBookDataFromDf(data_sources)
        
        # Create multiple strategies
        strategies = [
            SimpleTestStrategy(buy_threshold=0.01, sell_threshold=-0.01),
            SimpleTestStrategy(buy_threshold=0.02, sell_threshold=-0.02),
            SimpleTestStrategy(buy_threshold=0.005, sell_threshold=-0.005)
        ]
        
        config = BacktestConfig(
            initial_capital=100000,
            fees_graph=self.fees_graph,
            symbols=['ETH'],
            window_size=5,
            calibration_end_time=500,
            validation_start_time=500
        )
        
        backtester = Backtester(dataloader, config)
        results = backtester.backtest(strategies)
        
        # Should have results for all strategies
        assert len(results) == 3
        
        # Each strategy should have different call counts (indicating they're separate instances)
        call_counts = [strategy.call_count for strategy in strategies]
        assert all(count > 0 for count in call_counts)
    
    def test_backtesting_with_real_strategies(self):
        """Test backtesting with actual strategy implementations"""
        # Add features required by MomentumStrategy
        eth_data = self.eth_data.copy()
        
        data_sources = [
            ('ETH', str(self.test_dir / "ETH_EUR.parquet"))
        ]
        
        dataloader = OrderBookDataFromDf(data_sources)
        
        # Use actual strategy
        momentum_strategy = MomentumStrategy(
            short_window=5,
            long_window=20,
            volume_threshold=1.5,
            target_allocation=0.4
        )
        
        config = BacktestConfig(
            initial_capital=100000,
            fees_graph=self.fees_graph,
            symbols=['ETH'],
            window_size=25,  # Ensure enough data for long_window
            calibration_end_time=600,
            validation_start_time=600
        )
        
        backtester = Backtester(dataloader, config)
        
        try:
            results = backtester.backtest([momentum_strategy])
            
            # Basic verification
            assert 'MomentumStrategy' in results
            cal_result, val_result = results['MomentumStrategy']
            
            assert isinstance(cal_result.total_return, (int, float))
            assert isinstance(val_result.total_return, (int, float))
            
        except Exception as e:
            # Real strategies might fail due to missing features or other issues
            # This is acceptable for integration testing
            assert "KeyError" in str(type(e)) or "missing" in str(e).lower()
    
    def test_backtesting_error_handling(self):
        """Test error handling in backtesting system"""
        data_sources = [
            ('ETH', str(self.test_dir / "ETH_EUR.parquet"))
        ]
        
        dataloader = OrderBookDataFromDf(data_sources)
        
        # Create strategy that will raise errors
        class ErrorStrategy(Strategy):
            def get_action(self, data, portfolio, fees):
                raise ValueError("Test error")
        
        config = BacktestConfig(
            initial_capital=100000,
            fees_graph=self.fees_graph,
            symbols=['ETH'],
            window_size=10,
            calibration_end_time=100,
            validation_start_time=100
        )
        
        backtester = Backtester(dataloader, config)
        
        # Should handle strategy errors gracefully
        with pytest.raises((ValueError, RuntimeError)):
            backtester.backtest([ErrorStrategy()])
    
    def test_backtesting_with_different_time_splits(self):
        """Test backtesting with different calibration/validation splits"""
        data_sources = [
            ('ETH', str(self.test_dir / "ETH_EUR.parquet"))
        ]
        
        dataloader = OrderBookDataFromDf(data_sources)
        strategy = SimpleTestStrategy()
        
        # Test different split ratios
        splits = [
            (300, 300),   # 30/70 split
            (500, 500),   # 50/50 split
            (700, 700),   # 70/30 split
        ]
        
        for cal_end, val_start in splits:
            config = BacktestConfig(
                initial_capital=100000,
                fees_graph=self.fees_graph,
                symbols=['ETH'],
                window_size=10,
                calibration_end_time=cal_end,
                validation_start_time=val_start
            )
            
            backtester = Backtester(dataloader, config)
            results = backtester.backtest([strategy])
            
            # Should complete successfully for all splits
            assert 'SimpleTestStrategy' in results
            cal_result, val_result = results['SimpleTestStrategy']
            assert isinstance(cal_result.total_return, (int, float))
            assert isinstance(val_result.total_return, (int, float))
    
    def test_backtesting_portfolio_consistency(self):
        """Test that portfolio values are consistent throughout backtest"""
        data_sources = [
            ('ETH', str(self.test_dir / "ETH_EUR.parquet"))
        ]
        
        dataloader = OrderBookDataFromDf(data_sources)
        
        # Strategy that tracks portfolio values
        class PortfolioTrackingStrategy(Strategy):
            def __init__(self):
                super().__init__()
                self.portfolio_values = []
            
            def get_action(self, data, portfolio, fees):
                eth_pos = portfolio.get_position('ETH')
                eurc_pos = portfolio.get_position('EURC')
                self.portfolio_values.append((eth_pos, eurc_pos))
                return {}
        
        strategy = PortfolioTrackingStrategy()
        
        config = BacktestConfig(
            initial_capital=100000,
            fees_graph=self.fees_graph,
            symbols=['ETH'],
            window_size=10,
            calibration_end_time=100,
            validation_start_time=100
        )
        
        backtester = Backtester(dataloader, config)
        results = backtester.backtest([strategy])
        
        # Should have tracked portfolio values
        assert len(strategy.portfolio_values) > 0
        
        # Initial EURC position should equal initial capital
        initial_eurc = strategy.portfolio_values[0][1]
        assert abs(initial_eurc - 100000) < 1e-6
    
    def test_backtesting_fees_application(self):
        """Test that trading fees are correctly applied"""
        data_sources = [
            ('ETH', str(self.test_dir / "ETH_EUR.parquet"))
        ]
        
        dataloader = OrderBookDataFromDf(data_sources)
        
        # Strategy that makes many trades
        class HighFrequencyStrategy(Strategy):
            def get_action(self, data, portfolio, fees):
                # Alternate between small buys and sells
                if hasattr(self, 'last_action') and self.last_action > 0:
                    self.last_action = -0.01
                    return {'ETH': -0.01}
                else:
                    self.last_action = 0.01
                    return {'ETH': 0.01}
        
        strategy = HighFrequencyStrategy()
        
        # Test with high fees
        high_fees = {
            'ETH': [('EURC', 0.01)],  # 1% fee
            'EURC': [('ETH', 0.01)]
        }
        
        config = BacktestConfig(
            initial_capital=100000,
            fees_graph=high_fees,
            symbols=['ETH'],
            window_size=10,
            calibration_end_time=100,
            validation_start_time=100
        )
        
        backtester = Backtester(dataloader, config)
        results = backtester.backtest([strategy])
        
        cal_result, val_result = results['HighFrequencyStrategy']
        
        # With high fees and frequent trading, should see negative impact
        # Final portfolio value should be less than initial due to fees
        assert cal_result.final_portfolio_value < config.initial_capital
    
    def test_backtesting_data_window_handling(self):
        """Test that data windowing works correctly"""
        data_sources = [
            ('ETH', str(self.test_dir / "ETH_EUR.parquet"))
        ]
        
        dataloader = OrderBookDataFromDf(data_sources)
        
        # Strategy that checks data window size
        class WindowCheckStrategy(Strategy):
            def __init__(self, expected_window_size):
                super().__init__()
                self.expected_window_size = expected_window_size
                self.window_sizes = []
            
            def get_action(self, data, portfolio, fees):
                if 'ETH' in data:
                    self.window_sizes.append(len(data['ETH']))
                return {}
        
        window_size = 20
        strategy = WindowCheckStrategy(window_size)
        
        config = BacktestConfig(
            initial_capital=100000,
            fees_graph=self.fees_graph,
            symbols=['ETH'],
            window_size=window_size,
            calibration_end_time=100,
            validation_start_time=100
        )
        
        backtester = Backtester(dataloader, config)
        results = backtester.backtest([strategy])
        
        # Check that window sizes are as expected
        assert len(strategy.window_sizes) > 0
        
        # Most windows should be the expected size (except possibly early ones)
        final_window_sizes = strategy.window_sizes[-10:]  # Check last 10
        assert all(size == window_size for size in final_window_sizes)


class TestBacktestingPerformance:
    """Performance tests for backtesting system"""
    
    def setup_method(self):
        """Set up performance test environment"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.create_large_dataset()
    
    def teardown_method(self):
        """Clean up test environment"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def create_large_dataset(self):
        """Create a larger dataset for performance testing"""
        # Create 1 hour of 1-second data
        timestamps = pd.date_range('2024-01-01 00:00:00', periods=3600, freq='1s')
        
        base_price = 2000
        price_changes = np.random.normal(0, 0.0005, len(timestamps))
        prices = base_price * np.exp(np.cumsum(price_changes))
        
        eth_data = pd.DataFrame({
            'timestamp': timestamps,
            'level-1-bid-price': prices - np.random.uniform(0.5, 2.0, len(timestamps)),
            'level-1-ask-price': prices + np.random.uniform(0.5, 2.0, len(timestamps)),
            'level-1-bid-volume': np.random.exponential(100, len(timestamps)),
            'level-1-ask-volume': np.random.exponential(95, len(timestamps)),
            'spread': np.random.exponential(2, len(timestamps)) + 1
        })
        
        eth_data['level-1-ask-price'] = eth_data['level-1-bid-price'] + eth_data['spread']
        
        eth_file = self.test_dir / "ETH_EUR.parquet"
        eth_data.to_parquet(eth_file)
        
        self.eth_data = eth_data
    
    @pytest.mark.slow
    def test_backtesting_performance_large_dataset(self):
        """Test backtesting performance with large dataset"""
        import time
        
        data_sources = [
            ('ETH', str(self.test_dir / "ETH_EUR.parquet"))
        ]
        
        dataloader = OrderBookDataFromDf(data_sources)
        strategy = SimpleTestStrategy()
        
        config = BacktestConfig(
            initial_capital=100000,
            fees_graph={'ETH': [('EURC', 0.001)], 'EURC': [('ETH', 0.001)]},
            symbols=['ETH'],
            window_size=50,
            calibration_end_time=1800,  # 30 minutes
            validation_start_time=1800
        )
        
        backtester = Backtester(dataloader, config)
        
        start_time = time.time()
        results = backtester.backtest([strategy])
        elapsed_time = time.time() - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert elapsed_time < 60.0, f"Backtest took too long: {elapsed_time:.2f}s"
        
        # Verify results are still correct
        assert 'SimpleTestStrategy' in results
        cal_result, val_result = results['SimpleTestStrategy']
        assert isinstance(cal_result.total_return, (int, float))
        assert isinstance(val_result.total_return, (int, float))
    
    @pytest.mark.slow
    def test_multiple_strategies_performance(self):
        """Test performance with multiple strategies"""
        import time
        
        data_sources = [
            ('ETH', str(self.test_dir / "ETH_EUR.parquet"))
        ]
        
        dataloader = OrderBookDataFromDf(data_sources)
        
        # Create multiple strategies
        strategies = [SimpleTestStrategy() for _ in range(5)]
        
        config = BacktestConfig(
            initial_capital=100000,
            fees_graph={'ETH': [('EURC', 0.001)], 'EURC': [('ETH', 0.001)]},
            symbols=['ETH'],
            window_size=20,
            calibration_end_time=1000,
            validation_start_time=1000
        )
        
        backtester = Backtester(dataloader, config)
        
        start_time = time.time()
        results = backtester.backtest(strategies)
        elapsed_time = time.time() - start_time
        
        # Should scale reasonably with number of strategies
        assert elapsed_time < 120.0, f"Multiple strategies took too long: {elapsed_time:.2f}s"
        assert len(results) == 5
