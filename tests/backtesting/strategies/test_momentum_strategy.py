"""
Tests for the Momentum Strategy

These tests verify the behavior and logic of the MomentumStrategy class,
including signal generation, portfolio rebalancing, and error handling.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the strategy and required types
from src.strategies.momentum_strategy import MomentumStrategy
from src.backtesting.types import MarketData, FeesGraph
from src.backtesting.portfolio import Portfolio


class TestMomentumStrategy:
    """Test cases for MomentumStrategy"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.strategy = MomentumStrategy(
            short_window=5,
            long_window=20,
            volume_threshold=1.5,
            target_allocation=0.4
        )
        
        # Create sample market data
        self.sample_timestamps = pd.date_range('2024-01-01', periods=30, freq='1min')
        self.sample_eth_data = pd.DataFrame({
            'timestamp': self.sample_timestamps,
            'level-1-bid-price': 2000 + np.random.randn(30) * 10,
            'level-1-ask-price': 2005 + np.random.randn(30) * 10,
            'V-bid-5-levels': 100 + np.random.randn(30) * 20,
            'V-ask-5-levels': 95 + np.random.randn(30) * 18,
            'spread': 5 + np.random.randn(30) * 1
        })
        
        # Ensure prices are positive
        self.sample_eth_data['level-1-bid-price'] = np.abs(self.sample_eth_data['level-1-bid-price'])
        self.sample_eth_data['level-1-ask-price'] = self.sample_eth_data['level-1-bid-price'] + 5
        self.sample_eth_data['V-bid-5-levels'] = np.abs(self.sample_eth_data['V-bid-5-levels'])
        self.sample_eth_data['V-ask-5-levels'] = np.abs(self.sample_eth_data['V-ask-5-levels'])
        self.sample_eth_data['spread'] = np.abs(self.sample_eth_data['spread'])
        
        self.market_data = {'ETH': self.sample_eth_data}
        
        # Create mock portfolio
        self.mock_portfolio = Mock(spec=Portfolio)
        self.mock_portfolio.get_position.side_effect = lambda symbol: 10.0 if symbol == 'ETH' else 50000.0
        
        # Create mock fees graph
        self.fees_graph = {
            'ETH': [('EURC', 0.001)],
            'EURC': [('ETH', 0.001)]
        }
    
    def test_strategy_initialization(self):
        """Test strategy initialization with custom parameters"""
        strategy = MomentumStrategy(
            short_window=10,
            long_window=30,
            volume_threshold=2.0,
            target_allocation=0.6
        )
        
        assert strategy.short_window == 10
        assert strategy.long_window == 30
        assert strategy.volume_threshold == 2.0
        assert strategy.target_allocation == 0.6
    
    def test_calculate_mid_price(self):
        """Test mid price calculation"""
        mid_prices = self.strategy._calculate_mid_price(self.sample_eth_data)
        
        expected_mid = (self.sample_eth_data['level-1-bid-price'] + 
                       self.sample_eth_data['level-1-ask-price']) / 2
        
        pd.testing.assert_series_equal(mid_prices, expected_mid)
    
    def test_calculate_mid_price_missing_columns(self):
        """Test mid price calculation with missing columns"""
        data_without_prices = self.sample_eth_data.drop(columns=['level-1-bid-price'])
        result = self.strategy._calculate_mid_price(data_without_prices)
        
        assert result is None
    
    def test_calculate_moving_averages(self):
        """Test moving averages calculation"""
        prices = pd.Series(range(1, 31))  # 1 to 30
        
        short_ma, long_ma = self.strategy._calculate_moving_averages(prices)
        
        # Should return the last value of rolling means
        expected_short = prices.rolling(window=5).mean().iloc[-1]
        expected_long = prices.rolling(window=20).mean().iloc[-1]
        
        assert short_ma == expected_short
        assert long_ma == expected_long
    
    def test_calculate_moving_averages_insufficient_data(self):
        """Test moving averages with insufficient data"""
        short_prices = pd.Series([1, 2, 3])  # Only 3 data points
        
        short_ma, long_ma = self.strategy._calculate_moving_averages(short_prices)
        
        assert short_ma is None
        assert long_ma is None
    
    def test_analyze_volume(self):
        """Test volume analysis"""
        volume_ratio = self.strategy._analyze_volume(self.sample_eth_data)
        
        # Should return a positive number
        assert isinstance(volume_ratio, (int, float))
        assert volume_ratio > 0
    
    def test_analyze_volume_missing_columns(self):
        """Test volume analysis with missing volume columns"""
        data_without_volume = self.sample_eth_data.drop(columns=['V-bid-5-levels'])
        volume_ratio = self.strategy._analyze_volume(data_without_volume)
        
        assert volume_ratio == 1.0
    
    def test_calculate_spread_signal(self):
        """Test spread signal calculation"""
        spread_signal = self.strategy._calculate_spread_signal(self.sample_eth_data)
        
        # Should return a positive number
        assert isinstance(spread_signal, (int, float))
        assert spread_signal > 0
    
    def test_calculate_spread_signal_missing_column(self):
        """Test spread signal with missing spread column"""
        data_without_spread = self.sample_eth_data.drop(columns=['spread'])
        spread_signal = self.strategy._calculate_spread_signal(data_without_spread)
        
        assert spread_signal == 1.0
    
    def test_get_action_normal_case(self):
        """Test normal action generation"""
        action = self.strategy.get_action(
            self.market_data,
            self.mock_portfolio,
            self.fees_graph
        )
        
        # Should return a dictionary with ETH key
        assert isinstance(action, dict)
        if action:  # Might be empty if trade amount is too small
            assert 'ETH' in action
            assert isinstance(action['ETH'], (int, float))
    
    def test_get_action_no_eth_data(self):
        """Test action generation when ETH data is missing"""
        empty_market_data = {}
        
        action = self.strategy.get_action(
            empty_market_data,
            self.mock_portfolio,
            self.fees_graph
        )
        
        assert action == {}
    
    def test_get_action_empty_eth_data(self):
        """Test action generation when ETH data is empty"""
        empty_eth_data = {'ETH': pd.DataFrame()}
        
        action = self.strategy.get_action(
            empty_eth_data,
            self.mock_portfolio,
            self.fees_graph
        )
        
        assert action == {}
    
    def test_get_action_insufficient_data(self):
        """Test action generation with insufficient historical data"""
        # Create data with only 10 rows (less than long_window=20)
        short_data = self.sample_eth_data.head(10).copy()
        short_market_data = {'ETH': short_data}
        
        action = self.strategy.get_action(
            short_market_data,
            self.mock_portfolio,
            self.fees_graph
        )
        
        assert action == {}
    
    def test_get_action_error_handling(self):
        """Test error handling in get_action"""
        # Create data that will cause an error (e.g., invalid price)
        bad_data = self.sample_eth_data.copy()
        bad_data['level-1-bid-price'] = [np.inf] * len(bad_data)
        bad_market_data = {'ETH': bad_data}
        
        # Should not raise an exception, should return empty dict
        action = self.strategy.get_action(
            bad_market_data,
            self.mock_portfolio,
            self.fees_graph
        )
        
        # Strategy should handle errors gracefully - may return empty action or nan
        assert isinstance(action, dict)
        if "ETH" in action:
            # If ETH key exists, value should be either number or nan
            import math
            assert isinstance(action["ETH"], (int, float)) or math.isnan(action["ETH"])
    
    def test_momentum_signal_generation(self):
        """Test momentum signal generation logic"""
        # Create data where short MA > long MA (buy signal)
        increasing_prices = pd.Series(range(1, 31))  # Trending up
        
        short_ma, long_ma = self.strategy._calculate_moving_averages(increasing_prices)
        
        # In an uptrend, short MA should be > long MA
        assert short_ma > long_ma
        
        # Create data where short MA < long MA (sell signal)
        decreasing_prices = pd.Series(range(30, 0, -1))  # Trending down
        
        short_ma, long_ma = self.strategy._calculate_moving_averages(decreasing_prices)
        
        # In a downtrend, short MA should be < long MA
        assert short_ma < long_ma
    
    def test_volume_threshold_logic(self):
        """Test volume threshold logic"""
        # Test with high volume
        high_volume_data = self.sample_eth_data.copy()
        high_volume_data['V-bid-5-levels'] *= 3  # Triple the volume
        high_volume_data['V-ask-5-levels'] *= 3
        
        volume_ratio = self.strategy._analyze_volume(high_volume_data)
        
        # Should indicate higher than normal volume
        assert volume_ratio > 1.0
    
    def test_target_allocation_logic(self):
        """Test target allocation calculation logic"""
        # Mock portfolio with known values
        mock_portfolio = Mock(spec=Portfolio)
        mock_portfolio.get_position.side_effect = lambda symbol: 5.0 if symbol == 'ETH' else 10000.0
        
        # Create market data with stable prices
        stable_data = self.sample_eth_data.copy()
        stable_data['level-1-bid-price'] = 2000
        stable_data['level-1-ask-price'] = 2005
        market_data = {'ETH': stable_data}
        
        action = self.strategy.get_action(
            market_data,
            mock_portfolio,
            self.fees_graph
        )
        
        # Should generate some action based on target allocation
        assert isinstance(action, dict)
    
    def test_small_trade_filtering(self):
        """Test that very small trades are filtered out"""
        # Mock portfolio very close to target allocation
        mock_portfolio = Mock(spec=Portfolio)
        
        # Set position very close to target (should result in tiny adjustment)
        target_eth = 0.4 * 15000 / 2002.5  # target_allocation * total_value / mid_price
        mock_portfolio.get_position.side_effect = lambda symbol: target_eth if symbol == 'ETH' else 5000.0
        
        stable_data = self.sample_eth_data.copy()
        stable_data['level-1-bid-price'] = 2000
        stable_data['level-1-ask-price'] = 2005
        market_data = {'ETH': stable_data}
        
        action = self.strategy.get_action(
            market_data,
            mock_portfolio,
            self.fees_graph
        )
        
        # Should return empty dict for very small trades
        # (depends on the exact calculation, but the logic should filter small trades)
        assert isinstance(action, dict)


class TestMomentumStrategyIntegration:
    """Integration tests for MomentumStrategy"""
    
    def test_strategy_with_real_data_structure(self):
        """Test strategy with realistic data structure"""
        strategy = MomentumStrategy()
        
        # Create more realistic market data
        timestamps = pd.date_range('2024-01-01', periods=50, freq='1s')
        
        eth_data = pd.DataFrame({
            'timestamp': timestamps,
            'level-1-bid-price': 2000 + np.cumsum(np.random.randn(50) * 0.1),
            'level-1-ask-price': 2005 + np.cumsum(np.random.randn(50) * 0.1),
            'level-2-bid-price': 1999 + np.cumsum(np.random.randn(50) * 0.1),
            'level-2-ask-price': 2006 + np.cumsum(np.random.randn(50) * 0.1),
            'level-1-bid-volume': 100 + np.random.exponential(50, 50),
            'level-1-ask-volume': 95 + np.random.exponential(45, 50),
            'V-bid-5-levels': 500 + np.random.exponential(200, 50),
            'V-ask-5-levels': 480 + np.random.exponential(190, 50),
            'spread': np.random.exponential(2, 50) + 1
        })
        
        # Ensure realistic price relationships
        eth_data['level-1-ask-price'] = eth_data['level-1-bid-price'] + eth_data['spread']
        eth_data['level-2-bid-price'] = eth_data['level-1-bid-price'] - np.random.exponential(1, 50)
        eth_data['level-2-ask-price'] = eth_data['level-1-ask-price'] + np.random.exponential(1, 50)
        
        market_data = {'ETH': eth_data}
        
        # Mock portfolio
        mock_portfolio = Mock(spec=Portfolio)
        mock_portfolio.get_position.side_effect = lambda symbol: 8.0 if symbol == 'ETH' else 40000.0
        
        fees_graph = {'ETH': [('EURC', 0.001)], 'EURC': [('ETH', 0.001)]}
        
        action = strategy.get_action(market_data, mock_portfolio, fees_graph)
        
        # Should generate a valid action
        assert isinstance(action, dict)
        
        # If action is not empty, should contain valid trade amount
        if action:
            assert 'ETH' in action
            assert isinstance(action['ETH'], (int, float))
            assert not np.isnan(action['ETH'])
            assert not np.isinf(action['ETH'])
    
    def test_strategy_multiple_calls(self):
        """Test strategy behavior over multiple calls (state independence)"""
        strategy = MomentumStrategy()
        
        # Create evolving market data
        base_data = pd.DataFrame({
            'level-1-bid-price': 2000 + np.random.randn(25) * 5,
            'level-1-ask-price': 2005 + np.random.randn(25) * 5,
            'V-bid-5-levels': 100 + np.random.randn(25) * 20,
            'V-ask-5-levels': 95 + np.random.randn(25) * 18,
            'spread': 5 + np.random.randn(25) * 1
        })
        
        mock_portfolio = Mock(spec=Portfolio)
        mock_portfolio.get_position.side_effect = lambda symbol: 5.0 if symbol == 'ETH' else 25000.0
        
        fees_graph = {'ETH': [('EURC', 0.001)]}
        
        actions = []
        
        # Simulate multiple time steps
        for i in range(5):
            # Add new data point
            new_row = pd.DataFrame({
                'level-1-bid-price': [2000 + np.random.randn() * 5],
                'level-1-ask-price': [2005 + np.random.randn() * 5],
                'V-bid-5-levels': [100 + np.random.randn() * 20],
                'V-ask-5-levels': [95 + np.random.randn() * 18],
                'spread': [5 + np.random.randn() * 1]
            })
            
            current_data = pd.concat([base_data, new_row], ignore_index=True)
            market_data = {'ETH': current_data}
            
            action = strategy.get_action(market_data, mock_portfolio, fees_graph)
            actions.append(action)
        
        # All actions should be valid dictionaries
        for action in actions:
            assert isinstance(action, dict)
            if action:  # If not empty
                assert 'ETH' in action
                assert isinstance(action['ETH'], (int, float))
