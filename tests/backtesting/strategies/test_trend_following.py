"""
Tests for Trend Following Strategies

These tests verify the behavior and logic of the TrendFollowing strategy classes,
including TFCumulativeReturnStrategy, TFSharpeRatioStrategy, and TFImbalanceStrategy.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the strategies and required types
from src.strategies.trend_following import (
    TFCumulativeReturnStrategy,
    TFSharpeRatioStrategy,
    TFImbalanceStrategy
)
from src.backtesting.types import MarketData, FeesGraph
from src.backtesting.portfolio import Portfolio


class TestTFCumulativeReturnStrategy:
    """Test cases for TFCumulativeReturnStrategy"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.strategy = TFCumulativeReturnStrategy(window_size=5)
        
        # Create sample market data with cumulative return feature
        self.sample_eth_data = pd.DataFrame({
            'return-vs-volatility-5-ms': [
                [0.1], [0.05], [-0.02], [0.15], [-0.08],
                [0.12], [0.03], [-0.05], [0.08], [0.0]
            ]
        })
        
        self.market_data = {'ETH': self.sample_eth_data}
        
        # Create mock portfolio and fees
        self.mock_portfolio = Mock(spec=Portfolio)
        self.fees_graph = {'ETH': [('EURC', 0.001)]}
    
    def test_strategy_initialization(self):
        """Test strategy initialization with custom window size"""
        strategy = TFCumulativeReturnStrategy(window_size=10)
        assert strategy.window_size == 10
    
    def test_strategy_initialization_default(self):
        """Test strategy initialization with default window size"""
        strategy = TFCumulativeReturnStrategy()
        assert strategy.window_size == 5
    
    def test_get_action_positive_return(self):
        """Test action generation with positive cumulative return"""
        # Set up data with positive return
        positive_data = pd.DataFrame({
            'return-vs-volatility-5-ms': [[0.15]]
        })
        market_data = {'ETH': positive_data}
        
        action = self.strategy.get_action(
            market_data,
            self.mock_portfolio,
            self.fees_graph
        )
        
        assert action == {'ETH': 0.1 * 0.15}
        assert action['ETH'] > 0  # Should be a buy signal
    
    def test_get_action_negative_return(self):
        """Test action generation with negative cumulative return"""
        # Set up data with negative return
        negative_data = pd.DataFrame({
            'return-vs-volatility-5-ms': [[-0.12]]
        })
        market_data = {'ETH': negative_data}
        
        action = self.strategy.get_action(
            market_data,
            self.mock_portfolio,
            self.fees_graph
        )
        
        assert action == {'ETH': 0.1 * -0.12}
        assert action['ETH'] < 0  # Should be a sell signal
    
    def test_get_action_zero_return(self):
        """Test action generation with zero cumulative return"""
        # Set up data with zero return
        zero_data = pd.DataFrame({
            'return-vs-volatility-5-ms': [[0.0]]
        })
        market_data = {'ETH': zero_data}
        
        action = self.strategy.get_action(
            market_data,
            self.mock_portfolio,
            self.fees_graph
        )
        
        assert action == {'ETH': 0.0}
    
    def test_get_action_with_different_window_size(self):
        """Test action generation with different window size"""
        strategy = TFCumulativeReturnStrategy(window_size=10)
        
        data = pd.DataFrame({
            'return-vs-volatility-10-ms': [[0.08]]
        })
        market_data = {'ETH': data}
        
        action = strategy.get_action(
            market_data,
            self.mock_portfolio,
            self.fees_graph
        )
        
        assert action == {'ETH': 0.1 * 0.08}


class TestTFSharpeRatioStrategy:
    """Test cases for TFSharpeRatioStrategy"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.strategy = TFSharpeRatioStrategy(window_size=5)
        
        # Create sample market data with Sharpe ratio feature
        self.sample_eth_data = pd.DataFrame({
            'sharpe-ratio-quantile-calibrated-5-ms': [1.2, 0.8, -0.5, 1.8, 0.3]
        })
        
        self.market_data = {'ETH': self.sample_eth_data}
        
        # Create mock portfolio and fees
        self.mock_portfolio = Mock(spec=Portfolio)
        self.fees_graph = {'ETH': [('EURC', 0.001)]}
    
    def test_strategy_initialization(self):
        """Test strategy initialization with custom window size"""
        strategy = TFSharpeRatioStrategy(window_size=15)
        assert strategy.window_size == 15
    
    def test_strategy_initialization_default(self):
        """Test strategy initialization with default window size"""
        strategy = TFSharpeRatioStrategy()
        assert strategy.window_size == 5
    
    def test_get_action_positive_sharpe(self):
        """Test action generation with positive Sharpe ratio"""
        # Set up data with positive Sharpe ratio
        positive_data = pd.DataFrame({
            'sharpe-ratio-quantile-calibrated-5-ms': [1.5]
        })
        market_data = {'ETH': positive_data}
        
        action = self.strategy.get_action(
            market_data,
            self.mock_portfolio,
            self.fees_graph
        )
        
        assert action == {'ETH': 0.1 * 1.5}
        assert action['ETH'] > 0  # Should be a buy signal
    
    def test_get_action_negative_sharpe(self):
        """Test action generation with negative Sharpe ratio"""
        # Set up data with negative Sharpe ratio
        negative_data = pd.DataFrame({
            'sharpe-ratio-quantile-calibrated-5-ms': [-0.8]
        })
        market_data = {'ETH': negative_data}
        
        action = self.strategy.get_action(
            market_data,
            self.mock_portfolio,
            self.fees_graph
        )
        
        assert action == {'ETH': 0.1 * -0.8}
        assert action['ETH'] < 0  # Should be a sell signal
    
    def test_get_action_zero_sharpe(self):
        """Test action generation with zero Sharpe ratio"""
        # Set up data with zero Sharpe ratio
        zero_data = pd.DataFrame({
            'sharpe-ratio-quantile-calibrated-5-ms': [0.0]
        })
        market_data = {'ETH': zero_data}
        
        action = self.strategy.get_action(
            market_data,
            self.mock_portfolio,
            self.fees_graph
        )
        
        assert action == {'ETH': 0.0}
    
    def test_get_action_with_different_window_size(self):
        """Test action generation with different window size"""
        strategy = TFSharpeRatioStrategy(window_size=20)
        
        data = pd.DataFrame({
            'sharpe-ratio-quantile-calibrated-20-ms': [0.9]
        })
        market_data = {'ETH': data}
        
        action = strategy.get_action(
            market_data,
            self.mock_portfolio,
            self.fees_graph
        )
        
        assert action == {'ETH': 0.1 * 0.9}


class TestTFImbalanceStrategy:
    """Test cases for TFImbalanceStrategy"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.strategy = TFImbalanceStrategy(levels=5)
        
        # Create sample market data with bid-ask imbalance feature
        self.sample_eth_data = pd.DataFrame({
            'bid-ask-imbalance-5-levels': [0.8, 0.6, 0.2, 0.9, 0.5]
        })
        
        self.market_data = {'ETH': self.sample_eth_data}
        
        # Create mock portfolio and fees
        self.mock_portfolio = Mock(spec=Portfolio)
        self.fees_graph = {'ETH': [('EURC', 0.001)]}
    
    def test_strategy_initialization(self):
        """Test strategy initialization with custom levels"""
        strategy = TFImbalanceStrategy(levels=10)
        assert strategy.levels == 10
    
    def test_strategy_initialization_default(self):
        """Test strategy initialization with default levels"""
        strategy = TFImbalanceStrategy()
        assert strategy.levels == 5
    
    def test_get_action_high_imbalance_buy(self):
        """Test action generation with high imbalance (buy signal)"""
        # Set up data with imbalance > 0.7 (buy signal)
        high_imbalance_data = pd.DataFrame({
            'bid-ask-imbalance-5-levels': [0.8]
        })
        market_data = {'ETH': high_imbalance_data}
        
        action = self.strategy.get_action(
            market_data,
            self.mock_portfolio,
            self.fees_graph
        )
        
        assert action == {'ETH': 0.1}
    
    def test_get_action_low_imbalance_sell(self):
        """Test action generation with low imbalance (sell signal)"""
        # Set up data with imbalance < 0.3 (sell signal)
        low_imbalance_data = pd.DataFrame({
            'bid-ask-imbalance-5-levels': [0.2]
        })
        market_data = {'ETH': low_imbalance_data}
        
        action = self.strategy.get_action(
            market_data,
            self.mock_portfolio,
            self.fees_graph
        )
        
        assert action == {'ETH': -0.1}
    
    def test_get_action_neutral_imbalance_hold(self):
        """Test action generation with neutral imbalance (hold signal)"""
        # Set up data with imbalance between 0.3 and 0.7 (hold)
        neutral_imbalance_data = pd.DataFrame({
            'bid-ask-imbalance-5-levels': [0.5]
        })
        market_data = {'ETH': neutral_imbalance_data}
        
        action = self.strategy.get_action(
            market_data,
            self.mock_portfolio,
            self.fees_graph
        )
        
        assert action == {'ETH': 0.0}
    
    def test_get_action_boundary_values(self):
        """Test action generation at boundary values"""
        # Test exactly at 0.7 threshold
        boundary_high_data = pd.DataFrame({
            'bid-ask-imbalance-5-levels': [0.7]
        })
        market_data = {'ETH': boundary_high_data}
        
        action = self.strategy.get_action(
            market_data,
            self.mock_portfolio,
            self.fees_graph
        )
        
        assert action == {'ETH': 0.0}  # Should be hold (not > 0.7)
        
        # Test exactly at 0.3 threshold
        boundary_low_data = pd.DataFrame({
            'bid-ask-imbalance-5-levels': [0.3]
        })
        market_data = {'ETH': boundary_low_data}
        
        action = self.strategy.get_action(
            market_data,
            self.mock_portfolio,
            self.fees_graph
        )
        
        assert action == {'ETH': 0.0}  # Should be hold (not < 0.3)
    
    def test_get_action_with_different_levels(self):
        """Test action generation with different levels parameter"""
        strategy = TFImbalanceStrategy(levels=10)
        
        data = pd.DataFrame({
            'bid-ask-imbalance-10-levels': [0.8]
        })
        market_data = {'ETH': data}
        
        action = strategy.get_action(
            market_data,
            self.mock_portfolio,
            self.fees_graph
        )
        
        assert action == {'ETH': 0.1}


class TestTrendFollowingStrategiesIntegration:
    """Integration tests for all trend following strategies"""
    
    def test_all_strategies_with_realistic_data(self):
        """Test all strategies with realistic market data"""
        # Create realistic market data
        eth_data = pd.DataFrame({
            'return-vs-volatility-5-ms': [[0.08], [0.12], [-0.05], [0.03], [0.15]],
            'sharpe-ratio-quantile-calibrated-5-ms': [1.2, 1.8, -0.3, 0.7, 2.1],
            'bid-ask-imbalance-5-levels': [0.75, 0.65, 0.25, 0.85, 0.45]
        })
        
        market_data = {'ETH': eth_data}
        mock_portfolio = Mock(spec=Portfolio)
        fees_graph = {'ETH': [('EURC', 0.001)]}
        
        # Test TFCumulativeReturnStrategy
        cumulative_strategy = TFCumulativeReturnStrategy(window_size=5)
        action1 = cumulative_strategy.get_action(market_data, mock_portfolio, fees_graph)
        
        assert isinstance(action1, dict)
        assert 'ETH' in action1
        assert action1['ETH'] == 0.1 * 0.15  # Last return value
        
        # Test TFSharpeRatioStrategy
        sharpe_strategy = TFSharpeRatioStrategy(window_size=5)
        action2 = sharpe_strategy.get_action(market_data, mock_portfolio, fees_graph)
        
        assert isinstance(action2, dict)
        assert 'ETH' in action2
        assert action2['ETH'] == 0.1 * 2.1  # Last Sharpe ratio value
        
        # Test TFImbalanceStrategy
        imbalance_strategy = TFImbalanceStrategy(levels=5)
        action3 = imbalance_strategy.get_action(market_data, mock_portfolio, fees_graph)
        
        assert isinstance(action3, dict)
        assert 'ETH' in action3
        assert action3['ETH'] == 0.0  # 0.45 is neutral (between 0.3 and 0.7)
    
    def test_strategies_with_missing_data(self):
        """Test strategies behavior with missing or invalid data"""
        # Test with missing columns
        incomplete_data = pd.DataFrame({
            'some-other-feature': [1, 2, 3, 4, 5]
        })
        
        market_data = {'ETH': incomplete_data}
        mock_portfolio = Mock(spec=Portfolio)
        fees_graph = {'ETH': [('EURC', 0.001)]}
        
        cumulative_strategy = TFCumulativeReturnStrategy(window_size=5)
        
        # Should raise KeyError or handle gracefully
        with pytest.raises(KeyError):
            cumulative_strategy.get_action(market_data, mock_portfolio, fees_graph)
    
    def test_strategies_with_empty_data(self):
        """Test strategies behavior with empty data"""
        empty_data = pd.DataFrame()
        market_data = {'ETH': empty_data}
        mock_portfolio = Mock(spec=Portfolio)
        fees_graph = {'ETH': [('EURC', 0.001)]}
        
        cumulative_strategy = TFCumulativeReturnStrategy(window_size=5)
        
        # Should raise IndexError or handle gracefully
        with pytest.raises(IndexError):
            cumulative_strategy.get_action(market_data, mock_portfolio, fees_graph)
    
    def test_strategies_consistency(self):
        """Test that strategies produce consistent results with same input"""
        eth_data = pd.DataFrame({
            'return-vs-volatility-5-ms': [[0.1]],
            'sharpe-ratio-quantile-calibrated-5-ms': [1.5],
            'bid-ask-imbalance-5-levels': [0.8]
        })
        
        market_data = {'ETH': eth_data}
        mock_portfolio = Mock(spec=Portfolio)
        fees_graph = {'ETH': [('EURC', 0.001)]}
        
        # Test multiple calls with same input
        cumulative_strategy = TFCumulativeReturnStrategy(window_size=5)
        
        action1 = cumulative_strategy.get_action(market_data, mock_portfolio, fees_graph)
        action2 = cumulative_strategy.get_action(market_data, mock_portfolio, fees_graph)
        action3 = cumulative_strategy.get_action(market_data, mock_portfolio, fees_graph)
        
        # Should be identical
        assert action1 == action2 == action3
    
    def test_strategies_mathematical_correctness(self):
        """Test mathematical correctness of strategy calculations"""
        # Test that the calculation is exactly 0.1 * feature_value
        test_values = [0.05, -0.1, 0.0, 0.2, -0.15]
        
        for value in test_values:
            # Test cumulative return strategy
            eth_data = pd.DataFrame({
                'return-vs-volatility-5-ms': [[value]]
            })
            market_data = {'ETH': eth_data}
            mock_portfolio = Mock(spec=Portfolio)
            fees_graph = {'ETH': [('EURC', 0.001)]}
            
            strategy = TFCumulativeReturnStrategy(window_size=5)
            action = strategy.get_action(market_data, mock_portfolio, fees_graph)
            
            expected = 0.1 * value
            assert abs(action['ETH'] - expected) < 1e-10  # Account for floating point precision
            
            # Test Sharpe ratio strategy
            eth_data = pd.DataFrame({
                'sharpe-ratio-quantile-calibrated-5-ms': [value]
            })
            market_data = {'ETH': eth_data}
            
            sharpe_strategy = TFSharpeRatioStrategy(window_size=5)
            action = sharpe_strategy.get_action(market_data, mock_portfolio, fees_graph)
            
            expected = 0.1 * value
            assert abs(action['ETH'] - expected) < 1e-10
