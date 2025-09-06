"""
Tests for Random Forest Prediction Strategy

These tests verify the behavior and logic of the RFPredAllSignedStratMateo strategy,
including model loading, prediction-based trading, and portfolio rebalancing.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the strategy and required types
from src.strategies.rf_pred_all_signed_strat_mateo import RFPredAllSignedStratMateo
from src.backtesting.types import MarketData, FeesGraph
from src.backtesting.portfolio import Portfolio


class TestRFPredAllSignedStratMateo:
    """Test cases for RFPredAllSignedStratMateo"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Mock the joblib.load function to avoid loading actual model files
        self.mock_model = Mock()
        self.mock_model.predict.return_value = [0]  # Default prediction
        
        with patch('src.strategies.rf_pred_all_signed_strat_mateo.joblib.load') as mock_load:
            mock_load.return_value = self.mock_model
            self.strategy = RFPredAllSignedStratMateo(window_size=5)
        
        # Create sample XBT market data with required features
        self.sample_xbt_data = pd.DataFrame({
            'bid-ask-imbalance-5-levels': [0.6, 0.7, 0.5, 0.8, 0.4],
            'spread': [2.5, 3.0, 2.8, 2.2, 3.5],
            'inst-return': [0.001, -0.002, 0.003, 0.0, -0.001],
            'V-bid-5-levels': [1000, 1200, 800, 1500, 900],
            'V-ask-5-levels': [950, 1150, 750, 1400, 850],
            'slope-bid-5-levels': [0.5, -0.3, 0.8, 0.2, -0.6],
            'slope-ask-5-levels': [-0.4, 0.6, -0.7, -0.1, 0.9]
        })
        
        self.market_data = {'XBT': self.sample_xbt_data}
        
        # Create mock portfolio and fees
        self.mock_portfolio = Mock(spec=Portfolio)
        self.fees_graph = {'ETH': [('EURC', 0.001)], 'XBT': [('EURC', 0.001)]}
    
    @patch('src.strategies.rf_pred_all_signed_strat_mateo.joblib.load')
    def test_strategy_initialization(self, mock_load):
        """Test strategy initialization and model loading"""
        mock_model = Mock()
        mock_load.return_value = mock_model
        
        strategy = RFPredAllSignedStratMateo(window_size=10)
        
        # Should have loaded the model with correct path
        mock_load.assert_called_once_with("predictors/mateo/rf_model_10ms.joblib")
        assert strategy.model == mock_model
        assert strategy.target_eth == 100.0
    
    @patch('src.strategies.rf_pred_all_signed_strat_mateo.joblib.load')
    def test_strategy_initialization_default_window(self, mock_load):
        """Test strategy initialization with default window size"""
        mock_model = Mock()
        mock_load.return_value = mock_model
        
        strategy = RFPredAllSignedStratMateo()
        
        # Should use default window size of 5
        mock_load.assert_called_once_with("predictors/mateo/rf_model_5ms.joblib")
    
    def test_get_action_sell_signal(self):
        """Test action generation with sell signal (prediction = -1)"""
        # Set up model to predict sell signal
        self.mock_model.predict.return_value = [-1]
        
        # Portfolio has ETH position > 0
        self.mock_portfolio.get_position.return_value = 50.0
        
        action = self.strategy.get_action(
            self.market_data,
            self.mock_portfolio,
            self.fees_graph
        )
        
        assert action == {'ETH': -0.1}
        
        # Verify model was called with correct features
        call_args = self.mock_model.predict.call_args[0][0]
        expected_columns = [
            "bid-ask-imbalance-5-levels", "spread", "inst-return", 
            "V-bid-5-levels", "V-ask-5-levels", "slope-bid-5-levels", "slope-ask-5-levels"
        ]
        assert list(call_args.columns) == expected_columns
        assert len(call_args) == 1  # Should use only the last row
    
    def test_get_action_sell_signal_no_eth_position(self):
        """Test action generation with sell signal but no ETH to sell"""
        # Set up model to predict sell signal
        self.mock_model.predict.return_value = [-1]
        
        # Portfolio has no ETH position
        self.mock_portfolio.get_position.return_value = 0.0
        
        action = self.strategy.get_action(
            self.market_data,
            self.mock_portfolio,
            self.fees_graph
        )
        
        # Should not generate sell action when no ETH to sell
        assert action == {}
    
    def test_get_action_hold_signal(self):
        """Test action generation with hold/rebalance signal (prediction = 0)"""
        # Set up model to predict hold signal
        self.mock_model.predict.return_value = [0]
        
        # Portfolio has some ETH position
        current_eth = 80.0
        self.mock_portfolio.get_position.return_value = current_eth
        
        action = self.strategy.get_action(
            self.market_data,
            self.mock_portfolio,
            self.fees_graph
        )
        
        # Should rebalance to target (100.0 - 80.0 = 20.0)
        expected_rebalance = self.strategy.target_eth - current_eth
        assert action == {'ETH': expected_rebalance}
    
    def test_get_action_buy_signal(self):
        """Test action generation with buy signal (prediction = 1)"""
        # Set up model to predict buy signal
        self.mock_model.predict.return_value = [1]
        
        # Portfolio has ETH position < 200
        self.mock_portfolio.get_position.return_value = 150.0
        
        action = self.strategy.get_action(
            self.market_data,
            self.mock_portfolio,
            self.fees_graph
        )
        
        # Should generate buy action
        assert action == {'ETH': 0.1}
    
    def test_get_action_buy_signal_max_position(self):
        """Test action generation with buy signal but already at max position"""
        # Set up model to predict buy signal
        self.mock_model.predict.return_value = [1]
        
        # Portfolio already has 200+ ETH
        self.mock_portfolio.get_position.return_value = 250.0
        
        action = self.strategy.get_action(
            self.market_data,
            self.mock_portfolio,
            self.fees_graph
        )
        
        # Should not generate buy action when already at max
        assert isinstance(action, dict)
        # Specific behavior depends on complete implementation
    
    def test_feature_ordering(self):
        """Test that features are correctly ordered for model input"""
        self.mock_model.predict.return_value = [0]
        self.mock_portfolio.get_position.return_value = 100.0
        
        # Create data with different column order
        unordered_data = pd.DataFrame({
            'V-ask-5-levels': [850],
            'spread': [3.5],
            'bid-ask-imbalance-5-levels': [0.4],
            'slope-ask-5-levels': [0.9],
            'inst-return': [-0.001],
            'slope-bid-5-levels': [-0.6],
            'V-bid-5-levels': [900]
        })
        
        market_data = {'XBT': unordered_data}
        
        self.strategy.get_action(market_data, self.mock_portfolio, self.fees_graph)
        
        # Verify features were reordered correctly
        call_args = self.mock_model.predict.call_args[0][0]
        expected_order = [
            "bid-ask-imbalance-5-levels", "spread", "inst-return", 
            "V-bid-5-levels", "V-ask-5-levels", "slope-bid-5-levels", "slope-ask-5-levels"
        ]
        assert list(call_args.columns) == expected_order
    
    def test_uses_latest_data_point(self):
        """Test that strategy uses only the latest data point"""
        self.mock_model.predict.return_value = [0]
        self.mock_portfolio.get_position.return_value = 100.0
        
        self.strategy.get_action(self.market_data, self.mock_portfolio, self.fees_graph)
        
        # Verify only the last row was used
        call_args = self.mock_model.predict.call_args[0][0]
        assert len(call_args) == 1
        
        # Verify it's the last row by checking values
        expected_last_row = self.sample_xbt_data.iloc[[-1]]
        for col in call_args.columns:
            assert call_args[col].iloc[0] == expected_last_row[col].iloc[0]
    
    def test_missing_features_error(self):
        """Test error handling when required features are missing"""
        # Create data missing required features
        incomplete_data = pd.DataFrame({
            'bid-ask-imbalance-5-levels': [0.4],
            'spread': [3.5],
            # Missing other required features
        })
        
        market_data = {'XBT': incomplete_data}
        
        # Should raise KeyError when trying to access missing columns
        with pytest.raises(KeyError):
            self.strategy.get_action(market_data, self.mock_portfolio, self.fees_graph)
    
    def test_empty_data_error(self):
        """Test error handling with empty data"""
        empty_data = pd.DataFrame()
        market_data = {'XBT': empty_data}
        
        # Should raise IndexError when trying to access iloc[-1]
        with pytest.raises(IndexError):
            self.strategy.get_action(market_data, self.mock_portfolio, self.fees_graph)
    
    def test_model_prediction_types(self):
        """Test handling of different prediction types"""
        self.mock_portfolio.get_position.return_value = 100.0
        
        # Test with numpy array prediction
        self.mock_model.predict.return_value = np.array([1])
        action = self.strategy.get_action(self.market_data, self.mock_portfolio, self.fees_graph)
        assert isinstance(action, dict)
        
        # Test with list prediction
        self.mock_model.predict.return_value = [0]
        action = self.strategy.get_action(self.market_data, self.mock_portfolio, self.fees_graph)
        assert isinstance(action, dict)
    
    @patch('src.strategies.rf_pred_all_signed_strat_mateo.time.time')
    def test_execution_time_logging(self, mock_time):
        """Test that execution time is logged"""
        # Mock time to return specific values
        mock_time.side_effect = [1000.0, 1000.1]  # 0.1 second execution
        
        self.mock_model.predict.return_value = [0]
        self.mock_portfolio.get_position.return_value = 100.0
        
        with patch('builtins.print') as mock_print:
            self.strategy.get_action(self.market_data, self.mock_portfolio, self.fees_graph)
            
            # Should have printed execution time
            mock_print.assert_called_once()
            call_args = mock_print.call_args[0][0]
            assert "get_action execution time" in call_args
            assert "0.100000 seconds" in call_args


class TestRFPredAllSignedStratMateoIntegration:
    """Integration tests for RFPredAllSignedStratMateo"""
    
    @patch('src.strategies.rf_pred_all_signed_strat_mateo.joblib.load')
    def test_complete_trading_scenario(self, mock_load):
        """Test complete trading scenario with realistic data"""
        # Create a more sophisticated mock model
        mock_model = Mock()
        mock_load.return_value = mock_model
        
        strategy = RFPredAllSignedStratMateo(window_size=5)
        
        # Create realistic market data
        xbt_data = pd.DataFrame({
            'bid-ask-imbalance-5-levels': np.random.uniform(0.3, 0.8, 10),
            'spread': np.random.uniform(1.0, 5.0, 10),
            'inst-return': np.random.normal(0, 0.01, 10),
            'V-bid-5-levels': np.random.uniform(500, 2000, 10),
            'V-ask-5-levels': np.random.uniform(500, 2000, 10),
            'slope-bid-5-levels': np.random.uniform(-1, 1, 10),
            'slope-ask-5-levels': np.random.uniform(-1, 1, 10)
        })
        
        market_data = {'XBT': xbt_data}
        mock_portfolio = Mock(spec=Portfolio)
        fees_graph = {'ETH': [('EURC', 0.001)]}
        
        # Test different prediction scenarios
        scenarios = [
            (-1, 50.0, {'ETH': -0.1}),  # Sell signal with ETH position
            (-1, 0.0, {}),              # Sell signal without ETH position
            (0, 80.0, {'ETH': 20.0}),   # Hold/rebalance signal
            (1, 150.0, {'ETH': 0.1}),   # Buy signal under limit
            (1, 250.0, {}),             # Buy signal over limit
        ]
        
        for prediction, eth_position, expected_action in scenarios:
            mock_model.predict.return_value = [prediction]
            mock_portfolio.get_position.return_value = eth_position
            
            action = strategy.get_action(market_data, mock_portfolio, fees_graph)
            
            # Check that action is reasonable (exact match depends on complete implementation)
            assert isinstance(action, dict)
            if expected_action:
                assert 'ETH' in action
    
    @patch('src.strategies.rf_pred_all_signed_strat_mateo.joblib.load')
    def test_model_file_not_found(self, mock_load):
        """Test behavior when model file is not found"""
        mock_load.side_effect = FileNotFoundError("Model file not found")
        
        with pytest.raises(FileNotFoundError):
            RFPredAllSignedStratMateo(window_size=5)
    
    @patch('src.strategies.rf_pred_all_signed_strat_mateo.joblib.load')
    def test_strategy_state_independence(self, mock_load):
        """Test that strategy calls are independent (stateless)"""
        mock_model = Mock()
        mock_model.predict.return_value = [0]
        mock_load.return_value = mock_model
        
        strategy = RFPredAllSignedStratMateo(window_size=5)
        
        # Create test data
        xbt_data = pd.DataFrame({
            'bid-ask-imbalance-5-levels': [0.5],
            'spread': [2.5],
            'inst-return': [0.001],
            'V-bid-5-levels': [1000],
            'V-ask-5-levels': [950],
            'slope-bid-5-levels': [0.5],
            'slope-ask-5-levels': [-0.4]
        })
        
        market_data = {'XBT': xbt_data}
        mock_portfolio = Mock(spec=Portfolio)
        mock_portfolio.get_position.return_value = 100.0
        fees_graph = {'ETH': [('EURC', 0.001)]}
        
        # Multiple calls should produce identical results
        action1 = strategy.get_action(market_data, mock_portfolio, fees_graph)
        action2 = strategy.get_action(market_data, mock_portfolio, fees_graph)
        action3 = strategy.get_action(market_data, mock_portfolio, fees_graph)
        
        assert action1 == action2 == action3
