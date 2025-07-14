"""
Test order processor functionality for the backtesting module.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock
from src.backtesting.order_processor import OrderProcessor
from src.backtesting.portfolio import Portfolio
from src.backtesting.types import FeesGraph, TimeStep
from tests.backtesting.test_types import create_sample_fees_graph


@pytest.fixture
def sample_fees_graph():
    """Create a sample fees graph for testing."""
    return create_sample_fees_graph(['BTC', 'ETH'])


@pytest.fixture
def sample_portfolio():
    """Create a sample portfolio for testing."""
    portfolio = Portfolio(['BTC', 'ETH'], 10000.0)
    return portfolio


@pytest.fixture
def mock_dataloader():
    """Create a mock dataloader for testing."""
    mock_loader = Mock()
    
    # Sample market data
    sample_data = pd.DataFrame({
        'timestamp': [1641024000.0],
        'level-1-bid-price': [50000.0],
        'level-1-ask-price': [50100.0],
        'level-1-bid-volume': [1.5],
        'level-1-ask-volume': [2.0]
    })
    
    mock_loader.get_coin_at_timestep.return_value = sample_data
    mock_loader.get_time_step_values.return_value = {
        'BTC': np.array([1641024000.0, 1641024001.0, 1641024002.0]),
        'ETH': np.array([1641024000.0, 1641024001.0, 1641024002.0])
    }
    
    return mock_loader


@pytest.fixture
def order_processor(sample_fees_graph, mock_dataloader):
    """Create an order processor for testing."""
    return OrderProcessor(sample_fees_graph, mock_dataloader)


class TestOrderProcessor:
    """Test OrderProcessor class functionality."""
    
    def test_initialization(self, sample_fees_graph, mock_dataloader):
        """Test order processor initialization."""
        processor = OrderProcessor(sample_fees_graph, mock_dataloader)
        
        assert processor.fees_graph == sample_fees_graph
        assert processor.dataloader == mock_dataloader
        assert processor.timesteps is not None
    
    def test_process_buy_order_success(self, order_processor, sample_portfolio):
        """Test successful buy order processing."""
        # Setup mock data for BTC
        order_processor.dataloader.get_coin_at_timestep.return_value = pd.DataFrame({
            'timestamp': [1641024000.0],
            'level-1-bid-price': [50000.0],
            'level-1-ask-price': [50100.0]
        })
        
        # Process buy order
        trade = order_processor.process_order(
            coin='BTC',
            amount=0.1,
            action_type='buy',
            execution_timestamp=1641024000.0,
            portfolio=sample_portfolio
        )
        
        assert trade is not None
        assert trade['action'] == 'buy'
        assert trade['coin'] == 'BTC'
        assert trade['amount'] == 0.1
        assert trade['effective_price'] == 50100.0  # Ask price for buy
        assert 'fee' in trade
        assert 'cost' in trade
    
    def test_process_sell_order_success(self, order_processor, sample_portfolio):
        """Test successful sell order processing."""
        # Give portfolio some BTC to sell
        sample_portfolio.positions['BTC'] = 1.0
        
        # Setup mock data for BTC
        order_processor.dataloader.get_coin_at_timestep.return_value = pd.DataFrame({
            'timestamp': [1641024000.0],
            'level-1-bid-price': [50000.0],
            'level-1-ask-price': [50100.0]
        })
        
        # Process sell order
        trade = order_processor.process_order(
            coin='BTC',
            amount=0.1,
            action_type='sell',
            execution_timestamp=1641024000.0,
            portfolio=sample_portfolio
        )
        
        assert trade is not None
        assert trade['action'] == 'sell'
        assert trade['coin'] == 'BTC'
        assert trade['amount'] == 0.1
        assert trade['effective_price'] == 50000.0  # Bid price for sell
        assert 'fee' in trade
        assert 'proceeds' in trade
    
    def test_process_buy_order_insufficient_funds(self, order_processor):
        """Test buy order with insufficient funds."""
        # Create portfolio with very little money
        poor_portfolio = Portfolio(['BTC'], 100.0)
        
        # Setup mock data for expensive BTC
        order_processor.dataloader.get_coin_at_timestep.return_value = pd.DataFrame({
            'timestamp': [1641024000.0],
            'level-1-bid-price': [50000.0],
            'level-1-ask-price': [50100.0]
        })
        
        # Try to buy 1 BTC (costs ~50100, only have 100)
        trade = order_processor.process_order(
            coin='BTC',
            amount=1.0,
            action_type='buy',
            execution_timestamp=1641024000.0,
            portfolio=poor_portfolio
        )
        
        assert trade is None
    
    def test_process_sell_order_insufficient_coins(self, order_processor, sample_portfolio):
        """Test sell order with insufficient coins."""
        # Portfolio starts with 0 BTC
        assert sample_portfolio.positions['BTC'] == 0.0
        
        # Setup mock data
        order_processor.dataloader.get_coin_at_timestep.return_value = pd.DataFrame({
            'timestamp': [1641024000.0],
            'level-1-bid-price': [50000.0],
            'level-1-ask-price': [50100.0]
        })
        
        # Try to sell BTC we don't have
        trade = order_processor.process_order(
            coin='BTC',
            amount=0.1,
            action_type='sell',
            execution_timestamp=1641024000.0,
            portfolio=sample_portfolio
        )
        
        assert trade is None
    
    def test_process_order_empty_market_data(self, order_processor, sample_portfolio):
        """Test order processing with empty market data."""
        # Setup empty market data
        order_processor.dataloader.get_coin_at_timestep.return_value = pd.DataFrame()
        
        trade = order_processor.process_order(
            coin='BTC',
            amount=0.1,
            action_type='buy',
            execution_timestamp=1641024000.0,
            portfolio=sample_portfolio
        )
        
        assert trade is None
    
    def test_process_order_invalid_action_type(self, order_processor, sample_portfolio):
        """Test order processing with invalid action type returns None."""
        # Setup mock data
        order_processor.dataloader.get_coin_at_timestep.return_value = pd.DataFrame({
            'timestamp': [1641024000.0],
            'level-1-bid-price': [50000.0],
            'level-1-ask-price': [50100.0]
        })
        
        # Invalid action type should return None (exception is caught internally)
        result = order_processor.process_order(
            coin='BTC',
            amount=1.0,
            action_type='invalid',
            execution_timestamp=1641024000.0,
            portfolio=sample_portfolio
        )
        
        assert result is None
    
    def test_process_buy_order_with_fees(self, order_processor, sample_portfolio):
        """Test that buy orders correctly apply fees."""
        # Setup mock data
        order_processor.dataloader.get_coin_at_timestep.return_value = pd.DataFrame({
            'timestamp': [1641024000.0],
            'level-1-bid-price': [50000.0],
            'level-1-ask-price': [50100.0]
        })
        
        initial_eurc = sample_portfolio.positions['EURC']
        
        # Process buy order
        trade = order_processor.process_order(
            coin='BTC',
            amount=0.1,
            action_type='buy',
            execution_timestamp=1641024000.0,
            portfolio=sample_portfolio
        )
        
        assert trade is not None
        
        # Check that portfolio was updated correctly
        assert sample_portfolio.positions['BTC'] == 0.1
        
        # EURC should be reduced by cost + fees
        cost_with_fees = trade['cost'] + trade['fee']
        expected_eurc = initial_eurc - cost_with_fees
        assert abs(sample_portfolio.positions['EURC'] - expected_eurc) < 1e-6
    
    def test_process_sell_order_with_fees(self, order_processor, sample_portfolio):
        """Test that sell orders correctly apply fees."""
        # Give portfolio some BTC
        sample_portfolio.positions['BTC'] = 1.0
        initial_eurc = sample_portfolio.positions['EURC']
        
        # Setup mock data
        order_processor.dataloader.get_coin_at_timestep.return_value = pd.DataFrame({
            'timestamp': [1641024000.0],
            'level-1-bid-price': [50000.0],
            'level-1-ask-price': [50100.0]
        })
        
        # Process sell order
        trade = order_processor.process_order(
            coin='BTC',
            amount=0.1,
            action_type='sell',
            execution_timestamp=1641024000.0,
            portfolio=sample_portfolio
        )
        
        assert trade is not None
        
        # Check that portfolio was updated correctly
        assert sample_portfolio.positions['BTC'] == 0.9
        
        # EURC should be increased by proceeds - fees
        proceeds_minus_fees = trade['proceeds'] - trade['fee']
        expected_eurc = initial_eurc + proceeds_minus_fees
        assert abs(sample_portfolio.positions['EURC'] - expected_eurc) < 1e-6


class TestOrderProcessorEdgeCases:
    """Test edge cases and error conditions for order processor."""
    
    def test_process_order_with_exception_in_dataloader(self, order_processor, sample_portfolio):
        """Test handling of exceptions from dataloader."""
        # Make dataloader raise an exception
        order_processor.dataloader.get_coin_at_timestep.side_effect = Exception("Dataloader error")
        
        trade = order_processor.process_order(
            coin='BTC',
            amount=0.1,
            action_type='buy',
            execution_timestamp=1641024000.0,
            portfolio=sample_portfolio
        )
        
        assert trade is None
    
    def test_process_order_with_very_small_amount(self, order_processor, sample_portfolio):
        """Test processing orders with very small amounts."""
        # Setup mock data
        order_processor.dataloader.get_coin_at_timestep.return_value = pd.DataFrame({
            'timestamp': [1641024000.0],
            'level-1-bid-price': [50000.0],
            'level-1-ask-price': [50100.0]
        })
        
        # Very small buy order
        trade = order_processor.process_order(
            coin='BTC',
            amount=1e-8,  # Very small amount
            action_type='buy',
            execution_timestamp=1641024000.0,
            portfolio=sample_portfolio
        )
        
        # Should still process the order
        assert trade is not None
        assert trade['amount'] == 1e-8
    
    def test_process_order_timestamp_handling(self, order_processor, sample_portfolio):
        """Test that timestamps are handled correctly in order processing."""
        execution_time = 1641024001.5  # Between available timesteps
        
        # Setup mock data
        order_processor.dataloader.get_coin_at_timestep.return_value = pd.DataFrame({
            'timestamp': [1641024001.0],
            'level-1-bid-price': [50000.0],
            'level-1-ask-price': [50100.0]
        })
        
        trade = order_processor.process_order(
            coin='BTC',
            amount=0.1,
            action_type='buy',
            execution_timestamp=execution_time,
            portfolio=sample_portfolio
        )
        
        assert trade is not None
        assert trade['execution_timestamp'] == execution_time


@pytest.mark.integration
class TestOrderProcessorIntegration:
    """Integration tests for order processor."""
    
    def test_multiple_orders_sequence(self, order_processor, sample_portfolio):
        """Test processing multiple orders in sequence."""
        # Setup mock data
        order_processor.dataloader.get_coin_at_timestep.return_value = pd.DataFrame({
            'timestamp': [1641024000.0],
            'level-1-bid-price': [50000.0],
            'level-1-ask-price': [50100.0]
        })
        
        initial_eurc = sample_portfolio.positions['EURC']
        
        # Execute multiple buy orders
        trades = []
        for i in range(3):
            trade = order_processor.process_order(
                coin='BTC',
                amount=0.01,
                action_type='buy',
                execution_timestamp=1641024000.0 + i,
                portfolio=sample_portfolio
            )
            assert trade is not None
            trades.append(trade)
        
        # Check final portfolio state
        assert sample_portfolio.positions['BTC'] == 0.03  # 3 * 0.01
        assert sample_portfolio.positions['EURC'] < initial_eurc  # Reduced by costs + fees
        
        # All trades should have consistent structure
        for trade in trades:
            assert 'action' in trade
            assert 'coin' in trade
            assert 'amount' in trade
            assert 'effective_price' in trade
            assert 'fee' in trade
    
    def test_buy_then_sell_cycle(self, order_processor, sample_portfolio):
        """Test a complete buy-then-sell cycle."""
        # Setup mock data
        order_processor.dataloader.get_coin_at_timestep.return_value = pd.DataFrame({
            'timestamp': [1641024000.0],
            'level-1-bid-price': [50000.0],
            'level-1-ask-price': [50100.0]
        })
        
        initial_eurc = sample_portfolio.positions['EURC']
        
        # Buy BTC
        buy_trade = order_processor.process_order(
            coin='BTC',
            amount=0.1,
            action_type='buy',
            execution_timestamp=1641024000.0,
            portfolio=sample_portfolio
        )
        assert buy_trade is not None
        assert sample_portfolio.positions['BTC'] == 0.1
        
        # Sell BTC
        sell_trade = order_processor.process_order(
            coin='BTC',
            amount=0.1,
            action_type='sell',
            execution_timestamp=1641024001.0,
            portfolio=sample_portfolio
        )
        assert sell_trade is not None
        assert sample_portfolio.positions['BTC'] == 0.0
        
        # Final EURC should be less than initial due to bid-ask spread and fees
        final_eurc = sample_portfolio.positions['EURC']
        assert final_eurc < initial_eurc
        
        # But not drastically less (sanity check)
        loss = initial_eurc - final_eurc
        assert loss < 100  # Reasonable loss for small trade
