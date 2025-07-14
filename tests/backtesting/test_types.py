"""
Test types and type aliases for the backtesting module.
"""
import pytest
import pandas as pd
import numpy as np
from src.backtesting.types import (
    Coin, Filepath, OrderBookData, MarketData, TimeStep, Action, FeePrice, FeesGraph
)


class TestTypeAliases:
    """Test that type aliases work correctly."""
    
    def test_coin_type(self):
        """Test Coin type alias."""
        coin: Coin = "BTC"
        assert isinstance(coin, str)
        assert coin == "BTC"
    
    def test_filepath_type(self):
        """Test Filepath type alias."""
        filepath: Filepath = "/path/to/file.csv"
        assert isinstance(filepath, str)
        assert filepath == "/path/to/file.csv"
    
    def test_orderbook_data_type(self):
        """Test OrderBookData type alias."""
        data: OrderBookData = pd.DataFrame({
            'timestamp': [1.0, 2.0],
            'level-1-bid-price': [100.0, 101.0],
            'level-1-ask-price': [100.5, 101.5]
        })
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 2
    
    def test_market_data_type(self):
        """Test MarketData type alias."""
        market_data: MarketData = {
            'BTC': pd.DataFrame({'price': [50000.0]}),
            'ETH': pd.DataFrame({'price': [3000.0]})
        }
        assert isinstance(market_data, dict)
        assert 'BTC' in market_data
        assert 'ETH' in market_data
        assert isinstance(market_data['BTC'], pd.DataFrame)
    
    def test_timestep_type(self):
        """Test TimeStep type alias."""
        timestep: TimeStep = 1641024000.0
        assert isinstance(timestep, float)
        assert timestep > 0
    
    def test_action_type(self):
        """Test Action type alias."""
        action: Action = {
            'BTC': 0.5,
            'ETH': -1.0
        }
        assert isinstance(action, dict)
        assert action['BTC'] == 0.5
        assert action['ETH'] == -1.0
    
    def test_fee_price_type(self):
        """Test FeePrice type alias."""
        fee: FeePrice = 0.001
        assert isinstance(fee, float)
        assert 0 <= fee <= 1  # Reasonable fee range
    
    def test_fees_graph_type(self):
        """Test FeesGraph type alias."""
        fees: FeesGraph = {
            'BTC': [('EURC', 0.001), ('ETH', 0.002)],
            'EURC': [('BTC', 0.001), ('ETH', 0.001)]
        }
        assert isinstance(fees, dict)
        assert 'BTC' in fees
        assert isinstance(fees['BTC'], list)
        assert isinstance(fees['BTC'][0], tuple)
        assert len(fees['BTC'][0]) == 2


def create_sample_orderbook_data(num_rows: int = 10) -> pd.DataFrame:
    """Helper function to create sample orderbook data for testing."""
    timestamps = np.linspace(1641024000.0, 1641024600.0, num_rows)
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'level-1-bid-price': np.random.uniform(100, 110, num_rows),
        'level-1-ask-price': np.random.uniform(110, 120, num_rows),
        'level-1-bid-volume': np.random.uniform(1, 10, num_rows),
        'level-1-ask-volume': np.random.uniform(1, 10, num_rows),
    })


def create_sample_market_data(coins: list[str], num_rows: int = 10) -> MarketData:
    """Helper function to create sample market data for testing."""
    return {
        coin: create_sample_orderbook_data(num_rows)
        for coin in coins
    }


def create_sample_fees_graph(coins: list[str]) -> FeesGraph:
    """Helper function to create sample fees graph for testing."""
    fees_graph = {}
    base_fee = 0.001
    
    for coin in coins:
        fees_graph[coin] = []
        for target_coin in coins:
            if coin != target_coin:
                fees_graph[coin].append((target_coin, base_fee))
    
    # Add EURC as base currency
    if 'EURC' not in coins:
        fees_graph['EURC'] = [(coin, base_fee) for coin in coins]
        for coin in coins:
            fees_graph[coin].append(('EURC', base_fee))
    
    return fees_graph


class TestDataHelpers:
    """Test the helper functions for creating test data."""
    
    def test_create_sample_orderbook_data(self):
        """Test sample orderbook data creation."""
        data = create_sample_orderbook_data(5)
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 5
        assert 'timestamp' in data.columns
        assert 'level-1-bid-price' in data.columns
        assert 'level-1-ask-price' in data.columns
        assert all(data['level-1-ask-price'] >= data['level-1-bid-price'])
    
    def test_create_sample_market_data(self):
        """Test sample market data creation."""
        coins = ['BTC', 'ETH']
        market_data = create_sample_market_data(coins, 8)
        
        assert isinstance(market_data, dict)
        assert len(market_data) == 2
        assert 'BTC' in market_data
        assert 'ETH' in market_data
        assert len(market_data['BTC']) == 8
        assert len(market_data['ETH']) == 8
    
    def test_create_sample_fees_graph(self):
        """Test sample fees graph creation."""
        coins = ['BTC', 'ETH']
        fees_graph = create_sample_fees_graph(coins)
        
        assert isinstance(fees_graph, dict)
        assert 'BTC' in fees_graph
        assert 'ETH' in fees_graph
        assert 'EURC' in fees_graph
        
        # Check that each coin has trading pairs
        for coin in coins:
            assert len(fees_graph[coin]) >= 2  # At least one other coin + EURC
            for target_coin, fee in fees_graph[coin]:
                assert isinstance(target_coin, str)
                assert isinstance(fee, float)
                assert 0 <= fee <= 1
