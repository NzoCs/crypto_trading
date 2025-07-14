"""
Test portfolio functionality for the backtesting module.
"""
import pytest
import pandas as pd
import numpy as np
from src.backtesting.portfolio import Portfolio, estimate_price, get_bid_ask_spread, get_fee_for_trade
from tests.backtesting.test_types import create_sample_orderbook_data, create_sample_fees_graph


class TestEstimatePrice:
    """Test price estimation functions."""
    
    def test_estimate_price_ask(self):
        """Test ask price estimation."""
        data = pd.DataFrame({
            'level-1-ask-price': [100.5, 101.0, 101.5],
            'level-1-bid-price': [100.0, 100.5, 101.0]
        })
        
        price = estimate_price(data, 'ask')
        assert price == 101.5  # Should return last ask price
    
    def test_estimate_price_bid(self):
        """Test bid price estimation."""
        data = pd.DataFrame({
            'level-1-ask-price': [100.5, 101.0, 101.5],
            'level-1-bid-price': [100.0, 100.5, 101.0]
        })
        
        price = estimate_price(data, 'bid')
        assert price == 101.0  # Should return last bid price
    
    def test_estimate_price_empty_data(self):
        """Test price estimation with empty data."""
        data = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Cannot estimate price from empty data"):
            estimate_price(data, 'ask')
    
    def test_estimate_price_missing_columns(self):
        """Test price estimation with missing price columns."""
        data = pd.DataFrame({'other_column': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Cannot estimate price: no valid price columns found"):
            estimate_price(data, 'ask')


class TestGetBidAskSpread:
    """Test bid-ask spread calculation."""
    
    def test_get_bid_ask_spread_normal(self):
        """Test normal bid-ask spread calculation."""
        data = pd.DataFrame({
            'level-1-ask-price': [100.5, 101.0, 101.5],
            'level-1-bid-price': [100.0, 100.5, 101.0]
        })
        
        spread = get_bid_ask_spread(data)
        assert spread == 0.5  # 101.5 - 101.0
    
    def test_get_bid_ask_spread_empty_data(self):
        """Test spread calculation with empty data."""
        data = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Cannot calculate spread from empty data"):
            get_bid_ask_spread(data)
    
    def test_get_bid_ask_spread_missing_columns(self):
        """Test spread calculation with missing columns."""
        data = pd.DataFrame({'other_column': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Cannot calculate spread: no level-1 bid/ask prices found"):
            get_bid_ask_spread(data)


class TestGetFeeForTrade:
    """Test fee calculation for trades."""
    
    def test_get_fee_for_trade_normal(self):
        """Test normal fee calculation."""
        fees_graph = {
            'BTC': [('EURC', 0.001), ('ETH', 0.002)],
            'EURC': [('BTC', 0.001)]
        }
        
        fee = get_fee_for_trade('BTC', 'EURC', fees_graph)
        assert fee == 0.001
        
        fee = get_fee_for_trade('BTC', 'ETH', fees_graph)
        assert fee == 0.002
    
    def test_get_fee_for_trade_missing_source(self):
        """Test fee calculation with missing source coin."""
        fees_graph = {
            'BTC': [('EURC', 0.001)]
        }
        
        with pytest.raises(ValueError, match="No fee information available for trading from ETH"):
            get_fee_for_trade('ETH', 'EURC', fees_graph)
    
    def test_get_fee_for_trade_missing_target(self):
        """Test fee calculation with missing target coin."""
        fees_graph = {
            'BTC': [('EURC', 0.001)]
        }
        
        with pytest.raises(ValueError, match="No direct trading path from BTC to ETH"):
            get_fee_for_trade('BTC', 'ETH', fees_graph)


class TestPortfolio:
    """Test Portfolio class functionality."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test portfolio using pytest fixture."""
        self.coins = ['BTC', 'ETH']
        self.initial_capital = 10000.0
        self.portfolio = Portfolio(self.coins, self.initial_capital)
        self.fees_graph = create_sample_fees_graph(self.coins)
    
    def test_portfolio_initialization(self):
        """Test portfolio initialization."""
        portfolio = Portfolio(['BTC', 'ETH'], 10000.0)
        
        assert portfolio.positions['EURC'] == 10000.0
        assert portfolio.positions['BTC'] == 0.0
        assert portfolio.positions['ETH'] == 0.0
        assert portfolio.coins == ['BTC', 'ETH']
    
    def test_get_position(self):
        """Test getting position for a coin."""
        portfolio = Portfolio(['BTC'], 5000.0)
        
        assert portfolio.get_position('EURC') == 5000.0
        assert portfolio.get_position('BTC') == 0.0
        assert portfolio.get_position('ETH') == 0.0  # Not in portfolio
    
    def test_update_position_valid(self):
        """Test updating position with valid amount."""
        portfolio = Portfolio(['BTC'], 1000.0)
        
        portfolio.update_position('BTC', 0.5)
        assert portfolio.get_position('BTC') == 0.5
        
        portfolio.update_position('BTC', 0.3)
        assert portfolio.get_position('BTC') == 0.8
    
    def test_update_position_negative_result(self):
        """Test updating position resulting in negative balance."""
        portfolio = Portfolio(['BTC'], 1000.0)
        
        with pytest.raises(ValueError, match="Cannot have negative position for BTC"):
            portfolio.update_position('BTC', -0.5)
    
    def test_update_position_unknown_coin(self):
        """Test updating position for unknown coin."""
        portfolio = Portfolio(['BTC'], 1000.0)
        
        with pytest.raises(ValueError, match="Coin ETH not found in portfolio"):
            portfolio.update_position('ETH', 0.5)
    
    def test_can_execute_trade_sufficient_funds(self):
        """Test trade feasibility with sufficient funds."""
        portfolio = Portfolio(['BTC'], 10000.0)
        fees_graph = {'EURC': [('BTC', 0.001)]}
        
        # Should be able to trade 1000 EURC for BTC with 0.1% fee
        assert portfolio.can_execute_trade('EURC', 'BTC', 1000.0, fees_graph)
    
    def test_can_execute_trade_insufficient_funds(self):
        """Test trade feasibility with insufficient funds."""
        portfolio = Portfolio(['BTC'], 1000.0)
        fees_graph = {'EURC': [('BTC', 0.001)]}
        
        # Should not be able to trade 2000 EURC for BTC (only have 1000)
        assert not portfolio.can_execute_trade('EURC', 'BTC', 2000.0, fees_graph)
    
    def test_can_execute_trade_missing_coin(self):
        """Test trade feasibility with missing coin."""
        portfolio = Portfolio(['BTC'], 1000.0)
        fees_graph = {'EURC': [('BTC', 0.001)]}
        
        # Portfolio doesn't have ETH
        assert not portfolio.can_execute_trade('ETH', 'BTC', 100.0, fees_graph)
    
    def test_execute_trade_buy_success(self):
        """Test successful buy trade execution."""
        portfolio = Portfolio(['BTC'], 10000.0)
        fees_graph = {'EURC': [('BTC', 0.001)]}
        
        # Buy 0.1 BTC at 50000 EURC/BTC with 0.1% fee
        success = portfolio.execute_trade('EURC', 'BTC', 50000.0, 0.1, fees_graph)
        
        assert success
        assert portfolio.get_position('BTC') == 0.1
        # Should have spent 0.1 * 50000 * (1 + 0.001) = 5005 EURC
        assert portfolio.get_position('EURC') == 10000.0 - 5005.0
    
    def test_execute_trade_sell_success(self):
        """Test successful sell trade execution."""
        portfolio = Portfolio(['BTC'], 10000.0)
        portfolio.positions['BTC'] = 0.2  # Start with some BTC
        fees_graph = {'BTC': [('EURC', 0.001)]}
        
        # Sell 0.1 BTC at 50000 EURC/BTC with 0.1% fee
        success = portfolio.execute_trade('BTC', 'EURC', 50000.0, 0.1, fees_graph, reverse=True)
        
        assert success
        assert portfolio.get_position('BTC') == 0.1  # 0.2 - 0.1
        # Should have received 0.1 * 50000 * (1 - 0.001) = 4995 EURC
        assert portfolio.get_position('EURC') == 10000.0 + 4995.0
    
    def test_execute_trade_insufficient_funds(self):
        """Test trade execution with insufficient funds."""
        portfolio = Portfolio(['BTC'], 1000.0)
        fees_graph = {'EURC': [('BTC', 0.001)]}
        
        # Try to buy 1 BTC at 50000 EURC/BTC (need 50050 EURC, only have 1000)
        success = portfolio.execute_trade('EURC', 'BTC', 50000.0, 1.0, fees_graph)
        
        assert not success
        assert portfolio.get_position('EURC') == 1000.0  # Unchanged
        assert portfolio.get_position('BTC') == 0.0      # Unchanged
    
    def test_get_value_single_coin(self):
        """Test portfolio valuation with single coin."""
        portfolio = Portfolio(['BTC'], 5000.0)
        portfolio.positions['BTC'] = 0.1
        
        market_data = {
            'BTC': pd.DataFrame({
                'level-1-bid-price': [50000.0]
            })
        }
        
        total_value = portfolio.get_value(market_data)
        # 5000 EURC + 0.1 BTC * 50000 = 10000 EURC
        assert total_value == 10000.0
    
    def test_get_value_multiple_coins(self):
        """Test portfolio valuation with multiple coins."""
        portfolio = Portfolio(['BTC', 'ETH'], 2000.0)
        portfolio.positions['BTC'] = 0.1
        portfolio.positions['ETH'] = 1.0
        
        market_data = {
            'BTC': pd.DataFrame({'level-1-bid-price': [50000.0]}),
            'ETH': pd.DataFrame({'level-1-bid-price': [3000.0]})
        }
        
        total_value = portfolio.get_value(market_data)
        # 2000 EURC + 0.1 BTC * 50000 + 1.0 ETH * 3000 = 10000 EURC
        assert total_value == 10000.0
    
    def test_get_value_missing_market_data(self):
        """Test portfolio valuation with missing market data."""
        portfolio = Portfolio(['BTC', 'ETH'], 5000.0)
        portfolio.positions['BTC'] = 0.1
        portfolio.positions['ETH'] = 1.0
        
        market_data = {
            'BTC': pd.DataFrame({'level-1-bid-price': [50000.0]})
            # Missing ETH data
        }
        
        total_value = portfolio.get_value(market_data)
        # 5000 EURC + 0.1 BTC * 50000 + 0 (ETH value ignored) = 10000 EURC
        assert total_value == 10000.0
    
    def test_portfolio_str_representation(self):
        """Test portfolio string representation."""
        portfolio = Portfolio(['BTC'], 1000.0)
        portfolio.positions['BTC'] = 0.5
        
        portfolio_str = str(portfolio)
        assert "Portfolio positions:" in portfolio_str
        assert "EURC: 1000.00000000" in portfolio_str
        assert "BTC: 0.50000000" in portfolio_str
    
    def test_can_execute_trade_buy_vs_sell_logic(self, sample_fees_graph):
        """Test that can_execute_trade correctly handles buy vs sell logic."""
        portfolio = Portfolio(['BTC'], 10000.0)
        portfolio.positions['BTC'] = 0.1  # Give some BTC
        fees_graph = sample_fees_graph
        
        # Test buying (EURC -> BTC): should add fees to required amount
        # Want to spend 5000 EURC, with 0.1% fee need 5005 EURC total
        can_buy = portfolio.can_execute_trade('EURC', 'BTC', 5000.0, fees_graph)
        assert can_buy  # Should work, we have 10000 EURC
        
        can_buy_expensive = portfolio.can_execute_trade('EURC', 'BTC', 9999.0, fees_graph)  
        assert not can_buy_expensive  # Should fail, would need 9999 * 1.001 = 10008.999 EURC
        
        # Test selling (BTC -> EURC): should NOT add fees to required amount
        # Want to sell 0.1 BTC, should only need exactly 0.1 BTC (fees deducted from proceeds)
        can_sell = portfolio.can_execute_trade('BTC', 'EURC', 0.1, fees_graph)
        assert can_sell  # Should work, we have exactly 0.1 BTC
        
        can_sell_too_much = portfolio.can_execute_trade('BTC', 'EURC', 0.11, fees_graph)
        assert not can_sell_too_much  # Should fail, we only have 0.1 BTC


# Pytest fixtures for common test data
@pytest.fixture
def sample_portfolio():
    """Create a sample portfolio for testing."""
    return Portfolio(['BTC', 'ETH'], 10000.0)


@pytest.fixture
def sample_fees_graph():
    """Create a sample fees graph for testing."""
    return create_sample_fees_graph(['BTC', 'ETH'])


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    return {
        'BTC': pd.DataFrame({
            'level-1-bid-price': [50000.0, 50100.0],
            'level-1-ask-price': [50050.0, 50150.0]
        }),
        'ETH': pd.DataFrame({
            'level-1-bid-price': [3000.0, 3010.0],
            'level-1-ask-price': [3005.0, 3015.0]
        })
    }


class TestPortfolioIntegration:
    """Integration tests for portfolio functionality."""
    
    def test_portfolio_complete_trading_cycle(self, sample_portfolio, sample_fees_graph, sample_market_data):
        """Test a complete trading cycle: buy -> hold -> sell."""
        portfolio = sample_portfolio
        fees_graph = sample_fees_graph
        
        # Initial state
        assert portfolio.get_position('EURC') == 10000.0
        assert portfolio.get_position('BTC') == 0.0
        
        # Buy BTC - cost should be 0.1 * 50000 * (1 + 0.001) = 5005
        success = portfolio.execute_trade('EURC', 'BTC', 50000.0, 0.1, fees_graph)
        assert success
        assert portfolio.get_position('BTC') == 0.1
        assert abs(portfolio.get_position('EURC') - 4995.0) < 1e-6  # 10000 - 5005
        
        # Check portfolio value
        portfolio_value = portfolio.get_value(sample_market_data)
        btc_value = 0.1 * sample_market_data['BTC']['level-1-bid-price'].iloc[-1]
        expected_value = portfolio.get_position('EURC') + btc_value
        assert abs(portfolio_value - expected_value) < 1e-6
        
        # Now selling should work with the bug fix
        bid_price = sample_market_data['BTC']['level-1-bid-price'].iloc[-1]
        
        # Check that we can sell (should work now)
        can_sell = portfolio.can_execute_trade('BTC', 'EURC', 0.1, fees_graph)
        assert can_sell  # Should be True after the fix
        
        # Execute the sell
        success = portfolio.execute_trade('BTC', 'EURC', bid_price, 0.1, fees_graph, reverse=True)
        assert success
        assert portfolio.get_position('BTC') == 0.0
        
        # Should have less EURC than initially due to fees and bid-ask spread
        final_eurc = portfolio.get_position('EURC')
        assert final_eurc < 10000.0
        
        # But should have more than what we had after buying (since we sold)
        assert final_eurc > 4995.0
    
    def test_portfolio_sell_fee_calculation_bug(self, sample_portfolio, sample_fees_graph):
        """Test that the fix for can_execute_trade selling bug works correctly."""
        portfolio = sample_portfolio
        fees_graph = sample_fees_graph
        
        # Give portfolio some BTC
        portfolio.positions['BTC'] = 0.1
        
        # This should now work after the fix
        can_sell = portfolio.can_execute_trade('BTC', 'EURC', 0.1, fees_graph)
        
        # Should be able to sell all our BTC now
        assert can_sell  # Fixed: should be True now
        
        # Also test that buying still works correctly
        can_buy = portfolio.can_execute_trade('EURC', 'BTC', 5000.0, fees_graph)
        assert can_buy  # Should still work for buying
    
    def test_can_execute_trade_buy_vs_sell_logic(self, sample_fees_graph):
        """Test that can_execute_trade correctly handles buy vs sell logic."""
        portfolio = Portfolio(['BTC'], 10000.0)
        portfolio.positions['BTC'] = 0.1  # Give some BTC
        fees_graph = sample_fees_graph
        
        # Test buying (EURC -> BTC): should add fees to required amount
        # Want to spend 5000 EURC, with 0.1% fee need 5005 EURC total
        can_buy = portfolio.can_execute_trade('EURC', 'BTC', 5000.0, fees_graph)
        assert can_buy  # Should work, we have 10000 EURC
        
        can_buy_expensive = portfolio.can_execute_trade('EURC', 'BTC', 9999.0, fees_graph)  
        assert not can_buy_expensive  # Should fail, would need 9999 * 1.001 = 10008.999 EURC
        
        # Test selling (BTC -> EURC): should NOT add fees to required amount
        # Want to sell 0.1 BTC, should only need exactly 0.1 BTC (fees deducted from proceeds)
        can_sell = portfolio.can_execute_trade('BTC', 'EURC', 0.1, fees_graph)
        assert can_sell  # Should work, we have exactly 0.1 BTC
        
        can_sell_too_much = portfolio.can_execute_trade('BTC', 'EURC', 0.11, fees_graph)
        assert not can_sell_too_much  # Should fail, we only have 0.1 BTC
