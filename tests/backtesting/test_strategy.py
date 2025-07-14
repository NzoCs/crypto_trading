"""
Test strategy functionality for the backtesting module.
"""
import pytest
import pandas as pd
from abc import ABC
from src.backtesting.strategy import Strategy
from src.backtesting.portfolio import Portfolio
from src.backtesting.types import MarketData, Action, FeesGraph
from tests.backtesting.test_types import create_sample_market_data, create_sample_fees_graph


class TestStrategy:
    """Test the base Strategy class."""
    
    def test_strategy_is_abstract(self):
        """Test that Strategy is an abstract base class."""
        assert issubclass(Strategy, ABC)
        
        # Should not be able to instantiate directly
        with pytest.raises(TypeError):
            Strategy()
    
    def test_strategy_has_get_action_method(self):
        """Test that Strategy has the required get_action method."""
        assert hasattr(Strategy, 'get_action')
        assert callable(getattr(Strategy, 'get_action'))


class ConcreteStrategy(Strategy):
    """Concrete strategy implementation for testing."""
    
    def __init__(self, return_value=None):
        super().__init__()
        self.return_value = return_value or {}
        self.call_count = 0
        self.last_call_args = None
    
    def get_action(self, data: MarketData, current_portfolio: Portfolio, fees_graph: FeesGraph) -> Action:
        """Simple implementation that returns a predefined action."""
        self.call_count += 1
        self.last_call_args = {
            'data': data,
            'current_portfolio': current_portfolio,
            'fees_graph': fees_graph
        }
        return self.return_value


class BuyAndHoldStrategy(Strategy):
    """Buy and hold strategy for testing."""
    
    def __init__(self, target_coin: str, target_allocation: float):
        super().__init__()
        self.target_coin = target_coin
        self.target_allocation = target_allocation
        self.initialized = False
    
    def get_action(self, data: MarketData, current_portfolio: Portfolio, fees_graph: FeesGraph) -> Action:
        """Buy target coin once and hold."""
        if not self.initialized and self.target_coin in data:
            # Calculate how much to buy
            portfolio_value = current_portfolio.get_value(data)
            target_value = portfolio_value * self.target_allocation
            
            if self.target_coin in data and not data[self.target_coin].empty:
                try:
                    # Estimate current price
                    current_price = data[self.target_coin]['level-1-ask-price'].iloc[-1]
                    target_amount = target_value / current_price
                    
                    self.initialized = True
                    return {self.target_coin: target_amount}
                except (KeyError, IndexError):
                    pass
        
        return {}


class MomentumStrategy(Strategy):
    """Simple momentum strategy for testing."""
    
    def __init__(self, lookback_periods: int = 5, threshold: float = 0.01):
        super().__init__()
        self.lookback_periods = lookback_periods
        self.threshold = threshold
    
    def get_action(self, data: MarketData, current_portfolio: Portfolio, fees_graph: FeesGraph) -> Action:
        """Buy if price momentum is positive, sell if negative."""
        actions = {}
        
        for coin, coin_data in data.items():
            if coin == 'EURC' or coin_data.empty or len(coin_data) < self.lookback_periods:
                continue
            
            try:
                # Calculate price momentum
                prices = coin_data['level-1-bid-price'].tail(self.lookback_periods)
                if len(prices) >= 2:
                    momentum = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
                    
                    current_position = current_portfolio.get_position(coin)
                    
                    if momentum > self.threshold and current_position == 0:
                        # Buy signal - invest 10% of portfolio
                        portfolio_value = current_portfolio.get_value(data)
                        target_value = portfolio_value * 0.1
                        current_price = coin_data['level-1-ask-price'].iloc[-1]
                        actions[coin] = target_value / current_price
                    
                    elif momentum < -self.threshold and current_position > 0:
                        # Sell signal - sell all
                        actions[coin] = -current_position
            
            except (KeyError, IndexError, ZeroDivisionError):
                continue
        
        return actions


class TestConcreteStrategy:
    """Test concrete strategy implementations."""
    
    @pytest.fixture
    def sample_portfolio(self):
        return Portfolio(['BTC', 'ETH'], 10000.0)
    
    @pytest.fixture
    def sample_market_data(self):
        return create_sample_market_data(['BTC', 'ETH'], 10)
    
    @pytest.fixture
    def sample_fees_graph(self):
        return create_sample_fees_graph(['BTC', 'ETH'])
    
    def test_concrete_strategy_instantiation(self):
        """Test that concrete strategies can be instantiated."""
        strategy = ConcreteStrategy()
        assert isinstance(strategy, Strategy)
        assert strategy.call_count == 0
    
    def test_concrete_strategy_get_action(self, sample_portfolio, sample_market_data, sample_fees_graph):
        """Test calling get_action on concrete strategy."""
        expected_action = {'BTC': 0.1, 'ETH': -0.05}
        strategy = ConcreteStrategy(expected_action)
        
        action = strategy.get_action(sample_market_data, sample_portfolio, sample_fees_graph)
        
        assert action == expected_action
        assert strategy.call_count == 1
        assert strategy.last_call_args is not None
        assert strategy.last_call_args['data'] == sample_market_data
        assert strategy.last_call_args['current_portfolio'] == sample_portfolio
        assert strategy.last_call_args['fees_graph'] == sample_fees_graph
    
    def test_buy_and_hold_strategy(self, sample_portfolio, sample_market_data, sample_fees_graph):
        """Test buy and hold strategy implementation."""
        strategy = BuyAndHoldStrategy('BTC', 0.5)  # 50% allocation to BTC
        
        # First call should return buy action
        action = strategy.get_action(sample_market_data, sample_portfolio, sample_fees_graph)
        
        assert 'BTC' in action
        assert action['BTC'] > 0  # Should be buying
        assert strategy.initialized
        
        # Second call should return empty action (already initialized)
        action2 = strategy.get_action(sample_market_data, sample_portfolio, sample_fees_graph)
        assert action2 == {}
    
    def test_momentum_strategy_no_momentum(self, sample_portfolio, sample_fees_graph):
        """Test momentum strategy with flat prices (no momentum)."""
        # Create market data with flat prices
        flat_data = {
            'BTC': pd.DataFrame({
                'level-1-bid-price': [50000.0] * 10,
                'level-1-ask-price': [50100.0] * 10
            }),
            'ETH': pd.DataFrame({
                'level-1-bid-price': [3000.0] * 10,
                'level-1-ask-price': [3010.0] * 10
            })
        }
        
        strategy = MomentumStrategy(lookback_periods=5, threshold=0.01)
        action = strategy.get_action(flat_data, sample_portfolio, sample_fees_graph)
        
        # Should not trade with flat prices
        assert action == {}
    
    def test_momentum_strategy_positive_momentum(self, sample_portfolio, sample_fees_graph):
        """Test momentum strategy with positive momentum."""
        # Create market data with rising prices - make momentum > 1%
        # Price goes from 50000 to 52000 (4% increase) over 10 periods
        rising_data = {
            'BTC': pd.DataFrame({
                'level-1-bid-price': [50000.0 + i*200 for i in range(10)],  # Increased step size
                'level-1-ask-price': [50100.0 + i*200 for i in range(10)]
            })
        }
        
        strategy = MomentumStrategy(lookback_periods=5, threshold=0.01)
        action = strategy.get_action(rising_data, sample_portfolio, sample_fees_graph)
        
        # Should buy BTC due to positive momentum
        assert 'BTC' in action
        assert action['BTC'] > 0
    
    def test_momentum_strategy_negative_momentum(self, sample_portfolio, sample_fees_graph):
        """Test momentum strategy with negative momentum."""
        # Give portfolio some BTC to sell
        sample_portfolio.positions['BTC'] = 1.0
        
        # Create market data with falling prices - make momentum < -1%
        # Price goes from 50000 to 48000 (4% decrease) over 10 periods
        falling_data = {
            'BTC': pd.DataFrame({
                'level-1-bid-price': [50000.0 - i*200 for i in range(10)],  # Increased step size
                'level-1-ask-price': [50100.0 - i*200 for i in range(10)]
            })
        }
        
        strategy = MomentumStrategy(lookback_periods=5, threshold=0.01)
        action = strategy.get_action(falling_data, sample_portfolio, sample_fees_graph)
        
        # Should sell BTC due to negative momentum
        assert 'BTC' in action
        assert action['BTC'] < 0
    
    def test_strategy_with_empty_data(self, sample_portfolio, sample_fees_graph):
        """Test strategy behavior with empty market data."""
        empty_data = {'BTC': pd.DataFrame(), 'ETH': pd.DataFrame()}
        
        strategy = ConcreteStrategy({'BTC': 0.1})
        action = strategy.get_action(empty_data, sample_portfolio, sample_fees_graph)
        
        # Strategy should still return its predefined action
        assert action == {'BTC': 0.1}
    
    def test_strategy_with_insufficient_data(self, sample_portfolio, sample_fees_graph):
        """Test momentum strategy with insufficient historical data."""
        # Create market data with only 2 data points (less than lookback_periods)
        insufficient_data = {
            'BTC': pd.DataFrame({
                'level-1-bid-price': [50000.0, 50100.0],
                'level-1-ask-price': [50100.0, 50200.0]
            })
        }
        
        strategy = MomentumStrategy(lookback_periods=5, threshold=0.01)
        action = strategy.get_action(insufficient_data, sample_portfolio, sample_fees_graph)
        
        # Should not trade with insufficient data
        assert action == {}


class TestStrategyErrorHandling:
    """Test error handling in strategy implementations."""
    
    @pytest.fixture
    def sample_portfolio(self):
        return Portfolio(['BTC'], 10000.0)
    
    @pytest.fixture
    def sample_fees_graph(self):
        return create_sample_fees_graph(['BTC'])
    
    def test_strategy_with_malformed_data(self, sample_portfolio, sample_fees_graph):
        """Test strategy behavior with malformed market data."""
        malformed_data = {
            'BTC': pd.DataFrame({
                'wrong_column': [1, 2, 3]  # Missing expected price columns
            })
        }
        
        strategy = MomentumStrategy()
        
        # Should not crash, should return empty action
        action = strategy.get_action(malformed_data, sample_portfolio, sample_fees_graph)
        assert action == {}
    
    def test_strategy_with_nan_values(self, sample_portfolio, sample_fees_graph):
        """Test strategy behavior with NaN values in data."""
        nan_data = {
            'BTC': pd.DataFrame({
                'level-1-bid-price': [50000.0, float('nan'), 50200.0],
                'level-1-ask-price': [50100.0, float('nan'), 50300.0]
            })
        }
        
        strategy = MomentumStrategy()
        
        # Should handle NaN values gracefully
        action = strategy.get_action(nan_data, sample_portfolio, sample_fees_graph)
        assert isinstance(action, dict)


@pytest.mark.integration
class TestStrategyIntegration:
    """Integration tests for strategy functionality."""
    
    def test_multiple_strategies_same_data(self):
        """Test multiple strategies with the same market data."""
        portfolio = Portfolio(['BTC', 'ETH'], 10000.0)
        market_data = create_sample_market_data(['BTC', 'ETH'], 10)
        fees_graph = create_sample_fees_graph(['BTC', 'ETH'])
        
        strategies = [
            ConcreteStrategy({'BTC': 0.1}),
            BuyAndHoldStrategy('BTC', 0.3),
            MomentumStrategy(lookback_periods=3, threshold=0.005)
        ]
        
        actions = []
        for strategy in strategies:
            action = strategy.get_action(market_data, portfolio, fees_graph)
            actions.append(action)
            assert isinstance(action, dict)
        
        # Each strategy should return a valid action dict
        assert len(actions) == 3
        for action in actions:
            assert isinstance(action, dict)
            for coin, amount in action.items():
                assert isinstance(coin, str)
                assert isinstance(amount, (int, float))
    
    def test_strategy_state_persistence(self):
        """Test that strategy state persists between calls."""
        strategy = ConcreteStrategy()
        portfolio = Portfolio(['BTC'], 10000.0)
        market_data = create_sample_market_data(['BTC'], 5)
        fees_graph = create_sample_fees_graph(['BTC'])
        
        # Make multiple calls
        for i in range(3):
            strategy.get_action(market_data, portfolio, fees_graph)
        
        # Call count should persist
        assert strategy.call_count == 3
    
    def test_buy_and_hold_initialization_logic(self):
        """Test that buy and hold strategy only initializes once."""
        strategy = BuyAndHoldStrategy('BTC', 0.5)
        portfolio = Portfolio(['BTC'], 10000.0)
        market_data = create_sample_market_data(['BTC'], 5)
        fees_graph = create_sample_fees_graph(['BTC'])
        
        # First call should initialize
        action1 = strategy.get_action(market_data, portfolio, fees_graph)
        assert strategy.initialized
        assert len(action1) > 0
        
        # Subsequent calls should not trade
        action2 = strategy.get_action(market_data, portfolio, fees_graph)
        assert action2 == {}
        
        action3 = strategy.get_action(market_data, portfolio, fees_graph)
        assert action3 == {}
