"""
Test dataloader functionality for the backtesting module.
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from src.backtesting.dataloader import OrderBookDataFromDf
from src.backtesting.types import Coin, Filepath, TimeStep


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    timestamps = np.linspace(1641024000.0, 1641024600.0, 100)  # 100 data points over 10 minutes
    
    return pd.DataFrame({
        'level-1-bid-price': np.random.uniform(50000, 51000, 100),
        'level-1-ask-price': np.random.uniform(51000, 52000, 100),
        'level-1-bid-volume': np.random.uniform(1, 10, 100),
        'level-1-ask-volume': np.random.uniform(1, 10, 100),
        'mid_price': np.random.uniform(50500, 51500, 100)
    }, index=pd.Index(timestamps, name='timestamp'))


@pytest.fixture
def sample_parquet_files(sample_data):
    """Create temporary parquet files for testing."""
    temp_dir = tempfile.mkdtemp()
    
    # Create BTC data
    btc_file = os.path.join(temp_dir, 'BTC_EUR.parquet')
    btc_data = sample_data.copy()
    btc_data['level-1-bid-price'] *= 1.0  # BTC prices
    btc_data.to_parquet(btc_file)
    
    # Create ETH data (different price range)
    eth_file = os.path.join(temp_dir, 'ETH_EUR.parquet')
    eth_data = sample_data.copy()
    eth_data['level-1-bid-price'] *= 0.06  # ETH prices (~3000)
    eth_data['level-1-ask-price'] *= 0.06
    eth_data['mid_price'] *= 0.06
    eth_data.to_parquet(eth_file)
    
    sources = [
        ('BTC', btc_file),
        ('ETH', eth_file)
    ]
    
    yield sources
    
    # Cleanup
    os.remove(btc_file)
    os.remove(eth_file)
    os.rmdir(temp_dir)


@pytest.fixture
def dataloader(sample_parquet_files):
    """Create a dataloader instance for testing."""
    return OrderBookDataFromDf(sample_parquet_files)


class TestOrderBookDataFromDf:
    """Test the OrderBookDataFromDf class."""
    
    def test_initialization(self, sample_parquet_files):
        """Test dataloader initialization."""
        dataloader = OrderBookDataFromDf(sample_parquet_files)
        
        assert len(dataloader.dfs) == 2
        assert 'BTC' in dataloader.dfs
        assert 'ETH' in dataloader.dfs
        assert dataloader.coins == ['BTC', 'ETH']
        
        # Check that timestamp column was added
        for coin in ['BTC', 'ETH']:
            assert 'timestamp' in dataloader.dfs[coin].columns
            assert len(dataloader.dfs[coin]) > 0
    
    def test_get_time_step_values(self, dataloader):
        """Test getting timestep values for all coins."""
        timestep_values = dataloader.get_time_step_values()
        
        assert isinstance(timestep_values, dict)
        assert 'BTC' in timestep_values
        assert 'ETH' in timestep_values
        
        for coin, timestamps in timestep_values.items():
            assert isinstance(timestamps, np.ndarray)
            assert len(timestamps) > 0
            assert np.all(np.diff(timestamps) >= 0)  # Should be sorted
    
    def test_get_coin_at_timestep(self, dataloader):
        """Test getting data for a specific coin at a timestep index."""
        # Get data at index 5
        btc_data = dataloader.get_coin_at_timestep('BTC', 5)
        
        assert isinstance(btc_data, pd.DataFrame)
        assert len(btc_data) == 1  # Should return single row
        assert 'timestamp' in btc_data.columns
        assert 'level-1-bid-price' in btc_data.columns
    
    def test_get_book_from_range(self, dataloader):
        """Test getting data within a time range for a single coin."""
        # Get all timesteps for BTC
        timesteps = dataloader.get_time_step_values()['BTC']
        start_time = timesteps[10]
        end_time = timesteps[20]
        
        btc_data = dataloader.get_book_from_range('BTC', start_time, end_time)
        
        assert isinstance(btc_data, pd.DataFrame)
        assert len(btc_data) > 0
        assert btc_data['timestamp'].min() >= start_time
        assert btc_data['timestamp'].max() <= end_time
    
    def test_get_book_from_range_invalid_time_order(self, dataloader):
        """Test getting data with invalid time order (start > end)."""
        timesteps = dataloader.get_time_step_values()['BTC']
        start_time = timesteps[20]
        end_time = timesteps[10]  # Start after end
        
        with pytest.raises(ValueError, match="Start time .* cannot be after end time"):
            dataloader.get_book_from_range('BTC', start_time, end_time)
    
    def test_get_book_from_range_out_of_bounds(self, dataloader):
        """Test getting data outside available range."""
        timesteps = dataloader.get_time_step_values()['BTC']
        start_time = timesteps.min() - 1000  # Before available data
        end_time = timesteps.max() + 1000    # After available data
        
        with pytest.raises(ValueError, match="Requested time range .* is outside the available data range"):
            dataloader.get_book_from_range('BTC', start_time, end_time)
    
    def test_get_books_from_range(self, dataloader):
        """Test getting data within a time range for all coins."""
        # Get a common time range
        all_timesteps = dataloader.get_time_step_values()
        min_start = max(ts.min() for ts in all_timesteps.values())
        max_end = min(ts.max() for ts in all_timesteps.values())
        
        # Get data from the middle of the range
        range_size = (max_end - min_start) * 0.1
        start_time = min_start + range_size
        end_time = max_end - range_size
        
        market_data = dataloader.get_books_from_range(start_time, end_time)
        
        assert isinstance(market_data, dict)
        assert 'BTC' in market_data
        assert 'ETH' in market_data
        
        for coin, data in market_data.items():
            assert isinstance(data, pd.DataFrame)
            if not data.empty:
                assert data['timestamp'].min() >= start_time
                assert data['timestamp'].max() <= end_time
    
    def test_chronological_iterator(self, dataloader):
        """Test the chronological iterator functionality."""
        iterator = dataloader.chronological_iterator()
        
        timestamps = []
        coin_indices_list = []
        
        # Collect first 10 iterations
        for i, (timestamp, coin_indices) in enumerate(iterator):
            if i >= 10:
                break
            
            timestamps.append(timestamp)
            coin_indices_list.append(coin_indices.copy())
            
            # Verify timestamp is a float
            assert isinstance(timestamp, (float, np.floating))
            
            # Verify coin_indices is a dict with integer values
            assert isinstance(coin_indices, dict)
            for coin, idx in coin_indices.items():
                assert isinstance(idx, (int, np.integer))
                assert idx >= 0
        
        # Verify timestamps are in chronological order
        assert len(timestamps) > 0
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i-1]
    
    def test_chronological_iterator_covers_all_data(self, dataloader):
        """Test that chronological iterator covers all unique timestamps."""
        # Get all unique timestamps from all coins
        all_timesteps = dataloader.get_time_step_values()
        all_unique_timestamps = set()
        for coin_timestamps in all_timesteps.values():
            all_unique_timestamps.update(coin_timestamps)
        
        # Collect all timestamps from iterator
        iterator_timestamps = set()
        for timestamp, _ in dataloader.chronological_iterator():
            iterator_timestamps.add(timestamp)
        
        # Should have the same unique timestamps
        assert iterator_timestamps == all_unique_timestamps
    
    def test_empty_dataloader(self):
        """Test dataloader behavior with empty source list."""
        empty_dataloader = OrderBookDataFromDf([])
        
        assert len(empty_dataloader.dfs) == 0
        assert empty_dataloader.coins == []
        
        timestep_values = empty_dataloader.get_time_step_values()
        assert timestep_values == {}
        
        # Iterator should not yield anything
        iterator = empty_dataloader.chronological_iterator()
        count = sum(1 for _ in iterator)
        assert count == 0


class TestDataLoaderEdgeCases:
    """Test edge cases and error conditions for the dataloader."""
    
    def test_missing_file(self):
        """Test handling of missing parquet files."""
        sources = [('BTC', 'nonexistent_file.parquet')]
        
        with pytest.raises(FileNotFoundError):
            OrderBookDataFromDf(sources)
    
    def test_get_coin_at_timestep_out_of_bounds(self, dataloader):
        """Test getting data at out-of-bounds timestep index."""
        # Try to get data beyond available indices
        max_idx = len(dataloader.dfs['BTC'])
        
        with pytest.raises(IndexError):
            dataloader.get_coin_at_timestep('BTC', max_idx + 10)
    
    def test_get_coin_at_timestep_unknown_coin(self, dataloader):
        """Test getting data for unknown coin."""
        with pytest.raises(KeyError):
            dataloader.get_coin_at_timestep('UNKNOWN', 0)


@pytest.mark.integration
class TestDataLoaderIntegration:
    """Integration tests for dataloader functionality."""
    
    def test_full_workflow(self, dataloader):
        """Test a complete workflow with the dataloader."""
        # Get timestamp values
        timestep_values = dataloader.get_time_step_values()
        assert len(timestep_values) > 0
        
        # Get a time range
        all_timestamps = list(timestep_values.values())[0]
        start_time = all_timestamps[10]
        end_time = all_timestamps[20]
        
        # Get data for the range
        market_data = dataloader.get_books_from_range(start_time, end_time)
        assert len(market_data) > 0
        
        # Verify data quality
        for coin, data in market_data.items():
            if not data.empty:
                assert 'timestamp' in data.columns
                assert 'level-1-bid-price' in data.columns
                assert 'level-1-ask-price' in data.columns
                
                # Verify bid <= ask
                assert all(data['level-1-bid-price'] <= data['level-1-ask-price'])
    
    def test_chronological_iterator_with_market_data(self, dataloader):
        """Test using chronological iterator to process market data sequentially."""
        processed_count = 0
        last_timestamp = None
        
        for timestamp, coin_indices in dataloader.chronological_iterator():
            # Verify chronological ordering
            if last_timestamp is not None:
                assert timestamp >= last_timestamp
            last_timestamp = timestamp
            
            # Try to get market data at this timestamp
            for coin, idx in coin_indices.items():
                if idx < len(dataloader.dfs[coin]):
                    data = dataloader.get_coin_at_timestep(coin, idx)
                    assert not data.empty
                    assert 'timestamp' in data.columns
            
            processed_count += 1
            if processed_count >= 20:  # Limit for test performance
                break
        
        assert processed_count > 0
