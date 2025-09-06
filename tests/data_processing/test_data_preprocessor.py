"""
Tests for the data preprocessing module.

This module tests the data preprocessing functions including vectorized preprocessing,
duplicate cleaning, pivoting to wide format, and the complete preprocessing pipeline.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.data_processing.data_preprocessor import (
    preprocessing_vectorized,
    clean_duplicates,
    pivot_to_wide_format,
    preprocess_crypto_data,
    preprocess_data_folder
)


class TestPreprocessingVectorized:
    """Test the preprocessing_vectorized function."""
    
    def test_preprocessing_vectorized_basic(self):
        """Test basic vectorized preprocessing."""
        # Create test data
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=5, freq='1s'),
            'price': [100.0, 101.0, 102.0, 103.0, 104.0],
            'volume': [10.0, 15.0, 20.0, 25.0, 30.0],
            'level': [1, 1, 1, 1, 1],
            'side': ['bid', 'ask', 'bid', 'ask', 'bid']
        })
        
        result = preprocessing_vectorized(df, block_size=2)
        
        assert 'row_id' in result.columns
        assert len(result) == 5
        
        # Check row_id assignment: [0, 0, 1, 1, 2]
        expected_row_ids = [0, 0, 1, 1, 2]
        assert result['row_id'].tolist() == expected_row_ids
        
        # Check timestamp alignment (max per block)
        assert result.loc[0, 'timestamp'] == result.loc[1, 'timestamp']  # Block 0
        assert result.loc[2, 'timestamp'] == result.loc[3, 'timestamp']  # Block 1
    
    def test_preprocessing_vectorized_exact_block_size(self):
        """Test preprocessing when data length is exact multiple of block size."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=4, freq='1s'),
            'value': [1, 2, 3, 4]
        })
        
        result = preprocessing_vectorized(df, block_size=2)
        
        # Should have 2 blocks: [0, 0, 1, 1]
        expected_row_ids = [0, 0, 1, 1]
        assert result['row_id'].tolist() == expected_row_ids
    
    def test_preprocessing_vectorized_single_row(self):
        """Test preprocessing with single row."""
        df = pd.DataFrame({
            'timestamp': [pd.Timestamp('2023-01-01')],
            'value': [100.0]
        })
        
        result = preprocessing_vectorized(df, block_size=5)
        
        assert result['row_id'].tolist() == [0]
        assert len(result) == 1
    
    def test_preprocessing_vectorized_empty_dataframe(self):
        """Test preprocessing with empty DataFrame."""
        df = pd.DataFrame(columns=['timestamp', 'value'])
        
        result = preprocessing_vectorized(df, block_size=10)
        
        assert 'row_id' in result.columns
        assert len(result) == 0


class TestCleanDuplicates:
    """Test the clean_duplicates function."""
    
    def test_clean_duplicates_no_duplicates(self):
        """Test cleaning when no duplicates exist."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=3, freq='1s'),
            'row_id': [0, 0, 1],
            'level': [1, 2, 1],
            'side': ['bid', 'bid', 'ask'],
            'price': [100.0, 101.0, 102.0]
        })
        
        result = clean_duplicates(df)
        
        assert len(result) == len(df)
        pd.testing.assert_frame_equal(result, df)
    
    def test_clean_duplicates_with_duplicates(self):
        """Test cleaning when duplicates exist."""
        df = pd.DataFrame({
            'timestamp': [
                pd.Timestamp('2023-01-01'),
                pd.Timestamp('2023-01-01'),  # Duplicate
                pd.Timestamp('2023-01-02')
            ],
            'row_id': [0, 0, 1],
            'level': [1, 1, 1],
            'side': ['bid', 'bid', 'ask'],
            'price': [100.0, 100.5, 102.0],  # Different prices
            'volume': [10.0, 15.0, 20.0]
        })
        
        with patch('builtins.print') as mock_print:
            result = clean_duplicates(df)
        
        # Should remove one row (the duplicate)
        assert len(result) == 2
        mock_print.assert_called_once()
        assert "Found 2 duplicate rows" in mock_print.call_args[0][0]
    
    def test_clean_duplicates_empty_dataframe(self):
        """Test cleaning empty DataFrame."""
        df = pd.DataFrame(columns=['timestamp', 'row_id', 'level', 'side'])
        
        result = clean_duplicates(df)
        
        assert len(result) == 0
        pd.testing.assert_frame_equal(result, df)


class TestPivotToWideFormat:
    """Test the pivot_to_wide_format function."""
    
    def test_pivot_to_wide_format_basic(self):
        """Test basic pivoting to wide format."""
        df = pd.DataFrame({
            'timestamp': [
                pd.Timestamp('2023-01-01'),
                pd.Timestamp('2023-01-01'),
                pd.Timestamp('2023-01-01'),
                pd.Timestamp('2023-01-01')
            ],
            'row_id': [0, 0, 0, 0],
            'level': [1, 1, 2, 2],
            'side': ['bid', 'ask', 'bid', 'ask'],
            'price': [100.0, 101.0, 99.5, 101.5],
            'volume': [10.0, 12.0, 8.0, 9.0]
        })
        
        result = pivot_to_wide_format(df)
        
        # Check column format
        expected_columns = [
            'level-1-ask-price', 'level-1-ask-volume',
            'level-1-bid-price', 'level-1-bid-volume',
            'level-2-ask-price', 'level-2-ask-volume',
            'level-2-bid-price', 'level-2-bid-volume'
        ]
        
        assert sorted(result.columns) == sorted(expected_columns)
        assert len(result) == 1  # One unique timestamp-row_id combination
        
        # Check specific values
        assert result.iloc[0]['level-1-bid-price'] == 100.0
        assert result.iloc[0]['level-1-ask-price'] == 101.0
        assert result.iloc[0]['level-2-bid-volume'] == 8.0
    
    def test_pivot_to_wide_format_multiple_timestamps(self):
        """Test pivoting with multiple timestamps."""
        df = pd.DataFrame({
            'timestamp': [
                pd.Timestamp('2023-01-01'),
                pd.Timestamp('2023-01-01'),
                pd.Timestamp('2023-01-02'),
                pd.Timestamp('2023-01-02')
            ],
            'row_id': [0, 0, 1, 1],
            'level': [1, 1, 1, 1],
            'side': ['bid', 'ask', 'bid', 'ask'],
            'price': [100.0, 101.0, 102.0, 103.0],
            'volume': [10.0, 12.0, 15.0, 18.0]
        })
        
        result = pivot_to_wide_format(df)
        
        assert len(result) == 2  # Two unique timestamp-row_id combinations
        assert 'level-1-bid-price' in result.columns
        assert 'level-1-ask-price' in result.columns
        
        # Check values for each timestamp
        first_row = result.iloc[0]
        second_row = result.iloc[1]
        
        assert first_row['level-1-bid-price'] == 100.0
        assert first_row['level-1-ask-price'] == 101.0
        assert second_row['level-1-bid-price'] == 102.0
        assert second_row['level-1-ask-price'] == 103.0


class TestPreprocessCryptoData:
    """Test the complete preprocessing pipeline."""
    
    def test_preprocess_crypto_data_full_pipeline(self):
        """Test the complete preprocessing pipeline with temporary files."""
        # Create temporary input file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_input:
            # Write test CSV data
            tmp_input.write("timestamp,level,side,price,volume\n")
            tmp_input.write("2023-01-01 00:00:00,1,bid,100.0,10.0\n")
            tmp_input.write("2023-01-01 00:00:01,1,ask,101.0,12.0\n")
            tmp_input.write("2023-01-01 00:00:02,1,bid,100.5,15.0\n")
            tmp_input.write("2023-01-01 00:00:03,1,ask,101.5,18.0\n")
            tmp_input_path = tmp_input.name
        
        # Create temporary output file path
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp_output:
            tmp_output_path = tmp_output.name
        
        try:
            # Process the data
            with patch('builtins.print'):  # Suppress print statements
                result = preprocess_crypto_data(
                    input_file=tmp_input_path,
                    output_file=tmp_output_path,
                    coin='TEST',
                    block_size=2
                )
            
            # Check result
            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0
            assert result.index.names == ['timestamp', 'row_id']
            
            # Check that output file was created
            assert Path(tmp_output_path).exists()
            
            # Read back the saved file to verify
            saved_df = pd.read_parquet(tmp_output_path)
            pd.testing.assert_frame_equal(result, saved_df)
            
        finally:
            # Clean up temporary files
            Path(tmp_input_path).unlink(missing_ok=True)
            Path(tmp_output_path).unlink(missing_ok=True)
    
    def test_preprocess_crypto_data_file_not_found(self):
        """Test preprocessing with non-existent input file."""
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp_output:
            tmp_output_path = tmp_output.name
        
        try:
            with pytest.raises(FileNotFoundError):
                preprocess_crypto_data(
                    input_file="non_existent_file.csv",
                    output_file=tmp_output_path,
                    coin='TEST'
                )
        finally:
            Path(tmp_output_path).unlink(missing_ok=True)


class TestPreprocessDataFolder:
    """Test the batch preprocessing function."""
    
    def test_preprocess_data_folder_basic(self):
        """Test processing a folder with multiple CSV files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            input_dir.mkdir()
            
            # Create test CSV files
            for coin in ['BTC', 'ETH']:
                csv_file = input_dir / f"{coin}_EUR.csv"
                with open(csv_file, 'w') as f:
                    f.write("timestamp,level,side,price,volume\n")
                    f.write(f"2023-01-01 00:00:00,1,bid,100.0,10.0\n")
                    f.write(f"2023-01-01 00:00:01,1,ask,101.0,12.0\n")
            
            with patch('builtins.print'):  # Suppress print statements
                results = preprocess_data_folder(
                    input_folder=input_dir,
                    output_folder=output_dir,
                    coins=['BTC', 'ETH'],
                    block_size=2
                )
            
            # Check results
            assert isinstance(results, dict)
            assert 'BTC' in results
            assert 'ETH' in results
            assert len(results) == 2
            
            # Check output files exist
            assert (output_dir / "BTC_EUR.parquet").exists()
            assert (output_dir / "ETH_EUR.parquet").exists()
    
    def test_preprocess_data_folder_input_not_found(self):
        """Test processing with non-existent input folder."""
        with pytest.raises(FileNotFoundError):
            preprocess_data_folder(
                input_folder="non_existent_folder",
                output_folder="output_folder"
            )
    
    def test_preprocess_data_folder_missing_files(self):
        """Test processing when some expected files are missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            input_dir.mkdir()
            
            # Create only one file when two are requested
            csv_file = input_dir / "BTC_EUR.csv"
            with open(csv_file, 'w') as f:
                f.write("timestamp,level,side,price,volume\n")
                f.write("2023-01-01 00:00:00,1,bid,100.0,10.0\n")
            
            with patch('builtins.print') as mock_print:
                results = preprocess_data_folder(
                    input_folder=input_dir,
                    output_folder=output_dir,
                    coins=['BTC', 'ETH']  # ETH file doesn't exist
                )
            
            # Should process BTC successfully
            assert 'BTC' in results
            assert 'ETH' not in results
            assert len(results) == 1
            
            # Should print warning about missing file
            warnings = [call for call in mock_print.call_args_list 
                       if 'Warning' in str(call)]
            assert len(warnings) > 0


if __name__ == '__main__':
    pytest.main([__file__])
