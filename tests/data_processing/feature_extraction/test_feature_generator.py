"""
Tests for the feature generation module.

This module tests the FeatureGenerator class and related functionality,
including data preprocessing and feature generation pipelines.
"""

import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch, MagicMock
from typing import Any

from src.data_processing.types import (
    RawData,
    CleanedData,
    Feature,
    FeatureSet,
    FeatureName,
    FeatureGenerationError,
    DataValidationError
)
from src.data_processing.feature_extraction.base import BaseFeature

# Mock the imports that may not be available
with patch.dict('sys.modules', {
    'feature_extraction.features': MagicMock(),
    'features': MagicMock(),
}):
    from src.data_processing.feature_extraction.feature_generator import (
        preprocess_data,
        load_and_generate_features,
        save_features,
        recompute_feature
    )
    
    # Import FeatureGenerator separately with mocked default features
    # Note: With the new registry system, features are auto-registered via __init__.py
    from src.data_processing.feature_extraction.feature_generator import FeatureGenerator


class MockFeature(BaseFeature):
    """Mock feature for testing."""
    
    def __init__(self, name: FeatureName, should_fail: bool = False, value_multiplier: float = 1.0):
        super().__init__(name, f"Mock feature: {name}")
        self.should_fail = should_fail
        self.value_multiplier = value_multiplier
    
    def generate(self, df_cleaned: CleanedData, **kwargs: Any) -> Feature:
        """Generate a mock feature."""
        if self.should_fail:
            raise ValueError(f"Mock failure for {self.name}")
        
        # Simple mock feature: first numeric column * multiplier
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            result = pd.Series(np.ones(len(df_cleaned)) * self.value_multiplier, 
                         index=df_cleaned.index, name=self.name)
        else:
            result = df_cleaned[numeric_cols[0]] * self.value_multiplier
        result.name = self.name
        
        return result


class TestPreprocessData:
    """Test the preprocess_data function."""
    
    def test_preprocess_data_basic(self):
        """Test basic data preprocessing."""
        # Create multi-index DataFrame simulating raw order book data
        timestamps = pd.date_range('2023-01-01', periods=2, freq='1s')
        row_ids = [0, 1, 0, 1]  # Two rows per timestamp
        
        index = pd.MultiIndex.from_arrays([
        [timestamps[0], timestamps[0], timestamps[1], timestamps[1]],
        row_ids
        ], names=['timestamp', 'row_id'])
        
        raw_data = pd.DataFrame({
        'level-1-bid-price': [100.0, 100.5, 101.0, 101.5],
        'level-1-bid-volume': [10.0, 20.0, 15.0, 25.0],
        'level-1-ask-price': [101.0, 101.5, 102.0, 102.5],
        'level-1-ask-volume': [12.0, 18.0, 14.0, 22.0]
        }, index=index)
        
        result = preprocess_data(raw_data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # Two unique timestamps
        assert 'level-1-bid-price' in result.columns
        assert 'level-1-bid-volume' in result.columns
        assert 'level-1-ask-price' in result.columns
        assert 'level-1-ask-volume' in result.columns
        
        # Check weighted average calculation for first timestamp
        # Price: (100.0 * 10.0 + 100.5 * 20.0) / (10.0 + 20.0) = 100.33...
        expected_price = (100.0 * 10.0 + 100.5 * 20.0) / (10.0 + 20.0)
        assert abs(result.iloc[0]['level-1-bid-price'] - expected_price) < 0.01
        
        # Check volume average for first timestamp: (10.0 + 20.0) / 2 = 15.0
        expected_volume = (10.0 + 20.0) / 2
        assert result.iloc[0]['level-1-bid-volume'] == expected_volume
    
    def test_preprocess_data_empty_dataframe(self):
        """Test preprocessing with empty DataFrame."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(DataValidationError, match="Input DataFrame is empty"):
            preprocess_data(empty_df)
    
    def test_preprocess_data_missing_columns(self):
        """Test preprocessing with missing standard columns."""
        # DataFrame without standard order book columns
        timestamps = pd.date_range('2023-01-01', periods=2, freq='1s')
        index = pd.MultiIndex.from_arrays([
        [timestamps[0], timestamps[1]],
        [0, 0]
        ])
        
        df_no_standard_cols = pd.DataFrame({
        'custom-col': [1, 2]
        }, index=index)
        
        result = preprocess_data(df_no_standard_cols)
        
        # Should not fail, but result will be empty (no standard columns found)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert len(result.columns) == 0  # No standard columns processed
    
    def test_preprocess_data_single_timestamp(self):
        """Test preprocessing with single timestamp."""
        timestamp = pd.Timestamp('2023-01-01')
        index = pd.MultiIndex.from_arrays([
        [timestamp, timestamp],
        [0, 1]
        ])
        
        raw_data = pd.DataFrame({
        'level-1-bid-price': [100.0, 100.2],
        'level-1-bid-volume': [10.0, 15.0]
        }, index=index)
        
        result = preprocess_data(raw_data)
        
        assert len(result) == 1
        assert result.index[0] == timestamp
        # Weighted average: (100.0 * 10.0 + 100.2 * 15.0) / (10.0 + 15.0)
        expected_price = (100.0 * 10.0 + 100.2 * 15.0) / (10.0 + 15.0)
        assert abs(result.iloc[0]['level-1-bid-price'] - expected_price) < 0.01


class TestFeatureGenerator:
    """Test the FeatureGenerator class."""
    
    def test_feature_generator_creation(self):
        """Test creating a FeatureGenerator instance."""
        generator = FeatureGenerator()
        assert isinstance(generator.features, dict)
        # With auto-registration, we should have default features registered
        assert len(generator.features) > 0  # Auto-registered features are available
        
        # Verify some expected features are present
        feature_names = generator.list_features()
        assert "spread" in feature_names  # Basic spread feature should be auto-registered
        assert isinstance(feature_names, list)
    
    def test_register_feature(self):
        """Test registering a feature."""
        generator = FeatureGenerator()
        feature = MockFeature("test_feature")
        
        generator.register_feature(feature)
        
        assert "test_feature" in generator.features
        assert generator.features["test_feature"] == feature
    
    def test_unregister_feature(self):
        """Test unregistering a feature."""
        generator = FeatureGenerator()
        feature = MockFeature("test_feature")
        
        generator.register_feature(feature)
        assert "test_feature" in generator.features
        
        generator.unregister_feature("test_feature")
        assert "test_feature" not in generator.features
    
    def test_list_features(self):
        """Test listing registered features."""
        generator = FeatureGenerator()
        
        # Get initial count of auto-registered features
        initial_features = set(generator.list_features())
        initial_count = len(initial_features)
        
        feature1 = MockFeature("feature1")
        feature2 = MockFeature("feature2")
        
        generator.register_feature(feature1)
        generator.register_feature(feature2)
        
        feature_list = generator.list_features()
        assert isinstance(feature_list, list)
        
        # Check that our new features are added to the existing ones
        feature_set = set(feature_list)
        assert "feature1" in feature_set
        assert "feature2" in feature_set
        assert len(feature_list) == initial_count + 2  # Two new features added
        
        # Ensure initial features are still there
        assert initial_features.issubset(feature_set)
    
    def test_get_feature_info(self):
        """Test getting feature information."""
        generator = FeatureGenerator()
        feature = MockFeature("test_feature")
            
        generator.register_feature(feature)
            
        info = generator.get_feature_info("test_feature")
        assert "Mock feature: test_feature" in info
            
        # Test non-existent feature
        info_missing = generator.get_feature_info("missing_feature")
        assert "not found" in info_missing
    
    def test_generate_single_feature(self):
        """Test generating a single feature."""
        generator = FeatureGenerator()
        feature = MockFeature("test_feature", value_multiplier=2.0)
            
        generator.register_feature(feature)
            
        df_cleaned = pd.DataFrame({
            'price': [100.0, 101.0, 102.0],
            'volume': [10.0, 15.0, 20.0]
        })
            
        result = generator.generate_feature("test_feature", df_cleaned)
            
        assert isinstance(result, pd.Series)
        assert result.name == "test_feature"
        assert len(result) == 3
        # Should be price * 2.0
        expected = pd.Series([200.0, 202.0, 204.0], name="test_feature")
        pd.testing.assert_series_equal(result, expected)
    
    def test_generate_feature_not_registered(self):
        """Test generating a feature that's not registered."""
        generator = FeatureGenerator()
            
        df_cleaned = pd.DataFrame({'price': [100.0]})
            
        with pytest.raises(FeatureGenerationError, match="not registered"):
            generator.generate_feature("missing_feature", df_cleaned)
    
    def test_generate_feature_failure(self):
        """Test generating a feature that fails."""
        generator = FeatureGenerator()
        feature = MockFeature("failing_feature", should_fail=True)
            
        generator.register_feature(feature)
            
        df_cleaned = pd.DataFrame({'price': [100.0]})
            
        with pytest.raises(FeatureGenerationError, match="Generation failed"):
            generator.generate_feature("failing_feature", df_cleaned)
    
    def test_generate_multiple_features(self):
        """Test generating multiple features."""
        generator = FeatureGenerator()
            
        feature1 = MockFeature("feature1", value_multiplier=1.0)
        feature2 = MockFeature("feature2", value_multiplier=2.0)
            
        generator.register_feature(feature1)
        generator.register_feature(feature2)
            
        df_cleaned = pd.DataFrame({
            'price': [100.0, 101.0],
            'volume': [10.0, 15.0]
        })
            
        result = generator.generate_features(df_cleaned, ["feature1", "feature2"])
            
        assert isinstance(result, pd.DataFrame)
        assert "feature1" in result.columns
        assert "feature2" in result.columns
        assert "price" in result.columns  # Original columns preserved
        assert "volume" in result.columns
            
        # Check feature values
        pd.testing.assert_series_equal(result["feature1"], 
                                     pd.Series([100.0, 101.0], name="feature1"))
        pd.testing.assert_series_equal(result["feature2"], 
                                     pd.Series([200.0, 202.0], name="feature2"))
    
    def test_generate_features_with_failure(self):
        """Test generating multiple features when one fails."""
        generator = FeatureGenerator()
            
        feature1 = MockFeature("good_feature", value_multiplier=1.0)
        feature2 = MockFeature("bad_feature", should_fail=True)
            
        generator.register_feature(feature1)
        generator.register_feature(feature2)
            
        df_cleaned = pd.DataFrame({'price': [100.0, 101.0]})
            
        with pytest.warns(UserWarning, match="Failed to generate feature 'bad_feature'"):
            result = generator.generate_features(df_cleaned, ["good_feature", "bad_feature"])
            
        # Good feature should be generated
        assert "good_feature" in result.columns
        # Bad feature should not be in result
        assert "bad_feature" not in result.columns
    
    def test_generate_all_features(self):
        """Test generating all registered features."""
        generator = FeatureGenerator()
            
        feature1 = MockFeature("feature1")
        feature2 = MockFeature("feature2")
            
        generator.register_feature(feature1)
        generator.register_feature(feature2)
            
        df_cleaned = pd.DataFrame({'price': [100.0]})
            
        result = generator.generate_all_features(df_cleaned)
            
        assert isinstance(result, pd.DataFrame)
        assert "feature1" in result.columns
        assert "feature2" in result.columns


class TestUtilityFunctions:
    """Test utility functions."""
    
    @patch('src.data_processing.feature_extraction.feature_generator.pd.read_parquet')
    @patch('src.data_processing.feature_extraction.feature_generator.os.path.exists')
    def test_load_and_generate_features(self, mock_exists, mock_read_parquet):
        """Test load_and_generate_features function."""
        mock_exists.return_value = True
        
        # Mock raw data
        timestamps = pd.date_range('2023-01-01', periods=2, freq='1s')
        index = pd.MultiIndex.from_arrays([
        [timestamps[0], timestamps[1]],
        [0, 0]
        ])
        
        mock_raw_data = pd.DataFrame({
        'level-1-bid-price': [100.0, 101.0],
        'level-1-bid-volume': [10.0, 15.0]
        }, index=index)
        
        mock_read_parquet.return_value = mock_raw_data
        
        with patch.object(FeatureGenerator, 'generate_features') as mock_generate:
            mock_generate.return_value = pd.DataFrame({'mock_feature': [1, 2]})
            
            result = load_and_generate_features('ETH', 1, ['mock_feature'])
            
            assert isinstance(result, pd.DataFrame)
            mock_read_parquet.assert_called_once()
            mock_generate.assert_called_once()
    
    @patch('src.data_processing.feature_extraction.feature_generator.os.path.exists')
    def test_load_and_generate_features_file_not_found(self, mock_exists):
        """Test load_and_generate_features with missing file."""
        mock_exists.return_value = False
        
        with pytest.raises(FileNotFoundError):
            load_and_generate_features('ETH', 1)
    
    @patch('src.data_processing.feature_extraction.feature_generator.os.makedirs')
    def test_save_features(self, mock_makedirs):
        """Test save_features function."""
        features_df = pd.DataFrame({
        'feature1': [1.0, 2.0, 3.0],
        'feature2': [4.0, 5.0, 6.0]
        })
        
        with patch.object(features_df, 'to_parquet') as mock_to_parquet:
            save_features(features_df, 'ETH', 1)
            
        mock_makedirs.assert_called_once()
        mock_to_parquet.assert_called_once()
    
    def test_save_features_failure(self):
        """Test save_features when saving fails."""
        features_df = pd.DataFrame({'feature': [1, 2, 3]})
        
        with patch.object(features_df, 'to_parquet', side_effect=Exception("Save failed")):
            with pytest.raises(IOError, match="Failed to save features"):
                save_features(features_df, 'ETH', 1)


class TestIntegration:
    """Integration tests for the full pipeline."""
    
    def test_full_pipeline_mock(self):
        """Test the full feature generation pipeline with mocked data."""
        # Create realistic mock data
        timestamps = pd.date_range('2023-01-01', periods=3, freq='1s')
        index = pd.MultiIndex.from_arrays([
        [timestamps[0], timestamps[0], timestamps[1], timestamps[1], timestamps[2], timestamps[2]],
        [0, 1, 0, 1, 0, 1]
        ])
        
        raw_data = pd.DataFrame({
        'level-1-bid-price': [100.0, 100.1, 101.0, 101.1, 102.0, 102.1],
        'level-1-bid-volume': [10.0, 12.0, 15.0, 18.0, 20.0, 22.0],
        'level-1-ask-price': [101.0, 101.1, 102.0, 102.1, 103.0, 103.1],
        'level-1-ask-volume': [8.0, 10.0, 12.0, 14.0, 16.0, 18.0]
        }, index=index)
        
        # Preprocess data
        cleaned_data = preprocess_data(raw_data)
        
        # Create feature generator with mock features
        generator = FeatureGenerator()
            
        # Add simple mock features
        simple_feature = MockFeature("simple_feature", value_multiplier=1.0)
        doubled_feature = MockFeature("doubled_feature", value_multiplier=2.0)
            
        generator.register_feature(simple_feature)
        generator.register_feature(doubled_feature)
            
        # Generate features
        result = generator.generate_all_features(cleaned_data)
            
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3  # Three timestamps
        assert "simple_feature" in result.columns
        assert "doubled_feature" in result.columns
        assert "level-1-bid-price" in result.columns  # Original columns preserved
            
        # Verify feature calculations
        # simple_feature should equal the first numeric column (level-1-bid-price)
        expected_simple = cleaned_data['level-1-bid-price']
        expected_simple.name = 'simple_feature'
        pd.testing.assert_series_equal(result['simple_feature'], expected_simple)
            
        # doubled_feature should be 2x the first numeric column
        expected_doubled = cleaned_data['level-1-bid-price'] * 2
        expected_doubled.name = 'doubled_feature'
        pd.testing.assert_series_equal(result['doubled_feature'], expected_doubled)


if __name__ == '__main__':
    pytest.main([__file__])
