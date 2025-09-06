"""
Tests for the base feature extraction classes.

This module tests the BaseFeature abstract class and its functionality.
"""

import pytest
import pandas as pd
import numpy as np
from abc import ABC
from typing import Any

from src.data_processing.types import (
    CleanedData,
    Feature,
    FeatureName,
    FeatureMetadata,
    FeatureGenerationError
)
from src.data_processing.base_feature import BaseFeature


class ConcreteFeature(BaseFeature):
    """Concrete implementation of BaseFeature for testing."""
    
    def __init__(self, name: FeatureName, description: str = "", should_fail: bool = False):
        super().__init__(name, description)
        self.should_fail = should_fail
    
    def generate(self, df_cleaned: CleanedData, **kwargs: Any) -> Feature:
        """Generate a simple test feature."""
        if self.should_fail:
            raise ValueError("Intentional failure for testing")
        
        # Simple feature: sum of all numeric columns
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return pd.Series(np.zeros(len(df_cleaned)), index=df_cleaned.index, name=self.name)
        
        result = df_cleaned[numeric_cols].sum(axis=1)
        result.name = self.name
        return result


class TestBaseFeature:
    """Test BaseFeature abstract class."""
    
    def test_base_feature_is_abstract(self):
        """Test that BaseFeature cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseFeature("test", "test description")
    
    def test_concrete_feature_creation(self):
        """Test creating a concrete feature implementation."""
        feature = ConcreteFeature("test_feature", "Test description")
        
        assert feature.name == "test_feature"
        assert feature.description == "Test description"
        assert feature.metadata is None
    
    def test_concrete_feature_generation(self):
        """Test feature generation with concrete implementation."""
        feature = ConcreteFeature("sum_feature", "Sum of all columns")
        
        # Create test data
        df_cleaned = pd.DataFrame({
            'level-1-bid-price': [100.0, 101.0, 102.0],
            'level-1-bid-volume': [10.0, 15.0, 20.0],
            'level-1-ask-price': [101.0, 102.0, 103.0],
            'level-1-ask-volume': [12.0, 18.0, 22.0]
        })
        
        result = feature.generate(df_cleaned)
        
        assert isinstance(result, pd.Series)
        assert result.name == "sum_feature"
        assert len(result) == 3
        # Check that it's actually summing the columns
        expected = df_cleaned.sum(axis=1)
        pd.testing.assert_series_equal(result, expected, check_names=False)
    
    def test_feature_generation_with_kwargs(self):
        """Test feature generation with additional parameters."""
        feature = ConcreteFeature("test_feature")
        
        df_cleaned = pd.DataFrame({
            'price': [100.0, 101.0],
            'volume': [10.0, 15.0]
        })
        
        # Should work with additional kwargs (even if not used)
        result = feature.generate(df_cleaned, window=20, multiplier=2.0)
        
        assert isinstance(result, pd.Series)
        assert len(result) == 2
    
    def test_feature_generation_failure(self):
        """Test feature generation when it fails."""
        feature = ConcreteFeature("failing_feature", should_fail=True)
        
        df_cleaned = pd.DataFrame({'test': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Intentional failure"):
            feature.generate(df_cleaned)
    
    def test_feature_metadata_property(self):
        """Test metadata property getter."""
        feature = ConcreteFeature("test_feature")
        
        assert feature.metadata is None
        
        # Set metadata
        metadata = FeatureMetadata(
            name="test_feature",
            description="Test",
            data_type=float,
            requires_columns=["price"]
        )
        feature.set_metadata(metadata)
        
        assert feature.metadata == metadata
        assert feature.metadata.name == "test_feature"
    
    def test_feature_set_metadata(self):
        """Test setting feature metadata."""
        feature = ConcreteFeature("test_feature")
        
        metadata = FeatureMetadata(
            name="test_feature",
            description="Test feature",
            data_type=float,
            requires_columns=["price", "volume"],
            window_size=20,
            side="bid",
            level=1
        )
        
        feature.set_metadata(metadata)
        
        assert feature.metadata == metadata
        assert feature.metadata.window_size == 20
        assert feature.metadata.side == "bid"
        assert feature.metadata.level == 1
    
    def test_feature_repr(self):
        """Test feature string representation."""
        feature = ConcreteFeature("test_feature", "Test description")
        
        repr_str = repr(feature)
        assert "ConcreteFeature" in repr_str
        assert "test_feature" in repr_str
    
    def test_feature_with_empty_dataframe(self):
        """Test feature generation with empty DataFrame."""
        feature = ConcreteFeature("empty_test")
        
        df_empty = pd.DataFrame()
        
        result = feature.generate(df_empty)
        
        assert isinstance(result, pd.Series)
        assert len(result) == 0
        assert result.name == "empty_test"
    
    def test_feature_with_non_numeric_data(self):
        """Test feature generation with non-numeric data."""
        feature = ConcreteFeature("string_test")
        
        df_strings = pd.DataFrame({
            'text1': ['a', 'b', 'c'],
            'text2': ['x', 'y', 'z']
        })
        
        result = feature.generate(df_strings)
        
        assert isinstance(result, pd.Series)
        assert len(result) == 3
        assert result.name == "string_test"
        # Should be zeros since no numeric columns
        assert all(result == 0)
    
    def test_feature_with_mixed_data_types(self):
        """Test feature generation with mixed data types."""
        feature = ConcreteFeature("mixed_test")
        
        df_mixed = pd.DataFrame({
            'numeric1': [1.0, 2.0, 3.0],
            'text': ['a', 'b', 'c'],
            'numeric2': [10, 20, 30],
            'boolean': [True, False, True]
        })
        
        result = feature.generate(df_mixed)
        
        assert isinstance(result, pd.Series)
        assert len(result) == 3
        assert result.name == "mixed_test"
        # Should sum only the columns that pandas considers numeric (numeric1, numeric2)
        # Boolean columns are not included in select_dtypes(include=[np.number])
        expected = df_mixed[['numeric1', 'numeric2']].sum(axis=1)
        expected = expected.astype(float)  # Ensure float dtype to match result
        pd.testing.assert_series_equal(result, expected, check_names=False)


class TestFeatureValidation:
    """Test feature validation and error handling."""
    
    def test_feature_name_validation(self):
        """Test feature name validation."""
        # Valid names
        feature1 = ConcreteFeature("valid_name")
        feature2 = ConcreteFeature("another-valid-name")
        feature3 = ConcreteFeature("ValidName123")
        
        assert feature1.name == "valid_name"
        assert feature2.name == "another-valid-name"
        assert feature3.name == "ValidName123"
    
    def test_feature_with_index_mismatch(self):
        """Test feature generation preserves index."""
        feature = ConcreteFeature("index_test")
        
        # Create DataFrame with custom index
        custom_index = pd.date_range('2023-01-01', periods=3, freq='1h')  # Use lowercase 'h'
        df_cleaned = pd.DataFrame({
            'price': [100.0, 101.0, 102.0],
            'volume': [10.0, 15.0, 20.0]
        }, index=custom_index)
        
        result = feature.generate(df_cleaned)
        
        assert isinstance(result, pd.Series)
        pd.testing.assert_index_equal(result.index, custom_index)
        assert result.name == "index_test"


if __name__ == '__main__':
    pytest.main([__file__])
