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
from src.data_processing.base_feature import BaseFeature
from src.data_processing.feature_generator import FeatureGenerator


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


class TestFeatureGenerator:
    """Test the FeatureGenerator class."""
    
    def test_feature_generator_creation(self):
        """Test creating a FeatureGenerator instance."""
        generator = FeatureGenerator()
        assert isinstance(generator.features, dict)
        
        # Verify features can be listed
        feature_names = generator.list_features()
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


if __name__ == '__main__':
    pytest.main([__file__])
