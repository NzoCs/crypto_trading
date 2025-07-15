"""
Tests for the data types module.

This module tests the type definitions and custom classes used throughout
the data processing pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from typing import get_origin, get_args

from src.data_processing.types import (
    RawData,
    Feature,
    FeatureSet,
    CleanedData,
    Price,
    Volume,
    Timestamp,
    FeatureConfig,
    FeatureGeneratorFunc,
    Side,
    Level,
    WindowSize,
    FeatureName,
    FeatureGenerationError,
    DataValidationError,
    FeatureMetadata,
    FeatureRegistry,
)


class TestTypeAliases:
    """Test type aliases work correctly."""
    
    def test_raw_data_alias(self):
        """Test RawData type alias."""
        df = pd.DataFrame({'price': [1.0, 2.0], 'volume': [100, 200]})
        
        # Should accept DataFrame
        raw_data: RawData = df
        assert isinstance(raw_data, pd.DataFrame)
        assert len(raw_data) == 2
    
    def test_feature_alias(self):
        """Test Feature type alias."""
        series = pd.Series([1.0, 2.0, 3.0], name='test_feature')
        
        # Should accept Series
        feature: Feature = series
        assert isinstance(feature, pd.Series)
        assert feature.name == 'test_feature'
    
    def test_feature_set_alias(self):
        """Test FeatureSet type alias."""
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [4.0, 5.0, 6.0]
        })
        
        # Should accept DataFrame
        feature_set: FeatureSet = df
        assert isinstance(feature_set, pd.DataFrame)
        assert list(feature_set.columns) == ['feature1', 'feature2']
    
    def test_cleaned_data_alias(self):
        """Test CleanedData type alias."""
        df = pd.DataFrame({
            'level-1-bid-price': [100.0, 101.0],
            'level-1-bid-volume': [10.0, 15.0]
        })
        
        # Should accept DataFrame
        cleaned_data: CleanedData = df
        assert isinstance(cleaned_data, pd.DataFrame)
    
    def test_numeric_types(self):
        """Test numeric type aliases."""
        # Price type
        price1: Price = 100.5
        price2: Price = np.float64(100.5)
        assert isinstance(price1, (float, int))
        assert isinstance(price2, np.floating)
        
        # Volume type
        volume1: Volume = 1000.0
        volume2: Volume = np.float32(1000.0)
        assert isinstance(volume1, (float, int))
        assert isinstance(volume2, np.floating)
    
    def test_specification_types(self):
        """Test specification type aliases."""
        side: Side = 'bid'
        assert isinstance(side, str)
        assert side in ['bid', 'ask']
        
        level: Level = 1
        assert isinstance(level, int)
        assert level > 0
        
        window: WindowSize = 20
        assert isinstance(window, int)
        assert window > 0
        
        name: FeatureName = 'test_feature'
        assert isinstance(name, str)


class TestFeatureMetadata:
    """Test FeatureMetadata class."""
    
    def test_feature_metadata_creation(self):
        """Test creating FeatureMetadata instance."""
        metadata = FeatureMetadata(
            name='test_feature',
            description='A test feature',
            data_type=float,
            requires_columns=['price', 'volume'],
            window_size=20,
            side='bid',
            level=1
        )
        
        assert metadata.name == 'test_feature'
        assert metadata.description == 'A test feature'
        assert metadata.data_type == float
        assert metadata.requires_columns == ['price', 'volume']
        assert metadata.window_size == 20
        assert metadata.side == 'bid'
        assert metadata.level == 1
    
    def test_feature_metadata_optional_fields(self):
        """Test FeatureMetadata with optional fields."""
        metadata = FeatureMetadata(
            name='simple_feature',
            description='Simple feature',
            data_type=int,
            requires_columns=['data']
        )
        
        assert metadata.name == 'simple_feature'
        assert metadata.window_size is None
        assert metadata.side is None
        assert metadata.level is None
    
    def test_feature_metadata_repr(self):
        """Test FeatureMetadata string representation."""
        metadata = FeatureMetadata(
            name='test',
            description='Test',
            data_type=float,
            requires_columns=[]
        )
        
        repr_str = repr(metadata)
        assert 'FeatureMetadata' in repr_str
        assert 'test' in repr_str
        assert 'float' in repr_str


class TestCustomExceptions:
    """Test custom exception classes."""
    
    def test_feature_generation_error(self):
        """Test FeatureGenerationError exception."""
        original_error = ValueError("Original error")
        
        error = FeatureGenerationError(
            feature_name='test_feature',
            message='Generation failed',
            original_error=original_error
        )
        
        assert error.feature_name == 'test_feature'
        assert error.original_error == original_error
        assert 'test_feature' in str(error)
        assert 'Generation failed' in str(error)
    
    def test_feature_generation_error_without_original(self):
        """Test FeatureGenerationError without original error."""
        error = FeatureGenerationError(
            feature_name='test_feature',
            message='Simple failure'
        )
        
        assert error.feature_name == 'test_feature'
        assert error.original_error is None
        assert 'test_feature' in str(error)
        assert 'Simple failure' in str(error)
    
    def test_data_validation_error(self):
        """Test DataValidationError exception."""
        data_shape = (100, 5)
        
        error = DataValidationError(
            message='Invalid data format',
            data_shape=data_shape
        )
        
        assert error.data_shape == data_shape
        assert 'Invalid data format' in str(error)
    
    def test_data_validation_error_without_shape(self):
        """Test DataValidationError without data shape."""
        error = DataValidationError('Simple validation error')
        
        assert error.data_shape is None
        assert 'Simple validation error' in str(error)


class TestProtocolCompatibility:
    """Test that our types work with protocol definitions."""
    
    def test_feature_generator_protocol_compatibility(self):
        """Test that a class can implement FeatureGeneratorProtocol."""
        from src.data_processing.types import FeatureGeneratorProtocol
        
        class MockFeatureGenerator:
            def __init__(self):
                self.name = 'mock_feature'
                self.description = 'Mock feature for testing'
            
            def generate(self, df_cleaned: CleanedData, **kwargs) -> Feature:
                return pd.Series([1.0, 2.0, 3.0], name=self.name)
        
        # Should be compatible with protocol
        generator = MockFeatureGenerator()
        assert hasattr(generator, 'name')
        assert hasattr(generator, 'description')
        assert hasattr(generator, 'generate')
        
        # Test actual usage
        test_data = pd.DataFrame({'test': [1, 2, 3]})
        result = generator.generate(test_data)
        assert isinstance(result, pd.Series)
        assert result.name == 'mock_feature'


if __name__ == '__main__':
    pytest.main([__file__])
