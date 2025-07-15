"""
Data processing module for financial order book analysis.

This module provides type-safe data processing capabilities for:
- Raw data preprocessing
- Feature extraction
- Data validation

Key type aliases:
- RawData: Raw order book data (pandas DataFrame)
- Feature: Computed feature series (pandas Series)  
- FeatureSet: Collection of features (pandas DataFrame)
- CleanedData: Preprocessed order book data (pandas DataFrame)
"""

from .types import (
    # Core data types
    RawData,
    Feature,
    FeatureSet,
    CleanedData,
    
    # Numeric types
    Price,
    Volume,
    Timestamp,
    
    # Configuration types
    FeatureConfig,
    FeatureGeneratorFunc,
    FeatureGeneratorProtocol,
    
    # Specification types
    Side,
    Level,
    WindowSize,
    FeatureName,
    
    # Error types
    FeatureGenerationError,
    DataValidationError,
    
    # Validation types
    DataValidator,
    
    # Metadata types
    FeatureMetadata,
    FeatureRegistry,
)

__all__ = [
    # Core data types
    "RawData",
    "Feature", 
    "FeatureSet",
    "CleanedData",
    
    # Numeric types
    "Price",
    "Volume", 
    "Timestamp",
    
    # Configuration types
    "FeatureConfig",
    "FeatureGeneratorFunc",
    "FeatureGeneratorProtocol",
    
    # Specification types
    "Side",
    "Level",
    "WindowSize", 
    "FeatureName",
    
    # Error types
    "FeatureGenerationError",
    "DataValidationError",
    
    # Validation types
    "DataValidator",
    
    # Metadata types
    "FeatureMetadata",
    "FeatureRegistry",
]
