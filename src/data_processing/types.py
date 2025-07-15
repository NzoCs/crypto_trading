"""
Type definitions for the data processing module.

This module defines type aliases and custom types used throughout the data processing
pipeline to ensure type safety and code clarity.
"""

from typing import TypeAlias, Union, Any, Dict, List, Optional, Callable, Protocol
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

# Raw data types - aliases for pandas structures
RawData: TypeAlias = pd.DataFrame
"""
Type alias for raw financial order book data.
Represents a pandas DataFrame containing raw order book data with columns like:
- level-{i}-{side}-price: Price at level i for bid/ask side
- level-{i}-{side}-volume: Volume at level i for bid/ask side
- timestamp information in the index
"""

Feature: TypeAlias = pd.Series
"""
Type alias for a computed feature.
Represents a pandas Series containing a single feature's values indexed by timestamp.
"""

FeatureSet: TypeAlias = pd.DataFrame
"""
Type alias for a collection of features.
Represents a pandas DataFrame where each column is a feature and rows are indexed by timestamp.
"""

CleanedData: TypeAlias = pd.DataFrame
"""
Type alias for preprocessed/cleaned order book data.
Represents a pandas DataFrame containing cleaned and aggregated order book data
ready for feature extraction.
"""

# Numeric types commonly used in financial calculations
Price: TypeAlias = Union[float, np.floating[Any]]
"""Type alias for price values."""

Volume: TypeAlias = Union[float, np.floating[Any]]
"""Type alias for volume values."""

Timestamp: TypeAlias = Union[pd.Timestamp, str, int]
"""Type alias for timestamp values."""

# Feature generation function signature
FeatureGeneratorFunc: TypeAlias = Callable[[CleanedData], Feature]
"""
Type alias for feature generation functions.
Functions that take cleaned data and return a computed feature.
"""

# Configuration types
FeatureConfig: TypeAlias = Dict[str, Any]
"""Type alias for feature configuration parameters."""

class FeatureGeneratorProtocol(Protocol):
    """Protocol for feature generator classes."""
    
    name: str
    description: str
    
    def generate(self, df_cleaned: CleanedData, **kwargs: Any) -> Feature:
        """Generate the feature from cleaned data."""
        ...

# Side type for bid/ask specification
Side: TypeAlias = str  # Literal['bid', 'ask'] - using str for broader compatibility
"""Type alias for order book side specification (bid/ask)."""

# Level specification for order book levels
Level: TypeAlias = int
"""Type alias for order book level specification (1, 2, 3, ...)."""

# Window size for rolling calculations
WindowSize: TypeAlias = int
"""Type alias for time window sizes in feature calculations."""

# Feature name identifier
FeatureName: TypeAlias = str
"""Type alias for feature name identifiers."""

# Error handling types
class FeatureGenerationError(Exception):
    """Custom exception for feature generation errors."""
    
    def __init__(self, feature_name: str, message: str, original_error: Optional[Exception] = None):
        self.feature_name = feature_name
        self.original_error = original_error
        super().__init__(f"Feature '{feature_name}': {message}")

class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    
    def __init__(self, message: str, data_shape: Optional[tuple] = None):
        self.data_shape = data_shape
        super().__init__(message)

# Data validation types
class DataValidator(Protocol):
    """Protocol for data validation functions."""
    
    def validate(self, data: Union[RawData, CleanedData]) -> bool:
        """Validate the input data structure and content."""
        ...
    
    def get_validation_errors(self, data: Union[RawData, CleanedData]) -> List[str]:
        """Get list of validation error messages."""
        ...

# Feature metadata
class FeatureMetadata:
    """Metadata for feature information."""
    
    def __init__(
        self,
        name: FeatureName,
        description: str,
        data_type: type,
        requires_columns: List[str],
        window_size: Optional[WindowSize] = None,
        side: Optional[Side] = None,
        level: Optional[Level] = None
    ):
        self.name = name
        self.description = description
        self.data_type = data_type
        self.requires_columns = requires_columns
        self.window_size = window_size
        self.side = side
        self.level = level
    
    def __repr__(self) -> str:
        return f"FeatureMetadata(name='{self.name}', type={self.data_type.__name__})"

# Type for feature registry
FeatureRegistry: TypeAlias = Dict[FeatureName, FeatureGeneratorProtocol]
"""Type alias for the feature registry mapping feature names to generators."""
