"""
IFCOB - Intelligent Financial Collective Order Book

A comprehensive financial trading and backtesting framework.
"""

__version__ = "0.1.0"
__author__ = "IFCOB Team"

# Import main modules for easier access
from . import data_processing
from . import prediction_models
from . import backtesting

# Import commonly used types and classes
from .data_processing import (
    RawData,
    Feature,
    FeatureSet,
    CleanedData,
    Price,
    Volume,
    Timestamp,
    FeatureConfig,
    Side,
    Level,
    WindowSize,
    FeatureName,
)

__all__ = [
    "__version__",
    "__author__",
    "data_processing",
    "prediction_models", 
    "backtesting",
    "RawData",
    "Feature",
    "FeatureSet", 
    "CleanedData",
    "Price",
    "Volume",
    "Timestamp",
    "FeatureConfig",
    "Side",
    "Level",
    "WindowSize",
    "FeatureName",
]