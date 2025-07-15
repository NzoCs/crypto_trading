"""
Feature Extraction Module

This module provides automatic feature extraction capabilities for financial order book data.
All feature classes are automatically registered when this module is imported.

The module includes:
- BaseFeature: Abstract base class for all features
- FeatureGenerator: Main class for managing and generating features
- FeatureRegistryMeta: Metaclass for automatic feature registration
- All concrete feature implementations

Usage:
    from src.data_processing.feature_extraction import FeatureGenerator
    
    generator = FeatureGenerator()
    features = generator.generate_all_features(cleaned_data)
"""

# Import the registry system first
from .feature_registry import FeatureRegistryMeta, get_feature_registry_info

# Import base classes
from .base import BaseFeature

# Import the main feature generator
from .feature_generator import FeatureGenerator

# Import all feature implementations to trigger auto-registration via metaclass
# This ensures all features are available when the module is imported
from .features import (
    # Basic market features
    spread,
    bid_ask_imbalance,
    midprice,
    
    # Order book features
    book_slope,
    vwap,
    liquidity_ratio,
    volume,
    cumulative_volume,
    
    # Technical analysis features
    volatility,
    momentum,
    trend,
    
    # Return-based features
    inst_return,
    returns_all_signed_for_xms,
    cumulative_return,
    cumulative_return_transfer_entropy,
    
    # Advanced features
    sharpe_ratio,
    sharpe_ratio_transfer_entropy,
    priceincreases,
    timeavg,
    itincreases,
)

# Import specific feature classes for direct access
from .features.spread import SpreadFeature
from .features.bid_ask_imbalance import BidAskImbalanceFeature
from .features.book_slope import BookSlopeFeature
from .features.vwap import VWAPFeature
from .features.liquidity_ratio import LiquidityRatioFeature
from .features.volatility import VolatilityFeature
from .features.momentum import MomentumFeature
from .features.trend import TrendFeature
from .features.volume import VolumeFeature
from .features.cumulative_volume import CumulativeVolumeFeature
from .features.inst_return import InstReturnFeature
from .features.returns_all_signed_for_xms import ReturnsAllSignedForXmsFeature
from .features.cumulative_return import CumulativeReturnVsVolatilityFeature
from .features.cumulative_return_transfer_entropy import CumulativeReturnTransferEntropy
from .features.sharpe_ratio import SharpeRatioClassificationFeature
from .features.sharpe_ratio_transfer_entropy import SharpeRatioTransferEntropy
from .features.priceincreases import PriceIncreasesFeature
from .features.timeavg import TimeAvgFeature
from .features.midprice import MidPriceFeature
from .features.itincreases import ItIncreasesFeature

# Export main classes and functions
__all__ = [
    # Main classes
    'FeatureGenerator',
    'BaseFeature',
    'FeatureRegistryMeta',
    
    # Utility functions
    'get_feature_registry_info',
    
    # All feature classes
    'SpreadFeature',
    'BidAskImbalanceFeature',
    'BookSlopeFeature',
    'VWAPFeature',
    'LiquidityRatioFeature',
    'VolatilityFeature',
    'MomentumFeature',
    'TrendFeature',
    'VolumeFeature',
    'CumulativeVolumeFeature',
    'InstReturnFeature',
    'ReturnsAllSignedForXmsFeature',
    'CumulativeReturnVsVolatilityFeature',
    'CumulativeReturnTransferEntropy',
    'SharpeRatioClassificationFeature',
    'SharpeRatioTransferEntropy',
    'PriceIncreasesFeature',
    'TimeAvgFeature',
    'MidPriceFeature',
    'ItIncreasesFeature',
]

# Print registration info when module is imported
import warnings

def _show_registration_info():
    """Show information about registered features when module is imported."""
    try:
        registry_info = get_feature_registry_info()
        total_features = registry_info['total_classes']
        print(f"✓ Feature extraction module loaded with {total_features} registered feature classes")
        
        if total_features > 0:
            print(f"  Registered features: {', '.join(registry_info['class_names'][:5])}{'...' if total_features > 5 else ''}")
    except Exception as e:
        warnings.warn(f"Could not display registration info: {e}")

# Show registration info (can be disabled by setting environment variable)
import os
if os.getenv('FEATURE_EXTRACTION_SILENT') != '1':
    _show_registration_info()
