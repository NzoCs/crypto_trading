"""
Feature Generation Module for Financial Order Book Data

This module provides a flexible and extensible system for generating features
from preprocessed order book data. Features can be generated individually or
in batches, and new features can be easily added.

This module is fully type-annotated and type-checked with mypy for robustness.
"""

import sys
import os
import traceback
import pandas as pd
import numpy as np
from typing import Dict, List, Callable, Optional, Any
from abc import ABC, abstractmethod
import warnings

# Import our custom types
from ..types import (
    RawData,
    CleanedData,
    Feature,
    FeatureSet,
    FeatureName,
    FeatureRegistry,
    FeatureGenerationError,
    DataValidationError,
    Price,
    Volume,
    WindowSize,
    Level,
    Side
)

from .base import BaseFeature
from .feature_registry import FeatureRegistryMeta

# Note: Feature auto-registration is now handled by the __init__.py file
# when the feature_extraction module is imported. All features are 
# automatically discovered and registered via the metaclass system.

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class FeatureGenerator:
    """
    Main feature generator class that manages all features.
    
    This class maintains a registry of feature generators and provides methods
    to generate features individually or in batches with full type safety.
    """
    
    def __init__(self) -> None:
        self.features: FeatureRegistry = {}
        self._register_default_features()
    
    def _register_default_features(self) -> None:
        """Register all default features using the auto-discovered feature classes."""
        registered_feature_classes = FeatureRegistryMeta.get_registered_features()
        
        if not registered_feature_classes:
            warnings.warn(
                "No feature classes found in registry. "
                "FeatureGenerator initialized without default features."
            )
            return
        
        print(f"Found {len(registered_feature_classes)} registered feature classes: {list(registered_feature_classes.keys())}")
        
        # Default parameters
        n_levels: Level = 5
        window: WindowSize = 20
        
        # Helper function to safely register features
        def safe_register(feature_class_name: str, *args, **kwargs):
            """Safely register a feature instance if the class is available."""
            try:
                feature_instance = FeatureRegistryMeta.create_feature_instance(feature_class_name, *args, **kwargs)
                self.register_feature(feature_instance)
                print(f"Registered: {feature_instance.name}")
            except Exception as e:
                warnings.warn(f"Failed to register {feature_class_name}: {str(e)}")
        
        # Register all default features safely using the new registry system
        safe_register('BidAskImbalanceFeature', n_levels)
        safe_register('SpreadFeature')
        
        # Book slope features
        safe_register('BookSlopeFeature', 'bid', n_levels)
        safe_register('BookSlopeFeature', 'ask', n_levels)
        
        # VWAP features
        safe_register('VWAPFeature', 'bid', n_levels)
        safe_register('VWAPFeature', 'ask', n_levels)
        
        # Liquidity ratio
        safe_register('LiquidityRatioFeature', n_levels)
        
        # Rate features
        safe_register('VolatilityFeature', window)
        safe_register('MomentumFeature', window)
        safe_register('TrendFeature', window)
        
        # Volume features
        safe_register('VolumeFeature', 'bid', 1, window)
        safe_register('VolumeFeature', 'ask', 1, window)
        
        # Cumulative volume features
        safe_register('CumulativeVolumeFeature', 'bid', n_levels)
        safe_register('CumulativeVolumeFeature', 'ask', n_levels)

        safe_register('InstReturnFeature')

        safe_register('ReturnsAllSignedForXmsFeature', 5)
        safe_register('ReturnsAllSignedForXmsFeature', 10)
        safe_register('ReturnsAllSignedForXmsFeature', 20)

        # Cumulative Return 
        safe_register('CumulativeReturnVsVolatilityFeature', 5)
        safe_register('CumulativeReturnVsVolatilityFeature', 10)
        safe_register('CumulativeReturnVsVolatilityFeature', 20)
        safe_register('CumulativeReturnTransferEntropy', 5)

        # Sharpe ratio
        safe_register('SharpeRatioClassificationFeature', 5)
        safe_register('SharpeRatioTransferEntropy', 5)

        # Price increases
        safe_register('PriceIncreasesFeature', 1000, 10)
        safe_register('PriceIncreasesFeature', 500, 10)
        safe_register('PriceIncreasesFeature', 250, 10)
        safe_register('PriceIncreasesFeature', 100, 10)
        safe_register('PriceIncreasesFeature', 50, 10)

        safe_register('PriceIncreasesFeature', 1000, 5)
        safe_register('PriceIncreasesFeature', 500, 5)
        safe_register('PriceIncreasesFeature', 250, 5)
        safe_register('PriceIncreasesFeature', 100, 5)
        safe_register('PriceIncreasesFeature', 50, 5)

        # Complex features that depend on other features
        try:
            cumulative_volume_bid = FeatureRegistryMeta.create_feature_instance('CumulativeVolumeFeature', 'bid', n_levels)
            cumulative_volume_ask = FeatureRegistryMeta.create_feature_instance('CumulativeVolumeFeature', 'ask', n_levels)
            book_slope_bid = FeatureRegistryMeta.create_feature_instance('BookSlopeFeature', 'bid', n_levels)
            book_slope_ask = FeatureRegistryMeta.create_feature_instance('BookSlopeFeature', 'ask', n_levels)
            liquidity_ratio = FeatureRegistryMeta.create_feature_instance('LiquidityRatioFeature', n_levels)
            
            safe_register('TimeAvgFeature', cumulative_volume_bid, 250)
            safe_register('TimeAvgFeature', cumulative_volume_ask, 250)
            safe_register('TimeAvgFeature', book_slope_bid, 250)
            safe_register('TimeAvgFeature', book_slope_ask, 250)
            safe_register('TimeAvgFeature', liquidity_ratio, 250)
        except Exception as e:
            warnings.warn(f"Failed to register complex TimeAvg features: {str(e)}")

        try:
            mid_price = FeatureRegistryMeta.create_feature_instance('MidPriceFeature')
            time_avg_mid_price = FeatureRegistryMeta.create_feature_instance('TimeAvgFeature', mid_price, 10)
            
            safe_register('TimeAvgFeature', mid_price, 10)
            safe_register('ItIncreasesFeature', time_avg_mid_price, 200, 5)
        except Exception as e:
            warnings.warn(f"Failed to register complex ItIncreases features: {str(e)}")
        
        print(f"Feature registration complete. Total registered features: {len(self.features)}")


    def register_feature(self, feature: BaseFeature) -> None:
        """
        Register a new feature.
        
        Args:
            feature: Feature generator instance to register
        """
        self.features[feature.name] = feature
    
    def unregister_feature(self, feature_name: FeatureName) -> None:
        """
        Unregister a feature.
        
        Args:
            feature_name: Name of the feature to unregister
        """
        if feature_name in self.features:
            del self.features[feature_name]
    
    def list_features(self) -> List[FeatureName]:
        """
        List all registered feature names.
        
        Returns:
            List of registered feature names
        """
        return list(self.features.keys())
    
    def list_available_feature_classes(self) -> List[str]:
        """
        List all available feature classes from the registry metaclass.
        
        Returns:
            List of available feature class names
        """
        return list(FeatureRegistryMeta.get_registered_features().keys())
    
    def create_feature_instance_from_registry(self, class_name: str, *args, **kwargs) -> BaseFeature:
        """
        Create a feature instance from the registry and optionally register it.
        
        Args:
            class_name: Name of the feature class
            *args: Arguments to pass to the feature constructor
            **kwargs: Keyword arguments to pass to the feature constructor
            
        Returns:
            Created feature instance
        """
        return FeatureRegistryMeta.create_feature_instance(class_name, *args, **kwargs)
    
    def get_feature_info(self, feature_name: FeatureName) -> str:
        """
        Get description of a feature.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Feature description or error message
        """
        if feature_name in self.features:
            return self.features[feature_name].description
        return f"Feature '{feature_name}' not found"
    
    def generate_feature(self, feature_name: FeatureName, df_cleaned: CleanedData, **kwargs: Any) -> Feature:
        """
        Generate a single feature.
        
        Args:
            feature_name: Name of the feature to generate
            df_cleaned: Preprocessed order book data
            **kwargs: Additional parameters for feature generation
            
        Returns:
            Generated feature as a pandas Series
            
        Raises:
            FeatureGenerationError: If feature is not registered or generation fails
        """
        if feature_name not in self.features:
            raise FeatureGenerationError(feature_name, "Feature is not registered")
        
        try:
            return self.features[feature_name].generate(df_cleaned, **kwargs)
        except Exception as e:
            raise FeatureGenerationError(feature_name, f"Generation failed: {str(e)}", e)
    
    def generate_features(
        self, 
        df_cleaned: CleanedData, 
        feature_names: Optional[List[FeatureName]] = None, 
        **kwargs: Any
    ) -> FeatureSet:
        """
        Generate multiple features.
        
        Args:
            df_cleaned: Preprocessed order book data
            feature_names: List of specific features to generate (None for all)
            **kwargs: Additional parameters for feature generation
            
        Returns:
            DataFrame containing generated features
        """
        if feature_names is None:
            feature_names = self.list_features()
        
        features_df = pd.DataFrame(index=df_cleaned.index)
        
        for feature_name in feature_names:
            if feature_name in self.features:
                try:
                    features_df[feature_name] = self.generate_feature(feature_name, df_cleaned, **kwargs)
                except FeatureGenerationError as e:
                    traceback.print_exc()
                    warnings.warn(f"Failed to generate feature '{feature_name}': {str(e)}")
            else:
                warnings.warn(f"Feature '{feature_name}' is not registered")
        
        for col in df_cleaned.columns:
            features_df[col] = df_cleaned[col]
        
        return features_df
    
    def generate_all_features(self, df_cleaned: CleanedData, **kwargs: Any) -> FeatureSet:
        """
        Generate all registered features.
        
        Args:
            df_cleaned: Preprocessed order book data
            **kwargs: Additional parameters for feature generation
            
        Returns:
            DataFrame containing all generated features
        """
        return self.generate_features(df_cleaned, **kwargs)


def preprocess_data(df: RawData) -> CleanedData:
    """
    Preprocess the raw order book data to create df_cleaned.
    
    This function computes weighted average prices and volumes for each level and side
    by grouping data by timestamp (ignoring row_id).
    
    Args:
        df: Raw order book data containing price and volume levels
        
    Returns:
        CleanedData: Preprocessed data ready for feature extraction
        
    Raises:
        DataValidationError: If input data is invalid or missing required columns
    """
    if df.empty:
        raise DataValidationError("Input DataFrame is empty")
    
    try:
        df_cleaned: CleanedData = pd.DataFrame(index=df.index.get_level_values(0).unique())
        
        # For each level and side, compute the volume-weighted average price per timestamp
        for side in ['bid', 'ask']:
            for i in range(1, 11):  # Assuming 10 levels
                price_col = f'level-{i}-{side}-price'
                volume_col = f'level-{i}-{side}-volume'
                
                # Check if columns exist in the data
                if price_col in df.columns and volume_col in df.columns:
                    # Weighted average price per timestamp
                    weighted_avg_price: pd.Series = (
                        df[price_col] * df[volume_col]
                    ).groupby(level=0).sum() / df[volume_col].groupby(level=0).sum()
                    
                    # Average volume per timestamp
                    avg_volume: pd.Series = df[volume_col].groupby(level=0).mean()
                    
                    # Assign to df_cleaned
                    df_cleaned[price_col] = weighted_avg_price
                    df_cleaned[volume_col] = avg_volume
        
        return df_cleaned
    except Exception as e:
        raise DataValidationError(f"Failed to preprocess data: {str(e)}", df.shape)


def load_and_generate_features(
    coin: str, 
    data_version: int, 
    feature_names: Optional[List[FeatureName]] = None
) -> FeatureSet:
    """
    Load data and generate features in one step.
    
    Args:
        coin: Coin symbol (e.g., 'ETH', 'XBT')
        data_version: Data version number (e.g., 0, 1, 2)
        feature_names: List of specific features to generate (None for all)
    
    Returns:
        DataFrame with generated features
        
    Raises:
        FileNotFoundError: If data file doesn't exist
        DataValidationError: If data loading or preprocessing fails
    """
    # Load preprocessed data
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, 'data', 'preprocessed', f'DATA_{data_version}', f'{coin}_EUR.parquet')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    try:
        df: RawData = pd.read_parquet(data_path)
        
        # Preprocess data
        df_cleaned: CleanedData = preprocess_data(df)
        
        # Generate features
        generator = FeatureGenerator()
        features: FeatureSet = generator.generate_features(df_cleaned, feature_names)
        
        return features
    except Exception as e:
        raise DataValidationError(f"Failed to load and generate features: {str(e)}")


def save_features(features: FeatureSet, coin: str, data_version: int, output_dir: Optional[str] = None) -> None:
    """
    Save features to parquet file.
    
    Args:
        features: DataFrame containing computed features
        coin: Coin symbol (e.g., 'ETH', 'XBT')
        data_version: Data version number
        output_dir: Output directory (None for default)
        
    Raises:
        IOError: If saving fails
    """
    import os
    
    if output_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        output_dir = os.path.join(project_root, 'data', 'features')
    
    output_path = os.path.join(output_dir, f'DATA_{data_version}')
    os.makedirs(output_path, exist_ok=True)
    
    file_path = os.path.join(output_path, f'{coin}_EUR.parquet')
    
    try:
        features.to_parquet(file_path, index=True)
        print(f"Features saved to {file_path}")
    except Exception as e:
        raise IOError(f"Failed to save features to {file_path}: {str(e)}")


def recompute_feature(
    coin: str, 
    data_version: int, 
    feature_name: FeatureName, 
    output_dir: Optional[str] = None
) -> None:
    """
    Recompute a single feature for a coin and data version, updating the features file in place.
    Other features are left untouched and still present.

    If the features file exists, it is assumed to already contain the correct index and columns (from clean_df).
    If the required columns for the feature are missing, the function will load and preprocess the raw data to obtain them.

    Args:
        coin: Coin symbol (e.g., 'ETH', 'XBT')
        data_version: Data version number (e.g., 0, 1, 2)
        feature_name: Name of the feature to recompute
        output_dir: Directory where features are stored (default: project_root/data/features)
        
    Raises:
        FeatureGenerationError: If feature is not registered
        FileNotFoundError: If required data files don't exist
        IOError: If file operations fail
    """
    import os
    import pandas as pd

    # Set up paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if output_dir is None:
        output_dir = os.path.join(project_root, 'data', 'features')
    output_path = os.path.join(output_dir, f'DATA_{data_version}')
    os.makedirs(output_path, exist_ok=True)
    features_file = os.path.join(output_path, f'{coin}_EUR.parquet')

    generator = FeatureGenerator()
    if feature_name not in generator.features:
        raise FeatureGenerationError(feature_name, "Feature is not registered")

    # If features file exists, use its index and columns for recomputation
    if os.path.exists(features_file):
        try:
            features: FeatureSet = pd.read_parquet(features_file)
            # Try to recompute using existing features file
            new_feature: Feature = generator.features[feature_name].generate(features)
            features[feature_name] = new_feature
            features.to_parquet(features_file, index=True)
            print(f"Feature '{feature_name}' recomputed (in-place) and saved to {features_file}")
            return
        except KeyError as e:
            print(f"Missing columns in features file: {e}. Falling back to raw data.")
            # Fallback to raw data below
        except Exception as e:
            raise FeatureGenerationError(feature_name, f"Failed to recompute from existing features: {str(e)}", e)

    # If features file does not exist or required columns are missing, load and preprocess data
    data_path = os.path.join(project_root, 'data', 'preprocessed', f'DATA_{data_version}', f'{coin}_EUR.parquet')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Raw data file not found: {data_path}")
    
    try:
        df: RawData = pd.read_parquet(data_path)
        df_cleaned: CleanedData = preprocess_data(df)
        new_feature = generator.generate_feature(feature_name, df_cleaned)
        
        if os.path.exists(features_file):
            features = pd.read_parquet(features_file)
            features[feature_name] = new_feature
        else:
            features = pd.DataFrame(index=df_cleaned.index)
            features[feature_name] = new_feature
            
        features.to_parquet(features_file, index=True)
        print(f"Feature '{feature_name}' recomputed and saved to {features_file}")
    except Exception as e:
        raise FeatureGenerationError(feature_name, f"Failed to recompute from raw data: {str(e)}", e)

# Example CLI usage (add to your CLI script):
# from feature_extraction.feature_generator import recompute_feature
# recompute_feature('ETH', 1, 'spread')
    
if __name__ == "__main__":
    # Example usage
    try:
        generator = FeatureGenerator()
        
        # List all features
        print("Available features:")
        for feature_name in generator.list_features():
            print(f"  - {feature_name}: {generator.get_feature_info(feature_name)}")
        
        # Generate features for ETH data
        features: FeatureSet = load_and_generate_features('ETH', 2)
        print(f"\nGenerated {len(features.columns)} features for {len(features)} timestamps")
        print(f"Feature columns: {list(features.columns)}")
    except (DataValidationError, FeatureGenerationError, FileNotFoundError) as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
