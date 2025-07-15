from abc import ABC, abstractmethod
from typing import Any
import pandas as pd

# Import our custom types
from ..types import (
    CleanedData,
    Feature,
    FeatureName,
    FeatureMetadata,
    FeatureGeneratorProtocol
)

# Import the registry metaclass
from .feature_registry import FeatureRegistryMeta


class BaseFeature(ABC, metaclass=FeatureRegistryMeta):
    """
    Base class for all feature generators.
    
    This class defines the interface that all feature generators must implement
    and provides type safety through proper type annotations.
    
    All subclasses are automatically registered in the FeatureRegistry
    using the metaclass system.
    """
    
    def __init__(self, name: FeatureName, description: str = "") -> None:
        self.name = name
        self.description = description
        self._metadata: FeatureMetadata | None = None
    
    @abstractmethod
    def generate(self, df_cleaned: CleanedData, **kwargs: Any) -> Feature:
        """
        Generate the feature from cleaned data.
        
        Args:
            df_cleaned: Preprocessed order book data
            **kwargs: Additional parameters for feature generation
            
        Returns:
            Feature: A pandas Series containing the computed feature values
            
        Raises:
            FeatureGenerationError: If feature generation fails
        """
        pass
    
    @property
    def metadata(self) -> FeatureMetadata | None:
        """Get feature metadata if available."""
        return self._metadata
    
    def set_metadata(self, metadata: FeatureMetadata) -> None:
        """Set feature metadata."""
        self._metadata = metadata
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
