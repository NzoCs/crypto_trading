"""
Feature Registry Module

This module provides the metaclass system for automatic feature registration.
All feature classes that inherit from BaseFeature will be automatically
registered when they are imported, eliminating the need for manual registration.
"""

from typing import Dict, Any
from abc import ABCMeta
import warnings


class FeatureRegistryMeta(ABCMeta):
    """
    Metaclass that automatically registers feature classes in a global registry.
    
    This metaclass inherits from ABCMeta to resolve metaclass conflicts with ABC.
    When a class inherits from BaseFeature and uses this metaclass, it will be
    automatically registered in the FEATURE_REGISTRY when the class is defined.
    This eliminates the need for manual imports and registration.
    """
    
    # Global registry to store all feature classes
    FEATURE_REGISTRY: Dict[str, type] = {}
    
    def __new__(mcs, name: str, bases: tuple, namespace: dict, **kwargs):
        # Create the class normally
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        
        # Only register concrete feature classes (not the base class itself)
        if name != 'BaseFeature' and bases and any(hasattr(base, 'generate') for base in bases):
            # Register the class in the global registry
            mcs.FEATURE_REGISTRY[name] = cls
            print(f"Auto-registered feature class: {name}")
        
        return cls
    
    @classmethod
    def get_registered_features(mcs) -> Dict[str, type]:
        """Get all registered feature classes."""
        return mcs.FEATURE_REGISTRY.copy()
    
    @classmethod
    def clear_registry(mcs) -> None:
        """Clear the feature registry (useful for testing)."""
        mcs.FEATURE_REGISTRY.clear()
    
    @classmethod
    def create_feature_instance(mcs, class_name: str, *args, **kwargs):
        """Create an instance of a registered feature class."""
        if class_name not in mcs.FEATURE_REGISTRY:
            raise ValueError(
                f"Feature class '{class_name}' not found in registry. "
                f"Available: {list(mcs.FEATURE_REGISTRY.keys())}"
            )
        
        feature_class = mcs.FEATURE_REGISTRY[class_name]
        return feature_class(*args, **kwargs)
    
    @classmethod
    def register_feature_class(mcs, feature_class: type) -> None:
        """Manually register a feature class (useful for testing)."""
        class_name = feature_class.__name__
        mcs.FEATURE_REGISTRY[class_name] = feature_class
        print(f"Manually registered feature class: {class_name}")
    
    @classmethod
    def unregister_feature_class(mcs, class_name: str) -> None:
        """Manually unregister a feature class (useful for testing)."""
        if class_name in mcs.FEATURE_REGISTRY:
            del mcs.FEATURE_REGISTRY[class_name]
            print(f"Unregistered feature class: {class_name}")
    
    @classmethod
    def is_registered(mcs, class_name: str) -> bool:
        """Check if a feature class is registered."""
        return class_name in mcs.FEATURE_REGISTRY
    
    @classmethod
    def get_registry_size(mcs) -> int:
        """Get the number of registered feature classes."""
        return len(mcs.FEATURE_REGISTRY)


def get_feature_registry_info() -> Dict[str, Any]:
    """
    Get comprehensive information about the feature registry.
    
    Returns:
        Dictionary with registry information including class names, counts, etc.
    """
    registry = FeatureRegistryMeta.get_registered_features()
    
    return {
        'total_classes': len(registry),
        'class_names': list(registry.keys()),
        'class_info': {
            name: {
                'module': cls.__module__,
                'qualname': cls.__qualname__,
                'doc': cls.__doc__
            }
            for name, cls in registry.items()
        }
    }
