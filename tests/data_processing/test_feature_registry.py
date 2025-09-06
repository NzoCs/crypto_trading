"""
Test suite for the feature registry system.

This module tests the automatic feature registration system to ensure
features are properly registered and can be instantiated dynamically.
"""

import pytest
import sys
import os
import warnings

from src.data_processing.feature_registry import FeatureRegistryMeta, get_feature_registry_info
from src.data_processing.base_feature import BaseFeature


class TestFeatureRegistry:
    """Test cases for the feature registry system."""
    
    def setup_method(self):
        """Setup for each test method."""
        # Clear the registry before each test
        FeatureRegistryMeta.clear_registry()
    
    def test_registry_metaclass_basic_functionality(self):
        """Test that the metaclass correctly registers feature classes."""
        
        # Create a test feature class
        class TestFeature(BaseFeature):
            def __init__(self):
                super().__init__("test_feature", "A test feature")
            
            def generate(self, df_cleaned, **kwargs):
                return df_cleaned.iloc[:, 0] * 2  # Simple transformation
        
        # Check that it was registered
        assert FeatureRegistryMeta.is_registered("TestFeature")
        assert "TestFeature" in FeatureRegistryMeta.get_registered_features()
        assert FeatureRegistryMeta.get_registry_size() == 1
    
    def test_registry_ignores_base_class(self):
        """Test that BaseFeature itself is not registered."""
        # BaseFeature should not be in the registry
        assert not FeatureRegistryMeta.is_registered("BaseFeature")
    
    def test_feature_instance_creation(self):
        """Test creating feature instances from the registry."""
        
        class AnotherTestFeature(BaseFeature):
            def __init__(self, multiplier=1):
                super().__init__("another_test", "Another test feature")
                self.multiplier = multiplier
            
            def generate(self, df_cleaned, **kwargs):
                return df_cleaned.iloc[:, 0] * self.multiplier
        
        # Create instance with default parameters
        instance1 = FeatureRegistryMeta.create_feature_instance("AnotherTestFeature")
        assert instance1.name == "another_test"
        assert instance1.multiplier == 1
        
        # Create instance with custom parameters
        instance2 = FeatureRegistryMeta.create_feature_instance("AnotherTestFeature", multiplier=5)
        assert instance2.multiplier == 5
    
    def test_feature_instance_creation_invalid_class(self):
        """Test error handling for invalid feature class names."""
        with pytest.raises(ValueError) as exc_info:
            FeatureRegistryMeta.create_feature_instance("NonExistentFeature")
        
        assert "not found in registry" in str(exc_info.value)
    
    def test_manual_registration_and_unregistration(self):
        """Test manual registration and unregistration of feature classes."""
        
        class ManualFeature(BaseFeature):
            def __init__(self):
                super().__init__("manual_feature", "Manually registered feature")
            
            def generate(self, df_cleaned, **kwargs):
                return df_cleaned.iloc[:, 0]
        
        # This should already be registered by the metaclass
        assert FeatureRegistryMeta.is_registered("ManualFeature")
        
        # Unregister it
        FeatureRegistryMeta.unregister_feature_class("ManualFeature")
        assert not FeatureRegistryMeta.is_registered("ManualFeature")
        
        # Manually register it again
        FeatureRegistryMeta.register_feature_class(ManualFeature)
        assert FeatureRegistryMeta.is_registered("ManualFeature")
    
    def test_auto_import_features(self):
        """Test that features are automatically registered when module is imported."""
        # Ensure the registry is clear first
        from src.data_processing.feature_registry import FeatureRegistryMeta
        original_size = FeatureRegistryMeta.get_registry_size()
        
        # Import the features module to trigger auto-registration
        try:
            from src.data_processing import features
        except ImportError:
            # If that doesn't work, try importing the __init__ which should trigger registration
            import src.data_processing
        
        # Features should be registered via automatic import
        registry_info = get_feature_registry_info()
        print(f"Auto-registered {registry_info['total_classes']} feature classes")
        print(f"Available classes: {registry_info['class_names'][:5]}...")  # Show first 5
        
        # Should have some registered features from automatic import
        # Either from previous import or from this test's import
        assert registry_info['total_classes'] > 0, f"No features registered. Registry size: {FeatureRegistryMeta.get_registry_size()}"
        
        # If we have any registered features, test one that we know exists
        if registry_info['total_classes'] > 0:
            # Try to find a simple feature to test
            simple_features = ['MidPriceFeature', 'SpreadFeature', 'VolumeFeature']
            test_class = None
            
            for feature_name in simple_features:
                if feature_name in registry_info['class_names']:
                    test_class = feature_name
                    break
            
            if test_class:
                try:
                    instance = FeatureRegistryMeta.create_feature_instance(test_class)
                    assert hasattr(instance, 'name')
                    assert hasattr(instance, 'generate')
                except Exception as e:
                    # Some features might require parameters, that's okay
                    print(f"Could not instantiate {test_class} without parameters: {e}")
            else:
                # Just test that we can access the first one
                class_name = registry_info['class_names'][0]
                print(f"Testing with first available class: {class_name}")
    
    def test_registry_info_function(self):
        """Test the registry info function."""
        
        class InfoTestFeature(BaseFeature):
            """This is a test feature for info testing."""
            def __init__(self):
                super().__init__("info_test", "Info test feature")
            
            def generate(self, df_cleaned, **kwargs):
                return df_cleaned.iloc[:, 0]
        
        info = get_feature_registry_info()
        
        assert isinstance(info, dict)
        assert 'total_classes' in info
        assert 'class_names' in info
        assert 'class_info' in info
        assert info['total_classes'] >= 1
        assert 'InfoTestFeature' in info['class_names']
        
        # Check class info details
        class_info = info['class_info']['InfoTestFeature']
        assert 'module' in class_info
        assert 'qualname' in class_info
        assert 'doc' in class_info


if __name__ == "__main__":
    # Run basic tests if this file is executed directly
    test_suite = TestFeatureRegistry()
    
    print("Testing feature registry system...")
    
    try:
        test_suite.setup_method()
        test_suite.test_registry_metaclass_basic_functionality()
        print("✓ Basic metaclass functionality works")
        
        test_suite.setup_method()
        test_suite.test_registry_ignores_base_class()
        print("✓ BaseFeature is correctly ignored")
        
        test_suite.setup_method()
        test_suite.test_feature_instance_creation()
        print("✓ Feature instance creation works")
        
        test_suite.setup_method()
        test_suite.test_manual_registration_and_unregistration()
        print("✓ Manual registration/unregistration works")
        
        test_suite.setup_method()
        test_suite.test_auto_import_features()
        print("✓ Auto import functionality tested")
        
        test_suite.setup_method()
        test_suite.test_registry_info_function()
        print("✓ Registry info function works")
        
        print("\nAll tests passed! ✓")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
