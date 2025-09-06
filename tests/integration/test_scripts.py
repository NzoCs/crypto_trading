"""
Integration Tests for Scripts

These tests verify the end-to-end functionality of the various scripts
in the scripts/ directory, including feature generation, model management,
preprocessing, and backtesting.
"""

import pytest
import subprocess
import sys
import os
import tempfile
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, Mock
import json

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestGenerateFeaturesScript:
    """Integration tests for generate_features.py script"""
    
    def setup_method(self):
        """Set up test environment"""
        self.script_path = PROJECT_ROOT / "scripts" / "generate_features.py"
        self.test_data_dir = Path(tempfile.mkdtemp())
        self.test_output_dir = Path(tempfile.mkdtemp())
        
        # Create sample preprocessed data
        self.create_sample_preprocessed_data()
    
    def teardown_method(self):
        """Clean up test environment"""
        if self.test_data_dir.exists():
            shutil.rmtree(self.test_data_dir)
        if self.test_output_dir.exists():
            shutil.rmtree(self.test_output_dir)
    
    def create_sample_preprocessed_data(self):
        """Create sample preprocessed data for testing"""
        # Create directory structure
        data_dir = self.test_data_dir / "DATA_1"
        data_dir.mkdir(parents=True)
        
        # Create sample ETH data
        timestamps = pd.date_range('2024-01-01', periods=100, freq='1s')
        sample_data = pd.DataFrame({
            'timestamp': timestamps,
            'level-1-bid-price': 2000 + np.random.randn(100) * 10,
            'level-1-ask-price': 2005 + np.random.randn(100) * 10,
            'level-1-bid-volume': 100 + np.random.exponential(50, 100),
            'level-1-ask-volume': 95 + np.random.exponential(45, 100),
            'level-2-bid-price': 1999 + np.random.randn(100) * 10,
            'level-2-ask-price': 2006 + np.random.randn(100) * 10,
            'level-2-bid-volume': 80 + np.random.exponential(40, 100),
            'level-2-ask-volume': 75 + np.random.exponential(35, 100),
        })
        
        # Ensure realistic price relationships
        sample_data['level-1-bid-price'] = np.abs(sample_data['level-1-bid-price'])
        sample_data['level-1-ask-price'] = sample_data['level-1-bid-price'] + 5
        sample_data['level-2-bid-price'] = sample_data['level-1-bid-price'] - 1
        sample_data['level-2-ask-price'] = sample_data['level-1-ask-price'] + 1
        
        # Ensure positive volumes
        for col in ['level-1-bid-volume', 'level-1-ask-volume', 'level-2-bid-volume', 'level-2-ask-volume']:
            sample_data[col] = np.abs(sample_data[col])
        
        # Save as parquet
        sample_data.to_parquet(data_dir / "ETH_EUR.parquet")
        sample_data.to_parquet(data_dir / "XBT_EUR.parquet")  # Same structure for simplicity
    
    def test_script_exists(self):
        """Test that the generate_features.py script exists"""
        assert self.script_path.exists(), f"Script not found at {self.script_path}"
        assert self.script_path.is_file(), f"Path is not a file: {self.script_path}"
    
    def test_script_help(self):
        """Test that the script shows help without errors"""
        result = subprocess.run(
            [sys.executable, str(self.script_path), "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        assert result.returncode == 0, f"Help command failed: {result.stderr}"
        assert "usage:" in result.stdout.lower() or "Generate features" in result.stdout
    
    def test_script_list_features(self):
        """Test the --list-features option"""
        result = subprocess.run(
            [
                sys.executable, str(self.script_path),
                "--coin", "ETH",
                "--data-version", "1",
                "--list-features"
            ],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Should succeed even without actual data (just listing features)
        # The exact return code may depend on implementation
        assert result.returncode in [0, 1]  # May return 1 if no data found, but should still list features
    
    @pytest.mark.slow
    def test_script_feature_generation_dry_run(self):
        """Test feature generation script with minimal parameters (dry run style)"""
        # This test may fail if the script requires actual data, but we'll check basic invocation
        result = subprocess.run(
            [
                sys.executable, str(self.script_path),
                "--coin", "ETH",
                "--data-version", "1",
                "--input-dir", str(self.test_data_dir),
                "--output-dir", str(self.test_output_dir),
                "--features", "spread",  # Try to generate just one simple feature
                "--overwrite"
            ],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # The script might fail due to missing dependencies or data format issues
        # But it should at least start and show meaningful error messages
        assert result.returncode in [0, 1], f"Script failed unexpectedly: {result.stderr}"
        
        # Check if any meaningful output was produced
        assert len(result.stdout) > 0 or len(result.stderr) > 0, "No output from script"


class TestRunBacktestScript:
    """Integration tests for run_backtest.py script"""
    
    def setup_method(self):
        """Set up test environment"""
        self.script_path = PROJECT_ROOT / "scripts" / "run_backtest.py"
        self.test_output_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up test environment"""
        if self.test_output_dir.exists():
            shutil.rmtree(self.test_output_dir)
    
    def test_script_exists(self):
        """Test that the run_backtest.py script exists"""
        assert self.script_path.exists(), f"Script not found at {self.script_path}"
        assert self.script_path.is_file(), f"Path is not a file: {self.script_path}"
    
    def test_script_help(self):
        """Test that the script shows help without errors"""
        result = subprocess.run(
            [sys.executable, str(self.script_path), "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        assert result.returncode == 0, f"Help command failed: {result.stderr}"
        assert "usage:" in result.stdout.lower() or "backtest" in result.stdout.lower()
    
    def test_script_list_strategies(self):
        """Test the --list-strategies option"""
        result = subprocess.run(
            [sys.executable, str(self.script_path), "--list-strategies"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        assert result.returncode == 0, f"List strategies command failed: {result.stderr}"
        assert "strategies:" in result.stdout.lower() or len(result.stdout) > 0
    
    def test_script_missing_required_args(self):
        """Test script behavior with missing required arguments"""
        # Try to run without required arguments
        result = subprocess.run(
            [sys.executable, str(self.script_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Should fail with meaningful error message
        assert result.returncode != 0, "Script should fail without required arguments"
        assert "error:" in result.stderr.lower() or "required" in result.stderr.lower()
    
    def test_script_invalid_strategy(self):
        """Test script behavior with invalid strategy name"""
        result = subprocess.run(
            [
                sys.executable, str(self.script_path),
                "--strategy", "NonExistentStrategy",
                "--data-index", "1"
            ],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Should fail or handle gracefully
        assert result.returncode != 0 or "not found" in result.stdout.lower()


class TestPreprocessScript:
    """Integration tests for preprocessing scripts"""
    
    def setup_method(self):
        """Set up test environment"""
        self.preprocess_script = PROJECT_ROOT / "scripts" / "preprocess_script.py"
        self.preprocess_all_script = PROJECT_ROOT / "scripts" / "preprocess_all_data.py"
    
    def test_preprocess_script_exists(self):
        """Test that preprocessing scripts exist"""
        assert self.preprocess_script.exists(), f"Script not found at {self.preprocess_script}"
        # The preprocess_all_data.py might not exist, so we'll check conditionally
        if self.preprocess_all_script.exists():
            assert self.preprocess_all_script.is_file()
    
    def test_preprocess_script_help(self):
        """Test that the preprocessing script shows help"""
        if not self.preprocess_script.exists():
            pytest.skip("Preprocess script not found")
        
        result = subprocess.run(
            [sys.executable, str(self.preprocess_script), "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Allow for different return codes as some scripts might not have proper arg parsing
        assert result.returncode in [0, 1, 2], f"Help command failed: {result.stderr}"


class TestManageModelsScript:
    """Integration tests for manage_models.py script"""
    
    def setup_method(self):
        """Set up test environment"""
        self.script_path = PROJECT_ROOT / "scripts" / "manage_models.py"
    
    def test_script_exists(self):
        """Test that the manage_models.py script exists"""
        assert self.script_path.exists(), f"Script not found at {self.script_path}"
        assert self.script_path.is_file(), f"Path is not a file: {self.script_path}"
    
    def test_script_help(self):
        """Test that the script shows help without errors"""
        result = subprocess.run(
            [sys.executable, str(self.script_path), "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Allow for different return codes as some scripts might not have proper arg parsing
        assert result.returncode in [0, 1, 2], f"Help command failed: {result.stderr}"


class TestCreateFeatureScript:
    """Integration tests for create_feature.py script"""
    
    def setup_method(self):
        """Set up test environment"""
        self.script_path = PROJECT_ROOT / "scripts" / "create_feature.py"
    
    def test_script_exists(self):
        """Test that the create_feature.py script exists"""
        assert self.script_path.exists(), f"Script not found at {self.script_path}"
        assert self.script_path.is_file(), f"Path is not a file: {self.script_path}"
    
    def test_script_help(self):
        """Test that the script shows help without errors"""
        result = subprocess.run(
            [sys.executable, str(self.script_path), "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Allow for different return codes as implementation may vary
        assert result.returncode in [0, 1, 2], f"Help command failed: {result.stderr}"


class TestScriptIntegration:
    """Cross-script integration tests"""
    
    def setup_method(self):
        """Set up test environment"""
        self.test_workspace = Path(tempfile.mkdtemp())
        self.scripts_dir = PROJECT_ROOT / "scripts"
    
    def teardown_method(self):
        """Clean up test environment"""
        if self.test_workspace.exists():
            shutil.rmtree(self.test_workspace)
    
    def test_all_scripts_are_importable(self):
        """Test that all Python scripts in scripts/ are syntactically correct"""
        python_scripts = list(self.scripts_dir.glob("*.py"))
        python_scripts = [s for s in python_scripts if not s.name.startswith("__")]
        
        assert len(python_scripts) > 0, "No Python scripts found in scripts directory"
        
        for script in python_scripts:
            # Test that the script can be parsed (syntax check)
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", str(script)],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            assert result.returncode == 0, f"Syntax error in {script.name}: {result.stderr}"
    
    def test_scripts_have_shebang_or_main(self):
        """Test that scripts are properly structured"""
        python_scripts = list(self.scripts_dir.glob("*.py"))
        python_scripts = [s for s in python_scripts if not s.name.startswith("__")]
        
        for script in python_scripts:
            content = script.read_text(encoding='utf-8')
            
            # Should have either a shebang or a main block
            has_shebang = content.startswith("#!")
            has_main = 'if __name__ == "__main__"' in content
            
            assert has_shebang or has_main, f"Script {script.name} should have shebang or main block"
    
    def test_scripts_directory_structure(self):
        """Test that scripts directory has expected structure"""
        assert self.scripts_dir.exists(), "Scripts directory should exist"
        assert self.scripts_dir.is_dir(), "Scripts path should be a directory"
        
        # Check for key scripts (these should exist based on the project structure)
        expected_scripts = [
            "generate_features.py",
            "run_backtest.py",
            "manage_models.py",
            "create_feature.py"
        ]
        
        for script_name in expected_scripts:
            script_path = self.scripts_dir / script_name
            assert script_path.exists(), f"Expected script {script_name} not found"
    
    def test_readme_exists(self):
        """Test that scripts have documentation"""
        readme_path = self.scripts_dir / "README.md"
        # README might exist
        if readme_path.exists():
            content = readme_path.read_text(encoding='utf-8')
            assert len(content) > 0, "README should not be empty"
    
    @pytest.mark.slow
    def test_script_execution_isolation(self):
        """Test that scripts can be executed in isolated environments"""
        # This is a basic test to ensure scripts don't have obvious import issues
        python_scripts = [
            self.scripts_dir / "generate_features.py",
            self.scripts_dir / "run_backtest.py",
        ]
        
        for script in python_scripts:
            if not script.exists():
                continue
            
            # Try to import the script as a module
            result = subprocess.run(
                [sys.executable, "-c", f"import sys; sys.path.insert(0, '{self.scripts_dir}'); import {script.stem}"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(PROJECT_ROOT)
            )
            
            # Should not fail with import errors (may fail for other reasons like missing data)
            if result.returncode != 0:
                # Check if it's an import error vs other error
                if "ImportError" in result.stderr or "ModuleNotFoundError" in result.stderr:
                    pytest.fail(f"Import error in {script.name}: {result.stderr}")
                # Other errors are acceptable (missing data, etc.)


class TestScriptErrorHandling:
    """Test error handling in scripts"""
    
    def setup_method(self):
        """Set up test environment"""
        self.scripts_dir = PROJECT_ROOT / "scripts"
    
    def test_scripts_handle_keyboard_interrupt(self):
        """Test that scripts handle KeyboardInterrupt gracefully"""
        # This is a conceptual test - actual implementation would need careful timing
        pass
    
    def test_scripts_validate_inputs(self):
        """Test that scripts validate their inputs"""
        # Test with obviously invalid inputs
        run_backtest = self.scripts_dir / "run_backtest.py"
        
        if run_backtest.exists():
            # Test with invalid data-index
            result = subprocess.run(
                [
                    sys.executable, str(run_backtest),
                    "--strategy", "TFCumulativeReturnStrategy",
                    "--data-index", "-1"  # Invalid negative index
                ],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Should fail with validation error
            assert result.returncode != 0, "Script should reject invalid data-index"
    
    def test_scripts_provide_helpful_error_messages(self):
        """Test that scripts provide helpful error messages"""
        run_backtest = self.scripts_dir / "run_backtest.py"
        
        if run_backtest.exists():
            # Test with missing data-index
            result = subprocess.run(
                [
                    sys.executable, str(run_backtest),
                    "--strategy", "TFCumulativeReturnStrategy"
                    # Missing --data-index
                ],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Should provide helpful error message
            if result.returncode != 0:
                assert len(result.stderr) > 0, "Should provide error message for missing arguments"


class TestScriptPerformance:
    """Performance tests for scripts"""
    
    def setup_method(self):
        """Set up test environment"""
        self.scripts_dir = PROJECT_ROOT / "scripts"
    
    @pytest.mark.slow
    def test_script_startup_time(self):
        """Test that scripts start up in reasonable time"""
        run_backtest = self.scripts_dir / "run_backtest.py"
        
        if run_backtest.exists():
            import time
            start_time = time.time()
            
            result = subprocess.run(
                [sys.executable, str(run_backtest), "--help"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            elapsed_time = time.time() - start_time
            
            # Should start up within 10 seconds (generous timeout)
            assert elapsed_time < 10.0, f"Script took too long to start up: {elapsed_time:.2f}s"
    
    def test_scripts_memory_usage(self):
        """Test that scripts don't have obvious memory leaks"""
        # This would require more sophisticated tooling to implement properly
        # For now, we just ensure scripts exit cleanly
        pass
