#!/usr/bin/env python3
"""
Tests for user intent recognition.

Verify that NumPack correctly distinguishes:
1. Single access: lazy_array[i] - respect user intent, no intervention
2. Batch access: lazy_array[indices] - optimize with a single FFI call
3. Complex indexing: slices, boolean masks, etc. - use existing logic
"""

import pytest
import numpy as np
import time
import tempfile
import shutil
from pathlib import Path

from numpack import NumPack


class TestUserIntentRecognition:
    """Tests for user intent recognition and corresponding optimization strategies."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = Path(self.temp_dir) / "test_intent"
        
        # Create test data
        self.rows, self.cols = 50000, 100
        self.test_data = {
            'test_array': np.random.rand(self.rows, self.cols).astype(np.float32)
        }
        
        # Save test data
        self.npk = NumPack(str(self.test_file), drop_if_exists=True)
        self.npk.open()  # Open explicitly
        self.npk.save(self.test_data)
        
    def teardown_method(self):
        """Clean up test environment."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_single_access_intent(self):
        """Test single access intent recognition."""
        lazy_array = self.npk.load('test_array', lazy=True)
        
        # Correct single access usage - should be recognized as SingleAccess
        single_index = 42
        result = lazy_array[single_index]
        
        assert result.shape == (self.cols,), f"Single access result shape error: {result.shape}"
        
        # Verify data correctness
        expected = self.test_data['test_array'][single_index]
        np.testing.assert_array_almost_equal(result, expected, decimal=5)
        print("Single access intent recognized correctly")

    def test_batch_access_intent(self):
        """Test batch access intent recognition."""
        lazy_array = self.npk.load('test_array', lazy=True)
        
        # Correct batch access usage - should be recognized as BatchAccess
        batch_indices = [10, 25, 50, 100, 200]
        result = lazy_array[batch_indices]
        
        assert result.shape == (len(batch_indices), self.cols), f"Batch access result shape error: {result.shape}"
        
        # Verify data correctness
        expected = self.test_data['test_array'][batch_indices]
        np.testing.assert_array_almost_equal(result, expected, decimal=5)
        print("Batch access intent recognized correctly")

    def test_numpy_array_batch_access(self):
        """Test batch access using NumPy array indexing."""
        lazy_array = self.npk.load('test_array', lazy=True)
        
        # NumPy array indexing - should be recognized as BatchAccess
        indices = np.array([5, 15, 35, 75, 150])
        result = lazy_array[indices]
        
        assert result.shape == (len(indices), self.cols), f"NumPy array index result shape error: {result.shape}"
        
        # Verify data correctness
        expected = self.test_data['test_array'][indices]
        np.testing.assert_array_almost_equal(result, expected, decimal=5)
        print("NumPy array index batch access correct")

    def test_slice_access(self):
        """Test slice access - should be recognized as ComplexIndex."""
        lazy_array = self.npk.load('test_array', lazy=True)
        
        # Slice access
        result = lazy_array[10:20]
        
        assert result.shape == (10, self.cols), f"Slice access result shape error: {result.shape}"
        
        # Verify data correctness
        expected = self.test_data['test_array'][10:20]
        np.testing.assert_array_almost_equal(result, expected, decimal=5)
        print("Slice access correct")

    def test_user_intent_examples(self):
        """Demonstrate correct user intent usage examples."""
        lazy_array = self.npk.load('test_array', lazy=True)
        
        print("\nUser Intent Examples:")
        
        # Scenario 1: Clear single access
        print("Scenario 1 - Clear single access:")
        print("  Usage: row = lazy_array[42]")
        row = lazy_array[42]
        print(f"  Result: {row.shape}")
        
        # Scenario 2: Clear batch access
        print("Scenario 2 - Clear batch access:")
        print("  Usage: rows = lazy_array[[10, 20, 30]]")
        rows = lazy_array[[10, 20, 30]]
        print(f"  Result: {rows.shape}")
        
        # Scenario 3: NumPy array indexing
        print("Scenario 3 - NumPy array indexing:")
        indices = np.array([5, 15, 25])
        print(f"  Usage: rows = lazy_array[np.array({indices.tolist()})]")
        rows = lazy_array[indices]
        print(f"  Result: {rows.shape}")
        
        # Scenario 4: Slice access
        print("Scenario 4 - Slice access:")
        print("  Usage: rows = lazy_array[10:15]")
        rows = lazy_array[10:15]
        print(f"  Result: {rows.shape}")
        
        print("\nAll user intent example tests passed")

if __name__ == "__main__":
    # Run tests
    test = TestUserIntentRecognition()
    test.setup_method()
    
    try:
        test.test_single_access_intent()
        test.test_batch_access_intent()
        test.test_numpy_array_batch_access()
        test.test_slice_access()
        test.test_user_intent_examples()
        
        print("\nAll user intent recognition tests passed!")
        
    finally:
        test.teardown_method() 