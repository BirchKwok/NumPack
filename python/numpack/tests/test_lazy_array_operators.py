"""
Tests for LazyArray arithmetic operator support.

This test file verifies that LazyArray now supports various computational operations like NumPy memmap.
"""

import numpy as np
import pytest
import tempfile
import os
import numpack as npk
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import conftest
ALL_DTYPES = conftest.ALL_DTYPES
create_test_array = conftest.create_test_array

@pytest.mark.parametrize("dtype,test_values", ALL_DTYPES)
def test_lazy_array_arithmetic_operators(dtype, test_values):
    """Test basic arithmetic operators (all data types)."""
    # Skip certain operations for complex types (e.g., floor division, modulo)
    if np.issubdtype(dtype, np.complexfloating):
        pytest.skip("Complex types don't support floor division and modulo")
    
    # Create test data
    with tempfile.TemporaryDirectory() as tmpdir:
        original_data = create_test_array(dtype, (5,))
        
        # Save as LazyArray
        array_path = os.path.join(tmpdir, f"test_arithmetic_{dtype.__name__}.npk")
        with npk.NumPack(array_path, warn_no_context=False) as pack:
            pack.save({"test_array": original_data})
            
            # Load as LazyArray
            lazy_array = pack.load("test_array", lazy=True)

            # Test addition: lazy_array + scalar
            if np.issubdtype(dtype, np.integer):
                scalar = 2
            elif dtype == np.bool_:
                scalar = True
            else:
                scalar = 2.5
            
            result = lazy_array + scalar
            expected = original_data + scalar
            if dtype == np.bool_:
                np.testing.assert_array_equal(result, expected)
            elif dtype == np.float16:
                # Float16 operations in LazyArray may be promoted to Float32; allow larger tolerance
                np.testing.assert_allclose(result, expected, atol=1e-3)
            else:
                np.testing.assert_allclose(result, expected)

            # Test multiplication: lazy_array * scalar
            result = lazy_array * scalar
            expected = original_data * scalar
            if dtype == np.bool_:
                np.testing.assert_array_equal(result, expected)
            elif dtype == np.float16:
                np.testing.assert_allclose(result, expected, atol=1e-3)
            else:
                np.testing.assert_allclose(result, expected)

            # Test division (floating point types only)
            if np.issubdtype(dtype, np.floating):
                result = lazy_array / 2.0
                expected = original_data / 2.0
                if dtype == np.float16:
                    np.testing.assert_allclose(result, expected, atol=1e-3)
                else:
                    np.testing.assert_allclose(result, expected)

            # Test floor division and modulo (integer types only)
            if np.issubdtype(dtype, np.integer):
                result = lazy_array // 2
                expected = original_data // 2
                np.testing.assert_array_equal(result, expected)
                
                result = lazy_array % 2
                expected = original_data % 2
                np.testing.assert_array_equal(result, expected)

            # Test power operation (numeric types only)
            if not dtype == np.bool_:
                result = lazy_array ** 2
                expected = original_data ** 2
                if np.issubdtype(dtype, np.complexfloating):
                    np.testing.assert_allclose(result, expected)
                elif dtype == np.float16:
                    np.testing.assert_allclose(result, expected, atol=1e-3)
                else:
                    np.testing.assert_allclose(result, expected)


def test_lazy_array_comparison_operators():
    """Test comparison operators."""
    print("Testing comparison operators...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test array
        original_data = np.array([1, 2, 3, 4, 5], dtype=np.float32)

        # Save as LazyArray
        array_path = os.path.join(tmpdir, "test_comparison.npk")
        with npk.NumPack(array_path, warn_no_context=False) as pack:
            pack.save({"test_array": original_data})
            
            # Load as LazyArray
            lazy_array = pack.load("test_array", lazy=True)

            # Test equal
            result = lazy_array == 3
            expected = original_data == 3
            np.testing.assert_array_equal(result, expected)
            print("✓ Equal operator test passed")

            # Test not equal
            result = lazy_array != 3
            expected = original_data != 3
            np.testing.assert_array_equal(result, expected)
            print("✓ Not equal operator test passed")

            # Test less than
            result = lazy_array < 3
            expected = original_data < 3
            np.testing.assert_array_equal(result, expected)
            print("✓ Less than operator test passed")

            # Test less than or equal
            result = lazy_array <= 3
            expected = original_data <= 3
            np.testing.assert_array_equal(result, expected)
            print("✓ Less than or equal operator test passed")

            # Test greater than
            result = lazy_array > 3
            expected = original_data > 3
            np.testing.assert_array_equal(result, expected)
            print("✓ Greater than operator test passed")

            # Test greater than or equal
            result = lazy_array >= 3
            expected = original_data >= 3
            np.testing.assert_array_equal(result, expected)
            print("✓ Greater than or equal operator test passed")


def test_lazy_array_unary_operators():
    """Test unary operators."""
    print("Testing unary operators...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test array
        original_data = np.array([1, -2, 3, -4, 5], dtype=np.float32)

        # Save as LazyArray
        array_path = os.path.join(tmpdir, "test_unary.npk")
        with npk.NumPack(array_path, warn_no_context=False) as pack:
            pack.save({"test_array": original_data})
            
            # Load as LazyArray
            lazy_array = pack.load("test_array", lazy=True)

            # Test unary positive
            result = +lazy_array
            expected = +original_data
            np.testing.assert_array_equal(result, expected)
            print("✓ Unary positive operator test passed")

            # Test unary negative
            result = -lazy_array
            expected = -original_data
            np.testing.assert_array_equal(result, expected)
            print("✓ Unary negative operator test passed")


def test_lazy_array_bitwise_operators():
    """Test bitwise operators (integer types only)."""
    print("Testing bitwise operators...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create integer test array
        original_data = np.array([1, 2, 4, 8, 16], dtype=np.int32)

        # Save as LazyArray
        array_path = os.path.join(tmpdir, "test_bitwise.npk")
        with npk.NumPack(array_path, warn_no_context=False) as pack:
            pack.save({"test_array": original_data})
            
            # Load as LazyArray
            lazy_array = pack.load("test_array", lazy=True)

            # Test bitwise AND
            result = lazy_array & 3
            expected = original_data & 3
            np.testing.assert_array_equal(result, expected)
            print("✓ Bitwise AND operator test passed")

            # Test bitwise OR
            result = lazy_array | 2
            expected = original_data | 2
            np.testing.assert_array_equal(result, expected)
            print("✓ Bitwise OR operator test passed")

            # Test bitwise XOR
            result = lazy_array ^ 1
            expected = original_data ^ 1
            np.testing.assert_array_equal(result, expected)
            print("✓ Bitwise XOR operator test passed")

            # Test left shift
            result = lazy_array << 1
            expected = original_data << 1
            np.testing.assert_array_equal(result, expected)
            print("✓ Left shift operator test passed")

            # Test right shift
            result = lazy_array >> 1
            expected = original_data >> 1
            np.testing.assert_array_equal(result, expected)
            print("✓ Right shift operator test passed")

            # Test bitwise NOT
            result = ~lazy_array
            expected = ~original_data
            np.testing.assert_array_equal(result, expected)
            print("✓ Bitwise NOT operator test passed")


def test_lazy_array_inplace_operators():
    """Test in-place operators (converts LazyArray to NumPy array)."""
    print("Testing in-place operators...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test array
        original_data = np.array([1, 2, 3, 4, 5], dtype=np.float32)

        # Save as LazyArray
        array_path = os.path.join(tmpdir, "test_inplace.npk")
        with npk.NumPack(array_path, warn_no_context=False) as pack:
            pack.save({"test_array": original_data})
            
            # Load as LazyArray
            lazy_array = pack.load("test_array", lazy=True)

            # Test in-place addition (should convert to NumPy array)
            lazy_array += 2.5
            assert isinstance(lazy_array, np.ndarray), "In-place operation should return NumPy array"
            expected = original_data + 2.5
            np.testing.assert_array_equal(lazy_array, expected)
            print("✓ In-place addition operator test passed")
            
            # Reload to test in-place multiplication
            lazy_array2 = pack.load("test_array", lazy=True)
            lazy_array2 *= 2.5
            assert isinstance(lazy_array2, np.ndarray), "In-place operation should return NumPy array"
            expected = original_data * 2.5
            np.testing.assert_array_equal(lazy_array2, expected)
            print("✓ In-place multiplication operator test passed")


def test_lazy_array_bitwise_type_checking():
    """Test type checking for bitwise operators."""
    print("Testing bitwise operator type checking...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create floating point test array
        original_data = np.array([1.5, 2.5, 3.5], dtype=np.float32)

        # Save as LazyArray
        array_path = os.path.join(tmpdir, "test_bitwise_type_check.npk")
        with npk.NumPack(array_path, warn_no_context=False) as pack:
            pack.save({"test_array": original_data})
            
            # Load as LazyArray
            lazy_array = pack.load("test_array", lazy=True)

            # Test that bitwise operators on floats should raise error
            try:
                result = lazy_array & 1
                assert False, "Float bitwise operation should raise TypeError"
            except TypeError as e:
                assert "Bitwise operations are only supported for integer arrays" in str(e)
                print("✓ Float bitwise operation correctly raises TypeError")


def test_lazy_array_complex_operations():
    """Test complex operations."""
    print("Testing complex operations...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test array
        original_data = np.array([1, 2, 3, 4, 5], dtype=np.float32)

        # Save as LazyArray
        array_path = os.path.join(tmpdir, "test_complex.npk")
        with npk.NumPack(array_path, warn_no_context=False) as pack:
            pack.save({"test_array": original_data})
            
            # Load as LazyArray
            lazy_array = pack.load("test_array", lazy=True)

            # Test chain operations
            result = (lazy_array + 2) * 3 - 1
            expected = (original_data + 2) * 3 - 1
            np.testing.assert_array_equal(result, expected)
            print("✓ Chain operation test passed")

            # Test combination with comparison operations
            mask = (lazy_array > 2) & (lazy_array < 5)
            expected_mask = (original_data > 2) & (original_data < 5)
            np.testing.assert_array_equal(mask, expected_mask)
            print("✓ Operation with comparison test passed")

            # Test math function composition
            result = np.sqrt(lazy_array ** 2 + 1)
            expected = np.sqrt(original_data ** 2 + 1)
            np.testing.assert_array_almost_equal(result, expected)
            print("✓ Math function composition test passed")


if __name__ == "__main__":
    print("Starting LazyArray arithmetic operator support tests...")
    print("=" * 60)

    try:
        test_lazy_array_arithmetic_operators()
        test_lazy_array_comparison_operators()
        test_lazy_array_unary_operators()
        test_lazy_array_bitwise_operators()
        test_lazy_array_inplace_operators()
        test_lazy_array_bitwise_type_checking()
        test_lazy_array_complex_operations()

        print("=" * 60)
        print("All tests passed! LazyArray now supports arithmetic operators like NumPy memmap.")
        print()
        print("Usage example:")
        print("```python")
        print("import numpack as npk")
        print("import numpy as np")
        print()
        print("# Load LazyArray")
        print("lazy_array = npk.load('data', lazy=True)")
        print()
        print("# Now you can use various arithmetic operators")
        print("result = lazy_array * 4.1  # This works now!")
        print("result = lazy_array + np.array([1, 2, 3])")
        print("mask = lazy_array > 5")
        print("result = lazy_array ** 2 + lazy_array * 2 + 1")
        print("```")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()