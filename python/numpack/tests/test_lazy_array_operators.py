"""
测试 LazyArray 的算术操作符支持

这个测试文件验证 LazyArray 现在支持像 NumPy memmap 一样的各种计算操作
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
    """测试基本算术操作符（所有数据类型）"""
    # 跳过复数类型的某些操作（如地板除法、取模）
    if np.issubdtype(dtype, np.complexfloating):
        pytest.skip("Complex types don't support floor division and modulo")
    
    # 创建测试数据
    with tempfile.TemporaryDirectory() as tmpdir:
        original_data = create_test_array(dtype, (5,))
        
        # 保存为 LazyArray
        array_path = os.path.join(tmpdir, f"test_arithmetic_{dtype.__name__}.npk")
        with npk.NumPack(array_path, warn_no_context=False) as pack:
            pack.save({"test_array": original_data})
            
            # 加载为 LazyArray
            lazy_array = pack.load("test_array", lazy=True)

            # 测试加法: lazy_array + scalar
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
                # Float16 运算在 LazyArray 中可能被提升为 Float32，允许较大误差
                np.testing.assert_allclose(result, expected, atol=1e-3)
            else:
                np.testing.assert_allclose(result, expected)

            # 测试乘法: lazy_array * scalar
            result = lazy_array * scalar
            expected = original_data * scalar
            if dtype == np.bool_:
                np.testing.assert_array_equal(result, expected)
            elif dtype == np.float16:
                np.testing.assert_allclose(result, expected, atol=1e-3)
            else:
                np.testing.assert_allclose(result, expected)

            # 测试除法（仅浮点类型）
            if np.issubdtype(dtype, np.floating):
                result = lazy_array / 2.0
                expected = original_data / 2.0
                if dtype == np.float16:
                    np.testing.assert_allclose(result, expected, atol=1e-3)
                else:
                    np.testing.assert_allclose(result, expected)

            # 测试地板除法和取模（仅整数类型）
            if np.issubdtype(dtype, np.integer):
                result = lazy_array // 2
                expected = original_data // 2
                np.testing.assert_array_equal(result, expected)
                
                result = lazy_array % 2
                expected = original_data % 2
                np.testing.assert_array_equal(result, expected)

            # 测试幂运算（仅数值类型）
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
    """测试比较操作符"""
    print("Testing comparison operators...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建测试数组
        original_data = np.array([1, 2, 3, 4, 5], dtype=np.float32)

        # 保存为 LazyArray
        array_path = os.path.join(tmpdir, "test_comparison.npk")
        with npk.NumPack(array_path, warn_no_context=False) as pack:
            pack.save({"test_array": original_data})
            
            # 加载为 LazyArray
            lazy_array = pack.load("test_array", lazy=True)

            # 测试等于
            result = lazy_array == 3
            expected = original_data == 3
            np.testing.assert_array_equal(result, expected)
            print("✓ Equal operator test passed")

            # 测试不等于
            result = lazy_array != 3
            expected = original_data != 3
            np.testing.assert_array_equal(result, expected)
            print("✓ Not equal operator test passed")

            # 测试小于
            result = lazy_array < 3
            expected = original_data < 3
            np.testing.assert_array_equal(result, expected)
            print("✓ Less than operator test passed")

            # 测试小于等于
            result = lazy_array <= 3
            expected = original_data <= 3
            np.testing.assert_array_equal(result, expected)
            print("✓ Less than or equal operator test passed")

            # 测试大于
            result = lazy_array > 3
            expected = original_data > 3
            np.testing.assert_array_equal(result, expected)
            print("✓ Greater than operator test passed")

            # 测试大于等于
            result = lazy_array >= 3
            expected = original_data >= 3
            np.testing.assert_array_equal(result, expected)
            print("✓ Greater than or equal operator test passed")


def test_lazy_array_unary_operators():
    """测试一元操作符"""
    print("Testing unary operators...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建测试数组
        original_data = np.array([1, -2, 3, -4, 5], dtype=np.float32)

        # 保存为 LazyArray
        array_path = os.path.join(tmpdir, "test_unary.npk")
        with npk.NumPack(array_path, warn_no_context=False) as pack:
            pack.save({"test_array": original_data})
            
            # 加载为 LazyArray
            lazy_array = pack.load("test_array", lazy=True)

            # 测试一元正号
            result = +lazy_array
            expected = +original_data
            np.testing.assert_array_equal(result, expected)
            print("✓ Unary positive operator test passed")

            # 测试一元负号
            result = -lazy_array
            expected = -original_data
            np.testing.assert_array_equal(result, expected)
            print("✓ Unary negative operator test passed")


def test_lazy_array_bitwise_operators():
    """测试位操作符（仅适用于整数类型）"""
    print("Testing bitwise operators...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建整数测试数组
        original_data = np.array([1, 2, 4, 8, 16], dtype=np.int32)

        # 保存为 LazyArray
        array_path = os.path.join(tmpdir, "test_bitwise.npk")
        with npk.NumPack(array_path, warn_no_context=False) as pack:
            pack.save({"test_array": original_data})
            
            # 加载为 LazyArray
            lazy_array = pack.load("test_array", lazy=True)

            # 测试位与
            result = lazy_array & 3
            expected = original_data & 3
            np.testing.assert_array_equal(result, expected)
            print("✓ Bitwise AND operator test passed")

            # 测试位或
            result = lazy_array | 2
            expected = original_data | 2
            np.testing.assert_array_equal(result, expected)
            print("✓ Bitwise OR operator test passed")

            # 测试位异或
            result = lazy_array ^ 1
            expected = original_data ^ 1
            np.testing.assert_array_equal(result, expected)
            print("✓ Bitwise XOR operator test passed")

            # 测试左移
            result = lazy_array << 1
            expected = original_data << 1
            np.testing.assert_array_equal(result, expected)
            print("✓ Left shift operator test passed")

            # 测试右移
            result = lazy_array >> 1
            expected = original_data >> 1
            np.testing.assert_array_equal(result, expected)
            print("✓ Right shift operator test passed")

            # 测试一元取反
            result = ~lazy_array
            expected = ~original_data
            np.testing.assert_array_equal(result, expected)
            print("✓ Bitwise NOT operator test passed")


def test_lazy_array_inplace_operators():
    """测试原地操作符（会将LazyArray转换为NumPy数组）"""
    print("Testing in-place operators...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建测试数组
        original_data = np.array([1, 2, 3, 4, 5], dtype=np.float32)

        # 保存为 LazyArray
        array_path = os.path.join(tmpdir, "test_inplace.npk")
        with npk.NumPack(array_path, warn_no_context=False) as pack:
            pack.save({"test_array": original_data})
            
            # 加载为 LazyArray
            lazy_array = pack.load("test_array", lazy=True)

            # 测试原地加法（应该转换为NumPy数组）
            lazy_array += 2.5
            assert isinstance(lazy_array, np.ndarray), "In-place operation should return NumPy array"
            expected = original_data + 2.5
            np.testing.assert_array_equal(lazy_array, expected)
            print("✓ In-place addition operator test passed")
            
            # 重新加载测试原地乘法
            lazy_array2 = pack.load("test_array", lazy=True)
            lazy_array2 *= 2.5
            assert isinstance(lazy_array2, np.ndarray), "In-place operation should return NumPy array"
            expected = original_data * 2.5
            np.testing.assert_array_equal(lazy_array2, expected)
            print("✓ In-place multiplication operator test passed")


def test_lazy_array_bitwise_type_checking():
    """测试位操作符的类型检查"""
    print("Testing bitwise operator type checking...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建浮点测试数组
        original_data = np.array([1.5, 2.5, 3.5], dtype=np.float32)

        # 保存为 LazyArray
        array_path = os.path.join(tmpdir, "test_bitwise_type_check.npk")
        with npk.NumPack(array_path, warn_no_context=False) as pack:
            pack.save({"test_array": original_data})
            
            # 加载为 LazyArray
            lazy_array = pack.load("test_array", lazy=True)

            # 测试位操作符对浮点数应该抛出错误
            try:
                result = lazy_array & 1
                assert False, "Float bitwise operation should raise TypeError"
            except TypeError as e:
                assert "Bitwise operations are only supported for integer arrays" in str(e)
                print("✓ Float bitwise operation correctly raises TypeError")


def test_lazy_array_complex_operations():
    """测试复杂运算"""
    print("Testing complex operations...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建测试数组
        original_data = np.array([1, 2, 3, 4, 5], dtype=np.float32)

        # 保存为 LazyArray
        array_path = os.path.join(tmpdir, "test_complex.npk")
        with npk.NumPack(array_path, warn_no_context=False) as pack:
            pack.save({"test_array": original_data})
            
            # 加载为 LazyArray
            lazy_array = pack.load("test_array", lazy=True)

            # 测试链式运算
            result = (lazy_array + 2) * 3 - 1
            expected = (original_data + 2) * 3 - 1
            np.testing.assert_array_equal(result, expected)
            print("✓ Chain operation test passed")

            # 测试与比较操作结合
            mask = (lazy_array > 2) & (lazy_array < 5)
            expected_mask = (original_data > 2) & (original_data < 5)
            np.testing.assert_array_equal(mask, expected_mask)
            print("✓ Operation with comparison test passed")

            # 测试数学函数组合
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