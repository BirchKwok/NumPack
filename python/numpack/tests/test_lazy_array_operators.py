"""
测试 LazyArray 的算术操作符支持

这个测试文件验证 LazyArray 现在支持像 NumPy memmap 一样的各种计算操作
"""

import numpy as np
import tempfile
import os
import numpack as npk

def test_lazy_array_arithmetic_operators():
    """测试基本算术操作符"""
    print("Testing basic arithmetic operators...")

    # 创建测试数据
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建测试数组
        original_data = np.array([1, 2, 3, 4, 5], dtype=np.float32)

        # 保存为 LazyArray
        array_path = os.path.join(tmpdir, "test_arithmetic.npk")
        with npk.NumPack(array_path, warn_no_context=False) as pack:
            pack.save({"test_array": original_data})
            
            # 加载为 LazyArray（在context manager内部）
            lazy_array = pack.load("test_array", lazy=True)

            # 测试加法: lazy_array + scalar
            result = lazy_array + 2.5
            expected = original_data + 2.5
            np.testing.assert_array_equal(result, expected)
            print("✓ Addition operator test passed")

            # 测试减法: lazy_array - scalar
            result = lazy_array - 1.5
            expected = original_data - 1.5
            np.testing.assert_array_equal(result, expected)
            print("✓ Subtraction operator test passed")

            # 测试乘法: lazy_array * scalar
            result = lazy_array * 3.0
            expected = original_data * 3.0
            np.testing.assert_array_equal(result, expected)
            print("✓ Multiplication operator test passed")

            # 测试除法: lazy_array / scalar
            result = lazy_array / 2.0
            expected = original_data / 2.0
            np.testing.assert_array_equal(result, expected)
            print("✓ Division operator test passed")

            # 测试地板除法: lazy_array // scalar
            result = lazy_array // 2
            expected = original_data // 2
            np.testing.assert_array_equal(result, expected)
            print("✓ Floor division operator test passed")

            # 测试取模: lazy_array % scalar
            result = lazy_array % 2
            expected = original_data % 2
            np.testing.assert_array_equal(result, expected)
            print("✓ Modulo operator test passed")

            # 测试幂运算: lazy_array ** scalar
            result = lazy_array ** 2
            expected = original_data ** 2
            np.testing.assert_array_equal(result, expected)
            print("✓ Power operator test passed")

            # 测试与 NumPy 数组的运算
            other_array = np.array([10, 20, 30, 40, 50], dtype=np.float32)
            result = lazy_array + other_array
            expected = original_data + other_array
            np.testing.assert_array_equal(result, expected)
            print("✓ NumPy array operation test passed")


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
        print("✅ All tests passed! LazyArray now supports arithmetic operators like NumPy memmap.")
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
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()