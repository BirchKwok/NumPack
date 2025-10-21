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
    print("测试基本算术操作符...")

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
            print("✓ 加法操作符测试通过")

            # 测试减法: lazy_array - scalar
            result = lazy_array - 1.5
            expected = original_data - 1.5
            np.testing.assert_array_equal(result, expected)
            print("✓ 减法操作符测试通过")

            # 测试乘法: lazy_array * scalar
            result = lazy_array * 3.0
            expected = original_data * 3.0
            np.testing.assert_array_equal(result, expected)
            print("✓ 乘法操作符测试通过")

            # 测试除法: lazy_array / scalar
            result = lazy_array / 2.0
            expected = original_data / 2.0
            np.testing.assert_array_equal(result, expected)
            print("✓ 除法操作符测试通过")

            # 测试地板除法: lazy_array // scalar
            result = lazy_array // 2
            expected = original_data // 2
            np.testing.assert_array_equal(result, expected)
            print("✓ 地板除法操作符测试通过")

            # 测试取模: lazy_array % scalar
            result = lazy_array % 2
            expected = original_data % 2
            np.testing.assert_array_equal(result, expected)
            print("✓ 取模操作符测试通过")

            # 测试幂运算: lazy_array ** scalar
            result = lazy_array ** 2
            expected = original_data ** 2
            np.testing.assert_array_equal(result, expected)
            print("✓ 幂运算操作符测试通过")

            # 测试与 NumPy 数组的运算
            other_array = np.array([10, 20, 30, 40, 50], dtype=np.float32)
            result = lazy_array + other_array
            expected = original_data + other_array
            np.testing.assert_array_equal(result, expected)
            print("✓ 与 NumPy 数组运算测试通过")


def test_lazy_array_comparison_operators():
    """测试比较操作符"""
    print("测试比较操作符...")

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
            print("✓ 等于操作符测试通过")

            # 测试不等于
            result = lazy_array != 3
            expected = original_data != 3
            np.testing.assert_array_equal(result, expected)
            print("✓ 不等于操作符测试通过")

            # 测试小于
            result = lazy_array < 3
            expected = original_data < 3
            np.testing.assert_array_equal(result, expected)
            print("✓ 小于操作符测试通过")

            # 测试小于等于
            result = lazy_array <= 3
            expected = original_data <= 3
            np.testing.assert_array_equal(result, expected)
            print("✓ 小于等于操作符测试通过")

            # 测试大于
            result = lazy_array > 3
            expected = original_data > 3
            np.testing.assert_array_equal(result, expected)
            print("✓ 大于操作符测试通过")

            # 测试大于等于
            result = lazy_array >= 3
            expected = original_data >= 3
            np.testing.assert_array_equal(result, expected)
            print("✓ 大于等于操作符测试通过")


def test_lazy_array_unary_operators():
    """测试一元操作符"""
    print("测试一元操作符...")

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
            print("✓ 一元正号操作符测试通过")

            # 测试一元负号
            result = -lazy_array
            expected = -original_data
            np.testing.assert_array_equal(result, expected)
            print("✓ 一元负号操作符测试通过")


def test_lazy_array_bitwise_operators():
    """测试位操作符（仅适用于整数类型）"""
    print("测试位操作符...")

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
            print("✓ 位与操作符测试通过")

            # 测试位或
            result = lazy_array | 2
            expected = original_data | 2
            np.testing.assert_array_equal(result, expected)
            print("✓ 位或操作符测试通过")

            # 测试位异或
            result = lazy_array ^ 1
            expected = original_data ^ 1
            np.testing.assert_array_equal(result, expected)
            print("✓ 位异或操作符测试通过")

            # 测试左移
            result = lazy_array << 1
            expected = original_data << 1
            np.testing.assert_array_equal(result, expected)
            print("✓ 左移操作符测试通过")

            # 测试右移
            result = lazy_array >> 1
            expected = original_data >> 1
            np.testing.assert_array_equal(result, expected)
            print("✓ 右移操作符测试通过")

            # 测试一元取反
            result = ~lazy_array
            expected = ~original_data
            np.testing.assert_array_equal(result, expected)
            print("✓ 一元取反操作符测试通过")


def test_lazy_array_inplace_operators():
    """测试原地操作符（会将LazyArray转换为NumPy数组）"""
    print("测试原地操作符...")

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
            assert isinstance(lazy_array, np.ndarray), "原地操作应该返回NumPy数组"
            expected = original_data + 2.5
            np.testing.assert_array_equal(lazy_array, expected)
            print("✓ 原地加法操作符测试通过")
            
            # 重新加载测试原地乘法
            lazy_array2 = pack.load("test_array", lazy=True)
            lazy_array2 *= 2.5
            assert isinstance(lazy_array2, np.ndarray), "原地操作应该返回NumPy数组"
            expected = original_data * 2.5
            np.testing.assert_array_equal(lazy_array2, expected)
            print("✓ 原地乘法操作符测试通过")


def test_lazy_array_bitwise_type_checking():
    """测试位操作符的类型检查"""
    print("测试位操作符类型检查...")

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
                assert False, "浮点数位操作应该抛出 TypeError"
            except TypeError as e:
                assert "Bitwise operations are only supported for integer arrays" in str(e)
                print("✓ 浮点数位操作正确抛出 TypeError")


def test_lazy_array_complex_operations():
    """测试复杂运算"""
    print("测试复杂运算...")

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
            print("✓ 链式运算测试通过")

            # 测试与比较操作结合
            mask = (lazy_array > 2) & (lazy_array < 5)
            expected_mask = (original_data > 2) & (original_data < 5)
            np.testing.assert_array_equal(mask, expected_mask)
            print("✓ 运算与比较结合测试通过")

            # 测试数学函数组合
            result = np.sqrt(lazy_array ** 2 + 1)
            expected = np.sqrt(original_data ** 2 + 1)
            np.testing.assert_array_almost_equal(result, expected)
            print("✓ 数学函数组合测试通过")


if __name__ == "__main__":
    print("开始测试 LazyArray 算术操作符支持...")
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
        print("✅ 所有测试通过！LazyArray 现在支持像 NumPy memmap 一样的算术操作符。")
        print()
        print("使用示例：")
        print("```python")
        print("import numpack as npk")
        print("import numpy as np")
        print()
        print("# 加载 LazyArray")
        print("lazy_array = npk.load('data', lazy=True)")
        print()
        print("# 现在可以使用各种算术操作符")
        print("result = lazy_array * 4.1  # 这现在可以工作了！")
        print("result = lazy_array + np.array([1, 2, 3])")
        print("mask = lazy_array > 5")
        print("result = lazy_array ** 2 + lazy_array * 2 + 1")
        print("```")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()