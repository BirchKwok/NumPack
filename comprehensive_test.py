#!/usr/bin/env python3
"""
全面的 LazyArray 操作符测试
"""

import numpy as np
import tempfile
import sys

def test_comprehensive_operators():
    """全面的 LazyArray 操作符测试"""
    print("开始全面的 LazyArray 算术操作符测试...")

    try:
        # 导入 NumPack
        from numpack import _lib_numpack
        NumPack = _lib_numpack.NumPack
        print("✓ NumPack 导入成功")

        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"✓ 临时目录: {tmpdir}")

            # 创建测试数据
            test_data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
            print(f"✓ 测试数据: {test_data}")

            # 保存数据
            npk = NumPack(tmpdir)
            arrays = {'test_array': test_data}
            npk.save(arrays, None)
            print("✓ 数据保存成功")

            # 加载为 LazyArray
            lazy_array = npk.load('test_array', lazy=True)
            print(f"✓ LazyArray 加载成功: {type(lazy_array)}")

            print("\n=== 测试算术操作符 ===")

            # 加法
            result_add = lazy_array + 2
            expected_add = test_data + 2
            np.testing.assert_array_almost_equal(result_add, expected_add)
            print(f"✓ 加法: lazy_array + 2 = {result_add}")

            # 减法
            result_sub = lazy_array - 1
            expected_sub = test_data - 1
            np.testing.assert_array_almost_equal(result_sub, expected_sub)
            print(f"✓ 减法: lazy_array - 1 = {result_sub}")

            # 乘法 (用户原始测试)
            result_mul = lazy_array * 4.1
            expected_mul = test_data * 4.1
            np.testing.assert_array_almost_equal(result_mul, expected_mul)
            print(f"✓ 乘法: lazy_array * 4.1 = {result_mul}")

            # 除法
            result_div = lazy_array / 2
            expected_div = test_data / 2
            np.testing.assert_array_almost_equal(result_div, expected_div)
            print(f"✓ 除法: lazy_array / 2 = {result_div}")

            # 幂运算
            result_pow = lazy_array ** 2
            expected_pow = test_data ** 2
            np.testing.assert_array_almost_equal(result_pow, expected_pow)
            print(f"✓ 幂运算: lazy_array ** 2 = {result_pow}")

            print("\n=== 测试比较操作符 ===")

            # 大于
            result_gt = lazy_array > 3
            expected_gt = test_data > 3
            np.testing.assert_array_equal(result_gt, expected_gt)
            print(f"✓ 大于: lazy_array > 3 = {result_gt}")

            # 小于
            result_lt = lazy_array < 3
            expected_lt = test_data < 3
            np.testing.assert_array_equal(result_lt, expected_lt)
            print(f"✓ 小于: lazy_array < 3 = {result_lt}")

            # 等于
            result_eq = lazy_array == 3
            expected_eq = test_data == 3
            np.testing.assert_array_equal(result_eq, expected_eq)
            print(f"✓ 等于: lazy_array == 3 = {result_eq}")

            # 大于等于
            result_ge = lazy_array >= 3
            expected_ge = test_data >= 3
            np.testing.assert_array_equal(result_ge, expected_ge)
            print(f"✓ 大于等于: lazy_array >= 3 = {result_ge}")

            print("\n=== 测试一元操作符 ===")

            # 取负
            result_neg = -lazy_array
            expected_neg = -test_data
            np.testing.assert_array_almost_equal(result_neg, expected_neg)
            print(f"✓ 取负: -lazy_array = {result_neg}")

            print("\n=== 测试反向操作符 ===")

            # 右操作数加法
            result_radd = 10 + lazy_array
            expected_radd = 10 + test_data
            np.testing.assert_array_almost_equal(result_radd, expected_radd)
            print(f"✓ 右加法: 10 + lazy_array = {result_radd}")

            # 右操作数乘法
            result_rmul = 3 * lazy_array
            expected_rmul = 3 * test_data
            np.testing.assert_array_almost_equal(result_rmul, expected_rmul)
            print(f"✓ 右乘法: 3 * lazy_array = {result_rmul}")

            print("\n=== 测试错误处理 ===")

            # 测试就地操作（应该报错）
            try:
                lazy_array += 1
                print("❌ 就地加法应该报错")
                return False
            except NotImplementedError as e:
                print(f"✓ 就地加法正确报错: {e}")

            try:
                lazy_array *= 2
                print("❌ 就地乘法应该报错")
                return False
            except NotImplementedError as e:
                print(f"✓ 就地乘法正确报错: {e}")

            print("\n=== 测试数组间操作 ===")

            # 与 NumPy 数组操作
            other_array = np.array([10, 20, 30, 40, 50], dtype=np.float32)
            result_array = lazy_array + other_array
            expected_array = test_data + other_array
            np.testing.assert_array_almost_equal(result_array, expected_array)
            print(f"✓ 数组加法: lazy_array + numpy_array = {result_array}")

            print("\n🎉 所有测试通过！LazyArray 算术操作符完全正常工作！")
            return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_comprehensive_operators()
    sys.exit(0 if success else 1)