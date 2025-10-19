#!/usr/bin/env python3
"""
测试用户原始失败案例
"""

import numpy as np
import tempfile
import sys

# 添加当前目录到 Python 路径

def test_user_case():
    """测试用户的原始失败案例"""
    print("测试用户原始失败案例: a *= 4.1")

    try:
        # 导入 NumPack
        from numpack import _lib_numpack
        NumPack = _lib_numpack.NumPack
        print("✓ NumPack 导入成功")

        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建测试数据
            a_data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
            print(f"✓ 创建测试数据: {a_data}")

            # 保存数据
            npk = NumPack(tmpdir)
            arrays = {'a': a_data}
            npk.save(arrays, None)
            print("✓ 数据保存成功")

            # 加载为 LazyArray
            a = npk.load('a', lazy=True)
            print(f"✓ LazyArray 加载成功: {type(a)}")

            # 测试用户原始操作
            print("测试: a *= 4.1")
            try:
                a *= 4.1  # 这应该会报错，因为 LazyArray 是只读的
                print("❌ 意外成功：a *= 4.1 应该报错")
                return False
            except (TypeError, NotImplementedError) as e:
                print(f"✓ 预期错误：{e}")

            # 测试替代方案
            print("测试: b = a * 4.1")
            b = a * 4.1
            print(f"✓ 成功：b = a * 4.1 = {b}")

            # 验证结果
            expected = a_data * 4.1
            np.testing.assert_array_almost_equal(b, expected)
            print("✓ 结果验证通过")

            print("\n🎉 用户案例测试通过！LazyArray 现在支持算术操作符！")
            return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_user_case()
    sys.exit(0 if success else 1)