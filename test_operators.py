#!/usr/bin/env python3
"""
简单的测试脚本，验证 LazyArray 操作符修复是否有效
"""

import numpy as np
import tempfile
import os
import sys

# 添加当前目录到 Python 路径
sys.path.insert(0, '/Users/guobingming/projects/NumPack/python')
sys.path.insert(0, '/Users/guobingming/projects/NumPack')

def test_lazy_array_operators():
    """测试 LazyArray 操作符"""
    print("开始测试 LazyArray 算术操作符...")

    try:
        # 导入 NumPack
        import numpack as npk
        print("✓ NumPack 导入成功")

        # 创建临时目录
        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"✓ 临时目录: {tmpdir}")

            # 创建测试数据
            test_data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
            print(f"✓ 测试数据: {test_data}")

            # 保存数据
            arrays = {'test_array': test_data}
            pack = npk.pack(tmpdir, arrays)
            print("✓ 数据保存成功")

            # 加载为 LazyArray
            lazy_array = npk.load(tmpdir, 'test_array', lazy=True)
            print(f"✓ LazyArray 加载成功: {type(lazy_array)}")

            # 测试基本操作符
            try:
                result = lazy_array * 4.1
                print(f"✓ 乘法操作成功: lazy_array * 4.1 = {result}")

                # 验证结果
                expected = test_data * 4.1
                np.testing.assert_array_almost_equal(result, expected)
                print("✓ 结果验证通过")

                # 测试其他操作符
                result_add = lazy_array + 2
                print(f"✓ 加法操作: lazy_array + 2 = {result_add}")

                result_sub = lazy_array - 1
                print(f"✓ 减法操作: lazy_array - 1 = {result_sub}")

                result_gt = lazy_array > 3
                print(f"✓ 比较操作: lazy_array > 3 = {result_gt}")

                print("\n🎉 所有测试通过！LazyArray 算术操作符工作正常！")
                return True

            except Exception as e:
                print(f"❌ 操作符测试失败: {e}")
                import traceback
                traceback.print_exc()
                return False

    except ImportError as e:
        print(f"❌ 导入 NumPack 失败: {e}")
        print("请确保 NumPack 已正确编译和安装")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_lazy_array_operators()
    sys.exit(0 if success else 1)