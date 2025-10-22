#!/usr/bin/env python
"""
LazyArray 原地操作符示例

LazyArray 现在支持原地操作符（+=, -=, *=, /=, //=, %=, **=）。

注意：由于 LazyArray 是只读的内存映射数组，原地操作符会自动转换为
非原地版本，并返回一个新的 numpy 数组。这是符合预期的行为。

例如：
    a *= 4.1  # 等价于 a = a * 4.1
    
结果会是一个 numpy 数组，而不是 LazyArray。
"""

import numpy as np
import numpack as npk
from pathlib import Path
import shutil

def main():
    """演示 LazyArray 的原地操作符"""
    
    # 创建测试数据
    test_dir = Path("example_inplace_npk")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    try:
        # 创建并保存数据
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        
        with npk.NumPack(test_dir) as pack:
            pack.save({'array': data})
        
        # 加载 LazyArray 并使用原地操作符
        with npk.NumPack(test_dir) as pack:
            # 示例 1: 乘法
            lazy_array = pack.load('array', lazy=True)
            print(f"Original data:\n{np.array(lazy_array)}\n")
            
            # 使用原地乘法操作符
            result = lazy_array
            result *= 2.5
            print(f"After result *= 2.5:\n{result}")
            print(f"Result type: {type(result)}\n")
            
            # 示例 2: 加法
            lazy_array = pack.load('array', lazy=True)
            result = lazy_array
            result += 100
            print(f"After result += 100:\n{result}\n")
            
            # 示例 3: 连续操作
            lazy_array = pack.load('array', lazy=True)
            result = lazy_array
            result *= 2
            result += 10
            result /= 3
            print(f"After chain operations (×2, +10, ÷3):\n{result}\n")
            
    finally:
        # 清理
        if test_dir.exists():
            shutil.rmtree(test_dir)

if __name__ == "__main__":
    main()

