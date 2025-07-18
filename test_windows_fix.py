#!/usr/bin/env python3
"""
测试Windows平台的文件句柄清理功能
"""

import os
import sys
import tempfile
import numpy as np
import gc
import time
from pathlib import Path

# 确保能导入numpack
sys.path.insert(0, 'python')

def test_windows_cleanup():
    """测试Windows平台的清理功能"""
    print("测试Windows平台文件句柄清理...")
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"使用临时目录: {temp_dir}")
        
        try:
            from numpack import NumPack, force_cleanup_windows_handles
            print("成功导入numpack模块")
        except ImportError as e:
            print(f"导入失败: {e}")
            return False
        
        # 创建NumPack实例
        npk = NumPack(temp_dir)
        
        # 测试1: 基本保存和加载
        print("\n测试1: 基本保存和加载")
        data = {'test_array': np.random.rand(100, 32).astype(np.float32)}
        npk.save(data)
        
        # 正常加载
        loaded = npk.load('test_array')
        print(f"正常加载成功，形状: {loaded.shape}")
        
        # 懒加载测试
        print("\n测试2: 懒加载和迭代器")
        lazy_arr = npk.load('test_array', lazy=True)
        print(f"懒加载成功，形状: {lazy_arr.shape}")
        
        # 测试迭代器
        rows = []
        for i, row in enumerate(lazy_arr):
            rows.append(row)
            if i >= 2:  # 只取前3行
                break
        
        print(f"迭代器测试成功，获取了 {len(rows)} 行")
        
        # 测试Windows特定清理
        print("\n测试3: Windows特定清理")
        if os.name == 'nt':
            try:
                force_cleanup_windows_handles()
                print("Windows清理函数调用成功")
            except Exception as e:
                print(f"Windows清理函数调用失败: {e}")
        else:
            print("非Windows平台，跳过Windows特定清理测试")
        
        # 强制垃圾回收
        print("\n测试4: 垃圾回收测试")
        for i in range(5):
            gc.collect()
            time.sleep(0.01)
        print("垃圾回收完成")
        
        # 测试多个LazyArray
        print("\n测试5: 多个LazyArray测试")
        for i in range(3):
            data_name = f'test_array_{i}'
            test_data = {data_name: np.random.rand(50, 16).astype(np.float32)}
            npk.save(test_data)
            
            lazy_arr = npk.load(data_name, lazy=True)
            # 快速访问几行
            _ = lazy_arr[0]
            _ = lazy_arr[10:20]
            print(f"LazyArray {i} 测试完成")
        
        print("\n所有测试完成!")
        
        # 最终清理
        if os.name == 'nt':
            try:
                force_cleanup_windows_handles()
            except:
                pass
            
            for _ in range(3):
                gc.collect()
                time.sleep(0.02)
            
            time.sleep(0.1)
        
        return True

if __name__ == '__main__':
    try:
        success = test_windows_cleanup()
        if success:
            print("\n✅ 测试成功完成!")
            sys.exit(0)
        else:
            print("\n❌ 测试失败!")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 