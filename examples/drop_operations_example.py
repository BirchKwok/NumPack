#!/usr/bin/env python3
"""
NumPack Drop操作使用示例

演示如何使用drop功能删除数组中的行或整个数组
"""

import numpy as np
from numpack import NumPack
import tempfile
import shutil


def example_drop_rows():
    """示例：删除数组中的特定行"""
    print("="*60)
    print("Example 1: Delete specific rows from array")
    print("="*60)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        with NumPack(temp_dir, drop_if_exists=True) as npk:
            # 创建测试数据
            data = np.arange(100).reshape(10, 10).astype(np.float32)
            npk.save({'data': data})
            print(f"Original data: shape={npk.get_shape('data')}")
            
            # 删除单行
            npk.drop('data', 5)
            print(f"After deleting row 5: shape={npk.get_shape('data')}")
            
            # 删除多行
            npk.drop('data', [0, 2, 4])
            print(f"After deleting rows 0,2,4: shape={npk.get_shape('data')}")
            
            # 加载数据验证
            result = npk.load('data')
            print(f"Final data shape: {result.shape}")
            print(f"Remaining rows: {result.shape[0]}\n")
    finally:
        shutil.rmtree(temp_dir)


def example_drop_array():
    """示例：删除整个数组"""
    print("="*60)
    print("Example 2: Delete entire arrays")
    print("="*60)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        with NumPack(temp_dir, drop_if_exists=True) as npk:
            # 创建多个数组
            data1 = np.random.rand(100, 10).astype(np.float32)
            data2 = np.random.rand(200, 10).astype(np.float32)
            data3 = np.random.rand(300, 10).astype(np.float32)
            
            npk.save({'data1': data1, 'data2': data2, 'data3': data3})
            print(f"Created arrays: {npk.get_member_list()}")
            
            # 删除单个数组
            npk.drop('data2')
            print(f"After deleting data2: {npk.get_member_list()}")
            
            # 删除多个数组
            npk.drop(['data1', 'data3'])
            print(f"After deleting data1 and data3: {npk.get_member_list()}\n")
    finally:
        shutil.rmtree(temp_dir)


def example_drop_and_append():
    """示例：删除后追加新数据"""
    print("="*60)
    print("Example 3: Append new data after deletion")
    print("="*60)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        with NumPack(temp_dir, drop_if_exists=True) as npk:
            # 创建初始数据
            data = np.arange(1000).reshape(100, 10).astype(np.float32)
            npk.save({'data': data})
            print(f"Initial data: shape={npk.get_shape('data')}")
            
            # 删除最后20行
            npk.drop('data', list(range(80, 100)))
            print(f"After deleting last 20 rows: shape={npk.get_shape('data')}")
            
            # 追加30行新数据
            new_data = np.ones((30, 10), dtype=np.float32) * 999
            npk.append({'data': new_data})
            print(f"After appending 30 rows: shape={npk.get_shape('data')}")
            
            # 验证数据
            result = npk.load('data')
            print(f"Final shape: {result.shape}")
            print(f"First 80 rows are from original data")
            print(f"Last 30 rows are newly appended data (value=999)")
            print(f"Verification: value of last row = {result[-1, 0]}\n")
    finally:
        shutil.rmtree(temp_dir)


def example_drop_with_negative_index():
    """示例：使用负数索引删除"""
    print("="*60)
    print("Example 4: Delete using negative indices")
    print("="*60)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        with NumPack(temp_dir, drop_if_exists=True) as npk:
            data = np.arange(100).reshape(10, 10).astype(np.float32)
            npk.save({'data': data})
            print(f"Original data: shape={npk.get_shape('data')}")
            
            # 使用负数索引删除最后一行
            npk.drop('data', -1)
            print(f"After deleting last row (-1): shape={npk.get_shape('data')}")
            
            # 删除最后3行
            npk.drop('data', [-1, -2, -3])
            print(f"After deleting last 3 rows: shape={npk.get_shape('data')}\n")
    finally:
        shutil.rmtree(temp_dir)


def example_physical_compact():
    """示例：物理整合（compact）删除的行"""
    print("="*60)
    print("Example 5: Physical compact of deleted rows")
    print("="*60)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        with NumPack(temp_dir, drop_if_exists=True) as npk:
            # 创建数据
            data = np.random.rand(1000, 10).astype(np.float32)
            npk.save({'data': data})
            
            # 逻辑删除一些行
            npk.drop('data', list(range(0, 500)))
            print(f"After logical deletion of 500 rows: shape={npk.get_shape('data')}")
            print("Note: Data file still contains 1000 rows, using bitmap to mark deletion")
            
            # 物理整合：真正删除这些行
            npk.update('data')  # update方法会触发compact
            print(f"After physical compact: shape={npk.get_shape('data')}")
            print("Data file now contains only 500 rows, bitmap is removed\n")
    finally:
        shutil.rmtree(temp_dir)


def main():
    """运行所有示例"""
    print("\n" + "="*60)
    print("NumPack Drop Operations Usage Examples")
    print("="*60 + "\n")
    
    example_drop_rows()
    example_drop_array()
    example_drop_and_append()
    example_drop_with_negative_index()
    example_physical_compact()
    
    print("="*60)
    print("All examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()


