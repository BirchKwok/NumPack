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
    print("示例1: 删除数组中的特定行")
    print("="*60)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        with NumPack(temp_dir, drop_if_exists=True) as npk:
            # 创建测试数据
            data = np.arange(100).reshape(10, 10).astype(np.float32)
            npk.save({'data': data})
            print(f"原始数据: shape={npk.get_shape('data')}")
            
            # 删除单行
            npk.drop('data', 5)
            print(f"删除第5行后: shape={npk.get_shape('data')}")
            
            # 删除多行
            npk.drop('data', [0, 2, 4])
            print(f"删除第0,2,4行后: shape={npk.get_shape('data')}")
            
            # 加载数据验证
            result = npk.load('data')
            print(f"最终数据形状: {result.shape}")
            print(f"剩余行数: {result.shape[0]}\n")
    finally:
        shutil.rmtree(temp_dir)


def example_drop_array():
    """示例：删除整个数组"""
    print("="*60)
    print("示例2: 删除整个数组")
    print("="*60)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        with NumPack(temp_dir, drop_if_exists=True) as npk:
            # 创建多个数组
            data1 = np.random.rand(100, 10).astype(np.float32)
            data2 = np.random.rand(200, 10).astype(np.float32)
            data3 = np.random.rand(300, 10).astype(np.float32)
            
            npk.save({'data1': data1, 'data2': data2, 'data3': data3})
            print(f"创建的数组: {npk.get_member_list()}")
            
            # 删除单个数组
            npk.drop('data2')
            print(f"删除data2后: {npk.get_member_list()}")
            
            # 删除多个数组
            npk.drop(['data1', 'data3'])
            print(f"删除data1和data3后: {npk.get_member_list()}\n")
    finally:
        shutil.rmtree(temp_dir)


def example_drop_and_append():
    """示例：删除后追加新数据"""
    print("="*60)
    print("示例3: 删除后追加新数据")
    print("="*60)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        with NumPack(temp_dir, drop_if_exists=True) as npk:
            # 创建初始数据
            data = np.arange(1000).reshape(100, 10).astype(np.float32)
            npk.save({'data': data})
            print(f"初始数据: shape={npk.get_shape('data')}")
            
            # 删除最后20行
            npk.drop('data', list(range(80, 100)))
            print(f"删除最后20行后: shape={npk.get_shape('data')}")
            
            # 追加30行新数据
            new_data = np.ones((30, 10), dtype=np.float32) * 999
            npk.append({'data': new_data})
            print(f"追加30行后: shape={npk.get_shape('data')}")
            
            # 验证数据
            result = npk.load('data')
            print(f"最终形状: {result.shape}")
            print(f"前80行来自原始数据")
            print(f"后30行是新追加的数据（值=999）")
            print(f"验证: 最后一行的值 = {result[-1, 0]}\n")
    finally:
        shutil.rmtree(temp_dir)


def example_drop_with_negative_index():
    """示例：使用负数索引删除"""
    print("="*60)
    print("示例4: 使用负数索引删除")
    print("="*60)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        with NumPack(temp_dir, drop_if_exists=True) as npk:
            data = np.arange(100).reshape(10, 10).astype(np.float32)
            npk.save({'data': data})
            print(f"原始数据: shape={npk.get_shape('data')}")
            
            # 使用负数索引删除最后一行
            npk.drop('data', -1)
            print(f"删除最后一行（-1）后: shape={npk.get_shape('data')}")
            
            # 删除最后3行
            npk.drop('data', [-1, -2, -3])
            print(f"删除最后3行后: shape={npk.get_shape('data')}\n")
    finally:
        shutil.rmtree(temp_dir)


def example_physical_compact():
    """示例：物理整合（compact）删除的行"""
    print("="*60)
    print("示例5: 物理整合删除的行")
    print("="*60)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        with NumPack(temp_dir, drop_if_exists=True) as npk:
            # 创建数据
            data = np.random.rand(1000, 10).astype(np.float32)
            npk.save({'data': data})
            
            # 逻辑删除一些行
            npk.drop('data', list(range(0, 500)))
            print(f"逻辑删除500行后: shape={npk.get_shape('data')}")
            print("注意: 数据文件仍包含1000行，使用bitmap标记删除")
            
            # 物理整合：真正删除这些行
            npk.update('data')  # update方法会触发compact
            print(f"物理整合后: shape={npk.get_shape('data')}")
            print("数据文件现在只包含500行，bitmap被删除\n")
    finally:
        shutil.rmtree(temp_dir)


def main():
    """运行所有示例"""
    print("\n" + "="*60)
    print("NumPack Drop操作使用示例")
    print("="*60 + "\n")
    
    example_drop_rows()
    example_drop_array()
    example_drop_and_append()
    example_drop_with_negative_index()
    example_physical_compact()
    
    print("="*60)
    print("所有示例完成!")
    print("="*60)


if __name__ == "__main__":
    main()


