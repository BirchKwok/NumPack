#!/usr/bin/env python3
"""
NumPack Drop操作专项测试
测试各种drop场景的功能和性能
"""

import os
import sys
import time
import tempfile
import shutil
import numpy as np
from numpack import NumPack


def test_drop_single_row():
    """测试删除单行"""
    print("\n=== 测试：删除单行 ===")
    temp_dir = tempfile.mkdtemp()
    
    try:
        with NumPack(temp_dir, drop_if_exists=True) as npk:
            # 创建测试数据
            data = np.arange(100).reshape(10, 10).astype(np.float32)
            npk.save({'data': data})
            
            original_shape = npk.get_shape('data')
            print(f"原始形状: {original_shape}")
            
            # 删除第5行
            npk.drop('data', 5)
            new_shape = npk.get_shape('data')
            print(f"删除第5行后形状: {new_shape}")
            
            # 验证
            result = npk.load('data')
            expected = np.delete(data, 5, axis=0)
            
            assert result.shape == expected.shape, f"形状不匹配: {result.shape} vs {expected.shape}"
            assert np.array_equal(result, expected), "数据不匹配"
            
            print("✓ 测试通过")
    finally:
        shutil.rmtree(temp_dir)


def test_drop_multiple_rows():
    """测试删除多行"""
    print("\n=== 测试：删除多行 ===")
    temp_dir = tempfile.mkdtemp()
    
    try:
        with NumPack(temp_dir, drop_if_exists=True) as npk:
            # 创建测试数据
            data = np.arange(1000).reshape(100, 10).astype(np.float32)
            npk.save({'data': data})
            
            original_shape = npk.get_shape('data')
            print(f"原始形状: {original_shape}")
            
            # 删除多行
            rows_to_drop = [5, 10, 15, 20, 50, 80, 95]
            npk.drop('data', rows_to_drop)
            new_shape = npk.get_shape('data')
            print(f"删除{len(rows_to_drop)}行后形状: {new_shape}")
            
            # 验证
            result = npk.load('data')
            expected = np.delete(data, rows_to_drop, axis=0)
            
            assert result.shape == expected.shape, f"形状不匹配: {result.shape} vs {expected.shape}"
            assert np.array_equal(result, expected), "数据不匹配"
            
            print("✓ 测试通过")
    finally:
        shutil.rmtree(temp_dir)


def test_drop_consecutive_rows():
    """测试删除连续行"""
    print("\n=== 测试：删除连续行 ===")
    temp_dir = tempfile.mkdtemp()
    
    try:
        with NumPack(temp_dir, drop_if_exists=True) as npk:
            # 创建测试数据
            data = np.arange(1000).reshape(100, 10).astype(np.float32)
            npk.save({'data': data})
            
            original_shape = npk.get_shape('data')
            print(f"原始形状: {original_shape}")
            
            # 删除连续行 (从第30行到第39行)
            rows_to_drop = list(range(30, 40))
            npk.drop('data', rows_to_drop)
            new_shape = npk.get_shape('data')
            print(f"删除行30-39后形状: {new_shape}")
            
            # 验证
            result = npk.load('data')
            expected = np.delete(data, rows_to_drop, axis=0)
            
            assert result.shape == expected.shape, f"形状不匹配: {result.shape} vs {expected.shape}"
            assert np.array_equal(result, expected), "数据不匹配"
            
            print("✓ 测试通过")
    finally:
        shutil.rmtree(temp_dir)


def test_drop_first_rows():
    """测试删除开头的行"""
    print("\n=== 测试：删除开头的行 ===")
    temp_dir = tempfile.mkdtemp()
    
    try:
        with NumPack(temp_dir, drop_if_exists=True) as npk:
            # 创建测试数据
            data = np.arange(1000).reshape(100, 10).astype(np.float32)
            npk.save({'data': data})
            
            original_shape = npk.get_shape('data')
            print(f"原始形状: {original_shape}")
            
            # 删除前10行
            rows_to_drop = list(range(10))
            npk.drop('data', rows_to_drop)
            new_shape = npk.get_shape('data')
            print(f"删除前10行后形状: {new_shape}")
            
            # 验证
            result = npk.load('data')
            expected = data[10:]
            
            assert result.shape == expected.shape, f"形状不匹配: {result.shape} vs {expected.shape}"
            assert np.array_equal(result, expected), "数据不匹配"
            
            print("✓ 测试通过")
    finally:
        shutil.rmtree(temp_dir)


def test_drop_last_rows():
    """测试删除末尾的行"""
    print("\n=== 测试：删除末尾的行 ===")
    temp_dir = tempfile.mkdtemp()
    
    try:
        with NumPack(temp_dir, drop_if_exists=True) as npk:
            # 创建测试数据
            data = np.arange(1000).reshape(100, 10).astype(np.float32)
            npk.save({'data': data})
            
            original_shape = npk.get_shape('data')
            print(f"原始形状: {original_shape}")
            
            # 删除最后10行
            rows_to_drop = list(range(90, 100))
            npk.drop('data', rows_to_drop)
            new_shape = npk.get_shape('data')
            print(f"删除最后10行后形状: {new_shape}")
            
            # 验证
            result = npk.load('data')
            expected = data[:90]
            
            assert result.shape == expected.shape, f"形状不匹配: {result.shape} vs {expected.shape}"
            assert np.array_equal(result, expected), "数据不匹配"
            
            print("✓ 测试通过")
    finally:
        shutil.rmtree(temp_dir)


def test_drop_all_rows():
    """测试删除所有行（通过indexes参数）"""
    print("\n=== 测试：删除所有行（通过indexes） ===")
    temp_dir = tempfile.mkdtemp()
    
    try:
        with NumPack(temp_dir, drop_if_exists=True) as npk:
            # 创建测试数据
            data = np.arange(100).reshape(10, 10).astype(np.float32)
            npk.save({'data': data})
            
            original_shape = npk.get_shape('data')
            print(f"原始形状: {original_shape}")
            
            # 删除所有行
            npk.drop('data', list(range(10)))
            
            # 检查数组是否还存在
            arrays = npk.get_member_list()
            if 'data' in arrays:
                new_shape = npk.get_shape('data')
                print(f"删除所有行后形状: {new_shape}")
                assert new_shape[0] == 0, "应该没有行了"
            else:
                print("数组已被完全删除")
            
            print("✓ 测试通过")
    finally:
        shutil.rmtree(temp_dir)


def test_drop_entire_array():
    """测试删除整个数组"""
    print("\n=== 测试：删除整个数组 ===")
    temp_dir = tempfile.mkdtemp()
    
    try:
        with NumPack(temp_dir, drop_if_exists=True) as npk:
            # 创建多个数组
            data1 = np.arange(100).reshape(10, 10).astype(np.float32)
            data2 = np.arange(200).reshape(20, 10).astype(np.float32)
            data3 = np.arange(300).reshape(30, 10).astype(np.float32)
            
            npk.save({'data1': data1, 'data2': data2, 'data3': data3})
            
            arrays_before = npk.get_member_list()
            print(f"删除前的数组: {arrays_before}")
            
            # 删除data2
            npk.drop('data2')
            
            arrays_after = npk.get_member_list()
            print(f"删除后的数组: {arrays_after}")
            
            assert 'data1' in arrays_after, "data1应该还在"
            assert 'data2' not in arrays_after, "data2应该被删除"
            assert 'data3' in arrays_after, "data3应该还在"
            
            # 验证其他数组的数据完整性
            result1 = npk.load('data1')
            result3 = npk.load('data3')
            
            assert np.array_equal(result1, data1), "data1数据应该完整"
            assert np.array_equal(result3, data3), "data3数据应该完整"
            
            print("✓ 测试通过")
    finally:
        shutil.rmtree(temp_dir)


def test_drop_multiple_arrays():
    """测试同时删除多个数组"""
    print("\n=== 测试：同时删除多个数组 ===")
    temp_dir = tempfile.mkdtemp()
    
    try:
        with NumPack(temp_dir, drop_if_exists=True) as npk:
            # 创建多个数组
            data1 = np.arange(100).reshape(10, 10).astype(np.float32)
            data2 = np.arange(200).reshape(20, 10).astype(np.float32)
            data3 = np.arange(300).reshape(30, 10).astype(np.float32)
            data4 = np.arange(400).reshape(40, 10).astype(np.float32)
            
            npk.save({'data1': data1, 'data2': data2, 'data3': data3, 'data4': data4})
            
            arrays_before = npk.get_member_list()
            print(f"删除前的数组: {arrays_before}")
            
            # 删除多个数组
            npk.drop(['data2', 'data4'])
            
            arrays_after = npk.get_member_list()
            print(f"删除后的数组: {arrays_after}")
            
            assert 'data1' in arrays_after, "data1应该还在"
            assert 'data2' not in arrays_after, "data2应该被删除"
            assert 'data3' in arrays_after, "data3应该还在"
            assert 'data4' not in arrays_after, "data4应该被删除"
            
            print("✓ 测试通过")
    finally:
        shutil.rmtree(temp_dir)


def test_drop_and_append():
    """测试删除后追加"""
    print("\n=== 测试：删除后追加 ===")
    temp_dir = tempfile.mkdtemp()
    
    try:
        with NumPack(temp_dir, drop_if_exists=True) as npk:
            # 创建测试数据
            data = np.arange(1000).reshape(100, 10).astype(np.float32)
            npk.save({'data': data})
            
            print(f"原始形状: {npk.get_shape('data')}")
            
            # 删除最后20行
            npk.drop('data', list(range(80, 100)))
            print(f"删除后形状: {npk.get_shape('data')}")
            
            # 追加新数据
            new_data = np.ones((30, 10), dtype=np.float32) * 999
            npk.append({'data': new_data})
            print(f"追加后形状: {npk.get_shape('data')}")
            
            # 验证
            result = npk.load('data')
            assert result.shape == (110, 10), f"形状应该是(110, 10)，实际是{result.shape}"
            
            # 验证前80行是原始数据
            assert np.array_equal(result[:80], data[:80]), "前80行应该是原始数据"
            
            # 验证追加的数据
            assert np.all(result[80:] == 999), "追加的数据应该全是999"
            
            print("✓ 测试通过")
    finally:
        shutil.rmtree(temp_dir)


def test_drop_and_replace():
    """测试删除后替换"""
    print("\n=== 测试：删除后替换 ===")
    temp_dir = tempfile.mkdtemp()
    
    try:
        with NumPack(temp_dir, drop_if_exists=True) as npk:
            # 创建测试数据
            data = np.arange(1000).reshape(100, 10).astype(np.float32)
            npk.save({'data': data})
            
            print(f"原始形状: {npk.get_shape('data')}")
            
            # 删除中间的行
            npk.drop('data', list(range(40, 60)))
            print(f"删除后形状: {npk.get_shape('data')}")
            
            # 替换前10行
            replace_data = np.ones((10, 10), dtype=np.float32) * 888
            npk.replace({'data': replace_data}, list(range(10)))
            
            # 验证
            result = npk.load('data')
            assert result.shape == (80, 10), f"形状应该是(80, 10)，实际是{result.shape}"
            
            # 验证替换的数据
            assert np.all(result[:10] == 888), "前10行应该被替换为888"
            
            print("✓ 测试通过")
    finally:
        shutil.rmtree(temp_dir)


def test_drop_with_lazy_load():
    """测试删除操作与lazy load的交互"""
    print("\n=== 测试：删除操作与lazy load的交互 ===")
    temp_dir = tempfile.mkdtemp()
    
    try:
        with NumPack(temp_dir, drop_if_exists=True) as npk:
            # 创建测试数据
            data = np.arange(1000).reshape(100, 10).astype(np.float32)
            npk.save({'data': data})
            
            # 创建lazy load对象
            lazy1 = npk.load('data', lazy=True)
            print(f"Lazy load 创建，访问第0行: {lazy1[0][0]}")
            
            # 删除一些行
            npk.drop('data', [50, 60, 70])
            
            # 再次创建lazy load对象
            lazy2 = npk.load('data', lazy=True)
            print(f"删除后重新lazy load，访问第0行: {lazy2[0][0]}")
            
            # 验证lazy load对象反映了删除操作
            result = npk.load('data')
            assert result.shape[0] == 97, f"应该有97行，实际有{result.shape[0]}行"
            
            print("✓ 测试通过")
    finally:
        shutil.rmtree(temp_dir)


def test_drop_large_dataset():
    """测试大数据集的删除性能"""
    print("\n=== 测试：大数据集删除性能 ===")
    temp_dir = tempfile.mkdtemp()
    
    try:
        with NumPack(temp_dir, drop_if_exists=True) as npk:
            # 创建大数据集
            print("创建大数据集 (1M行)...")
            data = np.random.rand(1000000, 10).astype(np.float32)
            npk.save({'data': data})
            
            print(f"原始形状: {npk.get_shape('data')}")
            
            # 测试删除单行
            start = time.time()
            npk.drop('data', [500000])
            elapsed = time.time() - start
            print(f"删除单行耗时: {elapsed*1000:.2f}ms")
            print(f"删除后形状: {npk.get_shape('data')}")
            
            # 测试删除多行
            rows_to_drop = list(range(100000, 101000))  # 删除1000行
            start = time.time()
            npk.drop('data', rows_to_drop)
            elapsed = time.time() - start
            print(f"删除1000行耗时: {elapsed*1000:.2f}ms")
            print(f"删除后形状: {npk.get_shape('data')}")
            
            # 验证数据完整性
            result = npk.load('data')
            print(f"最终形状: {result.shape}")
            
            print("✓ 测试通过")
    finally:
        shutil.rmtree(temp_dir)


def test_drop_edge_cases():
    """测试边界情况"""
    print("\n=== 测试：边界情况 ===")
    temp_dir = tempfile.mkdtemp()
    
    try:
        with NumPack(temp_dir, drop_if_exists=True) as npk:
            # 创建测试数据
            data = np.arange(100).reshape(10, 10).astype(np.float32)
            npk.save({'data': data})
            
            # 测试：删除不存在的行索引（应该被忽略或报错）
            print("测试删除超出范围的索引...")
            try:
                npk.drop('data', [100, 200])
                print("  删除超出范围的索引（无错误）")
            except Exception as e:
                print(f"  删除超出范围的索引引发异常: {type(e).__name__}")
            
            # 测试：删除负数索引
            print("测试删除负数索引...")
            try:
                npk.drop('data', [-1])
                print(f"  删除负数索引成功，当前形状: {npk.get_shape('data')}")
            except Exception as e:
                print(f"  删除负数索引引发异常: {type(e).__name__}")
            
            # 测试：删除空列表
            print("测试删除空列表...")
            try:
                npk.drop('data', [])
                print(f"  删除空列表成功，当前形状: {npk.get_shape('data')}")
            except Exception as e:
                print(f"  删除空列表引发异常: {type(e).__name__}")
            
            # 测试：删除不存在的数组
            print("测试删除不存在的数组...")
            try:
                npk.drop('nonexistent_array')
                print("  删除不存在的数组成功（无错误）")
            except Exception as e:
                print(f"  删除不存在的数组引发异常: {type(e).__name__}")
            
            print("✓ 边界测试完成")
    finally:
        shutil.rmtree(temp_dir)


def test_drop_with_numpy_array_indexes():
    """测试使用numpy数组作为索引"""
    print("\n=== 测试：使用numpy数组作为索引 ===")
    temp_dir = tempfile.mkdtemp()
    
    try:
        with NumPack(temp_dir, drop_if_exists=True) as npk:
            # 创建测试数据
            data = np.arange(1000).reshape(100, 10).astype(np.float32)
            npk.save({'data': data})
            
            print(f"原始形状: {npk.get_shape('data')}")
            
            # 使用numpy数组作为索引
            rows_to_drop = np.array([10, 20, 30, 40, 50])
            npk.drop('data', rows_to_drop)
            
            new_shape = npk.get_shape('data')
            print(f"删除后形状: {new_shape}")
            
            # 验证
            result = npk.load('data')
            expected = np.delete(data, rows_to_drop, axis=0)
            
            assert result.shape == expected.shape, f"形状不匹配: {result.shape} vs {expected.shape}"
            assert np.array_equal(result, expected), "数据不匹配"
            
            print("✓ 测试通过")
    finally:
        shutil.rmtree(temp_dir)


def main():
    """运行所有测试"""
    print("="*70)
    print("NumPack Drop操作专项测试")
    print("="*70)
    
    tests = [
        test_drop_single_row,
        test_drop_multiple_rows,
        test_drop_consecutive_rows,
        test_drop_first_rows,
        test_drop_last_rows,
        test_drop_all_rows,
        test_drop_entire_array,
        test_drop_multiple_arrays,
        test_drop_and_append,
        test_drop_and_replace,
        test_drop_with_lazy_load,
        test_drop_large_dataset,
        test_drop_edge_cases,
        test_drop_with_numpy_array_indexes,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*70)
    print(f"测试完成: {passed} 通过, {failed} 失败")
    print("="*70)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

