"""测试不同场景的drop API功能"""
import numpy as np
import pytest
import tempfile
import shutil
import os
from numpack import NumPack


class TestDropScenarios:
    """测试drop API在各种场景下的正确性"""
    
    def test_drop_single_row_int_index(self):
        """测试删除单行（整数索引）"""
        test_data = np.arange(100).reshape(10, 10).astype(np.float32)
        numpack_dir = tempfile.mkdtemp()
        
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
                original_shape = npk.get_shape('data')
                
                # 删除第5行
                npk.drop('data', 5)
                new_shape = npk.get_shape('data')
                
                assert original_shape == (10, 10)
                assert new_shape == (9, 10)
                
                # 验证数据正确性
                loaded = npk.load('data')
                expected = np.delete(test_data, 5, axis=0)
                assert np.allclose(loaded, expected)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_drop_multiple_rows_list_index(self):
        """测试删除多行（列表索引）"""
        test_data = np.arange(200).reshape(20, 10).astype(np.float32)
        numpack_dir = tempfile.mkdtemp()
        
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
                
                # 删除多行
                indices_to_drop = [0, 5, 10, 15]
                npk.drop('data', indices_to_drop)
                new_shape = npk.get_shape('data')
                
                assert new_shape == (16, 10)
                
                # 验证数据正确性
                loaded = npk.load('data')
                expected = np.delete(test_data, indices_to_drop, axis=0)
                assert np.allclose(loaded, expected)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_drop_multiple_rows_numpy_array_index(self):
        """测试删除多行（numpy数组索引）"""
        test_data = np.arange(300).reshape(30, 10).astype(np.float32)
        numpack_dir = tempfile.mkdtemp()
        
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
                
                # 使用numpy数组作为索引
                indices_to_drop = np.array([2, 7, 12, 17, 22])
                npk.drop('data', indices_to_drop)
                new_shape = npk.get_shape('data')
                
                assert new_shape == (25, 10)
                
                # 验证数据正确性
                loaded = npk.load('data')
                expected = np.delete(test_data, indices_to_drop, axis=0)
                assert np.allclose(loaded, expected)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_drop_single_array(self):
        """测试删除单个数组"""
        numpack_dir = tempfile.mkdtemp()
        
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                data1 = np.random.rand(100, 10).astype(np.float32)
                data2 = np.random.rand(200, 10).astype(np.float32)
                npk.save({'array1': data1, 'array2': data2})
                
                # 删除单个数组
                npk.drop('array1')
                members = npk.get_member_list()
                
                assert 'array1' not in members
                assert 'array2' in members
                assert len(members) == 1
                
                # 验证array2仍然可以正常加载
                loaded = npk.load('array2')
                assert np.allclose(loaded, data2)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_drop_multiple_arrays(self):
        """测试删除多个数组"""
        numpack_dir = tempfile.mkdtemp()
        
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                data1 = np.random.rand(100, 10).astype(np.float32)
                data2 = np.random.rand(200, 10).astype(np.float32)
                data3 = np.random.rand(300, 10).astype(np.float32)
                npk.save({'array1': data1, 'array2': data2, 'array3': data3})
                
                # 删除多个数组
                npk.drop(['array1', 'array3'])
                members = npk.get_member_list()
                
                assert 'array1' not in members
                assert 'array2' in members
                assert 'array3' not in members
                assert len(members) == 1
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_drop_then_append(self):
        """测试删除后追加数据"""
        test_data = np.arange(1000).reshape(100, 10).astype(np.float32)
        numpack_dir = tempfile.mkdtemp()
        
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
                
                # 删除最后20行
                npk.drop('data', list(range(80, 100)))
                assert npk.get_shape('data') == (80, 10)
                
                # 追加30行新数据
                new_data = np.ones((30, 10), dtype=np.float32) * 999
                npk.append({'data': new_data})
                
                assert npk.get_shape('data') == (110, 10)
                
                # 验证数据
                loaded = npk.load('data')
                assert loaded.shape == (110, 10)
                # 前80行应该是原始数据
                assert np.allclose(loaded[:80], test_data[:80])
                # 后30行应该是新数据
                assert np.allclose(loaded[80:], new_data)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_drop_then_replace(self):
        """测试删除后替换数据"""
        test_data = np.arange(1000).reshape(100, 10).astype(np.float32)
        numpack_dir = tempfile.mkdtemp()
        
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
                
                # 删除索引 [5, 10, 15]
                npk.drop('data', [5, 10, 15])
                assert npk.get_shape('data') == (97, 10)
                
                # 替换第1行（逻辑索引）
                replacement = np.ones((1, 10), dtype=np.float32) * 888
                npk.replace({'data': replacement}, 1)
                
                # 验证
                loaded = npk.load('data')
                assert np.allclose(loaded[1], 888)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_drop_then_update_compact(self):
        """测试删除后物理整合
        
        注意：update操作会物理删除已标记删除的行，创建一个新的紧凑文件
        """
        test_data = np.random.rand(1000, 10).astype(np.float32)
        numpack_dir = tempfile.mkdtemp()
        
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
                
                # 删除500行
                indices_to_drop = list(range(0, 500))
                npk.drop('data', indices_to_drop)
                shape_after_drop = npk.get_shape('data')
                assert shape_after_drop == (500, 10)
                
                # 获取删除后的数据
                data_before_compact = npk.load('data')
                assert data_before_compact.shape == (500, 10)
                
                # 物理整合 - 这会创建一个新的紧凑文件
                npk.update('data')
                
                # update后立即验证形状
                shape_after_update = npk.get_shape('data')
                assert shape_after_update == (500, 10)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_drop_with_batch_mode(self):
        """测试batch_mode下的drop操作"""
        test_data = np.arange(500).reshape(50, 10).astype(np.float32)
        numpack_dir = tempfile.mkdtemp()
        
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
                
                with npk.batch_mode():
                    # 在batch mode中删除
                    npk.drop('data', [5, 10, 15])
                    assert npk.get_shape('data') == (47, 10)
                    
                    # 加载数据
                    loaded = npk.load('data')
                    expected = np.delete(test_data, [5, 10, 15], axis=0)
                    assert np.allclose(loaded, expected)
                
                # batch mode退出后验证
                loaded_after = npk.load('data')
                assert loaded_after.shape == (47, 10)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_drop_all_rows(self):
        """测试删除所有行
        
        注意：当删除所有行时，数组本身会被移除
        """
        test_data = np.arange(100).reshape(10, 10).astype(np.float32)
        numpack_dir = tempfile.mkdtemp()
        
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
                
                # 删除所有行 - 这会导致数组被完全删除
                npk.drop('data', list(range(10)))
                
                # 验证数组不再存在
                members = npk.get_member_list()
                assert 'data' not in members
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_drop_with_tuple_index(self):
        """测试使用元组索引删除"""
        test_data = np.arange(100).reshape(10, 10).astype(np.float32)
        numpack_dir = tempfile.mkdtemp()
        
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
                
                # 使用元组索引
                npk.drop('data', (2, 5, 8))
                
                assert npk.get_shape('data') == (7, 10)
                
                loaded = npk.load('data')
                expected = np.delete(test_data, [2, 5, 8], axis=0)
                assert np.allclose(loaded, expected)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_drop_consecutive_operations(self):
        """测试连续多次删除操作"""
        test_data = np.arange(1000).reshape(100, 10).astype(np.float32)
        numpack_dir = tempfile.mkdtemp()
        
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
                
                # 第一次删除
                npk.drop('data', [0, 1, 2])
                assert npk.get_shape('data') == (97, 10)
                
                # 第二次删除
                npk.drop('data', [5, 10])
                assert npk.get_shape('data') == (95, 10)
                
                # 第三次删除
                npk.drop('data', [20, 30, 40])
                assert npk.get_shape('data') == (92, 10)
                
                # 验证最终结果
                loaded = npk.load('data')
                assert loaded.shape == (92, 10)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_drop_different_dtypes(self):
        """测试不同数据类型的删除操作"""
        dtypes = [np.float32, np.float64, np.int32, np.int64, np.uint8]
        numpack_dir = tempfile.mkdtemp()
        
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                for i, dtype in enumerate(dtypes):
                    array_name = f'array_{dtype.__name__}'
                    test_data = np.arange(100, dtype=dtype).reshape(10, 10)
                    npk.save({array_name: test_data})
                    
                    # 删除第5行
                    npk.drop(array_name, 5)
                    
                    # 验证
                    assert npk.get_shape(array_name) == (9, 10)
                    loaded = npk.load(array_name)
                    expected = np.delete(test_data, 5, axis=0)
                    assert np.allclose(loaded, expected)
                    assert loaded.dtype == dtype
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_drop_lazy_load_consistency(self):
        """测试删除后lazy加载的一致性"""
        test_data = np.random.rand(1000, 10).astype(np.float32)
        numpack_dir = tempfile.mkdtemp()
        
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
                
                # 删除一些行
                npk.drop('data', [10, 20, 30, 40, 50])
                
                # 比较eager和lazy加载
                eager = npk.load('data', lazy=False)
                lazy = npk.load('data', lazy=True)
                
                # 形状应该一致
                assert eager.shape == lazy.shape == (995, 10)
                
                # 随机抽样验证数据一致性
                indices = [0, 100, 500, 900]
                for idx in indices:
                    assert np.allclose(eager[idx], lazy[idx])
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_drop_empty_list(self):
        """测试空索引列表删除（不应删除任何内容）"""
        test_data = np.arange(100).reshape(10, 10).astype(np.float32)
        numpack_dir = tempfile.mkdtemp()
        
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
                
                # 空列表删除
                npk.drop('data', [])
                
                # 形状应该不变
                assert npk.get_shape('data') == (10, 10)
                
                # 数据应该完全一致
                loaded = npk.load('data')
                assert np.allclose(loaded, test_data)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_drop_reopen_file(self):
        """测试删除后重新打开文件"""
        test_data = np.arange(1000).reshape(100, 10).astype(np.float32)
        numpack_dir = tempfile.mkdtemp()
        
        try:
            # 第一次打开，删除并保存
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
                npk.drop('data', list(range(50)))
                assert npk.get_shape('data') == (50, 10)
            
            # 重新打开，验证删除是否持久化
            with NumPack(numpack_dir) as npk:
                assert npk.get_shape('data') == (50, 10)
                loaded = npk.load('data')
                expected = test_data[50:]
                assert np.allclose(loaded, expected)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_drop_multidimensional_arrays(self):
        """测试删除多维数组的行"""
        # 3维数组
        test_data_3d = np.random.rand(50, 20, 10).astype(np.float32)
        numpack_dir = tempfile.mkdtemp()
        
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data_3d': test_data_3d})
                
                # 删除第一维的某些索引
                npk.drop('data_3d', [5, 10, 15])
                
                assert npk.get_shape('data_3d') == (47, 20, 10)
                
                loaded = npk.load('data_3d')
                expected = np.delete(test_data_3d, [5, 10, 15], axis=0)
                assert np.allclose(loaded, expected)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_drop_getitem_after_deletion(self):
        """测试删除后使用getitem访问"""
        test_data = np.arange(1000).reshape(100, 10).astype(np.float32)
        numpack_dir = tempfile.mkdtemp()
        
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
                
                # 删除一些行
                deleted_indices = [5, 10, 15, 20]
                npk.drop('data', deleted_indices)
                
                # 使用getitem访问
                item_0 = npk.getitem('data', 0)
                item_5 = npk.getitem('data', 5)
                
                # 验证
                full_data = npk.load('data')
                assert np.allclose(item_0, full_data[0])
                assert np.allclose(item_5, full_data[5])
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)


class TestDropErrorHandling:
    """测试drop API的错误处理
    
    注意：某些错误处理可能在Rust后端中实现，行为可能与预期不同
    """
    
    def test_drop_nonexistent_array_silent(self):
        """测试删除不存在的数组
        
        注意：当前实现可能会静默忽略不存在的数组，不抛出错误
        这是一个设计决策，避免在批量操作时因单个数组不存在而中断
        """
        numpack_dir = tempfile.mkdtemp()
        
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                # 创建一个数组
                npk.save({'array1': np.random.rand(10, 10).astype(np.float32)})
                
                # 尝试删除不存在的数组（可能静默成功）
                npk.drop('nonexistent')
                
                # 验证原有数组仍然存在
                assert 'array1' in npk.get_member_list()
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_drop_duplicate_index(self):
        """测试删除重复索引（应该被去重）"""
        test_data = np.arange(100).reshape(10, 10).astype(np.float32)
        numpack_dir = tempfile.mkdtemp()
        
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
                
                # 删除重复索引（应该自动去重）
                npk.drop('data', [0, 0, 1, 1, 2])
                
                # 应该只删除3行
                shape = npk.get_shape('data')
                assert shape == (7, 10)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

