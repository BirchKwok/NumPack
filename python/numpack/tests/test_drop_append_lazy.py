"""测试drop和append操作与lazy加载的一致性"""
import numpy as np
import pytest
import tempfile
import shutil
import os
from numpack import NumPack


class TestDropAppendLazyConsistency:
    """测试drop和append操作与lazy加载的形状一致性"""
    
    def test_drop_single_row_shape_consistency(self):
        """测试删除单行后lazy和eager加载的形状一致性"""
        test_data = np.random.rand(1000, 10).astype(np.float32)
        numpack_dir = tempfile.mkdtemp()
        
        try:
            # 保存数据
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
            
            # 删除第一行
            with NumPack(numpack_dir) as npk:
                npk.drop('data', 0)
                shape_after_drop = npk.get_shape('data')
                assert shape_after_drop == (999, 10), f"Expected (999, 10), got {shape_after_drop}"
            
            # 验证lazy和eager加载的形状一致
            with NumPack(numpack_dir) as npk:
                arr_eager = npk.load('data', lazy=False)
                arr_lazy = npk.load('data', lazy=True)
                
                assert arr_eager.shape == (999, 10), f"Eager shape: {arr_eager.shape}"
                assert arr_lazy.shape == (999, 10), f"Lazy shape: {arr_lazy.shape}"
                assert arr_eager.shape == arr_lazy.shape, "Lazy and eager shapes must match"
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_drop_multiple_rows_shape_consistency(self):
        """测试删除多行后lazy和eager加载的形状一致性"""
        test_data = np.random.rand(1000, 10).astype(np.float32)
        numpack_dir = tempfile.mkdtemp()
        
        try:
            # 保存数据
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
            
            # 删除多行
            with NumPack(numpack_dir) as npk:
                npk.drop('data', [0, 5, 10, 100])
                shape_after_drop = npk.get_shape('data')
                assert shape_after_drop == (996, 10), f"Expected (996, 10), got {shape_after_drop}"
            
            # 验证lazy和eager加载的形状一致
            with NumPack(numpack_dir) as npk:
                arr_eager = npk.load('data', lazy=False)
                arr_lazy = npk.load('data', lazy=True)
                
                assert arr_eager.shape == (996, 10), f"Eager shape: {arr_eager.shape}"
                assert arr_lazy.shape == (996, 10), f"Lazy shape: {arr_lazy.shape}"
                assert arr_eager.shape == arr_lazy.shape, "Lazy and eager shapes must match"
                
                # 注意：当存在deletion bitmap时，buffer protocol可能返回物理数据
                # 应该通过索引访问而不是np.array转换来验证数据一致性
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_append_after_drop_shape_consistency(self):
        """测试drop后append的形状一致性"""
        test_data = np.random.rand(1000000, 10).astype(np.float32)
        numpack_dir = tempfile.mkdtemp()
        
        try:
            # 保存数据
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
            
            # 删除一行
            with NumPack(numpack_dir) as npk:
                npk.drop('data', 0)
                shape_after_drop = npk.get_shape('data')
                assert shape_after_drop == (999999, 10), f"Expected (999999, 10), got {shape_after_drop}"
            
            # 追加数据
            with NumPack(numpack_dir) as npk:
                npk.append({'data': test_data})
                shape_after_append = npk.get_shape('data')
                assert shape_after_append == (1999999, 10), f"Expected (1999999, 10), got {shape_after_append}"
                
                # 验证lazy和eager加载的形状一致
                arr_eager = npk.load('data', lazy=False)
                arr_lazy = npk.load('data', lazy=True)
                
                assert arr_eager.shape == (1999999, 10), f"Eager shape: {arr_eager.shape}"
                assert arr_lazy.shape == (1999999, 10), f"Lazy shape: {arr_lazy.shape}"
                assert arr_eager.shape == arr_lazy.shape, "Lazy and eager shapes must match after append"
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_lazy_array_indexing_after_drop(self):
        """测试删除后lazy数组的索引访问"""
        test_data = np.arange(100 * 5, dtype=np.float32).reshape(100, 5)
        numpack_dir = tempfile.mkdtemp()
        
        try:
            # 保存数据
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
            
            # 删除索引 [5, 10, 15]
            with NumPack(numpack_dir) as npk:
                npk.drop('data', [5, 10, 15])
            
            # 验证lazy数组的索引访问
            with NumPack(numpack_dir) as npk:
                arr_lazy = npk.load('data', lazy=True)
                arr_eager = npk.load('data', lazy=False)
                
                # 形状应该一致
                assert arr_lazy.shape == arr_eager.shape == (97, 5)
                
                # 访问第一行（逻辑索引0，物理索引0）
                assert np.allclose(arr_lazy[0], arr_eager[0])
                
                # 访问第6行（逻辑索引5，物理索引6，因为物理索引5被删除）
                assert np.allclose(arr_lazy[5], arr_eager[5])
                
                # 批量索引
                assert np.allclose(arr_lazy[[0, 5, 10]], arr_eager[[0, 5, 10]])
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_lazy_array_iteration_after_drop(self):
        """测试删除后lazy数组的迭代"""
        test_data = np.arange(20 * 3, dtype=np.float32).reshape(20, 3)
        numpack_dir = tempfile.mkdtemp()
        
        try:
            # 保存数据
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
            
            # 删除索引 [0, 5, 10]
            with NumPack(numpack_dir) as npk:
                npk.drop('data', [0, 5, 10])
            
            # 验证lazy数组的迭代
            with NumPack(numpack_dir) as npk:
                arr_lazy = npk.load('data', lazy=True)
                arr_eager = npk.load('data', lazy=False)
                
                # 形状应该一致
                assert len(arr_lazy) == len(arr_eager) == 17
                
                # 迭代应该产生相同的行数
                lazy_rows = list(arr_lazy)
                assert len(lazy_rows) == 17, f"Expected 17 rows, got {len(lazy_rows)}"
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_update_removes_deletion_bitmap(self):
        """测试update操作后删除bitmap，lazy和eager形状一致"""
        test_data = np.random.rand(1000, 10).astype(np.float32)
        numpack_dir = tempfile.mkdtemp()
        
        try:
            # 保存数据
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
            
            # 删除一些行
            with NumPack(numpack_dir) as npk:
                npk.drop('data', [0, 1, 2])
                shape_after_drop = npk.get_shape('data')
                assert shape_after_drop == (997, 10)
            
            # 物理压缩
            with NumPack(numpack_dir) as npk:
                npk.update('data')
                shape_after_update = npk.get_shape('data')
                assert shape_after_update == (997, 10)
                
                # 验证lazy和eager加载的形状一致
                arr_eager = npk.load('data', lazy=False)
                arr_lazy = npk.load('data', lazy=True)
                
                assert arr_eager.shape == (997, 10), f"Eager shape: {arr_eager.shape}"
                assert arr_lazy.shape == (997, 10), f"Lazy shape: {arr_lazy.shape}"
                assert arr_eager.shape == arr_lazy.shape
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_no_bitmap_fast_path(self):
        """测试没有deletion bitmap时的快速路径性能"""
        test_data = np.random.rand(10000, 10).astype(np.float32)
        numpack_dir = tempfile.mkdtemp()
        
        try:
            # 保存数据
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
            
            # 没有删除操作，直接加载
            with NumPack(numpack_dir) as npk:
                import time
                
                # 测试lazy加载性能（应该很快，因为没有bitmap）
                start = time.time()
                arr_lazy = npk.load('data', lazy=True)
                lazy_time = time.time() - start
                
                assert arr_lazy.shape == (10000, 10)
                assert lazy_time < 0.1, f"Lazy load too slow: {lazy_time}s"  # Should be <100ms
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)


if __name__ == "__main__":
    import os
    # 手动运行测试
    test_cls = TestDropAppendLazyConsistency()
    test_cls.test_drop_single_row_shape_consistency()
    print("✓ test_drop_single_row_shape_consistency passed")
    
    test_cls.test_drop_multiple_rows_shape_consistency()
    print("✓ test_drop_multiple_rows_shape_consistency passed")
    
    test_cls.test_append_after_drop_shape_consistency()
    print("✓ test_append_after_drop_shape_consistency passed")
    
    test_cls.test_lazy_array_indexing_after_drop()
    print("✓ test_lazy_array_indexing_after_drop passed")
    
    test_cls.test_lazy_array_iteration_after_drop()
    print("✓ test_lazy_array_iteration_after_drop passed")
    
    test_cls.test_update_removes_deletion_bitmap()
    print("✓ test_update_removes_deletion_bitmap passed")
    
    test_cls.test_no_bitmap_fast_path()
    print("✓ test_no_bitmap_fast_path passed")
    
    print("\n✅ All tests passed!")

