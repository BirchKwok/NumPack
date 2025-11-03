"""测试NumPack API的线程安全性"""
import numpy as np
import pytest
import tempfile
import shutil
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from numpack import NumPack


class TestThreadSafety:
    """测试NumPack在多线程环境下的安全性"""
    
    def test_concurrent_reads_same_array(self):
        """测试多个线程同时读取同一个数组"""
        test_data = np.random.rand(1000, 10).astype(np.float32)
        numpack_dir = tempfile.mkdtemp()
        num_threads = 10
        
        try:
            # 先保存数据
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
            
            # 多线程读取
            def read_worker(thread_id):
                with NumPack(numpack_dir) as npk:
                    loaded = npk.load('data')
                    # 验证数据完整性
                    return np.allclose(loaded, test_data) and loaded.shape == test_data.shape
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(read_worker, i) for i in range(num_threads)]
                results = [f.result() for f in as_completed(futures)]
            
            # 所有读取都应该成功
            assert all(results), "Some threads failed to read correctly"
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_concurrent_reads_different_arrays(self):
        """测试多个线程读取不同数组"""
        num_arrays = 20
        numpack_dir = tempfile.mkdtemp()
        
        try:
            # 创建多个数组
            test_arrays = {}
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                for i in range(num_arrays):
                    array_name = f'array_{i}'
                    test_arrays[array_name] = np.random.rand(100, 10).astype(np.float32)
                npk.save(test_arrays)
            
            # 多线程读取不同数组
            def read_worker(array_id):
                array_name = f'array_{array_id}'
                with NumPack(numpack_dir) as npk:
                    loaded = npk.load(array_name)
                    expected = test_arrays[array_name]
                    return np.allclose(loaded, expected)
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(read_worker, i) for i in range(num_arrays)]
                results = [f.result() for f in as_completed(futures)]
            
            assert all(results), "Some threads failed to read correctly"
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_concurrent_writes_different_arrays(self):
        """测试多个线程写入不同数组
        
        注意：并发写入需要适当的同步机制。
        在实际应用中，建议使用锁来序列化写入操作。
        """
        num_threads = 10
        numpack_dir = tempfile.mkdtemp()
        
        try:
            # 初始化空文件
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                pass
            
            # 多线程写入不同数组 - 使用锁来确保线程安全
            test_data = {}
            data_lock = threading.Lock()
            write_lock = threading.Lock()  # 序列化写入操作
            
            def write_worker(thread_id):
                array_name = f'array_{thread_id}'
                data = np.random.rand(100, 10).astype(np.float32)
                
                with data_lock:  # 保护共享的test_data字典
                    test_data[array_name] = data.copy()
                
                # 使用锁来序列化写入操作，避免并发冲突
                with write_lock:
                    with NumPack(numpack_dir) as npk:
                        npk.save({array_name: data})
                return True
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(write_worker, i) for i in range(num_threads)]
                results = [f.result() for f in as_completed(futures)]
            
            assert all(results), "Some writes failed"
            
            # 验证所有数据都正确写入
            with NumPack(numpack_dir) as npk:
                members = npk.get_member_list()
                assert len(members) == num_threads, f"Expected {num_threads} arrays, got {len(members)}: {members}"
                
                for array_name, expected_data in test_data.items():
                    loaded = npk.load(array_name)
                    assert np.allclose(loaded, expected_data)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_concurrent_append_same_array(self):
        """测试多个线程对同一数组进行append操作"""
        num_threads = 5
        rows_per_thread = 20
        numpack_dir = tempfile.mkdtemp()
        
        try:
            # 初始化数组
            initial_data = np.random.rand(100, 10).astype(np.float32)
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': initial_data})
            
            # 多线程append（需要锁来序列化append操作）
            append_lock = threading.Lock()
            
            def append_worker(thread_id):
                data = np.ones((rows_per_thread, 10), dtype=np.float32) * thread_id
                
                with append_lock:  # 序列化append操作
                    with NumPack(numpack_dir) as npk:
                        npk.append({'data': data})
                return True
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(append_worker, i) for i in range(num_threads)]
                results = [f.result() for f in as_completed(futures)]
            
            assert all(results), "Some appends failed"
            
            # 验证最终大小
            with NumPack(numpack_dir) as npk:
                final_shape = npk.get_shape('data')
                expected_rows = 100 + num_threads * rows_per_thread
                assert final_shape == (expected_rows, 10)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_concurrent_drop_different_rows(self):
        """测试多个线程删除不同行"""
        numpack_dir = tempfile.mkdtemp()
        num_threads = 5
        
        try:
            # 创建足够大的数组
            initial_data = np.arange(1000).reshape(100, 10).astype(np.float32)
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': initial_data})
            
            # 多线程删除不同的行（需要锁来序列化）
            drop_lock = threading.Lock()
            
            def drop_worker(thread_id):
                # 每个线程删除不同的索引范围
                start_idx = thread_id * 10
                end_idx = start_idx + 5
                indices = list(range(start_idx, end_idx))
                
                with drop_lock:  # 序列化drop操作
                    with NumPack(numpack_dir) as npk:
                        npk.drop('data', indices)
                return True
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(drop_worker, i) for i in range(num_threads)]
                results = [f.result() for f in as_completed(futures)]
            
            assert all(results), "Some drops failed"
            
            # 验证删除结果
            with NumPack(numpack_dir) as npk:
                final_shape = npk.get_shape('data')
                expected_rows = 100 - (num_threads * 5)
                assert final_shape == (expected_rows, 10)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_concurrent_replace_operations(self):
        """测试多个线程进行replace操作"""
        numpack_dir = tempfile.mkdtemp()
        num_threads = 10
        
        try:
            # 初始化数组
            initial_data = np.zeros((100, 10), dtype=np.float32)
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': initial_data})
            
            # 多线程replace不同的行
            replace_lock = threading.Lock()
            
            def replace_worker(thread_id):
                # 每个线程替换不同的行
                index = thread_id
                replacement = np.ones((1, 10), dtype=np.float32) * (thread_id + 1)
                
                with replace_lock:  # 序列化replace操作
                    with NumPack(numpack_dir) as npk:
                        npk.replace({'data': replacement}, index)
                return True
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(replace_worker, i) for i in range(num_threads)]
                results = [f.result() for f in as_completed(futures)]
            
            assert all(results), "Some replaces failed"
            
            # 验证替换结果
            with NumPack(numpack_dir) as npk:
                loaded = npk.load('data')
                for i in range(num_threads):
                    expected_value = i + 1
                    assert np.allclose(loaded[i], expected_value)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_concurrent_mixed_operations(self):
        """测试混合并发操作（读、写、更新）"""
        numpack_dir = tempfile.mkdtemp()
        num_operations = 30
        
        try:
            # 初始化多个数组
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                for i in range(10):
                    data = np.random.rand(100, 10).astype(np.float32)
                    npk.save({f'array_{i}': data})
            
            operation_lock = threading.Lock()
            
            def mixed_worker(op_id):
                op_type = op_id % 3
                array_id = op_id % 10
                array_name = f'array_{array_id}'
                
                try:
                    with NumPack(numpack_dir) as npk:
                        if op_type == 0:  # Read
                            loaded = npk.load(array_name)
                            return ('read', loaded.shape[0] > 0)
                        elif op_type == 1:  # Get shape
                            shape = npk.get_shape(array_name)
                            return ('shape', shape[1] == 10)
                        else:  # Get member list
                            members = npk.get_member_list()
                            return ('list', len(members) == 10)
                except Exception as e:
                    return ('error', False)
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(mixed_worker, i) for i in range(num_operations)]
                results = [f.result() for f in as_completed(futures)]
            
            # 所有操作都应该成功
            for op_type, success in results:
                assert success, f"Operation {op_type} failed"
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_concurrent_lazy_load(self):
        """测试多线程lazy加载"""
        numpack_dir = tempfile.mkdtemp()
        num_threads = 10
        
        try:
            # 创建测试数据
            test_data = np.random.rand(1000, 10).astype(np.float32)
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
            
            # 多线程lazy加载
            def lazy_load_worker(thread_id):
                with NumPack(numpack_dir) as npk:
                    lazy_array = npk.load('data', lazy=True)
                    # 访问一些数据
                    row = lazy_array[thread_id % 100]
                    return row.shape == (10,)
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(lazy_load_worker, i) for i in range(num_threads)]
                results = [f.result() for f in as_completed(futures)]
            
            assert all(results), "Some lazy loads failed"
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_concurrent_batch_mode(self):
        """测试多线程使用batch_mode"""
        numpack_dir = tempfile.mkdtemp()
        
        try:
            # 初始化数据
            test_data = np.random.rand(100, 10).astype(np.float32)
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
            
            # 注意：batch_mode本身不是线程安全的，每个线程应该有自己的NumPack实例
            def batch_mode_worker(thread_id):
                try:
                    with NumPack(numpack_dir) as npk:
                        with npk.batch_mode():
                            loaded = npk.load('data')
                            # 简单验证
                            return loaded.shape == (100, 10)
                except Exception as e:
                    return False
            
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(batch_mode_worker, i) for i in range(5)]
                results = [f.result() for f in as_completed(futures)]
            
            assert all(results), "Some batch mode operations failed"
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_concurrent_getitem(self):
        """测试多线程使用getitem访问"""
        numpack_dir = tempfile.mkdtemp()
        num_threads = 10
        
        try:
            # 创建测试数据
            test_data = np.arange(1000).reshape(100, 10).astype(np.float32)
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
            
            # 多线程getitem
            def getitem_worker(thread_id):
                with NumPack(numpack_dir) as npk:
                    # 访问不同的索引
                    indices = [thread_id, (thread_id + 10) % 100]
                    items = npk.getitem('data', indices)
                    return items.shape == (2, 10)
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(getitem_worker, i) for i in range(num_threads)]
                results = [f.result() for f in as_completed(futures)]
            
            assert all(results), "Some getitem operations failed"
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_thread_safety_with_context_manager(self):
        """测试使用context manager时的线程安全性"""
        numpack_dir = tempfile.mkdtemp()
        num_threads = 10
        
        try:
            # 初始化
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                initial_data = np.random.rand(100, 10).astype(np.float32)
                npk.save({'data': initial_data})
            
            # 多线程使用context manager
            def context_worker(thread_id):
                try:
                    with NumPack(numpack_dir) as npk:
                        # 读取
                        loaded = npk.load('data')
                        # 获取形状
                        shape = npk.get_shape('data')
                        # 验证
                        return shape == (100, 10) and loaded.shape == shape
                except Exception as e:
                    return False
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(context_worker, i) for i in range(num_threads)]
                results = [f.result() for f in as_completed(futures)]
            
            assert all(results), "Some context manager operations failed"
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_multiple_numpack_instances_same_file(self):
        """测试多个NumPack实例同时访问同一文件"""
        numpack_dir = tempfile.mkdtemp()
        num_instances = 5
        
        try:
            # 初始化数据
            test_data = np.random.rand(100, 10).astype(np.float32)
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
            
            # 多个实例同时读取
            def instance_worker(instance_id):
                try:
                    # 每个线程创建自己的NumPack实例
                    with NumPack(numpack_dir) as npk:
                        loaded = npk.load('data')
                        return np.allclose(loaded, test_data)
                except Exception as e:
                    return False
            
            with ThreadPoolExecutor(max_workers=num_instances) as executor:
                futures = [executor.submit(instance_worker, i) for i in range(num_instances)]
                results = [f.result() for f in as_completed(futures)]
            
            assert all(results), "Some instances failed to read correctly"
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_concurrent_stream_load(self):
        """测试多线程使用stream_load"""
        numpack_dir = tempfile.mkdtemp()
        num_threads = 5
        
        try:
            # 创建测试数据
            test_data = np.arange(1000).reshape(100, 10).astype(np.float32)
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
            
            # 多线程stream_load
            def stream_load_worker(thread_id):
                try:
                    with NumPack(numpack_dir) as npk:
                        count = 0
                        for chunk in npk.stream_load('data', buffer_size=10):
                            count += chunk.shape[0]
                        return count == 100
                except Exception as e:
                    return False
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(stream_load_worker, i) for i in range(num_threads)]
                results = [f.result() for f in as_completed(futures)]
            
            assert all(results), "Some stream_load operations failed"
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)


class TestThreadSafetyStress:
    """压力测试线程安全性"""
    
    def test_high_concurrency_reads(self):
        """高并发读取压力测试"""
        numpack_dir = tempfile.mkdtemp()
        num_threads = 50
        iterations_per_thread = 10
        
        try:
            # 创建测试数据
            test_data = np.random.rand(1000, 10).astype(np.float32)
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
            
            # 高并发读取
            def stress_read_worker(thread_id):
                success_count = 0
                for i in range(iterations_per_thread):
                    try:
                        with NumPack(numpack_dir) as npk:
                            loaded = npk.load('data')
                            if loaded.shape == test_data.shape:
                                success_count += 1
                    except Exception:
                        pass
                return success_count
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(stress_read_worker, i) for i in range(num_threads)]
                results = [f.result() for f in as_completed(futures)]
            
            # 至少大部分操作应该成功
            total_operations = num_threads * iterations_per_thread
            successful_operations = sum(results)
            success_rate = successful_operations / total_operations
            
            assert success_rate > 0.95, f"Success rate too low: {success_rate}"
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_rapid_open_close(self):
        """快速打开关闭压力测试"""
        numpack_dir = tempfile.mkdtemp()
        num_threads = 20
        
        try:
            # 初始化数据
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': np.random.rand(100, 10).astype(np.float32)})
            
            # 快速打开关闭
            def rapid_open_close_worker(thread_id):
                success_count = 0
                for i in range(5):
                    try:
                        with NumPack(numpack_dir) as npk:
                            members = npk.get_member_list()
                            if 'data' in members:
                                success_count += 1
                    except Exception:
                        pass
                return success_count
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(rapid_open_close_worker, i) for i in range(num_threads)]
                results = [f.result() for f in as_completed(futures)]
            
            # 大部分操作应该成功
            assert sum(results) >= num_threads * 4  # 至少80%成功
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

