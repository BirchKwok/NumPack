import os
import numpy as np
import tempfile
import threading
import multiprocessing as mp
import fcntl
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from numpack import save_nnp, load_nnp, replace_arrays, append_arrays, drop_arrays

def acquire_lock(file_path, exclusive=True):
    """获取文件锁"""
    lock_path = file_path + '.lock'
    max_attempts = 10
    attempt = 0
    
    while attempt < max_attempts:
        try:
            # 创建或打开锁文件
            lock_file = open(lock_path, 'w')
            
            # 尝试获取锁
            if exclusive:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                
            return lock_file
        except IOError:
            # 如果获取锁失败，等待一段时间后重试
            attempt += 1
            if attempt < max_attempts:
                time.sleep(0.1)
            else:
                raise
        except Exception as e:
            if os.path.exists(lock_path):
                os.remove(lock_path)
            raise e

def release_lock(lock_file):
    """释放文件锁"""
    try:
        # 释放锁
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()
        
        # 删除锁文件
        lock_path = lock_file.name
        if os.path.exists(lock_path):
            os.remove(lock_path)
    except Exception:
        pass

def test_basic_operations():
    # 创建测试数据
    array1 = np.random.rand(100, 100).astype(np.float32)
    array2 = np.random.rand(50, 200).astype(np.float32)
    arrays = {
        'array1': array1,
        'array2': array2
    }
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(suffix='.nnp', delete=False) as f:
        filename = f.name
    
    try:
        # 保存数组
        print(f"保存数组到文件: {filename}")
        print(f"array1 前5个元素: {array1.flatten()[:5]}")
        save_nnp(filename, arrays)
        
        # 加载数组
        loaded_arrays = load_nnp(filename)
        loaded_array1 = loaded_arrays['array1']
        loaded_array2 = loaded_arrays['array2']
        
        print(f"加载的 array1 前5个元素: {loaded_array1.flatten()[:5]}")
        print(f"array1 和 loaded_array1 的最大差值: {np.max(np.abs(array1 - loaded_array1))}")
        
        # 验证数组内容
        assert np.array_equal(array1, loaded_array1), "array1 的内容不匹配"
        assert np.array_equal(array2, loaded_array2), "array2 的内容不匹配"
        
        # 验证数组形状
        assert array1.shape == loaded_array1.shape, "array1 的形状不匹配"
        assert array2.shape == loaded_array2.shape, "array2 的形状不匹配"
        
        # 测试替换数组
        new_array1 = np.random.rand(100, 100).astype(np.float32)
        replace_arrays(filename, {'array1': new_array1}, slice(None))
        loaded_arrays = load_nnp(filename)
        assert np.array_equal(new_array1, loaded_arrays['array1']), "替换后的 array1 不匹配"
        
        # 测试追加数组
        array3 = np.random.rand(75, 150).astype(np.float32)
        append_arrays(filename, {'array3': array3})
        loaded_arrays = load_nnp(filename)
        assert np.array_equal(array3, loaded_arrays['array3']), "追加的 array3 不匹配"
        
        # 测试删除数组
        drop_arrays(filename, slice(None), ['array2'])
        loaded_arrays = load_nnp(filename)
        assert 'array2' not in loaded_arrays, "array2 未被删除"
        assert len(loaded_arrays) == 2, "数组数量不正确"
        
    finally:
        # 清理临时文件
        if os.path.exists(filename):
            os.remove(filename)

def concurrent_read_write(filename, thread_id):
    """并发读写测试函数"""
    try:
        print(f"线程/进程 {thread_id} 开始执行")
        
        # 创建新数组
        new_array = np.random.rand(100, 100).astype(np.float32)
        array_name = f'array_thread_{thread_id}'
        
        # 获取独占锁并写入
        print(f"线程/进程 {thread_id} 开始写入 {array_name}")
        lock_file = acquire_lock(filename, exclusive=True)
        try:
            append_arrays(filename, {array_name: new_array})
            print(f"线程/进程 {thread_id} 成功写入 {array_name}")
        finally:
            release_lock(lock_file)
        
        # 获取共享锁并读取验证
        lock_file = acquire_lock(filename, exclusive=False)
        try:
            arrays = load_nnp(filename)
            assert array_name in arrays, f"{array_name} 未找到"
            assert np.array_equal(new_array, arrays[array_name]), f"{array_name} 内容不匹配"
            print(f"线程/进程 {thread_id} 验证成功")
        finally:
            release_lock(lock_file)
        
        return True
    except Exception as e:
        print(f"线程/进程 {thread_id} 发生错误: {str(e)}")
        return False

def test_multithreading():
    """多线程测试"""
    print("\n开始多线程测试...")
    
    # 创建初始数据
    array1 = np.random.rand(100, 100).astype(np.float32)
    arrays = {'array1': array1}
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(suffix='.nnp', delete=False) as f:
        filename = f.name
    
    try:
        # 保存初始数组
        print(f"保存初始数组到文件: {filename}")
        save_nnp(filename, arrays)
        
        # 创建线程池
        num_threads = 4
        print(f"创建 {num_threads} 个线程")
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(concurrent_read_write, filename, i) 
                      for i in range(num_threads)]
            
            # 等待所有线程完成
            results = [future.result() for future in futures]
            
        # 验证所有线程都成功完成
        assert all(results), "某些线程操作失败"
        
        # 验证最终文件状态
        final_arrays = load_nnp(filename)
        assert len(final_arrays) == num_threads + 1, "最终数组数量不正确"
        print("多线程测试完成")
        
    finally:
        if os.path.exists(filename):
            os.remove(filename)

def test_multiprocessing():
    """多进程测试"""
    print("\n开始多进程测试...")
    
    # 创建初始数据
    array1 = np.random.rand(100, 100).astype(np.float32)
    arrays = {'array1': array1}
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(suffix='.nnp', delete=False) as f:
        filename = f.name
    
    try:
        # 保存初始数组
        print(f"保存初始数组到文件: {filename}")
        save_nnp(filename, arrays)
        
        # 创建进程池
        num_processes = 4
        print(f"创建 {num_processes} 个进程")
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [executor.submit(concurrent_read_write, filename, i) 
                      for i in range(num_processes)]
            
            # 等待所有进程完成
            results = [future.result() for future in futures]
            
        # 验证所有进程都成功完成
        assert all(results), "某些进程操作失败"
        
        # 验证最终文件状态
        final_arrays = load_nnp(filename)
        assert len(final_arrays) == num_processes + 1, "最终数组数量不正确"
        print("多进程测试完成")
        
    finally:
        if os.path.exists(filename):
            os.remove(filename)

if __name__ == '__main__':
    print("运行基本操作测试...")
    test_basic_operations()
    print("基本操作测试通过！")
    
    print("\n运行多线程测试...")
    test_multithreading()
    print("多线程测试通过！")
    
    print("\n运行多进程测试...")
    test_multiprocessing()
    print("多进程测试通过！") 