import os
import sys
import time
import logging
import numpy as np
from functools import wraps
from numpack import NumPack

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clean_file_when_finished(*filenames):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            finally:
                for filename in filenames:
                    try:
                        if os.path.isdir(filename):
                            for f in os.listdir(filename):
                                os.remove(os.path.join(filename, f))
                            os.rmdir(filename)
                        else:
                            os.remove(filename)
                        logger.info(f"清理测试文件: {filename}")
                    except FileNotFoundError:
                        pass
        return wrapper
    return decorator

@clean_file_when_finished('test_large', 'test_large.npz', 'test_large_array1.npy', 'test_large_array2.npy')
def test_large_data():
    """测试大数据处理"""
    logger.info("=== 测试大数据处理 ===")
    
    try:
        # 创建大数据
        size = 1000000  # 100万行
        arrays = {
            'array1': np.random.rand(size, 10).astype(np.float32),
            'array2': np.random.rand(size // 2, 5).astype(np.float32)
        }
        
        # 创建目录
        os.makedirs('test_large', exist_ok=True)
        
        # 测试 NumPack 保存
        logger.info(f"测试 NumPack 保存大数组 (array1: {arrays['array1'].shape}, array2: {arrays['array2'].shape})...")
        start_time = time.time()
        npk = NumPack('test_large')
        npk.save_arrays(arrays)
        save_time = time.time() - start_time
        logger.info(f"NumPack 保存耗时: {save_time:.2f}秒")
        
        # 测试 NumPy npz 保存
        logger.info("测试 NumPy npz 保存...")
        start_time = time.time()
        np.savez('test_large.npz', **arrays)
        npz_save_time = time.time() - start_time
        logger.info(f"NumPy npz 保存耗时: {npz_save_time:.2f}秒")
        logger.info(f"保存性能对比 (npz): NumPack/NumPy = {save_time/npz_save_time:.2f}x")
        
        # 测试 NumPy npy 保存
        logger.info("测试 NumPy npy 保存...")
        start_time = time.time()
        np.save('test_large_array1.npy', arrays['array1'])
        np.save('test_large_array2.npy', arrays['array2'])
        npy_save_time = time.time() - start_time
        logger.info(f"NumPy npy 保存耗时: {npy_save_time:.2f}秒")
        logger.info(f"保存性能对比 (npy): NumPack/NumPy = {save_time/npy_save_time:.2f}x")
        
        # 测试 NumPack 完整加载
        logger.info("\n\n测试 NumPack 完整加载...")
        start_time = time.time()
        loaded = npk.load_arrays(mmap_mode=False)
        load_time = time.time() - start_time
        logger.info(f"NumPack 加载耗时: {load_time:.2f}秒")
        
        # 测试 NumPack 选择性加载
        logger.info("测试 NumPack 选择性加载...")
        start_time = time.time()
        loaded_partial = npk.load_arrays(['array1'], mmap_mode=False)
        load_partial_time = time.time() - start_time
        logger.info(f"NumPack 选择性加载耗时: {load_partial_time:.2f}秒")
        
        # 测试 NumPy npz 加载
        logger.info("测试 NumPy npz 加载...")
        start_time = time.time()
        npz_loaded = np.load('test_large.npz')
        _, _ = npz_loaded['array1'], npz_loaded['array2']
        npz_load_time = time.time() - start_time
        logger.info(f"NumPy npz 加载耗时: {npz_load_time:.2f}秒")
        logger.info(f"加载性能对比 (npz): NumPack/NumPy = {load_time/npz_load_time:.2f}x")
        
        # 测试 NumPy npz 按需加载
        logger.info("测试 NumPy npz 按需加载...")
        start_time = time.time()
        npz_loaded = np.load('test_large.npz')
        npz_array1 = npz_loaded['array1']
        npz_array1_load_time = time.time() - start_time
        logger.info(f"NumPy npz 按需加载单个数组耗时: {npz_array1_load_time:.2f}秒")
        logger.info(f"按需加载性能对比 (npz): NumPack/NumPy = {load_partial_time/npz_array1_load_time:.2f}x")
        
        # 测试 NumPy npy 加载
        logger.info("\n\n测试 NumPy npy 加载...")
        start_time = time.time()
        npy_loaded = {
            'array1': np.load('test_large_array1.npy'),
            'array2': np.load('test_large_array2.npy')
        }
        npy_load_time = time.time() - start_time
        logger.info(f"NumPy npy 加载耗时: {npy_load_time:.2f}秒")
        logger.info(f"加载性能对比 (npy): NumPack/NumPy = {load_time/npy_load_time:.2f}x")
        
        # 测试 NumPack 内存映射加载
        logger.info("\n\n测试 NumPack 内存映射加载...")
        start_time = time.time()
        lazy_loaded = npk.load_arrays(mmap_mode=True)
        lazy_load_time = time.time() - start_time
        logger.info(f"NumPack 内存映射加载耗时: {lazy_load_time:.2f}秒")
        logger.info(f"内存映射加载性能对比 (npy): NumPack/NumPy = {lazy_load_time/npy_load_time:.2f}x")
        
        # 测试 NumPy npz 内存映射加载
        logger.info("测试 NumPy npz 内存映射加载...")
        start_time = time.time()
        npz_mmap = np.load('test_large.npz', mmap_mode='r')
        _, _ = npz_mmap['array1'], npz_mmap['array2']
        npz_mmap_time = time.time() - start_time
        logger.info(f"NumPy npz 内存映射加载耗时: {npz_mmap_time:.2f}秒")
        logger.info(f"内存映射加载性能对比 (npz): NumPack/NumPy = {lazy_load_time/npz_mmap_time:.2f}x")
        
        # 测试 NumPy npy 内存映射加载
        logger.info("测试 NumPy npy 内存映射加载...")
        start_time = time.time()
        npy_mmap = {
            'array1': np.load('test_large_array1.npy', mmap_mode='r'),
            'array2': np.load('test_large_array2.npy', mmap_mode='r')
        }
        npy_mmap_time = time.time() - start_time
        logger.info(f"NumPy npy 内存映射加载耗时: {npy_mmap_time:.2f}秒")
        logger.info(f"内存映射加载性能对比 (npy): NumPack/NumPy = {lazy_load_time/npy_mmap_time:.2f}x")
        
        # 验证 NumPack 数据
        for name, array in arrays.items():
            assert np.allclose(array, loaded[name])
            logger.info(f"NumPack 数组 '{name}' 验证通过")
            
        # 验证 NumPy npz 数据
        for name, array in arrays.items():
            assert np.allclose(array, npz_loaded[name])
            logger.info(f"NumPy npz 数组 '{name}' 验证通过")
            
        # 验证 NumPy npy 数据
        for name, array in arrays.items():
            assert np.allclose(array, npy_loaded[name])
            logger.info(f"NumPy npy 数组 '{name}' 验证通过")
        
        # 测试 NumPack 替换操作
        logger.info("测试 NumPack 大数据替换...")
        replace_data = {
            'array1': np.random.rand(size, 10).astype(np.float32)
        }
        start_time = time.time()
        npk.replace_arrays(replace_data, slice(None))
        replace_time = time.time() - start_time
        logger.info(f"NumPack 替换操作耗时: {replace_time:.2f}秒")
        
        # NumPy npz 不支持原地替换，需要重新保存整个文件
        logger.info("测试 NumPy npz 大数据替换...")
        npz_data = dict(np.load('test_large.npz'))
        npz_data.update(replace_data)
        start_time = time.time()
        np.savez('test_large.npz', **npz_data)
        npz_replace_time = time.time() - start_time
        logger.info(f"NumPy npz 替换操作耗时: {npz_replace_time:.2f}秒")
        logger.info(f"替换性能对比 (npz): NumPack/NumPy = {replace_time/npz_replace_time:.2f}x")
        
        # NumPy npy 不支持原地替换，需要重新保存文件
        logger.info("测试 NumPy npy 大数据替换...")
        start_time = time.time()
        np.save('test_large_array1.npy', replace_data['array1'])
        npy_replace_time = time.time() - start_time
        logger.info(f"NumPy npy 替换操作耗时: {npy_replace_time:.2f}秒")
        logger.info(f"替换性能对比 (npy): NumPack/NumPy = {replace_time/npy_replace_time:.2f}x")
        
        # 比较文件大小
        npk_size = sum(os.path.getsize(os.path.join('test_large', f)) for f in os.listdir('test_large')) / (1024 * 1024)  # MB
        npz_size = os.path.getsize('test_large.npz') / (1024 * 1024)  # MB
        npy_size = sum(os.path.getsize(f'test_large_{name}.npy') / (1024 * 1024) 
                      for name in ['array1', 'array2'])  # MB
        logger.info(f"\n文件大小对比:")
        logger.info(f"NumPack: {npk_size:.2f} MB")
        logger.info(f"NumPy npz: {npz_size:.2f} MB")
        logger.info(f"NumPy npy: {npy_size:.2f} MB")
        logger.info(f"大小比例 (vs npz): NumPack/NumPy = {npk_size/npz_size:.2f}x")
        logger.info(f"大小比例 (vs npy): NumPack/NumPy = {npk_size/npy_size:.2f}x")
        
        logger.info("大数据测试完成")
        
    except Exception as e:
        logger.error(f"大数据测试失败: {str(e)}")
        raise

@clean_file_when_finished('test_append', 'test_append.npz')
def test_append_operations():
    """测试追加操作"""
    logger.info("=== 测试追加操作 ===")
    
    try:
        # 创建初始数据
        size = 1000000  # 100万行
        arrays = {
            'array1': np.random.rand(size, 10).astype(np.float32),
            'array2': np.random.rand(size // 2, 5).astype(np.float32)
        }
        
        # 创建目录
        os.makedirs('test_append', exist_ok=True)
        
        # 保存初始数据
        npk = NumPack('test_append')
        npk.save_arrays(arrays)
        np.savez('test_append.npz', **arrays)
        
        # 创建要追加的数据
        append_data = {
            'array3': np.random.rand(size // 4, 8).astype(np.float32),
            'array4': np.random.rand(size // 8, 3).astype(np.float32)
        }
        
        # 测试 NumPack 追加
        logger.info("测试 NumPack 追加数组...")
        start_time = time.time()
        npk.save_arrays(append_data)
        append_time = time.time() - start_time
        logger.info(f"NumPack 追加操作耗时: {append_time:.2f}秒")
        
        # NumPy npz 不支持追加，需要重新保存整个文件
        logger.info("测试 NumPy npz 追加...")
        npz_data = dict(np.load('test_append.npz'))
        npz_data.update(append_data)
        start_time = time.time()
        np.savez('test_append.npz', **npz_data)
        npz_append_time = time.time() - start_time
        logger.info(f"NumPy npz 追加操作耗时: {npz_append_time:.2f}秒")
        logger.info(f"追加性能对比: NumPack/NumPy = {append_time/npz_append_time:.2f}x")
        
        # 加载并验证
        loaded = npk.load_arrays(mmap_mode=False)
        
        # 验证原有数据
        for name, array in arrays.items():
            assert np.allclose(array, loaded[name])
            logger.info(f"原有数组 '{name}' 验证通过")
        
        # 验证追加的数据
        for name, array in append_data.items():
            assert np.allclose(array, loaded[name])
            logger.info(f"追加数组 '{name}' 验证通过")
        
        logger.info("追加操作测试完成")
        
    except Exception as e:
        logger.error(f"测试追加操作失败: {str(e)}")
        raise

@clean_file_when_finished('test_random_access', 'test_random_access.npz', 'test_random_access_array1.npy', 'test_random_access_array2.npy')
def test_random_access():
    """测试随机访问性能"""
    logger.info("=== 测试随机访问性能 ===")
    
    try:
        # 创建测试数据
        size = 1000000  # 100万行
        arrays = {
            'array1': np.random.rand(size, 10).astype(np.float32),
            'array2': np.random.rand(size, 5).astype(np.float32)
        }
        
        # 创建目录
        os.makedirs('test_random_access', exist_ok=True)
        
        # 保存数组
        npk = NumPack('test_random_access')
        npk.save_arrays(arrays)
        np.savez('test_random_access.npz', **arrays)
        np.save('test_random_access_array1.npy', arrays['array1'])
        np.save('test_random_access_array2.npy', arrays['array2'])
        
        # 测试场景：
        # 1. 完全随机访问（随机抽取10000行）
        random_indices = np.random.randint(0, size, 10000)
        logger.info("\n测试完全随机访问性能...")
        
        # NumPack 随机访问
        start_time = time.time()
        lazy_arrays = npk.load_arrays(mmap_mode=True)
        array1 = lazy_arrays['array1']
        array2 = lazy_arrays['array2']
        numpack_random = {
            'array1': array1[random_indices],
            'array2': array2[random_indices]
        }
        numpack_random_time = time.time() - start_time
        logger.info(f"NumPack 随机访问耗时: {numpack_random_time:.2f}秒")
        
        # NumPy npz 随机访问
        start_time = time.time()
        npz_data = np.load('test_random_access.npz', mmap_mode='r')
        npz_random = {
            'array1': npz_data['array1'][random_indices],
            'array2': npz_data['array2'][random_indices]
        }
        npz_random_time = time.time() - start_time
        logger.info(f"NumPy npz 随机访问耗时: {npz_random_time:.2f}秒")
        logger.info(f"随机访问性能对比 (npz): NumPack/NumPy = {numpack_random_time/npz_random_time:.2f}x")
        
        # NumPy npy 随机访问
        start_time = time.time()
        npy_random = {
            'array1': np.load('test_random_access_array1.npy', mmap_mode='r')[random_indices],
            'array2': np.load('test_random_access_array2.npy', mmap_mode='r')[random_indices]
        }
        npy_random_time = time.time() - start_time
        logger.info(f"NumPy npy 随机访问耗时: {npy_random_time:.2f}秒")
        logger.info(f"随机访问性能对比 (npy): NumPack/NumPy = {numpack_random_time/npy_random_time:.2f}x")
        
        # 验证数据正确性
        for name in arrays:
            assert np.allclose(numpack_random[name], npz_random[name])
            assert np.allclose(numpack_random[name], npy_random[name])
            logger.info(f"随机访问数组 '{name}' 验证通过")
        
        logger.info("随机访问性能测试完成")
        
    except Exception as e:
        logger.error(f"随机访问性能测试失败: {str(e)}")
        raise

if __name__ == '__main__':
    try:
        logger.info("开始运行性能测试...")
        test_large_data()
        test_append_operations()
        test_random_access()
        logger.info("所有测试完成！")
    except Exception as e:
        logger.error(f"测试过程中发生错误: {str(e)}")
        sys.exit(1) 