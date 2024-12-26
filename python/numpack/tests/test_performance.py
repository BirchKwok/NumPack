import os
import sys
import time
import logging
import numpy as np
from functools import wraps
from numpack import save_nnp, load_nnp, replace_arrays, append_arrays, drop_arrays, getitem

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
                        os.remove(filename)
                        logger.info(f"清理测试文件: {filename}")
                    except FileNotFoundError:
                        pass
        return wrapper
    return decorator

@clean_file_when_finished('test_large.nnp', 'test_large.npz', 'test_large_array1.npy', 'test_large_array2.npy')
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
        
        # 测试 NumPack 保存
        logger.info(f"测试 NumPack 保存大数组 (array1: {arrays['array1'].shape}, array2: {arrays['array2'].shape})...")
        start_time = time.time()
        save_nnp('test_large.nnp', arrays)
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
        logger.info("测试 NumPack 完整加载...")
        start_time = time.time()
        loaded = load_nnp('test_large.nnp', mmap_mode=False)
        load_time = time.time() - start_time
        logger.info(f"NumPack 加载耗时: {load_time:.2f}秒")
        
        # 测试 NumPack 选择性加载
        logger.info("测试 NumPack 选择性加载...")
        start_time = time.time()
        loaded_partial = load_nnp('test_large.nnp', array_names=['array1'])
        load_partial_time = time.time() - start_time
        logger.info(f"NumPack 选择性加载耗时: {load_partial_time:.2f}秒")
        
        # 测试 NumPy npz 加载
        logger.info("测试 NumPy npz 加载...")
        start_time = time.time()
        npz_loaded = dict(np.load('test_large.npz'))
        npz_load_time = time.time() - start_time
        logger.info(f"NumPy npz 加载耗时: {npz_load_time:.2f}秒")
        logger.info(f"加载性能对比 (npz): NumPack/NumPy = {load_time/npz_load_time:.2f}x")
        
        # 测试 NumPy npy 加载
        logger.info("测试 NumPy npy 加载...")
        start_time = time.time()
        npy_loaded = {
            'array1': np.load('test_large_array1.npy'),
            'array2': np.load('test_large_array2.npy')
        }
        npy_load_time = time.time() - start_time
        logger.info(f"NumPy npy 加载耗时: {npy_load_time:.2f}秒")
        logger.info(f"加载性能对比 (npy): NumPack/NumPy = {load_time/npy_load_time:.2f}x")
        
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
        
        # 测试 NumPack 内存映射加载
        logger.info("测试 NumPack 内存映射加载...")
        start_time = time.time()
        mmap_loaded = load_nnp('test_large.nnp', mmap_mode=True)
        mmap_time = time.time() - start_time
        logger.info(f"NumPack 内存映射加载耗时: {mmap_time:.2f}秒")
        
        # 测试 NumPy npz 内存映射加载
        logger.info("测试 NumPy npz 内存映射加载...")
        start_time = time.time()
        npz_mmap = np.load('test_large.npz', mmap_mode='r')
        _, _ = npz_mmap['array1'], npz_mmap['array2']
        npz_mmap_time = time.time() - start_time
        logger.info(f"NumPy npz 内存映射加载耗时: {npz_mmap_time:.2f}秒")
        logger.info(f"内存映射加载性能对比 (npz): NumPack/NumPy = {mmap_time/npz_mmap_time:.2f}x")
        
        # 测试 NumPy npy 内存映射加载
        logger.info("测试 NumPy npy 内存映射加载...")
        start_time = time.time()
        npy_mmap = {
            'array1': np.load('test_large_array1.npy', mmap_mode='r'),
            'array2': np.load('test_large_array2.npy', mmap_mode='r')
        }
        npy_mmap_time = time.time() - start_time
        logger.info(f"NumPy npy 内存映射加载耗时: {npy_mmap_time:.2f}秒")
        logger.info(f"内存映射加载性能对比 (npy): NumPack/NumPy = {mmap_time/npy_mmap_time:.2f}x")
        
        # 测试 NumPack 替换操作
        logger.info("测试 NumPack 大数据替换...")
        replace_data = {
            'array1': np.random.rand(size, 10).astype(np.float32)
        }
        start_time = time.time()
        replace_arrays('test_large.nnp', replace_data, slice(None), 'array1')
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
        
        # 测试 NumPack 删除操作
        logger.info("测试 NumPack 大数据删除...")
        start_time = time.time()
        drop_arrays('test_large.nnp', slice(size//2, None), 'array1')
        drop_time = time.time() - start_time
        logger.info(f"NumPack 删除操作耗时: {drop_time:.2f}秒")
        
        # NumPy npz 不支持原地删除，需要重新保存整个文件
        logger.info("测试 NumPy npz 大数据删除...")
        npz_data = dict(np.load('test_large.npz'))
        npz_data['array1'] = npz_data['array1'][:size//2]
        start_time = time.time()
        np.savez('test_large.npz', **npz_data)
        npz_drop_time = time.time() - start_time
        logger.info(f"NumPy npz 删除操作耗时: {npz_drop_time:.2f}秒")
        logger.info(f"删除性能对比 (npz): NumPack/NumPy = {drop_time/npz_drop_time:.2f}x")
        
        # NumPy npy 删除部分数据并重新保存
        logger.info("测试 NumPy npy 大数据删除...")
        start_time = time.time()
        array1 = np.load('test_large_array1.npy')
        array1 = array1[:size//2]
        np.save('test_large_array1.npy', array1)
        npy_drop_time = time.time() - start_time
        logger.info(f"NumPy npy 删除操作耗时: {npy_drop_time:.2f}秒")
        logger.info(f"删除性能对比 (npy): NumPack/NumPy = {drop_time/npy_drop_time:.2f}x")
        
        # 比较文件大小
        nnp_size = os.path.getsize('test_large.nnp') / (1024 * 1024)  # MB
        npz_size = os.path.getsize('test_large.npz') / (1024 * 1024)  # MB
        npy_size = sum(os.path.getsize(f'test_large_{name}.npy') / (1024 * 1024) 
                      for name in ['array1', 'array2'])  # MB
        logger.info(f"\n文件大小对比:")
        logger.info(f"NumPack: {nnp_size:.2f} MB")
        logger.info(f"NumPy npz: {npz_size:.2f} MB")
        logger.info(f"NumPy npy: {npy_size:.2f} MB")
        logger.info(f"大小比例 (vs npz): NumPack/NumPy = {nnp_size/npz_size:.2f}x")
        logger.info(f"大小比例 (vs npy): NumPack/NumPy = {nnp_size/npy_size:.2f}x")
        
        logger.info("大数据测试完成")
        
    except Exception as e:
        logger.error(f"大数据测试失败: {str(e)}")
        raise

@clean_file_when_finished('test_append.nnp', 'test_append.npz')
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
        
        # 保存初始数据
        save_nnp('test_append.nnp', arrays)
        np.savez('test_append.npz', **arrays)
        
        # 创建要追加的数据
        append_data = {
            'array3': np.random.rand(size // 4, 8).astype(np.float32),
            'array4': np.random.rand(size // 8, 3).astype(np.float32)
        }
        
        # 测试 NumPack 追加
        logger.info("测试 NumPack 追加数组...")
        start_time = time.time()
        append_arrays('test_append.nnp', append_data)
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
        loaded = load_nnp('test_append.nnp')
        
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

@clean_file_when_finished('test_random_access.nnp', 'test_random_access.npz', 'test_random_access_array1.npy', 'test_random_access_array2.npy')
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
        
        # 保存数据
        save_nnp('test_random_access.nnp', arrays)
        np.savez('test_random_access.npz', **arrays)
        np.save('test_random_access_array1.npy', arrays['array1'])
        np.save('test_random_access_array2.npy', arrays['array2'])
        
        # 生成随机索引
        random_indices = np.random.randint(0, size, 10000)  # 随机访问1万行
        sequential_indices = np.arange(10000)  # 顺序访问1万行
        slice_indices = slice(size//4, size//2)  # 切片访问25万行
        
        # 测试随机索引访问
        logger.info("测试随机索引访问性能...")
        
        # NumPack 随机访问
        start_time = time.time()
        numpack_random = getitem('test_random_access.nnp', random_indices)
        numpack_random_time = time.time() - start_time
        logger.info(f"NumPack 随机访问耗时: {numpack_random_time:.2f}秒")
        
        # NumPy npz 随机访问
        start_time = time.time()
        npz_data = np.load('test_random_access.npz')
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
            'array1': np.load('test_random_access_array1.npy')[random_indices],
            'array2': np.load('test_random_access_array2.npy')[random_indices]
        }
        npy_random_time = time.time() - start_time
        logger.info(f"NumPy npy 随机访问耗时: {npy_random_time:.2f}秒")
        logger.info(f"随机访问性能对比 (npy): NumPack/NumPy = {numpack_random_time/npy_random_time:.2f}x")
        
        # 测试顺序索引访问
        logger.info("\n测试顺序索引访问性能...")
        
        # NumPack 顺序访问
        start_time = time.time()
        numpack_seq = getitem('test_random_access.nnp', sequential_indices)
        numpack_seq_time = time.time() - start_time
        logger.info(f"NumPack 顺序访问耗时: {numpack_seq_time:.2f}秒")
        
        # NumPy npz 顺序访问
        start_time = time.time()
        npz_seq = {
            'array1': npz_data['array1'][sequential_indices],
            'array2': npz_data['array2'][sequential_indices]
        }
        npz_seq_time = time.time() - start_time
        logger.info(f"NumPy npz 顺序访问耗时: {npz_seq_time:.2f}秒")
        logger.info(f"顺序访问性能对比 (npz): NumPack/NumPy = {numpack_seq_time/npz_seq_time:.2f}x")
        
        # NumPy npy 顺序访问
        start_time = time.time()
        npy_seq = {
            'array1': np.load('test_random_access_array1.npy')[sequential_indices],
            'array2': np.load('test_random_access_array2.npy')[sequential_indices]
        }
        npy_seq_time = time.time() - start_time
        logger.info(f"NumPy npy 顺序访问耗时: {npy_seq_time:.2f}秒")
        logger.info(f"顺序访问性能对比 (npy): NumPack/NumPy = {numpack_seq_time/npy_seq_time:.2f}x")
        
        # 测试切片访问
        logger.info("\n测试切片访问性能...")
        
        # NumPack 切片访问
        start_time = time.time()
        numpack_slice = getitem('test_random_access.nnp', slice_indices)
        numpack_slice_time = time.time() - start_time
        logger.info(f"NumPack 切片访问耗时: {numpack_slice_time:.2f}秒")
        
        # NumPy npz 切片访问
        start_time = time.time()
        npz_slice = {
            'array1': npz_data['array1'][slice_indices],
            'array2': npz_data['array2'][slice_indices]
        }
        npz_slice_time = time.time() - start_time
        logger.info(f"NumPy npz 切片访问耗时: {npz_slice_time:.2f}秒")
        logger.info(f"切片访问性能对比 (npz): NumPack/NumPy = {numpack_slice_time/npz_slice_time:.2f}x")
        
        # NumPy npy 切片访问
        start_time = time.time()
        npy_slice = {
            'array1': np.load('test_random_access_array1.npy')[slice_indices],
            'array2': np.load('test_random_access_array2.npy')[slice_indices]
        }
        npy_slice_time = time.time() - start_time
        logger.info(f"NumPy npy 切片访问耗时: {npy_slice_time:.2f}秒")
        logger.info(f"切片访问性能对比 (npy): NumPack/NumPy = {numpack_slice_time/npy_slice_time:.2f}x")
        
        # 验证数据正确性
        for name in arrays:
            # 验证随机访问结果
            assert np.allclose(numpack_random[name], npz_random[name])
            assert np.allclose(numpack_random[name], npy_random[name])
            logger.info(f"随机访问数组 '{name}' 验证通过")
            
            # 验证顺序访问结果
            assert np.allclose(numpack_seq[name], npz_seq[name])
            assert np.allclose(numpack_seq[name], npy_seq[name])
            logger.info(f"顺序访问数组 '{name}' 验证通过")
            
            # 验证切片访问结果
            assert np.allclose(numpack_slice[name], npz_slice[name])
            assert np.allclose(numpack_slice[name], npy_slice[name])
            logger.info(f"切片访问数组 '{name}' 验证通过")
        
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