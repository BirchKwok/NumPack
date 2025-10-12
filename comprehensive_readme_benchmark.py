#!/usr/bin/env python3
"""
完整的 README 基准测试脚本 - 测试所有性能指标
包含：Storage、Modification、Drop、Append、Random Access、Matrix Computation
"""

import os
import sys
import time
import tempfile
import shutil
import gc
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

def format_time(seconds):
    """格式化时间"""
    if seconds < 0.0001:
        return "0.000"
    return f"{seconds:.3f}"

def get_file_size_mb(filepath):
    """获取文件大小（MB）"""
    if os.path.isdir(filepath):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(filepath):
            for filename in filenames:
                total_size += os.path.getsize(os.path.join(dirpath, filename))
        return total_size / 1024 / 1024
    else:
        return os.path.getsize(filepath) / 1024 / 1024

def test_storage_operations(temp_dir):
    """测试存储操作（1,000,000 × 128 float32）"""
    print("\n=== Storage Operations (1M × 128) ===")
    from numpack import NumPack
    
    rows, cols = 1_000_000, 128
    array = np.random.rand(rows, cols).astype(np.float32)
    data_size_mb = array.nbytes / 1024 / 1024
    
    results = {}
    
    # NumPack
    numpack_dir = os.path.join(temp_dir, 'storage_numpack')
    npk = NumPack(numpack_dir, drop_if_exists=True)
    npk.open()
    
    gc.collect()
    start = time.perf_counter()
    npk.save({'array': array})
    results['numpack_write'] = time.perf_counter() - start
    
    gc.collect()
    start = time.perf_counter()
    loaded = npk.load('array')
    results['numpack_read'] = time.perf_counter() - start
    
    gc.collect()
    start = time.perf_counter()
    lazy = npk.load('array', lazy=True)
    results['numpack_lazy'] = time.perf_counter() - start
    
    npk.close()
    results['numpack_size'] = get_file_size_mb(numpack_dir)
    
    # NumPy
    npy_file = os.path.join(temp_dir, 'test.npy')
    gc.collect()
    start = time.perf_counter()
    np.save(npy_file, array)
    results['npy_write'] = time.perf_counter() - start
    
    gc.collect()
    start = time.perf_counter()
    loaded = np.load(npy_file)
    results['npy_read'] = time.perf_counter() - start
    
    results['npy_size'] = get_file_size_mb(npy_file)
    
    # NPZ
    npz_file = os.path.join(temp_dir, 'test.npz')
    gc.collect()
    start = time.perf_counter()
    np.savez(npz_file, array=array)
    results['npz_write'] = time.perf_counter() - start
    
    gc.collect()
    start = time.perf_counter()
    with np.load(npz_file) as data:
        loaded = data['array'][:]
    results['npz_read'] = time.perf_counter() - start
    
    results['npz_size'] = get_file_size_mb(npz_file)
    results['data_size_mb'] = data_size_mb
    
    return results

def test_replace_operations(temp_dir):
    """测试替换操作"""
    print("\n=== Replace Operations ===")
    from numpack import NumPack
    
    rows, cols = 1_000_000, 128
    array = np.random.rand(rows, cols).astype(np.float32)
    
    results = {}
    
    # NumPack - 每个测试使用独立目录
    
    # Single row replace
    numpack_dir_1 = os.path.join(temp_dir, 'replace_numpack_1')
    npk = NumPack(numpack_dir_1, drop_if_exists=True)
    npk.open()
    npk.save({'array': array})
    single_row = np.random.rand(1, cols).astype(np.float32)
    gc.collect()
    start = time.perf_counter()
    npk.replace({'array': single_row}, [rows // 2])
    results['numpack_single_row'] = time.perf_counter() - start
    npk.close()
    
    # Continuous rows (10K)
    numpack_dir_2 = os.path.join(temp_dir, 'replace_numpack_2')
    npk = NumPack(numpack_dir_2, drop_if_exists=True)
    npk.open()
    npk.save({'array': array})
    continuous_rows = np.random.rand(10000, cols).astype(np.float32)
    gc.collect()
    start = time.perf_counter()
    npk.replace({'array': continuous_rows}, slice(0, 10000))
    results['numpack_continuous_10k'] = time.perf_counter() - start
    npk.close()
    
    # Random rows (10K)
    numpack_dir_3 = os.path.join(temp_dir, 'replace_numpack_3')
    npk = NumPack(numpack_dir_3, drop_if_exists=True)
    npk.open()
    npk.save({'array': array})
    random_indices = np.random.choice(rows, 10000, replace=False).tolist()
    random_rows = np.random.rand(10000, cols).astype(np.float32)
    gc.collect()
    start = time.perf_counter()
    npk.replace({'array': random_rows}, random_indices)
    results['numpack_random_10k'] = time.perf_counter() - start
    npk.close()
    
    # Large data (500K)
    numpack_dir_4 = os.path.join(temp_dir, 'replace_numpack_4')
    npk = NumPack(numpack_dir_4, drop_if_exists=True)
    npk.open()
    npk.save({'array': array})
    large_data = np.random.rand(500000, cols).astype(np.float32)
    gc.collect()
    start = time.perf_counter()
    npk.replace({'array': large_data}, slice(0, 500000))
    results['numpack_large_500k'] = time.perf_counter() - start
    npk.close()
    
    # NumPy NPZ (只测单行，其他操作太慢)
    npz_file = os.path.join(temp_dir, 'replace_test.npz')
    np.savez(npz_file, array=array)
    
    gc.collect()
    start = time.perf_counter()
    npz_data = dict(np.load(npz_file))
    npz_data['array'][rows // 2] = single_row
    np.savez(npz_file, **npz_data)
    results['npz_single_row'] = time.perf_counter() - start
    
    # NumPy NPY
    npy_file = os.path.join(temp_dir, 'replace_test.npy')
    np.save(npy_file, array)
    
    gc.collect()
    start = time.perf_counter()
    npy_data = np.load(npy_file)
    npy_data[rows // 2] = single_row
    np.save(npy_file, npy_data)
    results['npy_single_row'] = time.perf_counter() - start
    
    return results

def test_drop_operations(temp_dir):
    """测试删除操作"""
    print("\n=== Drop Operations ===")
    from numpack import NumPack
    
    rows, cols = 1_000_000, 128
    array = np.random.rand(rows, cols).astype(np.float32)
    
    results = {}
    
    # NumPack tests - 每个测试使用独立的目录
    
    # Drop array
    numpack_dir_1 = os.path.join(temp_dir, 'drop_numpack_1')
    npk = NumPack(numpack_dir_1, drop_if_exists=True)
    npk.open()
    npk.save({'array': array})
    gc.collect()
    start = time.perf_counter()
    npk.drop('array')
    results['numpack_drop_array'] = time.perf_counter() - start
    npk.close()
    
    # Drop first row
    numpack_dir_2 = os.path.join(temp_dir, 'drop_numpack_2')
    npk = NumPack(numpack_dir_2, drop_if_exists=True)
    npk.open()
    npk.save({'array': array})
    gc.collect()
    start = time.perf_counter()
    npk.drop('array', [0])
    results['numpack_drop_first'] = time.perf_counter() - start
    npk.close()
    
    # Drop last row
    numpack_dir_3 = os.path.join(temp_dir, 'drop_numpack_3')
    npk = NumPack(numpack_dir_3, drop_if_exists=True)
    npk.open()
    npk.save({'array': array})
    gc.collect()
    start = time.perf_counter()
    npk.drop('array', [rows - 1])
    results['numpack_drop_last'] = time.perf_counter() - start
    npk.close()
    
    # Drop middle row
    numpack_dir_4 = os.path.join(temp_dir, 'drop_numpack_4')
    npk = NumPack(numpack_dir_4, drop_if_exists=True)
    npk.open()
    npk.save({'array': array})
    gc.collect()
    start = time.perf_counter()
    npk.drop('array', [rows // 2])
    results['numpack_drop_middle'] = time.perf_counter() - start
    npk.close()
    
    # Drop front continuous (10K)
    numpack_dir_5 = os.path.join(temp_dir, 'drop_numpack_5')
    npk = NumPack(numpack_dir_5, drop_if_exists=True)
    npk.open()
    npk.save({'array': array})
    gc.collect()
    start = time.perf_counter()
    npk.drop('array', list(range(10000)))
    results['numpack_drop_front_10k'] = time.perf_counter() - start
    npk.close()
    
    # Drop middle continuous (10K)
    numpack_dir_6 = os.path.join(temp_dir, 'drop_numpack_6')
    npk = NumPack(numpack_dir_6, drop_if_exists=True)
    npk.open()
    npk.save({'array': array})
    middle_start = rows // 2 - 5000
    gc.collect()
    start = time.perf_counter()
    npk.drop('array', list(range(middle_start, middle_start + 10000)))
    results['numpack_drop_middle_10k'] = time.perf_counter() - start
    npk.close()
    
    # Drop end continuous (10K)
    numpack_dir_7 = os.path.join(temp_dir, 'drop_numpack_7')
    npk = NumPack(numpack_dir_7, drop_if_exists=True)
    npk.open()
    npk.save({'array': array})
    gc.collect()
    start = time.perf_counter()
    npk.drop('array', list(range(rows - 10000, rows)))
    results['numpack_drop_end_10k'] = time.perf_counter() - start
    npk.close()
    
    # Drop random rows (10K)
    numpack_dir_8 = os.path.join(temp_dir, 'drop_numpack_8')
    npk = NumPack(numpack_dir_8, drop_if_exists=True)
    npk.open()
    npk.save({'array': array})
    random_indices = np.random.choice(rows, 10000, replace=False).tolist()
    gc.collect()
    start = time.perf_counter()
    npk.drop('array', random_indices)
    results['numpack_drop_random_10k'] = time.perf_counter() - start
    npk.close()
    
    # Drop near non-continuous (10K)
    numpack_dir_9 = os.path.join(temp_dir, 'drop_numpack_9')
    npk = NumPack(numpack_dir_9, drop_if_exists=True)
    npk.open()
    npk.save({'array': array})
    near_indices = list(range(0, 20000, 2))
    gc.collect()
    start = time.perf_counter()
    npk.drop('array', near_indices)
    results['numpack_drop_near_10k'] = time.perf_counter() - start
    npk.close()
    
    # NumPy comparisons
    npz_file = os.path.join(temp_dir, 'drop_test.npz')
    npy_file = os.path.join(temp_dir, 'drop_test.npy')
    
    # NPZ drop array
    np.savez(npz_file, array=array)
    gc.collect()
    start = time.perf_counter()
    npz_data = dict(np.load(npz_file))
    del npz_data['array']
    np.savez(npz_file, **npz_data)
    results['npz_drop_array'] = time.perf_counter() - start
    
    # NPY drop array
    np.save(npy_file, array)
    gc.collect()
    start = time.perf_counter()
    os.remove(npy_file)
    results['npy_drop_array'] = time.perf_counter() - start
    
    # NPZ/NPY drop rows (use mask)
    np.savez(npz_file, array=array)
    mask = np.ones(rows, dtype=bool)
    mask[random_indices] = False
    
    gc.collect()
    start = time.perf_counter()
    npz_data = dict(np.load(npz_file))
    npz_data['array'] = npz_data['array'][mask]
    np.savez(npz_file, **npz_data)
    results['npz_drop_rows'] = time.perf_counter() - start
    
    np.save(npy_file, array)
    gc.collect()
    start = time.perf_counter()
    npy_data = np.load(npy_file)
    npy_data = npy_data[mask]
    np.save(npy_file, npy_data)
    results['npy_drop_rows'] = time.perf_counter() - start
    
    return results

def test_append_operations(temp_dir):
    """测试追加操作"""
    print("\n=== Append Operations ===")
    from numpack import NumPack
    
    rows, cols = 1_000_000, 128
    array = np.random.rand(rows, cols).astype(np.float32)
    small_append = np.random.rand(1000, cols).astype(np.float32)
    large_append = np.random.rand(500000, cols).astype(np.float32)
    
    results = {}
    
    # NumPack - 每个测试使用独立目录
    
    # Small append
    numpack_dir_1 = os.path.join(temp_dir, 'append_numpack_1')
    npk = NumPack(numpack_dir_1, drop_if_exists=True)
    npk.open()
    npk.save({'array': array})
    gc.collect()
    start = time.perf_counter()
    npk.append({'array': small_append})
    results['numpack_small_append'] = time.perf_counter() - start
    npk.close()
    
    # Large append
    numpack_dir_2 = os.path.join(temp_dir, 'append_numpack_2')
    npk = NumPack(numpack_dir_2, drop_if_exists=True)
    npk.open()
    npk.save({'array': array})
    gc.collect()
    start = time.perf_counter()
    npk.append({'array': large_append})
    results['numpack_large_append'] = time.perf_counter() - start
    npk.close()
    
    # NumPy NPZ
    npz_file = os.path.join(temp_dir, 'append_test.npz')
    
    np.savez(npz_file, array=array)
    gc.collect()
    start = time.perf_counter()
    npz_data = dict(np.load(npz_file))
    npz_data['array'] = np.vstack([npz_data['array'], small_append])
    np.savez(npz_file, **npz_data)
    results['npz_small_append'] = time.perf_counter() - start
    
    np.savez(npz_file, array=array)
    gc.collect()
    start = time.perf_counter()
    npz_data = dict(np.load(npz_file))
    npz_data['array'] = np.vstack([npz_data['array'], large_append])
    np.savez(npz_file, **npz_data)
    results['npz_large_append'] = time.perf_counter() - start
    
    # NumPy NPY
    npy_file = os.path.join(temp_dir, 'append_test.npy')
    
    np.save(npy_file, array)
    gc.collect()
    start = time.perf_counter()
    npy_data = np.load(npy_file)
    npy_data = np.vstack([npy_data, small_append])
    np.save(npy_file, npy_data)
    results['npy_small_append'] = time.perf_counter() - start
    
    np.save(npy_file, array)
    gc.collect()
    start = time.perf_counter()
    npy_data = np.load(npy_file)
    npy_data = np.vstack([npy_data, large_append])
    np.save(npy_file, npy_data)
    results['npy_large_append'] = time.perf_counter() - start
    
    return results

def test_random_access(temp_dir):
    """测试随机访问（10K indices）"""
    print("\n=== Random Access (10K indices) ===")
    from numpack import NumPack
    
    rows, cols = 1_000_000, 128
    array = np.random.rand(rows, cols).astype(np.float32)
    indices = np.random.randint(0, rows, 10000)
    
    results = {}
    
    # NumPack
    numpack_dir = os.path.join(temp_dir, 'random_numpack')
    npk = NumPack(numpack_dir, drop_if_exists=True)
    npk.open()
    npk.save({'array': array})
    
    gc.collect()
    start = time.perf_counter()
    data = npk.getitem('array', indices.tolist())
    results['numpack_random_access'] = time.perf_counter() - start
    
    npk.close()
    
    # NumPy NPZ (mmap)
    npz_file = os.path.join(temp_dir, 'random_test.npz')
    np.savez(npz_file, array=array)
    
    gc.collect()
    start = time.perf_counter()
    with np.load(npz_file, mmap_mode='r') as data:
        result = data['array'][indices]
    results['npz_random_access'] = time.perf_counter() - start
    
    # NumPy NPY (mmap)
    npy_file = os.path.join(temp_dir, 'random_test.npy')
    np.save(npy_file, array)
    
    gc.collect()
    start = time.perf_counter()
    mmap_data = np.load(npy_file, mmap_mode='r')
    result = mmap_data[indices]
    results['npy_random_access'] = time.perf_counter() - start
    
    return results

def test_matrix_computation(temp_dir):
    """测试矩阵运算"""
    print("\n=== Matrix Computation (Inner Product) ===")
    from numpack import NumPack
    
    rows, cols = 1_000_000, 128
    array = np.random.rand(rows, cols).astype(np.float32)
    query = np.random.rand(1, cols).astype(np.float32)
    
    results = {}
    
    # NumPack lazy mode
    numpack_dir = os.path.join(temp_dir, 'matrix_numpack')
    npk = NumPack(numpack_dir, drop_if_exists=True)
    npk.open()
    npk.save({'array': array})
    
    lazy_array = npk.load('array', lazy=True)
    gc.collect()
    start = time.perf_counter()
    result = np.inner(query, lazy_array)
    results['numpack_inner_product'] = time.perf_counter() - start
    
    npk.close()
    
    # NumPy NPZ mmap
    npz_file = os.path.join(temp_dir, 'matrix_test.npz')
    np.savez(npz_file, array=array)
    
    gc.collect()
    start = time.perf_counter()
    with np.load(npz_file, mmap_mode='r') as data:
        result = np.inner(query, data['array'])
    results['npz_inner_product'] = time.perf_counter() - start
    
    # NumPy NPY mmap
    npy_file = os.path.join(temp_dir, 'matrix_test.npy')
    np.save(npy_file, array)
    
    gc.collect()
    start = time.perf_counter()
    mmap_data = np.load(npy_file, mmap_mode='r')
    result = np.inner(query, mmap_data)
    results['npy_inner_product'] = time.perf_counter() - start
    
    # In-memory
    gc.collect()
    start = time.perf_counter()
    result = np.inner(query, array)
    results['memory_inner_product'] = time.perf_counter() - start
    
    return results

def print_results(all_results):
    """打印所有结果"""
    print("\n" + "="*80)
    print("COMPREHENSIVE BENCHMARK RESULTS")
    print("="*80)
    
    # Storage Operations
    r = all_results['storage']
    print("\n### Storage Operations (1,000,000 × 128 float32)")
    print(f"Data Size: {r['data_size_mb']:.2f} MB\n")
    print("| Format    | Write (s) | Read (s) | Write MB/s | Read MB/s | Size (MB) |")
    print("|-----------|-----------|----------|------------|-----------|-----------|")
    
    for name, write_key, read_key, size_key in [
        ('NumPack', 'numpack_write', 'numpack_read', 'numpack_size'),
        ('NumPack Lazy', None, 'numpack_lazy', None),
        ('NumPy .npy', 'npy_write', 'npy_read', 'npy_size'),
        ('NumPy .npz', 'npz_write', 'npz_read', 'npz_size'),
    ]:
        write_time = format_time(r[write_key]) if write_key and write_key in r else '-'
        read_time = format_time(r[read_key]) if read_key in r else '-'
        write_mbps = f"{int(r['data_size_mb'] / r[write_key]):,}" if write_key and write_key in r else '-'
        read_mbps = f"{int(r['data_size_mb'] / r[read_key]):,}" if read_key in r and r[read_key] > 0.001 else '-'
        size = f"{r[size_key]:.2f}" if size_key and size_key in r else '-'
        print(f"| {name:<9} | {write_time:>9} | {read_time:>8} | {write_mbps:>10} | {read_mbps:>9} | {size:>9} |")
    
    # Replace Operations
    r = all_results['replace']
    print("\n### Data Modification Operations")
    print("| Operation | NumPack (Rust) | NumPy NPZ | NumPy NPY |")
    print("|-----------|----------------|-----------|-----------|")
    print(f"| Single Row Replace | {format_time(r['numpack_single_row'])}s | {format_time(r['npz_single_row'])}s | {format_time(r['npy_single_row'])}s |")
    print(f"| Continuous Rows (10K) | {format_time(r['numpack_continuous_10k'])}s | - | - |")
    print(f"| Random Rows (10K) | {format_time(r['numpack_random_10k'])}s | - | - |")
    print(f"| Large Data Replace (500K) | {format_time(r['numpack_large_500k'])}s | - | - |")
    
    # Drop Operations
    r = all_results['drop']
    print("\n### Drop Operations")
    print("| Operation | NumPack (Rust) | NumPy NPZ | NumPy NPY |")
    print("|-----------|----------------|-----------|-----------|")
    print(f"| Drop Array | {format_time(r['numpack_drop_array'])}s | {format_time(r['npz_drop_array'])}s | {format_time(r['npy_drop_array'])}s |")
    print(f"| Drop First Row | {format_time(r['numpack_drop_first'])}s | {format_time(r['npz_drop_rows'])}s | {format_time(r['npy_drop_rows'])}s |")
    print(f"| Drop Last Row | {format_time(r['numpack_drop_last'])}s | {format_time(r['npz_drop_rows'])}s | {format_time(r['npy_drop_rows'])}s |")
    print(f"| Drop Middle Row | {format_time(r['numpack_drop_middle'])}s | {format_time(r['npz_drop_rows'])}s | {format_time(r['npy_drop_rows'])}s |")
    print(f"| Drop Front Continuous (10K) | {format_time(r['numpack_drop_front_10k'])}s | {format_time(r['npz_drop_rows'])}s | {format_time(r['npy_drop_rows'])}s |")
    print(f"| Drop Middle Continuous (10K) | {format_time(r['numpack_drop_middle_10k'])}s | {format_time(r['npz_drop_rows'])}s | {format_time(r['npy_drop_rows'])}s |")
    print(f"| Drop End Continuous (10K) | {format_time(r['numpack_drop_end_10k'])}s | {format_time(r['npz_drop_rows'])}s | {format_time(r['npy_drop_rows'])}s |")
    print(f"| Drop Random Rows (10K) | {format_time(r['numpack_drop_random_10k'])}s | {format_time(r['npz_drop_rows'])}s | {format_time(r['npy_drop_rows'])}s |")
    print(f"| Drop Near Non-continuous (10K) | {format_time(r['numpack_drop_near_10k'])}s | {format_time(r['npz_drop_rows'])}s | {format_time(r['npy_drop_rows'])}s |")
    
    # Append Operations
    r = all_results['append']
    print("\n### Append Operations")
    print("| Operation | NumPack (Rust) | NumPy NPZ | NumPy NPY |")
    print("|-----------|----------------|-----------|-----------|")
    print(f"| Small Append (1K rows) | {format_time(r['numpack_small_append'])}s | {format_time(r['npz_small_append'])}s | {format_time(r['npy_small_append'])}s |")
    print(f"| Large Append (500K rows) | {format_time(r['numpack_large_append'])}s | {format_time(r['npz_large_append'])}s | {format_time(r['npy_large_append'])}s |")
    
    # Random Access
    r = all_results['random_access']
    print("\n### Random Access Performance (10K indices)")
    print("| Operation | NumPack (Rust) | NumPy NPZ | NumPy NPY |")
    print("|-----------|----------------|-----------|-----------|")
    print(f"| Random Access | {format_time(r['numpack_random_access'])}s | {format_time(r['npz_random_access'])}s | {format_time(r['npy_random_access'])}s |")
    
    # Matrix Computation
    r = all_results['matrix']
    print("\n### Matrix Computation Performance (Inner Product)")
    print("| Operation | NumPack (Rust) | NumPy NPZ | NumPy NPY | In-Memory |")
    print("|-----------|----------------|-----------|-----------|-----------|")
    print(f"| Inner Product | {format_time(r['numpack_inner_product'])}s | {format_time(r['npz_inner_product'])}s | {format_time(r['npy_inner_product'])}s | {format_time(r['memory_inner_product'])}s |")
    
    print("\n" + "="*80)

def main():
    """主函数"""
    print("="*80)
    print("Comprehensive README Benchmark")
    print("Testing ALL performance metrics from README")
    print("="*80)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        all_results = {
            'storage': test_storage_operations(temp_dir),
            'replace': test_replace_operations(temp_dir),
            'drop': test_drop_operations(temp_dir),
            'append': test_append_operations(temp_dir),
            'random_access': test_random_access(temp_dir),
            'matrix': test_matrix_computation(temp_dir),
        }
        
        print_results(all_results)
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == '__main__':
    main()

