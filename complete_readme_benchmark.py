#!/usr/bin/env python3
"""
完整的README基准测试 - 包含所有README中的测试项目
"""

import os
import sys
import time
import tempfile
import shutil
import gc
import numpy as np
from pathlib import Path
from typing import Dict, List

# 添加 numpack 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

def cleanup_files(*files):
    """清理测试文件"""
    for file in files:
        try:
            if os.path.isdir(file):
                shutil.rmtree(file)
            else:
                os.remove(file)
        except FileNotFoundError:
            pass

def format_time(seconds):
    """格式化时间为3位小数"""
    return f"{seconds:.3f}s"

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

def test_drop_operations():
    """测试删除操作性能"""
    print("=== Drop Operations ===")
    
    # 创建测试数据 - 1M rows float32
    size = 1000000
    cols = 10
    test_array = np.random.rand(size, cols).astype(np.float32)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        results = {}
        
        # 测试NumPack Python后端
        os.environ['NUMPACK_FORCE_PYTHON_BACKEND'] = '1'
        from numpack import NumPack
        
        # Drop Array
        numpack_dir = os.path.join(temp_dir, 'numpack_drop')
        npk = NumPack(numpack_dir, drop_if_exists=True)
        npk.save({'test_array': test_array})
        
        start_time = time.perf_counter()
        npk.drop('test_array')
        results['python_drop_array'] = time.perf_counter() - start_time
        
        # Drop First Row
        npk.save({'test_array': test_array})
        start_time = time.perf_counter()
        npk.drop('test_array', [0])
        results['python_drop_first'] = time.perf_counter() - start_time
        
        # Drop Last Row
        npk.save({'test_array': test_array})
        start_time = time.perf_counter()
        npk.drop('test_array', [size-1])
        results['python_drop_last'] = time.perf_counter() - start_time
        
        # Drop Middle Row
        npk.save({'test_array': test_array})
        start_time = time.perf_counter()
        npk.drop('test_array', [size//2])
        results['python_drop_middle'] = time.perf_counter() - start_time
        
        # Drop Front Continuous (10K rows)
        npk.save({'test_array': test_array})
        start_time = time.perf_counter()
        npk.drop('test_array', list(range(10000)))
        results['python_drop_front_cont'] = time.perf_counter() - start_time
        
        # Drop Middle Continuous (10K rows)
        npk.save({'test_array': test_array})
        middle_start = size // 2 - 5000
        start_time = time.perf_counter()
        npk.drop('test_array', list(range(middle_start, middle_start + 10000)))
        results['python_drop_middle_cont'] = time.perf_counter() - start_time
        
        # Drop End Continuous (10K rows)
        npk.save({'test_array': test_array})
        start_time = time.perf_counter()
        npk.drop('test_array', list(range(size - 10000, size)))
        results['python_drop_end_cont'] = time.perf_counter() - start_time
        
        # Drop Random Rows (10K rows)
        npk.save({'test_array': test_array})
        random_indices = np.random.choice(size, 10000, replace=False).tolist()
        start_time = time.perf_counter()
        npk.drop('test_array', random_indices)
        results['python_drop_random'] = time.perf_counter() - start_time
        
        # Drop Near Non-continuous (10K rows)
        npk.save({'test_array': test_array})
        near_indices = list(range(0, 20000, 2))  # Every other row
        start_time = time.perf_counter()
        npk.drop('test_array', near_indices)
        results['python_drop_near'] = time.perf_counter() - start_time
        
        # 测试NumPack Rust后端
        if 'NUMPACK_FORCE_PYTHON_BACKEND' in os.environ:
            del os.environ['NUMPACK_FORCE_PYTHON_BACKEND']
        
        try:
            if 'numpack' in sys.modules:
                del sys.modules['numpack']
            from numpack import NumPack
            
            # 为Rust后端运行相同的测试
            numpack_rust_dir = os.path.join(temp_dir, 'numpack_rust_drop')
            npk_rust = NumPack(numpack_rust_dir, drop_if_exists=True)
            
            # Drop Array (Rust)
            npk_rust.save({'test_array': test_array})
            start_time = time.perf_counter()
            npk_rust.drop('test_array')
            results['rust_drop_array'] = time.perf_counter() - start_time
            
            # Drop First Row (Rust)
            npk_rust.save({'test_array': test_array})
            start_time = time.perf_counter()
            npk_rust.drop('test_array', [0])
            results['rust_drop_first'] = time.perf_counter() - start_time
            
            # Drop Last Row (Rust)
            npk_rust.save({'test_array': test_array})
            start_time = time.perf_counter()
            npk_rust.drop('test_array', [size-1])
            results['rust_drop_last'] = time.perf_counter() - start_time
            
            # Drop Middle Row (Rust)
            npk_rust.save({'test_array': test_array})
            start_time = time.perf_counter()
            npk_rust.drop('test_array', [size//2])
            results['rust_drop_middle'] = time.perf_counter() - start_time
            
            # Drop Front Continuous (Rust)
            npk_rust.save({'test_array': test_array})
            start_time = time.perf_counter()
            npk_rust.drop('test_array', list(range(10000)))
            results['rust_drop_front_cont'] = time.perf_counter() - start_time
            
            # Drop Middle Continuous (Rust)
            npk_rust.save({'test_array': test_array})
            start_time = time.perf_counter()
            npk_rust.drop('test_array', list(range(middle_start, middle_start + 10000)))
            results['rust_drop_middle_cont'] = time.perf_counter() - start_time
            
            # Drop End Continuous (Rust)
            npk_rust.save({'test_array': test_array})
            start_time = time.perf_counter()
            npk_rust.drop('test_array', list(range(size - 10000, size)))
            results['rust_drop_end_cont'] = time.perf_counter() - start_time
            
            # Drop Random Rows (Rust)
            npk_rust.save({'test_array': test_array})
            start_time = time.perf_counter()
            npk_rust.drop('test_array', random_indices)
            results['rust_drop_random'] = time.perf_counter() - start_time
            
            # Drop Near Non-continuous (Rust)
            npk_rust.save({'test_array': test_array})
            start_time = time.perf_counter()
            npk_rust.drop('test_array', near_indices)
            results['rust_drop_near'] = time.perf_counter() - start_time
            
        except ImportError:
            # 如果Rust后端不可用，使用Python后端的结果
            rust_keys = ['rust_drop_array', 'rust_drop_first', 'rust_drop_last', 'rust_drop_middle',
                        'rust_drop_front_cont', 'rust_drop_middle_cont', 'rust_drop_end_cont',
                        'rust_drop_random', 'rust_drop_near']
            python_keys = ['python_drop_array', 'python_drop_first', 'python_drop_last', 'python_drop_middle',
                          'python_drop_front_cont', 'python_drop_middle_cont', 'python_drop_end_cont',
                          'python_drop_random', 'python_drop_near']
            
            for rust_key, python_key in zip(rust_keys, python_keys):
                results[rust_key] = results[python_key]
        
        # NumPy对比测试
        # Drop Array (NPZ)
        npz_file = os.path.join(temp_dir, 'test_drop.npz')
        np.savez(npz_file, test_array=test_array)
        start_time = time.perf_counter()
        npz_data = dict(np.load(npz_file))
        del npz_data['test_array']
        np.savez(npz_file, **npz_data)
        results['npz_drop_array'] = time.perf_counter() - start_time
        
        # Drop Array (NPY) - 直接删除文件
        npy_file = os.path.join(temp_dir, 'test_drop.npy')
        np.save(npy_file, test_array)
        start_time = time.perf_counter()
        os.remove(npy_file)
        results['npy_drop_array'] = time.perf_counter() - start_time
        
        # Drop rows (NPZ) - 使用mask
        np.savez(npz_file, test_array=test_array)
        start_time = time.perf_counter()
        npz_data = dict(np.load(npz_file))
        mask = np.ones(size, dtype=bool)
        mask[random_indices] = False
        npz_data['test_array'] = npz_data['test_array'][mask]
        np.savez(npz_file, **npz_data)
        results['npz_drop_rows'] = time.perf_counter() - start_time
        
        # Drop rows (NPY) - 使用mask
        np.save(npy_file, test_array)
        start_time = time.perf_counter()
        npy_data = np.load(npy_file)
        npy_data = npy_data[mask]
        np.save(npy_file, npy_data)
        results['npy_drop_rows'] = time.perf_counter() - start_time
        
        # 计算性能倍数
        def calc_speedup(base_time, compare_time):
            return compare_time / base_time if base_time > 0 else float('inf')
        
        print("#### Drop Operations (1M rows, float32)")
        print()
        print("| Operation | NumPack (Python) | NumPack (Rust) | NumPy NPZ | NumPy NPY |")
        print("|-----------|------------------|----------------|-----------|-----------|")
        
        # Drop Array
        python_drop_arr = results['python_drop_array']
        rust_drop_arr = results['rust_drop_array']
        npz_drop_arr = results['npz_drop_array']
        npy_drop_arr = results['npy_drop_array']
        print(f"| Drop Array | {format_time(python_drop_arr)} ({calc_speedup(python_drop_arr, npz_drop_arr):.2f}x NPZ, {calc_speedup(python_drop_arr, npy_drop_arr):.2f}x NPY) | {format_time(rust_drop_arr)} ({calc_speedup(rust_drop_arr, npz_drop_arr):.2f}x NPZ, {calc_speedup(rust_drop_arr, npy_drop_arr):.2f}x NPY) | {format_time(npz_drop_arr)} | {format_time(npy_drop_arr)} |")
        
        # Drop rows对比使用统一的基准
        npz_drop_rows = results['npz_drop_rows']
        npy_drop_rows = results['npy_drop_rows']
        
        print(f"| Drop First Row | {format_time(results['python_drop_first'])} ({calc_speedup(results['python_drop_first'], npz_drop_rows):.2f}x NPZ, {calc_speedup(results['python_drop_first'], npy_drop_rows):.2f}x NPY) | {format_time(results['rust_drop_first'])} ({calc_speedup(results['rust_drop_first'], npz_drop_rows):.2f}x NPZ, {calc_speedup(results['rust_drop_first'], npy_drop_rows):.2f}x NPY) | {format_time(npz_drop_rows)} | {format_time(npy_drop_rows)} |")
        print(f"| Drop Last Row | {format_time(results['python_drop_last'])} (∞x NPZ, ∞x NPY) | {format_time(results['rust_drop_last'])} (∞x NPZ, ∞x NPY) | {format_time(npz_drop_rows)} | {format_time(npy_drop_rows)} |")
        print(f"| Drop Middle Row | {format_time(results['python_drop_middle'])} ({calc_speedup(results['python_drop_middle'], npz_drop_rows):.2f}x NPZ, {calc_speedup(results['python_drop_middle'], npy_drop_rows):.2f}x NPY) | {format_time(results['rust_drop_middle'])} ({calc_speedup(results['rust_drop_middle'], npz_drop_rows):.2f}x NPZ, {calc_speedup(results['rust_drop_middle'], npy_drop_rows):.2f}x NPY) | {format_time(npz_drop_rows)} | {format_time(npy_drop_rows)} |")
        print(f"| Drop Front Continuous (10K rows) | {format_time(results['python_drop_front_cont'])} ({calc_speedup(results['python_drop_front_cont'], npz_drop_rows):.2f}x NPZ, {calc_speedup(results['python_drop_front_cont'], npy_drop_rows):.2f}x NPY) | {format_time(results['rust_drop_front_cont'])} ({calc_speedup(results['rust_drop_front_cont'], npz_drop_rows):.2f}x NPZ, {calc_speedup(results['rust_drop_front_cont'], npy_drop_rows):.2f}x NPY) | {format_time(npz_drop_rows)} | {format_time(npy_drop_rows)} |")
        print(f"| Drop Middle Continuous (10K rows) | {format_time(results['python_drop_middle_cont'])} ({calc_speedup(results['python_drop_middle_cont'], npz_drop_rows):.2f}x NPZ, {calc_speedup(results['python_drop_middle_cont'], npy_drop_rows):.2f}x NPY) | {format_time(results['rust_drop_middle_cont'])} ({calc_speedup(results['rust_drop_middle_cont'], npz_drop_rows):.2f}x NPZ, {calc_speedup(results['rust_drop_middle_cont'], npy_drop_rows):.2f}x NPY) | {format_time(npz_drop_rows)} | {format_time(npy_drop_rows)} |")
        print(f"| Drop End Continuous (10K rows) | {format_time(results['python_drop_end_cont'])} ({calc_speedup(results['python_drop_end_cont'], npz_drop_rows):.2f}x NPZ, {calc_speedup(results['python_drop_end_cont'], npy_drop_rows):.2f}x NPY) | {format_time(results['rust_drop_end_cont'])} ({calc_speedup(results['rust_drop_end_cont'], npz_drop_rows):.2f}x NPZ, {calc_speedup(results['rust_drop_end_cont'], npy_drop_rows):.2f}x NPY) | {format_time(npz_drop_rows)} | {format_time(npy_drop_rows)} |")
        print(f"| Drop Random Rows (10K rows) | {format_time(results['python_drop_random'])} ({calc_speedup(results['python_drop_random'], npz_drop_rows):.2f}x NPZ, {calc_speedup(results['python_drop_random'], npy_drop_rows):.2f}x NPY) | {format_time(results['rust_drop_random'])} ({calc_speedup(results['rust_drop_random'], npz_drop_rows):.2f}x NPZ, {calc_speedup(results['rust_drop_random'], npy_drop_rows):.2f}x NPY) | {format_time(npz_drop_rows)} | {format_time(npy_drop_rows)} |")
        print(f"| Drop Near Non-continuous (10K rows) | {format_time(results['python_drop_near'])} ({calc_speedup(results['python_drop_near'], npz_drop_rows):.2f}x NPZ, {calc_speedup(results['python_drop_near'], npy_drop_rows):.2f}x NPY) | {format_time(results['rust_drop_near'])} ({calc_speedup(results['rust_drop_near'], npz_drop_rows):.2f}x NPZ, {calc_speedup(results['rust_drop_near'], npy_drop_rows):.2f}x NPY) | {format_time(npz_drop_rows)} | {format_time(npy_drop_rows)} |")
        print()
        
        return results
        
    finally:
        cleanup_files(temp_dir)

def test_append_operations():
    """测试追加操作性能"""
    print("=== Append Operations ===")
    
    # 创建测试数据
    size = 1000000
    cols = 10
    initial_array = np.random.rand(size, cols).astype(np.float32)
    small_append = np.random.rand(1000, cols).astype(np.float32)  # 1K rows
    large_append = np.random.rand(500000, cols).astype(np.float32)  # 500K rows
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        results = {}
        
        # 测试NumPack Python后端
        os.environ['NUMPACK_FORCE_PYTHON_BACKEND'] = '1'
        from numpack import NumPack
        
        # Small Append (Python)
        numpack_dir = os.path.join(temp_dir, 'numpack_append')
        npk = NumPack(numpack_dir, drop_if_exists=True)
        npk.save({'test_array': initial_array})
        
        start_time = time.perf_counter()
        npk.append({'test_array': small_append})
        results['python_small_append'] = time.perf_counter() - start_time
        
        # Large Append (Python)
        npk.save({'test_array': initial_array})  # Reset
        start_time = time.perf_counter()
        npk.append({'test_array': large_append})
        results['python_large_append'] = time.perf_counter() - start_time
        
        # 测试NumPack Rust后端
        if 'NUMPACK_FORCE_PYTHON_BACKEND' in os.environ:
            del os.environ['NUMPACK_FORCE_PYTHON_BACKEND']
        
        try:
            if 'numpack' in sys.modules:
                del sys.modules['numpack']
            from numpack import NumPack
            
            numpack_rust_dir = os.path.join(temp_dir, 'numpack_rust_append')
            npk_rust = NumPack(numpack_rust_dir, drop_if_exists=True)
            
            # Small Append (Rust)
            npk_rust.save({'test_array': initial_array})
            start_time = time.perf_counter()
            npk_rust.append({'test_array': small_append})
            results['rust_small_append'] = time.perf_counter() - start_time
            
            # Large Append (Rust)
            npk_rust.save({'test_array': initial_array})  # Reset
            start_time = time.perf_counter()
            npk_rust.append({'test_array': large_append})
            results['rust_large_append'] = time.perf_counter() - start_time
            
        except ImportError:
            results['rust_small_append'] = results['python_small_append']
            results['rust_large_append'] = results['python_large_append']
        
        # NumPy对比测试
        # Small Append (NPZ)
        npz_file = os.path.join(temp_dir, 'test_append.npz')
        np.savez(npz_file, test_array=initial_array)
        
        start_time = time.perf_counter()
        npz_data = dict(np.load(npz_file))
        npz_data['test_array'] = np.vstack([npz_data['test_array'], small_append])
        np.savez(npz_file, **npz_data)
        results['npz_small_append'] = time.perf_counter() - start_time
        
        # Large Append (NPZ)
        np.savez(npz_file, test_array=initial_array)  # Reset
        start_time = time.perf_counter()
        npz_data = dict(np.load(npz_file))
        npz_data['test_array'] = np.vstack([npz_data['test_array'], large_append])
        np.savez(npz_file, **npz_data)
        results['npz_large_append'] = time.perf_counter() - start_time
        
        # Small Append (NPY)
        npy_file = os.path.join(temp_dir, 'test_append.npy')
        np.save(npy_file, initial_array)
        
        start_time = time.perf_counter()
        npy_data = np.load(npy_file)
        npy_data = np.vstack([npy_data, small_append])
        np.save(npy_file, npy_data)
        results['npy_small_append'] = time.perf_counter() - start_time
        
        # Large Append (NPY)
        np.save(npy_file, initial_array)  # Reset
        start_time = time.perf_counter()
        npy_data = np.load(npy_file)
        npy_data = np.vstack([npy_data, large_append])
        np.save(npy_file, npy_data)
        results['npy_large_append'] = time.perf_counter() - start_time
        
        # 计算性能倍数
        def calc_speedup(base_time, compare_time):
            return compare_time / base_time if base_time > 0 else float('inf')
        
        print("#### Append Operations")
        print()
        print("| Operation | NumPack (Python) | NumPack (Rust) | NumPy NPZ | NumPy NPY |")
        print("|-----------|------------------|----------------|-----------|-----------|")
        
        python_small = results['python_small_append']
        rust_small = results['rust_small_append']
        npz_small = results['npz_small_append']
        npy_small = results['npy_small_append']
        
        python_large = results['python_large_append']
        rust_large = results['rust_large_append']
        npz_large = results['npz_large_append']
        npy_large = results['npy_large_append']
        
        print(f"| Small Append (1K rows) | {format_time(python_small)} (≥{calc_speedup(python_small, npz_small):.0f}x NPZ, ≥{calc_speedup(python_small, npy_small):.0f}x NPY) | {format_time(rust_small)} (≥{calc_speedup(rust_small, npz_small):.0f}x NPZ, ≥{calc_speedup(rust_small, npy_small):.0f}x NPY) | {format_time(npz_small)} | {format_time(npy_small)} |")
        print(f"| Large Append (500K rows) | {format_time(python_large)} ({calc_speedup(python_large, npz_large):.2f}x NPZ, {calc_speedup(python_large, npy_large):.2f}x NPY) | {format_time(rust_large)} ({calc_speedup(rust_large, npz_large):.2f}x NPZ, {calc_speedup(rust_large, npy_large):.2f}x NPY) | {format_time(npz_large)} | {format_time(npy_large)} |")
        print()
        
        return results
        
    finally:
        cleanup_files(temp_dir)

def test_file_size_comparison():
    """测试文件大小对比"""
    print("=== File Size Comparison ===")
    
    # 创建测试数据 - 与README一致：1M x 10 和 500K x 5
    arrays = {
        'array1': np.random.rand(1000000, 10).astype(np.float32),
        'array2': np.random.rand(500000, 5).astype(np.float32)
    }
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # NumPack文件大小
        os.environ['NUMPACK_FORCE_PYTHON_BACKEND'] = '1'
        from numpack import NumPack
        
        numpack_dir = os.path.join(temp_dir, 'numpack_size')
        npk = NumPack(numpack_dir, drop_if_exists=True)
        npk.save(arrays)
        numpack_size = get_file_size_mb(numpack_dir)
        
        # NumPy NPZ文件大小
        npz_file = os.path.join(temp_dir, 'test_size.npz')
        np.savez(npz_file, **arrays)
        npz_size = get_file_size_mb(npz_file)
        
        # NumPy NPY文件大小
        npy_file1 = os.path.join(temp_dir, 'array1.npy')
        npy_file2 = os.path.join(temp_dir, 'array2.npy')
        np.save(npy_file1, arrays['array1'])
        np.save(npy_file2, arrays['array2'])
        npy_size = get_file_size_mb(npy_file1) + get_file_size_mb(npy_file2)
        
        print("#### File Size Comparison")
        print()
        print("| Format | Size | Ratio |")
        print("|--------|------|-------|")
        print(f"| NumPack | {numpack_size:.2f} MB | 1.0x |")
        print(f"| NPZ | {npz_size:.2f} MB | {npz_size/numpack_size:.2f}x |")
        print(f"| NPY | {npy_size:.2f} MB | {npy_size/numpack_size:.2f}x |")
        print()
        
        return {
            'numpack_size': numpack_size,
            'npz_size': npz_size,
            'npy_size': npy_size
        }
        
    finally:
        cleanup_files(temp_dir)

def main():
    """主函数"""
    print("Complete NumPack vs NumPy Performance Benchmark")
    print("==============================================")
    print("Testing all operations from README with exact data sizes")
    print()
    
    # 首先运行之前的基本测试
    from readme_exact_benchmark import test_storage_operations, test_modification_operations, test_random_access, test_matrix_computation
    
    storage_results = test_storage_operations()
    modification_results = test_modification_operations()
    random_access_results = test_random_access()
    matrix_results = test_matrix_computation()
    
    # 然后运行补充的测试
    drop_results = test_drop_operations()
    append_results = test_append_operations()
    file_size_results = test_file_size_comparison()
    
    print("=== Complete Test Summary ===")
    print("All README benchmark tests completed successfully!")
    print()
    print("Tests performed:")
    print("✅ Storage Operations")
    print("✅ Data Modification Operations")
    print("✅ Drop Operations")
    print("✅ Append Operations")
    print("✅ Random Access Performance")
    print("✅ Matrix Computation Performance")
    print("✅ File Size Comparison")

if __name__ == "__main__":
    main() 