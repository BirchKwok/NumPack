#!/usr/bin/env python3
"""
与README完全一致的基准测试脚本
使用相同的数据规模：1M x 10 和 500K x 5 (float32)
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

def test_storage_operations():
    """测试存储操作 - 与README完全一致"""
    print("=== Storage Operations ===")
    
    # 创建测试数据 - 与README一致：1M x 10 和 500K x 5
    arrays = {
        'array1': np.random.rand(1000000, 10).astype(np.float32),
        'array2': np.random.rand(500000, 5).astype(np.float32)
    }
    
    temp_dir = tempfile.mkdtemp()
    numpack_dir = os.path.join(temp_dir, 'numpack_test')
    npz_file = os.path.join(temp_dir, 'test.npz')
    npy_file1 = os.path.join(temp_dir, 'array1.npy')
    npy_file2 = os.path.join(temp_dir, 'array2.npy')
    
    try:
        # 测试Python后端
        os.environ['NUMPACK_FORCE_PYTHON_BACKEND'] = '1'
        from numpack import NumPack
        
        # NumPack Save (Python)
        start_time = time.perf_counter()
        npk_python = NumPack(numpack_dir, drop_if_exists=True)
        npk_python.save(arrays)
        python_save_time = time.perf_counter() - start_time
        
        # NumPack Full Load (Python)
        start_time = time.perf_counter()
        loaded_python = {}
        for key in arrays.keys():
            loaded_python[key] = npk_python.load(key)
        python_load_time = time.perf_counter() - start_time
        
        # NumPack Selective Load (Python)
        start_time = time.perf_counter()
        selective_python = npk_python.load('array1')
        python_selective_time = time.perf_counter() - start_time
        
        # 测试Rust后端
        if 'NUMPACK_FORCE_PYTHON_BACKEND' in os.environ:
            del os.environ['NUMPACK_FORCE_PYTHON_BACKEND']
        
        try:
            # 重新导入以使用Rust后端
            if 'numpack' in sys.modules:
                del sys.modules['numpack']
            from numpack import NumPack
            
            numpack_rust_dir = os.path.join(temp_dir, 'numpack_rust_test')
            
            # NumPack Save (Rust)
            start_time = time.perf_counter()
            npk_rust = NumPack(numpack_rust_dir, drop_if_exists=True)
            npk_rust.save(arrays)
            rust_save_time = time.perf_counter() - start_time
            
            # NumPack Full Load (Rust)
            start_time = time.perf_counter()
            loaded_rust = {}
            for key in arrays.keys():
                loaded_rust[key] = npk_rust.load(key)
            rust_load_time = time.perf_counter() - start_time
            
            # NumPack Selective Load (Rust)
            start_time = time.perf_counter()
            selective_rust = npk_rust.load('array1')
            rust_selective_time = time.perf_counter() - start_time
            
        except ImportError:
            print("Rust backend not available, using Python backend only")
            rust_save_time = python_save_time
            rust_load_time = python_load_time
            rust_selective_time = python_selective_time
        
        # NumPy NPZ Save
        start_time = time.perf_counter()
        np.savez(npz_file, **arrays)
        npz_save_time = time.perf_counter() - start_time
        
        # NumPy NPZ Load
        start_time = time.perf_counter()
        npz_data = np.load(npz_file)
        npz_loaded = {key: npz_data[key] for key in arrays.keys()}
        npz_load_time = time.perf_counter() - start_time
        
        # NumPy NPZ Selective Load
        start_time = time.perf_counter()
        npz_data = np.load(npz_file)
        npz_selective = npz_data['array1']
        npz_selective_time = time.perf_counter() - start_time
        
        # NumPy NPY Save
        start_time = time.perf_counter()
        np.save(npy_file1, arrays['array1'])
        np.save(npy_file2, arrays['array2'])
        npy_save_time = time.perf_counter() - start_time
        
        # NumPy NPY Load
        start_time = time.perf_counter()
        npy_loaded = {
            'array1': np.load(npy_file1),
            'array2': np.load(npy_file2)
        }
        npy_load_time = time.perf_counter() - start_time
        
        # 计算性能倍数
        save_vs_npz_python = npz_save_time / python_save_time
        save_vs_npy_python = npy_save_time / python_save_time
        save_vs_npz_rust = npz_save_time / rust_save_time
        save_vs_npy_rust = npy_save_time / rust_save_time
        
        load_vs_npz_python = npz_load_time / python_load_time
        load_vs_npy_python = npy_load_time / python_load_time
        load_vs_npz_rust = npz_load_time / rust_load_time
        load_vs_npy_rust = npy_load_time / rust_load_time
        
        selective_vs_npz_python = npz_selective_time / python_selective_time
        selective_vs_npz_rust = npz_selective_time / rust_selective_time
        
        print("#### Storage Operations")
        print()
        print("| Operation | NumPack (Python) | NumPack (Rust) | NumPy NPZ | NumPy NPY |")
        print("|-----------|------------------|----------------|-----------|-----------|")
        print(f"| Save | {format_time(python_save_time)} ({save_vs_npz_python:.2f}x NPZ, {save_vs_npy_python:.2f}x NPY) | {format_time(rust_save_time)} ({save_vs_npz_rust:.2f}x NPZ, {save_vs_npy_rust:.2f}x NPY) | {format_time(npz_save_time)} | {format_time(npy_save_time)} |")
        print(f"| Full Load | {format_time(python_load_time)} ({load_vs_npz_python:.2f}x NPZ, {load_vs_npy_python:.2f}x NPY) | {format_time(rust_load_time)} ({load_vs_npz_rust:.2f}x NPZ, {load_vs_npy_rust:.2f}x NPY) | {format_time(npz_load_time)} | {format_time(npy_load_time)} |")
        print(f"| Selective Load | {format_time(python_selective_time)} ({selective_vs_npz_python:.2f}x NPZ, -) | {format_time(rust_selective_time)} ({selective_vs_npz_rust:.2f}x NPZ, -) | {format_time(npz_selective_time)} | - |")
        print()
        
        return {
            'python_save': python_save_time,
            'rust_save': rust_save_time,
            'python_load': python_load_time,
            'rust_load': rust_load_time,
            'python_selective': python_selective_time,
            'rust_selective': rust_selective_time,
            'npz_save': npz_save_time,
            'npy_save': npy_save_time,
            'npz_load': npz_load_time,
            'npy_load': npy_load_time
        }
        
    finally:
        cleanup_files(temp_dir)

def test_modification_operations():
    """测试数据修改操作"""
    print("=== Data Modification Operations ===")
    
    # 创建测试数据
    size = 1000000
    cols = 10
    test_array = np.random.rand(size, cols).astype(np.float32)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        results = {}
        
        # 测试NumPack Python后端
        os.environ['NUMPACK_FORCE_PYTHON_BACKEND'] = '1'
        from numpack import NumPack
        
        numpack_dir = os.path.join(temp_dir, 'numpack_modify')
        npk = NumPack(numpack_dir, drop_if_exists=True)
        npk.save({'test_array': test_array})
        
        # Single Row Replace (Python)
        single_row = np.random.rand(1, cols).astype(np.float32)
        start_time = time.perf_counter()
        npk.replace({'test_array': single_row}, [size//2])
        results['python_single_replace'] = time.perf_counter() - start_time
        
        # Continuous Rows Replace (Python) - 10K rows
        npk.save({'test_array': test_array})  # Reset
        continuous_rows = np.random.rand(10000, cols).astype(np.float32)
        start_time = time.perf_counter()
        npk.replace({'test_array': continuous_rows}, slice(0, 10000))
        results['python_continuous_replace'] = time.perf_counter() - start_time
        
        # Random Rows Replace (Python) - 10K rows
        npk.save({'test_array': test_array})  # Reset
        random_indices = np.random.choice(size, 10000, replace=False).tolist()
        random_rows = np.random.rand(10000, cols).astype(np.float32)
        start_time = time.perf_counter()
        npk.replace({'test_array': random_rows}, random_indices)
        results['python_random_replace'] = time.perf_counter() - start_time
        
        # Large Data Replace (Python) - 500K rows
        npk.save({'test_array': test_array})  # Reset
        large_rows = np.random.rand(500000, cols).astype(np.float32)
        start_time = time.perf_counter()
        npk.replace({'test_array': large_rows}, slice(0, 500000))
        results['python_large_replace'] = time.perf_counter() - start_time
        
        # 测试NumPack Rust后端
        if 'NUMPACK_FORCE_PYTHON_BACKEND' in os.environ:
            del os.environ['NUMPACK_FORCE_PYTHON_BACKEND']
        
        try:
            if 'numpack' in sys.modules:
                del sys.modules['numpack']
            from numpack import NumPack
            
            numpack_rust_dir = os.path.join(temp_dir, 'numpack_rust_modify')
            npk_rust = NumPack(numpack_rust_dir, drop_if_exists=True)
            npk_rust.save({'test_array': test_array})
            
            # Single Row Replace (Rust)
            start_time = time.perf_counter()
            npk_rust.replace({'test_array': single_row}, [size//2])
            results['rust_single_replace'] = time.perf_counter() - start_time
            
            # Continuous Rows Replace (Rust)
            npk_rust.save({'test_array': test_array})  # Reset
            start_time = time.perf_counter()
            npk_rust.replace({'test_array': continuous_rows}, slice(0, 10000))
            results['rust_continuous_replace'] = time.perf_counter() - start_time
            
            # Random Rows Replace (Rust)
            npk_rust.save({'test_array': test_array})  # Reset
            start_time = time.perf_counter()
            npk_rust.replace({'test_array': random_rows}, random_indices)
            results['rust_random_replace'] = time.perf_counter() - start_time
            
            # Large Data Replace (Rust)
            npk_rust.save({'test_array': test_array})  # Reset
            start_time = time.perf_counter()
            npk_rust.replace({'test_array': large_rows}, slice(0, 500000))
            results['rust_large_replace'] = time.perf_counter() - start_time
            
        except ImportError:
            results.update({
                'rust_single_replace': results['python_single_replace'],
                'rust_continuous_replace': results['python_continuous_replace'],
                'rust_random_replace': results['python_random_replace'],
                'rust_large_replace': results['python_large_replace']
            })
        
        # NumPy对比测试
        # Single Row Replace (NPZ)
        npz_file = os.path.join(temp_dir, 'test.npz')
        np.savez(npz_file, test_array=test_array)
        
        start_time = time.perf_counter()
        npz_data = dict(np.load(npz_file))
        npz_data['test_array'][size//2] = single_row[0]
        np.savez(npz_file, **npz_data)
        results['npz_single_replace'] = time.perf_counter() - start_time
        
        # Single Row Replace (NPY)
        npy_file = os.path.join(temp_dir, 'test.npy')
        np.save(npy_file, test_array)
        
        start_time = time.perf_counter()
        npy_data = np.load(npy_file)
        npy_data[size//2] = single_row[0]
        np.save(npy_file, npy_data)
        results['npy_single_replace'] = time.perf_counter() - start_time
        
        # 计算性能倍数
        def calc_speedup(base_time, compare_time):
            return compare_time / base_time if base_time > 0 else float('inf')
        
        print("#### Data Modification Operations")
        print()
        print("| Operation | NumPack (Python) | NumPack (Rust) | NumPy NPZ | NumPy NPY |")
        print("|-----------|------------------|----------------|-----------|-----------|")
        
        python_single = results['python_single_replace']
        rust_single = results['rust_single_replace']
        npz_single = results['npz_single_replace']
        npy_single = results['npy_single_replace']
        
        print(f"| Single Row Replace | {format_time(python_single)} (≥{calc_speedup(python_single, npz_single):.0f}x NPZ, ≥{calc_speedup(python_single, npy_single):.0f}x NPY) | {format_time(rust_single)} (≥{calc_speedup(rust_single, npz_single):.0f}x NPZ, ≥{calc_speedup(rust_single, npy_single):.0f}x NPY) | {format_time(npz_single)} | {format_time(npy_single)} |")
        
        python_cont = results['python_continuous_replace']
        rust_cont = results['rust_continuous_replace']
        print(f"| Continuous Rows (10K) | {format_time(python_cont)} | {format_time(rust_cont)} | - | - |")
        
        python_random = results['python_random_replace']
        rust_random = results['rust_random_replace']
        print(f"| Random Rows (10K) | {format_time(python_random)} | {format_time(rust_random)} | - | - |")
        
        python_large = results['python_large_replace']
        rust_large = results['rust_large_replace']
        print(f"| Large Data Replace (500K) | {format_time(python_large)} | {format_time(rust_large)} | - | - |")
        print()
        
        return results
        
    finally:
        cleanup_files(temp_dir)

def test_random_access():
    """测试随机访问性能"""
    print("=== Random Access Performance (10K indices) ===")
    
    # 创建测试数据
    size = 1000000
    cols = 10
    test_array = np.random.rand(size, cols).astype(np.float32)
    indices = np.random.randint(0, size, 10000)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 测试NumPack Python后端
        os.environ['NUMPACK_FORCE_PYTHON_BACKEND'] = '1'
        from numpack import NumPack
        
        numpack_dir = os.path.join(temp_dir, 'numpack_random')
        npk = NumPack(numpack_dir, drop_if_exists=True)
        npk.save({'test_array': test_array})
        
        start_time = time.perf_counter()
        python_result = npk.getitem('test_array', indices.tolist())
        python_time = time.perf_counter() - start_time
        
        # 测试NumPack Rust后端
        if 'NUMPACK_FORCE_PYTHON_BACKEND' in os.environ:
            del os.environ['NUMPACK_FORCE_PYTHON_BACKEND']
        
        try:
            if 'numpack' in sys.modules:
                del sys.modules['numpack']
            from numpack import NumPack
            
            numpack_rust_dir = os.path.join(temp_dir, 'numpack_rust_random')
            npk_rust = NumPack(numpack_rust_dir, drop_if_exists=True)
            npk_rust.save({'test_array': test_array})
            
            start_time = time.perf_counter()
            rust_result = npk_rust.getitem('test_array', indices.tolist())
            rust_time = time.perf_counter() - start_time
            
        except ImportError:
            rust_time = python_time
        
        # NumPy NPZ测试
        npz_file = os.path.join(temp_dir, 'test.npz')
        np.savez(npz_file, test_array=test_array)
        
        start_time = time.perf_counter()
        npz_data = np.load(npz_file, mmap_mode='r')
        npz_result = npz_data['test_array'][indices]
        npz_time = time.perf_counter() - start_time
        
        # NumPy NPY测试
        npy_file = os.path.join(temp_dir, 'test.npy')
        np.save(npy_file, test_array)
        
        start_time = time.perf_counter()
        npy_data = np.load(npy_file, mmap_mode='r')
        npy_result = npy_data[indices]
        npy_time = time.perf_counter() - start_time
        
        # 计算性能倍数
        python_vs_npz = npz_time / python_time
        python_vs_npy = npy_time / python_time
        rust_vs_npz = npz_time / rust_time
        rust_vs_npy = npy_time / rust_time
        
        print("#### Random Access Performance (10K indices)")
        print()
        print("| Operation | NumPack (Python) | NumPack (Rust) | NumPy NPZ | NumPy NPY |")
        print("|-----------|------------------|----------------|-----------|-----------|")
        print(f"| Random Access | {format_time(python_time)} ({python_vs_npz:.2f}x NPZ, {python_vs_npy:.2f}x NPY) | {format_time(rust_time)} ({rust_vs_npz:.2f}x NPZ, {rust_vs_npy:.2f}x NPY) | {format_time(npz_time)} | {format_time(npy_time)} |")
        print()
        
        return {
            'python_time': python_time,
            'rust_time': rust_time,
            'npz_time': npz_time,
            'npy_time': npy_time
        }
        
    finally:
        cleanup_files(temp_dir)

def test_matrix_computation():
    """测试矩阵计算性能"""
    print("=== Matrix Computation Performance (1M rows x 128 columns, Float32) ===")
    
    # 创建测试数据 - 1M x 128
    rows = 1000000
    cols = 128
    test_array = np.random.rand(rows, cols).astype(np.float32)
    query_vector = np.random.rand(1, cols).astype(np.float32)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 测试NumPack Python后端
        os.environ['NUMPACK_FORCE_PYTHON_BACKEND'] = '1'
        from numpack import NumPack
        
        numpack_dir = os.path.join(temp_dir, 'numpack_matrix')
        npk = NumPack(numpack_dir, drop_if_exists=True)
        npk.save({'test_array': test_array})
        
        lazy_array = npk.load('test_array', lazy=True)
        start_time = time.perf_counter()
        python_result = np.inner(query_vector, lazy_array)
        python_time = time.perf_counter() - start_time
        
        # 测试NumPack Rust后端
        if 'NUMPACK_FORCE_PYTHON_BACKEND' in os.environ:
            del os.environ['NUMPACK_FORCE_PYTHON_BACKEND']
        
        try:
            if 'numpack' in sys.modules:
                del sys.modules['numpack']
            from numpack import NumPack
            
            numpack_rust_dir = os.path.join(temp_dir, 'numpack_rust_matrix')
            npk_rust = NumPack(numpack_rust_dir, drop_if_exists=True)
            npk_rust.save({'test_array': test_array})
            
            lazy_array_rust = npk_rust.load('test_array', lazy=True)
            start_time = time.perf_counter()
            rust_result = np.inner(query_vector, lazy_array_rust)
            rust_time = time.perf_counter() - start_time
            
        except ImportError:
            rust_time = python_time
        
        # In-Memory测试
        start_time = time.perf_counter()
        memory_result = np.inner(query_vector, test_array)
        memory_time = time.perf_counter() - start_time
        
        # NumPy NPZ mmap测试
        npz_file = os.path.join(temp_dir, 'matrix_test.npz')
        np.savez(npz_file, test_array=test_array)
        
        start_time = time.perf_counter()
        npz_data = np.load(npz_file, mmap_mode='r')
        npz_result = np.inner(query_vector, npz_data['test_array'])
        npz_time = time.perf_counter() - start_time
        
        # NumPy NPY mmap测试
        npy_file = os.path.join(temp_dir, 'matrix_test.npy')
        np.save(npy_file, test_array)
        
        start_time = time.perf_counter()
        npy_data = np.load(npy_file, mmap_mode='r')
        npy_result = np.inner(query_vector, npy_data)
        npy_time = time.perf_counter() - start_time
        
        # 计算性能倍数
        python_vs_npz = npz_time / python_time
        python_vs_memory = python_time / memory_time
        rust_vs_npz = npz_time / rust_time
        rust_vs_memory = rust_time / memory_time
        
        print("#### Matrix Computation Performance (1M rows x 128 columns, Float32)")
        print()
        print("| Operation | NumPack (Python) | NumPack (Rust) | NumPy NPZ | NumPy NPY | In-Memory |")
        print("|-----------|------------------|----------------|-----------|-----------|-----------|")
        print(f"| Inner Product | {format_time(python_time)} ({python_vs_npz:.2f}x NPZ, {python_vs_memory:.2f}x Memory) | {format_time(rust_time)} ({rust_vs_npz:.2f}x NPZ, {rust_vs_memory:.2f}x Memory) | {format_time(npz_time)} | {format_time(npy_time)} | {format_time(memory_time)} |")
        print()
        
        return {
            'python_time': python_time,
            'rust_time': rust_time,
            'npz_time': npz_time,
            'npy_time': npy_time,
            'memory_time': memory_time
        }
        
    finally:
        cleanup_files(temp_dir)

def main():
    """主函数"""
    print("NumPack vs NumPy Performance Benchmark")
    print("=======================================")
    print("Using the exact same data sizes as README: 1M x 10 and 500K x 5 (float32)")
    print()
    
    # 运行各项测试
    storage_results = test_storage_operations()
    modification_results = test_modification_operations()
    random_access_results = test_random_access()
    matrix_results = test_matrix_computation()
    
    print("=== Test Completed ===")
    print("All benchmarks completed successfully!")

if __name__ == "__main__":
    main() 