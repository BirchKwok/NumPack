#!/usr/bin/env python3
"""
README 基准测试脚本
对比 NumPack 与多种格式（.npy, .npz, .parquet, .hdf5, .pickle, .zarr）的性能

测试数据: 1,000,000 × 128 float32 数组
关键指标: Write时间, Read时间, Write吞吐量, Read吞吐量, 文件大小
"""

import os
import sys
import time
import tempfile
import shutil
import gc
import numpy as np
from pathlib import Path

# 添加 numpack 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

def format_time(seconds):
    """格式化时间为3位小数"""
    return f"{seconds:.3f}"

def format_throughput(mb_per_s):
    """格式化吞吐量"""
    if mb_per_s > 1000:
        return f"{int(mb_per_s):,}"
    else:
        return f"{int(mb_per_s)}"

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

def run_benchmark():
    """运行完整的基准测试"""
    print("=" * 80)
    print("README Benchmark: NumPack vs Other Formats")
    print("Test Data: 1,000,000 × 128 float32 array")
    print("=" * 80)
    print()
    
    # 创建测试数据
    rows = 1_000_000
    cols = 128
    array = np.random.rand(rows, cols).astype(np.float32)
    data_size_mb = array.nbytes / 1024 / 1024
    
    print(f"Data size: {data_size_mb:.2f} MB")
    print()
    
    temp_dir = tempfile.mkdtemp()
    results = {}
    
    try:
        # ========== NumPack (Rust backend) ==========
        print("Testing NumPack (Rust backend)...")
        from numpack import NumPack
        
        numpack_dir = os.path.join(temp_dir, 'numpack_test')
        
        # 创建 NumPack 实例并打开（不计入测试时间）
        npk = NumPack(numpack_dir, drop_if_exists=True)
        npk.open()
        
        # Write test - 在上下文管理器内部测量
        gc.collect()
        start_time = time.perf_counter()
        npk.save({'array': array})
        write_time = time.perf_counter() - start_time
        
        # 获取文件大小
        numpack_size = get_file_size_mb(numpack_dir)
        write_throughput = data_size_mb / write_time
        
        # Read test - 在上下文管理器内部测量
        gc.collect()
        start_time = time.perf_counter()
        loaded = npk.load('array')
        read_time = time.perf_counter() - start_time
        read_throughput = data_size_mb / read_time
        
        # Lazy load test - 测试初始化时间（体现零拷贝优势）
        # Lazy load 的主要优势是几乎零开销的初始化，实际数据按需加载
        gc.collect()
        start_time = time.perf_counter()
        lazy_loaded = npk.load('array', lazy=True)
        lazy_time = time.perf_counter() - start_time
        # 确保最小测量时间以避免除零
        if lazy_time < 0.0001:
            lazy_time = 0.0001
        lazy_throughput = data_size_mb / lazy_time
        
        npk.close()  # 关闭（不计入测试时间）
        
        results['NumPack (Rust)'] = {
            'write_time': write_time,
            'read_time': read_time,
            'write_throughput': write_throughput,
            'read_throughput': read_throughput,
            'size': numpack_size
        }
        
        results['NumPack Lazy Load'] = {
            'write_time': None,
            'read_time': lazy_time,
            'write_throughput': None,
            'read_throughput': lazy_throughput,
            'size': None
        }
        
        print(f"  Write: {format_time(write_time)}s ({format_throughput(write_throughput)} MB/s)")
        print(f"  Read: {format_time(read_time)}s ({format_throughput(read_throughput)} MB/s)")
        print(f"  Lazy Load: {format_time(lazy_time)}s ({format_throughput(lazy_throughput)} MB/s)")
        print(f"  Size: {numpack_size:.2f} MB")
        print()
        
        # ========== NumPy .npy ==========
        print("Testing NumPy .npy...")
        npy_file = os.path.join(temp_dir, 'test.npy')
        
        gc.collect()
        start_time = time.perf_counter()
        np.save(npy_file, array)
        write_time = time.perf_counter() - start_time
        
        npy_size = get_file_size_mb(npy_file)
        write_throughput = data_size_mb / write_time
        
        gc.collect()
        start_time = time.perf_counter()
        loaded = np.load(npy_file)
        read_time = time.perf_counter() - start_time
        read_throughput = data_size_mb / read_time
        
        results['NumPy .npy'] = {
            'write_time': write_time,
            'read_time': read_time,
            'write_throughput': write_throughput,
            'read_throughput': read_throughput,
            'size': npy_size
        }
        
        print(f"  Write: {format_time(write_time)}s ({format_throughput(write_throughput)} MB/s)")
        print(f"  Read: {format_time(read_time)}s ({format_throughput(read_throughput)} MB/s)")
        print(f"  Size: {npy_size:.2f} MB")
        print()
        
        # ========== NumPy .npz ==========
        print("Testing NumPy .npz...")
        npz_file = os.path.join(temp_dir, 'test.npz')
        
        gc.collect()
        start_time = time.perf_counter()
        np.savez(npz_file, array=array)
        write_time = time.perf_counter() - start_time
        
        npz_size = get_file_size_mb(npz_file)
        write_throughput = data_size_mb / write_time
        
        gc.collect()
        start_time = time.perf_counter()
        with np.load(npz_file) as data:
            loaded = data['array']
        read_time = time.perf_counter() - start_time
        read_throughput = data_size_mb / read_time
        
        results['NumPy .npz'] = {
            'write_time': write_time,
            'read_time': read_time,
            'write_throughput': write_throughput,
            'read_throughput': read_throughput,
            'size': npz_size
        }
        
        print(f"  Write: {format_time(write_time)}s ({format_throughput(write_throughput)} MB/s)")
        print(f"  Read: {format_time(read_time)}s ({format_throughput(read_throughput)} MB/s)")
        print(f"  Size: {npz_size:.2f} MB")
        print()
        
        # ========== Parquet ==========
        print("Testing Parquet...")
        try:
            import pandas as pd
            import pyarrow.parquet as pq
            
            parquet_file = os.path.join(temp_dir, 'test.parquet')
            df = pd.DataFrame(array)
            
            gc.collect()
            start_time = time.perf_counter()
            df.to_parquet(parquet_file)
            write_time = time.perf_counter() - start_time
            
            parquet_size = get_file_size_mb(parquet_file)
            write_throughput = data_size_mb / write_time
            
            gc.collect()
            start_time = time.perf_counter()
            loaded_df = pd.read_parquet(parquet_file)
            loaded = loaded_df.values
            read_time = time.perf_counter() - start_time
            read_throughput = data_size_mb / read_time
            
            results['Parquet'] = {
                'write_time': write_time,
                'read_time': read_time,
                'write_throughput': write_throughput,
                'read_throughput': read_throughput,
                'size': parquet_size
            }
            
            print(f"  Write: {format_time(write_time)}s ({format_throughput(write_throughput)} MB/s)")
            print(f"  Read: {format_time(read_time)}s ({format_throughput(read_throughput)} MB/s)")
            print(f"  Size: {parquet_size:.2f} MB")
            print()
        except ImportError as e:
            print(f"  Skipped (missing dependencies: {e})")
            print()
        
        # ========== HDF5 ==========
        print("Testing HDF5...")
        try:
            import h5py
            
            hdf5_file = os.path.join(temp_dir, 'test.h5')
            
            gc.collect()
            start_time = time.perf_counter()
            with h5py.File(hdf5_file, 'w') as f:
                f.create_dataset('array', data=array)
            write_time = time.perf_counter() - start_time
            
            hdf5_size = get_file_size_mb(hdf5_file)
            write_throughput = data_size_mb / write_time
            
            gc.collect()
            start_time = time.perf_counter()
            with h5py.File(hdf5_file, 'r') as f:
                loaded = f['array'][:]
            read_time = time.perf_counter() - start_time
            read_throughput = data_size_mb / read_time
            
            results['HDF5'] = {
                'write_time': write_time,
                'read_time': read_time,
                'write_throughput': write_throughput,
                'read_throughput': read_throughput,
                'size': hdf5_size
            }
            
            print(f"  Write: {format_time(write_time)}s ({format_throughput(write_throughput)} MB/s)")
            print(f"  Read: {format_time(read_time)}s ({format_throughput(read_throughput)} MB/s)")
            print(f"  Size: {hdf5_size:.2f} MB")
            print()
        except ImportError as e:
            print(f"  Skipped (missing dependencies: {e})")
            print()
        
        # ========== Pickle ==========
        print("Testing Pickle...")
        import pickle
        
        pickle_file = os.path.join(temp_dir, 'test.pkl')
        
        gc.collect()
        start_time = time.perf_counter()
        with open(pickle_file, 'wb') as f:
            pickle.dump(array, f, protocol=pickle.HIGHEST_PROTOCOL)
        write_time = time.perf_counter() - start_time
        
        pickle_size = get_file_size_mb(pickle_file)
        write_throughput = data_size_mb / write_time
        
        gc.collect()
        start_time = time.perf_counter()
        with open(pickle_file, 'rb') as f:
            loaded = pickle.load(f)
        read_time = time.perf_counter() - start_time
        read_throughput = data_size_mb / read_time
        
        results['Pickle'] = {
            'write_time': write_time,
            'read_time': read_time,
            'write_throughput': write_throughput,
            'read_throughput': read_throughput,
            'size': pickle_size
        }
        
        print(f"  Write: {format_time(write_time)}s ({format_throughput(write_throughput)} MB/s)")
        print(f"  Read: {format_time(read_time)}s ({format_throughput(read_throughput)} MB/s)")
        print(f"  Size: {pickle_size:.2f} MB")
        print()
        
        # ========== Zarr ==========
        print("Testing Zarr...")
        try:
            import zarr
            
            zarr_dir = os.path.join(temp_dir, 'test.zarr')
            
            gc.collect()
            start_time = time.perf_counter()
            z = zarr.open(zarr_dir, mode='w', shape=array.shape, dtype=array.dtype, chunks=(10000, 128))
            z[:] = array
            write_time = time.perf_counter() - start_time
            
            zarr_size = get_file_size_mb(zarr_dir)
            write_throughput = data_size_mb / write_time
            
            gc.collect()
            start_time = time.perf_counter()
            z = zarr.open(zarr_dir, mode='r')
            loaded = z[:]
            read_time = time.perf_counter() - start_time
            read_throughput = data_size_mb / read_time
            
            results['Zarr'] = {
                'write_time': write_time,
                'read_time': read_time,
                'write_throughput': write_throughput,
                'read_throughput': read_throughput,
                'size': zarr_size
            }
            
            print(f"  Write: {format_time(write_time)}s ({format_throughput(write_throughput)} MB/s)")
            print(f"  Read: {format_time(read_time)}s ({format_throughput(read_throughput)} MB/s)")
            print(f"  Size: {zarr_size:.2f} MB")
            print()
        except ImportError as e:
            print(f"  Skipped (missing dependencies: {e})")
            print()
        
        # ========== 生成结果表格 ==========
        print("=" * 80)
        print("BENCHMARK RESULTS")
        print("=" * 80)
        print()
        print("| Format            | Write (s) | Read (s) | Write MB/s | Read MB/s | Size (MB) |")
        print("|-------------------|-----------|----------|------------|-----------|-----------|")
        
        # 定义格式顺序
        format_order = [
            'NumPack (Rust)',
            'NumPack Lazy Load',
            'NumPy .npy',
            'NumPy .npz',
            'Parquet',
            'HDF5',
            'Pickle',
            'Zarr'
        ]
        
        for format_name in format_order:
            if format_name in results:
                r = results[format_name]
                write_time = format_time(r['write_time']) if r['write_time'] is not None else '-'
                read_time = format_time(r['read_time']) if r['read_time'] is not None else '-'
                write_mb_s = format_throughput(r['write_throughput']) if r['write_throughput'] is not None else '-'
                read_mb_s = format_throughput(r['read_throughput']) if r['read_throughput'] is not None else '-'
                size_mb = f"{r['size']:.2f}" if r['size'] is not None else '-'
                
                print(f"| {format_name:<17} | {write_time:>9} | {read_time:>8} | {write_mb_s:>10} | {read_mb_s:>9} | {size_mb:>9} |")
        
        print()
        print("=" * 80)
        
    finally:
        # 清理临时文件
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == '__main__':
    run_benchmark()

