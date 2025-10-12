#!/usr/bin/env python3
"""
README 基准测试脚本 - 多次运行取平均值
对比 NumPack 与多种格式的性能

测试数据: 1,000,000 × 128 float32 数组
运行3次取平均值以确保结果稳定性
"""

import os
import sys
import time
import tempfile
import shutil
import gc
import numpy as np
from collections import defaultdict

# 添加 numpack 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

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

def run_single_benchmark():
    """运行单次基准测试"""
    # 创建测试数据
    rows = 1_000_000
    cols = 128
    array = np.random.rand(rows, cols).astype(np.float32)
    data_size_mb = array.nbytes / 1024 / 1024
    
    temp_dir = tempfile.mkdtemp()
    results = {}
    
    try:
        # ========== NumPack (Rust backend) ==========
        from numpack import NumPack
        
        numpack_dir = os.path.join(temp_dir, 'numpack_test')
        npk = NumPack(numpack_dir, drop_if_exists=True)
        npk.open()
        
        gc.collect()
        start_time = time.perf_counter()
        npk.save({'array': array})
        write_time = time.perf_counter() - start_time
        
        numpack_size = get_file_size_mb(numpack_dir)
        
        gc.collect()
        start_time = time.perf_counter()
        loaded = npk.load('array')
        read_time = time.perf_counter() - start_time
        
        gc.collect()
        start_time = time.perf_counter()
        lazy_loaded = npk.load('array', lazy=True)
        lazy_time = time.perf_counter() - start_time
        
        npk.close()
        
        results['NumPack (Rust)'] = {
            'write_time': write_time,
            'read_time': read_time,
            'size': numpack_size,
            'data_size_mb': data_size_mb
        }
        
        results['NumPack Lazy Load'] = {
            'read_time': lazy_time,
            'data_size_mb': data_size_mb
        }
        
        # ========== NumPy .npy ==========
        npy_file = os.path.join(temp_dir, 'test.npy')
        
        gc.collect()
        start_time = time.perf_counter()
        np.save(npy_file, array)
        write_time = time.perf_counter() - start_time
        
        npy_size = get_file_size_mb(npy_file)
        
        gc.collect()
        start_time = time.perf_counter()
        loaded = np.load(npy_file)
        read_time = time.perf_counter() - start_time
        
        results['NumPy .npy'] = {
            'write_time': write_time,
            'read_time': read_time,
            'size': npy_size,
            'data_size_mb': data_size_mb
        }
        
        # ========== NumPy .npz ==========
        npz_file = os.path.join(temp_dir, 'test.npz')
        
        gc.collect()
        start_time = time.perf_counter()
        np.savez(npz_file, array=array)
        write_time = time.perf_counter() - start_time
        
        npz_size = get_file_size_mb(npz_file)
        
        gc.collect()
        start_time = time.perf_counter()
        with np.load(npz_file) as data:
            loaded = data['array'][:]
        read_time = time.perf_counter() - start_time
        
        results['NumPy .npz'] = {
            'write_time': write_time,
            'read_time': read_time,
            'size': npz_size,
            'data_size_mb': data_size_mb
        }
        
        # ========== Parquet ==========
        try:
            import pandas as pd
            
            parquet_file = os.path.join(temp_dir, 'test.parquet')
            df = pd.DataFrame(array)
            
            gc.collect()
            start_time = time.perf_counter()
            df.to_parquet(parquet_file)
            write_time = time.perf_counter() - start_time
            
            parquet_size = get_file_size_mb(parquet_file)
            
            gc.collect()
            start_time = time.perf_counter()
            loaded_df = pd.read_parquet(parquet_file)
            loaded = loaded_df.values
            read_time = time.perf_counter() - start_time
            
            results['Parquet'] = {
                'write_time': write_time,
                'read_time': read_time,
                'size': parquet_size,
                'data_size_mb': data_size_mb
            }
        except ImportError:
            pass
        
        # ========== HDF5 ==========
        try:
            import h5py
            
            hdf5_file = os.path.join(temp_dir, 'test.h5')
            
            gc.collect()
            start_time = time.perf_counter()
            with h5py.File(hdf5_file, 'w') as f:
                f.create_dataset('array', data=array)
            write_time = time.perf_counter() - start_time
            
            hdf5_size = get_file_size_mb(hdf5_file)
            
            gc.collect()
            start_time = time.perf_counter()
            with h5py.File(hdf5_file, 'r') as f:
                loaded = f['array'][:]
            read_time = time.perf_counter() - start_time
            
            results['HDF5'] = {
                'write_time': write_time,
                'read_time': read_time,
                'size': hdf5_size,
                'data_size_mb': data_size_mb
            }
        except ImportError:
            pass
        
        # ========== Pickle ==========
        import pickle
        
        pickle_file = os.path.join(temp_dir, 'test.pkl')
        
        gc.collect()
        start_time = time.perf_counter()
        with open(pickle_file, 'wb') as f:
            pickle.dump(array, f, protocol=pickle.HIGHEST_PROTOCOL)
        write_time = time.perf_counter() - start_time
        
        pickle_size = get_file_size_mb(pickle_file)
        
        gc.collect()
        start_time = time.perf_counter()
        with open(pickle_file, 'rb') as f:
            loaded = pickle.load(f)
        read_time = time.perf_counter() - start_time
        
        results['Pickle'] = {
            'write_time': write_time,
            'read_time': read_time,
            'size': pickle_size,
            'data_size_mb': data_size_mb
        }
        
        # ========== Zarr ==========
        try:
            import zarr
            
            zarr_dir = os.path.join(temp_dir, 'test.zarr')
            
            gc.collect()
            start_time = time.perf_counter()
            z = zarr.open(zarr_dir, mode='w', shape=array.shape, dtype=array.dtype, chunks=(10000, 128))
            z[:] = array
            write_time = time.perf_counter() - start_time
            
            zarr_size = get_file_size_mb(zarr_dir)
            
            gc.collect()
            start_time = time.perf_counter()
            z = zarr.open(zarr_dir, mode='r')
            loaded = z[:]
            read_time = time.perf_counter() - start_time
            
            results['Zarr'] = {
                'write_time': write_time,
                'read_time': read_time,
                'size': zarr_size,
                'data_size_mb': data_size_mb
            }
        except ImportError:
            pass
        
        return results
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    """运行多次测试并计算平均值"""
    print("=" * 80)
    print("README Benchmark: NumPack vs Other Formats (3次运行平均值)")
    print("Test Data: 1,000,000 × 128 float32 array")
    print("=" * 80)
    print()
    
    num_runs = 3
    all_results = []
    
    for i in range(num_runs):
        print(f"运行第 {i+1}/{num_runs} 次...")
        results = run_single_benchmark()
        all_results.append(results)
        print(f"  完成")
    
    # 计算平均值
    print("\n计算平均值...")
    averaged_results = {}
    
    # 获取所有格式
    all_formats = set()
    for results in all_results:
        all_formats.update(results.keys())
    
    for format_name in all_formats:
        format_results = [r[format_name] for r in all_results if format_name in r]
        if not format_results:
            continue
        
        avg_result = {}
        for key in format_results[0].keys():
            if key == 'data_size_mb':
                avg_result[key] = format_results[0][key]
            else:
                values = [r[key] for r in format_results]
                avg_result[key] = sum(values) / len(values)
        
        averaged_results[format_name] = avg_result
    
    # 生成结果表格
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS (平均值)")
    print("=" * 80)
    print()
    print("| Format            | Write (s) | Read (s) | Write MB/s | Read MB/s | Size (MB) |")
    print("|-------------------|-----------|----------|------------|-----------|-----------|")
    
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
        if format_name not in averaged_results:
            continue
        
        r = averaged_results[format_name]
        data_size_mb = r['data_size_mb']
        
        if 'write_time' in r:
            write_time = f"{r['write_time']:.3f}"
            write_mb_s = f"{int(data_size_mb / r['write_time']):,}"
        else:
            write_time = "-"
            write_mb_s = "-"
        
        if 'read_time' in r:
            read_time = f"{r['read_time']:.3f}"
            read_mb_s = f"{int(data_size_mb / r['read_time']):,}"
        else:
            read_time = "-"
            read_mb_s = "-"
        
        if 'size' in r:
            size_mb = f"{r['size']:.2f}"
        else:
            size_mb = "-"
        
        print(f"| {format_name:<17} | {write_time:>9} | {read_time:>8} | {write_mb_s:>10} | {read_mb_s:>9} | {size_mb:>9} |")
    
    print()
    print("=" * 80)

if __name__ == '__main__':
    main()

