#!/usr/bin/env python3
"""
NumPack 全面性能基准测试
包含drop操作和多库横向对比
"""

import os
import sys
import time
import timeit
import tempfile
import shutil
import gc
import numpy as np
from pathlib import Path
import warnings

from numpack import NumPack

# 导入其他格式库
try:
    import h5py
    HAS_HDF5 = True
except ImportError:
    HAS_HDF5 = False
    print("警告: h5py未安装，跳过HDF5测试")

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False
    print("警告: pyarrow未安装，跳过Parquet测试")

try:
    import zarr
    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False
    print("警告: zarr未安装，跳过Zarr测试")

import pickle


def format_time(seconds):
    """格式化时间显示"""
    if seconds < 1e-6:
        return f"{seconds*1e9:.1f}ns"
    elif seconds < 1e-3:
        return f"{seconds*1e6:.1f}µs"
    elif seconds < 1:
        return f"{seconds*1e3:.1f}ms"
    else:
        return f"{seconds:.2f}s"


def format_speedup(ratio):
    """格式化加速比"""
    if ratio >= 1:
        return f"{ratio:.1f}x faster"
    else:
        return f"{1/ratio:.1f}x slower"


class BenchmarkRunner:
    """Benchmark运行器"""
    
    def __init__(self, size_name, shape, repeat=10):
        self.size_name = size_name
        self.shape = shape
        self.repeat = repeat
        self.test_data = np.random.rand(*shape).astype(np.float32)
        self.temp_dir = tempfile.mkdtemp()
        self.results = {}
        
    def cleanup(self):
        """清理临时文件"""
        shutil.rmtree(self.temp_dir)
    
    def run_all(self):
        """运行所有benchmark测试"""
        print(f"\n{'='*90}")
        print(f"{self.size_name} - Shape: {self.shape} - Size: {self.test_data.nbytes / 1024 / 1024:.1f}MB")
        print(f"{'='*90}")
        
        self.benchmark_numpack()
        self.benchmark_npy()
        self.benchmark_npz()
        
        if HAS_HDF5:
            self.benchmark_hdf5()
        
        if HAS_ZARR:
            self.benchmark_zarr()
            
        if HAS_PARQUET:
            self.benchmark_parquet()
        
        self.print_comparison()
    
    def benchmark_numpack(self):
        """NumPack性能测试"""
        print("\n--- NumPack (Rust Backend) ---")
        
        numpack_dir = os.path.join(self.temp_dir, 'numpack')
        results = {}
        
        with NumPack(numpack_dir, drop_if_exists=True) as npk:
            # Save操作
            save_time = timeit.timeit(
                lambda: (npk.save({'temp': self.test_data}), npk.drop('temp'))[1],
                number=self.repeat
            ) / self.repeat
            results['save'] = save_time
            
            # 准备测试数据
            npk.save({'data': self.test_data})
            
            # Load操作
            load_time = timeit.timeit(
                lambda: npk.load('data'),
                number=self.repeat * 10
            ) / (self.repeat * 10)
            results['load'] = load_time
            
            # Lazy Load操作
            lazy_load_time = timeit.timeit(
                lambda: npk.load('data', lazy=True),
                number=self.repeat * 100
            ) / (self.repeat * 100)
            results['lazy_load'] = lazy_load_time
            
            # GetItem操作
            lazy = npk.load('data', lazy=True)
            getitem_single_time = timeit.timeit(
                lambda: lazy[0],
                number=self.repeat * 100
            ) / (self.repeat * 100)
            results['getitem_single'] = getitem_single_time
            
            slice_size = min(100, self.shape[0])
            getitem_slice_time = timeit.timeit(
                lambda: lazy[:slice_size],
                number=self.repeat * 100
            ) / (self.repeat * 100)
            results['getitem_slice'] = getitem_slice_time
            
            # Replace操作
            replace_size = min(100, self.shape[0])
            replace_data = np.random.rand(replace_size, *self.shape[1:]).astype(np.float32)
            replace_time = timeit.timeit(
                lambda: npk.replace({'data': replace_data}, list(range(replace_size))),
                number=self.repeat
            ) / self.repeat
            results['replace'] = replace_time
            
            # Append操作
            append_data = np.random.rand(replace_size, *self.shape[1:]).astype(np.float32)
            
            total_time = 0
            for _ in range(self.repeat):
                start = time.time()
                npk.append({'data': append_data})
                total_time += time.time() - start
                # 恢复（不计时）- 删除刚追加的行
                current_rows = npk.get_shape('data')[0]
                npk.drop('data', list(range(current_rows - replace_size, current_rows)))
            
            append_time = total_time / self.repeat
            results['append'] = append_time
            
            # Drop操作 - 单行
            total_time = 0
            for _ in range(self.repeat):
                # 准备（不计时）- 追加一行以便删除
                npk.append({'data': append_data[:1]})
                start = time.time()
                npk.drop('data', [npk.get_shape('data')[0] - 1])
                total_time += time.time() - start
            
            drop_single_time = total_time / self.repeat
            results['drop_single'] = drop_single_time
            
            # Drop操作 - 多行
            drop_count = min(100, self.shape[0] // 10)
            total_time = 0
            for _ in range(self.repeat):
                # 准备（不计时）- 先append一些数据
                npk.append({'data': np.random.rand(drop_count, *self.shape[1:]).astype(np.float32)})
                current_rows = npk.get_shape('data')[0]
                start = time.time()
                npk.drop('data', list(range(current_rows - drop_count, current_rows)))
                total_time += time.time() - start
            
            drop_multi_time = total_time / self.repeat
            results['drop_multi'] = drop_multi_time
            
            # Drop操作 - 整个数组
            total_time = 0
            for _ in range(self.repeat):
                # 准备（不计时）- 创建临时数组
                npk.save({'temp_array': self.test_data})
                start = time.time()
                npk.drop('temp_array')
                total_time += time.time() - start
            
            drop_array_time = total_time / self.repeat
            results['drop_array'] = drop_array_time
        
        self.results['numpack'] = results
        
        print(f"  Save:            {format_time(results['save'])}")
        print(f"  Load:            {format_time(results['load'])}")
        print(f"  Lazy Load:       {format_time(results['lazy_load'])}")
        print(f"  GetItem[0]:      {format_time(results['getitem_single'])}")
        print(f"  GetItem[:100]:   {format_time(results['getitem_slice'])}")
        print(f"  Replace 100:     {format_time(results['replace'])}")
        print(f"  Append 100:      {format_time(results['append'])}")
        print(f"  Drop Single:     {format_time(results['drop_single'])}")
        print(f"  Drop {drop_count} rows:    {format_time(results['drop_multi'])}")
        print(f"  Drop Array:      {format_time(results['drop_array'])}")
    
    def benchmark_npy(self):
        """NPY格式性能测试"""
        print("\n--- NPY (NumPy Binary) ---")
        
        npy_file = os.path.join(self.temp_dir, 'test.npy')
        results = {}
        
        # Save操作
        save_time = timeit.timeit(
            lambda: np.save(npy_file, self.test_data),
            number=self.repeat
        ) / self.repeat
        results['save'] = save_time
        
        # 准备测试数据
        np.save(npy_file, self.test_data)
        
        # Load操作
        load_time = timeit.timeit(
            lambda: np.load(npy_file),
            number=self.repeat * 10
        ) / (self.repeat * 10)
        results['load'] = load_time
        
        # Lazy Load (mmap)
        lazy_load_time = timeit.timeit(
            lambda: np.load(npy_file, mmap_mode='r'),
            number=self.repeat * 100
        ) / (self.repeat * 100)
        results['lazy_load'] = lazy_load_time
        
        # GetItem操作
        mmap = np.load(npy_file, mmap_mode='r')
        getitem_single_time = timeit.timeit(
            lambda: mmap[0],
            number=self.repeat * 100
        ) / (self.repeat * 100)
        results['getitem_single'] = getitem_single_time
        
        slice_size = min(100, self.shape[0])
        getitem_slice_time = timeit.timeit(
            lambda: mmap[:slice_size],
            number=self.repeat * 100
        ) / (self.repeat * 100)
        results['getitem_slice'] = getitem_slice_time
        
        # Replace操作 - 需要load-modify-save
        replace_size = min(100, self.shape[0])
        replace_data = np.random.rand(replace_size, *self.shape[1:]).astype(np.float32)
        def replace_npy():
            data = np.load(npy_file)
            data[:replace_size] = replace_data
            np.save(npy_file, data)
        
        replace_time = timeit.timeit(replace_npy, number=self.repeat) / self.repeat
        results['replace'] = replace_time
        
        # Append操作
        append_data = np.random.rand(replace_size, *self.shape[1:]).astype(np.float32)
        
        total_time = 0
        for _ in range(self.repeat):
            start = time.time()
            data = np.load(npy_file)
            new_data = np.vstack([data, append_data])
            np.save(npy_file, new_data)
            total_time += time.time() - start
            # 恢复（不计时）
            np.save(npy_file, self.test_data)
        
        append_time = total_time / self.repeat
        results['append'] = append_time
        
        # Drop单行 - 需要load-delete-save
        total_time = 0
        for _ in range(self.repeat):
            start = time.time()
            data = np.load(npy_file)
            new_data = np.delete(data, -1, axis=0)
            np.save(npy_file, new_data)
            total_time += time.time() - start
            # 恢复（不计时）
            np.save(npy_file, self.test_data)
        
        drop_single_time = total_time / self.repeat
        results['drop_single'] = drop_single_time
        
        # Drop多行
        drop_count = min(100, self.shape[0] // 10)
        total_time = 0
        for _ in range(self.repeat):
            start = time.time()
            data = np.load(npy_file)
            new_data = np.delete(data, range(len(data) - drop_count, len(data)), axis=0)
            np.save(npy_file, new_data)
            total_time += time.time() - start
            # 恢复（不计时）
            np.save(npy_file, self.test_data)
        
        drop_multi_time = total_time / self.repeat
        results['drop_multi'] = drop_multi_time
        
        # Drop整个数组 (删除文件)
        total_time = 0
        for _ in range(self.repeat):
            start = time.time()
            if os.path.exists(npy_file):
                os.remove(npy_file)
            total_time += time.time() - start
            # 恢复（不计时）
            np.save(npy_file, self.test_data)
        
        drop_array_time = total_time / self.repeat
        results['drop_array'] = drop_array_time
        
        self.results['npy'] = results
        
        print(f"  Save:            {format_time(results['save'])}")
        print(f"  Load:            {format_time(results['load'])}")
        print(f"  Lazy Load:       {format_time(results['lazy_load'])}")
        print(f"  GetItem[0]:      {format_time(results['getitem_single'])}")
        print(f"  GetItem[:100]:   {format_time(results['getitem_slice'])}")
        print(f"  Replace 100:     {format_time(results['replace'])}")
        print(f"  Append 100:      {format_time(results['append'])}")
        print(f"  Drop Single:     {format_time(results['drop_single'])}")
        print(f"  Drop {drop_count} rows:    {format_time(results['drop_multi'])}")
        print(f"  Drop Array:      {format_time(results['drop_array'])}")
    
    def benchmark_npz(self):
        """NPZ格式性能测试"""
        print("\n--- NPZ (NumPy Compressed) ---")
        
        npz_file = os.path.join(self.temp_dir, 'test.npz')
        results = {}
        
        # Save操作
        save_time = timeit.timeit(
            lambda: np.savez(npz_file, data=self.test_data),
            number=self.repeat
        ) / self.repeat
        results['save'] = save_time
        
        # 准备测试数据
        np.savez(npz_file, data=self.test_data)
        
        # Load操作
        def load_npz():
            with np.load(npz_file) as npz:
                return npz['data'][:]
        
        load_time = timeit.timeit(load_npz, number=self.repeat * 10) / (self.repeat * 10)
        results['load'] = load_time
        
        # Replace操作
        replace_size = min(100, self.shape[0])
        replace_data = np.random.rand(replace_size, *self.shape[1:]).astype(np.float32)
        def replace_npz():
            with np.load(npz_file) as npz:
                data = npz['data'].copy()
            data[:replace_size] = replace_data
            np.savez(npz_file, data=data)
        
        replace_time = timeit.timeit(replace_npz, number=self.repeat) / self.repeat
        results['replace'] = replace_time
        
        # Append操作
        append_data = np.random.rand(replace_size, *self.shape[1:]).astype(np.float32)
        
        total_time = 0
        for _ in range(self.repeat):
            start = time.time()
            with np.load(npz_file) as npz:
                data = npz['data'].copy()
            new_data = np.vstack([data, append_data])
            np.savez(npz_file, data=new_data)
            total_time += time.time() - start
            # 恢复（不计时）
            np.savez(npz_file, data=self.test_data)
        
        append_time = total_time / self.repeat
        results['append'] = append_time
        
        # Drop单行
        total_time = 0
        for _ in range(self.repeat):
            start = time.time()
            with np.load(npz_file) as npz:
                data = npz['data'].copy()
            new_data = np.delete(data, -1, axis=0)
            np.savez(npz_file, data=new_data)
            total_time += time.time() - start
            # 恢复（不计时）
            np.savez(npz_file, data=self.test_data)
        
        drop_single_time = total_time / self.repeat
        results['drop_single'] = drop_single_time
        
        # Drop多行
        drop_count = min(100, self.shape[0] // 10)
        total_time = 0
        for _ in range(self.repeat):
            start = time.time()
            with np.load(npz_file) as npz:
                data = npz['data'].copy()
            new_data = np.delete(data, range(len(data) - drop_count, len(data)), axis=0)
            np.savez(npz_file, data=new_data)
            total_time += time.time() - start
            # 恢复（不计时）
            np.savez(npz_file, data=self.test_data)
        
        drop_multi_time = total_time / self.repeat
        results['drop_multi'] = drop_multi_time
        
        self.results['npz'] = results
        
        print(f"  Save:            {format_time(results['save'])}")
        print(f"  Load:            {format_time(results['load'])}")
        print(f"  Lazy Load:       N/A (压缩格式)")
        print(f"  Replace 100:     {format_time(results['replace'])}")
        print(f"  Append 100:      {format_time(results['append'])}")
        print(f"  Drop Single:     {format_time(results['drop_single'])}")
        print(f"  Drop {drop_count} rows:    {format_time(results['drop_multi'])}")
    
    def benchmark_hdf5(self):
        """HDF5格式性能测试"""
        print("\n--- HDF5 ---")
        
        hdf5_file = os.path.join(self.temp_dir, 'test.h5')
        results = {}
        
        # Save操作
        def save_hdf5():
            with h5py.File(hdf5_file, 'w') as f:
                f.create_dataset('data', data=self.test_data)
        
        save_time = timeit.timeit(save_hdf5, number=self.repeat) / self.repeat
        results['save'] = save_time
        
        # 准备测试数据
        with h5py.File(hdf5_file, 'w') as f:
            f.create_dataset('data', data=self.test_data)
        
        # Load操作
        def load_hdf5():
            with h5py.File(hdf5_file, 'r') as f:
                return f['data'][:]
        
        load_time = timeit.timeit(load_hdf5, number=self.repeat * 10) / (self.repeat * 10)
        results['load'] = load_time
        
        # Lazy load (打开文件句柄)
        def lazy_load_hdf5():
            f = h5py.File(hdf5_file, 'r')
            ds = f['data']
            f.close()
        
        lazy_load_time = timeit.timeit(lazy_load_hdf5, number=self.repeat * 100) / (self.repeat * 100)
        results['lazy_load'] = lazy_load_time
        
        # GetItem操作
        with h5py.File(hdf5_file, 'r') as f:
            ds = f['data']
            getitem_single_time = timeit.timeit(
                lambda: ds[0],
                number=self.repeat * 100
            ) / (self.repeat * 100)
            results['getitem_single'] = getitem_single_time
            
            slice_size = min(100, self.shape[0])
            getitem_slice_time = timeit.timeit(
                lambda: ds[:slice_size],
                number=self.repeat * 100
            ) / (self.repeat * 100)
            results['getitem_slice'] = getitem_slice_time
        
        # Replace操作
        replace_size = min(100, self.shape[0])
        replace_data = np.random.rand(replace_size, *self.shape[1:]).astype(np.float32)
        def replace_hdf5():
            with h5py.File(hdf5_file, 'r+') as f:
                f['data'][:replace_size] = replace_data
        
        replace_time = timeit.timeit(replace_hdf5, number=self.repeat) / self.repeat
        results['replace'] = replace_time
        
        # Append操作 - HDF5需要resize
        append_data = np.random.rand(replace_size, *self.shape[1:]).astype(np.float32)
        
        # 需要创建可调整大小的数据集
        with h5py.File(hdf5_file, 'w') as f:
            f.create_dataset('data', data=self.test_data, maxshape=(None,) + self.shape[1:])
        
        total_time = 0
        for _ in range(self.repeat):
            start = time.time()
            with h5py.File(hdf5_file, 'r+') as f:
                old_shape = f['data'].shape
                f['data'].resize((old_shape[0] + replace_size,) + old_shape[1:])
                f['data'][old_shape[0]:] = append_data
            total_time += time.time() - start
            # 恢复（不计时）
            with h5py.File(hdf5_file, 'r+') as f:
                f['data'].resize(self.test_data.shape)
        
        append_time = total_time / self.repeat
        results['append'] = append_time
        
        # Drop单行
        total_time = 0
        for _ in range(self.repeat):
            start = time.time()
            with h5py.File(hdf5_file, 'r+') as f:
                old_shape = f['data'].shape
                f['data'].resize((old_shape[0] - 1,) + old_shape[1:])
            total_time += time.time() - start
            # 恢复（不计时）
            with h5py.File(hdf5_file, 'r+') as f:
                f['data'].resize(self.test_data.shape)
        
        drop_single_time = total_time / self.repeat
        results['drop_single'] = drop_single_time
        
        # Drop多行
        drop_count = min(100, self.shape[0] // 10)
        total_time = 0
        for _ in range(self.repeat):
            start = time.time()
            with h5py.File(hdf5_file, 'r+') as f:
                old_shape = f['data'].shape
                f['data'].resize((old_shape[0] - drop_count,) + old_shape[1:])
            total_time += time.time() - start
            # 恢复（不计时）
            with h5py.File(hdf5_file, 'r+') as f:
                f['data'].resize(self.test_data.shape)
        
        drop_multi_time = total_time / self.repeat
        results['drop_multi'] = drop_multi_time
        
        self.results['hdf5'] = results
        
        print(f"  Save:            {format_time(results['save'])}")
        print(f"  Load:            {format_time(results['load'])}")
        print(f"  Lazy Load:       {format_time(results['lazy_load'])}")
        print(f"  GetItem[0]:      {format_time(results['getitem_single'])}")
        print(f"  GetItem[:100]:   {format_time(results['getitem_slice'])}")
        print(f"  Replace 100:     {format_time(results['replace'])}")
        print(f"  Append 100:      {format_time(results['append'])}")
        print(f"  Drop Single:     {format_time(results['drop_single'])}")
        print(f"  Drop {drop_count} rows:    {format_time(results['drop_multi'])}")
    
    def benchmark_zarr(self):
        """Zarr格式性能测试"""
        print("\n--- Zarr ---")
        
        zarr_dir = os.path.join(self.temp_dir, 'zarr_data')
        results = {}
        
        # Save操作
        def save_zarr():
            z = zarr.open(zarr_dir, mode='w', shape=self.shape, 
                         chunks=(min(1000, self.shape[0]),) + self.shape[1:],
                         dtype='f4')
            z[:] = self.test_data
        
        save_time = timeit.timeit(save_zarr, number=self.repeat) / self.repeat
        results['save'] = save_time
        
        # 准备测试数据
        z = zarr.open(zarr_dir, mode='w', shape=self.shape, 
                     chunks=(min(1000, self.shape[0]),) + self.shape[1:],
                     dtype='f4')
        z[:] = self.test_data
        
        # Load操作
        def load_zarr():
            z = zarr.open(zarr_dir, mode='r')
            return z[:]
        
        load_time = timeit.timeit(load_zarr, number=self.repeat * 10) / (self.repeat * 10)
        results['load'] = load_time
        
        # Lazy load (打开句柄)
        lazy_load_time = timeit.timeit(
            lambda: zarr.open(zarr_dir, mode='r'),
            number=self.repeat * 100
        ) / (self.repeat * 100)
        results['lazy_load'] = lazy_load_time
        
        # GetItem操作
        z = zarr.open(zarr_dir, mode='r')
        getitem_single_time = timeit.timeit(
            lambda: z[0],
            number=self.repeat * 100
        ) / (self.repeat * 100)
        results['getitem_single'] = getitem_single_time
        
        slice_size = min(100, self.shape[0])
        getitem_slice_time = timeit.timeit(
            lambda: z[:slice_size],
            number=self.repeat * 100
        ) / (self.repeat * 100)
        results['getitem_slice'] = getitem_slice_time
        
        # Replace操作
        replace_size = min(100, self.shape[0])
        replace_data = np.random.rand(replace_size, *self.shape[1:]).astype(np.float32)
        def replace_zarr():
            z = zarr.open(zarr_dir, mode='r+')
            z[:replace_size] = replace_data
        
        replace_time = timeit.timeit(replace_zarr, number=self.repeat) / self.repeat
        results['replace'] = replace_time
        
        # Append操作
        append_data = np.random.rand(replace_size, *self.shape[1:]).astype(np.float32)
        
        total_time = 0
        for _ in range(self.repeat):
            start = time.time()
            z = zarr.open(zarr_dir, mode='r+')
            z.append(append_data)
            total_time += time.time() - start
            # 恢复（不计时）
            z.resize(self.shape)
        
        append_time = total_time / self.repeat
        results['append'] = append_time
        
        # Drop单行
        total_time = 0
        for _ in range(self.repeat):
            start = time.time()
            z = zarr.open(zarr_dir, mode='r+')
            z.resize((z.shape[0] - 1,) + z.shape[1:])
            total_time += time.time() - start
            # 恢复（不计时）
            z.resize(self.shape)
        
        drop_single_time = total_time / self.repeat
        results['drop_single'] = drop_single_time
        
        # Drop多行
        drop_count = min(100, self.shape[0] // 10)
        total_time = 0
        for _ in range(self.repeat):
            start = time.time()
            z = zarr.open(zarr_dir, mode='r+')
            z.resize((z.shape[0] - drop_count,) + z.shape[1:])
            total_time += time.time() - start
            # 恢复（不计时）
            z.resize(self.shape)
        
        drop_multi_time = total_time / self.repeat
        results['drop_multi'] = drop_multi_time
        
        self.results['zarr'] = results
        
        print(f"  Save:            {format_time(results['save'])}")
        print(f"  Load:            {format_time(results['load'])}")
        print(f"  Lazy Load:       {format_time(results['lazy_load'])}")
        print(f"  GetItem[0]:      {format_time(results['getitem_single'])}")
        print(f"  GetItem[:100]:   {format_time(results['getitem_slice'])}")
        print(f"  Replace 100:     {format_time(results['replace'])}")
        print(f"  Append 100:      {format_time(results['append'])}")
        print(f"  Drop Single:     {format_time(results['drop_single'])}")
        print(f"  Drop {drop_count} rows:    {format_time(results['drop_multi'])}")
    
    def benchmark_parquet(self):
        """Parquet格式性能测试"""
        print("\n--- Parquet (PyArrow) ---")
        
        parquet_file = os.path.join(self.temp_dir, 'test.parquet')
        results = {}
        
        # 将数据转换为表格格式
        table_data = {f'col_{i}': self.test_data[:, i] for i in range(self.test_data.shape[1])}
        table = pa.table(table_data)
        
        # Save操作
        def save_parquet():
            pq.write_table(table, parquet_file)
        
        save_time = timeit.timeit(save_parquet, number=self.repeat) / self.repeat
        results['save'] = save_time
        
        # 准备测试数据
        pq.write_table(table, parquet_file)
        
        # Load操作
        def load_parquet():
            table = pq.read_table(parquet_file)
            return table.to_pandas().values
        
        load_time = timeit.timeit(load_parquet, number=self.repeat * 10) / (self.repeat * 10)
        results['load'] = load_time
        
        # Parquet不支持真正的lazy load和in-place修改
        print(f"  Save:            {format_time(results['save'])}")
        print(f"  Load:            {format_time(results['load'])}")
        print(f"  Lazy Load:       N/A (不支持)")
        print(f"  Replace:         N/A (不支持in-place修改)")
        print(f"  Append:          N/A (需要重写整个文件)")
        print(f"  Drop:            N/A (需要重写整个文件)")
        
        self.results['parquet'] = results
    
    def print_comparison(self):
        """打印性能对比"""
        print(f"\n{'='*90}")
        print("性能对比总结 (NumPack vs 其他库)")
        print(f"{'='*90}")
        
        operations = ['save', 'load', 'lazy_load', 'getitem_single', 'getitem_slice',
                     'replace', 'append', 'drop_single', 'drop_multi', 'drop_array']
        
        operation_names = {
            'save': 'Save',
            'load': 'Load',
            'lazy_load': 'Lazy Load',
            'getitem_single': 'GetItem[0]',
            'getitem_slice': 'GetItem[:100]',
            'replace': 'Replace 100',
            'append': 'Append 100',
            'drop_single': 'Drop Single Row',
            'drop_multi': f'Drop {min(100, self.shape[0]//10)} Rows',
            'drop_array': 'Drop Array'
        }
        
        numpack_results = self.results.get('numpack', {})
        
        for op in operations:
            if op not in numpack_results:
                continue
            
            print(f"\n{operation_names[op]}:")
            print(f"  NumPack:  {format_time(numpack_results[op])}", end='')
            
            comparisons = []
            for lib in ['npy', 'npz', 'hdf5', 'zarr']:
                if lib in self.results and op in self.results[lib]:
                    lib_time = self.results[lib][op]
                    ratio = lib_time / numpack_results[op]
                    comparisons.append(f"{lib.upper()}: {format_time(lib_time)} ({format_speedup(ratio)})")
            
            if comparisons:
                print(" | " + " | ".join(comparisons))
            else:
                print()
        
        print(f"\n{'='*90}")


def main():
    """主函数"""
    print("="*90)
    print("NumPack 全面性能基准测试")
    print("包含Drop操作和多库横向对比")
    print("="*90)
    
    # 测试配置
    test_configs = [
        ("超大数据集 (10M rows)", (10000000, 10), 5),   # 减少重复次数
        ("大数据集 (1M rows)", (1000000, 10), 10),
        ("中数据集 (100K rows)", (100000, 10), 10),
        ("小数据集 (10K rows)", (10000, 10), 10),
    ]
    
    for size_name, shape, repeat in test_configs:
        runner = BenchmarkRunner(size_name, shape, repeat=repeat)
        try:
            runner.run_all()
        finally:
            runner.cleanup()
        
        print("\n")
        gc.collect()
    
    print("="*90)
    print("测试完成!")
    print("="*90)


if __name__ == "__main__":
    main()

