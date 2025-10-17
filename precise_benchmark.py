#!/usr/bin/env python3
"""
精确的性能基准测试 - 使用timeit方法，排除文件打开关闭的开销
"""

import os
import sys
import time
import timeit
import tempfile
import shutil
import gc
import numpy as np

from numpack import NumPack

# 导入其他格式库
try:
    import h5py
    HAS_HDF5 = True
except ImportError:
    HAS_HDF5 = False

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False

try:
    import zarr
    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False

import pickle


def format_time_us(seconds):
    """格式化时间为微秒"""
    return f"{seconds*1000000:.1f}µs"


def run_precise_benchmark(size_name, shape):
    """运行精确的benchmark测试"""
    print(f"\n{'='*80}")
    print(f"{size_name} - Shape: {shape}")
    print(f"{'='*80}")
    
    test_data = np.random.rand(*shape).astype(np.float32)
    temp_dir = tempfile.mkdtemp()
    
    results = {}
    
    try:
        # 准备文件
        numpack_dir = os.path.join(temp_dir, 'numpack')
        npy_file = os.path.join(temp_dir, 'test.npy')
        npz_file = os.path.join(temp_dir, 'test.npz')
        pkl_file = os.path.join(temp_dir, 'test.pkl')
        
        # 保存数据
        with NumPack(numpack_dir, drop_if_exists=True) as npk:
            npk.save({'data': test_data})
        np.save(npy_file, test_data)
        np.savez(npz_file, data=test_data)
        with open(pkl_file, 'wb') as f:
            pickle.dump(test_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # === NumPack测试 ===
        print("\n--- NumPack ---")
        with NumPack(numpack_dir) as npk:
            # Save (在新文件中测试)
            save_time = timeit.timeit(
                lambda: (npk.save({'temp': test_data}), npk.drop('temp'))[1],
                number=10
            ) / 10
            
            # Load
            load_time = timeit.timeit(
                lambda: npk.load('data'),
                number=100
            ) / 100
            
            # Lazy Load (关键!)
            lazy_load_time = timeit.timeit(
                lambda: npk.load('data', lazy=True),
                number=1000
            ) / 1000
            
            # GetItem
            lazy = npk.load('data', lazy=True)
            getitem_single_time = timeit.timeit(
                lambda: lazy[0],
                number=1000
            ) / 1000
            
            slice_size = min(100, shape[0])
            getitem_slice_time = timeit.timeit(
                lambda: lazy[:slice_size],
                number=1000
            ) / 1000
            
            # Replace
            replace_size = min(100, shape[0])
            replace_data = np.random.rand(replace_size, *shape[1:]).astype(np.float32)
            replace_time = timeit.timeit(
                lambda: npk.replace({'data': replace_data}, list(range(replace_size))),
                number=10
            ) / 10
            
            # Append
            append_data = np.random.rand(replace_size, *shape[1:]).astype(np.float32)
            
            def append_test():
                npk.append({'data': append_data})
                # 删除追加的数据以重置
                current_rows = npk.get_shape('data')[0]
                npk.drop('data', list(range(current_rows - replace_size, current_rows)))
            
            append_time = timeit.timeit(append_test, number=10) / 10
            
        print(f"  Save:         {format_time_us(save_time)}")
        print(f"  Load:         {format_time_us(load_time)}")
        print(f"  Lazy Load:    {format_time_us(lazy_load_time)}")
        print(f"  GetItem[0]:   {format_time_us(getitem_single_time)}")
        print(f"  GetItem[:100]:{format_time_us(getitem_slice_time)}")
        print(f"  Replace:      {format_time_us(replace_time)}")
        print(f"  Append:       {format_time_us(append_time)}")
        
        results['numpack'] = {
            'lazy_load': lazy_load_time,
            'load': load_time,
            'getitem_single': getitem_single_time,
            'replace': replace_time,
            'append': append_time,
        }
        
        # === NPY测试 ===
        print("\n--- NPY ---")
        
        # Load
        load_time = timeit.timeit(
            lambda: np.load(npy_file),
            number=100
        ) / 100
        
        # Lazy Load (mmap) - 重复打开文件
        lazy_load_time = timeit.timeit(
            lambda: np.load(npy_file, mmap_mode='r'),
            number=1000
        ) / 1000
        
        # GetItem (使用已打开的mmap)
        mmap = np.load(npy_file, mmap_mode='r')
        getitem_single_time = timeit.timeit(
            lambda: mmap[0],
            number=1000
        ) / 1000
        
        getitem_slice_time = timeit.timeit(
            lambda: mmap[:100],
            number=1000
        ) / 1000
        
        # Replace (需要load-modify-save)
        def replace_npy():
            data = np.load(npy_file)
            data[:100] = replace_data
            np.save(npy_file, data)
        
        replace_time = timeit.timeit(replace_npy, number=10) / 10
        
        # Append
        def append_npy():
            data = np.load(npy_file)
            new_data = np.vstack([data, append_data])
            np.save(npy_file, new_data)
            # 恢复
            np.save(npy_file, test_data)
        
        append_time = timeit.timeit(append_npy, number=10) / 10
        
        print(f"  Load:         {format_time_us(load_time)}")
        print(f"  Lazy Load:    {format_time_us(lazy_load_time)}")
        print(f"  GetItem[0]:   {format_time_us(getitem_single_time)}")
        print(f"  GetItem[:100]:{format_time_us(getitem_slice_time)}")
        print(f"  Replace:      {format_time_us(replace_time)}")
        print(f"  Append:       {format_time_us(append_time)}")
        
        results['npy'] = {
            'lazy_load': lazy_load_time,
            'load': load_time,
            'getitem_single': getitem_single_time,
            'replace': replace_time,
            'append': append_time,
        }
        
        # === NPZ测试 ===
        print("\n--- NPZ ---")
        
        # Load
        def load_npz():
            with np.load(npz_file) as npz:
                return npz['data'][:]
        
        load_time = timeit.timeit(load_npz, number=100) / 100
        
        print(f"  Load:         {format_time_us(load_time)}")
        print(f"  Lazy Load:    N/A (不支持mmap)")
        
        # Replace
        def replace_npz():
            with np.load(npz_file) as npz:
                data = npz['data'].copy()
            data[:100] = replace_data
            np.savez(npz_file, data=data)
        
        replace_time = timeit.timeit(replace_npz, number=10) / 10
        
        # Append
        def append_npz():
            with np.load(npz_file) as npz:
                data = npz['data'].copy()
            new_data = np.vstack([data, append_data])
            np.savez(npz_file, data=new_data)
            # 恢复
            np.savez(npz_file, data=test_data)
        
        append_time = timeit.timeit(append_npz, number=10) / 10
        
        print(f"  Replace:      {format_time_us(replace_time)}")
        print(f"  Append:       {format_time_us(append_time)}")
        
        results['npz'] = {
            'load': load_time,
            'replace': replace_time,
            'append': append_time,
        }
        
        # === 性能对比 ===
        print(f"\n{'='*80}")
        print("性能对比 (NumPack vs NPY)")
        print(f"{'='*80}")
        
        if 'lazy_load' in results['numpack'] and 'lazy_load' in results['npy']:
            ratio = results['npy']['lazy_load'] / results['numpack']['lazy_load']
            print(f"Lazy Load:  NumPack {format_time_us(results['numpack']['lazy_load'])} vs NPY {format_time_us(results['npy']['lazy_load'])} = {ratio:.1f}x")
        
        if 'load' in results['numpack'] and 'load' in results['npy']:
            ratio = results['npy']['load'] / results['numpack']['load']
            print(f"Load:       NumPack {format_time_us(results['numpack']['load'])} vs NPY {format_time_us(results['npy']['load'])} = {ratio:.2f}x")
        
        if 'replace' in results['numpack'] and 'replace' in results['npy']:
            ratio = results['npy']['replace'] / results['numpack']['replace']
            print(f"Replace:    NumPack {format_time_us(results['numpack']['replace'])} vs NPY {format_time_us(results['npy']['replace'])} = {ratio:.1f}x")
        
        if 'append' in results['numpack'] and 'append' in results['npy']:
            ratio = results['npy']['append'] / results['numpack']['append']
            print(f"Append:     NumPack {format_time_us(results['numpack']['append'])} vs NPY {format_time_us(results['npy']['append'])} = {ratio:.1f}x")
        
        print(f"{'='*80}")
        
    finally:
        shutil.rmtree(temp_dir)


def main():
    """主函数"""
    print("精确的NumPack性能基准测试")
    print("使用timeit方法，排除文件I/O开销")
    print("="*80)
    
    run_precise_benchmark("大数据集 (1M行)", (1000000, 10))
    run_precise_benchmark("中数据集 (100K行)", (100000, 10))
    
    print("\n" + "="*80)
    print("测试完成!")
    print("="*80)


if __name__ == "__main__":
    main()

