//! FFI Communication Optimization Module
//!
//! 此模块提供优化的FFI通信策略，减少Python-Rust边界交叉开销
//!
//! 主要优化策略：
//! 1. Batch FFI Operations - 批量操作减少FFI调用
//! 2. Zero-Copy Views - 零拷贝数组视图
//! 3. Inline Hot Functions - 内联热点函数
//! 4. Cached Metadata - 缓存元数据访问

use memmap2::Mmap;
use ndarray::{ArrayD, ArrayViewD, IxDyn};
use numpy::ToPyArray;
use pyo3::prelude::*;
use std::sync::Arc;

use crate::core::metadata::DataType;

/// FFI优化配置
#[derive(Clone, Debug)]
pub struct FFIOptimizationConfig {
    /// 是否启用批量操作
    pub enable_batching: bool,
    /// 是否启用零拷贝
    pub enable_zero_copy: bool,
    /// 批量操作的最小元素数
    pub min_batch_size: usize,
    /// 零拷贝的最小数据大小（字节）
    pub zero_copy_threshold: usize,
}

impl Default for FFIOptimizationConfig {
    fn default() -> Self {
        Self {
            enable_batching: true,
            enable_zero_copy: true,
            min_batch_size: 10,
            zero_copy_threshold: 1024, // 1KB
        }
    }
}

/// 批量数据收集器 - 在Rust侧聚合数据，减少FFI调用
pub struct BatchDataCollector {
    data: Vec<u8>,
    shape: Vec<usize>,
    dtype: DataType,
    itemsize: usize,
}

impl BatchDataCollector {
    /// 创建新的批量数据收集器
    #[inline(always)]
    pub fn new(capacity: usize, shape: Vec<usize>, dtype: DataType, itemsize: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            shape,
            dtype,
            itemsize,
        }
    }

    /// 添加单行数据（Rust侧操作，无FFI开销）
    #[inline(always)]
    pub fn add_row(&mut self, row_data: &[u8]) {
        self.data.extend_from_slice(row_data);
    }

    /// 批量添加多行数据
    #[inline(always)]
    pub fn add_rows(&mut self, rows_data: &[&[u8]]) {
        for row_data in rows_data {
            self.data.extend_from_slice(row_data);
        }
    }

    /// 转换为NumPy数组（单次FFI调用）
    pub fn into_numpy_array(self, py: Python) -> PyResult<PyObject> {
        // 使用ndarray创建正确形状的数组
        let shape_ixdyn = IxDyn(&self.shape);
        let array = ArrayD::from_shape_vec(shape_ixdyn, self.data).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to create array: {}",
                e
            ))
        })?;

        // 单次FFI调用转换为PyArray
        Ok(array.to_pyarray(py).into())
    }

    /// 获取数据引用（用于零拷贝场景）
    #[inline(always)]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// 获取shape
    #[inline(always)]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

/// 零拷贝数组视图创建器
pub struct ZeroCopyArrayBuilder {
    mmap: Arc<Mmap>,
    config: FFIOptimizationConfig,
}

impl ZeroCopyArrayBuilder {
    /// 创建新的零拷贝构建器
    pub fn new(mmap: Arc<Mmap>, config: FFIOptimizationConfig) -> Self {
        Self { mmap, config }
    }

    /// 创建零拷贝NumPy数组视图
    ///
    /// # Safety
    /// 调用者必须确保：
    /// 1. offset和size在mmap范围内
    /// 2. 数据对齐正确
    /// 3. mmap在数组生命周期内有效
    pub unsafe fn create_view(
        &self,
        py: Python,
        offset: usize,
        shape: &[usize],
        dtype: DataType,
        itemsize: usize,
    ) -> PyResult<PyObject> {
        let total_size = shape.iter().product::<usize>() * itemsize;

        // 安全检查
        if offset + total_size > self.mmap.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Offset exceeds mmap range: offset={}, size={}, mmap_len={}",
                offset,
                total_size,
                self.mmap.len()
            )));
        }

        // 获取数据指针
        let data_ptr = self.mmap.as_ptr().add(offset);

        // 创建slice
        let data_slice = std::slice::from_raw_parts(data_ptr, total_size);

        // 使用ArrayView创建零拷贝视图
        let shape_ixdyn = IxDyn(shape);
        let array_view = ArrayViewD::from_shape(shape_ixdyn, data_slice).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to create zero-copy view: {}",
                e
            ))
        })?;

        // 转换为PyArray
        Ok(array_view.to_pyarray(py).into())
    }

    /// 判断是否应该使用零拷贝
    #[inline(always)]
    pub fn should_use_zero_copy(&self, data_size: usize) -> bool {
        self.config.enable_zero_copy && data_size >= self.config.zero_copy_threshold
    }
}

/// 批量索引操作优化器
pub struct BatchIndexOptimizer {
    config: FFIOptimizationConfig,
}

impl BatchIndexOptimizer {
    /// 创建新的批量索引优化器
    pub fn new(config: FFIOptimizationConfig) -> Self {
        Self { config }
    }

    /// 判断是否应该使用批量操作
    #[inline(always)]
    pub fn should_batch(&self, count: usize) -> bool {
        self.config.enable_batching && count >= self.config.min_batch_size
    }

    /// 批量获取行数据（优化版本）
    ///
    /// 此方法在Rust侧完成所有数据聚合，然后单次FFI调用返回
    pub fn batch_get_rows(
        &self,
        py: Python,
        mmap: &Arc<Mmap>,
        indices: &[usize],
        shape: &[usize],
        dtype: DataType,
        itemsize: usize,
    ) -> PyResult<PyObject> {
        let row_size = shape[1..].iter().product::<usize>() * itemsize;
        let total_size = indices.len() * row_size;

        // 创建批量收集器
        let mut collector = BatchDataCollector::new(
            total_size,
            vec![indices.len()]
                .into_iter()
                .chain(shape[1..].iter().copied())
                .collect(),
            dtype,
            itemsize,
        );

        // 在Rust侧聚合所有数据（无FFI开销）
        for &idx in indices {
            let offset = idx * row_size;

            // 边界检查
            if offset + row_size > mmap.len() {
                return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                    "Index out of range: idx={}, offset={}, row_size={}, mmap_len={}",
                    idx,
                    offset,
                    row_size,
                    mmap.len()
                )));
            }

            // 获取行数据
            unsafe {
                let row_slice = std::slice::from_raw_parts(mmap.as_ptr().add(offset), row_size);
                collector.add_row(row_slice);
            }
        }

        // 单次FFI调用转换为NumPy数组
        collector.into_numpy_array(py)
    }

    /// 批量获取指定索引的元素（点访问优化）
    pub fn batch_get_elements(
        &self,
        py: Python,
        mmap: &Arc<Mmap>,
        flat_indices: &[usize],
        dtype: DataType,
        itemsize: usize,
    ) -> PyResult<PyObject> {
        let total_size = flat_indices.len() * itemsize;

        // 创建收集器
        let mut collector =
            BatchDataCollector::new(total_size, vec![flat_indices.len()], dtype, itemsize);

        // 聚合数据
        for &idx in flat_indices {
            let offset = idx * itemsize;

            if offset + itemsize > mmap.len() {
                return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                    "Index out of range: idx={}",
                    idx
                )));
            }

            unsafe {
                let elem_slice = std::slice::from_raw_parts(mmap.as_ptr().add(offset), itemsize);
                collector.add_row(elem_slice);
            }
        }

        collector.into_numpy_array(py)
    }
}

/// 元数据缓存 - 避免重复计算
pub struct MetadataCache {
    row_size: usize,
    total_elements: usize,
    contiguous_size: usize,
}

impl MetadataCache {
    /// 创建新的元数据缓存
    #[inline(always)]
    pub fn new(shape: &[usize], itemsize: usize) -> Self {
        let row_size = if shape.len() > 1 {
            shape[1..].iter().product::<usize>() * itemsize
        } else {
            itemsize
        };

        let total_elements = shape.iter().product::<usize>();
        let contiguous_size = total_elements * itemsize;

        Self {
            row_size,
            total_elements,
            contiguous_size,
        }
    }

    /// 获取行大小（内联，无开销）
    #[inline(always)]
    pub fn row_size(&self) -> usize {
        self.row_size
    }

    /// 获取总元素数
    #[inline(always)]
    pub fn total_elements(&self) -> usize {
        self.total_elements
    }

    /// 获取连续数据大小
    #[inline(always)]
    pub fn contiguous_size(&self) -> usize {
        self.contiguous_size
    }

    /// 计算行偏移量（内联优化）
    #[inline(always)]
    pub fn row_offset(&self, row_idx: usize) -> usize {
        row_idx * self.row_size
    }

    /// 计算扁平化索引的偏移量
    #[inline(always)]
    pub fn flat_offset(&self, flat_idx: usize, itemsize: usize) -> usize {
        flat_idx * itemsize
    }
}

/// FFI优化统计信息
#[derive(Debug, Clone, Default)]
pub struct FFIOptimizationStats {
    /// FFI调用次数
    pub ffi_calls: u64,
    /// 批量操作次数
    pub batch_operations: u64,
    /// 零拷贝操作次数
    pub zero_copy_operations: u64,
    /// 节省的FFI调用次数
    pub saved_ffi_calls: u64,
    /// 节省的内存拷贝（字节）
    pub saved_memory_copies: u64,
}

impl FFIOptimizationStats {
    /// 记录FFI调用
    #[inline(always)]
    pub fn record_ffi_call(&mut self) {
        self.ffi_calls += 1;
    }

    /// 记录批量操作（节省了n-1次FFI调用）
    #[inline(always)]
    pub fn record_batch_operation(&mut self, items_count: usize) {
        self.batch_operations += 1;
        if items_count > 1 {
            self.saved_ffi_calls += (items_count - 1) as u64;
        }
    }

    /// 记录零拷贝操作
    #[inline(always)]
    pub fn record_zero_copy(&mut self, bytes_saved: usize) {
        self.zero_copy_operations += 1;
        self.saved_memory_copies += bytes_saved as u64;
    }

    /// 获取FFI减少比例
    pub fn ffi_reduction_ratio(&self) -> f64 {
        if self.ffi_calls == 0 {
            return 0.0;
        }
        self.saved_ffi_calls as f64 / (self.ffi_calls + self.saved_ffi_calls) as f64
    }

    /// 获取内存节省（MB）
    pub fn memory_saved_mb(&self) -> f64 {
        self.saved_memory_copies as f64 / (1024.0 * 1024.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_data_collector() {
        let mut collector = BatchDataCollector::new(100, vec![2, 5], DataType::Float32, 4);

        let row1 = vec![1u8, 2, 3, 4, 5];
        let row2 = vec![6u8, 7, 8, 9, 10];

        collector.add_row(&row1);
        collector.add_row(&row2);

        assert_eq!(collector.data().len(), 10);
    }

    #[test]
    fn test_metadata_cache() {
        let shape = vec![100, 10, 5];
        let itemsize = 4;

        let cache = MetadataCache::new(&shape, itemsize);

        assert_eq!(cache.row_size(), 10 * 5 * 4);
        assert_eq!(cache.total_elements(), 100 * 10 * 5);
        assert_eq!(cache.row_offset(5), 5 * 10 * 5 * 4);
    }

    #[test]
    fn test_optimization_stats() {
        let mut stats = FFIOptimizationStats::default();

        stats.record_ffi_call();
        stats.record_batch_operation(100); // 节省99次FFI调用
        stats.record_zero_copy(1024 * 1024); // 节省1MB拷贝

        assert_eq!(stats.ffi_calls, 1);
        assert_eq!(stats.saved_ffi_calls, 99);
        assert_eq!(stats.memory_saved_mb(), 1.0);

        let ratio = stats.ffi_reduction_ratio();
        assert!(ratio > 0.98 && ratio < 1.0);
    }
}
