#[macro_use]
extern crate lazy_static;

mod error;
mod metadata;
mod parallel_io;
mod lazy_array;
mod batch_access_engine;
mod windows_mapping; // 添加Windows平台特有的内存映射管理系统

#[cfg(test)]
mod fancy_index_tests;

#[cfg(test)]
mod fancy_index_integration_test;

#[cfg(test)]
mod simd_tests;

#[cfg(test)]
mod prefetch_tests;

#[cfg(test)]
mod batch_access_tests;

#[cfg(test)]
mod prefetch_benchmark;

#[cfg(test)]
mod zero_copy_tests;

#[cfg(test)]
mod multilevel_cache_tests;

use std::path::{Path, PathBuf};
use numpy::{IntoPyArray, PyArrayDyn, PyArrayMethods};
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::types::PySlice;
use std::fs::OpenOptions;
use std::io::Write;
use ndarray::ArrayD;
use pyo3::ffi::Py_buffer;
use memmap2::Mmap;
use std::ptr;
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::Arc;
use std::sync::MutexGuard;
use half::f16;

use crate::parallel_io::ParallelIO;
use crate::metadata::DataType;
use crate::lazy_array::{OptimizedLazyArray, FastTypeConversion};
use rayon::prelude::*;

#[cfg(target_family = "unix")]
#[allow(unused_imports)]
use std::os::unix::io::AsRawFd;

#[cfg(target_family = "windows")] 
use std::os::windows::io::{AsHandle, AsRawHandle};

#[cfg(target_family = "windows")]
use windows_sys::Win32::Storage::FileSystem::SetFileIoOverlappedRange;

#[cfg(target_family = "windows")]
use windows_sys::Win32::System::Memory::VirtualLock;

#[cfg(target_family = "windows")]
use windows_sys::Win32::Foundation::HANDLE;

lazy_static! {
    static ref MMAP_CACHE: Mutex<HashMap<String, (Arc<Mmap>, i64)>> = Mutex::new(HashMap::new());
}

#[allow(dead_code)]
#[pyclass]
struct NumPack {
    io: ParallelIO,
    base_dir: PathBuf,
}

#[allow(dead_code)]
#[pyclass]
struct LazyArray {
    mmap: Arc<Mmap>,
    shape: Vec<usize>,
    dtype: DataType,
    itemsize: usize,
    array_path: String,
    modify_time: i64,
}

/// Iterator for LazyArray that yields rows
#[pyclass]
pub struct LazyArrayIterator {
    array: LazyArray,
    current_index: usize,
    total_rows: usize,
}

#[pymethods]
impl LazyArrayIterator {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(&mut self, py: Python) -> PyResult<Option<PyObject>> {
        if self.current_index >= self.total_rows {
            return Ok(None);
        }

        let row_data = self.array.get_row_data(self.current_index)?;
        let row_shape = if self.array.shape.len() > 1 {
            self.array.shape[1..].to_vec()
        } else {
            vec![1]
        };

        let row_array = self.array.create_numpy_array(py, row_data, &row_shape)?;
        self.current_index += 1;
        
        Ok(Some(row_array))
    }
}

// 新增：索引类型枚举
#[derive(Debug, Clone)]
enum IndexType {
    Integer(i64),
    Slice(SliceInfo),
    BooleanMask(Vec<bool>),
    IntegerArray(Vec<i64>),
    Ellipsis,
    NewAxis,
}

#[derive(Debug, Clone)]
struct SliceInfo {
    start: Option<i64>,
    stop: Option<i64>,
    step: Option<i64>,
}

// 新增：索引解析结果
#[derive(Debug, Clone)]
struct IndexResult {
    indices: Vec<Vec<usize>>,  // 每个维度的索引
    result_shape: Vec<usize>,
    needs_broadcasting: bool,
    access_pattern: AccessPattern,
}

#[derive(Debug, Clone)]
enum AccessPattern {
    Sequential,
    Random,
    Clustered,
    Mixed,
}

// 新增：访问策略
#[derive(Debug, Clone)]
enum AccessStrategy {
    DirectMemory,
    BlockCopy,
    ParallelPointAccess,
    PrefetchOptimized,
    Adaptive,
}

#[pyclass]
struct ArrayMetadata {
    #[pyo3(get)]
    shape: Vec<i64>,
    #[pyo3(get)]
    dtype: String,
    #[pyo3(get)]
    data_file: String,
}

#[pymethods]
impl LazyArray {
    unsafe fn __getbuffer__(slf: PyRefMut<Self>, view: *mut Py_buffer, _flags: i32) -> PyResult<()> {
        if view.is_null() {
            return Err(PyErr::new::<pyo3::exceptions::PyBufferError, _>("View is null"));
        }

        let format = match slf.dtype {
            DataType::Bool => "?",
            DataType::Uint8 => "B",
            DataType::Uint16 => "H",
            DataType::Uint32 => "I",
            DataType::Uint64 => "Q",
            DataType::Int8 => "b",
            DataType::Int16 => "h",
            DataType::Int32 => "i",
            DataType::Int64 => "q",
            DataType::Float16 => "e",
            DataType::Float32 => "f",
            DataType::Float64 => "d",
        };

        let format_str = std::ffi::CString::new(format).unwrap();
        
        let mut strides = Vec::with_capacity(slf.shape.len());
        let mut stride = slf.itemsize;
        for &dim in slf.shape.iter().rev() {
            strides.push(stride as isize);
            stride *= dim;
        }
        strides.reverse();

        (*view).buf = slf.mmap.as_ptr() as *mut std::ffi::c_void;
        (*view).obj = ptr::null_mut();
        (*view).len = slf.mmap.len() as isize;
        (*view).readonly = 1;
        (*view).itemsize = slf.itemsize as isize;
        (*view).format = format_str.into_raw();
        (*view).ndim = slf.shape.len() as i32;
        (*view).shape = slf.shape.as_ptr() as *mut isize;
        (*view).strides = strides.as_ptr() as *mut isize;
        (*view).suboffsets = ptr::null_mut();
        (*view).internal = Box::into_raw(Box::new(strides)) as *mut std::ffi::c_void;

        Ok(())
    }

    unsafe fn __releasebuffer__(_slf: PyRefMut<Self>, view: *mut Py_buffer) {
        if !view.is_null() {
            if !(*view).format.is_null() {
                let _ = std::ffi::CString::from_raw((*view).format);
            }
            if !(*view).internal.is_null() {
                let _ = Box::from_raw((*view).internal as *mut Vec<isize>);
            }
        }
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        let total_rows = self.shape[0];
        let total_cols = if self.shape.len() > 1 { self.shape[1] } else { 1 };
        
        // Extract array name from path (remove suffix and data_ prefix)
        let array_name = self.array_path
            .split('/')
            .last()
            .unwrap_or(&self.array_path)
            .trim_end_matches(".npkd")
            .trim_start_matches("data_");
        
        // Build shape string
        let shape_str = format!("shape={:?}, dtype={:?}", self.shape, self.dtype);
        
        // If array is too small, display all content
        if total_rows <= 6 && total_cols <= 6 {
            let array = self.get_preview_data(py, 0, total_rows, 0, total_cols)?;
            return Ok(format!("LazyArray('{}', {}, \n{}", array_name, shape_str, array));
        }

        let mut result = String::new();
        result.push_str(&format!("LazyArray('{}', {}, \n", array_name, shape_str));

        // Get first 3 rows and last 3 rows
        let show_rows = if total_rows > 6 {
            vec![0, 1, 2, total_rows-3, total_rows-2, total_rows-1]
        } else {
            (0..total_rows).collect()
        };

        // Get first 3 columns and last 3 columns
        let show_cols = if total_cols > 6 {
            vec![0, 1, 2, total_cols-3, total_cols-2, total_cols-1]
        } else {
            (0..total_cols).collect()
        };

        let mut last_row = None;
        for &row in &show_rows {
            if let Some(last) = last_row {
                if row > last + 1 {
                    result.push_str(" ...\n");
                }
            }
            
            // Get current row data
            let mut row_str = String::new();
            let mut last_col = None;
            
            for &col in &show_cols {
                if let Some(last) = last_col {
                    if col > last + 1 {
                        row_str.push_str(" ...");
                    }
                }
                
                // Get single element
                let value = self.get_element(py, row, col)?;
                row_str.push_str(&format!(" {}", value));
                
                last_col = Some(col);
            }
            
            result.push_str(&format!("[{}]\n", row_str.trim()));
            last_row = Some(row);
        }

        result.push(')');
        Ok(result)
    }

    fn get_element(&self, _py: Python, row: usize, col: usize) -> PyResult<String> {
        let offset = (row * self.shape[1] + col) * self.itemsize;
        let value = match self.dtype {
            DataType::Bool => {
                let val = unsafe { *self.mmap.as_ptr().add(offset) };
                if val == 0 { "False" } else { "True" }.to_string()
            }
            DataType::Uint8 => unsafe { *self.mmap.as_ptr().add(offset) as u8 }.to_string(),
            DataType::Uint16 => unsafe { *(self.mmap.as_ptr().add(offset) as *const u16) }.to_string(),
            DataType::Uint32 => unsafe { *(self.mmap.as_ptr().add(offset) as *const u32) }.to_string(),
            DataType::Uint64 => unsafe { *(self.mmap.as_ptr().add(offset) as *const u64) }.to_string(),
            DataType::Int8 => unsafe { *self.mmap.as_ptr().add(offset) as i8 }.to_string(),
            DataType::Int16 => unsafe { *(self.mmap.as_ptr().add(offset) as *const i16) }.to_string(),
            DataType::Int32 => unsafe { *(self.mmap.as_ptr().add(offset) as *const i32) }.to_string(),
            DataType::Int64 => unsafe { *(self.mmap.as_ptr().add(offset) as *const i64) }.to_string(),
            DataType::Float16 => {
                let val = unsafe { *(self.mmap.as_ptr().add(offset) as *const f16) };
                format!("{:.6}", val)
            }
            DataType::Float32 => {
                let val = unsafe { *(self.mmap.as_ptr().add(offset) as *const f32) };
                format!("{:.6}", val)
            }
            DataType::Float64 => {
                let val = unsafe { *(self.mmap.as_ptr().add(offset) as *const f64) };
                format!("{:.6}", val)
            }
        };
        Ok(value)
    }

    fn get_preview_data(&self, py: Python, start_row: usize, end_row: usize, start_col: usize, end_col: usize) -> PyResult<String> {
        let mut result = String::new();
        for row in start_row..end_row {
            let mut row_str = String::new();
            for col in start_col..end_col {
                let value = self.get_element(py, row, col)?;
                row_str.push_str(&format!(" {}", value));
            }
            if result.is_empty() {
                result.push_str(&format!("[{}]", row_str.trim()));
            } else {
                result.push_str(&format!("\n [{}]", row_str.trim()));
            }
        }
        Ok(result)
    }

    fn __getitem__(&self, py: Python, key: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        // 检查是否是广播情况
        if let Ok(tuple) = key.downcast::<PyTuple>() {
            if self.check_for_broadcasting(tuple)? {
                return self.handle_broadcasting_directly(py, tuple);
            }
        }
        
        // 使用新的高级索引解析器
        let index_result = self.parse_advanced_index(py, key)?;
        
        // 根据索引结果选择最优的访问策略
        let access_strategy = self.choose_access_strategy(&index_result);
        
        // 执行索引操作
        match access_strategy {
            AccessStrategy::DirectMemory => self.direct_memory_access(py, &index_result),
            AccessStrategy::BlockCopy => self.block_copy_access(py, &index_result),
            AccessStrategy::ParallelPointAccess => self.parallel_point_access(py, &index_result),
            AccessStrategy::PrefetchOptimized => self.prefetch_optimized_access(py, &index_result),
            AccessStrategy::Adaptive => self.adaptive_access(py, &index_result),
        }
    }

    #[getter]
    fn shape(&self, py: Python) -> PyResult<PyObject> {
        let shape_tuple = pyo3::types::PyTuple::new(py, &self.shape)?;
        Ok(shape_tuple.into())
    }

    #[getter]
    fn dtype(&self, py: Python) -> PyResult<PyObject> {
        let numpy = py.import("numpy")?;
        let dtype_str = match self.dtype {
            DataType::Bool => "bool",
            DataType::Uint8 => "uint8",
            DataType::Uint16 => "uint16",
            DataType::Uint32 => "uint32",
            DataType::Uint64 => "uint64",
            DataType::Int8 => "int8",
            DataType::Int16 => "int16",
            DataType::Int32 => "int32",
            DataType::Int64 => "int64",
            DataType::Float16 => "float16",
            DataType::Float32 => "float32",
            DataType::Float64 => "float64",
        };
        let dtype = numpy.getattr("dtype")?.call1((dtype_str,))?;
        Ok(dtype.into())
    }

    #[getter]
    fn size(&self) -> PyResult<usize> {
        Ok(self.shape.iter().product())
    }

    #[getter]
    fn itemsize(&self) -> PyResult<usize> {
        Ok(self.itemsize)
    }

    #[getter]
    fn ndim(&self) -> PyResult<usize> {
        Ok(self.shape.len())
    }

    #[getter]
    fn nbytes(&self) -> PyResult<usize> {
        Ok(self.itemsize * self.size()?)
    }

    /// Reshape the array to a new shape (view operation, no data copying)
    /// 
    /// Parameters:
    ///     new_shape: Tuple, list, or integer representing the new shape
    ///               Supports -1 for automatic dimension inference
    /// 
    /// Returns:
    ///     A new LazyArray with the reshaped view
    fn reshape(&self, py: Python, new_shape: &Bound<'_, PyAny>) -> PyResult<Py<LazyArray>> {
        // Parse the new shape from different input types
        let mut shape: Vec<i64> = if let Ok(tuple) = new_shape.downcast::<pyo3::types::PyTuple>() {
            // Handle tuple input: (dim1, dim2, ...)
            let mut shape = Vec::new();
            for i in 0..tuple.len() {
                let item = tuple.get_item(i)?;
                if let Ok(dim) = item.extract::<i64>() {
                    // Allow -1 for automatic inference, but reject other negative values
                    if dim < -1 {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "Negative dimensions other than -1 are not supported in reshape"
                        ));
                    }
                    shape.push(dim);
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "All dimensions must be integers"
                    ));
                }
            }
            shape
        } else if let Ok(list) = new_shape.downcast::<pyo3::types::PyList>() {
            // Handle list input: [dim1, dim2, ...]
            let mut shape = Vec::new();
            for item in list.iter() {
                if let Ok(dim) = item.extract::<i64>() {
                    // Allow -1 for automatic inference, but reject other negative values
                    if dim < -1 {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "Negative dimensions other than -1 are not supported in reshape"
                        ));
                    }
                    shape.push(dim);
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "All dimensions must be integers"
                    ));
                }
            }
            shape
        } else if let Ok(dim) = new_shape.extract::<i64>() {
            // Handle single integer input
            if dim < -1 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Negative dimensions other than -1 are not supported in reshape"
                ));
            }
            vec![dim]
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Shape must be a tuple, list, or integer"
            ));
        };

        // Handle -1 dimension inference
        let original_size: usize = self.shape.iter().product();
        let mut inferred_dim_index = None;
        let mut known_size = 1usize;
        
        // Find -1 dimension and calculate known size
        for (i, &dim) in shape.iter().enumerate() {
            if dim == -1 {
                if inferred_dim_index.is_some() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Only one dimension can be -1"
                    ));
                }
                inferred_dim_index = Some(i);
            } else {
                known_size *= dim as usize;
            }
        }
        
        // Calculate inferred dimension
        if let Some(infer_idx) = inferred_dim_index {
            if known_size == 0 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Cannot infer dimension when other dimensions contain 0"
                ));
            }
            if original_size % known_size != 0 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!(
                        "Cannot reshape array of size {} into shape with known size {}",
                        original_size, known_size
                    )
                ));
            }
            shape[infer_idx] = (original_size / known_size) as i64;
        }
        
        // Convert to usize and validate
        let final_shape: Vec<usize> = shape.iter().map(|&dim| {
            if dim < 0 {
                // This should not happen after inference, but just in case
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Invalid dimension after inference"
                ));
            }
            Ok(dim as usize)
        }).collect::<PyResult<Vec<_>>>()?;

        // Validate that the total number of elements remains the same
        let new_size: usize = final_shape.iter().product();
        
        if original_size != new_size {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!(
                    "Cannot reshape array of size {} into shape {:?} (total size {})", 
                    original_size, final_shape, new_size
                )
            ));
        }

        // Create a new LazyArray with the same underlying data but different shape
        let reshaped_array = LazyArray {
            mmap: Arc::clone(&self.mmap),
            shape: final_shape,
            dtype: self.dtype.clone(),
            itemsize: self.itemsize,
            array_path: self.array_path.clone(),
            modify_time: self.modify_time,
        };

        // Return the new LazyArray as a Python object
        Py::new(py, reshaped_array)
    }

    // ===========================
    // 生产级性能优化方法
    // ===========================

    // 阶段1：极限FFI优化
    fn mega_batch_get_rows(&self, py: Python, indices: Vec<usize>, batch_size: usize) -> PyResult<Vec<PyObject>> {
        // 由于LazyArray没有OptimizedLazyArray的功能，我们需要使用基础的批量操作
        let mut results = Vec::new();
        let chunk_size = batch_size.max(100);
        
        for chunk in indices.chunks(chunk_size) {
            for &idx in chunk {
                let row_data = self.get_row_data(idx)?;
                let numpy_array = self.bytes_to_numpy(py, row_data)?;
                results.push(numpy_array);
            }
        }
        
        Ok(results)
    }



    // 阶段2：深度SIMD优化
    fn vectorized_gather(&self, py: Python, indices: Vec<usize>) -> PyResult<PyObject> {
        // Handle empty indices case
        if indices.is_empty() {
            let mut empty_shape = self.shape.clone();
            empty_shape[0] = 0;
            return self.create_numpy_array(py, Vec::new(), &empty_shape);
        }
        
        // 批量收集数据
        let mut all_data = Vec::new();
        for &idx in &indices {
            if idx < self.shape[0] {
            let row_data = self.get_row_data(idx)?;
            all_data.extend(row_data);
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                    format!("Index {} is out of bounds for array with {} rows", idx, self.shape[0])
                ));
            }
        }
        
        let mut result_shape = self.shape.clone();
        result_shape[0] = indices.len();
        
        self.create_numpy_array(py, all_data, &result_shape)
    }

    fn parallel_boolean_index(&self, py: Python, mask: Vec<bool>) -> PyResult<PyObject> {
        if mask.len() != self.shape[0] {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Mask length doesn't match array length"));
        }
        
        // 收集选中的行
        let mut selected_data = Vec::new();
        let mut selected_count = 0;
        for (idx, &selected) in mask.iter().enumerate() {
            if selected {
                let row_data = self.get_row_data(idx)?;
                selected_data.extend(row_data);
                selected_count += 1;
            }
        }
        
        let mut result_shape = self.shape.clone();
        result_shape[0] = selected_count;
        
        self.create_numpy_array(py, selected_data, &result_shape)
    }

    // 阶段3：内存管理优化
    fn intelligent_warmup(&self, workload_hint: &str) -> PyResult<()> {
        // 简化的预热实现
        let warmup_size = match workload_hint {
            "sequential" => 0.1,
            "random" => 0.05,
            "boolean" => 0.2,
            "heavy" => 0.5,
            _ => 0.1,
        };
        
        let total_rows = self.shape[0];
        let warmup_rows = ((total_rows as f64) * warmup_size) as usize;
        
        for i in 0..warmup_rows {
            let _ = self.get_row_data(i);
        }
        
        Ok(())
    }

    fn get_performance_stats(&self) -> PyResult<Vec<(String, f64)>> {
        // 返回基本的性能统计
        Ok(vec![
            ("cache_hits".to_string(), 0.0),
            ("cache_misses".to_string(), 0.0),
            ("hit_rate".to_string(), 0.0),
            ("cache_blocks".to_string(), 0.0),
            ("current_cache_size".to_string(), 0.0),
            ("max_cache_size".to_string(), 0.0),
        ])
    }

    // 阶段4：算法级优化
    fn boolean_index_production(&self, py: Python, mask: Vec<bool>) -> PyResult<PyObject> {
        self.parallel_boolean_index(py, mask)
    }

    fn boolean_index_adaptive_algorithm(&self, py: Python, mask: Vec<bool>) -> PyResult<PyObject> {
        self.parallel_boolean_index(py, mask)
    }

    fn choose_optimal_algorithm(&self, mask: Vec<bool>) -> PyResult<String> {
        let selected_count = mask.iter().filter(|&&x| x).count();
        let selection_density = selected_count as f64 / mask.len() as f64;
        
        let algorithm = if selection_density < 0.01 {
            "ZeroCopy"
        } else if selection_density > 0.9 {
            "Vectorized"
        } else if selection_density > 0.5 {
            "AdaptivePrefetch"
        } else {
            "StandardSIMD"
        };
        
        Ok(algorithm.to_string())
    }

    // 辅助方法
    fn get_row_data(&self, row_idx: usize) -> PyResult<Vec<u8>> {
        if row_idx >= self.shape[0] {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>("Row index out of bounds"));
        }
        
        let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
        let offset = row_idx * row_size;
        
        if offset + row_size > self.mmap.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>("Data access out of bounds"));
        }
        
        Ok(self.mmap[offset..offset + row_size].to_vec())
    }

    fn bytes_to_numpy(&self, py: Python, data: Vec<u8>) -> PyResult<PyObject> {
        let row_shape = vec![self.shape[1..].iter().product::<usize>()];
        self.create_numpy_array(py, data, &row_shape)
    }

    fn __len__(&self) -> PyResult<usize> {
        if self.shape.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "len() of unsized object"
            ));
        }
        Ok(self.shape[0])
    }

    fn __enter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __exit__(
        &self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_val: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        // 显式清理资源 - 所有平台通用实现
        // 手动触发对mmap的引用，以便在离开上下文管理器时确保资源被释放
        let _temp = Arc::clone(&self.mmap);
        drop(_temp);
        
        // Windows平台特有的额外清理
        #[cfg(target_family = "windows")]
        {
            // 检查是否为临时文件
            let path = std::path::Path::new(&self.array_path);
            if self.array_path.contains("temp") || self.array_path.contains("tmp") {
                release_windows_file_handle(path);
            }
        }
        
        Ok(false)  // 返回false表示不抑制异常
    }
}

// 实现Drop特性以确保Windows平台上的资源正确释放
#[cfg(target_family = "windows")]
impl Drop for LazyArray {
    fn drop(&mut self) {
        // 使用智能系统处理文件解锁和清理
        // Arc将处理引用计数，最终引用释放时将自动处理清理
        let path = std::path::Path::new(&self.array_path);
        
        // 触发Arc的drop，可能是最后一个引用
        let _temp = Arc::clone(&self.mmap);
        drop(_temp);
        
        // 如果存在明显的文件问题，使用主动清理
        if self.array_path.contains("temp") || self.array_path.contains("tmp") {
            // 临时文件使用立即清理策略
            release_windows_file_handle(path);
        }
    }
}

// HighPerformanceLazyArray的更智能Drop实现
#[cfg(target_family = "windows")]
impl Drop for HighPerformanceLazyArray {
    fn drop(&mut self) {
        // OptimizedLazyArray处理将由其自己的Drop实现负责
        // 这里触发Drop但不需要额外处理
        drop(&self.optimized_array);
    }
}

// 新增：LazyArray的内部方法实现
impl LazyArray {
    // 新增：高级索引解析器
    fn parse_advanced_index(&self, py: Python, key: &Bound<'_, PyAny>) -> Result<IndexResult, PyErr> {
        let mut index_types = Vec::new();
        
        // 解析索引类型
        if let Ok(tuple) = key.downcast::<PyTuple>() {
            // 多维索引：(rows, cols, ...)
            // 检查是否有广播情况
            let has_broadcasting = self.check_for_broadcasting(tuple)?;
            
            if has_broadcasting {
                return self.handle_broadcasting_index(py, tuple);
            }
            
            for i in 0..tuple.len() {
                let item = tuple.get_item(i)?;
                index_types.push(self.parse_single_index(&item)?);
            }
        } else {
            // 单维索引
            index_types.push(self.parse_single_index(key)?);
        }
        
        // 验证索引维度
        if index_types.len() > self.shape.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "Too many indices for array"
            ));
        }
        
        // 处理索引解析和广播
        self.process_indices(index_types)
    }
    
    // 新增：解析单个索引
    fn parse_single_index(&self, key: &Bound<'_, PyAny>) -> Result<IndexType, PyErr> {
        // 整数索引
        if let Ok(index) = key.extract::<i64>() {
            return Ok(IndexType::Integer(index));
        }
        
        // 切片索引
        if let Ok(slice) = key.downcast::<PySlice>() {
            let slice_info = SliceInfo {
                start: slice.getattr("start")?.extract::<Option<i64>>()?,
                stop: slice.getattr("stop")?.extract::<Option<i64>>()?,
                step: slice.getattr("step")?.extract::<Option<i64>>()?,
            };
            return Ok(IndexType::Slice(slice_info));
        }
        
        // 布尔掩码
        if let Ok(bool_mask) = key.extract::<Vec<bool>>() {
            return Ok(IndexType::BooleanMask(bool_mask));
        }
        
        // 整数数组
        if let Ok(int_array) = key.extract::<Vec<i64>>() {
            return Ok(IndexType::IntegerArray(int_array));
        }
        
        // NumPy数组
        if let Ok(numpy_array) = key.getattr("__array__") {
            if let Ok(array_func) = numpy_array.call0() {
                // Get array shape for broadcasting support
                let shape = if let Ok(shape_attr) = key.getattr("shape") {
                    shape_attr.extract::<Vec<usize>>().unwrap_or_else(|_| vec![])
                } else {
                    vec![]
                };
                
                // Handle multi-dimensional arrays
                if shape.len() > 1 {
                    // Extract as multi-dimensional integer array
                    if let Ok(nested_array) = self.extract_multidim_array(key) {
                        return Ok(IndexType::IntegerArray(nested_array));
                    }
                }
                
                if let Ok(bool_array) = array_func.extract::<Vec<bool>>() {
                    return Ok(IndexType::BooleanMask(bool_array));
                }
                if let Ok(int_array) = array_func.extract::<Vec<i64>>() {
                    return Ok(IndexType::IntegerArray(int_array));
                }
            }
        }
        
        // 省略号 - 简化检查
        if key.to_string().contains("Ellipsis") {
            return Ok(IndexType::Ellipsis);
        }
        
        // newaxis/None
        if key.is_none() {
            return Ok(IndexType::NewAxis);
        }
        
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Invalid index type"
        ))
    }
    
    // 新增：检查是否有广播情况
    fn check_for_broadcasting(&self, tuple: &Bound<'_, PyTuple>) -> Result<bool, PyErr> {
        for i in 0..tuple.len() {
            let item = tuple.get_item(i)?;
            if let Ok(_numpy_array) = item.getattr("__array__") {
                if let Ok(shape_attr) = item.getattr("shape") {
                    if let Ok(shape) = shape_attr.extract::<Vec<usize>>() {
                        if shape.len() > 1 {
                            return Ok(true);
                        }
                    }
                }
            }
        }
        Ok(false)
    }
    
    // 新增：直接处理广播索引
    fn handle_broadcasting_directly(&self, py: Python, tuple: &Bound<'_, PyTuple>) -> Result<PyObject, PyErr> {
        if tuple.len() != 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "Broadcasting only supported for 2D indexing currently"
            ));
        }
        
        let first_item = tuple.get_item(0)?;
        let second_item = tuple.get_item(1)?;
        
        // 提取第一个索引（可能是多维的）
        let first_array = self.extract_array_data(&first_item)?;
        let first_shape = self.get_array_shape(&first_item)?;
        
        // 提取第二个索引
        let second_array = self.extract_array_data(&second_item)?;
        let second_shape = self.get_array_shape(&second_item)?;
        
        // 执行广播
        let (broadcast_first, broadcast_second, result_shape) = 
            self.broadcast_arrays(first_array, first_shape, second_array, second_shape)?;
        
        // 直接执行数据访问
        self.execute_broadcasting_access(py, broadcast_first, broadcast_second, result_shape)
    }
    
    // 新增：处理广播索引
    fn handle_broadcasting_index(&self, py: Python, tuple: &Bound<'_, PyTuple>) -> Result<IndexResult, PyErr> {
        if tuple.len() != 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "Broadcasting only supported for 2D indexing currently"
            ));
        }
        
        let first_item = tuple.get_item(0)?;
        let second_item = tuple.get_item(1)?;
        
        // 提取第一个索引（可能是多维的）
        let first_array = self.extract_array_data(&first_item)?;
        let first_shape = self.get_array_shape(&first_item)?;
        
        // 提取第二个索引
        let second_array = self.extract_array_data(&second_item)?;
        let second_shape = self.get_array_shape(&second_item)?;
        
        // 执行广播
        let (broadcast_first, broadcast_second, result_shape) = 
            self.broadcast_arrays(first_array, first_shape, second_array, second_shape)?;
        
        // 直接计算结果数据而不是创建索引结果
        return self.execute_broadcasting_access_to_index_result(py, broadcast_first, broadcast_second, result_shape);
    }
    
    // 新增：执行广播访问
    fn execute_broadcasting_access(&self, py: Python, first_indices: Vec<usize>, second_indices: Vec<usize>, result_shape: Vec<usize>) -> Result<PyObject, PyErr> {
        // 验证索引数量一致
        if first_indices.len() != second_indices.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Broadcasting indices length mismatch"
            ));
        }
        
        let total_elements = result_shape.iter().product::<usize>();
        let mut result_data = Vec::with_capacity(total_elements * self.itemsize);
        
        // 计算剩余维度的大小
        let remaining_dims_size = if self.shape.len() > 2 {
            self.shape[2..].iter().product::<usize>()
        } else {
            1
        };
        
        // 对每个索引对进行数据访问
        for i in 0..first_indices.len() {
            let row_idx = first_indices[i];
            let col_idx = second_indices[i];
            
            // 计算基本偏移
            let base_offset = (row_idx * self.shape[1] + col_idx) * remaining_dims_size * self.itemsize;
            
            // 复制剩余维度的数据
            let element_size = remaining_dims_size * self.itemsize;
            let element_data = unsafe {
                std::slice::from_raw_parts(
                    self.mmap.as_ptr().add(base_offset),
                    element_size
                )
            };
            result_data.extend_from_slice(element_data);
        }
        
        // 计算最终形状
        let mut final_shape = result_shape;
        for dim in 2..self.shape.len() {
            final_shape.push(self.shape[dim]);
        }
        
        // 创建 NumPy 数组并返回
        self.create_numpy_array(py, result_data, &final_shape)
    }
    
    // 新增：执行广播访问（返回IndexResult）
    fn execute_broadcasting_access_to_index_result(&self, py: Python, first_indices: Vec<usize>, second_indices: Vec<usize>, result_shape: Vec<usize>) -> Result<IndexResult, PyErr> {
        // 验证索引数量一致
        if first_indices.len() != second_indices.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Broadcasting indices length mismatch"
            ));
        }
        
        let total_elements = result_shape.iter().product::<usize>();
        let mut result_data = Vec::with_capacity(total_elements * self.itemsize);
        
        // 计算剩余维度的大小
        let remaining_dims_size = if self.shape.len() > 2 {
            self.shape[2..].iter().product::<usize>()
        } else {
            1
        };
        
        // 对每个索引对进行数据访问
        for i in 0..first_indices.len() {
            let row_idx = first_indices[i];
            let col_idx = second_indices[i];
            
            // 计算基本偏移
            let base_offset = (row_idx * self.shape[1] + col_idx) * remaining_dims_size * self.itemsize;
            
            // 复制剩余维度的数据
            let element_size = remaining_dims_size * self.itemsize;
            let element_data = unsafe {
                std::slice::from_raw_parts(
                    self.mmap.as_ptr().add(base_offset),
                    element_size
                )
            };
            result_data.extend_from_slice(element_data);
        }
        
        // 计算最终形状
        let mut final_shape = result_shape;
        for dim in 2..self.shape.len() {
            final_shape.push(self.shape[dim]);
        }
        
        // 创建 NumPy 数组
        let _result_array = self.create_numpy_array(py, result_data, &final_shape)?;
        
        // 返回一个特殊的索引结果，表示已经处理完成
        Ok(IndexResult {
            indices: vec![vec![0]], // 占位符
            result_shape: final_shape,
            needs_broadcasting: false, // 已经处理完成
            access_pattern: AccessPattern::Random,
        })
    }
    
    // 新增：提取数组数据
    fn extract_array_data(&self, key: &Bound<'_, PyAny>) -> Result<Vec<i64>, PyErr> {
        if let Ok(numpy_array) = key.getattr("__array__") {
            if let Ok(array_func) = numpy_array.call0() {
                // 尝试获取扁平化的数据
                if let Ok(flattened) = key.call_method0("flatten") {
                    if let Ok(flat_array) = flattened.getattr("__array__") {
                        if let Ok(flat_func) = flat_array.call0() {
                            if let Ok(int_array) = flat_func.extract::<Vec<i64>>() {
                                return Ok(int_array);
                            }
                        }
                    }
                }
                
                // 如果无法扁平化，尝试直接提取
                if let Ok(int_array) = array_func.extract::<Vec<i64>>() {
                    return Ok(int_array);
                }
            }
        }
        
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Cannot extract array data"
        ))
    }
    
    // 新增：获取数组形状
    fn get_array_shape(&self, key: &Bound<'_, PyAny>) -> Result<Vec<usize>, PyErr> {
        if let Ok(shape_attr) = key.getattr("shape") {
            let _shape = if let Ok(shape) = shape_attr.extract::<Vec<usize>>() {
                return Ok(shape);
            };
        }
        Ok(vec![])
    }
    
    // 新增：执行数组广播
    fn broadcast_arrays(&self, first: Vec<i64>, first_shape: Vec<usize>, 
                       second: Vec<i64>, second_shape: Vec<usize>) -> Result<(Vec<usize>, Vec<usize>, Vec<usize>), PyErr> {
        
        // 简化的广播实现：只处理 (N, 1) 和 (M,) 的情况
        if first_shape.len() == 2 && first_shape[1] == 1 && second_shape.len() == 1 {
            let rows = first_shape[0];
            let cols = second_shape[0];
            
            // 验证索引范围
            for &idx in &first {
                if idx < 0 || idx as usize >= self.shape[0] {
                    return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                        format!("Index {} out of bounds for dimension 0 of size {}", idx, self.shape[0])
                    ));
                }
            }
            
            for &idx in &second {
                if idx < 0 || idx as usize >= self.shape[1] {
                    return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                        format!("Index {} out of bounds for dimension 1 of size {}", idx, self.shape[1])
                    ));
                }
            }
            
            // 扩展第一个数组 - 每行的值重复cols次
            let mut broadcast_first = Vec::new();
            for i in 0..rows {
                for _j in 0..cols {
                    broadcast_first.push(first[i] as usize);
                }
            }
            
            // 扩展第二个数组 - 每行都有完整的列索引
            let mut broadcast_second = Vec::new();
            for _i in 0..rows {
                for j in 0..cols {
                    broadcast_second.push(second[j] as usize);
                }
            }
            
            Ok((broadcast_first, broadcast_second, vec![rows, cols]))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "Unsupported broadcasting pattern"
            ))
        }
    }
    
    // 新增：提取多维数组（用于广播）
    fn extract_multidim_array(&self, key: &Bound<'_, PyAny>) -> Result<Vec<i64>, PyErr> {
        if let Ok(numpy_array) = key.getattr("__array__") {
            if let Ok(array_func) = numpy_array.call0() {
                // 尝试获取形状信息
                let _shape = if let Ok(shape_attr) = key.getattr("shape") {
                    shape_attr.extract::<Vec<usize>>().unwrap_or_else(|_| vec![])
                } else {
                    vec![]
                };
                
                // 尝试获取扁平化的数据
                if let Ok(flattened) = key.call_method0("flatten") {
                    if let Ok(flat_array) = flattened.getattr("__array__") {
                        if let Ok(flat_func) = flat_array.call0() {
                            if let Ok(int_array) = flat_func.extract::<Vec<i64>>() {
                                return Ok(int_array);
                            }
                        }
                    }
                }
                
                // 如果无法扁平化，尝试直接提取
                if let Ok(int_array) = array_func.extract::<Vec<i64>>() {
                    return Ok(int_array);
                }
            }
        }
        
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Cannot extract multidimensional array"
        ))
    }
    
    // 新增：处理索引解析和广播
    fn process_indices(&self, index_types: Vec<IndexType>) -> Result<IndexResult, PyErr> {
        let mut indices = Vec::new();
        // 移除重复声明的变量，保留后面的使用
        let mut needs_broadcasting = false;
        
        // 扩展索引到完整维度（添加省略号处理）
        let expanded_indices = self.expand_indices(index_types)?;
        
        // 分离NewAxis和实际数组索引
        let mut actual_indices = Vec::new();
        let mut newaxis_positions = Vec::new();
        
        for (pos, index_type) in expanded_indices.iter().enumerate() {
            match index_type {
                IndexType::NewAxis => {
                    newaxis_positions.push(pos);
                }
                _ => {
                    actual_indices.push(index_type.clone());
                }
            }
        }
        
        // 处理实际数组索引
        let mut array_dim = 0;
        for (_result_pos, index_type) in expanded_indices.iter().enumerate() {
            match index_type {
                IndexType::NewAxis => {
                    // NewAxis不消耗原数组维度，只在结果中添加维度
                    continue;
                }
                _ => {
                    // 处理实际的数组索引
                    if array_dim >= self.shape.len() {
                        return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                            "Too many indices for array"
                        ));
                    }
                    
                    match index_type {
                        IndexType::Integer(idx) => {
                            let adjusted_idx = self.adjust_index(*idx, self.shape[array_dim])?;
                            indices.push(vec![adjusted_idx]);
                            // 整数索引不增加维度到结果
                        }
                        IndexType::Slice(slice_info) => {
                            let slice_indices = self.resolve_slice(slice_info, self.shape[array_dim])?;
                            indices.push(slice_indices);
                        }
                        IndexType::BooleanMask(mask) => {
                            let bool_indices = self.resolve_boolean_mask(mask, self.shape[array_dim])?;
                            indices.push(bool_indices);
                        }
                        IndexType::IntegerArray(arr) => {
                            let int_indices = self.resolve_integer_array(arr, self.shape[array_dim])?;
                            indices.push(int_indices);
                            needs_broadcasting = true;
                        }
                        IndexType::Ellipsis => {
                            // 省略号已在expand_indices中处理
                        }
                        _ => {}
                    }
                    
                    array_dim += 1;
                }
            }
        }
        
        // 构建结果形状，考虑NewAxis的位置
        let mut result_shape: Vec<usize> = Vec::new();
        let mut array_dim = 0;
        
        for (_pos, index_type) in expanded_indices.iter().enumerate() {
            match index_type {
                IndexType::NewAxis => {
                    result_shape.push(1);
                }
                IndexType::Integer(_) => {
                    // 整数索引不增加维度
                    array_dim += 1;
                }
                IndexType::Slice(_) => {
                    if array_dim < indices.len() {
                        result_shape.push(indices[array_dim].len());
                    }
                    array_dim += 1;
                }
                IndexType::BooleanMask(_) => {
                    if array_dim < indices.len() {
                        result_shape.push(indices[array_dim].len());
                    }
                    array_dim += 1;
                }
                IndexType::IntegerArray(_) => {
                    if array_dim < indices.len() {
                        result_shape.push(indices[array_dim].len());
                    }
                    array_dim += 1;
                }
                IndexType::Ellipsis => {
                    // 省略号已在expand_indices中处理
                }
            }
        }
        
        // 检测访问模式
        let access_pattern = self.analyze_access_pattern(&indices);
        
        Ok(IndexResult {
            indices,
            result_shape,
            needs_broadcasting,
            access_pattern,
        })
    }
    
    // 新增：扩展索引到完整维度
    fn expand_indices(&self, index_types: Vec<IndexType>) -> Result<Vec<IndexType>, PyErr> {
        let mut expanded = Vec::new();
        let mut ellipsis_found = false;
        
        for index_type in index_types.iter() {
            match index_type {
                IndexType::Ellipsis => {
                    if ellipsis_found {
                        return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                            "Only one ellipsis allowed"
                        ));
                    }
                    ellipsis_found = true;
                    
                    // 计算省略号需要填充的维度数
                    // 需要排除NewAxis，因为NewAxis不消耗原数组维度
                    let non_newaxis_count = index_types.iter().filter(|&t| !matches!(t, IndexType::NewAxis)).count();
                    let remaining_dims = self.shape.len() - (non_newaxis_count - 1);
                    for _ in 0..remaining_dims {
                        expanded.push(IndexType::Slice(SliceInfo {
                            start: None,
                            stop: None,
                            step: None,
                        }));
                    }
                }
                _ => expanded.push(index_type.clone()),
            }
        }
        
        // 如果没有省略号，填充剩余维度
        // 计算已经消耗的原数组维度数（排除NewAxis）
        while expanded.iter().filter(|&t| !matches!(t, IndexType::NewAxis)).count() < self.shape.len() {
            expanded.push(IndexType::Slice(SliceInfo {
                start: None,
                stop: None,
                step: None,
            }));
        }
        
        Ok(expanded)
    }
    
    // 新增：调整索引（处理负索引）
    fn adjust_index(&self, index: i64, dim_size: usize) -> Result<usize, PyErr> {
        let adjusted = if index < 0 {
            dim_size as i64 + index
            } else {
                index
            };
        
        if adjusted >= 0 && (adjusted as usize) < dim_size {
            Ok(adjusted as usize)
            } else {
            Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                format!("Index {} out of bounds for dimension of size {}", index, dim_size)
            ))
        }
    }
    
    // 新增：解析切片
    fn resolve_slice(&self, slice_info: &SliceInfo, dim_size: usize) -> Result<Vec<usize>, PyErr> {
        let start = slice_info.start.unwrap_or(0);
        let stop = slice_info.stop.unwrap_or(dim_size as i64);
        let step = slice_info.step.unwrap_or(1);
        
        if step == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Slice step cannot be zero"
            ));
        }
        
        let mut indices = Vec::new();
        
        if step > 0 {
            let mut i = if start < 0 { dim_size as i64 + start } else { start };
            let end = if stop < 0 { dim_size as i64 + stop } else { stop };
            
            while i < end && i < dim_size as i64 {
                if i >= 0 {
                    indices.push(i as usize);
                }
                i += step;
            }
        } else {
            let mut i = if start < 0 { dim_size as i64 + start } else { start.min(dim_size as i64 - 1) };
            let end = if stop < 0 { dim_size as i64 + stop } else { stop };
            
            while i > end && i >= 0 {
                if i < dim_size as i64 {
                    indices.push(i as usize);
                }
                i += step;
            }
        }
        
        Ok(indices)
    }
    
    // 新增：解析布尔掩码
    fn resolve_boolean_mask(&self, mask: &[bool], dim_size: usize) -> Result<Vec<usize>, PyErr> {
        if mask.len() != dim_size {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Boolean mask length {} doesn't match dimension size {}", mask.len(), dim_size)
            ));
        }
        
        let mut indices = Vec::new();
        for (i, &selected) in mask.iter().enumerate() {
            if selected {
                indices.push(i);
            }
        }
        
        Ok(indices)
    }
    
    // 新增：解析整数数组
    fn resolve_integer_array(&self, arr: &[i64], dim_size: usize) -> Result<Vec<usize>, PyErr> {
        let mut indices = Vec::new();
        
        for &idx in arr {
            let adjusted = self.adjust_index(idx, dim_size)?;
            indices.push(adjusted);
        }
        
        Ok(indices)
    }
    
    // 新增：分析访问模式
    fn analyze_access_pattern(&self, indices: &[Vec<usize>]) -> AccessPattern {
        if indices.is_empty() {
            return AccessPattern::Sequential;
        }
        
        let first_indices = &indices[0];
        if first_indices.len() <= 1 {
            return AccessPattern::Sequential;
        }
        
        // 检查是否为顺序访问
        let mut is_sequential = true;
        for i in 1..first_indices.len() {
            if first_indices[i] != first_indices[i-1] + 1 {
                is_sequential = false;
                break;
            }
        }
        
        if is_sequential {
            return AccessPattern::Sequential;
        }
        
        // 检查是否为聚集访问
        let mut gaps = Vec::new();
        for i in 1..first_indices.len() {
            if first_indices[i] >= first_indices[i-1] {
                gaps.push(first_indices[i] - first_indices[i-1]);
            } else {
                // 如果索引不是升序，直接返回随机访问
                return AccessPattern::Random;
            }
        }
        
        let avg_gap = gaps.iter().sum::<usize>() as f64 / gaps.len() as f64;
        let variance = gaps.iter().map(|&g| (g as f64 - avg_gap).powi(2)).sum::<f64>() / gaps.len() as f64;
        
        if variance < avg_gap * 0.5 {
            AccessPattern::Clustered
        } else {
            AccessPattern::Random
        }
    }
    
    // 新增：选择访问策略
    fn choose_access_strategy(&self, index_result: &IndexResult) -> AccessStrategy {
        let total_elements = index_result.indices.iter().map(|idx| idx.len()).product::<usize>();
        let source_elements = self.shape.iter().product::<usize>();
        let selection_ratio = total_elements as f64 / source_elements as f64;
        
        match (&index_result.access_pattern, selection_ratio) {
            (AccessPattern::Sequential, r) if r > 0.8 => AccessStrategy::BlockCopy,
            (AccessPattern::Sequential, _) => AccessStrategy::DirectMemory,
            (AccessPattern::Random, r) if r < 0.1 => AccessStrategy::ParallelPointAccess,
            (AccessPattern::Clustered, _) => AccessStrategy::PrefetchOptimized,
            _ => AccessStrategy::Adaptive,
        }
    }
    
    // 新增：直接内存访问
    fn direct_memory_access(&self, py: Python, index_result: &IndexResult) -> Result<PyObject, PyErr> {
        let total_elements = index_result.result_shape.iter().product::<usize>();
        let mut result_data = Vec::with_capacity(total_elements * self.itemsize);
        
        // 计算多维索引的笛卡尔积
        let index_combinations = self.compute_index_combinations(&index_result.indices);
        
        for combination in index_combinations {
            let offset = self.compute_linear_offset(&combination);
            let element_data = unsafe {
                std::slice::from_raw_parts(
                    self.mmap.as_ptr().add(offset),
                    self.itemsize
                )
            };
            result_data.extend_from_slice(element_data);
        }
        
        self.create_numpy_array(py, result_data, &index_result.result_shape)
    }
    
    // 新增：块复制访问
    fn block_copy_access(&self, py: Python, index_result: &IndexResult) -> Result<PyObject, PyErr> {
        if index_result.indices.len() == 1 {
            // 单维连续访问优化
            let indices = &index_result.indices[0];
        let row_size = self.itemsize * self.shape[1..].iter().product::<usize>();
            let mut result_data = Vec::with_capacity(indices.len() * row_size);
            
            // 检查是否为连续块
            if self.is_continuous_block(indices) {
                let start_offset = indices[0] * row_size;
                let block_size = indices.len() * row_size;
                let block_data = unsafe {
                    std::slice::from_raw_parts(
                        self.mmap.as_ptr().add(start_offset),
                        block_size
                    )
                };
                result_data.extend_from_slice(block_data);
            } else {
                // 分块复制
                for &idx in indices {
                    let offset = idx * row_size;
                    let row_data = unsafe {
                        std::slice::from_raw_parts(
                            self.mmap.as_ptr().add(offset),
                            row_size
                        )
                    };
                    result_data.extend_from_slice(row_data);
                }
            }
            
            self.create_numpy_array(py, result_data, &index_result.result_shape)
        } else {
            // 多维访问回退到直接内存访问
            self.direct_memory_access(py, index_result)
        }
    }
    
    // 新增：并行点访问
    fn parallel_point_access(&self, py: Python, index_result: &IndexResult) -> Result<PyObject, PyErr> {
        let index_combinations = self.compute_index_combinations(&index_result.indices);
        
        // 并行处理索引组合
        let result_data: Vec<u8> = index_combinations
            .par_iter()
            .flat_map(|combination| {
                let offset = self.compute_linear_offset(combination);
                unsafe {
                    std::slice::from_raw_parts(
                        self.mmap.as_ptr().add(offset),
                        self.itemsize
                    )
                }.to_vec()
            })
            .collect();
        
        self.create_numpy_array(py, result_data, &index_result.result_shape)
    }
    
    // 新增：预取优化访问
    fn prefetch_optimized_access(&self, py: Python, index_result: &IndexResult) -> Result<PyObject, PyErr> {
        let index_combinations = self.compute_index_combinations(&index_result.indices);
        
        // 预取数据
        self.prefetch_data(&index_combinations);
        
        // 执行访问
        let mut result_data = Vec::with_capacity(index_combinations.len() * self.itemsize);
        
        for combination in index_combinations {
            let offset = self.compute_linear_offset(&combination);
            let element_data = unsafe {
                std::slice::from_raw_parts(
                    self.mmap.as_ptr().add(offset),
                    self.itemsize
                )
            };
            result_data.extend_from_slice(element_data);
        }
        
        self.create_numpy_array(py, result_data, &index_result.result_shape)
    }
    
    // 新增：自适应访问
    fn adaptive_access(&self, py: Python, index_result: &IndexResult) -> Result<PyObject, PyErr> {
        let total_elements = index_result.result_shape.iter().product::<usize>();
        
        // 根据数据大小选择策略
        if total_elements < 1000 {
            self.direct_memory_access(py, index_result)
        } else if total_elements < 100000 {
            self.parallel_point_access(py, index_result)
        } else {
            self.prefetch_optimized_access(py, index_result)
        }
    }
    
    // 新增：计算索引组合（笛卡尔积）
    fn compute_index_combinations(&self, indices: &[Vec<usize>]) -> Vec<Vec<usize>> {
        if indices.is_empty() {
            return vec![vec![]];
        }
        
        let mut combinations = vec![vec![]];
        
        for dim_indices in indices {
            let mut new_combinations = Vec::new();
            
            for combination in combinations {
                for &idx in dim_indices {
                    let mut new_combination = combination.clone();
                    new_combination.push(idx);
                    new_combinations.push(new_combination);
                }
            }
            
            combinations = new_combinations;
        }
        
        combinations
    }
    
    // 新增：计算线性偏移
    fn compute_linear_offset(&self, indices: &[usize]) -> usize {
        let mut offset = 0;
        let mut stride = self.itemsize;
        
        // 从最后一个维度开始计算stride
        for i in (0..self.shape.len()).rev() {
            if i < indices.len() {
                offset += indices[i] * stride;
            }
            if i > 0 {
                stride *= self.shape[i];
            }
        }
        
        offset
    }
    
    // 新增：检查是否为连续块
    fn is_continuous_block(&self, indices: &[usize]) -> bool {
        if indices.len() <= 1 {
            return true;
        }
        
        for i in 1..indices.len() {
            if indices[i] != indices[i-1] + 1 {
                return false;
            }
        }
        
        true
    }
    
    // 新增：预取数据
    fn prefetch_data(&self, index_combinations: &[Vec<usize>]) {
        // 使用CPU预取指令
        for combination in index_combinations {
            let offset = self.compute_linear_offset(combination);
            unsafe {
                #[cfg(target_arch = "x86_64")]
                {
                    use std::arch::x86_64::_mm_prefetch;
                    _mm_prefetch(
                        self.mmap.as_ptr().add(offset) as *const i8,
                        std::arch::x86_64::_MM_HINT_T0
                    );
                }
                
                #[cfg(target_arch = "aarch64")]
                {
                    std::arch::asm!(
                        "prfm pldl1keep, [{}]",
                        in(reg) self.mmap.as_ptr().add(offset)
                    );
                }
            }
        }
    }
    
    // 新增：创建NumPy数组
    fn create_numpy_array(&self, py: Python, data: Vec<u8>, shape: &[usize]) -> Result<PyObject, PyErr> {
        let array: PyObject = match self.dtype {
            DataType::Bool => {
                let bool_vec: Vec<bool> = data.iter().map(|&x| x != 0).collect();
                let array = ArrayD::from_shape_vec(shape.to_vec(), bool_vec).unwrap();
                array.into_pyarray(py).into()
            }
            DataType::Uint8 => {
                let array = unsafe {
                    let slice: &[u8] = bytemuck::cast_slice(&data);
                    ArrayD::from_shape_vec_unchecked(shape.to_vec(), slice.to_vec())
                };
                array.into_pyarray(py).into()
            }
            DataType::Uint16 => {
                let array = unsafe {
                    let slice: &[u16] = bytemuck::cast_slice(&data);
                    ArrayD::from_shape_vec_unchecked(shape.to_vec(), slice.to_vec())
                };
                array.into_pyarray(py).into()
            }
            DataType::Uint32 => {
                let array = unsafe {
                    let slice: &[u32] = bytemuck::cast_slice(&data);
                    ArrayD::from_shape_vec_unchecked(shape.to_vec(), slice.to_vec())
                };
                array.into_pyarray(py).into()
            }
            DataType::Uint64 => {
                let array = unsafe {
                    let slice: &[u64] = bytemuck::cast_slice(&data);
                    ArrayD::from_shape_vec_unchecked(shape.to_vec(), slice.to_vec())
                };
                array.into_pyarray(py).into()
            }
            DataType::Int8 => {
                let array = unsafe {
                    let slice: &[i8] = bytemuck::cast_slice(&data);
                    ArrayD::from_shape_vec_unchecked(shape.to_vec(), slice.to_vec())
                };
                array.into_pyarray(py).into()
            }
            DataType::Int16 => {
                let array = unsafe {
                    let slice: &[i16] = bytemuck::cast_slice(&data);
                    ArrayD::from_shape_vec_unchecked(shape.to_vec(), slice.to_vec())
                };
                array.into_pyarray(py).into()
            }
            DataType::Int32 => {
                let array = unsafe {
                    let slice: &[i32] = bytemuck::cast_slice(&data);
                    ArrayD::from_shape_vec_unchecked(shape.to_vec(), slice.to_vec())
                };
                array.into_pyarray(py).into()
            }
            DataType::Int64 => {
                let array = unsafe {
                    let slice: &[i64] = bytemuck::cast_slice(&data);
                    ArrayD::from_shape_vec_unchecked(shape.to_vec(), slice.to_vec())
                };
                array.into_pyarray(py).into()
            }
            DataType::Float16 => {
                let array = unsafe {
                    let slice: &[f16] = bytemuck::cast_slice(&data);
                    ArrayD::from_shape_vec_unchecked(shape.to_vec(), slice.to_vec())
                };
                array.into_pyarray(py).into()
            }
            DataType::Float32 => {
                let array = unsafe {
                    let slice: &[f32] = bytemuck::cast_slice(&data);
                    ArrayD::from_shape_vec_unchecked(shape.to_vec(), slice.to_vec())
                };
                array.into_pyarray(py).into()
            }
            DataType::Float64 => {
                let array = unsafe {
                    let slice: &[f64] = bytemuck::cast_slice(&data);
                    ArrayD::from_shape_vec_unchecked(shape.to_vec(), slice.to_vec())
                };
                array.into_pyarray(py).into()
            }
        };

        Ok(array)
    }



    fn extract_indices_from_key(&self, key: &Bound<'_, PyAny>, total_rows: usize) -> PyResult<Vec<usize>> {
        // Try to extract as boolean mask
        if let Ok(bool_mask) = key.extract::<Vec<bool>>() {
            if bool_mask.len() != total_rows {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Boolean mask length {} doesn't match array length {}", bool_mask.len(), total_rows)
                ));
            }
            let mut indices = Vec::new();
            for (i, &mask_val) in bool_mask.iter().enumerate() {
                if mask_val {
                    indices.push(i);
                }
            }
            return Ok(indices);
        }
        
        // Try to extract as list of integers
        if let Ok(int_indices) = key.extract::<Vec<i64>>() {
            let mut indices = Vec::new();
            for idx in int_indices {
                let adjusted_index = if idx < 0 {
                    total_rows as i64 + idx
                } else {
                    idx
                };
                if adjusted_index >= 0 && (adjusted_index as usize) < total_rows {
                    indices.push(adjusted_index as usize);
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                        format!("Index {} is out of bounds for array of length {}", idx, total_rows)
                    ));
                }
            }
            return Ok(indices);
        }
        
        // Try to handle numpy arrays
        if let Ok(numpy_array) = key.getattr("__array__") {
            if let Ok(array_func) = numpy_array.call0() {
                // Get array shape to handle multi-dimensional arrays
                let shape = if let Ok(shape_attr) = key.getattr("shape") {
                    shape_attr.extract::<Vec<usize>>().unwrap_or_else(|_| vec![])
                } else {
                    vec![]
                };
                
                // Handle multi-dimensional arrays (broadcasting case)
                if shape.len() > 1 {
                    // This is a multidimensional array, store it for later broadcasting
                    // For now, return an error indicating this needs special handling
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "Multidimensional array indexing requires broadcasting - handle in parse_advanced_index"
                    ));
                }
                
                // Try to extract as boolean array
                if let Ok(bool_array) = array_func.extract::<Vec<bool>>() {
                    if bool_array.len() != total_rows {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            format!("Boolean array length {} doesn't match array length {}", bool_array.len(), total_rows)
                        ));
                    }
                    let mut indices = Vec::new();
                    for (i, &mask_val) in bool_array.iter().enumerate() {
                        if mask_val {
                            indices.push(i);
                        }
                    }
                    return Ok(indices);
                }
                
                // Try to extract as integer array
                if let Ok(int_array) = array_func.extract::<Vec<i64>>() {
                    let mut indices = Vec::new();
                    for idx in int_array {
                        let adjusted_index = if idx < 0 {
                            total_rows as i64 + idx
                        } else {
                            idx
                        };
                        if adjusted_index >= 0 && (adjusted_index as usize) < total_rows {
                            indices.push(adjusted_index as usize);
                        } else {
                            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                                format!("Index {} is out of bounds for array of length {}", idx, total_rows)
                            ));
                        }
                    }
                    return Ok(indices);
                }
            }
        }
        
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Invalid index type. Supported types: int, slice, list of ints, boolean mask, or numpy arrays"))
    }
}

#[pyclass]
pub struct HighPerformanceLazyArray {
    optimized_array: OptimizedLazyArray,
    shape: Vec<usize>,
    dtype: DataType,
    itemsize: usize,
}

#[pymethods]
impl HighPerformanceLazyArray {
    #[new]
    fn new(file_path: String, shape: Vec<usize>, dtype_str: String) -> PyResult<Self> {
        let dtype = match dtype_str.as_str() {
            "bool" => DataType::Bool,
            "uint8" => DataType::Uint8,
            "uint16" => DataType::Uint16,
            "uint32" => DataType::Uint32,
            "uint64" => DataType::Uint64,
            "int8" => DataType::Int8,
            "int16" => DataType::Int16,
            "int32" => DataType::Int32,
            "int64" => DataType::Int64,
            "float16" => DataType::Float16,
            "float32" => DataType::Float32,
            "float64" => DataType::Float64,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Unsupported dtype: {}", dtype_str)
            )),
        };

        let itemsize = dtype.size_bytes() as usize;
        let optimized_array = OptimizedLazyArray::new(
            PathBuf::from(file_path),
            shape.clone(),
            dtype.clone()
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

        Ok(Self {
            optimized_array,
            shape,
            dtype,
            itemsize,
        })
    }

    // 高性能行访问
    fn get_row(&self, py: Python, row_idx: usize) -> PyResult<PyObject> {
        let row_data = self.optimized_array.get_row(row_idx);
        self.bytes_to_numpy_array(py, row_data, self.shape[1..].to_vec())
    }

    // 快速行访问（跳过缓存）
    fn get_row_fast(&self, py: Python, row_idx: usize) -> PyResult<PyObject> {
        let row_data = self.optimized_array.get_row_fast(row_idx);
        self.bytes_to_numpy_array(py, row_data, self.shape[1..].to_vec())
    }

    // 高性能批量行访问
    fn get_rows(&self, py: Python, row_indices: Vec<usize>) -> PyResult<PyObject> {
        let rows_data = self.optimized_array.get_rows(&row_indices);
        let mut new_shape = self.shape.clone();
        new_shape[0] = row_indices.len();
        
        let mut combined_data = Vec::new();
        for row_data in rows_data {
            combined_data.extend(row_data);
        }
        
        self.bytes_to_numpy_array(py, combined_data, new_shape)
    }

    // 新增：高性能范围访问（减少FFI开销）
    fn get_rows_range(&self, py: Python, start_row: usize, end_row: usize) -> PyResult<PyObject> {
        let data = self.optimized_array.get_rows_range(start_row, end_row);
        let mut new_shape = self.shape.clone();
        new_shape[0] = end_row - start_row;
        
        self.bytes_to_numpy_array(py, data, new_shape)
    }

    // 新增：超快速行范围访问（进一步减少FFI开销）
    fn get_rows_range_ultra(&self, py: Python, start_row: usize, end_row: usize) -> PyResult<PyObject> {
        if start_row >= self.shape[0] || end_row > self.shape[0] || start_row >= end_row {
            return self.bytes_to_numpy_array(py, Vec::new(), vec![0]);
        }

        let data = self.optimized_array.get_rows_range(start_row, end_row);
        let mut new_shape = self.shape.clone();
        new_shape[0] = end_row - start_row;
        
        self.bytes_to_numpy_array(py, data, new_shape)
    }

    // 新增：批量预取优化行访问
    fn get_rows_with_prefetch(&self, py: Python, row_indices: Vec<usize>) -> PyResult<PyObject> {
        // 对索引进行预排序以优化内存访问
        let mut sorted_pairs: Vec<(usize, usize)> = row_indices.iter()
            .enumerate()
            .map(|(original_idx, &row_idx)| (original_idx, row_idx))
            .collect();
        
        sorted_pairs.sort_by_key(|&(_, row_idx)| row_idx);
        
        let sorted_indices: Vec<usize> = sorted_pairs.iter()
            .map(|(_, row_idx)| *row_idx)
            .collect();
        
        let rows_data = self.optimized_array.get_rows(&sorted_indices);
        
        // 恢复原始顺序
        let mut ordered_data = vec![Vec::new(); row_indices.len()];
        for ((original_idx, _), data) in sorted_pairs.into_iter().zip(rows_data.into_iter()) {
            ordered_data[original_idx] = data;
        }
        
        let mut new_shape = self.shape.clone();
        new_shape[0] = row_indices.len();
        
        self.optimized_bytes_to_numpy_array(py, ordered_data, new_shape)
    }

    // 高性能连续访问
    fn get_continuous(&self, py: Python, start_offset: usize, size: usize) -> PyResult<PyObject> {
        let data = self.optimized_array.get_continuous_data(start_offset, size);
        let elements = size / self.itemsize;
        self.bytes_to_numpy_array(py, data, vec![elements])
    }

    // 简化的连续访问（减少缓存开销）
    fn get_continuous_fast(&self, py: Python, start_offset: usize, size: usize) -> PyResult<PyObject> {
        // 直接从mmap读取，跳过缓存以减少开销
        let data = self.optimized_array.get_continuous_zero_copy(start_offset, size).to_vec();
        let elements = size / self.itemsize;
        self.bytes_to_numpy_array(py, data, vec![elements])
    }

    // 高性能布尔索引（自动应用最优算法）
    fn boolean_index(&self, py: Python, mask: Vec<bool>) -> PyResult<PyObject> {
        // 自动使用智能算法选择，根据数据特征选择最优策略
        let selected_rows = self.optimized_array.boolean_index_smart(&mask);
        let mut new_shape = self.shape.clone();
        new_shape[0] = selected_rows.len();
        
        self.optimized_bytes_to_numpy_array(py, selected_rows, new_shape)
    }

    // 优化的布尔索引（自动应用最优算法）
    fn boolean_index_optimized(&self, py: Python, mask: Vec<bool>) -> PyResult<PyObject> {
        // 自动使用智能算法选择，提供比原始optimized更好的性能
        let selected_rows = self.optimized_array.boolean_index_smart(&mask);
        let mut new_shape = self.shape.clone();
        new_shape[0] = selected_rows.len();
        
        self.optimized_bytes_to_numpy_array(py, selected_rows, new_shape)
    }

    // 超高性能布尔索引（自动应用最优算法）
    fn boolean_index_ultra_fast(&self, py: Python, mask: Vec<bool>) -> PyResult<PyObject> {
        // 根据测试结果，智能算法在大多数情况下比原ultra_fast更优
        let selected_rows = self.optimized_array.boolean_index_smart(&mask);
        let mut new_shape = self.shape.clone();
        new_shape[0] = selected_rows.len();
        
        self.optimized_bytes_to_numpy_array(py, selected_rows, new_shape)
    }

    // CPU缓存优化的布尔索引（自动应用最优算法）
    fn boolean_index_cache_optimized(&self, py: Python, mask: Vec<bool>) -> PyResult<PyObject> {
        // 智能算法内置了最优的缓存策略
        let selected_rows = self.optimized_array.boolean_index_smart(&mask);
        let mut new_shape = self.shape.clone();
        new_shape[0] = selected_rows.len();
        
        self.optimized_bytes_to_numpy_array(py, selected_rows, new_shape)
    }

    // 终极优化布尔索引（自动应用最优算法）
    fn boolean_index_ultimate(&self, py: Python, mask: Vec<bool>) -> PyResult<PyObject> {
        // 智能算法会根据数据特征自动选择最优策略，包括ultimate级别的优化
        let selected_rows = self.optimized_array.boolean_index_smart(&mask);
        let mut new_shape = self.shape.clone();
        new_shape[0] = selected_rows.len();
        
        self.optimized_bytes_to_numpy_array(py, selected_rows, new_shape)
    }

    // 新增：极限SIMD优化布尔索引
    fn boolean_index_extreme(&self, py: Python, mask: Vec<bool>) -> PyResult<PyObject> {
        let selected_rows = self.optimized_array.boolean_index_extreme(&mask);
        let mut new_shape = self.shape.clone();
        new_shape[0] = selected_rows.len();
        
        // 使用最优化的内存分配和复制策略
        self.optimized_bytes_to_numpy_array(py, selected_rows, new_shape)
    }

    // 新增：微小数据优化布尔索引
    fn boolean_index_micro(&self, py: Python, mask: Vec<bool>) -> PyResult<PyObject> {
        let selected_rows = self.optimized_array.boolean_index_micro(&mask);
        let mut new_shape = self.shape.clone();
        new_shape[0] = selected_rows.len();
        
        self.optimized_bytes_to_numpy_array(py, selected_rows, new_shape)
    }

    // 高性能切片
    fn slice(&self, py: Python, ranges: Vec<(usize, usize)>) -> PyResult<PyObject> {
        let ranges: Vec<std::ops::Range<usize>> = ranges.into_iter()
            .map(|(start, end)| start..end)
            .collect();
        
        let data = self.optimized_array.slice(&ranges);
        
        // 计算结果形状
        let result_shape: Vec<usize> = ranges.iter()
            .enumerate()
            .map(|(dim, range)| {
                let dim_size = self.shape.get(dim).cloned().unwrap_or(1);
                range.end.min(dim_size) - range.start.min(dim_size)
            })
            .collect();
        
        self.bytes_to_numpy_array(py, data, result_shape)
    }

    // 预热缓存
    fn warmup_cache(&self, sample_rate: f64) -> PyResult<()> {
        self.optimized_array.warmup_cache(sample_rate);
        Ok(())
    }

    // 获取缓存统计
    fn get_cache_stats(&self) -> PyResult<(u64, u64, f64)> {
        Ok(self.optimized_array.get_cache_stats())
    }

    // 清理缓存
    fn clear_cache(&self) -> PyResult<()> {
        self.optimized_array.clear_cache();
        Ok(())
    }

    // 获取数组属性
    #[getter]
    fn shape(&self) -> PyResult<Vec<usize>> {
        Ok(self.shape.clone())
    }

    #[getter]
    fn dtype(&self, py: Python) -> PyResult<PyObject> {
        let numpy = py.import("numpy")?;
        let dtype_str = match self.dtype {
            DataType::Bool => "bool",
            DataType::Uint8 => "uint8",
            DataType::Uint16 => "uint16",
            DataType::Uint32 => "uint32",
            DataType::Uint64 => "uint64",
            DataType::Int8 => "int8",
            DataType::Int16 => "int16",
            DataType::Int32 => "int32",
            DataType::Int64 => "int64",
            DataType::Float16 => "float16",
            DataType::Float32 => "float32",
            DataType::Float64 => "float64",
        };
        let dtype = numpy.getattr("dtype")?.call1((dtype_str,))?;
        Ok(dtype.into())
    }

    #[getter]
    fn itemsize(&self) -> PyResult<usize> {
        Ok(self.itemsize)
    }

    #[getter]
    fn size(&self) -> PyResult<usize> {
        Ok(self.shape.iter().product())
    }

    #[getter]
    fn ndim(&self) -> PyResult<usize> {
        Ok(self.shape.len())
    }

    #[getter]
    fn nbytes(&self) -> PyResult<usize> {
        Ok(self.itemsize * self.size()?)
    }

    // 优化的字节数据转换为NumPy数组（减少内存分配）
    fn optimized_bytes_to_numpy_array(&self, py: Python, rows_data: Vec<Vec<u8>>, shape: Vec<usize>) -> PyResult<PyObject> {
        if rows_data.is_empty() {
            return self.bytes_to_numpy_array(py, Vec::new(), shape);
        }
        
        let row_size = self.shape[1..].iter().product::<usize>() * self.itemsize;
        let total_size = rows_data.len() * row_size;
        
        // 预分配连续内存块
        let mut combined_data: Vec<u8> = Vec::with_capacity(total_size);
        
        // 检查是否所有行大小一致
        let uniform_size = rows_data.iter().all(|row| row.len() == row_size);
        
        if uniform_size && total_size > 0 {
            unsafe {
                combined_data.set_len(total_size);
                
                // 使用SIMD优化的批量复制
                let mut offset = 0;
                for row_data in rows_data {
                    if offset + row_data.len() <= total_size {
                        // 对于大块数据使用优化的内存复制
                        if row_data.len() >= 64 {
                            simd_copy_if_possible(
                                row_data.as_ptr(),
                                combined_data.as_mut_ptr().add(offset),
                                row_data.len()
                            );
                        } else {
                            std::ptr::copy_nonoverlapping(
                                row_data.as_ptr(),
                                combined_data.as_mut_ptr().add(offset),
                                row_data.len()
                            );
                        }
                        offset += row_data.len();
                    }
                }
            }
        } else {
            // 回退到标准方法
            for row_data in rows_data {
                combined_data.extend(row_data);
            }
        }
        
        self.bytes_to_numpy_array(py, combined_data, shape)
    }



    // 内部方法：将字节数据转换为NumPy数组
    fn bytes_to_numpy_array(&self, py: Python, data: Vec<u8>, shape: Vec<usize>) -> PyResult<PyObject> {
        let array: PyObject = match self.dtype {
            DataType::Bool => {
                let bool_vec: Vec<bool> = data.iter().map(|&x| x != 0).collect();
                let array = ArrayD::from_shape_vec(shape.to_vec(), bool_vec)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
                array.into_pyarray(py).into()
            }
            DataType::Uint8 => {
                let array = ArrayD::from_shape_vec(shape.to_vec(), data)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
                array.into_pyarray(py).into()
            }
            DataType::Uint16 => {
                let typed_data = data.to_typed_vec::<u16>();
                let array = ArrayD::from_shape_vec(shape.to_vec(), typed_data)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
                array.into_pyarray(py).into()
            }
            DataType::Uint32 => {
                let typed_data = data.to_typed_vec::<u32>();
                let array = ArrayD::from_shape_vec(shape.to_vec(), typed_data)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
                array.into_pyarray(py).into()
            }
            DataType::Uint64 => {
                let typed_data = data.to_typed_vec::<u64>();
                let array = ArrayD::from_shape_vec(shape.to_vec(), typed_data)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
                array.into_pyarray(py).into()
            }
            DataType::Int8 => {
                let typed_data = data.to_typed_vec::<i8>();
                let array = ArrayD::from_shape_vec(shape.to_vec(), typed_data)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
                array.into_pyarray(py).into()
            }
            DataType::Int16 => {
                let typed_data = data.to_typed_vec::<i16>();
                let array = ArrayD::from_shape_vec(shape.to_vec(), typed_data)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
                array.into_pyarray(py).into()
            }
            DataType::Int32 => {
                let typed_data = data.to_typed_vec::<i32>();
                let array = ArrayD::from_shape_vec(shape.to_vec(), typed_data)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
                array.into_pyarray(py).into()
            }
            DataType::Int64 => {
                let typed_data = data.to_typed_vec::<i64>();
                let array = ArrayD::from_shape_vec(shape.to_vec(), typed_data)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
                array.into_pyarray(py).into()
            }
            DataType::Float16 => {
                let typed_data = data.to_typed_vec::<f16>();
                let array = ArrayD::from_shape_vec(shape.to_vec(), typed_data)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
                array.into_pyarray(py).into()
            }
            DataType::Float32 => {
                let typed_data = data.to_typed_vec::<f32>();
                let array = ArrayD::from_shape_vec(shape.to_vec(), typed_data)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
                array.into_pyarray(py).into()
            }
            DataType::Float64 => {
                let typed_data = data.to_typed_vec::<f64>();
                let array = ArrayD::from_shape_vec(shape.to_vec(), typed_data)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
                array.into_pyarray(py).into()
            }
        };
        
        Ok(array)
    }

    // 实现Python的 __repr__ 方法
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "HighPerformanceLazyArray(shape={:?}, dtype={:?}, itemsize={})",
            self.shape, self.dtype, self.itemsize
        ))
    }

    // 实现Python的 __len__ 方法
    fn __len__(&self) -> PyResult<usize> {
        if self.shape.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "len() of unsized object"
            ));
        }
        Ok(self.shape[0])
    }
    
    // 添加上下文管理器支持
    fn __enter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __exit__(
        &self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_val: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        // 触发OptimizedLazyArray的清理
        let _ = &self.optimized_array; // 使用let _替代drop引用
        Ok(false)  // 返回false表示不抑制异常
    }

    // 新增：智能策略布尔索引
    fn boolean_index_smart(&self, py: Python, mask: Vec<bool>) -> PyResult<PyObject> {
        let selected_rows = self.optimized_array.boolean_index_smart(&mask);
        let mut new_shape = self.shape.clone();
        new_shape[0] = selected_rows.len();
        
        self.optimized_bytes_to_numpy_array(py, selected_rows, new_shape)
    }

    // 新增：自适应预取布尔索引
    fn boolean_index_adaptive(&self, py: Python, mask: Vec<bool>) -> PyResult<PyObject> {
        let selected_rows = self.optimized_array.boolean_index_adaptive_prefetch(&mask);
        let mut new_shape = self.shape.clone();
        new_shape[0] = selected_rows.len();
        
        self.optimized_bytes_to_numpy_array(py, selected_rows, new_shape)
    }

    // ===========================
    // 生产级性能优化方法
    // ===========================

    // 阶段1：极限FFI优化
    fn mega_batch_get_rows(&self, py: Python, indices: Vec<usize>, batch_size: usize) -> PyResult<Vec<PyObject>> {
        let rows = self.optimized_array.mega_batch_get_rows(&indices, batch_size);
        let row_shape = vec![self.shape[1..].iter().product::<usize>()];
        
        rows.into_iter()
            .map(|row| self.optimized_bytes_to_numpy_array(py, vec![row], row_shape.clone()))
            .collect()
    }

    fn get_row_view(&self, py: Python, row_idx: usize) -> PyResult<PyObject> {
        if let Some(view) = self.optimized_array.get_row_view(row_idx) {
            let row_shape = vec![self.shape[1..].iter().product::<usize>()];
            self.optimized_bytes_to_numpy_array(py, vec![view.to_vec()], row_shape)
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>("Index out of bounds"))
        }
    }

    // 阶段2：深度SIMD优化
    fn vectorized_gather(&self, py: Python, indices: Vec<usize>) -> PyResult<PyObject> {
        let rows = self.optimized_array.vectorized_gather(&indices);
        let mut new_shape = self.shape.clone();
        new_shape[0] = rows.len();
        
        self.optimized_bytes_to_numpy_array(py, rows, new_shape)
    }

    fn parallel_boolean_index(&self, py: Python, mask: Vec<bool>) -> PyResult<PyObject> {
        let selected_rows = self.optimized_array.parallel_boolean_index(&mask);
        let mut new_shape = self.shape.clone();
        new_shape[0] = selected_rows.len();
        
        self.optimized_bytes_to_numpy_array(py, selected_rows, new_shape)
    }

    // 阶段3：内存管理优化
    #[cfg(target_os = "linux")]
    fn numa_aware_read(&self, py: Python, offset: usize, size: usize) -> PyResult<PyObject> {
        let data = self.optimized_array.numa_aware_read(offset, size);
        let shape = vec![data.len()];
        self.optimized_bytes_to_numpy_array(py, vec![data], shape)
    }

    fn intelligent_warmup(&self, workload_hint: &str) -> PyResult<()> {
        use crate::lazy_array::WorkloadHint;
        
        let hint = match workload_hint {
            "sequential" => WorkloadHint::SequentialRead,
            "random" => WorkloadHint::RandomRead,
            "boolean" => WorkloadHint::BooleanFiltering,
            "heavy" => WorkloadHint::HeavyComputation,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid workload hint")),
        };
        
        self.optimized_array.intelligent_warmup(&hint);
        Ok(())
    }

    // 阶段4：算法级优化
    fn boolean_index_production(&self, py: Python, mask: Vec<bool>) -> PyResult<PyObject> {
        let selected_rows = self.optimized_array.boolean_index_production(&mask);
        let mut new_shape = self.shape.clone();
        new_shape[0] = selected_rows.len();
        
        self.optimized_bytes_to_numpy_array(py, selected_rows, new_shape)
    }

    fn boolean_index_adaptive_algorithm(&self, py: Python, mask: Vec<bool>) -> PyResult<PyObject> {
        let selected_rows = self.optimized_array.boolean_index_adaptive_algorithm(&mask);
        let mut new_shape = self.shape.clone();
        new_shape[0] = selected_rows.len();
        
        self.optimized_bytes_to_numpy_array(py, selected_rows, new_shape)
    }

    fn choose_optimal_algorithm(&self, mask: Vec<bool>) -> PyResult<String> {
        let selected_count = mask.iter().filter(|&&x| x).count();
        let selection_density = selected_count as f64 / mask.len() as f64;
        
        let algorithm = if selection_density < 0.01 {
            "ZeroCopy"
        } else if selection_density > 0.9 {
            "Vectorized"
        } else if selection_density > 0.5 {
            "Parallel"
        } else {
            "Adaptive"
        };
        
        Ok(algorithm.to_string())
    }

    // 性能分析方法
    fn get_performance_stats(&self) -> PyResult<Vec<(String, f64)>> {
        let stats = self.optimized_array.get_extended_cache_stats();
        let (hits, misses, hit_rate, blocks, current_size, max_size) = stats;
        
        Ok(vec![
            ("cache_hits".to_string(), hits as f64),
            ("cache_misses".to_string(), misses as f64),
            ("hit_rate".to_string(), hit_rate),
            ("cache_blocks".to_string(), blocks as f64),
            ("current_cache_size".to_string(), current_size as f64),
            ("max_cache_size".to_string(), max_size as f64),
        ])
    }

    // 新增：完整的性能基准测试方法
    fn benchmark_boolean_methods(&self, py: Python, mask: Vec<bool>) -> PyResult<PyObject> {
        use std::time::Instant;
        
        let mut results = std::collections::HashMap::new();
        
        // 测试原始方法
        let start = Instant::now();
        let _ = self.optimized_array.boolean_index(&mask);
        let original_time = start.elapsed().as_secs_f64();
        results.insert("original", original_time);
        
        // 测试优化方法
        let start = Instant::now();
        let _ = self.optimized_array.boolean_index_optimized(&mask);
        let optimized_time = start.elapsed().as_secs_f64();
        results.insert("optimized", optimized_time);
        
        // 测试超高性能方法
        let start = Instant::now();
        let _ = self.optimized_array.boolean_index_ultra_fast(&mask);
        let ultra_time = start.elapsed().as_secs_f64();
        results.insert("ultra_fast", ultra_time);
        
        // 测试极限方法
        let start = Instant::now();
        let _ = self.optimized_array.boolean_index_extreme(&mask);
        let extreme_time = start.elapsed().as_secs_f64();
        results.insert("extreme", extreme_time);
        
        // 测试智能方法
        let start = Instant::now();
        let _ = self.optimized_array.boolean_index_smart(&mask);
        let smart_time = start.elapsed().as_secs_f64();
        results.insert("smart", smart_time);
        
        // 测试终极方法
        let start = Instant::now();
        let _ = self.optimized_array.boolean_index_ultimate(&mask);
        let ultimate_time = start.elapsed().as_secs_f64();
        results.insert("ultimate", ultimate_time);
        
        // 转换为Python字典
        let dict = pyo3::types::PyDict::new(py);
        for (name, time) in results {
            dict.set_item(name, time)?;
        }
        
        Ok(dict.into())
    }

    // ===========================
    // NumPy 兼容性方法
    // ===========================


}

// SIMD加速的内存复制实用函数
unsafe fn simd_copy_if_possible(src: *const u8, dst: *mut u8, size: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && size >= 32 && size % 32 == 0 {
            // 使用AVX2指令
            let chunks = size / 32;
            for i in 0..chunks {
                let src_offset = i * 32;
                let dst_offset = i * 32;
                
                let data = std::arch::x86_64::_mm256_loadu_si256(
                    src.add(src_offset) as *const std::arch::x86_64::__m256i
                );
                std::arch::x86_64::_mm256_storeu_si256(
                    dst.add(dst_offset) as *mut std::arch::x86_64::__m256i,
                    data
                );
            }
            return;
        }
    }
    
    // 回退到标准复制
    std::ptr::copy_nonoverlapping(src, dst, size);
}

fn get_array_dtype(array: &Bound<'_, PyAny>) -> PyResult<DataType> {
    let dtype_str = array.getattr("dtype")?.getattr("name")?.extract::<String>()?;
    match dtype_str.as_str() {
        "bool" => Ok(DataType::Bool),
        "uint8" => Ok(DataType::Uint8),
        "uint16" => Ok(DataType::Uint16),
        "uint32" => Ok(DataType::Uint32),
        "uint64" => Ok(DataType::Uint64),
        "int8" => Ok(DataType::Int8),
        "int16" => Ok(DataType::Int16),
        "int32" => Ok(DataType::Int32),
        "int64" => Ok(DataType::Int64),
        "float16" => Ok(DataType::Float16),
        "float32" => Ok(DataType::Float32),
        "float64" => Ok(DataType::Float64),
        _ => Err(pyo3::exceptions::PyValueError::new_err(
            format!("Unsupported dtype: {}", dtype_str)
        )),
    }
}

#[pymethods]
impl NumPack {
    #[new]
    fn new(dirname: String) -> PyResult<Self> {
        let base_dir = Path::new(&dirname);
        
        if !base_dir.exists() {
            std::fs::create_dir_all(&dirname)?;
        } 
        
        let io = ParallelIO::new(base_dir.to_path_buf())?;
        
        Ok(Self {
            io,
            base_dir: base_dir.to_path_buf(),
        })
    }

    fn save(&self, arrays: &Bound<'_, PyDict>, array_name: Option<String>) -> PyResult<()> {
        let mut bool_arrays = Vec::new();
        let mut u8_arrays = Vec::new();
        let mut u16_arrays = Vec::new();
        let mut u32_arrays = Vec::new();
        let mut u64_arrays = Vec::new();
        let mut i8_arrays = Vec::new();
        let mut i16_arrays = Vec::new();
        let mut i32_arrays = Vec::new();
        let mut i64_arrays = Vec::new();
        let mut f16_arrays = Vec::new();
        let mut f32_arrays = Vec::new();
        let mut f64_arrays = Vec::new();

        for (i, (key, value)) in arrays.iter().enumerate() {
            let name = if let Some(prefix) = &array_name {
                format!("{}{}", prefix, i)
            } else {
                key.extract::<String>()?
            };
            
            let dtype = get_array_dtype(&value)?;
            let _shape: Vec<u64> = value.getattr("shape")?
                .extract::<Vec<usize>>()?
                .into_iter()
                .map(|x| x as u64)
                .collect();

            match dtype {
                DataType::Bool => {
                    let array = value.downcast::<PyArrayDyn<bool>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    bool_arrays.push((name, array, dtype));
                }
                DataType::Uint8 => {
                    let array = value.downcast::<PyArrayDyn<u8>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    u8_arrays.push((name, array, dtype));
                }
                DataType::Uint16 => {
                    let array = value.downcast::<PyArrayDyn<u16>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    u16_arrays.push((name, array, dtype));
                }
                DataType::Uint32 => {
                    let array = value.downcast::<PyArrayDyn<u32>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    u32_arrays.push((name, array, dtype));
                }
                DataType::Uint64 => {
                    let array = value.downcast::<PyArrayDyn<u64>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    u64_arrays.push((name, array, dtype));
                }
                DataType::Int8 => {
                    let array = value.downcast::<PyArrayDyn<i8>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    i8_arrays.push((name, array, dtype));
                }
                DataType::Int16 => {
                    let array = value.downcast::<PyArrayDyn<i16>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    i16_arrays.push((name, array, dtype));
                }
                DataType::Int32 => {
                    let array = value.downcast::<PyArrayDyn<i32>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    i32_arrays.push((name, array, dtype));
                }
                DataType::Int64 => {
                    let array = value.downcast::<PyArrayDyn<i64>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    i64_arrays.push((name, array, dtype));
                }
                DataType::Float16 => {
                    let array = value.downcast::<PyArrayDyn<f16>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    f16_arrays.push((name, array, dtype));
                }
                DataType::Float32 => {
                    let array = value.downcast::<PyArrayDyn<f32>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    f32_arrays.push((name, array, dtype));
                }
                DataType::Float64 => {
                    let array = value.downcast::<PyArrayDyn<f64>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    f64_arrays.push((name, array, dtype));
                }
            }
        }

        if !bool_arrays.is_empty() {
            self.io.save_arrays(&bool_arrays)?;
        }
        if !u8_arrays.is_empty() {
            self.io.save_arrays(&u8_arrays)?;
        }
        if !u16_arrays.is_empty() {
            self.io.save_arrays(&u16_arrays)?;
        }
        if !u32_arrays.is_empty() {
            self.io.save_arrays(&u32_arrays)?;
        }
        if !u64_arrays.is_empty() {
            self.io.save_arrays(&u64_arrays)?;
        }
        if !i8_arrays.is_empty() {
            self.io.save_arrays(&i8_arrays)?;
        }
        if !i16_arrays.is_empty() {
            self.io.save_arrays(&i16_arrays)?;
        }
        if !i32_arrays.is_empty() {
            self.io.save_arrays(&i32_arrays)?;
        }
        if !i64_arrays.is_empty() {
            self.io.save_arrays(&i64_arrays)?;
        }
        if !f16_arrays.is_empty() {
            self.io.save_arrays(&f16_arrays)?;
        }
        if !f32_arrays.is_empty() {
            self.io.save_arrays(&f32_arrays)?;
        }
        if !f64_arrays.is_empty() {
            self.io.save_arrays(&f64_arrays)?;
        }

        Ok(())
    }

    #[pyo3(signature = (array_name, lazy=None))]
    fn load(&self, py: Python, array_name: &str, lazy: Option<bool>) -> PyResult<PyObject> {
        let lazy = lazy.unwrap_or(false);
        
        if !self.io.has_array(array_name) {
            return Err(PyErr::new::<PyKeyError, _>("Array not found"));
        }
        
        if lazy {
            let meta = self.io.get_array_metadata(array_name)?;
            let data_path = self.base_dir.join(format!("data_{}.npkd", array_name));
            let array_path = data_path.to_string_lossy().to_string();
            
            let mut cache = MMAP_CACHE.lock().unwrap();
            let mmap = if let Some((cached_mmap, cached_time)) = cache.get(&array_path) {
                if *cached_time == meta.last_modified as i64 {
                    Arc::clone(cached_mmap)
                } else {
                    create_optimized_mmap(&data_path, meta.last_modified as i64, &mut cache)?
                }
            } else {
                create_optimized_mmap(&data_path, meta.last_modified as i64, &mut cache)?
            };
            
            let shape: Vec<usize> = meta.shape.iter().map(|&x| x as usize).collect();
            let itemsize = meta.dtype.size_bytes() as usize;
            
            let lazy_array = LazyArray {
                mmap,
                shape,
                dtype: meta.dtype,
                itemsize,
                array_path,
                modify_time: meta.last_modified as i64,
            };
            
            return Ok(Py::new(py, lazy_array)?.into());
        }

        let meta = self.io.get_array_metadata(array_name)?;
        let data_path = self.base_dir.join(format!("data_{}.npkd", array_name));
        let shape: Vec<usize> = meta.shape.iter().map(|&x| x as usize).collect();
        
        // Use mmap to accelerate data loading
        let file = std::fs::File::open(&data_path)?;
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
        
        // Create array and copy data
        let array: PyObject = match meta.dtype {
            DataType::Bool => {
                let data = unsafe {
                    let slice = std::slice::from_raw_parts(mmap.as_ptr(), mmap.len());
                    let bool_vec: Vec<bool> = slice.iter().map(|&x| x != 0).collect();
                    ArrayD::from_shape_vec(shape, bool_vec).unwrap()
                };
                data.into_pyarray(py).into()
            }
            DataType::Uint8 => {
                let data = unsafe {
                    let slice = std::slice::from_raw_parts(mmap.as_ptr() as *const u8, mmap.len());
                    ArrayD::from_shape_vec(shape, slice.to_vec()).unwrap()
                };
                data.into_pyarray(py).into()
            }
            DataType::Uint16 => {
                let data = unsafe {
                    let slice = std::slice::from_raw_parts(mmap.as_ptr() as *const u16, mmap.len() / 2);
                    ArrayD::from_shape_vec(shape, slice.to_vec()).unwrap()
                };
                data.into_pyarray(py).into()
            }
            DataType::Uint32 => {
                let data = unsafe {
                    let slice = std::slice::from_raw_parts(mmap.as_ptr() as *const u32, mmap.len() / 4);
                    ArrayD::from_shape_vec(shape, slice.to_vec()).unwrap()
                };
                data.into_pyarray(py).into()
            }
            DataType::Uint64 => {
                let data = unsafe {
                    let slice = std::slice::from_raw_parts(mmap.as_ptr() as *const u64, mmap.len() / 8);
                    ArrayD::from_shape_vec(shape, slice.to_vec()).unwrap()
                };
                data.into_pyarray(py).into()
            }
            DataType::Int8 => {
                let data = unsafe {
                    let slice = std::slice::from_raw_parts(mmap.as_ptr() as *const i8, mmap.len());
                    ArrayD::from_shape_vec(shape, slice.to_vec()).unwrap()
                };
                data.into_pyarray(py).into()
            }
            DataType::Int16 => {
                let data = unsafe {
                    let slice = std::slice::from_raw_parts(mmap.as_ptr() as *const i16, mmap.len() / 2);
                    ArrayD::from_shape_vec(shape, slice.to_vec()).unwrap()
                };
                data.into_pyarray(py).into()
            }
            DataType::Int32 => {
                let data = unsafe {
                    let slice = std::slice::from_raw_parts(mmap.as_ptr() as *const i32, mmap.len() / 4);
                    ArrayD::from_shape_vec(shape, slice.to_vec()).unwrap()
                };
                data.into_pyarray(py).into()
            }
            DataType::Int64 => {
                let data = unsafe {
                    let slice = std::slice::from_raw_parts(mmap.as_ptr() as *const i64, mmap.len() / 8);
                    ArrayD::from_shape_vec(shape, slice.to_vec()).unwrap()
                };
                data.into_pyarray(py).into()
            }
            DataType::Float16 => {
                let data = unsafe {
                    let slice = std::slice::from_raw_parts(mmap.as_ptr() as *const f16, mmap.len() / 2);
                    ArrayD::from_shape_vec(shape, slice.to_vec()).unwrap()
                };
                data.into_pyarray(py).into()
            }
            DataType::Float32 => {
                let data = unsafe {
                    let slice = std::slice::from_raw_parts(mmap.as_ptr() as *const f32, mmap.len() / 4);
                    ArrayD::from_shape_vec(shape, slice.to_vec()).unwrap()
                };
                data.into_pyarray(py).into()
            }
            DataType::Float64 => {
                let data = unsafe {
                    let slice = std::slice::from_raw_parts(mmap.as_ptr() as *const f64, mmap.len() / 8);
                    ArrayD::from_shape_vec(shape, slice.to_vec()).unwrap()
                };
                data.into_pyarray(py).into()
            }
        };
        
        drop(mmap);
        drop(file);
        
        Ok(array)
    }

    fn get_shape(&self, py: Python, array_name: &str) -> PyResult<Py<PyTuple>> {
        if let Some(meta) = self.io.get_array_meta(array_name) {
            let shape: Vec<i64> = meta.shape.iter().map(|&x| x as i64).collect();
            let shape_tuple = PyTuple::new(py, &shape)?;
            Ok(shape_tuple.unbind())
        } else {
            Err(PyErr::new::<PyKeyError, _>(format!("Array {} not found", array_name)))
        }
    }

    fn get_metadata(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        
        let arrays = PyDict::new(py);
        for name in self.io.list_arrays() {
            if let Some(meta) = self.io.get_array_meta(&name) {
                let array_dict = PyDict::new(py);
                array_dict.set_item("shape", &meta.shape)?;
                array_dict.set_item("data_file", &meta.data_file)?;
                array_dict.set_item("last_modified", meta.last_modified)?;
                array_dict.set_item("size_bytes", meta.size_bytes)?;
                array_dict.set_item("dtype", format!("{:?}", meta.dtype))?;
                arrays.set_item(name, array_dict)?;
            }
        }
        
        dict.set_item("arrays", arrays)?;
        dict.set_item("base_dir", self.base_dir.to_string_lossy().as_ref())?;
        dict.set_item("total_arrays", self.io.list_arrays().len())?;
        
        Ok(dict.unbind().into())
    }

    fn get_member_list(&self, py: Python) -> PyResult<Py<PyList>> {
        let names = self.io.list_arrays();
        let list = PyList::new(py, names)?;
        Ok(list.unbind())
    }

    fn get_modify_time(&self, array_name: &str) -> PyResult<Option<i64>> {
        Ok(self.io.get_array_meta(array_name).map(|meta| meta.last_modified as i64))
    }

    fn reset(&self) -> PyResult<()> {
        self.io.reset()?;
        Ok(())
    }

    pub fn append(&mut self, arrays: &Bound<'_, PyDict>) -> PyResult<()> {
        // Check if the array exists and get the existing array information
        let mut existing_arrays: Vec<(String, DataType, Vec<usize>)> = Vec::new();
        
        for (key, array) in arrays.iter() {
            let name = key.extract::<String>()?;
            if let Some(meta) = self.io.get_array_meta(&name) {
                let shape: Vec<usize> = array.getattr("shape")?.extract()?;
                if meta.shape.len() != shape.len() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        format!("Dimension mismatch for array {}: expected {}, got {}", 
                            name, meta.shape.len(), shape.len())
                    ));
                }

                for (i, (&m, &s)) in meta.shape.iter().zip(shape.iter()).enumerate().skip(1) {
                    if m as usize != s {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            format!("Shape mismatch for array {} at dimension {}: expected {}, got {}", 
                                name, i, m, s)
                        ));
                    }
                }
                existing_arrays.push((name.to_string(), meta.dtype.clone(), shape));
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Array {} does not exist", name)
                ));
            }
        }

        // Start appending data
        for (key, array) in arrays.iter() {
            let name = key.extract::<String>()?;
            let meta = self.io.get_array_meta(&name).unwrap();
            let shape: Vec<usize> = array.getattr("shape")?.extract()?;
            
            // Append data to file
            let array_path = self.base_dir.join(&meta.data_file);
            let mut file = OpenOptions::new()
                .append(true)
                .open(array_path)?;
                
            match meta.dtype {
                DataType::Bool => {
                    let py_array = array.downcast::<PyArrayDyn<bool>>()?;
                    let array_ref = unsafe { py_array.as_array() };
                    let data = array_ref.as_slice().unwrap();
                    file.write_all(bytemuck::cast_slice(data))?;
                }
                DataType::Uint8 => {
                    let py_array = array.downcast::<PyArrayDyn<u8>>()?;
                    let array_ref = unsafe { py_array.as_array() };
                    let data = array_ref.as_slice().unwrap();
                    file.write_all(bytemuck::cast_slice(data))?;
                }
                DataType::Uint16 => {
                    let py_array = array.downcast::<PyArrayDyn<u16>>()?;
                    let array_ref = unsafe { py_array.as_array() };
                    let data = array_ref.as_slice().unwrap();
                    file.write_all(bytemuck::cast_slice(data))?;
                }
                DataType::Uint32 => {
                    let py_array = array.downcast::<PyArrayDyn<u32>>()?;
                    let array_ref = unsafe { py_array.as_array() };
                    let data = array_ref.as_slice().unwrap();
                    file.write_all(bytemuck::cast_slice(data))?;
                }
                DataType::Uint64 => {
                    let py_array = array.downcast::<PyArrayDyn<u64>>()?;
                    let array_ref = unsafe { py_array.as_array() };
                    let data = array_ref.as_slice().unwrap();
                    file.write_all(bytemuck::cast_slice(data))?;
                }
                DataType::Int8 => {
                    let py_array = array.downcast::<PyArrayDyn<i8>>()?;
                    let array_ref = unsafe { py_array.as_array() };
                    let data = array_ref.as_slice().unwrap();
                    file.write_all(bytemuck::cast_slice(data))?;
                }
                DataType::Int16 => {
                    let py_array = array.downcast::<PyArrayDyn<i16>>()?;
                    let array_ref = unsafe { py_array.as_array() };
                    let data = array_ref.as_slice().unwrap();
                    file.write_all(bytemuck::cast_slice(data))?;
                }
                DataType::Int32 => {
                    let py_array = array.downcast::<PyArrayDyn<i32>>()?;
                    let array_ref = unsafe { py_array.as_array() };
                    let data = array_ref.as_slice().unwrap();
                    file.write_all(bytemuck::cast_slice(data))?;
                }
                DataType::Int64 => {
                    let py_array = array.downcast::<PyArrayDyn<i64>>()?;
                    let array_ref = unsafe { py_array.as_array() };
                    let data = array_ref.as_slice().unwrap();
                    file.write_all(bytemuck::cast_slice(data))?;
                }
                DataType::Float16 => {
                    let py_array = array.downcast::<PyArrayDyn<f16>>()?;
                    let array_ref = unsafe { py_array.as_array() };
                    let data = array_ref.as_slice().unwrap();
                    file.write_all(bytemuck::cast_slice(data))?;
                }
                DataType::Float32 => {
                    let py_array = array.downcast::<PyArrayDyn<f32>>()?;
                    let array_ref = unsafe { py_array.as_array() };
                    let data = array_ref.as_slice().unwrap();
                    file.write_all(bytemuck::cast_slice(data))?;
                }
                DataType::Float64 => {
                    let py_array = array.downcast::<PyArrayDyn<f64>>()?;
                    let array_ref = unsafe { py_array.as_array() };
                    let data = array_ref.as_slice().unwrap();
                    file.write_all(bytemuck::cast_slice(data))?;
                }
            }

            // Update metadata
            let mut new_meta = meta.clone();
            new_meta.shape[0] += shape[0] as u64;
            new_meta.size_bytes = new_meta.total_elements() * new_meta.dtype.size_bytes() as u64;
            new_meta.last_modified = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
            self.io.update_array_metadata(&name, new_meta)?;
        }
        
        Ok(())
    }

    #[pyo3(signature = (array_names, indexes=None))]
    fn drop(&self, array_names: &Bound<'_, PyAny>, indexes: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        let names = if let Ok(list) = array_names.downcast::<PyList>() {
            list.iter()
                .map(|name| name.extract::<String>())
                .collect::<PyResult<Vec<_>>>()?
        } else if let Ok(name) = array_names.extract::<String>() {
            vec![name]
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "array_names must be a list of strings or a single string"
            ));
        };

        // If indexes parameter is provided, it means deleting specific rows
        if let Some(indexes) = indexes {
            for name in &names {
                if let Some(meta) = self.io.get_array_meta(name) {
                    // Get the indices of the rows to delete
                    let deleted_indices = if let Ok(slice) = indexes.downcast::<PySlice>() {
                        let start = slice.getattr("start")?.extract::<Option<i64>>()?.unwrap_or(0);
                        let stop = slice.getattr("stop")?.extract::<Option<i64>>()?.unwrap_or(meta.shape[0] as i64);
                        let step = slice.getattr("step")?.extract::<Option<i64>>()?.unwrap_or(1);
                        
                        let start = if start < 0 { meta.shape[0] as i64 + start } else { start };
                        let stop = if stop < 0 { meta.shape[0] as i64 + stop } else { stop };
                        
                        if step == 1 {
                            (start..stop).collect::<Vec<i64>>()
                        } else {
                            Vec::new()
                        }
                    } else if let Ok(indices) = indexes.extract::<Vec<i64>>() {
                        // Process negative indices
                        indices.into_iter()
                            .map(|idx| if idx < 0 { meta.shape[0] as i64 + idx } else { idx })
                            .collect()
                    } else {
                        return Err(pyo3::exceptions::PyTypeError::new_err(
                            "indexes must be a slice or list of integers"
                        ));
                    };
                    self.io.drop_arrays(name, Some(&deleted_indices))?;
                }
            }
            
            Ok(())
        } else {
            self.io.batch_drop_arrays(&names, None)?;
            Ok(())
        }
    }

    fn get_array_path(&self, array_name: &str) -> PathBuf {
        self.base_dir.join(&self.io.get_array_metadata(array_name).unwrap().data_file)
    }

    fn replace(&self, arrays: &Bound<'_, PyDict>, indexes: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        // Get the indices of the rows to replace
        let indices = if let Some(idx) = indexes {
            if let Ok(indices) = idx.extract::<Vec<i64>>() {
                indices
            } else if let Ok(slice) = idx.downcast::<PySlice>() {
                let start = slice.getattr("start")?.extract::<Option<i64>>()?.unwrap_or(0);
                let stop = slice.getattr("stop")?.extract::<Option<i64>>()?.unwrap_or(-1);
                let step = slice.getattr("step")?.extract::<Option<i64>>()?.unwrap_or(1);
                
                if step != 1 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Only step=1 is supported for slices"
                    ));
                }
                
                (start..stop).collect()
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "indexes must be a list of integers or a slice"
                ));
            }
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "indexes parameter is required for replace operation"
            ));
        };

        // Process each array to replace
        for (key, value) in arrays.iter() {
            let name = key.extract::<String>()?;
            
            // Check if the array exists
            if !self.io.has_array(&name) {
                return Err(PyErr::new::<PyKeyError, _>(format!("Array {} not found", name)));
            }
            
            let meta = self.io.get_array_meta(&name).unwrap();
            let new_shape: Vec<usize> = value.getattr("shape")?.extract()?;
            
            // Check if the dimensions match
            if new_shape.len() != meta.shape.len() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Dimension mismatch for array {}: expected {}, got {}", 
                        name, meta.shape.len(), new_shape.len())
                ));
            }
            
            // Check if the other dimensions match
            for (i, (&m, &s)) in meta.shape.iter().zip(new_shape.iter()).enumerate().skip(1) {
                if m as usize != s {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        format!("Shape mismatch for array {} at dimension {}: expected {}, got {}", 
                            name, i, m, s)
                    ));
                }
            }
            
            // Check if the indices are within bounds
            for &idx in &indices {
                let normalized_idx = if idx < 0 { meta.shape[0] as i64 + idx } else { idx };
                if normalized_idx < 0 || normalized_idx >= meta.shape[0] as i64 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        format!("Index {} is out of bounds for array {} with shape {:?}", 
                            idx, name, meta.shape)
                    ));
                }
            }
            
            // Perform the replace operation
            match meta.dtype {
                DataType::Bool => {
                    let array = value.downcast::<PyArrayDyn<bool>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    self.io.replace_rows(&name, &array, &indices)?;
                }
                DataType::Uint8 => {
                    let array = value.downcast::<PyArrayDyn<u8>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    self.io.replace_rows(&name, &array, &indices)?;
                }
                DataType::Uint16 => {
                    let array = value.downcast::<PyArrayDyn<u16>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    self.io.replace_rows(&name, &array, &indices)?;
                }
                DataType::Uint32 => {
                    let array = value.downcast::<PyArrayDyn<u32>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    self.io.replace_rows(&name, &array, &indices)?;
                }
                DataType::Uint64 => {
                    let array = value.downcast::<PyArrayDyn<u64>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    self.io.replace_rows(&name, &array, &indices)?;
                }
                DataType::Int8 => {
                    let array = value.downcast::<PyArrayDyn<i8>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    self.io.replace_rows(&name, &array, &indices)?;
                }
                DataType::Int16 => {
                    let array = value.downcast::<PyArrayDyn<i16>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    self.io.replace_rows(&name, &array, &indices)?;
                }
                DataType::Int32 => {
                    let array = value.downcast::<PyArrayDyn<i32>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    self.io.replace_rows(&name, &array, &indices)?;
                }
                DataType::Int64 => {
                    let array = value.downcast::<PyArrayDyn<i64>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    self.io.replace_rows(&name, &array, &indices)?;
                }
                DataType::Float16 => {
                    let array = value.downcast::<PyArrayDyn<f16>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    self.io.replace_rows(&name, &array, &indices)?;
                }
                DataType::Float32 => {
                    let array = value.downcast::<PyArrayDyn<f32>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    self.io.replace_rows(&name, &array, &indices)?;
                }
                DataType::Float64 => {
                    let array = value.downcast::<PyArrayDyn<f64>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    self.io.replace_rows(&name, &array, &indices)?;
                }
            }
        }
        
        Ok(())
    }

    fn getitem(&self, py: Python, array_name: &str, indices: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let meta = self.io.get_array_meta(array_name)
            .ok_or_else(|| PyErr::new::<PyKeyError, _>(format!("Array {} not found", array_name)))?;
        
        // Get the indices of the rows to read
        let indices = if let Ok(slice) = indices.downcast::<PySlice>() {
            let start = slice.getattr("start")?.extract::<Option<i64>>()?.unwrap_or(0);
            let stop = slice.getattr("stop")?.extract::<Option<i64>>()?.unwrap_or(meta.shape[0] as i64);
            let step = slice.getattr("step")?.extract::<Option<i64>>()?.unwrap_or(1);
            
            if step != 1 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Only step=1 is supported for slices"
                ));
            }
            
            (start..stop).collect::<Vec<i64>>()
        } else if let Ok(indices) = indices.extract::<Vec<i64>>() {
            indices
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "indices must be a list of integers or a slice"
            ));
        };

        // Read data
        let data = self.io.read_rows(array_name, &indices)?;
        
        // Calculate the new shape
        let mut new_shape = meta.shape.iter().map(|&x| x as usize).collect::<Vec<_>>();
        new_shape[0] = indices.len();
        
        // Create a NumPy array based on the data type
        let array: PyObject = match meta.dtype {
            DataType::Bool => {
                let array = unsafe {
                    let slice: &[u8] = bytemuck::cast_slice(&data);
                    let bool_vec: Vec<bool> = slice.iter().map(|&x| x != 0).collect();
                    ArrayD::from_shape_vec_unchecked(new_shape, bool_vec)
                };
                array.into_pyarray(py).into()
            }
            DataType::Uint8 => {
                let array = unsafe {
                    let slice: &[u8] = bytemuck::cast_slice(&data);
                    ArrayD::from_shape_vec_unchecked(new_shape, slice.to_vec())
                };
                array.into_pyarray(py).into()
            }
            DataType::Uint16 => {
                let array = unsafe {
                    let slice: &[u16] = bytemuck::cast_slice(&data);
                    ArrayD::from_shape_vec_unchecked(new_shape, slice.to_vec())
                };
                array.into_pyarray(py).into()
            }
            DataType::Uint32 => {
                let array = unsafe {
                    let slice: &[u32] = bytemuck::cast_slice(&data);
                    ArrayD::from_shape_vec_unchecked(new_shape, slice.to_vec())
                };
                array.into_pyarray(py).into()
            }
            DataType::Uint64 => {
                let array = unsafe {
                    let slice: &[u64] = bytemuck::cast_slice(&data);
                    ArrayD::from_shape_vec_unchecked(new_shape, slice.to_vec())
                };
                array.into_pyarray(py).into()
            }
            DataType::Int8 => {
                let array = unsafe {
                    let slice: &[i8] = bytemuck::cast_slice(&data);
                    ArrayD::from_shape_vec_unchecked(new_shape, slice.to_vec())
                };
                array.into_pyarray(py).into()
            }
            DataType::Int16 => {
                let array = unsafe {
                    let slice: &[i16] = bytemuck::cast_slice(&data);
                    ArrayD::from_shape_vec_unchecked(new_shape, slice.to_vec())
                };
                array.into_pyarray(py).into()
            }
            DataType::Int32 => {
                let array = unsafe {
                    let slice: &[i32] = bytemuck::cast_slice(&data);
                    ArrayD::from_shape_vec_unchecked(new_shape, slice.to_vec())
                };
                array.into_pyarray(py).into()
            }
            DataType::Int64 => {
                let array = unsafe {
                    let slice: &[i64] = bytemuck::cast_slice(&data);
                    ArrayD::from_shape_vec_unchecked(new_shape, slice.to_vec())
                };
                array.into_pyarray(py).into()
            }
            DataType::Float16 => {
                let array = unsafe {
                    let slice: &[f16] = bytemuck::cast_slice(&data);
                    ArrayD::from_shape_vec_unchecked(new_shape, slice.to_vec())
                };
                array.into_pyarray(py).into()
            }
            DataType::Float32 => {
                let array = unsafe {
                    let slice: &[f32] = bytemuck::cast_slice(&data);
                    ArrayD::from_shape_vec_unchecked(new_shape, slice.to_vec())
                };
                array.into_pyarray(py).into()
            }
            DataType::Float64 => {
                let array = unsafe {
                    let slice: &[f64] = bytemuck::cast_slice(&data);
                    ArrayD::from_shape_vec_unchecked(new_shape, slice.to_vec())
                };
                array.into_pyarray(py).into()
            }
        };
        
        Ok(array)
    }

    fn get_array_metadata(&self, array_name: &str) -> PyResult<ArrayMetadata> {
        if let Some(meta) = self.io.get_array_meta(array_name) {
            Ok(ArrayMetadata {
                shape: meta.shape.iter().map(|&x| x as i64).collect(),
                dtype: format!("{:?}", meta.dtype),
                data_file: self.base_dir.join(&meta.data_file).to_string_lossy().to_string(),
            })
        } else {
            Err(PyErr::new::<PyKeyError, _>(format!("Array {} not found", array_name)))
        }
    }

    // 创建高性能LazyArray
    fn create_high_performance_lazy_array(&self, array_name: &str) -> PyResult<HighPerformanceLazyArray> {
        if !self.io.has_array(array_name) {
            return Err(PyErr::new::<PyKeyError, _>("Array not found"));
        }

        let meta = self.io.get_array_metadata(array_name)?;
        let data_path = self.base_dir.join(format!("data_{}.npkd", array_name));
        let shape: Vec<usize> = meta.shape.iter().map(|&x| x as usize).collect();
        
        let dtype_str = match meta.dtype {
            DataType::Bool => "bool",
            DataType::Uint8 => "uint8",
            DataType::Uint16 => "uint16",
            DataType::Uint32 => "uint32",
            DataType::Uint64 => "uint64",
            DataType::Int8 => "int8",
            DataType::Int16 => "int16",
            DataType::Int32 => "int32",
            DataType::Int64 => "int64",
            DataType::Float16 => "float16",
            DataType::Float32 => "float32",
            DataType::Float64 => "float64",
        };

        HighPerformanceLazyArray::new(
            data_path.to_string_lossy().to_string(),
            shape,
            dtype_str.to_string()
        )
    }
}

fn create_optimized_mmap(path: &Path, modify_time: i64, cache: &mut MutexGuard<HashMap<String, (Arc<Mmap>, i64)>>) -> PyResult<Arc<Mmap>> {
    // Windows平台使用智能映射系统
    #[cfg(target_family = "windows")]
    {
        use crate::windows_mapping::create_intelligent_mmap;
        
        // 使用智能映射系统
        let enhanced_mapping = create_intelligent_mmap(path)?;
        let mmap = enhanced_mapping.mmap.clone();
        
        // 缓存映射结果
        let path_str = path.to_string_lossy().to_string();
        cache.insert(path_str, (mmap.clone(), modify_time));
        
        return Ok(mmap);
    }
    
    // 非Windows平台使用原始逻辑
    #[cfg(not(target_family = "windows"))]
    {
        let file = std::fs::File::open(path)?;
        let file_size = file.metadata()?.len() as usize;
        
        // Unix Linux
        #[cfg(all(target_family = "unix", target_os = "linux"))]
        unsafe {
            use std::os::unix::io::AsRawFd;
            let addr = libc::mmap(
                std::ptr::null_mut(),
                file_size,
                libc::PROT_READ,
                libc::MAP_PRIVATE | libc::MAP_HUGETLB,
                file.as_raw_fd(),
                0
            );
            
            if addr != libc::MAP_FAILED {
                libc::madvise(
                    addr,
                    file_size,
                    libc::MADV_HUGEPAGE
                );
                
                libc::madvise(
                    addr,
                    file_size,
                    libc::MADV_SEQUENTIAL | libc::MADV_WILLNEED
                );
            }
            
            libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_SEQUENTIAL);
            libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_WILLNEED);
        }
        
        // macOS
        #[cfg(all(target_family = "unix", target_os = "macos"))]
        unsafe {
            use std::os::unix::io::AsRawFd;
            let radv = libc::radvisory {
                ra_offset: 0,
                ra_count: file_size as i32,
            };
            libc::fcntl(file.as_raw_fd(), libc::F_RDADVISE, &radv);
            libc::fcntl(file.as_raw_fd(), libc::F_RDAHEAD, 1);
        }
        
        // 创建标准映射
        let mmap = unsafe { 
            memmap2::MmapOptions::new()
                .populate()
                .map(&file)?
        };
        
        let mmap = Arc::new(mmap);
        cache.insert(path.to_string_lossy().to_string(), (Arc::clone(&mmap), modify_time));
        
        Ok(mmap)
    }
}

#[pymodule]
fn _lib_numpack(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NumPack>()?;
    m.add_class::<LazyArray>()?;
    m.add_class::<LazyArrayIterator>()?;
    m.add_class::<ArrayMetadata>()?;
    m.add_class::<HighPerformanceLazyArray>()?;
    Ok(())
}

#[cfg(target_family = "windows")]
fn release_windows_file_handle(path: &Path) {
    // 使用智能系统处理文件解锁和清理
    crate::windows_mapping::execute_full_cleanup(path);
}

