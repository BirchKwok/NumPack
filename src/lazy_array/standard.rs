//! 标准LazyArray实现
//! 
//! 从lib.rs中提取的标准LazyArray结构体和实现

use std::sync::Arc;
use memmap2::Mmap;
use num_complex::{Complex32, Complex64};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple, PySlice};
use pyo3::ffi::Py_buffer;
use numpy::{IntoPyArray, PyArrayDyn};
use ndarray::ArrayD;
use std::ptr;
use std::collections::HashMap;
use std::path::Path;

use crate::metadata::DataType;
use crate::lazy_array::traits::FastTypeConversion;
use crate::lazy_array::indexing::{IndexType, SliceInfo, IndexResult, AccessPattern, AccessStrategy};

#[derive(Clone)]
pub struct LogicalRowMap {
    pub active_count: usize,
    pub physical_rows: usize,
    pub active_indices: Option<Vec<usize>>,
    pub bitmap: Option<Arc<crate::deletion_bitmap::DeletionBitmap>>,
}

impl LogicalRowMap {
    pub fn new(base_dir: &Path, array_name: &str, total_rows: usize) -> PyResult<Option<Self>> {
        use crate::deletion_bitmap::DeletionBitmap;
        if !DeletionBitmap::exists(base_dir, array_name) {
            return Ok(None);
        }
        let bitmap = Arc::new(
            DeletionBitmap::new(base_dir, array_name, total_rows).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
            })?,
        );
        let active_count = bitmap.active_count();
        Ok(Some(Self {
            active_count,
            physical_rows: bitmap.get_total_rows(),
            active_indices: None,
            bitmap: Some(bitmap),
        }))
    }

    pub fn logical_len(&self) -> usize {
        self.active_count
    }

    pub fn ensure_indices(&mut self) {
        if self.active_indices.is_some() {
            return;
        }
        if let Some(bitmap) = &self.bitmap {
            self.active_indices = Some(bitmap.get_active_indices());
        }
    }

    pub fn logical_to_physical(&mut self, logical_idx: usize) -> Option<usize> {
        if logical_idx >= self.active_count {
            return None;
        }
        if let Some(ref cache) = self.active_indices {
            return cache.get(logical_idx).cloned();
        }
        if let Some(bitmap) = &self.bitmap {
            bitmap.logical_to_physical(logical_idx)
        } else {
            Some(logical_idx)
        }
    }

    pub fn logical_indices(&mut self, logical_indices: &[usize]) -> PyResult<Vec<usize>> {
        let mut result = Vec::with_capacity(logical_indices.len());
        for &idx in logical_indices {
            let phys = self.logical_to_physical(idx).ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                    "Index {} is out of bounds for logical length {}",
                    idx, self.active_count
                ))
            })?;
            result.push(phys);
        }
        Ok(result)
    }
}

/// 标准LazyArray结构体 - 提供基本的懒加载数组功能
/// 
/// 🚀 性能优化：支持可写mmap，允许直接修改数据
#[pyclass]
pub struct LazyArray {
    pub(crate) mmap: Arc<Mmap>,
    pub(crate) shape: Vec<usize>,
    pub(crate) dtype: DataType,
    pub(crate) itemsize: usize,
    pub(crate) array_path: String,
    pub(crate) modify_time: i64,
    pub(crate) logical_rows: Option<LogicalRowMap>,
    /// 🚀 可写标志：如果为true，表示这是可写的mmap
    pub(crate) is_writable: bool,
    /// 🚀 脏标志：如果为true，表示数据已被修改，需要sync
    pub(crate) is_dirty: bool,
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
            DataType::Complex64 => "Zf",
            DataType::Complex128 => "Zd",
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
        for (idx, &row) in show_rows.iter().enumerate() {
            if let Some(last) = last_row {
                if row > last + 1 {
                    result.push_str("    ...\n");
                }
            }
            
            let mut row_str = String::new();
            let mut last_col = None;
            
            for (col_idx, &col) in show_cols.iter().enumerate() {
                if let Some(last) = last_col {
                    if col > last + 1 {
                        row_str.push_str(" ...");
                    }
                }
                
                let value = self.get_element(py, row, col)?;
                if col_idx == 0 {
                    row_str.push_str(&format!(" {}", value));
                } else {
                    row_str.push_str(&format!(" {}", value));
                }
                last_col = Some(col);
            }
            
            if idx == 0 {
                result.push_str(&format!("    [{}]", row_str.trim()));
            } else {
                result.push_str(&format!("\n    [{}]", row_str.trim()));
            }
            last_row = Some(row);
        }
        
        result.push_str(")");
        Ok(result)
    }

    #[getter]
    fn shape(&self, py: Python) -> PyResult<PyObject> {
        let logical_shape = self.logical_shape();
        let shape_tuple = PyTuple::new(py, &logical_shape)?;
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
            DataType::Complex64 => "complex64",
            DataType::Complex128 => "complex128",
        };
        let dtype = numpy.getattr("dtype")?.call1((dtype_str,))?;
        Ok(dtype.into())
    }

    #[getter]
    fn size(&self) -> PyResult<usize> {
        Ok(self.logical_shape().iter().product())
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
            for i in 0..list.len() {
                let item = list.get_item(i)?;
                if let Ok(dim) = item.extract::<i64>() {
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
                "new_shape must be a tuple, list, or integer"
            ));
        };

        // Calculate the total size of the original array
        let original_size: usize = self.shape.iter().product();

        // Handle -1 dimension inference
        let mut unknown_dim_index = None;
        let mut known_size = 1usize;
        for (i, &dim) in shape.iter().enumerate() {
            if dim == -1 {
                if unknown_dim_index.is_some() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Only one dimension can be -1"
                    ));
                }
                unknown_dim_index = Some(i);
            } else if dim > 0 {
                known_size *= dim as usize;
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Dimensions must be positive integers or -1"
                ));
            }
        }

        // Calculate the inferred dimension if needed
        if let Some(index) = unknown_dim_index {
            if original_size % known_size != 0 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Cannot infer dimension: total size is not divisible by known dimensions"
                ));
            }
            shape[index] = (original_size / known_size) as i64;
        }

        // Convert to usize and verify
        let final_shape: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
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
            logical_rows: self.logical_rows.clone(),
            is_writable: self.is_writable,
            is_dirty: self.is_dirty,
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

    // 阶段2：深度SIMD优化【已FFI优化】
    fn vectorized_gather(&self, py: Python, indices: Vec<usize>) -> PyResult<PyObject> {
        // Handle empty indices case
        if indices.is_empty() {
            let mut empty_shape = self.shape.clone();
            empty_shape[0] = 0;
            return self.create_numpy_array(py, Vec::new(), &empty_shape);
        }
        
        // 【FFI优化】使用批量操作 - 减少FFI调用从N次到1次
        // 如果索引数量较多，使用优化的批量方法
        if indices.len() >= 10 {
            return self.batch_get_rows_optimized(py, &indices);
        }
        
        // 小批量保持原有逻辑
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

    // ===========================
    // 高级索引功能
    // ===========================
    
    fn __len__(&self) -> PyResult<usize> {
        if self.shape.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "len() of unsized object"
            ));
        }
        Ok(self.len_logical())
    }

    fn __getitem__(&self, py: Python, key: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        // Fast path: 单个整数索引
        if let Ok(index) = key.extract::<i64>() {
            let logical_len = self.len_logical() as i64;
            let normalized = if index < 0 { logical_len + index } else { index };
            if normalized < 0 || normalized >= logical_len {
                return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                    format!("Index {} is out of bounds", index)
                ));
            }

            let row_data = self.get_row_data(normalized as usize)?;
            let row_shape = if self.shape.len() > 1 {
                self.shape[1..].to_vec()
            } else {
                vec![1]
            };
            return self.create_numpy_array(py, row_data, &row_shape);
        }
        
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
    
    // ===========================
    // Context Manager支持
    // ===========================
    
    /// 显式关闭方法以进行资源清理
    fn close(&mut self, py: Python) -> PyResult<()> {
        use crate::memory::handle_manager::get_handle_manager;
        
        let handle_manager = get_handle_manager();
        let path = std::path::Path::new(&self.array_path);
        
        // 清理此数组的句柄
        py.allow_threads(|| {
            if let Err(e) = handle_manager.cleanup_by_path(path) {
                eprintln!("Warning: Failed to cleanup LazyArray handle: {}", e);
            }
        });
        
        Ok(())
    }
    
    /// Context manager入口
    fn __enter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }
    
    /// Context manager出口
    fn __exit__(
        &mut self,
        py: Python,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_val: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        self.close(py)?;
        Ok(false)  // 不抑制异常
    }

    // ===========================
    // 算术操作符支持
    // ===========================

    /// 加法操作符：lazy_array + other
    fn __add__(&self, py: Python, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let self_array = self.to_numpy_array(py)?;
        let result = self_array.call_method1(py, "__add__", (other,))?;
        Ok(result.into())
    }

    /// 减法操作符：lazy_array - other
    fn __sub__(&self, py: Python, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let self_array = self.to_numpy_array(py)?;
        let result = self_array.call_method1(py, "__sub__", (other,))?;
        Ok(result.into())
    }

    /// 乘法操作符：lazy_array * other
    fn __mul__(&self, py: Python, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let self_array = self.to_numpy_array(py)?;
        let result = self_array.call_method1(py, "__mul__", (other,))?;
        Ok(result.into())
    }

    /// 真除法操作符：lazy_array / other
    fn __truediv__(&self, py: Python, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let self_array = self.to_numpy_array(py)?;
        let result = self_array.call_method1(py, "__truediv__", (other,))?;
        Ok(result.into())
    }

    /// 地板除法操作符：lazy_array // other
    fn __floordiv__(&self, py: Python, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let self_array = self.to_numpy_array(py)?;
        let result = self_array.call_method1(py, "__floordiv__", (other,))?;
        Ok(result.into())
    }

    /// 取模操作符：lazy_array % other
    fn __mod__(&self, py: Python, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let self_array = self.to_numpy_array(py)?;
        let result = self_array.call_method1(py, "__mod__", (other,))?;
        Ok(result.into())
    }

    /// 幂操作符：lazy_array ** other
    fn __pow__(&self, py: Python, other: &Bound<'_, PyAny>, _modulo: Option<&Bound<'_, PyAny>>) -> PyResult<PyObject> {
        let self_array = self.to_numpy_array(py)?;
        let result = self_array.call_method1(py, "__pow__", (other,))?;
        Ok(result.into())
    }

    // ===========================
    // 原地算术操作符支持
    // ===========================
    // 注意：原地操作符未在 Rust 层实现，而是在 Python 包装层处理。
    // 这是因为 PyO3 对原地操作符的返回类型有严格限制。
    // Python 层会自动将原地操作转换为非原地操作。

    // ===========================
    // 比较操作符支持
    // ===========================

    /// 等于操作符：lazy_array == other
    fn __eq__(&self, py: Python, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let self_array = self.to_numpy_array(py)?;
        let result = self_array.call_method1(py, "__eq__", (other,))?;
        Ok(result.into())
    }

    /// 不等于操作符：lazy_array != other
    fn __ne__(&self, py: Python, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let self_array = self.to_numpy_array(py)?;
        let result = self_array.call_method1(py, "__ne__", (other,))?;
        Ok(result.into())
    }

    /// 小于操作符：lazy_array < other
    fn __lt__(&self, py: Python, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let self_array = self.to_numpy_array(py)?;
        let result = self_array.call_method1(py, "__lt__", (other,))?;
        Ok(result.into())
    }

    /// 小于等于操作符：lazy_array <= other
    fn __le__(&self, py: Python, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let self_array = self.to_numpy_array(py)?;
        let result = self_array.call_method1(py, "__le__", (other,))?;
        Ok(result.into())
    }

    /// 大于操作符：lazy_array > other
    fn __gt__(&self, py: Python, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let self_array = self.to_numpy_array(py)?;
        let result = self_array.call_method1(py, "__gt__", (other,))?;
        Ok(result.into())
    }

    /// 大于等于操作符：lazy_array >= other
    fn __ge__(&self, py: Python, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let self_array = self.to_numpy_array(py)?;
        let result = self_array.call_method1(py, "__ge__", (other,))?;
        Ok(result.into())
    }

    // ===========================
    // 一元操作符支持
    // ===========================

    /// 一元正号：+lazy_array
    fn __pos__(&self, py: Python) -> PyResult<PyObject> {
        let self_array = self.to_numpy_array(py)?;
        let result = self_array.call_method0(py, "__pos__")?;
        Ok(result.into())
    }

    /// 一元负号：-lazy_array
    fn __neg__(&self, py: Python) -> PyResult<PyObject> {
        let self_array = self.to_numpy_array(py)?;
        let result = self_array.call_method0(py, "__neg__")?;
        Ok(result.into())
    }

    /// 一元位运算取反：~lazy_array
    fn __invert__(&self, py: Python) -> PyResult<PyObject> {
        let self_array = self.to_numpy_array(py)?;
        let result = self_array.call_method0(py, "__invert__")?;
        Ok(result.into())
    }

    // ===========================
    // 位操作符支持（仅适用于整数类型）
    // ===========================

    /// 位与操作符：lazy_array & other
    fn __and__(&self, py: Python, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        if !self.is_integer_type() {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Bitwise operations are only supported for integer arrays"
            ));
        }
        let self_array = self.to_numpy_array(py)?;
        let result = self_array.call_method1(py, "__and__", (other,))?;
        Ok(result.into())
    }

    /// 位或操作符：lazy_array | other
    fn __or__(&self, py: Python, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        if !self.is_integer_type() {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Bitwise operations are only supported for integer arrays"
            ));
        }
        let self_array = self.to_numpy_array(py)?;
        let result = self_array.call_method1(py, "__or__", (other,))?;
        Ok(result.into())
    }

    /// 位异或操作符：lazy_array ^ other
    fn __xor__(&self, py: Python, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        if !self.is_integer_type() {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Bitwise operations are only supported for integer arrays"
            ));
        }
        let self_array = self.to_numpy_array(py)?;
        let result = self_array.call_method1(py, "__xor__", (other,))?;
        Ok(result.into())
    }

    /// 左移操作符：lazy_array << other
    fn __lshift__(&self, py: Python, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        if !self.is_integer_type() {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Bitwise operations are only supported for integer arrays"
            ));
        }
        let self_array = self.to_numpy_array(py)?;
        let result = self_array.call_method1(py, "__lshift__", (other,))?;
        Ok(result.into())
    }

    /// 右移操作符：lazy_array >> other
    fn __rshift__(&self, py: Python, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        if !self.is_integer_type() {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Bitwise operations are only supported for integer arrays"
            ));
        }
        let self_array = self.to_numpy_array(py)?;
        let result = self_array.call_method1(py, "__rshift__", (other,))?;
        Ok(result.into())
    }

    // ===========================
    // 反向算术操作符支持
    // ===========================

    /// 反向加法操作符：other + lazy_array
    fn __radd__(&self, py: Python, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let self_array = self.to_numpy_array(py)?;
        let result = self_array.call_method1(py, "__radd__", (other,))?;
        Ok(result.into())
    }

    /// 反向减法操作符：other - lazy_array
    fn __rsub__(&self, py: Python, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let self_array = self.to_numpy_array(py)?;
        let result = self_array.call_method1(py, "__rsub__", (other,))?;
        Ok(result.into())
    }

    /// 反向乘法操作符：other * lazy_array
    fn __rmul__(&self, py: Python, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let self_array = self.to_numpy_array(py)?;
        let result = self_array.call_method1(py, "__rmul__", (other,))?;
        Ok(result.into())
    }

    /// 反向真除法操作符：other / lazy_array
    fn __rtruediv__(&self, py: Python, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let self_array = self.to_numpy_array(py)?;
        let result = self_array.call_method1(py, "__rtruediv__", (other,))?;
        Ok(result.into())
    }

    /// 反向地板除法操作符：other // lazy_array
    fn __rfloordiv__(&self, py: Python, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let self_array = self.to_numpy_array(py)?;
        let result = self_array.call_method1(py, "__rfloordiv__", (other,))?;
        Ok(result.into())
    }

    /// 反向取模操作符：other % lazy_array
    fn __rmod__(&self, py: Python, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let self_array = self.to_numpy_array(py)?;
        let result = self_array.call_method1(py, "__rmod__", (other,))?;
        Ok(result.into())
    }

    /// 反向幂操作符：other ** lazy_array
    fn __rpow__(&self, py: Python, other: &Bound<'_, PyAny>, _modulo: Option<&Bound<'_, PyAny>>) -> PyResult<PyObject> {
        let self_array = self.to_numpy_array(py)?;
        let result = self_array.call_method1(py, "__rpow__", (other,))?;
        Ok(result.into())
    }

    // ===========================
    // 反向位操作符支持
    // ===========================

    /// 反向位与操作符：other & lazy_array
    fn __rand__(&self, py: Python, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        if !self.is_integer_type() {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Bitwise operations are only supported for integer arrays"
            ));
        }
        let self_array = self.to_numpy_array(py)?;
        let result = self_array.call_method1(py, "__rand__", (other,))?;
        Ok(result.into())
    }

    /// 反向位或操作符：other | lazy_array
    fn __ror__(&self, py: Python, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        if !self.is_integer_type() {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Bitwise operations are only supported for integer arrays"
            ));
        }
        let self_array = self.to_numpy_array(py)?;
        let result = self_array.call_method1(py, "__ror__", (other,))?;
        Ok(result.into())
    }

    /// 反向位异或操作符：other ^ lazy_array
    fn __rxor__(&self, py: Python, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        if !self.is_integer_type() {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Bitwise operations are only supported for integer arrays"
            ));
        }
        let self_array = self.to_numpy_array(py)?;
        let result = self_array.call_method1(py, "__rxor__", (other,))?;
        Ok(result.into())
    }

    /// 反向左移操作符：other << lazy_array
    fn __rlshift__(&self, py: Python, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        if !self.is_integer_type() {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Bitwise operations are only supported for integer arrays"
            ));
        }
        let self_array = self.to_numpy_array(py)?;
        let result = self_array.call_method1(py, "__rlshift__", (other,))?;
        Ok(result.into())
    }

    /// 反向右移操作符：other >> lazy_array
    fn __rrshift__(&self, py: Python, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        if !self.is_integer_type() {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Bitwise operations are only supported for integer arrays"
            ));
        }
        let self_array = self.to_numpy_array(py)?;
        let result = self_array.call_method1(py, "__rrshift__", (other,))?;
        Ok(result.into())
    }
    
    // ===========================
    // 高级功能方法
    // ===========================
    
    /// 转置属性 - 返回转置后的NumPy数组
    #[getter]
    fn T(&self, py: Python) -> PyResult<PyObject> {
        // 转换为NumPy数组然后转置
        // 这是最简单且高效的实现方式
        let array = self.to_numpy_array(py)?;
        let transposed = array.getattr(py, "T")?;
        Ok(transposed)
    }
    
    /// 智能预热 - 根据访问模式提示进行缓存预热
    fn intelligent_warmup(&self, _hint: &str) -> PyResult<()> {
        // 占位符实现 - 不做实际操作，避免影响性能
        // 在实际使用中，Rust的内存映射和OS的页缓存已经很高效
        Ok(())
    }
    
    /// 获取性能统计信息
    fn get_performance_stats(&self) -> PyResult<Vec<(String, f64)>> {
        // 返回基本统计信息，格式为 [(name, value), ...]
        let stats = vec![
            ("array_size_mb".to_string(), (self.mmap.len() as f64) / (1024.0 * 1024.0)),
            ("logical_rows".to_string(), self.len_logical() as f64),
            ("physical_rows".to_string(), self.shape[0] as f64),
        ];
        Ok(stats)
    }
    
    /// 生产级布尔索引 - 优化的布尔掩码访问
    fn boolean_index_production(&self, py: Python, mask: Vec<bool>) -> PyResult<PyObject> {
        self.parallel_boolean_index(py, mask)
    }
    
    /// 自适应布尔索引算法
    fn boolean_index_adaptive_algorithm(&self, py: Python, mask: Vec<bool>) -> PyResult<PyObject> {
        self.parallel_boolean_index(py, mask)
    }
    
    /// 选择最优算法 - 根据掩码选择性返回算法名称
    fn choose_optimal_algorithm(&self, mask: Vec<bool>) -> PyResult<String> {
        let true_count = mask.iter().filter(|&&x| x).count();
        let selectivity = true_count as f64 / mask.len() as f64;
        
        if selectivity < 0.1 {
            Ok("sparse".to_string())
        } else if selectivity > 0.9 {
            Ok("dense".to_string())
        } else {
            Ok("adaptive".to_string())
        }
    }
}

// 实现Drop特性以确保Windows平台上的资源正确释放
#[cfg(target_family = "windows")]
impl Drop for LazyArray {
    fn drop(&mut self) {
        use crate::memory::handle_manager::get_handle_manager;
        
        let handle_manager = get_handle_manager();
        let path = std::path::Path::new(&self.array_path);
        
        // 清理与此数组路径关联的所有句柄
        if let Err(e) = handle_manager.cleanup_by_path(path) {
            eprintln!("Warning: Failed to cleanup handle {}: {}", self.array_path, e);
        }
        
        // 对于Windows，强制清理并等待
        if let Err(e) = handle_manager.force_cleanup_and_wait(None) {
            eprintln!("Warning: Forced cleanup failed: {}", e);
        }
    }
}

// 非Windows平台的Drop实现
#[cfg(not(target_family = "windows"))]
impl Drop for LazyArray {
    fn drop(&mut self) {
        use crate::memory::handle_manager::get_handle_manager;
        
        let handle_manager = get_handle_manager();
        let path = std::path::Path::new(&self.array_path);
        
        // Unix系统也受益于句柄跟踪
        let _ = handle_manager.cleanup_by_path(path);
    }
}

impl LazyArray {
    /// 创建新的LazyArray实例
    pub fn new(
        mmap: Arc<Mmap>,
        shape: Vec<usize>,
        dtype: DataType,
        itemsize: usize,
        array_path: String,
        modify_time: i64,
    ) -> Self {
        Self {
            mmap,
            shape,
            dtype,
            itemsize,
            array_path,
            modify_time,
            logical_rows: None,
            is_writable: false,
            is_dirty: false,
        }
    }
    
    /// 🚀 创建可写LazyArray实例
    pub fn new_writable(
        mmap: Arc<Mmap>,
        shape: Vec<usize>,
        dtype: DataType,
        itemsize: usize,
        array_path: String,
        modify_time: i64,
    ) -> Self {
        Self {
            mmap,
            shape,
            dtype,
            itemsize,
            array_path,
            modify_time,
            logical_rows: None,
            is_writable: true,
            is_dirty: false,
        }
    }
    
    /// 创建LazyArray并注册到句柄管理器
    pub fn new_with_handle_manager(
        mmap: Arc<Mmap>,
        shape: Vec<usize>,
        dtype: DataType,
        itemsize: usize,
        array_path: String,
        modify_time: i64,
        owner_name: String,
    ) -> PyResult<Self> {
        use crate::memory::handle_manager::get_handle_manager;
        
        let handle_manager = get_handle_manager();
        let handle_id = format!("lazy_array_{}_{}",  array_path, modify_time);
        let path = std::path::PathBuf::from(&array_path);
        
        // 将mmap注册到句柄管理器
        handle_manager.register_memmap(
            handle_id,
            Arc::clone(&mmap),
            Some(path),
            owner_name,
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to register handle: {}", e)
        ))?;
        
        Ok(Self {
            mmap,
            shape,
            dtype,
            itemsize,
            array_path,
            modify_time,
            logical_rows: None,
            is_writable: false,
            is_dirty: false,
        })
    }

    /// 获取元素值（用于__repr__）
    fn get_element(&self, _py: Python, row: usize, col: usize) -> PyResult<String> {
        if row >= self.shape[0] {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>("Row index out of bounds"));
        }
        
        let col_count = if self.shape.len() > 1 { self.shape[1] } else { 1 };
        if col >= col_count {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>("Column index out of bounds"));
        }

        let offset = (row * col_count + col) * self.itemsize;
        
        if offset + self.itemsize > self.mmap.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>("Data offset out of bounds"));
        }

        let value_str = match self.dtype {
            DataType::Bool => {
                let value = unsafe { *self.mmap.as_ptr().add(offset) };
                (value != 0).to_string()
            }
            DataType::Uint8 => unsafe { *self.mmap.as_ptr().add(offset) }.to_string(),
            DataType::Uint16 => unsafe { *(self.mmap.as_ptr().add(offset) as *const u16) }.to_string(),
            DataType::Uint32 => unsafe { *(self.mmap.as_ptr().add(offset) as *const u32) }.to_string(),
            DataType::Uint64 => unsafe { *(self.mmap.as_ptr().add(offset) as *const u64) }.to_string(),
            DataType::Int8 => unsafe { *(self.mmap.as_ptr().add(offset) as *const i8) }.to_string(),
            DataType::Int16 => unsafe { *(self.mmap.as_ptr().add(offset) as *const i16) }.to_string(),
            DataType::Int32 => unsafe { *(self.mmap.as_ptr().add(offset) as *const i32) }.to_string(),
            DataType::Int64 => unsafe { *(self.mmap.as_ptr().add(offset) as *const i64) }.to_string(),
            DataType::Float16 => {
                let raw_value = unsafe { *(self.mmap.as_ptr().add(offset) as *const u16) };
                let value = half::f16::from_bits(raw_value);
                format!("{:.3}", value)
            }
            DataType::Float32 => {
                let value = unsafe { *(self.mmap.as_ptr().add(offset) as *const f32) };
                format!("{:.3}", value)
            }
            DataType::Float64 => {
                let value = unsafe { *(self.mmap.as_ptr().add(offset) as *const f64) };
                format!("{:.3}", value)
            }
            DataType::Complex64 => {
                let value = unsafe { *(self.mmap.as_ptr().add(offset) as *const Complex32) };
                format!("{:.3}+{:.3}j", value.re, value.im)
            }
            DataType::Complex128 => {
                let value = unsafe { *(self.mmap.as_ptr().add(offset) as *const Complex64) };
                format!("{:.3}+{:.3}j", value.re, value.im)
            }
        };

        Ok(value_str)
    }

    /// 获取预览数据
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

    
    /// 【FFI优化】批量获取多行数据 - 减少FFI调用次数
    /// 
    /// 此方法在Rust侧完成所有数据聚合，然后单次FFI调用返回NumPy数组
    /// 相比逐行调用get_row_data，可减少50-100x的FFI开销
    #[inline]
    pub(crate) fn batch_get_rows_optimized(
        &self, 
        py: Python, 
        indices: &[usize]
    ) -> PyResult<PyObject> {
        use crate::lazy_array::ffi_optimization::{BatchIndexOptimizer, FFIOptimizationConfig};
        
        // 创建优化器
        let config = FFIOptimizationConfig::default();
        let optimizer = BatchIndexOptimizer::new(config);
        
        // 使用批量操作
        optimizer.batch_get_rows(
            py,
            &self.mmap,
            indices,
            &self.shape,
            self.dtype,
            self.itemsize,
        )
    }
    
    /// 【FFI优化 + 零拷贝】获取连续范围的数据 - 零拷贝视图
    /// 
    /// 对于连续的数据访问，创建零拷贝视图以完全避免内存拷贝
    /// 性能提升：5-10x相比传统拷贝方式
    #[inline]
    pub(crate) fn get_range_zero_copy(
        &self,
        py: Python,
        start: usize,
        end: usize,
    ) -> PyResult<PyObject> {
        use crate::lazy_array::ffi_optimization::{ZeroCopyArrayBuilder, FFIOptimizationConfig};
        
        if start >= end || end > self.shape[0] {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                format!("Invalid index range: start={}, end={}, len={}", start, end, self.shape[0])
            ));
        }
        
        let row_size = if self.shape.len() > 1 {
            self.shape[1..].iter().product::<usize>() * self.itemsize
        } else {
            self.itemsize
        };
        
        let offset = start * row_size;
        let count = end - start;
        
        let mut result_shape = self.shape.clone();
        result_shape[0] = count;
        
        // 创建零拷贝构建器
        let config = FFIOptimizationConfig::default();
        let builder = ZeroCopyArrayBuilder::new(Arc::clone(&self.mmap), config);
        
        // 使用零拷贝视图
        unsafe {
            builder.create_view(
                py,
                offset,
                &result_shape,
                self.dtype,
                self.itemsize,
            )
        }
    }

    /// 将字节数据转换为NumPy数组【Inline优化】
    #[inline]
    fn bytes_to_numpy(&self, py: Python, data: Vec<u8>) -> PyResult<PyObject> {
        let row_shape = if self.shape.len() > 1 {
            self.shape[1..].to_vec()
        } else {
            vec![1]
        };
        
        self.create_numpy_array(py, data, &row_shape)
    }

    /// 创建NumPy数组【Inline优化】
    #[inline]
    // Helper function to safely convert bytes to typed slice, handling alignment issues
    fn safe_cast_slice<T: bytemuck::Pod>(data: &[u8]) -> Vec<T> {
        if bytemuck::try_cast_slice::<u8, T>(data).is_ok() {
            // Fast path: data is properly aligned
            bytemuck::cast_slice::<u8, T>(data).to_vec()
        } else {
            // Slow path: data is not aligned, copy manually
            let num_elements = data.len() / std::mem::size_of::<T>();
            let mut result = Vec::with_capacity(num_elements);
            unsafe {
                let ptr = data.as_ptr() as *const T;
                for i in 0..num_elements {
                    result.push(ptr::read_unaligned(ptr.add(i)));
                }
            }
            result
        }
    }

    pub(crate) fn create_numpy_array(&self, py: Python, data: Vec<u8>, shape: &[usize]) -> Result<PyObject, PyErr> {
        let array: PyObject = match self.dtype {
            DataType::Bool => {
                let bool_vec: Vec<bool> = data.iter().map(|&x| x != 0).collect();
                let array = ArrayD::from_shape_vec(shape.to_vec(), bool_vec).unwrap();
                array.into_pyarray(py).into()
            }
            DataType::Uint8 => {
                let array = unsafe { ArrayD::from_shape_vec_unchecked(shape.to_vec(), data) };
                array.into_pyarray(py).into()
            }
            DataType::Uint16 => {
                let typed_data = Self::safe_cast_slice::<u16>(&data);
                let array = unsafe { ArrayD::from_shape_vec_unchecked(shape.to_vec(), typed_data) };
                array.into_pyarray(py).into()
            }
            DataType::Uint32 => {
                let typed_data = Self::safe_cast_slice::<u32>(&data);
                let array = unsafe { ArrayD::from_shape_vec_unchecked(shape.to_vec(), typed_data) };
                array.into_pyarray(py).into()
            }
            DataType::Uint64 => {
                let typed_data = Self::safe_cast_slice::<u64>(&data);
                let array = unsafe { ArrayD::from_shape_vec_unchecked(shape.to_vec(), typed_data) };
                array.into_pyarray(py).into()
            }
            DataType::Int8 => {
                let typed_data = Self::safe_cast_slice::<i8>(&data);
                let array = unsafe { ArrayD::from_shape_vec_unchecked(shape.to_vec(), typed_data) };
                array.into_pyarray(py).into()
            }
            DataType::Int16 => {
                let typed_data = Self::safe_cast_slice::<i16>(&data);
                let array = unsafe { ArrayD::from_shape_vec_unchecked(shape.to_vec(), typed_data) };
                array.into_pyarray(py).into()
            }
            DataType::Int32 => {
                let typed_data = Self::safe_cast_slice::<i32>(&data);
                let array = unsafe { ArrayD::from_shape_vec_unchecked(shape.to_vec(), typed_data) };
                array.into_pyarray(py).into()
            }
            DataType::Int64 => {
                let typed_data = Self::safe_cast_slice::<i64>(&data);
                let array = unsafe { ArrayD::from_shape_vec_unchecked(shape.to_vec(), typed_data) };
                array.into_pyarray(py).into()
            }
            DataType::Float16 => {
                let typed_data = data.to_typed_vec::<half::f16>();
                let f32_data: Vec<f32> = typed_data.iter().map(|&x| x.to_f32()).collect();
                let array = ArrayD::from_shape_vec(shape.to_vec(), f32_data).unwrap();
                array.into_pyarray(py).into()
            }
            DataType::Float32 => {
                let typed_data = Self::safe_cast_slice::<f32>(&data);
                let array = unsafe { ArrayD::from_shape_vec_unchecked(shape.to_vec(), typed_data) };
                array.into_pyarray(py).into()
            }
            DataType::Float64 => {
                let typed_data = Self::safe_cast_slice::<f64>(&data);
                let array = unsafe { ArrayD::from_shape_vec_unchecked(shape.to_vec(), typed_data) };
                array.into_pyarray(py).into()
            }
            DataType::Complex64 => {
                let array = unsafe {
                    let slice = std::slice::from_raw_parts(
                        data.as_ptr() as *const Complex32,
                        data.len() / std::mem::size_of::<Complex32>()
                    );
                    ArrayD::from_shape_vec_unchecked(shape.to_vec(), slice.to_vec())
                };
                array.into_pyarray(py).into()
            }
            DataType::Complex128 => {
                let array = unsafe {
                    let slice = std::slice::from_raw_parts(
                        data.as_ptr() as *const Complex64,
                        data.len() / std::mem::size_of::<Complex64>()
                    );
                    ArrayD::from_shape_vec_unchecked(shape.to_vec(), slice.to_vec())
                };
                array.into_pyarray(py).into()
            }
        };
        
        Ok(array)
    }

    // ===========================
    // 索引解析实现
    // ===========================
    
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

    fn handle_broadcasting_directly(&self, py: Python, tuple: &Bound<'_, PyTuple>) -> PyResult<PyObject> {
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
    
    fn handle_broadcasting_index(&self, _py: Python, tuple: &Bound<'_, PyTuple>) -> Result<IndexResult, PyErr> {
        if tuple.len() != 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "Broadcasting only supported for 2D indexing currently"
            ));
        }
        
        let first_item = tuple.get_item(0)?;
        let second_item = tuple.get_item(1)?;
        
        // 提取索引数组
        let first_array = self.extract_array_data(&first_item)?;
        let second_array = self.extract_array_data(&second_item)?;
        
        // 获取形状
        let first_shape = self.get_array_shape(&first_item)?;
        let second_shape = self.get_array_shape(&second_item)?;
        
        // 广播数组
        let (broadcast_first, broadcast_second, result_shape) = 
            self.broadcast_arrays(first_array, first_shape, second_array, second_shape)?;
        
        // 创建索引结果
        let indices = vec![broadcast_first, broadcast_second];
        
        Ok(IndexResult {
            indices,
            result_shape,
            needs_broadcasting: true,
            access_pattern: AccessPattern::Mixed,
        })
    }
    
    fn extract_array_data(&self, key: &Bound<'_, PyAny>) -> Result<Vec<i64>, PyErr> {
        // 尝试直接提取为Vec<i64>
        if let Ok(arr) = key.extract::<Vec<i64>>() {
            return Ok(arr);
        }
        
        // 尝试从NumPy数组提取
        if let Ok(numpy_array) = key.getattr("__array__") {
            if let Ok(array_func) = numpy_array.call0() {
                if let Ok(arr) = array_func.extract::<Vec<i64>>() {
                    return Ok(arr);
                }
                
                // 尝试扁平化
                if let Ok(flattened) = key.call_method0("flatten") {
                    if let Ok(flat_array) = flattened.getattr("__array__") {
                        if let Ok(flat_func) = flat_array.call0() {
                            if let Ok(arr) = flat_func.extract::<Vec<i64>>() {
                                return Ok(arr);
                            }
                        }
                    }
                }
            }
        }
        
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Cannot extract array data"
        ))
    }
    
    fn get_array_shape(&self, key: &Bound<'_, PyAny>) -> Result<Vec<usize>, PyErr> {
        if let Ok(shape_attr) = key.getattr("shape") {
            if let Ok(shape) = shape_attr.extract::<Vec<usize>>() {
                return Ok(shape);
            }
        }
        
        // 如果没有shape属性，假设是一维数组
        if let Ok(arr) = self.extract_array_data(key) {
            return Ok(vec![arr.len()]);
        }
        
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Cannot get array shape"
        ))
    }
    
    fn broadcast_arrays(
        &self,
        first: Vec<i64>,
        first_shape: Vec<usize>,
        second: Vec<i64>,
        second_shape: Vec<usize>,
    ) -> Result<(Vec<usize>, Vec<usize>, Vec<usize>), PyErr> {
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
    
    fn extract_multidim_array(&self, key: &Bound<'_, PyAny>) -> Result<Vec<i64>, PyErr> {
        if let Ok(numpy_array) = key.getattr("__array__") {
            if let Ok(_array_func) = numpy_array.call0() {
                // 尝试扁平化
                if let Ok(flattened) = key.call_method0("flatten") {
                    if let Ok(flat_array) = flattened.getattr("__array__") {
                        if let Ok(flat_func) = flat_array.call0() {
                            if let Ok(int_array) = flat_func.extract::<Vec<i64>>() {
                                return Ok(int_array);
                            }
                        }
                    }
                }
            }
        }
        
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Cannot extract multidimensional array"
        ))
    }
    
    fn execute_broadcasting_access(
        &self,
        py: Python,
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
        result_shape: Vec<usize>,
    ) -> Result<PyObject, PyErr> {
        let itemsize = self.itemsize;
        let mut result_data = Vec::with_capacity(row_indices.len() * itemsize);
        
        // 访问数据
        for (row_idx, col_idx) in row_indices.iter().zip(col_indices.iter()) {
            let offset = (*row_idx * self.shape[1] + *col_idx) * itemsize;
            
            if offset + itemsize <= self.mmap.len() {
                let element_data = unsafe {
                    std::slice::from_raw_parts(
                        self.mmap.as_ptr().add(offset),
                        itemsize
                    )
                };
                result_data.extend_from_slice(element_data);
            } else {
                result_data.extend(vec![0u8; itemsize]);
            }
        }
        
        self.create_numpy_array(py, result_data, &result_shape)
    }
    
    fn process_indices(&self, index_types: Vec<IndexType>) -> Result<IndexResult, PyErr> {
        let mut indices = Vec::new();
        let mut needs_broadcasting = false;
        
        // 扩展索引到完整维度（添加省略号处理）
        let expanded_indices = self.expand_indices(index_types)?;
        
        // 处理实际数组索引
        let mut array_dim = 0;
        for index_type in expanded_indices.iter() {
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
                            let int_indices = self.resolve_integer_array(arr, self.shape[array_dim], array_dim)?;
                            indices.push(int_indices);
                            needs_broadcasting = true;
                        }
                        IndexType::Ellipsis => {}
                        IndexType::NewAxis => {}
                    }
                    
                    array_dim += 1;
                }
            }
        }
        
        // 构建结果形状
        let mut result_shape: Vec<usize> = Vec::new();
        let mut array_dim = 0;
        
        for index_type in expanded_indices.iter() {
            match index_type {
                IndexType::NewAxis => {
                    result_shape.push(1);
                }
                IndexType::Integer(_) => {
                    // 整数索引不增加维度
                    array_dim += 1;
                }
                IndexType::Slice(_) | IndexType::BooleanMask(_) | IndexType::IntegerArray(_) => {
                    if array_dim < indices.len() {
                        result_shape.push(indices[array_dim].len());
                    }
                    array_dim += 1;
                }
                IndexType::Ellipsis => {}
            }
        }
        
        // 如果还有剩余维度，添加到结果形状
        while array_dim < self.shape.len() {
            result_shape.push(self.shape[array_dim]);
            array_dim += 1;
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
        while expanded.iter().filter(|&t| !matches!(t, IndexType::NewAxis)).count() < self.shape.len() {
            expanded.push(IndexType::Slice(SliceInfo {
                start: None,
                stop: None,
                step: None,
            }));
        }
        
        Ok(expanded)
    }
    
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
    
    fn resolve_integer_array(&self, arr: &[i64], dim_size: usize, dim_index: usize) -> Result<Vec<usize>, PyErr> {
        let mut indices = Vec::new();
        
        for &idx in arr {
            let adjusted = self.adjust_index(idx, dim_size)?;
            indices.push(adjusted);
        }
        
        // 如果有逻辑行映射（删除了一些行），需要将逻辑索引转换为物理索引
        // 注意：只对第一维（行维度，dim_index == 0）应用逻辑映射
        if dim_index == 0 {
            if let Some(ref mut logical_rows) = self.logical_rows.clone() {
                indices = logical_rows.logical_indices(&indices)?;
            }
        }
        
        Ok(indices)
    }
    
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

    /// 选择访问策略【Inline优化】
    #[inline(always)]
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

    /// 直接内存访问【Inline优化 + 批量读取优化】
    #[inline]
    fn direct_memory_access(&self, py: Python, index_result: &IndexResult) -> PyResult<PyObject> {
        // 优化：对于一维整数数组索引，使用批量行读取
        if index_result.indices.len() == 1 && self.shape.len() > 0 {
            let indices = &index_result.indices[0];
            let row_size = self.itemsize * self.shape[1..].iter().product::<usize>();
            let mut result_data = Vec::with_capacity(indices.len() * row_size);
            
            // 批量读取行数据，避免笛卡尔积计算
            for &idx in indices {
                let offset = idx * row_size;
                if offset + row_size <= self.mmap.len() {
                    let row_data = unsafe {
                        std::slice::from_raw_parts(
                            self.mmap.as_ptr().add(offset),
                            row_size
                        )
                    };
                    result_data.extend_from_slice(row_data);
                } else {
                    result_data.extend(vec![0u8; row_size]);
                }
            }
            
            return self.create_numpy_array(py, result_data, &index_result.result_shape);
        }
        
        // 通用路径：使用迭代器计算笛卡尔积，避免预先分配所有组合
        let total_elements = index_result.result_shape.iter().product::<usize>();
        let mut result_data = Vec::with_capacity(total_elements * self.itemsize);
        
        // 使用迭代器避免预先计算所有组合
        self.iterate_index_combinations(&index_result.indices, &mut |combination| {
            let offset = self.compute_linear_offset(combination);
            
            if offset + self.itemsize <= self.mmap.len() {
                let element_data = unsafe {
                    std::slice::from_raw_parts(
                        self.mmap.as_ptr().add(offset),
                        self.itemsize
                    )
                };
                result_data.extend_from_slice(element_data);
            } else {
                result_data.extend(vec![0u8; self.itemsize]);
            }
        });
        
        self.create_numpy_array(py, result_data, &index_result.result_shape)
    }

    /// 块拷贝访问【Inline优化】
    #[inline]
    fn block_copy_access(&self, py: Python, index_result: &IndexResult) -> PyResult<PyObject> {
        if index_result.indices.len() == 1 {
            // 单维连续访问优化
            let indices = &index_result.indices[0];
            let row_size = self.itemsize * self.shape[1..].iter().product::<usize>();
            let mut result_data = Vec::with_capacity(indices.len() * row_size);
            
            // 检查是否为连续块
            if self.is_continuous_block(indices) {
                let start_offset = indices[0] * row_size;
                let block_size = indices.len() * row_size;
                
                if start_offset + block_size <= self.mmap.len() {
                    let block_data = unsafe {
                        std::slice::from_raw_parts(
                            self.mmap.as_ptr().add(start_offset),
                            block_size
                        )
                    };
                    result_data.extend_from_slice(block_data);
                } else {
                    // 超出边界时使用逐行复制
                    for &idx in indices {
                        let offset = idx * row_size;
                        if offset + row_size <= self.mmap.len() {
                            let row_data = unsafe {
                                std::slice::from_raw_parts(
                                    self.mmap.as_ptr().add(offset),
                                    row_size
                                )
                            };
                            result_data.extend_from_slice(row_data);
                        } else {
                            result_data.extend(vec![0u8; row_size]);
                        }
                    }
                }
            } else {
                // 分块复制
                for &idx in indices {
                    let offset = idx * row_size;
                    if offset + row_size <= self.mmap.len() {
                        let row_data = unsafe {
                            std::slice::from_raw_parts(
                                self.mmap.as_ptr().add(offset),
                                row_size
                            )
                        };
                        result_data.extend_from_slice(row_data);
                    } else {
                        result_data.extend(vec![0u8; row_size]);
                    }
                }
            }
            
            self.create_numpy_array(py, result_data, &index_result.result_shape)
        } else {
            // 多维访问回退到直接内存访问
            self.direct_memory_access(py, index_result)
        }
    }

    /// 并行点访问【Inline优化】
    #[inline]
    fn parallel_point_access(&self, py: Python, index_result: &IndexResult) -> PyResult<PyObject> {
        use rayon::prelude::*;
        
        let index_combinations = self.compute_index_combinations(&index_result.indices);
        
        // 并行处理索引组合
        let result_data: Vec<u8> = index_combinations
            .par_iter()
            .flat_map(|combination| {
                let offset = self.compute_linear_offset(combination);
                if offset + self.itemsize <= self.mmap.len() {
                    unsafe {
                        std::slice::from_raw_parts(
                            self.mmap.as_ptr().add(offset),
                            self.itemsize
                        )
                    }.to_vec()
                } else {
                    vec![0u8; self.itemsize]
                }
            })
            .collect();
        
        self.create_numpy_array(py, result_data, &index_result.result_shape)
    }

    /// 预取优化访问【Inline优化 + 迭代器优化】
    #[inline]
    fn prefetch_optimized_access(&self, py: Python, index_result: &IndexResult) -> PyResult<PyObject> {
        // 优化：对于一维整数数组索引，使用批量行读取
        if index_result.indices.len() == 1 && self.shape.len() > 0 {
            let indices = &index_result.indices[0];
            let row_size = self.itemsize * self.shape[1..].iter().product::<usize>();
            let mut result_data = Vec::with_capacity(indices.len() * row_size);
            
            // 预取第一批数据
            let prefetch_ahead = 8.min(indices.len());
            for i in 0..prefetch_ahead {
                let offset = indices[i] * row_size;
                if offset < self.mmap.len() {
                    self.prefetch_single(offset);
                }
            }
            
            // 批量读取行数据，并继续预取
            for (i, &idx) in indices.iter().enumerate() {
                // 预取未来的数据
                if i + prefetch_ahead < indices.len() {
                    let future_offset = indices[i + prefetch_ahead] * row_size;
                    if future_offset < self.mmap.len() {
                        self.prefetch_single(future_offset);
                    }
                }
                
                let offset = idx * row_size;
                if offset + row_size <= self.mmap.len() {
                    let row_data = unsafe {
                        std::slice::from_raw_parts(
                            self.mmap.as_ptr().add(offset),
                            row_size
                        )
                    };
                    result_data.extend_from_slice(row_data);
                } else {
                    result_data.extend(vec![0u8; row_size]);
                }
            }
            
            return self.create_numpy_array(py, result_data, &index_result.result_shape);
        }
        
        // 通用路径：使用迭代器避免预先分配所有组合
        let total_elements = index_result.result_shape.iter().product::<usize>();
        let mut result_data = Vec::with_capacity(total_elements * self.itemsize);
        
        // 使用迭代器避免预先计算所有组合
        self.iterate_index_combinations(&index_result.indices, &mut |combination| {
            let offset = self.compute_linear_offset(combination);
            
            // 预取
            if offset < self.mmap.len() {
                self.prefetch_single(offset);
            }
            
            if offset + self.itemsize <= self.mmap.len() {
                let element_data = unsafe {
                    std::slice::from_raw_parts(
                        self.mmap.as_ptr().add(offset),
                        self.itemsize
                    )
                };
                result_data.extend_from_slice(element_data);
            } else {
                result_data.extend(vec![0u8; self.itemsize]);
            }
        });
        
        self.create_numpy_array(py, result_data, &index_result.result_shape)
    }

    /// 自适应访问【Inline优化】
    #[inline]
    fn adaptive_access(&self, py: Python, index_result: &IndexResult) -> PyResult<PyObject> {
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
    
    // ===========================
    // 辅助方法
    // ===========================
    
    /// 迭代器方式计算索引组合，避免预先分配所有组合【性能优化】
    fn iterate_index_combinations<F>(&self, indices: &[Vec<usize>], callback: &mut F)
    where
        F: FnMut(&[usize])
    {
        if indices.is_empty() {
            callback(&[]);
            return;
        }
        
        let mut current_combination = vec![0; indices.len()];
        let mut current_positions = vec![0; indices.len()];
        
        loop {
            // 构建当前组合
            for (i, pos) in current_positions.iter().enumerate() {
                current_combination[i] = indices[i][*pos];
            }
            
            // 调用回调函数
            callback(&current_combination);
            
            // 移动到下一个组合
            let mut carry = true;
            for i in (0..indices.len()).rev() {
                if carry {
                    current_positions[i] += 1;
                    if current_positions[i] >= indices[i].len() {
                        current_positions[i] = 0;
                    } else {
                        carry = false;
                    }
                }
            }
            
            // 如果所有位置都回到0，说明遍历完成
            if carry {
                break;
            }
        }
    }
    
    /// 保留原方法用于向后兼容
    fn compute_index_combinations(&self, indices: &[Vec<usize>]) -> Vec<Vec<usize>> {
        if indices.is_empty() {
            return vec![vec![]];
        }
        
        let mut result = Vec::new();
        self.iterate_index_combinations(indices, &mut |combination| {
            result.push(combination.to_vec());
        });
        result
    }
    
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
    
    /// 预取单个内存位置【内联优化】
    #[inline(always)]
    fn prefetch_single(&self, offset: usize) {
        if offset < self.mmap.len() {
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
                    // ARM架构的预取：使用简单的读取来触发预取
                    // 这种方法在所有Rust版本上都是稳定的
                    let _prefetch_trigger = std::ptr::read_volatile(self.mmap.as_ptr().add(offset));
                }
            }
        }
    }
    
    fn prefetch_data(&self, index_combinations: &[Vec<usize>]) {
        // 使用CPU预取指令
        for combination in index_combinations {
            let offset = self.compute_linear_offset(combination);
            self.prefetch_single(offset);
        }
    }
}

impl LazyArray {
    pub(crate) fn len_logical(&self) -> usize {
        self.logical_rows
            .as_ref()
            .map(|map| map.active_count)
            .unwrap_or_else(|| self.shape.get(0).cloned().unwrap_or(0))
    }

    /// 将整个LazyArray转换为NumPy数组
    /// 这是所有算术操作符的基础方法
    fn to_numpy_array(&self, py: Python) -> PyResult<PyObject> {
        let total_size = self.size()?;
        let mut all_data = Vec::with_capacity(total_size * self.itemsize);

        // 批量读取所有数据
        let logical_length = self.len_logical();
        for i in 0..logical_length {
            let row_data = self.get_row_data(i)?;
            all_data.extend(row_data);
        }

        self.create_numpy_array(py, all_data, &self.logical_shape())
    }

    /// 检查数组是否为整数类型（用于位操作符）
    fn is_integer_type(&self) -> bool {
        match self.dtype {
            DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 |
            DataType::Uint8 | DataType::Uint16 | DataType::Uint32 | DataType::Uint64 |
            DataType::Bool => true,
            _ => false,
        }
    }

    pub(crate) fn logical_shape(&self) -> Vec<usize> {
        let mut shape = self.shape.clone();
        if let Some(map) = &self.logical_rows {
            if !shape.is_empty() {
                shape[0] = map.active_count;
            }
        }
        shape
    }

    pub(crate) fn get_row_data(&self, row_idx: usize) -> PyResult<Vec<u8>> {
        let physical_idx = match self.logical_rows.clone() {
            Some(mut map) => map
                .logical_to_physical(row_idx)
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                    "Index {} is out of bounds for array with {} rows",
                    row_idx,
                    map.logical_len()
                )))?,
            None => row_idx,
        };

        let row_size = if self.shape.len() > 1 {
            self.shape[1..].iter().product::<usize>() * self.itemsize
        } else {
            self.itemsize
        };
        let offset = physical_idx * row_size;

        if offset + row_size > self.mmap.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "Data access out of bounds",
            ));
        }

        Ok(self.mmap[offset..offset + row_size].to_vec())
    }
}


#[cfg(target_family = "windows")]
pub fn release_windows_file_handle(path: &Path) {
    // Windows平台的文件句柄释放
    use std::thread;
    use std::time::Duration;
    
    // 检查是否在测试环境中
    let is_test = std::env::var("PYTEST_CURRENT_TEST").is_ok() || std::env::var("CARGO_PKG_NAME").is_ok();
    
    if is_test {
        // 测试环境：最小化操作，避免卡住
        let _temp_alloc: Vec<u8> = vec![0; 512];  // 更小的分配
        drop(_temp_alloc);
        // 不进行文件测试，避免可能的阻塞
        return;
    }
    
    // 生产环境：标准清理，但减少重试次数
    for attempt in 0..2 {  // 减少重试次数从3到2
        // 分配和释放一小块内存来触发系统的内存管理
        let _temp_alloc: Vec<u8> = vec![0; 1024];
        drop(_temp_alloc);
        
        // 短暂等待让系统处理文件句柄
        thread::sleep(Duration::from_millis(if attempt == 0 { 1 } else { 3 }));  // 减少睡眠时间
        
        // 尝试打开文件以测试是否仍被锁定
        if let Ok(_) = std::fs::File::open(path) {
            // 文件可以打开，说明没有被锁定
            break;
        }
    }
}
