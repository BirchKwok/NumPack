#[macro_use]
extern crate lazy_static;

// 核心模块
mod core; // 错误处理和核心类型定义
mod io;
mod storage; // 存储和元数据管理 // IO操作（并行IO、批量访问）

// 功能模块
mod access_pattern;
mod cache;
mod indexing;
mod lazy_array;
mod memory;
mod numpack;
mod performance;
mod vector_engine; // 向量计算引擎（SimSIMD）

// 测试模块
#[cfg(test)]
mod tests;

use half::f16;
use memmap2::{Mmap, MmapOptions};
use ndarray::ArrayD;
use num_complex::{Complex32, Complex64};
use numpy::{IntoPyArray, PyArrayDyn, PyArrayMethods};
use pyo3::exceptions::PyKeyError;
use pyo3::ffi::Py_buffer;
use pyo3::prelude::*;
use pyo3::types::PySlice;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;

use crate::core::DataType;
use crate::io::ParallelIO;
use crate::lazy_array::indexing::{
    AccessPattern, AccessStrategy, IndexParser, IndexResult, IndexType, SliceInfo,
};
use crate::lazy_array::OptimizedLazyArray;

// Windows 平台专用：mmap 清理函数
#[cfg(windows)]
use crate::numpack::core::clear_mmap_cache_for_array;

use crate::lazy_array::LogicalRowMap;
// use crate::storage::DeletionBitmap; // 未使用，暂时注释
use rayon::prelude::*;

#[cfg(target_family = "unix")]
#[allow(unused_imports)]
use std::os::unix::io::AsRawFd;

lazy_static! {
    static ref MMAP_CACHE: Mutex<HashMap<String, (Arc<Mmap>, i64)>> = Mutex::new(HashMap::new());

    /// 【性能优化】元数据缓存 - 避免重复的元数据查询
    /// 缓存格式: (dtype, shape, itemsize, modify_time)
    static ref METADATA_CACHE: Mutex<HashMap<String, (DataType, Vec<usize>, usize, i64)>> =
        Mutex::new(HashMap::new());
}

// 清理临时文件缓存（所有平台通用）
fn clear_temp_files_from_cache() {
    let mut cache = MMAP_CACHE.lock().unwrap();
    cache.clear();
}

#[allow(dead_code)]
#[pyclass]
struct NumPack {
    io: ParallelIO,
    base_dir: PathBuf,
}

#[pyclass]
pub struct LazyArray {
    mmap: Arc<Mmap>,
    shape: Vec<usize>,
    dtype: DataType,
    itemsize: usize,
    array_path: String,
    modify_time: i64,
    // 内置优化引擎，可选字段
    optimized_engine: Option<OptimizedLazyArray>,
    // 逻辑视图：可选的删除位图信息
    logical_rows: Option<LogicalRowMap>,
}

#[allow(dead_code)]
#[pyclass]
struct StreamLoader {
    base_dir: PathBuf,
    array_name: String,
    total_rows: i64,
    buffer_size: i64,
    current_index: i64,
    dtype: DataType,
    shape: Vec<usize>,
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

// 新增：用户意图分类
#[derive(Debug, Clone)]
enum UserIntent {
    SingleAccess(i64),     // 单次访问：lazy_array[i]
    BatchAccess(Vec<i64>), // 批量访问：lazy_array[indices]
    ComplexIndex,          // 复杂索引：切片、布尔掩码等
}

/// 批量访问模式枚举 - 用于动态策略选择
#[derive(Debug, Clone, Copy, PartialEq)]
enum BatchAccessPattern {
    /// 顺序访问：索引连续或近似连续
    Sequential,
    /// 聚簇访问：索引分布在几个连续块中
    Clustered,
    /// 稀疏访问：索引均匀分布在整个范围
    Sparse,
    /// 随机访问：索引分布无规律
    Random,
    /// 热点访问：大部分访问集中在少数区域
    Hot,
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
    unsafe fn __getbuffer__(
        slf: PyRefMut<Self>,
        view: *mut Py_buffer,
        _flags: i32,
    ) -> PyResult<()> {
        if view.is_null() {
            return Err(PyErr::new::<pyo3::exceptions::PyBufferError, _>(
                "View is null",
            ));
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

            DataType::Complex64 => "complex64",

            DataType::Complex128 => "complex128",
        };

        let format_str = std::ffi::CString::new(format).unwrap();

        let mut strides = Vec::with_capacity(slf.shape.len());
        let mut stride = slf.itemsize;
        for &dim in slf.shape.iter().rev() {
            strides.push(stride as isize);
            stride *= dim;
        }
        strides.reverse();

        // 安全检查：确保 mmap 有效

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
        let total_cols = if self.shape.len() > 1 {
            self.shape[1]
        } else {
            1
        };

        // Extract array name from path (remove suffix and data_ prefix)
        let array_name = self
            .array_path
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
            return Ok(format!(
                "LazyArray('{}', {}, \n{}",
                array_name, shape_str, array
            ));
        }

        let mut result = String::new();
        result.push_str(&format!("LazyArray('{}', {}, \n", array_name, shape_str));

        // Get first 3 rows and last 3 rows
        let show_rows = if total_rows > 6 {
            vec![0, 1, 2, total_rows - 3, total_rows - 2, total_rows - 1]
        } else {
            (0..total_rows).collect()
        };

        // Get first 3 columns and last 3 columns
        let show_cols = if total_cols > 6 {
            vec![0, 1, 2, total_cols - 3, total_cols - 2, total_cols - 1]
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

        // 边界检查
        if offset + self.itemsize > self.mmap.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "Data offset out of bounds",
            ));
        }

        let value = match self.dtype {
            DataType::Bool => {
                let val = unsafe { *self.mmap.as_ptr().add(offset) };
                if val == 0 { "False" } else { "True" }.to_string()
            }
            DataType::Uint8 => unsafe { *self.mmap.as_ptr().add(offset) as u8 }.to_string(),
            DataType::Uint16 => {
                unsafe { *(self.mmap.as_ptr().add(offset) as *const u16) }.to_string()
            }
            DataType::Uint32 => {
                unsafe { *(self.mmap.as_ptr().add(offset) as *const u32) }.to_string()
            }
            DataType::Uint64 => {
                unsafe { *(self.mmap.as_ptr().add(offset) as *const u64) }.to_string()
            }
            DataType::Int8 => unsafe { *self.mmap.as_ptr().add(offset) as i8 }.to_string(),
            DataType::Int16 => {
                unsafe { *(self.mmap.as_ptr().add(offset) as *const i16) }.to_string()
            }
            DataType::Int32 => {
                unsafe { *(self.mmap.as_ptr().add(offset) as *const i32) }.to_string()
            }
            DataType::Int64 => {
                unsafe { *(self.mmap.as_ptr().add(offset) as *const i64) }.to_string()
            }
            DataType::Float16 => {
                let raw_val = unsafe { *(self.mmap.as_ptr().add(offset) as *const u16) };
                let val = half::f16::from_bits(raw_val);
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
            DataType::Complex64 => {
                let val = unsafe { *(self.mmap.as_ptr().add(offset) as *const Complex32) };
                format!("{:.6}+{:.6}j", val.re, val.im)
            }
            DataType::Complex128 => {
                let val = unsafe { *(self.mmap.as_ptr().add(offset) as *const Complex64) };
                format!("{:.6}+{:.6}j", val.re, val.im)
            }
        };
        Ok(value)
    }

    fn get_preview_data(
        &self,
        py: Python,
        start_row: usize,
        end_row: usize,
        start_col: usize,
        end_col: usize,
    ) -> PyResult<String> {
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
        // 明确的用户意图识别
        let user_intent = self.classify_user_intent(key);

        match user_intent {
            UserIntent::SingleAccess(index) => {
                // 尊重用户明确的单次访问意图 - 不干预
                self.handle_single_access(py, index)
            }
            UserIntent::BatchAccess(indices) => {
                // 用户明确的批量访问 - 一次性FFI调用
                self.handle_batch_access(py, indices)
            }
            UserIntent::ComplexIndex => {
                // 复杂索引（切片、布尔掩码等）- 使用现有逻辑
                self.handle_complex_index(py, key)
            }
        }
    }

    #[getter]
    fn shape(&self, py: Python) -> PyResult<PyObject> {
        let logical_shape = self.logical_shape();
        let shape_tuple = pyo3::types::PyTuple::new(py, &logical_shape)?;
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

    #[getter]
    fn T(&self) -> PyResult<LazyArray> {
        // 只允许二维数组进行转置
        if self.shape.len() != 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Transpose is only supported for 2D arrays",
            ));
        }

        // 创建转置后的形状（交换行列）
        let mut transposed_shape = self.shape.clone();
        transposed_shape.swap(0, 1);

        // 创建转置的LazyArray实例
        Ok(LazyArray {
            mmap: Arc::clone(&self.mmap),
            shape: transposed_shape,
            dtype: self.dtype.clone(),
            itemsize: self.itemsize,
            array_path: format!("{}_T", self.array_path),
            modify_time: self.modify_time,
            optimized_engine: None, // 转置视图暂不支持优化引擎
            logical_rows: self.logical_rows.clone(),
        })
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
                            "Negative dimensions other than -1 are not supported in reshape",
                        ));
                    }
                    shape.push(dim);
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "All dimensions must be integers",
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
                            "Negative dimensions other than -1 are not supported in reshape",
                        ));
                    }
                    shape.push(dim);
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "All dimensions must be integers",
                    ));
                }
            }
            shape
        } else if let Ok(dim) = new_shape.extract::<i64>() {
            // Handle single integer input
            if dim < -1 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Negative dimensions other than -1 are not supported in reshape",
                ));
            }
            vec![dim]
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Shape must be a tuple, list, or integer",
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
                        "Only one dimension can be -1",
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
                    "Cannot infer dimension when other dimensions contain 0",
                ));
            }
            if original_size % known_size != 0 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Cannot reshape array of size {} into shape with known size {}",
                    original_size, known_size
                )));
            }
            shape[infer_idx] = (original_size / known_size) as i64;
        }

        // Convert to usize and validate
        let final_shape: Vec<usize> = shape
            .iter()
            .map(|&dim| {
                if dim < 0 {
                    // This should not happen after inference, but just in case
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Invalid dimension after inference",
                    ));
                }
                Ok(dim as usize)
            })
            .collect::<PyResult<Vec<_>>>()?;

        // Validate that the total number of elements remains the same
        let new_size: usize = final_shape.iter().product();

        if original_size != new_size {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Cannot reshape array of size {} into shape {:?} (total size {})",
                original_size, final_shape, new_size
            )));
        }

        // Create a new LazyArray with the same underlying data but different shape
        let reshaped_array = LazyArray {
            mmap: Arc::clone(&self.mmap),
            shape: final_shape,
            dtype: self.dtype.clone(),
            itemsize: self.itemsize,
            array_path: self.array_path.clone(),
            modify_time: self.modify_time,
            optimized_engine: None, // 重塑视图暂不支持优化引擎
            logical_rows: self.logical_rows.clone(),
        };

        // Return the new LazyArray as a Python object
        Py::new(py, reshaped_array)
    }

    // ===========================
    // 生产级性能优化方法
    // ===========================

    /// Fetch rows in large batches while favouring optimised engines when available.
    ///
    /// Parameters
    /// ----------
    /// indices : Sequence[int]
    ///     Logical row indices to gather.
    /// batch_size : int
    ///     Maximum number of rows processed in a single batch; values below 100
    ///     default to 100 to amortise overhead.
    ///
    /// Returns
    /// -------
    /// List[numpy.ndarray]
    ///     Materialised NumPy rows in the original dtype.
    ///
    /// Raises
    /// ------
    /// IndexError
    ///     If any index is out of bounds.
    fn mega_batch_get_rows(
        &self,
        py: Python,
        indices: Vec<usize>,
        batch_size: usize,
    ) -> PyResult<Vec<PyObject>> {
        // 优先使用优化引擎
        if let Some(ref engine) = self.optimized_engine {
            let rows_data = engine.mega_batch_get_rows(&indices, batch_size);
            if !rows_data.is_empty() {
                let mut results = Vec::new();
                for row_data in rows_data {
                    let numpy_array = self.bytes_to_numpy(py, row_data)?;
                    results.push(numpy_array);
                }
                return Ok(results);
            }
        }

        // 回退到基础实现
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

    /// Materialise the requested rows using the most vectorised path available.
    ///
    /// Parameters
    /// ----------
    /// indices : Sequence[int]
    ///     Logical row indices to gather.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     Array containing the stacked rows in their original dtype.
    ///
    /// Raises
    /// ------
    /// IndexError
    ///     If any index is out of bounds.
    fn vectorized_gather(&self, py: Python, indices: Vec<usize>) -> PyResult<PyObject> {
        // Handle empty indices case
        if indices.is_empty() {
            let mut empty_shape = self.shape.clone();
            empty_shape[0] = 0;
            return self.create_numpy_array(py, Vec::new(), &empty_shape);
        }

        // 优先使用优化引擎
        if let Some(ref engine) = self.optimized_engine {
            let rows_data = engine.vectorized_gather(&indices);
            if !rows_data.is_empty() {
                let mut all_data = Vec::new();
                for row in rows_data {
                    all_data.extend(row);
                }
                let mut result_shape = self.shape.clone();
                result_shape[0] = indices.len();
                return self.create_numpy_array(py, all_data, &result_shape);
            }
        }

        // 回退到基础实现
        let mut all_data = Vec::new();
        for &idx in &indices {
            if idx < self.shape[0] {
                let row_data = self.get_row_data(idx)?;
                all_data.extend(row_data);
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                    "Index {} is out of bounds for array with {} rows",
                    idx, self.shape[0]
                )));
            }
        }

        let mut result_shape = self.shape.clone();
        result_shape[0] = indices.len();

        self.create_numpy_array(py, all_data, &result_shape)
    }

    /// Boolean indexing helper that always prefers the most parallel path.
    ///
    /// Parameters
    /// ----------
    /// mask : Sequence[bool]
    ///     Boolean mask evaluated against the logical axis.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     Array containing all rows selected by the mask.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the mask length does not match the logical length.
    fn parallel_boolean_index(&self, py: Python, mask: Vec<bool>) -> PyResult<PyObject> {
        if mask.len() != self.shape[0] {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Mask length doesn't match array length",
            ));
        }

        // 优先使用优化引擎
        if let Some(ref engine) = self.optimized_engine {
            let selected_rows = engine.parallel_boolean_index(&mask);
            if !selected_rows.is_empty() {
                let mut selected_data = Vec::new();
                for row in selected_rows {
                    selected_data.extend(row);
                }
                let selected_count = mask.iter().filter(|&&x| x).count();
                let mut result_shape = self.shape.clone();
                result_shape[0] = selected_count;
                return self.create_numpy_array(py, selected_data, &result_shape);
            }
        }

        // 回退到基础实现
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

    /// Workload-aware warm-up helper. Acts as a pass-through when no
    /// specialised engine is available.
    ///
    /// Parameters
    /// ----------
    /// workload_hint : str
    ///     Describes the anticipated access pattern; recognised values are
    ///     ``"sequential"``, ``"random"``, ``"boolean"``, and ``"heavy"``.
    ///
    /// Returns
    /// -------
    /// None
    fn intelligent_warmup(&self, workload_hint: &str) -> PyResult<()> {
        if let Some(ref engine) = self.optimized_engine {
            use crate::access_pattern::AccessHint;
            let hint = match workload_hint {
                "sequential" => AccessHint::WillAccessAll,
                "random" => AccessHint::WillAccessSparse(0.05),
                "boolean" => AccessHint::WillAccessSparse(0.2),
                "heavy" => AccessHint::WillAccessAll,
                _ => AccessHint::WillAccessAll,
            };
            engine.warmup_intelligent(&hint);
            return Ok(());
        }

        // Fallback to a simple warm-up heuristic when no engine is present.
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

    /// Alias to the production-grade boolean indexing path.
    ///
    /// Parameters
    /// ----------
    /// mask : Sequence[bool]
    ///     Boolean mask evaluated against the logical axis.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     Array containing all rows selected by the mask.
    fn boolean_index_production(&self, py: Python, mask: Vec<bool>) -> PyResult<PyObject> {
        self.parallel_boolean_index(py, mask)
    }

    /// Alias exposing the adaptive boolean indexing algorithm.
    ///
    /// Parameters
    /// ----------
    /// mask : Sequence[bool]
    ///     Boolean mask evaluated against the logical axis.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     Array containing all rows selected by the mask.
    fn boolean_index_adaptive_algorithm(&self, py: Python, mask: Vec<bool>) -> PyResult<PyObject> {
        self.parallel_boolean_index(py, mask)
    }

    /// Return the boolean indexing algorithm that would be selected for the mask.
    ///
    /// Parameters
    /// ----------
    /// mask : Sequence[bool]
    ///     Boolean mask evaluated against the logical axis.
    ///
    /// Returns
    /// -------
    /// str
    ///     Name of the strategy chosen for the provided mask.
    fn choose_optimal_algorithm(&self, mask: Vec<bool>) -> PyResult<String> {
        let selected_count = mask.iter().filter(|&&x| x).count();
        let selection_density = selected_count as f64 / mask.len() as f64;

        let algorithm = if self.optimized_engine.is_some() {
            if selection_density < 0.005 {
                "OptimizedZeroCopy"
            } else if selection_density > 0.95 {
                "OptimizedVectorized"
            } else if selection_density > 0.6 {
                "OptimizedAdaptivePrefetch"
            } else {
                "OptimizedSIMD"
            }
        } else if selection_density < 0.01 {
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
        let physical_idx = match self.logical_rows.clone() {
            Some(mut map) => map.logical_to_physical(row_idx).ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                    "Index {} is out of bounds for array with {} rows",
                    row_idx,
                    map.logical_len()
                ))
            })?,
            None => row_idx,
        };

        if let Some(ref engine) = self.optimized_engine {
            let row_data = engine.get_row(physical_idx);
            if !row_data.is_empty() {
                return Ok(row_data);
            }
        }

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

    fn bytes_to_numpy(&self, py: Python, data: Vec<u8>) -> PyResult<PyObject> {
        let row_shape = if self.shape.len() > 1 {
            vec![self.shape[1..].iter().product::<usize>()]
        } else {
            vec![1]
        };
        self.create_numpy_array(py, data, &row_shape)
    }

    fn __len__(&self) -> PyResult<usize> {
        if self.shape.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "len() of unsized object",
            ));
        }
        Ok(self.len_logical())
    }

    fn len_logical(&self) -> usize {
        self.logical_rows
            .as_ref()
            .map(|map| map.active_count)
            .unwrap_or_else(|| self.shape.get(0).cloned().unwrap_or(0))
    }

    fn physical_rows(&self) -> usize {
        self.logical_rows
            .as_ref()
            .map(|map| map.physical_rows)
            .unwrap_or_else(|| self.shape.get(0).cloned().unwrap_or(0))
    }

    fn logical_shape(&self) -> Vec<usize> {
        let mut shape = self.shape.clone();
        if let Some(map) = &self.logical_rows {
            if !shape.is_empty() {
                shape[0] = map.active_count;
            }
        }
        shape
    }
}

// 实现Drop特性以确保资源正确释放
impl Drop for LazyArray {
    fn drop(&mut self) {
        // 清理缓存中的引用
        if let Ok(mut cache) = MMAP_CACHE.try_lock() {
            if let Some((cached_mmap, _)) = cache.get(&self.array_path) {
                if Arc::ptr_eq(&self.mmap, cached_mmap) {
                    cache.remove(&self.array_path);
                }
            }
        }
    }
}

// 新增：LazyArray的内部方法实现
impl LazyArray {
    fn logical_rows_mut(&mut self) -> Option<&mut LogicalRowMap> {
        self.logical_rows.as_mut()
    }

    fn logical_rows(&self) -> Option<&LogicalRowMap> {
        self.logical_rows.as_ref()
    }

    // 新增：高级索引解析器
    fn parse_advanced_index(
        &self,
        py: Python,
        key: &Bound<'_, PyAny>,
    ) -> Result<IndexResult, PyErr> {
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
                "Too many indices for array",
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
                    shape_attr
                        .extract::<Vec<usize>>()
                        .unwrap_or_else(|_| vec![])
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
            "Invalid index type",
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
    fn handle_broadcasting_directly(
        &self,
        py: Python,
        tuple: &Bound<'_, PyTuple>,
    ) -> Result<PyObject, PyErr> {
        if tuple.len() != 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "Broadcasting only supported for 2D indexing currently",
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
    fn handle_broadcasting_index(
        &self,
        py: Python,
        tuple: &Bound<'_, PyTuple>,
    ) -> Result<IndexResult, PyErr> {
        if tuple.len() != 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "Broadcasting only supported for 2D indexing currently",
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
        return self.execute_broadcasting_access_to_index_result(
            py,
            broadcast_first,
            broadcast_second,
            result_shape,
        );
    }

    // 新增：执行广播访问
    fn execute_broadcasting_access(
        &self,
        py: Python,
        first_indices: Vec<usize>,
        second_indices: Vec<usize>,
        result_shape: Vec<usize>,
    ) -> Result<PyObject, PyErr> {
        // 验证索引数量一致
        if first_indices.len() != second_indices.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Broadcasting indices length mismatch",
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
            let base_offset =
                (row_idx * self.shape[1] + col_idx) * remaining_dims_size * self.itemsize;

            // 复制剩余维度的数据
            let element_size = remaining_dims_size * self.itemsize;
            let element_data = unsafe {
                std::slice::from_raw_parts(self.mmap.as_ptr().add(base_offset), element_size)
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
    fn execute_broadcasting_access_to_index_result(
        &self,
        py: Python,
        first_indices: Vec<usize>,
        second_indices: Vec<usize>,
        result_shape: Vec<usize>,
    ) -> Result<IndexResult, PyErr> {
        // 验证索引数量一致
        if first_indices.len() != second_indices.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Broadcasting indices length mismatch",
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
            let base_offset =
                (row_idx * self.shape[1] + col_idx) * remaining_dims_size * self.itemsize;

            // 复制剩余维度的数据
            let element_size = remaining_dims_size * self.itemsize;
            let element_data = unsafe {
                std::slice::from_raw_parts(self.mmap.as_ptr().add(base_offset), element_size)
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
            "Cannot extract array data",
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
                    return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                        "Index {} out of bounds for dimension 0 of size {}",
                        idx, self.shape[0]
                    )));
                }
            }

            for &idx in &second {
                if idx < 0 || idx as usize >= self.shape[1] {
                    return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                        "Index {} out of bounds for dimension 1 of size {}",
                        idx, self.shape[1]
                    )));
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
                "Unsupported broadcasting pattern",
            ))
        }
    }

    // 新增：提取多维数组（用于广播）
    fn extract_multidim_array(&self, key: &Bound<'_, PyAny>) -> Result<Vec<i64>, PyErr> {
        if let Ok(numpy_array) = key.getattr("__array__") {
            if let Ok(array_func) = numpy_array.call0() {
                // 尝试获取形状信息
                let _shape = if let Ok(shape_attr) = key.getattr("shape") {
                    shape_attr
                        .extract::<Vec<usize>>()
                        .unwrap_or_else(|_| vec![])
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
            "Cannot extract multidimensional array",
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
                            "Too many indices for array",
                        ));
                    }

                    match index_type {
                        IndexType::Integer(idx) => {
                            let adjusted_idx = self.adjust_index(*idx, self.shape[array_dim])?;
                            indices.push(vec![adjusted_idx]);
                            // 整数索引不增加维度到结果
                        }
                        IndexType::Slice(slice_info) => {
                            let slice_indices =
                                self.resolve_slice(slice_info, self.shape[array_dim])?;
                            indices.push(slice_indices);
                        }
                        IndexType::BooleanMask(mask) => {
                            let bool_indices =
                                self.resolve_boolean_mask(mask, self.shape[array_dim])?;
                            indices.push(bool_indices);
                        }
                        IndexType::IntegerArray(arr) => {
                            let int_indices =
                                self.resolve_integer_array(arr, self.shape[array_dim])?;
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
                            "Only one ellipsis allowed",
                        ));
                    }
                    ellipsis_found = true;

                    // 计算省略号需要填充的维度数
                    // 需要排除NewAxis，因为NewAxis不消耗原数组维度
                    let non_newaxis_count = index_types
                        .iter()
                        .filter(|&t| !matches!(t, IndexType::NewAxis))
                        .count();
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
        while expanded
            .iter()
            .filter(|&t| !matches!(t, IndexType::NewAxis))
            .count()
            < self.shape.len()
        {
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
            Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Index {} out of bounds for dimension of size {}",
                index, dim_size
            )))
        }
    }

    // 新增：解析切片
    fn resolve_slice(&self, slice_info: &SliceInfo, dim_size: usize) -> PyResult<Vec<usize>> {
        crate::lazy_array::indexing::resolve_slice(slice_info, dim_size)
    }

    // 新增：解析布尔掩码
    fn resolve_boolean_mask(&self, mask: &[bool], dim_size: usize) -> Result<Vec<usize>, PyErr> {
        if mask.len() != dim_size {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Boolean mask length {} doesn't match dimension size {}",
                mask.len(),
                dim_size
            )));
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
        IndexParser::detect_access_pattern(indices)
    }

    // 新增：选择访问策略
    fn choose_access_strategy(&self, index_result: &IndexResult) -> AccessStrategy {
        let total_elements = index_result
            .indices
            .iter()
            .map(|idx| idx.len())
            .product::<usize>();
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

    // 新增：直接内存访问（Windows安全版本）
    fn direct_memory_access(
        &self,
        py: Python,
        index_result: &IndexResult,
    ) -> Result<PyObject, PyErr> {
        let total_elements = index_result.result_shape.iter().product::<usize>();
        let mut result_data = Vec::with_capacity(total_elements * self.itemsize);

        // 计算多维索引的笛卡尔积
        let index_combinations = self.compute_index_combinations(&index_result.indices);

        for combination in index_combinations {
            let offset = self.compute_linear_offset(&combination);

            // 简化的内存访问
            if offset + self.itemsize <= self.mmap.len() {
                let element_data = unsafe {
                    std::slice::from_raw_parts(self.mmap.as_ptr().add(offset), self.itemsize)
                };
                result_data.extend_from_slice(element_data);
            } else {
                // 超出边界时填充零
                result_data.extend(vec![0u8; self.itemsize]);
            }
        }

        self.create_numpy_array(py, result_data, &index_result.result_shape)
    }

    // 新增：块复制访问（Windows安全版本）
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

                if start_offset + block_size <= self.mmap.len() {
                    let block_data = unsafe {
                        std::slice::from_raw_parts(self.mmap.as_ptr().add(start_offset), block_size)
                    };
                    result_data.extend_from_slice(block_data);
                } else {
                    // 超出边界时使用逐行复制
                    for &idx in indices {
                        let offset = idx * row_size;
                        if offset + row_size <= self.mmap.len() {
                            let row_data = unsafe {
                                std::slice::from_raw_parts(self.mmap.as_ptr().add(offset), row_size)
                            };
                            result_data.extend_from_slice(row_data);
                        } else {
                            result_data.extend(vec![0u8; row_size]);
                        }
                    }
                }
            } else {
                // 分块复制（Windows安全版本）
                for &idx in indices {
                    let offset = idx * row_size;
                    if offset + row_size <= self.mmap.len() {
                        let row_data = unsafe {
                            std::slice::from_raw_parts(self.mmap.as_ptr().add(offset), row_size)
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

    // 新增：并行点访问
    fn parallel_point_access(
        &self,
        py: Python,
        index_result: &IndexResult,
    ) -> Result<PyObject, PyErr> {
        let index_combinations = self.compute_index_combinations(&index_result.indices);

        // 并行处理索引组合
        let result_data: Vec<u8> = index_combinations
            .par_iter()
            .flat_map(|combination| {
                let offset = self.compute_linear_offset(combination);
                unsafe { std::slice::from_raw_parts(self.mmap.as_ptr().add(offset), self.itemsize) }
                    .to_vec()
            })
            .collect();

        self.create_numpy_array(py, result_data, &index_result.result_shape)
    }

    // 新增：预取优化访问
    fn prefetch_optimized_access(
        &self,
        py: Python,
        index_result: &IndexResult,
    ) -> Result<PyObject, PyErr> {
        let index_combinations = self.compute_index_combinations(&index_result.indices);

        // 预取数据
        self.prefetch_data(&index_combinations);

        // 执行访问
        let mut result_data = Vec::with_capacity(index_combinations.len() * self.itemsize);

        for combination in index_combinations {
            let offset = self.compute_linear_offset(&combination);
            let element_data = unsafe {
                std::slice::from_raw_parts(self.mmap.as_ptr().add(offset), self.itemsize)
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
            if indices[i] != indices[i - 1] + 1 {
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
                        std::arch::x86_64::_MM_HINT_T0,
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

    #[inline]
    fn safe_cast_vec<T: bytemuck::Pod + bytemuck::AnyBitPattern + bytemuck::NoUninit>(data: Vec<u8>) -> Vec<T> {
        match bytemuck::allocation::try_cast_vec::<u8, T>(data) {
            Ok(typed_vec) => typed_vec,
            Err((_err, data)) => {
                let count = data.len() / std::mem::size_of::<T>();
                let mut result: Vec<T> = Vec::with_capacity(count);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr(),
                        result.as_mut_ptr() as *mut u8,
                        data.len()
                    );
                    result.set_len(count);
                }
                result
            }
        }
    }

    fn create_numpy_array(
        &self,
        py: Python,
        data: Vec<u8>,
        shape: &[usize],
    ) -> Result<PyObject, PyErr> {
        let array: PyObject = match self.dtype {
            DataType::Bool => {
                let bool_vec: Vec<bool> = data.iter().map(|&x| x != 0).collect();
                let array = ArrayD::from_shape_vec(shape.to_vec(), bool_vec).unwrap();
                array.into_pyarray(py).into()
            }
            DataType::Uint8 => {
                let array = unsafe {
                    ArrayD::from_shape_vec_unchecked(shape.to_vec(), data)
                };
                array.into_pyarray(py).into()
            }
            DataType::Uint16 => {
                let typed_vec: Vec<u16> = Self::safe_cast_vec(data);
                let array = unsafe {
                    ArrayD::from_shape_vec_unchecked(shape.to_vec(), typed_vec)
                };
                array.into_pyarray(py).into()
            }
            DataType::Uint32 => {
                let typed_vec: Vec<u32> = Self::safe_cast_vec(data);
                let array = unsafe {
                    ArrayD::from_shape_vec_unchecked(shape.to_vec(), typed_vec)
                };
                array.into_pyarray(py).into()
            }
            DataType::Uint64 => {
                let typed_vec: Vec<u64> = Self::safe_cast_vec(data);
                let array = unsafe {
                    ArrayD::from_shape_vec_unchecked(shape.to_vec(), typed_vec)
                };
                array.into_pyarray(py).into()
            }
            DataType::Int8 => {
                let typed_vec: Vec<i8> = Self::safe_cast_vec(data);
                let array = unsafe {
                    ArrayD::from_shape_vec_unchecked(shape.to_vec(), typed_vec)
                };
                array.into_pyarray(py).into()
            }
            DataType::Int16 => {
                let typed_vec: Vec<i16> = Self::safe_cast_vec(data);
                let array = unsafe {
                    ArrayD::from_shape_vec_unchecked(shape.to_vec(), typed_vec)
                };
                array.into_pyarray(py).into()
            }
            DataType::Int32 => {
                let typed_vec: Vec<i32> = Self::safe_cast_vec(data);
                let array = unsafe {
                    ArrayD::from_shape_vec_unchecked(shape.to_vec(), typed_vec)
                };
                array.into_pyarray(py).into()
            }
            DataType::Int64 => {
                let typed_vec: Vec<i64> = Self::safe_cast_vec(data);
                let array = unsafe {
                    ArrayD::from_shape_vec_unchecked(shape.to_vec(), typed_vec)
                };
                array.into_pyarray(py).into()
            }
            DataType::Float16 => {
                let typed_vec: Vec<f16> = Self::safe_cast_vec(data);
                let array = unsafe {
                    ArrayD::from_shape_vec_unchecked(shape.to_vec(), typed_vec)
                };
                array.into_pyarray(py).into()
            }
            DataType::Float32 => {
                let typed_vec: Vec<f32> = Self::safe_cast_vec(data);
                let array = unsafe {
                    ArrayD::from_shape_vec_unchecked(shape.to_vec(), typed_vec)
                };
                array.into_pyarray(py).into()
            }
            DataType::Float64 => {
                let typed_vec: Vec<f64> = Self::safe_cast_vec(data);
                let array = unsafe {
                    ArrayD::from_shape_vec_unchecked(shape.to_vec(), typed_vec)
                };
                array.into_pyarray(py).into()
            }
            DataType::Complex64 => {
                let array = unsafe {
                    let slice = std::slice::from_raw_parts(
                        data.as_ptr() as *const Complex32,
                        data.len() / std::mem::size_of::<Complex32>(),
                    );
                    ArrayD::from_shape_vec_unchecked(shape.to_vec(), slice.to_vec())
                };
                array.into_pyarray(py).into()
            }
            DataType::Complex128 => {
                let array = unsafe {
                    let slice = std::slice::from_raw_parts(
                        data.as_ptr() as *const Complex64,
                        data.len() / std::mem::size_of::<Complex64>(),
                    );
                    ArrayD::from_shape_vec_unchecked(shape.to_vec(), slice.to_vec())
                };
                array.into_pyarray(py).into()
            }
        };

        Ok(array)
    }

    fn extract_indices_from_key(
        &self,
        key: &Bound<'_, PyAny>,
        total_rows: usize,
    ) -> PyResult<Vec<usize>> {
        // Try to extract as boolean mask
        if let Ok(bool_mask) = key.extract::<Vec<bool>>() {
            if bool_mask.len() != total_rows {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Boolean mask length {} doesn't match array length {}",
                    bool_mask.len(),
                    total_rows
                )));
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
                    return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                        "Index {} is out of bounds for array of length {}",
                        idx, total_rows
                    )));
                }
            }
            return Ok(indices);
        }

        // Try to handle numpy arrays
        if let Ok(numpy_array) = key.getattr("__array__") {
            if let Ok(array_func) = numpy_array.call0() {
                // Get array shape to handle multi-dimensional arrays
                let shape = if let Ok(shape_attr) = key.getattr("shape") {
                    shape_attr
                        .extract::<Vec<usize>>()
                        .unwrap_or_else(|_| vec![])
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
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Boolean array length {} doesn't match array length {}",
                            bool_array.len(),
                            total_rows
                        )));
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
                            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                                "Index {} is out of bounds for array of length {}",
                                idx, total_rows
                            )));
                        }
                    }
                    return Ok(indices);
                }
            }
        }

        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Invalid index type. Supported types: int, slice, list of ints, boolean mask, or numpy arrays"))
    }
}

// LazyArray内部方法实现（非Python接口）
impl LazyArray {
    /// 用户意图分类（内部方法）
    fn classify_user_intent(&self, key: &Bound<'_, PyAny>) -> UserIntent {
        // 单个整数 - 明确的单次访问意图
        if let Ok(index) = key.extract::<i64>() {
            return UserIntent::SingleAccess(index);
        }

        // 检查布尔掩码 - 优先级高于其他数组检查
        if let Ok(bool_mask) = key.extract::<Vec<bool>>() {
            // 布尔掩码总是复杂索引
            return UserIntent::ComplexIndex;
        }

        // NumPy数组 - 需要区分布尔数组和整数数组
        if let Ok(_) = key.getattr("__array__") {
            // 检查dtype
            if let Ok(dtype) = key.getattr("dtype") {
                if let Ok(dtype_str) = dtype.str() {
                    let dtype_string = dtype_str.to_string();
                    if dtype_string.contains("bool") {
                        // 布尔数组 - 复杂索引
                        return UserIntent::ComplexIndex;
                    }
                }
            }

            // 尝试作为整数数组处理
            if let Ok(array) = key.call_method0("__array__") {
                if let Ok(indices) = array.extract::<Vec<i64>>() {
                    return UserIntent::BatchAccess(indices);
                }
            }
        }

        // 列表或数组 - 明确的批量访问意图
        if let Ok(list) = key.downcast::<PyList>() {
            if let Ok(indices) = list.extract::<Vec<i64>>() {
                return UserIntent::BatchAccess(indices);
            }
        }

        // 其他复杂索引（切片、布尔掩码等）
        UserIntent::ComplexIndex
    }

    /// 处理单次访问（尊重用户意图）
    fn handle_single_access(&self, py: Python, index: i64) -> PyResult<PyObject> {
        let logical_len = self.len_logical() as i64;
        let normalized = if index < 0 {
            logical_len + index
        } else {
            index
        };
        if normalized < 0 || normalized >= logical_len {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Index {} is out of bounds",
                index
            )));
        }

        let row_data = self.get_row_data(normalized as usize)?;
        let row_shape = if self.shape.len() > 1 {
            self.shape[1..].to_vec()
        } else {
            vec![1]
        };
        self.create_numpy_array(py, row_data, &row_shape)
    }

    /// 快速访问模式检测（极简版：最小化开销）
    /// 
    /// 只做最基本的检测，避免复杂统计计算
    #[inline]
    fn detect_access_pattern_fast(&self, indices: &[(usize, usize)], total_rows: usize) -> BatchAccessPattern {
        let n = indices.len();
        if n < 20 {
            return BatchAccessPattern::Sequential;
        }
        
        // 快速采样：只检测前32个样本（减少开销）
        const SAMPLE_SIZE: usize = 32;
        let sample_count = n.min(SAMPLE_SIZE);
        
        // 计算连续性和小间隔比率
        let mut consecutive = 0usize;
        let mut small_gap = 0usize;
        
        for i in 1..sample_count {
            let prev = indices[i - 1].1;
            let curr = indices[i].1;
            let gap = if curr > prev { curr - prev } else { prev - curr };
            
            if gap == 1 {
                consecutive += 1;
            }
            if gap <= 16 {  // 放宽小间隔阈值
                small_gap += 1;
            }
        }
        
        let pairs = sample_count - 1;
        if pairs == 0 {
            return BatchAccessPattern::Sequential;
        }
        
        // 快速判断 - 优化阈值
        if consecutive * 5 > pairs * 4 {  // >80% 连续
            return BatchAccessPattern::Sequential;
        }
        if small_gap * 10 > pairs * 6 {  // >60% 小间隔 -> Clustered
            return BatchAccessPattern::Clustered;
        }
        
        // 检测热点：检查范围
        let min_idx = indices[..sample_count].iter().map(|&(_, p)| p).min().unwrap_or(0);
        let max_idx = indices[..sample_count].iter().map(|&(_, p)| p).max().unwrap_or(0);
        let range = max_idx.saturating_sub(min_idx);
        
        if total_rows > 0 && range < total_rows / 10 {
            return BatchAccessPattern::Hot;
        }
        
        BatchAccessPattern::Random
    }
    
    /// 检测批量访问的访问模式（完整版：采样检测）
    /// 
    /// 通过分析索引分布特征来判断访问模式：
    /// - Sequential: 连续性 > 80%
    /// - Clustered: 连续性 > 40% 或 有明显的聚簇特征
    /// - Hot: 大部分索引集中在小范围内
    /// - Sparse: 索引间隔均匀
    /// - Random: 其他情况
    #[allow(dead_code)]
    fn detect_access_pattern(&self, indices: &[(usize, usize)], total_rows: usize) -> BatchAccessPattern {
        if indices.len() < 3 {
            return BatchAccessPattern::Sequential;
        }
        
        // 优化：对大批量使用采样检测（最多检测200个样本）
        const MAX_SAMPLE_SIZE: usize = 200;
        let sample_indices: Vec<usize> = if indices.len() <= MAX_SAMPLE_SIZE {
            indices.iter().map(|&(_, p)| p).collect()
        } else {
            // 均匀采样
            let step = indices.len() / MAX_SAMPLE_SIZE;
            indices.iter()
                .enumerate()
                .filter(|(i, _)| i % step == 0)
                .take(MAX_SAMPLE_SIZE)
                .map(|(_, &(_, p))| p)
                .collect()
        };
        
        if sample_indices.len() < 2 {
            return BatchAccessPattern::Sequential;
        }
        
        // 计算统计信息
        let mut consecutive_pairs = 0;
        let mut total_pairs = 0;
        let mut small_gap_count = 0; // 间隔 <= 5 的对数
        
        for window in sample_indices.windows(2) {
            let gap = if window[1] > window[0] {
                window[1] - window[0]
            } else {
                window[0] - window[1]
            };
            
            if gap == 1 {
                consecutive_pairs += 1;
            }
            if gap <= 5 {
                small_gap_count += 1;
            }
            total_pairs += 1;
        }
        
        if total_pairs == 0 {
            return BatchAccessPattern::Sequential;
        }
        
        let consecutive_ratio = consecutive_pairs as f64 / total_pairs as f64;
        let clustered_ratio = small_gap_count as f64 / total_pairs as f64;
        
        // 快速判断顺序和聚簇模式
        if consecutive_ratio > 0.8 {
            return BatchAccessPattern::Sequential;
        }
        if clustered_ratio > 0.5 || consecutive_ratio > 0.4 {
            return BatchAccessPattern::Clustered;
        }
        
        // 检测热点模式：大部分索引集中在小范围内
        let min_idx = *sample_indices.iter().min().unwrap_or(&0);
        let max_idx = *sample_indices.iter().max().unwrap_or(&0);
        let index_range = max_idx.saturating_sub(min_idx);
        
        // 热点模式：索引范围小于总数据量的10%，且索引密度高
        // 密度 = 索引数量 / 范围
        let density = if index_range > 0 {
            sample_indices.len() as f64 / index_range as f64
        } else {
            1.0
        };
        
        // 高密度（>0.3）且小范围才是热点模式
        if total_rows > 0 && index_range < total_rows / 10 && density > 0.3 {
            return BatchAccessPattern::Hot;
        }
        
        // 检查是否为稀疏模式（均匀分布）
        if sample_indices.len() > 1 && index_range > 0 {
            let expected_gap = index_range as f64 / (sample_indices.len() - 1) as f64;
            let mut total_gap = 0usize;
            for window in sample_indices.windows(2) {
                let gap = if window[1] > window[0] {
                    window[1] - window[0]
                } else {
                    window[0] - window[1]
                };
                total_gap += gap;
            }
            let avg_gap = total_gap as f64 / total_pairs as f64;
            
            // 如果平均间隔接近预期间隔，说明是均匀分布
            if expected_gap > 0.0 && (avg_gap - expected_gap).abs() / expected_gap < 0.3 {
                return BatchAccessPattern::Sparse;
            }
        }
        
        BatchAccessPattern::Random
    }
    
    /// 处理批量访问（优化的一次性FFI调用）
    /// 
    /// 优化策略（动态选择）：
    /// 1. 访问模式自动检测 - 分析索引分布判断最优策略
    /// 2. 索引排序优化 - 仅对随机/稀疏模式启用，聚簇模式绕过
    /// 3. 连续块检测 - 检测连续索引块进行批量读取
    /// 4. 预取提示 - 在Unix系统上使用madvise提供预读提示
    fn handle_batch_access(&self, py: Python, indices: Vec<i64>) -> PyResult<PyObject> {
        let logical_len = self.len_logical() as i64;
        let num_indices = indices.len();
        
        // 快速路径：空索引
        if num_indices == 0 {
            let mut result_shape = self.shape.clone();
            if !result_shape.is_empty() {
                result_shape[0] = 0;
            }
            return self.create_numpy_array(py, Vec::new(), &result_shape);
        }
        
        // 归一化索引并创建 (original_position, physical_index) 对
        let mut indexed_positions: Vec<(usize, usize)> = Vec::with_capacity(num_indices);
        
        for (orig_pos, &idx) in indices.iter().enumerate() {
            let normalized = if idx < 0 { logical_len + idx } else { idx };
            if normalized < 0 || normalized >= logical_len {
                return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                    "Index {} is out of bounds",
                    idx
                )));
            }
            indexed_positions.push((orig_pos, normalized as usize));
        }
        
        // 转换逻辑索引到物理索引
        if let Some(mut map) = self.logical_rows.clone() {
            let logical_indices: Vec<usize> = indexed_positions.iter().map(|&(_, li)| li).collect();
            let physical_indices = map.logical_indices(&logical_indices)?;
            for (i, phys_idx) in physical_indices.into_iter().enumerate() {
                indexed_positions[i].1 = phys_idx;
            }
        }
        
        let row_size = if self.shape.len() > 1 {
            self.shape[1..].iter().product::<usize>() * self.itemsize
        } else {
            self.itemsize
        };
        
        // 动态策略选择 - 根据数据规模调整阈值
        let total_rows = self.shape[0];
        let data_size_bytes = total_rows * row_size;
        
        // 大数据集（>10MB）使用更保守的策略
        let is_large_dataset = data_size_bytes > 10 * 1024 * 1024;
        
        // 优化：提高阈值减少小批量开销
        let sort_threshold = if is_large_dataset { 200 } else { 100 };
        let pattern_detect_threshold = 50; // 提高模式检测阈值
        let parallel_threshold = 50000; // 并行处理阈值（仅对超大批量启用）
        
        // 检测访问模式（仅对足够大的批量进行检测）
        let access_pattern = if num_indices >= pattern_detect_threshold {
            self.detect_access_pattern_fast(&indexed_positions, total_rows)
        } else {
            BatchAccessPattern::Sequential // 小批量默认按顺序处理
        };
        
        // 根据访问模式和数据规模决定优化策略
        // 优化：只对小数据集的Random/Sparse模式启用排序
        // Clustered模式已有良好局部性，不需要排序
        let use_sorted_access = num_indices >= sort_threshold && 
            !is_large_dataset &&
            access_pattern == BatchAccessPattern::Random;
        
        // 优化：使用连续内存缓冲区，避免Vec<Vec<u8>>的多次分配
        let total_size = num_indices * row_size;
        let mut all_data = vec![0u8; total_size];
        
        if use_sorted_access {
            // 就地排序indexed_positions（避免clone）
            indexed_positions.sort_unstable_by_key(|&(_, phys_idx)| phys_idx);
            
            // 预取提示
            #[cfg(unix)]
            if access_pattern == BatchAccessPattern::Random {
                self.advise_sequential_access(&indexed_positions, row_size);
            }
            
            // 直接写入连续缓冲区
            self.process_sorted_batch_fast(&indexed_positions, row_size, &mut all_data)?;
        } else if num_indices >= parallel_threshold && row_size >= 32 {
            // 优化：大批量使用并行处理
            self.process_parallel_batch_fast(&indexed_positions, row_size, &mut all_data)?;
        } else {
            // 预取提示
            #[cfg(unix)]
            if matches!(access_pattern, BatchAccessPattern::Clustered | BatchAccessPattern::Hot) 
                && num_indices >= sort_threshold {
                self.advise_willneed_access(&indexed_positions, row_size);
            }
            
            // 直接写入连续缓冲区
            self.process_direct_batch_fast(&indexed_positions, row_size, &mut all_data)?;
        }

        let mut result_shape = self.shape.clone();
        if !result_shape.is_empty() {
            result_shape[0] = num_indices;
        }

        self.create_numpy_array(py, all_data, &result_shape)
    }
    
    /// 快速处理排序后的批量访问（直接写入连续缓冲区）
    fn process_sorted_batch_fast(
        &self,
        sorted_positions: &[(usize, usize)],
        row_size: usize,
        output: &mut [u8],
    ) -> PyResult<()> {
        let mmap_len = self.mmap.len();
        let mut i = 0;
        
        while i < sorted_positions.len() {
            let (orig_pos, phys_idx) = sorted_positions[i];
            
            // 检测连续块的长度
            let mut block_len = 1;
            while i + block_len < sorted_positions.len() {
                let (_, next_phys_idx) = sorted_positions[i + block_len];
                if next_phys_idx == phys_idx + block_len {
                    block_len += 1;
                } else {
                    break;
                }
            }
            
            let src_offset = phys_idx * row_size;
            
            if block_len >= 2 {
                // 连续块：批量读取后分发
                let block_size = block_len * row_size;
                if src_offset + block_size <= mmap_len {
                    let block_data = &self.mmap[src_offset..src_offset + block_size];
                    for j in 0..block_len {
                        let (orig_pos_j, _) = sorted_positions[i + j];
                        let dst_start = orig_pos_j * row_size;
                        let src_start = j * row_size;
                        output[dst_start..dst_start + row_size]
                            .copy_from_slice(&block_data[src_start..src_start + row_size]);
                    }
                }
                i += block_len;
            } else {
                // 单行：直接复制
                if src_offset + row_size <= mmap_len {
                    let dst_start = orig_pos * row_size;
                    output[dst_start..dst_start + row_size]
                        .copy_from_slice(&self.mmap[src_offset..src_offset + row_size]);
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                        "Data access out of bounds",
                    ));
                }
                i += 1;
            }
        }
        Ok(())
    }
    
    /// 快速直接批量访问（直接写入连续缓冲区）
    fn process_direct_batch_fast(
        &self,
        indexed_positions: &[(usize, usize)],
        row_size: usize,
        output: &mut [u8],
    ) -> PyResult<()> {
        let mmap_len = self.mmap.len();
        let mmap_slice = &self.mmap[..];
        
        for &(orig_pos, phys_idx) in indexed_positions {
            let src_offset = phys_idx * row_size;
            if src_offset + row_size > mmap_len {
                return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                    "Data access out of bounds",
                ));
            }
            let dst_start = orig_pos * row_size;
            output[dst_start..dst_start + row_size]
                .copy_from_slice(&mmap_slice[src_offset..src_offset + row_size]);
        }
        Ok(())
    }
    
    /// 并行批量访问（使用rayon并行化大批量内存复制）
    /// 优化版：减少内存分配，使用更大的chunk和预分配缓冲区
    fn process_parallel_batch_fast(
        &self,
        indexed_positions: &[(usize, usize)],
        row_size: usize,
        output: &mut [u8],
    ) -> PyResult<()> {
        let mmap_len = self.mmap.len();
        let mmap_slice = &self.mmap[..];
        let num_indices = indexed_positions.len();
        
        // 首先验证所有索引都在范围内
        for &(_, phys_idx) in indexed_positions {
            let src_offset = phys_idx * row_size;
            if src_offset + row_size > mmap_len {
                return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                    "Data access out of bounds",
                ));
            }
        }
        
        // 优化：对于大批量，使用更大的chunk减少并行开销
        // 每个chunk至少处理1024行或总数/CPU核心数
        let num_threads = rayon::current_num_threads().max(1);
        let chunk_size = (num_indices / num_threads).max(1024).min(num_indices);
        
        // 使用分块并行处理 - 每个chunk预分配固定缓冲区
        let chunks: Vec<_> = indexed_positions.chunks(chunk_size).collect();
        
        // 并行处理每个chunk，直接写入预分配的缓冲区
        let chunk_buffers: Vec<Vec<u8>> = chunks
            .par_iter()
            .map(|chunk| {
                // 每个chunk使用一个连续缓冲区
                let mut buffer = vec![0u8; chunk.len() * row_size];
                for (i, &(_, phys_idx)) in chunk.iter().enumerate() {
                    let src_offset = phys_idx * row_size;
                    let dst_offset = i * row_size;
                    buffer[dst_offset..dst_offset + row_size]
                        .copy_from_slice(&mmap_slice[src_offset..src_offset + row_size]);
                }
                buffer
            })
            .collect();
        
        // 串行写入最终输出（保持原始顺序）
        let mut chunk_start = 0;
        for (chunk_idx, chunk) in chunks.iter().enumerate() {
            let buffer = &chunk_buffers[chunk_idx];
            for (i, &(orig_pos, _)) in chunk.iter().enumerate() {
                let src_offset = i * row_size;
                let dst_start = orig_pos * row_size;
                output[dst_start..dst_start + row_size]
                    .copy_from_slice(&buffer[src_offset..src_offset + row_size]);
            }
            chunk_start += chunk.len();
        }
        
        Ok(())
    }
    
    /// 处理排序后的批量访问（检测连续块）- 旧版本保留兼容
    #[allow(dead_code)]
    fn process_sorted_batch(
        &self,
        sorted_positions: &[(usize, usize)],
        row_size: usize,
        result_rows: &mut [Vec<u8>],
    ) -> PyResult<()> {
        let mut i = 0;
        while i < sorted_positions.len() {
            let (orig_pos, phys_idx) = sorted_positions[i];
            
            // 检测连续块的长度
            let mut block_len = 1;
            while i + block_len < sorted_positions.len() {
                let (_, next_phys_idx) = sorted_positions[i + block_len];
                if next_phys_idx == phys_idx + block_len {
                    block_len += 1;
                } else {
                    break;
                }
            }
            
            // 如果是连续块（>=2行），批量读取
            if block_len >= 2 {
                let start_offset = phys_idx * row_size;
                let block_size = block_len * row_size;
                
                if start_offset + block_size <= self.mmap.len() {
                    let block_data = &self.mmap[start_offset..start_offset + block_size];
                    
                    // 分配到各个原始位置
                    for j in 0..block_len {
                        let (orig_pos_j, _) = sorted_positions[i + j];
                        let row_start = j * row_size;
                        result_rows[orig_pos_j] = block_data[row_start..row_start + row_size].to_vec();
                    }
                }
                i += block_len;
            } else {
                // 单行读取
                self.read_single_row(phys_idx, row_size, orig_pos, result_rows)?;
                i += 1;
            }
        }
        Ok(())
    }
    
    /// 直接批量访问（按原始顺序）- 旧版本保留兼容
    #[allow(dead_code)]
    fn process_direct_batch(
        &self,
        indexed_positions: &[(usize, usize)],
        row_size: usize,
        result_rows: &mut [Vec<u8>],
    ) -> PyResult<()> {
        for &(orig_pos, phys_idx) in indexed_positions {
            self.read_single_row(phys_idx, row_size, orig_pos, result_rows)?;
        }
        Ok(())
    }
    
    /// 读取单行数据 - 旧版本保留兼容
    #[allow(dead_code)]
    fn read_single_row(
        &self,
        phys_idx: usize,
        row_size: usize,
        orig_pos: usize,
        result_rows: &mut [Vec<u8>],
    ) -> PyResult<()> {
        // 优先使用优化引擎
        if let Some(ref engine) = self.optimized_engine {
            let row_data = engine.get_row(phys_idx);
            if !row_data.is_empty() {
                result_rows[orig_pos] = row_data;
                return Ok(());
            }
        }
        
        let offset = phys_idx * row_size;
        if offset + row_size > self.mmap.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "Data access out of bounds",
            ));
        }
        result_rows[orig_pos] = self.mmap[offset..offset + row_size].to_vec();
        Ok(())
    }
    
    /// Unix系统上的madvise预取提示
    #[cfg(unix)]
    fn advise_sequential_access(&self, sorted_positions: &[(usize, usize)], row_size: usize) {
        use std::os::unix::io::AsRawFd;
        
        if sorted_positions.is_empty() {
            return;
        }
        
        // 获取访问范围
        let first_phys = sorted_positions.first().map(|&(_, p)| p).unwrap_or(0);
        let last_phys = sorted_positions.last().map(|&(_, p)| p).unwrap_or(0);
        
        let start_offset = first_phys * row_size;
        let end_offset = (last_phys + 1) * row_size;
        let range_size = end_offset.saturating_sub(start_offset);
        
        // 只对较大范围提供提示（避免过多系统调用）
        if range_size > 64 * 1024 {  // 64KB阈值
            unsafe {
                let ptr = self.mmap.as_ptr().add(start_offset);
                // MADV_SEQUENTIAL = 2, 提示顺序访问
                libc::madvise(ptr as *mut libc::c_void, range_size, libc::MADV_SEQUENTIAL);
            }
        }
    }
    
    /// Windows系统上的空实现
    #[cfg(not(unix))]
    fn advise_sequential_access(&self, _sorted_positions: &[(usize, usize)], _row_size: usize) {
        // Windows不支持madvise，使用空实现
    }
    
    /// Unix系统上的WILLNEED预取提示（用于聚簇访问模式）
    #[cfg(unix)]
    fn advise_willneed_access(&self, indexed_positions: &[(usize, usize)], row_size: usize) {
        if indexed_positions.is_empty() {
            return;
        }
        
        // 找出所有需要访问的区域，合并相近的区域
        let mut regions: Vec<(usize, usize)> = Vec::new();
        let mut sorted_indices: Vec<usize> = indexed_positions.iter().map(|&(_, p)| p).collect();
        sorted_indices.sort();
        
        let mut region_start = sorted_indices[0];
        let mut region_end = sorted_indices[0];
        
        for &idx in &sorted_indices[1..] {
            // 如果间隔小于32行，合并到同一区域
            if idx <= region_end + 32 {
                region_end = idx;
            } else {
                regions.push((region_start, region_end));
                region_start = idx;
                region_end = idx;
            }
        }
        regions.push((region_start, region_end));
        
        // 对每个区域发送WILLNEED提示
        for (start_idx, end_idx) in regions {
            let start_offset = start_idx * row_size;
            let end_offset = (end_idx + 1) * row_size;
            let range_size = end_offset.saturating_sub(start_offset);
            
            // 只对较大区域发送提示（>16KB）
            if range_size > 16 * 1024 && start_offset + range_size <= self.mmap.len() {
                unsafe {
                    let ptr = self.mmap.as_ptr().add(start_offset);
                    // MADV_WILLNEED = 3, 提示即将访问
                    libc::madvise(ptr as *mut libc::c_void, range_size, libc::MADV_WILLNEED);
                }
            }
        }
    }
    
    /// Windows系统上的空实现
    #[cfg(not(unix))]
    fn advise_willneed_access(&self, _indexed_positions: &[(usize, usize)], _row_size: usize) {
        // Windows不支持madvise，使用空实现
    }

    /// 处理复杂索引（保持现有逻辑）
    fn handle_complex_index(&self, py: Python, key: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        // 检查是否是广播情况
        if let Ok(tuple) = key.downcast::<PyTuple>() {
            if self.check_for_broadcasting(tuple)? {
                return self.handle_broadcasting_directly(py, tuple);
            }
        }

        // 使用现有的高级索引解析器
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
}

// 简化的内存复制函数
#[allow(dead_code)]
unsafe fn safe_memory_copy(src: *const u8, dst: *mut u8, size: usize) {
    // Windows平台使用简单的安全复制
    #[cfg(target_os = "windows")]
    {
        std::ptr::copy_nonoverlapping(src, dst, size);
        return;
    }

    // 非Windows平台可以使用SIMD优化
    #[cfg(all(target_arch = "x86_64", not(target_os = "windows")))]
    {
        if is_x86_feature_detected!("avx2") && size >= 32 && size % 32 == 0 {
            let chunks = size / 32;
            for i in 0..chunks {
                let src_offset = i * 32;
                let dst_offset = i * 32;

                let data = std::arch::x86_64::_mm256_loadu_si256(
                    src.add(src_offset) as *const std::arch::x86_64::__m256i
                );
                std::arch::x86_64::_mm256_storeu_si256(
                    dst.add(dst_offset) as *mut std::arch::x86_64::__m256i,
                    data,
                );
            }
            return;
        }
    }

    // 标准复制
    std::ptr::copy_nonoverlapping(src, dst, size);
}

fn get_array_dtype(array: &Bound<'_, PyAny>) -> PyResult<DataType> {
    let dtype_str = array
        .getattr("dtype")?
        .getattr("name")?
        .extract::<String>()?;
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
        "complex64" => Ok(DataType::Complex64),
        "complex128" => Ok(DataType::Complex128),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unsupported dtype: {}",
            dtype_str
        ))),
    }
}

// 辅助函数（非PyO3方法）
impl NumPack {
    #[inline]
    fn safe_cast_vec_numpack<T: bytemuck::Pod + bytemuck::AnyBitPattern + bytemuck::NoUninit>(data: Vec<u8>) -> Vec<T> {
        match bytemuck::allocation::try_cast_vec::<u8, T>(data) {
            Ok(typed_vec) => typed_vec,
            Err((_err, data)) => {
                let count = data.len() / std::mem::size_of::<T>();
                let mut result: Vec<T> = Vec::with_capacity(count);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr(),
                        result.as_mut_ptr() as *mut u8,
                        data.len()
                    );
                    result.set_len(count);
                }
                result
            }
        }
    }

    /// 辅助函数：根据dtype创建numpy数组
    /// 【零复制优化】使用try_cast_vec直接转换Vec，避免额外复制
    fn create_numpy_array_from_dtype(
        &self,
        py: Python,
        data: Vec<u8>,
        shape: &[usize],
        dtype: DataType,
    ) -> PyResult<PyObject> {
        let array: PyObject = match dtype {
            DataType::Bool => {
                let bool_vec: Vec<bool> = data.iter().map(|&x| x != 0).collect();
                let array = ArrayD::from_shape_vec(shape.to_vec(), bool_vec)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
                array.into_pyarray(py).into()
            }
            DataType::Uint8 => {
                let array = unsafe {
                    ArrayD::from_shape_vec_unchecked(shape.to_vec(), data)
                };
                array.into_pyarray(py).into()
            }
            DataType::Uint16 => {
                let typed_vec: Vec<u16> = Self::safe_cast_vec_numpack(data);
                let array = unsafe {
                    ArrayD::from_shape_vec_unchecked(shape.to_vec(), typed_vec)
                };
                array.into_pyarray(py).into()
            }
            DataType::Uint32 => {
                let typed_vec: Vec<u32> = Self::safe_cast_vec_numpack(data);
                let array = unsafe {
                    ArrayD::from_shape_vec_unchecked(shape.to_vec(), typed_vec)
                };
                array.into_pyarray(py).into()
            }
            DataType::Uint64 => {
                let typed_vec: Vec<u64> = Self::safe_cast_vec_numpack(data);
                let array = unsafe {
                    ArrayD::from_shape_vec_unchecked(shape.to_vec(), typed_vec)
                };
                array.into_pyarray(py).into()
            }
            DataType::Int8 => {
                let typed_vec: Vec<i8> = Self::safe_cast_vec_numpack(data);
                let array = unsafe {
                    ArrayD::from_shape_vec_unchecked(shape.to_vec(), typed_vec)
                };
                array.into_pyarray(py).into()
            }
            DataType::Int16 => {
                let typed_vec: Vec<i16> = Self::safe_cast_vec_numpack(data);
                let array = unsafe {
                    ArrayD::from_shape_vec_unchecked(shape.to_vec(), typed_vec)
                };
                array.into_pyarray(py).into()
            }
            DataType::Int32 => {
                let typed_vec: Vec<i32> = Self::safe_cast_vec_numpack(data);
                let array = unsafe {
                    ArrayD::from_shape_vec_unchecked(shape.to_vec(), typed_vec)
                };
                array.into_pyarray(py).into()
            }
            DataType::Int64 => {
                let typed_vec: Vec<i64> = Self::safe_cast_vec_numpack(data);
                let array = unsafe {
                    ArrayD::from_shape_vec_unchecked(shape.to_vec(), typed_vec)
                };
                array.into_pyarray(py).into()
            }
            DataType::Float16 => {
                let typed_vec: Vec<f16> = Self::safe_cast_vec_numpack(data);
                let array = unsafe {
                    ArrayD::from_shape_vec_unchecked(shape.to_vec(), typed_vec)
                };
                array.into_pyarray(py).into()
            }
            DataType::Float32 => {
                let typed_vec: Vec<f32> = Self::safe_cast_vec_numpack(data);
                let array = unsafe {
                    ArrayD::from_shape_vec_unchecked(shape.to_vec(), typed_vec)
                };
                array.into_pyarray(py).into()
            }
            DataType::Float64 => {
                let typed_vec: Vec<f64> = Self::safe_cast_vec_numpack(data);
                let array = unsafe {
                    ArrayD::from_shape_vec_unchecked(shape.to_vec(), typed_vec)
                };
                array.into_pyarray(py).into()
            }
            DataType::Complex64 => {
                let array = unsafe {
                    let slice = std::slice::from_raw_parts(
                        data.as_ptr() as *const Complex32,
                        data.len() / std::mem::size_of::<Complex32>(),
                    );
                    ArrayD::from_shape_vec_unchecked(shape.to_vec(), slice.to_vec())
                };
                array.into_pyarray(py).into()
            }
            DataType::Complex128 => {
                let array = unsafe {
                    let slice = std::slice::from_raw_parts(
                        data.as_ptr() as *const Complex64,
                        data.len() / std::mem::size_of::<Complex64>(),
                    );
                    ArrayD::from_shape_vec_unchecked(shape.to_vec(), slice.to_vec())
                };
                array.into_pyarray(py).into()
            }
        };
        Ok(array)
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
        let mut complex64_arrays = Vec::new();
        let mut complex128_arrays = Vec::new();

        for (i, (key, value)) in arrays.iter().enumerate() {
            let name = if let Some(prefix) = &array_name {
                format!("{}{}", prefix, i)
            } else {
                key.extract::<String>()?
            };

            let dtype = get_array_dtype(&value)?;
            let _shape: Vec<u64> = value
                .getattr("shape")?
                .extract::<Vec<usize>>()?
                .into_iter()
                .map(|x| x as u64)
                .collect();

            match dtype {
                DataType::Bool => {
                    let array = value.downcast::<PyArrayDyn<bool>>()?;
                    let readonly = array.readonly();
                    let array = unsafe { readonly.as_array().to_owned() };
                    bool_arrays.push((name, array, dtype));
                }
                DataType::Uint8 => {
                    let array = value.downcast::<PyArrayDyn<u8>>()?;
                    let readonly = array.readonly();
                    let array = unsafe { readonly.as_array().to_owned() };
                    u8_arrays.push((name, array, dtype));
                }
                DataType::Uint16 => {
                    let array = value.downcast::<PyArrayDyn<u16>>()?;
                    let readonly = array.readonly();
                    let array = unsafe { readonly.as_array().to_owned() };
                    u16_arrays.push((name, array, dtype));
                }
                DataType::Uint32 => {
                    let array = value.downcast::<PyArrayDyn<u32>>()?;
                    let readonly = array.readonly();
                    let array = unsafe { readonly.as_array().to_owned() };
                    u32_arrays.push((name, array, dtype));
                }
                DataType::Uint64 => {
                    let array = value.downcast::<PyArrayDyn<u64>>()?;
                    let readonly = array.readonly();
                    let array = unsafe { readonly.as_array().to_owned() };
                    u64_arrays.push((name, array, dtype));
                }
                DataType::Int8 => {
                    let array = value.downcast::<PyArrayDyn<i8>>()?;
                    let readonly = array.readonly();
                    let array = unsafe { readonly.as_array().to_owned() };
                    i8_arrays.push((name, array, dtype));
                }
                DataType::Int16 => {
                    let array = value.downcast::<PyArrayDyn<i16>>()?;
                    let readonly = array.readonly();
                    let array = unsafe { readonly.as_array().to_owned() };
                    i16_arrays.push((name, array, dtype));
                }
                DataType::Int32 => {
                    let array = value.downcast::<PyArrayDyn<i32>>()?;
                    let readonly = array.readonly();
                    let array = unsafe { readonly.as_array().to_owned() };
                    i32_arrays.push((name, array, dtype));
                }
                DataType::Int64 => {
                    let array = value.downcast::<PyArrayDyn<i64>>()?;
                    let readonly = array.readonly();
                    let array = unsafe { readonly.as_array().to_owned() };
                    i64_arrays.push((name, array, dtype));
                }
                DataType::Float16 => {
                    let array = value.downcast::<PyArrayDyn<f16>>()?;
                    let readonly = array.readonly();
                    let array = unsafe { readonly.as_array().to_owned() };
                    f16_arrays.push((name, array, dtype));
                }
                DataType::Float32 => {
                    let array = value.downcast::<PyArrayDyn<f32>>()?;
                    let readonly = array.readonly();
                    let array = unsafe { readonly.as_array().to_owned() };
                    f32_arrays.push((name, array, dtype));
                }
                DataType::Float64 => {
                    let array = value.downcast::<PyArrayDyn<f64>>()?;
                    let readonly = array.readonly();
                    let array = unsafe { readonly.as_array().to_owned() };
                    f64_arrays.push((name, array, dtype));
                }
                DataType::Complex64 => {
                    let array = value.downcast::<PyArrayDyn<Complex32>>()?;
                    let readonly = array.readonly();
                    let array = unsafe { readonly.as_array().to_owned() };
                    complex64_arrays.push((name, array, dtype));
                }
                DataType::Complex128 => {
                    let array = value.downcast::<PyArrayDyn<Complex64>>()?;
                    let readonly = array.readonly();
                    let array = unsafe { readonly.as_array().to_owned() };
                    complex128_arrays.push((name, array, dtype));
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
        if !complex64_arrays.is_empty() {
            self.io.save_arrays(&complex64_arrays)?;
        }
        if !complex128_arrays.is_empty() {
            self.io.save_arrays(&complex128_arrays)?;
        }

        // 清理所有保存的数组的元数据缓存
        // 因为 save 操作会更新文件和元数据（shape、modify_time 等）
        for (i, (key, _value)) in arrays.iter().enumerate() {
            let name = if let Some(prefix) = &array_name {
                format!("{}{}", prefix, i)
            } else {
                key.extract::<String>()?
            };

            let meta_cache_key = format!("{}:{}", self.base_dir.display(), name);
            let mut meta_cache = METADATA_CACHE.lock().unwrap();
            meta_cache.remove(&meta_cache_key);
        }

        Ok(())
    }

    #[pyo3(signature = (array_name, lazy=None))]
    fn load(&self, py: Python, array_name: &str, lazy: Option<bool>) -> PyResult<PyObject> {
        let lazy = lazy.unwrap_or(false);

        if lazy {
            // 【性能优化】快速路径 - 最小化操作
            return self.load_lazy_optimized(py, array_name);
        }

        // 【性能关键优化】超快速加载路径 - 最小化所有开销
        // 1. 元数据查询（带缓存）
        let meta_cache_key = format!("{}:{}", self.base_dir.display(), array_name);
        let cached_meta = {
            let meta_cache = METADATA_CACHE.lock().unwrap();
            meta_cache.get(&meta_cache_key).cloned()
        };

        let (dtype, shape, itemsize, modify_time) = if let Some((
            cached_dtype,
            cached_shape,
            cached_itemsize,
            cached_mtime,
        )) = cached_meta
        {
            (cached_dtype, cached_shape, cached_itemsize, cached_mtime)
        } else {
            let meta = match self.io.get_array_meta(array_name) {
                Some(m) => m,
                None => return Err(PyErr::new::<PyKeyError, _>("Array not found")),
            };

            let dtype = meta.get_dtype();
            let shape: Vec<usize> = meta.shape.iter().map(|&x| x as usize).collect();
            let itemsize = dtype.size_bytes() as usize;
            let modify_time = meta.last_modified as i64;

            let mut meta_cache = METADATA_CACHE.lock().unwrap();
            meta_cache.insert(
                meta_cache_key.clone(),
                (dtype, shape.clone(), itemsize, modify_time),
            );

            (dtype, shape, itemsize, modify_time)
        };

        // 2. 平台特定的数据加载策略
        let mut filename = String::with_capacity(5 + array_name.len() + 5); // "data_" + name + ".npkd"
        filename.push_str("data_");
        filename.push_str(array_name);
        filename.push_str(".npkd");
        let data_path = self.base_dir.join(&filename);
        let array_path_string = data_path.to_string_lossy().to_string();

        // Windows 平台专用逻辑：
        // 在 Windows 上，eager load 不使用 mmap 缓存，以避免错误 1224
        // （文件被 mmap 打开时无法执行 save/drop/append 等修改操作）
        #[cfg(windows)]
        let use_mmap_cache = false;

        #[cfg(not(windows))]
        let use_mmap_cache = true;

        let mmap_arc = if use_mmap_cache {
            // Unix 平台（macOS、Linux）：使用 mmap 缓存优化性能
            let mut mmap_cache = MMAP_CACHE.lock().unwrap();
            if let Some((cached_mmap, cached_time)) = mmap_cache.get(&array_path_string) {
                if *cached_time == modify_time {
                    Arc::clone(cached_mmap)
                } else {
                    create_optimized_mmap(&data_path, modify_time, &mut mmap_cache)?
                }
            } else {
                create_optimized_mmap(&data_path, modify_time, &mut mmap_cache)?
            }
        } else {
            // Windows 平台：直接创建 mmap，不缓存
            // 这样在 save/drop/append 时不会有遗留的 mmap 句柄
            let file = File::open(&data_path)?;
            let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
            Arc::new(mmap)
        };

        let mmap_bytes = mmap_arc.as_ref().as_ref();
        let total_elements: usize = shape.iter().product();
        let expected_bytes = total_elements.checked_mul(itemsize).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Array size overflow detected")
        })?;

        if expected_bytes > mmap_bytes.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Data file is smaller than expected: expected {} bytes, found {} bytes",
                expected_bytes,
                mmap_bytes.len()
            )));
        }

        // 【极致性能优化】超快速加载 - 直接复制到NumPy数组
        macro_rules! load_to_numpy {
            ($rust_type:ty) => {{
                let vec: Vec<$rust_type> = py.allow_threads(|| unsafe {
                    let mut vec = Vec::<$rust_type>::with_capacity(total_elements);
                    let src_ptr = mmap_bytes.as_ptr() as *const $rust_type;
                    ptr::copy_nonoverlapping(src_ptr, vec.as_mut_ptr(), total_elements);
                    vec.set_len(total_elements);
                    vec
                });

                ArrayD::from_shape_vec(shape.clone(), vec)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
                    .into_pyarray(py)
                    .into()
            }};
        }

        // 检查是否存在deletion bitmap
        use crate::storage::deletion_bitmap::DeletionBitmap;
        let base_dir = self.io.get_base_dir();
        let (bitmap_opt, actual_physical_rows) = if DeletionBitmap::exists(base_dir, array_name) {
            let bitmap = DeletionBitmap::new(base_dir, array_name, shape[0])
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            let physical_rows = bitmap.get_total_rows();
            (Some(bitmap), physical_rows)
        } else {
            (None, shape[0])
        };

        // 如果有bitmap，使用实际的物理行数更新shape
        let mut actual_shape = shape.clone();
        if bitmap_opt.is_some() {
            actual_shape[0] = actual_physical_rows;
        }

        let array: PyObject = if let Some(bitmap) = bitmap_opt {
            // 有deletion bitmap，需要过滤已删除的行
            let active_indices = bitmap.get_active_indices();
            let active_count = active_indices.len();

            if active_count == 0 {
                // 所有行都被删除了，返回空数组
                let empty_shape = if shape.len() > 1 {
                    let mut new_shape = shape.clone();
                    new_shape[0] = 0;
                    new_shape
                } else {
                    vec![0]
                };

                match dtype {
                    DataType::Bool => ArrayD::from_shape_vec(empty_shape, Vec::<bool>::new())
                        .map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
                        })?
                        .into_pyarray(py)
                        .into(),
                    DataType::Uint8 => ArrayD::from_shape_vec(empty_shape, Vec::<u8>::new())
                        .map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
                        })?
                        .into_pyarray(py)
                        .into(),
                    DataType::Uint16 => ArrayD::from_shape_vec(empty_shape, Vec::<u16>::new())
                        .map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
                        })?
                        .into_pyarray(py)
                        .into(),
                    DataType::Uint32 => ArrayD::from_shape_vec(empty_shape, Vec::<u32>::new())
                        .map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
                        })?
                        .into_pyarray(py)
                        .into(),
                    DataType::Uint64 => ArrayD::from_shape_vec(empty_shape, Vec::<u64>::new())
                        .map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
                        })?
                        .into_pyarray(py)
                        .into(),
                    DataType::Int8 => ArrayD::from_shape_vec(empty_shape, Vec::<i8>::new())
                        .map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
                        })?
                        .into_pyarray(py)
                        .into(),
                    DataType::Int16 => ArrayD::from_shape_vec(empty_shape, Vec::<i16>::new())
                        .map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
                        })?
                        .into_pyarray(py)
                        .into(),
                    DataType::Int32 => ArrayD::from_shape_vec(empty_shape, Vec::<i32>::new())
                        .map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
                        })?
                        .into_pyarray(py)
                        .into(),
                    DataType::Int64 => ArrayD::from_shape_vec(empty_shape, Vec::<i64>::new())
                        .map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
                        })?
                        .into_pyarray(py)
                        .into(),
                    DataType::Float16 => ArrayD::from_shape_vec(empty_shape, Vec::<f16>::new())
                        .map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
                        })?
                        .into_pyarray(py)
                        .into(),
                    DataType::Float32 => ArrayD::from_shape_vec(empty_shape, Vec::<f32>::new())
                        .map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
                        })?
                        .into_pyarray(py)
                        .into(),
                    DataType::Float64 => ArrayD::from_shape_vec(empty_shape, Vec::<f64>::new())
                        .map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
                        })?
                        .into_pyarray(py)
                        .into(),
                    DataType::Complex64 => {
                        ArrayD::from_shape_vec(empty_shape, Vec::<Complex32>::new())
                            .map_err(|e| {
                                PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
                            })?
                            .into_pyarray(py)
                            .into()
                    }
                    DataType::Complex128 => {
                        ArrayD::from_shape_vec(empty_shape, Vec::<Complex64>::new())
                            .map_err(|e| {
                                PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
                            })?
                            .into_pyarray(py)
                            .into()
                    }
                }
            } else {
                // 计算行大小（使用actual_shape以获取正确的物理行数）
                let row_size = if actual_shape.len() > 1 {
                    actual_shape[1..].iter().product::<usize>() * itemsize
                } else {
                    itemsize
                };

                // 构建新的shape（只包含活跃行）
                let mut new_shape = actual_shape.clone();
                new_shape[0] = active_count;
                let new_total_elements = new_shape.iter().product::<usize>();

                // 根据数据类型过滤行
                macro_rules! filter_rows {
                    ($rust_type:ty) => {{
                        let mut vec = Vec::<$rust_type>::with_capacity(new_total_elements);
                        py.allow_threads(|| unsafe {
                            let src_ptr = mmap_bytes.as_ptr() as *const $rust_type;
                            let elements_per_row = row_size / itemsize;

                            for &physical_idx in &active_indices {
                                let src_offset = physical_idx * elements_per_row;
                                ptr::copy_nonoverlapping(
                                    src_ptr.add(src_offset),
                                    vec.as_mut_ptr().add(vec.len()),
                                    elements_per_row,
                                );
                                vec.set_len(vec.len() + elements_per_row);
                            }
                        });

                        ArrayD::from_shape_vec(new_shape.clone(), vec)
                            .map_err(|e| {
                                PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
                            })?
                            .into_pyarray(py)
                            .into()
                    }};
                }

                match dtype {
                    DataType::Bool => {
                        let mut bool_vec = Vec::<bool>::with_capacity(new_total_elements);
                        py.allow_threads(|| {
                            let elements_per_row = row_size;
                            for &physical_idx in &active_indices {
                                let src_offset = physical_idx * elements_per_row;
                                for i in 0..elements_per_row {
                                    bool_vec.push(mmap_bytes[src_offset + i] != 0);
                                }
                            }
                        });
                        ArrayD::from_shape_vec(new_shape.clone(), bool_vec)
                            .map_err(|e| {
                                PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
                            })?
                            .into_pyarray(py)
                            .into()
                    }
                    DataType::Uint8 => filter_rows!(u8),
                    DataType::Uint16 => filter_rows!(u16),
                    DataType::Uint32 => filter_rows!(u32),
                    DataType::Uint64 => filter_rows!(u64),
                    DataType::Int8 => filter_rows!(i8),
                    DataType::Int16 => filter_rows!(i16),
                    DataType::Int32 => filter_rows!(i32),
                    DataType::Int64 => filter_rows!(i64),
                    DataType::Float16 => filter_rows!(f16),
                    DataType::Float32 => filter_rows!(f32),
                    DataType::Float64 => filter_rows!(f64),
                    DataType::Complex64 => filter_rows!(Complex32),
                    DataType::Complex128 => filter_rows!(Complex64),
                }
            }
        } else {
            // 没有deletion bitmap，正常加载
            match dtype {
                DataType::Bool => {
                    let bool_vec: Vec<bool> =
                        py.allow_threads(|| mmap_bytes.iter().map(|&x| x != 0).collect());
                    ArrayD::from_shape_vec(shape.clone(), bool_vec)
                        .map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
                        })?
                        .into_pyarray(py)
                        .into()
                }
                DataType::Uint8 => load_to_numpy!(u8),
                DataType::Uint16 => load_to_numpy!(u16),
                DataType::Uint32 => load_to_numpy!(u32),
                DataType::Uint64 => load_to_numpy!(u64),
                DataType::Int8 => load_to_numpy!(i8),
                DataType::Int16 => load_to_numpy!(i16),
                DataType::Int32 => load_to_numpy!(i32),
                DataType::Int64 => load_to_numpy!(i64),
                DataType::Float16 => load_to_numpy!(f16),
                DataType::Float32 => load_to_numpy!(f32),
                DataType::Float64 => load_to_numpy!(f64),
                DataType::Complex64 => load_to_numpy!(Complex32),
                DataType::Complex128 => load_to_numpy!(Complex64),
            }
        };

        Ok(array)
    }

    fn get_shape(&self, py: Python, array_name: &str) -> PyResult<Py<PyTuple>> {
        if let Some(meta) = self.io.get_array_meta(array_name) {
            let mut shape: Vec<i64> = meta.shape.iter().map(|&x| x as i64).collect();

            // 检查是否存在deletion bitmap
            // 如果存在，shape[0]应该返回逻辑行数（active_count）而不是物理行数
            if !shape.is_empty() {
                use crate::storage::deletion_bitmap::DeletionBitmap;
                let base_dir = self.io.get_base_dir();
                if DeletionBitmap::exists(base_dir, array_name) {
                    if let Ok(bitmap) = DeletionBitmap::new(base_dir, array_name, shape[0] as usize)
                    {
                        shape[0] = bitmap.active_count() as i64;
                    }
                }
            }

            let shape_tuple = PyTuple::new(py, &shape)?;
            Ok(shape_tuple.unbind())
        } else {
            Err(PyErr::new::<PyKeyError, _>(format!(
                "Array {} not found",
                array_name
            )))
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
                array_dict.set_item("dtype", format!("{:?}", meta.get_dtype()))?;
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
        Ok(self
            .io
            .get_array_meta(array_name)
            .map(|meta| meta.last_modified as i64))
    }

    fn reset(&self) -> PyResult<()> {
        self.io.reset()?;
        Ok(())
    }

    /// 强制同步元数据到磁盘
    ///
    /// 用于Batch Mode等批量操作结束时，确保元数据持久化
    fn sync_metadata(&self) -> PyResult<()> {
        self.io.sync_metadata()?;
        Ok(())
    }

    /// 物理整合数组：将逻辑删除的行真正删除
    ///
    /// 该方法会创建一个新的数组文件，只包含未删除的行，然后替换原文件。
    /// 适用于在执行大量删除操作后释放磁盘空间。
    ///
    /// Parameters:
    ///     array_name (str): 要整合的数组名称
    ///
    /// Example:
    ///     ```python
    ///     # 删除一些行
    ///     npk.drop('my_array', indexes=[0, 1, 2])
    ///     
    ///     # 物理整合，真正删除这些行并释放空间
    ///     npk.update('my_array')
    ///     ```
    fn update(&self, array_name: &str) -> PyResult<()> {
        self.io.compact_array(array_name)?;
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
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Dimension mismatch for array {}: expected {}, got {}",
                        name,
                        meta.shape.len(),
                        shape.len()
                    )));
                }

                for (i, (&m, &s)) in meta.shape.iter().zip(shape.iter()).enumerate().skip(1) {
                    if m as usize != s {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Shape mismatch for array {} at dimension {}: expected {}, got {}",
                            name, i, m, s
                        )));
                    }
                }
                existing_arrays.push((name.to_string(), meta.get_dtype(), shape));
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Array {} does not exist",
                    name
                )));
            }
        }

        // Start appending data
        for (key, array) in arrays.iter() {
            let name = key.extract::<String>()?;
            let meta = self.io.get_array_meta(&name).unwrap();
            let shape: Vec<usize> = array.getattr("shape")?.extract()?;

            // Append data to file
            let array_path = self.base_dir.join(&meta.data_file);
            let mut file = OpenOptions::new().append(true).open(array_path)?;

            match meta.get_dtype() {
                DataType::Bool => {
                    let py_array = array.downcast::<PyArrayDyn<bool>>()?;
                    let readonly = py_array.readonly();
                    let array_ref = unsafe { readonly.as_array() };
                    let data = array_ref.as_slice().unwrap();
                    file.write_all(bytemuck::cast_slice(data))?;
                }
                DataType::Uint8 => {
                    let py_array = array.downcast::<PyArrayDyn<u8>>()?;
                    let readonly = py_array.readonly();
                    let array_ref = unsafe { readonly.as_array() };
                    let data = array_ref.as_slice().unwrap();
                    file.write_all(bytemuck::cast_slice(data))?;
                }
                DataType::Uint16 => {
                    let py_array = array.downcast::<PyArrayDyn<u16>>()?;
                    let readonly = py_array.readonly();
                    let array_ref = unsafe { readonly.as_array() };
                    let data = array_ref.as_slice().unwrap();
                    file.write_all(bytemuck::cast_slice(data))?;
                }
                DataType::Uint32 => {
                    let py_array = array.downcast::<PyArrayDyn<u32>>()?;
                    let readonly = py_array.readonly();
                    let array_ref = unsafe { readonly.as_array() };
                    let data = array_ref.as_slice().unwrap();
                    file.write_all(bytemuck::cast_slice(data))?;
                }
                DataType::Uint64 => {
                    let py_array = array.downcast::<PyArrayDyn<u64>>()?;
                    let readonly = py_array.readonly();
                    let array_ref = unsafe { readonly.as_array() };
                    let data = array_ref.as_slice().unwrap();
                    file.write_all(bytemuck::cast_slice(data))?;
                }
                DataType::Int8 => {
                    let py_array = array.downcast::<PyArrayDyn<i8>>()?;
                    let readonly = py_array.readonly();
                    let array_ref = unsafe { readonly.as_array() };
                    let data = array_ref.as_slice().unwrap();
                    file.write_all(bytemuck::cast_slice(data))?;
                }
                DataType::Int16 => {
                    let py_array = array.downcast::<PyArrayDyn<i16>>()?;
                    let readonly = py_array.readonly();
                    let array_ref = unsafe { readonly.as_array() };
                    let data = array_ref.as_slice().unwrap();
                    file.write_all(bytemuck::cast_slice(data))?;
                }
                DataType::Int32 => {
                    let py_array = array.downcast::<PyArrayDyn<i32>>()?;
                    let readonly = py_array.readonly();
                    let array_ref = unsafe { readonly.as_array() };
                    let data = array_ref.as_slice().unwrap();
                    file.write_all(bytemuck::cast_slice(data))?;
                }
                DataType::Int64 => {
                    let py_array = array.downcast::<PyArrayDyn<i64>>()?;
                    let readonly = py_array.readonly();
                    let array_ref = unsafe { readonly.as_array() };
                    let data = array_ref.as_slice().unwrap();
                    file.write_all(bytemuck::cast_slice(data))?;
                }
                DataType::Float16 => {
                    let py_array = array.downcast::<PyArrayDyn<f16>>()?;
                    let readonly = py_array.readonly();
                    let array_ref = unsafe { readonly.as_array() };
                    let data = array_ref.as_slice().unwrap();
                    file.write_all(bytemuck::cast_slice(data))?;
                }
                DataType::Float32 => {
                    let py_array = array.downcast::<PyArrayDyn<f32>>()?;
                    let readonly = py_array.readonly();
                    let array_ref = unsafe { readonly.as_array() };
                    let data = array_ref.as_slice().unwrap();
                    file.write_all(bytemuck::cast_slice(data))?;
                }
                DataType::Float64 => {
                    let py_array = array.downcast::<PyArrayDyn<f64>>()?;
                    let readonly = py_array.readonly();
                    let array_ref = unsafe { readonly.as_array() };
                    let data = array_ref.as_slice().unwrap();
                    file.write_all(bytemuck::cast_slice(data))?;
                }
                DataType::Complex64 => {
                    let py_array = array.downcast::<PyArrayDyn<Complex32>>()?;
                    let readonly = py_array.readonly();
                    let array_ref = unsafe { readonly.as_array() };
                    let data = array_ref.as_slice().unwrap();
                    let bytes = unsafe {
                        std::slice::from_raw_parts(
                            data.as_ptr() as *const u8,
                            data.len() * std::mem::size_of::<Complex32>(),
                        )
                    };
                    file.write_all(bytes)?;
                }
                DataType::Complex128 => {
                    let py_array = array.downcast::<PyArrayDyn<Complex64>>()?;
                    let readonly = py_array.readonly();
                    let array_ref = unsafe { readonly.as_array() };
                    let data = array_ref.as_slice().unwrap();
                    let bytes = unsafe {
                        std::slice::from_raw_parts(
                            data.as_ptr() as *const u8,
                            data.len() * std::mem::size_of::<Complex64>(),
                        )
                    };
                    file.write_all(bytes)?;
                }
            }

            // Update metadata
            let mut new_meta = meta.clone();
            new_meta.shape[0] += shape[0] as u64;
            new_meta.size_bytes =
                new_meta.total_elements() * new_meta.get_dtype().size_bytes() as u64;
            new_meta.last_modified = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_micros() as u64;

            // 如果存在deletion bitmap，需要扩展它以包含新增的行
            use crate::storage::deletion_bitmap::DeletionBitmap;
            if DeletionBitmap::exists(&self.base_dir, &name) {
                let old_total_rows = meta.shape[0] as usize;

                // 加载bitmap（使用追加前的行数）
                let mut bitmap = DeletionBitmap::new(&self.base_dir, &name, old_total_rows)?;
                // 扩展bitmap以包含新追加的行
                bitmap.extend(shape[0]);
                bitmap.save()?;
            }

            self.io.update_array_metadata(&name, new_meta)?;

            // 清除元数据缓存，以便下次load时使用新的元数据
            let meta_cache_key = format!("{}:{}", self.base_dir.display(), name);
            {
                let mut meta_cache = METADATA_CACHE.lock().unwrap();
                meta_cache.remove(&meta_cache_key);
            }
        }

        Ok(())
    }

    #[pyo3(signature = (array_names, indexes=None))]
    fn drop(
        &self,
        array_names: &Bound<'_, PyAny>,
        indexes: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let names = if let Ok(list) = array_names.downcast::<PyList>() {
            list.iter()
                .map(|name| name.extract::<String>())
                .collect::<PyResult<Vec<_>>>()?
        } else if let Ok(name) = array_names.extract::<String>() {
            vec![name]
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "array_names must be a list of strings or a single string",
            ));
        };

        // If indexes parameter is provided, it means deleting specific rows
        if let Some(indexes) = indexes {
            for name in &names {
                if let Some(meta) = self.io.get_array_meta(name) {
                    // Get the indices of the rows to delete
                    let deleted_indices = if let Ok(slice) = indexes.downcast::<PySlice>() {
                        let start = slice
                            .getattr("start")?
                            .extract::<Option<i64>>()?
                            .unwrap_or(0);
                        let stop = slice
                            .getattr("stop")?
                            .extract::<Option<i64>>()?
                            .unwrap_or(meta.shape[0] as i64);
                        let step = slice
                            .getattr("step")?
                            .extract::<Option<i64>>()?
                            .unwrap_or(1);

                        let start = if start < 0 {
                            meta.shape[0] as i64 + start
                        } else {
                            start
                        };
                        let stop = if stop < 0 {
                            meta.shape[0] as i64 + stop
                        } else {
                            stop
                        };

                        if step == 1 {
                            (start..stop).collect::<Vec<i64>>()
                        } else {
                            Vec::new()
                        }
                    } else if let Ok(indices) = indexes.extract::<Vec<i64>>() {
                        // Process negative indices
                        indices
                            .into_iter()
                            .map(|idx| {
                                if idx < 0 {
                                    meta.shape[0] as i64 + idx
                                } else {
                                    idx
                                }
                            })
                            .collect()
                    } else {
                        return Err(pyo3::exceptions::PyTypeError::new_err(
                            "indexes must be a slice or list of integers",
                        ));
                    };

                    // Windows 平台：修改文件前清理 mmap 缓存（lazy load 可能创建的）
                    // Unix 平台：系统允许同时 mmap 和修改文件，不需要清理
                    #[cfg(windows)]
                    clear_mmap_cache_for_array(&self.base_dir, name);

                    self.io.drop_arrays(name, Some(&deleted_indices))?;

                    // 清除元数据缓存
                    let meta_cache_key = format!("{}:{}", self.base_dir.display(), name);
                    {
                        let mut meta_cache = METADATA_CACHE.lock().unwrap();
                        meta_cache.remove(&meta_cache_key);
                    }
                }
            }

            Ok(())
        } else {
            // Windows 平台：批量删除数组前清理 mmap 缓存
            // Unix 平台：系统允许同时 mmap 和修改文件，不需要清理
            #[cfg(windows)]
            for name in &names {
                clear_mmap_cache_for_array(&self.base_dir, name);
            }

            self.io.batch_drop_arrays(&names, None)?;

            // 清除所有被删除数组的元数据缓存
            for name in &names {
                let meta_cache_key = format!("{}:{}", self.base_dir.display(), name);
                {
                    let mut meta_cache = METADATA_CACHE.lock().unwrap();
                    meta_cache.remove(&meta_cache_key);
                }
            }

            Ok(())
        }
    }

    fn get_array_path(&self, array_name: &str) -> PathBuf {
        self.base_dir
            .join(&self.io.get_array_metadata(array_name).unwrap().data_file)
    }

    fn replace(
        &self,
        arrays: &Bound<'_, PyDict>,
        indexes: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        // Get the indices of the rows to replace
        let indices = if let Some(idx) = indexes {
            if let Ok(indices) = idx.extract::<Vec<i64>>() {
                indices
            } else if let Ok(slice) = idx.downcast::<PySlice>() {
                let start = slice
                    .getattr("start")?
                    .extract::<Option<i64>>()?
                    .unwrap_or(0);
                let stop = slice
                    .getattr("stop")?
                    .extract::<Option<i64>>()?
                    .unwrap_or(-1);
                let step = slice
                    .getattr("step")?
                    .extract::<Option<i64>>()?
                    .unwrap_or(1);

                if step != 1 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Only step=1 is supported for slices",
                    ));
                }

                (start..stop).collect()
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "indexes must be a list of integers or a slice",
                ));
            }
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "indexes parameter is required for replace operation",
            ));
        };

        // Process each array to replace
        for (key, value) in arrays.iter() {
            let name = key.extract::<String>()?;

            // Check if the array exists
            if !self.io.has_array(&name) {
                return Err(PyErr::new::<PyKeyError, _>(format!(
                    "Array {} not found",
                    name
                )));
            }

            let meta = self.io.get_array_meta(&name).unwrap();
            let new_shape: Vec<usize> = value.getattr("shape")?.extract()?;

            // Check if the dimensions match
            if new_shape.len() != meta.shape.len() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Dimension mismatch for array {}: expected {}, got {}",
                    name,
                    meta.shape.len(),
                    new_shape.len()
                )));
            }

            // Check if the other dimensions match
            for (i, (&m, &s)) in meta.shape.iter().zip(new_shape.iter()).enumerate().skip(1) {
                if m as usize != s {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Shape mismatch for array {} at dimension {}: expected {}, got {}",
                        name, i, m, s
                    )));
                }
            }

            // Check if the indices are within bounds
            for &idx in &indices {
                let normalized_idx = if idx < 0 {
                    meta.shape[0] as i64 + idx
                } else {
                    idx
                };
                if normalized_idx < 0 || normalized_idx >= meta.shape[0] as i64 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Index {} is out of bounds for array {} with shape {:?}",
                        idx, name, meta.shape
                    )));
                }
            }

            // Perform the replace operation
            match meta.get_dtype() {
                DataType::Bool => {
                    let array = value.downcast::<PyArrayDyn<bool>>()?;
                    let readonly = array.readonly();
                    let array = unsafe { readonly.as_array().to_owned() };
                    self.io.replace_rows(&name, &array, &indices)?;
                }
                DataType::Uint8 => {
                    let array = value.downcast::<PyArrayDyn<u8>>()?;
                    let readonly = array.readonly();
                    let array = unsafe { readonly.as_array().to_owned() };
                    self.io.replace_rows(&name, &array, &indices)?;
                }
                DataType::Uint16 => {
                    let array = value.downcast::<PyArrayDyn<u16>>()?;
                    let readonly = array.readonly();
                    let array = unsafe { readonly.as_array().to_owned() };
                    self.io.replace_rows(&name, &array, &indices)?;
                }
                DataType::Uint32 => {
                    let array = value.downcast::<PyArrayDyn<u32>>()?;
                    let readonly = array.readonly();
                    let array = unsafe { readonly.as_array().to_owned() };
                    self.io.replace_rows(&name, &array, &indices)?;
                }
                DataType::Uint64 => {
                    let array = value.downcast::<PyArrayDyn<u64>>()?;
                    let readonly = array.readonly();
                    let array = unsafe { readonly.as_array().to_owned() };
                    self.io.replace_rows(&name, &array, &indices)?;
                }
                DataType::Int8 => {
                    let array = value.downcast::<PyArrayDyn<i8>>()?;
                    let readonly = array.readonly();
                    let array = unsafe { readonly.as_array().to_owned() };
                    self.io.replace_rows(&name, &array, &indices)?;
                }
                DataType::Int16 => {
                    let array = value.downcast::<PyArrayDyn<i16>>()?;
                    let readonly = array.readonly();
                    let array = unsafe { readonly.as_array().to_owned() };
                    self.io.replace_rows(&name, &array, &indices)?;
                }
                DataType::Int32 => {
                    let array = value.downcast::<PyArrayDyn<i32>>()?;
                    let readonly = array.readonly();
                    let array = unsafe { readonly.as_array().to_owned() };
                    self.io.replace_rows(&name, &array, &indices)?;
                }
                DataType::Int64 => {
                    let array = value.downcast::<PyArrayDyn<i64>>()?;
                    let readonly = array.readonly();
                    let array = unsafe { readonly.as_array().to_owned() };
                    self.io.replace_rows(&name, &array, &indices)?;
                }
                DataType::Float16 => {
                    let array = value.downcast::<PyArrayDyn<f16>>()?;
                    let readonly = array.readonly();
                    let array = unsafe { readonly.as_array().to_owned() };
                    self.io.replace_rows(&name, &array, &indices)?;
                }
                DataType::Float32 => {
                    let array = value.downcast::<PyArrayDyn<f32>>()?;
                    let readonly = array.readonly();
                    let array = unsafe { readonly.as_array().to_owned() };
                    self.io.replace_rows(&name, &array, &indices)?;
                }
                DataType::Float64 => {
                    let array = value.downcast::<PyArrayDyn<f64>>()?;
                    let readonly = array.readonly();
                    let array = unsafe { readonly.as_array().to_owned() };
                    self.io.replace_rows(&name, &array, &indices)?;
                }
                DataType::Complex64 => {
                    let array = value.downcast::<PyArrayDyn<Complex32>>()?;
                    let readonly = array.readonly();
                    let array = unsafe { readonly.as_array().to_owned() };
                    self.io.replace_rows(&name, &array, &indices)?;
                }
                DataType::Complex128 => {
                    let array = value.downcast::<PyArrayDyn<Complex64>>()?;
                    let readonly = array.readonly();
                    let array = unsafe { readonly.as_array().to_owned() };
                    self.io.replace_rows(&name, &array, &indices)?;
                }
            }
        }

        Ok(())
    }

    fn getitem(
        &self,
        py: Python,
        array_name: &str,
        indices: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        let meta = self.io.get_array_meta(array_name).ok_or_else(|| {
            PyErr::new::<PyKeyError, _>(format!("Array {} not found", array_name))
        })?;

        // 【性能优化】使用mmap路径读取数据
        let shape: Vec<usize> = meta.shape.iter().map(|&x| x as usize).collect();
        let dtype = meta.get_dtype();
        let itemsize = dtype.size_bytes() as usize;
        let row_size = shape[1..].iter().product::<usize>() * itemsize;
        let physical_rows = shape[0];
        let modify_time = meta.last_modified as i64;
        
        // 检查是否存在deletion bitmap，获取逻辑行数
        use crate::storage::deletion_bitmap::DeletionBitmap;
        let base_dir = self.io.get_base_dir();
        let bitmap_opt = if DeletionBitmap::exists(base_dir, array_name) {
            Some(DeletionBitmap::new(base_dir, array_name, physical_rows)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?)
        } else {
            None
        };
        let logical_rows = bitmap_opt.as_ref().map_or(physical_rows, |b| b.active_count());
        
        // 构建数据文件路径
        let data_path = self.base_dir.join(format!("data_{}.npkd", array_name));
        let array_path_string = data_path.to_string_lossy().to_string();
        
        // 获取或创建mmap
        let mmap_arc = {
            let mut mmap_cache = MMAP_CACHE.lock().unwrap();
            if let Some((cached_mmap, cached_time)) = mmap_cache.get(&array_path_string) {
                if *cached_time == modify_time {
                    Arc::clone(cached_mmap)
                } else {
                    create_optimized_mmap(&data_path, modify_time, &mut mmap_cache)?
                }
            } else {
                create_optimized_mmap(&data_path, modify_time, &mut mmap_cache)?
            }
        };
        
        let mmap_slice = &mmap_arc[..];
        let mmap_len = mmap_slice.len();

        // 【快速路径】检测slice并使用单次memcpy（仅当无bitmap时）
        if let Ok(slice) = indices.downcast::<PySlice>() {
            let start = slice
                .getattr("start")?
                .extract::<Option<i64>>()?
                .unwrap_or(0);
            let stop = slice
                .getattr("stop")?
                .extract::<Option<i64>>()?
                .unwrap_or(logical_rows as i64);
            let step = slice
                .getattr("step")?
                .extract::<Option<i64>>()?
                .unwrap_or(1);

            if step != 1 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Only step=1 is supported for slices",
                ));
            }

            // 标准化逻辑索引
            let start_idx = if start < 0 { (logical_rows as i64 + start).max(0) as usize } else { start as usize };
            let stop_idx = if stop < 0 { (logical_rows as i64 + stop).max(0) as usize } else { (stop as usize).min(logical_rows) };
            
            if start_idx >= stop_idx || start_idx >= logical_rows {
                // 空切片
                let mut new_shape = shape.clone();
                new_shape[0] = 0;
                return self.create_numpy_array_from_dtype(py, vec![], &new_shape, dtype);
            }
            
            let num_rows = stop_idx - start_idx;
            
            // 如果存在bitmap，需要转换逻辑索引到物理索引
            if let Some(ref bitmap) = bitmap_opt {
                // 先转换所有逻辑索引到物理索引
                let mut physical_indices: Vec<usize> = Vec::with_capacity(num_rows);
                for logical_idx in start_idx..stop_idx {
                    let physical_idx = bitmap.logical_to_physical(logical_idx)
                        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                            format!("Logical index {} out of bounds", logical_idx)
                        ))?;
                    physical_indices.push(physical_idx);
                }
                
                // 【优化】检测连续物理索引块，使用批量memcpy
                let total_size = num_rows * row_size;
                let mut data = vec![0u8; total_size];
                let mut dst_offset = 0;
                let mut i = 0;
                
                while i < physical_indices.len() {
                    // 找到连续块的结束位置
                    let block_start = i;
                    let physical_start = physical_indices[i];
                    while i + 1 < physical_indices.len() 
                          && physical_indices[i + 1] == physical_indices[i] + 1 {
                        i += 1;
                    }
                    let block_len = i - block_start + 1;
                    
                    // 批量复制整个连续块
                    let src_offset = physical_start * row_size;
                    let block_size = block_len * row_size;
                    if src_offset + block_size > mmap_len {
                        return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>("Data access out of bounds"));
                    }
                    data[dst_offset..dst_offset + block_size]
                        .copy_from_slice(&mmap_slice[src_offset..src_offset + block_size]);
                    
                    dst_offset += block_size;
                    i += 1;
                }
                
                let mut new_shape = shape.clone();
                new_shape[0] = num_rows;
                return self.create_numpy_array_from_dtype(py, data, &new_shape, dtype);
            }
            
            // 无bitmap时，直接连续读取
            let src_offset = start_idx * row_size;
            let total_size = num_rows * row_size;
            
            if src_offset + total_size > mmap_len {
                return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                    "Slice access out of bounds"
                ));
            }
            
            // 【零复制优化】直接从mmap创建typed Vec，跳过中间的Vec<u8>
            let mut new_shape = shape.clone();
            new_shape[0] = num_rows;
            
            // 针对最常用类型的快速路径
            return match dtype {
                DataType::Float32 => {
                    let count = total_size / std::mem::size_of::<f32>();
                    let mut typed_data: Vec<f32> = Vec::with_capacity(count);
                    unsafe {
                        typed_data.set_len(count);
                        std::ptr::copy_nonoverlapping(
                            mmap_slice[src_offset..].as_ptr(),
                            typed_data.as_mut_ptr() as *mut u8,
                            total_size
                        );
                    }
                    let array = unsafe {
                        ArrayD::from_shape_vec_unchecked(new_shape, typed_data)
                    };
                    Ok(array.into_pyarray(py).into())
                }
                DataType::Float64 => {
                    let count = total_size / std::mem::size_of::<f64>();
                    let mut typed_data: Vec<f64> = Vec::with_capacity(count);
                    unsafe {
                        typed_data.set_len(count);
                        std::ptr::copy_nonoverlapping(
                            mmap_slice[src_offset..].as_ptr(),
                            typed_data.as_mut_ptr() as *mut u8,
                            total_size
                        );
                    }
                    let array = unsafe {
                        ArrayD::from_shape_vec_unchecked(new_shape, typed_data)
                    };
                    Ok(array.into_pyarray(py).into())
                }
                DataType::Int32 => {
                    let count = total_size / std::mem::size_of::<i32>();
                    let mut typed_data: Vec<i32> = Vec::with_capacity(count);
                    unsafe {
                        typed_data.set_len(count);
                        std::ptr::copy_nonoverlapping(
                            mmap_slice[src_offset..].as_ptr(),
                            typed_data.as_mut_ptr() as *mut u8,
                            total_size
                        );
                    }
                    let array = unsafe {
                        ArrayD::from_shape_vec_unchecked(new_shape, typed_data)
                    };
                    Ok(array.into_pyarray(py).into())
                }
                DataType::Int64 => {
                    let count = total_size / std::mem::size_of::<i64>();
                    let mut typed_data: Vec<i64> = Vec::with_capacity(count);
                    unsafe {
                        typed_data.set_len(count);
                        std::ptr::copy_nonoverlapping(
                            mmap_slice[src_offset..].as_ptr(),
                            typed_data.as_mut_ptr() as *mut u8,
                            total_size
                        );
                    }
                    let array = unsafe {
                        ArrayD::from_shape_vec_unchecked(new_shape, typed_data)
                    };
                    Ok(array.into_pyarray(py).into())
                }
                DataType::Uint8 => {
                    let data = mmap_slice[src_offset..src_offset + total_size].to_vec();
                    let array = unsafe {
                        ArrayD::from_shape_vec_unchecked(new_shape, data)
                    };
                    Ok(array.into_pyarray(py).into())
                }
                _ => {
                    // 其他类型走通用路径
                    let data = mmap_slice[src_offset..src_offset + total_size].to_vec();
                    self.create_numpy_array_from_dtype(py, data, &new_shape, dtype)
                }
            };
        }
        
        // 【普通路径】索引列表访问
        let indices: Vec<i64> = if let Ok(idx_list) = indices.extract::<Vec<i64>>() {
            idx_list
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "indices must be a list of integers or a slice",
            ));
        };
        
        let num_indices = indices.len();
        
        // 先转换所有逻辑索引到物理索引
        let mut physical_indices: Vec<usize> = Vec::with_capacity(num_indices);
        for &idx in &indices {
            let logical_idx = if idx < 0 { logical_rows as i64 + idx } else { idx };
            if logical_idx < 0 || logical_idx as usize >= logical_rows {
                return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                    "Index {} is out of bounds", idx
                )));
            }
            
            let physical_idx = if let Some(ref bitmap) = bitmap_opt {
                bitmap.logical_to_physical(logical_idx as usize)
                    .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                        format!("Logical index {} out of bounds", logical_idx)
                    ))?
            } else {
                logical_idx as usize
            };
            physical_indices.push(physical_idx);
        }
        
        // 【优化】检测连续物理索引块，使用批量memcpy
        let total_size = num_indices * row_size;
        let mut data = vec![0u8; total_size];
        let mut dst_offset = 0;
        let mut i = 0;
        
        while i < physical_indices.len() {
            // 找到连续块的结束位置
            let block_start = i;
            let physical_start = physical_indices[i];
            while i + 1 < physical_indices.len() 
                  && physical_indices[i + 1] == physical_indices[i] + 1 {
                i += 1;
            }
            let block_len = i - block_start + 1;
            
            // 批量复制整个连续块
            let src_offset = physical_start * row_size;
            let block_size = block_len * row_size;
            if src_offset + block_size > mmap_len {
                return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>("Data access out of bounds"));
            }
            data[dst_offset..dst_offset + block_size]
                .copy_from_slice(&mmap_slice[src_offset..src_offset + block_size]);
            
            dst_offset += block_size;
            i += 1;
        }

        // Calculate the new shape
        let mut new_shape = meta.shape.iter().map(|&x| x as usize).collect::<Vec<_>>();
        new_shape[0] = indices.len();

        // Create a NumPy array based on the data type
        let array: PyObject = match meta.get_dtype() {
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
            DataType::Complex64 => {
                let array = unsafe {
                    let slice = std::slice::from_raw_parts(
                        data.as_ptr() as *const Complex32,
                        data.len() / std::mem::size_of::<Complex32>(),
                    );
                    ArrayD::from_shape_vec_unchecked(new_shape, slice.to_vec())
                };
                array.into_pyarray(py).into()
            }
            DataType::Complex128 => {
                let array = unsafe {
                    let slice = std::slice::from_raw_parts(
                        data.as_ptr() as *const Complex64,
                        data.len() / std::mem::size_of::<Complex64>(),
                    );
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
                data_file: self
                    .base_dir
                    .join(&meta.data_file)
                    .to_string_lossy()
                    .to_string(),
            })
        } else {
            Err(PyErr::new::<PyKeyError, _>(format!(
                "Array {} not found",
                array_name
            )))
        }
    }

    // 创建高性能LazyArray
    fn stream_load(
        &self,
        py: Python,
        array_name: &str,
        buffer_size: i64,
    ) -> PyResult<StreamLoader> {
        if !self.io.has_array(array_name) {
            return Err(PyErr::new::<PyKeyError, _>(format!(
                "Array {} not found",
                array_name
            )));
        }

        let meta = self.io.get_array_meta(array_name).unwrap();
        let total_rows = meta.shape[0] as i64;
        let shape: Vec<usize> = meta.shape.iter().map(|&x| x as usize).collect();

        Ok(StreamLoader {
            base_dir: self.base_dir.clone(),
            array_name: array_name.to_string(),
            total_rows,
            buffer_size,
            current_index: 0,
            dtype: meta.get_dtype(),
            shape,
        })
    }

    // ===========================
    // 性能优化的lazy load
    // ===========================

    /// 【性能优化】超快速lazy load实现
    ///
    /// 优化策略：
    /// 1. 元数据缓存 - 避免重复元数据查询
    /// 2. 快速cache查找 - 跳过文件系统检查
    /// 3. 延迟创建优化引擎
    /// 4. 最小化字符串分配
    /// 5. 立即释放锁减少竞争
    #[inline(always)]
    fn load_lazy_optimized(&self, py: Python, array_name: &str) -> PyResult<PyObject> {
        let data_path = self.base_dir.join(format!("data_{}.npkd", array_name));
        let array_path_string = data_path.to_string_lossy().to_string();

        // 【优化1】元数据缓存查找 - 避免重复IO操作
        let meta_cache_key = format!("{}:{}", self.base_dir.display(), array_name);

        let (dtype, shape, itemsize, modify_time, mmap) = {
            // 尝试从元数据缓存获取
            let meta_cache = METADATA_CACHE.lock().unwrap();
            if let Some((cached_dtype, cached_shape, cached_itemsize, cached_mtime)) =
                meta_cache.get(&meta_cache_key)
            {
                // 元数据cache命中 - 超快速路径
                let dtype = *cached_dtype;
                let shape = cached_shape.clone();
                let itemsize = *cached_itemsize;
                let mtime = *cached_mtime;
                drop(meta_cache); // 立即释放元数据cache锁

                // 获取mmap（应该也在cache中）
                let mut mmap_cache = MMAP_CACHE.lock().unwrap();
                let mmap = if let Some((cached_mmap, cached_time)) =
                    mmap_cache.get(&array_path_string)
                {
                    if *cached_time == mtime {
                        let mmap_ref = Arc::clone(cached_mmap);
                        drop(mmap_cache); // 立即释放mmap cache锁
                        mmap_ref
                    } else {
                        // 时间不匹配，需要重建（罕见情况）
                        let new_mmap = create_optimized_mmap(&data_path, mtime, &mut mmap_cache)?;
                        drop(mmap_cache);
                        new_mmap
                    }
                } else {
                    // Mmap未cache，但元数据已cache（不太可能）
                    let new_mmap = create_optimized_mmap(&data_path, mtime, &mut mmap_cache)?;
                    drop(mmap_cache);
                    new_mmap
                };

                (dtype, shape, itemsize, mtime, mmap)
            } else {
                // 元数据未cache - 需要完整加载
                drop(meta_cache); // 释放读锁

                let meta = self.io.get_array_metadata(array_name)?;
                let dtype = meta.get_dtype();
                let shape: Vec<usize> = meta.shape.iter().map(|&x| x as usize).collect();
                let itemsize = dtype.size_bytes() as usize;
                let mtime = meta.last_modified as i64;

                // 获取或创建mmap
                let mut mmap_cache = MMAP_CACHE.lock().unwrap();
                let mmap = if let Some((cached_mmap, cached_time)) =
                    mmap_cache.get(&array_path_string)
                {
                    if *cached_time == mtime {
                        Arc::clone(cached_mmap)
                    } else {
                        create_optimized_mmap(&data_path, mtime, &mut mmap_cache)?
                    }
                } else {
                    // 首次加载 - 需要文件检查
                    if !data_path.exists() {
                        return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
                            format!("Data file not found: {}", data_path.display()),
                        ));
                    }
                    if data_path.is_dir() {
                        return Err(PyErr::new::<pyo3::exceptions::PyIsADirectoryError, _>(
                            format!("Expected file but found directory: {}", data_path.display()),
                        ));
                    }
                    create_optimized_mmap(&data_path, mtime, &mut mmap_cache)?
                };
                drop(mmap_cache);

                // 缓存元数据
                let mut meta_cache = METADATA_CACHE.lock().unwrap();
                meta_cache.insert(meta_cache_key, (dtype, shape.clone(), itemsize, mtime));
                drop(meta_cache);

                (dtype, shape, itemsize, mtime, mmap)
            }
        };

        // 【关键修复】检测并加载deletion bitmap以支持逻辑视图
        let base_dir = self.io.get_base_dir();
        let logical_map = LogicalRowMap::new(base_dir, array_name, shape[0])?;

        // 【优化2】使用新的 LazyArray（支持算术操作符）
        let mut lazy_array = crate::lazy_array::standard::LazyArray::new(
            mmap,
            shape,
            dtype,
            itemsize,
            array_path_string,
            modify_time,
        );

        // 设置 logical_rows 字段
        lazy_array.logical_rows = logical_map;

        // 【优化3】快速对象创建
        Ok(Py::new(py, lazy_array)?.into())
    }

    // ===========================
    // Array Cloning
    // ===========================

    /// Clone an existing array to a new array with a different name
    ///
    /// This method creates a complete copy of the source array, including its data and metadata.
    /// The cloned array is independent of the original and can be modified separately.
    ///
    /// Parameters:
    ///     source_name (str): Name of the source array to clone
    ///     target_name (str): Name for the cloned array
    ///
    /// Example:
    ///     ```python
    ///     # Clone an existing array
    ///     npk.clone('original_array', 'cloned_array')
    ///     
    ///     # The cloned array can now be modified independently
    ///     data = npk.load('cloned_array')
    ///     ```
    ///
    /// Raises:
    ///     PyKeyError: If source array doesn't exist
    ///     PyValueError: If target array already exists
    fn clone(&self, py: Python, source_name: &str, target_name: &str) -> PyResult<()> {
        // Check if source array exists
        if !self.io.has_array(source_name) {
            return Err(PyErr::new::<PyKeyError, _>(format!(
                "Source array '{}' not found",
                source_name
            )));
        }

        // Check if target array already exists
        if self.io.has_array(target_name) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Target array '{}' already exists. Please use a different name or drop the existing array first.", target_name)
            ));
        }

        // Get source array metadata
        let meta = self.io.get_array_meta(source_name).ok_or_else(|| {
            PyErr::new::<PyKeyError, _>(format!(
                "Failed to get metadata for array '{}'",
                source_name
            ))
        })?;

        let dtype = meta.get_dtype();

        // Load the source array data (eager mode)
        let source_data = self.load(py, source_name, Some(false))?;

        // Create a dictionary to save the cloned data
        let dict = PyDict::new(py);
        dict.set_item(target_name, source_data)?;

        // Save the cloned array with the new name
        self.save(&dict, None)?;

        // Clear metadata cache for the new array to ensure fresh reads
        let meta_cache_key = format!("{}:{}", self.base_dir.display(), target_name);
        let mut meta_cache = METADATA_CACHE.lock().unwrap();
        meta_cache.remove(&meta_cache_key);

        Ok(())
    }

    // ===========================
    // Context Manager支持
    // ===========================

    /// 显式关闭NumPack实例并释放所有资源
    fn close(&mut self, py: Python) -> PyResult<()> {
        // 【性能优化】快速close - 只同步元数据，其他资源由RAII自动清理
        // 移除耗时的handle_manager.force_cleanup_and_wait操作

        py.allow_threads(|| {
            if let Err(e) = self.io.sync_metadata() {
                // 静默忽略"目录不存在"错误（这是测试清理时的正常情况）
                let error_msg = e.to_string();
                let is_not_found_error = error_msg.contains("No such file or directory")
                    || error_msg.contains("cannot find the path")
                    || error_msg.contains("system cannot find the path")
                    || error_msg.contains("os error 2");

                if !is_not_found_error {
                    eprintln!("Warning: Metadata synchronization failed: {}", e);
                }
            }
        });

        // 清除临时文件缓存（轻量级操作）
        clear_temp_files_from_cache();

        // Rust的RAII会自动清理文件句柄和内存映射
        // 无需显式等待清理完成

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
        Ok(false) // 不抑制异常
    }
}

#[pymethods]
impl StreamLoader {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python) -> PyResult<Option<PyObject>> {
        if self.current_index >= self.total_rows {
            return Ok(None);
        }

        let end_index = std::cmp::min(self.current_index + self.buffer_size, self.total_rows);
        let batch_size = (end_index - self.current_index) as usize;

        // 直接从文件加载数据
        let data_file = self.base_dir.join(format!("data_{}.npkd", self.array_name));
        let file = std::fs::File::open(&data_file)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        // 计算偏移量和大小
        let element_size = self.dtype.size_bytes();
        let row_size = self.shape[1..].iter().product::<usize>() * element_size;
        let offset = self.current_index as usize * row_size;
        let size = batch_size * row_size;

        if offset + size > mmap.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "Data offset out of bounds",
            ));
        }

        // 创建新的形状
        let mut new_shape = self.shape.clone();
        new_shape[0] = batch_size;

        // 根据数据类型创建numpy数组
        let array = match self.dtype {
            DataType::Bool => {
                let data = unsafe {
                    let slice =
                        std::slice::from_raw_parts(mmap.as_ptr().add(offset) as *const bool, size);
                    ArrayD::from_shape_vec(new_shape, slice.to_vec()).unwrap()
                };
                data.into_pyarray(py).into()
            }
            DataType::Uint8 => {
                let data = unsafe {
                    let slice =
                        std::slice::from_raw_parts(mmap.as_ptr().add(offset) as *const u8, size);
                    ArrayD::from_shape_vec(new_shape, slice.to_vec()).unwrap()
                };
                data.into_pyarray(py).into()
            }
            DataType::Uint16 => {
                let data = unsafe {
                    let slice = std::slice::from_raw_parts(
                        mmap.as_ptr().add(offset) as *const u16,
                        size / 2,
                    );
                    ArrayD::from_shape_vec(new_shape, slice.to_vec()).unwrap()
                };
                data.into_pyarray(py).into()
            }
            DataType::Uint32 => {
                let data = unsafe {
                    let slice = std::slice::from_raw_parts(
                        mmap.as_ptr().add(offset) as *const u32,
                        size / 4,
                    );
                    ArrayD::from_shape_vec(new_shape, slice.to_vec()).unwrap()
                };
                data.into_pyarray(py).into()
            }
            DataType::Uint64 => {
                let data = unsafe {
                    let slice = std::slice::from_raw_parts(
                        mmap.as_ptr().add(offset) as *const u64,
                        size / 8,
                    );
                    ArrayD::from_shape_vec(new_shape, slice.to_vec()).unwrap()
                };
                data.into_pyarray(py).into()
            }
            DataType::Int8 => {
                let data = unsafe {
                    let slice =
                        std::slice::from_raw_parts(mmap.as_ptr().add(offset) as *const i8, size);
                    ArrayD::from_shape_vec(new_shape, slice.to_vec()).unwrap()
                };
                data.into_pyarray(py).into()
            }
            DataType::Int16 => {
                let data = unsafe {
                    let slice = std::slice::from_raw_parts(
                        mmap.as_ptr().add(offset) as *const i16,
                        size / 2,
                    );
                    ArrayD::from_shape_vec(new_shape, slice.to_vec()).unwrap()
                };
                data.into_pyarray(py).into()
            }
            DataType::Int32 => {
                let data = unsafe {
                    let slice = std::slice::from_raw_parts(
                        mmap.as_ptr().add(offset) as *const i32,
                        size / 4,
                    );
                    ArrayD::from_shape_vec(new_shape, slice.to_vec()).unwrap()
                };
                data.into_pyarray(py).into()
            }
            DataType::Int64 => {
                let data = unsafe {
                    let slice = std::slice::from_raw_parts(
                        mmap.as_ptr().add(offset) as *const i64,
                        size / 8,
                    );
                    ArrayD::from_shape_vec(new_shape, slice.to_vec()).unwrap()
                };
                data.into_pyarray(py).into()
            }
            DataType::Float16 => {
                let data = unsafe {
                    let slice = std::slice::from_raw_parts(
                        mmap.as_ptr().add(offset) as *const u16,
                        size / 2,
                    );
                    let f16_data: Vec<half::f16> =
                        slice.iter().map(|&x| half::f16::from_bits(x)).collect();
                    // 转换为f32用于Python兼容性
                    let f32_data: Vec<f32> = f16_data.iter().map(|&x| x.to_f32()).collect();
                    ArrayD::from_shape_vec(new_shape, f32_data).unwrap()
                };
                data.into_pyarray(py).into()
            }
            DataType::Float32 => {
                let data = unsafe {
                    let slice = std::slice::from_raw_parts(
                        mmap.as_ptr().add(offset) as *const f32,
                        size / 4,
                    );
                    ArrayD::from_shape_vec(new_shape, slice.to_vec()).unwrap()
                };
                data.into_pyarray(py).into()
            }
            DataType::Float64 => {
                let data = unsafe {
                    let slice = std::slice::from_raw_parts(
                        mmap.as_ptr().add(offset) as *const f64,
                        size / 8,
                    );
                    ArrayD::from_shape_vec(new_shape, slice.to_vec()).unwrap()
                };
                data.into_pyarray(py).into()
            }
            DataType::Complex64 => {
                let data = unsafe {
                    let slice = std::slice::from_raw_parts(
                        mmap.as_ptr().add(offset) as *const Complex32,
                        size / 8,
                    );
                    ArrayD::from_shape_vec(new_shape, slice.to_vec()).unwrap()
                };
                data.into_pyarray(py).into()
            }
            DataType::Complex128 => {
                let data = unsafe {
                    let slice = std::slice::from_raw_parts(
                        mmap.as_ptr().add(offset) as *const Complex64,
                        size / 16,
                    );
                    ArrayD::from_shape_vec(new_shape, slice.to_vec()).unwrap()
                };
                data.into_pyarray(py).into()
            }
        };

        self.current_index = end_index;
        Ok(Some(array))
    }
}

fn create_optimized_mmap(
    path: &Path,
    modify_time: i64,
    cache: &mut MutexGuard<HashMap<String, (Arc<Mmap>, i64)>>,
) -> PyResult<Arc<Mmap>> {
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
            0,
        );

        if addr != libc::MAP_FAILED {
            libc::madvise(addr, file_size, libc::MADV_HUGEPAGE);

            libc::madvise(addr, file_size, libc::MADV_SEQUENTIAL | libc::MADV_WILLNEED);
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

    // 标准内存映射处理

    let mmap = unsafe { memmap2::MmapOptions::new().populate().map(&file)? };

    let mmap = Arc::new(mmap);
    cache.insert(
        path.to_string_lossy().to_string(),
        (Arc::clone(&mmap), modify_time),
    );

    Ok(mmap)
}

#[pymodule]
fn _lib_numpack(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // 注册核心类
    m.add_class::<NumPack>()?;
    // 注释掉旧的 LazyArray，使用新的模块化版本
    // m.add_class::<LazyArray>()?;
    m.add_class::<LazyArrayIterator>()?;
    m.add_class::<ArrayMetadata>()?;
    m.add_class::<StreamLoader>()?;

    // 注册清理函数
    m.add_function(wrap_pyfunction!(force_cleanup_windows_handles, m)?)?;

    // 注册新模块的Python绑定
    numpack::python_bindings::register_python_bindings(m)?;

    // 注册向量引擎模块
    vector_engine::python_bindings::register_vector_engine_module(m)?;

    Ok(())
}

// 通用清理函数
#[pyfunction]
fn force_cleanup_windows_handles() -> PyResult<()> {
    // 清理临时文件缓存
    clear_temp_files_from_cache();
    Ok(())
}

// 通用文件句柄释放函数（兼容性保留）
#[allow(dead_code)]
fn release_windows_file_handle(_path: &Path) {
    // 不再需要特殊处理，保留函数名以兼容现有代码
}

/// 安全创建内存 slice 的辅助函数
#[allow(dead_code)]
fn safe_slice_from_mmap(mmap: &Mmap, offset: usize, size: usize) -> Result<&[u8], PyErr> {
    if offset + size > mmap.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
            "Data offset out of bounds",
        ));
    }
    Ok(unsafe { std::slice::from_raw_parts(mmap.as_ptr().add(offset), size) })
}

/// 安全复制内存数据的辅助函数 - 防止 Windows 内存访问冲突
#[allow(dead_code)]
fn safe_copy_from_mmap(mmap: &Mmap, offset: usize, size: usize) -> Result<Vec<u8>, PyErr> {
    if offset + size > mmap.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
            "Data offset out of bounds",
        ));
    }

    let mut result = Vec::with_capacity(size);
    unsafe {
        std::ptr::copy_nonoverlapping(mmap.as_ptr().add(offset), result.as_mut_ptr(), size);
        result.set_len(size);
    }
    Ok(result)
}
