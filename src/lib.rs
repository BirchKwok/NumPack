#[macro_use]
extern crate lazy_static;

mod error;
mod metadata;
mod parallel_io;

use std::path::{Path, PathBuf};
use numpy::{IntoPyArray, PyArrayDyn};
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

use crate::parallel_io::ParallelIO;
use crate::metadata::DataType;

#[cfg(target_family = "unix")]
use std::os::unix::io::AsRawFd;

#[cfg(target_family = "windows")] 
use std::os::windows::io::AsHandle;

#[cfg(target_family = "windows")]
use windows_sys::Win32::Storage::FileSystem::SetFileIoOverlappedRange;

#[cfg(target_family = "windows")]
use windows_sys::Win32::System::Memory::VirtualLock;

lazy_static! {
    static ref MMAP_CACHE: Mutex<HashMap<String, (Arc<Mmap>, u64)>> = Mutex::new(HashMap::new());
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
    modify_time: u64,
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

    fn get_element(&self, py: Python, row: usize, col: usize) -> PyResult<String> {
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

    fn __getitem__(&self, py: Python, key: &PyAny) -> PyResult<PyObject> {
        let total_rows = self.shape[0];
        let mut indexes = Vec::new();

        if let Ok(slice) = key.downcast::<PySlice>() {
            let indices = slice.indices((total_rows as i32).try_into().unwrap())?;
            for i in (indices.start..indices.stop).step_by(indices.step as usize) {
                if i >= 0 && (i as usize) < total_rows {
                    indexes.push(i as usize);
                }
            }
        } else if let Ok(index) = key.extract::<i64>() {
            let adjusted_index = if index < 0 {
                total_rows as i64 + index
            } else {
                index
            };
            if adjusted_index >= 0 && (adjusted_index as usize) < total_rows {
                indexes.push(adjusted_index as usize);
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>("Index out of bounds"));
            }
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Invalid index type"));
        }

        let data = unsafe {
            let data_ptr = self.mmap.as_ptr();
            let data_len = self.mmap.len();
            std::slice::from_raw_parts(data_ptr, data_len)
        };

        let mut new_shape = self.shape.clone();
        new_shape[0] = indexes.len();

        let row_size = self.itemsize * self.shape[1..].iter().product::<usize>();
        let mut result_data = Vec::with_capacity(indexes.len() * row_size);

        for &idx in &indexes {
            let start = idx * row_size;
            let end = start + row_size;
            result_data.extend_from_slice(&data[start..end]);
        }

        let array: PyObject = match self.dtype {
            DataType::Bool => {
                let bool_vec: Vec<bool> = result_data.iter().map(|&x| x != 0).collect();
                let array = ArrayD::from_shape_vec(new_shape, bool_vec).unwrap();
                array.into_pyarray(py).into()
            }
            DataType::Uint8 => {
                let array = unsafe {
                    let slice: &[u8] = bytemuck::cast_slice(&result_data);
                    ArrayD::from_shape_vec_unchecked(new_shape, slice.to_vec())
                };
                array.into_pyarray(py).into()
            }
            DataType::Uint16 => {
                let array = unsafe {
                    let slice: &[u16] = bytemuck::cast_slice(&result_data);
                    ArrayD::from_shape_vec_unchecked(new_shape, slice.to_vec())
                };
                array.into_pyarray(py).into()
            }
            DataType::Uint32 => {
                let array = unsafe {
                    let slice: &[u32] = bytemuck::cast_slice(&result_data);
                    ArrayD::from_shape_vec_unchecked(new_shape, slice.to_vec())
                };
                array.into_pyarray(py).into()
            }
            DataType::Uint64 => {
                let array = unsafe {
                    let slice: &[u64] = bytemuck::cast_slice(&result_data);
                    ArrayD::from_shape_vec_unchecked(new_shape, slice.to_vec())
                };
                array.into_pyarray(py).into()
            }
            DataType::Int8 => {
                let array = unsafe {
                    let slice: &[i8] = bytemuck::cast_slice(&result_data);
                    ArrayD::from_shape_vec_unchecked(new_shape, slice.to_vec())
                };
                array.into_pyarray(py).into()
            }
            DataType::Int16 => {
                let array = unsafe {
                    let slice: &[i16] = bytemuck::cast_slice(&result_data);
                    ArrayD::from_shape_vec_unchecked(new_shape, slice.to_vec())
                };
                array.into_pyarray(py).into()
            }
            DataType::Int32 => {
                let array = unsafe {
                    let slice: &[i32] = bytemuck::cast_slice(&result_data);
                    ArrayD::from_shape_vec_unchecked(new_shape, slice.to_vec())
                };
                array.into_pyarray(py).into()
            }
            DataType::Int64 => {
                let array = unsafe {
                    let slice: &[i64] = bytemuck::cast_slice(&result_data);
                    ArrayD::from_shape_vec_unchecked(new_shape, slice.to_vec())
                };
                array.into_pyarray(py).into()
            }
            DataType::Float32 => {
                let array = unsafe {
                    let slice: &[f32] = bytemuck::cast_slice(&result_data);
                    ArrayD::from_shape_vec_unchecked(new_shape, slice.to_vec())
                };
                array.into_pyarray(py).into()
            }
            DataType::Float64 => {
                let array = unsafe {
                    let slice: &[f64] = bytemuck::cast_slice(&result_data);
                    ArrayD::from_shape_vec_unchecked(new_shape, slice.to_vec())
                };
                array.into_pyarray(py).into()
            }
        };

        Ok(array)
    }

    #[getter]
    fn shape(&self) -> PyResult<Vec<usize>> {
        Ok(self.shape.clone())
    }
}

fn get_array_dtype(array: &PyAny) -> PyResult<DataType> {
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

    fn save(&self, arrays: &PyDict, array_name: Option<String>) -> PyResult<()> {
        let mut bool_arrays = Vec::new();
        let mut u8_arrays = Vec::new();
        let mut u16_arrays = Vec::new();
        let mut u32_arrays = Vec::new();
        let mut u64_arrays = Vec::new();
        let mut i8_arrays = Vec::new();
        let mut i16_arrays = Vec::new();
        let mut i32_arrays = Vec::new();
        let mut i64_arrays = Vec::new();
        let mut f32_arrays = Vec::new();
        let mut f64_arrays = Vec::new();

        for (i, (key, value)) in arrays.iter().enumerate() {
            let name = if let Some(prefix) = &array_name {
                format!("{}{}", prefix, i)
            } else {
                key.extract::<String>()?
            };
            
            let dtype = get_array_dtype(value)?;
            let _shape: Vec<u64> = value.getattr("shape")?
                .extract::<Vec<usize>>()?
                .into_iter()
                .map(|x| x as u64)
                .collect();

            match dtype {
                DataType::Bool => {
                    let array = value.extract::<&PyArrayDyn<bool>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    bool_arrays.push((name, array, dtype));
                }
                DataType::Uint8 => {
                    let array = value.extract::<&PyArrayDyn<u8>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    u8_arrays.push((name, array, dtype));
                }
                DataType::Uint16 => {
                    let array = value.extract::<&PyArrayDyn<u16>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    u16_arrays.push((name, array, dtype));
                }
                DataType::Uint32 => {
                    let array = value.extract::<&PyArrayDyn<u32>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    u32_arrays.push((name, array, dtype));
                }
                DataType::Uint64 => {
                    let array = value.extract::<&PyArrayDyn<u64>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    u64_arrays.push((name, array, dtype));
                }
                DataType::Int8 => {
                    let array = value.extract::<&PyArrayDyn<i8>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    i8_arrays.push((name, array, dtype));
                }
                DataType::Int16 => {
                    let array = value.extract::<&PyArrayDyn<i16>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    i16_arrays.push((name, array, dtype));
                }
                DataType::Int32 => {
                    let array = value.extract::<&PyArrayDyn<i32>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    i32_arrays.push((name, array, dtype));
                }
                DataType::Int64 => {
                    let array = value.extract::<&PyArrayDyn<i64>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    i64_arrays.push((name, array, dtype));
                }
                DataType::Float32 => {
                    let array = value.extract::<&PyArrayDyn<f32>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    f32_arrays.push((name, array, dtype));
                }
                DataType::Float64 => {
                    let array = value.extract::<&PyArrayDyn<f64>>()?;
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
                if *cached_time == meta.last_modified {
                    Arc::clone(cached_mmap)
                } else {
                    create_optimized_mmap(&data_path, meta.last_modified, &mut cache)?
                }
            } else {
                create_optimized_mmap(&data_path, meta.last_modified, &mut cache)?
            };
            
            let shape: Vec<usize> = meta.shape.iter().map(|&x| x as usize).collect();
            let itemsize = meta.dtype.size_bytes() as usize;
            
            let lazy_array = LazyArray {
                mmap,
                shape,
                dtype: meta.dtype,
                itemsize,
                array_path,
                modify_time: meta.last_modified,
            };
            
            return Ok(lazy_array.into_py(py));
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
            let shape_tuple = PyTuple::new(py, &shape);
            Ok(shape_tuple.into())
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
        
        Ok(dict.into())
    }

    fn get_member_list(&self, py: Python) -> PyResult<Py<PyList>> {
        let names = self.io.list_arrays();
        let list = PyList::new(py, names);
        Ok(list.into())
    }

    fn get_modify_time(&self, array_name: &str) -> PyResult<Option<u64>> {
        Ok(self.io.get_array_meta(array_name).map(|meta| meta.last_modified))
    }

    fn reset(&self) -> PyResult<()> {
        self.io.reset()?;
        Ok(())
    }

    pub fn append(&mut self, arrays: Vec<(&str, &PyAny)>) -> PyResult<()> {
        // Check if the array exists and get the existing array information
        let mut existing_arrays: Vec<(String, DataType, Vec<usize>)> = Vec::new();
        
        for (name, array) in &arrays {
            if let Some(meta) = self.io.get_array_meta(name) {
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
        for (name, array) in arrays {
            let meta = self.io.get_array_meta(name).unwrap();
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
            self.io.update_array_metadata(name, new_meta)?;
        }
        
        Ok(())
    }

    #[pyo3(signature = (array_names, indexes=None))]
    fn drop(&self, array_names: &PyAny, indexes: Option<&PyAny>) -> PyResult<()> {
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

    fn replace(&self, arrays: &PyDict, indexes: Option<&PyAny>) -> PyResult<()> {
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
                    let array = value.extract::<&PyArrayDyn<bool>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    self.io.replace_rows(&name, &array, &indices)?;
                }
                DataType::Uint8 => {
                    let array = value.extract::<&PyArrayDyn<u8>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    self.io.replace_rows(&name, &array, &indices)?;
                }
                DataType::Uint16 => {
                    let array = value.extract::<&PyArrayDyn<u16>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    self.io.replace_rows(&name, &array, &indices)?;
                }
                DataType::Uint32 => {
                    let array = value.extract::<&PyArrayDyn<u32>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    self.io.replace_rows(&name, &array, &indices)?;
                }
                DataType::Uint64 => {
                    let array = value.extract::<&PyArrayDyn<u64>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    self.io.replace_rows(&name, &array, &indices)?;
                }
                DataType::Int8 => {
                    let array = value.extract::<&PyArrayDyn<i8>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    self.io.replace_rows(&name, &array, &indices)?;
                }
                DataType::Int16 => {
                    let array = value.extract::<&PyArrayDyn<i16>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    self.io.replace_rows(&name, &array, &indices)?;
                }
                DataType::Int32 => {
                    let array = value.extract::<&PyArrayDyn<i32>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    self.io.replace_rows(&name, &array, &indices)?;
                }
                DataType::Int64 => {
                    let array = value.extract::<&PyArrayDyn<i64>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    self.io.replace_rows(&name, &array, &indices)?;
                }
                DataType::Float32 => {
                    let array = value.extract::<&PyArrayDyn<f32>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    self.io.replace_rows(&name, &array, &indices)?;
                }
                DataType::Float64 => {
                    let array = value.extract::<&PyArrayDyn<f64>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    self.io.replace_rows(&name, &array, &indices)?;
                }
            }
        }
        
        Ok(())
    }

    fn getitem(&self, py: Python, array_name: &str, indices: &PyAny) -> PyResult<PyObject> {
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
}

fn create_optimized_mmap(path: &Path, modify_time: u64, cache: &mut MutexGuard<HashMap<String, (Arc<Mmap>, u64)>>) -> PyResult<Arc<Mmap>> {
    let file = std::fs::File::open(path)?;
    let file_size = file.metadata()?.len() as usize;
    
    // Unix系统特定的优化
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
    
    // macOS系统特定的优化
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

    // Windows系统特定的优化
    #[cfg(target_family = "windows")]
    unsafe {
        // 创建内存映射视图
        if let Ok(mmap_view) = memmap2::MmapOptions::new()
            .populate() // 预取页面到内存
            .map(&file) 
        {
            // 锁定内存区域以防止页面交换
            let _ = VirtualLock(
                mmap_view.as_ptr() as *mut _,
                mmap_view.len()
            );
        }

        // 设置文件IO重叠范围以优化性能
        let handle = file.as_handle();
        let _ = SetFileIoOverlappedRange(
            handle.as_ptr() as _,  // 修复: 使用as_ptr()获取原始句柄
            std::ptr::null(),      // 修复: 使用null指针
            file_size.min(u32::MAX as usize) as u32  // 确保不超过u32的范围
        );
    }
    
    // 通用的mmap创建代码
    let mmap = unsafe { 
        memmap2::MmapOptions::new()
            .populate()
            .map(&file)?
    };
    
    let mmap = Arc::new(mmap);
    cache.insert(path.to_string_lossy().to_string(), (Arc::clone(&mmap), modify_time));
    
    Ok(mmap)
}

#[pymodule]
fn _lib_numpack(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<NumPack>()?;
    m.add_class::<LazyArray>()?;
    m.add_class::<ArrayMetadata>()?;
    Ok(())
}

