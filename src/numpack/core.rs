//! NumPack核心功能实现
//!
//! 从lib.rs中提取的NumPack和ArrayMetadata结构体和实现

use half::f16;
use memmap2::Mmap;
use ndarray::ArrayD;
use num_complex::{Complex32, Complex64};
use numpy::{IntoPyArray, PyArrayDyn, PyArrayMethods};
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySlice, PyTuple};
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::core::metadata::DataType;
use crate::io::parallel_io::ParallelIO;
use crate::lazy_array::standard::LazyArray;

lazy_static::lazy_static! {
    static ref MMAP_CACHE: Mutex<HashMap<String, (Arc<Mmap>, i64)>> = Mutex::new(HashMap::new());
}

/// 清理指定文件的mmap缓存（Windows平台专用）
/// 在 Windows 上修改文件前必须调用，以避免错误 1224
/// Unix 平台不需要（系统允许同时 mmap 和修改文件）
#[cfg(windows)]
pub(crate) fn clear_mmap_cache_for_file(file_path: &str) {
    if let Ok(mut cache) = MMAP_CACHE.lock() {
        cache.remove(file_path);
    }
}

/// 清理指定数组的所有相关文件的mmap缓存（Windows平台专用）
#[cfg(windows)]
pub(crate) fn clear_mmap_cache_for_array(base_dir: &Path, array_name: &str) {
    let data_path = base_dir.join(format!("data_{}.npkd", array_name));
    clear_mmap_cache_for_file(&data_path.to_string_lossy());
}

/// NumPack主结构体 - 提供高性能数组存储和管理功能
#[pyclass]
pub struct NumPack {
    io: ParallelIO,
    base_dir: PathBuf,
}

/// 数组元数据结构体
#[pyclass]
pub struct ArrayMetadata {
    #[pyo3(get)]
    pub shape: Vec<i64>,
    #[pyo3(get)]
    pub dtype: String,
    #[pyo3(get)]
    pub data_file: String,
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
                    let array = unsafe { array.readonly().as_array().to_owned() };
                    bool_arrays.push((name, array, dtype));
                }
                DataType::Uint8 => {
                    let array = value.downcast::<PyArrayDyn<u8>>()?;
                    let array = unsafe { array.readonly().as_array().to_owned() };
                    u8_arrays.push((name, array, dtype));
                }
                DataType::Uint16 => {
                    let array = value.downcast::<PyArrayDyn<u16>>()?;
                    let array = unsafe { array.readonly().as_array().to_owned() };
                    u16_arrays.push((name, array, dtype));
                }
                DataType::Uint32 => {
                    let array = value.downcast::<PyArrayDyn<u32>>()?;
                    let array = unsafe { array.readonly().as_array().to_owned() };
                    u32_arrays.push((name, array, dtype));
                }
                DataType::Uint64 => {
                    let array = value.downcast::<PyArrayDyn<u64>>()?;
                    let array = unsafe { array.readonly().as_array().to_owned() };
                    u64_arrays.push((name, array, dtype));
                }
                DataType::Int8 => {
                    let array = value.downcast::<PyArrayDyn<i8>>()?;
                    let array = unsafe { array.readonly().as_array().to_owned() };
                    i8_arrays.push((name, array, dtype));
                }
                DataType::Int16 => {
                    let array = value.downcast::<PyArrayDyn<i16>>()?;
                    let array = unsafe { array.readonly().as_array().to_owned() };
                    i16_arrays.push((name, array, dtype));
                }
                DataType::Int32 => {
                    let array = value.downcast::<PyArrayDyn<i32>>()?;
                    let array = unsafe { array.readonly().as_array().to_owned() };
                    i32_arrays.push((name, array, dtype));
                }
                DataType::Int64 => {
                    let array = value.downcast::<PyArrayDyn<i64>>()?;
                    let array = unsafe { array.readonly().as_array().to_owned() };
                    i64_arrays.push((name, array, dtype));
                }
                DataType::Float16 => {
                    let array = value.downcast::<PyArrayDyn<f16>>()?;
                    let array = unsafe { array.readonly().as_array().to_owned() };
                    f16_arrays.push((name, array, dtype));
                }
                DataType::Float32 => {
                    let array = value.downcast::<PyArrayDyn<f32>>()?;
                    let array = unsafe { array.readonly().as_array().to_owned() };
                    f32_arrays.push((name, array, dtype));
                }
                DataType::Float64 => {
                    let array = value.downcast::<PyArrayDyn<f64>>()?;
                    let array = unsafe { array.readonly().as_array().to_owned() };
                    f64_arrays.push((name, array, dtype));
                }
                DataType::Complex64 => {
                    let array = value.downcast::<PyArrayDyn<Complex32>>()?;
                    let array = unsafe { array.readonly().as_array().to_owned() };
                    complex64_arrays.push((name, array, dtype));
                }
                DataType::Complex128 => {
                    let array = value.downcast::<PyArrayDyn<Complex64>>()?;
                    let array = unsafe { array.readonly().as_array().to_owned() };
                    complex128_arrays.push((name, array, dtype));
                }
            }
        }

        // Save arrays by dtype
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

            // Windows平台文件存在性检查，防止路径被误解析为目录
            if !data_path.exists() {
                return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
                    format!("Data file not found: {}", data_path.display()),
                ));
            }

            // 确保路径是文件而不是目录
            if data_path.is_dir() {
                return Err(PyErr::new::<pyo3::exceptions::PyIsADirectoryError, _>(
                    format!("Expected file but found directory: {}", data_path.display()),
                ));
            }

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
            let itemsize = meta.get_dtype().size_bytes() as usize;

            let lazy_array = LazyArray::new(
                mmap,
                shape,
                meta.get_dtype(),
                itemsize,
                array_path,
                meta.last_modified as i64,
            );

            return Ok(Py::new(py, lazy_array)?.into());
        }

        let meta = self.io.get_array_metadata(array_name)?;
        let data_path = self.base_dir.join(format!("data_{}.npkd", array_name));

        // Windows平台文件存在性检查，防止路径被误解析为目录
        if !data_path.exists() {
            return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
                format!("Data file not found: {}", data_path.display()),
            ));
        }

        // 确保路径是文件而不是目录
        if data_path.is_dir() {
            return Err(PyErr::new::<pyo3::exceptions::PyIsADirectoryError, _>(
                format!("Expected file but found directory: {}", data_path.display()),
            ));
        }

        let shape: Vec<usize> = meta.shape.iter().map(|&x| x as usize).collect();

        // Use mmap to accelerate data loading
        let file = std::fs::File::open(&data_path)?;
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };

        // Create array and copy data
        let array: PyObject = match meta.get_dtype() {
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
                    let slice =
                        std::slice::from_raw_parts(mmap.as_ptr() as *const u16, mmap.len() / 2);
                    ArrayD::from_shape_vec(shape, slice.to_vec()).unwrap()
                };
                data.into_pyarray(py).into()
            }
            DataType::Uint32 => {
                let data = unsafe {
                    let slice =
                        std::slice::from_raw_parts(mmap.as_ptr() as *const u32, mmap.len() / 4);
                    ArrayD::from_shape_vec(shape, slice.to_vec()).unwrap()
                };
                data.into_pyarray(py).into()
            }
            DataType::Uint64 => {
                let data = unsafe {
                    let slice =
                        std::slice::from_raw_parts(mmap.as_ptr() as *const u64, mmap.len() / 8);
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
                    let slice =
                        std::slice::from_raw_parts(mmap.as_ptr() as *const i16, mmap.len() / 2);
                    ArrayD::from_shape_vec(shape, slice.to_vec()).unwrap()
                };
                data.into_pyarray(py).into()
            }
            DataType::Int32 => {
                let data = unsafe {
                    let slice =
                        std::slice::from_raw_parts(mmap.as_ptr() as *const i32, mmap.len() / 4);
                    ArrayD::from_shape_vec(shape, slice.to_vec()).unwrap()
                };
                data.into_pyarray(py).into()
            }
            DataType::Int64 => {
                let data = unsafe {
                    let slice =
                        std::slice::from_raw_parts(mmap.as_ptr() as *const i64, mmap.len() / 8);
                    ArrayD::from_shape_vec(shape, slice.to_vec()).unwrap()
                };
                data.into_pyarray(py).into()
            }
            DataType::Float16 => {
                let data = unsafe {
                    let slice =
                        std::slice::from_raw_parts(mmap.as_ptr() as *const f16, mmap.len() / 2);
                    ArrayD::from_shape_vec(shape, slice.to_vec()).unwrap()
                };
                data.into_pyarray(py).into()
            }
            DataType::Float32 => {
                let data = unsafe {
                    let slice =
                        std::slice::from_raw_parts(mmap.as_ptr() as *const f32, mmap.len() / 4);
                    ArrayD::from_shape_vec(shape, slice.to_vec()).unwrap()
                };
                data.into_pyarray(py).into()
            }
            DataType::Float64 => {
                let data = unsafe {
                    let slice =
                        std::slice::from_raw_parts(mmap.as_ptr() as *const f64, mmap.len() / 8);
                    ArrayD::from_shape_vec(shape, slice.to_vec()).unwrap()
                };
                data.into_pyarray(py).into()
            }
            DataType::Complex64 => {
                let data = unsafe {
                    let slice = std::slice::from_raw_parts(
                        mmap.as_ptr() as *const Complex32,
                        mmap.len() / 8,
                    );
                    ArrayD::from_shape_vec(shape, slice.to_vec()).unwrap()
                };
                data.into_pyarray(py).into()
            }
            DataType::Complex128 => {
                let data = unsafe {
                    let slice = std::slice::from_raw_parts(
                        mmap.as_ptr() as *const Complex64,
                        mmap.len() / 16,
                    );
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

    #[pyo3(signature = (array_name, arrays))]
    fn append(&self, array_name: &str, arrays: &Bound<'_, PyDict>) -> PyResult<()> {
        if !self.io.has_array(array_name) {
            return Err(PyErr::new::<PyKeyError, _>(format!(
                "Array {} not found",
                array_name
            )));
        }

        let meta = self.io.get_array_metadata(array_name)?;

        for (name, array) in arrays.iter() {
            let name = name.extract::<String>()?;
            if name != array_name {
                continue;
            }

            let dtype = get_array_dtype(&array)?;
            if dtype != meta.get_dtype() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Array dtype mismatch: expected {:?}, got {:?}",
                    meta.get_dtype(),
                    dtype
                )));
            }

            let shape: Vec<u64> = array
                .getattr("shape")?
                .extract::<Vec<usize>>()?
                .into_iter()
                .map(|x| x as u64)
                .collect();

            if shape.len() != meta.shape.len() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Array dimension mismatch",
                ));
            }

            for i in 1..shape.len() {
                if shape[i] != meta.shape[i] {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Array shape mismatch (except first dimension)",
                    ));
                }
            }

            // Append data to file
            let data_path = self.base_dir.join(format!("data_{}.npkd", array_name));
            let mut file = OpenOptions::new()
                .write(true)
                .append(true)
                .open(&data_path)?;

            match dtype {
                DataType::Bool => {
                    let py_array = array.downcast::<PyArrayDyn<bool>>()?;
                    let readonly = py_array.readonly();
                    let array_ref = unsafe { readonly.as_array() };
                    let data = array_ref.as_slice().unwrap();
                    let bytes: Vec<u8> = data.iter().map(|&b| if b { 1 } else { 0 }).collect();
                    file.write_all(&bytes)?;
                }
                DataType::Uint8 => {
                    let py_array = array.downcast::<PyArrayDyn<u8>>()?;
                    let readonly = py_array.readonly();
                    let array_ref = unsafe { readonly.as_array() };
                    let data = array_ref.as_slice().unwrap();
                    file.write_all(data)?;
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
            self.io.update_array_metadata(&name, new_meta)?;
        }

        Ok(())
    }

    #[pyo3(signature = (array_names, indexes=None))]
    fn getbatch(
        &self,
        py: Python,
        array_names: Vec<String>,
        indexes: Option<Vec<i64>>,
    ) -> PyResult<PyObject> {
        let dict = PyDict::new(py);

        for array_name in array_names {
            if !self.io.has_array(&array_name) {
                return Err(PyErr::new::<PyKeyError, _>(format!(
                    "Array {} not found",
                    array_name
                )));
            }

            let result = if let Some(ref indices) = indexes {
                let py_list = PyList::new(py, indices)?;
                self.getitem(py, &array_name, py_list.as_any())?
            } else {
                self.load(py, &array_name, Some(false))?
            };

            dict.set_item(array_name, result)?;
        }

        Ok(dict.unbind().into())
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

        // Get the indices of the rows to read
        let indices = if let Ok(slice) = indices.downcast::<PySlice>() {
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

            if step != 1 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Only step=1 is supported for slices",
                ));
            }

            (start..stop).collect::<Vec<i64>>()
        } else if let Ok(indices) = indices.extract::<Vec<i64>>() {
            indices
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "indices must be a list of integers or a slice",
            ));
        };

        // Read data
        let data = self.io.read_rows(array_name, &indices)?;

        // Calculate the new shape
        let mut new_shape = meta.shape.iter().map(|&x| x as usize).collect::<Vec<_>>();
        new_shape[0] = indices.len();

        // 优化策略：使用 zero-copy 转换，避免 to_vec() 的内存复制
        // 直接从 Vec<u8> 转换为目标类型的 Vec<T>
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
                // 优化：直接使用 data，无需 to_vec()
                let array = unsafe { ArrayD::from_shape_vec_unchecked(new_shape, data) };
                array.into_pyarray(py).into()
            }
            DataType::Uint16 => {
                let array = unsafe {
                    let len = data.len() / std::mem::size_of::<u16>();
                    let ptr = data.as_ptr() as *mut u16;
                    let capacity = data.capacity() / std::mem::size_of::<u16>();
                    std::mem::forget(data);
                    let vec_u16 = Vec::from_raw_parts(ptr, len, capacity);
                    ArrayD::from_shape_vec_unchecked(new_shape, vec_u16)
                };
                array.into_pyarray(py).into()
            }
            DataType::Uint32 => {
                let array = unsafe {
                    let len = data.len() / std::mem::size_of::<u32>();
                    let ptr = data.as_ptr() as *mut u32;
                    let capacity = data.capacity() / std::mem::size_of::<u32>();
                    std::mem::forget(data);
                    let vec_u32 = Vec::from_raw_parts(ptr, len, capacity);
                    ArrayD::from_shape_vec_unchecked(new_shape, vec_u32)
                };
                array.into_pyarray(py).into()
            }
            DataType::Uint64 => {
                let array = unsafe {
                    let len = data.len() / std::mem::size_of::<u64>();
                    let ptr = data.as_ptr() as *mut u64;
                    let capacity = data.capacity() / std::mem::size_of::<u64>();
                    std::mem::forget(data);
                    let vec_u64 = Vec::from_raw_parts(ptr, len, capacity);
                    ArrayD::from_shape_vec_unchecked(new_shape, vec_u64)
                };
                array.into_pyarray(py).into()
            }
            DataType::Int8 => {
                let array = unsafe {
                    let len = data.len();
                    let ptr = data.as_ptr() as *mut i8;
                    let capacity = data.capacity();
                    std::mem::forget(data);
                    let vec_i8 = Vec::from_raw_parts(ptr, len, capacity);
                    ArrayD::from_shape_vec_unchecked(new_shape, vec_i8)
                };
                array.into_pyarray(py).into()
            }
            DataType::Int16 => {
                let array = unsafe {
                    let len = data.len() / std::mem::size_of::<i16>();
                    let ptr = data.as_ptr() as *mut i16;
                    let capacity = data.capacity() / std::mem::size_of::<i16>();
                    std::mem::forget(data);
                    let vec_i16 = Vec::from_raw_parts(ptr, len, capacity);
                    ArrayD::from_shape_vec_unchecked(new_shape, vec_i16)
                };
                array.into_pyarray(py).into()
            }
            DataType::Int32 => {
                let array = unsafe {
                    let len = data.len() / std::mem::size_of::<i32>();
                    let ptr = data.as_ptr() as *mut i32;
                    let capacity = data.capacity() / std::mem::size_of::<i32>();
                    std::mem::forget(data);
                    let vec_i32 = Vec::from_raw_parts(ptr, len, capacity);
                    ArrayD::from_shape_vec_unchecked(new_shape, vec_i32)
                };
                array.into_pyarray(py).into()
            }
            DataType::Int64 => {
                let array = unsafe {
                    let len = data.len() / std::mem::size_of::<i64>();
                    let ptr = data.as_ptr() as *mut i64;
                    let capacity = data.capacity() / std::mem::size_of::<i64>();
                    std::mem::forget(data);
                    let vec_i64 = Vec::from_raw_parts(ptr, len, capacity);
                    ArrayD::from_shape_vec_unchecked(new_shape, vec_i64)
                };
                array.into_pyarray(py).into()
            }
            DataType::Float16 => {
                let array = unsafe {
                    let len = data.len() / std::mem::size_of::<f16>();
                    let ptr = data.as_ptr() as *mut f16;
                    let capacity = data.capacity() / std::mem::size_of::<f16>();
                    std::mem::forget(data);
                    let vec_f16 = Vec::from_raw_parts(ptr, len, capacity);
                    ArrayD::from_shape_vec_unchecked(new_shape, vec_f16)
                };
                array.into_pyarray(py).into()
            }
            DataType::Float32 => {
                let array = unsafe {
                    let len = data.len() / std::mem::size_of::<f32>();
                    let ptr = data.as_ptr() as *mut f32;
                    let capacity = data.capacity() / std::mem::size_of::<f32>();
                    std::mem::forget(data);
                    let vec_f32 = Vec::from_raw_parts(ptr, len, capacity);
                    ArrayD::from_shape_vec_unchecked(new_shape, vec_f32)
                };
                array.into_pyarray(py).into()
            }
            DataType::Float64 => {
                let array = unsafe {
                    let len = data.len() / std::mem::size_of::<f64>();
                    let ptr = data.as_ptr() as *mut f64;
                    let capacity = data.capacity() / std::mem::size_of::<f64>();
                    std::mem::forget(data);
                    let vec_f64 = Vec::from_raw_parts(ptr, len, capacity);
                    ArrayD::from_shape_vec_unchecked(new_shape, vec_f64)
                };
                array.into_pyarray(py).into()
            }
            DataType::Complex64 => {
                let array = unsafe {
                    let len = data.len() / std::mem::size_of::<Complex32>();
                    let ptr = data.as_ptr() as *mut Complex32;
                    let capacity = data.capacity() / std::mem::size_of::<Complex32>();
                    std::mem::forget(data);
                    let vec_c64 = Vec::from_raw_parts(ptr, len, capacity);
                    ArrayD::from_shape_vec_unchecked(new_shape, vec_c64)
                };
                array.into_pyarray(py).into()
            }
            DataType::Complex128 => {
                let array = unsafe {
                    let len = data.len() / std::mem::size_of::<Complex64>();
                    let ptr = data.as_ptr() as *mut Complex64;
                    let capacity = data.capacity() / std::mem::size_of::<Complex64>();
                    std::mem::forget(data);
                    let vec_c128 = Vec::from_raw_parts(ptr, len, capacity);
                    ArrayD::from_shape_vec_unchecked(new_shape, vec_c128)
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
                dtype: format!("{:?}", meta.get_dtype()),
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
}

impl NumPack {
    /// 获取基础目录
    pub fn base_dir(&self) -> &PathBuf {
        &self.base_dir
    }

    /// 获取ParallelIO引用
    pub fn io(&self) -> &ParallelIO {
        &self.io
    }
}

/// 获取数组的数据类型
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
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unsupported dtype: {}",
            dtype_str
        ))),
    }
}

/// 创建优化的内存映射
#[cfg(target_family = "unix")]
fn create_optimized_mmap(
    path: &PathBuf,
    modify_time: i64,
    cache: &mut std::sync::MutexGuard<HashMap<String, (Arc<Mmap>, i64)>>,
) -> Result<Arc<Mmap>, PyErr> {
    let file = std::fs::File::open(path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

    let file_size = file
        .metadata()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?
        .len();

    if file_size == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Empty file",
        ));
    }

    // 针对Unix系统的预读优化
    #[cfg(target_os = "macos")]
    unsafe {
        use std::os::unix::io::AsRawFd;
        let radv = libc::radvisory {
            ra_offset: 0,
            ra_count: file_size as i32,
        };
        libc::fcntl(file.as_raw_fd(), libc::F_RDADVISE, &radv);
        libc::fcntl(file.as_raw_fd(), libc::F_RDAHEAD, 1);
    }

    // 针对Linux系统的预读优化
    #[cfg(target_os = "linux")]
    unsafe {
        use std::os::unix::io::AsRawFd;
        // Linux使用posix_fadvise进行预读优化
        libc::posix_fadvise(
            file.as_raw_fd(),
            0,
            file_size as i64,
            libc::POSIX_FADV_WILLNEED,
        );
        libc::posix_fadvise(
            file.as_raw_fd(),
            0,
            file_size as i64,
            libc::POSIX_FADV_SEQUENTIAL,
        );
    }

    // 创建标准映射
    let mmap = unsafe {
        memmap2::MmapOptions::new()
            .populate()
            .map(&file)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?
    };

    let mmap = Arc::new(mmap);
    cache.insert(
        path.to_string_lossy().to_string(),
        (Arc::clone(&mmap), modify_time),
    );

    Ok(mmap)
}

/// Windows平台的内存映射实现
#[cfg(target_family = "windows")]
fn create_optimized_mmap(
    path: &PathBuf,
    modify_time: i64,
    cache: &mut std::sync::MutexGuard<HashMap<String, (Arc<Mmap>, i64)>>,
) -> Result<Arc<Mmap>, PyErr> {
    let file = std::fs::File::open(path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

    let file_size = file
        .metadata()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?
        .len();

    if file_size == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Empty file",
        ));
    }

    // Windows平台使用标准映射
    let mmap = unsafe {
        memmap2::MmapOptions::new()
            .map(&file)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?
    };

    let mmap = Arc::new(mmap);
    cache.insert(
        path.to_string_lossy().to_string(),
        (Arc::clone(&mmap), modify_time),
    );

    Ok(mmap)
}
