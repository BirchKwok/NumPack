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

use crate::parallel_io::ParallelIO;
use crate::metadata::DataType;

#[allow(dead_code)]
#[pyclass]
struct NumPack {
    io: ParallelIO,
    base_dir: PathBuf,
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

    #[pyo3(signature = (array_names=None, excluded_indices=None))]
    fn load(&self, py: Python, array_names: Option<&PyList>, excluded_indices: Option<&PyAny>) -> PyResult<PyObject> {
        if let Some(names) = array_names {
            for name in names.iter() {
                if !self.io.has_array(name.extract::<String>()?.as_str()) {
                    return Err(PyErr::new::<PyKeyError, _>("Array not found"));
                }
            }
        }

        let views = self.io.get_array_views(array_names.map(|names| {
            names.iter()
                .map(|name| name.extract::<String>())
                .collect::<PyResult<Vec<_>>>()
        }).transpose()?.as_deref())?;

        // Convert excluded_indices to Vec<i64>
        let excluded = if let Some(indices) = excluded_indices {
            if let Ok(slice) = indices.downcast::<PySlice>() {
                let start = slice.getattr("start")?.extract::<Option<i64>>()?.unwrap_or(0);
                let stop = slice.getattr("stop")?.extract::<Option<i64>>()?.unwrap_or(-1);
                let step = slice.getattr("step")?.extract::<Option<i64>>()?.unwrap_or(1);
                
                if step == 1 {
                    Some((start..stop).collect::<Vec<i64>>())
                } else {
                    None
                }
            } else if let Ok(indices) = indices.extract::<Vec<i64>>() {
                Some(indices)
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "excluded_indices must be a slice or list of integers"
                ));
            }
        } else {
            None
        };

        // Load to memory in non-mmap mode
        let dict = PyDict::new(py);
        for (name, mut view) in views {
            let final_excluded = excluded.clone();

            let array: PyObject = match view.meta.dtype {
                DataType::Bool => {
                    let array = view.into_array::<bool>(final_excluded.as_deref())?;
                    array.into_pyarray(py).into()
                }
                DataType::Uint8 => {
                    let array = view.into_array::<u8>(final_excluded.as_deref())?;
                    array.into_pyarray(py).into()
                }
                DataType::Uint16 => {
                    let array = view.into_array::<u16>(final_excluded.as_deref())?;
                    array.into_pyarray(py).into()
                }
                DataType::Uint32 => {
                    let array = view.into_array::<u32>(final_excluded.as_deref())?;
                    array.into_pyarray(py).into()
                }
                DataType::Uint64 => {
                    let array = view.into_array::<u64>(final_excluded.as_deref())?;
                    array.into_pyarray(py).into()
                }
                DataType::Int8 => {
                    let array = view.into_array::<i8>(final_excluded.as_deref())?;
                    array.into_pyarray(py).into()
                }
                DataType::Int16 => {
                    let array = view.into_array::<i16>(final_excluded.as_deref())?;
                    array.into_pyarray(py).into()
                }
                DataType::Int32 => {
                    let array = view.into_array::<i32>(final_excluded.as_deref())?;
                    array.into_pyarray(py).into()
                }
                DataType::Int64 => {
                    let array = view.into_array::<i64>(final_excluded.as_deref())?;
                    array.into_pyarray(py).into()
                }
                DataType::Float32 => {
                    let array = view.into_array::<f32>(final_excluded.as_deref())?;
                    array.into_pyarray(py).into()
                }
                DataType::Float64 => {
                    let array = view.into_array::<f64>(final_excluded.as_deref())?;
                    array.into_pyarray(py).into()
                }
            };
            dict.set_item(name, array)?;
        }
        Ok(dict.into_py(py))
    }

    fn get_shape(&self, py: Python, array_name: &str) -> PyResult<Option<Py<PyTuple>>> {
        if let Some(meta) = self.io.get_array_meta(array_name) {
            let shape: Vec<i64> = meta.shape.iter().map(|&x| x as i64).collect();
            let shape_tuple = PyTuple::new(py, &shape);
            Ok(Some(shape_tuple.into()))
        } else {
            Ok(None)
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
}

#[pymodule]
fn _lib_numpack(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<NumPack>()?;
    Ok(())
}
