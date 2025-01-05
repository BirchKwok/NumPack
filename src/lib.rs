mod error;
mod metadata;
mod parallel_io;

use std::path::{Path, PathBuf};
use numpy::{PyArray2, IntoPyArray};
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::types::PySlice;
use std::fs::OpenOptions;
use std::io::Write;

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

fn get_array_shape(array: &PyAny) -> PyResult<(usize, usize)> {
    let shape = array.getattr("shape")?;
    let rows = shape.get_item(0)?.extract::<usize>()?;
    let cols = shape.get_item(1)?.extract::<usize>()?;
    Ok((rows, cols))
}

impl PartialEq for DataType {
    fn eq(&self, other: &Self) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
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
            match dtype {
                DataType::Bool => {
                    let array = value.extract::<&PyArray2<bool>>()?;
                    bool_arrays.push((name, unsafe { array.as_array().to_owned() }, dtype));
                }
                DataType::Uint8 => {
                    let array = value.extract::<&PyArray2<u8>>()?;
                    u8_arrays.push((name, unsafe { array.as_array().to_owned() }, dtype));
                }
                DataType::Uint16 => {
                    let array = value.extract::<&PyArray2<u16>>()?;
                    u16_arrays.push((name, unsafe { array.as_array().to_owned() }, dtype));
                }
                DataType::Uint32 => {
                    let array = value.extract::<&PyArray2<u32>>()?;
                    u32_arrays.push((name, unsafe { array.as_array().to_owned() }, dtype));
                }
                DataType::Uint64 => {
                    let array = value.extract::<&PyArray2<u64>>()?;
                    u64_arrays.push((name, unsafe { array.as_array().to_owned() }, dtype));
                }
                DataType::Int8 => {
                    let array = value.extract::<&PyArray2<i8>>()?;
                    i8_arrays.push((name, unsafe { array.as_array().to_owned() }, dtype));
                }
                DataType::Int16 => {
                    let array = value.extract::<&PyArray2<i16>>()?;
                    i16_arrays.push((name, unsafe { array.as_array().to_owned() }, dtype));
                }
                DataType::Int32 => {
                    let array = value.extract::<&PyArray2<i32>>()?;
                    i32_arrays.push((name, unsafe { array.as_array().to_owned() }, dtype));
                }
                DataType::Int64 => {
                    let array = value.extract::<&PyArray2<i64>>()?;
                    i64_arrays.push((name, unsafe { array.as_array().to_owned() }, dtype));
                }
                DataType::Float32 => {
                    let array = value.extract::<&PyArray2<f32>>()?;
                    f32_arrays.push((name, unsafe { array.as_array().to_owned() }, dtype));
                }
                DataType::Float64 => {
                    let array = value.extract::<&PyArray2<f64>>()?;
                    f64_arrays.push((name, unsafe { array.as_array().to_owned() }, dtype));
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

    fn replace(&self, _py: Python, arrays: &PyDict, indexes: &PyAny) -> PyResult<()> {
        // Get index list
        let indices = if let Ok(slice) = indexes.extract::<&PySlice>() {
            let start = slice.getattr("start")?.extract::<Option<i64>>()?.unwrap_or(0);
            let stop = slice.getattr("stop")?.extract::<Option<i64>>()?.unwrap_or(-1);
            let step = slice.getattr("step")?.extract::<Option<i64>>()?.unwrap_or(1);
            
            if step != 1 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Step size must be 1 for replacement"
                ));
            }
            
            (start..stop).collect::<Vec<i64>>()
        } else if let Ok(idx_list) = indexes.extract::<Vec<usize>>() {
            idx_list.into_iter().map(|x| x as i64).collect()
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Indexes must be a slice or list of integers"
            ));
        };

        // Process each array
        for (key, value) in arrays.iter() {
            let name = key.extract::<String>()?;
            let dtype = self.io.get_array_meta(&name)
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                    format!("Array {} not found", name)
                ))?
                .dtype;

            match dtype {
                DataType::Bool => {
                    let array = value.extract::<&PyArray2<bool>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    self.io.replace_rows(&name, &array, &indices)?;
                }
                DataType::Uint8 => {
                    let array = value.extract::<&PyArray2<u8>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    self.io.replace_rows(&name, &array, &indices)?;
                }
                DataType::Uint16 => {
                    let array = value.extract::<&PyArray2<u16>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    self.io.replace_rows(&name, &array, &indices)?;
                }
                DataType::Uint32 => {
                    let array = value.extract::<&PyArray2<u32>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    self.io.replace_rows(&name, &array, &indices)?;
                }
                DataType::Uint64 => {
                    let array = value.extract::<&PyArray2<u64>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    self.io.replace_rows(&name, &array, &indices)?;
                }
                DataType::Int8 => {
                    let array = value.extract::<&PyArray2<i8>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    self.io.replace_rows(&name, &array, &indices)?;
                }
                DataType::Int16 => {
                    let array = value.extract::<&PyArray2<i16>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    self.io.replace_rows(&name, &array, &indices)?;
                }
                DataType::Int32 => {
                    let array = value.extract::<&PyArray2<i32>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    self.io.replace_rows(&name, &array, &indices)?;
                }
                DataType::Int64 => {
                    let array = value.extract::<&PyArray2<i64>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    self.io.replace_rows(&name, &array, &indices)?;
                }
                DataType::Float32 => {
                    let array = value.extract::<&PyArray2<f32>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    self.io.replace_rows(&name, &array, &indices)?;
                }
                DataType::Float64 => {
                    let array = value.extract::<&PyArray2<f64>>()?;
                    let array = unsafe { array.as_array().to_owned() };
                    self.io.replace_rows(&name, &array, &indices)?;
                }
            }
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
            // let py = indexes.py();
            
            for name in &names {
                if let Some(meta) = self.io.get_array_meta(name) {
                    let shape = (meta.rows as usize, meta.cols as usize);
                    
                    // Get the indices of the rows to delete
                    let deleted_indices = if let Ok(slice) = indexes.downcast::<PySlice>() {
                        let start = slice.getattr("start")?.extract::<Option<i64>>()?.unwrap_or(0);
                        let stop = slice.getattr("stop")?.extract::<Option<i64>>()?.unwrap_or(shape.0 as i64);
                        let step = slice.getattr("step")?.extract::<Option<i64>>()?.unwrap_or(1);
                        
                        let start = if start < 0 { shape.0 as i64 + start } else { start };
                        let stop = if stop < 0 { shape.0 as i64 + stop } else { stop };
                        
                        if step == 1 {
                            (start..stop).collect::<Vec<i64>>()
                        } else {
                            Vec::new()
                        }
                    } else if let Ok(indices) = indexes.extract::<Vec<i64>>() {
                        // Process negative indices
                        indices.into_iter()
                            .map(|idx| if idx < 0 { shape.0 as i64 + idx } else { idx })
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

    fn get_shape(&self, py: Python, array_name: &str) -> PyResult<Option<Py<PyTuple>>> {
        if let Some(meta) = self.io.get_array_meta(array_name) {
            let shape = PyTuple::new(py, &[meta.rows as i64, meta.cols as i64]);
            Ok(Some(shape.into()))
        } else {
            Ok(None)
        }
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
        let mut existing_arrays: Vec<(String, DataType, (usize, usize))> = Vec::new();
        
        for (name, array) in &arrays {
            if let Some(meta) = self.io.get_array_meta(name) {
                let shape = get_array_shape(array)?;
                if meta.cols != shape.1 as u64 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        format!("Column count mismatch for array {}: expected {}, got {}", 
                            name, meta.cols, shape.1)
                    ));
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
            let shape = get_array_shape(array)?;
            
            // Append data to file
            let array_path = self.base_dir.join(&meta.data_file);
            let mut file = OpenOptions::new()
                .append(true)
                .open(array_path)?;
                
            match meta.dtype {
                DataType::Bool => {
                    let py_array = array.downcast::<PyArray2<bool>>()?;
                    let array_ref = unsafe { py_array.as_array() };
                    let data = array_ref.as_slice().unwrap();
                    file.write_all(bytemuck::cast_slice(data))?;
                }
                DataType::Uint8 => {
                    let py_array = array.downcast::<PyArray2<u8>>()?;
                    let array_ref = unsafe { py_array.as_array() };
                    let data = array_ref.as_slice().unwrap();
                    file.write_all(bytemuck::cast_slice(data))?;
                }
                DataType::Uint16 => {
                    let py_array = array.downcast::<PyArray2<u16>>()?;
                    let array_ref = unsafe { py_array.as_array() };
                    let data = array_ref.as_slice().unwrap();
                    file.write_all(bytemuck::cast_slice(data))?;
                }
                DataType::Uint32 => {
                    let py_array = array.downcast::<PyArray2<u32>>()?;
                    let array_ref = unsafe { py_array.as_array() };
                    let data = array_ref.as_slice().unwrap();
                    file.write_all(bytemuck::cast_slice(data))?;
                }
                DataType::Uint64 => {
                    let py_array = array.downcast::<PyArray2<u64>>()?;
                    let array_ref = unsafe { py_array.as_array() };
                    let data = array_ref.as_slice().unwrap();
                    file.write_all(bytemuck::cast_slice(data))?;
                }
                DataType::Int8 => {
                    let py_array = array.downcast::<PyArray2<i8>>()?;
                    let array_ref = unsafe { py_array.as_array() };
                    let data = array_ref.as_slice().unwrap();
                    file.write_all(bytemuck::cast_slice(data))?;
                }
                DataType::Int16 => {
                    let py_array = array.downcast::<PyArray2<i16>>()?;
                    let array_ref = unsafe { py_array.as_array() };
                    let data = array_ref.as_slice().unwrap();
                    file.write_all(bytemuck::cast_slice(data))?;
                }
                DataType::Int32 => {
                    let py_array = array.downcast::<PyArray2<i32>>()?;
                    let array_ref = unsafe { py_array.as_array() };
                    let data = array_ref.as_slice().unwrap();
                    file.write_all(bytemuck::cast_slice(data))?;
                }
                DataType::Int64 => {
                    let py_array = array.downcast::<PyArray2<i64>>()?;
                    let array_ref = unsafe { py_array.as_array() };
                    let data = array_ref.as_slice().unwrap();
                    file.write_all(bytemuck::cast_slice(data))?;
                }
                DataType::Float32 => {
                    let py_array = array.downcast::<PyArray2<f32>>()?;
                    let array_ref = unsafe { py_array.as_array() };
                    let data = array_ref.as_slice().unwrap();
                    file.write_all(bytemuck::cast_slice(data))?;
                }
                DataType::Float64 => {
                    let py_array = array.downcast::<PyArray2<f64>>()?;
                    let array_ref = unsafe { py_array.as_array() };
                    let data = array_ref.as_slice().unwrap();
                    file.write_all(bytemuck::cast_slice(data))?;
                }
            }

            // Update metadata
            let mut new_meta = meta.clone();
            new_meta.rows += shape.0 as u64;
            new_meta.size_bytes = new_meta.rows * new_meta.cols * new_meta.dtype.size_bytes() as u64;
            new_meta.last_modified = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
            self.io.update_array_metadata(name, new_meta)?;
        }
        
        Ok(())
    }

    fn getitem(&self, py: Python, indexes: &PyAny, array_names: Option<Vec<String>>) -> PyResult<Py<PyDict>> {
        let result = PyDict::new(py);
        let member_list = if let Some(names) = array_names {
            names
        } else {
            self.io.list_arrays()
        };

        for name in member_list {
            let meta = self.io.get_array_metadata(&name)?;
            let indices = indexes.extract::<Vec<i64>>()?;
            
            // Read specified rows of data
            let data = self.io.read_rows(&name, &indices)?;
            let shape = [indices.len(), meta.cols as usize];
            
            // Create NumPy array based on data type
            let array: Py<PyAny> = match meta.dtype {
                DataType::Bool => unsafe {
                    let py_array = PyArray2::<bool>::new(py, shape, false);
                    let mut raw_array = py_array.as_raw_array_mut();
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr() as *const bool,
                        raw_array.as_mut_ptr(),
                        shape[0] * shape[1]
                    );
                    py_array.into()
                },
                DataType::Uint8 => unsafe {
                    let py_array = PyArray2::<u8>::new(py, shape, false);
                    let mut raw_array = py_array.as_raw_array_mut();
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr() as *const u8,
                        raw_array.as_mut_ptr(),
                        shape[0] * shape[1]
                    );
                    py_array.into()
                },
                DataType::Uint16 => unsafe {
                    let py_array = PyArray2::<u16>::new(py, shape, false);
                    let mut raw_array = py_array.as_raw_array_mut();
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr() as *const u16,
                        raw_array.as_mut_ptr(),
                        shape[0] * shape[1]
                    );
                    py_array.into()
                },
                DataType::Uint32 => unsafe {
                    let py_array = PyArray2::<u32>::new(py, shape, false);
                    let mut raw_array = py_array.as_raw_array_mut();
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr() as *const u32,
                        raw_array.as_mut_ptr(),
                        shape[0] * shape[1]
                    );
                    py_array.into()
                },
                DataType::Uint64 => unsafe {
                    let py_array = PyArray2::<u64>::new(py, shape, false);
                    let mut raw_array = py_array.as_raw_array_mut();
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr() as *const u64,
                        raw_array.as_mut_ptr(),
                        shape[0] * shape[1]
                    );
                    py_array.into()
                },
                DataType::Int8 => unsafe {
                    let py_array = PyArray2::<i8>::new(py, shape, false);
                    let mut raw_array = py_array.as_raw_array_mut();
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr() as *const i8,
                        raw_array.as_mut_ptr(),
                        shape[0] * shape[1]
                    );
                    py_array.into()
                },
                DataType::Int16 => unsafe {
                    let py_array = PyArray2::<i16>::new(py, shape, false);
                    let mut raw_array = py_array.as_raw_array_mut();
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr() as *const i16,
                        raw_array.as_mut_ptr(),
                        shape[0] * shape[1]
                    );
                    py_array.into()
                },
                DataType::Int32 => unsafe {
                    let py_array = PyArray2::<i32>::new(py, shape, false);
                    let mut raw_array = py_array.as_raw_array_mut();
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr() as *const i32,
                        raw_array.as_mut_ptr(),
                        shape[0] * shape[1]
                    );
                    py_array.into()
                },
                DataType::Int64 => unsafe {
                    let py_array = PyArray2::<i64>::new(py, shape, false);
                    let mut raw_array = py_array.as_raw_array_mut();
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr() as *const i64,
                        raw_array.as_mut_ptr(),
                        shape[0] * shape[1]
                    );
                    py_array.into()
                },
                DataType::Float32 => unsafe {
                    let py_array = PyArray2::<f32>::new(py, shape, false);
                    let mut raw_array = py_array.as_raw_array_mut();
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr() as *const f32,
                        raw_array.as_mut_ptr(),
                        shape[0] * shape[1]
                    );
                    py_array.into()
                },
                DataType::Float64 => unsafe {
                    let py_array = PyArray2::<f64>::new(py, shape, false);
                    let mut raw_array = py_array.as_raw_array_mut();
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr() as *const f64,
                        raw_array.as_mut_ptr(),
                        shape[0] * shape[1]
                    );
                    py_array.into()
                },
            };
            result.set_item(&name, array)?;
        }

        Ok(result.into())
    }

    fn get_array_path(&self, array_name: &str) -> PathBuf {
        self.base_dir.join(&self.io.get_array_metadata(array_name).unwrap().data_file)
    }

    fn get_metadata(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        
        let arrays = PyDict::new(py);
        for name in self.io.list_arrays() {
            if let Some(meta) = self.io.get_array_meta(&name) {
                let array_dict = PyDict::new(py);
                array_dict.set_item("rows", meta.rows)?;
                array_dict.set_item("cols", meta.cols)?;
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
}

#[pymodule]
fn _lib_numpack(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<NumPack>()?;
    Ok(())
}
