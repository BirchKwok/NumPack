mod error;
mod metadata;
mod parallel_io;

use std::path::{Path, PathBuf};
use numpy::{PyArray2, IntoPyArray};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use ndarray::Array2;
use ndarray::prelude::*;
use pyo3::types::PySlice;

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
        "float16" => Ok(DataType::Float16),
        "float32" => Ok(DataType::Float32),
        "float64" => Ok(DataType::Float64),
        _ => Err(pyo3::exceptions::PyValueError::new_err(
            format!("Unsupported dtype: {}", dtype_str)
        )),
    }
}

fn convert_to_float16(py: Python, array: &PyArray2<f32>) -> PyResult<PyObject> {
    let np = py.import("numpy")?;
    let args = (array, "float16");
    let result = np.getattr("asarray")?.call1(args)?;
    Ok(result.into())
}

fn convert_from_float16(py: Python, array: &PyAny) -> PyResult<Array2<f32>> {
    let np = py.import("numpy")?;
    let args = (array, "float32");
    let result = np.getattr("asarray")?.call1(args)?;
    let array = result.extract::<&PyArray2<f32>>()?;
    Ok(unsafe { array.as_array().to_owned() })
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

    fn save_arrays(&self, arrays: &PyDict, array_name: Option<String>) -> PyResult<()> {
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
                DataType::Float16 => {
                    let array = Python::with_gil(|py| convert_from_float16(py, value))?;
                    f32_arrays.push((name, array, dtype));
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

    #[pyo3(signature = (array_names=None, mmap_mode=false))]
    fn load_arrays(&self, py: Python, array_names: Option<&PyList>, mmap_mode: bool) -> PyResult<PyObject> {
        let views = self.io.get_array_views(array_names.map(|names| {
            names.iter()
                .map(|name| name.extract::<String>())
                .collect::<PyResult<Vec<_>>>()
        }).transpose()?.as_deref(), mmap_mode)?;

        if mmap_mode {
            // 在mmap模式下，返回原始的内存映射视图
            let dict = PyDict::new(py);
            for (name, mut view) in views {
                let array = view.get_mmap_array(py)?;
                dict.set_item(name, array)?;
            }
            Ok(dict.into_py(py))
        } else {
            // 在非mmap模式下，加载到内存
            let dict = PyDict::new(py);
            for (name, mut view) in views {
                let array: PyObject = match view.meta.dtype {
                    DataType::Bool => {
                        let array = view.into_array::<bool>()?;
                        array.into_pyarray(py).into()
                    }
                    DataType::Uint8 => {
                        let array = view.into_array::<u8>()?;
                        array.into_pyarray(py).into()
                    }
                    DataType::Uint16 => {
                        let array = view.into_array::<u16>()?;
                        array.into_pyarray(py).into()
                    }
                    DataType::Uint32 => {
                        let array = view.into_array::<u32>()?;
                        array.into_pyarray(py).into()
                    }
                    DataType::Uint64 => {
                        let array = view.into_array::<u64>()?;
                        array.into_pyarray(py).into()
                    }
                    DataType::Int8 => {
                        let array = view.into_array::<i8>()?;
                        array.into_pyarray(py).into()
                    }
                    DataType::Int16 => {
                        let array = view.into_array::<i16>()?;
                        array.into_pyarray(py).into()
                    }
                    DataType::Int32 => {
                        let array = view.into_array::<i32>()?;
                        array.into_pyarray(py).into()
                    }
                    DataType::Int64 => {
                        let array = view.into_array::<i64>()?;
                        array.into_pyarray(py).into()
                    }
                    DataType::Float16 => {
                        let array = view.into_array::<f32>()?;
                        let py_array = array.into_pyarray(py);
                        convert_to_float16(py, py_array)?
                    }
                    DataType::Float32 => {
                        let array = view.into_array::<f32>()?;
                        array.into_pyarray(py).into()
                    }
                    DataType::Float64 => {
                        let array = view.into_array::<f64>()?;
                        array.into_pyarray(py).into()
                    }
                };
                dict.set_item(name, array)?;
            }
            Ok(dict.into_py(py))
        }
    }

    fn replace_arrays(&self, py: Python, arrays: &PyDict, indexes: &PyAny, array_names: Option<&PyList>) -> PyResult<()> {
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

        let mut all_names = Vec::new();

        // 首先加载现有数组
        let existing_arrays = self.load_arrays(
            py,
            array_names,
            false
        )?;

        for (key, value) in arrays.iter() {
            let name = key.extract::<String>()?;
            if let Some(names) = array_names {
                if !names.contains(&name)? {
                    continue;
                }
            }
            
            all_names.push(name.clone());
            let dtype = get_array_dtype(value)?;

            // 获取现有数组
            let existing_array = if let Ok(dict) = existing_arrays.extract::<&PyDict>(py) {
                dict.get_item(&name)
            } else {
                None
            };

            match dtype {
                DataType::Bool => {
                    let array = value.extract::<&PyArray2<bool>>()?;
                    let final_array = if let Some(existing) = existing_array {
                        let array_ref = existing.extract::<&PyArray2<bool>>()?;
                        let mut existing = unsafe { array_ref.as_array().to_owned() };
                        if let Ok(slice) = indexes.extract::<&PySlice>() {
                            let start = slice.getattr("start")?.extract::<Option<i64>>()?.unwrap_or(0);
                            let stop = slice.getattr("stop")?.extract::<Option<i64>>()?.unwrap_or(existing.shape()[0] as i64);
                            let step = slice.getattr("step")?.extract::<Option<i64>>()?.unwrap_or(1);
                            
                            let start = if start < 0 { (existing.shape()[0] as i64 + start) as usize } else { start as usize };
                            let stop = if stop < 0 { (existing.shape()[0] as i64 + stop) as usize } else { stop as usize };
                            
                            if step == 1 {
                                unsafe { existing.slice_mut(s![start..stop, ..]).assign(&array.as_array()) };
                            }
                        } else if let Ok(indices) = indexes.extract::<Vec<i64>>() {
                            for (i, &idx) in indices.iter().enumerate() {
                                let idx = if idx < 0 { (existing.shape()[0] as i64 + idx) as usize } else { idx as usize };
                                unsafe { existing.slice_mut(s![idx, ..]).assign(&array.as_array().slice(s![i, ..])) };
                            }
                        }
                        existing
                    } else {
                        unsafe { array.as_array().to_owned() }
                    };
                    bool_arrays.push((name, final_array, dtype));
                }
                DataType::Uint8 => {
                    let array = value.extract::<&PyArray2<u8>>()?;
                    let final_array = if let Some(existing) = existing_array {
                        let array_ref = existing.extract::<&PyArray2<u8>>()?;
                        let mut existing = unsafe { array_ref.as_array().to_owned() };
                        if let Ok(slice) = indexes.extract::<&PySlice>() {
                            let start = slice.getattr("start")?.extract::<Option<i64>>()?.unwrap_or(0);
                            let stop = slice.getattr("stop")?.extract::<Option<i64>>()?.unwrap_or(existing.shape()[0] as i64);
                            let step = slice.getattr("step")?.extract::<Option<i64>>()?.unwrap_or(1);
                            
                            let start = if start < 0 { (existing.shape()[0] as i64 + start) as usize } else { start as usize };
                            let stop = if stop < 0 { (existing.shape()[0] as i64 + stop) as usize } else { stop as usize };
                            
                            if step == 1 {
                                unsafe { existing.slice_mut(s![start..stop, ..]).assign(&array.as_array()) };
                            }
                        } else if let Ok(indices) = indexes.extract::<Vec<i64>>() {
                            for (i, &idx) in indices.iter().enumerate() {
                                let idx = if idx < 0 { (existing.shape()[0] as i64 + idx) as usize } else { idx as usize };
                                unsafe { existing.slice_mut(s![idx, ..]).assign(&array.as_array().slice(s![i, ..])) };
                            }
                        }
                        existing
                    } else {
                        unsafe { array.as_array().to_owned() }
                    };
                    u8_arrays.push((name, final_array, dtype));
                }
                DataType::Uint16 => {
                    let array = value.extract::<&PyArray2<u16>>()?;
                    let final_array = if let Some(existing) = existing_array {
                        let array_ref = existing.extract::<&PyArray2<u16>>()?;
                        let mut existing = unsafe { array_ref.as_array().to_owned() };
                        if let Ok(slice) = indexes.extract::<&PySlice>() {
                            let start = slice.getattr("start")?.extract::<Option<i64>>()?.unwrap_or(0);
                            let stop = slice.getattr("stop")?.extract::<Option<i64>>()?.unwrap_or(existing.shape()[0] as i64);
                            let step = slice.getattr("step")?.extract::<Option<i64>>()?.unwrap_or(1);
                            
                            let start = if start < 0 { (existing.shape()[0] as i64 + start) as usize } else { start as usize };
                            let stop = if stop < 0 { (existing.shape()[0] as i64 + stop) as usize } else { stop as usize };
                            
                            if step == 1 {
                                unsafe { existing.slice_mut(s![start..stop, ..]).assign(&array.as_array()) };
                            }
                        } else if let Ok(indices) = indexes.extract::<Vec<i64>>() {
                            for (i, &idx) in indices.iter().enumerate() {
                                let idx = if idx < 0 { (existing.shape()[0] as i64 + idx) as usize } else { idx as usize };
                                unsafe { existing.slice_mut(s![idx, ..]).assign(&array.as_array().slice(s![i, ..])) };
                            }
                        }
                        existing
                    } else {
                        unsafe { array.as_array().to_owned() }
                    };
                    u16_arrays.push((name, final_array, dtype));
                }
                DataType::Uint32 => {
                    let array = value.extract::<&PyArray2<u32>>()?;
                    let final_array = if let Some(existing) = existing_array {
                        let array_ref = existing.extract::<&PyArray2<u32>>()?;
                        let mut existing = unsafe { array_ref.as_array().to_owned() };
                        if let Ok(slice) = indexes.extract::<&PySlice>() {
                            let start = slice.getattr("start")?.extract::<Option<i64>>()?.unwrap_or(0);
                            let stop = slice.getattr("stop")?.extract::<Option<i64>>()?.unwrap_or(existing.shape()[0] as i64);
                            let step = slice.getattr("step")?.extract::<Option<i64>>()?.unwrap_or(1);
                            
                            let start = if start < 0 { (existing.shape()[0] as i64 + start) as usize } else { start as usize };
                            let stop = if stop < 0 { (existing.shape()[0] as i64 + stop) as usize } else { stop as usize };
                            
                            if step == 1 {
                                unsafe { existing.slice_mut(s![start..stop, ..]).assign(&array.as_array()) };
                            }
                        } else if let Ok(indices) = indexes.extract::<Vec<i64>>() {
                            for (i, &idx) in indices.iter().enumerate() {
                                let idx = if idx < 0 { (existing.shape()[0] as i64 + idx) as usize } else { idx as usize };
                                unsafe { existing.slice_mut(s![idx, ..]).assign(&array.as_array().slice(s![i, ..])) };
                            }
                        }
                        existing
                    } else {
                        unsafe { array.as_array().to_owned() }
                    };
                    u32_arrays.push((name, final_array, dtype));
                }
                DataType::Uint64 => {
                    let array = value.extract::<&PyArray2<u64>>()?;
                    let final_array = if let Some(existing) = existing_array {
                        let array_ref = existing.extract::<&PyArray2<u64>>()?;
                        let mut existing = unsafe { array_ref.as_array().to_owned() };
                        if let Ok(slice) = indexes.extract::<&PySlice>() {
                            let start = slice.getattr("start")?.extract::<Option<i64>>()?.unwrap_or(0);
                            let stop = slice.getattr("stop")?.extract::<Option<i64>>()?.unwrap_or(existing.shape()[0] as i64);
                            let step = slice.getattr("step")?.extract::<Option<i64>>()?.unwrap_or(1);
                            
                            let start = if start < 0 { (existing.shape()[0] as i64 + start) as usize } else { start as usize };
                            let stop = if stop < 0 { (existing.shape()[0] as i64 + stop) as usize } else { stop as usize };
                            
                            if step == 1 {
                                unsafe { existing.slice_mut(s![start..stop, ..]).assign(&array.as_array()) };
                            }
                        } else if let Ok(indices) = indexes.extract::<Vec<i64>>() {
                            for (i, &idx) in indices.iter().enumerate() {
                                let idx = if idx < 0 { (existing.shape()[0] as i64 + idx) as usize } else { idx as usize };
                                unsafe { existing.slice_mut(s![idx, ..]).assign(&array.as_array().slice(s![i, ..])) };
                            }
                        }
                        existing
                    } else {
                        unsafe { array.as_array().to_owned() }
                    };
                    u64_arrays.push((name, final_array, dtype));
                }
                DataType::Int8 => {
                    let array = value.extract::<&PyArray2<i8>>()?;
                    let final_array = if let Some(existing) = existing_array {
                        let array_ref = existing.extract::<&PyArray2<i8>>()?;
                        let mut existing = unsafe { array_ref.as_array().to_owned() };
                        if let Ok(slice) = indexes.extract::<&PySlice>() {
                            let start = slice.getattr("start")?.extract::<Option<i64>>()?.unwrap_or(0);
                            let stop = slice.getattr("stop")?.extract::<Option<i64>>()?.unwrap_or(existing.shape()[0] as i64);
                            let step = slice.getattr("step")?.extract::<Option<i64>>()?.unwrap_or(1);
                            
                            let start = if start < 0 { (existing.shape()[0] as i64 + start) as usize } else { start as usize };
                            let stop = if stop < 0 { (existing.shape()[0] as i64 + stop) as usize } else { stop as usize };
                            
                            if step == 1 {
                                unsafe { existing.slice_mut(s![start..stop, ..]).assign(&array.as_array()) };
                            }
                        } else if let Ok(indices) = indexes.extract::<Vec<i64>>() {
                            for (i, &idx) in indices.iter().enumerate() {
                                let idx = if idx < 0 { (existing.shape()[0] as i64 + idx) as usize } else { idx as usize };
                                unsafe { existing.slice_mut(s![idx, ..]).assign(&array.as_array().slice(s![i, ..])) };
                            }
                        }
                        existing
                    } else {
                        unsafe { array.as_array().to_owned() }
                    };
                    i8_arrays.push((name, final_array, dtype));
                }
                DataType::Int16 => {
                    let array = value.extract::<&PyArray2<i16>>()?;
                    let final_array = if let Some(existing) = existing_array {
                        let array_ref = existing.extract::<&PyArray2<i16>>()?;
                        let mut existing = unsafe { array_ref.as_array().to_owned() };
                        if let Ok(slice) = indexes.extract::<&PySlice>() {
                            let start = slice.getattr("start")?.extract::<Option<i64>>()?.unwrap_or(0);
                            let stop = slice.getattr("stop")?.extract::<Option<i64>>()?.unwrap_or(existing.shape()[0] as i64);
                            let step = slice.getattr("step")?.extract::<Option<i64>>()?.unwrap_or(1);
                            
                            let start = if start < 0 { (existing.shape()[0] as i64 + start) as usize } else { start as usize };
                            let stop = if stop < 0 { (existing.shape()[0] as i64 + stop) as usize } else { stop as usize };
                            
                            if step == 1 {
                                unsafe { existing.slice_mut(s![start..stop, ..]).assign(&array.as_array()) };
                            }
                        } else if let Ok(indices) = indexes.extract::<Vec<i64>>() {
                            for (i, &idx) in indices.iter().enumerate() {
                                let idx = if idx < 0 { (existing.shape()[0] as i64 + idx) as usize } else { idx as usize };
                                unsafe { existing.slice_mut(s![idx, ..]).assign(&array.as_array().slice(s![i, ..])) };
                            }
                        }
                        existing
                    } else {
                        unsafe { array.as_array().to_owned() }
                    };
                    i16_arrays.push((name, final_array, dtype));
                }
                DataType::Int32 => {
                    let array = value.extract::<&PyArray2<i32>>()?;
                    let final_array = if let Some(existing) = existing_array {
                        let array_ref = existing.extract::<&PyArray2<i32>>()?;
                        let mut existing = unsafe { array_ref.as_array().to_owned() };
                        if let Ok(slice) = indexes.extract::<&PySlice>() {
                            let start = slice.getattr("start")?.extract::<Option<i64>>()?.unwrap_or(0);
                            let stop = slice.getattr("stop")?.extract::<Option<i64>>()?.unwrap_or(existing.shape()[0] as i64);
                            let step = slice.getattr("step")?.extract::<Option<i64>>()?.unwrap_or(1);
                            
                            let start = if start < 0 { (existing.shape()[0] as i64 + start) as usize } else { start as usize };
                            let stop = if stop < 0 { (existing.shape()[0] as i64 + stop) as usize } else { stop as usize };
                            
                            if step == 1 {
                                unsafe { existing.slice_mut(s![start..stop, ..]).assign(&array.as_array()) };
                            }
                        } else if let Ok(indices) = indexes.extract::<Vec<i64>>() {
                            for (i, &idx) in indices.iter().enumerate() {
                                let idx = if idx < 0 { (existing.shape()[0] as i64 + idx) as usize } else { idx as usize };
                                unsafe { existing.slice_mut(s![idx, ..]).assign(&array.as_array().slice(s![i, ..])) };
                            }
                        }
                        existing
                    } else {
                        unsafe { array.as_array().to_owned() }
                    };
                    i32_arrays.push((name, final_array, dtype));
                }
                DataType::Int64 => {
                    let array = value.extract::<&PyArray2<i64>>()?;
                    let final_array = if let Some(existing) = existing_array {
                        let array_ref = existing.extract::<&PyArray2<i64>>()?;
                        let mut existing = unsafe { array_ref.as_array().to_owned() };
                        if let Ok(slice) = indexes.extract::<&PySlice>() {
                            let start = slice.getattr("start")?.extract::<Option<i64>>()?.unwrap_or(0);
                            let stop = slice.getattr("stop")?.extract::<Option<i64>>()?.unwrap_or(existing.shape()[0] as i64);
                            let step = slice.getattr("step")?.extract::<Option<i64>>()?.unwrap_or(1);
                            
                            let start = if start < 0 { (existing.shape()[0] as i64 + start) as usize } else { start as usize };
                            let stop = if stop < 0 { (existing.shape()[0] as i64 + stop) as usize } else { stop as usize };
                            
                            if step == 1 {
                                unsafe { existing.slice_mut(s![start..stop, ..]).assign(&array.as_array()) };
                            }
                        } else if let Ok(indices) = indexes.extract::<Vec<i64>>() {
                            for (i, &idx) in indices.iter().enumerate() {
                                let idx = if idx < 0 { (existing.shape()[0] as i64 + idx) as usize } else { idx as usize };
                                unsafe { existing.slice_mut(s![idx, ..]).assign(&array.as_array().slice(s![i, ..])) };
                            }
                        }
                        existing
                    } else {
                        unsafe { array.as_array().to_owned() }
                    };
                    i64_arrays.push((name, final_array, dtype));
                }
                DataType::Float16 => {
                    let array = Python::with_gil(|py| convert_from_float16(py, value))?;
                    let final_array = if let Some(existing) = existing_array {
                        let mut existing = Python::with_gil(|py| convert_from_float16(py, existing))?;
                        if let Ok(slice) = indexes.extract::<&PySlice>() {
                            let start = slice.getattr("start")?.extract::<Option<i64>>()?.unwrap_or(0);
                            let stop = slice.getattr("stop")?.extract::<Option<i64>>()?.unwrap_or(existing.shape()[0] as i64);
                            let step = slice.getattr("step")?.extract::<Option<i64>>()?.unwrap_or(1);
                            
                            let start = if start < 0 { (existing.shape()[0] as i64 + start) as usize } else { start as usize };
                            let stop = if stop < 0 { (existing.shape()[0] as i64 + stop) as usize } else { stop as usize };
                            
                            if step == 1 {
                                existing.slice_mut(s![start..stop, ..]).assign(&array);
                            }
                        } else if let Ok(indices) = indexes.extract::<Vec<i64>>() {
                            for (i, &idx) in indices.iter().enumerate() {
                                let idx = if idx < 0 { (existing.shape()[0] as i64 + idx) as usize } else { idx as usize };
                                existing.slice_mut(s![idx, ..]).assign(&array.slice(s![i, ..]));
                            }
                        }
                        existing
                    } else {
                        array
                    };
                    f32_arrays.push((name, final_array, dtype));
                }
                DataType::Float32 => {
                    let array = value.extract::<&PyArray2<f32>>()?;
                    let final_array = if let Some(existing) = existing_array {
                        let array_ref = existing.extract::<&PyArray2<f32>>()?;
                        let mut existing = unsafe { array_ref.as_array().to_owned() };
                        if let Ok(slice) = indexes.extract::<&PySlice>() {
                            let start = slice.getattr("start")?.extract::<Option<i64>>()?.unwrap_or(0);
                            let stop = slice.getattr("stop")?.extract::<Option<i64>>()?.unwrap_or(existing.shape()[0] as i64);
                            let step = slice.getattr("step")?.extract::<Option<i64>>()?.unwrap_or(1);
                            
                            let start = if start < 0 { (existing.shape()[0] as i64 + start) as usize } else { start as usize };
                            let stop = if stop < 0 { (existing.shape()[0] as i64 + stop) as usize } else { stop as usize };
                            
                            if step == 1 {
                                unsafe { existing.slice_mut(s![start..stop, ..]).assign(&array.as_array()) };
                            }
                        } else if let Ok(indices) = indexes.extract::<Vec<i64>>() {
                            for (i, &idx) in indices.iter().enumerate() {
                                let idx = if idx < 0 { (existing.shape()[0] as i64 + idx) as usize } else { idx as usize };
                                unsafe { existing.slice_mut(s![idx, ..]).assign(&array.as_array().slice(s![i, ..])) };
                            }
                        }
                        existing
                    } else {
                        unsafe { array.as_array().to_owned() }
                    };
                    f32_arrays.push((name, final_array, dtype));
                }
                DataType::Float64 => {
                    let array = value.extract::<&PyArray2<f64>>()?;
                    let final_array = if let Some(existing) = existing_array {
                        let array_ref = existing.extract::<&PyArray2<f64>>()?;
                        let mut existing = unsafe { array_ref.as_array().to_owned() };
                        if let Ok(slice) = indexes.extract::<&PySlice>() {
                            let start = slice.getattr("start")?.extract::<Option<i64>>()?.unwrap_or(0);
                            let stop = slice.getattr("stop")?.extract::<Option<i64>>()?.unwrap_or(existing.shape()[0] as i64);
                            let step = slice.getattr("step")?.extract::<Option<i64>>()?.unwrap_or(1);
                            
                            let start = if start < 0 { (existing.shape()[0] as i64 + start) as usize } else { start as usize };
                            let stop = if stop < 0 { (existing.shape()[0] as i64 + stop) as usize } else { stop as usize };
                            
                            if step == 1 {
                                unsafe { existing.slice_mut(s![start..stop, ..]).assign(&array.as_array()) };
                            }
                        } else if let Ok(indices) = indexes.extract::<Vec<i64>>() {
                            for (i, &idx) in indices.iter().enumerate() {
                                let idx = if idx < 0 { (existing.shape()[0] as i64 + idx) as usize } else { idx as usize };
                                unsafe { existing.slice_mut(s![idx, ..]).assign(&array.as_array().slice(s![i, ..])) };
                            }
                        }
                        existing
                    } else {
                        unsafe { array.as_array().to_owned() }
                    };
                    f64_arrays.push((name, final_array, dtype));
                }
            }
        }

        self.io.mark_deleted(&all_names)?;
        
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

    fn drop_arrays(&self, array_names: &PyList) -> PyResult<()> {
        let names = array_names.iter()
            .map(|name| name.extract::<String>())
            .collect::<PyResult<Vec<_>>>()?;
        
        self.io.mark_deleted(&names)?;
        Ok(())
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
}

#[pymodule]
fn _lib_numpack(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<NumPack>()?;
    Ok(())
}
