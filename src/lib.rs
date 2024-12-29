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

    #[pyo3(signature = (array_names, indexes=None))]
    fn drop_arrays(&self, array_names: &PyAny, indexes: Option<&PyAny>) -> PyResult<()> {
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

        // 如果有indexes参数，说明是删除特定行
        if let Some(indexes) = indexes {
            let py = indexes.py();
            
            // 加载原始数组
            let arrays = self.load_arrays(py, Some(&PyList::new(py, &names)), false)?;
            let dict = arrays.downcast::<PyDict>(py)?;
            
            for name in &names {
                if let Some(array) = dict.get_item(name) {
                    let array = array.downcast::<PyAny>()?;
                    let shape = array.getattr("shape")?.extract::<(usize, usize)>()?;
                    
                    // 获取要删除的行数
                    let rows_to_delete = if let Ok(slice) = indexes.downcast::<PySlice>() {
                        let start = slice.getattr("start")?.extract::<Option<i64>>()?.unwrap_or(0);
                        let stop = slice.getattr("stop")?.extract::<Option<i64>>()?.unwrap_or(shape.0 as i64);
                        let step = slice.getattr("step")?.extract::<Option<i64>>()?.unwrap_or(1);
                        
                        let start = if start < 0 { (shape.0 as i64 + start) as usize } else { start as usize };
                        let stop = if stop < 0 { (shape.0 as i64 + stop) as usize } else { stop as usize };
                        
                        if step == 1 {
                            stop - start
                        } else {
                            0
                        }
                    } else if let Ok(indices) = indexes.extract::<Vec<i64>>() {
                        indices.len()
                    } else {
                        return Err(pyo3::exceptions::PyTypeError::new_err(
                            "indexes must be a slice or list of integers"
                        ));
                    };

                    // 更新元数据中的行数
                    if let Some(meta) = self.io.get_array_meta(name) {
                        let mut new_meta = meta.clone();
                        new_meta.rows = new_meta.rows.saturating_sub(rows_to_delete as u64);
                        // 更新元数据
                        self.io.update_array_metadata(name, new_meta)?;
                    }
                }
            }
            
            Ok(())
        } else {
            // 批量标记删除
            let deleted_count = self.io.batch_mark_deleted(&names)?;
            
            // 如果删除的数组数量达到阈值，执行增量压缩
            if deleted_count > 0 && self.io.should_compact(4 * 1024 * 1024 * 1024) { // 4GB阈值
                // 每次处理100个数组
                let compacted = self.io.incremental_compact(100)?;
                if !compacted.is_empty() {
                    // 删除已压缩的数组文件
                    for name in compacted {
                        if let Some(meta) = self.io.get_array_meta(&name) {
                            let file_path = self.base_dir.join(&meta.data_file);
                            if let Err(e) = std::fs::remove_file(file_path) {
                                eprintln!("Warning: Failed to remove file for array {}: {}", name, e);
                            }
                        }
                    }
                }
            }
            
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
}

#[pymodule]
fn _lib_numpack(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<NumPack>()?;
    Ok(())
}
