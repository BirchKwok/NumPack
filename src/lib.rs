mod error;
mod io;
mod types;

use std::path::Path;
use std::collections::HashMap;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use numpy::{PyArray2, IntoPyArray};

use crate::io::{save_arrays, load_arrays};

#[pyfunction]
#[pyo3(signature = (filename, arrays, array_name=None))]
fn save_nnp(_py: Python, filename: String, arrays: &PyDict, array_name: Option<String>) -> PyResult<()> {
    let mut array_map = HashMap::new();
    for (i, (key, value)) in arrays.iter().enumerate() {
        let name = if let Some(prefix) = &array_name {
            format!("{}{}", prefix, i)
        } else {
            key.extract::<String>()?
        };
        let array = value.extract::<&PyArray2<f32>>()?;
        array_map.insert(name, array.to_owned());
    }

    save_arrays(Path::new(&filename), &array_map)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (filename, array_names=None, mmap_mode=false))]
fn load_nnp(py: Python, filename: String, array_names: Option<&PyList>, mmap_mode: bool) -> PyResult<Py<PyDict>> {
    let arrays = load_arrays(Path::new(&filename))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    
    let dict = PyDict::new(py);
    for (name, array) in arrays {
        if let Some(names) = array_names {
            if !names.contains(name.as_str())? {
                continue;
            }
        }
        let py_array = array.into_pyarray(py);
        dict.set_item(name, py_array)?;
    }
    Ok(dict.into())
}

#[pyfunction]
#[pyo3(signature = (filename, arrays, indexes, array_names=None))]
fn replace_arrays(py: Python, filename: String, arrays: &PyDict, indexes: &PyAny, array_names: Option<&PyList>) -> PyResult<()> {
    // 首先加载现有数组
    let mut existing = load_arrays(Path::new(&filename))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    
    // 处理要替换的数组
    for (key, value) in arrays.iter() {
        let name = key.extract::<String>()?;
        if let Some(names) = array_names {
            if !names.contains(name.as_str())? {
                continue;
            }
        }
        let array = value.extract::<&PyArray2<f32>>()?;
        
        // 如果数组已存在，则替换它
        if let Some(existing_array) = existing.get_mut(&name) {
            // TODO: 根据 indexes 参数替换部分数据
            *existing_array = unsafe { array.as_array().to_owned() };
        }
    }
    
    // 保存更新后的数组
    let mut array_map = HashMap::new();
    for (name, array) in existing {
        array_map.insert(name, array.into_pyarray(py).to_owned());
    }
    
    save_arrays(Path::new(&filename), &array_map)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (filename, arrays, array_names=None))]
fn append_arrays(py: Python, filename: String, arrays: &PyDict, array_names: Option<&PyList>) -> PyResult<()> {
    // 首先加载现有数组
    let mut existing = load_arrays(Path::new(&filename))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    
    // 添加新数组
    for (key, value) in arrays.iter() {
        let name = key.extract::<String>()?;
        if let Some(names) = array_names {
            if !names.contains(name.as_str())? {
                continue;
            }
        }
        let array = value.extract::<&PyArray2<f32>>()?;
        if existing.contains_key(&name) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Array '{}' already exists", name)
            ));
        }
        existing.insert(name, unsafe { array.as_array().to_owned() });
    }
    
    // 保存更新后的数组
    let mut array_map = HashMap::new();
    for (name, array) in existing {
        array_map.insert(name, array.into_pyarray(py).to_owned());
    }
    
    save_arrays(Path::new(&filename), &array_map)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (filename, indexes, array_names=None))]
fn drop_arrays(py: Python, filename: String, indexes: &PyAny, array_names: Option<&PyList>) -> PyResult<()> {
    // 首先加载现有数组
    let mut existing = load_arrays(Path::new(&filename))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    
    // 删除指定的数组
    if let Some(names) = array_names {
        for name in names.iter() {
            if let Ok(name) = name.extract::<String>() {
                existing.remove(&name);
            }
        }
    }
    
    // 如果没有剩余数组，直接返回
    if existing.is_empty() {
        return Ok(());
    }
    
    // 保存剩余的数组
    let mut array_map = HashMap::new();
    for (name, array) in existing {
        array_map.insert(name, array.into_pyarray(py).to_owned());
    }
    
    save_arrays(Path::new(&filename), &array_map)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

#[pymodule]
fn _lib_numpack(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(save_nnp, m)?)?;
    m.add_function(wrap_pyfunction!(load_nnp, m)?)?;
    m.add_function(wrap_pyfunction!(replace_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(append_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(drop_arrays, m)?)?;
    Ok(())
}