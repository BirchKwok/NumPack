//! LazyArray迭代器实现
//! 
//! 从lib.rs中提取的LazyArrayIterator结构体和实现

use pyo3::prelude::*;
use crate::lazy_array::standard::LazyArray;

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

impl LazyArrayIterator {
    /// 创建新的LazyArrayIterator实例
    pub fn new(array: LazyArray) -> Self {
        let total_rows = array.shape[0];
        Self {
            array,
            current_index: 0,
            total_rows,
        }
    }

    /// 重置迭代器到开始位置
    pub fn reset(&mut self) {
        self.current_index = 0;
    }

    /// 获取当前位置
    pub fn position(&self) -> usize {
        self.current_index
    }

    /// 获取总行数
    pub fn total_rows(&self) -> usize {
        self.total_rows
    }

    /// 检查是否还有更多元素
    pub fn has_next(&self) -> bool {
        self.current_index < self.total_rows
    }

    /// 跳过指定数量的行
    pub fn skip(&mut self, count: usize) {
        self.current_index = (self.current_index + count).min(self.total_rows);
    }

    /// 设置迭代器位置
    pub fn set_position(&mut self, position: usize) {
        self.current_index = position.min(self.total_rows);
    }
}
