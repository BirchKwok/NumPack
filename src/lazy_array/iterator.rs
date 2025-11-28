//! LazyArray迭代器实现
//!
//! 从lib.rs中提取的LazyArrayIterator结构体和实现

use crate::lazy_array::standard::LazyArray;
use pyo3::prelude::*;

/// Iterator for LazyArray that yields rows
#[pyclass(module = "numpack")]
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
        let total_rows = array.len_logical();
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

// Windows平台的Drop实现 - 确保迭代器释放时正确清理资源
#[cfg(target_family = "windows")]
impl Drop for LazyArrayIterator {
    fn drop(&mut self) {
        // 迭代器的Drop应该确保其持有的LazyArray也被正确清理
        // 通过显式触发array的清理来确保资源释放
        let array_path = &self.array.array_path;

        // 检查是否为临时文件
        let is_temp = array_path.contains("temp") || array_path.contains("tmp");
        if is_temp {
            // 对于临时文件，强制执行清理流程
            let path = std::path::Path::new(array_path);

            // 先强制释放对mmap的引用
            std::mem::drop(std::sync::Arc::clone(&self.array.mmap));

            // 触发Windows文件句柄释放
            crate::lazy_array::standard::release_windows_file_handle(path);

            // 短暂等待确保Windows有时间处理文件释放
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }
}

// 非Windows平台的简单Drop实现
#[cfg(not(target_family = "windows"))]
impl Drop for LazyArrayIterator {
    fn drop(&mut self) {
        // 非Windows平台不需要特殊处理
    }
}
