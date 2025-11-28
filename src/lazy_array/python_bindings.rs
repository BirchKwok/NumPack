//! LazyArray Python绑定模块
//!
//! 从lib.rs中提取的LazyArray相关Python绑定代码

use pyo3::prelude::*;

/// 注册LazyArray相关的Python类到模块
pub fn register_lazy_array_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<crate::lazy_array::standard::LazyArray>()?;
    m.add_class::<crate::lazy_array::iterator::LazyArrayIterator>()?;
    Ok(())
}

/// 便捷的类型别名，用于Python绑定
pub type PyLazyArray = crate::lazy_array::standard::LazyArray;
pub type PyLazyArrayIterator = crate::lazy_array::iterator::LazyArrayIterator;

/// 用于lib.rs中的便捷导入
pub fn setup_lazy_array_bindings() -> Vec<&'static str> {
    vec!["LazyArray", "LazyArrayIterator"]
}
