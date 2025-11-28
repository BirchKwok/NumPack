//! NumPack Python绑定注册
//!
//! 统一的Python模块注册，避免与core.rs中的绑定冲突

use pyo3::prelude::*;

/// 注册NumPack Python绑定
/// 注意：避免与core.rs中现有的Python绑定冲突
pub fn register_python_bindings(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // 注册新的 LazyArray 类（支持算术操作符）
    m.add_class::<crate::lazy_array::standard::LazyArray>()?;
    m.add_class::<crate::lazy_array::iterator::LazyArrayIterator>()?;

    Ok(())
}
