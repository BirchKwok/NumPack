//! NumPack Python绑定注册
//! 
//! 统一的Python模块注册，避免与core.rs中的绑定冲突

use pyo3::prelude::*;

/// 注册NumPack Python绑定
/// 注意：避免与core.rs中现有的Python绑定冲突
pub fn register_python_bindings(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    // 注册类型（如果core.rs中未注册的话）
    // 注意：ArrayMetadata可能已经在其他地方注册了，先注释掉避免冲突
    // m.add_class::<ArrayMetadata>()?;
    
    Ok(())
}
