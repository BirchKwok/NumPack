//! NumPack主要功能模块
//! 提供高性能数组存储和管理功能

#[cfg(feature = "python")]
pub mod core;
#[cfg(feature = "python")]
pub mod python_bindings;

#[cfg(feature = "python")]
#[allow(unused_imports)]
pub use core::*;
// Python绑定不重新导出，仅供lib.rs使用
