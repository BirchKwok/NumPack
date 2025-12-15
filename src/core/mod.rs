//! 核心基础设施模块
//!
//! 提供错误处理和核心数据类型定义

pub mod error;
pub mod metadata;

// 重新导出常用类型
#[allow(unused_imports)]
pub use error::*;
pub use metadata::*;
