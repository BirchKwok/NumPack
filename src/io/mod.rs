//! IO操作模块
//!
//! 提供并行IO和批量访问引擎

pub mod batch_access_engine;
pub mod parallel_io;

// 重新导出常用类型
pub use parallel_io::*;
