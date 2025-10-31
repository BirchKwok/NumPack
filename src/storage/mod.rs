//! 存储和元数据管理模块
//! 
//! 提供二进制元数据存储和删除位图管理

pub mod binary_metadata;
pub mod deletion_bitmap;

// 重新导出常用类型
pub use binary_metadata::*;
pub use deletion_bitmap::*;

