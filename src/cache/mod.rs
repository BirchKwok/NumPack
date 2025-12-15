//! 缓存系统模块
//!
//! 提供智能缓存、预取优化和多级缓存功能
//! 专门针对大型数组的内存访问模式优化

pub mod adaptive_cache;
pub mod adaptive_compression;
pub mod compressed_cache;
// pub mod compression_cache; // 已删除：完全未使用
pub mod lru_cache;
pub mod multilevel_cache;
pub mod performance_monitor;
pub mod prefetch;
pub mod smart_cache;

// 暂时注释掉未使用的导入，避免警告
// pub use smart_cache::*;
pub use prefetch::*;
// pub use multilevel_cache::*;
// pub use adaptive_cache::*;
// pub use lru_cache::*;
// pub use compressed_cache::*;
// pub use performance_monitor::*;
