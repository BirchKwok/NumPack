//! 缓存系统模块
//! 
//! 提供智能缓存、预取优化和多级缓存功能
//! 专门针对大型数组的内存访问模式优化

pub mod smart_cache;
pub mod prefetch;
pub mod multilevel_cache;
pub mod adaptive_cache;
pub mod compression_cache;
pub mod lru_cache;
pub mod compressed_cache;
pub mod performance_monitor;

// 暂时注释掉未使用的导入，避免警告
// pub use smart_cache::*;
pub use prefetch::*;
// pub use multilevel_cache::*;
// pub use adaptive_cache::*;
// pub use compression_cache::*;
// pub use lru_cache::*;
// pub use compressed_cache::*;
// pub use performance_monitor::*; 