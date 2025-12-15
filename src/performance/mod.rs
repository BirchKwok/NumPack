//! 性能监控和分析模块
//!
//! 提供性能指标收集、算法选择和系统监控功能

pub mod algorithm_selector;
pub mod gil_optimization;
pub mod metrics;
pub mod monitor;
pub mod profiler;

// 只导出实际使用的模块
#[allow(unused_imports)]
pub use gil_optimization::{GILOperation, GILOptimizer, GILReleaseConfig};
#[allow(unused_imports)]
pub use monitor::*;
// 暂时注释掉未使用的导入，避免警告
// pub use metrics::*;
// pub use profiler::*;
// pub use algorithm_selector::*;
