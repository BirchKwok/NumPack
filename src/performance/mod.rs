//! 性能监控和分析模块
//! 
//! 提供性能指标收集、算法选择和系统监控功能

pub mod metrics;
pub mod profiler;
pub mod monitor;
pub mod algorithm_selector;
pub mod gil_optimization;

// 只导出实际使用的模块
pub use monitor::*;
pub use gil_optimization::{
    GILOptimizer,
    GILReleaseConfig,
    GILOptimizationStats,
    GILOperation,
};
// 暂时注释掉未使用的导入，避免警告
// pub use metrics::*;
// pub use profiler::*;
// pub use algorithm_selector::*; 