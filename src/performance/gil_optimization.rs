//! GIL优化模块
//! 
//! 提供智能的GIL释放策略，在I/O密集型操作期间释放GIL以实现真正的并行性

use pyo3::prelude::*;
use std::time::{Duration, Instant};

/// GIL释放配置
#[derive(Clone, Debug)]
pub struct GILReleaseConfig {
    /// 最小操作大小（字节），低于此值不释放GIL
    pub min_data_size_for_release: usize,
    /// 最小预期操作时间（微秒），低于此值不释放GIL
    pub min_operation_time_us: u64,
    /// 是否在save操作时释放GIL
    pub release_on_save: bool,
    /// 是否在load操作时释放GIL
    pub release_on_load: bool,
    /// 是否在批量操作时释放GIL
    pub release_on_batch: bool,
}

impl Default for GILReleaseConfig {
    fn default() -> Self {
        Self {
            min_data_size_for_release: 100 * 1024,  // 100KB
            min_operation_time_us: 1000,             // 1ms
            release_on_save: true,
            release_on_load: true,
            release_on_batch: true,
        }
    }
}

/// GIL优化统计
#[derive(Debug, Clone, Default)]
pub struct GILOptimizationStats {
    /// GIL释放次数
    pub gil_releases: u64,
    /// GIL持有时间（微秒）
    pub gil_held_time_us: u64,
    /// GIL释放后的时间（微秒）
    pub gil_released_time_us: u64,
    /// 因GIL释放节省的时间（微秒）
    pub time_saved_us: u64,
}

impl GILOptimizationStats {
    /// 记录一次GIL释放操作
    #[inline(always)]
    pub fn record_gil_release(&mut self, released_duration: Duration) {
        self.gil_releases += 1;
        self.gil_released_time_us += released_duration.as_micros() as u64;
    }
    
    /// 记录GIL持有时间
    #[inline(always)]
    pub fn record_gil_held(&mut self, held_duration: Duration) {
        self.gil_held_time_us += held_duration.as_micros() as u64;
    }
    
    /// 计算GIL释放比例
    pub fn gil_release_ratio(&self) -> f64 {
        let total_time = self.gil_held_time_us + self.gil_released_time_us;
        if total_time == 0 {
            return 0.0;
        }
        self.gil_released_time_us as f64 / total_time as f64
    }
}

/// GIL优化器 - 智能决策何时释放GIL
pub struct GILOptimizer {
    config: GILReleaseConfig,
    stats: GILOptimizationStats,
}

impl GILOptimizer {
    /// 创建新的GIL优化器
    pub fn new(config: GILReleaseConfig) -> Self {
        Self {
            config,
            stats: GILOptimizationStats::default(),
        }
    }
    
    /// 创建使用默认配置的优化器
    pub fn default() -> Self {
        Self::new(GILReleaseConfig::default())
    }
    
    /// 判断是否应该释放GIL
    #[inline(always)]
    pub fn should_release_gil(&self, operation_type: GILOperation, data_size: usize) -> bool {
        // 检查数据大小阈值
        if data_size < self.config.min_data_size_for_release {
            return false;
        }
        
        // 根据操作类型决定
        match operation_type {
            GILOperation::Save => self.config.release_on_save,
            GILOperation::Load => self.config.release_on_load,
            GILOperation::BatchAccess => self.config.release_on_batch,
            GILOperation::FileIO => true,  // 文件I/O总是释放
            GILOperation::Computation => data_size > 1024 * 1024,  // 大于1MB的计算
        }
    }
    
    /// 执行带GIL释放的操作
    /// 
    /// # Example
    /// ```rust
    /// let optimizer = GILOptimizer::default();
    /// let result = optimizer.execute_with_gil_release(
    ///     py,
    ///     GILOperation::FileIO,
    ///     1024 * 1024,
    ///     || {
    ///         // 这里的代码在释放GIL的情况下执行
    ///         perform_io_operation()
    ///     }
    /// )?;
    /// ```
    pub fn execute_with_gil_release<F, R>(
        &mut self,
        py: Python,
        operation_type: GILOperation,
        data_size: usize,
        f: F,
    ) -> PyResult<R>
    where
        F: FnOnce() -> PyResult<R> + Send,
        R: Send,
    {
        if self.should_release_gil(operation_type, data_size) {
            // 释放GIL执行操作
            let start = Instant::now();
            let result = py.allow_threads(f);
            let duration = start.elapsed();
            
            self.stats.record_gil_release(duration);
            
            result
        } else {
            // 保持GIL执行
            let start = Instant::now();
            let result = f();
            let duration = start.elapsed();
            
            self.stats.record_gil_held(duration);
            
            result
        }
    }
    
    /// 获取统计信息
    pub fn get_stats(&self) -> &GILOptimizationStats {
        &self.stats
    }
    
    /// 重置统计信息
    pub fn reset_stats(&mut self) {
        self.stats = GILOptimizationStats::default();
    }
}

/// GIL操作类型
#[derive(Debug, Clone, Copy)]
pub enum GILOperation {
    /// Save操作
    Save,
    /// Load操作
    Load,
    /// 批量访问
    BatchAccess,
    /// 文件I/O
    FileIO,
    /// 计算操作
    Computation,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_should_release_gil() {
        let optimizer = GILOptimizer::default();
        
        // 小数据不应释放
        assert!(!optimizer.should_release_gil(GILOperation::Save, 1024));
        
        // 大数据应该释放
        assert!(optimizer.should_release_gil(GILOperation::Save, 1024 * 1024));
        
        // FileIO总是释放
        assert!(optimizer.should_release_gil(GILOperation::FileIO, 1024));
    }
    
    #[test]
    fn test_gil_stats() {
        let mut stats = GILOptimizationStats::default();
        
        stats.record_gil_release(Duration::from_millis(10));
        stats.record_gil_held(Duration::from_millis(5));
        
        assert_eq!(stats.gil_releases, 1);
        assert_eq!(stats.gil_released_time_us, 10000);
        assert_eq!(stats.gil_held_time_us, 5000);
        
        let ratio = stats.gil_release_ratio();
        assert!((ratio - 0.666).abs() < 0.01);
    }
}

