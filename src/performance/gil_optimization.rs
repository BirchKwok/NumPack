//! GIL优化模块
//!
//! 提供智能的GIL释放策略，在I/O密集型操作期间释放GIL以实现真正的并行性

use pyo3::prelude::*;

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
            min_data_size_for_release: 100 * 1024, // 100KB
            min_operation_time_us: 1000,           // 1ms
            release_on_save: true,
            release_on_load: true,
            release_on_batch: true,
        }
    }
}

/// GIL优化器 - 智能决策何时释放GIL
pub struct GILOptimizer {
    config: GILReleaseConfig,
}

impl GILOptimizer {
    /// 创建新的GIL优化器
    pub fn new(config: GILReleaseConfig) -> Self {
        Self { config }
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
            GILOperation::FileIO => true, // 文件I/O总是释放
            GILOperation::Computation => data_size > 1024 * 1024, // 大于1MB的计算
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
            py.allow_threads(f)
        } else {
            // 保持GIL执行
            f()
        }
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
}
