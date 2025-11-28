//! 系统监控器
//!
//! 从lazy_array_original.rs中提取的系统监控功能

use std::time::{Duration, Instant};

/// 系统监控器，监控CPU和内存使用情况
#[derive(Debug)]
pub struct SystemMonitor {
    last_cpu_check: Instant,
    last_memory_check: Instant,
    cpu_utilization: f64,
    memory_usage: usize,
}

impl SystemMonitor {
    pub fn new() -> Self {
        Self {
            last_cpu_check: Instant::now(),
            last_memory_check: Instant::now(),
            cpu_utilization: 0.0,
            memory_usage: 0,
        }
    }

    /// 获取CPU利用率（0.0-1.0）
    pub fn get_cpu_utilization(&mut self) -> f64 {
        let now = Instant::now();
        if now.duration_since(self.last_cpu_check) > Duration::from_secs(1) {
            // 简化的CPU使用率估算
            // 在实际实现中，这里应该使用系统API
            self.cpu_utilization = self.estimate_cpu_usage();
            self.last_cpu_check = now;
        }
        self.cpu_utilization
    }

    /// 获取内存使用量（字节）
    pub fn get_memory_usage(&mut self) -> usize {
        let now = Instant::now();
        if now.duration_since(self.last_memory_check) > Duration::from_secs(5) {
            // 简化的内存使用估算
            self.memory_usage = self.estimate_memory_usage();
            self.last_memory_check = now;
        }
        self.memory_usage
    }

    /// 估算CPU使用率
    fn estimate_cpu_usage(&self) -> f64 {
        // 简化实现：基于当前线程数估算
        #[cfg(feature = "rayon")]
        {
            let thread_count = rayon::current_num_threads();
            (thread_count as f64 * 0.1).min(1.0)
        }
        #[cfg(not(feature = "rayon"))]
        {
            // 如果没有rayon，使用默认值
            0.3
        }
    }

    /// 估算内存使用量
    fn estimate_memory_usage(&self) -> usize {
        // 简化实现：返回固定值
        // 在实际实现中应该使用系统API获取真实内存使用
        #[cfg(target_os = "linux")]
        {
            self.get_linux_memory_usage().unwrap_or(1024 * 1024 * 100)
        }
        #[cfg(target_os = "windows")]
        {
            self.get_windows_memory_usage().unwrap_or(1024 * 1024 * 100)
        }
        #[cfg(target_os = "macos")]
        {
            self.get_macos_memory_usage().unwrap_or(1024 * 1024 * 100)
        }
        #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
        {
            1024 * 1024 * 100 // 100MB
        }
    }

    /// Linux系统内存使用量获取
    #[cfg(target_os = "linux")]
    fn get_linux_memory_usage(&self) -> Option<usize> {
        use std::fs;

        let status = fs::read_to_string("/proc/self/status").ok()?;
        for line in status.lines() {
            if line.starts_with("VmRSS:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    if let Ok(kb) = parts[1].parse::<usize>() {
                        return Some(kb * 1024); // 转换为字节
                    }
                }
            }
        }
        None
    }

    /// Windows系统内存使用量获取
    #[cfg(target_os = "windows")]
    fn get_windows_memory_usage(&self) -> Option<usize> {
        // 在实际实现中，这里应该使用Windows API
        // 例如GetProcessMemoryInfo
        Some(1024 * 1024 * 100) // 占位符
    }

    /// macOS系统内存使用量获取
    #[cfg(target_os = "macos")]
    fn get_macos_memory_usage(&self) -> Option<usize> {
        // 在实际实现中，这里应该使用mach API
        Some(1024 * 1024 * 100) // 占位符
    }

    /// 重置监控状态
    pub fn reset(&mut self) {
        self.last_cpu_check = Instant::now();
        self.last_memory_check = Instant::now();
        self.cpu_utilization = 0.0;
        self.memory_usage = 0;
    }

    /// 获取系统监控摘要
    pub fn get_summary(&mut self) -> SystemSummary {
        SystemSummary {
            cpu_utilization: self.get_cpu_utilization(),
            memory_usage_bytes: self.get_memory_usage(),
            memory_usage_mb: self.get_memory_usage() / (1024 * 1024),
            last_updated: Instant::now(),
        }
    }

    /// 检查系统是否处于高负载状态
    pub fn is_high_load(&mut self) -> bool {
        let cpu = self.get_cpu_utilization();
        let memory_gb = self.get_memory_usage() as f64 / (1024.0 * 1024.0 * 1024.0);

        cpu > 0.8 || memory_gb > 2.0 // CPU超过80%或内存超过2GB
    }

    /// 检查系统是否适合执行高性能操作
    pub fn is_suitable_for_high_performance(&mut self) -> bool {
        let cpu = self.get_cpu_utilization();
        let memory_gb = self.get_memory_usage() as f64 / (1024.0 * 1024.0 * 1024.0);

        cpu < 0.5 && memory_gb < 1.0 // CPU低于50%且内存低于1GB
    }
}

/// 系统监控摘要
#[derive(Debug, Clone)]
pub struct SystemSummary {
    pub cpu_utilization: f64,
    pub memory_usage_bytes: usize,
    pub memory_usage_mb: usize,
    pub last_updated: Instant,
}
