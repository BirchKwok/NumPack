//! 访问模式相关类型定义
//!
//! 从lazy_array_original.rs中提取的访问模式相关类型

#[derive(Debug, Clone)]
pub enum AccessPatternType {
    Sequential, // 顺序访问
    Random,     // 随机访问
    Clustered,  // 聚集访问
    Mixed,      // 混合访问
}

// 生产级性能优化相关类型定义
#[derive(Debug, Clone)]
pub enum AccessPattern {
    Sequential(usize, usize),     // 顺序访问(start, end)
    Random(Vec<usize>),           // 随机访问
    Strided(usize, usize, usize), // 步长访问(start, stride, count)
}

#[derive(Debug, Clone)]
pub enum AccessHint {
    WillAccessAll,                 // 将访问全部数据
    WillAccessRange(usize, usize), // 将访问特定范围
    WillAccessSparse(f64),         // 稀疏访问(比例)
    WillAccessHot(Vec<usize>),     // 热点访问
}

// 优化算法选择
#[derive(Debug, Clone)]
pub enum OptimizationAlgorithm {
    StandardSIMD,
    AVX512,
    AdaptivePrefetch,
    ZeroCopy,
    Vectorized,
}

// 工作负载提示
#[derive(Debug, Clone)]
pub enum WorkloadHint {
    SequentialRead,
    RandomRead,
    BooleanFiltering,
    HeavyComputation,
}

#[derive(Debug, Clone)]
pub enum SizeCategory {
    Micro,  // < 100 elements
    Small,  // 100 - 1K elements
    Medium, // 1K - 100K elements
    Large,  // 100K - 10M elements
    Huge,   // > 10M elements
}

#[derive(Debug, Clone)]
pub enum AccessFrequency {
    Rare,    // < 1/min
    Low,     // 1-10/min
    Medium,  // 10-100/min
    High,    // 100-1000/min
    Extreme, // > 1000/min
}

// 访问模式分析结果
#[derive(Debug, Clone)]
pub struct AccessPatternAnalysis {
    pub pattern_type: AccessPatternType,
    pub locality_score: f64, // 局部性评分 0-1
    pub density: f64,        // 密度 0-1
    pub size_category: SizeCategory,
    pub frequency: AccessFrequency,
}
