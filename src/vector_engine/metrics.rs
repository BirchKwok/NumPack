//! 距离度量类型定义

use std::fmt;

/// 支持的距离度量类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetricType {
    /// 点积 (Dot Product)
    /// 计算两个向量的内积：sum(a[i] * b[i])
    DotProduct,

    /// 余弦相似度 (Cosine Similarity)
    /// 计算：(a · b) / (||a|| * ||b||)
    /// 范围：[-1, 1]，值越大越相似
    Cosine,

    /// L2 距离 (Euclidean Distance)
    /// 计算：sqrt(sum((a[i] - b[i])^2))
    L2Distance,

    /// L2 距离平方 (Squared Euclidean Distance)
    /// 计算：sum((a[i] - b[i])^2)
    /// 避免开方运算，速度更快
    L2Squared,

    /// 汉明距离 (Hamming Distance)
    /// 用于二值向量，计算不同位的数量
    Hamming,

    /// Jaccard 距离 (Jaccard Distance)
    /// 计算：1 - |A ∩ B| / |A ∪ B|
    Jaccard,

    /// KL 散度 (Kullback-Leibler Divergence)
    /// 衡量两个概率分布的差异
    KL,

    /// JS 散度 (Jensen-Shannon Divergence)
    /// KL 散度的对称版本
    JS,

    /// 内积 (Inner Product)
    /// 与 DotProduct 相同，但语义不同
    InnerProduct,
}

impl MetricType {
    /// 从字符串解析度量类型
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "dot" | "dot_product" | "dotproduct" => Some(MetricType::DotProduct),
            "cos" | "cosine" | "cosine_similarity" => Some(MetricType::Cosine),
            "l2" | "euclidean" | "l2_distance" => Some(MetricType::L2Distance),
            "l2sq" | "l2_squared" | "squared_euclidean" => Some(MetricType::L2Squared),
            "hamming" => Some(MetricType::Hamming),
            "jaccard" => Some(MetricType::Jaccard),
            "kl" | "kl_divergence" => Some(MetricType::KL),
            "js" | "js_divergence" => Some(MetricType::JS),
            "inner" | "inner_product" => Some(MetricType::InnerProduct),
            _ => None,
        }
    }

    /// 转换为字符串
    pub fn as_str(&self) -> &'static str {
        match self {
            MetricType::DotProduct => "dot",
            MetricType::Cosine => "cosine",
            MetricType::L2Distance => "l2",
            MetricType::L2Squared => "l2sq",
            MetricType::Hamming => "hamming",
            MetricType::Jaccard => "jaccard",
            MetricType::KL => "kl",
            MetricType::JS => "js",
            MetricType::InnerProduct => "inner",
        }
    }

    /// 是否是相似度度量（值越大越相似）
    ///
    /// Returns:
    ///     true: 相似度度量（Cosine, DotProduct, InnerProduct）
    ///     false: 距离度量（L2, Hamming, KL, JS）
    pub fn is_similarity(&self) -> bool {
        matches!(
            self,
            MetricType::DotProduct | MetricType::Cosine | MetricType::InnerProduct
        )
    }

    /// 是否需要归一化
    pub fn requires_normalization(&self) -> bool {
        matches!(self, MetricType::Cosine)
    }
}

impl fmt::Display for MetricType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_str() {
        assert_eq!(MetricType::from_str("dot"), Some(MetricType::DotProduct));
        assert_eq!(MetricType::from_str("cosine"), Some(MetricType::Cosine));
        assert_eq!(MetricType::from_str("l2"), Some(MetricType::L2Distance));
        assert_eq!(MetricType::from_str("invalid"), None);
    }

    #[test]
    fn test_is_similarity() {
        assert!(MetricType::DotProduct.is_similarity());
        assert!(MetricType::Cosine.is_similarity());
        assert!(!MetricType::L2Distance.is_similarity());
        assert!(!MetricType::L2Squared.is_similarity());
    }

    #[test]
    fn test_requires_normalization() {
        assert!(MetricType::Cosine.requires_normalization());
        assert!(!MetricType::DotProduct.requires_normalization());
    }
}
