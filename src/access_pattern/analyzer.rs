//! 访问模式分析器
//!
//! 从lazy_array_original.rs中提取的访问模式分析器实现

use super::types::*;
use std::time::{Duration, Instant};

// 访问模式分析器
#[derive(Debug)]
pub struct AccessPatternAnalyzer {
    recent_accesses: Vec<(usize, Instant)>,
    pattern_history: Vec<AccessPatternAnalysis>,
    max_history: usize,
}

impl AccessPatternAnalyzer {
    pub fn new() -> Self {
        Self {
            recent_accesses: Vec::new(),
            pattern_history: Vec::new(),
            max_history: 1000,
        }
    }

    pub fn analyze_access(&mut self, offset: usize, size: usize) -> AccessPatternAnalysis {
        let now = Instant::now();
        self.recent_accesses.push((offset, now));

        // 保持历史记录在合理范围内
        if self.recent_accesses.len() > self.max_history {
            self.recent_accesses.drain(0..self.max_history / 2);
        }

        let pattern_type = self.detect_pattern_type();
        let locality_score = self.calculate_locality_score();
        let density = self.calculate_density();
        let size_category = self.categorize_size(size);
        let frequency = self.calculate_frequency();

        let analysis = AccessPatternAnalysis {
            pattern_type,
            locality_score,
            density,
            size_category,
            frequency,
        };

        self.pattern_history.push(analysis.clone());
        if self.pattern_history.len() > 100 {
            self.pattern_history.drain(0..50);
        }

        analysis
    }

    fn detect_pattern_type(&self) -> AccessPatternType {
        if self.recent_accesses.len() < 3 {
            return AccessPatternType::Random;
        }

        let mut sequential_count = 0;
        let mut clustered_count = 0;

        for window in self.recent_accesses.windows(2) {
            let diff = if window[1].0 > window[0].0 {
                window[1].0 - window[0].0
            } else {
                window[0].0 - window[1].0
            };

            if diff < 1024 {
                // 1KB内认为是顺序或聚集
                if diff < 64 {
                    // 64字节内认为是顺序
                    sequential_count += 1;
                } else {
                    clustered_count += 1;
                }
            }
        }

        let total = self.recent_accesses.len() - 1;
        if sequential_count as f64 / total as f64 > 0.7 {
            AccessPatternType::Sequential
        } else if clustered_count as f64 / total as f64 > 0.5 {
            AccessPatternType::Clustered
        } else if sequential_count + clustered_count > total / 2 {
            AccessPatternType::Mixed
        } else {
            AccessPatternType::Random
        }
    }

    fn calculate_locality_score(&self) -> f64 {
        if self.recent_accesses.len() < 2 {
            return 0.0;
        }

        let mut total_distance = 0usize;
        let mut count = 0;

        for window in self.recent_accesses.windows(2) {
            let distance = if window[1].0 > window[0].0 {
                window[1].0 - window[0].0
            } else {
                window[0].0 - window[1].0
            };
            total_distance += distance;
            count += 1;
        }

        let avg_distance = total_distance as f64 / count as f64;
        // 距离越小，局部性越好
        (1.0 / (1.0 + avg_distance / 1024.0)).min(1.0)
    }

    fn calculate_density(&self) -> f64 {
        if self.recent_accesses.is_empty() {
            return 0.0;
        }

        let min_offset = self
            .recent_accesses
            .iter()
            .map(|(o, _)| *o)
            .min()
            .unwrap_or(0);
        let max_offset = self
            .recent_accesses
            .iter()
            .map(|(o, _)| *o)
            .max()
            .unwrap_or(0);

        if max_offset == min_offset {
            return 1.0;
        }

        let range = max_offset - min_offset;
        let unique_accesses = self.recent_accesses.len();

        unique_accesses as f64 / (range as f64 / 1024.0 + 1.0)
    }

    fn categorize_size(&self, size: usize) -> SizeCategory {
        match size {
            0..=100 => SizeCategory::Micro,
            101..=1000 => SizeCategory::Small,
            1001..=100000 => SizeCategory::Medium,
            100001..=10000000 => SizeCategory::Large,
            _ => SizeCategory::Huge,
        }
    }

    fn calculate_frequency(&self) -> AccessFrequency {
        if self.recent_accesses.len() < 2 {
            return AccessFrequency::Rare;
        }

        let now = Instant::now();
        let minute_ago = now - Duration::from_secs(60);

        let recent_count = self
            .recent_accesses
            .iter()
            .filter(|(_, time)| *time > minute_ago)
            .count();

        match recent_count {
            0 => AccessFrequency::Rare,
            1..=10 => AccessFrequency::Low,
            11..=100 => AccessFrequency::Medium,
            101..=1000 => AccessFrequency::High,
            _ => AccessFrequency::Extreme,
        }
    }
}
