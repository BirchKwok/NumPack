//! 访问模式预测器
//!
//! 从lazy_array_original.rs中提取的访问模式预测器实现

use std::collections::HashMap;

pub struct AccessPatternPredictor {
    pattern_history: Vec<Vec<usize>>,
    stride_patterns: HashMap<usize, usize>, // stride -> frequency
    sequence_patterns: HashMap<Vec<usize>, usize>, // sequence -> frequency
    max_history: usize,
    min_confidence: f64,
}

impl AccessPatternPredictor {
    pub fn new() -> Self {
        Self {
            pattern_history: Vec::new(),
            stride_patterns: HashMap::new(),
            sequence_patterns: HashMap::new(),
            max_history: 50,
            min_confidence: 0.6,
        }
    }

    pub fn learn_pattern(&mut self, indices: &[usize]) {
        // 记录访问模式历史
        self.pattern_history.push(indices.to_vec());
        if self.pattern_history.len() > self.max_history {
            self.pattern_history.remove(0);
        }

        // 学习步长模式
        self.learn_stride_patterns(indices);

        // 学习序列模式
        self.learn_sequence_patterns(indices);
    }

    fn learn_stride_patterns(&mut self, indices: &[usize]) {
        if indices.len() < 2 {
            return;
        }

        for window in indices.windows(2) {
            if window[1] > window[0] {
                let stride = window[1] - window[0];
                *self.stride_patterns.entry(stride).or_insert(0) += 1;
            }
        }
    }

    fn learn_sequence_patterns(&mut self, indices: &[usize]) {
        if indices.len() < 3 {
            return;
        }

        for window in indices.windows(3) {
            let pattern = window.to_vec();
            *self.sequence_patterns.entry(pattern).or_insert(0) += 1;
        }
    }

    pub fn predict_next_accesses(
        &self,
        current_indices: &[usize],
        window_size: usize,
    ) -> Vec<usize> {
        let mut predictions = Vec::new();

        // 基于步长模式预测
        if let Some(stride_predictions) = self.predict_by_stride(current_indices, window_size) {
            predictions.extend(stride_predictions);
        }

        // 基于序列模式预测
        if let Some(sequence_predictions) = self.predict_by_sequence(current_indices, window_size) {
            predictions.extend(sequence_predictions);
        }

        // 去重并排序
        predictions.sort_unstable();
        predictions.dedup();
        predictions.truncate(window_size);

        predictions
    }

    fn predict_by_stride(&self, indices: &[usize], window_size: usize) -> Option<Vec<usize>> {
        if indices.len() < 2 {
            return None;
        }

        // 找到最常见的步长
        let mut stride_scores: Vec<(usize, usize)> = self
            .stride_patterns
            .iter()
            .map(|(&stride, &freq)| (stride, freq))
            .collect();
        stride_scores.sort_by(|a, b| b.1.cmp(&a.1));

        if let Some(&(best_stride, frequency)) = stride_scores.first() {
            let total_patterns = self.stride_patterns.values().sum::<usize>();
            let confidence = frequency as f64 / total_patterns as f64;

            if confidence >= self.min_confidence {
                let last_index = *indices.last().unwrap();
                let predictions: Vec<usize> = (1..=window_size)
                    .map(|i| last_index + i * best_stride)
                    .collect();
                return Some(predictions);
            }
        }

        None
    }

    fn predict_by_sequence(&self, indices: &[usize], window_size: usize) -> Option<Vec<usize>> {
        if indices.len() < 2 {
            return None;
        }

        // 查找匹配的序列模式
        let suffix = if indices.len() >= 3 {
            &indices[indices.len() - 3..]
        } else {
            &indices[indices.len() - 2..]
        };

        for (pattern, &frequency) in &self.sequence_patterns {
            if pattern.len() >= suffix.len() && pattern[..suffix.len()] == *suffix {
                let total_patterns = self.sequence_patterns.values().sum::<usize>();
                let confidence = frequency as f64 / total_patterns as f64;

                if confidence >= self.min_confidence {
                    // 基于找到的模式预测下一个访问
                    if let Some(&next_in_pattern) = pattern.get(suffix.len()) {
                        let last_index = *indices.last().unwrap();
                        let offset = next_in_pattern.saturating_sub(pattern[suffix.len() - 1]);
                        let predicted_start = last_index + offset;

                        let predictions: Vec<usize> =
                            (0..window_size).map(|i| predicted_start + i).collect();
                        return Some(predictions);
                    }
                }
            }
        }

        None
    }

    pub fn get_confidence(&self, indices: &[usize]) -> f64 {
        let stride_confidence = self.get_stride_confidence(indices);
        let sequence_confidence = self.get_sequence_confidence(indices);
        stride_confidence.max(sequence_confidence)
    }

    fn get_stride_confidence(&self, indices: &[usize]) -> f64 {
        if indices.len() < 2 {
            return 0.0;
        }

        let recent_stride = indices[indices.len() - 1] - indices[indices.len() - 2];
        let frequency = self.stride_patterns.get(&recent_stride).unwrap_or(&0);
        let total = self.stride_patterns.values().sum::<usize>();

        if total > 0 {
            *frequency as f64 / total as f64
        } else {
            0.0
        }
    }

    fn get_sequence_confidence(&self, indices: &[usize]) -> f64 {
        if indices.len() < 3 {
            return 0.0;
        }

        let recent_sequence = &indices[indices.len() - 3..];
        let frequency = self.sequence_patterns.get(recent_sequence).unwrap_or(&0);
        let total = self.sequence_patterns.values().sum::<usize>();

        if total > 0 {
            *frequency as f64 / total as f64
        } else {
            0.0
        }
    }
}
