use crate::core::error::{NpkError, NpkResult};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// DeletionBitmap 管理数组的逻辑删除标记
/// 使用位图（bitmap）来标记哪些行已被删除
/// 位值: 0 = 已删除, 1 = 活跃（未删除）
#[derive(Debug, Clone)]
pub struct DeletionBitmap {
    /// 位图数据，以64位字（u64）存储
    /// 每个bit代表一行，0表示删除，1表示活跃
    words: Vec<u64>,
    /// 总行数
    total_rows: usize,
    /// 已删除的行数
    deleted_count: usize,
    /// bitmap文件路径
    file_path: PathBuf,
}

impl DeletionBitmap {
    /// 创建新的bitmap或加载已存在的bitmap
    pub fn new(base_dir: &Path, array_name: &str, total_rows: usize) -> NpkResult<Self> {
        let file_path = Self::get_bitmap_path(base_dir, array_name);

        if file_path.exists() {
            // 加载已存在的bitmap
            Self::load(&file_path, total_rows)
        } else {
            // 创建新的bitmap，所有行初始为活跃状态
            let num_words = (total_rows + 63) / 64;
            let mut words = vec![u64::MAX; num_words];

            // 处理最后一个字，如果总行数不是64的倍数
            if total_rows % 64 != 0 {
                let last_word_bits = total_rows % 64;
                let mask = (1u64 << last_word_bits) - 1;
                words[num_words - 1] = mask;
            }

            Ok(Self {
                words,
                total_rows,
                deleted_count: 0,
                file_path,
            })
        }
    }

    /// 获取bitmap文件路径
    pub fn get_bitmap_path(base_dir: &Path, array_name: &str) -> PathBuf {
        base_dir.join(format!("deleted_{}.npkb", array_name))
    }

    /// 从文件加载bitmap
    /// 文件格式：[total_rows (u64), deleted_count (u64), words...]
    /// 如果文件格式是旧的（没有头部），则使用expected_rows
    fn load(path: &Path, expected_rows: usize) -> NpkResult<Self> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        // 检查文件大小
        if buffer.len() < 16 {
            return Err(NpkError::InvalidMetadata(format!(
                "Bitmap file too small: {}",
                buffer.len()
            )));
        }

        // 尝试读取头部
        let stored_total_rows = u64::from_le_bytes(buffer[0..8].try_into().unwrap()) as usize;
        let stored_deleted_count = u64::from_le_bytes(buffer[8..16].try_into().unwrap()) as usize;

        // 检查是否是新格式（有头部）
        let expected_num_words = (stored_total_rows + 63) / 64;
        let expected_file_size = 16 + expected_num_words * 8;

        let (total_rows, deleted_count, words_start) = if buffer.len() == expected_file_size {
            // 新格式：有头部
            (stored_total_rows, stored_deleted_count, 16)
        } else {
            // 旧格式：没有头部，整个文件都是words
            if buffer.len() % 8 != 0 {
                return Err(NpkError::InvalidMetadata(format!(
                    "Invalid bitmap file size: {}",
                    buffer.len()
                )));
            }
            (expected_rows, 0, 0) // 延迟计算deleted_count
        };

        // 读取words
        let num_words = (buffer.len() - words_start) / 8;
        let mut words = Vec::with_capacity(num_words);

        for i in 0..num_words {
            let offset = words_start + i * 8;
            let word_bytes = &buffer[offset..offset + 8];
            let word = u64::from_le_bytes(word_bytes.try_into().unwrap());
            words.push(word);
        }

        // 如果是旧格式，计算deleted_count
        let deleted_count = if words_start == 0 {
            let mut count = 0;
            for row in 0..total_rows {
                if !Self::is_active_static(&words, row) {
                    count += 1;
                }
            }
            count
        } else {
            deleted_count
        };

        Ok(Self {
            words,
            total_rows,
            deleted_count,
            file_path: path.to_path_buf(),
        })
    }

    /// 保存bitmap到文件
    /// 文件格式：[total_rows (u64), deleted_count (u64), words...]
    pub fn save(&self) -> NpkResult<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&self.file_path)?;

        // 先写入头部：total_rows和deleted_count
        let mut buffer = Vec::with_capacity(16 + self.words.len() * 8);
        buffer.extend_from_slice(&(self.total_rows as u64).to_le_bytes());
        buffer.extend_from_slice(&(self.deleted_count as u64).to_le_bytes());

        // 然后写入words
        for &word in &self.words {
            buffer.extend_from_slice(&word.to_le_bytes());
        }

        file.write_all(&buffer)?;
        file.sync_all()?;

        Ok(())
    }

    /// 删除bitmap文件（在update操作后调用）
    pub fn delete_file(&self) -> NpkResult<()> {
        if self.file_path.exists() {
            std::fs::remove_file(&self.file_path)?;
        }
        Ok(())
    }

    /// 检查某一行是否为活跃状态（未删除）
    pub fn is_active(&self, row: usize) -> bool {
        if row >= self.total_rows {
            return false;
        }
        Self::is_active_static(&self.words, row)
    }

    fn is_active_static(words: &[u64], row: usize) -> bool {
        let word_idx = row / 64;
        let bit_idx = row % 64;
        if word_idx >= words.len() {
            return false;
        }
        (words[word_idx] >> bit_idx) & 1 == 1
    }

    /// 标记某一行为已删除
    pub fn mark_deleted(&mut self, row: usize) -> NpkResult<()> {
        if row >= self.total_rows {
            return Err(NpkError::IndexOutOfBounds(
                row as i64,
                self.total_rows as u64,
            ));
        }

        if self.is_active(row) {
            let word_idx = row / 64;
            let bit_idx = row % 64;
            self.words[word_idx] &= !(1u64 << bit_idx);
            self.deleted_count += 1;
        }

        Ok(())
    }

    /// 批量标记行为已删除
    pub fn mark_deleted_batch(&mut self, rows: &[usize]) -> NpkResult<()> {
        for &row in rows {
            self.mark_deleted(row)?;
        }
        Ok(())
    }

    /// 获取活跃行数（未删除的行数）
    pub fn active_count(&self) -> usize {
        self.total_rows - self.deleted_count
    }

    /// 获取已删除行数
    pub fn deleted_count(&self) -> usize {
        self.deleted_count
    }

    /// 获取总行数（物理行数，包括已删除的行）
    pub fn get_total_rows(&self) -> usize {
        self.total_rows
    }

    /// 将逻辑索引（用户视角）转换为物理索引（实际存储）
    /// 逻辑索引跳过已删除的行
    pub fn logical_to_physical(&self, logical_idx: usize) -> Option<usize> {
        if logical_idx >= self.active_count() {
            return None;
        }

        let mut active_count = 0;
        for physical_idx in 0..self.total_rows {
            if self.is_active(physical_idx) {
                if active_count == logical_idx {
                    return Some(physical_idx);
                }
                active_count += 1;
            }
        }

        None
    }

    /// 将物理索引转换为逻辑索引
    pub fn physical_to_logical(&self, physical_idx: usize) -> Option<usize> {
        if physical_idx >= self.total_rows || !self.is_active(physical_idx) {
            return None;
        }

        let mut logical_idx = 0;
        for i in 0..physical_idx {
            if self.is_active(i) {
                logical_idx += 1;
            }
        }

        Some(logical_idx)
    }

    /// 将逻辑索引数组转换为物理索引数组
    pub fn logical_to_physical_batch(&self, logical_indices: &[usize]) -> NpkResult<Vec<usize>> {
        let mut physical_indices = Vec::with_capacity(logical_indices.len());
        for &logical_idx in logical_indices {
            if let Some(physical_idx) = self.logical_to_physical(logical_idx) {
                physical_indices.push(physical_idx);
            } else {
                return Err(NpkError::IndexOutOfBounds(
                    logical_idx as i64,
                    self.active_count() as u64,
                ));
            }
        }
        Ok(physical_indices)
    }

    /// 获取所有活跃行的物理索引
    pub fn get_active_indices(&self) -> Vec<usize> {
        let mut indices = Vec::with_capacity(self.active_count());
        for i in 0..self.total_rows {
            if self.is_active(i) {
                indices.push(i);
            }
        }
        indices
    }

    /// 重置bitmap（所有行标记为活跃）
    pub fn reset(&mut self) {
        let num_words = (self.total_rows + 63) / 64;
        self.words = vec![u64::MAX; num_words];

        // 处理最后一个字
        if self.total_rows % 64 != 0 {
            let last_word_bits = self.total_rows % 64;
            let mask = (1u64 << last_word_bits) - 1;
            self.words[num_words - 1] = mask;
        }

        self.deleted_count = 0;
    }

    /// 检查bitmap文件是否存在
    pub fn exists(base_dir: &Path, array_name: &str) -> bool {
        Self::get_bitmap_path(base_dir, array_name).exists()
    }

    /// 更新总行数（在append操作后调用）
    pub fn extend(&mut self, additional_rows: usize) {
        let old_total = self.total_rows;
        self.total_rows += additional_rows;

        // 计算需要的字数
        let new_num_words = (self.total_rows + 63) / 64;
        let old_num_words = (old_total + 63) / 64;

        // 处理旧的最后一个字（如果还有剩余位）
        if old_total % 64 != 0 {
            let old_word_idx = old_num_words - 1;
            let old_bits = old_total % 64;
            // 填充旧最后一个字的剩余位
            let fill_mask = !((1u64 << old_bits) - 1);
            self.words[old_word_idx] |= fill_mask;
        }

        // 添加新的完整字（全部设为活跃）
        if new_num_words > old_num_words {
            self.words.resize(new_num_words, u64::MAX);
        }

        // 处理新的最后一个字的掩码（如果total_rows不是64的倍数）
        if self.total_rows % 64 != 0 {
            let last_word_idx = new_num_words - 1;
            let last_word_bits = self.total_rows % 64;
            let mask = (1u64 << last_word_bits) - 1;
            self.words[last_word_idx] = mask;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_bitmap_creation() {
        let temp_dir = TempDir::new().unwrap();
        let bitmap = DeletionBitmap::new(temp_dir.path(), "test_array", 100).unwrap();

        assert_eq!(bitmap.active_count(), 100);
        assert_eq!(bitmap.deleted_count(), 0);

        for i in 0..100 {
            assert!(bitmap.is_active(i));
        }
    }

    #[test]
    fn test_mark_deleted() {
        let temp_dir = TempDir::new().unwrap();
        let mut bitmap = DeletionBitmap::new(temp_dir.path(), "test_array", 100).unwrap();

        bitmap.mark_deleted(10).unwrap();
        bitmap.mark_deleted(20).unwrap();
        bitmap.mark_deleted(30).unwrap();

        assert_eq!(bitmap.active_count(), 97);
        assert_eq!(bitmap.deleted_count(), 3);
        assert!(!bitmap.is_active(10));
        assert!(!bitmap.is_active(20));
        assert!(!bitmap.is_active(30));
        assert!(bitmap.is_active(0));
        assert!(bitmap.is_active(99));
    }

    #[test]
    fn test_logical_to_physical() {
        let temp_dir = TempDir::new().unwrap();
        let mut bitmap = DeletionBitmap::new(temp_dir.path(), "test_array", 10).unwrap();

        // 删除索引 2, 5, 7
        bitmap.mark_deleted(2).unwrap();
        bitmap.mark_deleted(5).unwrap();
        bitmap.mark_deleted(7).unwrap();

        // 活跃的物理索引: 0, 1, 3, 4, 6, 8, 9
        // 对应的逻辑索引: 0, 1, 2, 3, 4, 5, 6
        assert_eq!(bitmap.logical_to_physical(0), Some(0));
        assert_eq!(bitmap.logical_to_physical(1), Some(1));
        assert_eq!(bitmap.logical_to_physical(2), Some(3));
        assert_eq!(bitmap.logical_to_physical(3), Some(4));
        assert_eq!(bitmap.logical_to_physical(4), Some(6));
        assert_eq!(bitmap.logical_to_physical(5), Some(8));
        assert_eq!(bitmap.logical_to_physical(6), Some(9));
        assert_eq!(bitmap.logical_to_physical(7), None);
    }

    #[test]
    fn test_physical_to_logical() {
        let temp_dir = TempDir::new().unwrap();
        let mut bitmap = DeletionBitmap::new(temp_dir.path(), "test_array", 10).unwrap();

        bitmap.mark_deleted(2).unwrap();
        bitmap.mark_deleted(5).unwrap();
        bitmap.mark_deleted(7).unwrap();

        assert_eq!(bitmap.physical_to_logical(0), Some(0));
        assert_eq!(bitmap.physical_to_logical(1), Some(1));
        assert_eq!(bitmap.physical_to_logical(2), None); // 已删除
        assert_eq!(bitmap.physical_to_logical(3), Some(2));
        assert_eq!(bitmap.physical_to_logical(4), Some(3));
        assert_eq!(bitmap.physical_to_logical(5), None); // 已删除
        assert_eq!(bitmap.physical_to_logical(6), Some(4));
        assert_eq!(bitmap.physical_to_logical(7), None); // 已删除
        assert_eq!(bitmap.physical_to_logical(8), Some(5));
        assert_eq!(bitmap.physical_to_logical(9), Some(6));
    }

    #[test]
    fn test_save_and_load() {
        let temp_dir = TempDir::new().unwrap();
        let mut bitmap = DeletionBitmap::new(temp_dir.path(), "test_array", 100).unwrap();

        bitmap.mark_deleted(10).unwrap();
        bitmap.mark_deleted(20).unwrap();
        bitmap.mark_deleted(30).unwrap();
        bitmap.save().unwrap();

        // 重新加载
        let loaded_bitmap = DeletionBitmap::load(
            &DeletionBitmap::get_bitmap_path(temp_dir.path(), "test_array"),
            100,
        )
        .unwrap();

        assert_eq!(loaded_bitmap.active_count(), 97);
        assert_eq!(loaded_bitmap.deleted_count(), 3);
        assert!(!loaded_bitmap.is_active(10));
        assert!(!loaded_bitmap.is_active(20));
        assert!(!loaded_bitmap.is_active(30));
    }

    #[test]
    fn test_extend() {
        let temp_dir = TempDir::new().unwrap();
        let mut bitmap = DeletionBitmap::new(temp_dir.path(), "test_array", 100).unwrap();

        bitmap.mark_deleted(10).unwrap();
        assert_eq!(bitmap.active_count(), 99);

        // 扩展50行
        bitmap.extend(50);
        assert_eq!(bitmap.total_rows, 150);
        assert_eq!(bitmap.active_count(), 149); // 99 + 50

        // 新增的行应该是活跃的
        for i in 100..150 {
            assert!(bitmap.is_active(i));
        }

        // 原来删除的行仍然是删除状态
        assert!(!bitmap.is_active(10));
    }
}
