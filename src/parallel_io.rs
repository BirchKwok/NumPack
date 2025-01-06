use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;
use memmap2::MmapOptions;
use rayon::prelude::*;
use std::sync::Arc;
use ndarray::{Array2, ArrayView2};
use numpy::Element;
use std::collections::HashSet;
use std::os::unix::fs::FileExt;

use crate::error::{NpkResult, NpkError};
use crate::metadata::{CachedMetadataStore, ArrayMetadata, DataType};


#[allow(dead_code)]
pub struct ArrayView {
    pub meta: ArrayMetadata,
    file: File,
    file_path: PathBuf,
}

fn normalize_index(idx: i64, total_rows: usize) -> Option<usize> {
    let normalized = if idx < 0 {
        total_rows as i64 + idx // 若 idx = -1，则表示最后一行
    } else {
        idx
    };
    // 排除越界索引
    if normalized >= 0 && normalized < total_rows as i64 {
        Some(normalized as usize)
    } else {
        None
    }
}

impl ArrayView {
    fn new(meta: ArrayMetadata, file: File, file_path: PathBuf) -> Self {
        Self {
            meta,
            file,
            file_path,
        }
    }

    fn get_retained_indices(&self, excluded_indices: Option<&[i64]>) -> Vec<usize> {
        let mut excluded_set = HashSet::new();
        let original_rows = self.meta.rows as i64;
                
        if let Some(excluded) = excluded_indices {
            for &idx in excluded {
                let normalized_idx = if idx < 0 { original_rows + idx } else { idx };
                if normalized_idx >= 0 && normalized_idx < original_rows {
                    excluded_set.insert(normalized_idx);
                }
            }
        }

        // If there are no indices to exclude, return all rows
        if excluded_set.is_empty() {
            return (0..self.meta.rows as usize).collect();
        }
        
        // Create a sorted list of excluded indices
        let mut excluded_vec: Vec<i64> = excluded_set.into_iter().collect();
        excluded_vec.sort_unstable();
        
        // Calculate retained indices
        let mut retained = Vec::with_capacity((original_rows - excluded_vec.len() as i64) as usize);
        
        // Iterate over all rows, keeping rows not in the exclusion list
        for i in 0..original_rows {
            if !excluded_vec.contains(&i) {
                retained.push(i as usize);
            }
        }
        
        retained
    }

    pub fn into_array<T: Element + Copy>(&mut self, excluded_indices: Option<&[i64]>) -> NpkResult<Array2<T>> {
        let element_size = std::mem::size_of::<T>();
        let row_size = (self.meta.cols as usize) * element_size;
        
        // Use memory mapping
        let mmap = unsafe { MmapOptions::new().map(&self.file)? };
        
        if let Some(excluded) = excluded_indices {
            // Calculate rows to retain
            let retained = self.get_retained_indices(Some(excluded));
            let new_rows = retained.len();
            let cols = self.meta.cols as usize;
            
            // Create new array
            let mut raw_data = vec![0u8; new_rows * cols * element_size];
            
            // Copy needed rows
            for (new_row, &old_row) in retained.iter().enumerate() {
                let src_offset = old_row * row_size;
                let dst_offset = new_row * row_size;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        mmap.as_ptr().add(src_offset),
                        raw_data.as_mut_ptr().add(dst_offset),
                        row_size
                    );
                }
            }
            
            // Convert to final array
            let array = unsafe {
                Array2::from_shape_vec_unchecked(
                    (new_rows, cols),
                    std::slice::from_raw_parts(
                        raw_data.as_ptr() as *const T,
                        new_rows * cols
                    ).to_vec()
                )
            };
            Ok(array)
        } else {
            // If there are no rows to exclude, return the entire array
            let ptr = mmap.as_ptr() as *const T;
            let shape = (self.meta.rows as usize, self.meta.cols as usize);
            unsafe {
                Ok(ArrayView2::from_shape_ptr(shape, ptr).to_owned())
            }
        }
    }

    pub fn physical_delete(&mut self, excluded_indices: &[i64]) -> NpkResult<()> {
        let element_size = self.meta.dtype.size_bytes() as usize;
        let row_size = (self.meta.cols as usize) * element_size;
        let total_rows = self.meta.rows as usize;
    
        // 1. 收集需要排除的行索引（去重 + 排序）
        let mut excluded_vec: Vec<usize> = excluded_indices
            .iter()
            .filter_map(|&idx| normalize_index(idx, total_rows))
            .collect();
        excluded_vec.sort_unstable();
        excluded_vec.dedup();
    
        // 如果没有任何行需要排除，直接返回
        if excluded_vec.is_empty() {
            return Ok(());
        }
    
        // 2. 计算删除后文件大小
        let new_rows = total_rows - excluded_vec.len();
        let new_size = new_rows * row_size;
    
        // 3. 创建临时文件，用于写入删除后的数据
        let temp_path = self.file_path.with_extension("tmp");
        let temp_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&temp_path)?;
    
        // 预先分配目标文件大小
        temp_file.set_len(new_size as u64)?;
    
        // 4. 映射源文件到内存（只读）
        let src_mmap = unsafe { MmapOptions::new().map(&self.file)? };
    
        // 使用较大的写缓冲，避免频繁 I/O 调用
        let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, temp_file);
    
        // 5. 将排除行索引合并成「连续区间」列表
        //    例如：excluded_vec = [2, 3, 4, 10, 11, 20]
        //    则分块 [(2,4), (10,11), (20,20)]
        let mut chunks = Vec::new();
        let mut chunk_start = excluded_vec[0];
        let mut chunk_end = chunk_start;
    
        for &idx in excluded_vec.iter().skip(1) {
            if idx == chunk_end + 1 {
                // 仍在同一个连续区间
                chunk_end = idx;
            } else {
                // 出现不连续，先把前一个区间保存
                chunks.push((chunk_start, chunk_end));
                // 开启新的区间
                chunk_start = idx;
                chunk_end = idx;
            }
        }
        // 别忘了将最后一个区间 push 进去
        chunks.push((chunk_start, chunk_end));
    
        // 6. 依次把「非排除」的块复制到临时文件
        let mut current_pos = 0; // 当前处理到哪一行了
        for &(start, end) in &chunks {
            if start > current_pos {
                let begin_offset = current_pos * row_size;
                let end_offset = start * row_size;
                writer.write_all(&src_mmap[begin_offset..end_offset])?;
            }
            // 跳过 [start..=end] 这个区间
            current_pos = end + 1;
        }
    
        // 如果还有剩余行在最后
        if current_pos < total_rows {
            let begin_offset = current_pos * row_size;
            let end_offset = total_rows * row_size;
            writer.write_all(&src_mmap[begin_offset..end_offset])?;
        }
    
        // 刷新缓冲区并关闭映射
        writer.flush()?;
        drop(writer);
        drop(src_mmap);
    
        // 7. 原子替换旧文件
        std::fs::rename(&temp_path, &self.file_path)?;
    
        // 8. 重新打开文件，更新元信息
        self.file = File::open(&self.file_path)?;
        self.meta.rows = new_rows as u64;
        self.meta.size_bytes = new_size as u64;
    
        Ok(())
    }    
}

#[allow(dead_code)]
pub struct ParallelIO {
    base_dir: PathBuf,
    metadata: Arc<CachedMetadataStore>,
    metadata_path: PathBuf,
}

impl ParallelIO {
    pub fn new(base_dir: PathBuf) -> NpkResult<Self> {
        let metadata_path = base_dir.join("metadata.npkm");
        let wal_path = Some(base_dir.join("metadata.wal"));
        let metadata = CachedMetadataStore::new(&metadata_path, wal_path)?;
        
        Ok(Self {
            base_dir,
            metadata: Arc::new(metadata),
            metadata_path,
        })
    }

    const WRITE_CHUNK_SIZE: usize = 8 * 1024 * 1024;  // 8MB write chunk size

    pub fn save_arrays<T: Element + Copy + Send + Sync>(&self, arrays: &[(String, Array2<T>, DataType)]) -> NpkResult<()> {
        // Parallel process array writing and collect metadata
        let metadata_updates: Vec<_> = arrays.par_iter()
            .map(|(name, array, dtype)| -> NpkResult<(String, ArrayMetadata)> {
                let data_file = format!("data_{}.npkd", name);
                let data_path = self.base_dir.join(&data_file);
                
                // Calculate total size
                let total_size = array.shape()[0] * array.shape()[1] * std::mem::size_of::<T>();
                
                // Create and preallocate file
                let file = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create(true)
                    .open(&data_path)?;
                file.set_len(total_size as u64)?;
                
                // Use direct write instead of memory mapping
                let mut writer = BufWriter::with_capacity(Self::WRITE_CHUNK_SIZE, &file);
                
                // Get data pointer
                let data_ptr = array.as_ptr() as *const u8;
                let mut offset = 0;
                
                // Write data in chunks
                while offset < total_size {
                    let chunk_size = std::cmp::min(Self::WRITE_CHUNK_SIZE, total_size - offset);
                    let chunk = unsafe {
                        std::slice::from_raw_parts(data_ptr.add(offset), chunk_size)
                    };
                    
                    writer.write_all(chunk)?;
                    
                    offset += chunk_size;
                }
                
                // Ensure data is written to disk
                writer.flush()?;
                
                // Create metadata
                let meta = ArrayMetadata::new(
                    name.clone(),
                    array.shape()[0] as u64,
                    array.shape()[1] as u64,
                    data_file,
                    *dtype,
                );
                
                Ok((name.clone(), meta))
            })
            .collect::<Result<Vec<_>, _>>()?;
        
        // Batch update metadata
        for (_name, meta) in metadata_updates {
            self.metadata.add_array(meta)?;
        }
        
        Ok(())
    }

    pub fn get_array_views(&self, names: Option<&[String]>) -> NpkResult<Vec<(String, ArrayView)>> {
        let arrays_to_load: Vec<_> = if let Some(names) = names {
            names.iter()
                .filter_map(|name| {
                    self.metadata.get_array(name)
                        .map(|meta| (name.clone(), meta))
                })
                .collect()
        } else {
            self.metadata.list_arrays()
                .into_iter()
                .filter_map(|name| {
                    self.metadata.get_array(&name)
                        .map(|meta| (name.clone(), meta))
                })
                .collect()
        };

        // Parallel create views
        arrays_to_load.into_par_iter()
            .map(|(name, meta)| {
                let data_path = self.base_dir.join(&meta.data_file);
                let file = File::open(&data_path)?;
                let view = ArrayView::new(meta, file, data_path.clone());
                Ok((name, view))
            })
            .collect()
    }

    pub fn get_array_meta(&self, name: &str) -> Option<ArrayMetadata> {
        self.metadata.get_array(name)
    }

    pub fn get_array_metadata(&self, name: &str) -> Result<ArrayMetadata, NpkError> {
        self.metadata.get_array(name)
            .ok_or_else(|| NpkError::ArrayNotFound(name.to_string()))
    }

    pub fn list_arrays(&self) -> Vec<String> {
        self.metadata.list_arrays()
    }

    pub fn reset(&self) -> NpkResult<()> {
        self.metadata.reset()?;
        // Delete array files
        for entry in std::fs::read_dir(&self.base_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "npkd") {
                std::fs::remove_file(path)?;
            }
        }
        Ok(())
    }

    pub fn update_array_metadata(&self, name: &str, meta: ArrayMetadata) -> NpkResult<()> {
        self.metadata.update_array_metadata(name, meta)
    }

    pub fn has_array(&self, name: &str) -> bool {
        self.metadata.has_array(name)
    }

    pub fn drop_arrays(&self, name: &str, excluded_indices: Option<&[i64]>) -> NpkResult<()> {
        if let Some(meta) = self.metadata.get_array(name) {
            let data_path = self.base_dir.join(&meta.data_file);
            let file = File::open(&data_path)?;
            let mut view = ArrayView::new(meta, file, data_path.clone());
            
            if let Some(indices) = excluded_indices {
                // Physical delete specified rows
                view.physical_delete(indices)?;
                // Update metadata
                self.metadata.update_array_metadata(name, view.meta)?;
            } else {
                // If no specified row indices, delete the entire array
                std::fs::remove_file(&data_path)?;
                // Delete array from metadata
                self.metadata.delete_array(name)?;
            }
        }
        Ok(())
    }

    pub fn batch_drop_arrays(&self, names: &[String], excluded_indices: Option<&[i64]>) -> NpkResult<()> {
        for name in names {
            self.drop_arrays(name, excluded_indices)?;
        }
        Ok(())
    }

    // Check if indices are continuous
    fn is_continuous_indices(indices: &[i64]) -> Option<(i64, usize)> {
        if indices.is_empty() {
            return None;
        }
        
        let start = indices[0];
        let len = indices.len();
        
        for (i, &idx) in indices.iter().enumerate() {
            if idx != start + i as i64 {
                return None;
            }
        }
        
        Some((start, len))
    }

    const REPLACE_CHUNK_SIZE: usize = 4096;  // Process 4096 rows per batch
    const BLOCK_SIZE: usize = 8 * 1024 * 1024;  // 8MB block size

    // Group indices by blocks to improve locality
    fn group_indices(indices: &[i64], row_size: usize, rows: u64) -> Vec<Vec<(usize, i64)>> {
        let mut indexed: Vec<_> = indices.iter()
            .enumerate()
            .map(|(i, &idx)| {
                let normalized_idx = if idx < 0 {
                    (rows as i64 + idx) as u64
                } else {
                    idx as u64
                };
                (i, normalized_idx)
            })
            .collect();

        // Sort by file offset
        indexed.sort_by_key(|&(_, idx)| idx);

        // Calculate block size in rows
        let rows_per_block = Self::BLOCK_SIZE / row_size;
        let mut groups = Vec::new();
        let mut current_group = Vec::new();
        let mut current_block = None;

        for (i, idx) in indexed {
            let block_idx = idx as usize / rows_per_block;
            
            match current_block {
                Some(block) => {
                    if block_idx != block || current_group.len() >= Self::REPLACE_CHUNK_SIZE {
                        if !current_group.is_empty() {
                            groups.push(std::mem::take(&mut current_group));
                        }
                        current_block = Some(block_idx);
                    }
                }
                None => {
                    current_block = Some(block_idx);
                }
            }
            current_group.push((i, idx as i64));
        }

        if !current_group.is_empty() {
            groups.push(current_group);
        }

        groups
    }

    pub fn replace_rows<T: Element + Copy + Send + Sync>(
        &self,
        name: &str,
        data: &Array2<T>,
        indices: &[i64]
    ) -> NpkResult<()> {
        let meta = self.get_array_meta(name).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, format!("Array {} not found", name))
        })?;
        
        let element_size = std::mem::size_of::<T>();
        let row_size = meta.cols as usize * element_size;
        
        // Open file for writing
        let file_path = self.base_dir.join(&meta.data_file);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&file_path)?;
            
        // Check if indices are continuous
        if let Some((start, len)) = Self::is_continuous_indices(indices) {
            let normalized_start = if start < 0 {
                (meta.rows as i64 + start) as u64
            } else {
                start as u64
            };
            
            if normalized_start + len as u64 > meta.rows {
                return Err(NpkError::IoError(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("Index range {}:{} is out of bounds", start, start + len as i64)
                )));
            }
            
            // For continuous indices, use one-time write
            let offset = normalized_start * row_size as u64;
            let data_slice = unsafe {
                std::slice::from_raw_parts(
                    data.as_ptr() as *const u8,
                    len * row_size
                )
            };
            file.write_at(data_slice, offset)?;
        } else {
            // For non-continuous indices, group them by blocks and process in parallel
            let groups = Self::group_indices(indices, row_size, meta.rows);
            
            // Process each group in parallel
            groups.par_iter().try_for_each(|group| -> NpkResult<()> {
                if group.is_empty() {
                    return Ok(());
                }
                
                // Calculate block range
                let first_idx = if group[0].1 < 0 {
                    meta.rows as i64 + group[0].1
                } else {
                    group[0].1
                } as u64;
                
                let last_idx = if group[group.len() - 1].1 < 0 {
                    meta.rows as i64 + group[group.len() - 1].1
                } else {
                    group[group.len() - 1].1
                } as u64;
                
                // Read the entire block
                let block_size = (last_idx - first_idx + 1) as usize * row_size;
                let mut block_buffer = vec![0u8; block_size];
                file.read_at(&mut block_buffer, first_idx * row_size as u64)?;
                
                // Update rows in the block
                for &(data_idx, file_idx) in group {
                    let normalized_idx = if file_idx < 0 {
                        (meta.rows as i64 + file_idx) as u64
                    } else {
                        file_idx as u64
                    };
                    
                    if normalized_idx >= meta.rows {
                        return Err(NpkError::IoError(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            format!("Index {} is out of bounds", file_idx)
                        )));
                    }
                    
                    let offset = (normalized_idx - first_idx) as usize * row_size;
                    unsafe {
                        let src_ptr = data.row(data_idx).as_ptr() as *const u8;
                        let dst_ptr = block_buffer.as_mut_ptr().add(offset);
                        std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, row_size);
                    }
                }
                
                // Write back the entire block
                file.write_at(&block_buffer, first_idx * row_size as u64)?;
                
                Ok(())
            })?;
        }
        
        Ok(())
    }

    pub fn read_rows(&self, name: &str, indexes: &[i64]) -> Result<Vec<u8>, NpkError> {
        let meta = self.get_array_metadata(name)?;
        let data_path = self.base_dir.join(format!("data_{}.npkd", name));
        
        let row_size = usize::try_from(meta.cols)
            .map_err(|e| NpkError::Other(e.to_string()))? * 
            usize::try_from(meta.dtype.size_bytes())
            .map_err(|e| NpkError::Other(e.to_string()))?;
        
        // Validate all indices
        for &idx in indexes {
            if idx < 0 || idx >= meta.rows as i64 {
                return Err(NpkError::IndexOutOfBounds(idx, meta.rows));
            }
        }

        // Open file and create memory mapping
        let file = std::fs::File::open(&data_path)?;
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
        let mut data = Vec::with_capacity(indexes.len() * row_size);

        // Copy data from memory mapping directly
        for &idx in indexes {
            let offset = (idx as usize) * row_size;
            let row_slice = &mmap[offset..offset + row_size];
            data.extend_from_slice(row_slice);
        }

        Ok(data)
    }
}


