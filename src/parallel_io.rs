use std::io::{self, BufWriter, Write};
use std::fs::{OpenOptions, File};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use rayon::prelude::*;
use memmap2::MmapOptions;
use ndarray::{ArrayD, ArrayViewD, IxDyn};
use numpy::Element;
use std::collections::{HashSet, VecDeque};
use std::os::unix::fs::FileExt;

use crate::error::{NpkResult, NpkError};
use crate::metadata::{CachedMetadataStore, ArrayMetadata, DataType};

const BUFFER_SIZE: usize = 8 * 1024 * 1024; // 8MB 缓冲区
const MAX_BUFFERS: usize = 4; // 最大缓冲区数量

// 添加缓冲区池结构
pub struct BufferPool {
    buffers: Mutex<VecDeque<Vec<u8>>>,
    buffer_size: usize,
    max_buffers: usize,
}

impl BufferPool {
    pub fn new(buffer_size: usize, max_buffers: usize) -> Self {
        Self {
            buffers: Mutex::new(VecDeque::with_capacity(max_buffers)),
            buffer_size,
            max_buffers,
        }
    }

    pub fn get_buffer(&self) -> Vec<u8> {
        let mut buffers = self.buffers.lock().unwrap();
        buffers.pop_front().unwrap_or_else(|| vec![0; self.buffer_size])
    }

    pub fn return_buffer(&self, mut buffer: Vec<u8>) {
        let mut buffers = self.buffers.lock().unwrap();
        if buffers.len() < self.max_buffers {
            buffer.clear();
            buffers.push_back(buffer);
        }
    }
}

lazy_static! {
    static ref BUFFER_POOL: Arc<BufferPool> = Arc::new(BufferPool::new(BUFFER_SIZE, MAX_BUFFERS));
}

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
    pub fn new(meta: ArrayMetadata, file: File, file_path: PathBuf) -> Self {
        Self {
            meta,
            file,
            file_path,
        }
    }

    fn get_retained_indices(&self, excluded_indices: Option<&[i64]>) -> Vec<usize> {
        let mut excluded_set = HashSet::new();
        let original_rows = self.meta.shape[0] as i64;
                
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
            return (0..self.meta.shape[0] as usize).collect();
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

    pub fn into_array<T: Element + Copy>(&mut self, excluded_indices: Option<&[i64]>) -> NpkResult<ArrayD<T>> {
        let element_size = std::mem::size_of::<T>();
        let shape: Vec<usize> = self.meta.shape.iter().map(|&x| x as usize).collect();
        let row_size = shape[1..].iter().product::<usize>() * element_size;
        
        if let Some(excluded) = excluded_indices {
            // Calculate rows to retain
            let retained = self.get_retained_indices(Some(excluded));
            let new_rows = retained.len();
            let mut new_shape = shape.clone();
            new_shape[0] = new_rows;
            
            // 使用内存映射读取源数据
            let mmap = unsafe { MmapOptions::new().map(&self.file)? };
            
            // 创建结果数组
            let mut result = unsafe { ArrayD::<T>::uninit(IxDyn(&new_shape)).assume_init() };
            let result_slice = unsafe { 
                std::slice::from_raw_parts_mut(
                    result.as_mut_ptr() as *mut u8,
                    new_rows * row_size
                )
            };
            
            // 使用缓冲区池进行分块复制
            let buffer = BUFFER_POOL.get_buffer();
            let chunk_size = buffer.len() / row_size * row_size;
            
            for chunk_start in (0..new_rows).step_by(chunk_size / row_size) {
                let chunk_end = std::cmp::min(chunk_start + chunk_size / row_size, new_rows);
                let chunk_size = (chunk_end - chunk_start) * row_size;
                
                // 复制数据到缓冲区
                for (i, &old_row) in retained[chunk_start..chunk_end].iter().enumerate() {
                    let src_offset = old_row * row_size;
                    let dst_offset = i * row_size;
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            mmap.as_ptr().add(src_offset),
                            buffer.as_ptr() as *mut u8,
                            row_size
                        );
                    }
                    result_slice[dst_offset..dst_offset + row_size]
                        .copy_from_slice(&buffer[..row_size]);
                }
            }
            
            // 返回缓冲区到池中
            BUFFER_POOL.return_buffer(buffer);
            
            Ok(result)
        } else {
            // 如果没有要排除的行，直接使用内存映射
            let mmap = unsafe { MmapOptions::new().map(&self.file)? };
            unsafe {
                Ok(ArrayViewD::from_shape_ptr(IxDyn(&shape), mmap.as_ptr() as *const T).to_owned())
            }
        }
    }

    pub fn physical_delete(&mut self, excluded_indices: &[i64]) -> NpkResult<()> {
        let element_size = self.meta.dtype.size_bytes() as usize;
        let shape: Vec<usize> = self.meta.shape.iter().map(|&x| x as usize).collect();
        let row_size = shape[1..].iter().product::<usize>() * element_size;
        let total_rows = shape[0];

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

        // 3. 创建临时文件
        let temp_path = self.file_path.with_extension("tmp");
        let temp_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&temp_path)?;

        // 预先分配目标文件大小
        temp_file.set_len(new_size as u64)?;

        const CHUNK_ROWS: usize = 1024 * 1024; // 每次处理100万行
        let chunks_count = (total_rows + CHUNK_ROWS - 1) / CHUNK_ROWS;
        
        // 4. 并行处理数据块
        let source_file = Arc::new(File::open(&self.file_path)?);
        let excluded_vec = Arc::new(excluded_vec);
        let temp_file = Arc::new(temp_file);
        
        let result: NpkResult<()> = (0..chunks_count).into_par_iter().try_for_each(|chunk_idx| {
            let start_row = chunk_idx * CHUNK_ROWS;
            let end_row = std::cmp::min(total_rows, (chunk_idx + 1) * CHUNK_ROWS);
            let chunk_row_count = end_row - start_row;
            
            // 为每个线程创建独立的缓冲区
            let read_size = chunk_row_count * row_size;
            let mut read_buffer = vec![0u8; read_size];
            let mut write_buffer = Vec::with_capacity(read_size);
            
            // 读取源文件块
            let read_offset = (start_row * row_size) as u64;
            source_file.read_exact_at(&mut read_buffer[..read_size], read_offset)?;
            
            // 处理当前块中的行
            for row in start_row..end_row {
                if !excluded_vec.binary_search(&row).is_ok() {
                    let row_start = (row - start_row) * row_size;
                    write_buffer.extend_from_slice(&read_buffer[row_start..row_start + row_size]);
                }
            }
            
            // 计算写入位置
            let mut write_row = 0;
            for i in 0..start_row {
                if !excluded_vec.binary_search(&i).is_ok() {
                    write_row += 1;
                }
            }
            
            // 写入临时文件
            let write_offset = (write_row * row_size) as u64;
            temp_file.write_all_at(&write_buffer, write_offset)?;
            
            Ok(())
        });

        // 检查并处理错误
        result?;

        // 5. 替换原文件
        std::fs::rename(&temp_path, &self.file_path)?;
        
        // 6. 更新元数据
        self.meta.shape[0] = new_rows as u64;
        self.meta.size_bytes = new_size as u64;
        self.meta.last_modified = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

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

    pub fn save_arrays<T: Element + Copy + Send + Sync>(&self, arrays: &[(String, ArrayD<T>, DataType)]) -> NpkResult<()> {
        // Parallel process array writing and collect metadata
        let metadata_updates: Vec<_> = arrays.par_iter()
            .map(|(name, array, dtype)| -> NpkResult<(String, ArrayMetadata)> {
                let data_file = format!("data_{}.npkd", name);
                let data_path = self.base_dir.join(&data_file);
                
                // Calculate total size
                let total_size = array.shape().iter().product::<usize>() * std::mem::size_of::<T>();
                
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
                    array.shape().iter().map(|&x| x as u64).collect(),
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

    fn process_batch<T: Element + Copy + Send + Sync>(
        mmap: &mut memmap2::MmapMut,
        data: &ArrayD<T>,
        batch: &[(usize, u64)],
        row_size: usize
    ) -> NpkResult<()> {
        if batch.is_empty() {
            return Ok(());
        }
        
        let mut buffer = BUFFER_POOL.get_buffer();
        
        // 如果是连续的索引，使用一次性写入
        if batch.len() > 1 && 
           batch.last().unwrap().1 - batch.first().unwrap().1 + 1 == batch.len() as u64 {
            let start_offset = (batch[0].1 as usize) * row_size;
            let total_size = batch.len() * row_size;
            
            // 确保缓冲区足够大
            if buffer.len() < total_size {
                buffer.resize(total_size, 0);
            }
            
            // 创建连续的数据缓冲区
            for (i, &(src_idx, _)) in batch.iter().enumerate() {
                let src_offset = (src_idx % data.shape()[0]) * row_size;
                let dst_offset = i * row_size;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        (data.as_ptr() as *const u8).add(src_offset),
                        buffer.as_mut_ptr().add(dst_offset),
                        row_size
                    );
                }
            }
            
            // 一次性写入
            mmap[start_offset..start_offset + total_size].copy_from_slice(&buffer[..total_size]);
        } else {
            // 对于不连续的索引，使用缓冲区进行分块写入
            for chunk in batch.chunks(buffer.len() / row_size) {
                for &(src_idx, dst_idx) in chunk {
                    let src_offset = (src_idx % data.shape()[0]) * row_size;
                    let dst_offset = (dst_idx as usize) * row_size;
                    
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            (data.as_ptr() as *const u8).add(src_offset),
                            buffer.as_mut_ptr(),
                            row_size
                        );
                    }
                    
                    mmap[dst_offset..dst_offset + row_size].copy_from_slice(&buffer[..row_size]);
                }
            }
        }
        
        // 返回缓冲区到池中
        BUFFER_POOL.return_buffer(buffer);
        Ok(())
    }

    pub fn replace_rows<T: Element + Copy + Send + Sync>(
        &self,
        name: &str,
        data: &ArrayD<T>,
        indices: &[i64]
    ) -> NpkResult<()> {
        let meta = self.get_array_meta(name).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, format!("Array {} not found", name))
        })?;
        
        let element_size = std::mem::size_of::<T>();
        // 计算每行的大小，对于一维数组，每行就是一个元素
        let row_size = if meta.shape.len() == 1 {
            element_size
        } else {
            meta.shape[1..].iter().product::<u64>() as usize * element_size
        };
        
        // Open file for writing
        let file_path = self.base_dir.join(&meta.data_file);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&file_path)?;
            
        // Check if indices are continuous
        if let Some((start, len)) = Self::is_continuous_indices(indices) {
            let normalized_start = if start < 0 {
                (meta.shape[0] as i64 + start) as u64
            } else {
                start as u64
            };
            
            if normalized_start + len as u64 > meta.shape[0] {
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
            let groups = Self::group_indices(indices, row_size, meta.shape[0]);
            
            // Process each group in parallel
            groups.par_iter().try_for_each(|group| -> NpkResult<()> {
                if group.is_empty() {
                    return Ok(());
                }
                
                // Calculate block range
                let first_idx = if group[0].1 < 0 {
                    meta.shape[0] as i64 + group[0].1
                } else {
                    group[0].1
                } as u64;
                
                let last_idx = if group[group.len() - 1].1 < 0 {
                    meta.shape[0] as i64 + group[group.len() - 1].1
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
                        (meta.shape[0] as i64 + file_idx) as u64
                    } else {
                        file_idx as u64
                    };
                    
                    if normalized_idx >= meta.shape[0] {
                        return Err(NpkError::IoError(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            format!("Index {} is out of bounds", file_idx)
                        )));
                    }
                    
                    let offset = (normalized_idx - first_idx) as usize * row_size;
                    unsafe {
                        let src_ptr = data.as_ptr().add(data_idx * row_size) as *const u8;
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
        
        let shape: Vec<usize> = meta.shape.iter().map(|&x| x as usize).collect();
        let row_size = shape[1..].iter().product::<usize>() * meta.dtype.size_bytes() as usize;
        
        // Validate all indices
        for &idx in indexes {
            if idx < 0 || idx >= meta.shape[0] as i64 {
                return Err(NpkError::IndexOutOfBounds(idx, meta.shape[0]));
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
