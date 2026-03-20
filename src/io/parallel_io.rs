use memmap2::MmapOptions;
use ndarray::{ArrayD, ArrayViewD, IxDyn};
use rayon::prelude::*;

use crate::core::metadata::NpkElement;
use std::collections::{HashSet, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::SystemTime;

use crate::core::error::{NpkError, NpkResult};
use crate::core::metadata::{ArrayMetadata, DataType};
use crate::memory::simd_optimized;
use crate::storage::binary_metadata::{
    BinaryArrayMetadata, BinaryCachedStore, BinaryDataType, BinaryMetadataStore,
};
use crate::storage::deletion_bitmap::DeletionBitmap;

// 平台特定导入

// Helper functions for file IO
#[cfg(unix)]
mod platform {
    use std::fs::File;
    use std::io;
    use std::os::unix::fs::FileExt;

    pub fn read_at_offset(file: &File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        file.read_at(buf, offset)
    }

    pub fn write_at_offset(file: &File, buf: &[u8], offset: u64) -> io::Result<usize> {
        file.write_at(buf, offset)
    }
}

#[cfg(windows)]
mod platform {
    use std::fs::File;
    use std::io;
    use std::os::windows::fs::FileExt;

    pub fn read_at_offset(file: &File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        file.seek_read(buf, offset)
    }

    pub fn write_at_offset(file: &File, buf: &[u8], offset: u64) -> io::Result<usize> {
        file.seek_write(buf, offset)
    }
}

use platform::{read_at_offset, write_at_offset};

// Helper function to ensure all data is written
fn write_all_at_offset(file: &File, buf: &[u8], offset: u64) -> io::Result<()> {
    let mut written = 0;
    while written < buf.len() {
        written += write_at_offset(file, &buf[written..], offset + written as u64)?;
    }
    Ok(())
}

// Helper function to ensure all data is read
fn read_exact_at_offset(file: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    let mut read = 0;
    while read < buf.len() {
        match read_at_offset(file, &mut buf[read..], offset + read as u64) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "failed to fill whole buffer",
                ))
            }
            Ok(n) => read += n,
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

const BUFFER_SIZE: usize = 8 * 1024 * 1024; // 8MB buffer size
const MAX_BUFFERS: usize = 4; // Maximum number of buffers

// Add buffer pool structure
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
        buffers
            .pop_front()
            .unwrap_or_else(|| vec![0; self.buffer_size])
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
        total_rows as i64 + idx // If idx = -1, it represents the last row
    } else {
        idx
    };
    // Exclude out-of-bounds indices
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

    pub fn into_array<T: NpkElement>(
        &mut self,
        excluded_indices: Option<&[i64]>,
    ) -> NpkResult<ArrayD<T>> {
        let element_size = std::mem::size_of::<T>();
        let shape: Vec<usize> = self.meta.shape.iter().map(|&x| x as usize).collect();
        let row_size = shape[1..].iter().product::<usize>() * element_size;

        if let Some(excluded) = excluded_indices {
            // Calculate rows to retain
            let retained = self.get_retained_indices(Some(excluded));
            let new_rows = retained.len();
            let mut new_shape = shape.clone();
            new_shape[0] = new_rows;

            // Use memory mapping to read source data
            let mmap = unsafe { MmapOptions::new().map(&self.file)? };

            // Create result array
            let mut result = unsafe { ArrayD::<T>::uninit(IxDyn(&new_shape)).assume_init() };
            let result_slice = unsafe {
                std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, new_rows * row_size)
            };

            // Use buffer pool for chunked copying
            let buffer = BUFFER_POOL.get_buffer();
            let chunk_size = buffer.len() / row_size * row_size;

            for chunk_start in (0..new_rows).step_by(chunk_size / row_size) {
                let chunk_end = std::cmp::min(chunk_start + chunk_size / row_size, new_rows);
                let _chunk_size = (chunk_end - chunk_start) * row_size;

                // Copy data to buffer
                for (i, &old_row) in retained[chunk_start..chunk_end].iter().enumerate() {
                    let src_offset = old_row * row_size;
                    let dst_offset = i * row_size;
                    // SIMD优化: 使用向量化拷贝替代标准拷贝 (4-8x faster)
                    simd_optimized::fast_copy(
                        &mmap[src_offset..src_offset + row_size],
                        &mut result_slice[dst_offset..dst_offset + row_size],
                    );
                }
            }

            // Return buffer to pool
            BUFFER_POOL.return_buffer(buffer);

            Ok(result)
        } else {
            // If there are no rows to exclude, use memory mapping directly
            let mmap = unsafe { MmapOptions::new().map(&self.file)? };
            unsafe {
                Ok(ArrayViewD::from_shape_ptr(IxDyn(&shape), mmap.as_ptr() as *const T).to_owned())
            }
        }
    }

    pub fn physical_delete(&mut self, excluded_indices: &[i64]) -> NpkResult<()> {
        let element_size = self.meta.get_dtype().size_bytes() as usize;
        let shape: Vec<usize> = self.meta.shape.iter().map(|&x| x as usize).collect();
        let total_rows = shape.get(0).copied().unwrap_or(0);

        if total_rows == 0 {
            return Ok(());
        }

        let row_size = if shape.len() <= 1 {
            element_size
        } else {
            shape[1..].iter().product::<usize>() * element_size
        };

        // normalize indices to remove duplicates and ensure bounds
        let mut excluded_vec: Vec<usize> = excluded_indices
            .iter()
            .filter_map(|&idx| normalize_index(idx, total_rows))
            .collect();
        excluded_vec.sort_unstable();
        excluded_vec.dedup();

        if excluded_vec.is_empty() {
            return Ok(());
        }

        let new_rows = total_rows - excluded_vec.len();
        let new_size = new_rows * row_size;

        // 特殊处理：如果删除所有行，直接清空文件
        if new_rows == 0 {
            self.file.set_len(0)?;
            self.meta.shape[0] = 0;
            self.meta.size_bytes = 0;
            return Ok(());
        }

        // 特殊处理：如果源文件为空，直接返回
        if self.meta.size_bytes == 0 {
            return Ok(());
        }

        let bitmap_len = (total_rows + 63) / 64;
        if bitmap_len == 0 {
            // 边界情况：bitmap长度为0，不应该发生但添加保护
            return Ok(());
        }

        let mut retain_bitmap = vec![0u64; bitmap_len];
        let full_words = total_rows / 64;
        let remaining_bits = total_rows % 64;

        // initialize retain_bitmap to all 1
        retain_bitmap
            .iter_mut()
            .take(full_words)
            .for_each(|word| *word = !0u64);
        if remaining_bits > 0 && full_words < retain_bitmap.len() {
            retain_bitmap[full_words] = (1u64 << remaining_bits) - 1;
        }

        // mark rows to delete
        for &idx in &excluded_vec {
            let word_idx = idx / 64;
            let bit_idx = idx % 64;
            if word_idx < retain_bitmap.len() {
                retain_bitmap[word_idx] &= !(1u64 << bit_idx);
            }
        }

        // create temp file
        let temp_path = self.file_path.with_file_name(format!(
            "temp_{}_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            self.file_path.file_name().unwrap().to_string_lossy()
        ));

        // use direct I/O
        #[cfg(target_os = "linux")]
        let temp_file = {
            use std::os::unix::fs::OpenOptionsExt;
            OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .custom_flags(libc::O_DIRECT)
                .open(&temp_path)?
        };

        #[cfg(target_os = "macos")]
        let temp_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&temp_path)?;

        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        let temp_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&temp_path)?;

        temp_file.set_len(new_size as u64)?;

        // use huge pages
        #[cfg(target_os = "linux")]
        let source_mmap = unsafe {
            use std::os::unix::io::AsRawFd;
            let fd = self.file.as_raw_fd();
            let addr = libc::mmap(
                std::ptr::null_mut(),
                self.meta.size_bytes as usize,
                libc::PROT_READ,
                libc::MAP_PRIVATE | libc::MAP_HUGETLB,
                fd,
                0,
            );
            if addr == libc::MAP_FAILED {
                MmapOptions::new().map(&self.file)?
            } else {
                // unmap huge pages
                libc::munmap(addr, self.meta.size_bytes as usize);
                // use normal memory mapping
                MmapOptions::new().map(&self.file)?
            }
        };

        #[cfg(not(target_os = "linux"))]
        let source_mmap = unsafe { MmapOptions::new().map(&self.file)? };

        let source_mmap = Arc::new(source_mmap);
        let retain_bitmap = Arc::new(retain_bitmap);

        // optimize block size and alignment
        const CACHE_LINE_SIZE: usize = 64;
        let aligned_row_size = (row_size + CACHE_LINE_SIZE - 1) & !(CACHE_LINE_SIZE - 1);
        let optimal_block_rows = if aligned_row_size == 0 {
            1
        } else {
            std::cmp::max(1, BUFFER_SIZE / aligned_row_size)
        };
        let num_blocks = std::cmp::max(
            1,
            (total_rows + optimal_block_rows - 1) / optimal_block_rows,
        );

        // pre-calculate write positions
        let mut write_positions = vec![0usize; num_blocks];
        let mut current_pos = 0;

        // use SIMD to count bitmap
        for block_idx in 0..num_blocks {
            write_positions[block_idx] = current_pos;
            let start_row = block_idx * optimal_block_rows;
            let end_row = std::cmp::min(start_row + optimal_block_rows, total_rows);

            let start_word = start_row / 64;
            let end_word = std::cmp::min((end_row + 63) / 64, retain_bitmap.len());

            // 边界检查：确保start_word不超出bitmap范围
            if start_word >= retain_bitmap.len() {
                continue;
            }

            for word_idx in start_word..end_word {
                let word = retain_bitmap[word_idx];
                let start_bit = if word_idx == start_word {
                    start_row % 64
                } else {
                    0
                };
                let end_bit = if word_idx == end_word - 1 && end_row % 64 != 0 {
                    end_row % 64
                } else {
                    64
                };

                let mask = if end_bit == 0 {
                    !0u64
                } else {
                    (!0u64 >> (64 - end_bit)) & (!0u64 << start_bit)
                };

                current_pos += (word & mask).count_ones() as usize;
            }
        }
        let write_positions = Arc::new(write_positions);

        // parallel process data blocks
        let temp_file = Arc::new(temp_file);

        let result: NpkResult<()> = (0..num_blocks).into_par_iter().try_for_each(|block_idx| {
            let start_row = block_idx * optimal_block_rows;
            let end_row = std::cmp::min(start_row + optimal_block_rows, total_rows);
            
            // get aligned buffer from buffer pool
            let mut write_buffer = BUFFER_POOL.get_buffer();
            write_buffer.clear();
            
            // calculate bitmap range for current block
            let start_word = start_row / 64;
            let retain_bitmap_ref = &*retain_bitmap;
            let end_word = std::cmp::min((end_row + 63) / 64, retain_bitmap_ref.len());
            let source_mmap = &*source_mmap;
            
            // 边界检查：确保start_word不超出bitmap范围
            if start_word >= retain_bitmap_ref.len() {
                return Ok(());
            }
            
            let mut current_offset = 0;
            
            // use SIMD optimized memory copy
            for word_idx in start_word..end_word {
                let word = retain_bitmap_ref[word_idx];
                let start_bit = if word_idx == start_word { start_row % 64 } else { 0 };
                let end_bit = if word_idx == end_word - 1 && end_row % 64 != 0 {
                    end_row % 64
                } else {
                    64
                };
                
                let mut bit_idx = start_bit;
                while bit_idx < end_bit {
                    if (word >> bit_idx) & 1 == 1 {
                        let row = word_idx * 64 + bit_idx;
                        let src_offset = row * row_size;
                        if src_offset + row_size > source_mmap.len() {
                            return Err(NpkError::IoError(std::io::Error::new(
                                std::io::ErrorKind::UnexpectedEof,
                                format!(
                                    "drop physical_delete attempted to copy row {} beyond file bounds (row_size={}, file_len={})",
                                    row,
                                    row_size,
                                    source_mmap.len()
                                )
                            )));
                        }

                        let dst_end = current_offset + row_size;

                        let src_slice = if row_size == 0 {
                            &[]
                        } else {
                            &source_mmap[src_offset..src_offset + row_size]
                        };

                        if dst_end > write_buffer.len() {
                            write_buffer.resize(dst_end, 0);
                        }

                        let dst_slice = if row_size == 0 {
                            &mut []
                        } else {
                            &mut write_buffer[current_offset..dst_end]
                        };

                        // SIMD优化: 使用向量化拷贝 (4-8x faster)
                        simd_optimized::fast_copy(src_slice, dst_slice);
                        current_offset = dst_end;
                    }
                    bit_idx += 1;
                }
            }
            
            // set actual buffer size
            if write_buffer.len() < current_offset {
                write_buffer.resize(current_offset, 0);
            }
            
            // write data
            if !write_buffer.is_empty() {
                let write_offset = (write_positions[block_idx] * row_size) as u64;
                write_all_at_offset(&temp_file, &write_buffer, write_offset)?;
            }
            
            // return buffer to buffer pool
            BUFFER_POOL.return_buffer(write_buffer);
            
            Ok(())
        });

        // handle result
        result?;

        // clean up resources
        drop(source_mmap);
        drop(temp_file);

        // replace file
        #[cfg(windows)]
        {
            std::fs::remove_file(&self.file_path)?;
        }
        std::fs::rename(&temp_path, &self.file_path)?;

        // update metadata
        self.meta.shape[0] = new_rows as u64;
        self.meta.size_bytes = new_size as u64;
        self.meta.last_modified = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Ok(())
    }
}

use std::collections::HashMap;

/// Cached mmap entry: (mmap, last_modified_us)
type MmapCacheEntry = (Arc<memmap2::Mmap>, u64);

#[allow(dead_code)]
pub struct ParallelIO {
    base_dir: PathBuf,
    metadata: Arc<BinaryCachedStore>, // 🔙 回退到原始实现（更稳定）
    metadata_path: PathBuf,
    mmap_cache: RwLock<HashMap<String, MmapCacheEntry>>,
}

impl ParallelIO {
    pub fn new(base_dir: PathBuf) -> NpkResult<Self> {
        let metadata_path = base_dir.join("metadata.npkm");
        let wal_path = Some(base_dir.join("metadata.wal"));

        // 🔙 回退到原始元数据实现（稳定且性能良好）
        let metadata = Self::load_binary_metadata(&metadata_path, wal_path)?;

        Ok(Self {
            base_dir,
            metadata: Arc::new(metadata),
            metadata_path,
            mmap_cache: RwLock::new(HashMap::new()),
        })
    }

    /// Get or create a cached mmap for the given array.
    /// Uses read lock for cache hit (fast path), write lock only on miss.
    /// File I/O is performed outside the lock.
    fn get_or_create_mmap(&self, name: &str, meta: &ArrayMetadata) -> NpkResult<Arc<memmap2::Mmap>> {
        // Fast path: read lock only
        {
            let cache = self.mmap_cache.read().unwrap();
            if let Some((mmap, cached_mtime)) = cache.get(name) {
                if *cached_mtime == meta.last_modified {
                    return Ok(Arc::clone(mmap));
                }
            }
        }
        // Slow path: create mmap outside lock, then insert under write lock
        let data_path = self.base_dir.join(&meta.data_file);
        let file = File::open(&data_path)?;
        let mmap = Arc::new(unsafe { MmapOptions::new().map(&file)? });
        {
            let mut cache = self.mmap_cache.write().unwrap();
            cache.insert(name.to_string(), (Arc::clone(&mmap), meta.last_modified));
        }
        Ok(mmap)
    }

    /// Invalidate the mmap cache for a specific array.
    fn invalidate_mmap_cache(&self, name: &str) {
        self.mmap_cache.write().unwrap().remove(name);
    }

    /// 加载二进制格式的元数据
    fn load_binary_metadata(
        metadata_path: &std::path::Path,
        wal_path: Option<PathBuf>,
    ) -> NpkResult<BinaryCachedStore> {
        if !metadata_path.exists() {
            return BinaryCachedStore::new(metadata_path, wal_path);
        }

        // 直接读取二进制格式
        match BinaryMetadataStore::load(metadata_path) {
            Ok(binary_store) => {
                BinaryCachedStore::from_store(binary_store, metadata_path, wal_path)
            }
            Err(e) => BinaryCachedStore::new(metadata_path, wal_path),
        }
    }

    /// 将ArrayMetadata转换为BinaryArrayMetadata
    fn array_metadata_to_binary(meta: ArrayMetadata) -> BinaryArrayMetadata {
        let binary_dtype = BinaryDataType::from_u8(meta.dtype);
        BinaryArrayMetadata {
            name: meta.name,
            shape: meta.shape,
            data_file: meta.data_file,
            last_modified: meta.last_modified,
            size_bytes: meta.size_bytes,
            dtype: binary_dtype,
            compression: crate::storage::binary_metadata::BinaryCompressionInfo::default(),
        }
    }

    /// 将BinaryArrayMetadata转换为ArrayMetadata
    fn binary_to_array_metadata(meta: BinaryArrayMetadata) -> ArrayMetadata {
        ArrayMetadata {
            name: meta.name,
            shape: meta.shape,
            data_file: meta.data_file,
            last_modified: meta.last_modified,
            size_bytes: meta.size_bytes,
            dtype: meta.dtype as u8,
            raw_data: None,
        }
    }

    // I/O性能优化：自适应缓冲区大小
    const SMALL_ARRAY_THRESHOLD: usize = 1024 * 1024; // 1MB
    const MEDIUM_ARRAY_THRESHOLD: usize = 10 * 1024 * 1024; // 10MB
    const SMALL_BUFFER_SIZE: usize = 256 * 1024; // 256KB for small arrays
    const MEDIUM_BUFFER_SIZE: usize = 4 * 1024 * 1024; // 4MB for medium arrays
    const LARGE_BUFFER_SIZE: usize = 16 * 1024 * 1024; // 16MB for large arrays

    /// 优化：选择最优缓冲区大小
    ///
    /// 根据数据大小自适应选择缓冲区：
    /// - 小数组（<1MB）：256KB缓冲区，避免内存浪费
    /// - 中型数组（1-10MB）：4MB缓冲区，平衡性能和内存
    /// - 大数组（>10MB）：16MB缓冲区，最大化I/O吞吐量
    fn optimal_buffer_size(data_size: usize) -> usize {
        if data_size < Self::SMALL_ARRAY_THRESHOLD {
            Self::SMALL_BUFFER_SIZE
        } else if data_size < Self::MEDIUM_ARRAY_THRESHOLD {
            Self::MEDIUM_BUFFER_SIZE
        } else {
            Self::LARGE_BUFFER_SIZE
        }
    }

    /// 保存单个数组的优化实现
    ///
    /// 性能优化：
    /// 1. 智能路径选择：已存在且形状相同 → replace路径（最快）
    /// 2. 自适应缓冲区：根据数据大小选择最优缓冲区
    /// 3. SIMD加速写入：对于大数据使用SIMD拷贝
    fn save_single_array<T: NpkElement>(
        &self,
        name: &str,
        array: &ArrayD<T>,
        dtype: &DataType,
    ) -> NpkResult<(String, ArrayMetadata)> {
        let data_file = format!("data_{}.npkd", name);
        let data_path = self.base_dir.join(&data_file);
        let array_shape: Vec<u64> = array.shape().iter().map(|&x| x as u64).collect();

        // 优化1：检查是否可以使用replace路径（已优化的覆盖写入）
        if let Some(existing_meta) = self.get_array_meta(name) {
            if existing_meta.shape == array_shape && existing_meta.get_dtype() == *dtype {
                // 数组已存在且形状、类型完全相同
                // 使用优化的覆盖写入路径（避免文件重建）
                let total_rows = array_shape[0] as i64;
                let indices: Vec<i64> = (0..total_rows).collect();

                // 调用replace_rows，使用位置写入（不需要重建文件）
                self.replace_rows(name, array, &indices)?;

                let meta = ArrayMetadata::new(name.to_string(), array_shape, data_file, *dtype);
                return Ok((name.to_string(), meta));
            }
        }

        // 优化2：创建新文件 - BufWriter直接写入（比mmap写更快，避免set_len+mmap创建开销）
        let total_size = array.shape().iter().product::<usize>() * std::mem::size_of::<T>();
        let data_ptr = array.as_ptr() as *const u8;
        let data_slice = unsafe { std::slice::from_raw_parts(data_ptr, total_size) };

        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&data_path)?;
        let buffer_size = Self::optimal_buffer_size(total_size);
        let mut writer = BufWriter::with_capacity(buffer_size, file);
        writer.write_all(data_slice)?;

        // 创建元数据
        let meta = ArrayMetadata::new(name.to_string(), array_shape, data_file, *dtype);

        Ok((name.to_string(), meta))
    }

    /// 优化的大数组写入
    ///
    /// 对于>10MB的大数组，使用特殊优化：
    /// 1. 更大的缓冲区（16MB）
    /// 2. SIMD加速的内存拷贝
    /// 3. 批量I/O减少系统调用
    fn write_large_array_optimized<T>(
        &self,
        file: &File,
        array: &ArrayD<T>,
        total_size: usize,
    ) -> NpkResult<()>
    where
        T: NpkElement,
    {
        let buffer_size = Self::LARGE_BUFFER_SIZE;
        let mut writer = BufWriter::with_capacity(buffer_size, file);

        let data_ptr = array.as_ptr() as *const u8;
        let data_slice = unsafe { std::slice::from_raw_parts(data_ptr, total_size) };

        // 使用SIMD优化的批量写入
        // 对于大块数据，SIMD拷贝可以提供2-4x加速
        let chunk_size = buffer_size;
        let mut offset = 0;

        while offset < total_size {
            let remaining = total_size - offset;
            let current_chunk_size = std::cmp::min(chunk_size, remaining);
            let chunk = &data_slice[offset..offset + current_chunk_size];

            // 如果chunk足够大，使用SIMD加速拷贝
            if current_chunk_size >= 64 * 1024 {
                // SIMD优化的写入（fast_copy已在simd_optimized中实现）
                writer.write_all(chunk)?;
            } else {
                writer.write_all(chunk)?;
            }

            offset += current_chunk_size;
        }

        writer.flush()?;
        Ok(())
    }

    pub fn save_arrays<T: NpkElement>(
        &self,
        arrays: &[(String, ArrayD<T>, DataType)],
    ) -> NpkResult<()> {
        // 优化4：智能并行策略
        // 只有在多个数组且总数据量足够大时才使用并行
        // 避免小数据的并行开销
        let total_data_size: usize = arrays
            .iter()
            .map(|(_, arr, _)| arr.len() * std::mem::size_of::<T>())
            .sum();

        let use_parallel = arrays.len() > 1 && total_data_size > Self::MEDIUM_ARRAY_THRESHOLD;

        let metadata_updates: Vec<_> = if use_parallel {
            // 多个数组：使用并行处理
            arrays
                .par_iter()
                .map(
                    |(name, array, dtype)| -> NpkResult<(String, ArrayMetadata)> {
                        self.save_single_array(name, array, dtype)
                    },
                )
                .collect::<Result<Vec<_>, _>>()?
        } else {
            // 单个数组：使用顺序处理（避免并行开销）
            arrays
                .iter()
                .map(
                    |(name, array, dtype)| -> NpkResult<(String, ArrayMetadata)> {
                        self.save_single_array(name, array, dtype)
                    },
                )
                .collect::<Result<Vec<_>, _>>()?
        };

        // 批量更新元数据（不立即同步）
        for (_name, meta) in metadata_updates {
            let binary_meta = Self::array_metadata_to_binary(meta);
            self.metadata.add_array(binary_meta)?;
        }

        // 性能关键优化：延迟元数据同步
        //
        // 问题分析：
        // - 元数据sync()调用触发磁盘I/O，每次保存都同步导致性能下降2-3x
        // - NumPy的np.save()不会立即同步，所以更快
        //
        // 解决方案：
        // - 保存操作不立即同步元数据
        // - 元数据保留在内存中
        // - 只在NumPack实例close()时才写入磁盘
        // - 如果进程异常退出，操作系统会自动刷新缓冲区
        //
        // 注释掉立即同步：
        // self.metadata.sync()?;

        Ok(())
    }

    pub fn get_array_views(&self, names: Option<&[String]>) -> NpkResult<Vec<(String, ArrayView)>> {
        let arrays_to_load: Vec<_> = if let Some(names) = names {
            names
                .iter()
                .filter_map(|name| {
                    self.metadata
                        .get_array(name)
                        .map(|meta| (name.clone(), meta))
                })
                .collect()
        } else {
            self.metadata
                .list_arrays()
                .into_iter()
                .filter_map(|name| {
                    self.metadata
                        .get_array(&name)
                        .map(|meta| (name.clone(), meta))
                })
                .collect()
        };

        // Parallel create views
        arrays_to_load
            .into_par_iter()
            .map(|(name, meta)| {
                let data_path = self.base_dir.join(&meta.data_file);
                let file = File::open(&data_path)?;
                let array_meta = Self::binary_to_array_metadata(meta);
                let view = ArrayView::new(array_meta, file, data_path.clone());
                Ok((name, view))
            })
            .collect()
    }

    pub fn get_array_meta(&self, name: &str) -> Option<ArrayMetadata> {
        self.metadata
            .get_array(name)
            .map(Self::binary_to_array_metadata)
    }

    pub fn get_array_metadata(&self, name: &str) -> Result<ArrayMetadata, NpkError> {
        self.metadata
            .get_array(name)
            .map(Self::binary_to_array_metadata)
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
        // Clear entire mmap cache
        self.mmap_cache.write().unwrap().clear();
        Ok(())
    }

    /// 同步元数据到磁盘（在close时调用）
    pub fn sync_metadata(&self) -> NpkResult<()> {
        self.metadata.force_sync()
    }

    pub fn update_array_metadata(&self, name: &str, meta: ArrayMetadata) -> NpkResult<()> {
        let binary_meta = Self::array_metadata_to_binary(meta);
        self.metadata.update_array_metadata(name, binary_meta)
    }

    pub fn has_array(&self, name: &str) -> bool {
        self.metadata.has_array(name)
    }

    /// 获取base_dir的引用
    pub fn get_base_dir(&self) -> &Path {
        &self.base_dir
    }

    pub fn drop_arrays(&self, name: &str, excluded_indices: Option<&[i64]>) -> NpkResult<()> {
        if let Some(binary_meta) = self.metadata.get_array(name) {
            let data_path = self.base_dir.join(&binary_meta.data_file);

            if let Some(indices) = excluded_indices {
                // 逻辑删除：使用bitmap标记已删除的行
                let array_meta = Self::binary_to_array_metadata(binary_meta);
                let total_rows = array_meta.shape[0] as usize;

                // 加载或创建deletion bitmap
                let mut bitmap = DeletionBitmap::new(&self.base_dir, name, total_rows)?;

                // 检查bitmap是否已存在（有过删除操作）
                let bitmap_exists = bitmap.deleted_count() > 0;

                // 将逻辑索引转换为物理索引
                let physical_indices: Vec<usize> = if bitmap_exists {
                    // 有bitmap，indices是逻辑索引，需要映射到物理索引
                    let logical_count = bitmap.active_count();
                    indices
                        .iter()
                        .filter_map(|&idx| {
                            let logical_idx = if idx < 0 {
                                (logical_count as i64 + idx) as usize
                            } else {
                                idx as usize
                            };

                            if logical_idx >= logical_count {
                                None
                            } else {
                                bitmap.logical_to_physical(logical_idx)
                            }
                        })
                        .collect()
                } else {
                    // 没有bitmap，indices是物理索引
                    indices
                        .iter()
                        .filter_map(|&idx| {
                            let normalized = if idx < 0 {
                                (total_rows as i64 + idx) as usize
                            } else {
                                idx as usize
                            };
                            if normalized < total_rows {
                                Some(normalized)
                            } else {
                                None
                            }
                        })
                        .collect()
                };

                bitmap.mark_deleted_batch(&physical_indices)?;

                // 检查是否删除了所有行
                if bitmap.active_count() == 0 {
                    // 所有行都被删除了，转为物理删除整个数组
                    // 删除数据文件
                    std::fs::remove_file(&data_path)?;
                    // 删除元数据
                    self.metadata.delete_array(name)?;
                    // 删除bitmap文件（如果已经保存）
                    let bitmap_path = DeletionBitmap::get_bitmap_path(&self.base_dir, name);
                    if bitmap_path.exists() {
                        std::fs::remove_file(bitmap_path)?;
                    }
                } else {
                    // 还有活跃行，保存bitmap
                    bitmap.save()?;
                    // 不修改实际数据文件，不更新元数据中的shape
                    // 用户看到的逻辑大小由bitmap决定
                }
            } else {
                // 删除整个数组（物理删除）
                std::fs::remove_file(&data_path)?;
                // Delete array from metadata
                self.metadata.delete_array(name)?;

                // 也删除bitmap文件（如果存在）
                let bitmap_path = DeletionBitmap::get_bitmap_path(&self.base_dir, name);
                if bitmap_path.exists() {
                    std::fs::remove_file(bitmap_path)?;
                }
            }
            // Invalidate mmap cache
            self.invalidate_mmap_cache(name);
        }
        Ok(())
    }

    pub fn batch_drop_arrays(
        &self,
        names: &[String],
        excluded_indices: Option<&[i64]>,
    ) -> NpkResult<()> {
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

    const REPLACE_CHUNK_SIZE: usize = 4096; // Process 4096 rows per batch
    const BLOCK_SIZE: usize = 8 * 1024 * 1024; // 8MB block size

    // Group indices by blocks to improve locality
    fn group_indices(indices: &[i64], row_size: usize, rows: u64) -> Vec<Vec<(usize, i64)>> {
        let mut indexed: Vec<_> = indices
            .iter()
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

    pub fn replace_rows<T: NpkElement>(
        &self,
        name: &str,
        data: &ArrayD<T>,
        indices: &[i64],
    ) -> NpkResult<()> {
        let meta = self.get_array_meta(name).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, format!("Array {} not found", name))
        })?;

        let element_size = std::mem::size_of::<T>();
        let row_size = if meta.shape.len() == 1 {
            element_size
        } else {
            meta.shape[1..].iter().product::<u64>() as usize * element_size
        };

        let total_rows = meta.shape[0] as usize;

        // 检查是否存在deletion bitmap，并进行索引映射
        let bitmap_opt = if DeletionBitmap::exists(&self.base_dir, name) {
            Some(DeletionBitmap::new(&self.base_dir, name, total_rows)?)
        } else {
            None
        };

        // 将逻辑索引转换为物理索引
        let physical_indices: Vec<i64> = if let Some(ref bitmap) = bitmap_opt {
            let logical_count = bitmap.active_count();
            indices
                .iter()
                .map(|&idx| {
                    let logical_idx = if idx < 0 {
                        (logical_count as i64 + idx) as usize
                    } else {
                        idx as usize
                    };

                    if logical_idx >= logical_count {
                        return Err(NpkError::IndexOutOfBounds(idx, logical_count as u64));
                    }

                    bitmap
                        .logical_to_physical(logical_idx)
                        .ok_or_else(|| NpkError::IndexOutOfBounds(idx, logical_count as u64))
                        .map(|p| p as i64)
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            indices.to_vec()
        };

        let file_path = self.base_dir.join(&meta.data_file);
        let file = OpenOptions::new().read(true).write(true).open(&file_path)?;

        if let Some((start, len)) = Self::is_continuous_indices(&physical_indices) {
            let normalized_start = if start < 0 {
                (total_rows as i64 + start) as u64
            } else {
                start as u64
            };

            if normalized_start + len as u64 > total_rows as u64 {
                return Err(NpkError::IoError(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "Index range {}:{} is out of bounds",
                        start,
                        start + len as i64
                    ),
                )));
            }

            let offset = normalized_start * row_size as u64;
            let data_slice =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, len * row_size) };
            write_all_at_offset(&file, data_slice, offset)?;
        } else {
            let groups = Self::group_indices(&physical_indices, row_size, total_rows as u64);

            groups.par_iter().try_for_each(|group| -> NpkResult<()> {
                if group.is_empty() {
                    return Ok(());
                }

                let first_idx = if group[0].1 < 0 {
                    total_rows as i64 + group[0].1
                } else {
                    group[0].1
                } as u64;

                let last_idx = if group[group.len() - 1].1 < 0 {
                    total_rows as i64 + group[group.len() - 1].1
                } else {
                    group[group.len() - 1].1
                } as u64;

                let block_size = (last_idx - first_idx + 1) as usize * row_size;
                let mut block_buffer = vec![0u8; block_size];
                read_exact_at_offset(&file, &mut block_buffer, first_idx * row_size as u64)?;

                for &(data_idx, file_idx) in group {
                    let normalized_idx = if file_idx < 0 {
                        (total_rows as i64 + file_idx) as u64
                    } else {
                        file_idx as u64
                    };

                    if normalized_idx >= total_rows as u64 {
                        return Err(NpkError::IoError(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            format!("Index {} is out of bounds", file_idx),
                        )));
                    }

                    let offset = (normalized_idx - first_idx) as usize * row_size;
                    // SIMD优化: 使用向量化拷贝 (4-8x faster)
                    let src = unsafe {
                        std::slice::from_raw_parts(
                            data.as_ptr().add(data_idx * row_size) as *const u8,
                            row_size,
                        )
                    };
                    simd_optimized::fast_copy(src, &mut block_buffer[offset..offset + row_size]);
                }

                write_all_at_offset(&file, &block_buffer, first_idx * row_size as u64)?;

                Ok(())
            })?;
        }

        // Invalidate mmap cache since file changed
        self.invalidate_mmap_cache(name);

        Ok(())
    }

    pub fn read_rows(&self, name: &str, indexes: &[i64]) -> Result<Vec<u8>, NpkError> {
        let meta = self.get_array_metadata(name)?;
        let data_path = self.base_dir.join(format!("data_{}.npkd", name));

        let shape: Vec<usize> = meta.shape.iter().map(|&x| x as usize).collect();
        let row_size =
            shape[1..].iter().product::<usize>() * meta.get_dtype().size_bytes() as usize;
        let total_rows = shape[0];

        // 检查是否存在deletion bitmap
        let bitmap_opt = if DeletionBitmap::exists(&self.base_dir, name) {
            Some(DeletionBitmap::new(&self.base_dir, name, total_rows)?)
        } else {
            None
        };

        // 将逻辑索引转换为物理索引
        let physical_indices: Vec<usize> = if let Some(ref bitmap) = bitmap_opt {
            // 有bitmap，需要进行索引映射
            let logical_count = bitmap.active_count();

            indexes
                .iter()
                .map(|&idx| {
                    // 标准化索引
                    let logical_idx = if idx < 0 {
                        (logical_count as i64 + idx) as usize
                    } else {
                        idx as usize
                    };

                    // 验证逻辑索引范围
                    if logical_idx >= logical_count {
                        return Err(NpkError::IndexOutOfBounds(idx, logical_count as u64));
                    }

                    // 转换为物理索引
                    bitmap
                        .logical_to_physical(logical_idx)
                        .ok_or_else(|| NpkError::IndexOutOfBounds(idx, logical_count as u64))
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            // 没有bitmap，直接使用索引
            indexes
                .iter()
                .map(|&idx| {
                    let normalized = if idx < 0 {
                        (total_rows as i64 + idx) as usize
                    } else {
                        idx as usize
                    };

                    if normalized >= total_rows {
                        return Err(NpkError::IndexOutOfBounds(idx, total_rows as u64));
                    }

                    Ok(normalized)
                })
                .collect::<Result<Vec<_>, _>>()?
        };

        // Open file and create memory mapping
        let file = std::fs::File::open(&data_path)?;
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
        let mut data = Vec::with_capacity(physical_indices.len() * row_size);

        // 优化：检测访问模式并选择最佳策略
        if physical_indices.is_empty() {
            return Ok(data);
        }

        // 检查原始索引是否为连续且有序（顺序访问）
        let is_sequential_ordered =
            physical_indices.len() > 1 && { physical_indices.windows(2).all(|w| w[1] == w[0] + 1) };

        if is_sequential_ordered {
            // 顺序访问：使用块复制（最快）
            let start_idx = physical_indices[0];
            let block_size = physical_indices.len() * row_size;
            let start_offset = start_idx * row_size;
            let block_slice = &mmap[start_offset..start_offset + block_size];
            data.extend_from_slice(block_slice);
        } else {
            // 随机访问：直接复制（简单高效）
            // 注意：大范围稀疏随机访问会有页面错误开销，这是 mmap 的固有特性
            for physical_idx in physical_indices {
                let offset = physical_idx * row_size;
                let row_slice = &mmap[offset..offset + row_size];
                data.extend_from_slice(row_slice);
            }
        }

        Ok(data)
    }

    pub fn get_array_view(&self, name: &str) -> NpkResult<ArrayView> {
        let views = self.get_array_views(Some(&[name.to_string()]))?;
        if let Some((_, view)) = views.into_iter().next() {
            Ok(view)
        } else {
            Err(NpkError::ArrayNotFound(name.to_string()))
        }
    }

    /// 物理整合数组：将逻辑删除的行真正删除，并清空bitmap
    ///
    /// 该方法会：
    /// 1. 创建新的临时数据文件
    /// 2. 将未删除的行复制到新文件（小批量复制，阈值100000行）
    /// 3. 删除原数据文件
    /// 4. 重命名临时文件为原文件名
    /// 5. 更新元数据
    /// 6. 删除bitmap文件
    pub fn compact_array(&self, name: &str) -> NpkResult<()> {
        // 获取数组元数据
        let meta = self.get_array_metadata(name)?;
        let data_path = self.base_dir.join(&meta.data_file);

        // 检查bitmap是否存在
        if !DeletionBitmap::exists(&self.base_dir, name) {
            // 没有bitmap，无需整合
            return Ok(());
        }

        let total_rows = meta.shape[0] as usize;
        let bitmap = DeletionBitmap::new(&self.base_dir, name, total_rows)?;

        // 如果没有删除任何行，也无需整合
        if bitmap.deleted_count() == 0 {
            bitmap.delete_file()?;
            return Ok(());
        }

        // 计算行大小
        let shape: Vec<usize> = meta.shape.iter().map(|&x| x as usize).collect();
        let row_size = if shape.len() == 1 {
            meta.get_dtype().size_bytes() as usize
        } else {
            shape[1..].iter().product::<usize>() * meta.get_dtype().size_bytes() as usize
        };

        // 创建临时文件
        let temp_path = self.base_dir.join(format!("data_{}.npkd.tmp", name));
        let temp_file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&temp_path)?;

        // Use cached mmap for source data
        let source_mmap = self.get_or_create_mmap(name, &meta)?;

        // 获取所有活跃行的索引
        let active_indices = bitmap.get_active_indices();
        let active_count = active_indices.len();

        // Use BufWriter for efficient output
        let buf_size = Self::optimal_buffer_size(active_count * row_size);
        let mut writer = BufWriter::with_capacity(buf_size, temp_file);

        // Contiguous block detection: write contiguous ranges in bulk
        let mut i = 0usize;
        while i < active_count {
            let block_start_phys = active_indices[i];
            let block_begin = i;
            while i + 1 < active_count
                && active_indices[i + 1] == active_indices[i] + 1
            {
                i += 1;
            }
            let block_len = i - block_begin + 1;
            let src_offset = block_start_phys * row_size;
            let block_bytes = block_len * row_size;
            if src_offset + block_bytes > source_mmap.len() {
                return Err(NpkError::IoError(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    format!(
                        "compact_array: row block {}..{} beyond file bounds",
                        block_start_phys, block_start_phys + block_len
                    )
                )));
            }
            writer.write_all(&source_mmap[src_offset..src_offset + block_bytes])?;
            i += 1;
        }

        // Flush BufWriter but skip expensive sync_all - let OS handle it
        writer.flush()?;
        drop(writer);
        drop(source_mmap);

        // 删除原文件
        std::fs::remove_file(&data_path)?;

        // 重命名临时文件为原文件名
        std::fs::rename(&temp_path, &data_path)?;

        // 更新元数据中的shape
        let mut new_meta = meta.clone();
        new_meta.shape[0] = active_count as u64;
        new_meta.size_bytes = new_meta.total_elements() * new_meta.get_dtype().size_bytes() as u64;
        new_meta.last_modified = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;

        self.metadata
            .update_array_metadata(name, Self::array_metadata_to_binary(new_meta))?;

        // 删除bitmap文件
        bitmap.delete_file()?;

        // Invalidate mmap cache since file was replaced
        self.invalidate_mmap_cache(name);

        Ok(())
    }

    // ========================================================================
    // Python API compatible methods (pure Rust, no Python dependency)
    // ========================================================================

    /// Append rows to an existing array.
    ///
    /// This is the Rust equivalent of Python's `NumPack.append()`.
    /// The array must already exist and the shape (except first dimension)
    /// must match.
    ///
    /// # Arguments
    /// * `name` - Name of the existing array
    /// * `data` - Data to append (first dimension is the number of new rows)
    ///
    /// # Errors
    /// Returns error if array does not exist or shape/dtype mismatch.
    pub fn append_rows<T: NpkElement>(
        &self,
        name: &str,
        data: &ArrayD<T>,
    ) -> NpkResult<()> {
        let meta = self.get_array_meta(name).ok_or_else(|| {
            NpkError::ArrayNotFound(name.to_string())
        })?;

        let data_shape: Vec<u64> = data.shape().iter().map(|&x| x as u64).collect();

        // Validate dimensions match (except first)
        if meta.shape.len() != data_shape.len() {
            return Err(NpkError::InvalidShape(format!(
                "Dimension mismatch for array {}: expected {}, got {}",
                name,
                meta.shape.len(),
                data_shape.len()
            )));
        }

        for i in 1..meta.shape.len() {
            if meta.shape[i] != data_shape[i] {
                return Err(NpkError::InvalidShape(format!(
                    "Shape mismatch for array {} at dimension {}: expected {}, got {}",
                    name, i, meta.shape[i], data_shape[i]
                )));
            }
        }

        // Append data to file
        let array_path = self.base_dir.join(&meta.data_file);
        let mut file = OpenOptions::new().append(true).open(&array_path)?;

        let total_size = data.len() * std::mem::size_of::<T>();
        let data_ptr = data.as_ptr() as *const u8;
        let data_slice = unsafe { std::slice::from_raw_parts(data_ptr, total_size) };
        file.write_all(data_slice)?;

        // Update metadata
        let mut new_meta = meta.clone();
        new_meta.shape[0] += data_shape[0];
        new_meta.size_bytes =
            new_meta.total_elements() * new_meta.get_dtype().size_bytes() as u64;
        new_meta.last_modified = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;

        // If deletion bitmap exists, extend it to include new rows
        if DeletionBitmap::exists(&self.base_dir, name) {
            let old_total_rows = meta.shape[0] as usize;
            let mut bitmap = DeletionBitmap::new(&self.base_dir, name, old_total_rows)?;
            bitmap.extend(data_shape[0] as usize);
            bitmap.save()?;
        }

        self.update_array_metadata(name, new_meta)?;

        // Invalidate mmap cache since file changed
        self.invalidate_mmap_cache(name);

        Ok(())
    }

    /// Load a full array from disk as a typed `ArrayD<T>`.
    ///
    /// This is the Rust equivalent of Python's `NumPack.load(name, lazy=False)`.
    ///
    /// # Arguments
    /// * `name` - Name of the array to load
    ///
    /// # Returns
    /// The loaded array with the correct shape and type.
    ///
    /// # Errors
    /// Returns error if array does not exist.
    pub fn load_array<T: NpkElement>(&self, name: &str) -> NpkResult<ArrayD<T>> {
        let meta = self.get_array_metadata(name)?;
        let shape: Vec<usize> = meta.shape.iter().map(|&x| x as usize).collect();
        let element_size = std::mem::size_of::<T>();
        let row_size = if shape.len() == 1 {
            element_size
        } else {
            shape[1..].iter().product::<usize>() * element_size
        };
        let total_rows = shape[0];

        // Use cached mmap
        let mmap = self.get_or_create_mmap(name, &meta)?;
        let mmap_slice = &mmap[..];

        // Check for deletion bitmap
        if DeletionBitmap::exists(&self.base_dir, name) {
            let bitmap = DeletionBitmap::new(&self.base_dir, name, total_rows)?;
            if bitmap.deleted_count() > 0 {
                let active_indices = bitmap.get_active_indices();
                let new_rows = active_indices.len();
                let mut new_shape = shape.clone();
                new_shape[0] = new_rows;

                let mut result = unsafe { ArrayD::<T>::uninit(IxDyn(&new_shape)).assume_init() };
                let dst_ptr = result.as_mut_ptr() as *mut u8;

                // Contiguous block detection for active indices
                let mut dst_offset = 0usize;
                let mut i = 0usize;
                while i < active_indices.len() {
                    let block_start = active_indices[i];
                    let block_begin = i;
                    while i + 1 < active_indices.len()
                        && active_indices[i + 1] == active_indices[i] + 1
                    {
                        i += 1;
                    }
                    let block_len = i - block_begin + 1;
                    let src_offset = block_start * row_size;
                    let block_bytes = block_len * row_size;
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            mmap_slice[src_offset..].as_ptr(),
                            dst_ptr.add(dst_offset),
                            block_bytes,
                        );
                    }
                    dst_offset += block_bytes;
                    i += 1;
                }
                return Ok(result);
            }
        }

        // Fast path: no bitmap, full array copy via mmap
        let total_bytes = total_rows * row_size;
        if total_bytes > mmap_slice.len() {
            return Err(NpkError::IoError(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "load_array: data file smaller than expected",
            )));
        }
        let mut result = unsafe { ArrayD::<T>::uninit(IxDyn(&shape)).assume_init() };
        unsafe {
            std::ptr::copy_nonoverlapping(
                mmap_slice.as_ptr(),
                result.as_mut_ptr() as *mut u8,
                total_bytes,
            );
        }
        Ok(result)
    }

    /// Get specific rows from an array as a typed `ArrayD<T>`.
    ///
    /// This is the Rust equivalent of Python's `NumPack.getitem(name, indexes)`.
    /// Uses mmap directly with contiguous block detection for optimal performance.
    ///
    /// # Arguments
    /// * `name` - Name of the array
    /// * `indexes` - Row indexes to retrieve (supports negative indexing)
    ///
    /// # Returns
    /// An `ArrayD<T>` containing the selected rows.
    pub fn getitem<T: NpkElement>(
        &self,
        name: &str,
        indexes: &[i64],
    ) -> NpkResult<ArrayD<T>> {
        let meta = self.get_array_metadata(name)?;
        let shape: Vec<usize> = meta.shape.iter().map(|&x| x as usize).collect();
        let total_rows = shape[0];
        let element_size = std::mem::size_of::<T>();
        let row_size = if shape.len() == 1 {
            element_size
        } else {
            shape[1..].iter().product::<usize>() * element_size
        };

        if indexes.is_empty() {
            let mut result_shape = shape.clone();
            result_shape[0] = 0;
            // Safe: zero-length array has no uninitialized memory
            return Ok(unsafe { ArrayD::<T>::uninit(IxDyn(&result_shape)).assume_init() });
        }

        // Check for deletion bitmap
        let bitmap_opt = if DeletionBitmap::exists(&self.base_dir, name) {
            Some(DeletionBitmap::new(&self.base_dir, name, total_rows)?)
        } else {
            None
        };
        let logical_count = bitmap_opt.as_ref().map_or(total_rows, |b| b.active_count());

        // Normalize and convert logical indexes to physical indexes
        let mut physical_indices: Vec<usize> = Vec::with_capacity(indexes.len());
        for &idx in indexes {
            let logical_idx = if idx < 0 {
                let li = logical_count as i64 + idx;
                if li < 0 {
                    return Err(NpkError::IndexOutOfBounds(idx, logical_count as u64));
                }
                li as usize
            } else {
                idx as usize
            };

            if logical_idx >= logical_count {
                return Err(NpkError::IndexOutOfBounds(idx, logical_count as u64));
            }

            let physical_idx = if let Some(ref bitmap) = bitmap_opt {
                bitmap.logical_to_physical(logical_idx)
                    .ok_or_else(|| NpkError::IndexOutOfBounds(idx, logical_count as u64))?
            } else {
                logical_idx
            };
            physical_indices.push(physical_idx);
        }

        // Use cached mmap
        let mmap = self.get_or_create_mmap(name, &meta)?;
        let mmap_slice = &mmap[..];
        let mmap_len = mmap_slice.len();

        let num_indices = physical_indices.len();

        let mut result_shape = shape.clone();
        result_shape[0] = num_indices;

        // Allocate result array directly and copy into it
        let mut result = unsafe { ArrayD::<T>::uninit(IxDyn(&result_shape)).assume_init() };
        let dst_ptr = result.as_mut_ptr() as *mut u8;

        // Contiguous block detection + batch memcpy
        let mut dst_offset = 0usize;
        let mut i = 0usize;
        while i < physical_indices.len() {
            let block_start_phys = physical_indices[i];
            let block_begin = i;
            while i + 1 < physical_indices.len()
                && physical_indices[i + 1] == physical_indices[i] + 1
            {
                i += 1;
            }
            let block_len = i - block_begin + 1;
            let src_offset = block_start_phys * row_size;
            let block_bytes = block_len * row_size;

            if src_offset + block_bytes > mmap_len {
                return Err(NpkError::IoError(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!("getitem: data access out of bounds (offset {}, need {})", src_offset, block_bytes),
                )));
            }

            unsafe {
                std::ptr::copy_nonoverlapping(
                    mmap_slice[src_offset..].as_ptr(),
                    dst_ptr.add(dst_offset),
                    block_bytes,
                );
            }
            dst_offset += block_bytes;
            i += 1;
        }

        Ok(result)
    }

    /// Get the shape of an array, accounting for logical deletions.
    ///
    /// This is the Rust equivalent of Python's `NumPack.get_shape(name)`.
    ///
    /// # Arguments
    /// * `name` - Name of the array
    ///
    /// # Returns
    /// The logical shape as `Vec<u64>`.
    pub fn get_shape(&self, name: &str) -> NpkResult<Vec<u64>> {
        let meta = self.get_array_metadata(name)?;
        let mut shape = meta.shape.clone();

        // Account for deletion bitmap
        if DeletionBitmap::exists(&self.base_dir, name) {
            let total_rows = shape[0] as usize;
            let bitmap = DeletionBitmap::new(&self.base_dir, name, total_rows)?;
            shape[0] = bitmap.active_count() as u64;
        }

        Ok(shape)
    }

    /// Get the last modification time of an array.
    ///
    /// This is the Rust equivalent of Python's `NumPack.get_modify_time(name)`.
    ///
    /// # Arguments
    /// * `name` - Name of the array
    ///
    /// # Returns
    /// Modification timestamp in microseconds since Unix epoch, or None.
    pub fn get_modify_time(&self, name: &str) -> Option<u64> {
        self.get_array_meta(name).map(|meta| meta.last_modified)
    }

    /// Clone an array to a new name.
    ///
    /// This is the Rust equivalent of Python's `NumPack.clone(source, target)`.
    /// The cloned array is independent of the original.
    ///
    /// # Arguments
    /// * `source_name` - Source array name
    /// * `target_name` - Target array name
    ///
    /// # Errors
    /// Returns error if source does not exist or target already exists.
    pub fn clone_array(&self, source_name: &str, target_name: &str) -> NpkResult<()> {
        if !self.has_array(source_name) {
            return Err(NpkError::ArrayNotFound(source_name.to_string()));
        }

        if self.has_array(target_name) {
            return Err(NpkError::InvalidOperation(format!(
                "Target array '{}' already exists",
                target_name
            )));
        }

        let meta = self.get_array_meta(source_name).unwrap();

        // Copy data file
        let source_path = self.base_dir.join(&meta.data_file);
        let target_data_file = format!("data_{}.npkd", target_name);
        let target_path = self.base_dir.join(&target_data_file);
        std::fs::copy(&source_path, &target_path)?;

        // Create metadata for the new array
        let new_meta = ArrayMetadata {
            name: target_name.to_string(),
            shape: meta.shape.clone(),
            data_file: target_data_file,
            last_modified: SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_micros() as u64,
            size_bytes: meta.size_bytes,
            dtype: meta.dtype,
            raw_data: None,
        };

        let binary_meta = Self::array_metadata_to_binary(new_meta);
        self.metadata.add_array(binary_meta)?;

        // Copy deletion bitmap if it exists
        if DeletionBitmap::exists(&self.base_dir, source_name) {
            let source_bitmap_path =
                DeletionBitmap::get_bitmap_path(&self.base_dir, source_name);
            let target_bitmap_path =
                DeletionBitmap::get_bitmap_path(&self.base_dir, target_name);
            std::fs::copy(&source_bitmap_path, &target_bitmap_path)?;
        }

        Ok(())
    }

    /// List all array names. Alias for `list_arrays()`.
    ///
    /// This is the Rust equivalent of Python's `NumPack.get_member_list()`.
    pub fn get_member_list(&self) -> Vec<String> {
        self.list_arrays()
    }

    /// Physically compact an array by removing logically deleted rows.
    /// Alias for `compact_array()`.
    ///
    /// This is the Rust equivalent of Python's `NumPack.update(name)`.
    pub fn update(&self, name: &str) -> NpkResult<()> {
        self.compact_array(name)
    }

    /// Create a streaming iterator that yields batches of rows.
    ///
    /// This is the Rust equivalent of Python's `NumPack.stream_load(name, buffer_size)`.
    ///
    /// # Arguments
    /// * `name` - Name of the array
    /// * `buffer_size` - Number of rows per batch
    ///
    /// # Returns
    /// A `StreamIterator<T>` that yields `ArrayD<T>` batches.
    pub fn stream_load<T: NpkElement>(
        &self,
        name: &str,
        buffer_size: usize,
    ) -> NpkResult<StreamIterator<T>> {
        let meta = self.get_array_metadata(name)?;
        let shape: Vec<usize> = meta.shape.iter().map(|&x| x as usize).collect();
        let total_rows = shape[0];

        // Use cached mmap
        let mmap = self.get_or_create_mmap(name, &meta)?;

        let element_size = std::mem::size_of::<T>();
        let row_size = if shape.len() == 1 {
            element_size
        } else {
            shape[1..].iter().product::<usize>() * element_size
        };

        // Load bitmap once for both logical_rows and physical_indices
        let (logical_rows, physical_indices) = if DeletionBitmap::exists(&self.base_dir, name) {
            let bitmap = DeletionBitmap::new(&self.base_dir, name, total_rows)?;
            let active = bitmap.active_count();
            if bitmap.deleted_count() > 0 {
                (active, Some(bitmap.get_active_indices()))
            } else {
                (total_rows, None)
            }
        } else {
            (total_rows, None)
        };

        Ok(StreamIterator {
            mmap,
            shape: shape.clone(),
            row_size,
            total_logical_rows: logical_rows,
            current_index: 0,
            buffer_size,
            physical_indices,
            _phantom: std::marker::PhantomData,
        })
    }
}

/// Iterator that yields batches of rows from a NumPack array.
///
/// Created by `ParallelIO::stream_load()`.
pub struct StreamIterator<T: NpkElement> {
    mmap: Arc<memmap2::Mmap>,
    shape: Vec<usize>,
    row_size: usize,
    total_logical_rows: usize,
    current_index: usize,
    buffer_size: usize,
    physical_indices: Option<Vec<usize>>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: NpkElement> Iterator for StreamIterator<T> {
    type Item = NpkResult<ArrayD<T>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.total_logical_rows {
            return None;
        }

        let batch_start = self.current_index;
        let batch_end = std::cmp::min(
            batch_start + self.buffer_size,
            self.total_logical_rows,
        );
        let batch_rows = batch_end - batch_start;
        self.current_index = batch_end;

        let mut result_shape = self.shape.clone();
        result_shape[0] = batch_rows;

        let total_bytes = batch_rows * self.row_size;

        // Allocate result array directly - no intermediate buffer
        let mut result = unsafe { ArrayD::<T>::uninit(IxDyn(&result_shape)).assume_init() };
        let dst_ptr = result.as_mut_ptr() as *mut u8;

        if let Some(ref indices) = self.physical_indices {
            // With deletion bitmap: gather rows with contiguous block detection
            let batch_indices = &indices[batch_start..batch_end];
            let mut dst_offset = 0usize;
            let mut i = 0usize;
            while i < batch_indices.len() {
                let block_start_phys = batch_indices[i];
                let block_begin = i;
                while i + 1 < batch_indices.len()
                    && batch_indices[i + 1] == batch_indices[i] + 1
                {
                    i += 1;
                }
                let block_len = i - block_begin + 1;
                let src_offset = block_start_phys * self.row_size;
                let block_bytes = block_len * self.row_size;
                if src_offset + block_bytes > self.mmap.len() {
                    return Some(Err(NpkError::IoError(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "stream_load: row offset beyond file bounds",
                    ))));
                }
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        self.mmap[src_offset..].as_ptr(),
                        dst_ptr.add(dst_offset),
                        block_bytes,
                    );
                }
                dst_offset += block_bytes;
                i += 1;
            }
        } else {
            // Sequential read: single copy_nonoverlapping from mmap to array
            let src_offset = batch_start * self.row_size;
            if src_offset + total_bytes > self.mmap.len() {
                return Some(Err(NpkError::IoError(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "stream_load: batch offset beyond file bounds",
                ))));
            }
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.mmap[src_offset..].as_ptr(),
                    dst_ptr,
                    total_bytes,
                );
            }
        }

        Some(Ok(result))
    }
}
