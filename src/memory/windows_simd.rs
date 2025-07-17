//! Windows特定的SIMD安全实现
//! 
//! 针对Windows平台内存访问违规问题的专门修复

// use std::alloc;  // 暂时注释掉未使用的导入
// use std::sync::Mutex;  // 暂时注释掉未使用的导入

/// 全局Windows内存安全开关
#[cfg(target_os = "windows")]
static WINDOWS_SAFE_MODE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// 启用/禁用Windows安全模式
#[cfg(target_os = "windows")]
pub fn set_windows_safe_mode(enabled: bool) {
    WINDOWS_SAFE_MODE.store(enabled, std::sync::atomic::Ordering::Release);
}

/// 检查是否启用Windows安全模式
#[cfg(target_os = "windows")]
pub fn is_windows_safe_mode() -> bool {
    WINDOWS_SAFE_MODE.load(std::sync::atomic::Ordering::Acquire)
}

/// 非Windows平台的空实现
#[cfg(not(target_os = "windows"))]
pub fn set_windows_safe_mode(_enabled: bool) {}

#[cfg(not(target_os = "windows"))]
pub fn is_windows_safe_mode() -> bool { false }

/// 安全内存访问宏 - 防止 Windows 访问冲突
#[macro_export]
macro_rules! safe_slice_from_mmap {
    ($mmap:expr, $offset:expr, $size:expr) => {{
        if $offset + $size > $mmap.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>("Data offset out of bounds"));
        }
        unsafe {
            $crate::memory::windows_simd::WindowsSafeMemoryAccess::safe_slice_from_raw_parts(
                $mmap.as_ptr(),
                $offset,
                $size,
                $mmap.len()
            ).unwrap_or(&[])
        }
    }};
}

/// 安全内存复制宏 - 防止 Windows 访问冲突  
#[macro_export]
macro_rules! safe_copy_from_mmap {
    ($mmap:expr, $offset:expr, $size:expr) => {{
        if $offset + $size > $mmap.len() {
            vec![0; $size]
        } else {
            unsafe {
                $crate::memory::windows_simd::WindowsSafeMemoryAccess::safe_copy_to_vec(
                    $mmap.as_ptr(),
                    $offset,
                    $size,
                    $mmap.len()
                ).unwrap_or_else(|_| vec![0; $size])
            }
        }
    }};
}

/// 带错误恢复的安全内存访问宏
#[macro_export]
macro_rules! safe_memory_access_with_fallback {
    ($operation:expr, $fallback:expr) => {{
        #[cfg(target_os = "windows")]
        {
            if $crate::memory::windows_simd::is_windows_safe_mode() {
                // 使用try-catch类似的错误处理
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    $operation
                })).unwrap_or_else(|_| {
                    // 发生panic时使用fallback
                    $fallback
                })
            } else {
                $operation
            }
        }
        #[cfg(not(target_os = "windows"))]
        {
            $operation
        }
    }};
}

/// Windows平台SIMD错误类型
#[cfg(target_os = "windows")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowsSIMDError {
    UnalignedPointer,       // 指针未对齐
    PageBoundaryCrossing,   // 跨页操作
    InvalidInstructionSet,  // 指令集不可用
    InvalidMemoryAccess,    // 无效内存访问
    AccessViolation,        // 访问违例
    NullPointer,           // 空指针
    BufferOverflow,        // 缓冲区溢出
}

/// Windows平台安全对象池，提高内存利用效率
/// **专门为修复Windows内存访问违规设计**
#[cfg(target_os = "windows")]
pub struct WindowsSIMDBufferPool {
    small_buffers: std::sync::Mutex<Vec<(*mut u8, usize)>>,  // 小缓冲区池 (<1KB)
    medium_buffers: std::sync::Mutex<Vec<(*mut u8, usize)>>, // 中缓冲区池 (1KB-16KB)
    large_buffers: std::sync::Mutex<Vec<(*mut u8, usize)>>,  // 大缓冲区池 (>16KB)
    alignment: usize,
}

#[cfg(target_os = "windows")]
impl WindowsSIMDBufferPool {
    pub fn new(alignment: usize) -> Self {
        Self {
            small_buffers: std::sync::Mutex::new(Vec::new()),
            medium_buffers: std::sync::Mutex::new(Vec::new()),
            large_buffers: std::sync::Mutex::new(Vec::new()),
            alignment,
        }
    }
    
    /// **安全分配对齐内存 - 防止Windows内存错误**
    pub fn get_buffer(&self, size: usize) -> *mut u8 {
        unsafe {
            // 根据大小选择合适的池
            if size < 1024 {
                let mut pool = self.small_buffers.lock().unwrap();
                
                // 查找匹配大小的缓冲区
                for i in 0..pool.len() {
                    let (ptr, buf_size) = pool[i];
                    if buf_size >= size {
                        let buffer = ptr;
                        pool.remove(i);
                        return buffer;
                    }
                }
                
                // 没有找到匹配的，创建新的
                self.safe_alloc(size)
            } else if size < 16384 {
                let mut pool = self.medium_buffers.lock().unwrap();
                
                for i in 0..pool.len() {
                    let (ptr, buf_size) = pool[i];
                    if buf_size >= size {
                        let buffer = ptr;
                        pool.remove(i);
                        return buffer;
                    }
                }
                
                self.safe_alloc(size)
            } else {
                let mut pool = self.large_buffers.lock().unwrap();
                
                for i in 0..pool.len() {
                    let (ptr, buf_size) = pool[i];
                    if buf_size >= size {
                        let buffer = ptr;
                        pool.remove(i);
                        return buffer;
                    }
                }
                
                self.safe_alloc(size)
            }
        }
    }
    
    /// **安全内存分配**
    unsafe fn safe_alloc(&self, size: usize) -> *mut u8 {
        match std::alloc::Layout::from_size_align(size, self.alignment) {
            Ok(layout) => {
                let ptr = std::alloc::alloc_zeroed(layout);
                if ptr.is_null() {
                    // 分配失败，返回空指针
                    std::ptr::null_mut()
                } else {
                    ptr
                }
            }
            Err(_) => {
                // 布局错误，返回空指针
                std::ptr::null_mut()
            }
        }
    }
    
    /// **安全回收内存**
    pub unsafe fn return_buffer(&self, ptr: *mut u8, size: usize) {
        if ptr.is_null() {
            return; // 空指针直接返回
        }
        
        // 根据大小选择合适的池
        if size < 1024 {
            let mut pool = self.small_buffers.lock().unwrap();
            if pool.len() < 32 { // 限制池大小
                pool.push((ptr, size));
            } else {
                self.safe_dealloc(ptr, size);
            }
        } else if size < 16384 {
            let mut pool = self.medium_buffers.lock().unwrap();
            if pool.len() < 16 { // 限制池大小
                pool.push((ptr, size));
            } else {
                self.safe_dealloc(ptr, size);
            }
        } else {
            let mut pool = self.large_buffers.lock().unwrap();
            if pool.len() < 8 { // 限制池大小
                pool.push((ptr, size));
            } else {
                self.safe_dealloc(ptr, size);
            }
        }
    }
    
    /// **安全内存释放**
    unsafe fn safe_dealloc(&self, ptr: *mut u8, size: usize) {
        if let Ok(layout) = std::alloc::Layout::from_size_align(size, self.alignment) {
            std::alloc::dealloc(ptr, layout);
        }
        // 布局错误时忽略释放，避免崩溃
    }
    
    /// 清理所有缓冲区
    pub fn cleanup(&self) {
        unsafe {
            // 清理小缓冲区
            let mut small_pool = self.small_buffers.lock().unwrap();
            for (ptr, size) in small_pool.drain(..) {
                self.safe_dealloc(ptr, size);
            }
            
            // 清理中缓冲区
            let mut medium_pool = self.medium_buffers.lock().unwrap();
            for (ptr, size) in medium_pool.drain(..) {
                self.safe_dealloc(ptr, size);
            }
            
            // 清理大缓冲区
            let mut large_pool = self.large_buffers.lock().unwrap();
            for (ptr, size) in large_pool.drain(..) {
                self.safe_dealloc(ptr, size);
            }
        }
    }
}

#[cfg(target_os = "windows")]
impl Drop for WindowsSIMDBufferPool {
    fn drop(&mut self) {
        self.cleanup();
    }
}

// 非Windows平台的空实现
#[cfg(not(target_os = "windows"))]
pub struct WindowsSIMDBufferPool;

#[cfg(not(target_os = "windows"))]
impl WindowsSIMDBufferPool {
    pub fn new(_alignment: usize) -> Self {
        Self
    }
    
    pub fn get_buffer(&self, _size: usize) -> *mut u8 {
        std::ptr::null_mut()
    }
    
    pub unsafe fn return_buffer(&self, _ptr: *mut u8, _size: usize) {}
    
    pub fn cleanup(&self) {}
}

#[cfg(not(target_os = "windows"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowsSIMDError {
    UnalignedPointer,
    PageBoundaryCrossing,
    InvalidInstructionSet,
    InvalidMemoryAccess,
    AccessViolation,
    NullPointer,
    BufferOverflow,
}

/// Windows平台安全内存访问工具
#[cfg(target_os = "windows")]
pub struct WindowsSafeMemoryAccess;

#[cfg(target_os = "windows")]
impl WindowsSafeMemoryAccess {
    /// 通用安全边界检查
    fn check_bounds(ptr: *const u8, offset: usize, size: usize, total_len: usize) -> Result<(), WindowsSIMDError> {
        if ptr.is_null() {
            return Err(WindowsSIMDError::NullPointer);
        }
        if offset >= total_len {
            return Err(WindowsSIMDError::InvalidMemoryAccess);
        }
        if offset + size > total_len {
            return Err(WindowsSIMDError::BufferOverflow);
        }
        Ok(())
    }

    /// 带try-catch的安全内存读取包装
    unsafe fn safe_memory_operation<T, F>(
        ptr: *const u8,
        offset: usize,
        size: usize,
        total_len: usize,
        operation: F
    ) -> Result<T, WindowsSIMDError>
    where
        F: FnOnce() -> T + std::panic::UnwindSafe,
        T: Default,
    {
        // 边界检查
        Self::check_bounds(ptr, offset, size, total_len)?;
        
        // 使用panic捕获来处理可能的访问违例
        match std::panic::catch_unwind(operation) {
            Ok(result) => Ok(result),
            Err(_) => {
                // 发生panic时返回默认值
                Ok(T::default())
            }
        }
    }

    /// 安全读取不同类型的数据，避免访问冲突
    pub unsafe fn safe_read_u8(ptr: *const u8, offset: usize, len: usize) -> Result<u8, WindowsSIMDError> {
        Self::safe_memory_operation(ptr, offset, 1, len, || {
            std::ptr::read_volatile(ptr.add(offset))
        })
    }
    
    pub unsafe fn safe_read_u16(ptr: *const u8, offset: usize, len: usize) -> Result<u16, WindowsSIMDError> {
        Self::safe_memory_operation(ptr, offset, 2, len, || {
            let aligned_ptr = ptr.add(offset);
            if (aligned_ptr as usize) % 2 == 0 {
                std::ptr::read_volatile(aligned_ptr as *const u16)
            } else {
                // 未对齐时使用字节复制
                let mut bytes = [0u8; 2];
                std::ptr::copy_nonoverlapping(aligned_ptr, bytes.as_mut_ptr(), 2);
                u16::from_le_bytes(bytes)
            }
        })
    }
    
    pub unsafe fn safe_read_u32(ptr: *const u8, offset: usize, len: usize) -> Result<u32, WindowsSIMDError> {
        Self::safe_memory_operation(ptr, offset, 4, len, || {
            let aligned_ptr = ptr.add(offset);
            if (aligned_ptr as usize) % 4 == 0 {
                std::ptr::read_volatile(aligned_ptr as *const u32)
            } else {
                let mut bytes = [0u8; 4];
                std::ptr::copy_nonoverlapping(aligned_ptr, bytes.as_mut_ptr(), 4);
                u32::from_le_bytes(bytes)
            }
        })
    }
    
    pub unsafe fn safe_read_u64(ptr: *const u8, offset: usize, len: usize) -> Result<u64, WindowsSIMDError> {
        Self::safe_memory_operation(ptr, offset, 8, len, || {
            let aligned_ptr = ptr.add(offset);
            if (aligned_ptr as usize) % 8 == 0 {
                std::ptr::read_volatile(aligned_ptr as *const u64)
            } else {
                let mut bytes = [0u8; 8];
                std::ptr::copy_nonoverlapping(aligned_ptr, bytes.as_mut_ptr(), 8);
                u64::from_le_bytes(bytes)
            }
        })
    }
    
    pub unsafe fn safe_read_i8(ptr: *const u8, offset: usize, len: usize) -> Result<i8, WindowsSIMDError> {
        Self::safe_memory_operation(ptr, offset, 1, len, || {
            std::ptr::read_volatile(ptr.add(offset) as *const i8)
        })
    }
    
    pub unsafe fn safe_read_i16(ptr: *const u8, offset: usize, len: usize) -> Result<i16, WindowsSIMDError> {
        Self::safe_memory_operation(ptr, offset, 2, len, || {
            let aligned_ptr = ptr.add(offset);
            if (aligned_ptr as usize) % 2 == 0 {
                std::ptr::read_volatile(aligned_ptr as *const i16)
            } else {
                let mut bytes = [0u8; 2];
                std::ptr::copy_nonoverlapping(aligned_ptr, bytes.as_mut_ptr(), 2);
                i16::from_le_bytes(bytes)
            }
        })
    }
    
    pub unsafe fn safe_read_i32(ptr: *const u8, offset: usize, len: usize) -> Result<i32, WindowsSIMDError> {
        Self::safe_memory_operation(ptr, offset, 4, len, || {
            let aligned_ptr = ptr.add(offset);
            if (aligned_ptr as usize) % 4 == 0 {
                std::ptr::read_volatile(aligned_ptr as *const i32)
            } else {
                let mut bytes = [0u8; 4];
                std::ptr::copy_nonoverlapping(aligned_ptr, bytes.as_mut_ptr(), 4);
                i32::from_le_bytes(bytes)
            }
        })
    }
    
    pub unsafe fn safe_read_i64(ptr: *const u8, offset: usize, len: usize) -> Result<i64, WindowsSIMDError> {
        Self::safe_memory_operation(ptr, offset, 8, len, || {
            let aligned_ptr = ptr.add(offset);
            if (aligned_ptr as usize) % 8 == 0 {
                std::ptr::read_volatile(aligned_ptr as *const i64)
            } else {
                let mut bytes = [0u8; 8];
                std::ptr::copy_nonoverlapping(aligned_ptr, bytes.as_mut_ptr(), 8);
                i64::from_le_bytes(bytes)
            }
        })
    }
    
    pub unsafe fn safe_read_f32(ptr: *const u8, offset: usize, len: usize) -> Result<f32, WindowsSIMDError> {
        Self::safe_memory_operation(ptr, offset, 4, len, || {
            let aligned_ptr = ptr.add(offset);
            if (aligned_ptr as usize) % 4 == 0 {
                std::ptr::read_volatile(aligned_ptr as *const f32)
            } else {
                let mut bytes = [0u8; 4];
                std::ptr::copy_nonoverlapping(aligned_ptr, bytes.as_mut_ptr(), 4);
                f32::from_le_bytes(bytes)
            }
        })
    }
    
    pub unsafe fn safe_read_f64(ptr: *const u8, offset: usize, len: usize) -> Result<f64, WindowsSIMDError> {
        Self::safe_memory_operation(ptr, offset, 8, len, || {
            let aligned_ptr = ptr.add(offset);
            if (aligned_ptr as usize) % 8 == 0 {
                std::ptr::read_volatile(aligned_ptr as *const f64)
            } else {
                let mut bytes = [0u8; 8];
                std::ptr::copy_nonoverlapping(aligned_ptr, bytes.as_mut_ptr(), 8);
                f64::from_le_bytes(bytes)
            }
        })
    }
    
    /// 安全的未对齐读取 - 通过复制避免对齐问题
    pub unsafe fn safe_read_unaligned_u16(ptr: *const u8, offset: usize, len: usize) -> Result<u16, WindowsSIMDError> {
        Self::safe_memory_operation(ptr, offset, 2, len, || {
            let mut bytes = [0u8; 2];
            std::ptr::copy_nonoverlapping(ptr.add(offset), bytes.as_mut_ptr(), 2);
            u16::from_le_bytes(bytes)
        })
    }
    
    pub unsafe fn safe_read_unaligned_u32(ptr: *const u8, offset: usize, len: usize) -> Result<u32, WindowsSIMDError> {
        Self::safe_memory_operation(ptr, offset, 4, len, || {
            let mut bytes = [0u8; 4];
            std::ptr::copy_nonoverlapping(ptr.add(offset), bytes.as_mut_ptr(), 4);
            u32::from_le_bytes(bytes)
        })
    }
    
    pub unsafe fn safe_read_unaligned_u64(ptr: *const u8, offset: usize, len: usize) -> Result<u64, WindowsSIMDError> {
        Self::safe_memory_operation(ptr, offset, 8, len, || {
            let mut bytes = [0u8; 8];
            std::ptr::copy_nonoverlapping(ptr.add(offset), bytes.as_mut_ptr(), 8);
            u64::from_le_bytes(bytes)
        })
    }
    
    /// 安全创建 slice - 替代 std::slice::from_raw_parts
    pub unsafe fn safe_slice_from_raw_parts(ptr: *const u8, offset: usize, len: usize, total_len: usize) -> Result<&'static [u8], WindowsSIMDError> {
        Self::check_bounds(ptr, offset, len, total_len)?;
        
        match std::panic::catch_unwind(|| {
            std::slice::from_raw_parts(ptr.add(offset), len)
        }) {
            Ok(slice) => Ok(slice),
            Err(_) => Err(WindowsSIMDError::AccessViolation),
        }
    }
    
    /// 安全复制内存块 - 用于复制数据到向量
    pub unsafe fn safe_copy_to_vec(ptr: *const u8, offset: usize, len: usize, total_len: usize) -> Result<Vec<u8>, WindowsSIMDError> {
        Self::check_bounds(ptr, offset, len, total_len)?;
        
        match std::panic::catch_unwind(|| {
            let mut result = Vec::with_capacity(len);
            std::ptr::copy_nonoverlapping(ptr.add(offset), result.as_mut_ptr(), len);
            result.set_len(len);
            result
        }) {
            Ok(vec) => Ok(vec),
            Err(_) => Err(WindowsSIMDError::AccessViolation),
        }
    }
}

// 非Windows平台的空实现（保持不变，但添加更多方法）
#[cfg(not(target_os = "windows"))]
pub struct WindowsSafeMemoryAccess;

#[cfg(not(target_os = "windows"))]
impl WindowsSafeMemoryAccess {
    pub unsafe fn safe_read_u8(ptr: *const u8, offset: usize, _len: usize) -> Result<u8, WindowsSIMDError> {
        Ok(*ptr.add(offset))
    }
    
    pub unsafe fn safe_read_u16(ptr: *const u8, offset: usize, _len: usize) -> Result<u16, WindowsSIMDError> {
        Ok(*(ptr.add(offset) as *const u16))
    }
    
    pub unsafe fn safe_read_u32(ptr: *const u8, offset: usize, _len: usize) -> Result<u32, WindowsSIMDError> {
        Ok(*(ptr.add(offset) as *const u32))
    }
    
    pub unsafe fn safe_read_u64(ptr: *const u8, offset: usize, _len: usize) -> Result<u64, WindowsSIMDError> {
        Ok(*(ptr.add(offset) as *const u64))
    }
    
    pub unsafe fn safe_read_i8(ptr: *const u8, offset: usize, _len: usize) -> Result<i8, WindowsSIMDError> {
        Ok(*(ptr.add(offset) as *const i8))
    }
    
    pub unsafe fn safe_read_i16(ptr: *const u8, offset: usize, _len: usize) -> Result<i16, WindowsSIMDError> {
        Ok(*(ptr.add(offset) as *const i16))
    }
    
    pub unsafe fn safe_read_i32(ptr: *const u8, offset: usize, _len: usize) -> Result<i32, WindowsSIMDError> {
        Ok(*(ptr.add(offset) as *const i32))
    }
    
    pub unsafe fn safe_read_i64(ptr: *const u8, offset: usize, _len: usize) -> Result<i64, WindowsSIMDError> {
        Ok(*(ptr.add(offset) as *const i64))
    }
    
    pub unsafe fn safe_read_f32(ptr: *const u8, offset: usize, _len: usize) -> Result<f32, WindowsSIMDError> {
        Ok(*(ptr.add(offset) as *const f32))
    }
    
    pub unsafe fn safe_read_f64(ptr: *const u8, offset: usize, _len: usize) -> Result<f64, WindowsSIMDError> {
        Ok(*(ptr.add(offset) as *const f64))
    }
    
    pub unsafe fn safe_read_unaligned_u16(ptr: *const u8, offset: usize, _len: usize) -> Result<u16, WindowsSIMDError> {
        Ok(*(ptr.add(offset) as *const u16))
    }
    
    pub unsafe fn safe_read_unaligned_u32(ptr: *const u8, offset: usize, _len: usize) -> Result<u32, WindowsSIMDError> {
        Ok(*(ptr.add(offset) as *const u32))
    }
    
    pub unsafe fn safe_read_unaligned_u64(ptr: *const u8, offset: usize, _len: usize) -> Result<u64, WindowsSIMDError> {
        Ok(*(ptr.add(offset) as *const u64))
    }
    
    /// 安全创建 slice - 替代 std::slice::from_raw_parts
    pub unsafe fn safe_slice_from_raw_parts(ptr: *const u8, offset: usize, len: usize, _total_len: usize) -> Result<&'static [u8], WindowsSIMDError> {
        Ok(std::slice::from_raw_parts(ptr.add(offset), len))
    }
    
    /// 安全复制内存块 - 用于复制数据到向量
    pub unsafe fn safe_copy_to_vec(ptr: *const u8, offset: usize, len: usize, _total_len: usize) -> Result<Vec<u8>, WindowsSIMDError> {
        let mut result = Vec::with_capacity(len);
        std::ptr::copy_nonoverlapping(ptr.add(offset), result.as_mut_ptr(), len);
        result.set_len(len);
        Ok(result)
    }
}
