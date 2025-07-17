//! Windows特定的SIMD安全实现
//! 
//! 针对Windows平台内存访问违规问题的专门修复

use std::alloc;
use std::sync::Mutex;

/// Windows平台SIMD错误类型
#[cfg(target_os = "windows")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowsSIMDError {
    UnalignedPointer,       // 指针未对齐
    PageBoundaryCrossing,   // 跨页操作
    InvalidInstructionSet,  // 指令集不可用
    InvalidMemoryAccess,    // 无效内存访问
}

/// Windows平台安全对象池，提高内存利用效率
/// **专门为修复Windows内存访问违规设计**
#[cfg(target_os = "windows")]
pub struct WindowsSIMDBufferPool {
    small_buffers: Mutex<Vec<(*mut u8, usize)>>,  // 小缓冲区池 (<1KB)
    medium_buffers: Mutex<Vec<(*mut u8, usize)>>, // 中缓冲区池 (1KB-16KB)
    large_buffers: Mutex<Vec<(*mut u8, usize)>>,  // 大缓冲区池 (>16KB)
    alignment: usize,
}

#[cfg(target_os = "windows")]
impl WindowsSIMDBufferPool {
    pub fn new(alignment: usize) -> Self {
        Self {
            small_buffers: Mutex::new(Vec::new()),
            medium_buffers: Mutex::new(Vec::new()),
            large_buffers: Mutex::new(Vec::new()),
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
        match alloc::Layout::from_size_align(size, self.alignment) {
            Ok(layout) => {
                let ptr = alloc::alloc_zeroed(layout);
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
        if let Ok(layout) = alloc::Layout::from_size_align(size, self.alignment) {
            alloc::dealloc(ptr, layout);
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
}
