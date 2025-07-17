//! Windows内存安全测试
//! 
//! 测试Windows平台的内存访问安全机制

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::windows_simd::{WindowsSafeMemoryAccess, WindowsSIMDError, set_windows_safe_mode, is_windows_safe_mode};
    use std::panic;

    #[test]
    fn test_windows_safe_mode_toggle() {
        // 测试安全模式开关
        set_windows_safe_mode(true);
        assert!(is_windows_safe_mode());
        
        set_windows_safe_mode(false);
        #[cfg(target_os = "windows")]
        assert!(!is_windows_safe_mode());
        
        #[cfg(not(target_os = "windows"))]
        assert!(!is_windows_safe_mode()); // 非Windows平台总是返回false
    }

    #[test]
    fn test_safe_memory_operations() {
        // 创建测试数据
        let test_data = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let ptr = test_data.as_ptr();
        let len = test_data.len();

        unsafe {
            // 测试安全读取操作
            let result_u8 = WindowsSafeMemoryAccess::safe_read_u8(ptr, 0, len);
            assert!(result_u8.is_ok());
            assert_eq!(result_u8.unwrap(), 1);

            let result_u16 = WindowsSafeMemoryAccess::safe_read_unaligned_u16(ptr, 0, len);
            assert!(result_u16.is_ok());
            
            let result_u32 = WindowsSafeMemoryAccess::safe_read_unaligned_u32(ptr, 0, len);
            assert!(result_u32.is_ok());

            // 测试边界检查
            let result_oob = WindowsSafeMemoryAccess::safe_read_u8(ptr, len, len);
            assert!(result_oob.is_err());

            // 测试安全slice创建
            let slice_result = WindowsSafeMemoryAccess::safe_slice_from_raw_parts(ptr, 0, 8, len);
            assert!(slice_result.is_ok());
            assert_eq!(slice_result.unwrap().len(), 8);

            // 测试安全复制
            let copy_result = WindowsSafeMemoryAccess::safe_copy_to_vec(ptr, 0, 8, len);
            assert!(copy_result.is_ok());
            assert_eq!(copy_result.unwrap().len(), 8);
        }
    }

    #[test]
    fn test_bounds_checking() {
        let test_data = vec![1u8, 2, 3, 4];
        let ptr = test_data.as_ptr();
        let len = test_data.len();

        unsafe {
            // 测试越界访问
            assert!(WindowsSafeMemoryAccess::safe_read_u8(ptr, len, len).is_err());
            assert!(WindowsSafeMemoryAccess::safe_read_u16(ptr, len - 1, len).is_err());
            assert!(WindowsSafeMemoryAccess::safe_read_u32(ptr, len - 3, len).is_err());
            
            // 测试空指针
            assert!(WindowsSafeMemoryAccess::safe_read_u8(std::ptr::null(), 0, 1).is_err());
        }
    }

    #[test]
    fn test_panic_recovery() {
        // 启用Windows安全模式
        set_windows_safe_mode(true);

        // 测试panic恢复机制
        let test_data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let ptr = test_data.as_ptr();
        let len = test_data.len();

        unsafe {
            // 这些操作应该在panic时返回默认值而不是崩溃
            let result = panic::catch_unwind(|| {
                WindowsSafeMemoryAccess::safe_read_u8(ptr, 0, len).unwrap_or(0)
            });
            assert!(result.is_ok());
        }
    }

    #[test]
    #[cfg(target_os = "windows")]
    fn test_windows_specific_features() {
        // 测试Windows特定的功能
        set_windows_safe_mode(true);
        
        let test_data = vec![0u8; 1024];
        let ptr = test_data.as_ptr();
        let len = test_data.len();

        unsafe {
            // 测试大块内存访问
            let large_slice = WindowsSafeMemoryAccess::safe_slice_from_raw_parts(ptr, 0, 512, len);
            assert!(large_slice.is_ok());

            // 测试对齐检查
            let aligned_u32 = WindowsSafeMemoryAccess::safe_read_u32(ptr, 0, len);
            assert!(aligned_u32.is_ok());

            // 测试未对齐访问
            let unaligned_u32 = WindowsSafeMemoryAccess::safe_read_unaligned_u32(ptr, 1, len);
            assert!(unaligned_u32.is_ok());
        }
    }

    #[test]
    fn test_error_types() {
        // 测试不同类型的错误
        let test_data = vec![1u8, 2, 3, 4];
        let ptr = test_data.as_ptr();
        let len = test_data.len();

        unsafe {
            // 测试缓冲区溢出错误
            let overflow_error = WindowsSafeMemoryAccess::safe_read_u64(ptr, 0, len);
            assert!(overflow_error.is_err());

            // 测试无效内存访问错误
            let invalid_access = WindowsSafeMemoryAccess::safe_read_u8(ptr, len + 1, len);
            assert!(invalid_access.is_err());

            // 测试空指针错误
            let null_ptr_error = WindowsSafeMemoryAccess::safe_read_u8(std::ptr::null(), 0, 1);
            assert!(null_ptr_error.is_err());
        }
    }
} 