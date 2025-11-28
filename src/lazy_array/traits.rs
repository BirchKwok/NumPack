//! LazyArray相关trait定义
//!
//! 这个文件将在Task 7中从lazy_array_original.rs中提取相关trait

// TODO: Task 7 - 从lazy_array_original.rs提取完整的trait定义
// 临时定义，让编译通过
pub trait FastTypeConversion {
    fn to_typed_slice<T>(&self) -> &[T]
    where
        T: Copy;
    fn to_typed_vec<T>(&self) -> Vec<T>
    where
        T: Copy;
}

impl FastTypeConversion for Vec<u8> {
    fn to_typed_slice<T>(&self) -> &[T]
    where
        T: Copy,
    {
        unsafe {
            std::slice::from_raw_parts(
                self.as_ptr() as *const T,
                self.len() / std::mem::size_of::<T>(),
            )
        }
    }

    fn to_typed_vec<T>(&self) -> Vec<T>
    where
        T: Copy,
    {
        self.to_typed_slice::<T>().to_vec()
    }
}
