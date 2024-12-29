use float16::f16;
use numpy::ndarray::{Array, ArrayBase, Data, Ix2};
use pyo3::prelude::*;
use pyo3::types::PyAny;

// 创建包装类型
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct F16Wrapper(pub f16);

// 实现转换 trait
impl From<f16> for F16Wrapper {
    fn from(value: f16) -> Self {
        F16Wrapper(value)
    }
}

impl From<F16Wrapper> for f16 {
    fn from(value: F16Wrapper) -> Self {
        value.0
    }
}

// 实现 FromPyObject trait for F16Wrapper
impl<'source> FromPyObject<'source> for F16Wrapper {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let value = ob.extract::<f32>()?;
        Ok(F16Wrapper(f16::from_f32(value)))
    }
}

// 实现 IntoPy trait for F16Wrapper
impl IntoPy<PyObject> for F16Wrapper {
    fn into_py(self, py: Python) -> PyObject {
        f32::from(self.0).into_py(py)
    }
}

// 转换函数
pub fn array_to_f16(array: ArrayBase<impl Data<Elem = f32>, Ix2>) -> Array<F16Wrapper, Ix2> {
    array.mapv(|x| F16Wrapper(f16::from_f32(x)))
}

pub fn array_from_f16(array: ArrayBase<impl Data<Elem = F16Wrapper>, Ix2>) -> Array<f32, Ix2> {
    array.mapv(|x| f32::from(x.0))
} 