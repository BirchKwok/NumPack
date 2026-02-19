// build.rs - Build script for NumPack

fn main() {
    // Configure pyo3 only when python feature is enabled
    #[cfg(feature = "python")]
    pyo3_build_config::use_pyo3_cfgs();
}
