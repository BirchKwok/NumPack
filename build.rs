// build.rs - Build script for NumPack
use std::env;

fn main() {
    // Configure pyo3
    pyo3_build_config::use_pyo3_cfgs();
    
    // 编译Cap'n Proto schema（如果启用）
    #[cfg(feature = "capnp-metadata")]
    {
        use capnpc::CompilerCommand;
        
        CompilerCommand::new()
            .src_prefix("src")
            .file("src/capnp_metadata.capnp")
            .run()
            .expect("Failed to compile Cap'n Proto schema");
    }
    
    // 告诉cargo在schema文件改变时重新编译
    println!("cargo:rerun-if-changed=src/capnp_metadata.capnp");
}

