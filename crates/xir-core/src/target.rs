use serde::{Deserialize, Serialize};

/// Target capabilities are intentionally data-driven so MUSA architectures can
/// be extended without changing the module interchange format.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TargetSpec {
    pub backend: String,
    pub arch: String,
    pub warp_size: u32,
    pub max_threads_per_block: u32,
    pub memory_spaces: Vec<String>,
    pub async_copy_kinds: Vec<String>,
    pub barrier_scopes: Vec<String>,
    pub mma_kinds: Vec<String>,
    pub supported_dtypes: Vec<String>,
    pub fragment_kinds: Vec<String>,
    pub compiler_features: Vec<String>,
}

impl TargetSpec {
    pub fn musa_mp31() -> Self {
        Self {
            backend: "musa".into(),
            arch: "mp31".into(),
            warp_size: 32,
            max_threads_per_block: 1024,
            memory_spaces: vec![
                "global".into(),
                "shared".into(),
                "local".into(),
                "register".into(),
                "accumulator".into(),
            ],
            async_copy_kinds: vec![],
            barrier_scopes: vec!["warp".into(), "warpgroup".into(), "block".into()],
            mma_kinds: vec!["sqmma".into()],
            supported_dtypes: vec!["f16".into(), "bf16".into(), "f32".into(), "i8".into()],
            fragment_kinds: vec!["musa.sqmma.accumulator".into()],
            compiler_features: vec![],
        }
    }
}

impl Default for TargetSpec {
    fn default() -> Self {
        Self::musa_mp31()
    }
}
