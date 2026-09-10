use serde::{Deserialize, Serialize};

/// Requested identity only. Imported modules cannot grant themselves capabilities.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TargetRequest {
    pub backend: String,
    pub arch: String,
}
impl TargetRequest {
    pub fn musa_mp31() -> Self {
        Self {
            backend: "musa".into(),
            arch: "mp31".into(),
        }
    }
    pub fn is_known(&self) -> bool {
        self.backend == "musa" && self.arch == "mp31"
    }
}
impl Default for TargetRequest {
    fn default() -> Self {
        Self::musa_mp31()
    }
}

/// Offline execution geometry for structural checks, not instruction/toolchain admission.
/// Full instruction, descriptor and synchronization contracts remain unimplemented.
pub(crate) const MP31_WARP_SIZE: u32 = 32;
