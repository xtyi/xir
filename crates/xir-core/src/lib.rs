//! Core data contracts for the agent-facing XIR schedule compiler.

pub mod ast;
pub mod builder;
pub mod diagnostic;
pub mod target;
pub mod types;

pub use ast::{Kernel, Module};
pub use builder::ModuleBuilder;
pub use diagnostic::{Diagnostic, DiagnosticBag, DiagnosticCode, DiagnosticSeverity, SourceSpan};
pub use target::TargetSpec;
pub use types::{
    AddressSpace, BufferId, BufferRef, BufferType, DType, Dim, EffectSet, FragmentType, IndexType,
    StorageId, StorageType, TypeRef, ValueId,
};

/// Serialize a module using the stable interchange representation.
pub fn serialize_module(module: &Module) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(module)
}

/// Deserialize a module from the stable interchange representation.
pub fn deserialize_module(input: &str) -> Result<Module, serde_json::Error> {
    serde_json::from_str(input)
}

/// Produce a compact, deterministic tree dump for diagnostics and golden tests.
pub fn dump_module(module: &Module) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "Module name={} version={}\n",
        module.name, module.version
    ));
    out.push_str(&format!(
        "  target={} arch={}\n",
        module.target.backend, module.target.arch
    ));
    for kernel in &module.kernels {
        out.push_str(&format!(
            "  Kernel name={} params={} body_nodes={}\n",
            kernel.name,
            kernel.params.len(),
            kernel.body.len()
        ));
    }
    out
}
