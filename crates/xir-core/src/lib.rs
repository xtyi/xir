//! Typed schedule IR and conservative construction/protocol validation.
pub mod ast;
pub mod builder;
pub mod diagnostic;
pub mod effects;
pub mod target;
pub mod types;
pub mod verify;

pub use ast::{Kernel, Module};
pub use builder::ModuleBuilder;
pub use diagnostic::{Diagnostic, DiagnosticBag, DiagnosticCode, DiagnosticSeverity, SourceSpan};
pub use target::TargetRequest;
pub use verify::{validate_module, verify_module, CheckStatus, VerificationReport};

pub fn serialize_module(module: &Module) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(module)
}

/// Decode only. Use `validate_module` or `ModuleBuilder::finish_checked` before use.
pub fn deserialize_module(input: &str) -> Result<Module, serde_json::Error> {
    serde_json::from_str(input)
}

/// A recursive dump of the canonical schema, including references and source metadata.
pub fn dump_module(module: &Module) -> String {
    serialize_module(module).expect("IR must contain serializable literals")
}
