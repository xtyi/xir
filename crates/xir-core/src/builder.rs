use crate::ast::*;
use crate::diagnostic::DiagnosticBag;
use crate::target::TargetRequest;
use crate::verify::validate_module;
use std::collections::HashSet;

/// Native and imported modules use the same construction checks.
#[derive(Debug)]
pub struct ModuleBuilder {
    module: Module,
}
impl ModuleBuilder {
    pub fn new(name: impl Into<String>, target_request: TargetRequest) -> Self {
        Self {
            module: Module {
                name: name.into(),
                target_request,
                ..Module::default()
            },
        }
    }
    pub fn from_module(module: Module) -> Self {
        Self { module }
    }
    pub fn add_kernel(&mut self, kernel: Kernel) {
        self.module.kernels.push(kernel);
    }
    /// Explicitly unchecked construction is useful for diagnostics and negative tests.
    pub fn finish_unchecked(mut self) -> Module {
        assign_missing_ids(&mut self.module);
        self.module
    }
    /// Construction/protocol checks only; this is not permission to execute a GPU binary.
    pub fn finish_checked(self) -> Result<Module, DiagnosticBag> {
        let module = self.finish_unchecked();
        let diagnostics = validate_module(&module);
        if diagnostics.has_errors() {
            Err(diagnostics)
        } else {
            Ok(module)
        }
    }
}
fn collect_ids(region: &Region, used: &mut HashSet<String>) {
    for node in region {
        if !node.meta.node_id.0.is_empty() {
            used.insert(node.meta.node_id.0.clone());
        }
        for child in node.regions() {
            collect_ids(child, used);
        }
    }
}
fn assign_region(region: &mut Region, prefix: &str, next: &mut u64, used: &mut HashSet<String>) {
    for node in region {
        if node.meta.node_id.0.is_empty() {
            loop {
                let candidate = format!("{prefix}.n{next}");
                *next += 1;
                if used.insert(candidate.clone()) {
                    node.meta.node_id = candidate.into();
                    break;
                }
            }
        }
        for child in node.regions_mut() {
            assign_region(child, prefix, next, used);
        }
    }
}
fn assign_missing_ids(module: &mut Module) {
    let mut used = HashSet::new();
    for kernel in &module.kernels {
        collect_ids(&kernel.body, &mut used);
    }
    let mut next = 0;
    for kernel in &mut module.kernels {
        assign_region(&mut kernel.body, &module.name, &mut next, &mut used);
    }
}
