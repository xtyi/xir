use crate::ast::{Kernel, Module, Parameter, ScheduleNode, ScheduleNodeKind};
use crate::diagnostic::{Diagnostic, DiagnosticBag, DiagnosticCode};
use crate::target::TargetSpec;

/// Native construction API used by the future PyO3 frontend.
#[derive(Debug)]
pub struct ModuleBuilder {
    module: Module,
    diagnostics: DiagnosticBag,
}

#[derive(Clone, Debug, Default)]
pub struct NodeIdGenerator {
    next: u32,
}

impl NodeIdGenerator {
    pub fn next(&mut self, prefix: &str) -> String {
        let id = format!("{}.n{}", prefix, self.next);
        self.next += 1;
        id
    }
}

impl ModuleBuilder {
    pub fn new(name: impl Into<String>, target: TargetSpec) -> Self {
        Self {
            module: Module {
                name: name.into(),
                version: 1,
                target,
                kernels: vec![],
            },
            diagnostics: DiagnosticBag::default(),
        }
    }

    pub fn kernel(&mut self, name: impl Into<String>) -> KernelBuilder<'_> {
        KernelBuilder {
            parent: self,
            kernel: Kernel {
                name: name.into(),
                params: vec![],
                body: vec![],
            },
        }
    }

    pub fn finish(mut self) -> Module {
        for kernel in &mut self.module.kernels {
            let mut ids = NodeIdGenerator::default();
            assign_ids(
                &mut kernel.body,
                &format!("{}.{}", self.module.name, kernel.name),
                &mut ids,
            );
        }
        self.module
    }

    pub fn finish_checked(mut self) -> Result<Module, DiagnosticBag> {
        for kernel in &mut self.module.kernels {
            let mut ids = NodeIdGenerator::default();
            assign_ids(
                &mut kernel.body,
                &format!("{}.{}", self.module.name, kernel.name),
                &mut ids,
            );
        }
        let mut names = std::collections::HashSet::new();
        for kernel in &self.module.kernels {
            if kernel.name.is_empty() {
                self.diagnostics.push(Diagnostic::error(
                    DiagnosticCode::InvalidType,
                    "kernel name cannot be empty",
                ));
            }
            if !names.insert(kernel.name.clone()) {
                self.diagnostics.push(Diagnostic::error(
                    DiagnosticCode::UnknownSymbol,
                    format!("duplicate kernel '{}'", kernel.name),
                ));
            }
            for node in &kernel.body {
                if let ScheduleNode {
                    kind: ScheduleNodeKind::PipelineDecl { name, stages },
                    ..
                } = node
                {
                    if *stages == 0 {
                        self.diagnostics.push(Diagnostic::error(
                            DiagnosticCode::InvalidType,
                            format!("pipeline '{}' must have at least one stage", name),
                        ));
                    }
                }
            }
        }
        if self.diagnostics.has_errors() {
            Err(self.diagnostics)
        } else {
            Ok(self.module)
        }
    }
}

fn assign_ids(nodes: &mut [ScheduleNode], prefix: &str, ids: &mut NodeIdGenerator) {
    for node in nodes {
        if node.meta.node_id.is_empty() {
            node.meta.node_id = ids.next(prefix);
        }
        match &mut node.kind {
            ScheduleNodeKind::RoleRegion { body, .. }
            | ScheduleNodeKind::StageFor { body, .. }
            | ScheduleNodeKind::For { body, .. } => assign_ids(body, &node.meta.node_id, ids),
            ScheduleNodeKind::If {
                then_body,
                else_body,
                ..
            } => {
                assign_ids(then_body, &node.meta.node_id, ids);
                assign_ids(else_body, &node.meta.node_id, ids);
            }
            _ => {}
        }
    }
}

pub struct KernelBuilder<'a> {
    parent: &'a mut ModuleBuilder,
    kernel: Kernel,
}

impl KernelBuilder<'_> {
    pub fn param(mut self, name: impl Into<String>, ty: impl Into<String>) -> Self {
        self.kernel.params.push(Parameter {
            name: name.into(),
            ty: ty.into(),
        });
        self
    }

    pub fn node(mut self, node: ScheduleNode) -> Self {
        self.kernel.body.push(node);
        self
    }

    pub fn body<F>(mut self, build: F) -> Self
    where
        F: FnOnce(&mut BodyBuilder),
    {
        let mut body = BodyBuilder { nodes: vec![] };
        build(&mut body);
        self.kernel.body.extend(body.nodes);
        self
    }

    pub fn buffer(
        self,
        name: impl Into<String>,
        memory_space: impl Into<String>,
        dtype: impl Into<String>,
        shape: Vec<u64>,
        stages: u32,
    ) -> Self {
        self.node(ScheduleNode::new(ScheduleNodeKind::BufferDecl {
            name: name.into(),
            memory_space: memory_space.into(),
            dtype: dtype.into(),
            shape,
            stages,
        }))
    }
    pub fn pipeline(self, name: impl Into<String>, stages: u32) -> Self {
        self.node(ScheduleNode::new(ScheduleNodeKind::PipelineDecl {
            name: name.into(),
            stages,
        }))
    }
    pub fn role(self, name: impl Into<String>, warps: Vec<u32>) -> Self {
        self.node(ScheduleNode::new(ScheduleNodeKind::RoleDecl {
            name: name.into(),
            warps,
        }))
    }
    pub fn async_copy(
        self,
        src: impl Into<String>,
        dst: impl Into<String>,
        barrier: impl Into<String>,
    ) -> Self {
        self.node(ScheduleNode::new(ScheduleNodeKind::AsyncCopy {
            src: src.into(),
            dst: dst.into(),
            barrier: barrier.into(),
        }))
    }
    pub fn wait(self, event: impl Into<String>, stage: Option<String>) -> Self {
        self.node(ScheduleNode::new(ScheduleNodeKind::Wait {
            event: event.into(),
            stage,
        }))
    }
    pub fn sqmma(
        self,
        accumulator: impl Into<String>,
        a: impl Into<String>,
        b: impl Into<String>,
    ) -> Self {
        self.node(ScheduleNode::new(ScheduleNodeKind::Sqmma {
            accumulator: accumulator.into(),
            a: a.into(),
            b: b.into(),
        }))
    }

    pub fn finish(self) {
        self.parent.module.kernels.push(self.kernel);
    }
}

pub struct BodyBuilder {
    nodes: Vec<ScheduleNode>,
}

impl BodyBuilder {
    pub fn async_copy(
        &mut self,
        src: impl Into<String>,
        dst: impl Into<String>,
        barrier: impl Into<String>,
    ) {
        self.nodes
            .push(ScheduleNode::new(ScheduleNodeKind::AsyncCopy {
                src: src.into(),
                dst: dst.into(),
                barrier: barrier.into(),
            }));
    }
    pub fn wait(&mut self, event: impl Into<String>, stage: Option<String>) {
        self.nodes.push(ScheduleNode::new(ScheduleNodeKind::Wait {
            event: event.into(),
            stage,
        }));
    }
    pub fn sqmma(
        &mut self,
        accumulator: impl Into<String>,
        a: impl Into<String>,
        b: impl Into<String>,
    ) {
        self.nodes.push(ScheduleNode::new(ScheduleNodeKind::Sqmma {
            accumulator: accumulator.into(),
            a: a.into(),
            b: b.into(),
        }));
    }
    pub fn assign(&mut self, name: impl Into<String>, value: impl Into<String>) {
        self.nodes.push(ScheduleNode::new(ScheduleNodeKind::Assign {
            name: name.into(),
            value: value.into(),
        }));
    }
    pub fn load(&mut self, result: impl Into<String>, source: impl Into<String>) {
        self.nodes.push(ScheduleNode::new(ScheduleNodeKind::Load {
            result: result.into(),
            source: source.into(),
        }));
    }
    pub fn store(&mut self, destination: impl Into<String>, value: impl Into<String>) {
        self.nodes.push(ScheduleNode::new(ScheduleNodeKind::Store {
            destination: destination.into(),
            value: value.into(),
        }));
    }
    pub fn binary(
        &mut self,
        result: impl Into<String>,
        op: impl Into<String>,
        lhs: impl Into<String>,
        rhs: impl Into<String>,
    ) {
        self.nodes.push(ScheduleNode::new(ScheduleNodeKind::Binary {
            result: result.into(),
            op: op.into(),
            lhs: lhs.into(),
            rhs: rhs.into(),
        }));
    }
    pub fn if_then<F>(&mut self, condition: impl Into<String>, build: F)
    where
        F: FnOnce(&mut BodyBuilder),
    {
        let mut body = BodyBuilder { nodes: vec![] };
        build(&mut body);
        self.nodes.push(ScheduleNode::new(ScheduleNodeKind::If {
            condition: condition.into(),
            then_body: body.nodes,
            else_body: vec![],
        }));
    }
    pub fn for_loop<F>(
        &mut self,
        iterator: impl Into<String>,
        begin: impl Into<String>,
        end: impl Into<String>,
        build: F,
    ) where
        F: FnOnce(&mut BodyBuilder),
    {
        let mut body = BodyBuilder { nodes: vec![] };
        build(&mut body);
        self.nodes.push(ScheduleNode::new(ScheduleNodeKind::For {
            iterator: iterator.into(),
            begin: begin.into(),
            end: end.into(),
            body: body.nodes,
        }));
    }
    pub fn ret(&mut self, value: Option<String>) {
        self.nodes
            .push(ScheduleNode::new(ScheduleNodeKind::Return { value }));
    }
}
