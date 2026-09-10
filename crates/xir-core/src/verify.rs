//! Conservative validation of the implemented structured subset.
//! A successful construction check is not a proof of target lowering or GPU correctness.
use crate::ast::*;
use crate::diagnostic::*;
use crate::effects::{derive_effects, NodeEffects};
use crate::target::MP31_WARP_SIZE;
use crate::types::*;
use serde::Serialize;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub enum CheckStatus {
    Pass,
    Fail,
    Unknown,
}
#[derive(Clone, Debug, Serialize)]
pub struct VerificationReport {
    pub status: CheckStatus,
    pub construction: CheckStatus,
    pub target: CheckStatus,
    pub module_revision: u64,
    pub diagnostics: Vec<Diagnostic>,
    pub effects: Vec<NodeEffects>,
    pub coverage_limits: Vec<String>,
}
pub fn verify_module(module: &Module) -> VerificationReport {
    let diagnostics = validate_module(module);
    let failed = diagnostics.has_errors();
    VerificationReport {
        status: if failed { CheckStatus::Fail } else { CheckStatus::Unknown },
        construction: if failed { CheckStatus::Fail } else { CheckStatus::Pass },
        target: CheckStatus::Unknown,
        module_revision: module.module_revision,
        diagnostics: diagnostics.diagnostics,
        effects: derive_effects(module),
        coverage_limits: vec![
            "Target instruction, representation and synchronization adapters are not implemented.".into(),
            "Dynamic bounds, external alias/extent guards and cross-thread access footprints are not fully verified.".into(),
            "No MUSA lowering, toolchain compilation, runtime validation or performance calibration has been performed.".into(),
        ],
    }
}
pub fn validate_module(module: &Module) -> DiagnosticBag {
    let mut bag = DiagnosticBag::default();
    if module.schema_version != SCHEMA_VERSION {
        bag.push(Diagnostic::error(
            DiagnosticCode::UnsupportedSchema,
            format!(
                "expected schema {SCHEMA_VERSION}, got {}",
                module.schema_version
            ),
        ));
    }
    if module.name.is_empty() {
        bag.push(Diagnostic::error(
            DiagnosticCode::InvalidType,
            "module name cannot be empty",
        ));
    }
    if !module.target_request.is_known() {
        bag.push(Diagnostic::error(
            DiagnosticCode::UnsupportedCapability,
            "unknown exact target; no fallback is permitted",
        ));
    }
    let mut kernel_names = HashSet::new();
    let mut node_ids = HashSet::new();
    for kernel in &module.kernels {
        if kernel.name.is_empty() || !kernel_names.insert(&kernel.name) {
            bag.push(Diagnostic::error(
                DiagnosticCode::DuplicateSymbol,
                "kernel names must be nonempty and unique",
            ));
        }
        let mut verifier = Verifier {
            kernel,
            bag: &mut bag,
            node_ids: &mut node_ids,
            definitions: HashSet::new(),
            event_definitions: HashSet::new(),
            ticket_definitions: HashSet::new(),
            executed_pipelines: HashSet::new(),
        };
        let mut context = Context::default();
        verifier.declarations(&mut context);
        verifier.region(&kernel.body, &mut context, Ending::Kernel);
    }
    bag
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum TicketKind {
    Producer,
    Consumer,
}
#[derive(Clone)]
struct Ticket {
    pipeline: PipelineId,
    kind: TicketKind,
    closed: bool,
}
#[derive(Clone)]
struct Event {
    writes: Option<(BufferId, TicketId)>,
    reads: Vec<MemoryRef>,
    accumulator: Option<AccumulatorId>,
    ready: bool,
    published: bool,
}
#[derive(Clone, Default)]
struct Context {
    values: HashMap<ValueId, TypeRef>,
    literals: HashMap<ValueId, Literal>,
    uniform: HashSet<ValueId>,
    role: Option<RoleId>,
    pipeline: Option<(PipelineId, ValueId)>,
    control_depth: usize,
    tickets: HashMap<TicketId, Ticket>,
    events: HashMap<EventId, Event>,
    initialized: HashSet<AccumulatorId>,
    pending_accumulators: HashMap<AccumulatorId, EventId>,
    loops: HashSet<PipelineId>,
    drained: HashSet<PipelineId>,
}
enum Ending {
    Kernel,
    Yield(Vec<TypeRef>),
    Role,
    Pipeline,
}
struct Verifier<'a> {
    kernel: &'a Kernel,
    bag: &'a mut DiagnosticBag,
    node_ids: &'a mut HashSet<NodeId>,
    definitions: HashSet<ValueId>,
    event_definitions: HashSet<EventId>,
    ticket_definitions: HashSet<TicketId>,
    executed_pipelines: HashSet<PipelineId>,
}
impl Verifier<'_> {
    fn error(
        &mut self,
        node: Option<&ScheduleNode>,
        code: DiagnosticCode,
        message: impl Into<String>,
    ) {
        let mut finding = Diagnostic::error(code, message);
        if let Some(node) = node {
            finding.node_id = Some(node.meta.node_id.clone());
            finding.source_span = node.meta.source_span.clone();
        }
        self.bag.push(finding);
    }
    fn expect(&mut self, ctx: &Context, id: &ValueId, ty: &TypeRef, node: Option<&ScheduleNode>) {
        match ctx.values.get(id) {
            None => self.error(
                node,
                DiagnosticCode::UnknownSymbol,
                format!("value '{}' is not defined in this execution region", id.0),
            ),
            Some(actual) if actual != ty => self.error(
                node,
                DiagnosticCode::InvalidType,
                format!("value '{}' has type {actual:?}, expected {ty:?}", id.0),
            ),
            _ => {}
        }
    }
    fn define(&mut self, ctx: &mut Context, value: &ValueDef, node: Option<&ScheduleNode>) {
        if value.id.0.is_empty() || !self.definitions.insert(value.id.clone()) {
            self.error(
                node,
                DiagnosticCode::DuplicateSymbol,
                format!("duplicate or empty value definition '{}'", value.id.0),
            );
        }
        ctx.values.insert(value.id.clone(), value.ty.clone());
    }
    fn index(&mut self, expr: &IndexExpr, ctx: &Context) {
        let mut ids = vec![];
        expr.values(&mut ids);
        for id in ids {
            self.expect(ctx, &id, &TypeRef::Index, None);
        }
        if let Err(message) = expr.evaluate() {
            self.error(None, DiagnosticCode::InvalidResource, message);
        }
    }
    fn unique(&mut self, names: impl Iterator<Item = String>, category: &str) {
        let mut seen = HashSet::new();
        for name in names {
            if name.is_empty() || !seen.insert(name.clone()) {
                self.error(
                    None,
                    DiagnosticCode::DuplicateSymbol,
                    format!("duplicate or empty {category} '{name}'"),
                );
            }
        }
    }
    fn declarations(&mut self, ctx: &mut Context) {
        for param in &self.kernel.params {
            self.define(ctx, param, None);
            if matches!(param.ty, TypeRef::Fragment(_)) {
                self.error(
                    None,
                    DiagnosticCode::InvalidType,
                    "kernel parameters cannot be distributed fragments",
                );
            }
            ctx.uniform.insert(param.id.clone());
        }
        self.unique(
            self.kernel.storages.iter().map(|s| s.id.0.clone()),
            "storage",
        );
        self.unique(self.kernel.buffers.iter().map(|b| b.id.0.clone()), "buffer");
        self.unique(self.kernel.roles.iter().map(|r| r.id.0.clone()), "role");
        self.unique(
            self.kernel.pipelines.iter().map(|p| p.id.0.clone()),
            "pipeline",
        );
        self.unique(
            self.kernel.accumulators.iter().map(|a| a.id.0.clone()),
            "accumulator",
        );
        let threads = self
            .kernel
            .block
            .iter()
            .try_fold(1u64, |n, d| n.checked_mul(u64::from(*d)));
        if threads.is_none() || self.kernel.block.contains(&0) {
            self.error(
                None,
                DiagnosticCode::InvalidParticipation,
                "invalid or overflowing block dimensions",
            );
        }
        for storage in &self.kernel.storages {
            self.index(&storage.ty.capacity_bytes, ctx);
            if !storage.ty.alignment_bytes.is_power_of_two() {
                self.error(
                    None,
                    DiagnosticCode::InvalidResource,
                    "storage alignment must be a nonzero power of two",
                );
            }
            if let Some(external) = &storage.external {
                match ctx.values.get(external) {
                    Some(TypeRef::Pointer(pointer)) if pointer.address_space == storage.ty.address_space => {}
                    _ => self.error(None, DiagnosticCode::InvalidResource, "external storage must reference a parameter pointer in the same address space"),
                }
            } else if storage.ty.address_space == AddressSpace::Global {
                self.error(
                    None,
                    DiagnosticCode::InvalidResource,
                    "global storage requires an external pointer binding",
                );
            }
        }
        for buffer in &self.kernel.buffers {
            self.index(&buffer.ty.offset_bytes, ctx);
            for dim in &buffer.ty.shape {
                self.index(dim, ctx);
            }
            if let MappingSpec::Strided { strides_bytes } = &buffer.ty.mapping {
                if strides_bytes.len() != buffer.ty.shape.len() {
                    self.error(
                        None,
                        DiagnosticCode::InvalidResource,
                        "buffer rank and stride rank differ",
                    );
                }
                for stride in strides_bytes {
                    self.index(stride, ctx);
                    if stride
                        .evaluate()
                        .ok()
                        .flatten()
                        .is_some_and(|stride| stride % buffer.ty.element.size_bytes() != 0)
                    {
                        self.error(
                            None,
                            DiagnosticCode::InvalidResource,
                            "buffer stride must preserve element alignment",
                        );
                    }
                }
            }
            let element_bytes = buffer.ty.element.size_bytes();
            if buffer
                .ty
                .offset_bytes
                .evaluate()
                .ok()
                .flatten()
                .is_some_and(|offset| offset % element_bytes != 0)
            {
                self.error(
                    None,
                    DiagnosticCode::InvalidResource,
                    "buffer byte offset is not element-aligned",
                );
            }
            if matches!(&buffer.ty.mapping, MappingSpec::Registered { name } if name.is_empty()) {
                self.error(
                    None,
                    DiagnosticCode::InvalidResource,
                    "registered mapping name cannot be empty",
                );
            }
            if let Some(storage) = self
                .kernel
                .storages
                .iter()
                .find(|s| s.id == buffer.ty.storage)
            {
                if let Some(external) = &storage.external {
                    if matches!(ctx.values.get(external), Some(TypeRef::Pointer(p)) if !p.mutable)
                        && buffer.ty.mutable
                    {
                        self.error(
                            None,
                            DiagnosticCode::InvalidResource,
                            "a buffer cannot make read-only external storage writable",
                        );
                    }
                }
                if u64::from(storage.ty.alignment_bytes) < buffer.ty.element.size_bytes() {
                    self.error(
                        None,
                        DiagnosticCode::InvalidResource,
                        "storage alignment is insufficient for the buffer element type",
                    );
                }
                match buffer_end(&buffer.ty, 1, 0) {
                    Err(message) => self.error(None, DiagnosticCode::InvalidResource, message),
                    Ok(Some(end))
                        if storage
                            .ty
                            .capacity_bytes
                            .evaluate()
                            .ok()
                            .flatten()
                            .is_some_and(|size| end > size) =>
                    {
                        self.error(
                            None,
                            DiagnosticCode::InvalidResource,
                            format!("buffer '{}' exceeds its storage capacity", buffer.id.0),
                        );
                    }
                    _ => {}
                }
            } else {
                self.error(
                    None,
                    DiagnosticCode::UnknownSymbol,
                    format!("unknown storage '{}'", buffer.ty.storage.0),
                );
            }
        }
        for role in &self.kernel.roles {
            let mut seen = HashSet::new();
            if role.warps.is_empty() {
                self.error(
                    None,
                    DiagnosticCode::InvalidParticipation,
                    "a role cannot be empty",
                );
            }
            for warp in &role.warps {
                if !seen.insert(warp)
                    || threads
                        .is_some_and(|n| (u64::from(*warp) + 1) * u64::from(MP31_WARP_SIZE) > n)
                {
                    self.error(
                        None,
                        DiagnosticCode::InvalidParticipation,
                        format!("role '{}' has duplicate or out-of-block warps", role.id.0),
                    );
                }
            }
        }
        for accumulator in &self.kernel.accumulators {
            if !self
                .kernel
                .roles
                .iter()
                .any(|r| r.id == accumulator.ty.participants)
            {
                self.error(
                    None,
                    DiagnosticCode::UnknownSymbol,
                    "accumulator has an unknown participant role",
                );
            }
            for dim in &accumulator.ty.logical_shape {
                self.index(dim, ctx);
            }
            if accumulator.ty.representation.is_empty() {
                self.error(
                    None,
                    DiagnosticCode::InvalidType,
                    "accumulator requires an explicit representation",
                );
            }
        }
        let mut owned_storages = HashMap::new();
        for pipe in &self.kernel.pipelines {
            for binding in &pipe.buffers {
                if let Some(buffer) = self.kernel.buffers.iter().find(|b| b.id == binding.buffer) {
                    if owned_storages
                        .insert(buffer.ty.storage.clone(), pipe.id.clone())
                        .is_some_and(|owner| owner != pipe.id)
                    {
                        self.error(
                            None,
                            DiagnosticCode::InvalidPipeline,
                            "sharing storage between pipelines is not supported",
                        );
                    }
                }
            }
            self.index(&pipe.iteration_count, ctx);
            if pipe.slots == 0 || pipe.buffers.is_empty() || pipe.producer == pipe.consumer {
                self.error(None, DiagnosticCode::InvalidPipeline, "pipeline requires positive slots, buffers and distinct producer/consumer roles");
            }
            for role in [&pipe.producer, &pipe.consumer] {
                if !self.kernel.roles.iter().any(|r| &r.id == role) {
                    self.error(
                        None,
                        DiagnosticCode::UnknownSymbol,
                        format!("pipeline references unknown role '{}'", role.0),
                    );
                }
            }
            for (position, a) in pipe.buffers.iter().enumerate() {
                for b in &pipe.buffers[position + 1..] {
                    let a_buffer = self.kernel.buffers.iter().find(|v| v.id == a.buffer);
                    let b_buffer = self.kernel.buffers.iter().find(|v| v.id == b.buffer);
                    if let (Some(ab), Some(bb)) = (a_buffer, b_buffer) {
                        if ab.ty.storage == bb.ty.storage {
                            // Bounding envelopes are conservative for interleaved slot layouts.
                            match (ab.ty.offset_bytes.evaluate(), buffer_end(&ab.ty, pipe.slots, a.slot_stride_bytes),
                                   bb.ty.offset_bytes.evaluate(), buffer_end(&bb.ty, pipe.slots, b.slot_stride_bytes)) {
                                (Ok(Some(a0)), Ok(Some(a1)), Ok(Some(b0)), Ok(Some(b1))) if a1 <= b0 || b1 <= a0 => {},
                                _ => self.error(None, DiagnosticCode::InvalidPipeline, "pipeline buffer slot envelopes may overlap; use separate storage or disjoint envelopes"),
                            }
                        }
                    }
                }
            }
            let mut bound = HashSet::new();
            for binding in &pipe.buffers {
                if !bound.insert(binding.buffer.clone()) {
                    self.error(
                        None,
                        DiagnosticCode::InvalidPipeline,
                        "a buffer is bound twice to a pipeline",
                    );
                }
                let Some(buffer) = self.kernel.buffers.iter().find(|b| b.id == binding.buffer)
                else {
                    self.error(
                        None,
                        DiagnosticCode::UnknownSymbol,
                        "pipeline references unknown buffer",
                    );
                    continue;
                };
                let Some(storage) = self
                    .kernel
                    .storages
                    .iter()
                    .find(|s| s.id == buffer.ty.storage)
                else {
                    continue;
                };
                if storage.ty.address_space != AddressSpace::Shared || !buffer.ty.mutable {
                    self.error(
                        None,
                        DiagnosticCode::InvalidPipeline,
                        "pipeline buffers must use writable shared storage",
                    );
                }
                if binding.slot_stride_bytes == 0
                    || u64::from(storage.ty.alignment_bytes) == 0
                    || binding.slot_stride_bytes % u64::from(storage.ty.alignment_bytes).max(1) != 0
                {
                    self.error(
                        None,
                        DiagnosticCode::InvalidPipeline,
                        "slot stride must be positive and preserve storage alignment",
                    );
                }
                if footprint(&buffer.ty)
                    .ok()
                    .flatten()
                    .is_some_and(|size| size > binding.slot_stride_bytes)
                {
                    self.error(
                        None,
                        DiagnosticCode::InvalidPipeline,
                        "pipeline slots overlap",
                    );
                }
                match buffer_end(&buffer.ty, pipe.slots, binding.slot_stride_bytes) {
                    Err(message) => self.error(None, DiagnosticCode::InvalidPipeline, message),
                    Ok(Some(end))
                        if storage
                            .ty
                            .capacity_bytes
                            .evaluate()
                            .ok()
                            .flatten()
                            .is_some_and(|size| end > size) =>
                    {
                        self.error(
                            None,
                            DiagnosticCode::InvalidPipeline,
                            "staged buffer exceeds storage capacity",
                        );
                    }
                    _ => {}
                }
            }
        }
    }

    fn region(&mut self, body: &Region, ctx: &mut Context, ending: Ending) {
        for (position, node) in body.iter().enumerate() {
            if node.meta.node_id.0.is_empty() || !self.node_ids.insert(node.meta.node_id.clone()) {
                self.error(
                    Some(node),
                    DiagnosticCode::DuplicateSymbol,
                    "node IDs must be nonempty and unique across the module",
                );
            }
            for operand in node.operands() {
                if !ctx.values.contains_key(operand) {
                    self.error(
                        Some(node),
                        DiagnosticCode::UnknownSymbol,
                        format!("value '{}' does not dominate this use", operand.0),
                    );
                }
            }
            use ScheduleNodeKind::*;
            match &node.kind {
                KernelReturn => {
                    if !matches!(ending, Ending::Kernel) || position + 1 != body.len() {
                        self.error(
                            Some(node),
                            DiagnosticCode::InvalidControlFlow,
                            "KernelReturn is only allowed at the end of the outer kernel region",
                        );
                    }
                }
                Yield { values } => {
                    if let Ending::Yield(expected) = &ending {
                        if position + 1 != body.len() || values.len() != expected.len() {
                            self.error(
                                Some(node),
                                DiagnosticCode::InvalidControlFlow,
                                "invalid region yield arity or position",
                            );
                        }
                        for (value, ty) in values.iter().zip(expected) {
                            self.expect(ctx, value, ty, Some(node));
                        }
                    } else {
                        self.error(
                            Some(node),
                            DiagnosticCode::InvalidControlFlow,
                            "Yield is not valid in this region",
                        );
                    }
                }
                Constant { value, result } => {
                    if !literal_matches(value, &result.ty) {
                        self.error(
                            Some(node),
                            DiagnosticCode::InvalidType,
                            "literal does not fit the declared result type",
                        );
                    }
                    ctx.literals.insert(result.id.clone(), value.clone());
                    ctx.uniform.insert(result.id.clone());
                }
                Binary {
                    op,
                    lhs,
                    rhs,
                    result,
                } => self.binary(ctx, node, op, lhs, rhs, result),
                Load {
                    source,
                    indices,
                    mask,
                    masked_value,
                    result,
                } => {
                    let buffer = self.memory(ctx, node, source, false);
                    self.expect(ctx, mask, &TypeRef::Predicate, Some(node));
                    self.expect(ctx, masked_value, &result.ty, Some(node));
                    self.indices(ctx, node, indices, buffer.as_ref());
                    if let Some(buffer) = buffer {
                        if result.ty != TypeRef::Scalar(buffer.ty.element) {
                            self.error(
                                Some(node),
                                DiagnosticCode::InvalidType,
                                "scalar load result must match buffer dtype",
                            );
                        }
                    }
                }
                Store {
                    destination,
                    indices,
                    value,
                    mask,
                } => {
                    let buffer = self.memory(ctx, node, destination, true);
                    self.expect(ctx, mask, &TypeRef::Predicate, Some(node));
                    self.indices(ctx, node, indices, buffer.as_ref());
                    if let Some(buffer) = buffer {
                        self.expect(ctx, value, &TypeRef::Scalar(buffer.ty.element), Some(node));
                    }
                }
                If {
                    condition,
                    results,
                    then_body,
                    else_body,
                } => {
                    self.expect(ctx, condition, &TypeRef::Predicate, Some(node));
                    let types: Vec<_> = results.iter().map(|v| v.ty.clone()).collect();
                    let mut left = ctx.clone();
                    left.control_depth += 1;
                    let mut right = left.clone();
                    self.region(then_body, &mut left, Ending::Yield(types.clone()));
                    self.region(else_body, &mut right, Ending::Yield(types));
                    // Runtime branch results are conservatively non-uniform.
                }
                For {
                    lower,
                    upper,
                    step,
                    induction,
                    carried,
                    body,
                } => {
                    for bound in [lower, upper, step] {
                        self.expect(ctx, bound, &TypeRef::Index, Some(node));
                    }
                    if !matches!(ctx.literals.get(step), Some(Literal::Index(n)) if *n > 0) {
                        self.error(
                            Some(node),
                            DiagnosticCode::UnsupportedSyntax,
                            "the supported For subset requires a constant positive index step",
                        );
                    }
                    let mut child = ctx.clone();
                    child.control_depth += 1;
                    if induction.ty != TypeRef::Index {
                        self.error(
                            Some(node),
                            DiagnosticCode::InvalidType,
                            "For induction must be Index",
                        );
                    }
                    self.define(&mut child, induction, Some(node));
                    if [lower, upper, step]
                        .iter()
                        .all(|id| ctx.uniform.contains(*id))
                    {
                        child.uniform.insert(induction.id.clone());
                    }
                    for item in carried {
                        self.expect(ctx, &item.initial, &item.argument.ty, Some(node));
                        if item.argument.ty != item.result.ty {
                            self.error(
                                Some(node),
                                DiagnosticCode::InvalidType,
                                "loop-carried argument and result types differ",
                            );
                        }
                        self.define(&mut child, &item.argument, Some(node));
                    }
                    self.region(
                        body,
                        &mut child,
                        Ending::Yield(carried.iter().map(|v| v.result.ty.clone()).collect()),
                    );
                }
                ConcurrentRoles { roles, join_scope } => {
                    self.concurrent(ctx, node, roles, join_scope)
                }
                PipelineFor {
                    pipeline,
                    iteration,
                    body,
                } => self.pipeline_loop(ctx, node, pipeline, iteration, body),
                PipelineAcquire {
                    pipeline,
                    iteration,
                    ticket,
                } => self.acquire(ctx, node, pipeline, iteration, ticket, TicketKind::Producer),
                PipelineConsume {
                    pipeline,
                    iteration,
                    ticket,
                } => self.acquire(ctx, node, pipeline, iteration, ticket, TicketKind::Consumer),
                PipelinePublish { ticket, events } => self.publish(ctx, node, ticket, events),
                PipelineRelease { ticket } => self.release(ctx, node, ticket),
                PipelineDrain { pipeline } => {
                    if ctx.control_depth != 0
                        || ctx.pipeline.is_some()
                        || !ctx.loops.contains(pipeline)
                    {
                        self.error(
                            Some(node),
                            DiagnosticCode::InvalidPipeline,
                            "drain must follow the matching role's complete pipeline loop",
                        );
                    }
                    ctx.drained.insert(pipeline.clone());
                }
                AsyncCopy {
                    source,
                    destination,
                    transfer,
                    completion,
                } => {
                    self.protocol_context(ctx, node);
                    let src = self.memory(ctx, node, source, false);
                    let dst = self.memory(ctx, node, destination, true);
                    if self.same_storage(source, destination) {
                        self.error(
                            Some(node),
                            DiagnosticCode::InvalidResource,
                            "AsyncCopy source and destination may overlap",
                        );
                    }
                    if transfer.0.is_empty() {
                        self.error(
                            Some(node),
                            DiagnosticCode::UnsupportedCapability,
                            "transfer requires an explicit target spec",
                        );
                    }
                    if let (Some(src), Some(dst)) = (src, dst) {
                        if src.ty.shape != dst.ty.shape || src.ty.element != dst.ty.element {
                            self.error(
                                Some(node),
                                DiagnosticCode::InvalidType,
                                "whole-view AsyncCopy requires matching shapes and dtypes",
                            );
                        }
                    }
                    let writes = match destination {
                        MemoryRef::Leased { buffer, ticket } => {
                            Some((buffer.clone(), ticket.clone()))
                        }
                        _ => {
                            self.error(
                                Some(node),
                                DiagnosticCode::InvalidTicket,
                                "this AsyncCopy subset requires a producer-leased destination",
                            );
                            None
                        }
                    };
                    for event in ctx.events.values() {
                        if !event.ready && event.writes == writes && writes.is_some() {
                            self.error(
                                Some(node),
                                DiagnosticCode::PendingAccess,
                                "an earlier async write to this leased buffer is still pending",
                            );
                        }
                    }
                    self.event(
                        ctx,
                        node,
                        completion,
                        Event {
                            writes,
                            reads: vec![source.clone()],
                            accumulator: None,
                            ready: false,
                            published: false,
                        },
                    );
                }
                AccumulatorInit { accumulator, value } => {
                    self.accumulator(ctx, node, accumulator);
                    if ctx.control_depth != 0 || ctx.pending_accumulators.contains_key(accumulator)
                    {
                        self.error(Some(node), DiagnosticCode::PendingAccess, "accumulator initialization requires unconditional execution and no pending update");
                    }
                    if let Some(acc) = self
                        .kernel
                        .accumulators
                        .iter()
                        .find(|a| &a.id == accumulator)
                    {
                        self.expect(
                            ctx,
                            value,
                            &TypeRef::Scalar(acc.ty.element.clone()),
                            Some(node),
                        );
                    }
                    ctx.initialized.insert(accumulator.clone());
                }
                AccumulatorRead {
                    accumulator,
                    result,
                } => {
                    self.accumulator(ctx, node, accumulator);
                    self.ready_accumulator(ctx, node, accumulator);
                    if let Some(acc) = self
                        .kernel
                        .accumulators
                        .iter()
                        .find(|a| &a.id == accumulator)
                    {
                        if result.ty != TypeRef::Fragment(acc.ty.clone()) {
                            self.error(
                                Some(node),
                                DiagnosticCode::InvalidType,
                                "accumulator read result must preserve its fragment type",
                            );
                        }
                    }
                }
                Sqmma {
                    accumulator,
                    a,
                    b,
                    instruction,
                    completion,
                } => {
                    self.protocol_context(ctx, node);
                    self.accumulator(ctx, node, accumulator);
                    self.ready_accumulator(ctx, node, accumulator);
                    for input in [a, b] {
                        if let Some(buffer) = self.memory(ctx, node, input, false) {
                            if !self.kernel.storages.iter().any(|s| {
                                s.id == buffer.ty.storage
                                    && s.ty.address_space == AddressSpace::Shared
                            }) {
                                self.error(
                                    Some(node),
                                    DiagnosticCode::InvalidResource,
                                    "the supported SQMMA SS contract requires shared-memory inputs",
                                );
                            }
                        }
                    }
                    if instruction.0.is_empty() {
                        self.error(
                            Some(node),
                            DiagnosticCode::UnsupportedCapability,
                            "SQMMA requires an explicit instruction spec",
                        );
                    }
                    if let Some(role) = self
                        .kernel
                        .roles
                        .iter()
                        .find(|r| Some(&r.id) == ctx.role.as_ref())
                    {
                        if role.warps.len() != 4
                            || role.warps[0] % 4 != 0
                            || !role.warps.windows(2).all(|p| p[1] == p[0] + 1)
                        {
                            self.error(Some(node), DiagnosticCode::InvalidParticipation, "MP31 SQMMA requires an aligned group of four consecutive warps in the supported subset");
                        }
                    }
                    self.event(
                        ctx,
                        node,
                        completion,
                        Event {
                            writes: None,
                            reads: vec![a.clone(), b.clone()],
                            accumulator: Some(accumulator.clone()),
                            ready: false,
                            published: false,
                        },
                    );
                    ctx.pending_accumulators
                        .insert(accumulator.clone(), completion.clone());
                }
                AwaitEvent { event } => {
                    self.protocol_context(ctx, node);
                    match ctx.events.get_mut(event) {
                        Some(info) => {
                            info.ready = true;
                            if let Some(accumulator) = &info.accumulator {
                                if ctx.pending_accumulators.get(accumulator) == Some(event) {
                                    ctx.pending_accumulators.remove(accumulator);
                                }
                            }
                        }
                        None => self.error(
                            Some(node),
                            DiagnosticCode::UnknownSymbol,
                            "event is not defined in this role/iteration",
                        ),
                    }
                }
            }
            for result in node.results() {
                self.define(ctx, result, Some(node));
            }
        }
        match ending {
            Ending::Kernel
                if !matches!(
                    body.last().map(|n| &n.kind),
                    Some(ScheduleNodeKind::KernelReturn)
                ) =>
            {
                self.error(
                    body.last(),
                    DiagnosticCode::InvalidControlFlow,
                    "kernel must end with KernelReturn",
                );
            }
            Ending::Yield(_)
                if !matches!(
                    body.last().map(|n| &n.kind),
                    Some(ScheduleNodeKind::Yield { .. })
                ) =>
            {
                self.error(
                    body.last(),
                    DiagnosticCode::InvalidControlFlow,
                    "structured value region must end with Yield",
                );
            }
            _ => {}
        }
    }

    fn binary(
        &mut self,
        ctx: &mut Context,
        node: &ScheduleNode,
        op: &BinaryOp,
        lhs: &ValueId,
        rhs: &ValueId,
        result: &ValueDef,
    ) {
        let Some(ty) = ctx.values.get(lhs).cloned() else {
            return;
        };
        self.expect(ctx, rhs, &ty, Some(node));
        let numeric = matches!(
            ty,
            TypeRef::Index
                | TypeRef::Scalar(
                    DType::I32
                        | DType::I64
                        | DType::U32
                        | DType::U64
                        | DType::F16
                        | DType::BF16
                        | DType::F32
                        | DType::F64
                )
        );
        let expected = match op {
            BinaryOp::And | BinaryOp::Or if ty == TypeRef::Predicate => Some(TypeRef::Predicate),
            BinaryOp::Equal if numeric || ty == TypeRef::Predicate => Some(TypeRef::Predicate),
            BinaryOp::Less if numeric => Some(TypeRef::Predicate),
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div if numeric => {
                Some(ty.clone())
            }
            _ => None,
        };
        if expected.as_ref() != Some(&result.ty) {
            self.error(
                Some(node),
                DiagnosticCode::InvalidType,
                "invalid binary operand or result types",
            );
        }
        if ctx.uniform.contains(lhs) && ctx.uniform.contains(rhs) {
            ctx.uniform.insert(result.id.clone());
        }
        if matches!(op, BinaryOp::Div)
            && matches!(
                ctx.literals.get(rhs),
                Some(Literal::Index(0) | Literal::Integer(0))
            )
        {
            self.error(Some(node), DiagnosticCode::InvalidType, "division by zero");
        }
        if let (Some(Literal::Index(a)), Some(Literal::Index(b))) =
            (ctx.literals.get(lhs), ctx.literals.get(rhs))
        {
            let folded = match op {
                BinaryOp::Add => a.checked_add(*b),
                BinaryOp::Sub => a.checked_sub(*b),
                BinaryOp::Mul => a.checked_mul(*b),
                BinaryOp::Div => a.checked_div(*b),
                _ => return,
            };
            if let Some(value) = folded {
                ctx.literals
                    .insert(result.id.clone(), Literal::Index(value));
            } else {
                self.error(
                    Some(node),
                    DiagnosticCode::InvalidType,
                    "constant index arithmetic overflows or divides by zero",
                );
            }
        }
    }

    fn indices(
        &mut self,
        ctx: &Context,
        node: &ScheduleNode,
        indices: &[ValueId],
        buffer: Option<&BufferDecl>,
    ) {
        for index in indices {
            self.expect(ctx, index, &TypeRef::Index, Some(node));
        }
        if buffer.is_some_and(|buffer| indices.len() != buffer.ty.shape.len()) {
            self.error(
                Some(node),
                DiagnosticCode::InvalidType,
                "access index count must match buffer rank",
            );
        }
    }

    fn memory(
        &mut self,
        ctx: &Context,
        node: &ScheduleNode,
        reference: &MemoryRef,
        write: bool,
    ) -> Option<BufferDecl> {
        let Some(buffer) = self
            .kernel
            .buffers
            .iter()
            .find(|b| &b.id == reference.buffer_id())
            .cloned()
        else {
            self.error(
                Some(node),
                DiagnosticCode::UnknownSymbol,
                "unknown buffer reference",
            );
            return None;
        };
        if write && !buffer.ty.mutable {
            self.error(
                Some(node),
                DiagnosticCode::InvalidResource,
                "write through a read-only buffer",
            );
        }
        match reference {
            MemoryRef::Buffer(_) => {
                // Storage identity also catches alternate views that bypass a pipeline lease.
                let pipeline_owned =
                    self.kernel
                        .pipelines
                        .iter()
                        .flat_map(|p| &p.buffers)
                        .any(|binding| {
                            self.kernel.buffers.iter().any(|b| {
                                b.id == binding.buffer && b.ty.storage == buffer.ty.storage
                            })
                        });
                if pipeline_owned {
                    self.error(
                        Some(node),
                        DiagnosticCode::InvalidTicket,
                        "pipeline storage must be accessed through a matching live ticket",
                    );
                }
            }
            MemoryRef::Leased { buffer: id, ticket } => {
                match ctx.tickets.get(ticket) {
                    Some(info) if !info.closed => {
                        if ctx.pipeline.as_ref().map(|p| &p.0) != Some(&info.pipeline) {
                            self.error(
                                Some(node),
                                DiagnosticCode::InvalidTicket,
                                "ticket belongs to a different pipeline iteration",
                            );
                        }
                        if !self.kernel.pipelines.iter().any(|p| {
                            p.id == info.pipeline && p.buffers.iter().any(|b| &b.buffer == id)
                        }) {
                            self.error(
                                Some(node),
                                DiagnosticCode::InvalidTicket,
                                "buffer is not bound to this ticket's pipeline",
                            );
                        }
                        let expected = if write {
                            TicketKind::Producer
                        } else {
                            TicketKind::Consumer
                        };
                        if info.kind != expected {
                            self.error(
                                Some(node),
                                DiagnosticCode::InvalidTicket,
                                "supported leases allow producer writes and consumer reads only",
                            );
                        }
                    }
                    _ => self.error(
                        Some(node),
                        DiagnosticCode::InvalidTicket,
                        "ticket is undefined or already closed",
                    ),
                }
            }
        }
        for event in ctx.events.values().filter(|e| !e.ready) {
            if write && event.reads.iter().any(|r| self.same_storage(r, reference)) {
                self.error(
                    Some(node),
                    DiagnosticCode::PendingAccess,
                    "write may overlap a pending async read",
                );
            }
            if let Some((destination, _)) = &event.writes {
                if self.same_storage(&MemoryRef::Buffer(destination.clone()), reference) {
                    self.error(
                        Some(node),
                        DiagnosticCode::PendingAccess,
                        "access may overlap a pending async write",
                    );
                }
            }
        }
        Some(buffer)
    }

    fn same_storage(&self, a: &MemoryRef, b: &MemoryRef) -> bool {
        let lookup = |id: &BufferId| {
            self.kernel
                .buffers
                .iter()
                .find(|buffer| &buffer.id == id)
                .map(|buffer| &buffer.ty.storage)
        };
        match (lookup(a.buffer_id()), lookup(b.buffer_id())) {
            (Some(sa), Some(sb)) if sa == sb => {
                let a = self
                    .kernel
                    .buffers
                    .iter()
                    .find(|v| &v.id == a.buffer_id())
                    .unwrap();
                let b = self
                    .kernel
                    .buffers
                    .iter()
                    .find(|v| &v.id == b.buffer_id())
                    .unwrap();
                !views_disjoint(&a.ty, &b.ty)
            }
            _ => false,
        }
    }

    fn protocol_context(&mut self, ctx: &Context, node: &ScheduleNode) {
        if ctx.pipeline.is_none() || ctx.control_depth != 0 {
            self.error(Some(node), DiagnosticCode::UnsupportedSyntax, "async protocol operations require an unconditional PipelineFor body in the supported subset");
        }
    }

    fn accumulator(&mut self, ctx: &Context, node: &ScheduleNode, id: &AccumulatorId) {
        match self.kernel.accumulators.iter().find(|a| &a.id == id) {
            None => self.error(
                Some(node),
                DiagnosticCode::UnknownSymbol,
                "unknown accumulator",
            ),
            Some(acc)
                if Some(&acc.ty.participants) != ctx.role.as_ref() || ctx.control_depth != 0 =>
            {
                self.error(
                    Some(node),
                    DiagnosticCode::InvalidParticipation,
                    "accumulator operation must execute unconditionally in its participant role",
                );
            }
            _ => {}
        }
    }

    fn ready_accumulator(&mut self, ctx: &Context, node: &ScheduleNode, id: &AccumulatorId) {
        if !ctx.initialized.contains(id) {
            self.error(
                Some(node),
                DiagnosticCode::InvalidResource,
                "accumulator is not initialized",
            );
        }
        if ctx.pending_accumulators.contains_key(id) {
            self.error(
                Some(node),
                DiagnosticCode::PendingAccess,
                "accumulator has a pending update; await its event first",
            );
        }
    }

    fn event(&mut self, ctx: &mut Context, node: &ScheduleNode, id: &EventId, event: Event) {
        if id.0.is_empty() || !self.event_definitions.insert(id.clone()) {
            self.error(
                Some(node),
                DiagnosticCode::DuplicateSymbol,
                "event IDs must be nonempty and unique within a kernel",
            );
        }
        ctx.events.insert(id.clone(), event);
    }

    fn acquire(
        &mut self,
        ctx: &mut Context,
        node: &ScheduleNode,
        pipeline: &PipelineId,
        iteration: &ValueId,
        ticket: &TicketId,
        kind: TicketKind,
    ) {
        self.protocol_context(ctx, node);
        self.expect(ctx, iteration, &TypeRef::Index, Some(node));
        if ctx.pipeline.as_ref() != Some(&(pipeline.clone(), iteration.clone())) {
            self.error(
                Some(node),
                DiagnosticCode::InvalidTicket,
                "ticket must use the enclosing pipeline and logical iteration",
            );
        }
        if let Some(pipe) = self.kernel.pipelines.iter().find(|p| &p.id == pipeline) {
            let expected = if kind == TicketKind::Producer {
                &pipe.producer
            } else {
                &pipe.consumer
            };
            if ctx.role.as_ref() != Some(expected) {
                self.error(
                    Some(node),
                    DiagnosticCode::InvalidParticipation,
                    "pipeline ticket requested by the wrong role",
                );
            }
        } else {
            self.error(
                Some(node),
                DiagnosticCode::UnknownSymbol,
                "unknown pipeline",
            );
        }
        if ticket.0.is_empty() || !self.ticket_definitions.insert(ticket.clone()) {
            self.error(
                Some(node),
                DiagnosticCode::DuplicateSymbol,
                "ticket IDs must be nonempty and unique within a kernel",
            );
        }
        ctx.tickets.insert(
            ticket.clone(),
            Ticket {
                pipeline: pipeline.clone(),
                kind,
                closed: false,
            },
        );
    }

    fn publish(
        &mut self,
        ctx: &mut Context,
        node: &ScheduleNode,
        ticket: &TicketId,
        events: &[EventId],
    ) {
        self.protocol_context(ctx, node);
        let Some(info) = ctx.tickets.get(ticket).cloned() else {
            self.error(
                Some(node),
                DiagnosticCode::InvalidTicket,
                "publish uses an undefined ticket",
            );
            return;
        };
        if info.closed || info.kind != TicketKind::Producer {
            self.error(
                Some(node),
                DiagnosticCode::InvalidTicket,
                "publish requires a live producer ticket",
            );
        }
        let mut seen = HashSet::new();
        let mut writes = HashSet::new();
        for id in events {
            if !seen.insert(id) {
                self.error(
                    Some(node),
                    DiagnosticCode::InvalidPipeline,
                    "publish lists an event more than once",
                );
            }
            match ctx.events.get_mut(id) {
                Some(event)
                    if !event.published
                        && event
                            .writes
                            .as_ref()
                            .is_some_and(|(_, owner)| owner == ticket) =>
                {
                    let buffer = event.writes.as_ref().unwrap().0.clone();
                    if !writes.insert(buffer) {
                        self.error(
                            Some(node),
                            DiagnosticCode::InvalidPipeline,
                            "publish contains duplicate buffer writes",
                        );
                    }
                    // Publication transfers completion responsibility; it does not make the event ready locally.
                    ctx.events.get_mut(id).unwrap().published = true;
                }
                _ => self.error(
                    Some(node),
                    DiagnosticCode::InvalidPipeline,
                    "publish event must be an unpublished write owned by this ticket",
                ),
            }
        }
        if let Some(pipe) = self.kernel.pipelines.iter().find(|p| p.id == info.pipeline) {
            let required: HashSet<_> = pipe.buffers.iter().map(|b| b.buffer.clone()).collect();
            if writes != required {
                self.error(
                    Some(node),
                    DiagnosticCode::InvalidPipeline,
                    "publish must cover every bound buffer exactly once",
                );
            }
        }
        ctx.tickets.get_mut(ticket).unwrap().closed = true;
    }

    fn release(&mut self, ctx: &mut Context, node: &ScheduleNode, ticket: &TicketId) {
        self.protocol_context(ctx, node);
        if !matches!(ctx.tickets.get(ticket), Some(info) if !info.closed && info.kind == TicketKind::Consumer)
        {
            self.error(
                Some(node),
                DiagnosticCode::InvalidTicket,
                "release requires a live consumer ticket",
            );
        }
        if ctx.events.values().any(|event| {
            !event.ready
                && event.reads.iter().any(
                    |r| matches!(r, MemoryRef::Leased { ticket: owner, .. } if owner == ticket),
                )
        }) {
            self.error(
                Some(node),
                DiagnosticCode::PendingAccess,
                "cannot release a slot while an async read still uses it",
            );
        }
        if let Some(info) = ctx.tickets.get_mut(ticket) {
            info.closed = true;
        }
    }

    fn pipeline_loop(
        &mut self,
        ctx: &mut Context,
        node: &ScheduleNode,
        pipeline: &PipelineId,
        iteration: &ValueDef,
        body: &Region,
    ) {
        if ctx.pipeline.is_some() || ctx.role.is_none() || ctx.control_depth != 0 {
            self.error(
                Some(node),
                DiagnosticCode::InvalidPipeline,
                "PipelineFor must be unconditional and directly inside a role",
            );
        }
        let Some(pipe) = self
            .kernel
            .pipelines
            .iter()
            .find(|p| &p.id == pipeline)
            .cloned()
        else {
            self.error(
                Some(node),
                DiagnosticCode::UnknownSymbol,
                "unknown pipeline",
            );
            return;
        };
        if ctx.role.as_ref() != Some(&pipe.producer) && ctx.role.as_ref() != Some(&pipe.consumer) {
            self.error(
                Some(node),
                DiagnosticCode::InvalidParticipation,
                "PipelineFor must execute in its producer or consumer role",
            );
        }
        if !ctx.loops.insert(pipeline.clone()) {
            self.error(
                Some(node),
                DiagnosticCode::InvalidPipeline,
                "a role may execute each pipeline only once",
            );
        }
        if !ctx.loops.iter().all(|p| p == pipeline) {
            self.error(Some(node), DiagnosticCode::UnsupportedSyntax, "multiple pipelines per role require a dependency/deadlock analysis that is not implemented");
        }
        let mut child = ctx.clone();
        child.pipeline = Some((pipeline.clone(), iteration.id.clone()));
        child.tickets.clear();
        child.events.clear();
        if iteration.ty != TypeRef::Index {
            self.error(
                Some(node),
                DiagnosticCode::InvalidType,
                "pipeline iteration must be Index",
            );
        }
        self.define(&mut child, iteration, Some(node));
        child.uniform.insert(iteration.id.clone());
        self.region(body, &mut child, Ending::Pipeline);
        if child.tickets.len() != 1
            || child
                .tickets
                .values()
                .any(|t| !t.closed || &t.pipeline != pipeline)
        {
            self.error(
                Some(node),
                DiagnosticCode::InvalidTicket,
                "every pipeline iteration must open and close exactly one role ticket",
            );
        }
        if child.events.values().any(|e| !e.ready && !e.published)
            || !child.pending_accumulators.is_empty()
        {
            self.error(
                Some(node),
                DiagnosticCode::PendingAccess,
                "pipeline iteration leaves an unobserved or unpublished async completion",
            );
        }
        // A zero-trip loop cannot initialize a resource for subsequent use.
        // Initialization inside an iteration is local to that iteration.
    }

    fn concurrent(
        &mut self,
        ctx: &Context,
        node: &ScheduleNode,
        roles: &[RoleRegion],
        join_scope: &Scope,
    ) {
        if ctx.role.is_some()
            || ctx.pipeline.is_some()
            || ctx.control_depth != 0
            || *join_scope != Scope::Block
            || roles.is_empty()
        {
            self.error(
                Some(node),
                DiagnosticCode::InvalidParticipation,
                "ConcurrentRoles requires nonempty top-level role regions and a block join",
            );
        }
        let mut participants = HashSet::new();
        let mut warps = HashSet::new();
        let mut uses: HashMap<PipelineId, HashSet<RoleId>> = HashMap::new();
        for region in roles {
            if !participants.insert(region.role.clone()) {
                self.error(
                    Some(node),
                    DiagnosticCode::InvalidParticipation,
                    "role appears twice in ConcurrentRoles",
                );
            }
            match self.kernel.roles.iter().find(|r| r.id == region.role) {
                Some(role) => {
                    for warp in &role.warps {
                        if !warps.insert(warp) {
                            self.error(
                                Some(node),
                                DiagnosticCode::InvalidParticipation,
                                "concurrent roles must have disjoint warps",
                            );
                        }
                    }
                }
                None => self.error(
                    Some(node),
                    DiagnosticCode::UnknownSymbol,
                    "unknown concurrent role",
                ),
            }
            let mut child = ctx.clone();
            child.role = Some(region.role.clone());
            child.loops.clear();
            child.drained.clear();
            self.region(&region.body, &mut child, Ending::Role);
            if child.loops != child.drained {
                self.error(
                    Some(node),
                    DiagnosticCode::InvalidPipeline,
                    "each role must drain every pipeline before the block join",
                );
            }
            if !child.pending_accumulators.is_empty() {
                self.error(
                    Some(node),
                    DiagnosticCode::PendingAccess,
                    "role exits with a pending accumulator update",
                );
            }
            for pipeline in child.loops {
                uses.entry(pipeline)
                    .or_default()
                    .insert(region.role.clone());
            }
        }
        for (id, actual) in uses {
            if let Some(pipe) = self.kernel.pipelines.iter().find(|p| p.id == id) {
                let expected = HashSet::from([pipe.producer.clone(), pipe.consumer.clone()]);
                if actual != expected {
                    self.error(Some(node), DiagnosticCode::InvalidPipeline, "both producer and consumer must execute and drain the same pipeline domain");
                }
            }
            if !self.executed_pipelines.insert(id) {
                self.error(Some(node), DiagnosticCode::InvalidPipeline, "pipeline reuse across concurrent groups needs explicit reinitialization and is not supported");
            }
        }
    }
}

fn literal_matches(value: &Literal, ty: &TypeRef) -> bool {
    match (value, ty) {
        (Literal::Index(_), TypeRef::Index) => true,
        (Literal::Bool(_), TypeRef::Predicate | TypeRef::Scalar(DType::Bool)) => true,
        (Literal::Integer(n), TypeRef::Scalar(dtype)) => match dtype {
            DType::I32 => i32::try_from(*n).is_ok(),
            DType::I64 => true,
            DType::U32 => u32::try_from(*n).is_ok(),
            DType::U64 => *n >= 0,
            _ => false,
        },
        (Literal::Float(n), TypeRef::Scalar(dtype)) => {
            n.is_finite()
                && match dtype {
                    DType::F64 => true,
                    DType::F32 | DType::BF16 => n.abs() <= f64::from(f32::MAX),
                    DType::F16 => n.abs() <= 65504.0,
                    _ => false,
                }
        }
        _ => false,
    }
}

fn footprint(buffer: &BufferType) -> Result<Option<u64>, &'static str> {
    let MappingSpec::Strided { strides_bytes } = &buffer.mapping else {
        return Ok(None);
    };
    if buffer.shape.len() != strides_bytes.len() {
        return Err("buffer rank and stride rank differ");
    }
    let mut size = buffer.element.size_bytes();
    let mut symbolic = false;
    for (dim, stride) in buffer.shape.iter().zip(strides_bytes) {
        match (dim.evaluate()?, stride.evaluate()?) {
            (Some(0), _) => return Ok(Some(0)),
            (Some(dim), Some(stride)) => {
                size = size
                    .checked_add(
                        (dim - 1)
                            .checked_mul(stride)
                            .ok_or("buffer footprint overflows")?,
                    )
                    .ok_or("buffer footprint overflows")?;
            }
            _ => symbolic = true,
        }
    }
    Ok(if symbolic { None } else { Some(size) })
}

fn buffer_end(buffer: &BufferType, slots: u32, stride: u64) -> Result<Option<u64>, &'static str> {
    if slots == 0 {
        return Err("pipeline must have at least one slot");
    }
    let extra = u64::from(slots - 1)
        .checked_mul(stride)
        .ok_or("staged buffer extent overflows")?;
    match (buffer.offset_bytes.evaluate()?, footprint(buffer)?) {
        (Some(offset), Some(size)) => Ok(Some(
            offset
                .checked_add(extra)
                .and_then(|v| v.checked_add(size))
                .ok_or("buffer extent overflows")?,
        )),
        _ => Ok(None),
    }
}

fn views_disjoint(a: &BufferType, b: &BufferType) -> bool {
    match (
        a.offset_bytes.evaluate(),
        buffer_end(a, 1, 0),
        b.offset_bytes.evaluate(),
        buffer_end(b, 1, 0),
    ) {
        (Ok(Some(a0)), Ok(Some(a1)), Ok(Some(b0)), Ok(Some(b1))) => a1 <= b0 || b1 <= a0,
        _ => false,
    }
}
