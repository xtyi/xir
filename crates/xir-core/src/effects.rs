//! Derived static access summaries. These are not a proof of non-aliasing or completion.
use crate::ast::*;
use crate::types::*;
use serde::Serialize;

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub enum AccessMode {
    Read,
    Write,
    ReadWrite,
}
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub enum AccessTarget {
    Memory(MemoryRef),
    Accumulator(AccumulatorId),
}
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct Access {
    pub target: AccessTarget,
    pub storage: Option<StorageId>,
    pub mode: AccessMode,
    pub pending_until: Option<EventId>,
}
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub enum Transition {
    Acquire(TicketId),
    Publish(TicketId),
    Consume(TicketId),
    Release(TicketId),
    Drain(PipelineId),
}
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct NodeEffects {
    pub node: NodeId,
    pub role: Option<RoleId>,
    pub pipeline_iteration: Option<(PipelineId, ValueId)>,
    pub produces: Vec<EventId>,
    pub publishes: Vec<EventId>,
    pub join_scope: Option<Scope>,
    pub accesses: Vec<Access>,
    pub observes: Vec<EventId>,
    pub transitions: Vec<Transition>,
}
pub fn derive_effects(module: &Module) -> Vec<NodeEffects> {
    fn walk(
        kernel: &Kernel,
        region: &Region,
        role: Option<&RoleId>,
        iteration: Option<&(PipelineId, ValueId)>,
        result: &mut Vec<NodeEffects>,
    ) {
        for node in region {
            let mut effect = NodeEffects {
                node: node.meta.node_id.clone(),
                role: role.cloned(),
                pipeline_iteration: iteration.cloned(),
                produces: vec![],
                publishes: vec![],
                join_scope: None,
                accesses: vec![],
                observes: vec![],
                transitions: vec![],
            };
            let memory = |reference: &MemoryRef, mode, event: Option<EventId>| Access {
                target: AccessTarget::Memory(reference.clone()),
                storage: kernel
                    .buffers
                    .iter()
                    .find(|b| &b.id == reference.buffer_id())
                    .map(|b| b.ty.storage.clone()),
                mode,
                pending_until: event,
            };
            let accumulator = |id: &AccumulatorId, mode, event| Access {
                target: AccessTarget::Accumulator(id.clone()),
                storage: None,
                mode,
                pending_until: event,
            };
            use ScheduleNodeKind::*;
            match &node.kind {
                Load { source, .. } => effect.accesses.push(memory(source, AccessMode::Read, None)),
                Store { destination, .. } => {
                    effect
                        .accesses
                        .push(memory(destination, AccessMode::Write, None))
                }
                AsyncCopy {
                    source,
                    destination,
                    completion,
                    ..
                } => {
                    effect.produces.push(completion.clone());
                    effect.accesses.push(memory(
                        source,
                        AccessMode::Read,
                        Some(completion.clone()),
                    ));
                    effect.accesses.push(memory(
                        destination,
                        AccessMode::Write,
                        Some(completion.clone()),
                    ));
                }
                Sqmma {
                    accumulator: acc,
                    a,
                    b,
                    completion,
                    ..
                } => {
                    effect.produces.push(completion.clone());
                    effect
                        .accesses
                        .push(memory(a, AccessMode::Read, Some(completion.clone())));
                    effect
                        .accesses
                        .push(memory(b, AccessMode::Read, Some(completion.clone())));
                    effect.accesses.push(accumulator(
                        acc,
                        AccessMode::ReadWrite,
                        Some(completion.clone()),
                    ));
                }
                AccumulatorInit {
                    accumulator: acc, ..
                } => effect
                    .accesses
                    .push(accumulator(acc, AccessMode::Write, None)),
                AccumulatorRead {
                    accumulator: acc, ..
                } => effect
                    .accesses
                    .push(accumulator(acc, AccessMode::Read, None)),
                AwaitEvent { event } => effect.observes.push(event.clone()),
                PipelineAcquire { ticket, .. } => {
                    effect.transitions.push(Transition::Acquire(ticket.clone()))
                }
                PipelineConsume { ticket, .. } => {
                    effect.transitions.push(Transition::Consume(ticket.clone()))
                }
                PipelinePublish { ticket, events } => {
                    effect.transitions.push(Transition::Publish(ticket.clone()));
                    effect.publishes.clone_from(events);
                }
                ConcurrentRoles { join_scope, .. } => effect.join_scope = Some(join_scope.clone()),
                PipelineRelease { ticket } => {
                    effect.transitions.push(Transition::Release(ticket.clone()))
                }
                PipelineDrain { pipeline } => {
                    effect.transitions.push(Transition::Drain(pipeline.clone()))
                }
                Constant { .. }
                | LaunchIndex { .. }
                | Binary { .. }
                | If { .. }
                | For { .. }
                | Yield { .. }
                | KernelReturn
                | PipelineFor { .. } => {}
            }
            result.push(effect);
            match &node.kind {
                ConcurrentRoles { roles, .. } => {
                    for region in roles {
                        walk(kernel, &region.body, Some(&region.role), iteration, result);
                    }
                }
                PipelineFor {
                    pipeline,
                    iteration: induction,
                    body,
                } => {
                    walk(
                        kernel,
                        body,
                        role,
                        Some(&(pipeline.clone(), induction.id.clone())),
                        result,
                    );
                }
                _ => {
                    for child in node.regions() {
                        walk(kernel, child, role, iteration, result);
                    }
                }
            }
        }
    }
    let mut result = vec![];
    for kernel in &module.kernels {
        walk(kernel, &kernel.body, None, None, &mut result);
    }
    result
}
