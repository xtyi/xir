use crate::diagnostic::SourceSpan;
use crate::target::TargetRequest;
use crate::types::*;
use serde::{Deserialize, Serialize};

pub const SCHEMA_VERSION: u32 = 2;
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Module {
    pub name: String,
    pub schema_version: u32,
    pub module_revision: u64,
    pub target_request: TargetRequest,
    pub kernels: Vec<Kernel>,
}
impl Default for Module {
    fn default() -> Self {
        Self {
            name: "module".into(),
            schema_version: SCHEMA_VERSION,
            module_revision: 0,
            target_request: TargetRequest::default(),
            kernels: vec![],
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Kernel {
    pub name: String,
    pub params: Vec<ValueDef>,
    pub block: [u32; 3],
    pub storages: Vec<StorageDecl>,
    pub buffers: Vec<BufferDecl>,
    pub roles: Vec<RoleDecl>,
    pub pipelines: Vec<PipelineDecl>,
    pub accumulators: Vec<AccumulatorDecl>,
    pub body: Region,
}
impl Kernel {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            params: vec![],
            block: [1, 1, 1],
            storages: vec![],
            buffers: vec![],
            roles: vec![],
            pipelines: vec![],
            accumulators: vec![],
            body: vec![],
        }
    }
}
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct NodeMeta {
    pub node_id: NodeId,
    pub node_revision: u64,
    pub source_span: Option<SourceSpan>,
    pub origin_ids: Vec<NodeId>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ValueDef {
    pub id: ValueId,
    pub ty: TypeRef,
}
impl ValueDef {
    pub fn new(id: &str, ty: TypeRef) -> Self {
        Self { id: id.into(), ty }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct StorageDecl {
    pub id: StorageId,
    pub ty: StorageType,
    pub external: Option<ValueId>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BufferDecl {
    pub id: BufferId,
    pub ty: BufferType,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RoleDecl {
    pub id: RoleId,
    pub warps: Vec<u32>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PipelineBufferBinding {
    pub buffer: BufferId,
    pub slot_stride_bytes: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PipelineDecl {
    pub id: PipelineId,
    pub iteration_count: IndexExpr,
    pub slots: u32,
    pub producer: RoleId,
    pub consumer: RoleId,
    pub buffers: Vec<PipelineBufferBinding>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AccumulatorDecl {
    pub id: AccumulatorId,
    pub ty: FragmentType,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CarriedValue {
    pub initial: ValueId,
    pub argument: ValueDef,
    pub result: ValueDef,
}

pub type Region = Vec<ScheduleNode>;
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoleRegion {
    pub role: RoleId,
    pub body: Region,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ScheduleNode {
    pub meta: NodeMeta,
    pub kind: ScheduleNodeKind,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", content = "data", deny_unknown_fields)]
pub enum Literal {
    Index(u64),
    Integer(i64),
    Float(f64),
    Bool(bool),
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Less,
    Equal,
    And,
    Or,
}
/// Logical launch coordinates, independent of target spelling.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum LaunchBuiltin {
    ThreadIdxX,
    BlockIdxX,
    BlockDimX,
    GridDimX,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", content = "data", deny_unknown_fields)]
pub enum ScheduleNodeKind {
    LaunchIndex {
        builtin: LaunchBuiltin,
        result: ValueDef,
    },
    Constant {
        value: Literal,
        result: ValueDef,
    },
    Binary {
        op: BinaryOp,
        lhs: ValueId,
        rhs: ValueId,
        result: ValueDef,
    },
    Load {
        source: MemoryRef,
        indices: Vec<ValueId>,
        mask: ValueId,
        masked_value: ValueId,
        result: ValueDef,
    },
    Store {
        destination: MemoryRef,
        indices: Vec<ValueId>,
        value: ValueId,
        mask: ValueId,
    },
    If {
        condition: ValueId,
        results: Vec<ValueDef>,
        then_body: Region,
        else_body: Region,
    },
    For {
        lower: ValueId,
        upper: ValueId,
        step: ValueId,
        induction: ValueDef,
        carried: Vec<CarriedValue>,
        body: Region,
    },
    Yield {
        values: Vec<ValueId>,
    },
    KernelReturn,
    ConcurrentRoles {
        roles: Vec<RoleRegion>,
        join_scope: Scope,
    },
    PipelineFor {
        pipeline: PipelineId,
        iteration: ValueDef,
        body: Region,
    },
    PipelineAcquire {
        pipeline: PipelineId,
        iteration: ValueId,
        ticket: TicketId,
    },
    PipelinePublish {
        ticket: TicketId,
        events: Vec<EventId>,
    },
    PipelineConsume {
        pipeline: PipelineId,
        iteration: ValueId,
        ticket: TicketId,
    },
    PipelineRelease {
        ticket: TicketId,
    },
    PipelineDrain {
        pipeline: PipelineId,
    },
    AsyncCopy {
        source: MemoryRef,
        destination: MemoryRef,
        transfer: TransferSpecId,
        completion: EventId,
    },
    AccumulatorInit {
        accumulator: AccumulatorId,
        value: ValueId,
    },
    AccumulatorRead {
        accumulator: AccumulatorId,
        result: ValueDef,
    },
    Sqmma {
        accumulator: AccumulatorId,
        a: MemoryRef,
        b: MemoryRef,
        instruction: InstructionSpecId,
        completion: EventId,
    },
    AwaitEvent {
        event: EventId,
    },
}
impl ScheduleNode {
    pub fn new(kind: ScheduleNodeKind) -> Self {
        Self {
            meta: NodeMeta::default(),
            kind,
        }
    }
    /// Ordinary SSA operands only; resource references remain strongly typed in payloads.
    pub fn operands(&self) -> Vec<&ValueId> {
        use ScheduleNodeKind::*;
        match &self.kind {
            Binary { lhs, rhs, .. } => vec![lhs, rhs],
            Load {
                indices,
                mask,
                masked_value,
                ..
            } => indices.iter().chain([mask, masked_value]).collect(),
            Store {
                indices,
                value,
                mask,
                ..
            } => indices.iter().chain([value, mask]).collect(),
            If { condition, .. } => vec![condition],
            For {
                lower,
                upper,
                step,
                carried,
                ..
            } => {
                let mut ids = vec![lower, upper, step];
                ids.extend(carried.iter().map(|c| &c.initial));
                ids
            }
            Yield { values } => values.iter().collect(),
            PipelineAcquire { iteration, .. } | PipelineConsume { iteration, .. } => {
                vec![iteration]
            }
            AccumulatorInit { value, .. } => vec![value],
            _ => vec![],
        }
    }
    pub fn results(&self) -> Vec<&ValueDef> {
        use ScheduleNodeKind::*;
        match &self.kind {
            Constant { result, .. }
            | LaunchIndex { result, .. }
            | Binary { result, .. }
            | Load { result, .. }
            | AccumulatorRead { result, .. } => vec![result],
            If { results, .. } => results.iter().collect(),
            For { carried, .. } => carried.iter().map(|c| &c.result).collect(),
            _ => vec![],
        }
    }
    pub fn regions(&self) -> Vec<&Region> {
        match &self.kind {
            ScheduleNodeKind::If {
                then_body,
                else_body,
                ..
            } => vec![then_body, else_body],
            ScheduleNodeKind::For { body, .. } | ScheduleNodeKind::PipelineFor { body, .. } => {
                vec![body]
            }
            ScheduleNodeKind::ConcurrentRoles { roles, .. } => {
                roles.iter().map(|r| &r.body).collect()
            }
            _ => vec![],
        }
    }
    pub fn regions_mut(&mut self) -> Vec<&mut Region> {
        match &mut self.kind {
            ScheduleNodeKind::If {
                then_body,
                else_body,
                ..
            } => vec![then_body, else_body],
            ScheduleNodeKind::For { body, .. } | ScheduleNodeKind::PipelineFor { body, .. } => {
                vec![body]
            }
            ScheduleNodeKind::ConcurrentRoles { roles, .. } => {
                roles.iter_mut().map(|r| &mut r.body).collect()
            }
            _ => vec![],
        }
    }
}
