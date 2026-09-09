use crate::target::TargetSpec;
use crate::types::{EffectSet, TypeRef};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Module {
    pub name: String,
    pub version: u32,
    pub target: TargetSpec,
    pub kernels: Vec<Kernel>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Kernel {
    pub name: String,
    pub params: Vec<Parameter>,
    /// Placeholder until the typed schedule node set is introduced in P2.
    pub body: Vec<ScheduleNode>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Parameter {
    pub name: String,
    pub ty: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct NodeMeta {
    pub node_id: String,
    pub source_span: Option<crate::diagnostic::SourceSpan>,
    pub effects: EffectSet,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ScheduleNode {
    pub meta: NodeMeta,
    pub kind: ScheduleNodeKind,
    pub operands: Vec<String>,
    pub result: Option<ValueDef>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ValueDef {
    pub value_id: String,
    pub ty: TypeRef,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", content = "data")]
pub enum ScheduleNodeKind {
    BufferDecl {
        name: String,
        memory_space: String,
        dtype: String,
        shape: Vec<u64>,
        stages: u32,
    },
    PipelineDecl {
        name: String,
        stages: u32,
    },
    RoleDecl {
        name: String,
        warps: Vec<u32>,
    },
    RoleRegion {
        role: String,
        body: Vec<ScheduleNode>,
    },
    StageFor {
        iterator: String,
        pipeline: String,
        body: Vec<ScheduleNode>,
    },
    AsyncCopy {
        src: String,
        dst: String,
        barrier: String,
    },
    Wait {
        event: String,
        stage: Option<String>,
    },
    Sqmma {
        accumulator: String,
        a: String,
        b: String,
    },
    Assign {
        name: String,
        value: String,
    },
    Load {
        result: String,
        source: String,
    },
    Store {
        destination: String,
        value: String,
    },
    Binary {
        result: String,
        op: String,
        lhs: String,
        rhs: String,
    },
    If {
        condition: String,
        then_body: Vec<ScheduleNode>,
        else_body: Vec<ScheduleNode>,
    },
    For {
        iterator: String,
        begin: String,
        end: String,
        body: Vec<ScheduleNode>,
    },
    Return {
        value: Option<String>,
    },
}

impl ScheduleNode {
    pub fn new(kind: ScheduleNodeKind) -> Self {
        Self {
            meta: NodeMeta::default(),
            kind,
            operands: vec![],
            result: None,
        }
    }
}
