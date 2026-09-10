use serde::{Deserialize, Serialize};

macro_rules! identifier {
    ($($name:ident),+ $(,)?) => {$ (
        #[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
        #[serde(transparent)]
        pub struct $name(pub String);
        impl From<&str> for $name {
            fn from(value: &str) -> Self { Self(value.to_owned()) }
        }
        impl From<String> for $name {
            fn from(value: String) -> Self { Self(value) }
        }
    )+};
}
identifier!(
    NodeId,
    ValueId,
    StorageId,
    BufferId,
    RoleId,
    PipelineId,
    AccumulatorId,
    EventId,
    TicketId,
    InstructionSpecId,
    TransferSpecId
);

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum DType {
    Bool,
    I32,
    I64,
    U32,
    U64,
    F16,
    BF16,
    F32,
    F64,
}
impl DType {
    pub fn size_bytes(&self) -> u64 {
        match self {
            Self::Bool => 1,
            Self::F16 | Self::BF16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::U64 | Self::F64 => 8,
        }
    }
}

/// Ordinary immutable values. Resources, tickets and events are separate namespaces.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub enum TypeRef {
    Scalar(DType),
    /// The initial index model is unsigned 64-bit arithmetic with checked bounds.
    Index,
    Predicate,
    Pointer(PointerType),
    Fragment(FragmentType),
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PointerType {
    pub element: DType,
    pub address_space: AddressSpace,
    pub mutable: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum AddressSpace {
    Global,
    Shared,
    Private,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Scope {
    Thread,
    Warp,
    Block,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", content = "data", deny_unknown_fields)]
pub enum IndexExpr {
    Constant(u64),
    Value(ValueId),
    Add(Box<IndexExpr>, Box<IndexExpr>),
    Mul(Box<IndexExpr>, Box<IndexExpr>),
    FloorDiv(Box<IndexExpr>, Box<IndexExpr>),
}
impl IndexExpr {
    pub fn constant(value: u64) -> Self {
        Self::Constant(value)
    }
    pub fn values(&self, output: &mut Vec<ValueId>) {
        match self {
            Self::Value(id) => output.push(id.clone()),
            Self::Add(a, b) | Self::Mul(a, b) | Self::FloorDiv(a, b) => {
                a.values(output);
                b.values(output);
            }
            Self::Constant(_) => {}
        }
    }
    /// None means symbolic, not zero or a successfully proven bound.
    pub fn evaluate(&self) -> Result<Option<u64>, &'static str> {
        match self {
            Self::Constant(n) => Ok(Some(*n)),
            Self::Value(_) => Ok(None),
            Self::Add(a, b) | Self::Mul(a, b) | Self::FloorDiv(a, b) => {
                let left = a.evaluate()?;
                let right = b.evaluate()?;
                if matches!(self, Self::FloorDiv(_, _)) && right == Some(0) {
                    return Err("index division by zero");
                }
                match (left, right) {
                    (Some(a), Some(b)) => {
                        let value = match self {
                            Self::Add(_, _) => a.checked_add(b),
                            Self::Mul(_, _) => a.checked_mul(b),
                            Self::FloorDiv(_, _) => a.checked_div(b),
                            _ => unreachable!(),
                        };
                        value.map(Some).ok_or("index arithmetic overflows u64")
                    }
                    _ => Ok(None),
                }
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct StorageType {
    pub address_space: AddressSpace,
    pub capacity_bytes: IndexExpr,
    pub alignment_bytes: u32,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", content = "data", deny_unknown_fields)]
pub enum MappingSpec {
    Strided {
        strides_bytes: Vec<IndexExpr>,
    },
    /// Admission of a registered mapping requires a future target adapter.
    Registered {
        name: String,
    },
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BufferType {
    pub storage: StorageId,
    pub offset_bytes: IndexExpr,
    pub element: DType,
    pub shape: Vec<IndexExpr>,
    pub mapping: MappingSpec,
    pub mutable: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FragmentType {
    pub element: DType,
    pub logical_shape: Vec<IndexExpr>,
    pub participants: RoleId,
    pub representation: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", content = "data", deny_unknown_fields)]
pub enum MemoryRef {
    Buffer(BufferId),
    Leased { buffer: BufferId, ticket: TicketId },
}
impl MemoryRef {
    pub fn buffer(id: &str) -> Self {
        Self::Buffer(id.into())
    }
    pub fn leased(buffer: &str, ticket: &str) -> Self {
        Self::Leased {
            buffer: buffer.into(),
            ticket: ticket.into(),
        }
    }
    pub fn buffer_id(&self) -> &BufferId {
        match self {
            Self::Buffer(id) | Self::Leased { buffer: id, .. } => id,
        }
    }
}
