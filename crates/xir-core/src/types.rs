use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum TypeRef {
    Scalar(DType),
    Index(IndexType),
    Predicate,
    Pointer(PointerType),
    Storage(StorageType),
    Buffer(BufferType),
    Fragment(FragmentType),
    Event(EventType),
    Resource(ResourceType),
    Tuple(Vec<TypeRef>),
    Void,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct StorageId(pub String);
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct BufferId(pub String);
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ValueId(pub String);

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum DType {
    Bool,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F16,
    BF16,
    F32,
    F64,
    F8E4M3,
    F8E5M2,
    Custom(String),
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct IndexType {
    pub width: u8,
    pub signed: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum AddressSpace {
    Global,
    Shared,
    Private,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct StorageType {
    pub address_space: AddressSpace,
    pub size_bytes: u64,
    pub alignment_bytes: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PointerType {
    pub element: Box<TypeRef>,
    pub address_space: AddressSpace,
    pub mutable: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Dim {
    Static(u64),
    Symbolic(Symbol),
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Symbol {
    pub name: String,
    pub kind: SymbolKind,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum SymbolKind {
    Shape,
    Stride,
    Scalar,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct BufferType {
    pub storage: StorageId,
    pub offset_bytes: Vec<String>,
    pub element: DType,
    pub shape: Vec<Dim>,
    pub strides_bytes: Vec<String>,
    pub mutable: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct BufferRef {
    pub buffer: BufferId,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct FragmentType {
    pub family: String,
    pub role: String,
    pub element: DType,
    pub logical_shape: Vec<Dim>,
    pub representation: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct EventType {
    pub kind: String,
    pub scope: String,
    pub stage: Option<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResourceType {
    pub kind: String,
    pub memory_space: Option<AddressSpace>,
    pub owner: Option<String>,
    pub lifetime: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct EffectSet {
    pub reads: Vec<String>,
    pub writes: Vec<String>,
    pub produces: Vec<String>,
    pub consumes: Vec<String>,
    pub scope: Option<String>,
}
