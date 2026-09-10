use xir_core::ast::*;
use xir_core::types::*;
use xir_core::{Kernel, Module, ModuleBuilder, TargetRequest};

pub fn node(kind: ScheduleNodeKind) -> ScheduleNode {
    ScheduleNode::new(kind)
}

/// A protocol example, not a complete numerical GEMM or a runnable GPU kernel.
/// Each logical iteration transfers the same two views.
pub fn pipeline_module() -> Module {
    use ScheduleNodeKind::*;
    let mut kernel = Kernel::new("pipeline_example");
    kernel.block = [160, 1, 1];
    kernel.params = vec![ValueDef::new("iterations", TypeRef::Index)];
    for name in ["a", "b"] {
        let pointer = format!("{name}_ptr");
        kernel.params.push(ValueDef::new(
            &pointer,
            TypeRef::Pointer(PointerType {
                element: DType::BF16,
                address_space: AddressSpace::Global,
                mutable: false,
            }),
        ));
        for (suffix, address_space, capacity, external) in [
            (
                "global",
                AddressSpace::Global,
                16384,
                Some(pointer.clone().into()),
            ),
            ("shared", AddressSpace::Shared, 32768, None),
        ] {
            let id = format!("{name}_{suffix}");
            kernel.storages.push(StorageDecl {
                id: id.clone().into(),
                external,
                ty: StorageType {
                    address_space,
                    capacity_bytes: IndexExpr::constant(capacity),
                    alignment_bytes: 128,
                },
            });
            let (shape, strides) = if name == "a" {
                ([128, 64], [128, 2])
            } else {
                ([64, 128], [256, 2])
            };
            kernel.buffers.push(BufferDecl {
                id: id.clone().into(),
                ty: BufferType {
                    storage: id.into(),
                    offset_bytes: IndexExpr::constant(0),
                    element: DType::BF16,
                    shape: shape.into_iter().map(IndexExpr::constant).collect(),
                    mapping: MappingSpec::Strided {
                        strides_bytes: strides.into_iter().map(IndexExpr::constant).collect(),
                    },
                    mutable: suffix == "shared",
                },
            });
        }
    }
    kernel.roles = vec![
        RoleDecl {
            id: "producer".into(),
            warps: vec![4],
        },
        RoleDecl {
            id: "consumer".into(),
            warps: vec![0, 1, 2, 3],
        },
    ];
    kernel.pipelines.push(PipelineDecl {
        id: "pipe".into(),
        iteration_count: IndexExpr::Value("iterations".into()),
        slots: 2,
        producer: "producer".into(),
        consumer: "consumer".into(),
        buffers: ["a_shared", "b_shared"]
            .into_iter()
            .map(|id| PipelineBufferBinding {
                buffer: id.into(),
                slot_stride_bytes: 16384,
            })
            .collect(),
    });
    let fragment = FragmentType {
        element: DType::F32,
        logical_shape: vec![IndexExpr::constant(128), IndexExpr::constant(128)],
        participants: "consumer".into(),
        representation: "mp31.sqmma.acc.128x128.f32".into(),
    };
    kernel.accumulators.push(AccumulatorDecl {
        id: "acc".into(),
        ty: fragment.clone(),
    });
    let producer = vec![
        node(PipelineFor {
            pipeline: "pipe".into(),
            iteration: ValueDef::new("producer_i", TypeRef::Index),
            body: vec![
                node(PipelineAcquire {
                    pipeline: "pipe".into(),
                    iteration: "producer_i".into(),
                    ticket: "write_ticket".into(),
                }),
                node(AsyncCopy {
                    source: MemoryRef::buffer("a_global"),
                    destination: MemoryRef::leased("a_shared", "write_ticket"),
                    transfer: "mp31.tme.load".into(),
                    completion: "copy_a".into(),
                }),
                node(AsyncCopy {
                    source: MemoryRef::buffer("b_global"),
                    destination: MemoryRef::leased("b_shared", "write_ticket"),
                    transfer: "mp31.tme.load".into(),
                    completion: "copy_b".into(),
                }),
                node(PipelinePublish {
                    ticket: "write_ticket".into(),
                    events: vec!["copy_a".into(), "copy_b".into()],
                }),
            ],
        }),
        node(PipelineDrain {
            pipeline: "pipe".into(),
        }),
    ];
    let consumer = vec![
        node(Constant {
            value: Literal::Float(0.0),
            result: ValueDef::new("zero", TypeRef::Scalar(DType::F32)),
        }),
        node(AccumulatorInit {
            accumulator: "acc".into(),
            value: "zero".into(),
        }),
        node(PipelineFor {
            pipeline: "pipe".into(),
            iteration: ValueDef::new("consumer_i", TypeRef::Index),
            body: vec![
                node(PipelineConsume {
                    pipeline: "pipe".into(),
                    iteration: "consumer_i".into(),
                    ticket: "read_ticket".into(),
                }),
                node(Sqmma {
                    accumulator: "acc".into(),
                    a: MemoryRef::leased("a_shared", "read_ticket"),
                    b: MemoryRef::leased("b_shared", "read_ticket"),
                    instruction: "MP31_128x128x64_F32BF16BF16_SS".into(),
                    completion: "mma".into(),
                }),
                node(AwaitEvent {
                    event: "mma".into(),
                }),
                node(PipelineRelease {
                    ticket: "read_ticket".into(),
                }),
            ],
        }),
        node(PipelineDrain {
            pipeline: "pipe".into(),
        }),
        node(AccumulatorRead {
            accumulator: "acc".into(),
            result: ValueDef::new("result_fragment", TypeRef::Fragment(fragment)),
        }),
    ];
    kernel.body = vec![
        node(ConcurrentRoles {
            roles: vec![
                RoleRegion {
                    role: "producer".into(),
                    body: producer,
                },
                RoleRegion {
                    role: "consumer".into(),
                    body: consumer,
                },
            ],
            join_scope: Scope::Block,
        }),
        node(KernelReturn),
    ];
    let mut builder = ModuleBuilder::new("pipeline", TargetRequest::musa_mp31());
    builder.add_kernel(kernel);
    builder
        .finish_checked()
        .expect("protocol example must pass construction validation")
}
