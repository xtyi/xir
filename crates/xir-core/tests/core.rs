#[path = "../examples/support/mod.rs"]
mod support;
use support::{node, pipeline_module};
fn index(id: &str, n: u64) -> ScheduleNode {
    node(ScheduleNodeKind::Constant {
        value: Literal::Index(n),
        result: ValueDef::new(id, TypeRef::Index),
    })
}
use xir_core::ast::*;
use xir_core::effects::{AccessMode, AccessTarget};
use xir_core::types::*;
use xir_core::{
    deserialize_module, serialize_module, validate_module, verify_module, CheckStatus,
    DiagnosticCode, ModuleBuilder,
};

fn normalized(kernel: Kernel) -> Module {
    let mut builder = ModuleBuilder::new("test", Default::default());
    builder.add_kernel(kernel);
    builder.finish_unchecked()
}
fn simple(body: Region) -> Module {
    let mut kernel = Kernel::new("test");
    kernel.body = body;
    normalized(kernel)
}
fn assert_error(module: &Module, code: DiagnosticCode) {
    let errors = validate_module(module);
    assert!(
        errors.diagnostics.iter().any(|d| d.code == code),
        "expected {code:?}, got {errors:?}"
    );
}
fn role_body(module: &mut Module, role: usize) -> &mut Region {
    let ScheduleNodeKind::ConcurrentRoles { roles, .. } = &mut module.kernels[0].body[0].kind
    else {
        panic!()
    };
    &mut roles[role].body
}
fn pipeline_body(module: &mut Module, role: usize) -> &mut Region {
    let position = if role == 0 { 0 } else { 2 };
    let ScheduleNodeKind::PipelineFor { body, .. } = &mut role_body(module, role)[position].kind
    else {
        panic!()
    };
    body
}

#[test]
fn canonical_json_round_trip_and_honest_verification() {
    let module = pipeline_module();
    assert_eq!(
        deserialize_module(&serialize_module(&module).unwrap()).unwrap(),
        module
    );
    let report = verify_module(&module);
    assert_eq!(report.construction, CheckStatus::Pass);
    assert_eq!(report.status, CheckStatus::Unknown);
    assert_eq!(report.target, CheckStatus::Unknown);
    assert!(report.coverage_limits.len() >= 3);
}

#[test]
fn recursive_validation_rejects_undefined_and_wrong_typed_values() {
    let mut module = pipeline_module();
    let ScheduleNodeKind::AccumulatorInit { value, .. } = &mut role_body(&mut module, 1)[1].kind
    else {
        panic!()
    };
    *value = "missing".into();
    assert_error(&module, DiagnosticCode::UnknownSymbol);
    *match &mut role_body(&mut module, 1)[1].kind {
        ScheduleNodeKind::AccumulatorInit { value, .. } => value,
        _ => panic!(),
    } = "iterations".into();
    assert_error(&module, DiagnosticCode::InvalidType);
}

#[test]
fn generated_ids_skip_existing_ids_and_duplicate_imports_fail() {
    let mut end = node(ScheduleNodeKind::KernelReturn);
    end.meta.node_id = "test.n0".into();
    let module = simple(vec![index("zero", 0), end]);
    assert_eq!(module.kernels[0].body[0].meta.node_id.0, "test.n1");
    assert!(!validate_module(&module).has_errors());
    let mut duplicate = module.clone();
    duplicate.kernels[0].body[1].meta.node_id = duplicate.kernels[0].body[0].meta.node_id.clone();
    assert_error(&duplicate, DiagnosticCode::DuplicateSymbol);
}

#[test]
fn old_schema_and_forged_fields_are_rejected() {
    let module = pipeline_module();
    let mut encoded = serde_json::to_value(&module).unwrap();
    encoded["kernels"][0]["body"][0]["meta"]["effects"] = serde_json::json!([]);
    assert!(deserialize_module(&encoded.to_string()).is_err());
    let mut encoded = serde_json::to_value(&module).unwrap();
    encoded["kernels"][0]["body"][0]["kind"]["data"]["extra"] = serde_json::json!(true);
    assert!(deserialize_module(&encoded.to_string()).is_err());
    let mut old = module.clone();
    old.schema_version = 1;
    assert_error(&old, DiagnosticCode::UnsupportedSchema);
    old.target_request.arch = "mp99".into();
    assert_error(&old, DiagnosticCode::UnsupportedCapability);
}

#[test]
fn structured_ssa_for_and_if_validate_yields_and_scopes() {
    use ScheduleNodeKind::*;
    let mut module = simple(vec![
        index("zero", 0),
        index("one", 1),
        index("limit", 8),
        node(For {
            lower: "zero".into(),
            upper: "limit".into(),
            step: "one".into(),
            induction: ValueDef::new("i", TypeRef::Index),
            carried: vec![CarriedValue {
                initial: "zero".into(),
                argument: ValueDef::new("sum_arg", TypeRef::Index),
                result: ValueDef::new("sum", TypeRef::Index),
            }],
            body: vec![
                node(Binary {
                    op: BinaryOp::Add,
                    lhs: "sum_arg".into(),
                    rhs: "i".into(),
                    result: ValueDef::new("next", TypeRef::Index),
                }),
                node(Yield {
                    values: vec!["next".into()],
                }),
            ],
        }),
        node(Constant {
            value: Literal::Bool(true),
            result: ValueDef::new("cond", TypeRef::Predicate),
        }),
        node(If {
            condition: "cond".into(),
            results: vec![ValueDef::new("chosen", TypeRef::Index)],
            then_body: vec![node(Yield {
                values: vec!["sum".into()],
            })],
            else_body: vec![node(Yield {
                values: vec!["zero".into()],
            })],
        }),
        node(KernelReturn),
    ]);
    assert!(
        !validate_module(&module).has_errors(),
        "{:?}",
        validate_module(&module)
    );
    if let If { else_body, .. } = &mut module.kernels[0].body[5].kind {
        else_body[0].kind = Yield {
            values: vec!["next".into()],
        };
    }
    assert_error(&module, DiagnosticCode::UnknownSymbol);
}

#[test]
fn missing_terminator_and_nonpositive_loop_step_fail() {
    assert_error(&simple(vec![]), DiagnosticCode::InvalidControlFlow);
    let module = simple(vec![
        index("zero", 0),
        node(ScheduleNodeKind::For {
            lower: "zero".into(),
            upper: "zero".into(),
            step: "zero".into(),
            induction: ValueDef::new("i", TypeRef::Index),
            carried: vec![],
            body: vec![node(ScheduleNodeKind::Yield { values: vec![] })],
        }),
        node(ScheduleNodeKind::KernelReturn),
    ]);
    assert_error(&module, DiagnosticCode::UnsupportedSyntax);
}

#[test]
fn sibling_roles_do_not_share_ssa_values_or_events() {
    let mut module = pipeline_module();
    pipeline_body(&mut module, 0).insert(0, index("producer_value", 0));
    let ScheduleNodeKind::AccumulatorInit { value, .. } = &mut role_body(&mut module, 1)[1].kind
    else {
        panic!()
    };
    *value = "producer_value".into();
    assert_error(&module, DiagnosticCode::UnknownSymbol);
    pipeline_body(&mut module, 1)[2].kind = ScheduleNodeKind::AwaitEvent {
        event: "copy_a".into(),
    };
    assert_error(&module, DiagnosticCode::UnknownSymbol);
}

#[test]
fn overlapping_roles_missing_partner_and_missing_drain_fail() {
    let mut module = pipeline_module();
    module.kernels[0].roles[0].warps = vec![0];
    assert_error(&module, DiagnosticCode::InvalidParticipation);
    let mut module = pipeline_module();
    role_body(&mut module, 0).pop();
    assert_error(&module, DiagnosticCode::InvalidPipeline);
    let mut module = pipeline_module();
    let ScheduleNodeKind::ConcurrentRoles { roles, .. } = &mut module.kernels[0].body[0].kind
    else {
        panic!()
    };
    roles.remove(0);
    assert_error(&module, DiagnosticCode::InvalidPipeline);
}

#[test]
fn zero_slots_overlap_capacity_and_overflow_fail() {
    let mut module = pipeline_module();
    module.kernels[0].pipelines[0].slots = 0;
    assert_error(&module, DiagnosticCode::InvalidPipeline);
    let mut module = pipeline_module();
    module.kernels[0].pipelines[0].buffers[0].slot_stride_bytes = 128;
    assert_error(&module, DiagnosticCode::InvalidPipeline);
    let mut module = pipeline_module();
    module.kernels[0].pipelines[0].slots = 3;
    assert_error(&module, DiagnosticCode::InvalidPipeline);
    let mut module = pipeline_module();
    module.kernels[0].buffers[0].ty.shape[0] = IndexExpr::constant(u64::MAX);
    assert_error(&module, DiagnosticCode::InvalidResource);
}

#[test]
fn raw_alias_cannot_bypass_lease_and_ticket_cannot_be_reused() {
    let mut module = pipeline_module();
    let mut alias = module.kernels[0].buffers[1].clone();
    alias.id = "alias".into();
    module.kernels[0].buffers.push(alias);
    let ScheduleNodeKind::Sqmma { a, .. } = &mut pipeline_body(&mut module, 1)[1].kind else {
        panic!()
    };
    *a = MemoryRef::buffer("alias");
    assert_error(&module, DiagnosticCode::InvalidTicket);
    let mut module = pipeline_module();
    pipeline_body(&mut module, 1).swap(1, 3);
    assert_error(&module, DiagnosticCode::InvalidTicket);
}

#[test]
fn release_and_accumulator_reads_require_mma_completion() {
    let mut module = pipeline_module();
    pipeline_body(&mut module, 1).remove(2);
    assert_error(&module, DiagnosticCode::PendingAccess);
    let mut module = pipeline_module();
    let read = role_body(&mut module, 1).pop().unwrap();
    pipeline_body(&mut module, 1).insert(2, read);
    assert_error(&module, DiagnosticCode::PendingAccess);
}

#[test]
fn publish_requires_matching_completion_coverage() {
    let mut module = pipeline_module();
    let ScheduleNodeKind::PipelinePublish { events, .. } =
        &mut pipeline_body(&mut module, 0)[3].kind
    else {
        panic!()
    };
    events.pop();
    assert_error(&module, DiagnosticCode::InvalidPipeline);
    let mut module = pipeline_module();
    let ScheduleNodeKind::PipelineAcquire { iteration, .. } =
        &mut pipeline_body(&mut module, 0)[0].kind
    else {
        panic!()
    };
    *iteration = "iterations".into();
    assert_error(&module, DiagnosticCode::InvalidTicket);
}

#[test]
fn conditional_protocol_is_explicitly_unsupported() {
    let mut module = pipeline_module();
    let acquire = pipeline_body(&mut module, 0).remove(0);
    pipeline_body(&mut module, 0).insert(
        0,
        node(ScheduleNodeKind::If {
            condition: "iterations".into(),
            results: vec![],
            then_body: vec![acquire, node(ScheduleNodeKind::Yield { values: vec![] })],
            else_body: vec![node(ScheduleNodeKind::Yield { values: vec![] })],
        }),
    );
    assert_error(&module, DiagnosticCode::UnsupportedSyntax);
}

#[test]
fn effect_contract_marks_pending_inputs_and_accumulator_and_is_recomputed() {
    let mut module = pipeline_module();
    let report = verify_module(&module);
    let mma = report
        .effects
        .iter()
        .find(|n| {
            n.accesses.iter().any(|a| {
                matches!(&a.target, AccessTarget::Accumulator(_)) && a.mode == AccessMode::ReadWrite
            })
        })
        .unwrap();
    assert_eq!(mma.accesses.len(), 3);
    assert!(mma
        .accesses
        .iter()
        .all(|a| a.pending_until == Some("mma".into())));
    let id = pipeline_body(&mut module, 0)[1].meta.node_id.clone();
    let ScheduleNodeKind::AsyncCopy { source, .. } = &mut pipeline_body(&mut module, 0)[1].kind
    else {
        panic!()
    };
    *source = MemoryRef::buffer("b_global");
    let changed = verify_module(&module);
    let copy = changed.effects.iter().find(|n| n.node == id).unwrap();
    assert!(copy
        .accesses
        .iter()
        .any(|a| a.target == AccessTarget::Memory(MemoryRef::buffer("b_global"))));
}

#[test]
fn literal_overflow_and_index_division_by_zero_fail() {
    let module = simple(vec![
        node(ScheduleNodeKind::Constant {
            value: Literal::Integer(i64::MAX),
            result: ValueDef::new("x", TypeRef::Scalar(DType::I32)),
        }),
        node(ScheduleNodeKind::KernelReturn),
    ]);
    assert_error(&module, DiagnosticCode::InvalidType);
    assert!(IndexExpr::FloorDiv(
        Box::new(IndexExpr::Value("n".into())),
        Box::new(IndexExpr::constant(0))
    )
    .evaluate()
    .is_err());
}

#[test]
fn repeated_waits_do_not_consume_immutable_event_handles() {
    let mut module = pipeline_module();
    pipeline_body(&mut module, 1).insert(
        3,
        node(ScheduleNodeKind::AwaitEvent {
            event: "mma".into(),
        }),
    );
    pipeline_body(&mut module, 0).push(node(ScheduleNodeKind::AwaitEvent {
        event: "copy_a".into(),
    }));
    let module = ModuleBuilder::from_module(module).finish_checked().unwrap();
    assert_eq!(verify_module(&module).construction, CheckStatus::Pass);
}

#[test]
fn shared_pipeline_domain_accepts_empty_and_rollover_trip_counts_structurally() {
    // This tests symbolic protocol construction, not a lowered barrier implementation.
    for slots in [1, 2, 3] {
        for count in [
            0,
            1,
            u64::from(slots - 1),
            u64::from(slots),
            u64::from(slots + 1),
            u64::from(2 * slots + 1),
        ] {
            let mut module = pipeline_module();
            module.kernels[0].pipelines[0].slots = slots;
            module.kernels[0].pipelines[0].iteration_count = IndexExpr::constant(count);
            for storage in &mut module.kernels[0].storages {
                if storage.ty.address_space == AddressSpace::Shared {
                    storage.ty.capacity_bytes = IndexExpr::constant(16384 * u64::from(slots));
                }
            }
            assert_eq!(
                verify_module(&module).construction,
                CheckStatus::Pass,
                "slots={slots}, count={count}"
            );
        }
    }
}

#[test]
fn malformed_storage_views_and_readonly_external_bindings_fail() {
    let mut module = pipeline_module();
    module.kernels[0].buffers[0].ty.mutable = true;
    assert_error(&module, DiagnosticCode::InvalidResource);
    let mut module = pipeline_module();
    module.kernels[0].buffers[0].ty.offset_bytes = IndexExpr::constant(1);
    assert_error(&module, DiagnosticCode::InvalidResource);
    let mut module = pipeline_module();
    module.kernels[0].buffers[0].ty.mapping = MappingSpec::Strided {
        strides_bytes: vec![IndexExpr::constant(1), IndexExpr::constant(1)],
    };
    assert_error(&module, DiagnosticCode::InvalidResource);
    let mut module = pipeline_module();
    module.kernels[0].buffers[0].ty.storage = "missing".into();
    assert_error(&module, DiagnosticCode::UnknownSymbol);
}

#[test]
fn overlapping_bound_buffers_and_insufficient_sqmma_participants_fail() {
    let mut module = pipeline_module();
    module.kernels[0].buffers[3].ty.storage = module.kernels[0].buffers[1].ty.storage.clone();
    assert_error(&module, DiagnosticCode::InvalidPipeline);
    let mut module = pipeline_module();
    module.kernels[0].roles[1].warps = vec![0];
    assert_error(&module, DiagnosticCode::InvalidParticipation);
}

#[test]
fn effects_retain_role_iteration_publication_and_join_domains() {
    let module = pipeline_module();
    let report = verify_module(&module);
    let mma = report
        .effects
        .iter()
        .find(|e| e.produces == vec![EventId::from("mma")])
        .unwrap();
    assert_eq!(mma.role, Some("consumer".into()));
    assert_eq!(
        mma.pipeline_iteration,
        Some(("pipe".into(), "consumer_i".into()))
    );
    let publish = report
        .effects
        .iter()
        .find(|e| !e.publishes.is_empty())
        .unwrap();
    assert_eq!(
        publish.publishes,
        vec![EventId::from("copy_a"), EventId::from("copy_b")]
    );
    assert!(publish.observes.is_empty());
    assert_eq!(report.effects[0].join_scope, Some(Scope::Block));
}

#[test]
fn a_zero_trip_loop_cannot_initialize_a_later_accumulator_read() {
    let mut module = pipeline_module();
    module.kernels[0].pipelines[0].iteration_count = IndexExpr::constant(0);
    let initialize = role_body(&mut module, 1).remove(1);
    // Removing the init shifted the pipeline loop to position 1.
    let ScheduleNodeKind::PipelineFor { body, .. } = &mut role_body(&mut module, 1)[1].kind else {
        panic!()
    };
    body.insert(1, initialize);
    assert_error(&module, DiagnosticCode::InvalidResource);
}
