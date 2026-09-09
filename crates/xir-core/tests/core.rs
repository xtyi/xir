use xir_core::ast::{Kernel, Parameter, ScheduleNodeKind};
use xir_core::{
    deserialize_module, dump_module, serialize_module, DType, Dim, FragmentType, Module,
    ModuleBuilder, TargetSpec, TypeRef,
};

#[test]
fn module_round_trips_through_json() {
    let module = Module {
        name: "smoke".into(),
        version: 1,
        target: TargetSpec::musa_mp31(),
        kernels: vec![Kernel {
            name: "add".into(),
            params: vec![Parameter {
                name: "x".into(),
                ty: "ptr<f32>".into(),
            }],
            body: vec![],
        }],
    };
    let encoded = serialize_module(&module).unwrap();
    assert_eq!(deserialize_module(&encoded).unwrap(), module);
}

#[test]
fn target_dump_builder_and_diagnostics_work() {
    let target = TargetSpec::default();
    assert_eq!(target.arch, "mp31");
    let mut builder = ModuleBuilder::new("m", target);
    builder.kernel("k").pipeline("p", 0).finish();
    let errors = builder.finish_checked().unwrap_err();
    assert!(errors.has_errors());
    assert!(dump_module(&Module::default()).contains("Module name="));
}

#[test]
fn builder_assigns_deterministic_nested_ids() {
    let mut builder = ModuleBuilder::new("m", TargetSpec::default());
    builder
        .kernel("k")
        .body(|body| body.for_loop("i", "0", "2", |inner| inner.assign("x", "i")))
        .finish();
    let module = builder.finish();
    assert_eq!(module.kernels[0].body[0].meta.node_id, "m.k.n0");
    assert!(matches!(
        module.kernels[0].body[0].kind,
        ScheduleNodeKind::For { .. }
    ));
}

#[test]
fn type_system_round_trips_fragment_type() {
    let ty = TypeRef::Fragment(FragmentType {
        family: "musa.sqmma".into(),
        role: "accumulator".into(),
        element: DType::F32,
        logical_shape: vec![Dim::Static(128), Dim::Static(128)],
        representation: "mp31.c.v1".into(),
    });
    let json = serde_json::to_string(&ty).unwrap();
    assert_eq!(serde_json::from_str::<TypeRef>(&json).unwrap(), ty);
}
