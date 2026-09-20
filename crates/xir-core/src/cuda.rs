//! Minimal CUDA lowering for a checked, flat FP32 elementwise subset.
//! The admission rules are intentionally narrower than the portable AST.
use crate::ast::*;
use crate::diagnostic::*;
use crate::types::*;
use crate::verify::validate_module;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::fmt::Write;

#[derive(Clone, Debug, Serialize)]
pub struct CudaArtifact {
    pub source: String,
    pub kernel_name: String,
    pub entrypoint: String,
    pub arch: String,
    pub block: [u32; 3],
    pub params: Vec<ValueDef>,
    pub extent_parameter: usize,
    pub source_map: Vec<(usize, NodeId)>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Coordinate {
    Thread,
    Block,
    BlockSize,
    GridSize,
    BlockStart,
    Linear,
}

struct Admitted<'a> {
    kernel: &'a Kernel,
    extent_parameter: usize,
    buffer_parameters: HashMap<BufferId, usize>,
}

fn reject(node: Option<&ScheduleNode>, message: impl Into<String>) -> Diagnostic {
    let mut error = Diagnostic::error(DiagnosticCode::UnsupportedCapability, message);
    if let Some(node) = node {
        error.node_id = Some(node.meta.node_id.clone());
        error.source_span = node.meta.source_span.clone();
    }
    error
}

fn is_extent_bytes(expr: &IndexExpr, extent: &ValueId) -> bool {
    matches!(expr, IndexExpr::Mul(a, b) if
        (a.as_ref() == &IndexExpr::Value(extent.clone()) && b.as_ref() == &IndexExpr::Constant(4)) ||
        (b.as_ref() == &IndexExpr::Value(extent.clone()) && a.as_ref() == &IndexExpr::Constant(4)))
}

fn admit(module: &Module) -> Result<Admitted<'_>, Diagnostic> {
    if !module.target_request.is_cuda() || module.kernels.len() != 1 {
        return Err(reject(
            None,
            "CUDA lowering currently requires cuda:sm_120 and exactly one kernel",
        ));
    }
    let kernel = &module.kernels[0];
    if kernel.block[0] == 0 || kernel.block[0] > 1024 || kernel.block[1..] != [1, 1] {
        return Err(reject(
            None,
            "CUDA elementwise lowering requires a 1D block of 1..=1024 threads",
        ));
    }
    if !kernel.roles.is_empty() || !kernel.pipelines.is_empty() || !kernel.accumulators.is_empty() {
        return Err(reject(
            None,
            "CUDA roles, pipelines and accumulators are not implemented",
        ));
    }
    let extents: Vec<_> = kernel
        .params
        .iter()
        .enumerate()
        .filter(|(_, p)| p.ty == TypeRef::Index)
        .collect();
    if extents.len() != 1 {
        return Err(reject(
            None,
            "CUDA elementwise ABI requires exactly one Index extent parameter",
        ));
    }
    let (extent_parameter, extent) = extents[0];
    let mut pointer_parameters = HashSet::new();
    for (position, param) in kernel.params.iter().enumerate() {
        if position == extent_parameter {
            continue;
        }
        if !matches!(&param.ty, TypeRef::Pointer(p) if p.element == DType::F32 && p.address_space == AddressSpace::Global)
        {
            return Err(reject(
                None,
                "CUDA elementwise ABI supports only global FP32 pointers and one Index extent",
            ));
        }
        pointer_parameters.insert(position);
    }
    let mut buffer_parameters = HashMap::new();
    let mut used_parameters = HashSet::new();
    let mut used_storage = HashSet::new();
    for buffer in &kernel.buffers {
        let storage = kernel
            .storages
            .iter()
            .find(|s| s.id == buffer.ty.storage)
            .unwrap();
        let external = storage
            .external
            .as_ref()
            .ok_or_else(|| reject(None, "CUDA buffers require external storage"))?;
        let parameter = kernel
            .params
            .iter()
            .position(|p| &p.id == external)
            .unwrap();
        if !pointer_parameters.contains(&parameter)
            || !used_parameters.insert(parameter)
            || !used_storage.insert(storage.id.clone())
        {
            return Err(reject(
                None,
                "CUDA subset requires one storage and one buffer per pointer parameter",
            ));
        }
        if storage.ty.address_space != AddressSpace::Global
            || storage.ty.alignment_bytes != 4
            || !is_extent_bytes(&storage.ty.capacity_bytes, &extent.id)
            || buffer.ty.element != DType::F32
            || buffer.ty.shape != vec![IndexExpr::Value(extent.id.clone())]
            || buffer.ty.offset_bytes != IndexExpr::Constant(0)
            || buffer.ty.mapping
                != (MappingSpec::Strided {
                    strides_bytes: vec![IndexExpr::Constant(4)],
                })
        {
            return Err(reject(None, "CUDA buffers must be contiguous FP32 [n], offset 0, stride/alignment 4, and storage capacity n * 4"));
        }
        buffer_parameters.insert(buffer.id.clone(), parameter);
    }
    if used_parameters != pointer_parameters || used_storage.len() != kernel.storages.len() {
        return Err(reject(
            None,
            "CUDA ABI requires all pointers and storages to have exactly one declared view",
        ));
    }
    let mut coordinates = HashMap::new();
    let mut masks = HashMap::new();
    let mut stores = 0;
    for node in &kernel.body {
        use ScheduleNodeKind::*;
        match &node.kind {
            LaunchIndex { builtin, result } => {
                coordinates.insert(
                    result.id.clone(),
                    match builtin {
                        LaunchBuiltin::ThreadIdxX => Coordinate::Thread,
                        LaunchBuiltin::BlockIdxX => Coordinate::Block,
                        LaunchBuiltin::BlockDimX => Coordinate::BlockSize,
                        LaunchBuiltin::GridDimX => Coordinate::GridSize,
                    },
                );
            }
            Constant { result, .. } => {
                if !matches!(
                    result.ty,
                    TypeRef::Index | TypeRef::Predicate | TypeRef::Scalar(DType::F32)
                ) {
                    return Err(reject(
                        Some(node),
                        "CUDA constants support Index, Predicate and F32 only",
                    ));
                }
            }
            Binary {
                op,
                lhs,
                rhs,
                result,
            } => {
                let a = coordinates.get(lhs).copied();
                let b = coordinates.get(rhs).copied();
                if result.ty == TypeRef::Index {
                    let coordinate = match (op, a, b) {
                        (BinaryOp::Mul, Some(Coordinate::Block), Some(Coordinate::BlockSize)) |
                        (BinaryOp::Mul, Some(Coordinate::BlockSize), Some(Coordinate::Block)) => Coordinate::BlockStart,
                        (BinaryOp::Add, Some(Coordinate::BlockStart), Some(Coordinate::Thread)) |
                        (BinaryOp::Add, Some(Coordinate::Thread), Some(Coordinate::BlockStart)) => Coordinate::Linear,
                        _ => return Err(reject(Some(node), "CUDA Index arithmetic currently supports blockIdx.x * blockDim.x + threadIdx.x only")),
                    };
                    coordinates.insert(result.id.clone(), coordinate);
                } else if !matches!(result.ty, TypeRef::Predicate | TypeRef::Scalar(DType::F32)) {
                    return Err(reject(Some(node), "unsupported CUDA binary result type"));
                }
                if *op == BinaryOp::Less && a == Some(Coordinate::Linear) && rhs == &extent.id {
                    masks.insert(result.id.clone(), lhs.clone());
                }
            }
            Load {
                source,
                indices,
                mask,
                ..
            }
            | Store {
                destination: source,
                indices,
                mask,
                ..
            } => {
                if !matches!(source, MemoryRef::Buffer(_))
                    || indices.len() != 1
                    || masks.get(mask) != indices.first()
                {
                    return Err(reject(Some(node), "CUDA memory accesses require the same linear index i and explicit mask i < n"));
                }
                if matches!(node.kind, Store { .. }) {
                    stores += 1;
                }
            }
            KernelReturn => {}
            _ => {
                return Err(reject(
                    Some(node),
                    "operation is outside the flat CUDA elementwise subset",
                ))
            }
        }
    }
    if stores == 0 {
        return Err(reject(None, "CUDA elementwise kernel must contain a Store"));
    }
    Ok(Admitted {
        kernel,
        extent_parameter,
        buffer_parameters,
    })
}

/// Target admission is separate from construction and external device validation.
pub fn validate_cuda(module: &Module) -> DiagnosticBag {
    let mut diagnostics = validate_module(module);
    if !diagnostics.has_errors() {
        if let Err(error) = admit(module) {
            diagnostics.push(error);
        }
    }
    diagnostics
}

fn c_type(ty: &TypeRef) -> &'static str {
    match ty {
        TypeRef::Index => "uint64_t",
        TypeRef::Predicate => "bool",
        TypeRef::Scalar(DType::F32) => "float",
        TypeRef::Pointer(p) if p.mutable => "float*",
        TypeRef::Pointer(_) => "const float*",
        _ => unreachable!("target admission validates every emitted type"),
    }
}

pub fn emit_cuda(module: &Module) -> Result<CudaArtifact, DiagnosticBag> {
    let errors = validate_cuda(module);
    if errors.has_errors() {
        return Err(errors);
    }
    let admitted = admit(module).expect("validated above");
    let kernel = admitted.kernel;
    let mut names: HashMap<ValueId, String> = kernel
        .params
        .iter()
        .enumerate()
        .map(|(i, p)| (p.id.clone(), format!("p{i}")))
        .collect();
    let signature = kernel
        .params
        .iter()
        .enumerate()
        .map(|(i, p)| format!("{} p{i}", c_type(&p.ty)))
        .collect::<Vec<_>>()
        .join(", ");
    let mut source = String::from("// Generated by XIR. Do not edit; regenerate from the DSL.\n#include <cuda_runtime.h>\n#include <cstdint>\n#include <cstdio>\n#include <limits>\n\n");
    writeln!(
        source,
        "extern \"C\" __global__ void xir_kernel_0({signature}) {{"
    )
    .unwrap();
    let mut source_map = vec![];
    for (position, node) in kernel.body.iter().enumerate() {
        source_map.push((source.lines().count() + 1, node.meta.node_id.clone()));
        writeln!(source, "    // schedule node {position}").unwrap();
        for result in node.results() {
            names.insert(result.id.clone(), format!("v{position}"));
        }
        let value = |id: &ValueId| names.get(id).unwrap().as_str();
        let memory = |reference: &MemoryRef| {
            format!("p{}", admitted.buffer_parameters[reference.buffer_id()])
        };
        use ScheduleNodeKind::*;
        match &node.kind {
            LaunchIndex { builtin, result } => {
                let builtin = match builtin {
                    LaunchBuiltin::ThreadIdxX => "threadIdx.x",
                    LaunchBuiltin::BlockIdxX => "blockIdx.x",
                    LaunchBuiltin::BlockDimX => "blockDim.x",
                    LaunchBuiltin::GridDimX => "gridDim.x",
                };
                writeln!(
                    source,
                    "    const uint64_t {} = static_cast<uint64_t>({builtin});",
                    value(&result.id)
                )
                .unwrap();
            }
            Constant {
                value: literal,
                result,
            } => {
                let literal = match literal {
                    Literal::Index(n) => format!("{n}ULL"),
                    Literal::Bool(b) => b.to_string(),
                    Literal::Float(n) => format!("static_cast<float>({n:.17e})"),
                    _ => unreachable!(),
                };
                writeln!(
                    source,
                    "    const {} {} = {literal};",
                    c_type(&result.ty),
                    value(&result.id)
                )
                .unwrap();
            }
            Binary {
                op,
                lhs,
                rhs,
                result,
            } => {
                let operator = match op {
                    BinaryOp::Add => "+",
                    BinaryOp::Sub => "-",
                    BinaryOp::Mul => "*",
                    BinaryOp::Div => "/",
                    BinaryOp::Less => "<",
                    BinaryOp::Equal => "==",
                    BinaryOp::And => "&&",
                    BinaryOp::Or => "||",
                };
                writeln!(
                    source,
                    "    const {} {} = {} {operator} {};",
                    c_type(&result.ty),
                    value(&result.id),
                    value(lhs),
                    value(rhs)
                )
                .unwrap();
            }
            Load {
                source: reference,
                indices,
                mask,
                masked_value,
                result,
            } => {
                writeln!(
                    source,
                    "    const float {} = {} ? {}[{}] : {};",
                    value(&result.id),
                    value(mask),
                    memory(reference),
                    value(&indices[0]),
                    value(masked_value)
                )
                .unwrap();
            }
            Store {
                destination,
                indices,
                value: stored,
                mask,
            } => {
                writeln!(
                    source,
                    "    if ({}) {{ {}[{}] = {}; }}",
                    value(mask),
                    memory(destination),
                    value(&indices[0]),
                    value(stored)
                )
                .unwrap();
            }
            KernelReturn => source.push_str("    return;\n"),
            _ => unreachable!(),
        }
    }
    source.push_str("}\n\n");
    source.push_str(include_str!("cuda_runtime.cuh"));
    // The wrapper uses parameter positions; source names and imported IDs are never C identifiers.
    writeln!(source, "\nextern \"C\" int xir_launch(void** host, const uint64_t* lengths, uint64_t n, char* error, size_t error_capacity) {{").unwrap();
    source.push_str("    if (!host || !lengths) return xir_error(error, error_capacity, \"missing ABI argument arrays\");\n");
    source.push_str("    if (n > std::numeric_limits<size_t>::max() / sizeof(float)) return xir_error(error, error_capacity, \"extent byte size overflows\");\n");
    for parameter in admitted
        .buffer_parameters
        .values()
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
    {
        writeln!(source, "    if (lengths[{parameter}] < n || lengths[{parameter}] > std::numeric_limits<size_t>::max() / sizeof(float) || (lengths[{parameter}] && !host[{parameter}])) return xir_error(error, error_capacity, \"invalid host buffer extent\");").unwrap();
    }
    source.push_str("    if (n == 0) return 0;\n    int device = 0;\n    XIR_CUDA(cudaGetDevice(&device));\n    cudaDeviceProp properties{};\n    XIR_CUDA(cudaGetDeviceProperties(&properties, device));\n    if (properties.major != 12 || properties.minor != 0) return xir_error(error, error_capacity, \"artifact requires compute capability 12.0\");\n");
    writeln!(
        source,
        "    constexpr uint32_t block = {};",
        kernel.block[0]
    )
    .unwrap();
    source.push_str("    const uint64_t grid = n / block + (n % block != 0);\n    if (block > static_cast<uint32_t>(properties.maxThreadsPerBlock) || block > static_cast<uint32_t>(properties.maxThreadsDim[0]) || grid > static_cast<uint64_t>(properties.maxGridSize[0])) return xir_error(error, error_capacity, \"launch exceeds device limits\");\n");
    for (position, param) in kernel.params.iter().enumerate() {
        if !matches!(param.ty, TypeRef::Pointer(_)) {
            continue;
        }
        writeln!(source, "    XirDeviceBuffer d{position};\n    const size_t bytes{position} = static_cast<size_t>(lengths[{position}]) * sizeof(float);\n    XIR_CUDA(cudaMalloc(&d{position}.pointer, bytes{position}));\n    XIR_CUDA(cudaMemcpy(d{position}.pointer, host[{position}], bytes{position}, cudaMemcpyHostToDevice));").unwrap();
    }
    let arguments = kernel
        .params
        .iter()
        .enumerate()
        .map(|(i, p)| {
            if p.ty == TypeRef::Index {
                "n".to_string()
            } else {
                format!("static_cast<{}>(d{i}.pointer)", c_type(&p.ty))
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    writeln!(
        source,
        "    xir_kernel_0<<<static_cast<uint32_t>(grid), block>>>({arguments});"
    )
    .unwrap();
    source.push_str("    XIR_CUDA(cudaGetLastError());\n    XIR_CUDA(cudaDeviceSynchronize());\n");
    for (position, param) in kernel.params.iter().enumerate() {
        if matches!(&param.ty, TypeRef::Pointer(p) if p.mutable) {
            writeln!(source, "    XIR_CUDA(cudaMemcpy(host[{position}], d{position}.pointer, bytes{position}, cudaMemcpyDeviceToHost));").unwrap();
        }
    }
    source.push_str("    return 0;\n}\n");
    Ok(CudaArtifact {
        source,
        kernel_name: "xir_kernel_0".into(),
        entrypoint: "xir_launch".into(),
        arch: module.target_request.arch.clone(),
        block: kernel.block,
        params: kernel.params.clone(),
        extent_parameter: admitted.extent_parameter,
        source_map,
    })
}
