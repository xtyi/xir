"""Compiler tests run without a GPU; device tests require XIR_TEST_CUDA=1."""
from array import array
import copy
import json
import os
from pathlib import Path
import runpy

import pytest
from xir import IRValidationError, KernelProgram, kernel_from_source
from xir.cuda import CompiledCudaKernel, CudaCompileError

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def add_program():
    return runpy.run_path(str(ROOT / "examples" / "cuda_elementwise_add.py"))["elementwise_add"]


def test_cuda_pipeline_emits_actual_nodes_and_keeps_target_admission_separate(add_program):
    report = add_program.verify()
    assert report["construction"] == report["target"] == "Pass"
    assert report["status"] == "Unknown"
    artifact = add_program.emit_cuda()
    assert artifact["arch"] == "sm_120"
    assert artifact["block"] == [256, 1, 1]
    assert artifact["extent_parameter"] == 3
    assert "static_cast<uint64_t>(blockIdx.x)" in artifact["source"]
    assert "static_cast<uint64_t>(threadIdx.x)" in artifact["source"]
    assert " ? p0[" in artifact["source"]
    assert " ? p1[" in artifact["source"]
    assert "p2[" in artifact["source"]
    assert "cudaDeviceSynchronize()" in artifact["source"]
    assert len(artifact["source_map"]) == len(add_program.to_dict()["kernels"][0]["body"])
    changed = kernel_from_source(add_program.source.replace("x + y", "x - y")).emit_cuda()
    assert changed["source"] != artifact["source"]
    assert " - " in changed["source"]


@pytest.mark.parametrize("mutation", ["mask", "index", "stride", "capacity", "block", "shared"])
def test_cuda_admission_rejects_unproved_access_or_unsupported_layout(add_program, mutation):
    module = add_program.to_dict()
    kernel = module["kernels"][0]
    nodes = kernel["body"]
    store = next(node["kind"]["data"] for node in nodes if node["kind"]["kind"] == "Store")
    if mutation == "mask":
        mask = next(node["kind"]["data"] for node in nodes if node["kind"]["kind"] == "Binary" and node["kind"]["data"]["op"] == "Less")
        mask["op"] = "Equal"
    elif mutation == "index":
        store["indices"] = [kernel["params"][3]["id"]]
    elif mutation == "stride":
        kernel["buffers"][2]["ty"]["mapping"]["data"]["strides_bytes"][0]["data"] = 8
    elif mutation == "capacity":
        kernel["storages"][2]["ty"]["capacity_bytes"] = {"kind": "Constant", "data": 4}
    elif mutation == "block":
        kernel["block"] = [32, 2, 1]
    else:
        kernel["params"][2]["ty"]["Pointer"]["address_space"] = "Shared"
        kernel["storages"][2]["ty"]["address_space"] = "Shared"
    with pytest.raises(IRValidationError):
        KernelProgram.from_dict(module).emit_cuda()


def test_cuda_unsupported_node_is_not_silently_dropped(add_program):
    source = add_program.source + "    for j in range(n):\n        pass\n"
    program = kernel_from_source(source)
    report = program.verify()
    assert report["construction"] == "Pass"
    assert report["target"] == report["status"] == "Fail"
    with pytest.raises(IRValidationError, match="outside the flat CUDA"):
        program.emit_cuda()


def test_cuda_emission_revalidates_imports_and_launch_index_types(add_program):
    module = add_program.to_dict()
    coordinate = next(node["kind"]["data"] for node in module["kernels"][0]["body"] if node["kind"]["kind"] == "LaunchIndex")
    coordinate["result"]["ty"] = {"Scalar": "F32"}
    with pytest.raises(IRValidationError, match="launch coordinate"):
        KernelProgram.from_dict(module).emit_cuda()


def test_musa_does_not_accidentally_use_cuda_lowering():
    program = kernel_from_source('def empty():\n    pass')
    with pytest.raises(IRValidationError, match="cuda:sm_120"):
        program.emit_cuda()


def test_missing_nvcc_reports_compiler_failure(add_program, tmp_path):
    with pytest.raises(CudaCompileError, match="not found"):
        add_program.compile_cuda(tmp_path, nvcc="/nonexistent/nvcc")


def test_invalid_host_arguments_are_rejected_before_native_execution(add_program):
    # No CUDA dependency is required to test the public ABI guards.
    compiled = object.__new__(CompiledCudaKernel)
    compiled._artifact = copy.deepcopy(add_program.emit_cuda())
    def unexpected_launch(*args):
        pytest.fail("invalid arguments reached the native launcher")
    compiled._launch = unexpected_launch
    a, b, c = (array("f", [1, 2]) for _ in range(3))
    for n in (-1, True, 0.5, 2**64):
        with pytest.raises(ValueError, match="extent"):
            compiled.run(a, b, c, n)
    with pytest.raises(ValueError, match="requires at least"):
        compiled.run(a, b, c, 3)
    with pytest.raises(TypeError, match="array"):
        compiled.run(array("d", [1, 2]), b, c, 2)
    with pytest.raises(ValueError, match="distinct"):
        compiled.run(a, b, a, 2)
    with pytest.raises(ValueError, match="arguments"):
        compiled.run(a, b, c)


@pytest.mark.skipif(os.environ.get("XIR_TEST_CUDA") != "1", reason="set XIR_TEST_CUDA=1 to compile and run on an SM120 GPU")
def test_cuda_device_add_and_non_add_expression(add_program, tmp_path):
    compiled = add_program.compile_cuda(tmp_path / "add")
    validate = runpy.run_path(str(ROOT / "examples" / "cuda_elementwise_add.py"))["validate"]
    results = validate(compiled, [0, 1, 31, 32, 33, 255, 256, 257, 1023, 1024, 1025, 1000003])
    assert len(results) == 24
    evidence = json.loads(compiled.library_path.with_name("compile.json").read_text())
    assert evidence["returncode"] == 0
    subtraction = kernel_from_source(add_program.source.replace("x + y", "x - y")).compile_cuda(tmp_path / "sub")
    a, b, c = array("f", [1, 2, 3]), array("f", [4, 5, 7]), array("f", [0, 0, 0])
    subtraction.run(a, b, c, 3)
    assert list(c) == [-3, -3, -4]
