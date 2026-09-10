import inspect
import json
from pathlib import Path
import runpy
import subprocess

import pytest
from xir import DSLParseError, IRValidationError, KernelProgram, NativeUnavailableError, kernel, kernel_from_source

ROOT = Path(__file__).resolve().parents[2]


def walk(body):
    for node in body:
        yield node
        data = node["kind"].get("data", {})
        for key in ("body", "then_body", "else_body"):
            yield from walk(data.get(key, []))
        for role in data.get("roles", []):
            yield from walk(role["body"])


def test_pipeline_source_builds_canonical_rust_schema():
    program = runpy.run_path(str(ROOT / "examples" / "pipeline.py"))["pipeline_example"]
    tree = program.to_dict()
    assert tree["schema_version"] == 2
    assert tree["kernels"][0]["block"] == [160, 1, 1]
    assert tree["kernels"][0]["roles"][0]["warps"] == [4]
    nodes = list(walk(tree["kernels"][0]["body"]))
    assert len({node["meta"]["node_id"] for node in nodes}) == len(nodes)
    assert all(node["meta"]["node_id"] for node in nodes)
    assert "Sqmma" in {node["kind"]["kind"] for node in nodes}
    report = program.verify()
    assert report["construction"] == "Pass"
    assert report["status"] == report["target"] == "Unknown"
    imported = KernelProgram.from_dict(json.loads(program.dump()))
    assert imported.to_dict() == tree


def test_for_and_if_construct_structured_ssa():
    program = kernel_from_source('''
    def sum_indices(n: "index"):
        total = index(0)
        for i in range(n):
            total = total + i
        if total < n:
            total = total + 1
        else:
            total = index(0)
    ''')
    body = program.to_dict()["kernels"][0]["body"]
    loop = next(node["kind"]["data"] for node in body if node["kind"]["kind"] == "For")
    carried = loop["carried"][0]
    assert len({carried["initial"], carried["argument"]["id"], carried["result"]["id"]}) == 3
    assert loop["body"][-1]["kind"]["kind"] == "Yield"
    conditional = next(node["kind"]["data"] for node in body if node["kind"]["kind"] == "If")
    assert len(conditional["results"]) == 1
    assert program.verify()["construction"] == "Pass"


def test_load_store_types_and_derived_accesses():
    program = kernel_from_source('''
    def copy_one(p: "ptr[f32]"):
        s = storage("Global", 16, 4, external=p)
        b = buffer(s, "f32", (4,), (4,), 0, True)
        x = load(b, (0,), True, 0.0)
        store(b, (1,), x, True)
    ''')
    report = program.verify()
    accesses = [access for node in report["effects"] for access in node["accesses"]]
    assert [access["mode"] for access in accesses] == ["Read", "Write"]
    assert accesses[0]["storage"] == accesses[1]["storage"]


@pytest.mark.parametrize("source", [
    'def bad(x):\n    pass',
    'async def bad():\n    pass',
    'def bad(x: "index" = 1):\n    pass',
    'def bad(*args):\n    pass',
    'def bad(*, x: "index"):\n    pass',
    'def bad():\n    x = index(1, 2)',
    'def bad():\n    x = index(value=1, unknown=2)',
    'def bad():\n    x = index(**{})',
    'def bad():\n    x = index(*[1])',
    'def bad():\n    for i in range(4):\n        pass\n    else:\n        pass',
    'def bad():\n    with role([0]):\n        pass',
    'def bad():\n    for i in stages(p):\n        pass',
    'def bad():\n    x = index(0)\n    s = storage("Shared", 16, 4)',
    'def bad():\n    x = index(0) / index(1)',
    'def bad():\n    x = -1 // 2',
    'def bad():\n    x = 1 < 2 < 3',
    'def bad():\n    x = [v for v in range(3)]',
    'def bad():\n    return 1',
    'def bad():\n    if True:\n        return',
    'def bad():\n    x = index(0)\n    x += 1',
    'def bad():\n    a = b = index(0)',
    'def bad():\n    x: "index" = i32(0)',
    'def bad():\n    x = index(1)\n    y = index(x)',
    'def bad():\n    index(1)',
    'def bad():\n    x = captured_value',
    'def bad():\n    pass\n\ndef another():\n    pass',
])
def test_unsupported_syntax_is_rejected_with_source_location(source):
    with pytest.raises(DSLParseError) as failure:
        kernel_from_source(source, filename="invalid.py").build()
    assert failure.value.filename == "invalid.py"
    assert failure.value.line >= 1


def test_native_checks_receive_frontend_values_and_source_spans():
    with pytest.raises(IRValidationError) as failure:
        kernel_from_source('def bad(n: "index"):\n    x = n + f32(1.0)', filename="typed.py").build()
    diagnostic = next(item for item in failure.value.diagnostics if item["code"] == "InvalidType")
    assert diagnostic["source_span"]["file"] == "typed.py"
    assert diagnostic["source_span"]["start_line"] == 2
    assert diagnostic["node_id"]


def test_native_pipeline_error_is_not_hidden_by_python():
    source = (ROOT / "examples" / "pipeline.py").read_text()
    # Capture the decorated function without executing the source or its imports.
    source = source[source.index("@kernel"):source.index('\n\nif __name__')]
    source = source.replace("                await_event(mma)\n", "")
    with pytest.raises(IRValidationError) as failure:
        kernel_from_source(source).build()
    assert "PendingAccess" in {item["code"] for item in failure.value.diagnostics}


def test_json_import_revalidates_instead_of_trusting_cached_effects():
    program = kernel_from_source('def test():\n    x = index(0)')
    module = program.to_dict()
    module["kernels"][0]["body"][0]["meta"]["effects"] = []
    with pytest.raises(IRValidationError):
        KernelProgram.from_dict(module).build()
    assert "effects" not in program.to_dict()["kernels"][0]["body"][0]["meta"]
    module = program.to_dict()
    module["kernels"][0]["body"][0]["kind"]["data"]["result"]["ty"] = "Predicate"
    with pytest.raises(IRValidationError):
        KernelProgram.from_dict(module).build()


def test_decorated_source_line_numbers_are_absolute():
    @kernel
    def located(n: "index"):
        x = n + f32(1.0)

    expected_line = inspect.currentframe().f_lineno - 2
    with pytest.raises(IRValidationError) as failure:
        located.build()
    diagnostic = next(item for item in failure.value.diagnostics if item["code"] == "InvalidType")
    assert diagnostic["source_span"]["start_line"] == expected_line
    assert diagnostic["source_span"]["file"] == __file__


def test_missing_native_is_explicit():
    with pytest.raises(NativeUnavailableError):
        kernel_from_source('def empty():\n    pass').build(verifier="/nonexistent/xir-verify")


def test_cli_rejects_malformed_json_and_unknown_command(rust_verifier):
    result = subprocess.run([str(rust_verifier), "construct"], input="{broken", text=True, capture_output=True, check=True)
    assert json.loads(result.stdout)["diagnostics"][0]["code"] == "ParseError"
    result = subprocess.run([str(rust_verifier), "compile"], text=True, capture_output=True)
    assert result.returncode == 2


def test_safe_source_does_not_execute_annotations_or_calls(tmp_path):
    sentinel = tmp_path / "should-not-exist"
    source = f'def bad(x: __import__("pathlib").Path({str(sentinel)!r}).touch()):\n    pass'
    with pytest.raises(DSLParseError):
        kernel_from_source(source).build()
    assert not sentinel.exists()


def test_branch_local_value_cannot_escape_without_a_join():
    with pytest.raises(DSLParseError):
        kernel_from_source('def bad(c: "predicate"):\n    if c:\n        local = index(0)\n    x = local').build()


def test_source_mapping_preserves_original_indentation():
    source = '    def indented(n: "index"):\n        x = n + f32(1.0)'
    with pytest.raises(IRValidationError) as failure:
        kernel_from_source(source).build()
    diagnostic = next(item for item in failure.value.diagnostics if item["code"] == "InvalidType")
    assert diagnostic["source_span"]["start_column"] == 12


def test_resource_constructor_does_not_drop_arguments():
    source = 'def bad():\n    s = storage("Shared", 16, 4, extra=True)'
    with pytest.raises(DSLParseError, match="keyword"):
        kernel_from_source(source).build()


def test_source_block_configuration_cannot_be_silently_overridden():
    source = '@kernel(block=(32, 1, 1))\ndef empty():\n    pass'
    with pytest.raises(DSLParseError, match="dimensions differ"):
        kernel_from_source(source, block=(64, 1, 1)).build()


def test_if_join_type_mismatch_is_rejected():
    source = 'def bad(c: "predicate"):\n    if c:\n        x = index(0)\n    else:\n        x = f32(0.0)'
    with pytest.raises(DSLParseError, match="matching types"):
        kernel_from_source(source).build()


def test_loop_induction_and_new_body_values_do_not_escape():
    for name in ("i", "inner"):
        source = f'def bad():\n    for i in range(0):\n        inner = i\n    result = {name}'
        with pytest.raises(DSLParseError, match="defined symbol"):
            kernel_from_source(source).build()
