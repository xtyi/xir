"""DSL -> typed IR -> generated CUDA -> nvcc -> GPU -> independent CPU oracle."""
from array import array
import argparse
import ctypes
import json
from pathlib import Path

from xir import kernel


@kernel(target="cuda:sm_120", block=(256, 1, 1))
def elementwise_add(a_ptr: "const_ptr[f32]", b_ptr: "const_ptr[f32]", c_ptr: "ptr[f32]", n: "index"):
    a_storage = storage("Global", n * 4, 4, external=a_ptr)
    b_storage = storage("Global", n * 4, 4, external=b_ptr)
    c_storage = storage("Global", n * 4, 4, external=c_ptr)
    a = buffer(a_storage, "f32", (n,), (4,), 0, False)
    b = buffer(b_storage, "f32", (n,), (4,), 0, False)
    c = buffer(c_storage, "f32", (n,), (4,), 0, True)
    i = block_idx_x() * block_dim_x() + thread_idx_x()
    active = i < n
    x = load(a, (i,), active, 0.0)
    y = load(b, (i,), active, 0.0)
    store(c, (i,), x + y, active)


def validate(compiled, sizes):
    results = []
    for n in sizes:
        if n < 0:
            raise ValueError("sizes must be nonnegative")
        # Both exact-sized and padded arrays exercise the tail contract.
        for padding in (0, 17):
            a = array("f", ((i % 251 - 125) / 13 for i in range(n + padding)))
            b = array("f", ((i % 127 - 63) / 7 for i in range(n + padding)))
            sentinel = -12345.25
            output = array("f", [sentinel]) * (n + padding)
            before_a, before_b = a.tobytes(), b.tobytes()
            compiled.run(a, b, output, n)
            # Python double addition, then an independent FP32 rounding step.
            for i in range(n):
                expected = ctypes.c_float(float(a[i]) + float(b[i])).value
                if output[i] != expected:
                    raise AssertionError(f"n={n}, padding={padding}, i={i}: got {output[i]}, expected {expected}")
            if any(value != sentinel for value in output[n:]):
                raise AssertionError(f"tail guard was modified: n={n}, padding={padding}")
            if a.tobytes() != before_a or b.tobytes() != before_b:
                raise AssertionError("read-only input was modified")
            results.append({"n": n, "padding": padding, "correct": True})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="build/cuda_elementwise_add")
    parser.add_argument("--sizes", nargs="+", type=int, default=[0, 1, 31, 32, 33, 255, 256, 257, 1023, 1024, 1025, 1000003])
    args = parser.parse_args()
    static_report = elementwise_add.verify()
    compiled = elementwise_add.compile_cuda(args.output_dir)
    results = validate(compiled, args.sizes)
    evidence = {"source": str(compiled.source_path), "library": str(compiled.library_path),
                "construction": static_report["construction"], "target": static_report["target"],
                "static_status": static_report["status"], "device_validation": "Pass", "cases": results}
    report_path = Path(compiled.library_path).with_name("validation.json")
    report_path.write_text(json.dumps(evidence, indent=2))
    print(json.dumps({"cases_passed": len(results), "generated_cuda": str(compiled.source_path), "validation": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
