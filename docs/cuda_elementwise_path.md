# CUDA Elementwise Add 最小闭环

当前可以从 XIR DSL 生成 CUDA、调用 nvcc 编译并在本机 GPU 执行 `C[i] = A[i] + B[i]`。这条路径先固定为 `cuda:sm_120`、一维连续 FP32、一个 Index 长度参数、256 threads/block。GPU 调用使用同步 host-array API，便于先检查每一层。

## 运行

```sh
cargo build --bin xir-verify
PYTHONPATH=python .venv/bin/python examples/cuda_elementwise_add.py
```

源码是 [examples/cuda_elementwise_add.py](../examples/cuda_elementwise_add.py)。它的计算部分为：

```python
i = block_idx_x() * block_dim_x() + thread_idx_x()
active = i < n
x = load(a, (i,), active, 0.0)
y = load(b, (i,), active, 0.0)
store(c, (i,), x + y, active)
```

A/B/C 的 Storage/Buffer 声明也在该函数中：capacity 为 `n * 4` 字节，shape 为 `(n,)`，stride/alignment 为 4，A/B 只读，C 可写。

也可以直接调用：

```python
from array import array
from examples.cuda_elementwise_add import elementwise_add

compiled = elementwise_add.compile_cuda("build/my_add")
a = array("f", [1, 2, 3])
b = array("f", [4, 5, 6])
c = array("f", [0, 0, 0])
compiled.run(a, b, c, 3)
assert list(c) == [5, 7, 9]
print(compiled.source_path)
```

`emit_cuda()` 只生成 artifact 字典，不调用 nvcc；`compile_cuda()` 生成并加载 shared library，但不启动 kernel；`run()` 按 DSL 参数顺序接受参数并同步执行。

## 各层实现

```text
@kernel(target="cuda:sm_120")
    -> Python AST parser
    -> schema v2 typed Schedule AST
    -> Rust construction verifier
    -> CUDA subset admission / access checks
    -> Rust node-by-node CUDA emitter
    -> nvcc -> shared library
    -> ctypes -> generated host wrapper
    -> cudaMalloc / H2D / launch / synchronize / D2H
    -> independent CPU FP32 oracle
```

| 层 | 实现位置 | 本次内容 |
|---|---|---|
| DSL frontend | `python/xir/dsl.py` | 显式 target、4 个 x-axis launch intrinsic、emit/compile API |
| Typed IR | `crates/xir-core/src/ast.rs` | `LaunchIndex { builtin, result: Index }`；沿用 Binary/Load/Store |
| Construction checks | `crates/xir-core/src/verify.rs` | LaunchIndex 类型检查、block-uniform / thread-varying 区分 |
| Target admission | `crates/xir-core/src/cuda.rs` | 验证连续 FP32 view、参数 ABI、线性索引和一致的 `i < n` mask；拒绝不支持的节点 |
| CUDA emitter | `crates/xir-core/src/cuda.rs` | 每个 AST node 生成对应语句；uint64_t 索引；masked load/store；生成 source map |
| Compilation | `python/xir/cuda.py` | nvcc shared-library 编译、日志、source/module/artifact 保存和编译失败诊断 |
| Runtime | `cuda_runtime.cuh` 与 emitter 生成的 wrapper | 校验设备/extent/launch，独立 device allocations、同步传输/执行和错误返回 |
| Validation | `examples/cuda_elementwise_add.py`、`python/tests/test_cuda.py` | CPU reference、边界、canary、不支持输入和真实 GPU 回归 |

生成代码没有固定 Add 模板。把 DSL 中 `x + y` 改为 `x - y` 会生成 Sub，回归测试实际编译运行了这两个不同程序。C identifier 使用 compiler 分配的名字；用户提供的 IR ID 不直接拼接成 C 源码。`artifact.json` 保存 generated line → NodeId 的映射。

## Artifact 与验证状态

每次编译写入 `build/cuda_elementwise_add/<digest>/`；digest 包含 IR、生成源码、nvcc 版本/路径与编译选项。

| 文件 | 用途 |
|---|---|
| `module.json` | 经 native construction checks 的 typed IR snapshot |
| `artifact.json` | 生成源码、ABI、target、block 和 source mapping |
| `kernel.cu` | 实际编译的 CUDA kernel 与最小 host wrapper |
| `kernel.so` | nvcc 生成的可加载库 |
| `compile.json` | 命令、nvcc 版本、返回码、stdout/stderr 和 identity digest |
| `validation.json` | 示例的数值、输入不变性与 tail canary 检查结果 |

编译使用 `--fmad=false`，避免将分离的 FP32 Mul/Add 擅自融合；不打开 fast math。当前没有编译缓存，重复调用仍编译，并原子替换 library，避免截断已加载的映射。

CUDA 子集的 `verify()` 可以返回 `construction: Pass, target: Pass`；总体仍为 `Unknown`，因为静态 report 不包含实际编译/设备证据。`emit_cuda` 和 `compile_cuda` 每次重新验证 snapshot，要求 construction 和该 CUDA target admission 通过。生成 wrapper 负责剩余的调用前提，设备数值结果单独记录，不把一次 example 通过扩展成任意程序已验证。

Launch 为一元素一线程，`grid = ceil(n / block)`。索引在乘法之前转换为 uint64_t；load 的三元表达式与 store 的分支只在 `i < n` 时访问内存。`n=0` 跳过 launch；超过设备 grid/block 限制会报错，没有将有限 grid 当作无限数据覆盖。

初始 ABI 只接受独立的 `array('f')`，长度至少 n。每个参数单独分配 device storage，输入均先 H2D，mutable 参数 D2H；host array 比 n 长时尾部也保留和取回，便于检查 canary。不是零拷贝或性能测试接口。直接手动 launch 生成的 raw kernel 时，调用者仍须满足相同 extent、device、launch 和 alias 前提。

## 本机验证记录

2026-09-10：RTX 5060 Laptop GPU，compute capability 12.0，nvcc 13.1.80。

- Rust：21 tests passed；Clippy `-D warnings` 通过。
- Python：`XIR_TEST_CUDA=1` 下 55 tests passed，包含实际 Add/Sub GPU 编译执行。
- Add：`n = 0, 1, 31, 32, 33, 255, 256, 257, 1023, 1024, 1025, 1000003`，每个长度分别使用精确分配和额外 17 元素的 canary，共 24 组通过。
- Reference 使用 Python double 加法再独立舍入到 FP32，逐元素相等；检查只读输入未变和输出尾部未变。
- Compute Sanitizer memcheck **未完成**：当前 WSL/WDDM 环境报 `Failed to initialize WDDM debugger interface` 和 `Device not supported`，退出码 99。没有将其计作通过，也没有修改宿主机调试设置。

重跑测试：

```sh
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
PYTHONPATH=python XIR_TEST_CUDA=1 .venv/bin/python -m pytest -q python/tests
PYTHONPATH=python compute-sanitizer --tool memcheck --error-exitcode 99 \
    .venv/bin/python examples/cuda_elementwise_add.py
```

不设置 `XIR_TEST_CUDA=1` 时只跳过 GPU integration case，其余 compiler/ABI tests 不要求 CUDA 环境。

## 后续边界

当前 CUDA emitter 不支持 For/If region、非连续 view、其他 dtype、共享内存、roles、pipeline、SQMMA 或其他架构；这些在 emission 前明确拒绝。Target profile 目前是一个固定子集函数，尚未泛化成 registry。MUSA pipeline 的 construction 示例保持原有边界，MUSA lowering/mtcc 尚未接入。未做性能调优或 benchmark。
