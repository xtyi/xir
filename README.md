# XIR

XIR 是面向显式 GPU schedule 的实验性 DSL。当前实现了 schema v2、Python source frontend、Rust typed AST、递归 construction/protocol checks，以及从节点重新推导的 effects。目标语义见 [Schedule AST IR 设计](schedule_ast_ir_design.md)。

**目前不能生成或运行 GPU kernel。** `construction: Pass` 只表示通过已实现的结构检查；target adapter、MUSA lowering、设备 oracle 尚未实现，总体 verification 状态为 `Unknown`。

## 构建与运行

```sh
cargo build --bin xir-verify
python3 -m venv .venv
.venv/bin/python -m pip install -e . pytest
PYTHONPATH=python .venv/bin/python examples/pipeline.py
```

Python 通过 JSON subprocess 调用 Rust `xir-verify`，没有第二套独立验证器。默认查找 `PATH` 和仓库 `target/debug/xir-verify`；安装包或自定义 Cargo target directory 时设置 `XIR_VERIFY_BIN`。缺少 native executable 会明确报错，不会回退成未验证的 Python tree。

```python
from xir import kernel_from_source

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
module = program.build().to_dict()
report = program.verify()
assert report["construction"] == "Pass"
assert report["status"] == "Unknown"
```

文件中的函数也可以使用 `@kernel` / `@kernel(block=(160, 1, 1))`。函数体只被解析，不执行；intrinsic 名称是 DSL 语法，不需要导入运行时函数。参数要求显式标注，例如 `"index"`、`"f32"`、`"predicate"`、`"ptr[f32]"`、`"const_ptr[bf16]"`。不捕获 Python globals/closures，不求值注解；不支持的语法和额外参数会被拒绝。源码不可获取时使用 `kernel_from_source()`。

`For` 使用 unsigned 64-bit Index，step 必须为正的常量；loop induction 和新建的 body-local values 采用词法作用域。已有普通变量的重绑定变为 loop-carried arguments/results 和 `Yield`；零次迭代的结果来自 initial value。`If` 通过显式 results/yields 合并普通值；两个分支的类型必须一致。这里描述的是 IR contract，尚无执行这些节点的解释器或 backend。

完整协议示例见 [examples/pipeline.py](examples/pipeline.py)，同一 schema 的 Rust 构造示例见 [support/mod.rs](crates/xir-core/examples/support/mod.rs)。它们每轮重复读取相同的两个输入 view，只演示协议，不是完成 K 维遍历和输出存储的数值 GEMM。

也可通过 JSON 导入和命令行访问相同的验证入口：

```sh
cargo run --quiet --example pipeline > /tmp/xir-pipeline.json
target/debug/xir-verify construct < /tmp/xir-pipeline.json
target/debug/xir-verify verify < /tmp/xir-pipeline.json
```

`construct` 分配缺失的 NodeId 并检查，返回 `{ok, module, diagnostics}`；`verify` 检查输入的完整 ID，返回 `{ok, report}`，其中 `ok` 表示 construction 是否通过，完整状态在 `report.status`。请求解析或 construction 错误在 JSON 中返回；命令/进程错误使用非零退出码。Python 对 construction 错误抛出包含原生诊断的 `IRValidationError`。`KernelProgram.from_dict(module)` 会复制并重新验证输入；`to_dict()` 返回副本。

## 已实现的检查与边界

| 项目 | 当前行为 |
|---|---|
| Values / resources | SSA values、storage、buffer、accumulator、event、ticket 使用独立 typed ID；节点 payload 是唯一引用来源 |
| Structured regions | 递归检查定义支配、结果类型、For carried values、If yields、终结节点和 role-local value 隔离 |
| Resources | 检查已知容量、溢出、元素对齐、read-only binding、rank/stride；symbolic extent 保持符号表达式 |
| ConcurrentRoles | 顶层、无条件进入，成员 warps 不重叠，block join；SQMMA 的当前结构子集要求对齐的四个连续 warps |
| Pipeline | 单 producer/consumer，共同引用一个迭代域；每轮 acquire/publish 或 consume/release 恰好闭合一个 ticket；两端均需 drain |
| Async lifetime | AsyncCopy/Sqmma 产生 event；SQMMA 的 accumulator 更新和输入读取保持 pending，await 后才能复用/读取；publish 委托 completion，不等于本地 await |
| Alias / slots | pipeline-owned storage 禁止通过另一个裸 view 绕过 lease；拒绝已知 pending 冲突及可能重叠的 pipeline buffer slot envelopes |
| Effects | 从 operation payload 全量重算 accesses、pending event、producer/observer/publication、role/iteration 和 join 信息；不接受 JSON 伪造 effects |
| Target | `musa:mp31` 只是请求身份，instruction/transfer/representation ID 尚未由 registry 解析；非空字符串不是合法指令证明 |

当前 protocol verifier 采用受限的直线 body：异步 protocol 操作不得嵌入普通 `If`/`For`；每个 role 只执行一个 pipeline，pipeline 不跨 concurrent group 重用。Producer lease 只支持写入、consumer lease 只支持读取；publish 必须用 whole-view copy events 覆盖全部绑定 buffer。Pipeline buffers 的 slot envelopes 必须可证明互不重叠，共用 storage 的交错 slots 暂不接受。每个 accumulator 同时最多一个 pending update，event 不跨 role/iteration 传递。

尚未实现：region-local dynamic views、distributed fragment store/reduce/conversion、完整 alias/线程访问 footprint 和 uniformity 证明、runtime guards、operation registry 与 target representation admission、barrier/fence elaboration、MUSA C emitter、`mtcc` 集成、设备数值与性能验证。不能将当前 report 当作无 race、无 deadlock 或可部署的证明。

Schema v2 是破坏性迁移：不兼容旧 `Assign`/`StageFor`/无类型 `wait`、公共可写 operands/effects、旧 `TargetSpec` capability 名称列表以及旧 Python tree。Serde 拒绝未知字段，schema version 必须为 2。Rust 的 `finish_unchecked()` 和 `deserialize_module()` 明确不提供语义验证；正常构造使用 `finish_checked()`。

## 验证

```sh
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
PYTHONPATH=python .venv/bin/python -m pytest -q python/tests
```

Python 测试 fixture 会构建 native CLI；无需 MUSA SDK/GPU。测试覆盖有效协议、缺失 await/release/drain、错误 publish、别名绕过、role 交叉引用、SSA 合并、资源容量/对齐、JSON 伪造字段以及前端 source mapping。静态 trip-count 测试不替代 slot rollover 的 lowering/设备测试。
