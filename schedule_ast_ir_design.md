# Schedule AST IR 设计草案

## 目标

Schedule AST 是 AI Agent 的主要编辑面。
它描述 GPU kernel 的计算、数据移动、并行角色、同步和资源生命周期，保留会影响性能的硬件决策，
同时隐藏地址编码、barrier phase、线程 ID 展开等机械细节。
最终目标是稳定地生成 MUSA C；底层指令选择、调度和寄存器分配交给 `mtcc`。

```text
Typed Schedule AST
    -> canonicalize / elaborate
    -> schedule verifier
    -> MUSA C emitter
    -> mtcc
```

## Python DSL 前端

Python 只作为 Agent 友好的源码接口，不作为核心 IR。推荐使用装饰器捕获函数源码，再把受限 Python 子集翻译成 Typed Schedule AST：

```text
@kernel function
    -> inspect.getsource()
    -> ast.parse()
    -> restricted-Python validation
    -> Typed Builder
    -> Typed Schedule AST
```

例如：

```python
@kernel
def gemm(A, B, C):
    smem_a = shared((128, 64), dtype=bf16, stages=3)
    pipe = pipeline(stages=3)
    loader = role(warps=[0])
    compute = role(warps=[1])

    with loader:
        for stage in stages(pipe):
            async_copy(A, smem_a[stage], barrier=ready)

    with compute:
        for stage in stages(pipe):
            wait(ready, stage)
            sqmma(acc, smem_a[stage], B)
```

前端不执行函数体，而是由 Python AST translator 调用受限的 Typed Builder：

```text
Python AST translator
    -> builder.begin_role(...)
    -> builder.begin_stage_loop(...)
    -> builder.emit_async_copy(...)
    -> builder.emit_sqmma(...)
    -> builder.end_region(...)
```

这样可以在构造阶段完成类型和引用检查，并允许未来增加 JSON/其他 Agent 协议，而不把核心 compiler 绑定到 Python 语法。第一版支持函数参数、资源声明、`with role`、`for stage in stages`、`if`、DSL intrinsic、常量/shape 表达式和显式 `return`；暂不支持任意 Python 调用、动态容器、异常、generator、runtime side effect 和未受控的对象访问。

装饰器返回 Kernel 对象而不是立即执行 GPU：

```python
program = gemm.build(target="musa:mp31")
program.verify()
source = program.emit_musa_c()
program.compile()
```

## 实施计划

分阶段任务、完成标准和待办记录在 [`plan/`](plan/) 目录中。核心原则是先固定 AST 语义和诊断接口，再逐步增加 MUSA capability 与 lowering；CFG/token SSA 只作为复杂分析的派生表示。

AST 节点都应包含 `source_span`、稳定的 `node_id` 和 `effects`，这样 verifier 可以把诊断定位到 Agent 实际编辑的节点。

值类型不放在公共节点 metadata 中。统一节点结构为：

```text
ScheduleNode
  meta: NodeMeta { node_id, source_span, effects }
  kind: ScheduleNodeKind
  operands: ValueId[]
  result: Option<ValueDef { value_id, ty }>
```

没有结果值的 `Store`、`Wait`、`Fence` 等节点将 `result` 置空；`Load`、`Binary`、`AsyncCopy` 和 SSA-like 的 `Sqmma` 才产生 `ValueDef`。`node_id` 标识语法/调度节点，`value_id` 标识数据流中的值或异步 token。

## 顶层结构

```text
Module
  TargetSpec
  TypeDecls
  KernelDecl*

KernelDecl
  name, params, grid, block, shared_memory_budget
  ResourceDecl*
  RoleDecl*
  PipelineDecl*
  Body: Region
```

`KernelDecl` 的参数分为 pointer/tensor descriptor、scalar、shape、stride 和 compile-time constant。shape、dtype、address space、alignment 和 aliasing 属性必须在类型中表达，不能只存在 Python 注释里。

## 节点分类

### 1. 资源声明

```text
BufferDecl(name, memory_space, dtype, shape, layout_hint,
           alignment, owner, lifetime)
BufferView(base, offset, shape, strides, stage, swizzle)
AccumulatorDecl(name, dtype, shape, fragment_kind, owner)
BarrierDecl(name, scope, producers, consumers, slots, initial_phase)
PipelineDecl(name, stages, carried_resources)
RoleDecl(name, members, lane_mapping, capabilities)
```

memory space 至少包括 `global`、`shared`、`local`、`register`、`accumulator`、`constant`。`fragment_kind` 用于描述 MUSA SQMMA 等专用片段，但不强迫 Agent 手写物理寄存器顺序。

`owner` 可以是 role、block、cluster 或 grid；`lifetime` 可以是 kernel、region、pipeline-stage。资源声明是 verifier 的事实来源，也是 MUSA C emitter 分配变量和属性的依据。

### 2. 并行与作用域

```text
RoleRegion(role, body)
ParallelFor(iv, begin, end, step, mapping, body)
LaneId(scope)
BlockId(axis)
ClusterId(axis)
Elect(condition, body)
Uniform(body)
```

`RoleRegion` 是核心节点，表达 warp/wave specialization 或 producer-consumer 分工。`mapping` 明确循环迭代如何映射到 thread、warp、warpgroup、block 或 cluster。普通的 `if lane_id == 0` 只能表示条件执行，不能隐式声明 collective role。

### 3. 控制流

```text
If(condition, then_region, else_region)
For(iv, begin, end, step, body)
While(condition, body)
StageFor(pipeline, stage_iv, body)
Break / Continue
Return(value?)
```

第一版应限制 `While`、`Break` 和跨 role 的非结构化跳转；结构化控制流更容易验证 barrier 和资源生命周期。`StageFor` 与普通 `For` 分开，便于检查 stage 数、循环携带资源和 barrier phase。

### 4. 内存移动

```text
Load(dst_type, source, indices, cache_policy?, vector_width?)
Store(destination, value, indices, cache_policy?, vector_width?)
AsyncCopy(src, dst_view, bytes, event, predicate?)
AsyncStore(src_view, dst, bytes, event?)
Prefetch(src, descriptor?, stage?)
Commit(event)
Wait(event, stage?, scope?)
Fence(kind, scope)
```

`AsyncCopy` 返回或更新显式 event/token，并记录读写资源。`Wait` 和 `Fence` 是独立节点；不能仅根据源代码出现顺序推断异步可见性。具体生成 `musaMemcpyAsync`、MUSA intrinsic、barrier API 或内联 asm 由 target lowering 决定。

### 5. 计算操作

```text
Elementwise(op, operands, result_type)
Cast(value, dtype, rounding?)
Reduce(op, value, axis, scope, algorithm?)
Mma(op, accumulator, a, b, shape, dtype, saturate?)
Sqmma(accumulator, a, b, tile_shape, operand_layout?, accumulate?)
Atomic(op, address, value, scope)
CallIntrinsic(name, operands, effects, capability)
```

`Mma` 是抽象矩阵计算节点，`Sqmma` 是 MUSA capability-gated 的专用节点。二者都表达逻辑 tile 和数据类型；物理 fragment 排列、寄存器表示和具体 intrinsic 由 lowering 决定。只有确实没有结构化节点时才允许 `CallIntrinsic`，并要求声明 effects 和 target capability。

### 6. 同步与协作

```text
Signal(barrier, stage?, count?)
Wait(event_or_barrier, stage?, predicate?)
Fence(kind = shared|global|proxy|instruction, scope)
BarrierInit(barrier)
BarrierArrive(barrier)
BarrierWait(barrier)
ClusterSync(scope)
```

同步节点必须带 scope，例如 `lane`、`warp`、`warpgroup`、`block`、`cluster`、`grid`。scope 影响 verifier、MUSA intrinsic 选择和必要的参与线程集合。

## 统一的节点属性

每个 effectful 节点具有：

```text
reads:  ResourceSlice*
writes: ResourceSlice*
produces: Event*
consumes: Event*
requires_scope: Scope
capability: TargetCapability?
```

`ResourceSlice` 至少包含 buffer、byte/element range、stage 和 representation。这样可以在 AST 上直接做内存越界、读写冲突、异步依赖和 producer-consumer 表示兼容检查；复杂分支和循环再派生 CFG/token SSA。

## TargetSpec 与能力门控

Target 不应写死为 CUDA 架构名，而应描述 MUSA device/toolchain 的能力：

```text
TargetSpec(
  backend = "musa",
  arch = "mp31",
  warp_size,
  max_threads_per_block,
  memory_spaces,
  async_copy_kinds,
  barrier_scopes,
  mma_kinds,
  supported_dtypes,
  fragment_kinds,
  alignment_rules,
  compiler_features)
```

Ampere 的 `cp.async`、Hopper 的 TMA/WGMMA、Blackwell 的 TMEM/tcgen05 可以作为抽象来源，但不直接成为核心 AST 节点。MUSA 后端只注册实际支持的 capability，例如 SQMMA、MUSA async copy、特定 accumulator fragment 和 barrier 语义；不支持时在构造或验证阶段报出精确 finding。

## Canonical 约束

第一版应坚持以下规则：

- 一种异步传输使用一种 canonical `AsyncCopy` 形式；不要同时支持多个等价 spelling。
- 所有跨 role 的数据交接必须通过显式 event/barrier。
- 不能用普通条件分支伪装 collective scope。
- view 的 offset、shape、stage 和 swizzle 必须是结构化字段。
- 专用 intrinsic 必须声明 effects、scope、operand representation 和 capability。
- 生成 MUSA C 前先完成资源和同步 elaboration；emitter 不负责猜测语义。

## MUSA C lowering 示例

AST：

```text
AsyncCopy(A, smem_a[stage], 8192, ready[stage])
Wait(ready[stage])
Sqmma(acc, smem_a[stage], smem_b[stage], tile=(128,128,64))
```

可能生成：

```cpp
musa_async_copy(smem_a_stage, A_ptr, 8192, &ready, stage);
musa_wait(&ready, stage);
musa_sqmma(acc, smem_a_stage, smem_b_stage);
```

barrier phase、地址计算、线程参与判断、fragment 临时变量和 launch attributes 由 elaboration/emitter 生成；fence 插入、指令选择、指令调度和寄存器分配交给 `mtcc`。

## 实现顺序

1. 先实现 typed nodes、symbol table、resource/effect 属性和 source diagnostics。
2. 加入 `RoleRegion`、`StageFor`、`AsyncCopy`、`Wait`、`Fence`、`Sqmma` 等最小 MUSA schedule vocabulary。
3. 实现 canonicalization 和 AST-level verifier。
4. 对包含循环/分支的 kernel 派生 CFG + token SSA，用于复杂生命周期和依赖分析；简单 kernel 不构造它。
5. 实现 MUSA C emitter 与 capability matrix，再逐步接入 `mtcc` 编译和真实设备反馈。
