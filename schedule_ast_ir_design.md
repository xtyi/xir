# Schedule AST IR 设计

## 1. 目标、范围与已确定的选择

Schedule AST 是 Agent 可编辑的结构化程序表示，描述计算、数据移动、并行角色、同步与资源生命周期。它保留 instruction family、tile、representation、role partition、pipeline depth 等硬件决策；compiler 推导地址编码、barrier allocation/phase 和线程身份判断。

本文固定 XIR 的目标语义与接口边界，不表示当前 Rust/Python 实现已经具备这些能力。文中的类型、节点与示例均为设计规范；具体字段名可以调整，语义不变量必须保留。设计依据是 [CAKE 的核对总结](cake_paper_summary_and_musa_dsl.md)，MUSA 指令约束另以文末固定版本的官方源码核对。

| 设计问题 | 本版决定 |
|---|---|
| Value/resource model | 普通计算值使用 structured SSA；可寻址存储和异步 accumulator 使用 mutable resource semantics |
| Role execution | 使用显式 `ConcurrentRoles`；组内 role 可并发，组尾显式指定 block join |
| Pipeline lifecycle | 区分 iteration、slot、generation；通过 producer/consumer ticket 表达访问与复用权限 |
| Operation effects | 从已注册的 operation contract 推导；effects 是可重建的分析结果，不是 Agent 填写的权威 metadata |
| Node representation | 各 operation 保存强类型字段，通过只读统一接口枚举 operands/results；不重复存储两套引用 |
| Representation | Storage、Buffer/View、Fragment 分层；影响正确性的 mapping/swizzle 是 contract，不是可忽略的 hint |
| Backend responsibility | XIR 决定同步协议并 lowering 到已知语义的 target operations；`mtcc` 完成其实现、最终指令调度和寄存器分配 |

当前先以独立的 CUDA FP32 elementwise 子集跑通 DSL → IR → CUDA → nvcc → GPU 的同步路径，具体支持范围见 [CUDA 实现记录](docs/cuda_elementwise_path.md)。这不改变后续异步协议的设计范围。

第一条完整异步路径面向 exact MP31 target 上的单 CTA GEMM。初始支持一个 producer role、一个 consumer role、固定 slot 数和有界迭代域；允许两者重叠执行。多消费者、嵌套并发组、跨 CTA 协议、非结构化控制流和同一 accumulator 的多条未完成 MMA 链作为后续扩展，不以未定义行为暂时接入。

编译流程为：

```text
Restricted Python / Typed Builder
    -> Typed Schedule AST
    -> canonicalize + resolve target contracts
    -> derive effects + verify schedule
    -> elaborate resources and synchronization
    -> verify target operation contracts
    -> MUSA C emitter
    -> mtcc
    -> external oracle + device measurement
```

CFG、dependency graph 和 token SSA 是从 AST 派生的分析表示。是否构建它们取决于分析需要；源程序的赋值、并发和异步语义不依赖这些表示是否已构建。

## 2. Value/resource model

### 2.1 普通计算值与 structured SSA

`ValueId` 标识一个不可变计算值，其类型可以是 scalar、index、predicate、pointer 或 synchronous fragment。每个 value 只有一个定义；使用必须满足 region dominance 和执行域约束。类型、shape 和 representation 相同不代表 value 可以跨 role 传递。

Python 局部变量可以重绑定，例如 `x = x + 1`。Frontend 将其转换为新的 `ValueId`，而不是在核心 IR 中保留含糊的 `Assign(name, value)`。`For` 的循环状态和 `If` 的合流状态必须显式表达：

```text
For(
    lower, upper, step,
    induction: ValueDef<Index>,
    carried: [CarriedValue { initial, argument: ValueDef, result: ValueDef }],
    body: Region
)
If(condition, results: [ValueDef], then_region, else_region)
Yield(values: [ValueId])
```

`For` 是半开区间，初始版本要求正 step；零次迭代返回 initial values。`Yield` 的数量、类型与 carried/results 一致，每条分支都必须给出合流值。Region arguments 与 operation results 是独立的定义位置，但每个定义只存储一次。GPU kernel 的 `KernelReturn` 无返回值，结果通过输出 buffer 写回；本版只允许在 kernel 最外层末尾返回。

Uniformity 是基于定义、控制流和 participant set 推导的分析属性。Kernel scalar 参数和编译期常量可提供已知 uniform 值；从每线程数据计算的值不能仅因 dtype 相同就当作 uniform。

### 2.2 Storage、Buffer/View 与 Fragment

| 对象 | 表达的内容 | 身份与可变性 |
|---|---|---|
| Storage | Allocation、address space、capacity、alignment | `StorageId`；内容可变，allocation identity 不变 |
| Buffer/View | 对某个 Storage 的逻辑解释：dtype、shape、offset、mapping | `BufferId`；view 不分配存储，不复制内容 |
| Fragment value | 分布在参与线程上的不可变逻辑值，以及 instruction representation | `ValueId`；包含 participant domain 与 representation |
| Accumulator | 具有指令专用表示和异步访问状态的可变计算资源 | `AccumulatorId`；不能作为普通可寻址 pointer 使用 |

初始 address spaces 为 `global`、`shared`、`private`。`register` 和 `accumulator` 不作为与它们等价的可寻址空间；只读性用 access 属性表示，不凭一个 `constant` 标签假定存在特定硬件空间。Private storage 的最终 placement 由 backend 决定。

```text
StorageDecl(
    id: StorageId, address_space, capacity_bytes: IndexExpr,
    alignment_bytes, allocation_scope, lifetime_region,
    access_policy, origin
)
BufferDecl(
    id: BufferId, storage: StorageId, dtype, shape: [IndexExpr],
    offset_bytes: IndexExpr, mapping: MappingSpec, access_mode
)
BufferView(id: BufferId, base: BufferId, slice: SliceSpec)
FragmentType(dtype, logical_shape, participants, representation)
AccumulatorDecl(id: AccumulatorId, fragment_type, access_role, lifetime_region)
```

`MappingSpec` 支持显式 byte strides，以及 compiler 注册的受约束 `LayoutTransform`。Swizzle、operand major mode 和 lane/value mapping 必须能解析到确定的 contract；它们与 cache preference 等性能 hint 分开。View 变换必须组合回底层 Storage，不能只比较两个 BufferId 是否相同来判断 alias。

不同的内部独立 allocation 默认不重叠。两个外部参数对应不同 StorageId 并不自动证明 no-alias；alias 前提必须来自可执行的调用约定、guard 或保守分析。Shared-memory packing/overlay 若改变物理重叠关系，必须重新验证容量与 lifetime。

### 2.3 Accumulator 的状态

本版 `Sqmma` 采用 mutable accumulator semantics，返回 completion event，不返回可立即读取的 fragment：

```text
Uninitialized -- AccumulatorInit --> Ready
Ready -- Sqmma --> Pending(event)
Pending(event) -- AwaitEvent(event) --> Ready
Ready -- AccumulatorRead --> Ready + synchronous Fragment value
```

`Sqmma(acc, a, b)` 对已初始化 accumulator 做 read-modify-write，并在其完成前保留对输入 view 的异步读取。`AccumulatorRead`、重新初始化、释放和其他不兼容访问要求 `Ready`。初始版本同一 accumulator 有 pending MMA 时不得再 issue 第二条 MMA；以后只有在 target 的 ordered accumulation contract 和相应分析齐备时才放宽。

同步 fragment 上的 elementwise 运算仍产生 SSA value。Physical register allocation、具体寄存器编号和 spilling 由 `mtcc` 决定，前端的 fragment/resource 声明不承诺这些结果。

### 2.4 Event、Barrier 与 Ticket 分开建模

| 对象 | 语义 |
|---|---|
| EventRef | 一次异步 operation instance 的完成句柄；关联该操作的 pending accesses |
| BarrierRef | Lowering 使用的可重复利用的同步资源；不是某次完成事件的身份 |
| ProducerTicket / ConsumerTicket | 特定 pipeline、iteration/generation 的访问权限；必须按协议结束，不能复制后重复使用 |

Event 是不可变的编译器句柄，不是跨线程自动传输的运行时 SSA payload。`AwaitEvent` 建立当前 observer domain 对相应 completion 的已知关系，不销毁 event；同一合法作用域内可重复等待。循环中的同一个静态 event 定义在不同迭代中对应不同 operation instance，不能只用 NodeId 或 slot 编号识别。

初始版本 event 不任意跨 role 或循环迭代逃逸。Producer 通过 `PipelinePublish` 将 transfer completion 关联到某个 generation，consumer 通过 `PipelineConsume` 观察该 generation；这不依赖把 producer 的寄存器值传给 consumer。Ticket 不可存入普通 buffer、参与算术或通过任意 tuple/region result 逃逸，必须在所属 iteration 内恰好 publish/release 一次。

### 2.5 动态参数与数值前提

Shape、stride、buffer extent、alignment 和 alias 前提进入 `KernelContract`。`IndexExpr` 使用有类型的常量、符号及受限运算，注明 byte/element 单位；不以任意字符串或 Python 表达式充当核心索引模型。第一版支持可判定的表达式子集，未知范围和可能溢出显式报告，不能按无限精度证明后直接生成会溢出的机器整数。

Load/Store 的 mask 只约束实际访问；masked-off load 必须给定结果值，且不能依赖先形成一个非法 pointer 再屏蔽访问的 lowering。动态前提必须由已验证的调用路径或 launch guard 保证，无法保证时返回 `unknown` 或拒绝该路径。

`NumericalContract` 单独记录 input/output dtype、accumulation precision、允许的近似/重排与 oracle 判据。Memory-safe、instruction-legal 与数值正确是不同性质。

## 3. Role 执行语义

### 3.1 显式并发组与 join

`RoleDecl` 声明 CTA 内的参与线程集合；`RoleRegion` 只允许作为 `ConcurrentRoles` 的直接子区域。本版同一并发组内各 role 的成员集合必须静态可确定且两两不重叠，不支持嵌套并发组。

```text
ConcurrentRoles(
    roles: [RoleRegion(role, body)],
    join_scope = block
)
```

整个 CTA 必须 uniform 地进入并退出该结构。不同 role 的 body 可以并发执行；它们在 AST 中的前后顺序不建立跨 role happens-before。每个线程在自己的 role 内遵循 program order，但异步 operation 的完成不由 issue 顺序自动保证。

`join_scope=block` 是显式的性能与同步决策。未分配给任何 role 的线程跳过 role body，但仍参加组尾 join。Join 对应的 target barrier 由 XIR lowering，不能默认由 `mtcc` 推导。Join 不替代 async completion：各 role 离开前必须结束自己的 ticket、drain 所属异步职责，之后才能在统一 join 后回收资源。

不同并发组按外层 program order 排列，前一组的显式 join 完成后才进入后一组。Role-local value 不作为并发组结果导出；需要传递的数据通过已声明的共享资源和同步协议交接。

Role 可以引用外层 dominating value，但该定义必须在每个实际使用它的线程上均可用；这是各线程使用自身已有的值，不是隐式 broadcast。Kernel uniform 参数可被各 role 使用，sibling role 内新产生的 value 则不能凭符号可见性直接读取。

### 3.2 Issuer、participant 与 memory scope

每个 collective/async operation 的 contract 分别记录：

- `execution_domain`：当前 active threads 及其控制流条件。
- `participants`：必须共同参与该操作的线程集合及 operand 一致性要求。
- `issuers`：实际发出硬件请求的线程集合或已注册的 election policy。
- `memory_scope`：建立 ordering/visibility 的范围。

这些信息不能压缩成一个 `requires_scope`。例如 TME transfer 可以由 role 内选定的 issuer 发起，而 shared data 被另一个 role 消费；issuer election 本身不证明所有参与者都完成了协议。

`If` 的条件必须满足所包含 collective 的 uniformity 要求。`Uniform(body)` 不保留为未经证明的声明；`Elect(condition, body)` 暂不作为通用扩展口，所需 election 放在经过检查的 operation contract 中。Block barrier 不能出现在只有部分 CTA 线程会执行的 role body 内。

初始 target admission 只允许有依据的线程/warp/block 范围。Warp squad 等 participant group 由具体 instruction contract 描述，不因名字类似就等同于 NVIDIA warpgroup；cluster/grid 协作未注册时明确拒绝。

MP31 SQMMA 的所查 traits 使用 128-thread mapping。因此示例采用 consumer warps `[4, 5, 6, 7]`，loader warp `[0]`，CTA 为 256 threads；warps `[1, 2, 3]` 空闲但参加 join。具体连续性、对齐和调用约束仍由 target contract 验证，不能只检查线程总数。[MP31 traits][mp31-traits]

## 4. Pipeline 生命周期

### 4.1 Iteration domain 与 slot allocation

用 `PipelineFor` 替代含糊的 `StageFor`。完整迭代域只声明一次，由两个 role 共同引用：

```text
PipelineDecl(
    id: PipelineId,
    iteration_count: IndexExpr,
    slots: PositiveConstInt,
    producer: RoleId,
    consumer: RoleId,
    buffers: [PipelineBufferBinding]
)
PipelineFor(pipeline: PipelineId, iteration: ValueDef<Index>, body: Region)
```

`iteration_count=N` 表示半开区间 `[0, N)`，step 固定为 1；N 必须非负且在 CTA 内 uniform。两个 role 各有一个对应的 PipelineFor，按相同逻辑序列处理每个 iteration。第一版不允许跳过 iteration、early return 或改变该循环步长。

```text
iteration = i
slot = i % slots
generation = i // slots
```

`K=1024, BK=64, slots=3` 对应 16 次迭代，而非三次。Slot 只是物理复用位置，generation 区分该位置上的不同使用周期。Barrier phase 的实际编码、初始值与 rollover 由 target protocol 生成，不作为 Agent 手填的逻辑 generation。

`PipelineBufferBinding` 关联逻辑 buffer template、底层 allocation 和 `slot_stride_bytes`。所有参与该 pipeline 的 buffers 对同一 iteration 使用同一个逻辑 slot。所需容量由 mapping footprint、alignment、slot stride 和 slot 数共同决定；不能对 swizzled/padded buffer 直接套用 `product(shape) * sizeof(dtype)`。

基础 `BufferRef` 不带 pipeline stage 字段。`PipelineView(ticket, buffer)` 派生普通 view 地址，并返回带 lease 的 `LeasedBufferRef`；lease 关联 ticket/generation。任意重新构造 alias view 都不能绕过底层 pipeline allocation 的访问权限检查。

### 4.2 Canonical handoff protocol

| 操作 | 前提 | 结果与职责 |
|---|---|---|
| `PipelineAcquire(pipe, i)` | Slot 尚未使用，或上一 generation 已由 consumer release | 返回 producer ticket，允许对本 generation 的绑定 buffers 写入 |
| `AsyncCopy(src, dst, transfer_spec)` | Source/destination 合法，destination 有有效写权限 | 返回 event；源读取和目标写入在完成前保持 pending |
| `PipelinePublish(ticket, events)` | Events 完整覆盖该 iteration 承诺交付的数据 | 结束 producer ticket，把这些 completion 关联到 ready 通知；不要求 producer 在此同步等待 |
| `PipelineConsume(pipe, i)` | 等待该 generation 已 publish 且关联 transfer 已完成 | 返回 consumer ticket，按 target contract 建立输入数据可见性 |
| `Sqmma(acc, a, b, instruction_spec)` | Accumulator ready，输入 view lease 有效，representation 与参与者合法 | 返回 event；保持对输入的 pending reads 与 accumulator 的 pending update |
| `AwaitEvent(event)` | 当前 observer domain 有权观察该 event | 建立 completion 事实和该 contract 所定义的 ordering/visibility |
| `PipelineRelease(ticket)` | Consumer 对绑定 slot 的所有访问已完成 | 结束 consumer ticket，将 slot 交还给下一 generation |
| `PipelineDrain(pipe)` | 当前 role 已完成全部 iteration | Producer 等待其发布的 slot 均完成 release；consumer 确认所有 ticket 和相关访问已结束 |

Publish 可以早于 transfer 的物理完成。Lowering 必须在 transfer 发出前按协议完成 expected transaction 等设置，并把最终 completion 接到对应 ready 通知；不能把“publish 被执行”编码成“数据已经完成”。多次 copy 的覆盖与 transaction accounting 必须检查，不能只给 events 求一个未经证明的 byte 总和。

逻辑状态为：

```text
Free(slot, generation)
    -> ProducerOwned
    -> PublishedInFlight
    -> Ready
    -> ConsumerOwned
    -> ConsumerAccessesComplete
    -> Free(slot, generation + 1)
```

Producer 的写入完成后 consumer 才能读；consumer 的最后一次读取完成后 producer 才能覆盖。SQMMA issue、transfer completion、MMA completion、consumer release 和 block join 是不同事件，不能相互替代。

### 4.3 Prologue、tail 与退出

初始所有 slot 为 Free，尚无 ready generation。Producer 可以在 consumer 消费之前填充至多 slots 个 iteration；占满后通过 acquire 等待释放。初始版本在 consumer 每次 SQMMA 后等待其完成再 release，仍允许 producer 的后续 transfer 与当前计算重叠。

N 为零时双方不进入循环，drain 不等待不存在的 generation；N 小于 slots 时只处理实际使用的 slot。尾部 drain 按实际迭代数处理，不能无条件等待每个物理 slot 都发生过一次通知。Producer/consumer 的计数不一致、publish/release 缺失和不满足参与者条件的等待均属于 verifier 错误。

基础 elementwise 支持 masked access；首个 TME/SQMMA kernel 可用明确的整除/extent guards 限制 shape。后续增加矩阵 tail 时，必须注册 zero-fill/padding、合法 transfer shape 和 transaction accounting，双方仍要执行匹配的 iteration 协议。不能简单用 predicate 跳过 copy，却继续等待原先的 bytes 或 arrivals。

本版在有限迭代域、已注册的 device progress 前提和该单 producer/consumer 协议内检查匹配与无环依赖，不声称证明任意 GPU 程序都不会 deadlock。

## 5. Operation effect contract

### 5.1 语义的唯一来源

每个 operation kind 对应 compiler 注册的 `OperationContract`：

```text
OperationContract
    schema_version
    operand_and_result_schema
    infer_types(op, context)
    validate_preconditions(op, context)
    resolve_target_contract(op, target)
    infer_effects(op, context, target)
    lower(op, target)
```

Agent 只能选择已注册的 operation、参数和目标允许的策略。Contract 定义类型、参与者、representation、同步以及 completion 规则；添加新 operation 必须同时添加这些规则与正反例。`RegisteredIntrinsic(intrinsic_id, ...)` 只能引用具有同等 contract 的注册项，不接受任意名字加用户填写 effects 的旁路。

Compiler evolution 可以修改 registry，但 registry revision、IR schema revision 和 target/toolchain profile 必须进入验证与编译记录。某项行为未建模时返回 `unknown`，不能用空 effects 代表“没有副作用”。

### 5.2 Derived effects 与 resource footprint

```text
EffectSummary
    immediate_accesses: [Access]
    pending_accesses: [AsyncAccess { event, access }]
    event_relations: [CompletionOrObservation]
    protocol_transitions: [Acquire | Publish | Consume | Release | Drain]
    execution_requirements: ExecutionRequirements

Access
    storage_or_accumulator
    footprint: SymbolicFootprint
    mode: Read | Write | ReadWrite
    execution_domain
    predicate
    representation
```

`SymbolicFootprint` 将 view 映射回 storage origin，包含索引域、byte 单位、必要的 mapping 和 lease/generation 关系。存在 alias、非仿射映射或动态范围时可以保守近似；若近似无法证明所需性质，报告 limitation，不假定两个访问不重叠。Accumulator 用独立 resource footprint 表示，不伪造 byte pointer。

`AwaitEvent` 改变已知 completion 状态；`Fence` 改变指定访问之间的 ordering/visibility。Fence 不凭空完成一个 async operation，Await 也不保证其 contract 之外的所有访存可见。

Effects、uniformity、alias 和 readiness 作为 side tables 或缓存保存。缓存至少以 module revision、operation revision、registry revision、target profile 及依赖分析版本标识；初始实现可全量重算。修改 operand、region、mapping 或 protocol 后必须使相关结果失效，不能信任 JSON 中携带的旧 effects。

### 5.3 Verification stages

1. Construction checks：强类型引用、定义唯一性、dominance、region result/argument、symbol/index 表达式合法性。
2. Schedule verification：role 执行域、ticket 状态、iteration 匹配、资源 alias/容量/生命周期、pending access 冲突与数值模式前提。
3. Target verification：instruction/transfer contract、participant/issuer、descriptor representation、barrier 资源、实际插入的 waits/fences 和 launch 约束。
4. External validation：工具链编译、生成代码检查、external oracle 与 device measurement。

诊断包含 rule code、primary/related NodeId、source span、resource、target、状态以及可解释的依据。`pass`、`fail`、`unknown` 分开；默认 checked compilation 在所需性质为 fail/unknown 时不进入正常运行路径。未校准的性能估计可以 unavailable，不构成程序非法。

上述检查只在明确建模的范围内成立，不能取代外部正确性验证，也不能仅凭 AST 宣称最终 binary 没有 backend 错误。

## 6. AST 数据结构与节点列表

### 6.1 Strongly typed payload 是唯一存储

```text
ScheduleNode
    meta: NodeMeta { node_id, node_revision, source_span, origin_ids }
    kind: ScheduleNodeKind

LoadOp
    source: MemoryRef
    indices: [ValueId]
    mask: ValueId
    masked_value: ValueId
    result: ValueDef

SqmmaOp
    accumulator: AccumulatorId
    a: MemoryRef
    b: MemoryRef
    instruction: InstructionSpecId
    completion: EventDef
```

Operation-specific payload 唯一保存 operands 和 definitions。通过只读 `operands()`、`definitions()`、`regions()` 接口为 traversal、serialization、verification 提供统一视图，`ScheduleNode` 不另存可写的 `operands/result/effects` 副本。`ValueDef`、`EventDef`、ticket definition 等按各自类别携带 typed ID 和类型；统一接口不把它们都当作可算术操作的 ValueId。

`MemoryRef` 可以是普通 `BufferRef` 或受 ticket 约束的 `LeasedBufferRef`。动态 view 是 region 内的结构化定义，其 offset 可以引用当前 induction value；定义和 lease 不能越过允许的作用域逃逸。资源声明使用 typed ID，name 只用于显示。

NodeId 在同一 module 的已有节点改写中保留；复制节点产生新 ID，并用 `origin_ids` 保留映射。Source frontend 重建模块时允许重新分配 ID，不承诺插入一行后自动恢复旧身份。确定性序列号不等于跨 revision 稳定身份，验证与诊断记录必须带 module revision。

### 6.2 Module 与 kernel contract

```text
Module
    module_id, module_revision
    schema_version
    registry_revision
    target_request: TargetRequest
    kernels: [KernelDecl]

KernelDecl
    id, name
    parameters: [TypedParameter]
    constexpr_parameters
    contract: KernelContract
    launch: LaunchSpec
    resources: [ResourceDecl]
    roles: [RoleDecl]
    pipelines: [PipelineDecl]
    body: Region

KernelContract
    parameter_extents_and_strides
    alignment_and_alias_preconditions
    shape_domain_and_guards
    numerical_contract

LaunchSpec
    grid, block, shared_memory_budget
```

参数的类型采用结构化 `TypeRef`；参数身份、runtime shape/stride 表达式与已验证前提分开保存，不把所有运行时事实硬塞进静态 dtype。Allocation、role、pipeline 集中声明一次，body 通过 typed ID 引用；依赖 induction value 的 BufferView 可在对应 region 内定义。`grid/block` 必须通过 exact target 的资源和范围检查。

### 6.3 Canonical Schedule operations

以下签名省略 NodeMeta 与部分明确的类型标注；每个输入和输出只在专有 payload 中存储一次。

| 分类 | 节点及关键字段 | 核心规则 |
|---|---|---|
| Declarations | `StorageDecl`、`BufferDecl`、`BufferView`、`AccumulatorDecl` | Allocation 与 view 分开；representation 必须解析 |
| Execution declarations | `RoleDecl(id, members)`、`PipelineDecl(...)` | Member set 与迭代域可检查，capability 不由 role 自行宣称 |
| Pure values | `Constant(value, result)`、`Binary(op, lhs, rhs, result)`、`Cast(input, mode, result)` | SSA；保留整数与浮点语义 |
| Indexing | `ThreadId(axis, result)`、`BlockId(axis, result)`、`LaneId(group, result)` | 明确结果的执行域和 uniformity；group 必须有已定义成员 |
| Memory | `Load(source, indices, mask, masked_value, result)`、`Store(destination, indices, value, mask)` | Predicate、extent、mapping 和 lease 均受检查；collective fragment store 采用匹配 representation |
| Structured control | `For(...)`、`If(...)`、`Yield(values)`、`KernelReturn()` | 显式 carried/results；本版无 role 内提前退出 |
| Concurrency | `ConcurrentRoles(role_regions, join_scope)`、`RoleRegion(role, body)` | RoleRegion 只能出现在并发组中，join 显式可见 |
| Pipeline control | `PipelineFor(pipe, iteration, body)` | 按 PipelineDecl 的完整迭代域执行，非物理 slot 遍历 |
| Producer protocol | `PipelineAcquire(pipe, i, ticket_def)`、`PipelinePublish(ticket, events)` | 每次 acquire 恰好 publish 一次，发布内容与 transfer 匹配 |
| Consumer protocol | `PipelineConsume(pipe, i, ticket_def)`、`PipelineRelease(ticket)` | 消费正确 generation；release 前无 pending slot access |
| Pipeline views/exit | `PipelineView(ticket, buffer, leased_view_def)`、`PipelineDrain(pipe)` | View 访问权限依赖 ticket；drain 只处理实际使用的 iterations |
| Async transfer | `AsyncCopy(source, destination, transfer_spec, completion)` | Explicit typed transfer contract；不使用无类型的 event-or-barrier 参数 |
| Accumulator | `AccumulatorInit(acc, value)`、`AccumulatorRead(acc, result)` | 初始化、ready/pending 与访问 role 有明确规则 |
| Async matrix | `Sqmma(acc, a, b, instruction_spec, completion)` | Mutable resource update，返回 EventDef；input pending reads 一并建模 |
| Completion/ordering | `AwaitEvent(event)`、`Fence(ordering_spec)` | 完成观察和 memory ordering 分开；scope 由 spec 明确 |
| Registered extension | `RegisteredIntrinsic(intrinsic_id, typed_arguments, definitions)` | 仅调用已注册且分析覆盖齐备的 contract |

`OrderingSpec` 至少记录作用于哪些访问类别、producer/consumer scope 或 domain，以及要求的 ordering/visibility；不把 `shared/global/proxy/instruction` 这种混合分类直接作为全部 fence 语义。

Generic `Mma` 暂不与 `Sqmma` 并列提供模糊的自动选择路径。Frontend 将来可以提供选型 sugar，但必须在 verification 前解析到确定的 InstructionSpec。`Reduce`、`Atomic`、`AsyncStore`、`Prefetch`、persistent/cluster operations 按 corpus 需求加入，分别补齐数值、参与者、memory order、completion 和 target 规则。

### 6.4 Schedule operations 与 target operations 的边界

逻辑 schedule 中不同时提供 `Signal`、`BarrierArrive`、两种 `Wait` 等重复入口。Canonicalization 只归一等价拼写，不把不同 scope、不同 completion 语义或有不同性能含义的 schedule 合并。

物理 barrier 的 allocate/init/arrive/wait、expected transaction 设置、phase progression、descriptor encoding 和 target fence 属于 elaboration 的输出。这些 target operations 可打印、可验证，并映射回源 NodeId；Agent 的默认编辑面仍是上述逻辑 schedule。

新增低层操作若需要成为可编辑原语，应作为显式语言扩展注册其 contract；不能通过修改生成代码后继续沿用旧 IR 的验证结果。

## 7. GEMM 协议示例

下面是 **Schedule IR pseudocode**，用于完整展示 A/B 两组 buffer、三个 slots 的 pipeline 角色与状态流转，不是当前 Python API 或可直接编译的 MUSA kernel。

示例 contract：单 CTA 处理 `C[128,128] = A[128,1024] @ B[1024,128]`；A/B 为连续 BF16，accumulator 与 C 为 FP32，tile 为 `(128,128,64)`，16 个 iterations，3 个 slots。参数 extent、alignment、输出与输入不重叠，以及表示映射必须在 launch 前检查。所查官方 MP31 源码包含对应 BF16 SQMMA form，但本示例未做设备验证。[SQMMA primitives][mp31-sqmma]

`spec` 是通过 target registry 验证的编译期配置，显式列出 `instruction`、`a_transfer/b_transfer`、`a_mapping/b_mapping/acc_mapping`、每个 slot 的 stride/alignment。它们不是 emitter 隐式选择的默认布局，也不是未经注册的字符串。示例只省略这些硬件编码细节。

```text
loader = RoleDecl(members = warps[0])
compute = RoleDecl(members = warps[4, 5, 6, 7])
launch = LaunchSpec(grid = (1, 1, 1), block = (256, 1, 1))

shared_a = StorageDecl(shared, 3 * spec.a_slot_stride, spec.a_alignment,
                       allocation_scope = block, access_roles = {loader, compute})
shared_b = StorageDecl(shared, 3 * spec.b_slot_stride, spec.b_alignment,
                       allocation_scope = block, access_roles = {loader, compute})
a_tile = BufferDecl(shared_a, bf16, (128, 64), offset_bytes = 0, mapping = spec.a_mapping)
b_tile = BufferDecl(shared_b, bf16, (64, 128), offset_bytes = 0, mapping = spec.b_mapping)
acc = AccumulatorDecl(f32, (128, 128), compute, spec.acc_mapping)
pipe = PipelineDecl(iteration_count = 16, slots = 3,
                    producer = loader, consumer = compute,
                    buffers = [(a_tile, spec.a_slot_stride),
                               (b_tile, spec.b_slot_stride)])

ConcurrentRoles(join_scope = block):
    RoleRegion(loader):
        PipelineFor(pipe, i):
            p = PipelineAcquire(pipe, i)
            a_dst = PipelineView(p, a_tile)
            b_dst = PipelineView(p, b_tile)
            a_src = BufferView(A, slice = [0:128, 64*i:64*(i+1)])
            b_src = BufferView(B, slice = [64*i:64*(i+1), 0:128])
            ea = AsyncCopy(a_src, a_dst, spec.a_transfer)
            eb = AsyncCopy(b_src, b_dst, spec.b_transfer)
            PipelinePublish(p, [ea, eb])
        PipelineDrain(pipe)

    RoleRegion(compute):
        AccumulatorInit(acc, zero)
        PipelineFor(pipe, i):
            c = PipelineConsume(pipe, i)
            a_input = PipelineView(c, a_tile)
            b_input = PipelineView(c, b_tile)
            done = Sqmma(acc, a_input, b_input, spec.instruction)
            AwaitEvent(done)
            PipelineRelease(c)
        PipelineDrain(pipe)
        output = AccumulatorRead(acc)
        Store(C, full_tile_indices, output, mask = true)

KernelReturn()
```

`zero`、`true` 和 `full_tile_indices` 是带类型的常量/索引简写；resource lifetime 限于本 kernel，accumulator 只允许 compute role 访问。Tile store 使用与 `output` fragment 相匹配的线程到元素映射，不能把一个分布式 fragment 当作每线程完整矩阵重复写入。

这个例子同时保留数据 ready 和 slot reusable 两条依赖。Producer 没有每次等待 copy，consumer 显式等待 MMA 后释放输入。尾部 pipeline drain 与 block join 都可在最终 lowering 中检查，不能只看生成代码是否包含某个 `wait` 名称。

## 8. Python DSL 前端

Python 作为源码接口，核心语义存在 Rust Typed Schedule AST。保留装饰器捕获源码的方式，同时提供显式 source text 入口，支持 Agent 生成源码及 `inspect.getsource()` 不可用的环境：

```text
@kernel function / kernel_from_source(text, filename, signature)
    -> source capture
    -> ast.parse
    -> restricted-Python validation
    -> symbol/type resolution + structured SSA construction
    -> Typed Builder
    -> Typed Schedule AST
```

Frontend 不执行 kernel 函数体，不通过任意 `eval` 推断表达式或 effects。DSL 名称从固定 registry 解析；closure/global 常量只能通过明确的 constexpr bindings 导入。参数类型由受限注解语法或显式 signature 提供；使用装饰器时应避免 Python 先执行任意注解表达式，例如采用 postponed annotations。

局部变量重绑定、普通 `if/for` 和资源别名可作为语法 sugar，但必须在构造 Typed AST 时生成明确的 SSA definitions、region results 和 resource references。`for i in pipeline_iterations(pipe)` 对应完整 PipelineFor，不再使用容易误解成物理 slots 的 `stages(pipe)`。

第一版只接入已注册的 canonical operations；不支持任意 Python 调用、异常、generator、动态容器和未受控的对象访问。Constexpr 分支在构造时求值，runtime 分支保留 If；不能把 runtime shape 当成 Python 常量执行。

下面仅表示期望的 host API 生命周期，未声明这些方法已经实现：

```python
program = gemm.build(target="musa:mp31", constexpr={"BK": 64})
report = program.verify()
source = program.emit_musa_c()
artifact = program.compile()
```

`emit_musa_c()` 与 `compile()` 必须检查当前 revision 的 verification 状态，不能依赖用户曾经手动调用过 `verify()`。构建对象本身不启动 GPU；实际调用入口还须检查 runtime guards、device 与 artifact target 是否匹配。

## 9. Target resolution 与 MUSA lowering

### 9.1 Target identity 与 capability 分开

```text
TargetRequest(backend, arch)
ResolvedTarget(
    identity,
    device_properties_or_offline_profile,
    toolchain_identity,
    capability_registry_revision,
    instruction_contracts,
    transfer_contracts,
    synchronization_contracts,
    resource_limits,
    analysis_coverage,
    performance_calibration
)
```

Agent 请求 target，compiler 从已注册 profile 和可用环境解析。导入的 JSON 即使声称支持某 capability，也不能替代 registry/toolchain 检查。离线 profile 可用于源码生成；没有实际工具链或设备时分别标明未编译、未执行。

支持 `bf16`、支持 `sqmma` 不等于支持它们与任意 tile、mapping、scope 的组合。Admission 查询必须针对完整 operation contract。Exact architecture/toolchain 不匹配时报告 unsupported，不静默切换另一种指令语义。

Launch limits、shared-memory/barrier 使用量、descriptor 与 fragment 约束纳入 target checks。Performance calibration 独立于 instruction admission；没有模型时不捏造 latency/occupancy 结论。

### 9.2 XIR 与 mtcc 的职责

| 工作 | 责任归属 |
|---|---|
| Instruction family、transfer mechanism、角色与 pipeline 结构 | Agent 显式选择，XIR 解析并检查 |
| Operand mapping、descriptor representation、numerical mode | XIR target contract；语义相关选择保留在 schedule/spec 中 |
| Barrier 生命周期、transaction accounting、phase、wait/fence 协议 | XIR elaboration，随后验证 target operations |
| Host descriptor setup、launch attributes、guard 检查 | XIR host-side lowering，与 device program 明确分开 |
| Device intrinsic 的最终实现与 instruction scheduling/register allocation | `mtcc` 在已发出的合法 target 语义下完成 |
| 数值正确性和性能结论 | External oracle、目标设备与测量记录 |

Kernel 内的 AsyncCopy lowering 到目标支持的 device transfer primitives，例如已验证的 TME tile load contract；不能用 host/stream 层的 `musaMemcpyAsync` 代替。Host runtime 拷贝属于独立的 host execution 层。[MUSA Runtime API][musa-runtime]

Elaboration 可以在 operation contract 指定的位置实现必须的 fence，包括通过具有明确语义的 SDK wrapper 实现；不得默认 `mtcc` 会替一个缺少同步语义的程序推断跨 role 或跨引擎协议。插入的同步必须保留 source mapping 和可检查的依据。

Target verification 还要检查 adapter 保留了逻辑 schedule 的 completion、happens-before 和 lease 约束：例如 ready 通知不能早于相应 transfer completion，empty 通知不能早于最后一次输入读取完成。仅检查每条生成指令各自合法，不足以验证整个 lowering。每个 adapter 都需要覆盖这些约束的正反例、生成代码检查和设备回归。

TME descriptor setup/transfer、SQMMA descriptor/fragment、completion 和 pipeline barrier 分别参照官方 [TME primitives][mp31-tme]、[SQMMA descriptor][mp31-desc]、[SQMMA primitives][mp31-sqmma] 与 [MP31 pipeline][mp31-pipeline]。源码中的可用实现是建模依据，不自动构成对所有 MUSA target 的保证。

## 10. 实现顺序与迁移要求

沿用 [`plan/`](plan/) 的阶段划分。Schema v2 迁移已同步 P0–P5 的任务描述；实际支持范围见 [项目 README](README.md)，不能仅凭节点名称认定完整 target contract 已实现。

| 顺序 | 要完成的工作 | 对应阶段 |
|---|---|---|
| 1 | 固定本版四项 contract 的数据类型、作用域与正反例；将 ValueId/StorageId/BufferId 等接入强类型节点 | P2 |
| 2 | 实现 For/If/Yield 与普通 load/store 的构造检查；以 elementwise 跑通类型、索引、mask 和 host guard | P1/P2/P6 的基础闭环 |
| 3 | 完成 operation registry、effects 推导与缓存失效；解析 mapping 和 exact target contracts | P3/P4 |
| 4 | 实现 ConcurrentRoles、pipeline tickets、accumulator readiness 与本例的有界协议验证 | P2/P4 |
| 5 | 将逻辑协议 elaborate 成可验证的 MP31 resource/transfer/synchronization operations | P3/P5 |
| 6 | 生成 MUSA C，接入 `mtcc`、oracle 与设备 benchmark；再按第二个 kernel family 扩展语言 | P5/P6/P7 |

Schema v2 已落地的迁移与后续边界：

- `ast.rs`：已移除公共可写 operands/result 副本；operation 字段使用 typed ID，Event/Ticket/Accumulator 使用独立引用；已用 PipelineFor 替换 StageFor。
- `types.rs`：已保留 Storage/Buffer 分层，并用结构化 IndexExpr/MappingSpec 替换字符串索引；fragment 带 participants，verifier 跟踪 accumulator/event/ticket 状态。动态 view、target mapping admission 和 slot/generation elaboration 尚未实现。
- `NodeMeta` 与分析：effects 已移至 derived analysis，保留 NodeId/source mapping/revision。初始实现每次全量重算，无旧 effects 缓存可复用。
- `target.rs`：保留 TargetRequest 请求身份，已移除旧能力名称列表。CUDA SM120 有固定 elementwise admission；MUSA resolved profile 与通用 operation registry 待做。静态 report 与外部编译/设备验证分开，construction/target Pass 不自动产生总体 verified 状态。
- Python frontend 与 builder：已通过 JSON CLI 统一源码解析、JSON import 和 native builder 的验证入口。普通 If/For、资源声明和显式并发 pipeline 有可运行的 construction 示例；缺少 native executable 时明确失败。

已新增纯 `LaunchIndex` 节点（ThreadIdxX/BlockIdxX/BlockDimX/GridDimX），结果为 Index，无 memory effects；CUDA adapter 根据 builtin lowering 到 launch coordinates。CUDA 的 byte extent/device/grid/block 前提通过生成 wrapper 检查，raw kernel 手动调用仍须满足这些前提。

当前异步 verifier 只支持 README 所述的直线 pipeline 子集；复杂分析需要时再派生 CFG/token SSA。MUSA 编译、数值正确性和性能只有真实执行后才能标记完成，静态 protocol 示例不替代这些验证。

## 11. Contract 验收样例

以下是完整设计的验收要求。Rust/Python 回归已覆盖其中的结构化 SSA、role 隔离、ticket/pending lifetime、publish coverage、静态资源和 effects 重算等子集；target mapping、真实 slot rollover、lowering 与设备数值行为仍未验证。测试入口与边界见 [README](README.md)。

| 场景 | 期望结果 |
|---|---|
| For 零次迭代、携带一个 scalar | 返回 initial value |
| If 某分支缺少结果或结果类型不匹配 | Construction failure，定位对应 yield/branch |
| 在另一个 role 中直接引用 role-local value | 拒绝隐式跨 role value transfer |
| 同组 role 成员重叠、group 非 uniform 进入 | 拒绝该并发结构 |
| 单 warp 作为所选 128-thread SQMMA 的 participant set | Target contract failure |
| Copy 已 issue 但还未建立 completion，consumer 直接读 | Pending-access/hand-off failure |
| SQMMA 后未等待即释放 slot 或读取 accumulator | Completion/lifetime failure |
| Producer publish event 来自其他 iteration，或少发布 B tile | Generation/data-coverage failure |
| 同一个 ticket publish/release 两次，或借由 alias view 越权访问 | Ticket/access-policy failure |
| `N = 0, 1, S-1, S, S+1, 2*S+1`，其中 `S >= 2` | Iteration coverage、slot rollover 和 drain 均匹配；不等待未使用的 generation |
| Producer/consumer 不同 trip count，或 predicate 跳过必须的交接 | Protocol verification failure |
| TME 写入 mapping 与 SQMMA descriptor 消费 mapping 不一致 | Representation failure，即使 shape/dtype 相同 |
| 两个外部 pointer 可能 alias，却要求无冲突并行访问 | 没有足够前提时为 unknown；有强制 guard/合法证明后才接受 |
| 修改 copy destination 后重跑 verifier | 使用重新推导的 footprint/effects，不读取旧缓存 |
| 未注册 intrinsic、JSON 伪造 capability 或缺少 analysis coverage | 拒绝或 unknown，不能作为 verified program 接受 |
| Target 编译通过，但 oracle mismatch | Runtime numerical validation failure，不以静态通过覆盖该结果 |

## 参考与证据范围

- [CAKE v1](https://arxiv.org/html/2608.12629v1)：提供 compiler/agent co-design、显式 schedule 和分析接口的方向；本文的 SSA/resource 分工、ConcurrentRoles、tickets 与节点签名是 XIR 的设计选择。
- [MLIR SCF](https://mlir.llvm.org/docs/Dialects/SCFDialect/#scffor-scfforop)：structured region 中 loop-carried values 与 yield 的参考；XIR 不因此依赖 MLIR。
- [Moore Threads MUTLASS](https://github.com/MooreThreads/mutlass/tree/78b349514831ed54a797b717f2fc0534646acff2)：MP31 事实固定于 commit `78b349514831ed54a797b717f2fc0534646acff2`。源码检查不等于本设计已通过 MUSA 编译或设备测试。

[mp31-traits]: https://github.com/MooreThreads/mutlass/blob/78b349514831ed54a797b717f2fc0534646acff2/include/mute/atom/mma_traits_mp31_sqmma.hpp
[mp31-sqmma]: https://github.com/MooreThreads/mutlass/blob/78b349514831ed54a797b717f2fc0534646acff2/include/mute/arch/mma_mp31_sqmma.hpp
[mp31-pipeline]: https://github.com/MooreThreads/mutlass/blob/78b349514831ed54a797b717f2fc0534646acff2/include/mutlass/pipeline/mp31_pipeline.hpp
[mp31-tme]: https://github.com/MooreThreads/mutlass/blob/78b349514831ed54a797b717f2fc0534646acff2/include/mute/arch/copy_mp31_tme.hpp
[mp31-desc]: https://github.com/MooreThreads/mutlass/blob/78b349514831ed54a797b717f2fc0534646acff2/include/mute/arch/mma_mp31_desc.hpp
[musa-runtime]: https://docs.mthreads.com/en/musa-sdk/musa-sdk-doc-online/history_version/v5.1.0/libraries/core_api/runtime_api_reference/
