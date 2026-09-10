# CAKE：面向 AI Agent 的 Compiler / IR 协同演进

本文基于 [CAKE: Compiler–Agent Co-Design for Frontier Kernel Evolution](https://arxiv.org/html/2608.12629v1)（arXiv:2608.12629v1，2026-08-12）整理，并于 2026-09-09 对照正文和附录复核。论文的目标是把 **Agent 可编辑的程序表示、静态验证、性能反馈和编译器演进**设计成一个闭环。论文中的后端是 NVIDIA CUDA/PTX；MUSA 是本文讨论的迁移方向，论文没有验证这一迁移。

## 1. 核心问题与总设计

现有 kernel agent 通常把编译器当黑盒：Agent 生成代码，系统只返回编译失败、正确性结果和一个端到端耗时。这样的反馈无法定位是哪一个同步、硬件约束或流水线决策导致失败；当新 workload 暴露能力缺口时，语言和编译器也不能自我扩展。

CAKE 的三项承诺是：

1. Agent 编辑经过类型检查、显式描述硬件调度的 Cake IR，而不是直接编辑 CUDA/PTX。
2. Harness 在编译前返回局部化的安全性/硬件一致性诊断，并用代价模型过滤候选；GPU 执行和 profiler 仍是最终依据。
3. Compiler 与 IR 本身也是演进对象。重复出现的运行时错误、非法 lowering 或代价模型误判，会被提炼为新的 verifier 规则、IR primitive、校准任务或可复用 tactic，并通过 kernel corpus 测试门控。

IR 不是预先设计完毕的语言：从生产 kernel corpus 提取 barrier choreography、pipeline staging、warp-role partitioning、TMA/TMEM 生命周期等模式；每移植一个 kernel family，成功会验证抽象，失败则推动 IR 或 lowering 扩展。这是 bottom-up、evidence-driven 的语言演进。
## 2. Cake IR 的程序模型

### 2.1 显式 machine schedule

Cake IR 同时保留计算语义和“机器应如何被驱动”的调度决策。一个 schedule 包含：

- 类型化的 compute、memory movement、synchronization、math 和 warp-control 操作；
- 命名的内存/同步/流水线资源及其 shape、dtype、ownership、lifetime；
- 命名的 warp groups（哪些 warp 承担 load、MMA、epilogue 等角色）；
- grid 配置和 pipeline stage 数；
- producer-consumer 之间显式的 barrier、wait、fence 和 handoff；
- 指令形式、操作数表示以及跨 memory tier 的移动。

关键分工是 **schedule 指定硬件决策，lowering 推导这些决策的机械实现**。Agent 仍会选择 role 的 warp 集合、SMEM view offset、TMEM column range、swizzle 等；lowering 据此生成 barrier 地址、phase 更新、最终地址计算、descriptor bit encoding 和 warp identity 判断。不能把“自动推导 TMEM offset”理解为“Agent 不决定 TMEM 放置”，也不能把它扩大成“编译器自动选择全部 layout、pipeline 和并行策略”。

下面保留论文 Figure 3 的 **schedule fragment**，用于说明声明与跨 role handoff，不能作为可运行的 FMHA 实现或完整 API 规范：

```python
@cake.schedule()
def fmha_fwd(lm, Q: LM.tma3d, O: LM.tma2d, seqlen_q: LM.i32):
    pool = lm.smem(98304)
    smem_q = pool.view(offset=0, shape=(128, 128), dtype=lm.bf16, stage=3)
    tmem_acc = lm.tmem(cols=0, width=128, shape=(128, 128), dtype=lm.f32)

    load = lm.role(warps=[0])
    mma = lm.role(warps=[1])
    pipe = lm.pipeline(stages=3)
    q_full = lm.barrier(count=3, prod=[load], cons=[mma],
                        init_count=1, pipeline=pipe)

    with load:
        for stage in lm.range(0, 3):
            smem_q.tma_load(Q, coords=(0, 0, stage),
                            stage=stage, barrier=q_full)
    with mma:
        for stage in lm.range(0, 3):
            lm.wait(q_full, stage=stage)
            lm.fence_proxy()
            lm.mma(tmem_acc, smem_q[stage], smem_q[stage], init=(stage == 0))
```

这个片段没有展示完整的 Q/K/V 数据流、softmax、输出写回或资源收尾。`range(0, 3)` 也只展示三个 stage，不能据此推断循环复用的完整协议；把同一个 view 同时传给两个 MMA operand 不能当作真实 attention 的数学定义。尤其不能把 `warps=[1]` 原样迁移到另一目标的 collective MMA：参与线程集合必须按具体指令核对。

### 2.2 Layout 的取舍

CAKE 刻意不把 layout 设计成独立的一等代数对象（区别于 CuTe layout、Triton linear layout、Axe named-axis）。Agent 直接写 concrete commitments：SMEM view offset、operand byte offset、TMEM column range、swizzle tag、TMA descriptor coordinate。Compiler 负责检查 producer/consumer representation 是否一致，并验证它们是否满足 target instruction/resource contract。

这样做的动机是缩小 Agent 的编辑面，避免让 Agent 操作复杂 layout algebra；代价是 compiler verifier 必须承担更多表示推理。诊断应指向具体 IR 决策和宽泛的 mismatch 类别，而不是只报 backend error。论文描述的是 verifier 的 contract 和 coverage，没有给出完整内部算法。

需要区分三件事：不暴露独立的 layout algebra、不需要 layout 信息、不在 compiler 内部使用 layout 分析。CAKE 明确选择了第一项，并没有由此得到后两项。对于 XIR，内部使用 mapping、受约束的 layout template 或专用 fragment 类型，与保持 Agent 编辑面具体并不冲突；这属于实现选择，不是论文规定。[§2.2、Appendix B.4](https://arxiv.org/html/2608.12629v1#A2.SS4)

### 2.3 声明式资源模型

论文把资源分为五类：shared-memory regions、tensor-memory regions、synchronization objects、warp roles、pipelines。声明记录 shape、dtype、所有权、生命周期、生产者/消费者和 stage 关系；硬件敏感选择保留在 IR，纯机械 metadata 在 lowering 阶段生成。资源声明既是 codegen 输入，也是静态分析的事实来源。

### 2.4 能力分类

操作词汇分成四类：

| 类别 | 内容 |
|---|---|
| Compute | 矩阵、elementwise、reduction，以及目标支持的精度模式 |
| Memory movement | global/on-chip 间显式传输，包括异步和 collective transfer |
| Synchronization | role、pipeline stage、hardware scope 间的 ordering/coordination |
| Control and scheduling | warp-role assignment、pipeline 管理、persistent execution、multi-block coordination |

## 3. Compiler / Harness 设计

### 3.1 分层处理与证据

Harness 的标准流程是：生成结构不同的 IR 候选；执行 IR construction checks、verifier hard gates 和 cost-model ranking；只把幸存者送到 GPU；用 external oracle、benchmark 和 profiler 评估；再把证据路由到候选、verifier、cost model 或 IR vocabulary。

静态检查是 pre-compile gate，检查类别包括：同步/排序、内存安全、数据流、资源使用、指令合法性、数据表示兼容性和 schedule structural invariants。每个 finding 应定位 IR 区域和违反的 contract 类别，通过稳定接口给出可操作的修复目标。支持范围之外的信息缺失必须显式报告，不能当作检查通过。

这里的“覆盖”指已支持的分析类别，不代表完整证明这些性质。Appendix C 明确指出可能出现 false positive 和 false negative。类型检查、schedule verifier、外部工具链编译、数值验证是不同层次的检查；通过前一层不保证后一层通过。[§3.1、Appendix C](https://arxiv.org/html/2608.12629v1#A3)

数值正确性通过不同 shape、输入分布与外部 reference 对比，最终还要在目标框架端到端验证。代价模型只用于排序、过滤和高层 bottleneck attribution；真实设备测量和 profiler 是 ground truth。

Table 1 还区分了 blocking gate、performance report 和 optimization hint。性能报告可以影响搜索调度，但性能差不等于程序非法；缺少某个 target 的模型校准，也不等于该 target 无法编译。

### 3.2 Compiler evolution

编译器演进有两个入口：

- Agent 检查生产 kernel 与硬件文档，提出新的 instruction/resource/descriptor/synchronization primitive；
- Agent 根据 sanitizer、错误样例、mismatch 和调试日志，将 recurring failure 转成静态检查、分析规则或 cost-model calibration。

新 primitive 必须同时提供 effect、legality rules、分析支持和 corpus tests；新 verifier 规则也必须在 corpus 上验证，防止误拒绝合法 kernel。IR、lowering、analysis 和 tests 是一个共同演进单元。

对 XIR 的建议是分开保存两个层面的版本：kernel candidate 的版本，以及 IR/verifier/lowering 的版本。修改 compiler 后，先重跑已知合法与非法样例，再重新评测受影响的候选；external oracle、容差和评分口径应由 workload contract 管理，不能随修复一起放宽。版本管理方式是工程建议，论文没有给出具体协议。

### 3.3 Target 与 lowering

Target 必须精确匹配。若 device 或 toolchain 不支持，compiler 报告 unsupported，而不是静默降级到另一种架构。结构化 schedule 可跨架构复用，但 instruction admission、resource contract 和 lowering 必须 target-specific。静态检查后 deterministic 地生成可检查的 CUDA/PTX，再交给标准工具链；生成源码保留为 escape hatch。

“结构可复用”不等于同一份 schedule 无修改地跨 target 保持合法性或性能。生成源码可脱离 Cake 集成，但若在生成代码中手工修改同步或访存，原 IR 上的分析结论不能自动覆盖这些修改。稳定地生成源码也不保证外部工具链的寄存器分配、指令调度和运行时间保持不变。

## 4. Agent 工作流与库化

论文区分从已有生产实现继续优化，以及在不能查看 low-level target implementation 的条件下合成 schedule。后者仍可使用数学描述、oracle、高层代码和 black-box baseline；论文的 frontier 是这个实验意义上的限制，不意味着算子或算法本身此前不存在。IR 最初从专家 kernel corpus 提取，也不意味着每次 clean-start run 都能查看被测 workload 的底层实现。[§1、§5](https://arxiv.org/html/2608.12629v1#S5)

Workload contract 固定 shape、oracle、容差、硬件和允许的 reference，保留每次结果以便审计和复用。搜索分为两个目标不同的阶段：

1. 单 shape 演进：允许激进 specialization，以稳定的正确性和 latency 信号找到强 seed。
2. Library generalization：把 seeds 分成 shape buckets，生成 specialized/shared variants，按显式 guards 加 fallback，验证边界、tail、guard overlap/missing 和 held-out shapes。

Dispatcher 不能通过新增“方便的 evaluation rows”制造覆盖率，合法 shape domain 必须在 tuning 前声明。泛化阶段按包含 dispatch 的固定 workload 表现评分，额外 route 应由实际收益支撑；候选单独很快不代表包含 dispatch 和 fallback 后仍快。这样把 kernel 优化和库级泛化解耦。[§6](https://arxiv.org/html/2608.12629v1#S6)

## 5. 八项 IR 设计原则

| 原则 | 含义 |
|---|---|
| P1 Ergonomic | 接近 NumPy/PyTorch 编辑体验，避免不必要的 destination-passing/grid bookkeeping |
| P2 Performance-transparent | 性能相关硬件决策可见，lowering 可检查 |
| P3 Canonical | 每类操作尽量只有一种 canonical spelling |
| P4 Statically type-checked | 用 typing rules 约束 lowering，在构造阶段拒绝 ill-typed 程序；不保证排除所有非法 lowering |
| P5 Analysis-friendly | 暴露静态分析所需信息 |
| P6 Test-gated | IR 变化必须通过 kernel-matrix 静态分析和编译测试 |
| P7 Analysis-consistent | IR data model 改动必须同步更新分析 |
| P8 Hardware-grounded | 为每个 operation 记录预期硬件行为 |

这些是 Appendix B.1 的设计目标，不是完整性证明。特别是 canonical form 不应抹掉具有不同性能含义的 schedule：减少同一操作的等价拼写，与保留不同 pipeline、role 分工和 memory placement 是兼容的。

## 6. 对 MUSA/XIR DSL 的设计建议

本节是根据上述思想提出的设计建议。MUSA 事实以官方 MUTLASS 源码为补充依据；此次只做文献和源码核对，没有进行 MUSA 编译或设备测试。

### 6.1 先确定 target contract，再确定词汇

CAKE 的 TMEM、TMA、WGMMA/tcgen05、CTA cluster 不能直接映射到 MUSA。应选定具体设备、architecture、SDK/`mtcc` 版本，建立 capability matrix，分别记录 instruction、resource、scope、分析覆盖和性能模型校准状态。

官方 MP31 实现提供了比笼统的“异步 copy + MMA”更具体的起点：

| 官方源码中可确认的事实 | 对 XIR 的设计含义 |
|---|---|
| TME 即 Tensor Memory Engine；tile load 接收 descriptor、coordinate、shared-memory destination 和 barrier ID | 将 TME transfer 的形状、目标 view、完成关系纳入 IR；TME 不等于 NVIDIA TMA，更不等于 TMEM |
| SQMMA 即 Squad-level MMA；所查 MP31 traits 的 `ThrID = Layout<_128>`，并定义 A/B descriptor 和 C register fragment | 按 instruction contract 检查 128-thread participant set 与 fragment mapping；不能照搬一个 warp 的 MMA role |
| SQMMA descriptor 包含 start address、leading stride 和 swizzle 信息 | 保留影响性能的 stride/swizzle 选择，自动生成 bit encoding，并验证 TME 写入表示与 SQMMA 读取表示一致 |
| `warpsquad_wait()` 调用 `__musa_sqmma_wait()`；MP31 TME pipeline 区分 full/empty barrier，并跟踪 index/phase | 分开描述 transfer completion、MMA completion 与 buffer reuse，不能用一个含糊的 `wait` 混合所有语义 |

上述依据分别是 [TME primitives][musa-tme]、[SQMMA primitives][musa-sqmma]、[SQMMA traits][musa-traits]、[descriptor][musa-desc] 和 [MP31 pipeline][musa-pipeline]。它们确认的是所查库实现及其约束，不是完整 ISA 规范，也不能证明其他 MUSA target 具备相同能力。

原总结中的 `warp/wave`、`warpgroup`、`cluster/多计算单元协作` 不宜作为已确认的 MUSA 通用能力。IR 可以预留 group/scope 扩展点，但每个 target 只能启用已核实的语义。也不要由“有 accumulator”推导出“存在 NVIDIA TMEM 式的独立存储层”。

### 6.2 Agent 决定什么，compiler 推导什么

| Agent 可编辑的 schedule 决策 | Compiler 应推导或检查的内容 |
|---|---|
| Tile shape、instruction family、dtype、accumulation/rounding policy | 支持的 instruction form、operand/result type、numerical contract |
| Buffer/view、stage count、stride、swizzle、显式 placement | 地址计算、alignment、容量、alias/lifetime、descriptor encoding |
| Role participant set、producer/consumer 分工 | Thread/warp identity 展开、collective participation、合法 scope |
| 异步依赖、buffer handoff、复用策略 | Barrier allocation、phase progression、completion 与 reuse 检查 |
| Grid、persistent scheduling、specialization guards | Launch legality、guard 对实际参数的约束、fallback 覆盖 |

不要把 register allocation 归入“声明后就能确定的机械 offset”。若 backend 生成 MUSA C，再交给 `mtcc`，最终 physical register allocation 仍由外部编译器完成。XIR 可以规定 logical fragment 及其 lane/value mapping，并报告最终资源使用，但不能仅凭前端声明承诺物理寄存器编号、无 spill 或精确 occupancy。

资源模型应区分 memory space 与 representation：例如 global/shared/thread-private storage 描述可访问性和存储语义，register fragment/accumulator 描述值如何分布或被指令消费。不要将 `shared/local` 混成一个空间；若采用 `local` 名称，应明确它在 XIR 中的定义。

### 6.3 先定义 effect 和生命周期，再选 API 拼写

仅有 `role`、`buffer`、`barrier`、`sqmma` 节点名称不足以支撑 verifier。建议每个 primitive 至少规定：

- Operand/result type、address space、shape、alignment 和 alias 前提。
- 实际参与的线程集合、发起指令的线程集合，以及控制流的 uniformity 要求；这些集合不必相同。
- 读写哪个 resource/view、在哪个 stage/iteration，是否有尚未完成的异步访问。
- 完成由什么事件表示，等待后保证哪些访问完成或可见，在哪个 scope 生效。
- 哪些行为可重排，哪些资源可以复用，哪种事实缺失会使分析返回 unknown。

动态参数的前提也必须进入 contract：例如 shape/stride 范围、pointer alignment、可访问 buffer extent、aliasing 和 tail mask。编译期无法证明的条件，应由可检查的 launch/dispatch guard 保证或报告分析限制；仅在类型里写下一个假设并不构成证明。Numerical contract 还需独立规定 accumulation precision、允许的重排/近似以及误差判据，避免把 memory-safe、instruction-legal 与数值正确混为一谈。

例如，循环复用的 staged buffer 可以用下列**语义示意**描述；这些状态不是 CAKE 或 MUSA 的实际 API：

```text
free(slot, generation)
    -> producer acquires slot
    -> transfer is issued
    -> transfer completes; consumer may access data
    -> consumer issues compute
    -> all accesses to the slot complete; consumer releases slot
    -> free(slot, next_generation)
```

对 `S` 个 slot，逻辑迭代 `i` 可映射为 `slot = i % S`、`generation = i // S`。物理 phase 如何编码、初始状态如何设定、是否还需 fence，必须由目标协议决定，不能只看到一个 phase bit 就套用同一种算法。

这里至少存在两条依赖：producer 完成后 consumer 才能读；consumer 对旧数据的最后一次访问完成后 producer 才能覆盖。如果 MMA 异步读取 shared memory，“已经发出 MMA”不代表可以复用输入 buffer；accumulator 的完成条件也应单独建模。`wait` 表示等待什么、`fence` 保证哪种 ordering/visibility，都需要明确，不能把两者当作同义词。官方 [MP31 pipeline][musa-pipeline] 的 full/empty 协议和 [SQMMA wait][musa-sqmma] 是可用于提取这些约束的起点。

还需要覆盖 pipeline prologue/drain、零次或少于 stage 数的迭代、tail predicate、分支中的 barrier participation，以及最后一次消费后的资源释放。对于当前无法分析的控制流，应明确限制支持范围。

### 6.4 将诊断与性能证据设计成稳定接口

建议把构造检查、schedule verifier、backend compilation 和 runtime validation 的结果分开，并区分 `pass`、`fail`、`unknown`。`unknown` 不是通过；是否允许继续做受控的设备验证，应由明确的 harness policy 决定。静态检查通过后仍需要 external oracle、runtime 检查和设备测量。

一个 **XIR 诊断格式建议**如下，字段和错误码均为示意：

```json
{
  "category": "synchronization",
  "status": "fail",
  "code": "XIR_BUFFER_REUSE_BEFORE_COMPLETION",
  "node_id": "copy_next_tile",
  "related_nodes": ["mma_current_tile"],
  "resource": "smem_a",
  "message": "The producer may overwrite a slot before the consumer access completes.",
  "evidence": "No completion dependency orders the next write after the pending read.",
  "hint": "Complete the pending read before releasing the slot."
}
```

诊断还应能携带 source span、target 和 analysis version。编译器能够确定的证据与启发式 hint 应分开：例如某条指令不支持该 dtype 可以是 hard gate，推测 pipeline stall 或建议增加 stage 则通常属于 performance report/hint。不能把“pipeline stall”直接等同于程序非法。

性能部分可以先报告可解释的静态资源量和 backend 资源结果，再用设备 microbenchmark 校准 ranking model。缺少校准时报告 unavailable；不要因为模型预测较差就永久排除所有新结构，应留一部分实测预算检查模型的盲点。保存 target/toolchain、IR/compiler revision、shape/stride/dtype、benchmark 协议与原始样本，才能区分 schedule 改进、工具链变化和测量噪声。

### 6.5 建议的最小闭环与扩展顺序

原建议一次纳入 SQMMA、异步 load/store、reduction 和复杂 role protocol，作为“最小闭环”仍然偏大。可以先把垂直链路跑通，再按 corpus 扩展：

1. **基础链路**：固定一个 exact target；实现 typed load/store、简单 elementwise、launch/indexing 和 tail mask。得到可检查的 MUSA C，经 `mtcc` 编译，完成 oracle 对比和设备计时。这一阶段验证基础设施，不代表已经实现 CAKE 的核心优势。
2. **第一个代表性 schedule**：选择一个有官方基线的 GEMM，明确 input/accumulator types、participant set、shared-memory representation 和完成条件；先得到容易验证的版本，再加入 role specialization 与多 stage 的 TME/SQMMA overlap。单 stage 也必须正确处理异步完成。
3. **Verifier 的反馈闭环**：保留已知合法案例和能定位的非法案例，例如 descriptor mismatch、缺少 completion dependency、越界 view、非法 participant set。补充 pipeline wraparound 与 tail 测试，避免只通过一个恰好整除的 shape。
4. **第二个不同结构的 family**：引入 reduction、attention 或不规则 tail，用实际缺口推动 primitive/effect 扩展。已有合法与非法 corpus 同步回归，检验抽象是否只适合第一个 GEMM。
5. **搜索与库化**：增加经校准的性能排序、结构化候选变异、held-out shape 和 dispatcher/fallback 验证。每个新增 route 都应有独立的正确性和性能证据。

每增加一个 primitive，都应同时提交 semantic/effect contract、target admission、lowering、静态规则和回归案例。若环境暂时没有 MUSA 设备，源码生成与静态分析可以先验证，设备编译、正确性和性能必须保留为未验证项。

## 7. 实验证据及其边界

以下均为论文报告值，本次没有复现实验。Speedup 应连同 workload、baseline 和统计方式一起阅读。[§5–6、Table 2、Appendix E](https://arxiv.org/html/2608.12629v1#S5)

| 实验 | 报告结果 | 解释时必须保留的条件 |
|---|---|---|
| Flash-KMeans clean start | Cake IR 为 `1.144× [1.041, 1.205]`；direct CUDA/PTX 为 `0.928× [0.852, 1.151]` | B200，固定 assign shape：`B=32, N=65536, K=1024, D=128`；每组 3 次运行，80M token 预算；数值为相对 tuned FlashML baseline 的 best validated speedup 的 median `[min, max]` |
| KDA prefill | 相对 official FlashKDA，6 个 B200 BF16 shapes 的 geometric mean 为 `2.05×` | FlashKDA 是不可查看实现的 timing baseline；bitwise correctness 属于该 validation contract，另有 SGLang serving 验证 |
| Known-kernel reproduction | 11 项固定比较中 10 项达到或超过 reference，另一项约为 `0.965×` | 允许查看 reference；这是复现并继续优化已有专家结构，与 implementation-hidden synthesis 的问题不同 |
| Dispatcher portfolio | KNN build/search、KMeans 分别为 `1.418× / 2.116× / 1.803×`，覆盖 `112 / 198 / 124` shapes | §6 报告的是 GB200 上包含 dispatch 的 per-shape GPU-span speedup 的 unweighted geometric mean，不能当成 Table 2 的 clean-start 结果 |

由这些实验可以支持“显式 IR 与 compiler feedback 值得作为 Agent 编程环境来设计”。但还不能推出以下结论：

- **不能归因于单一语言特性。** Clean-start 对比控制了 Agent、任务和 reference policy，但没有分别消融 layout algebra、verifier、cost model 和 compiler evolution；不能把整体差异单独归功于“不暴露 layout algebra”。
- **不能认定搜索到理论最优。** 预算内的 best candidate 与 plateau 都依赖搜索协议；3 次运行和一个固定 clean-start shape 不构成所有 workload 的普遍保证。
- **不能混用不同指标。** GPU span、API latency、serving throughput 和 token/evolve time 是不同指标；kernel speedup 不直接等于模型端到端加速。GB200 portfolio 与 B200 clean start 的差值也不是受控测得的“泛化成本”。
- **不能由源码行数证明生产力。** 更短的 IR 可以说明表示紧凑，但抽象层级、支持代码范围、调试成本和有效候选比例都不同。
- **不能外推 MUSA 性能。** 论文 §8 明确说 non-NVIDIA transfer 尚未测量。Appendix B.5/§8 的 timing model 只对 B200 和 H100 有校准；列出 architecture support 不等于每个 target、每类 kernel 都有同等实证覆盖。

因此，把 CAKE 作为 XIR 的设计起点是有依据的；具体取舍仍应通过 MUSA corpus、失败样例和受控实验检验。

## 8. 论文没有交付完整设计规范的部分

“论文没有具体设计细节”略显绝对：Figure 3、Table 1 和 Appendix B/C 已经给出编辑面、资源类别、设计原则、target 策略与分析边界。但这些还不足以照着实现一个完整 compiler。对 XIR 至少仍需明确：

| 尚需设计的内容 | 必须回答的问题 |
|---|---|
| IR representation 与 frontend | AST、SSA、region、token/effect graph 如何分工？Python 是 tracing、AST capture 还是仅作 builder？论文未规定 |
| Type/effect system | Shape/stride/alias、fragment、scope、uniformity、异步完成如何表示和组合？ |
| Schedule semantics | 跨 role 的执行关系是什么？循环、分支、stage 复用和 early exit 的合法性如何定义？ |
| Representation verification | Producer/consumer mapping 如何比较？哪些约束可静态判定，哪些必须拒绝或返回 unknown？ |
| Lowering contract | 哪些转换只是编码，哪些会改变 schedule？如何将生成代码或 backend 错误映射回 IR？ |
| Performance 与 evolution | Cost model 怎么校准、何时失效？compiler 变更怎样做回归、版本化和重新评测？ |

对新 DSL 而言，优先产出应是少量可执行的 semantic contracts、可定位的失败案例，以及它们到真实硬件行为的映射。语法可以随后完善；否则容易得到一个能描述 schedule、却无法可靠检查 schedule 的语言。

## 参考

- Ye et al., [CAKE: Compiler–Agent Co-Design for Frontier Kernel Evolution](https://arxiv.org/html/2608.12629v1), arXiv:2608.12629v1, 2026-08-12。本文锁定 v1，不把后续版本或 upstream PR 的当前状态混入论文结论。
- Moore Threads, [MUTLASS](https://github.com/MooreThreads/mutlass/tree/78b349514831ed54a797b717f2fc0534646acff2)。MUSA 源码核对固定在 commit `78b349514831ed54a797b717f2fc0534646acff2`，访问日期 2026-09-09。

[musa-tme]: https://github.com/MooreThreads/mutlass/blob/78b349514831ed54a797b717f2fc0534646acff2/include/mute/arch/copy_mp31_tme.hpp
[musa-sqmma]: https://github.com/MooreThreads/mutlass/blob/78b349514831ed54a797b717f2fc0534646acff2/include/mute/arch/mma_mp31_sqmma.hpp
[musa-traits]: https://github.com/MooreThreads/mutlass/blob/78b349514831ed54a797b717f2fc0534646acff2/include/mute/atom/mma_traits_mp31_sqmma.hpp
[musa-desc]: https://github.com/MooreThreads/mutlass/blob/78b349514831ed54a797b717f2fc0534646acff2/include/mute/arch/mma_mp31_desc.hpp
[musa-pipeline]: https://github.com/MooreThreads/mutlass/blob/78b349514831ed54a797b717f2fc0534646acff2/include/mutlass/pipeline/mp31_pipeline.hpp
