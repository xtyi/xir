# CAKE：面向 AI Agent 的 Compiler / IR 协同演进

本文基于 [CAKE: Compiler–Agent Co-Design for Frontier Kernel Evolution](https://arxiv.org/html/2608.12629v1)（arXiv:2608.12629v1，2026-08-12）整理。论文的目标不是提出一个传统意义上的“更高层 GPU DSL”，而是把 **Agent 可编辑的程序表示、静态验证、性能反馈和编译器演进**设计成一个闭环。论文中的后端是 NVIDIA CUDA/PTX；本文在末尾给出迁移到摩尔线程 MUSA 的设计启示。

## 1. 核心问题与总设计

现有 kernel agent 通常把编译器当黑盒：Agent 生成代码，系统只返回编译失败、正确性结果和一个端到端耗时。这样的反馈无法定位是哪一个同步、硬件约束或流水线决策导致失败；当新 workload 暴露能力缺口时，语言和编译器也不能自我扩展。

CAKE 的三项承诺是：

1. Agent 编辑经过类型检查、显式描述硬件调度的 Cake IR，而不是直接编辑 CUDA/PTX。
2. Harness 在编译前返回局部化的安全性/硬件一致性诊断，并用代价模型过滤候选；GPU 执行和 profiler 仍是最终依据。
3. Compiler 与 IR 本身也是演进对象。重复出现的运行时错误、非法 lowering 或代价模型误判，会被提炼为新的 verifier 规则、IR primitive、校准任务或可复用 tactic，并通过 kernel corpus 测试门控。

IR 不是预先设计完毕的语言：从生产 kernel corpus 提取 barrier choreography、pipeline staging、warp-role partitioning、TMA/TMEM 生命周期等模式；每移植一个 kernel family，成功会验证抽象，失败则推动 IR 或 lowering 扩展。这是 bottom-up、evidence-driven 的语言演进。

## 2. Cake IR 的程序模型

### 2.1 显式 machine schedule

Cake IR 表达“机器应如何被驱动”，而不是只表达算子数学语义。一个 schedule 同时包含：

- 类型化的 compute、memory movement、synchronization、math 和 warp-control 操作；
- 命名的内存/同步/流水线资源及其 shape、dtype、ownership、lifetime；
- 命名的 warp groups（哪些 warp 承担 load、MMA、epilogue 等角色）；
- grid 配置和 pipeline stage 数；
- producer-consumer 之间显式的 barrier、wait、fence 和 handoff；
- 指令形式、操作数表示以及跨 memory tier 的移动。

关键分工是 **schedule 写“做什么”，lowering 推导“怎么编码”**。Agent 不直接写 barrier 地址、phase bit、TMEM offset、descriptor encoding 或 warp identity；这些由声明推导，从而保持硬件决策可见、机械细节可自动生成。

论文示例的结构可以抽象为：

```python
@cake.schedule()
def fmha_fwd(lm, Q: LM.tma3d, O: LM.tma2d, seqlen_q: lm.i32):
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

### 2.2 Layout 的取舍

CAKE 刻意不把 layout 设计成独立的一等代数对象（区别于 CuTe layout、Triton linear layout、Axe named-axis）。Agent 直接写 concrete commitments：SMEM view offset、operand byte offset、TMEM column range、swizzle tag、TMA descriptor coordinate。Compiler 负责检查 producer/consumer representation 是否一致，并验证它们是否满足 target instruction/resource contract。

这样做的动机是缩小 Agent 的编辑面，避免让 Agent 操作复杂 layout algebra；代价是 compiler verifier 必须承担更多表示推理。诊断应指向具体 IR 决策和宽泛的 mismatch 类别，而不是只报 backend error。论文没有规定 verifier 的内部算法，只规定其 contract 和 coverage。

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

静态检查是 pre-compile gate，覆盖：同步/排序、内存安全、数据流、资源使用、指令合法性、数据表示兼容性和 schedule structural invariants。每个 finding 要包含 IR 区域、违反的 contract 类别和稳定的修复目标。支持范围之外的信息缺失必须显式报告，不能当作检查通过。

数值正确性通过不同 shape、输入分布与外部 reference 对比，最终还要在目标框架端到端验证。代价模型只用于排序、过滤和高层 bottleneck attribution；真实设备测量和 profiler 是 ground truth。

### 3.2 Compiler evolution

编译器演进有两个入口：

- Agent 检查生产 kernel 与硬件文档，提出新的 instruction/resource/descriptor/synchronization primitive；
- Agent 根据 sanitizer、错误样例、mismatch 和调试日志，将 recurring failure 转成静态检查、分析规则或 cost-model calibration。

新 primitive 必须同时提供 effect、legality rules、分析支持和 corpus tests；新 verifier 规则也必须在 corpus 上验证，防止误拒绝合法 kernel。IR、lowering、analysis 和 tests 是一个共同演进单元。

### 3.3 Target 与 lowering

Target 必须精确匹配。若 device 或 toolchain 不支持，compiler 报告 unsupported，而不是静默降级到另一种架构。结构化 schedule 可跨架构复用，但 instruction admission、resource contract 和 lowering 必须 target-specific。静态检查后 deterministic 地生成可检查的 CUDA/PTX，再交给标准工具链；生成源码保留为 escape hatch。

## 4. Agent 工作流与库化

Workload contract 固定 shape、oracle、容差、硬件和允许的 reference，保留每次结果以便审计和复用。搜索分为两个目标不同的阶段：

1. 单 shape 演进：允许激进 specialization，以稳定的正确性和 latency 信号找到强 seed。
2. Library generalization：把 seeds 分成 shape buckets，生成 specialized/shared variants，按显式 guards 加 fallback，验证边界、tail、guard overlap/missing 和 held-out shapes。

Dispatcher 不能通过新增“方便的 evaluation rows”制造覆盖率，合法 shape domain 必须在 tuning 前声明。这样把 kernel 优化和库级泛化解耦。

## 5. 八项 IR 设计原则

| 原则 | 含义 |
|---|---|
| P1 Ergonomic | 接近 NumPy/PyTorch 编辑体验，避免不必要的 destination-passing/grid bookkeeping |
| P2 Performance-transparent | 性能相关硬件决策可见，lowering 可检查 |
| P3 Canonical | 每类操作尽量只有一种 canonical spelling |
| P4 Statically type-checked | 构造阶段拒绝无法合法 lowering 的程序 |
| P5 Analysis-friendly | 暴露静态分析所需信息 |
| P6 Test-gated | IR 变化必须通过 kernel-matrix 静态分析和编译测试 |
| P7 Analysis-consistent | IR data model 改动必须同步更新分析 |
| P8 Hardware-grounded | 每个 operation 的预期硬件行为有文档依据 |

## 6. 对 MUSA DSL 的直接启示

### 6.1 应继承的结构

- 让 Agent 编辑 typed schedule IR，保留 MUSA 的 warp/wave、异步拷贝、barrier、pipeline、cluster/多计算单元协作、SQMMA/矩阵指令等决策。
- 建立 declarative resource model：global、shared/local、寄存器/accumulator、descriptor、barrier、pipeline、role/group，并记录 shape、dtype、scope、owner、stage、lifetime。
- lowering 自动推导 MUSA descriptor 编码、barrier token/phase、寄存器或 accumulator offset、线程/warp 身份和 launch metadata；Agent 不直接拼装这些机械细节。
- 把 compiler harness 作为产品核心：construction check -> verifier -> cost model -> MUSA 编译 -> 数值 oracle -> device benchmark/profiler。
- 将 MUSA 编译器错误、非法 SQMMA operand layout、barrier deadlock、shared-memory 越界、pipeline stall 等重复问题沉淀成可定位的 verifier finding。

### 6.2 不能直接照搬的部分

CAKE 的 TMEM、TMA、WGMMA/tcgen05、CTA cluster 和 NVIDIA 架构矩阵不能直接映射到 MUSA。MUSA backend 应先建立 capability matrix 和 exact-target policy：例如 MP31 的 SQMMA 指令形态、warpgroup 数、累加器物理片段、共享内存 bank/swizzle、异步 copy/barrier 语义、支持的 dtype/量化格式，都应作为 target contract 显式建模。

布局可以沿用“concrete commitment + compiler verification”的思想，但 MUSA SQMMA 的 operand/C-fragment 物理顺序若无法由简单 offset 表达，应提供受约束的 fragment/resource 类型和 source-grounded verifier；不能把复杂硬件排列留给 Agent 猜测。

### 6.3 建议的最小闭环

第一阶段只实现少量 canonical primitives：`role`、`buffer/view`、`pipeline`、`barrier`、异步 load/store、SQMMA、elementwise/reduction、wait/fence，以及 exact MUSA target。先让 verifier 能定位 resource、scope、stage、operand compatibility 和 memory safety，再接入 cost model。每增加一个 MUSA primitive，同时提交 lowering、静态规则、生成代码检查和 kernel corpus 回归。

## 7. 论文边界与解读

论文报告了在 B200 上的 kernel 演进结果，但静态分析只覆盖其建模域，不能证明所有 GPU 正确性，也不能捕获全部微架构行为。性能估计需要 target-specific calibration；未校准 target 应报告 coverage limitation。论文的主要可迁移成果是接口和演进方法，而不是 NVIDIA 指令本身。

## 参考

- Ye et al., “CAKE: Compiler–Agent Co-Design for Frontier Kernel Evolution”, arXiv:2608.12629v1, 2026-08-12：<https://arxiv.org/html/2608.12629v1>
