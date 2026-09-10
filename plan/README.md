# Compiler 实施计划

## 状态约定

- `完成`：代码、接口和对应验证已具备。
- `进行中`：接口或实现已经开始，但完成标准尚未满足。
- `待做`：尚未实现。

每个阶段独立记录完成事项和待做事项。阶段完成后更新对应文件，并保留失败样例和诊断输出。

## 阶段总览

| 阶段 | 目标 | 状态 |
|---|---|---|
| P0 | 工程骨架、目标和诊断协议 | 进行中 |
| P1 | Python 受限 DSL 前端和 Typed Builder | 进行中 |
| P2 | Typed Schedule AST 核心节点 | 进行中 |
| P3 | canonicalize / elaborate | 待做 |
| P4 | Schedule verifier 和派生 Effect IR | 进行中 |
| P5 | MUSA C emitter | 待做 |
| P6 | `mtcc` 编译集成和端到端 kernel | 待做 |
| P7 | Agent harness、性能反馈和增量演进 | 待做 |

## 阶段依赖

```text
P0 -> P1/P2 -> P4 construction/protocol subset
          \-> P3 target elaboration -> P4 target verification -> P5 -> P6 -> P7
                                    \-> optional CFG/token SSA analysis
```

CFG/token SSA 是 P4 的派生分析能力，不是独立的持久化语言层。

## 完成定义

每个阶段必须同时交付：接口/实现、正向样例、负向诊断样例、最小回归测试和一份更新后的阶段记录。没有真实 MUSA 环境时，生成代码和静态验证可以完成，但设备编译/性能结果必须标记为未验证。

## Schema v2 迁移状态

当前已接通 Python → canonical JSON → Rust Builder/verifier；数据模型、结构化 SSA、role/pipeline 生命周期与派生 effects 有正反例回归。Operation registry、target elaboration、MUSA C、mtcc 和设备测试尚未实现。具体运行方法和限制以 [项目 README](../README.md) 为准。
