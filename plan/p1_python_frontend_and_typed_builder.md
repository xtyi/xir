# P1：Python 受限 DSL 前端与 Typed Builder

状态：进行中

## 已完成

- [x] `@kernel`、源码行列定位和 `kernel_from_source()` fallback；不执行 source/annotations。
- [x] 显式 typed parameters、SSA value rebinding、For carried arguments/results、If results/yields。
- [x] Storage/Buffer/Role/Pipeline/Accumulator declarations，显式 ConcurrentRoles 与 PipelineFor。
- [x] Load/Store、ticket/event protocol、AsyncCopy、AccumulatorInit/Read、Sqmma 语法。
- [x] 严格检查 arity/keyword/unpacking；拒绝 defaults、closures、未知语法和旧 stages/wait。
- [x] 与 Rust 相同的 canonical JSON schema；JSON import 和源码构造均调用 `ModuleBuilder::finish_checked()`。
- [x] 将 source/value errors 与原生 construction errors 分开报告，缺少 native executable 明确失败。

## 待做

- [ ] Region-local dynamic views、fragment store/conversion 等语法随核心 contract 扩展。
- [ ] 如吞吐量需要，用长期 native process 或 PyO3 替换 subprocess transport。
- [ ] 各种等价表达式 canonicalization；目前只做受限 literal/index expression lowering。

## 当前边界

资源声明位于 executable statements 之前；For induction 和 body-local variables 使用词法作用域。Python 不实现独立 schedule verifier，代码及可运行示例见根目录 README。
