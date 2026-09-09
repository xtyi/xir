# P1：Python 受限 DSL 前端与 Typed Builder

状态：进行中

## 完成事项

- [x] 实现 `@kernel` 装饰器和源码捕获。
- [x] 使用 `ast.parse` 解析 Python 源码。
- [x] 实现受限语法检查：函数、变量、`with`、`for`、intrinsic 调用。
- [x] 实现最小 Typed Builder/Node 构造接口。
- [x] 支持 `shared`、`pipeline`、`role`、`stages` 的最小 API。
- [x] 建立 Rust 原生 `ModuleBuilder` / `KernelBuilder`，作为 Python binding 的唯一构造后端。

## 待做

- [ ] 设计闭包、默认参数和源码不可获取时的报错策略。
- [ ] 定义 shape/dtype 表达式的常量求值规则。
- [ ] 增加 JSON AST 导入入口，供非 Python Agent 使用。
- [ ] 接入 Rust core 的 JSON schema/serde binding。
- [ ] 用 PyO3 将 Rust Builder 暴露给 `xir` Python 包。

## 完成标准

示例 Python DSL 可以生成稳定的 Typed AST；任意未支持的 Python 语法都能在源码位置处被拒绝。
