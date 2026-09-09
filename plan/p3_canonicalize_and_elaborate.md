# P3：Canonicalize 与 Elaborate

状态：待做

## 完成事项

- [ ] 统一等价节点 spelling、索引表达式和 view 表达式。
- [ ] 解析资源/role/pipeline/barrier 符号引用。
- [ ] 物化 stage stride、stage lifetime、resource ownership。
- [ ] 为异步操作创建 event/token，并推导基本 effects。
- [ ] 生成 target capability 查询结果。

## 待做

- [ ] 明确 canonicalize 的幂等性和稳定排序规则。
- [ ] 设计 elaboration 前后 AST 的版本边界。
- [ ] 对动态 shape 保留运行时表达式，而不是错误地常量化。

## 完成标准

同一语义的不同 DSL 写法产生相同 canonical AST；elaborated AST 不再依赖隐式 stage、ownership 或 token 信息。
