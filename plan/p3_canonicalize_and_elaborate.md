# P3：Canonicalize 与 Elaborate

状态：进行中

## 已有基础（不代表本阶段完成）

- [x] 结构化 IndexExpr 和 typed references，不将动态参数错误地常量化。
- [x] Builder ID normalization、基础常量计算、原生 declaration/reference checks。
- [x] Event/ticket 由操作显式声明，基础 effects 在 P4 全量派生。

- [x] CUDA SM120 elementwise 的最小 target admission：检查 flat nodes、连续 FP32 view、线性 index/mask 和 ABI；通用 registry 尚未实现。

## 待做

- [ ] 等价 IR canonicalization、幂等性和保留 NodeId/origin 的变换规范。
- [ ] Operation registry：解析 instruction、transfer、representation 与 exact target contracts。
- [ ] 将共同 iteration domain elaborate 为 slot/generation、stage views 与资源生命周期。
- [ ] 将 publish/consume/release/drain elaborate 为有证据的 target completion/ordering operations。
- [ ] 记录 target/registry/analysis revision 与 elaboration source mapping。

## 完成标准

Elaborated AST 显式表达 target 所需的资源和同步语义，adapter 保留 completion、happens-before 与 lease 不变量。当前非空 spec ID 不能代替这些验证。
