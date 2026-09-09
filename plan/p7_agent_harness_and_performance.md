# P7：Agent Harness 与性能反馈

状态：待做

## 完成事项

- [ ] 定义候选生成、构造检查、verifier gate、cost ranking、GPU evaluation 的协议。
- [ ] 定义稳定的 correctness/performance/finding artifact 格式。
- [ ] 缓存 AST、elaborated IR、MUSA C 和编译结果。
- [ ] 将重复失败归类为 verifier rule、IR primitive 或 calibration task。

## 待做

- [ ] 实现 shape contract、oracle、tolerance、target 和 reference policy。
- [ ] 实现单 shape tuning 与 dispatcher generalization 的分离。
- [ ] 建立 kernel corpus 和 compiler change regression gate。

## 完成标准

Agent 可以根据结构化 finding 修改 AST，并在不重新编译明显无关候选的情况下完成可审计的正确性和性能迭代。
