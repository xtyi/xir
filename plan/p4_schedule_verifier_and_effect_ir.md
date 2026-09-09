# P4：Schedule Verifier 与 Effect IR

状态：待做

## 完成事项

- [ ] 实现资源容量、对齐、生命周期和 ownership 检查。
- [ ] 实现 role participation、scope 和结构化控制流检查。
- [ ] 实现 barrier/wait/fence、pipeline stage 和 async dependency 检查。
- [ ] 实现内存范围、address space、alias 和 fragment compatibility 检查。
- [ ] 构造轻量 Effect/Dependency Graph。

## 待做

- [ ] 复杂循环/分支时派生 CFG + token SSA。
- [ ] 增量重建受影响 region，避免每个 pass 全量分析。
- [ ] 建立 finding code 到修复策略的映射。
- [ ] 明确静态分析覆盖边界，未建模行为必须报告为 unknown。

## 完成标准

典型同步、越界、scope 和 SQMMA 表示错误在生成 MUSA C 前被拒绝，并能定位到具体节点和资源。
