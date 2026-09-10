# P0：工程骨架与诊断协议

状态：进行中

## 已完成

- [x] Rust workspace、`xir-core`、Python package 和测试入口。
- [x] Schema v2 的 `Module` / `Kernel` / `ValueDef` / `TargetRequest`，serde 拒绝未知字段。
- [x] 统一诊断：severity、code、message、source_span、node_id、related_nodes。
- [x] 递归 JSON dump、确定性 NodeId 分配与已有 ID 冲突检测。
- [x] `xir-verify construct|verify` JSON CLI，Python 调用同一 native validation 入口。
- [x] Rust round-trip / diagnostics 和 Python native integration 回归。

## 待做

- [ ] TargetRequest 对应的 resolved profile、SDK/registry revision 和 capability evidence。
- [ ] 分析驱动的 related_nodes/repair hints；目前诊断主要定位 node/source。
- [ ] 按实际需求增加长期进程协议或 PyO3 binding；当前 subprocess bridge 是真实可用入口。

## 当前边界

TargetRequest 只表达请求身份，不能用能力名称列表代替目标验证。具体 instruction/representation admission 属于 P3/P4。
