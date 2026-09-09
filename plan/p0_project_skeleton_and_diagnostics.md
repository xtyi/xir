# P0：工程骨架与诊断协议

状态：进行中

## 已完成

- [x] 建立 Rust workspace 和 `crates/xir-core` core crate。
- [x] 定义 `Module`、`Kernel`、`Parameter` 和 `TargetSpec` 的 serde JSON 格式。
- [x] 提供 `musa:mp31` 最小 target descriptor。
- [x] 定义统一诊断结构：severity、code、message、source_span、node_id、related_nodes、repair_hint。
- [x] 提供确定性的 module tree dump。

## 待做

- [ ] 增加 Python/Rust binding（优先 PyO3 或 maturin）。
- [ ] 为诊断结构增加 builder 和 JSONL 输出协议。
- [ ] 将 `body_nodes` 替换为 P2 的 typed schedule node 序列。
- [ ] 补充 target descriptor 的真实 MP31 capability 数据并接入验证。
- [x] 增加 Rust 单元测试（JSON round-trip、默认 target、AST dump）。
- [ ] 增加 Python smoke test。

## 当前边界

P0 只固定工程和数据契约，不实现 Python DSL、typed schedule nodes、canonicalization、elaboration 或 MUSA C lowering。
