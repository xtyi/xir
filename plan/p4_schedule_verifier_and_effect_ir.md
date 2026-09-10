# P4：Schedule Verifier 与 Effect IR

状态：进行中

## 已完成

- [x] 递归 construction checks：SSA dominance、If/For yields、终结节点、独立 role 作用域。
- [x] 静态 storage capacity / alignment / stride / overflow / read-only binding 检查。
- [x] 顶层 ConcurrentRoles 的 disjoint warps、block join 和当前 SQMMA participant geometry 检查。
- [x] 单 producer/consumer、共同 iteration domain 的直线 pipeline body 验证。
- [x] Ticket 全覆盖、publish completion coverage、consume/release/drain 和 pending accumulator/input lifetime。
- [x] Pipeline storage 的裸 alias lease 绕过检查；static ranges 和保守 slot-envelope conflict 检查。
- [x] Effects 全量派生，包含 role/iteration、access/pending completion、event production/observation/publication 与 join。
- [x] Pass/Fail/Unknown report 与明确 coverage_limits；结构检查通过时总体仍是 Unknown。

## 待做

- [ ] 完整 operation contract registry、descriptor/mapping/fragment/transfer compatibility。
- [ ] Dynamic bounds、external alias/extent guards、cross-thread access footprints 与 uniformity 证明。
- [ ] 验证 target adapter 的 ordering/visibility/ownership 保真，而不只检查单条指令名称。
- [ ] 多 pipeline 依赖、复杂循环/分支协议；有需要时派生 CFG/token SSA 与 dependency graph。
- [ ] 根据实测需求增加 revision-keyed 分析缓存，初始版本不缓存 effects。
- [ ] 带关联节点和修复建议的丰富诊断。

## 当前边界

每个 role 最多一个 PipelineFor，每轮恰好一个 ticket，异步 protocol 必须位于直线 body，单 accumulator 最多一个 pending update。当前没有证明所有 GPU race/deadlock 被排除，也没有 backend/device evidence；不能将 finish_checked 视为执行授权。
