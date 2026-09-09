# P2：Typed Schedule AST 核心节点

状态：进行中

## 完成事项

- [ ] 实现 Kernel、参数、Buffer、BufferView、Accumulator、Role、Pipeline、Barrier 声明。
- [ ] 实现 RoleRegion、StageFor、If、For 等结构化控制流。
- [ ] 实现 Load、Store、AsyncCopy、Wait、Fence、Signal。
- [ ] 实现 Elementwise、Reduce、Mma、Sqmma、Atomic。
- [x] 补充 `Assign`、`Load`、`Store`、`Binary`、`If`、`For`、`Return` 基础节点。
- [x] 提供 body closure builder，保留嵌套作用域关系。
- [x] Builder 支持 `finish_checked()` 并返回结构化 `DiagnosticBag`。
- [x] 增加 `NodeIdGenerator`，在 `finish`/`finish_checked` 时为嵌套节点分配确定性 ID。
- [x] 将公共节点 metadata 与值定义分离：`NodeMeta`、`operands`、`ValueDef`。
- [x] 实现第一版类型系统：scalar/index/predicate/pointer/buffer/tile/fragment/event/resource/void。
- [x] 将存储类型统一为 `BufferType`，移除独立 `MemRefType`/`ViewType`。
- [x] 增加 `BufferRef` 和 `StorageType` 数据模型。
- [x] 按物理存储与逻辑解释拆分 `StorageType` / `BufferType`，并移除 `TileType`。
- [x] 从基础 `BufferRef` 移除 pipeline-specific 的 `stage` 字段。
- [x] 使用 `StorageId`、`BufferId`、`ValueId` 替代裸字符串引用。
- [ ] 为 effectful 节点加入 reads/writes/produces/consumes/scope/capability 属性。

## 待做

- [ ] 固定 canonical spelling 和节点版本策略。
- [ ] 定义 fragment representation 与 MUSA SQMMA 类型约束。
- [ ] 定义 resource slice、alias 和 lifetime 数据结构。
- [ ] 将 Python 参数标注与 Rust `TypeRef` 对接。
- [ ] 实现 operation-level type inference 和 conversion rules。
- [ ] 将 `BufferType`/`BufferRef` 接入 `BufferDecl`、`Load`、`Store`、`AsyncCopy` 和 `Sqmma` 节点。
- [ ] 在 verifier 中检查 BufferRef 的 offset、shape、stride、stage 和 swizzle 兼容性。
- [ ] 增加受约束的非仿射 `LayoutTransform`，仅用于无法用 shape/stride 表达的硬件映射。
- [ ] 在 pipeline/view elaboration 中引入 stage 关系和 stage-specific views。

## 完成标准

核心节点可以构造、打印、序列化，并通过类型系统区分 address space、dtype、shape、scope 和 event token。
