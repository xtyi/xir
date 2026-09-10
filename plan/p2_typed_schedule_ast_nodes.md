# P2：Typed Schedule AST 核心节点

状态：进行中

## 已完成

- [x] 独立 Value/Storage/Buffer/Role/Pipeline/Accumulator/Event/Ticket typed ID。
- [x] StorageType / BufferType / FragmentType 分层，结构化 IndexExpr 和 MappingSpec。
- [x] Constant/Binary/Load/Store/If/For/Yield/KernelReturn 强类型 payload；operands/results 仅通过方法派生。
- [x] ConcurrentRoles / PipelineFor / Acquire / Publish / Consume / Release / Drain。
- [x] AsyncCopy 和 Sqmma 返回 EventId；AccumulatorInit/Read 显式建模可变 accumulator。
- [x] NodeMeta 保留 ID/source/origin/revision，移除可写 effects。
- [x] Rust finish_checked 递归验证与 schema v2 round-trip；Python 参数对接 TypeRef。

## 待做

- [ ] Region-local BufferView、dynamic subview 和 target-registered mapping admission。
- [ ] Fragment operation、distributed store、Reduce、Atomic 与 conversion contracts。
- [ ] 完整 operation registry/type inference，不通过通用 attrs 或裸 intrinsic string 绕过 contract。
- [ ] 逻辑 iteration 到 slot/generation 的 target elaboration；目前由结构化 PipelineFor 定义实例域。

## 当前边界

不再使用 Assign/StageFor、无类型 Wait、BufferRef.stage 或可写公共 operands/effects。Event 是不可变 completion 句柄，ticket 是线性 lease，两者不可混用。所支持 protocol 子集及尚未覆盖的部分见 README/P4。
