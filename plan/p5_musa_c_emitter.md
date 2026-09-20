# P5：CUDA / MUSA Emitter

状态：进行中

## CUDA 最小路径已完成

- [x] LaunchIndex、Constant、Binary、masked Load/Store、KernelReturn 的逐节点 CUDA lowering。
- [x] 生成 source mapping、CUDA kernel 与同步 host wrapper；unsupported nodes 在 emission 前拒绝。
- [x] Wrapper 校验 device/extent/grid/block，处理 n=0，独立分配显存并检查 CUDA errors。
- [x] Add/Sub 两种 DSL 表达式生成不同代码并实际执行；不是固定 Add 模板。

## MUSA 路径待完成

- [ ] 生成 kernel 签名、参数、shared/local/accumulator 声明。
- [ ] 生成 ConcurrentRoles 与 logical iteration/slot/generation 控制流。
- [ ] 生成 view offset、索引、对齐和地址空间转换。
- [ ] 按已验证 adapter 生成 MUSA transfer、completion、barrier/fence 和 SQMMA 调用。
- [ ] 保留 node_id 注释或 debug mapping，便于定位生成代码。

## 待做

- [ ] 仅接受通过当前 revision target verification 的 IR，不以 construction Pass 代替。
- [ ] 验证 ready 不早于 transfer completion，empty 不早于最后一次输入读取完成。

- [ ] 选择 MUSA intrinsic 与 inline asm 的边界。
- [ ] 处理不同 MP 架构的 capability-specific spelling。
- [ ] 对生成 C 做语法检查和稳定 golden diff。

## 完成标准

CUDA FP32 elementwise 已达成最小生成闭环。MUSA 完成标准仍为：最小 GEMM 和 pipeline kernel 能生成可读、确定性的 MUSA C；无 MUSA 工具链时只报告静态/生成代码验证结果。
