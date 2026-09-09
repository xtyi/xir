# P5：MUSA C Emitter

状态：待做

## 完成事项

- [ ] 生成 kernel 签名、参数、shared/local/accumulator 声明。
- [ ] 生成 role 和 stage 控制流。
- [ ] 生成 view offset、索引、对齐和地址空间转换。
- [ ] 生成 MUSA async copy、barrier、wait、fence 和 SQMMA 调用。
- [ ] 保留 node_id 注释或 debug mapping，便于定位生成代码。

## 待做

- [ ] 选择 MUSA intrinsic 与 inline asm 的边界。
- [ ] 处理不同 MP 架构的 capability-specific spelling。
- [ ] 对生成 C 做语法检查和稳定 golden diff。

## 完成标准

最小 GEMM 和 pipeline kernel 能生成可读、确定性的 MUSA C；无 MUSA 工具链时只报告静态/生成代码验证结果。
