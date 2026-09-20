# P6：nvcc / mtcc 集成与端到端 Kernel

状态：进行中

## CUDA 最小路径已完成

- [x] `compile_cuda()` 调用 nvcc，保存 IR/CUDA/source mapping/编译日志与 shared library。
- [x] `CompiledCudaKernel.run()` 通过 ctypes 执行同步 host-array ABI。
- [x] RTX 5060 Laptop SM120 + nvcc 13.1.80 上，Add 24 组边界/尾部测试及 Sub 回归通过。
- [x] 编译失败、非法参数、CUDA runtime errors、numerical mismatch 分别报告。
- [ ] Compute Sanitizer：本机 WDDM debugger interface 初始化失败，不计为通过。

详见 [CUDA 闭环与验证记录](../docs/cuda_elementwise_path.md)。尚未做 benchmark 或设备 tensor/stream API。

## MUSA 路径待完成

- [ ] 接入 `mtcc` 命令行或编译 API。
- [ ] 支持编译日志、错误位置和生成文件保留。
- [ ] 完成一个 MUSA device correctness oracle。
- [ ] 完成最小 benchmark 和结果记录格式。

## 待做

- [ ] 区分 compiler failure、runtime failure、numerical mismatch 和 timeout。
- [ ] 记录 `mtcc` 生成代码/汇编可用性，不把 DSL 静态检查当作机器码证明。
- [ ] 建立目标架构 capability matrix 的回归样例。

## 完成标准

至少一个 kernel 从 Python DSL 经过 MUSA C、`mtcc` 编译，在目标设备上完成正确性验证；所有不能执行的步骤明确标注未验证。
