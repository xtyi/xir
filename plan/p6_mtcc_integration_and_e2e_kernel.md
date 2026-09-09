# P6：`mtcc` 集成与端到端 Kernel

状态：待做

## 完成事项

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
