set(LLVM_TARGET_DEFINITIONS Passes.td)
# name 影响生成的 register 函数的名字, 由于这些函数被 include 到 xir 命名空间下,
# 所以使用 Transforms 也不会冲突, 否则应该使用 XIRTransforms
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Transforms)
add_public_tablegen_target(XIRTransformsPassIncGen)
