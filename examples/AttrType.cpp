#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"

using namespace mlir;

void attr_type_demo() {
  MLIRContext ctx{};
  Builder builder{&ctx};

  // 这里不需要保存额外的类型信息了, 所以用 NoneType?
  auto strAttr = builder.getStringAttr("hello world");
  if (strAttr.getType().isa<NoneType>()) {
    llvm::outs() << "StringAttr has a NoneType\n";
  } else {
    llvm::outs() << "other\n";
  }

  // 这里 IntegerAttr 使用 APInt 表示值, 确实需要使用一个 Type 表示类型信息, 多少bit, 有无符号
  auto intAttr = builder.getI32IntegerAttr(10);
  if (auto intTy = intAttr.getType().dyn_cast<IntegerType>()) {
    llvm::outs() << "IntegerAttr has a IntegerType\n";
    llvm::outs() << intTy.getWidth() << " " << intTy.getSignedness() << "\n";
  } else {
    llvm::outs() << "other\n";
  }

}
