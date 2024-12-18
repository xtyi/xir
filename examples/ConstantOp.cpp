#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

using namespace mlir;

void build_arith_constant_ops() {
  MLIRContext ctx;
  ctx.loadDialect<arith::ArithDialect>();

  OpBuilder builder(&ctx);

  int64_t constantValue = 42;
  auto intType = builder.getIntegerType(64);
  auto constantAttr = builder.getIntegerAttr(intType, constantValue);
  auto loc = builder.getUnknownLoc();

  //   auto constantOp = builder.create<arith::ConstantOp>(loc, intType, constantAttr);
  auto constantOp = builder.create<arith::ConstantOp>(loc, constantAttr);

  constantOp.dump();
}
