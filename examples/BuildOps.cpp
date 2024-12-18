#include "mlir/IR/Builders.h"
#include "mlir/IR/ValueRange.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;

void build_ops() {
  MLIRContext ctx;
  ctx.loadDialect<func::FuncDialect, arith::ArithDialect>();

  OpBuilder builder(&ctx);
  auto mod = builder.create<ModuleOp>(builder.getUnknownLoc());

  builder.setInsertionPointToEnd(mod.getBody());

  auto i32 = builder.getI32Type();
  auto funcType = builder.getFunctionType({i32, i32}, {i32});
  auto func = builder.create<func::FuncOp>(builder.getUnknownLoc(), "test", funcType);

  auto entry = func.addEntryBlock();
  auto args = entry->getArguments();

  builder.setInsertionPointToEnd(entry);

  auto addi = builder.create<arith::AddIOp>(builder.getUnknownLoc(), args[0], args[1]);
  builder.create<func::ReturnOp>(builder.getUnknownLoc(), ValueRange{addi});

  mod->print(llvm::outs());
}
