#include "mlir/Parser/Parser.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"


using namespace mlir;

void parse_and_print(int argc, char** argv) {
  MLIRContext ctx;
  ctx.loadDialect<func::FuncDialect, arith::ArithDialect>();
  auto src = parseSourceFile<ModuleOp>(argv[1], &ctx);

  src->print(llvm::outs());

  src->dump();
}
