#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;

int symbol_table_demo(int argc, char** argv) {
  mlir::MLIRContext ctx;
  ctx.loadDialect<func::FuncDialect, arith::ArithDialect>();
  auto module = parseSourceFile<ModuleOp>(argv[1], &ctx);

  mlir::SymbolTable symbolTable(module.get());
  auto foo = symbolTable.lookup("foo2");
  foo->dump();

  return 0;
}
