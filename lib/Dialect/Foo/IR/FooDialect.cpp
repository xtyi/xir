#include "xir/Dialect/Foo/IR/Foo.h"

#include "xir/Dialect/Foo/IR/FooOpsDialect.cpp.inc"



void mlir::foo::FooDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "xir/Dialect/Foo/IR/FooOps.cpp.inc"
      >();
}
