#ifndef FOO_H
#define FOO_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
// Foo Dialect
//===----------------------------------------------------------------------===//

#include "xir/Dialect/Foo/IR/FooOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Foo Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "xir/Dialect/Foo/IR/FooOps.h.inc"

#endif
