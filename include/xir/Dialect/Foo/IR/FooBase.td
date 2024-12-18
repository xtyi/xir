#ifndef FOO_BASE_TD
#define FOO_BASE_TD

include "mlir/IR/OpBase.td"

def FooDialect : Dialect {
  let name = "foo";
  let cppNamespace = "::mlir::foo";
  let summary = "Foo Dialect";
}

#endif
