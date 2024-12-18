add_mlir_dialect_library(XIRFooDialect
  FooOps.cpp
  FooDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${XIR_SRC_INCLUDE_DIR}/xir/Dialect/Foo

  DEPENDS
  MLIRFooOpsIncGen

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRIR
  )
