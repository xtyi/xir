# get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
# get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
  MLIRIR
  MLIRParser
  MLIRFuncDialect
  MLIRArithDialect
  # ${dialect_libs}
  # ${conversion_libs}
)

add_llvm_executable(demo demo.cpp
  AttrType.cpp
  ParseAndPrint.cpp
  BuildOps.cpp
  SymbolTableDemo.cpp
  EquivalenceClassesDemo.cpp
  ConstantOp.cpp
  LLVMContainerDemo.cpp)

llvm_update_compile_flags(demo)

target_link_libraries(demo PRIVATE ${LIBS})

mlir_check_all_link_libraries(demo)
