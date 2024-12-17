get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)


set(LIBS
  ${dialect_libs}
  ${conversion_libs}

  MLIROptLib
  XIRTransforms
  )


add_llvm_executable(xir-opt main.cpp)

llvm_update_compile_flags(xir-opt)

target_link_libraries(xir-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(xir-opt)
