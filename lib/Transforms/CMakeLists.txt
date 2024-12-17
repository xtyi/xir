add_mlir_library(XIRTransforms
  PrintIRStructure.cpp
  PrintDefUse.cpp
  TestApplyPatterns.cpp

  DEPENDS
  XIRTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRSupport
  )
