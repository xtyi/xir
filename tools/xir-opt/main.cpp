#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "xir/Transforms/Passes.h"

using namespace mlir;

int main(int argc, char **argv) {
  registerAllPasses();

  DialectRegistry registry;
  registerAllDialects(registry);
  xir::registerTransformsPasses();

  return failed(
      MlirOptMain(argc, argv, "StableHLO optimizer driver\n", registry));
}
