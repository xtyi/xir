#ifndef XIR_TRANSFORMS_PASSES_H
#define XIR_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

namespace mlir {

namespace xir {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "xir/Transforms/Passes.h.inc"

} // namespace xir

} // namespace mlir

#endif // XIR_TRANSFORMS_PASSES_H
