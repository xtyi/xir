#include "xir/Transforms/Passes.h"

namespace mlir {
namespace xir {
#define GEN_PASS_DEF_PRINTDEFUSE
#include "xir/Transforms/Passes.h.inc"

namespace {
/// A control-flow sink pass.
struct PrintDefUse : public impl::PrintDefUseBase<PrintDefUse> {
  void runOnOperation() override {
    // Recursively traverse the IR nested under the current operation and print
    // every single operation and their operands and users.
    getOperation()->walk([](Operation *op) {
      llvm::outs() << "Visiting op '" << op->getName() << "' with "
                   << op->getNumOperands() << " operands:\n";

      // Print information about the producer of each of the operands.
      for (Value operand : op->getOperands()) {
        if (Operation *producer = operand.getDefiningOp()) {
          llvm::outs() << "  - Operand produced by operation '"
                       << producer->getName() << "'\n";
        } else {
          // If there is no defining op, the Value is necessarily a Block
          // argument.
          auto blockArg = cast<BlockArgument>(operand);
          llvm::outs() << "  - Operand produced by Block argument, number "
                       << blockArg.getArgNumber() << "\n";
        }
      }

      // Print information about the user of each of the result.
      llvm::outs() << "Has " << op->getNumResults() << " results:\n";
      for (const auto &indexedResult : llvm::enumerate(op->getResults())) {
        Value result = indexedResult.value();
        llvm::outs() << "  - Result " << indexedResult.index();
        if (result.use_empty()) {
          llvm::outs() << " has no uses\n";
          continue;
        }
        if (result.hasOneUse()) {
          llvm::outs() << " has a single use: ";
        } else {
          llvm::outs() << " has "
                       << std::distance(result.getUses().begin(),
                                        result.getUses().end())
                       << " uses:\n";
        }
        for (Operation *userOp : result.getUsers()) {
          llvm::outs() << "    - " << userOp->getName() << "\n";
        }
      }
    });
  }
};

} // namespace
} // namespace xir
} // namespace mlir
