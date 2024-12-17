#include "xir/Transforms/Passes.h"

namespace mlir {
namespace xir {
#define GEN_PASS_DEF_TESTAPPLYPATTERNS
#include "xir/Transforms/Passes.h.inc"

namespace {
/// A control-flow sink pass.
struct TestApplyPatterns : public impl::TestApplyPatternsBase<TestApplyPatterns> {
  void runOnOperation() override {}
};


} // namespace
} // namespace xir
} // namespace mlir
