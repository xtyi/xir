#ifndef XIR_TRANSFORMS_PASSES
#define XIR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"


def PrintIRStructure : Pass<"print-ir-structure"> {
  let summary = "print ir structure";
  let description = [{

  }];
}

def PrintDefUse : Pass<"print-def-use"> {
  let summary = "print def use";
  let description = [{

  }];
}


def TestApplyPatterns: Pass<"test-apply-patterns"> {
  let summary = "test apply patterns";
  let description = [{}];
}

#endif // XIR_TRANSFORMS_PASSES
