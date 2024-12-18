#ifndef FOO_OPS_TD
#define FOO_OPS_TD

include "FooBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class FooOp<string mnemonic, list<Trait> traits = []> :
  Op<FooDialect, mnemonic, traits>;

def AddOp : FooOp<"add", [Pure]> {
  let summary = "add operation";
  let arguments = (ins Variadic<AnyInteger>:$inputs);
  let results = (outs AnyInteger:$result);
}

def SubOp : FooOp<"sub", [Pure]> {
  let summary = "sub operation";
  let arguments = (ins AnyInteger:$lhs, AnyInteger:$rhs);
  let results = (outs AnyInteger:$result);
}

def ConstantOp : FooOp<"const", [Pure]> {
  let summary = "const operation";
  let arguments = (ins APIntAttr:$value);
  let results = (outs AnyInteger:$result);
}

#endif
