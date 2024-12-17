// ./build/bin/xir-opt -print-def-use ./testdata/def-use.mlir
func.func @test(%a: i32) -> i32 {
  %c10 = arith.constant 10: i32
  %c = arith.addi %a, %c10 : i32

  func.return %c : i32
}
