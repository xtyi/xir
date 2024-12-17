func.func @test(%a: i32) -> i32 {
  %c10_0 = arith.constant 10: i32
  %c10_1 = arith.constant 10 : i32

  func.return %a : i32
}
