func.func @foo1() -> i32 {
  %c10 = arith.constant 10 : i32
  func.return %c10 : i32
}

func.func @foo2() -> i32 {
  %c20 = arith.constant 20 : i32
  func.return %c20 : i32
}
