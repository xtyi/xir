#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;




void dense_set_demo() {

  // DenseSet 支持列表初始化
  DenseSet<int> ds1{10};
  ds1.insert(100);
  outs() << "ds1:" << "\n";
  for (auto& elem: ds1) {
    outs() << elem << "\n";
  }

  // 调用构造函数，参数表示预留空间
  DenseSet<int> ds2(10);
  // insert 方法的返回值是一个 std::pair, first 指向插入位置的迭代器，second 是一个布尔值，表示插入是否成功
  auto [it, inserted] = ds2.insert(100);
  if (inserted) {
    outs() << "ds2:" << "\n";
    for (auto& elem: ds2) {
      outs() << elem << "\n";
    }
    outs() << "iterator: " << *it << "\n";
  }


}
