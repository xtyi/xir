#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/Support/raw_ostream.h"

using llvm::EquivalenceClasses;

void equivalence_classes_demo() {
  EquivalenceClasses<int> EC;
  EC.insert(1);
  EC.insert(2);
  EC.insert(3);
  EC.insert(4);

  llvm::outs() << "before union" << EC.isEquivalent(1, 2) << "\n";
  for (auto i : EC) {
    llvm::outs() << i.getData() << "\n";
  }

  EC.unionSets(1, 2);
  llvm::outs() << "after union" << EC.isEquivalent(1, 2) << "\n";

  llvm::outs() << *EC.findLeader(1) << "\n";
  llvm::outs() << *EC.findLeader(2) << "\n";
  llvm::outs() << *EC.findLeader(3) << "\n";
  llvm::outs() << *EC.findLeader(4) << "\n";

  llvm::outs() << EC.findValue(1)->getData() << "\n";
  llvm::outs() << EC.findValue(2)->getData() << "\n";
  llvm::outs() << EC.findValue(3)->getData() << "\n";
  llvm::outs() << EC.findValue(4)->getData() << "\n";
}
