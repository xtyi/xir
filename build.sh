if [ -d "build" ]; then
  rm -rf build
fi
mkdir build
cmake -B ./build -G Ninja \
  -DLLVM_INSTALL_PATH=./externals/llvm-project/build \
  -DXIR_ENABLE_LLD=ON \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++

cmake --build ./build
