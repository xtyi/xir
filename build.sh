if [ -d "build" ]; then
  rm -rf build
fi
mkdir build
cmake -B ./build -G Ninja -DLLVM_INSTALL_PATH=./externals/llvm-project/build -DXIR_ENABLE_LLD=ON
cmake --build ./build
