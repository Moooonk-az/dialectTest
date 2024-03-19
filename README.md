# dialectTest
ai compiler test

### preparation
```
pip3 install pybind11 Numpy
```

### clone code and init submodule llvm

```
git clone https://github.com/Moooonk-az/dialectTest.git
cd dialectTest
git submodule update --init
```

### build llvm first

```
cd dailectTest/llvm
mkidr build && cd build

# option -DMLIR_ENABLE_BINDINGS_PYTHON=ON and -DPython3_EXECUTABLE=$(which python3) is not neccessary
cmake -G Ninja ../llvm
-DLLVM_ENABLE_PROJECTS="mlir;clang"
-DLLVM_TARGETS_TO_BUILD="host"
-DLLVM_ENABLE_ASSERTIONS=ON
-DCMAKE_BUILD_TYPE=DEBUG
-DMLIR_ENABLE_BINDINGS_PYTHON=ON
-DPython3_EXECUTABLE=$(which python3)
```
### build dialectTest
```
cd dialectTest && mkdir build
```
