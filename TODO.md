# TensorASM TODO List
Updated: January 9, 2026 19:52

## 🔴 Critical (Today)
- [ ] Implement hw::MADD in tensorasm.cpp
- [ ] Implement hw::SOFTMAX in tensorasm.cpp  
- [ ] Implement hw::LOOKUP in tensorasm.cpp
- [ ] Add corresponding implementations in tensor_interpreter.cpp

## 🟡 High Priority (This Week)
- [ ] Add memory space tracking to interpreter
- [ ] Generate strided access for Tiled layout
- [ ] Create test file: test_intrinsics.ta
- [ ] Create test file: test_shapes.ta
- [ ] Create test file: test_errors.ta

## 🟢 Normal Priority (Next Week)
- [ ] Add constant folding optimization
- [ ] Implement SYNC properly with barriers
- [ ] Add profiling to interpreter
- [ ] Create VSCode syntax highlighter
- [ ] Write comprehensive README.md

## 📝 Documentation Needed
- [ ] Build instructions with dependencies
- [ ] API reference for hw namespace
- [ ] Tutorial for writing .ta programs
- [ ] Performance tuning guide

## 🧪 Test Cases Needed
- [ ] MADD with different shapes
- [ ] SOFTMAX on vectors and matrices
- [ ] LOOKUP with string tables
- [ ] Tiled layout access patterns
- [ ] Memory space violation detection
- [ ] Shape mismatch error handling
- [ ] Constant reassignment prevention

## 🐛 Known Bugs
- [ ] Warning about Global in MMUL but continues anyway
- [ ] No actual memory space simulation in interpreter
- [ ] Tiled layout parsed but not used in backend
- [ ] Constants can be shadowed by loop variables
