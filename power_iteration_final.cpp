#include <iostream>
#include <array>
#include <vector>

// Target: AppleM2
using Matrix = std::array<float, 1048576>;
using Vector = std::array<float, 1024>;
using Tile = std::array<float, 1024>;

namespace hw {
  void LOAD(auto&... args) { std::cout << "LOAD" << std::endl; }
  void STORE(auto&... args) { std::cout << "STORE" << std::endl; }
  void MMUL(auto&... args) { std::cout << "MMUL" << std::endl; }
  void REDUCE(auto&... args) { std::cout << "REDUCE" << std::endl; }
}

void PowerIteration(Matrix& A, Vector& v, Vector& out) {
  Tile tA = {};
  Tile tv = {};
  Tile tRes = {};
  for (int i = 0; i < 1024; i += 32) {
    hw::LOAD(tA, A[i]);
    hw::LOAD(tv, v[i]);
    hw::MMUL(tRes, tA, tv);
    hw::STORE(out[i], tRes);
  }
}

int main() {
  std::cout << "Starting Kernel execution..." << std::endl;
  Matrix A_dummy = {};
  Vector v_dummy = {};
  Vector out_dummy = {};
  PowerIteration(A_dummy, v_dummy, out_dummy);
  std::cout << "Kernel execution finished." << std::endl;
  return 0;
}
