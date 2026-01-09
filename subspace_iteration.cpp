#include <iostream>
#include <array>
#include <vector>

// Target: AppleM2
using Matrix = std::array<float, 1048576>;
using Subspace = std::array<float, 65536>;
using Tile = std::array<float, 1024>;

namespace hw {
  void LOAD(auto&... args) { std::cout << "LOAD" << std::endl; }
  void STORE(auto&... args) { std::cout << "STORE" << std::endl; }
  void MMUL(auto&... args) { std::cout << "MMUL" << std::endl; }
  void REDUCE(auto&... args) { std::cout << "REDUCE" << std::endl; }
}

void SubspaceStep(Matrix& A, Subspace& V, Subspace& V_new) {
  Tile tA = {};
  Tile tV = {};
  Tile tRes = {};
  for (int i = 0; i < 1024; i += 32) {
    for (int j = 0; j < 64; j += 32) {
      hw::LOAD(tA, A[i]);
      hw::LOAD(tV, V[j]);
      hw::MMUL(tRes, tA, tV);
      hw::STORE(V_new[i], tRes);
    }
  }
}

int main() {
  std::cout << "Starting Kernel execution..." << std::endl;
  Matrix A_dummy = {};
  Subspace V_dummy = {};
  Subspace V_new_dummy = {};
  SubspaceStep(A_dummy, V_dummy, V_new_dummy);
  std::cout << "Kernel execution finished." << std::endl;
  return 0;
}
