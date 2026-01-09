#include <iostream>
#include <array>
#include <vector>

// Target: AppleM2
using Matrix = std::array<float, 1048576>;
using Tile = std::array<float, 1024>;

namespace hw {
  void LOAD(auto&... args) { std::cout << "LOAD" << std::endl; }
  void STORE(auto&... args) { std::cout << "STORE" << std::endl; }
  void MMUL(auto&... args) { std::cout << "MMUL" << std::endl; }
  void REDUCE(auto&... args) { std::cout << "REDUCE" << std::endl; }
}

void QRStep(Matrix& Q, Matrix& R, Matrix& A_next) {
  Tile tQ = {};
  Tile tR = {};
  Tile tRes = {};
  for (int i = 0; i < 1024; i += 32) {
    hw::LOAD(tQ, Q[i]);
    hw::LOAD(tR, R[i]);
    hw::MMUL(tRes, tR, tQ);
    hw::STORE(A_next[i], tRes);
  }
}

int main() {
  std::cout << "Starting Kernel execution..." << std::endl;
  Matrix Q_dummy = {};
  Matrix R_dummy = {};
  Matrix A_next_dummy = {};
  QRStep(Q_dummy, R_dummy, A_next_dummy);
  std::cout << "Kernel execution finished." << std::endl;
  return 0;
}
