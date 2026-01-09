#include <iostream>
#include <Eigen/Dense>

namespace hw {
  template<typename Dst, typename SrcPtr>
  void LOAD(Dst& dst, SrcPtr src_ptr) {
    dst = Eigen::Map<const Dst>(src_ptr);
  }
  template<typename SrcPtr, typename Src>
  void STORE(SrcPtr dst_ptr, const Src& src) {
    Eigen::Map<Src>(dst_ptr) = src;
  }
  template<typename Acc, typename A, typename B>
  void MMUL(Acc& acc, const A& a, const B& b) {
    acc.noalias() += a * b;
  }
  template<typename T>
  void REDUCE(T& t) {
    (void)t.sum();
  }
}

using Matrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Subspace = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Tile = Eigen::Matrix<float, 32, 32, Eigen::RowMajor>;

void SubspaceStep(Matrix& A, Subspace& V, Subspace& V_new) {
  Tile tA; tA.setZero();
  Tile tV; tV.setZero();
  Tile tRes; tRes.setZero();
  for (int i = 0; i < 1024; i += 32) {
    for (int j = 0; j < 64; j += 32) {
      hw::LOAD(tA, A.data() + i);
      hw::LOAD(tV, V.data() + j);
      hw::MMUL(tRes, tA, tV);
      hw::STORE(V_new.data() + i, tRes);
    }
  }
}

int main() {
  Matrix A(1024, 1024); A.setRandom();
  Subspace V(1024, 64); V.setRandom();
  Subspace V_new(1024, 64); V_new.setRandom();
  SubspaceStep(A, V, V_new);
  std::cout << "Kernel execution successful." << std::endl;
  return 0;
}
