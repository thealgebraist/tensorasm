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
using Vector = Eigen::Vector<float, Eigen::Dynamic>;
using Tile = Eigen::Matrix<float, 32, 32, Eigen::RowMajor>;

void RayleighQuotient(Matrix& A, Vector& v) {
  Tile tA; tA.setZero();
  Tile tv; tv.setZero();
  Tile tAcc; tAcc.setZero();
  for (int i = 0; i < 1024; i += 32) {
    hw::LOAD(tA, A.data() + i);
    hw::LOAD(tv, v.data() + i);
    hw::MMUL(tAcc, tv, tA);
    hw::MMUL(tAcc, tAcc, tv);
    hw::REDUCE(tAcc);
  }
}

int main() {
  Matrix A(1024, 1024); A.setRandom();
  Vector v(1024); v.setRandom();
  RayleighQuotient(A, v);
  std::cout << "Kernel execution successful." << std::endl;
  return 0;
}
