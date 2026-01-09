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
using Tile = Eigen::Matrix<float, 32, 32, Eigen::RowMajor>;

void QRStep(Matrix& Q, Matrix& R, Matrix& A_next) {
  Tile tQ; tQ.setZero();
  Tile tR; tR.setZero();
  Tile tRes; tRes.setZero();
  for (int i = 0; i < 1024; i += 32) {
    hw::LOAD(tQ, Q.data() + i);
    hw::LOAD(tR, R.data() + i);
    hw::MMUL(tRes, tR, tQ);
    hw::STORE(A_next.data() + i, tRes);
  }
}

int main() {
  Matrix Q(1024, 1024); Q.setRandom();
  Matrix R(1024, 1024); R.setRandom();
  Matrix A_next(1024, 1024); A_next.setRandom();
  QRStep(Q, R, A_next);
  std::cout << "Kernel execution successful." << std::endl;
  return 0;
}
