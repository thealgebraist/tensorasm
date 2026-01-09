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
    std::cout << "Reduced sum: " << t.sum() << std::endl;
  }
}

// Target: AppleM2
using Matrix = Eigen::Matrix<float, 1024, 1024, Eigen::RowMajor>;
using Vector = Eigen::Vector<float, 1024>;
using Tile = Eigen::Matrix<float, 32, 32, Eigen::RowMajor>;

void PowerIteration(Matrix& A, Vector& v, Vector& out) {
  Tile tA; tA.setZero();
  Tile tv; tv.setZero();
  Tile tRes; tRes.setZero();
  for (int i = 0; i < 1024; i += 32) {
    hw::LOAD(tA, A.data() + i);
    hw::LOAD(tv, v.data() + i);
    hw::MMUL(tRes, tA, tv);
    hw::STORE(out.data() + i, tRes);
  }
}

int main() {
  Matrix A_dummy; A_dummy.setRandom();
  Vector v_dummy; v_dummy.setRandom();
  Vector out_dummy; out_dummy.setRandom();
  PowerIteration(A_dummy, v_dummy, out_dummy);
  return 0;
}
