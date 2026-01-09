#include <iostream>
#include <memory>
#include <Eigen/Dense>

namespace hw {
  template<typename Dst, typename SrcPtr>
  void LOAD(Dst& dst, SrcPtr src_ptr) {
    dst = Eigen::Map<const Dst>(src_ptr);
  }
  template<typename SrcPtr, typename Src>
  void STORE(SrcPtr dst_ptr, const Src& src) {
    Eigen::Map<Src> m(dst_ptr); m = src;
  }
  template<typename Acc, typename A, typename B>
  void MMUL(Acc& acc, const A& a, const B& b) {
    acc.noalias() += a * b;
  }
  template<typename Acc, typename A, typename B, typename C>
  void MADD(Acc& acc, const A& a, const B& b, const C& c) {
    acc.noalias() += a * b + c;
  }
  template<typename T>
  void REDUCE(T& t) {
    (void)t.sum();
  }
  template<typename T>
  void SOFTMAX(T& t) {
    t = t.array().exp() / t.array().exp().sum();
  }
  template<typename T, typename Table, typename Idx>
  void LOOKUP(T& t, const Table& table, const Idx& idx) {
    t = table.row(idx);
  }
  void SYNC(const std::string& name) {
    // Barrier for name
  }
}

using Acc = Eigen::Matrix<float, 2, 2, Eigen::RowMajor>;
using A = Eigen::Matrix<float, 2, 2, Eigen::RowMajor>;
using B = Eigen::Matrix<float, 2, 2, Eigen::RowMajor>;
using C = Eigen::Matrix<float, 2, 2, Eigen::RowMajor>;
using Out = Eigen::Matrix<float, 2, 2, Eigen::RowMajor>;

void test_madd(Acc& acc, A& a, B& b, C& c, Out& out) {
  acc = 0;
  hw::MADD(acc, a, b, c);
  hw::STORE(out.data(), acc);
}

int main() {
  Acc acc(2, 2); acc.setRandom();
  A a(2, 2); a.setRandom();
  B b(2, 2); b.setRandom();
  C c(2, 2); c.setRandom();
  Out out(2, 2); out.setRandom();
  test_madd(acc, a, b, c, out);
  std::cout << "Kernel execution successful." << std::endl;
  return 0;
}
