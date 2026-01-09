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

using Tile = Eigen::Matrix<float, 16, 16, Eigen::RowMajor>;
using Weight = Eigen::Matrix<float, 16, 16, Eigen::RowMajor>;
using Input = Eigen::Matrix<float, 16, 16, Eigen::RowMajor>;
using Output = Eigen::Matrix<float, 16, 16, Eigen::RowMajor>;

void matmul(Tile& acc, Weight& W, Input& x, Output& y) {
  hw::MMUL(acc, W, x);
  hw::STORE(y.data(), acc);
}

int main() {
  Tile acc(16, 16); acc.setRandom();
  Weight W(16, 16); W.setRandom();
  Input x(16, 16); x.setRandom();
  Output y(16, 16); y.setRandom();
  matmul(acc, W, x, y);
  std::cout << "Kernel execution successful." << std::endl;
  return 0;
}
