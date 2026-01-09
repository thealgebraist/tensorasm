#include <iostream>
#include <memory>
#include <type_traits>
#include <Eigen/Dense>

namespace hw {
  template<typename T, typename S>
  void ASSIGN(T& dst, const S& src) {
    if constexpr (std::is_arithmetic_v<S> && !std::is_arithmetic_v<T>) {
      dst.setConstant(src);
    } else {
      dst = src;
    }
  }
  template<typename Dst, typename SrcPtr>
  void LOAD(Dst& dst, SrcPtr src_ptr) {
    if constexpr (std::is_arithmetic_v<SrcPtr>) {
      dst.setConstant(src_ptr);
    } else {
      dst = Eigen::Map<const Dst>(src_ptr);
    }
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
    std::cout << t.sum() << std::endl;
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
using ATile = Eigen::Matrix<float, 2, 2, Eigen::RowMajor>;
using BTile = Eigen::Matrix<float, 2, 2, Eigen::RowMajor>;
using A = Eigen::Matrix<float, 2, 2, Eigen::RowMajor>;
using B = Eigen::Matrix<float, 2, 2, Eigen::RowMajor>;
using C = Eigen::Matrix<float, 2, 2, Eigen::RowMajor>;

void product(Acc& acc, A& a, B& b, C& c) {
  ATile at; at.setZero();
  BTile bt; bt.setZero();
  hw::ASSIGN(acc, 0);
  hw::LOAD(at, 1);
  hw::LOAD(bt, 2);
  hw::MMUL(acc, at, bt);
  hw::STORE(c.data(), acc);
  hw::REDUCE(c);
}

int main() {
  Acc acc(2, 2); acc.setRandom();
  A a(2, 2); a.setRandom();
  B b(2, 2); b.setRandom();
  C c(2, 2); c.setRandom();
  product(acc, a, b, c);
  std::cout << "Kernel execution successful." << std::endl;
  return 0;
}
