#define EIGEN_STACK_ALLOCATION_LIMIT 0
#include <iostream>
#include <memory>
#include <type_traits>
#include <cstdint>
#include <fstream>
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
    if constexpr (std::is_arithmetic_v<Src>) {
       *dst_ptr = src;
    } else {
      Eigen::Map<Src> m(dst_ptr); m = src;
    }
  }
  template<typename Acc, typename A, typename B>
  void MMUL(Acc& acc, const A& a, const B& b) {
    acc.noalias() += a * b;
  }
  template<typename Acc, typename A, typename B, typename C>
  void MADD(Acc& acc, const A& a, const B& b, const C& c) {
    acc.noalias() += a * b + c;
  }
  template<typename Dst, typename Src>
  void REDUCE(Dst& dst, const Src& src) {
    if constexpr (std::is_scalar_v<std::decay_t<Dst>>) {
      dst = src.sum();
    } else {
      dst.setConstant(src.sum());
    }
  }
  template<typename T>
  void SOFTMAX(T& t) {
    t = t.array().exp() / t.array().exp().sum();
  }
  template<typename T, typename Table, typename Idx>
  void LOOKUP(T& t, const Table& table, const Idx& idx) {
    if constexpr (std::is_arithmetic_v<Idx>) {
      t = table.row(idx);
    } else {
      t = table.row(static_cast<int>(idx.data()[0]));
    }
  }
  template<typename T>
  void EXP(T& t) { t = t.array().exp().matrix(); }
  template<typename T>
  void SQRT(T& t) { t = t.array().sqrt().matrix(); }
  template<typename Dst, typename Src>
  void TRANSPOSE(Dst& dst, const Src& src) { dst = src.transpose(); }
  template<typename T>
  void ACT(T& t, int type) {
    if (type == 1) { // GELU
      auto a = t.array();
      t = (0.5f * a * (1.0f + (0.7978845608f * (a + 0.044715f * a.pow(3))).tanh())).matrix();
    } else if (type == 2) { // SILU
      auto a = t.array();
      t = (a / (1.0f + (-a).exp())).matrix();
    }
  }
  template<typename T>
  void FILE_LOAD(T& t, const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (f) {
      f.read(reinterpret_cast<char*>(t.data()), t.size() * sizeof(typename T::Scalar));
    } else {
      std::cerr << "Warning: Could not load " << path << ", using random/zero values." << std::endl;
    }
  }
  void SYNC(const std::string& name) {
    // Barrier for name
  }
}

using Matrix4x4 = Eigen::Matrix<float, 4, 4, Eigen::RowMajor>;
using Vector4 = Eigen::Matrix<float, 4, 1>;
using MatTile = Eigen::Matrix<float, 4, 4, Eigen::RowMajor>;
using VecTile = Eigen::Matrix<float, 4, 1>;

void PowerIteration(Matrix4x4& A, Vector4& v, Vector4& out) {
  MatTile tA; tA.setZero();
  VecTile tv; tv.setZero();
  VecTile tRes; tRes.setZero();
  hw::LOAD(tA, A.data());
  hw::LOAD(tv, v.data());
  hw::ASSIGN(tRes, 0);
  hw::MMUL(tRes, tA, tv);
  hw::STORE(out.data(), tRes);
}

int main() {
  auto A_ptr = std::make_unique<Matrix4x4>();
  Matrix4x4& A = *A_ptr;
  hw::FILE_LOAD(A, "weights/A.bin");
  auto v_ptr = std::make_unique<Vector4>();
  Vector4& v = *v_ptr;
  hw::FILE_LOAD(v, "weights/v.bin");
  auto out_ptr = std::make_unique<Vector4>();
  Vector4& out = *out_ptr;
  hw::FILE_LOAD(out, "weights/out.bin");
  PowerIteration(A, v, out);
  std::cout << "Kernel execution successful." << std::endl;
  return 0;
}
