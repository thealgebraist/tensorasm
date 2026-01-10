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

using InputInt = Eigen::Matrix<int32_t, 1, 1>;
using OutputVec = Eigen::Matrix<float, 768, 1>;
using W_TextEmbed = Eigen::Matrix<float, 81, 768, Eigen::RowMajor>;
using T_Hidden = Eigen::Matrix<float, 768, 1>;
using T_Idx = Eigen::Matrix<int32_t, 1, 1>;

void text_encoder_lookup(InputInt& token_idx, W_TextEmbed& embed_w, OutputVec& output) {
  T_Idx idx; idx.setZero();
  hw::LOAD(idx, token_idx.data());
  T_Hidden h; h.setZero();
  hw::LOOKUP(h, embed_w, idx);
  hw::STORE(output.data(), h);
}

int main() {
  auto token_idx_ptr = std::make_unique<InputInt>();
  InputInt& token_idx = *token_idx_ptr;
  hw::FILE_LOAD(token_idx, "weights/token_idx.bin");
  auto embed_w_ptr = std::make_unique<W_TextEmbed>();
  W_TextEmbed& embed_w = *embed_w_ptr;
  hw::FILE_LOAD(embed_w, "weights/embed_w.bin");
  auto output_ptr = std::make_unique<OutputVec>();
  OutputVec& output = *output_ptr;
  hw::FILE_LOAD(output, "weights/output.bin");
  text_encoder_lookup(token_idx, embed_w, output);
  std::cout << "Kernel execution successful." << std::endl;
  return 0;
}
