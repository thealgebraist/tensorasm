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
  template<typename Dst, typename Src, typename Idx>
  void LOOKUP(Dst& dst, const Src& src, const Idx& idx) {
    if constexpr (std::is_arithmetic_v<Idx>) {
      dst = src.row(idx);
    } else {
      int i = idx(0);
      dst = src.row(i);
    }
  }
  template<typename T>
  void FILE_LOAD(T& t, const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (f) {
      f.read(reinterpret_cast<char*>(t.data()), t.size() * sizeof(typename T::Scalar));
      std::cout << "Loaded " << path << " (" << t.size() << " elements)" << std::endl;
    } else {
      std::cerr << "Warning: Could not load " << path << std::endl;
    }
  }
  template<typename T>
  void FILE_SAVE(const T& t, const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (f) {
      f.write(reinterpret_cast<const char*>(t.data()), t.size() * sizeof(typename T::Scalar));
      std::cout << "Saved " << path << " (" << t.size() << " elements)" << std::endl;
    } else {
      std::cerr << "Warning: Could not save " << path << std::endl;
    }
  }
}

using InputInt = Eigen::Matrix<int32_t, 1, 1>;
using OutputVec = Eigen::Matrix<float, 768, 1>;
using W_TextEmbed = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using T_Hidden = Eigen::Matrix<float, 768, 1>;
using T_Idx = Eigen::Matrix<int32_t, 1, 1>;

void text_encoder_lookup(InputInt& token_idx, W_TextEmbed& embed_w, OutputVec& output) {
  T_Idx idx; idx.setZero();
  hw::LOAD(idx, token_idx.data());
  std::cout << "Token index: " << idx(0) << std::endl;
  
  T_Hidden h; h.setZero();
  hw::LOOKUP(h, embed_w, idx);
  std::cout << "Embedding lookup: first 5 values = [" 
            << h(0) << ", " << h(1) << ", " << h(2) << ", " 
            << h(3) << ", " << h(4) << "]" << std::endl;
  
  hw::STORE(output.data(), h);
}

int main() {
  auto token_idx_ptr = std::make_unique<InputInt>();
  InputInt& token_idx = *token_idx_ptr;
  hw::FILE_LOAD(token_idx, "weights/token_idx.bin");
  
  auto embed_w_ptr = std::make_unique<W_TextEmbed>();
  W_TextEmbed& embed_w = *embed_w_ptr;
  embed_w.resize(81, 768);
  hw::FILE_LOAD(embed_w, "weights/embed_w.bin");
  
  auto output_ptr = std::make_unique<OutputVec>();
  OutputVec& output = *output_ptr;
  output.setZero();
  
  text_encoder_lookup(token_idx, embed_w, output);
  
  hw::FILE_SAVE(output, "weights/output.bin");
  
  std::cout << "Kernel execution successful." << std::endl;
  return 0;
}
