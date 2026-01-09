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

using InputMat = Eigen::Matrix<float, 1, 1, Eigen::RowMajor>;
using HiddenVec = Eigen::Matrix<float, 1, 16, Eigen::RowMajor>;
using OutputVec = Eigen::Matrix<float, 1, 16, Eigen::RowMajor>;
using W1Matrix = Eigen::Matrix<float, 1, 16, Eigen::RowMajor>;
using W2Matrix = Eigen::Matrix<float, 16, 16, Eigen::RowMajor>;
using HiddenVecT = Eigen::Matrix<float, 16, 1, Eigen::RowMajor>;
using InputMatT = Eigen::Matrix<float, 1, 1, Eigen::RowMajor>;
using InputTile = Eigen::Matrix<float, 1, 1, Eigen::RowMajor>;
using HiddenTile = Eigen::Matrix<float, 1, 16, Eigen::RowMajor>;
using OutputTile = Eigen::Matrix<float, 1, 16, Eigen::RowMajor>;
using W1Tile = Eigen::Matrix<float, 1, 16, Eigen::RowMajor>;
using W2Tile = Eigen::Matrix<float, 16, 16, Eigen::RowMajor>;
using HiddenTileT = Eigen::Matrix<float, 16, 1, Eigen::RowMajor>;
using InputTileT = Eigen::Matrix<float, 1, 1, Eigen::RowMajor>;

void ForwardPass(InputMat& x, W1Matrix& W1, HiddenVec& b1, W2Matrix& W2, OutputVec& b2, HiddenVec& z1_out, OutputVec& out) {
  InputTile tx; tx.setZero();
  W1Tile tW1; tW1.setZero();
  HiddenTile tb1; tb1.setZero();
  HiddenTile tz1; tz1.setZero();
  W2Tile tW2; tW2.setZero();
  OutputTile tb2; tb2.setZero();
  OutputTile tout; tout.setZero();
  hw::LOAD(tx, x.data());
  hw::LOAD(tW1, W1.data());
  hw::LOAD(tb1, b1.data());
  hw::ASSIGN(tz1, 0);
  hw::MMUL(tz1, tx, tW1);
  hw::ASSIGN(tz1, (tz1 + tb1));
  hw::STORE(z1_out.data(), tz1);
  hw::LOAD(tW2, W2.data());
  hw::LOAD(tb2, b2.data());
  hw::ASSIGN(tout, 0);
  hw::MMUL(tout, tz1, tW2);
  hw::ASSIGN(tout, (tout + tb2));
  hw::STORE(out.data(), tout);
}

void ComputeOutputGradient(OutputVec& pred, OutputVec& y_true, OutputVec& grad_out) {
  OutputTile tpred; tpred.setZero();
  OutputTile ttrue; ttrue.setZero();
  OutputTile tgrad; tgrad.setZero();
  hw::LOAD(tpred, pred.data());
  hw::LOAD(ttrue, y_true.data());
  hw::ASSIGN(tgrad, (tpred - ttrue));
  hw::STORE(grad_out.data(), tgrad);
}

void BackwardHiddenLayer(OutputVec& grad_output, W2Matrix& W2, HiddenVec& z1, HiddenVec& grad_hidden) {
  OutputTile tgrad_out; tgrad_out.setZero();
  W2Tile tW2; tW2.setZero();
  W2Tile tW2T; tW2T.setZero();
  HiddenTile tz1; tz1.setZero();
  HiddenTile tgrad_h; tgrad_h.setZero();
  HiddenTile tmask; tmask.setZero();
  hw::LOAD(tgrad_out, grad_output.data());
  hw::LOAD(tW2, W2.data());
  hw::TRANSPOSE(tW2T, tW2);
  hw::ASSIGN(tgrad_h, 0);
  hw::MMUL(tgrad_h, tgrad_out, tW2T);
  hw::LOAD(tz1, z1.data());
  hw::ASSIGN(tmask, tz1);
  hw::ASSIGN(tgrad_h, (tgrad_h * tmask));
  hw::STORE(grad_hidden.data(), tgrad_h);
}

void UpdateW2(W2Matrix& W2, HiddenVec& z1, OutputVec& grad_output, W2Matrix& W2_out, HiddenVecT& z1_T_buf) {
  HiddenTile tz1; tz1.setZero();
  HiddenTileT tz1T; tz1T.setZero();
  OutputTile tgrad; tgrad.setZero();
  W2Tile tdW2; tdW2.setZero();
  W2Tile tW2; tW2.setZero();
  hw::LOAD(tz1, z1.data());
  hw::TRANSPOSE(tz1T, tz1);
  hw::LOAD(tgrad, grad_output.data());
  hw::ASSIGN(tdW2, 0);
  hw::MMUL(tdW2, tz1T, tgrad);
  hw::LOAD(tW2, W2.data());
  hw::ASSIGN(tW2, (tW2 - tdW2));
  hw::STORE(W2_out.data(), tW2);
}

void UpdateW1(W1Matrix& W1, InputMat& x, HiddenVec& grad_hidden, W1Matrix& W1_out, InputMatT& x_T_buf) {
  InputTile tx; tx.setZero();
  InputTileT txT; txT.setZero();
  HiddenTile tgrad_h; tgrad_h.setZero();
  W1Tile tdW1; tdW1.setZero();
  W1Tile tW1; tW1.setZero();
  hw::LOAD(tx, x.data());
  hw::TRANSPOSE(txT, tx);
  hw::LOAD(tgrad_h, grad_hidden.data());
  hw::ASSIGN(tdW1, 0);
  hw::MMUL(tdW1, txT, tgrad_h);
  hw::LOAD(tW1, W1.data());
  hw::ASSIGN(tW1, (tW1 - tdW1));
  hw::STORE(W1_out.data(), tW1);
}

void UpdateBias(OutputVec& bias, OutputVec& gradient, OutputVec& bias_out) {
  OutputTile tbias; tbias.setZero();
  OutputTile tgrad; tgrad.setZero();
  hw::LOAD(tbias, bias.data());
  hw::LOAD(tgrad, gradient.data());
  hw::ASSIGN(tbias, (tbias - tgrad));
  hw::STORE(bias_out.data(), tbias);
}

void LoadWeights(W1Matrix& W1_file, HiddenVec& b1_file, W2Matrix& W2_file, OutputVec& b2_file, W1Matrix& W1, HiddenVec& b1, W2Matrix& W2, OutputVec& b2) {
  W1Tile tW1; tW1.setZero();
  HiddenTile tb1; tb1.setZero();
  W2Tile tW2; tW2.setZero();
  OutputTile tb2; tb2.setZero();
  hw::LOAD(tW1, W1_file.data());
  hw::LOAD(tb1, b1_file.data());
  hw::LOAD(tW2, W2_file.data());
  hw::LOAD(tb2, b2_file.data());
  hw::STORE(W1.data(), tW1);
  hw::STORE(b1.data(), tb1);
  hw::STORE(W2.data(), tW2);
  hw::STORE(b2.data(), tb2);
}

void SaveWeights(W1Matrix& W1, HiddenVec& b1, W2Matrix& W2, OutputVec& b2, W1Matrix& W1_file, HiddenVec& b1_file, W2Matrix& W2_file, OutputVec& b2_file) {
  W1Tile tW1; tW1.setZero();
  HiddenTile tb1; tb1.setZero();
  W2Tile tW2; tW2.setZero();
  OutputTile tb2; tb2.setZero();
  hw::LOAD(tW1, W1.data());
  hw::LOAD(tb1, b1.data());
  hw::LOAD(tW2, W2.data());
  hw::LOAD(tb2, b2.data());
  hw::STORE(W1_file.data(), tW1);
  hw::STORE(b1_file.data(), tb1);
  hw::STORE(W2_file.data(), tW2);
  hw::STORE(b2_file.data(), tb2);
}

int main() {
  auto x_ptr = std::make_unique<InputMat>();
  InputMat& x = *x_ptr;
  hw::FILE_LOAD(x, "weights/x.bin");
  auto W1_ptr = std::make_unique<W1Matrix>();
  W1Matrix& W1 = *W1_ptr;
  hw::FILE_LOAD(W1, "weights/W1.bin");
  auto b1_ptr = std::make_unique<HiddenVec>();
  HiddenVec& b1 = *b1_ptr;
  hw::FILE_LOAD(b1, "weights/b1.bin");
  auto W2_ptr = std::make_unique<W2Matrix>();
  W2Matrix& W2 = *W2_ptr;
  hw::FILE_LOAD(W2, "weights/W2.bin");
  auto b2_ptr = std::make_unique<OutputVec>();
  OutputVec& b2 = *b2_ptr;
  hw::FILE_LOAD(b2, "weights/b2.bin");
  auto z1_out_ptr = std::make_unique<HiddenVec>();
  HiddenVec& z1_out = *z1_out_ptr;
  hw::FILE_LOAD(z1_out, "weights/z1_out.bin");
  auto out_ptr = std::make_unique<OutputVec>();
  OutputVec& out = *out_ptr;
  hw::FILE_LOAD(out, "weights/out.bin");
  ForwardPass(x, W1, b1, W2, b2, z1_out, out);
  auto pred_ptr = std::make_unique<OutputVec>();
  OutputVec& pred = *pred_ptr;
  hw::FILE_LOAD(pred, "weights/pred.bin");
  auto y_true_ptr = std::make_unique<OutputVec>();
  OutputVec& y_true = *y_true_ptr;
  hw::FILE_LOAD(y_true, "weights/y_true.bin");
  auto grad_out_ptr = std::make_unique<OutputVec>();
  OutputVec& grad_out = *grad_out_ptr;
  hw::FILE_LOAD(grad_out, "weights/grad_out.bin");
  ComputeOutputGradient(pred, y_true, grad_out);
  auto grad_output_ptr = std::make_unique<OutputVec>();
  OutputVec& grad_output = *grad_output_ptr;
  hw::FILE_LOAD(grad_output, "weights/grad_output.bin");
  auto W2_ptr = std::make_unique<W2Matrix>();
  W2Matrix& W2 = *W2_ptr;
  hw::FILE_LOAD(W2, "weights/W2.bin");
  auto z1_ptr = std::make_unique<HiddenVec>();
  HiddenVec& z1 = *z1_ptr;
  hw::FILE_LOAD(z1, "weights/z1.bin");
  auto grad_hidden_ptr = std::make_unique<HiddenVec>();
  HiddenVec& grad_hidden = *grad_hidden_ptr;
  hw::FILE_LOAD(grad_hidden, "weights/grad_hidden.bin");
  BackwardHiddenLayer(grad_output, W2, z1, grad_hidden);
  auto W2_ptr = std::make_unique<W2Matrix>();
  W2Matrix& W2 = *W2_ptr;
  hw::FILE_LOAD(W2, "weights/W2.bin");
  auto z1_ptr = std::make_unique<HiddenVec>();
  HiddenVec& z1 = *z1_ptr;
  hw::FILE_LOAD(z1, "weights/z1.bin");
  auto grad_output_ptr = std::make_unique<OutputVec>();
  OutputVec& grad_output = *grad_output_ptr;
  hw::FILE_LOAD(grad_output, "weights/grad_output.bin");
  auto W2_out_ptr = std::make_unique<W2Matrix>();
  W2Matrix& W2_out = *W2_out_ptr;
  hw::FILE_LOAD(W2_out, "weights/W2_out.bin");
  auto z1_T_buf_ptr = std::make_unique<HiddenVecT>();
  HiddenVecT& z1_T_buf = *z1_T_buf_ptr;
  hw::FILE_LOAD(z1_T_buf, "weights/z1_T_buf.bin");
  UpdateW2(W2, z1, grad_output, W2_out, z1_T_buf);
  auto W1_ptr = std::make_unique<W1Matrix>();
  W1Matrix& W1 = *W1_ptr;
  hw::FILE_LOAD(W1, "weights/W1.bin");
  auto x_ptr = std::make_unique<InputMat>();
  InputMat& x = *x_ptr;
  hw::FILE_LOAD(x, "weights/x.bin");
  auto grad_hidden_ptr = std::make_unique<HiddenVec>();
  HiddenVec& grad_hidden = *grad_hidden_ptr;
  hw::FILE_LOAD(grad_hidden, "weights/grad_hidden.bin");
  auto W1_out_ptr = std::make_unique<W1Matrix>();
  W1Matrix& W1_out = *W1_out_ptr;
  hw::FILE_LOAD(W1_out, "weights/W1_out.bin");
  auto x_T_buf_ptr = std::make_unique<InputMatT>();
  InputMatT& x_T_buf = *x_T_buf_ptr;
  hw::FILE_LOAD(x_T_buf, "weights/x_T_buf.bin");
  UpdateW1(W1, x, grad_hidden, W1_out, x_T_buf);
  auto bias_ptr = std::make_unique<OutputVec>();
  OutputVec& bias = *bias_ptr;
  hw::FILE_LOAD(bias, "weights/bias.bin");
  auto gradient_ptr = std::make_unique<OutputVec>();
  OutputVec& gradient = *gradient_ptr;
  hw::FILE_LOAD(gradient, "weights/gradient.bin");
  auto bias_out_ptr = std::make_unique<OutputVec>();
  OutputVec& bias_out = *bias_out_ptr;
  hw::FILE_LOAD(bias_out, "weights/bias_out.bin");
  UpdateBias(bias, gradient, bias_out);
  auto W1_file_ptr = std::make_unique<W1Matrix>();
  W1Matrix& W1_file = *W1_file_ptr;
  hw::FILE_LOAD(W1_file, "weights/W1_file.bin");
  auto b1_file_ptr = std::make_unique<HiddenVec>();
  HiddenVec& b1_file = *b1_file_ptr;
  hw::FILE_LOAD(b1_file, "weights/b1_file.bin");
  auto W2_file_ptr = std::make_unique<W2Matrix>();
  W2Matrix& W2_file = *W2_file_ptr;
  hw::FILE_LOAD(W2_file, "weights/W2_file.bin");
  auto b2_file_ptr = std::make_unique<OutputVec>();
  OutputVec& b2_file = *b2_file_ptr;
  hw::FILE_LOAD(b2_file, "weights/b2_file.bin");
  auto W1_ptr = std::make_unique<W1Matrix>();
  W1Matrix& W1 = *W1_ptr;
  hw::FILE_LOAD(W1, "weights/W1.bin");
  auto b1_ptr = std::make_unique<HiddenVec>();
  HiddenVec& b1 = *b1_ptr;
  hw::FILE_LOAD(b1, "weights/b1.bin");
  auto W2_ptr = std::make_unique<W2Matrix>();
  W2Matrix& W2 = *W2_ptr;
  hw::FILE_LOAD(W2, "weights/W2.bin");
  auto b2_ptr = std::make_unique<OutputVec>();
  OutputVec& b2 = *b2_ptr;
  hw::FILE_LOAD(b2, "weights/b2.bin");
  LoadWeights(W1_file, b1_file, W2_file, b2_file, W1, b1, W2, b2);
  auto W1_ptr = std::make_unique<W1Matrix>();
  W1Matrix& W1 = *W1_ptr;
  hw::FILE_LOAD(W1, "weights/W1.bin");
  auto b1_ptr = std::make_unique<HiddenVec>();
  HiddenVec& b1 = *b1_ptr;
  hw::FILE_LOAD(b1, "weights/b1.bin");
  auto W2_ptr = std::make_unique<W2Matrix>();
  W2Matrix& W2 = *W2_ptr;
  hw::FILE_LOAD(W2, "weights/W2.bin");
  auto b2_ptr = std::make_unique<OutputVec>();
  OutputVec& b2 = *b2_ptr;
  hw::FILE_LOAD(b2, "weights/b2.bin");
  auto W1_file_ptr = std::make_unique<W1Matrix>();
  W1Matrix& W1_file = *W1_file_ptr;
  hw::FILE_LOAD(W1_file, "weights/W1_file.bin");
  auto b1_file_ptr = std::make_unique<HiddenVec>();
  HiddenVec& b1_file = *b1_file_ptr;
  hw::FILE_LOAD(b1_file, "weights/b1_file.bin");
  auto W2_file_ptr = std::make_unique<W2Matrix>();
  W2Matrix& W2_file = *W2_file_ptr;
  hw::FILE_LOAD(W2_file, "weights/W2_file.bin");
  auto b2_file_ptr = std::make_unique<OutputVec>();
  OutputVec& b2_file = *b2_file_ptr;
  hw::FILE_LOAD(b2_file, "weights/b2_file.bin");
  SaveWeights(W1, b1, W2, b2, W1_file, b1_file, W2_file, b2_file);
  std::cout << "Kernel execution successful." << std::endl;
  return 0;
}
