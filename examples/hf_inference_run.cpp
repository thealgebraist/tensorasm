
#include "hf_inference_kernels.cpp"

int main() {
    // Load weights from files
    auto W_ptr = std::make_unique<WeightMatrix>();
    WeightMatrix& W = *W_ptr;
    hw::FILE_LOAD(W, "weights/bert_tiny/query_W.bin");
    
    auto b_ptr = std::make_unique<BiasVec>();
    BiasVec& b = *b_ptr;
    hw::FILE_LOAD(b, "weights/bert_tiny/query_b.bin");
    
    // Create test input (random 16-dim vector)
    auto x_ptr = std::make_unique<InputVec>();
    InputVec& x = *x_ptr;
    x.setRandom();
    
    // Save input for verification
    std::ofstream fin("weights/bert_tiny/input.bin", std::ios::binary);
    fin.write(reinterpret_cast<const char*>(x.data()), x.size() * sizeof(float));
    fin.close();
    
    std::cout << "Input sample: " << x(0, 0) << ", " << x(0, 1) << ", " << x(0, 2) << std::endl;
    
    // Create output tensors
    auto out_ptr = std::make_unique<OutputVec>();
    OutputVec& out = *out_ptr;
    
    auto gelu_out_ptr = std::make_unique<OutputVec>();
    OutputVec& gelu_out = *gelu_out_ptr;
    
    std::cout << "\n=== Running DSL Inference ===" << std::endl;
    
    // Run inference using DSL kernel (computes attention + GELU in DSL)
    RunInference(x, W, b, out, gelu_out);
    
    std::cout << "\n=== DSL Inference Results ===" << std::endl;
    std::cout << "Output (first 8 values): ";
    for (int i = 0; i < 8; i++) {
        std::cout << out(0, i) << " ";
    }
    std::cout << std::endl;
    
    std::cout << "After GELU (first 8 values): ";
    for (int i = 0; i < 8; i++) {
        std::cout << gelu_out(0, i) << " ";
    }
    std::cout << std::endl;
    
    // Save output for verification
    std::ofstream fout("weights/bert_tiny/dsl_output.bin", std::ios::binary);
    fout.write(reinterpret_cast<const char*>(out.data()), out.size() * sizeof(float));
    fout.close();
    
    std::cout << "\n✓ Inference complete!" << std::endl;
    std::cout << "  DSL kernel performed: LOAD, MMUL, bias add, ACT(GELU), STORE" << std::endl;
    
    return 0;
}
