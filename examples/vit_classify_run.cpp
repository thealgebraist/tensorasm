
#include "vit_classify_kernels.cpp"

int main() {
    auto W_ptr = std::make_unique<WeightMatrix>();
    WeightMatrix& W = *W_ptr;
    hw::FILE_LOAD(W, "weights/vit/query_W.bin");
    
    auto b_ptr = std::make_unique<BiasVec>();
    BiasVec& b = *b_ptr;
    hw::FILE_LOAD(b, "weights/vit/query_b.bin");
    
    auto x_ptr = std::make_unique<InputVec>();
    InputVec& x = *x_ptr;
    hw::FILE_LOAD(x, "weights/vit/real_patch_features.bin");
    
    auto out_ptr = std::make_unique<OutputVec>();
    OutputVec& out = *out_ptr;
    
    auto gelu_out_ptr = std::make_unique<OutputVec>();
    OutputVec& gelu_out = *gelu_out_ptr;
    
    RunInference(x, W, b, out, gelu_out);
    
    std::ofstream fout("weights/vit/dsl_query_output.bin", std::ios::binary);
    fout.write(reinterpret_cast<const char*>(out.data()), out.size() * sizeof(float));
    fout.close();
    
    std::cout << "✓ DSL computed query projection on real image features" << std::endl;
    
    return 0;
}
