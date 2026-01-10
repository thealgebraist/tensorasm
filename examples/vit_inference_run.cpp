
#include "hf_inference_kernels.cpp"

int main() {
    auto W_ptr = std::make_unique<WeightMatrix>();
    WeightMatrix& W = *W_ptr;
    hw::FILE_LOAD(W, "weights/vit/query_W.bin");
    
    auto b_ptr = std::make_unique<BiasVec>();
    BiasVec& b = *b_ptr;
    hw::FILE_LOAD(b, "weights/vit/query_b.bin");
    
    auto x_ptr = std::make_unique<InputVec>();
    InputVec& x = *x_ptr;
    x.setRandom();
    
    std::ofstream fin("weights/vit/input.bin", std::ios::binary);
    fin.write(reinterpret_cast<const char*>(x.data()), x.size() * sizeof(float));
    fin.close();
    
    std::cout << "Input: " << x(0, 0) << ", " << x(0, 1) << ", " << x(0, 2) << std::endl;
    
    auto out_ptr = std::make_unique<OutputVec>();
    OutputVec& out = *out_ptr;
    
    auto gelu_out_ptr = std::make_unique<OutputVec>();
    OutputVec& gelu_out = *gelu_out_ptr;
    
    std::cout << "\nRunning DSL inference with ViT weights..." << std::endl;
    RunInference(x, W, b, out, gelu_out);
    
    std::cout << "Query output: ";
    for (int i = 0; i < 8; i++) std::cout << out(0, i) << " ";
    std::cout << std::endl;
    
    std::cout << "After GELU: ";
    for (int i = 0; i < 8; i++) std::cout << gelu_out(0, i) << " ";
    std::cout << std::endl;
    
    std::ofstream fout("weights/vit/dsl_output.bin", std::ios::binary);
    fout.write(reinterpret_cast<const char*>(out.data()), out.size() * sizeof(float));
    fout.close();
    
    std::cout << "\n✓ ViT inference complete (Google ViT ImageNet weights)" << std::endl;
    
    return 0;
}
