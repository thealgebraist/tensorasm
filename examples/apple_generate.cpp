
#include "apple_gen_kernels.cpp"

int main() {
    auto W_ptr = std::make_unique<WeightMatrix>();
    WeightMatrix& W = *W_ptr;
    hw::FILE_LOAD(W, "weights/stable_diffusion/cross_attn_q_W.bin");
    
    auto b_ptr = std::make_unique<BiasVec>();
    BiasVec& b = *b_ptr;
    hw::FILE_LOAD(b, "weights/stable_diffusion/cross_attn_q_b.bin");
    
    // Load 64 feature vectors (8x8 spatial grid)
    std::vector<float> features(64 * 16);
    std::ifstream fin("weights/stable_diffusion/apple_features.bin", std::ios::binary);
    fin.read(reinterpret_cast<char*>(features.data()), features.size() * sizeof(float));
    fin.close();
    
    std::cout << "Processing 64 spatial locations (8x8 grid)..." << std::endl;
    
    // Transform each spatial location with SD weights
    std::vector<float> transformed(64 * 16);
    
    for (int i = 0; i < 64; i++) {
        auto x_ptr = std::make_unique<InputVec>();
        InputVec& x = *x_ptr;
        
        for (int j = 0; j < 16; j++) {
            x(0, j) = features[i * 16 + j];
        }
        
        auto out_ptr = std::make_unique<OutputVec>();
        OutputVec& out = *out_ptr;
        
        // Apply SD cross-attention transformation
        AttentionQuery(x, W, b, out);
        
        for (int j = 0; j < 16; j++) {
            transformed[i * 16 + j] = out(0, j);
        }
    }
    
    std::ofstream fout("weights/stable_diffusion/apple_transformed.bin", std::ios::binary);
    fout.write(reinterpret_cast<char*>(transformed.data()), transformed.size() * sizeof(float));
    fout.close();
    
    std::cout << "✓ Transformed 64 feature vectors using SD 1.5 weights" << std::endl;
    std::cout << "  Each location: 16-dim -> 16-dim via cross-attention" << std::endl;
    
    return 0;
}
