
#include "vit_gen_kernels.cpp"

int main() {
    auto W_ptr = std::make_unique<WeightMatrix>();
    WeightMatrix& W = *W_ptr;
    hw::FILE_LOAD(W, "weights/vit/query_W.bin");
    
    auto b_ptr = std::make_unique<BiasVec>();
    BiasVec& b = *b_ptr;
    hw::FILE_LOAD(b, "weights/vit/query_b.bin");
    
    // Load features (16 patches x 16 dims)
    auto features_ptr = std::make_unique<WeightMatrix>();
    WeightMatrix& features = *features_ptr;
    hw::FILE_LOAD(features, "weights/vit/generation_features.bin");
    
    // Transform each patch feature
    std::vector<float> transformed_features(16 * 16);
    
    for (int i = 0; i < 16; i++) {
        auto x_ptr = std::make_unique<InputVec>();
        InputVec& x = *x_ptr;
        
        // Extract one patch feature
        for (int j = 0; j < 16; j++) {
            x(0, j) = features(i, j);
        }
        
        auto out_ptr = std::make_unique<OutputVec>();
        OutputVec& out = *out_ptr;
        
        // Transform using DSL kernel
        AttentionQuery(x, W, b, out);
        
        // Store result
        for (int j = 0; j < 16; j++) {
            transformed_features[i * 16 + j] = out(0, j);
        }
    }
    
    // Save transformed features
    std::ofstream fout("weights/vit/transformed_features.bin", std::ios::binary);
    fout.write(reinterpret_cast<char*>(transformed_features.data()), 
               transformed_features.size() * sizeof(float));
    fout.close();
    
    std::cout << "✓ Transformed 16 patch features using DSL kernel" << std::endl;
    std::cout << "  Each feature: 16-dim input -> 16-dim output" << std::endl;
    std::cout << "  Using authentic ViT query projection weights" << std::endl;
    
    return 0;
}
