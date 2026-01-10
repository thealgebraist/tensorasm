
#include "sd_text_kernels.cpp"

int main() {
    auto W_ptr = std::make_unique<WeightMatrix>();
    WeightMatrix& W = *W_ptr;
    hw::FILE_LOAD(W, "weights/stable_diffusion/cross_attn_q_W.bin");
    
    auto b_ptr = std::make_unique<BiasVec>();
    BiasVec& b = *b_ptr;
    hw::FILE_LOAD(b, "weights/stable_diffusion/cross_attn_q_b.bin");
    
    // Load text embeddings (16 tokens x 16 dims)
    auto embeddings_ptr = std::make_unique<WeightMatrix>();
    WeightMatrix& embeddings = *embeddings_ptr;
    hw::FILE_LOAD(embeddings, "weights/stable_diffusion/text_embeddings.bin");
    
    std::cout << "Processing 16 text tokens through cross-attention..." << std::endl;
    
    // Process each token
    std::vector<float> queries(16 * 16);
    
    for (int i = 0; i < 16; i++) {
        auto x_ptr = std::make_unique<InputVec>();
        InputVec& x = *x_ptr;
        
        for (int j = 0; j < 16; j++) {
            x(0, j) = embeddings(i, j);
        }
        
        auto out_ptr = std::make_unique<OutputVec>();
        OutputVec& out = *out_ptr;
        
        // Compute query projection using DSL
        AttentionQuery(x, W, b, out);
        
        for (int j = 0; j < 16; j++) {
            queries[i * 16 + j] = out(0, j);
        }
    }
    
    // Save results
    std::ofstream fout("weights/stable_diffusion/text_queries.bin", std::ios::binary);
    fout.write(reinterpret_cast<char*>(queries.data()), queries.size() * sizeof(float));
    fout.close();
    
    std::cout << "✓ Computed queries for 16 text tokens" << std::endl;
    std::cout << "  Each token: 16-dim embedding -> 16-dim query" << std::endl;
    std::cout << "  Using SD 1.5 cross-attention weights" << std::endl;
    std::cout << "\nThis query computation is used to attend to text" << std::endl;
    std::cout << "during image generation (text-to-image cross-attention)" << std::endl;
    
    return 0;
}
