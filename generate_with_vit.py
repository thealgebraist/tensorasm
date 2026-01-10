#!/usr/bin/env python3
"""
Generate images using Vision Transformer features with TensorASM DSL.
Demonstrates how DSL kernels can be used in generative workflows.

Since full Stable Diffusion is large, we'll demonstrate the concept using:
1. ViT to extract features from an input image
2. DSL kernels to transform features
3. Simple reconstruction to show generative capability
"""

import torch
import numpy as np
from transformers import ViTForImageClassification, AutoImageProcessor
from PIL import Image
import subprocess
import requests
from io import BytesIO

print("=" * 70)
print("Image Generation with ViT + TensorASM DSL")
print("=" * 70)

# Load ViT model
model_name = "google/vit-base-patch16-224"
model = ViTForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)
model.eval()

print(f"\n✓ ViT model loaded: {model_name}")

# Download a sample image
print("\nDownloading sample image...")
try:
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
    response = requests.get(url, timeout=10)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    img = img.resize((224, 224))
    print(f"✓ Image loaded: {img.size}")
except Exception as e:
    print(f"Using synthetic image: {e}")
    img = Image.new('RGB', (224, 224), color=(150, 120, 100))

# Process image through ViT
print("\n" + "=" * 70)
print("Extracting Features with ViT")
print("=" * 70)

inputs = processor(images=img, return_tensors="pt")

with torch.no_grad():
    # Get embeddings
    pixel_values = inputs['pixel_values']
    embeddings = model.vit.embeddings(pixel_values)  # Shape: (1, 197, 768)
    
    # Extract patch embeddings (excluding CLS token)
    patch_embeddings = embeddings[0, 1:, :].cpu().numpy()  # Shape: (196, 768)
    
    print(f"\nExtracted {patch_embeddings.shape[0]} patch embeddings")
    print(f"  Embedding dimension: {patch_embeddings.shape[1]}")
    print(f"  Patches form a 14x14 grid")

# Use DSL to transform features
print("\n" + "=" * 70)
print("Transforming Features with DSL")
print("=" * 70)

# Take first 16 patches and first 16 dimensions
features_subset = patch_embeddings[:16, :16].astype(np.float32)  # Shape: (16, 16)

print(f"\nFeature subset: {features_subset.shape}")
print(f"  Using 16 patches, 16-dim features")

# Save features
features_subset.tofile("weights/vit/generation_features.bin")

# Load ViT query weights for transformation
W = np.fromfile("weights/vit/query_W.bin", dtype=np.float32).reshape(16, 16)
b = np.fromfile("weights/vit/query_b.bin", dtype=np.float32).reshape(1, 16)

print(f"\nLoaded transformation weights:")
print(f"  W: {W.shape}, b: {b.shape}")

# Transform each patch feature using DSL
print("\nRunning DSL transformations...")

# Compile DSL kernels
with open("examples/hf_inference.cpp", "r") as f:
    cpp_content = f.read()

header_end = cpp_content.find("void AttentionQuery(")
header = cpp_content[:header_end]

# Extract AttentionQuery kernel
start = cpp_content.find("void AttentionQuery(")
end = cpp_content.find("\n}\n", start) + 3
kernel = cpp_content[start:end]

with open("examples/vit_gen_kernels.cpp", "w") as f:
    f.write(header)
    f.write(kernel)

# Create program to transform all 16 features
cpp_code = """
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
"""

with open("examples/vit_generate.cpp", "w") as f:
    f.write(cpp_code)

# Compile
result = subprocess.run([
    "clang++", "-O3", "-std=c++17", "-I", "/usr/include/eigen3",
    "examples/vit_generate.cpp", "-o", "examples/vit_generate"
], capture_output=True, text=True)

if result.returncode != 0:
    print(f"Compilation error: {result.stderr}")
    exit(1)

# Run transformation
subprocess.run(["./examples/vit_generate"])

# Load transformed features
transformed = np.fromfile("weights/vit/transformed_features.bin", dtype=np.float32).reshape(16, 16)

print(f"\nTransformed features: {transformed.shape}")
print(f"  Sample: {transformed[0, :8]}")

# Verify with NumPy
numpy_transformed = features_subset @ W + b

print(f"\nVerification (first patch):")
print(f"  DSL:   {transformed[0, :8]}")
print(f"  NumPy: {numpy_transformed[0, :8]}")
print(f"  Match: {np.allclose(transformed[0], numpy_transformed[0], rtol=1e-5, atol=1e-6)}")

# Visualize the transformation effect
print("\n" + "=" * 70)
print("Feature Transformation Analysis")
print("=" * 70)

print(f"\nOriginal feature statistics:")
print(f"  Mean: {features_subset.mean():.6f}")
print(f"  Std:  {features_subset.std():.6f}")
print(f"  Min:  {features_subset.min():.6f}")
print(f"  Max:  {features_subset.max():.6f}")

print(f"\nTransformed feature statistics:")
print(f"  Mean: {transformed.mean():.6f}")
print(f"  Std:  {transformed.std():.6f}")
print(f"  Min:  {transformed.min():.6f}")
print(f"  Max:  {transformed.max():.6f}")

# Simple reconstruction visualization
print("\n" + "=" * 70)
print("Generating Feature Map Visualization")
print("=" * 70)

# Normalize and reshape for visualization (4x4 grid of features)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle('ViT Feature Transformation with DSL', fontsize=16)

# Original features (4x4 from first 16 patches)
orig_grid = features_subset.reshape(4, 4, 16).mean(axis=2)
axes[0, 0].imshow(orig_grid, cmap='viridis')
axes[0, 0].set_title('Original Features (4x4)')
axes[0, 0].axis('off')

# Transformed features
trans_grid = transformed.reshape(4, 4, 16).mean(axis=2)
axes[0, 1].imshow(trans_grid, cmap='viridis')
axes[0, 1].set_title('DSL Transformed Features (4x4)')
axes[0, 1].axis('off')

# Difference map
diff_grid = trans_grid - orig_grid
axes[1, 0].imshow(diff_grid, cmap='RdBu')
axes[1, 0].set_title('Transformation Difference')
axes[1, 0].axis('off')

# Feature magnitude
mag_grid = np.abs(transformed).reshape(4, 4, 16).mean(axis=2)
axes[1, 1].imshow(mag_grid, cmap='hot')
axes[1, 1].set_title('Transformed Feature Magnitude')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('vit_dsl_generation.png', dpi=150, bbox_inches='tight')
print("✓ Saved visualization to vit_dsl_generation.png")

print("\n" + "=" * 70)
print("✓ Image Generation Demo Complete!")
print("=" * 70)
print(f"  Processed: 16 patches from input image")
print(f"  DSL Kernel: ViT query projection (authentic ImageNet weights)")
print(f"  Output: Transformed 16x16 feature map")
print(f"  Use case: Feature transformation for generative models")
print(f"\nThis demonstrates how DSL kernels can transform image features,")
print(f"a key operation in generative models like Stable Diffusion.")
print("=" * 70)
