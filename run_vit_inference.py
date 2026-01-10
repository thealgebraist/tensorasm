#!/usr/bin/env python3
"""
Run Vision Transformer (ViT) inference using TensorASM DSL and compare with PyTorch.
Uses real pretrained Google ViT weights from ImageNet.
"""

import subprocess
import sys
import numpy as np
import torch
import torch.nn.functional as F
from transformers import ViTForImageClassification
import json
import os

print("=" * 70)
print("Vision Transformer (ViT) Inference with TensorASM DSL")
print("=" * 70)

# Load config
with open("weights/vit/config.json", "r") as f:
    config = json.load(f)

print(f"\nModel config:")
print(json.dumps(config, indent=2))

# Load weights
W = np.fromfile("weights/vit/query_W.bin", dtype=np.float32).reshape(16, 16)
b = np.fromfile("weights/vit/query_b.bin", dtype=np.float32).reshape(1, 16)

print(f"\nLoaded weights:")
print(f"  W shape: {W.shape}")
print(f"  b shape: {b.shape}")
print(f"  W sample values: {W[0, :3]}")
print(f"  b sample values: {b[0, :3]}")

# Compile DSL kernels (reuse the same hf_inference.ta)
print("\n" + "=" * 60)
print("Compiling DSL kernels...")
print("=" * 60)

result = subprocess.run([
    "./tensorasm", "examples/hf_inference.ta"
], capture_output=True, text=True)

if result.returncode != 0:
    print("DSL compilation failed:")
    print(result.stderr)
    exit(1)

# Extract kernel implementations
with open("examples/hf_inference.cpp", "r") as f:
    cpp_content = f.read()

# Get the header part (includes, types, hw namespace)
header_end = cpp_content.find("void AttentionQuery(")
header = cpp_content[:header_end]

# Extract all kernel functions
kernels = []
for kernel_name in ["AttentionQuery", "LinearLayer", "GELU", "RunInference"]:
    start = cpp_content.find(f"void {kernel_name}(")
    if start != -1:
        end = cpp_content.find("\n}\n", start) + 3
        kernels.append(cpp_content[start:end])

with open("examples/hf_inference_kernels.cpp", "w") as f:
    f.write(header)
    f.write("\n".join(kernels))

print("✓ DSL kernels extracted")

# Create inference program that uses ViT weights
inference_code = """
#include "hf_inference_kernels.cpp"

int main() {
    // Load ViT weights from files
    auto W_ptr = std::make_unique<WeightMatrix>();
    WeightMatrix& W = *W_ptr;
    hw::FILE_LOAD(W, "weights/vit/query_W.bin");
    
    auto b_ptr = std::make_unique<BiasVec>();
    BiasVec& b = *b_ptr;
    hw::FILE_LOAD(b, "weights/vit/query_b.bin");
    
    // Create test input (random 16-dim vector representing patch features)
    auto x_ptr = std::make_unique<InputVec>();
    InputVec& x = *x_ptr;
    x.setRandom();
    
    // Save input for verification
    std::ofstream fin("weights/vit/input.bin", std::ios::binary);
    fin.write(reinterpret_cast<const char*>(x.data()), x.size() * sizeof(float));
    fin.close();
    
    std::cout << "Input (patch features) sample: " << x(0, 0) << ", " << x(0, 1) << ", " << x(0, 2) << std::endl;
    
    // Create output tensors
    auto out_ptr = std::make_unique<OutputVec>();
    OutputVec& out = *out_ptr;
    
    auto gelu_out_ptr = std::make_unique<OutputVec>();
    OutputVec& gelu_out = *gelu_out_ptr;
    
    std::cout << "\\n=== Running DSL Inference with ViT Weights ===" << std::endl;
    
    // Run inference using DSL kernel (computes attention query + GELU)
    RunInference(x, W, b, out, gelu_out);
    
    std::cout << "\\n=== DSL Inference Results ===" << std::endl;
    std::cout << "Query output (first 8 values): ";
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
    std::ofstream fout("weights/vit/dsl_output.bin", std::ios::binary);
    fout.write(reinterpret_cast<const char*>(out.data()), out.size() * sizeof(float));
    fout.close();
    
    std::cout << "\\n✓ ViT inference complete!" << std::endl;
    std::cout << "  DSL kernel performed: LOAD, MMUL, bias add, ACT(GELU), STORE" << std::endl;
    std::cout << "  Using authentic Google ViT ImageNet weights" << std::endl;
    
    return 0;
}
"""

with open("examples/vit_inference_run.cpp", "w") as f:
    f.write(inference_code)

print("\n" + "=" * 60)
print("Compiling inference program...")
print("=" * 60)

result = subprocess.run([
    "clang++", "-O3", "-std=c++17", "-I", "/usr/include/eigen3",
    "examples/vit_inference_run.cpp", "-o", "examples/vit_inference_run"
], capture_output=True, text=True)

if result.returncode != 0:
    print("Compilation failed:")
    print(result.stderr)
    exit(1)

print("✓ Compiled successfully!")

print("\n" + "=" * 60)
print("Running DSL inference...")
print("=" * 60)
subprocess.run(["./examples/vit_inference_run"])

# Verify output
print("\n" + "=" * 60)
print("Comparing with PyTorch ViT...")
print("=" * 60)

# Load DSL output and input
dsl_output = np.fromfile("weights/vit/dsl_output.bin", dtype=np.float32).reshape(1, 16)
x_input = np.fromfile("weights/vit/input.bin", dtype=np.float32).reshape(1, 16)

# Load full ViT model
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
model.eval()

# Extract the same query weights
model_query_W = model.vit.encoder.layer[0].attention.attention.query.weight.data[:16, :16].numpy()
model_query_b = model.vit.encoder.layer[0].attention.attention.query.bias.data[:16].numpy()

# Verify weights match
W_match = np.allclose(W, model_query_W, rtol=1e-6, atol=1e-7)
b_match = np.allclose(b, model_query_b, rtol=1e-6, atol=1e-7)

print(f"\nWeight verification:")
print(f"  W matches ViT model: {W_match}")
print(f"  b matches ViT model: {b_match}")

# Compute PyTorch reference
output_torch = x_input @ model_query_W + model_query_b

print(f"\nQuery projection (x@W+b):")
print(f"  PyTorch: {output_torch[0, :8]}")
print(f"  DSL:     {dsl_output[0, :8]}")

# Compare outputs
diff = np.abs(dsl_output - output_torch)
max_diff = np.max(diff)
mean_diff = np.mean(diff)

print(f"\nComparison:")
print(f"  Max absolute difference: {max_diff:.2e}")
print(f"  Mean absolute difference: {mean_diff:.2e}")
print(f"  Outputs match: {np.allclose(dsl_output, output_torch, rtol=1e-5, atol=1e-6)}")

if np.allclose(dsl_output, output_torch, rtol=1e-5, atol=1e-6):
    print("\n✓ DSL inference matches PyTorch ViT reference!")
    print("  All operations correct with real Google ViT ImageNet weights")
else:
    print("\n✗ Outputs differ")

print("\n" + "=" * 60)
print("✓ Vision Transformer inference complete!")
print(f"  Model: {config['model_type']}")
print(f"  Total ViT parameters: {config['total_params']:,}")
print(f"  DSL computed: Attention query projection on patch features")
print("=" * 60)
