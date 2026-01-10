#!/usr/bin/env python3
"""
Quick ViT DSL inference - compile and run only, compare separately.
"""

import subprocess
import numpy as np

print("=" * 70)
print("ViT Inference with TensorASM DSL")
print("=" * 70)

# Compile DSL (already done, just extract kernels)
with open("examples/hf_inference.cpp", "r") as f:
    cpp_content = f.read()

header_end = cpp_content.find("void AttentionQuery(")
header = cpp_content[:header_end]

kernels = []
for kernel_name in ["AttentionQuery", "LinearLayer", "GELU", "RunInference"]:
    start = cpp_content.find(f"void {kernel_name}(")
    if start != -1:
        end = cpp_content.find("\n}\n", start) + 3
        kernels.append(cpp_content[start:end])

with open("examples/hf_inference_kernels.cpp", "w") as f:
    f.write(header)
    f.write("\n".join(kernels))

print("✓ Kernels extracted")

# Create inference program
inference_code = """
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
    
    std::cout << "\\nRunning DSL inference with ViT weights..." << std::endl;
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
    
    std::cout << "\\n✓ ViT inference complete (Google ViT ImageNet weights)" << std::endl;
    
    return 0;
}
"""

with open("examples/vit_inference_run.cpp", "w") as f:
    f.write(inference_code)

print("Compiling...")
result = subprocess.run([
    "clang++", "-O3", "-std=c++17", "-I", "/usr/include/eigen3",
    "examples/vit_inference_run.cpp", "-o", "examples/vit_inference_run"
], capture_output=True, text=True)

if result.returncode != 0:
    print("Error:", result.stderr)
    exit(1)

print("✓ Compiled\n")
subprocess.run(["./examples/vit_inference_run"])

# Quick verification with numpy
print("\n" + "=" * 70)
print("Verification")
print("=" * 70)

W = np.fromfile("weights/vit/query_W.bin", dtype=np.float32).reshape(16, 16)
b = np.fromfile("weights/vit/query_b.bin", dtype=np.float32).reshape(1, 16)
x = np.fromfile("weights/vit/input.bin", dtype=np.float32).reshape(1, 16)
dsl_out = np.fromfile("weights/vit/dsl_output.bin", dtype=np.float32).reshape(1, 16)

numpy_out = x @ W + b

print(f"DSL output:   {dsl_out[0, :8]}")
print(f"NumPy output: {numpy_out[0, :8]}")
print(f"Match: {np.allclose(dsl_out, numpy_out, rtol=1e-5, atol=1e-6)}")

if np.allclose(dsl_out, numpy_out, rtol=1e-5, atol=1e-6):
    print("\n✓ DSL matches NumPy - using real Google ViT ImageNet weights!")
print("=" * 70)
