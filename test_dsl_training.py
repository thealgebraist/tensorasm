#!/usr/bin/env python3
"""
Use DSL-generated kernels for full FFNN training
"""
import numpy as np
import subprocess
import json
import os

# Compile the DSL
print("Compiling DSL kernels...")
subprocess.run(["./tensorasm", "examples/ffnn_train.ta"], 
               stdout=open("examples/ffnn_train_gen.cpp", "w"),
               check=True)

# Extract just the kernel functions without the main()
with open("examples/ffnn_train_gen.cpp", "r") as f:
    lines = f.readlines()

# Find where main() starts and remove it
kernel_code = []
in_main = False
for line in lines:
    if line.strip().startswith("int main()"):
        in_main = True
    if not in_main:
        kernel_code.append(line)

# Write kernels only
with open("examples/ffnn_train_kernels.cpp", "w") as f:
    f.writelines(kernel_code)

print("DSL kernels extracted to ffnn_train_kernels.cpp")
print(f"Total lines: {len(kernel_code)}")

# Now create a simple demonstration showing the kernels work
demo_code = """
#include "ffnn_train_kernels.cpp"

int main() {
    // Initialize weights
    auto W1_ptr = std::make_unique<W1Matrix>();
    W1Matrix& W1 = *W1_ptr;
    W1.setRandom();
    
    auto b1_ptr = std::make_unique<HiddenVec>();
    HiddenVec& b1 = *b1_ptr;
    b1.setZero();
    
    auto W2_ptr = std::make_unique<W2Matrix>();
    W2Matrix& W2 = *W2_ptr;
    W2.setRandom();
    
    auto b2_ptr = std::make_unique<OutputVec>();
    OutputVec& b2 = *b2_ptr;
    b2.setZero();
    
    // Test input
    auto x_ptr = std::make_unique<InputMat>();
    InputMat& x = *x_ptr;
    x(0, 0) = 1.5;
    
    // Forward pass buffers
    auto z1_ptr = std::make_unique<HiddenVec>();
    HiddenVec& z1 = *z1_ptr;
    
    auto y_pred_ptr = std::make_unique<OutputVec>();
    OutputVec& y_pred = *y_pred_ptr;
    
    // Run forward pass
    ForwardPass(x, W1, b1, W2, b2, z1, y_pred);
    
    std::cout << "Forward pass successful!" << std::endl;
    std::cout << "Input: " << x(0, 0) << std::endl;
    std::cout << "Output (first 4 values): ";
    for (int i = 0; i < 4; i++) {
        std::cout << y_pred(0, i) << " ";
    }
    std::cout << std::endl;
    
    // Test backward pass
    auto y_true_ptr = std::make_unique<OutputVec>();
    OutputVec& y_true = *y_true_ptr;
    for (int i = 0; i < 16; i++) {
        y_true(0, i) = std::sin((i + 1) * x(0, 0));
    }
    
    auto grad_out_ptr = std::make_unique<OutputVec>();
    OutputVec& grad_out = *grad_out_ptr;
    
    ComputeOutputGradient(y_pred, y_true, grad_out);
    
    std::cout << "Gradient computed!" << std::endl;
    std::cout << "Gradient (first 4 values): ";
    for (int i = 0; i < 4; i++) {
        std::cout << grad_out(0, i) << " ";
    }
    std::cout << std::endl;
    
    // Test weight updates
    auto grad_h_ptr = std::make_unique<HiddenVec>();
    HiddenVec& grad_h = *grad_h_ptr;
    
    BackwardHiddenLayer(grad_out, W2, z1, grad_h);
    
    auto W2_new_ptr = std::make_unique<W2Matrix>();
    W2Matrix& W2_new = *W2_new_ptr;
    
    auto z1_T_buf_ptr = std::make_unique<HiddenVecT>();
    HiddenVecT& z1_T_buf = *z1_T_buf_ptr;
    
    UpdateW2(W2, z1, grad_out, W2_new, z1_T_buf);
    
    auto W1_new_ptr = std::make_unique<W1Matrix>();
    W1Matrix& W1_new = *W1_new_ptr;
    
    auto x_T_buf_ptr = std::make_unique<InputMatT>();
    InputMatT& x_T_buf = *x_T_buf_ptr;
    
    UpdateW1(W1, x, grad_h, W1_new, x_T_buf);
    
    auto b1_new_ptr = std::make_unique<HiddenVec>();
    HiddenVec& b1_new = *b1_new_ptr;
    
    auto b2_new_ptr = std::make_unique<OutputVec>();
    OutputVec& b2_new = *b2_new_ptr;
    
    UpdateBias(b1, grad_h, b1_new);
    UpdateBias(b2, grad_out, b2_new);
    
    std::cout << "Weight updates successful!" << std::endl;
    
    // Test save/load
    auto W1_file_ptr = std::make_unique<W1Matrix>();
    W1Matrix& W1_file = *W1_file_ptr;
    hw::FILE_LOAD(W1_file, "weights/ffnn/W1.bin");
    
    auto b1_file_ptr = std::make_unique<HiddenVec>();
    HiddenVec& b1_file = *b1_file_ptr;
    hw::FILE_LOAD(b1_file, "weights/ffnn/b1.bin");
    
    auto W2_file_ptr = std::make_unique<W2Matrix>();
    W2Matrix& W2_file = *W2_file_ptr;
    hw::FILE_LOAD(W2_file, "weights/ffnn/W2.bin");
    
    auto b2_file_ptr = std::make_unique<OutputVec>();
    OutputVec& b2_file = *b2_file_ptr;
    hw::FILE_LOAD(b2_file, "weights/ffnn/b2.bin");
    
    auto W1_loaded_ptr = std::make_unique<W1Matrix>();
    W1Matrix& W1_loaded = *W1_loaded_ptr;
    
    auto b1_loaded_ptr = std::make_unique<HiddenVec>();
    HiddenVec& b1_loaded = *b1_loaded_ptr;
    
    auto W2_loaded_ptr = std::make_unique<W2Matrix>();
    W2Matrix& W2_loaded = *W2_loaded_ptr;
    
    auto b2_loaded_ptr = std::make_unique<OutputVec>();
    OutputVec& b2_loaded = *b2_loaded_ptr;
    
    LoadWeights(W1_file, b1_file, W2_file, b2_file, W1_loaded, b1_loaded, W2_loaded, b2_loaded);
    
    std::cout << "Weights loaded successfully!" << std::endl;
    std::cout << "W1 sample values: " << W1_loaded(0, 0) << ", " << W1_loaded(0, 1) << std::endl;
    
    std::cout << "\\n=== All DSL kernels working! ===" << std::endl;
    
    return 0;
}
"""

with open("examples/ffnn_train_demo.cpp", "w") as f:
    f.write(demo_code)

print("\nCompiling demo program...")
result = subprocess.run([
    "clang++", "-O3", "-std=c++17", "-I", "/usr/include/eigen3",
    "examples/ffnn_train_demo.cpp", "-o", "examples/ffnn_train_demo"
], capture_output=True, text=True)

if result.returncode == 0:
    print("✓ Demo compiled successfully!")
    print("\nRunning demo...")
    subprocess.run(["./examples/ffnn_train_demo"])
else:
    print("✗ Compilation failed:")
    print(result.stderr)
