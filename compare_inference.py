#!/usr/bin/env python3
"""
Compare DSL inference output with PyTorch reference implementation.
"""

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel
import json

print("=" * 70)
print("Comparing DSL vs PyTorch Inference")
print("=" * 70)

# Load config
with open("weights/bert_tiny/config.json", "r") as f:
    config = json.load(f)

print(f"\nModel: {config['model_type']}")
print(f"Input dim: {config['input_dim']}, Output dim: {config['output_dim']}")

# Load weights
W = np.fromfile("weights/bert_tiny/query_W.bin", dtype=np.float32).reshape(16, 16)
b = np.fromfile("weights/bert_tiny/query_b.bin", dtype=np.float32)

print(f"\nWeights loaded:")
print(f"  W shape: {W.shape}")
print(f"  b shape: {b.shape}")
print(f"  W sample: {W[0, :3]}")
print(f"  b sample: {b[:3]}")

# Load input (saved by C++ code)
x = np.fromfile("weights/bert_tiny/input.bin", dtype=np.float32).reshape(1, 16)
print(f"\nInput loaded:")
print(f"  x shape: {x.shape}")
print(f"  x sample: {x[0, :3]}")

# Load DSL output
dsl_output = np.fromfile("weights/bert_tiny/dsl_output.bin", dtype=np.float32).reshape(1, 16)

print("\n" + "=" * 70)
print("PyTorch Reference Computation")
print("=" * 70)

# Convert to PyTorch tensors
x_torch = torch.from_numpy(x)
W_torch = torch.from_numpy(W)
b_torch = torch.from_numpy(b)

# Compute: x @ W + b (attention query linear layer)
output_torch = x_torch @ W_torch + b_torch
print(f"\nLinear output (x@W+b):")
print(f"  PyTorch: {output_torch[0, :8].numpy()}")
print(f"  DSL:     {dsl_output[0, :8]}")

# Apply GELU activation
gelu_torch = F.gelu(output_torch)
print(f"\nAfter GELU activation:")
print(f"  PyTorch: {gelu_torch[0, :8].numpy()}")

print("\n" + "=" * 70)
print("Comparison Results")
print("=" * 70)

# Compare outputs
diff = np.abs(dsl_output - output_torch.numpy())
max_diff = np.max(diff)
mean_diff = np.mean(diff)

print(f"\nLinear layer (x@W+b) comparison:")
print(f"  Max absolute difference: {max_diff:.2e}")
print(f"  Mean absolute difference: {mean_diff:.2e}")
print(f"  Outputs match: {np.allclose(dsl_output, output_torch.numpy(), rtol=1e-5, atol=1e-6)}")

if np.allclose(dsl_output, output_torch.numpy(), rtol=1e-5, atol=1e-6):
    print("\n✓ DSL inference matches PyTorch reference!")
    print("  All operations (LOAD, MMUL, bias add, STORE) are correct.")
else:
    print("\n✗ Outputs differ")
    print(f"  First 8 values:")
    print(f"    PyTorch: {output_torch[0, :8].numpy()}")
    print(f"    DSL:     {dsl_output[0, :8]}")
    print(f"    Diff:    {diff[0, :8]}")

print("\n" + "=" * 70)
print("Full PyTorch Model Verification")
print("=" * 70)

# Load full BERT-tiny model for verification
model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
model.eval()

# Extract the same weights from the model
model_query_W = model.encoder.layer[0].attention.self.query.weight.data[:16, :16].numpy()
model_query_b = model.encoder.layer[0].attention.self.query.bias.data[:16].numpy()

# Verify weights match
W_match = np.allclose(W, model_query_W, rtol=1e-6, atol=1e-7)
b_match = np.allclose(b, model_query_b, rtol=1e-6, atol=1e-7)

print(f"\nWeight verification:")
print(f"  W matches model: {W_match}")
print(f"  b matches model: {b_match}")

# Compute using model weights directly
output_model = x @ model_query_W + model_query_b
output_match = np.allclose(dsl_output, output_model, rtol=1e-5, atol=1e-6)

print(f"  DSL output matches model computation: {output_match}")

if W_match and b_match and output_match:
    print("\n✓ Complete verification successful!")
    print("  DSL uses authentic BERT-tiny weights and computes correctly.")
else:
    print("\n⚠ Some verification checks failed")

print("\n" + "=" * 70)
