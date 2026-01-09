#!/usr/bin/env python3
import numpy as np
import struct

# Create a simple 4x4 matrix
A = np.array([
    [4.0, 1.0, 0.0, 0.0],
    [1.0, 3.0, 1.0, 0.0],
    [0.0, 1.0, 2.0, 1.0],
    [0.0, 0.0, 1.0, 1.0]
], dtype=np.float32)

# Create an initial vector
v = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

# Expected output (A * v)
out = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

# Save to binary files
A.tofile('weights/A.bin')
v.tofile('weights/v.bin')
out.tofile('weights/out.bin')

print("Generated test data:")
print(f"A =\n{A}")
print(f"v = {v}")
print(f"Expected A*v = {A @ v}")
