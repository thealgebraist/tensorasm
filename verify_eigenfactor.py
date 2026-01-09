#!/usr/bin/env python3
import numpy as np

# Load the output
out = np.fromfile('weights/out.bin', dtype=np.float32)

# Load the inputs
A = np.fromfile('weights/A.bin', dtype=np.float32).reshape(4, 4)
v = np.fromfile('weights/v.bin', dtype=np.float32)

# Compute expected result
expected = A @ v

print("Matrix A:")
print(A)
print("\nVector v:")
print(v)
print("\nOutput from kernel:")
print(out)
print("\nExpected (A*v):")
print(expected)
print("\nMatch:", np.allclose(out, expected))
