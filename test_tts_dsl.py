#!/usr/bin/env python3
"""
Test TTS DSL kernel with actual model weights
"""
import numpy as np
import os

def main():
    print("=== TTS DSL Test ===\n")
    
    weights_dir = "weights/tts"
    
    # Load embeddings
    print("1. Loading TTS embeddings...")
    text_embed = np.fromfile(f"{weights_dir}/text_embed.bin", dtype=np.float32)
    text_embed = text_embed.reshape(81, 768)
    print(f"   Embedding shape: {text_embed.shape}")
    
    # Create test token
    print("\n2. Creating test input...")
    token_idx = np.array([5], dtype=np.int32)
    print(f"   Token ID: {token_idx[0]}")
    
    # Save inputs for DSL kernel
    os.makedirs("weights", exist_ok=True)
    token_idx.tofile("weights/token_idx.bin")
    text_embed.tofile("weights/embed_w.bin")
    
    # Initialize output
    output = np.zeros(768, dtype=np.float32)
    output.tofile("weights/output.bin")
    
    print("\n3. Compiling and running DSL kernel...")
    os.system("./tensorasm examples/tts_lookup.ta > examples/tts_lookup.cpp")
    os.system("g++ -O3 -std=c++17 -I/usr/include/eigen3 examples/tts_lookup.cpp -o examples/tts_lookup_run")
    
    # Run the compiled kernel from the root directory
    result = os.system("./examples/tts_lookup_run")
    
    if result == 0:
        # Load output
        output = np.fromfile("weights/output.bin", dtype=np.float32)
        
        print("\n4. Results:")
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min():.4f}, {output.max():.4f}]")
        print(f"   Output mean: {output.mean():.4f}")
        print(f"   Output std: {output.std():.4f}")
        print(f"   First 5 values: {output[:5]}")
        
        # Verify against Python
        expected = text_embed[token_idx[0]]
        diff = np.abs(output - expected).max()
        print(f"\n5. Verification:")
        print(f"   Max difference from Python: {diff:.6f}")
        
        if diff < 1e-5:
            print("\n✓ SUCCESS: TTS model working with DSL!")
        else:
            print(f"\n✗ Warning: Difference {diff} might be too large")
    else:
        print("\n✗ Kernel execution failed")

if __name__ == "__main__":
    main()
