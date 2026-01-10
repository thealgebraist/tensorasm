#!/usr/bin/env python3
"""
Download a small vision model and extract weights for DSL inference.
"""

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModelForImageClassification
import numpy as np
import json
import os

print("=" * 70)
print("Downloading Small Vision Model for DSL")
print("=" * 70)

# Try google/vit-base-patch16-224-in21k first (might be large)
# Alternative: microsoft/resnet-18 or a simple custom CNN

print("\nSearching for small vision models...")
print("Trying: HuggingFace vision models")

try:
    from transformers import AutoImageProcessor, ViTForImageClassification
    
    # Try a small ViT model
    model_name = "google/vit-base-patch16-224"
    print(f"Loading {model_name}...")
    
    model = ViTForImageClassification.from_pretrained(model_name)
    processor = AutoImageProcessor.from_pretrained(model_name)
    model.eval()
    
    print("✓ ViT model loaded successfully!")
    
    # Get model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Extract embedding projection layer (patch embedding)
    # ViT processes image patches through a Conv2d layer
    patch_embed = model.vit.embeddings.patch_embeddings.projection
    
    print(f"\nPatch Embedding Layer:")
    print(f"  Weight shape: {patch_embed.weight.shape}")  
    print(f"  Bias shape: {patch_embed.bias.shape if patch_embed.bias is not None else 'None'}")
    
    # Extract first attention layer query projection (linear layer)
    qkv = model.vit.encoder.layer[0].attention.attention
    
    query_weight = qkv.query.weight.data.numpy()  
    query_bias = qkv.query.bias.data.numpy()
    
    print(f"\nFirst Attention Query Layer:")
    print(f"  Weight shape: {query_weight.shape}")
    print(f"  Bias shape: {query_bias.shape}")
    
    # Take a 16x16 subset for DSL
    weight = query_weight[:16, :16].astype(np.float32)
    bias = query_bias[:16].astype(np.float32)
    # Take a 16x16 subset for DSL
    weight = query_weight[:16, :16].astype(np.float32)
    bias = query_bias[:16].astype(np.float32)
    
    print(f"\nExtracted weights:")
    print(f"  Query weight subset: {weight.shape}")
    print(f"  Query bias subset: {bias.shape}")
    
    # Create output directory
    os.makedirs("weights/vit", exist_ok=True)
    
    # Save weights
    weight.tofile("weights/vit/query_W.bin")
    bias.tofile("weights/vit/query_b.bin")
    
    # Save as text too
    with open("weights/vit/query_W.txt", "w") as f:
        for row in weight:
            f.write(" ".join([f"{v:.8f}" for v in row]) + "\n")
    
    with open("weights/vit/query_b.txt", "w") as f:
        for b in bias:
            f.write(f"{b:.8f}\n")
    
    # Save config
    config = {
        "model_type": "vit-base-patch16-224",
        "layer": "encoder.layer[0].attention.query",
        "input_dim": 16,
        "output_dim": 16,
        "total_params": int(total_params),
        "shapes": {
            "W": [16, 16],
            "b": [16]
        }
    }
    
    with open("weights/vit/config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "=" * 70)
    print("✓ ViT weights saved!")
    print("=" * 70)
    print(f"  Location: weights/vit/")
    print(f"  Files: query_W.bin ({os.path.getsize('weights/vit/query_W.bin')} bytes)")
    print(f"         query_b.bin ({os.path.getsize('weights/vit/query_b.bin')} bytes)")
    print(f"  Weight sample: {weight[0, :3]}")
    print(f"  Bias sample: {bias[:3]}")
    
    print("\n✓ Real pretrained ViT model weights extracted!")
    print("  These are authentic Google ViT weights from ImageNet pretraining")

except Exception as e:
    print(f"Error loading ViT: {e}")
    import traceback
    traceback.print_exc()
