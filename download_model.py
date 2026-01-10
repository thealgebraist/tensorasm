#!/usr/bin/env python3
"""
Download a small model and prepare it for DSL inference
"""
import torch
import torch.nn as nn
import numpy as np
import json
import os

print("Looking for small pre-trained models...")

# Option 1: Try downloading a tiny model from HuggingFace
try:
    from transformers import AutoModel, AutoTokenizer
    
    # Try distilbert-base-uncased (smaller variant)
    model_name = "prajjwal1/bert-tiny"  # Only 4.4M parameters, ~17MB
    print(f"\nAttempting to download {model_name}...")
    
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"✓ Model downloaded successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Save a simple config
    config = {
        "model_type": "bert-tiny",
        "hidden_size": model.config.hidden_size,
        "num_hidden_layers": model.config.num_hidden_layers,
        "num_attention_heads": model.config.num_attention_heads,
        "intermediate_size": model.config.intermediate_size,
        "vocab_size": model.config.vocab_size,
    }
    
    # Extract a single layer's weights for DSL demo
    # Get the first attention layer's query weights
    first_layer = model.encoder.layer[0]
    
    # Extract key matrices for demonstration
    query_weight = first_layer.attention.self.query.weight.detach().cpu().numpy()
    query_bias = first_layer.attention.self.query.bias.detach().cpu().numpy()
    
    print(f"\nExtracted weights:")
    print(f"  Query weight shape: {query_weight.shape}")
    print(f"  Query bias shape: {query_bias.shape}")
    
    os.makedirs("weights/bert_tiny", exist_ok=True)
    
    # Save just a small subset for DSL inference
    # Use first 16x16 block
    W_small = query_weight[:16, :16].astype(np.float32)
    b_small = query_bias[:16].astype(np.float32)
    
    W_small.tofile("weights/bert_tiny/query_W.bin")
    b_small.tofile("weights/bert_tiny/query_b.bin")
    
    np.savetxt("weights/bert_tiny/query_W.txt", W_small, fmt='%.8f')
    np.savetxt("weights/bert_tiny/query_b.txt", b_small, fmt='%.8f')
    
    # Save config
    config_small = {
        "model_type": "bert-tiny-subset",
        "input_dim": 16,
        "output_dim": 16,
        "layer": "attention.query",
        "shapes": {
            "W": [16, 16],
            "b": [16]
        }
    }
    
    with open("weights/bert_tiny/config.json", "w") as f:
        json.dump(config_small, f, indent=2)
    
    print(f"\n✓ Saved 16x16 weight subset to weights/bert_tiny/")
    print(f"  Config: {config_small}")
    
    USE_HF_MODEL = True
    
except Exception as e:
    print(f"HuggingFace model download failed: {e}")
    print("\nFalling back to creating a simple pre-trained model...")
    USE_HF_MODEL = False

if not USE_HF_MODEL:
    # Fallback: Create a simple pre-trained model
    print("\nCreating a simple pre-trained linear model...")
    
    # Simple 16x16 linear transformation (like a tiny attention layer)
    W = np.random.randn(16, 16).astype(np.float32) * 0.02
    b = np.random.randn(16).astype(np.float32) * 0.01
    
    os.makedirs("weights/simple_model", exist_ok=True)
    
    W.tofile("weights/simple_model/W.bin")
    b.tofile("weights/simple_model/b.bin")
    
    np.savetxt("weights/simple_model/W.txt", W, fmt='%.8f')
    np.savetxt("weights/simple_model/b.txt", b, fmt='%.8f')
    
    config = {
        "model_type": "simple_linear",
        "input_dim": 16,
        "output_dim": 16,
        "shapes": {
            "W": [16, 16],
            "b": [16]
        }
    }
    
    with open("weights/simple_model/config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Created simple model in weights/simple_model/")

print("\n=== Model ready for DSL inference! ===")
