#!/usr/bin/env python3
"""
Convert TTS model weights from PyTorch to binary format for DSL

Usage: python tts_weights_to_bin.py
"""
import torch
import numpy as np
import json
import os
from pathlib import Path

def main():
    weights_dir = Path("weights/tts")
    
    if not weights_dir.exists():
        print("Error: weights/tts directory not found")
        print("Run: python download_tts_model.py first")
        return
    
    config_path = weights_dir / "config.json"
    model_path = weights_dir / "model.pt"
    
    if not model_path.exists():
        print(f"Error: {model_path} not found")
        print("Run: python download_tts_model.py first")
        return
    
    print(f"Loading model from {model_path}...")
    state_dict = torch.load(model_path, map_location="cpu")
    
    print(f"\nAvailable weights:")
    for i, key in enumerate(sorted(state_dict.keys())[:20]):
        shape = state_dict[key].shape
        print(f"  {i+1}. {key}: {tuple(shape)}")
    
    if len(state_dict) > 20:
        print(f"  ... and {len(state_dict) - 20} more")
    
    # Export encoder layer 0 weights
    print("\n\nExporting encoder layer 0 weights to binary...")
    
    weights_to_export = [
        ("speecht5.encoder.wrapped_encoder.layers.0.attention.q_proj.weight", "encoder_q_proj_0.bin"),
        ("speecht5.encoder.wrapped_encoder.layers.0.attention.k_proj.weight", "encoder_k_proj_0.bin"),
        ("speecht5.encoder.wrapped_encoder.layers.0.attention.v_proj.weight", "encoder_v_proj_0.bin"),
        ("speecht5.encoder.wrapped_encoder.layers.0.attention.out_proj.weight", "encoder_attn_out_0.bin"),
        ("speecht5.encoder.wrapped_encoder.layers.0.feed_forward.intermediate_dense.weight", "encoder_ff1_0.bin"),
        ("speecht5.encoder.wrapped_encoder.layers.0.feed_forward.output_dense.weight", "encoder_ff2_0.bin"),
        ("speecht5.encoder.wrapped_encoder.layers.0.attention.q_proj.bias", "encoder_q_bias_0.bin"),
        ("speecht5.encoder.wrapped_encoder.layers.0.attention.k_proj.bias", "encoder_k_bias_0.bin"),
        ("speecht5.encoder.wrapped_encoder.layers.0.attention.v_proj.bias", "encoder_v_bias_0.bin"),
        ("speecht5.encoder.wrapped_encoder.layers.0.feed_forward.intermediate_dense.bias", "encoder_ff1_bias_0.bin"),
    ]
    
    exported = 0
    for pt_key, bin_name in weights_to_export:
        if pt_key in state_dict:
            weight = state_dict[pt_key].cpu().numpy().astype(np.float32)
            out_path = weights_dir / bin_name
            weight.tofile(out_path)
            print(f"  ✓ {bin_name}: {weight.shape}")
            exported += 1
        else:
            print(f"  ✗ {pt_key} not found in model")
    
    # Try to export text embeddings
    embed_keys = [
        "speecht5.encoder.prenet.embed_tokens.weight",
        "speecht5.text_encoder_prenet.embed_tokens.weight",
        "speecht5.encoder.wrapped_encoder.embed_tokens.weight",
    ]
    
    for key in embed_keys:
        if key in state_dict:
            weight = state_dict[key].cpu().numpy().astype(np.float32)
            out_path = weights_dir / "text_embed.bin"
            weight.tofile(out_path)
            print(f"  ✓ text_embed.bin: {weight.shape}")
            exported += 1
            break
    
    print(f"\n✓ Exported {exported} weight tensors to {weights_dir}/")
    print("\nNext steps:")
    print("1. Create DSL kernel: examples/tts_inference.ta")
    print("2. Compile: ./tensorasm examples/tts_inference.ta")

if __name__ == "__main__":
    main()
