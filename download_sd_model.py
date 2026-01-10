#!/usr/bin/env python3
"""
Download Stable Diffusion 1.5 and extract weights for DSL inference.
Focus on extracting a manageable subset of the model.
"""

import torch
import numpy as np
from diffusers import StableDiffusionPipeline
import json
import os

print("=" * 70)
print("Downloading Stable Diffusion 1.5")
print("=" * 70)

model_id = "runwayml/stable-diffusion-v1-5"

print(f"\nLoading: {model_id}")
print("This may take a few minutes...")

try:
    # Load only the UNet component (lighter than full pipeline)
    from diffusers import UNet2DConditionModel
    
    print("Loading UNet model (main diffusion component)...")
    unet = UNet2DConditionModel.from_pretrained(
        model_id,
        subfolder="unet",
        torch_dtype=torch.float32
    )
    
    print("Loading UNet model (main diffusion component)...")
    unet = UNet2DConditionModel.from_pretrained(
        model_id,
        subfolder="unet",
        torch_dtype=torch.float32
    )
    
    print("\n✓ Stable Diffusion UNet loaded successfully!")
    
    # Count parameters
    unet_params = sum(p.numel() for p in unet.parameters())
    
    print(f"\nUNet Model:")
    print(f"  Parameters: {unet_params:,}")
    
    # Extract a small subset of weights from the UNet
    # Focus on the first cross-attention layer (text-to-image attention)
    print("\n" + "=" * 70)
    print("Extracting UNet Cross-Attention Weights")
    print("=" * 70)
    
    # Get first down block's cross-attention layer
    down_block = unet.down_blocks[1]  # Second down block has cross-attention
    
    # Find cross-attention layer
    if hasattr(down_block, 'attentions') and down_block.attentions is not None:
        attn_block = down_block.attentions[0]
        transformer_block = attn_block.transformer_blocks[0]
        cross_attn = transformer_block.attn2  # Cross-attention (text to image)
        
        # Extract query projection weights
        q_weight = cross_attn.to_q.weight.data.cpu().numpy()  # Shape: (out_dim, in_dim)
        q_bias = cross_attn.to_q.bias.data.cpu().numpy() if cross_attn.to_q.bias is not None else None
        
        print(f"\nCross-Attention Query Projection:")
        print(f"  Weight shape: {q_weight.shape}")
        print(f"  Bias shape: {q_bias.shape if q_bias is not None else 'None'}")
        
        # Take a 16x16 subset for DSL
        W = q_weight[:16, :16].astype(np.float32)
        b = q_bias[:16].astype(np.float32) if q_bias is not None else np.zeros(16, dtype=np.float32)
        
        print(f"\nExtracted subset:")
        print(f"  W: {W.shape}")
        print(f"  b: {b.shape}")
        
        # Create output directory
        os.makedirs("weights/stable_diffusion", exist_ok=True)
        
        # Save weights
        W.tofile("weights/stable_diffusion/cross_attn_q_W.bin")
        b.tofile("weights/stable_diffusion/cross_attn_q_b.bin")
        
        # Save as text
        with open("weights/stable_diffusion/cross_attn_q_W.txt", "w") as f:
            for row in W:
                f.write(" ".join([f"{v:.8f}" for v in row]) + "\n")
        
        with open("weights/stable_diffusion/cross_attn_q_b.txt", "w") as f:
            for val in b:
                f.write(f"{val:.8f}\n")
        
        # Save config
        config = {
            "model_type": "stable-diffusion-v1-5",
            "component": "unet.down_blocks[1].attentions[0].transformer_blocks[0].attn2",
            "layer": "cross_attention_query",
            "input_dim": 16,
            "output_dim": 16,
            "unet_params": int(unet_params),
            "shapes": {
                "W": [16, 16],
                "b": [16]
            }
        }
        
        with open("weights/stable_diffusion/config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        print("\n" + "=" * 70)
        print("✓ Stable Diffusion weights saved!")
        print("=" * 70)
        print(f"  Location: weights/stable_diffusion/")
        print(f"  Files: cross_attn_q_W.bin ({os.path.getsize('weights/stable_diffusion/cross_attn_q_W.bin')} bytes)")
        print(f"         cross_attn_q_b.bin ({os.path.getsize('weights/stable_diffusion/cross_attn_q_b.bin')} bytes)")
        print(f"  Weight sample: {W[0, :3]}")
        print(f"  Bias sample: {b[:3]}")
        
        print("\n✓ Ready for DSL inference!")
        print("  Extracted: Cross-attention query projection from SD 1.5 UNet")
        print("  These are authentic Stable Diffusion 1.5 weights")
    
    else:
        print("\n⚠ Could not find expected attention structure")
        print("  Model architecture may have changed")

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
