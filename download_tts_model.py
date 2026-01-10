#!/usr/bin/env python3
"""
Download a TTS model and prepare it for DSL inference

We'll use a simple TTS model from HuggingFace.
Options:
- facebook/fastspeech2-en-ljspeech (FastSpeech2)
- suno/bark-small (smaller Bark model)
- microsoft/speecht5_tts (SpeechT5)
"""
import torch
import numpy as np
import json
import os
from pathlib import Path
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

def download_speecht5_tts():
    """
    Download Microsoft's SpeechT5 TTS model
    """
    try:
        
        model_name = "microsoft/speecht5_tts"
        print(f"\nDownloading {model_name}...")
        
        processor = SpeechT5Processor.from_pretrained(model_name)
        model = SpeechT5ForTextToSpeech.from_pretrained(model_name)
        vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
        
        print(f"✓ Model downloaded successfully!")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Save config
        config = {
            "model_type": "speecht5_tts",
            "vocab_size": model.config.vocab_size,
            "hidden_size": model.config.hidden_size,
            "encoder_layers": model.config.encoder_layers,
            "decoder_layers": model.config.decoder_layers,
            "encoder_attention_heads": model.config.encoder_attention_heads,
            "decoder_attention_heads": model.config.decoder_attention_heads,
            "encoder_ffn_dim": model.config.encoder_ffn_dim,
            "decoder_ffn_dim": model.config.decoder_ffn_dim,
            "num_mel_bins": model.config.num_mel_bins,
            "max_text_positions": model.config.max_text_positions,
            "max_speech_positions": model.config.max_speech_positions,
        }
        
        os.makedirs("weights/tts", exist_ok=True)
        
        with open("weights/tts/config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"\nConfig saved to weights/tts/config.json")
        print(f"\nModel configuration:")
        for k, v in config.items():
            print(f"  {k}: {v}")
        
        # Print model structure to understand hierarchy
        print(f"\nModel structure:")
        for name, _ in model.named_modules():
            if 'encoder' in name and 'layer' in name and 'self_attn' in name:
                print(f"  {name}")
                break
        
        # Save full model for reference
        print(f"\nSaving model state dict...")
        torch.save(model.state_dict(), "weights/tts/model.pt")
        print(f"  ✓ Saved to weights/tts/model.pt")
        
        # List some weight keys
        print(f"\nSample weight keys:")
        state_dict = model.state_dict()
        for i, key in enumerate(sorted(state_dict.keys())[:10]):
            print(f"  {key}: {state_dict[key].shape}")

        
        return model, processor, vocoder, config
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def main():
    print("=== TTS Model Downloader ===")
    model, processor, vocoder, config = download_speecht5_tts()
    
    if model is not None:
        print("\n✓ Download complete!")
        print("Next steps:")
        print("1. Run: python tts_to_ta.py weights/tts/config.json > tts_types.ta")
        print("2. Run: python tts_weights_to_bin.py")
        print("3. Create TTS kernel in DSL: examples/tts_inference.ta")

if __name__ == "__main__":
    main()
