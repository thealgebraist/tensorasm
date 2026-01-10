#!/usr/bin/env python3
"""
Convert TTS model config.json to TensorCore DSL type definitions

Usage: python tts_to_ta.py weights/tts/config.json > tts_types.ta
"""
import json
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python tts_to_ta.py config.json")
        sys.exit(1)
        
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    
    model_type = cfg.get("model_type", "tts")
    vocab_size = cfg.get("vocab_size", 200)
    hidden_size = cfg.get("hidden_size", 768)
    encoder_layers = cfg.get("encoder_layers", 6)
    decoder_layers = cfg.get("decoder_layers", 6)
    encoder_attn_heads = cfg.get("encoder_attention_heads", 12)
    decoder_attn_heads = cfg.get("decoder_attention_heads", 12)
    encoder_ffn_dim = cfg.get("encoder_ffn_dim", 3072)
    decoder_ffn_dim = cfg.get("decoder_ffn_dim", 3072)
    num_mel_bins = cfg.get("num_mel_bins", 80)
    max_text_positions = cfg.get("max_text_positions", 450)
    max_speech_positions = cfg.get("max_speech_positions", 4096)
    
    d_head_enc = hidden_size // encoder_attn_heads
    d_head_dec = hidden_size // decoder_attn_heads
    
    print(f"// TTS Model: {model_type}")
    print(f"// Generated from config.json")
    print("")
    print(f"const VOCAB_SIZE = {vocab_size};")
    print(f"const HIDDEN_SIZE = {hidden_size};")
    print(f"const ENCODER_LAYERS = {encoder_layers};")
    print(f"const DECODER_LAYERS = {decoder_layers};")
    print(f"const ENCODER_HEADS = {encoder_attn_heads};")
    print(f"const DECODER_HEADS = {decoder_attn_heads};")
    print(f"const D_HEAD_ENC = {d_head_enc};")
    print(f"const D_HEAD_DEC = {d_head_dec};")
    print(f"const ENCODER_FFN = {encoder_ffn_dim};")
    print(f"const DECODER_FFN = {decoder_ffn_dim};")
    print(f"const MEL_BINS = {num_mel_bins};")
    print(f"const MAX_TEXT_LEN = {max_text_positions};")
    print(f"const MAX_SPEECH_LEN = {max_speech_positions};")
    print("")
    
    print("// ======== Encoder Weights (Global Memory) ========")
    print("")
    print("// Text Embedding")
    print(f"type W_TextEmbed = Tensor<f32, {{VOCAB_SIZE, HIDDEN_SIZE}}, Global, RowMajor>;")
    print(f"type W_PosEmbed = Tensor<f32, {{MAX_TEXT_LEN, HIDDEN_SIZE}}, Global, RowMajor>;")
    print("")
    
    print("// Encoder Layer Weights")
    print(f"type W_EncQ = Tensor<f32, {{HIDDEN_SIZE, HIDDEN_SIZE}}, Global, RowMajor>;")
    print(f"type W_EncK = Tensor<f32, {{HIDDEN_SIZE, HIDDEN_SIZE}}, Global, RowMajor>;")
    print(f"type W_EncV = Tensor<f32, {{HIDDEN_SIZE, HIDDEN_SIZE}}, Global, RowMajor>;")
    print(f"type W_EncAttnOut = Tensor<f32, {{HIDDEN_SIZE, HIDDEN_SIZE}}, Global, RowMajor>;")
    print(f"type W_EncFF1 = Tensor<f32, {{HIDDEN_SIZE, ENCODER_FFN}}, Global, RowMajor>;")
    print(f"type W_EncFF2 = Tensor<f32, {{ENCODER_FFN, HIDDEN_SIZE}}, Global, RowMajor>;")
    print(f"type B_EncQ = Tensor<f32, {{HIDDEN_SIZE}}, Global, RowMajor>;")
    print(f"type B_EncK = Tensor<f32, {{HIDDEN_SIZE}}, Global, RowMajor>;")
    print(f"type B_EncV = Tensor<f32, {{HIDDEN_SIZE}}, Global, RowMajor>;")
    print("")
    
    print("// ======== Decoder Weights (Global Memory) ========")
    print("")
    print("// Decoder Layer Weights")
    print(f"type W_DecQ = Tensor<f32, {{HIDDEN_SIZE, HIDDEN_SIZE}}, Global, RowMajor>;")
    print(f"type W_DecK = Tensor<f32, {{HIDDEN_SIZE, HIDDEN_SIZE}}, Global, RowMajor>;")
    print(f"type W_DecV = Tensor<f32, {{HIDDEN_SIZE, HIDDEN_SIZE}}, Global, RowMajor>;")
    print(f"type W_DecAttnOut = Tensor<f32, {{HIDDEN_SIZE, HIDDEN_SIZE}}, Global, RowMajor>;")
    print(f"type W_DecFF1 = Tensor<f32, {{HIDDEN_SIZE, DECODER_FFN}}, Global, RowMajor>;")
    print(f"type W_DecFF2 = Tensor<f32, {{DECODER_FFN, HIDDEN_SIZE}}, Global, RowMajor>;")
    print("")
    
    print("// Speech Decoder Head")
    print(f"type W_MelProj = Tensor<f32, {{HIDDEN_SIZE, MEL_BINS}}, Global, RowMajor>;")
    print(f"type B_MelProj = Tensor<f32, {{MEL_BINS}}, Global, RowMajor>;")
    print("")
    
    print("// ======== Compute Tiles (Tile Registers) ========")
    print("")
    print(f"type T_Hidden = Tensor<f32, {{HIDDEN_SIZE}}, TileReg, RowMajor>;")
    print(f"type T_EncFFN = Tensor<f32, {{ENCODER_FFN}}, TileReg, RowMajor>;")
    print(f"type T_DecFFN = Tensor<f32, {{DECODER_FFN}}, TileReg, RowMajor>;")
    print(f"type T_Mel = Tensor<f32, {{MEL_BINS}}, TileReg, RowMajor>;")
    print(f"type T_Head = Tensor<f32, {{D_HEAD_ENC}}, TileReg, RowMajor>;")
    print("")
    
    print("// ======== Activation Types ========")
    print("")
    print(f"type T_TextSeq = Tensor<f32, {{MAX_TEXT_LEN, HIDDEN_SIZE}}, Global, RowMajor>;")
    print(f"type T_MelSeq = Tensor<f32, {{MAX_SPEECH_LEN, MEL_BINS}}, Global, RowMajor>;")

if __name__ == "__main__":
    main()
