import json
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python hf_to_ta.py config.json")
        sys.exit(1)
        
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
        
    d_model = cfg.get("n_embd", cfg.get("hidden_size", 768))
    n_layer = cfg.get("n_layer", cfg.get("num_hidden_layers", 12))
    n_head = cfg.get("n_head", cfg.get("num_attention_heads", 12))
    vocab_size = cfg.get("vocab_size", 50257)
    intermediate_size = cfg.get("n_inner", cfg.get("intermediate_size", d_model * 4))
    
    print(f"// Model: {cfg.get('model_type', 'unknown')}")
    print(f"const D_MODEL = {d_model};")
    print(f"const N_LAYER = {n_layer};")
    print(f"const N_HEAD = {n_head};")
    print(f"const VOCAB_SIZE = {vocab_size};")
    print(f"const D_HEAD = {d_model // n_head};")
    print(f"const D_FF = {intermediate_size};")
    print("")
    
    print("// Weights (Global Memory)")
    print(f"type W_Embed = Tensor<f32, {{VOCAB_SIZE, D_MODEL}}, Global, RowMajor>;")
    print(f"type W_QKV = Tensor<f32, {{D_MODEL, {d_model * 3}}}, Global, RowMajor>;")
    print(f"type W_AttnOut = Tensor<f32, {{D_MODEL, D_MODEL}}, Global, RowMajor>;")
    print(f"type W_FF1 = Tensor<f32, {{D_MODEL, D_FF}}, Global, RowMajor>;")
    print(f"type W_FF2 = Tensor<f32, {{D_FF, D_MODEL}}, Global, RowMajor>;")
    print(f"type W_Norm = Tensor<f32, {{D_MODEL}}, Global, RowMajor>;")
    print("")
    
    print("// Compute Tiles (Tile Registers)")
    print(f"type T_Hidden = Tensor<f32, {{D_MODEL}}, TileReg, RowMajor>;")
    print(f"type T_QKV = Tensor<f32, {{{d_model * 3}}}, TileReg, RowMajor>;")
    print(f"type T_Head = Tensor<f32, {{D_HEAD}}, TileReg, RowMajor>;")
    print(f"type T_FF = Tensor<f32, {{D_FF}}, TileReg, RowMajor>;")

if __name__ == "__main__":
    main()
