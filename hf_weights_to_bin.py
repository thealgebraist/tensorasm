import numpy as np
import sys
import os
import json

def main():
    if len(sys.argv) < 2:
        print("Usage: python hf_weights_to_bin.py config.json [output_dir]")
        sys.exit(1)
        
    config_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "weights"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(config_path) as f:
        cfg = json.load(f)
        
    d_model = cfg.get("n_embd", cfg.get("hidden_size", 768))
    n_layer = cfg.get("n_layer", cfg.get("num_hidden_layers", 12))
    vocab_size = cfg.get("vocab_size", 50257)
    intermediate_size = cfg.get("n_inner", cfg.get("intermediate_size", d_model * 4))
    
    def save(name, shape):
        path = os.path.join(output_dir, f"{name}.bin")
        print(f"Saving {name} {shape} to {path}")
        # Initialize with small random values to avoid NANs in SOFTMAX
        data = (np.random.randn(*shape) * 0.01).astype(np.float32)
        data.tofile(path)

    save("wte", (vocab_size, d_model))
    save("wpe", (1024, d_model)) # Positional
    
    for i in range(n_layer):
        save(f"l{i}_ln1_w", (d_model,))
        save(f"l{i}_ln1_b", (d_model,))
        save(f"l{i}_qkv_w", (d_model, d_model * 3))
        save(f"l{i}_qkv_b", (d_model * 3,))
        save(f"l{i}_attn_out_w", (d_model, d_model))
        save(f"l{i}_attn_out_b", (d_model,))
        save(f"l{i}_ln2_w", (d_model,))
        save(f"l{i}_ln2_b", (d_model,))
        save(f"l{i}_ff1_w", (d_model, intermediate_size))
        save(f"l{i}_ff1_b", (intermediate_size,))
        save(f"l{i}_ff2_w", (intermediate_size, d_model))
        save(f"l{i}_ff2_b", (d_model,))

    save("ln_f_w", (d_model,))
    save("ln_f_b", (d_model,))

if __name__ == "__main__":
    main()
