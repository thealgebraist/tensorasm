import json
import sys

def main():
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("Error: config.json not found.")
        sys.exit(1)

    n_embd = config.get('n_embd', 768)
    n_layer = config.get('n_layer', 12)
    vocab_size = config.get('vocab_size', 50257)
    n_inner = config.get('n_inner', 3072)
    n_ctx = config.get('n_ctx', 1024) # Default GPT-2 context size

    print(f"// Generated from config.json")
    print(f"const VOCAB_SIZE = {vocab_size};")
    print(f"const N_EMBD = {n_embd};")
    print(f"const N_INNER = {n_inner};")
    print(f"const N_CTX = {n_ctx};")
    print(f"const N_LAYER = {n_layer};")
    print("")

    # Memory Spaces: Global for weights, TileReg for intermediate computations
    print(f"type EmbeddingTable = Tensor<f32, {{{vocab_size}, {n_embd}}}, Global, RowMajor>;")
    print(f"type PositionalEmbedding = Tensor<f32, {{{n_ctx}, {n_embd}}}, Global, RowMajor>;")
    print("")
    
    # QKV are often fused in GPT-2 weights as [3 * n_embd, n_embd] or similar
    print(f"type QKVWeight = Tensor<f32, {{{n_embd}, {3 * n_embd}}}, Global, RowMajor>;")
    print(f"type QKVBias = Tensor<f32, {{{3 * n_embd}}}, Global, RowMajor>;")
    print("")

    print(f"type AttnProjWeight = Tensor<f32, {{{n_embd}, {n_embd}}}, Global, RowMajor>;")
    print(f"type AttnProjBias = Tensor<f32, {{{n_embd}}}, Global, RowMajor>;")
    print("")

    print(f"type MLPWeight1 = Tensor<f32, {{{n_embd}, {n_inner}}}, Global, RowMajor>;")
    print(f"type MLPBias1 = Tensor<f32, {{{n_inner}}}, Global, RowMajor>;")
    print("")

    print(f"type MLPWeight2 = Tensor<f32, {{{n_inner}, {n_embd}}}, Global, RowMajor>;")
    print(f"type MLPBias2 = Tensor<f32, {{{n_embd}}}, Global, RowMajor>;")
    print("")

    print(f"type LayerNormWeight = Tensor<f32, {{{n_embd}}}, Global, RowMajor>;")
    print(f"type LayerNormBias = Tensor<f32, {{{n_embd}}}, Global, RowMajor>;")
    print("")

    # Register tiles for computations (e.g. 32x32 tiles)
    print(f"type Tile32x32 = Tensor<f32, {{32, 32}}, TileReg, RowMajor>;")
    print(f"type Vec32 = Tensor<f32, {{32}}, TileReg, RowMajor>;")

if __name__ == "__main__":
    main()
