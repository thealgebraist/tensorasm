# Implementation Instructions for TensorCore DSL

**Status: INFERENCE EXPANSION**

GOAL: Run inference on an actual huggingface model using only tensorcore dsl.

## 1. Accomplishments
- [x] **Type System**: `f32`, `f16`, `bf16`, `int8`, `int32` supported. Fixed-size enforcement ensures no dynamic allocation in generated kernels.
- [x] **Semantic Analysis**: Strict checks for hardware affinity (`TileReg` for MMUL/MADD) and shape compatibility.
- [x] **Compiler**: Generates efficient Eigen code with scalar broadcasting, row-based indexing, and stack-overflow protection for large tensors.
- [x] **Interpreter**: Fully synchronized with compiler logic, supporting `batch` loops, `LOOKUP`, and complex binary expressions.
- [x] **Robustness**: 100% pass rate on regression tests.

## 2. Inference Tasks (32 Tasks)
- [x] 1. **Target Model Selection**: Chosen **GPT-2-small** (124M).
- [x] 2. **Configuration Parser**: Created `hf_to_ta.py` to generate `.ta` types.
- [x] 3. **Weight Conversion**: Created `hf_weights_to_bin.py` to generate binary weights.
- [x] 4. **Binary Weight Loading**: Updated `hw::LOAD` and `main` generation in the compiler.
- [x] 5. **Interpreter Binary Loading**: Updated the interpreter's initialization to read from `weights/*.bin`.
- [x] 6. **EXP Intrinsic**: Added `EXP` intrinsic.
- [x] 7. **EXP Backend**: Implemented `EXP` in Eigen backend.
- [x] 8. **EXP Interpreter**: Implemented `EXP` in Interpreter.
- [x] 9. **SQRT Intrinsic**: Added `SQRT` intrinsic.
- [x] 10. **SQRT Backend**: Implemented `SQRT` in Eigen backend.
- [x] 11. **SQRT Interpreter**: Implemented `SQRT` in Interpreter.
- [x] 12. **TRANSPOSE Intrinsic**: Added `TRANSPOSE` intrinsic.
- [x] 13. **TRANSPOSE Backend**: Implemented `TRANSPOSE` in Eigen backend.
- [x] 14. **TRANSPOSE Interpreter**: Implemented `TRANSPOSE` in Interpreter.
- [ ] 15. **LayerNorm Composition**: Implement LayerNormalization in DSL.
- [ ] 16. **RMSNorm Composition**: Implement RMSNorm in DSL.
- [x] 17. **ACT Intrinsic**: Added `ACT` (Activation) intrinsic for GELU/SILU.
- [x] 18. **ACT Backend**: Implemented `ACT` in backends.
- [ ] 19. **Attention Kernel**: Compose a Multi-Head Attention kernel in DSL.
- [ ] 20. **MLP Kernel**: Compose an MLP kernel in DSL.
- [ ] 21. **Transformer Layer**: Integrate Attn + MLP + Norms in DSL.
- [ ] 22. **Sequential Layers**: Implement a mechanism to loop over Transformer layers.
- [ ] 23. **KV Cache Management**: Define static tensors for KV caching in the DSL.
- [ ] 24. **Positional Embeddings**: Implement RoPE or Learned Absolute Embeddings in DSL.
- [ ] 25. **Top-level Inference Kernel**: Connect Embedding -> Layers -> Head in DSL.
- [ ] 26. **Tokenizer Bridge**: Implement a wrapper to map text to `TokenIdx`.
- [ ] 27. **Decoding Loop**: Implementation of token generation loop in C++.
- [ ] 28. **Unit Verification**: Compare DSL output vs PyTorch for a single layer.
- [ ] 29. **Full Verification**: Compare full model output.
- [ ] 30. **Tiling Optimization**: Add explicit tiling for Attention QK^T.
- [ ] 31. **SIMD Intrinsics**: Add hardware-specific instructions (AMX/Neon) beyond Eigen.
- [ ] 32. **CLI Deployment**: End-to-end text generation CLI using the compiled DSL.

---
**Working on**: Task 15 & 16 - Implementing Norm compositions in DSL.

## 3. Python Data Generation Tasks (16)
- [ ] **Task 1**: Tokenize a sample prompt using the Hugging Face tokenizer and export token IDs for DSL ingestion.
- [ ] **Task 2**: Extract embeddings for those tokens via HF and serialize as DSL-compatible tensors.
- [ ] **Task 3**: Compute attention weights (QK^T / softmax) for one block and export matrices.
- [ ] **Task 4**: Generate logits via HF head and store both raw logits and normalized softmax outputs.
- [ ] **Task 5**: Extract positional encodings (RoPE or learned) from HF config and store them for DSL embedding.
- [ ] **Task 6**: Sample the top-1 prediction for the prompt (reference argmax).
- [ ] **Task 7**: Log per-token intermediate activations to compare against DSL batch loops.
- [ ] **Task 8**: Serialize HF model metadata (vocab size, hidden size, num heads) for DSL type generation.
- [ ] **Task 9**: Automate downloading/preparing HF checkpoints and converting them to binary blobs.
- [ ] **Task 10**: Normalize HF embeddings (layer norm) and output normalized tensors.
- [ ] **Task 11**: Save simulated global memory state (e.g., embedding table snapshots) for LOAD tests.
- [ ] **Task 12**: Export tokenizer merges/vocab (BPE) so DSL LOOKUP can mirror HF tokenization.
- [ ] **Task 13**: Record softmax sum validations (should equal 1.0) for each prompt.
- [ ] **Task 14**: Bundle multiple prompts/responses into dataset files for DSL loop testing.
- [ ] **Task 15**: Compute cosine similarity between HF and DSL outputs for regression verification.
- [ ] **Task 16**: Package all Python-generated data (tokens, weights, metadata) into an archive for reuse.