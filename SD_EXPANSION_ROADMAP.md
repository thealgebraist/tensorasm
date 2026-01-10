# TensorASM DSL Expansion Roadmap: Full Stable Diffusion 1.5 Inference

## Goal
Expand the TensorASM DSL to support complete Stable Diffusion 1.5 inference, from text prompt to 512x512 generated image.

---

## Phase 1: Core Intrinsics & Operations (Tasks 1-9)

### Task 1: Implement 2D Convolution Intrinsic
- **Priority**: Critical
- **Description**: Add CONV2D intrinsic for spatial convolutions (kernel_size, stride, padding)
- **Current Gap**: No convolution support; SD UNet uses extensive conv layers
- **Success Criteria**: 
  - Parse `CONV2D(out, input, weight, bias, stride, padding)`
  - Generate optimized Eigen/SIMD code for 2D convolutions
  - Support multiple kernel sizes (1x1, 3x3, 5x5)
- **Dependencies**: None
- **Estimated Effort**: 2 weeks

### Task 2: Implement FFT-Based Convolution
- **Priority**: High
- **Description**: Add FFT/IFFT intrinsics for Fourier-domain convolutions
- **Current Gap**: Spatial convolution can be slow for large kernels; FFT convolution is often faster and simpler
- **Success Criteria**:
  - Add `FFT(out, input)` and `IFFT(out, input)` intrinsics
  - Implement element-wise multiplication in frequency domain
  - Automatic selection: FFT for kernel_size ≥ 7x7, spatial for smaller
  - Integrate with existing CONV2D interface
  - Use FFTW or similar optimized library backend
- **Dependencies**: Task 1
- **Estimated Effort**: 1.5 weeks

### Task 3: Add GroupNorm Intrinsic
- **Priority**: Critical
- **Description**: Implement GROUP_NORM for normalization layers
- **Current Gap**: SD uses GroupNorm extensively (32 groups)
- **Success Criteria**:
  - `GROUP_NORM(out, input, gamma, beta, num_groups, eps)`
  - Correct mean/variance computation per group
  - Support different group sizes
- **Dependencies**: Task 1
- **Estimated Effort**: 1 week

### Task 4: Implement SiLU/Swish Activation
- **Priority**: High
- **Description**: Add SiLU activation to ACT intrinsic (type=2)
- **Current Gap**: SD UNet uses SiLU, currently only have GELU
- **Success Criteria**:
  - Extend `ACT(tensor, 2)` for SiLU
  - Optimize computation: x * sigmoid(x)
- **Dependencies**: None
- **Estimated Effort**: 3 days

### Task 5: Add LayerNorm Intrinsic
- **Priority**: High
- **Description**: Implement LAYER_NORM for transformer blocks
- **Current Gap**: Cross-attention uses LayerNorm
- **Success Criteria**:
  - `LAYER_NORM(out, input, gamma, beta, eps)`
  - Correct normalization across feature dimension
- **Dependencies**: None
- **Estimated Effort**: 1 week

### Task 6: Implement Batched Matrix Multiplication
- **Priority**: Critical
- **Description**: Extend MMUL to support batched operations
- **Current Gap**: Attention requires batch x heads x seq x dim operations
- **Success Criteria**:
  - `MMUL_BATCH(out, a, b, batch_size)`
  - Efficient batching for multi-head attention
  - Support broadcasting semantics
- **Dependencies**: None
- **Estimated Effort**: 1 week

### Task 7: Add Upsampling/Interpolation Intrinsic
- **Priority**: High
- **Description**: Implement UPSAMPLE for spatial upsampling
- **Current Gap**: UNet decoder needs 2x upsampling
- **Success Criteria**:
  - `UPSAMPLE(out, input, scale_factor, mode)`
  - Support nearest and bilinear interpolation
  - Efficient implementation for 2x, 4x scaling
- **Dependencies**: Task 1
- **Estimated Effort**: 1 week

### Task 8: Implement Concatenation Intrinsic
- **Priority**: High
- **Description**: Add CONCAT for tensor concatenation
- **Current Gap**: UNet skip connections need channel concatenation
- **Success Criteria**:
  - `CONCAT(out, input1, input2, axis)`
  - Support channel and spatial concatenation
  - Zero-copy when possible
- **Dependencies**: None
- **Estimated Effort**: 4 days

### Task 9: Add Slicing/Indexing Operations
- **Priority**: Medium
- **Description**: Implement tensor slicing for skip connections
- **Current Gap**: Need to extract/combine tensor regions
- **Success Criteria**:
  - `SLICE(out, input, start, end, axis)`
  - Efficient memory layout handling
- **Dependencies**: None
- **Estimated Effort**: 1 week

---

## Phase 2: Advanced Attention Mechanisms (Tasks 10-13)

### Task 10: Implement Multi-Head Attention Kernel
- **Priority**: Critical
- **Description**: Build complete multi-head attention in DSL
- **Current Gap**: Only have single query projection
- **Success Criteria**:
  - Q, K, V projections with multiple heads
  - Scaled dot-product attention
  - Output projection
  - Support 8 heads at 64-dim each
- **Dependencies**: Tasks 6, 9
- **Estimated Effort**: 2 weeks

### Task 11: Add Cross-Attention Kernel
- **Priority**: Critical
- **Description**: Implement cross-attention between image and text
- **Current Gap**: SD needs image x text cross-attention
- **Success Criteria**:
  - Separate Q (from image) and K,V (from text)
  - Proper masking support
  - Integration with transformer blocks
- **Dependencies**: Task 10
- **Estimated Effort**: 1.5 weeks

### Task 12: Implement Self-Attention Kernel
- **Priority**: High
- **Description**: Self-attention for spatial features
- **Current Gap**: UNet transformer blocks use self-attention
- **Success Criteria**:
  - Q, K, V from same input
  - Position encoding support
  - Efficient computation for 64x64 spatial resolution
- **Dependencies**: Task 10
- **Estimated Effort**: 1 week

### Task 13: Optimize Attention Memory Layout
- **Priority**: Medium
- **Description**: Implement efficient memory layout for attention
- **Current Gap**: Attention has complex memory access patterns
- **Success Criteria**:
  - Minimize cache misses
  - Support tiled computation
  - Fused kernels where possible
- **Dependencies**: Tasks 10-12
- **Estimated Effort**: 2 weeks

---

## Phase 3: Model Architecture Support (Tasks 14-19)

### Task 14: Implement ResNet Block Kernel
- **Priority**: Critical
- **Description**: Build complete ResNet block used in SD UNet
- **Current Gap**: UNet heavily uses ResNet blocks
- **Success Criteria**:
  - Two conv layers with GroupNorm and SiLU
  - Skip connection with optional projection
  - Time embedding injection
- **Dependencies**: Tasks 1-4
- **Estimated Effort**: 2 weeks

### Task 15: Build Transformer Block Kernel
- **Priority**: Critical
- **Description**: Complete transformer block with self and cross-attention
- **Current Gap**: Need full transformer for SD UNet
- **Success Criteria**:
  - Self-attention + cross-attention + FFN
  - Layer norms and residual connections
  - Supports both spatial and sequence inputs
- **Dependencies**: Tasks 5, 10-12
- **Estimated Effort**: 2.5 weeks

### Task 16: Implement Downsampling Block
- **Priority**: High
- **Description**: UNet encoder downsampling (conv stride=2 or pooling)
- **Current Gap**: Need to reduce spatial resolution
- **Success Criteria**:
  - 2x spatial downsampling
  - Preserve channels or expand
  - Efficient implementation
- **Dependencies**: Task 1
- **Estimated Effort**: 1 week

### Task 17: Implement Upsampling Block
- **Priority**: High
- **Description**: UNet decoder upsampling with skip connections
- **Current Gap**: Need to increase spatial resolution
- **Success Criteria**:
  - 2x spatial upsampling
  - Concatenate skip connections
  - Conv layers after upsampling
- **Dependencies**: Tasks 7-8
- **Estimated Effort**: 1.5 weeks

### Task 18: Add Time Embedding Support
- **Priority**: Critical
- **Description**: Implement sinusoidal time embeddings and injection
- **Current Gap**: SD diffusion needs timestep conditioning
- **Success Criteria**:
  - Sinusoidal embedding generation
  - MLP for time projection
  - Injection into ResNet blocks
- **Dependencies**: None
- **Estimated Effort**: 1 week

### Task 19: Build VAE Decoder Architecture
- **Priority**: Critical
- **Description**: Implement VAE decoder for latent->image
- **Current Gap**: Need to decode 4-channel latents to RGB
- **Success Criteria**:
  - 4x upsampling path (4->8->16->32->64->512)
  - ResNet blocks with attention at 32x32
  - Final conv to RGB output
- **Dependencies**: Tasks 1-4, 7, 16-17
- **Estimated Effort**: 3 weeks

---

## Phase 4: Memory & Scheduling (Tasks 20-25)

### Task 20: Implement Dynamic Tensor Allocation
- **Priority**: High
- **Description**: Support runtime tensor creation based on config
- **Current Gap**: All tensors currently static
- **Success Criteria**:
  - Parse dimensions from config.json
  - Allocate appropriate memory
  - Support different resolutions (256, 512, 1024)
- **Dependencies**: None
- **Estimated Effort**: 2 weeks

### Task 21: Add Memory Pool Management
- **Priority**: Medium
- **Description**: Efficient memory reuse across operations
- **Current Gap**: Each kernel allocates independently
- **Success Criteria**:
  - Analyze tensor lifetimes
  - Reuse memory for intermediate tensors
  - Reduce peak memory usage by 50%
- **Dependencies**: Task 20
- **Estimated Effort**: 2 weeks

### Task 22: Implement Computation Graph
- **Priority**: High
- **Description**: Build dependency graph for kernel execution
- **Current Gap**: Sequential execution only
- **Success Criteria**:
  - Parse kernel dependencies
  - Generate execution schedule
  - Support parallel execution where possible
- **Dependencies**: None
- **Estimated Effort**: 2.5 weeks

### Task 23: Add Operator Fusion
- **Priority**: Medium
- **Description**: Fuse compatible operations (conv+norm+act)
- **Current Gap**: Each operation is separate kernel
- **Success Criteria**:
  - Detect fusable patterns
  - Generate fused kernels
  - 20-30% speedup on common patterns
- **Dependencies**: Task 22
- **Estimated Effort**: 3 weeks

### Task 24: Implement Gradient Checkpointing
- **Priority**: Low
- **Description**: Trade compute for memory in large models
- **Current Gap**: May need for 512x512 inference
- **Success Criteria**:
  - Selective intermediate tensor storage
  - Recompute on demand
  - Configurable checkpoint frequency
- **Dependencies**: Tasks 21-22
- **Estimated Effort**: 2 weeks

### Task 25: Add Streaming Execution
- **Priority**: Medium
- **Description**: Pipeline execution for better throughput
- **Current Gap**: Batch size 1 only, synchronous
- **Success Criteria**:
  - Overlap data transfer and compute
  - Support mini-batching
  - Async kernel execution
- **Dependencies**: Task 22
- **Estimated Effort**: 2 weeks

---

## Phase 5: Complete Model Integration (Tasks 26-29)

### Task 26: Integrate CLIP Text Encoder
- **Priority**: Critical
- **Description**: Load and run CLIP for text embeddings
- **Current Gap**: Need to encode prompts
- **Success Criteria**:
  - Tokenizer integration
  - CLIP transformer in DSL (12 layers)
  - Output 77x768 embeddings
- **Dependencies**: Tasks 5, 10, 12
- **Estimated Effort**: 3 weeks

### Task 27: Build Complete UNet Pipeline
- **Priority**: Critical
- **Description**: Assemble full UNet with all components
- **Current Gap**: Only have single attention layer
- **Success Criteria**:
  - 4 down blocks, mid block, 4 up blocks
  - Cross-attention at each resolution
  - Time embedding throughout
  - 320/640/1280 channel progression
- **Dependencies**: Tasks 14-18
- **Estimated Effort**: 4 weeks

### Task 28: Implement Noise Scheduler
- **Priority**: Critical
- **Description**: DDPM/DDIM scheduling for diffusion steps
- **Current Gap**: No diffusion sampling
- **Success Criteria**:
  - DDPM noise schedule (1000 steps)
  - DDIM for fast sampling (50 steps)
  - Variance and noise computation
- **Dependencies**: None
- **Estimated Effort**: 1.5 weeks

### Task 29: Integrate VAE Encoder
- **Priority**: Medium
- **Description**: Encode images to latent space (for img2img)
- **Current Gap**: Only have decoder
- **Success Criteria**:
  - RGB -> 4-channel latent encoding
  - Inverse of decoder architecture
  - KL divergence sampling
- **Dependencies**: Task 19
- **Estimated Effort**: 2 weeks

---

## Phase 6: Optimization & Production (Tasks 30-33)

### Task 30: Implement FP16/BF16 Support
- **Priority**: High
- **Description**: Mixed precision for 2x speedup
- **Current Gap**: FP32 only
- **Success Criteria**:
  - FP16 compute, FP32 accumulate
  - Automatic loss scaling
  - 1.8-2.5x faster inference
- **Dependencies**: None
- **Estimated Effort**: 2 weeks

### Task 31: Add Quantization (INT8)
- **Priority**: Medium
- **Description**: 8-bit quantization for weights
- **Current Gap**: FP32 weights require 3.4GB
- **Success Criteria**:
  - Per-channel or per-tensor quantization
  - Calibration on sample data
  - <5% quality degradation
  - 4x memory reduction
- **Dependencies**: Task 30
- **Estimated Effort**: 3 weeks

### Task 32: Platform-Specific Optimizations
- **Priority**: High
- **Description**: Target-specific code generation
- **Current Gap**: Generic Eigen code
- **Success Criteria**:
  - Apple AMX tiles for M-series
  - CUDA kernels for NVIDIA
  - AVX512 for x86 servers
  - 2-4x speedup on target hardware
- **Dependencies**: All previous tasks
- **Estimated Effort**: 6 weeks

### Task 33: Build End-to-End Pipeline
- **Priority**: Critical
- **Description**: Complete text-to-image generation
- **Current Gap**: Only component tests
- **Success Criteria**:
  - Prompt -> CLIP -> UNet (50 steps) -> VAE -> Image
  - 512x512 generation in <30s on M2
  - Match quality of diffusers library
  - Support negative prompts and guidance scale
- **Dependencies**: All previous tasks
- **Estimated Effort**: 3 weeks

---

## Summary Statistics

- **Total Tasks**: 33
- **Critical Priority**: 11 tasks
- **High Priority**: 10 tasks  
- **Medium Priority**: 9 tasks
- **Low Priority**: 2 tasks

**Estimated Timeline**: 
- Phase 1-2: 3-4 months (foundational work)
- Phase 3-4: 4-5 months (architecture & optimization)
- Phase 5-6: 3-4 months (integration & production)
- **Total: 10-13 months** with 2-3 engineers

**Current Capabilities**:
- ✓ Basic matrix operations (LOAD, STORE, MMUL)
- ✓ Simple activations (GELU)
- ✓ Single attention query projection
- ✓ File I/O
- ✓ Eigen backend

**Target Capabilities**:
- Full Stable Diffusion 1.5 inference
- Text prompt -> 512x512 image in ~30 seconds
- Optimized for Apple Silicon, x86, and NVIDIA
- Mixed precision support
- Production-ready performance

---

## Risk Mitigation

1. **Memory constraints**: Implement Tasks 19-20 early, test on 512x512
2. **Performance gaps**: Benchmark after each phase, optimize hot paths
3. **Numerical stability**: Validate against PyTorch at each component
4. **Integration complexity**: Build incremental test suite from day 1
5. **Hardware portability**: Abstract intrinsics, target one platform first

---

## Success Metrics

- ✓ Generate 512x512 images from text prompts
- ✓ Match visual quality of HuggingFace Diffusers
- ✓ <30 second generation time on M2 MacBook
- ✓ <4GB memory usage
- ✓ All 33 tasks validated with unit tests
- ✓ Full SD 1.5 checkpoint loadable from .safetensors
