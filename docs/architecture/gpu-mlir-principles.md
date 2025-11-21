# ⚡ GPU & MLIR Principles for the Morphogen Stack

> Why they matter, and how they shape the kernel/compiler architecture

**Version:** 1.0
**Last Updated:** 2025-11-13
**Status:** Design Guidelines

---

## Introduction

Morphogen's compiler ultimately lowers typed, domain-rich graphs (streams, fields, transforms) into MLIR, then into GPU code (SPIR-V, Metal, CUDA). That means Morphogen must be structured so that its high-level semantics map cleanly to MLIR's structured IR and then to GPU SIMT/SIMD hardware realities.

These design notes explain how the GPU checklist affects Morphogen's kernel semantics, Transform Dialect, operator registry metadata, and lowering strategies.

**This document is not generic MLIR advice** — it reframes GPU/SIMT concepts as design rules for how Morphogen should structure lowering, operator templates, and IR generation.

---

## Design Principles

### 1. Express Parallelism Structurally, Not Implicitly

**Principle:** MLIR's GPU and affine dialects expect structured parallel loops. Morphogen must therefore ensure that operators, transforms, and fields expose their iteration spaces explicitly.

**Implications for Morphogen:**

- **Stream / Field types** must include shape, rate, domain, so a lowering pass can derive explicit loop bounds.

- **Transform Dialect ops** (FFT, DCT, k-space) must expand to regular affine loops or GPU block-mappable tiles.

- **Kernel scheduling metadata** must retain enough static shape and rate information to support block/thread mapping.

**Why This Matters:**

SIMT GPUs execute threads as uniform SIMD warps. If Morphogen emitted "opaque kernels," MLIR couldn't tile or map them correctly.

**Example:**

```mlir
// Good: explicit parallel structure
affine.parallel (%i, %j) = (0, 0) to (%N, %M) {
  %val = affine.load %field[%i, %j]
  %result = math.sin %val
  affine.store %result, %out[%i, %j]
}

// Bad: opaque function call
call @process_field(%field, %out) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
```

---

### 2. Operator Registry Must Encode Lowering-Friendly Metadata

**Principle:** For GPUs, MLIR needs static loop bounds, memory access patterns, vector widths, tile sizes, and shared-memory usage.

**Implications for Morphogen:**

Operator definitions in Morphogen's registry should include structural hints:

```json
{
  "name": "lpf",
  "category": "filter",
  "lowering": {
    "dialect": "morphogen.signal",
    "template": "lpf_svf",
    "tile_hint": [32, 1],
    "vector_hint": 4,
    "memory": "coalesced"
  }
}
```

**Why This Matters:**

Morphogen ops remain "semantic" at the DSL level, but their registry entries encode just enough structure for MLIR to tile → vectorize → GPU-map them.

**Registry Fields:**

| Field | Purpose | Example |
|-------|---------|---------|
| `tile_hint` | Suggested tile dimensions for GPU blocks | `[32, 8]` |
| `vector_hint` | SIMD width for vectorization | `4`, `8`, `16` |
| `memory` | Memory access pattern | `"coalesced"`, `"strided"`, `"random"` |
| `shared_mem` | Use shared memory | `true`, `false` |
| `loop_bounds` | Static bounds if known | `"[N, M, K]"` |

---

### 3. Transform Dialect Must Remain Intentionally Regular

**Principle:** GPU performance collapses when transforms are irregular. Morphogen's transform grammar already aims for explicit, parameterized operations.

**Implications for Morphogen:**

```morphogen
transform.to(... domain="frequency", method="fft")
transform.reparam(x, mapping)
```

To map well to GPUs:

- **FFT/STFT/DCT** must expand to static, tileable loops.

- **Reparam mappings** should resolve to affine index transforms whenever possible.

- **Wavelets, graph-spectral transforms**, etc. must declare static kernel sizes or structured decomposition.

**Why This Matters:**

MLIR's GPU pipeline is happiest when transforms resolve into predictable nested loops.

**Transform Structure:**

```
High-level transform:
  transform.to(signal, domain="frequency", method="fft", size=1024)

Lowered to structured loops:
  for stage in [0..log2(N)):
    for block in parallel:
      butterfly_computation(stage, block)
```

---

### 4. Memory Hierarchy Must Be Modeled Explicitly

**Principle:** Morphogen's Kernel treats streams/fields as typed `Stream<T, D, R>`. But for GPU lowering, we must also respect:

- **Global memory:** large, slow (100+ cycles)
- **Shared memory:** small, fast, banked (32KB-64KB per SM)
- **Register pressure:** expensive spills

**Implications for Morphogen:**

**a) Clear ABI for "logical buffers":**

```
buf(time, channel, spatial dims) → memref<…>
```

**b) Tiling passes that precede GPU lowering:**

Especially for convolution, FFT, wavelets, PDE stencils.

**c) Shared-memory annotation paths:**

Morphogen ops like `conv2d`, `wavelet`, and k-space transforms should optionally say:

```json
{
  "memory_hint": "shared",
  "cache_tile": [32, 32]
}
```

**Why This Matters:**

Without this structural information, MLIR may not lift scratchpads into shared memory, hurting performance.

**Memory Tiers & Access Patterns:**

| Memory Type | Size | Latency | Bandwidth | Use Case |
|-------------|------|---------|-----------|----------|
| **Global** | GB | 200-800 cycles | 900 GB/s | Input/output buffers |
| **Shared** | 48-64 KB | 20-40 cycles | ~10 TB/s | Tile caching, reductions |
| **Registers** | 64 KB/SM | 1 cycle | ~20 TB/s | Hot loop variables |

---

### 5. Follow the Canonical GPU Lowering Pipeline

**Principle:** Morphogen's compiler should adopt MLIR's standard GPU optimization pipeline.

**Pipeline Stages:**

```
1. Tiling (linalg/affine)
   Outer = GPU blocks
   Inner = warps/threads

2. Vectorization (vector dialect)
   Maps microkernels to SIMD inside a thread.

3. GPU dialect mapping
   gpu.launch, thread/block dims, shared memory.

4. NVVM/ROCDL/SPIRV lowering
   Emit PTX/Metal/SPIR-V.
```

This matches how Morphogen transforms, filters, and domain operators generally decompose.

**Why This Matters:**

Morphogen can guarantee determinism + reproducibility across backends only if its lowering path is structured and repeatable.

**Example Pipeline:**

```
Morphogen Graph IR
    ↓
morphogen.signal dialect
    ↓
linalg.generic (tiled)
    ↓
affine.parallel (mapped to blocks)
    ↓
scf.parallel (mapped to threads)
    ↓
vector dialect (SIMD within thread)
    ↓
gpu.launch_func
    ↓
nvvm/rocdl/spirv
    ↓
PTX/LLVM/SPIR-V
```

---

### 6. Static Shapes Are Gold

**Principle:** The kernel already tracks domain, rate, and shapes. To optimize for GPUs, prefer static dimensions whenever possible.

**Implications for Morphogen:**

- **Stream lengths** should be static within a block
- **Field dimensions** should be known at compile time whenever possible
- **Transform windows/overlaps** should be compile-time constants

**Where dynamics must exist, Morphogen should:**

- Allow symbolic shapes
- But encourage partial evaluation
- And propagate constants aggressively

**Why This Matters:**

STFT, convolution, DCT, and spectral transforms all depend on static tile sizes for speed.

**Static vs Dynamic Trade-offs:**

| Aspect | Static | Dynamic |
|--------|--------|---------|
| **Performance** | 2-5× faster | Baseline |
| **Compilation** | Slower | Faster |
| **Flexibility** | Limited | High |
| **Code size** | Larger | Smaller |
| **Morphogen preference** | **Strong default** | When necessary |

---

### 7. Avoid Divergence: Ops Must Be Warp-Friendly

**Principle:** SIMT GPUs serialize divergent branches. Morphogen ops should be branch-coherent, regular, and explicit about mask vs branch semantics.

**Guidelines:**

- **Envelope generators** → mask instead of per-sample branching
  ```morphogen
  // Good: masked computation
  let env = envexp(5ms) * (t < attack_time ? 1.0 : 0.0)

  // Better: predicated operation
  let env = select(t < attack_time, envexp(5ms), 0.0)
  ```

- **Saturators** → vectorized clamp functions
  ```morphogen
  // Good: no branching
  let clipped = clamp(signal, -1.0, 1.0)

  // Bad: branching inside hot loop
  if signal > 1.0 { signal = 1.0 }
  if signal < -1.0 { signal = -1.0 }
  ```

- **Conditionals at block level** → `scf.if` outside innermost loops
  ```mlir
  // Good: branch outside parallel region
  scf.if %cond {
    affine.parallel (%i) = (0) to (%N) { ... }
  }

  // Bad: branch inside parallel region
  affine.parallel (%i) = (0) to (%N) {
    scf.if %cond { ... } else { ... }
  }
  ```

**Why This Matters:**

Guaranteeing determinism is easier when kernel execution paths are uniform.

**Warp Divergence Cost:**

- **Uniform branch:** 0 extra cycles
- **2-way divergence:** 2× execution time
- **4-way divergence:** 4× execution time
- **Per-thread branch:** 32× execution time (worst case)

---

### 8. Determinism Profiles Must Map to GPU Semantics

**Principle:** GPU determinism is not trivial. But Morphogen's profile system can map directly.

**Profile Mappings:**

#### `strict` Profile
- Fixed tile sizes
- Fixed launch parameters
- Disallow atomics
- Disallow warp-wide reductions unless deterministic
- Bit-exact FFT providers (cuFFT determinism modes)

```json
{
  "profile": "strict",
  "gpu": {
    "block_size": [256, 1, 1],
    "tile_size": [32, 32],
    "atomics": false,
    "fft_mode": "deterministic"
  }
}
```

#### `repro` Profile
- Deterministic within FP; allow GPU math
- No nondeterministic atomics
- Reproducible reductions

```json
{
  "profile": "repro",
  "gpu": {
    "block_size": "auto",
    "tile_size": "auto",
    "atomics": "none",
    "reductions": "deterministic"
  }
}
```

#### `live` Profile
- Low-latency kernels
- Flexible tiling
- Possibly non-deterministic GPU reductions

```json
{
  "profile": "live",
  "gpu": {
    "block_size": "auto",
    "tile_size": "auto",
    "atomics": "allow",
    "reductions": "fast"
  }
}
```

**Why This Matters:**

This cleanly extends Morphogen's determinism tiers to GPU operators and providers.

---

### 9. Graph IR Must Describe Enough Structure for GPU Mapping

**Principle:** Morphogen Graph IR today is `{"nodes":[...], "edges":[...]}`. For GPU correctness/performance, the IR should also include structural metadata.

**Required Metadata:**

```json
{
  "nodes": [
    {
      "id": "fft1",
      "op": "transform.to",
      "params": {
        "domain": "frequency",
        "method": "fft",
        "size": 1024,
        "window": "hann"
      },
      "type": "Stream<f32, frequency, 48000Hz>",
      "gpu_hints": {
        "tile_size": [256],
        "shared_mem": true,
        "vector_width": 4
      }
    }
  ]
}
```

**Metadata Categories:**

- **Declared shapes/units/rates:** Enable static allocation
- **Transform parameters:** Window size, hop size, dims
- **Tiling hints:** Optional performance guidance
- **Vector width suggestions:** SIMD optimization
- **Memory locality intentions:** Cache behavior

**Why This Matters:**

This doesn't pollute the composer/performance surfaces — it's all validated and injected by the kernel.

---

### 10. Morphogen's Goals Match the MLIR/GPU Sweet Spot

**Why These Principles Matter:**

Morphogen is designed around:

- ✅ **Determinism**
- ✅ **Typed streams/fields**
- ✅ **Domain transforms**
- ✅ **Structured graphs**
- ✅ **Declarative operators**

**This is exactly what MLIR's structured IR and GPU pipeline expect.**

If Morphogen embraces the checklist above, it gains:

| Benefit | Impact |
|---------|--------|
| **Predictable performance** | Across CPU/GPU backends |
| **Portable kernels** | Write once, run anywhere |
| **Backend-agnostic transforms** | FFT/DCT/Wavelet work everywhere |
| **Reproducibility** | Core Morphogen promise delivered |
| **Simpler debugging** | Structured IR = better introspection |
| **Easier operator extensibility** | Clear lowering patterns |

---

## Summary

> **Morphogen's semantic kernel is already architecturally aligned with MLIR's structured, multi-dialect GPU programming model.**
>
> **These GPU principles simply turn that alignment into implementation reality.**

### Quick Reference: The 10 Commandments

1. ⚡ **Express parallelism structurally** — No opaque kernels
2. 📋 **Registry encodes lowering metadata** — Tile hints, vector widths
3. 🔄 **Transforms remain regular** — Static, tileable loops
4. 💾 **Model memory hierarchy explicitly** — Global/shared/registers
5. 🏗️ **Follow canonical GPU pipeline** — Tile → vectorize → GPU-map
6. 📐 **Static shapes are gold** — Prefer compile-time constants
7. 🌊 **Avoid warp divergence** — Uniform execution paths
8. 🎯 **Determinism profiles map to GPU** — Strict/repro/live semantics
9. 📊 **Graph IR describes structure** — Shapes, rates, hints
10. ✨ **Morphogen matches MLIR's sweet spot** — Semantic kernel + structured IR = 💯

---

## References

- [MLIR GPU Dialect Documentation](https://mlir.llvm.org/docs/Dialects/GPU/)
- [MLIR Affine Dialect](https://mlir.llvm.org/docs/Dialects/Affine/)
- [MLIR Vector Dialect](https://mlir.llvm.org/docs/Dialects/Vector/)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Morphogen Architecture](../ARCHITECTURE.md)
- [Morphogen Transform Dialect Spec](../specifications/transform.md)
- [Morphogen Graph IR Spec](../specifications/graph-ir.md)

---

**Next Steps:**

1. Audit existing operator registry for GPU metadata coverage
2. Add tile/vector hints to Transform Dialect operators
3. Implement GPU lowering passes in MLIR pipeline
4. Create GPU-specific conformance tests for each determinism profile
5. Document GPU provider ABI for cuFFT/rocFFT/Metal Performance Shaders

---

*This document is part of the Morphogen Stack v1.0 architecture. For questions or contributions, see the main [ARCHITECTURE.md](../ARCHITECTURE.md).*
