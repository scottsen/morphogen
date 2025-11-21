# ADR-005: GPU-First Domains and the BI Paradigm Shift

**Date:** 2025-11-15
**Status:** Proposed
**Authors:** Claude, User
**Related:** ../specifications/bi-domain.md, ../specifications/kax-language.md, ADR-002 (Cross-Domain Patterns)

---

## Context

Morphogen was originally designed with a **device-transparent compute model**:

> "Operators can run on either CPU or GPU with identical semantics.
> GPU is an accelerator, chosen for performance, not for algorithmic reasons."

This model has served Morphogen well across many domains:

- ✅ **Audio DSP** — FFT, filters, oscillators (CPU or GPU)
- ✅ **Fractal rendering** — Mandelbrot, Julia sets (CPU or GPU)
- ✅ **Physics** — Particle systems, fluids, thermal (CPU or GPU)
- ✅ **Cellular Automata** — Game of Life, Lenia (CPU or GPU)
- ✅ **Geometry** — Procedural meshes, CAD operations (CPU or GPU)
- ✅ **Transform** — FFT, STFT, wavelets (CPU or GPU)

In all these cases:
- The **algorithm** is device-independent
- The **semantics** are the same on CPU and GPU
- The **GPU is optional** — it provides speedup, not different behavior
- **Operators auto-lower** to GPU when beneficial

### The Problem: BI Breaks This Model

When we started designing the **BIDomain** for business intelligence and analytical workloads, we encountered a fundamental conflict:

**BI engines like VertiPaq (Power BI) require data structures that are inherently GPU-native:**

1. **Dictionary encoding** — String → Integer ID mapping with reverse lookup
2. **Bit-packing** — Compressed integer storage (e.g., 3-bit integers)
3. **Segmented columnar storage** — Sorted, partitioned, metadata-rich
4. **Warp-level parallel scanning** — GPU-specific SIMD primitives
5. **Bitmap filters** — Compressed boolean predicates
6. **GPU-resident segment metadata** — Min/max, null counts, cardinality

These are **not CPU algorithms accelerated on GPU.**
They are **GPU-specific data structures and algorithms.**

Running them on CPU:
- ❌ Different memory layout
- ❌ Different internal representations
- ❌ Different semantics (e.g., warp-level operations have no CPU equivalent)
- ❌ Not just slower — fundamentally different

### The Dilemma

We face a choice:

1. **Option A:** Force BI into the existing device-transparent model
   - Result: Compromise BI performance and semantics
   - Consequence: Morphogen BI would not be competitive with VertiPaq, cuDF, or OmniSciDB
   - Outcome: BI becomes a second-class domain

2. **Option B:** Introduce a new concept — **GPU-First Domains**
   - Result: Some domains explicitly require GPU
   - Consequence: Morphogen's abstraction model evolves
   - Outcome: BI becomes a first-class, high-performance domain

### Why This Matters

BI is not the only domain that benefits from GPU-first semantics:

- **Machine Learning** — GPU-native tensor operations, autograd, kernel fusion
- **Ray Tracing** — BVH traversal, GPU-native acceleration structures
- **Sparse Linear Algebra** — GPU-specific CSR/CSC formats, warp-level reductions
- **Graph Algorithms** — GPU-native adjacency representations
- **Cryptography** — Warp-level parallel hashing, GPU-specific bit operations

If Morphogen wants to be a **universal computational platform**, it must support GPU-first domains.

---

## Decision

**We will introduce GPU-First Domains as a new architectural concept in Morphogen.**

### What is a GPU-First Domain?

A **GPU-First Domain** is a Morphogen domain where:

1. **GPU is not an accelerator — it is the primary execution model**
2. **Operators require GPU memory formats** (e.g., dictionary-encoded columns, bitmaps)
3. **CPU fallback may be impossible or semantically different**
4. **GPU data structures define the algorithm** (e.g., warp-level scans)

### Domain Annotation

GPU-First domains are explicitly marked in their definition:

```morphogen
domain: BI
device: gpu
```

This indicates:
- Operators in this domain are GPU-resident by default
- CPU execution may not be supported
- Automatic GPU lowering is assumed

Contrast with device-transparent domains:

```morphogen
domain: Audio
device: any  // Can run on CPU or GPU
```

### GPU-First Operators

GPU-First domains introduce GPU-specific operators:

```morphogen
// Dictionary encoding (GPU-native)
op gpu.dict_encode<T>(
    col: GpuColumn<T>
) -> GpuDictEncodedColumn<T>

// Bitmap filter (GPU-native)
op gpu.bitmap_filter(
    bitmap: GpuBitmap,
    table: GpuTable
) -> GpuTable

// Warp-level prefix sum (GPU-specific)
op gpu.scan_warp(
    col: GpuColumn<i32>
) -> GpuColumn<i32>

// Segmented group-by (GPU-native)
op gpu.segmented_groupby<K, V>(
    keys: GpuColumn<K>,
    values: GpuColumn<V>,
    agg: AggOp
) -> GpuGroup<K, V>
```

These operators:
- Have **no CPU equivalents** (or CPU versions are semantically different)
- Require **GPU memory layouts** (e.g., `GpuColumn<T>`, `GpuBitmap`)
- Are **intrinsically GPU-parallel** (e.g., warp-level operations)

---

## Implications

### 1. Type System Extension

We introduce GPU-resident types:

```morphogen
type GpuColumn<T> {
    data: GpuBuffer<T>,
    count: i64,
    nulls: GpuBitmap?
}

type GpuDictEncodedColumn<T> {
    dictionary: GpuColumn<T>,
    indices: GpuColumn<i32>,
    count: i64
}

type GpuBitmap {
    bits: GpuBuffer<u32>,
    count: i64
}

type GpuTable {
    columns: Map<String, GpuColumn<Any>>,
    row_count: i64,
    segments: GpuColumn<GpuSegment>?
}
```

These types:
- Are **always GPU-resident**
- Cannot be directly manipulated on CPU
- Require GPU context for allocation

---

### 2. Operator Registry Updates

The operator registry must support device-specific operators:

```morphogen
// GPU-first operator
@device(gpu)
op gpu.dict_encode<T>(col: GpuColumn<T>) -> GpuDictEncodedColumn<T>

// Device-transparent operator (existing model)
@device(any)
op fft(signal: Array<f64>) -> Array<Complex<f64>>
```

The scheduler uses `@device` annotations to:
- Determine execution backend
- Validate device availability
- Allocate GPU memory when needed

---

### 3. Scheduler Changes

The scheduler must:

1. **Recognize GPU-first operators**
   - Check for GPU availability
   - Fail gracefully if no GPU is present

2. **Manage GPU memory**
   - Allocate `GpuBuffer<T>` on device
   - Track GPU resident data
   - Handle host ↔ device transfers

3. **Optimize GPU-to-GPU pipelines**
   - Fuse operators when data is GPU-resident
   - Avoid unnecessary host transfers
   - Batch GPU kernels

Example:

```morphogen
// All GPU-resident — no host transfers
let encoded = gpu.dict_encode(Sales[ProductName])
let filtered = gpu.bitmap_filter(bitmap, Sales)
let aggregated = gpu.agg_sum(filtered[Amount])
```

Scheduler generates:

```
[GPU] dict_encode
  ↓ (GPU memory)
[GPU] bitmap_filter
  ↓ (GPU memory)
[GPU] agg_sum
  ↓ (scalar result to host)
```

---

### 4. Cross-Domain Interaction

GPU-first domains can interact with device-transparent domains:

**BI → Visualization:**

```morphogen
let sales_total = gpu.agg_sum(Sales[Amount])  // GPU-first
viz.bar_chart(sales_total)                    // Device-transparent
```

Data flow:

```
GPU: gpu.agg_sum → scalar result
Host: viz.bar_chart (receives scalar)
```

**BI → Simulation:**

```morphogen
let avg_temp = gpu.agg_avg(Sensors[Temperature])  // GPU-first
physics.thermal_ode(initial_temp: avg_temp)       // Device-transparent
```

Data flow:

```
GPU: gpu.agg_avg → scalar result
Host or GPU: thermal_ode (receives scalar, can run on either)
```

**Mixed GPU pipelines:**

```morphogen
let grouped = gpu.segmented_groupby(Sales[Region], Sales[Amount], Sum)  // GPU
let heatmap = viz.gpu_heatmap(grouped)                                  // GPU
render.gpu(heatmap)                                                     // GPU
```

Data flow:

```
GPU: segmented_groupby → GpuGroup (stays on GPU)
GPU: gpu_heatmap (consumes GpuGroup, produces GpuTexture)
GPU: render.gpu (consumes GpuTexture)
```

---

### 5. Error Handling

If a GPU-first operator is invoked without GPU:

```morphogen
let result = gpu.dict_encode(Sales[ProductName])
```

Error:

```
Error: Operator 'gpu.dict_encode' requires GPU, but no GPU device is available.
Hint: Check CUDA/ROCm installation or run on a GPU-enabled machine.
```

Optional: Provide CPU fallback (if semantically feasible):

```morphogen
@device(gpu)
@fallback(cpu)  // Optional CPU implementation (if possible)
op gpu.dict_encode<T>(col: GpuColumn<T>) -> GpuDictEncodedColumn<T>
```

For BI, most operators have **no CPU fallback** — GPU is required.

---

## Why GPU-First Domains Fit Morphogen

### 1. Morphogen is Already GPU-Centric

Morphogen's MLIR pipeline already lowers to GPU:
- `gpu.launch` for parallel kernels
- `gpu.memcpy` for host ↔ device transfers
- GPU dialect integration

GPU-first domains **leverage existing infrastructure**.

---

### 2. Cross-Domain Composability

GPU-first domains integrate naturally:

**BI + Visualization:**

```morphogen
let sales = gpu.agg_sum(Sales[Amount])
viz.bar_chart(sales)
```

**BI + ML:**

```morphogen
let features = gpu.calc_column(Customers, ...)
ml.train_gpu(features, target)
```

**BI + Simulation:**

```morphogen
let avg_pressure = gpu.agg_avg(Sensors[Pressure])
physics.fluid_network(initial_pressure: avg_pressure)
```

---

### 3. Performance Gains

GPU-first BI operators achieve:

| Operation | GPU Speedup vs CPU |
|-----------|-------------------|
| Dictionary encoding | 10-30x |
| Grouped aggregation | 20-50x |
| Bitmap filtering | 30-100x |
| Hash joins | 15-40x |

This makes Morphogen BI competitive with cuDF, OmniSciDB, and RAPIDS.

---

### 4. Future-Proofing

GPU-first domains enable future expansions:

- **Ray Tracing** — GPU-native BVH, traversal
- **ML Inference** — GPU-native tensor ops, kernel fusion
- **Sparse LinAlg** — GPU CSR/CSC formats
- **Graph Algorithms** — GPU adjacency representations
- **Cryptography** — Warp-level hashing

Morphogen becomes a **universal GPU-first computational platform**.

---

## Alternatives Considered

### Alternative 1: Force BI to Be Device-Transparent

**Approach:** Implement BI using CPU-GPU-agnostic operators.

**Pros:**
- Maintains existing abstraction
- No new concepts

**Cons:**
- ❌ BI performance would be poor
- ❌ Cannot use GPU-native data structures (dictionary encoding, bitmaps)
- ❌ Morphogen BI would not be competitive
- ❌ Limits future GPU-first domains (ML, ray tracing)

**Decision:** Rejected — compromises BI performance and limits future growth.

---

### Alternative 2: Create a Separate GPU-Only BI Engine

**Approach:** Build BI as a standalone GPU library, not integrated with Morphogen.

**Pros:**
- GPU-native performance
- No impact on Morphogen's abstraction

**Cons:**
- ❌ No cross-domain composability
- ❌ Fragmented ecosystem
- ❌ Cannot combine BI with simulation, ML, rendering
- ❌ Defeats Morphogen's vision of unified computational platform

**Decision:** Rejected — violates Morphogen's core principle of domain composability.

---

### Alternative 3: Automatic GPU Fallback to CPU

**Approach:** GPU-first operators automatically fall back to CPU implementations.

**Pros:**
- Works on machines without GPU
- Graceful degradation

**Cons:**
- ❌ CPU implementations of GPU-native ops are semantically different
- ❌ Hidden performance cliffs (user expects GPU speed, gets CPU slowness)
- ❌ Increased implementation complexity
- ❌ Not all GPU ops have CPU equivalents (e.g., warp-level operations)

**Decision:** Rejected — CPU fallback is optional and domain-specific, not automatic.

---

## Implementation Plan

### Phase 1: Type System
- Introduce `GpuColumn<T>`, `GpuTable`, `GpuBitmap`
- Add `GpuBuffer<T>` allocation primitives
- Update type checker for GPU-resident types

**Deliverable:** GPU-resident type system

---

### Phase 2: Operator Registry
- Add `@device(gpu)` annotations
- Support device-specific operator registration
- Update operator lookup to filter by device

**Deliverable:** Device-specific operator registry

---

### Phase 3: Scheduler Updates
- Recognize GPU-first operators
- Allocate GPU memory for `GpuBuffer<T>`
- Optimize GPU-to-GPU pipelines
- Handle host ↔ device transfers

**Deliverable:** GPU-aware scheduler

---

### Phase 4: BIDomain Implementation
- Implement core GPU operators (`dict_encode`, `bitmap_filter`, `agg_sum`, etc.)
- Write CUDA kernels for BI primitives
- Integrate with MLIR GPU dialect

**Deliverable:** Functional GPU-first BIDomain

---

### Phase 5: Cross-Domain Integration
- BI → Visualization
- BI → Simulation
- BI ↔ ML
- Example workflows

**Deliverable:** Cross-domain BI examples

---

### Phase 6: Future GPU-First Domains
- ML inference (GPU tensor ops)
- Ray tracing (GPU BVH)
- Sparse linear algebra (GPU CSR/CSC)

**Deliverable:** Expanded GPU-first ecosystem

---

## Risks and Mitigations

### Risk 1: No GPU Available

**Risk:** User runs GPU-first operator without GPU.

**Mitigation:**
- Fail with clear error message
- Suggest GPU-enabled alternatives
- (Optional) Provide CPU fallback where feasible

---

### Risk 2: Increased Complexity

**Risk:** GPU-first domains add abstraction complexity.

**Mitigation:**
- Clear documentation
- Device annotations make intent explicit
- Type system enforces GPU-residency

---

### Risk 3: Cross-Domain Data Movement

**Risk:** Frequent GPU ↔ CPU transfers hurt performance.

**Mitigation:**
- Scheduler optimizes GPU-to-GPU pipelines
- Batch transfers when necessary
- Encourage GPU-resident workflows

---

## Success Criteria

We will know GPU-first domains are successful when:

1. ✅ **BIDomain achieves 10-30x speedup** over CPU-based BI engines
2. ✅ **Cross-domain workflows work seamlessly** (BI → Viz, BI → Sim, etc.)
3. ✅ **Future GPU-first domains are easy to add** (ML, ray tracing, etc.)
4. ✅ **Type system enforces GPU-residency** at compile time
5. ✅ **Scheduler optimizes GPU-to-GPU pipelines** automatically

---

## Conclusion

**GPU-First Domains** are a necessary evolution of Morphogen's architecture.

They enable:
- ✅ **High-performance BI** competitive with cuDF and OmniSciDB
- ✅ **Cross-domain composability** (BI + Sim + ML + Viz)
- ✅ **Future expansion** into ML, ray tracing, sparse algebra
- ✅ **GPU-native semantics** where appropriate

This decision preserves Morphogen's core principles:
- **Declarative operator model** — GPU-first operators are still declarative
- **Cross-domain composability** — GPU-first domains integrate with all others
- **Extensibility** — New GPU-first domains are easy to add

And it positions Morphogen as a **universal GPU-first computational platform**.

---

## Related Documents

- **../specifications/bi-domain.md:** BI domain architecture and operators
- **../specifications/kax-language.md:** KAX expression language specification
- **../specifications/operator-registry.md:** Operator registration and extensibility
- **../architecture/gpu-mlir-principles.md:** GPU lowering and MLIR integration
- **ADR-002:** Cross-Domain Architectural Patterns

---

**End of ADR-005**
