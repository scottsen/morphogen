# The Morphogen Stack — Finalized Architecture & Specs (v1.0 Draft)

> **For newcomers**: Start with [README.md](README.md) for the vision and [ECOSYSTEM_MAP.md](ECOSYSTEM_MAP.md) for domain coverage. This document is the **technical architecture** for implementers.

## Overview

Morphogen is architected as a **semantic kernel** that unifies multi-domain computation through:
- **One type system** with physical units (Stream, Field, Event, Agent)
- **One scheduler** handling multiple rates (audio @ 48kHz, control @ 60Hz, physics @ 240Hz)
- **One compiler** (MLIR → LLVM/GPU) for all domains
- **Two human-friendly surfaces** (Morphogen.Audio DSL + RiffStack performance environment)

This architecture enables **cross-domain composition** that's impossible in traditional systems: circuit simulation can drive audio synthesis, fluid dynamics can generate acoustic fields, geometry can define PDE boundary conditions — all in one deterministic execution environment.

**Goal**: Keep what's great, shed historical baggage, and define the **cleanest possible layering** for deterministic, creative, multi-domain computation.

---

## 0) Design Tenets

1. **Operator-theoretic foundation** — Every module, transform, and domain operation is an operator `O: X → X` with well-defined spectral properties. See **[docs/OPERATOR_PHILOSOPHY.md](docs/OPERATOR_PHILOSOPHY.md)** for the mathematical foundation.
2. **Semantic kernel, not a monolith** — one place defines time/space/rate/units/state/profiles/determinism.
3. **Transforms as a first-class grammar** — FFT is not special; *domain changes* are core operations that diagonalize operators to reveal their spectra.
4. **Typed, reproducible computation** — every stream/field/event has type, units, domain, and a determinism tier. Operators must declare their algebraic properties (linear, unitary, self-adjoint).
5. **Thin, pluggable backends** — CPU/GPU/Audio/FFT providers are replaceable modules implementing the same operator interfaces.
6. **Two user surfaces** — *Composer* (Morphogen.Audio) and *Performer* (RiffStack) share the same kernel and operator registry.

> 📍 **Operator Philosophy:** Everything in Morphogen is an operator. Systems are operators, dynamics are spectra. This unifies our discrete (digital) and continuous (Philbrick analog) computation under one mathematical framework. Read **[docs/OPERATOR_PHILOSOPHY.md](docs/OPERATOR_PHILOSOPHY.md)** for the deep foundation.

> 📍 **Ecosystem Overview:** For a comprehensive map of all potential Morphogen domains, modules, and expansion roadmap, see **[ECOSYSTEM_MAP.md](ECOSYSTEM_MAP.md)**

---

## 1) Layering (crisp separation of concerns)

```
┌───────────────────────────────────────────────────────┐
│                   Applications / IDEs                 │
└───────────────▲───────────────────────▲───────────────┘
                │                       │
        (A) RiffStack               (B) Morphogen.Audio
        Live YAML/RPN               Typed declarative DSL
        • patches, loopers          • scenes, modules, events
        • performance UX            • composition & rendering
                └───────────────▲──────┘
                                │  (Graph IR)
┌───────────────────────────────┴────────────────────────┐
│                     Morphogen Kernel                       │
│  • Types & Units (Stream<T,D,R>, Evt<A>, Space/Boundary)│
│  • Deterministic Multirate Scheduler                    │
│  • Transform Dialect (to/from/reparam)                  │
│  • Profiles (strict/repro/live)                         │
│  • Operator Registry (single source of truth)           │
│  • State & Snapshot ABI, Introspection                  │
└───────────────────────▲─────────▲──────────────────────┘
                        │         │
               Morphogen Compiler   Runtime Backends
               (MLIR dialects   (CPU/GPU/Audio/FFT/Convolution/
                & lowering)       Storage providers)
```

**Contract:** RiffStack and Morphogen.Audio *both* emit **Morphogen Graph IR** (typed node/edge JSON). Kernel validates, compiles, schedules, and runs.

---

## 2) Kernel: Responsibilities & APIs

### 2.1 Types & Units

* `Stream<T,Domain,Rate>` — unified supertype (signals, fields, images).
* `Evt<A>` — totally ordered, timestamped events.
* `Space/Grid` — dimension, spacing, centering.
* `Boundary/Interface` — domain couplers (Dirichlet/Neumann/periodic/reflect; flux_match/insulated).
* Units: SI + domain aliases (`Hz`, `dB`, `cents`, `px`, `k` wavenumber). Lossy casts require `@allow_unit_cast`.

### 2.2 Deterministic Multirate Scheduler

* Rates: `audio`, `control`, `visual`, `sim` with explicit `dt` or `sample_rate`.
* Partitions time by LCM/hop sizes; sample-accurate **event fences**.
* Cross-rate ops are explicit: `resample(to=rate, mode=nearest|linear|cubic)`.
* State consistency: double-buffering; hot-reload barriers.

### 2.3 Transform Dialect (Core Grammar)

* `transform.to(x, domain, method, attrs...)`
* `transform.from(x, domain, method, attrs...)`
* `transform.reparam(x, mapping)` (coordinate changes)
* Normalization policies & windowing are profile-driven, explicit in metadata.

**Minimum transforms (v1):** FFT/iFFT (time↔frequency), STFT/ISTFT, DCT/IDCT, Wavelet/IWavelet (family param), Space↔k-space, Graph Laplacian spectral, Mel/Inverse-Mel.

### 2.4 Profiles & Determinism

* Tiers: `strict` (bit-exact), `repro` (deterministic within FP), `live` (replayable, low-latency).
* Precedence: per-op > module > scene > profile > global.
* Profiles tune precision, oversampling, block size, convolution partitioning, FFT normalization.

### 2.5 Operator Registry (Single Source of Truth)

* Declarative metadata for all ops used by both frontends (and docs/CLI generated from it).
* Organized into **7 semantic layers**: Core → Transforms → Stochastic → Physics/Fields → Audio → Visuals/Fractals → Finance
* Each operator carries type signatures, units, determinism tier, numeric properties, and lowering templates

> **See:** [docs/SPEC-OPERATOR-REGISTRY.md](docs/SPEC-OPERATOR-REGISTRY.md) for complete 7-layer architecture and operator catalog

**Schema (excerpt):**

```json
{
  "name": "lpf",
  "category": "filter",
  "layer": 5,
  "inputs": [{"name":"sig","type":"Sig"}],
  "params": {
    "cutoff": {"type":"Ctl[Hz]", "default":"2kHz", "range":[20, 24000]},
    "q": {"type":"Ctl", "default":0.707, "range":[0.1, 20]}
  },
  "determinism": "strict",
  "profile_defaults": {"live":{}, "repro":{}, "strict":{}},
  "lowering": {"dialect":"morphogen.signal", "template":"lpf_svf"}
}
```

### 2.6 State, Snapshot, Introspection

* Snapshot ABI: buffers, seeds, profiles, graph hash.
* Hot-reload: patch graph at barriers (scene add/remove, parameter rebind).
* `introspect(graph) -> JSON` (nodes, edges, rates, counters).

---

## 3) Morphogen Compiler (MLIR)

* Dialects: `morphogen.stream`, `morphogen.signal`, `morphogen.field`, `morphogen.visual`, `morphogen.transform`, `morphogen.agent`.
* Passes: type/units, event fencing, fusion, vectorization, tiling, async I/O.
* Lowering targets: `linalg/affine/vector/gpu/async` → LLVM/SPIR-V/Metal.
* External calls: FFT/Conv providers (FFTW/MKL/cuFFT/rocFFT), device audio.

**GPU & MLIR Design Principles:** The compiler follows structured lowering patterns that align Morphogen's semantic kernel with MLIR's GPU pipeline. See [GPU & MLIR Principles](docs/GPU_MLIR_PRINCIPLES.md) for detailed design rules on parallelism, memory hierarchy, determinism profiles, and operator metadata.

---

## 4) Runtime Backends (Providers)

* **CPU** (LLVM JIT), **GPU** (SPIR-V/Metal/CUDA), **Audio Device** (block push/pull), **FFT/Conv** (pluggable).
* Provider ABI (C/Python): `init()`, `load(module)`, `run(block)`, `event_inject(evt)`, `shutdown()`.

---

## 5) Morphogen Graph IR (frontend-facing JSON)

**Purpose:** a neutral, typed graph that both RiffStack and Morphogen.Audio can emit.

```json
{
  "version": "1.0",
  "profile": "repro",
  "nodes": [
    {"id":"osc1","op":"sine","out":["s1"],"params":{"freq":"440Hz"}},
    {"id":"lpf1","op":"lpf","in":["s1"],"out":["s2"],"params":{"cutoff":"2kHz","q":0.8}},
    {"id":"pan","op":"pan","in":["s2"],"out":["L","R"],"params":{"pos":0.1}}
  ],
  "outputs": {"stereo":["L","R"]}
}
```

Kernel validates (types/units/domains), attaches rates, compiles, runs.

---

## 6) Morphogen.Audio (Composer Surface)

* **Types:** `Sig` ≡ `Stream<f32,1D,audio>`, `Ctl` ≡ `Stream<f32,0D,control>`, `Evt<A>`, `Note`.
* **Structure:** `scene`, `module`, `out stereo`, `score/at/loop`, `spawn`.
* **Ops:** oscillators, filters, envelopes, FX, physical (waveguide/membrane/bodyIR/pickup/amp/cab), transforms.
* **Determinism & profiles** respected by default.

*Example (declarative, fun-first, fully overridable):*

```morphogen
scene Duo {
  let seq = score [
    at 0s   note("A3",1,0.5s),
    at 0.5s note("C4",0.9,0.5s),
    at 1.0s note("E4",0.8,1.0s)
  ] |> loop(2s)

  module Guitar(n: Note): Sig {
    let exc = noise(seed=7) |> lpf(6kHz) |> envexp(5ms) * n.vel
    let str = string(n.pitch, 1.2s) exc
    mix( bodyIR("acoustic.ir")(str)*0.8,
         str |> pickup("humbucker",0.25) |> amp("brown",0.7) |> cab("4x12.ir")*0.7 )
  }

  out stereo = spawn(seq, (n)=>Guitar(n), max_voices=12) |> reverb(0.12) |> limiter(-1dB)
}
```

---

## 7) RiffStack (Performer Surface)

* **YAML patches + RPN expressions**, loopers, controls; compiles to Morphogen Graph IR.
* Uses kernel profiles (`live` default) and injects controls as `Evt<Control>`.

*Example:*

```yaml
version: 0.3
tracks:
  - id: lead
    expr: "saw 220 0.7 lowpass 1200 0.9 reverb 0.1 play"
loopers:
  - id: main
    input: lead
    length: 4 bars
controls:
  - trigger: space
    action: toggle_record main
```

---

## 8) Transform Mindset — Spectral Decomposition as First-Class

**Core Insight**: Transforms don't just "change domains" — they **diagonalize operators** to reveal their spectra.

When you run `fft(signal)`, you're not "converting to frequency." You're:
1. Decomposing the signal into an orthogonal basis (complex exponentials)
2. Projecting the signal operator onto its eigenbasis
3. Revealing the operator's spectrum (eigenvalues = frequencies)

**Why this matters**: The spectral view unifies all transforms under one mathematical framework.

### Minimal v1 Transform Set (Strict & Reproducible)

| Transform | Basis | Operator Diagonalized | Use Case |
|-----------|-------|----------------------|----------|
| `fft/ifft` | Complex exponentials | Convolution → Multiplication | Audio, signals |
| `stft/istft` | Windowed exponentials | Time-varying convolution | Spectrograms |
| `fft2d/ifft2d` | 2D/3D Fourier | Spatial convolution | Fields, images |
| `dct/idct` | Cosine basis | Compression operators | JPEG, cepstrum |
| `wavelet/iwavelet` | Wavelet family | Multi-scale analysis | Denoising, edges |
| `graph_spectral` | Laplacian eigenvectors | Graph diffusion | Networks, topology |
| `mel/imel` | Perceptual basis | Psychoacoustic operators | Audio perception |

All exposed uniformly:

```morphogen
# Explicit spectral decomposition
let spec = transform.to(sig, domain="frequency", method="fft", window="hann")
let shaped = spec * pink_shelf    # Diagonal operator in frequency domain
let out = transform.from(shaped, domain="time")

# This is literally: O_ifft ∘ O_multiply ∘ O_fft
# The FFT diagonalizes the convolution operator
```

> 📖 **Deep Dive**: See [docs/OPERATOR_PHILOSOPHY.md](docs/OPERATOR_PHILOSOPHY.md) for the mathematical foundation of spectral decomposition and why transforms are operator diagonalization.

---

## 9) Extension & Customization

* **New ops**: add to registry (metadata + lowering template or kernel hook).
* **New domains**: register name, coordinates, basis, default transforms.
* **New backends**: implement provider ABI & advertise capabilities.
* **New DSLs**: emit Graph IR or call kernel SDK; reuse registry.

---

## 10) Diagnostics & Conformance

* Lints: NaN/Inf, DC offset, clipping risk, unit mismatches.
* Golden artifacts per profile (WAV/PNG/NPY).
* Event fence tests (off-by-one proof).
* Deterministic RNG: Philox 4×32-10 seeded with `(graph_hash, id, tick, user_seed)`.

---

## 11) Migration Plan (from "what exists" to "this")

**Week 1–2 — Boundaries**

* Extract **Operator Registry** (JSON + codegen), retrofit a subset of ops.
* Move MLIR dialects/passes into `compiler/` module.

**Week 3–4 — Core additions**

* Implement **Transform Dialect** (fft/stft/dct/wavelet + space↔k-space).
* Expose **Morphogen Graph IR** loader/validator (JSON schema + CLI).

**Week 5 — Runtime bridges**

* Provider ABI for **audio** & **FFT/conv**; plan caching.
* Python SDK: `compile(graph)`, `play(profile)`, `inject(evt)`.

**Week 6 — Frontends**

* Morphogen.Audio emits Graph IR; RiffStack transpiles YAML/RPN → Graph IR.
* Golden tests and first public examples.

---

## 12) Why this is "the best version" of the stack

* **Conceptually minimal**: one kernel, one registry, one graph, one transform grammar.
* **Practically maximal**: supports deterministic audio today, PDE/visuals tomorrow, ML and geometry later — without revisiting fundamentals.
* **Friendly by default**: RiffStack stays playful; Morphogen.Audio stays expressive; both inherit safety and quality from the kernel.

---

### One-liner

> Morphogen is a **semantic, deterministic transform kernel** with two human-friendly faces: **Morphogen.Audio** for composition and **RiffStack** for performance — all powered by a single operator registry, a neutral graph IR, and first-class domain transforms.

---

## See Also

**Related Architecture Documents**:
- **[ECOSYSTEM_MAP.md](ECOSYSTEM_MAP.md)** — Complete map of kernel domains, domain libraries, and frontends
- **[docs/architecture/domain-architecture.md](docs/architecture/domain-architecture.md)** — Deep technical vision covering 20+ domains (2,266 lines)
- **[docs/architecture/gpu-mlir-principles.md](docs/architecture/gpu-mlir-principles.md)** — GPU lowering patterns and MLIR integration

**Domain Specifications**:
- **[docs/specifications/](docs/specifications/)** — 19 comprehensive domain specs (Circuit, Chemistry, Physics, Video, etc.)
- **[docs/specifications/graph-ir.md](docs/specifications/graph-ir.md)** — Graph IR specification (frontend-kernel boundary)
- **[docs/specifications/operator-registry.md](docs/specifications/operator-registry.md)** — Complete operator catalog

**Implementation Guides**:
- **[docs/guides/domain-implementation.md](docs/guides/domain-implementation.md)** — Step-by-step guide for adding new domains
- **[docs/adr/](docs/adr/)** — Architectural decision records explaining key design choices

**Professional Applications**:
- **[docs/reference/professional-domains.md](docs/reference/professional-domains.md)** — Value proposition across engineering, science, finance, creative fields

---

**Version:** 1.0 Draft
**Last Updated:** 2025-11-15
**Status:** Finalized Architecture
