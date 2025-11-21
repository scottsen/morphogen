# SPEC: Transform Composition - Composable Named Transforms

**Version:** 1.0
**Status:** RFC (Request for Comments)
**Last Updated:** 2025-11-21

---

## Overview

This specification extends Morphogen's [Transform Dialect](transform.md) with **composable named transforms** — reusable, chainable transformation pipelines that can be defined once and used throughout a program.

**Key Ideas:**
- **Named transforms** as first-class language constructs
- **Automatic inversion** for invertible transforms
- **Composition algebra** for building complex pipelines
- **Type-safe** transform chaining with representation tracking
- **Domain-polymorphic** transforms that work across domains

**Prerequisites:**
- [Transform Dialect](transform.md) — Base transform operations
- [Universal DSL Principles](../philosophy/universal-dsl-principles.md) — Design philosophy
- [Continuous-Discrete Semantics](../architecture/continuous-discrete-semantics.md) — Execution models

**Related:**
- [Operator Foundations](../philosophy/operator-foundations.md) — Spectral and operator theory
- [Categorical Structure](../philosophy/categorical-structure.md) — Functorial semantics

---

## Motivation

### Problem

**Currently:**
```morphogen
// Repetitive transform chains
let spec1 = signal1 |> fft |> magnitude |> mel_scale |> log
let spec2 = signal2 |> fft |> magnitude |> mel_scale |> log
let spec3 = signal3 |> fft |> magnitude |> mel_scale |> log
```

**Issues:**
- Code duplication
- No automatic inverse
- No reusability across programs
- No composition

---

### Solution: Named Composable Transforms

**Define once:**
```morphogen
@transform audio_to_mel_spectrogram {
    signal : Stream<f32, audio:time>
    -> fft
    -> magnitude
    -> mel_scale(n_mels=128)
    -> log(offset=1e-6)
    -> Stream<f32, audio:mel>
}
```

**Use everywhere:**
```morphogen
let mel_spec1 = audio_to_mel_spectrogram(signal1)
let mel_spec2 = audio_to_mel_spectrogram(signal2)

// Automatic inverse (when possible)
let reconstructed = inv(audio_to_mel_spectrogram)(mel_spec1)
```

**Benefits:**
- ✅ Reusable across program
- ✅ Automatic type inference
- ✅ Composable with other transforms
- ✅ Self-documenting
- ✅ Invertible (when theoretically possible)

---

## Syntax

### Basic Transform Definition

**Syntax:**
```morphogen
@transform <name> {
    <input_type>
    -> <transform1>
    -> <transform2>
    -> ...
    -> <output_type>
}
```

**Example:**
```morphogen
@transform time_to_frequency {
    Stream<f32, audio:time, 48kHz>
    -> fft(window="hann", norm="ortho")
    -> Stream<Complex<f32>, audio:frequency, 24kHz>
}
```

---

### Parametric Transforms

**Syntax:**
```morphogen
@transform <name>(params...) {
    ...
}
```

**Example:**
```morphogen
@transform mel_spectrogram(n_fft=2048, hop_length=512, n_mels=128) {
    Stream<f32, audio:time>
    -> stft(n_fft, hop_length, window="hann")
    -> magnitude
    -> mel_scale(n_mels)
    -> log(offset=1e-6)
    -> Stream<f32, audio:mel>
}
```

**Usage:**
```morphogen
// Use default parameters
let mel1 = mel_spectrogram(signal)

// Override parameters
let mel2 = mel_spectrogram(signal, n_mels=256, n_fft=4096)
```

---

### Transform Composition

**Compose transforms with `∘` (compose operator):**

```morphogen
@transform audio_features = mel_spectrogram ∘ normalize ∘ delta_features

// Equivalent to:
let features = signal
    |> mel_spectrogram
    |> normalize
    |> delta_features
```

**Composition is associative:**
```
(f ∘ g) ∘ h = f ∘ (g ∘ h)
```

---

### Inverse Transforms

**Automatic inverse (when mathematically possible):**

```morphogen
@transform fwd {
    x -> fft -> magnitude -> y
}

// Inverse is automatically defined (if invertible)
let y = fwd(x)
let x_reconstructed = inv(fwd)(y)  // May be approximate
```

**Invertibility properties:**

| Transform | Invertible? | Notes |
|-----------|------------|-------|
| `fft` | ✅ Yes | Exact (unitary) |
| `magnitude` | ❌ No | Loses phase information |
| `mel_scale` | ⚠️ Approximate | Non-linear warping, can pseudo-invert |
| `log` | ✅ Yes | `exp` is exact inverse |
| `normalize` | ⚠️ Conditional | If normalization stats are stored |

**Compiler behavior:**
- **Exact inverse:** Compiler generates exact inverse
- **Approximate inverse:** Compiler warns, generates best-effort inverse
- **No inverse:** Compile error if `inv()` is called

---

## Semantics

### Type Inference

**Input and output types are tracked:**

```morphogen
@transform mel_spectrogram {
    Stream<f32, audio:time, R>
    -> stft -> magnitude -> mel_scale -> log
    -> Stream<f32, audio:mel, R/hop_length>
}

// Type checker verifies:
input : Stream<f32, audio:time, 48kHz>
output : Stream<f32, audio:mel, 93Hz>  // 48000 / 512
```

**Type error example:**
```morphogen
@transform invalid {
    Stream<f32, audio:time>
    -> fft  // OK: time -> frequency
    -> diffuse(rate=0.1, dt=0.01)  // ERROR: diffuse expects Field2D, got Spectrum
}
```

---

### Representation Tracking

**Domains define valid representations:**

```morphogen
domain audio {
    representations: [time, frequency, cepstral, mel]

    // Valid transform paths
    time -> frequency: fft, stft
    frequency -> time: ifft, istft
    frequency -> mel: mel_scale
    time -> cepstral: dct
}
```

**Compiler enforces valid paths:**
```morphogen
// OK: time -> frequency -> mel
audio.time |> fft |> mel_scale

// ERROR: No direct path time -> mel (must go through frequency)
audio.time |> mel_scale  // Compile error
```

---

### Composition Laws

**Identity transform:**
```morphogen
@transform identity {
    x -> x
}

// Laws:
identity ∘ f = f
f ∘ identity = f
```

**Associativity:**
```morphogen
(f ∘ g) ∘ h = f ∘ (g ∘ h)
```

**Inverse laws (when invertible):**
```morphogen
inv(f) ∘ f = identity
f ∘ inv(f) = identity
```

**Composition inverse:**
```morphogen
inv(f ∘ g) = inv(g) ∘ inv(f)  // Reverse order
```

---

## Examples

### Example 1: Audio Feature Extraction

**Define feature extraction pipeline:**

```morphogen
@transform audio_to_mfcc(n_fft=2048, n_mels=128, n_mfcc=13) {
    Stream<f32, audio:time>
    -> stft(n_fft, hop_length=512)
    -> magnitude
    -> mel_scale(n_mels)
    -> log(offset=1e-6)
    -> dct(type=2, norm="ortho")
    -> take_first(n_mfcc)  // Keep first N coefficients
    -> Stream<f32, audio:mfcc, 93Hz>
}

// Use in program
use audio

@state recording : AudioBuffer = audio.load("speech.wav")

flow() {
    let mfcc = audio_to_mfcc(recording)
    output mfcc
}
```

---

### Example 2: Field Transforms (Spectral Methods)

**Define spectral solver for Poisson equation:**

```morphogen
@transform spectral_poisson_solve(laplacian_eigenvalues) {
    Field2D<f32>
    -> fft2d                            // Spatial -> k-space
    -> divide_elementwise(laplacian_eigenvalues)  // Solve in k-space
    -> ifft2d                           // k-space -> spatial
    -> Field2D<f32>
}

// Use for fast Poisson solve
use field

@state rhs : Field2D<f32> = initialize_source()
@state solution : Field2D<f32>

flow() {
    // Solve ∇²φ = rhs in Fourier space (O(N log N) instead of O(N²))
    solution = spectral_poisson_solve(rhs, laplacian_eigenvalues=compute_eigenvalues())
}
```

---

### Example 3: Phase Space Transforms (Physics)

**Define Hamiltonian phase space transform:**

```morphogen
@transform canonical_coordinates_to_hamiltonian {
    State<position: Vec3, momentum: Vec3>
    -> compute_kinetic_energy
    -> compute_potential_energy
    -> sum_energies
    -> Hamiltonian<f32>
}

// Enables analysis in energy space
use physics

@state particles : State<position, momentum>

flow(dt=0.01) {
    let H = canonical_coordinates_to_hamiltonian(particles)

    // Energy should be conserved (check)
    assert(abs(H - H_initial) < 1e-6)
}
```

---

### Example 4: Cross-Domain Transform Composition

**Chain transforms across domains:**

```morphogen
// Audio -> Visual pipeline
@transform audio_to_visual {
    Stream<f32, audio:time>
    -> mel_spectrogram(n_mels=64)      // Audio domain
    -> normalize(mean=0.5, std=0.2)
    -> to_image(colormap="viridis")    // Visual domain
    -> Stream<RGB, visual, 30Hz>
}

// Use for real-time visualization
use audio, visual

@state mic_input : Stream<f32, audio:time> = audio.record()

flow() {
    let viz = audio_to_visual(mic_input)
    visual.display(viz)
}
```

---

## Implementation

### Transform Registry

**Transforms are registered like operators:**

```python
# morphogen/stdlib/transforms/audio.py

@composable_transform(
    name="mel_spectrogram",
    domain="audio",
    input_repr="time",
    output_repr="mel",
    invertible="approximate"
)
def mel_spectrogram(
    signal,
    n_fft=2048,
    hop_length=512,
    n_mels=128
):
    """Convert audio signal to mel-scaled spectrogram."""
    # Pipeline: stft -> magnitude -> mel_scale -> log
    spec = stft(signal, n_fft=n_fft, hop_length=hop_length)
    mag = magnitude(spec)
    mel = mel_scale(mag, n_mels=n_mels)
    return log(mel + 1e-6)
```

---

### Inverse Generation

**Automatic inverse for invertible transforms:**

```python
# Compiler generates inverse
def inv_mel_spectrogram(mel_spec, n_fft=2048, hop_length=512, n_mels=128):
    """Approximate inverse of mel_spectrogram."""
    # Pipeline: exp -> inv_mel_scale -> istft
    mag = exp(mel_spec)
    spec_mag = inv_mel_scale(mag, n_mels=n_mels)

    # Phase reconstruction (approximate - use Griffin-Lim)
    spec = phase_reconstruction(spec_mag, method="griffin_lim")

    return istft(spec, hop_length=hop_length)
```

---

### Composition Optimization

**Compiler fuses composed transforms:**

```morphogen
@transform pipeline = f ∘ g ∘ h

// Compiled as single fused kernel (when possible)
let result = pipeline(input)

// Instead of:
// temp1 = h(input)
// temp2 = g(temp1)
// result = f(temp2)
```

**Fusion rules:**
- Consecutive spectral transforms → single FFT
- Consecutive element-wise ops → single kernel
- Consecutive filters → frequency-domain multiplication

---

## Advanced Features

### Conditional Transforms

**Choose transform based on runtime condition:**

```morphogen
@transform adaptive_denoise(noise_level) {
    if noise_level > 0.5:
        signal -> wavelet_denoise(threshold=0.3)
    else:
        signal -> gaussian_blur(sigma=1.0)
}
```

---

### Multi-Input Transforms

**Transforms with multiple inputs:**

```morphogen
@transform cross_correlation {
    (signal1: Stream<f32>, signal2: Stream<f32>)
    -> (fft(signal1), fft(signal2))
    -> multiply_conjugate
    -> ifft
    -> Stream<f32, correlation>
}
```

---

### Learned Transforms

**Transforms with learnable parameters:**

```morphogen
@transform learned_encoder(params: NeuralNetParams) {
    Image<RGB>
    -> apply_neural_net(params)
    -> Embedding<f32, 512>
}

// Parameters updated during training
flow() {
    let embedding = learned_encoder(image, params=trained_params)
}
```

---

## Catalog of Common Transforms

### Audio

| Transform | Input | Output | Invertible? |
|-----------|-------|--------|-------------|
| `fft` | time | frequency | ✅ Exact |
| `stft` | time | time-frequency | ✅ Exact |
| `mel_spectrogram` | time | mel | ⚠️ Approximate |
| `mfcc` | time | mfcc | ❌ No (lossy) |
| `chromagram` | time | chroma | ⚠️ Approximate |

### Fields

| Transform | Input | Output | Invertible? |
|-----------|-------|--------|-------------|
| `fft2d` | spatial | k-space | ✅ Exact |
| `wavelet2d` | spatial | wavelet | ✅ Exact |
| `dct2d` | spatial | dct | ✅ Exact |
| `eigenbasis` | standard | eigen | ✅ Exact |

### Physics

| Transform | Input | Output | Invertible? |
|-----------|-------|--------|-------------|
| `position_to_phase` | position | phase-space | ✅ Exact |
| `energy_to_action` | energy | action | ✅ Exact |
| `cart_to_polar` | cartesian | polar | ✅ Exact |

---

## Status & Roadmap

### ✅ Currently Supported

**Basic transforms:**
- `fft`, `ifft` (1D)
- `stft`, `istft` (2D time-frequency)
- `dct` (cepstral)
- `fft2d`, `ifft2d` (spatial)

**Operators exist, but not as composable named transforms.**

---

### 🚧 Planned (This Spec)

**Language features:**
- `@transform` declaration syntax
- Transform composition (`∘`)
- Automatic inverse (`inv()`)
- Parametric transforms
- Type-safe composition

**Transform catalog:**
- Audio: `mel_spectrogram`, `mfcc`, `chromagram`
- Field: Spectral Poisson solver, wavelet denoise
- Physics: Phase space, Hamiltonian, canonical

---

### 🔮 Future

**Advanced features:**
- Conditional transforms (runtime dispatch)
- Multi-input/multi-output transforms
- Learned transforms (neural networks)
- Adaptive transforms (parameter tuning)
- Transform equivalence (automatic simplification)

---

## Design Guidelines

### When to Define a Named Transform

**Define a named transform if:**
- ✅ Pipeline used multiple times in program
- ✅ Common pattern in domain (e.g., MFCC in audio)
- ✅ Reusable across programs
- ✅ Self-documenting (name explains what it does)

**Don't define if:**
- ❌ Used only once
- ❌ Highly specific to one use case
- ❌ Too simple (single operation)

---

### Invertibility Guidelines

**Mark as invertible if:**
- ✅ Mathematically invertible (FFT, rotation, etc.)
- ✅ Lossless (no information dropped)

**Mark as approximate if:**
- ⚠️ Information lost but reconstruction possible (mel scale)
- ⚠️ Phase lost but magnitude preserved
- ⚠️ Requires additional assumptions (Griffin-Lim)

**Mark as non-invertible if:**
- ❌ Fundamentally lossy (dimensionality reduction without inverse)
- ❌ No known inverse method

---

## Further Reading

**Specifications:**
- [Transform Dialect](transform.md) — Base transform operations
- [Type System](type-system.md) — Type inference and checking

**Philosophy:**
- [Universal DSL Principles](../philosophy/universal-dsl-principles.md) — Design foundations
- [Operator Foundations](../philosophy/operator-foundations.md) — Spectral theory

**Architecture:**
- [Continuous-Discrete Semantics](../architecture/continuous-discrete-semantics.md) — Execution models
- [Domain Architecture](../architecture/domain-architecture.md) — Domain specifications

**ADRs:**
- [Universal Domain Translation](../adr/012-universal-domain-translation.md) — Translation framework

---

## Summary

**Composable named transforms enable:**

1. **Reusability** — Define once, use everywhere
2. **Composition** — Chain transforms algebraically (`f ∘ g ∘ h`)
3. **Invertibility** — Automatic inverse when theoretically possible
4. **Type safety** — Representation tracking and validation
5. **Self-documentation** — Names capture intent

**Example:**
```morphogen
@transform audio_to_mel_spectrogram {
    signal -> stft -> magnitude -> mel_scale -> log
}

// Use anywhere
let mel = audio_to_mel_spectrogram(recording)

// Automatic approximate inverse
let reconstructed = inv(audio_to_mel_spectrogram)(mel)
```

**This makes transform-first thinking practical and powerful.**

---

**Next:** See [Universal Domain Translation](../adr/012-universal-domain-translation.md) for cross-domain translation semantics, or [Transform Dialect](transform.md) for base transform operations.
