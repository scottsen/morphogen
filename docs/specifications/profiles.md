# SPEC: Morphogen Profile System

**Version:** 1.0 Draft
**Status:** RFC
**Last Updated:** 2025-11-13

---

## Overview

The **Morphogen Profile System** governs determinism, precision, performance, and operator behavior across the entire execution pipeline. Profiles are the contract between user intent and kernel execution.

**Core Principle:** Profiles must have permanent, locked-down semantics. Once defined, behavior cannot change across Morphogen versions.

---

## Three Core Profiles

Morphogen defines **three execution profiles** that cover the determinism-performance tradeoff space:

| Profile | Determinism | Performance | Use Case |
|---------|-------------|-------------|----------|
| **strict** | Bit-exact | Slowest | Golden tests, archival, debugging |
| **repro** | Deterministic within FP | Balanced | Production audio, simulations |
| **live** | Replayable | Fastest | Live performance, interactive |

---

## Profile: strict

**Guarantee:** Bit-exact results across devices, OS, compiler versions, and runs.

### FP Behavior
- **Precision:** `f64` by default (unless explicitly overridden)
- **Flush-to-zero:** Disabled
- **Denormal handling:** Full IEEE 754 compliance
- **FMA (fused multiply-add):** Disabled (explicit rounding at each step)
- **Math library:** Reference implementations only (no vendor optimizations)

### FFT Behavior
- **Provider:** Reference implementation (Ooura or equivalent)
- **Normalization:** `ortho` (orthonormal, symmetric scaling)
- **Window coefficients:** Exact (no approximations)
- **Bit-reversal ordering:** Stable, deterministic

### Convolution Behavior
- **Partitioning:** Disabled (full direct convolution)
- **FFT-based:** Only if bit-exact FFT available

### Randomness
- **RNG:** Philox 4×32-10 (counter-based, deterministic)
- **Seeding:** Explicit seed required (no default seed)
- **Seed formula:** `hash64(global_seed, operator_id, tick, local_seed)`

### Block Size
- **Default:** 64 samples (power of 2)
- **Oversampling:** 1× (no upsampling unless explicit)

### Solver Behavior
- **Iterative solvers:** Fixed iteration count (no adaptive termination)
- **Tolerance:** Ignored (always run max_iters)
- **Convergence:** Deterministic ordering of operations

### Implicit Casts
- **Allowed:** None (all casts must be explicit)

---

## Profile: repro

**Guarantee:** Deterministic within floating-point precision. Same input → same output (within ~1e-7 relative error for f32).

### FP Behavior
- **Precision:** `f32` by default
- **Flush-to-zero:** Enabled (performance optimization)
- **Denormal handling:** Flush to zero
- **FMA:** Enabled (faster, but different rounding)
- **Math library:** Vendor-optimized (SVML, ARM NEON, etc.)

### FFT Behavior
- **Provider:** Vendor libraries allowed (FFTW, MKL, vDSP, cuFFT)
- **Normalization:** `ortho` (but vendor-specific algorithms allowed)
- **Window coefficients:** Vendor-optimized (within 1e-7 of reference)
- **Bit-reversal ordering:** Vendor-specific (must be deterministic)

### Convolution Behavior
- **Partitioning:** Allowed (overlap-add, overlap-save)
- **FFT-based:** Allowed (if faster than direct)

### Randomness
- **RNG:** Philox 4×32-10 (same as strict)
- **Seeding:** Explicit seed required
- **Seed formula:** Same as strict

### Block Size
- **Default:** 128 samples
- **Oversampling:** Allowed (2×, 4× for nonlinear ops)

### Solver Behavior
- **Iterative solvers:** Adaptive termination allowed (with tolerance)
- **Tolerance:** Default 1e-6 (user configurable)
- **Convergence:** Early exit when tolerance met

### Implicit Casts
- **Allowed:** Safe promotions only (f32 → f64, i32 → i64)

---

## Profile: live

**Guarantee:** Replayable (same input → same output), but not bit-exact. Optimized for lowest latency.

### FP Behavior
- **Precision:** `f32` (or `f16` on supported hardware)
- **Flush-to-zero:** Enabled
- **Denormal handling:** Flush to zero
- **FMA:** Enabled
- **Math library:** Fastest vendor implementation

### FFT Behavior
- **Provider:** Fastest available (even approximations allowed)
- **Normalization:** `backward` (1/N on inverse only, faster)
- **Window coefficients:** Approximations allowed
- **Bit-reversal ordering:** Any stable ordering

### Convolution Behavior
- **Partitioning:** Aggressive (small blocks for low latency)
- **FFT-based:** Always preferred

### Randomness
- **RNG:** Philox 4×32-10 (same algorithm, but fast-path)
- **Seeding:** Auto-seed from timestamp if not provided
- **Seed formula:** Same as strict (if explicit seed)

### Block Size
- **Default:** 32-64 samples (minimize latency)
- **Oversampling:** Skipped unless critical (nonlinear only)

### Solver Behavior
- **Iterative solvers:** Aggressive early exit
- **Tolerance:** Relaxed (1e-4)
- **Convergence:** Approximate solutions accepted

### Implicit Casts
- **Allowed:** All safe casts + lossy casts with warning

---

## Profile Configuration Schema

### Global Profile

```json
{
  "profile": "repro",
  "profile_config": {
    "precision": "f32",
    "flush_to_zero": true,
    "fft_provider": "fftw",
    "fft_norm": "ortho",
    "block_size": 128,
    "oversampling": 2,
    "rng_seed": 42,
    "solver_tolerance": 1e-6,
    "max_iterations": 100
  }
}
```

---

### Per-Operator Overrides

Operators can override profile settings:

```json
{
  "id": "reverb1",
  "op": "convolution",
  "params": {"ir": "@resource:hall_reverb.wav"},
  "profile_overrides": {
    "precision": "f64",
    "fft_provider": "reference",
    "determinism": "strict"
  }
}
```

**Precedence:** Operator > Module > Scene > Global profile

---

## Profile Semantics by Component

### 1. Type System

| Setting | strict | repro | live |
|---------|--------|-------|------|
| Default precision | f64 | f32 | f32 |
| Implicit casts | None | Safe only | All with warning |
| Unit checking | Strict | Strict | Relaxed (warnings) |

### 2. Scheduler

| Setting | strict | repro | live |
|---------|--------|-------|------|
| Block size | 64 | 128 | 32-64 |
| Event quantization | Sample-accurate | Sample-accurate | Block-accurate |
| Jitter handling | Error | Snap to boundary | Ignore |

### 3. FFT Transforms

| Setting | strict | repro | live |
|---------|--------|-------|------|
| Provider | Reference | Vendor (FFTW) | Fastest |
| Normalization | ortho | ortho | backward |
| Accuracy | Bit-exact | 1e-7 relative | 1e-4 relative |

### 4. Solvers (Iterative)

| Setting | strict | repro | live |
|---------|--------|-------|------|
| Iterations | Fixed (max) | Adaptive | Aggressive exit |
| Tolerance | Ignored | 1e-6 | 1e-4 |
| Convergence | Full | Early exit | Approximate |

### 5. Randomness

| Setting | strict | repro | live |
|---------|--------|-------|------|
| Algorithm | Philox 4×32-10 | Philox 4×32-10 | Philox 4×32-10 |
| Seeding | Explicit required | Explicit required | Auto-seed allowed |
| Determinism | Bit-exact | Bit-exact | Replayable |

---

## Operator Determinism Metadata

Every operator in the registry declares its **determinism tier**:

```json
{
  "name": "fft",
  "determinism_tiers": {
    "strict": {
      "provider": "reference",
      "norm": "ortho",
      "accuracy": "bit-exact"
    },
    "repro": {
      "provider": "fftw",
      "norm": "ortho",
      "accuracy": "1e-7"
    },
    "live": {
      "provider": "fastest",
      "norm": "backward",
      "accuracy": "1e-4"
    }
  }
}
```

### Validation Rule

```python
def validate_operator_in_profile(op, profile):
    """Ensure operator is compatible with profile."""
    if profile == "strict" and op.determinism_tier != "strict":
        raise ProfileError(
            f"Operator {op.name} (tier={op.determinism_tier}) "
            f"not allowed in profile=strict"
        )
```

---

## Cross-Profile Compatibility

### Golden Test Vectors

Every operator must provide **golden test vectors** for strict profile:

```json
{
  "operator": "fft",
  "test_vector": {
    "input": [1.0, 0.5, 0.25, 0.125],
    "params": {"window": "hann", "norm": "ortho"},
    "expected_output": [
      {"re": 1.875, "im": 0.0},
      {"re": 0.46193977, "im": -0.19134172},
      {"re": 0.125, "im": 0.0},
      {"re": 0.46193977, "im": 0.19134172}
    ],
    "profile": "strict",
    "tolerance": 0.0  // Bit-exact
  }
}
```

### Regression Testing

Profile behavior must remain **stable across Morphogen versions**:

```python
def test_profile_stability():
    """Ensure profiles produce identical results across versions."""
    input_signal = load_golden_input("sine_440hz.wav")

    # strict profile must be bit-exact
    output_v1 = run_with_profile(input_signal, profile="strict", version="0.4.0")
    output_v2 = run_with_profile(input_signal, profile="strict", version="0.5.0")
    assert output_v1 == output_v2  // Bit-exact

    # repro profile must match within tolerance
    output_v1 = run_with_profile(input_signal, profile="repro", version="0.4.0")
    output_v2 = run_with_profile(input_signal, profile="repro", version="0.5.0")
    assert allclose(output_v1, output_v2, rtol=1e-7)
```

---

## Profile Selection Guidelines

### When to Use strict

- ✅ Archival projects (must reproduce exactly in 10 years)
- ✅ Scientific simulations (bit-exact reproducibility required)
- ✅ Debugging (eliminate all non-determinism sources)
- ✅ Golden test generation

❌ Avoid for: Real-time audio (too slow), live visuals

---

### When to Use repro

- ✅ Production audio (DAW projects, mastering)
- ✅ Physics simulations (deterministic but fast)
- ✅ Procedural generation (same seed → same output)
- ✅ Offline rendering

❌ Avoid for: Live performance (latency), strict archival

---

### When to Use live

- ✅ Live performance (synthesizers, VJ tools)
- ✅ Interactive visuals (real-time feedback)
- ✅ Game audio (low latency critical)
- ✅ Streaming applications

❌ Avoid for: Archival, regression tests, scientific reproducibility

---

## Profile Switching

### Hot Reload with Profile Change

Switching profiles may require state reinitialization:

```python
def switch_profile(graph, old_profile, new_profile):
    """Switch execution profile with state migration."""

    if old_profile == new_profile:
        return  # No change

    # Snapshot current state
    snapshot = save_snapshot(graph, profile=old_profile)

    # Migrate state to new profile
    if old_profile.precision != new_profile.precision:
        # Convert buffer precision
        snapshot = convert_precision(snapshot, new_profile.precision)

    # Reload with new profile
    load_snapshot(graph, snapshot, profile=new_profile)
```

**Constraints:**
- `strict ↔ repro`: State migration allowed (precision change)
- `repro ↔ live`: State migration allowed (approximation)
- `strict → live`: Lossy (warning)
- `live → strict`: Not guaranteed bit-exact

---

## Profile Inheritance

Profiles follow a hierarchy:

```
Global Profile (scene-level)
  ↓
Module Profile (module-level)
  ↓
Operator Override (operator-level)
```

**Example:**

```json
{
  "version": "1.0",
  "profile": "repro",  // Global default

  "modules": [
    {
      "id": "reverb_module",
      "profile": "strict",  // Module override

      "nodes": [
        {
          "id": "reverb1",
          "op": "convolution",
          "profile_overrides": {
            "precision": "f64"  // Operator override
          }
        }
      ]
    }
  ]
}
```

**Resolution:**
- `reverb1` runs with: `strict` profile + `f64` precision
- Other operators in `reverb_module` run with: `strict` profile + default precision
- Operators outside `reverb_module` run with: `repro` profile

---

## Implementation Checklist

### Phase 1: Core Profile System
- [ ] Define profile data structures
- [ ] Implement profile parser
- [ ] Add profile validation
- [ ] Profile inheritance resolver

### Phase 2: Operator Integration
- [ ] Add determinism tier to operator registry
- [ ] Implement per-operator profile overrides
- [ ] Golden test vector validation

### Phase 3: Runtime Enforcement
- [ ] FP mode switching (flush-to-zero, FMA)
- [ ] FFT provider selection
- [ ] RNG seed management
- [ ] Solver termination control

---

## Testing Strategy

### Golden Tests (strict profile)

Every operator must pass bit-exact tests:

```python
@pytest.mark.profile("strict")
def test_fft_strict():
    input = np.array([1.0, 0.5, 0.25, 0.125])
    expected = load_golden("fft_golden.npy")
    output = morphogen.fft(input, window="hann", norm="ortho", profile="strict")
    assert np.array_equal(output, expected)  # Bit-exact
```

### Determinism Tests (repro profile)

```python
@pytest.mark.profile("repro")
def test_fft_repro_determinism():
    input = np.random.randn(1024)
    output1 = morphogen.fft(input, profile="repro", seed=42)
    output2 = morphogen.fft(input, profile="repro", seed=42)
    assert np.allclose(output1, output2, rtol=1e-7)
```

### Performance Tests (live profile)

```python
@pytest.mark.profile("live")
@pytest.mark.benchmark
def test_fft_live_performance(benchmark):
    input = np.random.randn(1024)
    result = benchmark(lambda: morphogen.fft(input, profile="live"))
    assert result.avg_time < 0.001  # < 1ms
```

---

## Summary

The Morphogen Profile System provides:

✅ **Three determinism tiers** — strict, repro, live
✅ **Permanent semantics** — Locked-down behavior across versions
✅ **Fine-grained control** — Global, module, operator overrides
✅ **Explicit tradeoffs** — Performance vs reproducibility
✅ **Validation** — Golden tests, regression tests, profile compatibility

Profiles are the contract that makes Morphogen both **fast** and **correct**.

---

## References

- `type-system.md` — Determinism tiers affect type behavior
- `operator-registry.md` — Operators declare determinism tiers
- `scheduler.md` — Scheduler uses profile block sizes
- `transform.md` — Transform ops use profile FFT settings
