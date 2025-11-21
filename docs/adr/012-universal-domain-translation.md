# ADR 012: Universal Domain Translation Framework

**Status:** Proposed
**Date:** 2025-11-21
**Decision Makers:** Architecture Team
**Consulted:** Domain experts, language designers

---

## Context

Morphogen enables cross-domain composition (audio + physics + fields + agents), but currently lacks a **formal framework for domain translation**. Translation between domains is ad-hoc, semantics are implicit, and invariant preservation is not enforced.

**Current limitations:**

1. **No translation semantics** — Unclear what `agents_to_field()` preserves
2. **No invariant tracking** — Can't verify mass/energy/momentum conservation
3. **No representation spaces** — Domains don't declare natural representations
4. **No composable translations** — Can't chain domain-crossing operations
5. **Ambiguous semantics** — Does `field_to_agents()` sample or spawn?

**As Morphogen evolves toward universal cross-domain computation, we need:**
- **Explicit translation semantics**
- **Invariant preservation guarantees**
- **Composable domain-crossing operators**
- **Type-safe representation tracking**

---

## Decision

We adopt a **Universal Domain Translation Framework** with the following components:

1. **Representation Spaces** — Domains declare natural representations
2. **Translation Operators** — Explicit domain-crossing with declared semantics
3. **Invariant Specifications** — What must be preserved/dropped/approximated
4. **Composition Algebra** — Translators compose like functors
5. **Type System Integration** — Representation tags in types

---

## Detailed Design

### 1. Representation Spaces

**Each domain declares its natural representation spaces:**

```morphogen
domain audio {
    representations: [time, frequency, cepstral, mel, wavelet]
    natural: time  // Default representation

    // Transforms between representations (within domain)
    transforms: {
        time <-> frequency: (fft, ifft),
        time -> cepstral: dct,
        frequency -> mel: mel_scale
    }
}

domain field {
    representations: [spatial, k_space, wavelet, eigenbasis]
    natural: spatial

    transforms: {
        spatial <-> k_space: (fft2d, ifft2d),
        spatial <-> wavelet: (wavelet_transform, wavelet_inverse),
        spatial -> eigenbasis: eigendecompose
    }
}
```

**Benefits:**
- Makes domain structure explicit
- Guides optimization (choose best representation for operation)
- Enables automatic transform insertion

---

### 2. Translation Operators

**Explicit domain-crossing with declared semantics:**

```morphogen
@translate agents -> field {
    name: "agents_to_density_field"
    method: "kernel_density_estimation"

    // Invariants
    preserves: {
        total_mass: ∑ agent.mass = ∫ field dx,
        center_of_mass: mean(agent.pos) = ∫ x·field dx / ∫ field dx
    }
    drops: {
        individual_identity: true,
        velocity_distribution: true
    }
    approximate: true
    error_bound: 1e-3

    // Parameters
    kernel: "gaussian"
    bandwidth: 0.5
}
```

**Usage:**
```morphogen
use agent, field

@state particles : Agents<Particle>
@state density : Field2D<f32>

flow(dt=0.01) {
    // Explicit translation with verified invariants
    density = translate(particles, agents -> field,
                        method="kernel_density_estimation",
                        kernel="gaussian")

    // Compiler verifies: total_mass preserved
    assert( sum(density) ≈ sum(particles.mass) )
}
```

---

### 3. Invariant Specifications

**Three categories of invariants:**

**Preserved (must be exact or within tolerance):**
```morphogen
preserves: {
    total_mass: ∑ agents.mass = ∫ field dx,
    momentum: ∑ agents.momentum = ∫ field·velocity dx,
    energy: total_energy(agents) = total_energy(field)
}
```

**Dropped (intentionally lost, document why):**
```morphogen
drops: {
    individual_identity: "Coarse-graining to continuous field",
    velocity_variance: "Only mean velocity preserved",
    spatial_correlations: "Averaged over kernel bandwidth"
}
```

**Approximated (not exact, but controlled error):**
```morphogen
approximate: {
    spatial_distribution: "KDE approximates true density",
    error_metric: "L2_norm",
    error_bound: 1e-3
}
```

---

### 4. Translation Catalog

**Common translation patterns:**

| From | To | Method | Preserves | Drops |
|------|----|----|-----------|-------|
| **Agents** | **Field** | Kernel density | Mass, COM | Identity, velocity dist |
| **Field** | **Agents** | Monte Carlo sampling | Mass | Exact positions |
| **Audio** | **Visual** | Spectrogram → image | Frequency content | Phase |
| **Physics** | **Audio** | Pressure → waveform | Energy, frequency | Spatial distribution |
| **Graph** | **Field** | Graph Laplacian → PDE | Connectivity | Discrete structure |

---

### 5. Type System Integration

**Representation tags in types:**

```morphogen
// Before translation
particles : Agents<Particle, discrete>

// After translation
density : Field2D<f32, continuous, spatial>

// Type signature of translator
translate : Agents<T, discrete> -> Field2D<f32, continuous, R>
    where R ∈ representations(field)
```

**Compiler enforces:**
- Valid domain crossings
- Representation compatibility
- Invariant verification (when computable)

---

## Examples

### Example 1: Agents → Field (Density)

```morphogen
use agent, field

@state swarm : Agents<Particle> = alloc(count=10000)
@state density : Field2D<f32>

@translate agents_to_density {
    from: Agents<Particle>
    to: Field2D<f32>
    method: "kernel_density_estimation"

    preserves: {
        total_mass: true,
        center_of_mass: true
    }
    drops: {
        individual_identity: true,
        velocity: true
    }

    kernel: "gaussian"
    bandwidth: 0.5
}

flow(dt=0.01) {
    // Continuous representation for diffusion
    density = agents_to_density(swarm)
    density = diffuse(density, rate=0.1, dt)

    // Back to discrete (sampling)
    swarm = density_to_agents(density, count=10000)
}
```

---

### Example 2: Field → Audio (Sonification)

```morphogen
use field, audio

@state temp : Field2D<f32> = initialize()

@translate field_to_audio {
    from: Field2D<f32>
    to: Stream<f32, audio:time>
    method: "spatial_to_temporal"

    // Map spatial frequencies to audible frequencies
    preserves: {
        spectral_energy: "Parseval's theorem"
    }
    drops: {
        2D_structure: "Collapse to 1D"
    }

    sample_rate: 48000Hz
    duration: 2.0s
    frequency_scale: "logarithmic"
}

flow() {
    let sound = field_to_audio(temp)
    audio.play(sound)
}
```

---

### Example 3: Physics → Acoustics → Audio

**Multi-hop translation:**

```morphogen
use physics, acoustics, audio

@state flow : FluidField1D
@state acoustic : AcousticField1D
@state sound : Stream<f32, audio:time>

// Translation 1: Fluid → Acoustics
@translate fluid_to_acoustic {
    from: FluidField1D
    to: AcousticField1D
    method: "impedance_coupling"

    preserves: {
        pressure_energy: true,
        wavelength: true
    }
    drops: {
        vorticity: "Acoustics is irrotational"
    }
}

// Translation 2: Acoustics → Audio
@translate acoustic_to_audio {
    from: AcousticField1D
    to: Stream<f32, audio:time>
    method: "microphone_sampling"

    preserves: {
        frequency_content: true,
        amplitude_envelope: true
    }
    drops: {
        spatial_field_structure: "Point measurement"
    }

    mic_position: 1.5m
    sample_rate: 48000Hz
}

flow(dt=0.0001) {
    // Physics domain
    flow = advance_fluid(flow, dt)

    // Physics → Acoustics
    acoustic = fluid_to_acoustic(flow)

    // Acoustics → Audio
    sound = acoustic_to_audio(acoustic)

    // Real-time audio output
    audio.play(sound)
}
```

**Compiler verifies:**
- Energy conservation across physics → acoustics
- Frequency content preserved across acoustics → audio

---

## Implementation Strategy

### Phase 1: Representation Spaces (v0.12)

**Deliverables:**
- [ ] Domain representation declaration syntax
- [ ] Representation tags in type system
- [ ] Intra-domain transform catalog (FFT, wavelets, etc.)

**Timeline:** 2 weeks

---

### Phase 2: Translation Operators (v0.13)

**Deliverables:**
- [ ] `@translate` declaration syntax
- [ ] Invariant specification language
- [ ] Translation registry and lookup
- [ ] Common translations (agents ↔ field, etc.)

**Timeline:** 4 weeks

---

### Phase 3: Invariant Verification (v0.14)

**Deliverables:**
- [ ] Runtime invariant checking (assertions)
- [ ] Compile-time verification (when possible)
- [ ] Error bounds and tolerances
- [ ] Logging and diagnostics

**Timeline:** 3 weeks

---

### Phase 4: Composition (v0.15)

**Deliverables:**
- [ ] Translation composition algebra
- [ ] Multi-hop translation optimization
- [ ] Automatic translation insertion
- [ ] Type-driven translation selection

**Timeline:** 4 weeks

---

## Consequences

### Positive

**✅ Explicit semantics**
- Translation behavior is documented, not implicit

**✅ Invariant guarantees**
- Compiler/runtime can verify conservation laws

**✅ Composability**
- Translations compose like functors

**✅ Optimization**
- Choose optimal representation for operations

**✅ Debuggability**
- When translation fails, reason is clear

**✅ Reusability**
- Named translations are reusable across programs

---

### Negative

**❌ Increased complexity**
- More language features to learn

**❌ Verbosity**
- Explicit declarations are longer

**❌ Runtime overhead**
- Invariant checking adds cost (can be disabled)

**❌ Implementation effort**
- Significant compiler and runtime work

---

### Mitigations

**Complexity:**
- Provide common translations in stdlib
- Excellent documentation and examples
- Defaults for simple cases

**Verbosity:**
- Allow shorthand for common patterns
- Type inference reduces annotations

**Runtime overhead:**
- Invariant checking only in debug mode
- Profile-driven optimization

**Implementation effort:**
- Phased rollout (4 phases over ~13 weeks)
- Reuse existing transform infrastructure

---

## Alternatives Considered

### Alternative 1: Implicit Translation

**Approach:** Let compiler insert translations automatically

**Pros:**
- Less verbose
- Easier to write

**Cons:**
- ❌ Ambiguous semantics (which translation method?)
- ❌ No invariant guarantees
- ❌ Hard to debug (what did compiler choose?)
- ❌ Unpredictable behavior

**Decision:** Rejected. Explicitness is critical for correctness.

---

### Alternative 2: Ad-Hoc Operators

**Approach:** Keep current `agents_to_field()` style functions

**Pros:**
- Simpler to implement
- No new language features

**Cons:**
- ❌ No composability
- ❌ No invariant tracking
- ❌ No reusability
- ❌ Doesn't scale to many domains

**Decision:** Rejected. Doesn't meet universality goal.

---

### Alternative 3: Monadic Translation

**Approach:** Use monads for domain translations

**Pros:**
- Compositional
- Type-safe

**Cons:**
- ❌ Requires advanced type system (higher-kinded types)
- ❌ Unfamiliar to most users
- ❌ Overkill for this problem

**Decision:** Rejected. Too complex for current needs.

---

## Open Questions

### Q1: How granular should invariants be?

**Options:**
- Global invariants only (total mass, energy)
- Local invariants (per-cell, per-agent)
- Statistical invariants (mean, variance)

**Decision:** TBD — Likely support all three with different verification strategies

---

### Q2: Should translations be reversible?

**Options:**
- Require bidirectional translations
- Allow one-way translations
- Support pseudo-inverses

**Decision:** TBD — Likely allow one-way but encourage bidirectional where possible

---

### Q3: How to handle multi-scale translations?

**Example:** Molecular → Cellular → Tissue

**Decision:** TBD — May need hierarchical translation composition

---

## Related Work

**Domain-Specific Translation:**
- **Model Transformation Languages** (ATL, QVT) — General purpose, but no physics
- **Ptolemy II** — Multi-domain modeling, but no type system
- **Modelica** — Multi-physics, but no domain translation framework

**Morphogen's Contribution:**
- First to combine type-safe domain translation with invariant preservation in a unified language

---

## References

**Internal:**
- [Universal DSL Principles](../philosophy/universal-dsl-principles.md) — Design philosophy
- [Continuous-Discrete Semantics](../architecture/continuous-discrete-semantics.md) — Execution models
- [Transform Composition](../specifications/transform-composition.md) — Composable transforms
- [Operator Foundations](../philosophy/operator-foundations.md) — Spectral theory

**External:**
- **"Functorial Semantics of Algebraic Theories"** — Lawvere
- **"Model Transformation Languages: An Overview"** — ACM Survey
- **"Hybrid Systems: Computation and Control"** — HSCC

---

## Decision

**Adopted:** Universal Domain Translation Framework as specified

**Rationale:**
- Enables verifiable cross-domain composition
- Scales to arbitrary number of domains
- Makes semantics explicit and debuggable
- Aligns with category-theoretic foundations

**Implementation:** Phased rollout over 4 versions (v0.12 - v0.15)

**Review Date:** After Phase 1 completion (2 weeks)

---

**Next Steps:**
1. Review and approve this ADR
2. Begin Phase 1 implementation (representation spaces)
3. Gather community feedback on syntax
4. Iterate on invariant specification language

---

**Approved by:** [Pending]
**Date:** [Pending]
