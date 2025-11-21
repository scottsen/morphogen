# ADR-003: Instrument Modeling Domain for Timbre Extraction and Synthesis

**Status:** Proposed
**Date:** 2025-11-15
**Decision Makers:** Scott Sen
**Related:** [specifications/timbre-extraction.md](../specifications/timbre-extraction.md), [ADR-002](002-cross-domain-architectural-patterns.md), [architecture/domain-architecture.md](../architecture/domain-architecture.md)

---

## Context

Morphogen's cross-domain operator architecture naturally supports audio synthesis, physical modeling, and spectral transforms. A critical gap exists: **timbre extraction and instrument modeling** — the ability to analyze an acoustic recording and create a reusable synthesis model.

This is one of the **holy grails of audio DSP**:
> Record acoustic guitar → extract timbre → synthesize new notes with the same sonic character

Real-world precedents exist:
- **Yamaha VL1/VL70m** (physical modeling synthesizers, 1994)
- **Karplus-Strong** algorithm (plucked string synthesis, 1983)
- **Google Magenta NSynth** (neural audio synthesis, 2017)
- **Modal synthesis** systems (IRCAM Modalys, 1990s)
- **Additive resynthesis** tools (SPEAR, AudioSculpt, 2000s)

**Morphogen can unify all of these techniques** in a way that is:
- Extensible (operator-driven)
- Domain-aware (audio + physics + transforms)
- GPU-accelerated (MLIR compilation)
- Deterministic (reproducible analysis/synthesis)

---

## Decision

We will add a **new Layer 7 domain: InstrumentModeling** that composes operators from existing domains (Audio, Transform, Physics, Stochastic) to provide:

1. **Timbre analysis** — Extract fundamental frequency, harmonic structure, modal resonances, excitation characteristics, decay rates, and noise signatures from audio recordings
2. **Instrument models** — Reusable, parameterized representations of instrument timbre
3. **Resynthesis** — Generate new notes with extracted timbre at arbitrary pitches
4. **Morphing** — Blend timbre characteristics between instruments
5. **Physics integration** — Use extracted parameters to drive physical models

**This domain will NOT:**
- Duplicate existing operators (reuses Transform, Audio, Stochastic)
- Require new MLIR dialects (composes existing dialects)
- Be a monolithic "black box" (exposes composable analysis operators)

---

## Architectural Design

### Layer Structure

```
Layer 7: InstrumentModeling (NEW)
├── Uses Layer 2: Transform (STFT, FFT, wavelet)
├── Uses Layer 3: Stochastic (noise modeling, Monte Carlo)
├── Uses Layer 4: Physics (modal analysis, integrators)
├── Uses Layer 5: Audio (filters, effects, synthesis)
└── Produces: Reusable InstrumentModel objects
```

### Three Sub-Domains

**1. AudioAnalysisDomain** (13 new operators)
- Purpose: Extract timbre features from recordings
- Examples: `harmonic.track_fundamental`, `modal.analyze`, `deconvolve`, `spectral.envelope`

**2. SynthesisDomain (Extended)** (6 operators, some existing)
- Purpose: Synthesize audio from extracted features
- Examples: `additive.synth`, `modal.synth`, `excitation.pluck`, `spectral.filter`

**3. InstrumentModelDomain** (5 high-level operators)
- Purpose: Manage instrument models (analyze, synthesize, morph, save/load)
- Examples: `instrument.analyze`, `instrument.synthesize`, `instrument.morph`

### Core Type: InstrumentModel

```morphogen
type InstrumentModel {
  id: String                       // Identifier
  type: Enum                       // "modal_string", "additive", etc.

  // Analysis results
  fundamental: Ctl[Hz]             // Base pitch
  harmonics: Field2D<f32>          // Time-varying harmonic amplitudes
  modes: ModalModel                // Resonant body modes
  body_ir: IR                      // Impulse response
  noise: NoiseModel                // Noise signature
  excitation: ExcitationModel      // Attack/pluck model
  decay_rates: Field1D<f32>        // Per-partial decay
  inharmonicity: f32               // Deviation from perfect harmonics

  // Synthesis parameters
  synth_params: Map<String, f32>   // Runtime controls
}
```

---

## Rationale

### Why This is a Perfect Fit for Morphogen

**1. Cross-Domain Composition**

Timbre extraction naturally uses multiple domains:
- **Transform:** STFT for spectral analysis
- **Stochastic:** Noise modeling, parameter estimation
- **Physics:** Modal decomposition, resonance analysis
- **Audio:** Filters, synthesis, effects

Morphogen's operator architecture makes this composition **natural and type-safe**.

---

**2. MLIR Acceleration Opportunities**

Key operators are embarrassingly parallel:
- STFT: Batched FFT on GPU
- Harmonic tracking: Parallel peak detection
- Modal analysis: GPU-accelerated least-squares fitting
- Additive synthesis: Parallel sinusoid banks
- Modal synthesis: Parallel damped oscillators

**Example MLIR lowering:**
```mlir
// Additive synthesis (sum of sinusoids)
%harmonics = morphogen.field.load %model.harmonics : !morphogen.field<f32>
%time = morphogen.stream.time : !morphogen.stream<f32>

// Lower to parallel vector operations
%result = linalg.generic {
  iterator_types = ["parallel", "reduction"]
} ins(%harmonics, %time) outs(%output) {
  // GPU-friendly sinusoid summation
  %sin = math.sin %phase : f32
  %scaled = arith.mulf %sin, %amplitude : f32
  linalg.yield %scaled : f32
}
```

---

**3. Deterministic Semantics**

All analysis and synthesis operators can be deterministic:
- **Strict:** Additive/modal synthesis (bit-exact)
- **Repro:** Modal fitting, deconvolution (FP precision)
- **Live:** Real-time adaptive analysis (replayable)

This enables:
- **Regression testing** (golden waveform comparison)
- **A/B testing** (fidelity measurement)
- **Reproducible research** (published models match paper results)

---

**4. Extensibility**

The operator architecture makes adding new techniques trivial:
- New analysis method? → Add operator to AudioAnalysisDomain
- New synthesis technique? → Add operator to SynthesisDomain
- New physical model? → Compose existing operators

**Examples of future extensions:**
- Neural timbre embeddings (Layer 7: Neural Operators)
- Spectral morphing (Transform domain extension)
- Real-time adaptive analysis (Stochastic + Audio)

---

**5. Real-World Impact**

This capability unlocks **high-value use cases**:

| Use Case | Enabled By | Value |
|----------|------------|-------|
| **MIDI instrument creation** | Analyze one note → synthesize any pitch | Musicians can "sample" instruments with one recording |
| **Timbre morphing** | Blend models | Novel hybrid instruments (guitar+violin) |
| **Luthier analysis** | Measure decay, resonance, inharmonicity | Quantify instrument quality |
| **Virtual acoustics** | Extract body IR | Apply guitar body to synths/drums |
| **Physics-informed synthesis** | Modal parameters → physical model | Expressive, controllable synthesis |
| **Archive preservation** | Digitize vintage instruments | Historical instrument banks |

---

## Alternatives Considered

### Alternative 1: External Tool (e.g., SPEAR, AudioSculpt)

**Rejected because:**
- ❌ Breaks Morphogen's unified workflow (external I/O, different language)
- ❌ No GPU acceleration
- ❌ Not deterministic (different tools, different results)
- ❌ Can't compose with Morphogen's physics/fields

---

### Alternative 2: Neural-Only Approach (NSynth-style)

**Rejected as sole solution because:**
- ❌ Not interpretable (black-box latent space)
- ❌ Requires large training datasets
- ❌ No physical parameters for luthiers/engineers
- ⚠️ **Could complement** physics-based approach later

**Potential future:** Hybrid approach (physics + neural embeddings)

---

### Alternative 3: Plugin/Library Integration (e.g., Essentia, librosa)

**Rejected because:**
- ❌ Python-only (no MLIR acceleration)
- ❌ Not type-safe (no unit checking)
- ❌ Not deterministic across versions
- ⚠️ **Could use** as reference implementation for validation

---

### Alternative 4: Wait for Neural Operators (Layer 7)

**Rejected because:**
- ❌ Physics-based methods are mature and well-understood now
- ❌ High value even without neural components
- ⚠️ **Can add** neural methods later (orthogonal)

---

## Implementation Strategy

### Phase 1: Foundation (v0.9)

**Goal:** Basic analysis and resynthesis

**Tasks:**
- [ ] Implement 5 core analysis operators:
  - `harmonic.track_fundamental`
  - `harmonic.track_partials`
  - `spectral.envelope`
  - `decay.fit_exponential`
  - `inharmonicity.measure`
- [ ] Implement 2 synthesis operators:
  - `additive.synth`
  - `spectral.filter`
- [ ] Define `InstrumentModel` type
- [ ] Golden tests (analyze → synthesize → measure error)

**Success criteria:** <10% spectral error on simple plucked string

---

### Phase 2: Full Pipeline (v0.10)

**Goal:** End-to-end record → model → synthesize

**Tasks:**
- [ ] Implement `instrument.analyze` (high-level pipeline)
- [ ] Implement `instrument.synthesize`
- [ ] Add serialization (`instrument.save`, `instrument.load`)
- [ ] Multi-note analysis (keymap support)
- [ ] Validation suite (10+ instruments)

**Success criteria:** <5% spectral error on guitar, violin, piano

---

### Phase 3: Advanced Features (v0.11)

**Goal:** Modal analysis, deconvolution, morphing

**Tasks:**
- [ ] Implement `modal.analyze` (damped sinusoid fitting)
- [ ] Implement `modal.synth`
- [ ] Implement `deconvolve` (excitation/resonator separation)
- [ ] Implement `instrument.morph`
- [ ] GPU acceleration (MLIR lowering for analysis ops)

**Success criteria:** Real-time synthesis on GPU (1000+ voices)

---

### Phase 4: Integration (v1.0)

**Goal:** Cross-domain workflows

**Tasks:**
- [ ] Integrate with Physics domain (extracted modes → physical model)
- [ ] Integrate with Fields (spectrograms as 2D fields)
- [ ] Integrate with Visual (real-time timbre visualization)
- [ ] Web export (instrument models → WebAudio)

---

## Risks and Mitigations

### Risk 1: Analysis Quality (Polyphonic Recordings)

**Problem:** Most analysis techniques assume monophonic input (one note at a time)

**Mitigation:**
- Phase 1: Focus on monophonic recordings (single notes)
- Future: Add polyphonic pitch detection (PYIN, multi-pitch tracking)

---

### Risk 2: Computational Cost (Real-Time)

**Problem:** Modal fitting, deconvolution are expensive

**Mitigation:**
- **Analysis is offline** (precompute models)
- **Synthesis is real-time** (GPU-accelerated)
- Trade-off: Slower analysis for faster synthesis

---

### Risk 3: Perceptual Fidelity

**Problem:** Spectral metrics don't always match human perception

**Mitigation:**
- Use perceptual metrics (Mel-frequency cepstral distance)
- Listening tests (A/B comparison)
- Golden waveform database (regression testing)

---

### Risk 4: Determinism (Iterative Algorithms)

**Problem:** Modal fitting uses iterative solvers (non-deterministic on GPU)

**Mitigation:**
- Use deterministic solvers (QR decomposition, not SGD)
- Profile override: `strict` uses CPU, `live` uses GPU
- Document FP precision limits

---

## Success Metrics

### Technical Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Spectral error** | <5% L2 norm | `spectral.distance(original, resynthesized)` |
| **Pitch accuracy** | <1 cent | `pitch.compare(original, resynthesized)` |
| **Timbre preservation** | >0.9 MFCC correlation | `mfcc.distance(original, resynthesized)` |
| **Real-time synthesis** | >1000 voices @ 48kHz | GPU benchmark |
| **Model size** | <1MB per instrument | Serialized InstrumentModel size |

---

### User Impact Metrics

| Metric | Target | Evidence |
|--------|--------|----------|
| **Use cases enabled** | 5+ documented workflows | Examples, tutorials |
| **Instruments modeled** | 20+ validated | Golden test suite |
| **Community adoption** | 10+ user-contributed models | Model repository |

---

## Cross-References

### Related ADRs
- **[ADR-001: Unified Reference Model](001-unified-reference-model.md)** — Anchor system applies to timbre features
- **[ADR-002: Cross-Domain Architectural Patterns](002-cross-domain-architectural-patterns.md)** — Operator registries, passes

### Related Specifications
- **[specifications/timbre-extraction.md](../specifications/timbre-extraction.md)** — Full technical specification
- **[specifications/transform.md](../specifications/transform.md)** — FFT, STFT operators
- **[specifications/operator-registry.md](../specifications/operator-registry.md)** — Operator metadata
- **[AUDIO_SPECIFICATION.md](../AUDIO_SPECIFICATION.md)** — Audio domain
- **[architecture/domain-architecture.md](../architecture/domain-architecture.md)** — Domain layers

### Academic References
- Karplus & Strong (1983) — Plucked string synthesis
- Smith (1992) — Physical modeling with digital waveguides
- Morrison & Adrien (1993) — Modal synthesis framework
- Serra & Smith (1990) — Spectral modeling synthesis
- Engel et al. (2017) — NSynth (neural audio synthesis)

---

## Decision Log

### 2025-11-15: Initial Proposal
- **Decision:** Add InstrumentModeling as Layer 7 domain
- **Rationale:** Perfect fit for Morphogen's cross-domain architecture
- **Status:** Proposed (awaiting implementation)

---

## Conclusion

Adding an **InstrumentModeling domain** to Morphogen is:

✅ **High-value** — Unlocks transformative use cases (MIDI instruments, timbre morphing, luthier tools)
✅ **Architecturally sound** — Composes existing domains (Audio, Transform, Physics)
✅ **Technically feasible** — Well-established DSP techniques
✅ **GPU-friendly** — Parallelizable analysis and synthesis
✅ **Deterministic** — Reproducible results
✅ **Extensible** — Easy to add new methods

**This is not just doable — it's a perfect showcase of Morphogen's cross-domain operator model.**

---

**Status:** ✅ Approved for implementation (v0.9+)

---

**End of ADR-003**
