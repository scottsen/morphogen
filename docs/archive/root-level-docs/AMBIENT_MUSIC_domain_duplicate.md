# Ambient Music & Generative Audio Domain

**Version:** 1.0 (Phase 0 - Design)
**Status:** Proposed
**Last Updated:** 2025-11-16

---

## Overview

The Ambient Music domain provides four specialized sub-domains for ambient and generative audio composition:

1. **Morphogen.Spectral** — Frequency-domain audio manipulation
2. **Morphogen.Ambience** — High-level ambient primitives (drones, granular, long-form evolution)
3. **Morphogen.Synthesis** — Modular-style DSP routing
4. **Morphogen.Composition** — Generative pattern generation (Markov, CA, swarms)

**Total:** 90 new operators across 4 domains

**Phase 0 Status:** Design and specification complete ✅
**Phase 1 Status:** Implementation pending

---

## Key Applications

**Creative Music Production:**
- Ambient music composition (Eno-style generative systems)
- Soundscape design for games and installations
- Generative background music (adaptive, infinite)
- Experimental electronic music

**Cross-Domain Art:**
- Visual + audio synthesis (fractals → melodies)
- Physics simulations → audio textures
- CA patterns → rhythmic sequences
- Swarm dynamics → spatial soundscapes

**Research & Education:**
- Generative music algorithms
- Spectral processing techniques
- Cross-modal synthesis
- GPU-accelerated DSP

**Live Performance:**
- Generative modular patches
- Adaptive ambient systems
- Long-form evolving compositions

---

## Why Morphogen for Ambient Music?

**Unique capabilities no other tool provides:**

| Morphogen Strength | Ambient Music Benefit |
|----------------|----------------------|
| **Cross-domain composition** | Physics/CA/fractals drive audio parameters |
| **GPU acceleration** | Real-time dense granular synthesis, spectral convolution |
| **Deterministic semantics** | Same seed = same evolution (reproducible compositions) |
| **Multi-hour time scales** | Ultra-slow LFOs (0.0001 Hz), drift modulators |
| **Declarative synthesis** | Compute graph = signal flow (no wire spaghetti) |
| **Operator fusion** | MLIR automatically optimizes DSP chains |

**Competing tools:**
- **Max/MSP, Pure Data:** No GPU, no CA, no physics coupling
- **SuperCollider:** No GPU, complex syntax, no cross-domain
- **VCV Rack:** CPU-only, no composition tools, no CA integration
- **DAWs:** Linear timelines, not generative/declarative
- **Eno-style apps (Bloom):** Closed systems, limited extensibility

**Morphogen unifies all of these capabilities.**

**Mathematical frameworks for composition:**
Morphogen's design is informed by mathematical structures that map naturally to music: group theory for transformations, topology for voice-leading, dynamical systems for rhythm evolution, and information theory for expectation management. See [Mathematical Music Frameworks](../reference/mathematical-music-frameworks.md) for details on how these domains enhance generative composition.

---

## Phase Roadmap

### Phase 0: Design & Specification ✅ COMPLETE

**Deliverables:**
- ✅ ADR-009 (Architecture Decision Record)
- ✅ Complete specification (90 operators, 1000+ lines)
- ✅ ECOSYSTEM_MAP.md updated
- ✅ Domain documentation (this file)

**Status:** Complete (2025-11-16)

---

### Phase 1: Spectral Domain (v0.6) — Estimated 3 months

**Operators (15):**
- `spectral.blur`, `spectral.morph`, `spectral.freeze`
- `harmonic.nebula`, `harmonic.drift`, `harmonic.spread`
- `vocode`, `spectral.filter`, `spectral.crossfade`
- `additive.resynth`, `spectral.shift`, `spectral.compress`
- `spectral.gate`, `spectral.enhance`, `spectral.smear`

**Core Types:**
```morphogen
type SpectralField = Field2D<complex, (freq: Hz, time: s)>
type Spectrogram = Field2D<f32, (freq: Hz, time: s)>
type HarmonicSeries = Field1D<f32, freq: Hz>
```

**MLIR Integration:**
- GPU-accelerated FFT (cuFFT/rocFFT)
- Spectral convolution on GPU
- Automatic operator fusion

**Examples:**
- Spectral morphing pad (noise → harmonic transition)
- Harmonic nebula synthesis (distributed harmonics)
- Spectral freeze effect (Clouds-style)

**Testing:**
- Unit tests for all 15 operators
- Golden spectrograms (regression testing)
- GPU vs CPU benchmarks (target >5x speedup)

**Status:** 🔲 Not started

---

### Phase 2: Ambience Domain (v0.7) — Estimated 4 months

**Operators (25):**

**Drones (4):**
- `drone.harmonic`, `drone.subharmonic`, `drone.pad`, `drone.shimmer`

**Granular (4):**
- `granular.cloud`, `granular.freeze`, `granular.reverse_cloud`, `granular.stretch`

**Textures (3):**
- `texture.evolving`, `texture.shimmer`, `texture.noise_field`

**Modulators (4):**
- `drift.noise`, `orbit.lfo`, `slow.random_walk`, `drift.curve`

**Evolvers (3):**
- `evolve.spectral`, `evolve.harmonic`, `evolve.density`

**Core Types:**
```morphogen
type Drone = {
  fundamental: Ctl[Hz],
  texture: Enum["smooth", "shimmer", "rough", "nebula"],
  drift: Ctl[cents/s],
  spread: Ctl[ratio]
}

type GranularCloud = {
  source: Sig,
  density: Ctl[grains/s],
  grain_size: Ctl[ms],
  randomness: Ctl[ratio],
  freeze: bool
}
```

**GPU Acceleration:**
- Parallel grain synthesis (1000s of grains)
- Dense granular clouds (>100 grains/second)
- Efficient sparse grain buffers

**Examples:**
- 20-minute evolving drone (ultra-slow modulation)
- CA-driven granular texture
- Multi-hour ambient composition

**Testing:**
- Deterministic grain playback (bit-exact)
- Long-form stability (24-hour test runs)
- Memory leak prevention

**Status:** 🔲 Not started

---

### Phase 3: Synthesis Domain (v0.8) — Estimated 3 months

**Operators (30):**

**Oscillators (5):**
- `vco`, `wavetable`, `fm`, `additive`, `phase_mod`

**Filters (5):**
- `multimode`, `formant`, `comb`, `phaser_filter`, `ladder`

**Modulation (5):**
- `lfo`, `envelope.follower`, `sample_and_hold`, `slew`, `quantizer`

**Routing (5):**
- `mix`, `crossfade`, `vca`, `pan`, `matrix_mixer`

**Core Types:**
```morphogen
type Patch = Graph<SynthOp>  // Declarative synthesis graph
type CV = Ctl  // Control voltage
type Gate = Evt<void>  // Trigger events
```

**Declarative Patching:**
- Compute graph = signal flow (no visual wiring)
- Type-safe connections (compile-time errors)
- Automatic operator fusion (MLIR optimization)
- Sample-accurate block execution

**Examples:**
- Generative modular patch (self-evolving)
- West Coast synthesis patch
- Feedback FM patch

**Testing:**
- Signal flow validation
- Type safety tests
- Patch serialization/deserialization

**Status:** 🔲 Not started

---

### Phase 4: Composition Domain (v0.9) — Estimated 4 months

**Operators (20):**

**Markov (3):**
- `markov.sequence`, `markov.parameter_walk`, `markov.interpolate`

**CA (4):**
- `ca.sequencer`, `life.triggers`, `lenia.modulation`, `rule30.noise`

**Stochastic (3):**
- `poisson.trigger`, `brownian.melody`, `drunk.walk`

**Patterns (4):**
- `euclidean.rhythm`, `fractal.timing`, `fibonacci.sequence`, `golden.ratio`

**Swarms (3):**
- `swarm.pitch_map`, `swarm.density_modulation`, `swarm.spatial_pan`

**Core Types:**
```morphogen
type MarkovMatrix = Field2D<f32, (state_from, state_to)>
type CARule = {
  rule: Enum["life", "rule30", "lenia", ...],
  params: Map<String, f32>
}
type Pattern = Evt<Note>
```

**Cross-Domain Integration:**
- CA → note triggering
- Swarm agents → melody generation
- Physics fields → parameter modulation
- Fractals → harmonic evolution

**Examples:**
- Markov chain melody generator
- Life CA sequencer
- Swarm-based spatial audio
- Fractal rhythm patterns

**Testing:**
- Deterministic sequence generation
- CA integration validation
- Swarm → audio mapping

**Status:** 🔲 Not started

---

### Phase 5: Cross-Domain Integration & Showcase (v1.0) — Estimated 2 months

**Deliverables:**
- 10 cross-domain examples (physics → audio, fractals → melody)
- Performance benchmarks (CPU vs GPU, speedup metrics)
- Comprehensive tutorial documentation
- Video demonstrations
- Golden artifacts (WAV + PNG + video renders)

**Cross-Domain Examples:**

**1. Fluid Vorticity → Granular Density**
```morphogen
fluid = navier_stokes.solve(...)
vorticity = field.curl(fluid.velocity)
grain_density = vorticity.max() * 80

grains = granular.cloud(
  source = drone.harmonic(110Hz),
  density = grain_density,
  grain_size = 60ms
)
```

**2. Mandelbrot → Harmonic Evolution**
```morphogen
mandelbrot = fractal.iterate(zoom_center, zoom_rate)
depth = mandelbrot.escape_time.mean()
harmonic_spread = depth / 100.0

pad = drone.harmonic(
  fundamental = 55Hz,
  spread = harmonic_spread,
  shimmer = 0.3
)
```

**3. Swarm → Melody Generation**
```morphogen
boids = agents.boids(num=40, cohesion=0.5)
notes = swarm.pitch_map(
  boids,
  pitch_range = (220Hz, 880Hz),
  quantize_to_scale = [0,2,3,5,7,8,10]  // Natural minor
)
```

**4. Reaction-Diffusion → Spectral Filtering**
```morphogen
rd = reaction_diffusion.solve(...)
spectral_env = field.to_spectrogram(rd.activator_field)
filtered = spectral.filter(drone, spectral_env)
```

**5. CA → Multi-Voice Composition**
```morphogen
life = ca.life(size=32, seed=100)
notes = ca.sequencer(life, mapping=(x,y) => {
  pitch = 220Hz * (1 + (x+y)/64.0)
  note(pitch, vel=0.7, dur=0.2s)
})
```

**Status:** 🔲 Not started

---

## Operator Quick Reference

### Spectral Domain (15 operators)

| Operator | Category | Description |
|----------|----------|-------------|
| `spectral.blur` | Manipulation | Gaussian blur in frequency domain |
| `spectral.morph` | Manipulation | Crossfade between spectrograms |
| `spectral.freeze` | Manipulation | Freeze spectral content |
| `harmonic.nebula` | Harmonics | Distributed harmonic cloud |
| `harmonic.drift` | Harmonics | Slow pitch drift |
| `harmonic.spread` | Harmonics | Harmonic spacing control |
| `vocode` | Filtering | Classic vocoder effect |
| `spectral.filter` | Filtering | Arbitrary spectral envelope |
| `spectral.crossfade` | Manipulation | Spectral crossfading |
| `additive.resynth` | Synthesis | Additive resynthesis |
| `spectral.shift` | Manipulation | Frequency shifting |
| `spectral.compress` | Dynamics | Spectral compression |
| `spectral.gate` | Dynamics | Spectral gating |
| `spectral.enhance` | Enhancement | Harmonic enhancement |
| `spectral.smear` | Manipulation | Time-domain smearing |

---

### Ambience Domain (25 operators)

| Operator | Category | Description |
|----------|----------|-------------|
| `drone.harmonic` | Drones | Harmonic drone generator |
| `drone.subharmonic` | Drones | Subharmonic bass drone |
| `drone.pad` | Drones | Lush ambient pad |
| `drone.shimmer` | Drones | Shimmer/reverb drone |
| `granular.cloud` | Granular | Dense granular cloud |
| `granular.freeze` | Granular | Freeze playback position |
| `granular.reverse_cloud` | Granular | Reverse granular synthesis |
| `granular.stretch` | Granular | Time-stretch granular |
| `texture.evolving` | Textures | Slowly evolving texture |
| `texture.shimmer` | Textures | Shimmer texture |
| `texture.noise_field` | Textures | Noise-based texture |
| `drift.noise` | Modulators | Ultra-slow drift (minutes) |
| `orbit.lfo` | Modulators | Orbital LFO (hours) |
| `slow.random_walk` | Modulators | Brownian parameter walk |
| `drift.curve` | Modulators | Custom drift curve |
| `evolve.spectral` | Evolvers | Spectral evolution |
| `evolve.harmonic` | Evolvers | Harmonic evolution |
| `evolve.density` | Evolvers | Density evolution |

---

### Synthesis Domain (30 operators)

| Operator | Category | Description |
|----------|----------|-------------|
| `vco` | Oscillators | Voltage-controlled oscillator |
| `wavetable` | Oscillators | Wavetable synthesis |
| `fm` | Oscillators | FM synthesis |
| `additive` | Oscillators | Additive synthesis |
| `phase_mod` | Oscillators | Phase modulation |
| `multimode` | Filters | Multimode filter (LP/HP/BP) |
| `formant` | Filters | Formant filter (vowels) |
| `comb` | Filters | Comb filter (metallic) |
| `phaser_filter` | Filters | Phaser effect |
| `ladder` | Filters | Moog ladder filter |
| `lfo` | Modulation | Low-frequency oscillator |
| `envelope.follower` | Modulation | Amplitude tracking |
| `sample_and_hold` | Modulation | S&H modulation |
| `slew` | Modulation | Slew limiter |
| `quantizer` | Modulation | Pitch quantization |
| `mix` | Routing | Signal mixer |
| `crossfade` | Routing | Crossfader |
| `vca` | Routing | Voltage-controlled amp |
| `pan` | Routing | Stereo panning |
| `matrix_mixer` | Routing | Matrix routing |

---

### Composition Domain (20 operators)

| Operator | Category | Description |
|----------|----------|-------------|
| `markov.sequence` | Markov | Markov chain sequencer |
| `markov.parameter_walk` | Markov | Parameter evolution |
| `markov.interpolate` | Markov | State interpolation |
| `ca.sequencer` | CA | CA-driven sequencing |
| `life.triggers` | CA | Life birth/death triggers |
| `lenia.modulation` | CA | Lenia field modulation |
| `rule30.noise` | CA | Rule30 as noise source |
| `poisson.trigger` | Stochastic | Poisson process triggers |
| `brownian.melody` | Stochastic | Brownian melodic walk |
| `drunk.walk` | Stochastic | Drunk walk modulation |
| `euclidean.rhythm` | Patterns | Euclidean rhythms |
| `fractal.timing` | Patterns | Fractal timing patterns |
| `fibonacci.sequence` | Patterns | Fibonacci sequences |
| `golden.ratio` | Patterns | Golden ratio timing |
| `swarm.pitch_map` | Swarms | Swarm → pitch mapping |
| `swarm.density_modulation` | Swarms | Swarm density → params |
| `swarm.spatial_pan` | Swarms | Swarm → spatial audio |

---

## Implementation Notes

### GPU Acceleration Strategy

**High-priority GPU operators:**

1. **Granular synthesis** — 1000s of parallel grains
   - Expected speedup: 5-10x vs CPU
   - MLIR lowering: `linalg.generic` with parallel iterator

2. **Spectral convolution** — FFT-based filtering
   - Expected speedup: 10-20x vs CPU
   - MLIR lowering: cuFFT/rocFFT integration

3. **Additive synthesis** — 100+ parallel sinusoids
   - Expected speedup: 5-8x vs CPU
   - MLIR lowering: vectorized oscillator bank

4. **Spectral blur** — 2D Gaussian convolution
   - Expected speedup: 8-15x vs CPU
   - MLIR lowering: separable convolution kernels

### Determinism Guarantees

All operators support three determinism tiers:

**Strict (bit-exact):**
- Markov sequencers
- CA-driven composition
- Additive synthesis
- Euclidean rhythms

**Repro (deterministic within FP precision):**
- Spectral blur
- Granular synthesis (grain windowing)
- Convolution reverb
- Filter processing

**Live (latency-optimized, replayable):**
- Real-time CA evolution
- Adaptive modulation
- Low-latency granular

### Memory Efficiency

**Sparse grain buffers:**
- Only active grains in memory
- Grain pool recycling
- Target: 1000+ grains with <100MB RAM

**Spectral streaming:**
- Partitioned STFT (overlapping windows)
- Incremental FFT computation
- Target: Real-time spectral processing at 96kHz

**Long-form state:**
- Efficient ultra-slow LFO storage
- Minimal state for hour-long periods
- Target: Multi-hour compositions in <1GB RAM

---

## Testing Strategy

### Unit Tests (Per Operator)

**Test categories:**
1. **Determinism:** Same seed → same output
2. **Parameter ranges:** No NaN/Inf for valid inputs
3. **Type safety:** Compile-time errors for invalid types
4. **Golden artifacts:** Reference WAV files

**Example:**
```python
def test_granular_cloud_deterministic():
    grains_1 = granular.cloud(source, density=50, seed=42)
    grains_2 = granular.cloud(source, density=50, seed=42)
    assert arrays_identical(grains_1, grains_2)
```

### Integration Tests (Pipelines)

**Test multi-operator compositions:**
```python
def test_spectral_morph_pipeline():
    spec_a = stft(noise(seed=1))
    spec_b = stft(sine(440))
    morphed = spectral.morph(spec_a, spec_b, mix=0.5)
    audio = istft(morphed)
    assert matches_golden("spectral_morph.wav", audio)
```

### Cross-Domain Tests

**Test physics → audio integration:**
```python
def test_ca_driven_granular():
    life = ca.life(size=32, seed=100)
    density = field.mean(life) * 100
    grains = granular.cloud(source, density=density)
    assert density_varies_over_time(grains)
```

### Performance Benchmarks

**GPU vs CPU speedup:**
```python
def benchmark_granular_cloud():
    t_cpu = time_execution(granular.cloud(..., backend="cpu"))
    t_gpu = time_execution(granular.cloud(..., backend="gpu"))
    speedup = t_cpu / t_gpu
    assert speedup > 5.0  # Target >5x on GPU
```

---

## Contributing

To contribute to the Ambient Music domain:

1. **Implement operators** in respective domain files:
   - `morphogen/stdlib/spectral.py`
   - `morphogen/stdlib/ambience.py`
   - `morphogen/stdlib/synthesis.py`
   - `morphogen/stdlib/composition.py`

2. **Add tests** in `tests/test_ambient_*.py`

3. **Update documentation:**
   - Operator reference in this file
   - Examples in `/docs/examples/`
   - ADR updates if architecture changes

4. **Create examples:**
   - Simple examples in `/examples/ambient/`
   - Cross-domain examples in `/docs/examples/`

See [docs/guides/domain-implementation.md](../guides/domain-implementation.md) for detailed guidelines.

---

## References

### Morphogen Documentation
- [ADR-009](../adr/009-ambient-music-generative-domains.md) — Architecture decision
- [docs/specifications/ambient-music.md](../specifications/ambient-music.md) — Complete specification
- [AUDIO_SPECIFICATION.md](../../AUDIO_SPECIFICATION.md) — Base audio domain
- [docs/architecture/domain-architecture.md](../architecture/domain-architecture.md) — Multi-domain architecture
- [docs/reference/mathematical-music-frameworks.md](../reference/mathematical-music-frameworks.md) — Mathematical frameworks for music representation

### Academic Research
- Curtis Roads, *"Microsound"* (2001) — Granular synthesis
- Brian Eno, *"Generative Music"* (1996) — Generative philosophy
- Julius O. Smith III, *"Spectral Audio Signal Processing"* — Spectral techniques

### Commercial Systems
- Mutable Instruments Clouds — Granular synthesis reference
- VCV Rack — Modular synthesis concepts
- SuperCollider — Generative music programming
- Ableton Live + Max4Live — Modular ambient patching

---

**Status Summary:**

✅ Phase 0 Complete (Design & Specification)
🔲 Phase 1 Pending (Spectral Domain)
🔲 Phase 2 Pending (Ambience Domain)
🔲 Phase 3 Pending (Synthesis Domain)
🔲 Phase 4 Pending (Composition Domain)
🔲 Phase 5 Pending (Cross-Domain Integration)

**Total Operators:** 90 across 4 domains
**Estimated Timeline:** 16 months (Phases 1-5)
**Current Status:** Awaiting Phase 1 implementation approval
