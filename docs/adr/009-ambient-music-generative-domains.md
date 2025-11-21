# ADR-009: Ambient Music & Generative Audio Domains

**Status:** Proposed
**Date:** 2025-11-16
**Decision Makers:** Scott Sen
**Related:** [AUDIO_SPECIFICATION.md](../../AUDIO_SPECIFICATION.md), [specifications/procedural-generation.md](../specifications/procedural-generation.md), [specifications/emergence.md](../specifications/emergence.md), [ADR-002](002-cross-domain-architectural-patterns.md)

---

## Context

Ambient music represents a perfect convergence of Morphogen's core capabilities:
- **Procedural generation** (slow-evolving textures, parameter drift)
- **Cellular automata & emergence** (generative sequencing, modulation)
- **DSP & spectral processing** (granular synthesis, spectral blur)
- **Cross-domain composition** (physics simulations → audio modulation)
- **GPU acceleration** (massive parallel synthesis, convolution)

Ambient music is fundamentally:
- **Generative** — Evolves over long timeframes (hours) via simple rules
- **Procedural** — Parameter-driven transformation, not performance
- **Compositional** — Layered modular synthesis (like Morphogen operators)
- **Simulation-friendly** — Time-evolving fields, agent-like soundscapes
- **GPU-accelerable** — Parallel grain synthesis, convolution, spectral ops

**No existing tool unifies these capabilities:**

| Tool | Strengths | Morphogen Advantage |
|------|-----------|-----------------|
| **Max/MSP, Pure Data** | Visual patching | No GPU, no CA, no physics coupling |
| **SuperCollider** | Procedural synthesis | No GPU, complex syntax, no cross-domain |
| **VCV Rack** | Modular synthesis | CPU-only, no composition, no ML |
| **DAWs (Ableton, FL Studio)** | Production workflows | Linear timelines, not declarative |
| **Eno-style tools (Bloom, Reflection)** | Generative UX | Closed systems, limited extensibility |

**Morphogen can become the first unified platform for ambient/generative audio** that bridges DSP, cellular automata, spectral geometry, physics, and GPU compute.

---

## Decision

We will expand Morphogen's audio capabilities with **four new specialized domains** for ambient and generative music:

### 1. **Morphogen.Spectral** (Frequency-Domain Audio)
High-level operators for spectral manipulation beyond basic FFT/STFT:
- Spectral blurring, morphing, crossfading
- Harmonic nebulae (distributed harmonics)
- Vocoding and spectral filtering
- Additive resynthesis with time-varying parameters
- Pitch-shifting drones
- Spectral freeze effects

### 2. **Morphogen.Ambience** (High-Level Generative Music)
Domain-specific primitives for ambient composition:
- Drone generators (harmonic pads, sub-bass, shimmer)
- Granular clouds (dense, frozen, reverse)
- Markov sequence generators
- CA-driven sequencers
- Stochastic triggers (Poisson processes, Brownian walks)
- Long-form evolution engines (drift, orbit LFOs)

### 3. **Morphogen.Synthesis** (Modular Patching)
Modular-style DSP with declarative signal flow:
- Oscillators, filters, envelopes (enhanced from Morphogen.Audio)
- CV/routing via compute graph (no wire spaghetti)
- Sample-accurate block execution
- Automatic operator fusion and GPU scheduling
- Hybrid of Pure Data + Max/MSP + SuperCollider philosophy

### 4. **Morphogen.Composition** (Generative Structures)
High-level compositional patterns:
- Markov chains (note sequences, parameter evolution)
- CA-based sequencers (Game of Life → trigger events)
- Swarm sequencers (agent positions → pitch/timing)
- Pattern generators (Euclidean rhythms, fractal timing)
- Multi-hour time evolution curves

**These domains will NOT:**
- Duplicate existing Morphogen.Audio operators (extends, doesn't replace)
- Require new MLIR dialects (composes existing Transform, Audio, Agent dialects)
- Be monolithic black boxes (exposes composable low-level primitives)

---

## Architectural Design

### Domain Layer Structure

```
Layer 9: Composition (NEW) — High-level generative patterns
├── Uses Layer 8: Ambience (drones, clouds, evolvers)
├── Uses Layer 7: Spectral (frequency-domain textures)
├── Uses Layer 6: Synthesis (modular DSP routing)
├── Uses Layer 5: Audio (oscillators, filters, effects)
├── Uses Layer 4: Emergence (CA, swarms, reaction-diffusion)
├── Uses Layer 3: Stochastic (random walks, Markov chains)
├── Uses Layer 2: Transform (FFT, STFT, wavelet, convolution)
└── Uses Layer 1: Time/Space/Event (kernel domains)
```

### New Operator Registries

#### 1. `morphogen.audio.spectral.*` (Spectral Domain)

**Core Types:**
```morphogen
type SpectralField = Field2D<complex, (freq, time)>
type Spectrogram = Field2D<f32, (freq, time)>
type HarmonicSeries = Field1D<f32, freq>
```

**Operators (15 new):**
```morphogen
// Spectral manipulation
spectral.blur(spectrogram, bandwidth=50Hz) -> Spectrogram
spectral.morph(spec_a, spec_b, mix=0.5) -> Spectrogram
spectral.freeze(spectrogram, freeze_freq_range=(200Hz, 2kHz)) -> Spectrogram

// Harmonic processing
harmonic.nebula(fundamental, spread=0.2, density=32) -> HarmonicSeries
harmonic.drift(series, rate=0.01Hz, depth=10cents) -> HarmonicSeries

// Vocoding
vocode(carrier, modulator, bands=16) -> Sig
spectral.filter(sig, envelope: Spectrogram) -> Sig

// Resynthesis
additive.resynth(harmonics: Field2D, phases: Field2D) -> Sig
spectral.crossfade(spec_a, spec_b, curve: Env) -> Spectrogram
```

#### 2. `morphogen.ambient.*` (Ambience Domain)

**Core Types:**
```morphogen
type Drone = {
  fundamental: Ctl[Hz],
  texture: Enum["smooth", "shimmer", "rough"],
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

**Operators (25 new):**
```morphogen
// Drone generators
drone.harmonic(fundamental, spread, shimmer=0.2) -> Stereo
drone.subharmonic(root, divisions=[2,3,4,5]) -> Sig
drone.pad(harmonics, bandwidth, modulation) -> Stereo

// Granular synthesis
granular.cloud(source, density, grain_size, pitch_shift=0) -> Sig
granular.freeze(source, freeze_position=0.5) -> Sig
granular.reverse_cloud(source, density, grain_size) -> Sig

// Texture generators
texture.evolving(seed, evolution_rate, spectral_range) -> Sig
texture.shimmer(fundamental, density, brightness) -> Stereo

// Long-form modulators
drift.noise(period_minutes=10, depth=1.0) -> Ctl
orbit.lfo(period_hours=1, orbit_shape="ellipse") -> Ctl
slow.random_walk(range=(-1,1), smoothing=0.8) -> Ctl
```

#### 3. `morphogen.synthesis.*` (Modular Synthesis Domain)

**Core Types:**
```morphogen
type Patch = Graph<SynthOp>  // Directed acyclic synthesis graph
type CV = Ctl  // Control voltage (modulation signal)
type Gate = Evt<void>  // Trigger/gate events
```

**Operators (30 new, many enhanced from Morphogen.Audio):**
```morphogen
// Enhanced oscillators
vco(freq, waveform="saw", sync: Gate) -> Sig
wavetable(table: Field1D, freq, morph: Ctl) -> Sig
fm(carrier_freq, mod_freq, mod_index) -> Sig

// Advanced filters
multimode(sig, mode="lp", cutoff, resonance, drive=0) -> Sig
formant(sig, vowel="a", morph: Ctl) -> Sig
comb(sig, delay_time, feedback, damping) -> Sig

// Modulation sources
lfo(rate, waveform, phase_offset=0) -> CV
envelope.follower(sig, attack, release) -> CV
sample_and_hold(sig, trigger: Gate) -> CV

// Routing
mix(inputs: List[Sig], gains: List[f32]) -> Sig
crossfade(sig_a, sig_b, position: CV) -> Sig
vca(sig, cv: Ctl) -> Sig  // Voltage-controlled amplifier
```

#### 4. `morphogen.composition.*` (Generative Composition Domain)

**Core Types:**
```morphogen
type MarkovMatrix = Field2D<f32, (state_from, state_to)>
type CARule = {
  rule: Enum["life", "rule30", "lenia", ...],
  params: Map<String, f32>
}
type Pattern = Evt<Note>  // Musical pattern (timed notes)
```

**Operators (20 new):**
```morphogen
// Markov sequencing
markov.sequence(matrix, initial_state, tempo) -> Evt<Note>
markov.parameter_walk(matrix, value_mapping) -> Ctl

// CA-based sequencing
ca.sequencer(ca_rule, grid_size, mapping: (x,y) -> Note) -> Evt<Note>
life.triggers(grid: Field2D<bool>, birth_note, death_note) -> Evt<Note>
lenia.modulation(field: Field2D<f32>, param_name) -> Ctl

// Stochastic generators
poisson.trigger(rate: Ctl[Hz], note_pool) -> Evt<Note>
brownian.melody(start_note, step_size, bounds) -> Evt<Note>

// Pattern generators
euclidean.rhythm(steps, pulses, rotation=0) -> Evt<Gate>
fractal.timing(iterations, ratio, base_duration) -> Evt<Gate>

// Swarm-based composition
swarm.pitch_map(agents: Agents, pitch_range, quantize_to_scale) -> Evt<Note>
swarm.density_modulation(agents: Agents, param_name) -> Ctl
```

---

## Ambient Pipeline Architecture

### Typical Ambient Composition Flow

```morphogen
// Stage 1: Sound source generation
source = noise(seed=42, type="pink") |> lpf(2kHz)

// Stage 2: Texture shaping (spectral domain)
spectrogram = stft(source)
blurred = spectral.blur(spectrogram, bandwidth=100Hz)
textured = spectral.morph(blurred, harmonic.nebula(220Hz, spread=0.3))

// Stage 3: Temporal evolution (CA modulation)
life_grid = ca.life(initial_state=random_grid(64,64))
drift_amount = ca.density_map(life_grid) * 20cents  // CA → pitch drift

// Stage 4: Granular processing
grains = granular.cloud(
  source = istft(textured),
  density = 50,
  grain_size = 100ms,
  pitch_shift = drift_amount
)

// Stage 5: Spatialization
wide = stereo.width(grains, width=1.5)
spatial = orbit.pan(wide, period=30s, orbit_radius=0.8)

// Stage 6: Long-form evolution
reverb_decay = drift.noise(period_minutes=15, depth=0.5) + 2.0
output = spatial |> reverb(mix=0.3, decay=reverb_decay)
```

### Cross-Domain Integration Examples

**Example 1: Fluid Simulation → Granular Modulation**
```morphogen
// Fluid vorticity drives grain density
fluid = navier_stokes.solve(...)
vorticity = field.curl(fluid.velocity)
grain_density = vorticity.max() * 100  // High vorticity = dense grains

grains = granular.cloud(
  source = drone.harmonic(110Hz, spread=0.2),
  density = grain_density,
  grain_size = 50ms
)
```

**Example 2: Mandelbrot Zoom → Spectral Evolution**
```morphogen
// Fractal depth drives harmonic spread
mandelbrot = fractal.iterate(zoom_center, zoom_rate)
depth = mandelbrot.escape_time.mean()
harmonic_spread = depth / 100.0  // Deeper = wider harmonics

pad = drone.harmonic(
  fundamental = 55Hz,
  spread = harmonic_spread,
  shimmer = 0.3
)
```

**Example 3: Swarm Agents → Melody Generation**
```morphogen
// Boid positions generate melodic material
boids = agents.boids(num=50, bounds=100, cohesion=0.5)
notes = swarm.pitch_map(
  boids,
  pitch_range = (55Hz, 880Hz),
  quantize_to_scale = [0,2,3,5,7,9,10]  // Natural minor
)

voice = (note: Note) => {
  sine(note.pitch)
  |> adsr(5ms, 200ms, 0.4, 500ms)
  |> reverb(0.2)
}

output = spawn(notes, voice, max_voices=8)
```

---

## Rationale

### Why This is a Perfect Fit for Morphogen

**1. Natural Operator Composition**

Ambient music is built from **layers of simple transformations** — exactly Morphogen's model:
```
Noise → Filter → Granular → Spectral Blur → Reverb → Spatial
```

Each stage = operator. GPU-accelerated. Deterministic. Composable.

**2. Cross-Domain Synergy**

Morphogen's unique strength: **physics/CA/fractals can drive audio parameters**
- Reaction-diffusion → filter cutoff modulation
- Cellular automata → triggering events
- Lenia creatures → pitch evolution
- Turbulence field → grain density

No other audio tool can do this.

**3. GPU Acceleration Opportunities**

Massive parallelism in ambient techniques:
- **Granular synthesis:** 1000s of independent grains → parallel processing
- **Spectral blur:** Convolution in frequency domain → GPU FFT
- **Additive synthesis:** 100+ sinusoids → parallel oscillator bank
- **Convolution reverb:** Partitioned FFT → GPU acceleration

MLIR enables automatic fusion:
```mlir
// Granular synthesis lowered to GPU
%grains = linalg.generic {
  iterator_types = ["parallel"]
} ins(%source, %density, %grain_size) outs(%output) {
  // Each grain computed independently on GPU
  %windowed = arith.mulf %source_sample, %window : f32
  linalg.yield %windowed : f32
}
```

**4. Deterministic Generative Music**

Ambient music needs **reproducible randomness** (same seed = same evolution):
- Markov chains → seeded RNG
- CA sequencers → deterministic rules
- Granular noise → Philox RNG (cross-platform bit-exact)

Morphogen's determinism tiers map perfectly:
- **Strict:** Additive synthesis, Markov sequences
- **Repro:** Spectral blur, convolution reverb
- **Live:** Real-time CA evolution, adaptive modulation

**5. Multi-Hour Evolution**

Ambient tracks evolve over hours. Morphogen's scheduler handles:
- Long-period modulators (orbit LFOs with hour-long periods)
- Slow parameter drift (0.001 Hz modulation)
- Deterministic long-form sequences
- Memory-efficient state (sparse grain buffers)

**6. Modular Philosophy Without Wire Spaghetti**

Traditional modular synths: visual mess of patch cables
Morphogen.Synthesis: **compute graph IS the signal flow**
- Automatic dependency analysis
- Operator fusion for efficiency
- Type-safe connections (no CV → audio accidents)
- Declarative (describe WHAT, not HOW)

---

## Consequences

### Positive

**1. Morphogen becomes the premier ambient music platform**
- Unified DSP, CA, spectral, physics, GPU
- No competing tool has this integration

**2. Showcase for cross-domain composition**
- Fluid sim → audio modulation
- Fractals → melodic generation
- CA → sequencing
- Physics → timbre evolution

**3. GPU acceleration differentiator**
- Real-time dense granular synthesis
- Massive additive synthesis (1000+ partials)
- Low-latency convolution reverb
- Spectral processing in real-time

**4. Generative music research platform**
- Reproducible experiments (deterministic)
- Extensible operator library
- Fast iteration (MLIR JIT compilation)
- Cross-domain experimentation

**5. Educational value**
- Learn DSP through composition
- Understand CA through sound
- Explore spectral processing visually
- Bridge music and mathematics

### Negative / Challenges

**1. Scope expansion risk**
- Adds 90+ new operators across 4 domains
- Requires extensive testing and documentation
- Potential maintenance burden

**Mitigation:**
- Phase implementation (start with Spectral, then Ambience, etc.)
- Reuse existing primitives where possible
- Focus on composability over monolithic features

**2. MLIR lowering complexity**
- Granular synthesis requires sophisticated scheduling
- Spectral ops need efficient GPU FFT
- Long-period modulators challenge scheduler

**Mitigation:**
- Start with CPU implementations
- Add GPU lowering incrementally
- Leverage existing Transform dialect (FFT already planned)

**3. Documentation and examples needed**
- Ambient music techniques unfamiliar to some users
- Cross-domain patterns need clear tutorials
- Large operator surface area

**Mitigation:**
- Comprehensive examples (Eno-style generative pieces)
- Tutorial series (beginner → advanced)
- Reference implementations (classic ambient patches)

**4. Performance optimization required**
- Dense granular synthesis is CPU-intensive
- Spectral processing can be memory-hungry
- Long-running compositions need efficient state

**Mitigation:**
- Profile-based optimization (strict=expensive, live=efficient)
- Sparse grain buffers (only active grains in memory)
- MLIR fusion eliminates intermediate allocations

### Neutral / Future Considerations

**1. Audio engine integration**
- Requires real-time audio backend (JACK, PortAudio, WebAudio)
- Hot reload during live performance
- Latency-adaptive scheduling

**Action:** Design snapshot ABI to support live reload (already planned)

**2. Control surface integration**
- MIDI controllers for live parameter tweaks
- OSC integration for TouchOSC, Max4Live
- CV/Gate hardware (Expert Sleepers, Motu)

**Action:** Event stream abstraction supports MIDI/OSC (future Layer 10 domain)

**3. Commercial ambient tools integration**
- Export to Ableton Live (via Max4Live device)
- Integration with Eurorack hardware (via CV outputs)
- Preset sharing ecosystem (community library)

**Action:** YAML import/export, preset versioning (future work)

---

## Implementation Phases

### Phase 1: Spectral Domain (v0.6)
- Core spectral operators (blur, morph, freeze)
- Harmonic nebula generators
- Vocoding primitives
- GPU-accelerated FFT/STFT (MLIR lowering)

**Deliverables:**
- 15 spectral operators
- Spectral field type
- Example: Spectral morphing pad
- Tests: Golden spectrograms

### Phase 2: Ambience Domain (v0.7)
- Drone generators (harmonic, subharmonic, pad)
- Granular synthesis operators
- Long-form modulators (drift, orbit LFO)
- Texture generators

**Deliverables:**
- 25 ambience operators
- Granular cloud type
- Example: 20-minute evolving drone
- Tests: Deterministic grain playback

### Phase 3: Synthesis Domain (v0.8)
- Enhanced oscillators (VCO, wavetable, FM)
- Advanced filters (multimode, formant, comb)
- Modulation sources (LFO, envelope follower, S&H)
- Declarative patch routing

**Deliverables:**
- 30 synthesis operators
- Patch graph type
- Example: Generative modular patch
- Tests: Signal flow validation

### Phase 4: Composition Domain (v0.9)
- Markov sequencers
- CA-driven composition
- Stochastic generators (Poisson, Brownian)
- Pattern generators (Euclidean, fractal)

**Deliverables:**
- 20 composition operators
- Markov matrix type
- Example: CA-sequenced melody
- Tests: Deterministic sequence generation

### Phase 5: Cross-Domain Integration (v1.0)
- Physics → audio modulation examples
- Fractal → melodic generation
- Swarm → spatial audio
- Complete ambient showcase (multi-domain composition)

**Deliverables:**
- 10 cross-domain examples
- Performance benchmarks (CPU vs GPU)
- Tutorial documentation
- Golden artifacts (WAV, PNG, videos)

---

## Alternatives Considered

### Alternative 1: Extend Morphogen.Audio Only (No New Domains)

**Pros:**
- Simpler architecture (fewer domains)
- Less documentation needed
- Easier to maintain

**Cons:**
- Loses semantic clarity (spectral ops mixed with oscillators)
- Harder to discover operators (100+ in one namespace)
- Misses opportunity for domain-specific types (GranularCloud, Drone)
- No clear separation of concerns

**Decision:** Rejected. Domain separation improves discoverability and type safety.

### Alternative 2: Single "Generative" Domain (Instead of 4)

**Pros:**
- Simpler mental model
- Single namespace

**Cons:**
- Mixes concerns (spectral ≠ granular ≠ composition)
- Operator explosion (90+ in one domain)
- Harder to optimize (mixed MLIR lowering strategies)

**Decision:** Rejected. Four focused domains better than one mega-domain.

### Alternative 3: Pure Library (No Morphogen Integration)

Build standalone ambient music library, don't integrate with Morphogen.

**Pros:**
- Faster initial development
- No kernel dependencies

**Cons:**
- Loses cross-domain composition (biggest differentiator)
- No GPU acceleration (MLIR required)
- No determinism guarantees
- Duplicates existing tools (SuperCollider, Max/MSP)

**Decision:** Rejected. Morphogen integration is the entire value proposition.

### Alternative 4: Focus on Preset Systems (Not Operators)

Provide high-level presets ("Eno Pad", "Shimmer Drone") instead of operators.

**Pros:**
- Easier for beginners
- Immediate results

**Cons:**
- Not extensible (closed presets)
- Hides composability
- Becomes "yet another preset library"
- Doesn't showcase Morphogen's architecture

**Decision:** Rejected. Operators + presets (both), operators first.

---

## Success Criteria

This decision will be considered successful if:

1. **Technical:**
   - 90+ ambient operators implemented and tested
   - GPU acceleration for granular + spectral ops (5x speedup vs CPU)
   - Deterministic reproduction across platforms (bit-exact in strict mode)
   - Multi-hour compositions run stably (no memory leaks)

2. **Usability:**
   - 10+ complete ambient examples (beginner → advanced)
   - Cross-domain integration examples (physics → audio, CA → sequencing)
   - Documentation coverage (every operator has examples)
   - Tutorial series completed (ambient music fundamentals)

3. **Community:**
   - 5+ community-contributed ambient patches
   - Positive feedback from ambient music community (Lines forum, r/ambient)
   - Integration requests (Eurorack, Ableton, hardware)

4. **Impact:**
   - Cited in generative music research papers
   - Used in live performances (concerts, installations)
   - Referenced as ambient music platform (vs SuperCollider, Max/MSP)

---

## Related Work

### Academic Research
- **Curtis Roads** — "Microsound" (2001) - Granular synthesis theory
- **Brian Eno** — "Generative Music" (1996) - Compositional philosophy
- **Julius O. Smith III** — "Physical Audio Signal Processing" - Waveguide synthesis
- **Lenia** (Bert Chan, 2019) - Continuous cellular automata for generative art

### Commercial Systems
- **Yamaha VL1** (1994) — Physical modeling synthesis
- **Ableton Live + Max4Live** — Modular ambient patching
- **Make Noise Morphagene** — Eurorack granular module
- **Mutable Instruments Clouds** — Eurorack granular texture module
- **Native Instruments Reaktor** — Modular synthesis environment

### Generative Tools
- **Brian Eno's "Bloom"** — iOS generative music app
- **Orca** — Esoteric programming language for sequencing
- **TidalCycles** — Live-coded ambient patterns
- **Sonic Pi** — Educational music programming

**Morphogen's differentiator:** None of these integrate DSP + CA + physics + GPU + determinism.

---

## References

### Specifications
- [AUDIO_SPECIFICATION.md](../../AUDIO_SPECIFICATION.md) — Morphogen.Audio domain
- [docs/specifications/emergence.md](../specifications/emergence.md) — CA, swarms, L-systems
- [docs/specifications/procedural-generation.md](../specifications/procedural-generation.md) — Noise, fractals, terrain
- [docs/specifications/transform.md](../specifications/transform.md) — FFT, STFT, wavelets

### Architecture
- [ADR-002](002-cross-domain-architectural-patterns.md) — Cross-domain integration patterns
- [docs/architecture/domain-architecture.md](../architecture/domain-architecture.md) — Multi-domain vision
- [docs/architecture/gpu-mlir-principles.md](../architecture/gpu-mlir-principles.md) — GPU lowering

### Examples
- [docs/examples/emergence-cross-domain.md](../examples/emergence-cross-domain.md) — CA → geometry → physics
- [examples/audio/](../../examples/audio/) — Audio synthesis examples

---

## Appendix: Operator Inventory

### Spectral Domain (15 operators)
- `spectral.blur`, `spectral.morph`, `spectral.freeze`
- `harmonic.nebula`, `harmonic.drift`, `harmonic.spread`
- `vocode`, `spectral.filter`, `spectral.crossfade`
- `additive.resynth`, `spectral.shift`, `spectral.compress`
- `spectral.gate`, `spectral.enhance`, `spectral.smear`

### Ambience Domain (25 operators)
- **Drones:** `drone.harmonic`, `drone.subharmonic`, `drone.pad`, `drone.shimmer`
- **Granular:** `granular.cloud`, `granular.freeze`, `granular.reverse_cloud`, `granular.stretch`
- **Textures:** `texture.evolving`, `texture.shimmer`, `texture.noise_field`
- **Modulators:** `drift.noise`, `orbit.lfo`, `slow.random_walk`, `drift.curve`
- **Evolvers:** `evolve.spectral`, `evolve.harmonic`, `evolve.density`

### Synthesis Domain (30 operators)
- **Oscillators:** `vco`, `wavetable`, `fm`, `additive`, `phase_mod`
- **Filters:** `multimode`, `formant`, `comb`, `phaser_filter`, `ladder`
- **Modulation:** `lfo`, `envelope.follower`, `sample_and_hold`, `slew`, `quantizer`
- **Routing:** `mix`, `crossfade`, `vca`, `pan`, `matrix_mixer`

### Composition Domain (20 operators)
- **Markov:** `markov.sequence`, `markov.parameter_walk`, `markov.interpolate`
- **CA:** `ca.sequencer`, `life.triggers`, `lenia.modulation`, `rule30.noise`
- **Stochastic:** `poisson.trigger`, `brownian.melody`, `drunk.walk`
- **Patterns:** `euclidean.rhythm`, `fractal.timing`, `fibonacci.sequence`
- **Swarms:** `swarm.pitch_map`, `swarm.density_modulation`, `swarm.spatial_pan`

**Total:** 90 new operators across 4 domains

---

**Status:** Awaiting approval
**Next Steps:**
1. Approve ADR
2. Create specification document (`docs/specifications/ambient-music.md`)
3. Implement Phase 1 (Spectral Domain)
4. Update `ECOSYSTEM_MAP.md` with new domains
