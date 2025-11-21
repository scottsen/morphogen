# SPEC: Timbre Extraction and Instrument Modeling

**Version:** 1.0 Draft
**Status:** RFC
**Last Updated:** 2025-11-15

---

## Overview

This specification defines Morphogen's **timbre extraction and instrument modeling** capability — a system for analyzing acoustic recordings, extracting timbre characteristics, and synthesizing new notes with the same sonic character.

This is one of the **holy grails of audio DSP**: turning a recording into a reusable synthesis model.

**What it does:**
- Record acoustic guitar → extract timbre → synthesize new notes
- Recreate the same note (test fidelity)
- Generate new notes with the same timbre
- Generate legato, bends, slides
- Convert the guitar into a MIDI instrument
- Morph instruments (guitar → violin, piano, synth hybrid)
- Measure exact timbre evolution (for luthiers and engineers)

**Why Morphogen is ideal:**
- Cross-domain operator architecture (audio + physics + transforms)
- MLIR compilation for GPU acceleration
- Deterministic semantics for reproducible analysis
- Type-safe composition of analysis → model → synthesis pipeline

---

## Real-World Precedents

This capability builds on established techniques from:

| System | Technique | Year |
|--------|-----------|------|
| **Yamaha VL1 / VL70m** | Physical modeling synthesis | 1994 |
| **Karplus–Strong** | Plucked string synthesis | 1983 |
| **NSynth (Google Magenta)** | Neural audio synthesis | 2017 |
| **Additive synthesis** | Harmonic resynthesis | 1950s+ |
| **Modal synthesis** | Resonant mode decomposition | 1990s+ |
| **Convolution-based resonators** | Impulse response modeling | 2000s+ |

Morphogen **unifies all of these** in a way that is:
- ✅ Extensible (operator-driven)
- ✅ Domain-aware (cross-domain composition)
- ✅ Physically rooted (modal analysis, harmonic decomposition)
- ✅ GPU-accelerated (MLIR lowering)

---

## What Needs to Be Extracted from a WAV

From a single recorded note (or a set of them), we mathematically extract:

### 1. Fundamental Frequency Evolution

**What:**
- Onset pitch (attack transient pitch)
- Micro-pitch variation (pitch wobble, drift)
- Vibrato (periodic pitch modulation)
- Attack noise (percussive component)

**Operators:**
- `harmonic.track_fundamental` — Track f0 over time
- `pitch.autocorrelation` — Pitch detection via autocorrelation
- `pitch.yin` — YIN pitch detector (robust)
- `vibrato.extract` — Extract vibrato parameters (rate, depth, phase)

---

### 2. Harmonic Amplitudes Over Time (Spectral Envelope)

**What:**
- Time-varying amplitude of each harmonic (the "timbre")
- Spectral centroid evolution
- Brightness evolution

**Operators:**
- `stft` ✅ (already exists)
- `harmonic.track_partials` — Track amplitudes of each harmonic over time
- `spectral.envelope` — Extract smooth spectral envelope
- `spectral.centroid` — Compute spectral centroid

---

### 3. Modal Resonances of the Guitar Body

**What:**
- Damped sinusoids (resonant peaks)
- Body modes (wood vibration, soundhole resonance)
- Low-frequency modes (Helmholtz resonance)
- Broadband noise (body creaks, fret buzz)

**Operators:**
- `modal.analyze` — Fit damped sinusoid modes: `A e^(-dt) sin(2πft + φ)`
- `modal.extract_modes` — Extract resonant peaks from frequency response
- `resonance.peaks` — Detect resonant peaks in spectrum

---

### 4. Excitation vs Resonance Separation

**What:**
- **Key insight:** `Recorded signal = Excitation ⊗ Resonator Filter`
- Deconvolution gives:
  - **Excitation** — pluck/noise/attack
  - **Resonator** — instrument body filter response

**Operators:**
- `deconvolve` — Separate excitation and resonator
- `excitation.extract` — Extract pluck/attack transient
- `resonator.ir` — Extract impulse response of body

**Technique:** Homomorphic deconvolution (cepstral domain)

---

### 5. Decay Rates

**What:**
- Each overtone decays at its own exponential rate
- Reveals: string material, damping, body coupling, inharmonicity

**Operators:**
- `decay.fit_exponential` — Fit exponential decay envelope per partial
- `decay.t60` — Compute T60 (time to decay 60dB)
- `inharmonicity.measure` — Measure deviation from perfect harmonics (piano strings, guitar)

---

### 6. Nonlinearities

**What:**
- Subtle clipping, saturation
- Fret buzz (noise bursts)
- Microphone distortion

**Operators:**
- `nonlinear.detect_clipping` — Detect hard/soft clipping
- `transient.detect` — Detect percussive transients
- `noise.extract_broadband` — Extract broadband noise layer

---

## Morphogen Domain Model

### AudioAnalysisDomain (NEW)

**Purpose:** Extract timbre features from audio recordings.

**Operators:**

| Operator | Category | Description | Inputs | Outputs |
|----------|----------|-------------|--------|---------|
| `stft` | Transform | Short-time Fourier transform | `Sig` | `Field2D<Complex>` |
| `harmonic.track_fundamental` | Pitch | Track fundamental frequency over time | `Sig` | `Ctl[Hz]` |
| `harmonic.track_partials` | Spectral | Track harmonic amplitudes over time | `Sig`, `f0: Ctl[Hz]` | `Field2D<f32>` (time × partial) |
| `modal.analyze` | Modal | Fit damped sinusoid modes | `Sig` | `ModalModel` |
| `envelope.extract` | Envelope | Extract amplitude/spectral envelopes | `Sig` | `Env` |
| `deconvolve` | Deconvolution | Separate excitation and resonator | `Sig` | `(excitation: Sig, resonator: IR)` |
| `noise.model` | Noise | Capture broadband noise signature | `Sig`, `bands: i32` | `NoiseModel` |
| `decay.fit_exponential` | Decay | Fit exponential decay per partial | `Field2D<f32>` | `Field1D<f32>` (decay rates) |
| `spectral.envelope` | Spectral | Extract smooth spectral envelope | `Field2D<Complex>` | `Field1D<f32>` |
| `inharmonicity.measure` | Spectral | Measure inharmonicity coefficient | `Sig`, `f0: Ctl[Hz]` | `f32` |

---

### SynthesisDomain (EXTENDED)

**Purpose:** Synthesize audio from extracted models.

**Operators:**

| Operator | Category | Description | Inputs | Outputs |
|----------|----------|-------------|--------|---------|
| `additive.synth` | Synthesis | Sum of harmonics with time-varying envelopes | `Field2D<f32>` (amplitudes) | `Sig` |
| `modal.synth` | Synthesis | Damped sinusoid resonator bank | `ModalModel` | `Sig` |
| `convolution.reverb` | Effect | Apply impulse response (body resonance) | `Sig`, `IR` | `Sig` |
| `excitation.pluck` | Synthesis | Pluck/noise generator | `type: Enum`, `params: Map` | `Sig` |
| `spectral.filter` | Filter | Reapply learned timbre shape | `Sig`, `envelope: Field1D<f32>` | `Sig` |
| `granular.resynth` | Synthesis | Granular resynthesis for textures | `Sig`, `grains: GrainParams` | `Sig` |

---

### InstrumentModelDomain (NEW — Layer 7)

**Purpose:** A reusable, parameterized "instrument" that stores extracted features and synthesizes new notes.

**Key Type:**

```morphogen
type InstrumentModel {
  id: String                          // "acoustic_guitar_1"
  type: Enum                          // "modal_string", "modal_membrane", "additive", etc.

  // Analysis results
  fundamental: Ctl[Hz]                // Base pitch of analyzed note
  harmonics: Field2D<f32>             // Harmonic amplitudes (time × partial)
  modes: ModalModel                   // Resonant modes
  body_ir: IR                         // Body impulse response
  noise: NoiseModel                   // Noise layer
  excitation: ExcitationModel         // Pluck/attack model
  decay_rates: Field1D<f32>           // Decay per partial
  inharmonicity: f32                  // Inharmonicity coefficient

  // Synthesis parameters
  synth_params: {
    pluck_position: f32<ratio>        // 0.0 = bridge, 1.0 = neck
    pluck_stiffness: f32              // 0.0 = soft, 1.0 = hard
    body_coupling: f32<ratio>         // How much body resonance to apply
    noise_level: f32<dB>              // Broadband noise mix
  }
}
```

**Operators:**

| Operator | Description | Inputs | Outputs |
|----------|-------------|--------|---------|
| `instrument.analyze` | Full analysis pipeline | `Sig` | `InstrumentModel` |
| `instrument.synthesize` | Generate new note from model | `InstrumentModel`, `pitch: Ctl[Hz]`, `velocity: f32` | `Sig` |
| `instrument.morph` | Morph between two models | `model_a: InstrumentModel`, `model_b: InstrumentModel`, `blend: f32` | `InstrumentModel` |
| `instrument.save` | Serialize model to disk | `InstrumentModel`, `path: String` | `void` |
| `instrument.load` | Load model from disk | `path: String` | `InstrumentModel` |

---

## Example: Full Pipeline (Morphogen Language)

### 1. Analyze Acoustic Guitar Recording

```morphogen
scene AnalyzeGuitar {
  // Load recording
  let recording = io.load("samples/guitar_E3.wav")

  // Extract fundamental frequency
  let f0 = harmonic.track_fundamental(recording)

  // Track harmonic amplitudes over time
  let harmonics = harmonic.track_partials(recording, f0, num_partials=20)

  // Fit resonant modes (body resonances)
  let modes = modal.analyze(recording, num_modes=20)

  // Extract envelopes
  let amp_envelope = envelope.extract(recording, type="amplitude")
  let spectral_envelope = spectral.envelope(stft(recording))

  // Separate excitation and resonator
  let (excitation, body_ir) = deconvolve(recording, f0)

  // Extract noise model
  let noise = noise.model(recording, bands=32)

  // Fit decay rates
  let decay_rates = decay.fit_exponential(harmonics)

  // Measure inharmonicity
  let inharm = inharmonicity.measure(recording, f0)

  // Build instrument model
  let model = InstrumentModel {
    id: "acoustic_guitar_1",
    type: "modal_string",
    fundamental: f0,
    harmonics: harmonics,
    modes: modes,
    body_ir: body_ir,
    noise: noise,
    excitation: excitation,
    decay_rates: decay_rates,
    inharmonicity: inharm,
    synth_params: {
      pluck_position: 0.18,
      pluck_stiffness: 0.97,
      body_coupling: 0.9,
      noise_level: -60dB
    }
  }

  // Save model
  instrument.save(model, "models/guitar_e3.kairo")
}
```

---

### 2. Synthesize New Notes from Model

```morphogen
scene SynthesizeFromModel {
  // Load model
  let guitar = instrument.load("models/guitar_e3.kairo")

  // Generate a new note (A3, same timbre as original E3)
  let note_a3 = instrument.synthesize(
    guitar,
    pitch = note("A3"),  // 220Hz
    velocity = 0.8
  )

  // Generate a melody
  let melody = score [
    at 0s note("A3", 0.8, 0.5s),
    at 0.5s note("C4", 0.7, 0.5s),
    at 1s note("E4", 0.6, 1s)
  ]

  let voice = (n: Note) => instrument.synthesize(
    guitar,
    pitch = n.pitch,
    velocity = n.vel
  )

  out stereo = spawn(melody, voice, max_voices=8) |> reverb(0.1)
}
```

---

### 3. Morph Guitar into Violin

```morphogen
scene MorphInstruments {
  let guitar = instrument.load("models/guitar_e3.kairo")
  let violin = instrument.load("models/violin_a4.kairo")

  // Morph between guitar and violin (50/50 blend)
  let hybrid = instrument.morph(guitar, violin, blend=0.5)

  // Synthesize with hybrid timbre
  let note = instrument.synthesize(
    hybrid,
    pitch = note("D4"),
    velocity = 0.9
  )

  out stereo = note |> reverb(0.15)
}
```

---

## Deconstruction Techniques (DSP Methods)

These are standard in acoustics and DSP; Morphogen just operators-them.

### 1. Harmonic + Noise Model (H+N Model)

**Split into:**
- **H:** Harmonic partials (sinusoidal components)
- **N:** Stochastic noise (broadband residual)

**Used in:** Guitar synthesis, voice synthesis

**Operators:** `harmonic.track_partials`, `noise.extract_broadband`

---

### 2. Modal Decomposition

**Fit damped sinusoids:**
```
A e^(-dt) sin(2πft + φ)
```

**Matches guitar body physics exactly.**

**Operators:** `modal.analyze`, `modal.synth`

**References:** Modal synthesis (Morrison & Adrien, 1993)

---

### 3. LPC / Spectral Envelope Extraction

**Linear Predictive Coding (LPC):**
- Models vocal tract (or instrument body) as all-pole filter
- Used in speech, instrument modeling, tube modeling

**Operators:** `spectral.envelope`, `lpc.analyze`

---

### 4. Deconvolution

**Solve:**
```
output = excitation ⊗ body_IR
→ body_IR = deconvolution(output, excitation)
```

**Homomorphic deconvolution (cepstral domain):**
```
cepstrum(signal) = cepstrum(excitation) + cepstrum(body)
```

**Operators:** `deconvolve`, `cepstral.transform`

---

### 5. Wavelet-Based Transient Separation

**Use wavelets to isolate attack transients:**
- Useful for pluck attack, bow noise, key click

**Operators:** `wavelet.transform`, `transient.detect`

---

## What Morphogen Needs to Add

### New Domains

1. **AudioAnalysisDomain** — Analysis operators for timbre extraction
2. **InstrumentModelDomain** — High-level instrument modeling (Layer 7)

### New Operators (35 new operators)

**AudioAnalysisDomain (13 operators):**
- `harmonic.track_fundamental`
- `harmonic.track_partials`
- `modal.analyze`
- `modal.extract_modes`
- `envelope.extract`
- `deconvolve`
- `excitation.extract`
- `resonator.ir`
- `noise.model`
- `decay.fit_exponential`
- `decay.t60`
- `spectral.envelope`
- `inharmonicity.measure`

**SynthesisDomain Extensions (6 operators):**
- `additive.synth`
- `modal.synth`
- `excitation.pluck`
- `spectral.filter`
- `granular.resynth`
- `convolution.reverb` (extend existing `conv`)

**InstrumentModelDomain (5 operators):**
- `instrument.analyze`
- `instrument.synthesize`
- `instrument.morph`
- `instrument.save`
- `instrument.load`

**Supporting Operators (11 operators):**
- `pitch.autocorrelation`
- `pitch.yin`
- `vibrato.extract`
- `spectral.centroid`
- `resonance.peaks`
- `nonlinear.detect_clipping`
- `transient.detect`
- `noise.extract_broadband`
- `lpc.analyze`
- `cepstral.transform`
- `wavelet.transform` (if not in Transform domain)

---

## GPU Acceleration (MLIR Lowering)

These operators are **ideal for GPU acceleration:**

| Operator | MLIR Strategy | Notes |
|----------|---------------|-------|
| `stft` | `linalg.matmul` + `fft.fft` | Batched FFT on GPU |
| `harmonic.track_partials` | `linalg.generic` | Parallel peak tracking |
| `modal.analyze` | `linalg.matmul` + iterative solver | Least-squares fitting on GPU |
| `additive.synth` | `linalg.generic` + vectorization | Parallel sinusoid summation |
| `modal.synth` | `linalg.generic` + vectorization | Parallel damped sinusoid bank |
| `deconvolve` | `fft.fft` + `arith.divf` + `fft.ifft` | FFT-based deconvolution |
| `spectral.filter` | `linalg.matmul` | Apply spectral mask in frequency domain |

**MLIR Dialects Used:**
- `morphogen.transform` (FFT, STFT)
- `linalg` (matrix ops, generic ops)
- `vector` (SIMD vectorization)
- `gpu` (GPU kernels)
- `scf` (loops for iterative solvers)

---

## Use Cases

### 1. Recreate the Same Note (Fidelity Test)

**Goal:** Analyze a recording, synthesize the same note, measure fidelity.

**Metrics:**
- Spectral distance (L2 norm in frequency domain)
- Perceptual distance (Mel-frequency cepstral distance)
- Listening test (A/B comparison)

---

### 2. Generate New Notes with Same Timbre

**Goal:** Convert guitar into a MIDI instrument.

**Workflow:**
1. Analyze one note (or multiple notes for multi-sampling)
2. Build instrument model
3. Synthesize any pitch on demand

---

### 3. Generate Legato, Bends, Slides

**Goal:** Expressive synthesis with continuous pitch variation.

**Operators:**
- `pitch.bend` — Apply pitch envelope
- `legato.transition` — Smooth transition between notes

---

### 4. Morph Instruments

**Goal:** Create hybrid timbres (guitar + violin, piano + synth).

**Operator:** `instrument.morph`

**Technique:** Interpolate harmonic amplitudes, blend body IRs, mix excitations

---

### 5. Measure Timbre Evolution (Luthier Tool)

**Goal:** Quantify how timbre changes over note duration.

**Metrics:**
- Spectral centroid over time
- Harmonic amplitude decay rates
- Inharmonicity

**Output:** Plots, numerical data for instrument makers

---

### 6. Use Body Resonance as Reverb

**Goal:** Extract guitar body IR, apply to other sounds.

**Operator:** `convolution.reverb`

**Use case:** Apply guitar body acoustics to synth, voice, drums

---

### 7. Physics-Driven Synthesis

**Goal:** Use extracted parameters to drive physical model.

**Operators:**
- `string.physical` — Physical string model (Karplus-Strong, waveguide)
- `modal.physical` — Physical modal synthesis

**Parameters derived from analysis:**
- String tension, damping, inharmonicity
- Body mode frequencies, Q factors

---

## Determinism Guarantees

| Operator | Tier | Rationale |
|----------|------|-----------|
| `stft` | Strict | Bit-exact FFT |
| `harmonic.track_fundamental` | Repro | Iterative peak tracking (FP precision) |
| `modal.analyze` | Repro | Least-squares fitting (FP precision) |
| `deconvolve` | Repro | Division in frequency domain (FP precision) |
| `additive.synth` | Strict | Deterministic sinusoid summation |
| `modal.synth` | Strict | Deterministic damped sinusoids |
| `instrument.synthesize` | Strict | Deterministic synthesis from model |

---

## Testing Strategy

### 1. Fidelity Tests

**Test:** Analyze → Synthesize → Measure Error

```morphogen
let original = io.load("guitar_e3.wav")
let model = instrument.analyze(original)
let resynthesized = instrument.synthesize(model, pitch=note("E3"), velocity=1.0)

let error = spectral.distance(original, resynthesized)
assert(error < 0.05)  // <5% spectral error
```

---

### 2. Determinism Tests

**Test:** Same input → Same output (bit-exact)

```morphogen
let model1 = instrument.analyze(recording)
let model2 = instrument.analyze(recording)
assert_eq!(model1, model2)
```

---

### 3. Pitch Transposition Tests

**Test:** Synthesize at different pitches, verify harmonic structure

```morphogen
let model = instrument.analyze("guitar_e3.wav")
let note_a3 = instrument.synthesize(model, pitch=220Hz, velocity=1.0)
let note_a4 = instrument.synthesize(model, pitch=440Hz, velocity=1.0)

// Verify harmonic structure is preserved (scaled by 2x)
assert(harmonics_match(note_a3, note_a4, ratio=2.0))
```

---

### 4. Morphing Tests

**Test:** Morph between two instruments, verify interpolation

```morphogen
let guitar = instrument.load("guitar.kairo")
let violin = instrument.load("violin.kairo")

let morph_50 = instrument.morph(guitar, violin, blend=0.5)
let note = instrument.synthesize(morph_50, pitch=440Hz, velocity=1.0)

// Verify spectral centroid is between guitar and violin
assert(guitar.spectral_centroid < note.spectral_centroid < violin.spectral_centroid)
```

---

## Roadmap

### Phase 1: Analysis Operators (v0.9)
- [ ] `stft` (already exists, verify)
- [ ] `harmonic.track_fundamental`
- [ ] `harmonic.track_partials`
- [ ] `spectral.envelope`
- [ ] `decay.fit_exponential`

**Goal:** Extract basic timbre features from recording

---

### Phase 2: Synthesis Operators (v0.9)
- [ ] `additive.synth`
- [ ] `modal.synth`
- [ ] `excitation.pluck`
- [ ] `spectral.filter`

**Goal:** Resynthesize notes from extracted features

---

### Phase 3: End-to-End Pipeline (v0.10)
- [ ] `instrument.analyze` (high-level pipeline)
- [ ] `instrument.synthesize`
- [ ] `InstrumentModel` type
- [ ] Serialization (`instrument.save`, `instrument.load`)

**Goal:** Full record → model → synthesize workflow

---

### Phase 4: Advanced Features (v0.11)
- [ ] `modal.analyze` (resonant mode fitting)
- [ ] `deconvolve` (excitation/resonator separation)
- [ ] `instrument.morph`
- [ ] GPU acceleration (MLIR lowering)

**Goal:** Advanced analysis, morphing, performance optimization

---

## Cross-Domain Integration

**Timbre extraction naturally integrates with other Morphogen domains:**

### Audio ↔ Physics
- Modal analysis reveals physical properties (mass, stiffness, damping)
- Use extracted modes to drive physical models

### Audio ↔ Fields
- Spectrograms are 2D fields (time × frequency)
- Apply field operators (diffusion, advection) to spectrograms

### Audio ↔ Stochastic
- Noise modeling uses stochastic processes
- Monte Carlo sampling for parameter estimation

### Audio ↔ Transform
- STFT, FFT are foundational for all spectral analysis
- Mel-frequency transforms for perceptual modeling

---

## References

### Academic Papers
- **Karplus & Strong (1983)** — "Digital Synthesis of Plucked String and Drum Timbres"
- **Smith (1992)** — "Physical Modeling Using Digital Waveguides"
- **Morrison & Adrien (1993)** — "MOSAIC: A Framework for Modal Synthesis"
- **Engel et al. (2017)** — "Neural Audio Synthesis of Musical Notes with WaveNet Autoencoders" (NSynth)
- **Serra & Smith (1990)** — "Spectral Modeling Synthesis"

### Existing Systems
- **Yamaha VL1** — Physical modeling synthesizer
- **Google Magenta NSynth** — Neural audio synthesis
- **Modalys** — Modal synthesis environment
- **IRCAM AudioSculpt** — Spectral analysis/resynthesis

### Morphogen Documentation
- **transform.md** — Transform operators (FFT, STFT)
- **operator-registry.md** — Operator metadata
- **AUDIO_SPECIFICATION.md** — Audio domain specification
- **../architecture/domain-architecture.md** — Domain organization

---

## Conclusion

Morphogen's timbre extraction and instrument modeling capability is:

✅ **Scientifically grounded** — Built on established DSP techniques (modal analysis, harmonic decomposition, deconvolution)
✅ **Operator-driven** — Composes analysis, synthesis, and modeling operators
✅ **Cross-domain** — Leverages audio, physics, transforms, and stochastic domains
✅ **GPU-accelerated** — MLIR lowering for performance
✅ **Deterministic** — Reproducible analysis and synthesis
✅ **Extensible** — Easy to add new analysis/synthesis methods

**This is a perfect fit for Morphogen's architecture.**

Not only can Morphogen do this — **this is an ideal showcase of Morphogen's cross-domain operator model.**

---

**End of Specification**
