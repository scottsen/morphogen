---
project: morphogen
type: architecture
status: design
date: 2025-12-10
author: TIA + Scott Senften
keywords:
  - music-architecture
  - semantic-layers
  - mlir
  - audio-analysis
  - audio-synthesis
beth_topics:
  - morphogen-music
  - semantic-audio
  - mlir-music
---

# 🎼 Music Semantic Layers Architecture

**A Unified Framework for Music Understanding, Generation, and Performance**

---

## Executive Summary

This document defines a **7-layer semantic architecture** for music computation that unifies:
- **Analysis** (understanding existing music)
- **Synthesis** (generating audio)
- **Composition** (creating musical structures)

Each layer represents a distinct **semantic abstraction level**, with clean interfaces between layers. This architecture enables:
- Composable operations at each level
- Bidirectional flow (analysis ↔ synthesis)
- MLIR compilation for optimization
- Clear separation of concerns

**Key Insight**: Music has natural semantic layers (physical → perceptual → symbolic → structural → compositional). MLIR's multi-level IR maps perfectly to these musical abstractions.

---

## 🏗️ The 7 Semantic Layers

### Overview

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 7: COMPOSITIONAL (Musical Intent)                     │
│ • Voice-leading constraints, harmonic progressions          │
│ • Generative rules, style parameters                        │
│ • Examples: "ii-V-I in jazz style", "smooth voice-leading"  │
├─────────────────────────────────────────────────────────────┤
│ Layer 6: STRUCTURAL (Musical Form)                          │
│ • Sections, phrases, hypermeter                             │
│ • Repetition, development, variation                        │
│ • Examples: "verse-chorus-bridge", "AABA form"              │
├─────────────────────────────────────────────────────────────┤
│ Layer 5: SYMBOLIC (Music Theory)                            │
│ • Notes, chords, keys, scales                               │
│ • Beats, meter, tempo                                       │
│ • Examples: "Cmin7", "120 BPM", "4/4 time"                  │
├─────────────────────────────────────────────────────────────┤
│ Layer 4: FEATURE (Perceptual)                               │
│ • Mel-frequency, chroma, spectral centroid                  │
│ • Timbre descriptors, musical features                      │
│ • Examples: "brightness", "harmonic content", "roughness"   │
├─────────────────────────────────────────────────────────────┤
│ Layer 3: SPECTRAL (Frequency Domain)                        │
│ • FFT, STFT, spectral operations                            │
│ • Harmonic-percussive separation                            │
│ • Examples: "frequency bins", "magnitude spectrum"          │
├─────────────────────────────────────────────────────────────┤
│ Layer 2: DSP (Signal Processing)                            │
│ • Filters, envelopes, effects                               │
│ • Oscillators, modulators, dynamics                         │
│ • Examples: "lowpass filter", "ADSR envelope", "reverb"     │
├─────────────────────────────────────────────────────────────┤
│ Layer 1: PHYSICAL (Waveform/Sample)                         │
│ • Audio buffers, samples, sample rate                       │
│ • Mono/stereo signals, raw PCM data                         │
│ • Examples: "48kHz float32 buffer", "stereo waveform"       │
└─────────────────────────────────────────────────────────────┘
```

**Information Flow**:
- **Analysis** (INPUT): Layer 1 → 2 → 3 → 4 → 5 → 6 → 7
- **Synthesis** (OUTPUT): Layer 7 → 6 → 5 → 2 → 1
- **Bidirectional**: Any layer can interface with adjacent layers

---

## 📐 Layer Definitions

### Layer 1: PHYSICAL (Waveform/Sample)

**Semantic Domain**: Raw audio data, physical representation

**Types**:
```python
AudioBuffer:
  - data: ndarray[float32]      # Sample data
  - sample_rate: int            # 44100, 48000, etc.
  - channels: int               # 1 (mono), 2 (stereo)
  - duration: float             # Seconds
```

**Operations**:
- Buffer manipulation: `slice`, `concat`, `resample`, `reverse`
- Format conversion: `mono_to_stereo`, `stereo_to_mono`
- I/O: `load`, `save`, `play`, `record`
- Basic arithmetic: `add`, `multiply`, `clip`, `normalize`

**MLIR Dialect**: `physical`
```mlir
!physical.buffer<f32>
!physical.sample

physical.buffer.alloc
physical.buffer.fill
physical.add
physical.mul
```

**Semantics**: No musical meaning. Pure signal data.

---

### Layer 2: DSP (Signal Processing)

**Semantic Domain**: Audio signal transformations, time-domain processing

**Types**:
```python
Signal:           # Processed audio
Filter:           # Filter coefficients
Envelope:         # Time-varying control
ControlSignal:    # Modulation/CV
```

**Operations**:

**Generators**:
- Oscillators: `sine`, `saw`, `square`, `triangle`
- Noise: `white`, `pink`, `brown`, `impulse`

**Processors**:
- Filters: `lowpass`, `highpass`, `bandpass`, `notch`, `eq`
- Dynamics: `limiter`, `compressor`, `expander`, `gate`
- Distortion: `drive`, `clip`, `waveshape`

**Effects**:
- Time: `delay`, `echo`, `chorus`, `flanger`
- Space: `reverb`, `convolution`
- Modulation: `vca`, `vcf`, `ring_mod`

**Envelopes**:
- `adsr`, `ar`, `env_exp`, `env_custom`

**MLIR Dialect**: `dsp`
```mlir
!dsp.signal
!dsp.filter
!dsp.envelope

dsp.oscillator
dsp.filter.biquad
dsp.envelope.adsr
dsp.delay
```

**Semantics**: Audio transformations. No pitch/chord awareness.

---

### Layer 3: SPECTRAL (Frequency Domain)

**Semantic Domain**: Frequency-domain representation and analysis

**Types**:
```python
Spectrum:         # Frequency magnitude
PhaseSpectrum:    # Frequency phase
Spectrogram:      # Time-frequency representation (STFT)
```

**Operations**:

**Transforms**:
- `fft`, `ifft` (Fast Fourier Transform)
- `stft`, `istft` (Short-Time Fourier Transform)
- `dct`, `idct` (Discrete Cosine Transform)

**Spectral Processing**:
- `spectral_gate` (noise reduction)
- `spectral_filter` (frequency masking)
- `convolution` (frequency-domain convolution)
- `hpss` (Harmonic-Percussive Source Separation)

**Analysis**:
- `spectrum`, `phase_spectrum`
- `spectral_peaks` (peak detection)
- `spectral_flux` (temporal change)

**MLIR Dialect**: `spectral`
```mlir
!spectral.spectrum
!spectral.spectrogram
!spectral.stft_matrix

spectral.fft
spectral.stft
spectral.hpss
spectral.gate
```

**Semantics**: Frequency representation. Bridge between time and perceptual domains.

---

### Layer 4: FEATURE (Perceptual/Musical)

**Semantic Domain**: Perceptual and musical features extracted from audio

**Types**:
```python
MelSpectrogram:   # Mel-frequency representation
Chroma:           # Pitch class energy (12 bins)
MFCC:             # Mel-Frequency Cepstral Coefficients
SSM:              # Self-Similarity Matrix (for structure)

SpectralFeatures:
  - centroid      # Spectral center of mass ("brightness")
  - rolloff       # High-frequency cutoff
  - flux          # Spectral change rate
  - flatness      # Tonality vs noise

TemporalFeatures:
  - rms           # Root mean square (loudness)
  - zcr           # Zero-crossing rate (noisiness)
  - onset_strength # Attack/transient energy
```

**Operations**:

**Perceptual Transforms**:
- `melspectrogram` (human frequency perception)
- `chroma` (pitch class representation)
- `mfcc` (timbre features for ML)
- `self_similarity_matrix` (repetition structure)

**Feature Extraction**:
- `spectral_centroid`, `spectral_rolloff`, `spectral_flux`
- `rms`, `zero_crossings`
- `onset_detect`, `onset_strength`

**Timbre Analysis**:
- `spectral_envelope` (formants)
- `track_fundamental` (pitch tracking)
- `track_partials` (harmonic tracking)
- `analyze_modes` (modal decomposition)
- `measure_inharmonicity`

**MLIR Dialect**: `features`
```mlir
!features.mel
!features.chroma
!features.mfcc
!features.ssm

features.melspectrogram
features.chroma
features.mfcc
features.ssm
features.spectral_centroid
```

**Semantics**: Perceptually meaningful features. Bridge to symbolic music.

---

### Layer 5: SYMBOLIC (Music Theory)

**Semantic Domain**: Musical concepts - notes, chords, rhythm, meter

**Types**:
```python
# Pitch
Pitch:            # MIDI note number or Hz
PitchClass:       # 0-11 (C, C#, D, ...)
Note:             # Pitch + duration + velocity

# Harmony
Chord:            # Root + quality + extensions
ChordClass:       # Chord category (maj, min, dim, aug, etc.)
ChordSequence:    # Sequence of chords with timing
Key:              # Tonic + mode (major/minor)
Scale:            # Set of pitch classes

# Rhythm
Beat:             # Beat time (in seconds or samples)
BeatGrid:         # Sequence of beats with downbeats
Tempo:            # BPM with confidence
Meter:            # Time signature (4/4, 3/4, etc.)
```

**Operations**:

**Extraction (Analysis - INPUT)**:
- `beat_track` (beat detection from audio)
- `estimate_tempo` (BPM estimation)
- `estimate_meter` (time signature)
- `chord_estimate` (chord recognition from chroma)
- `key_estimate` (key detection)

**Generation (Synthesis - OUTPUT)**:
- `note.create` (pitch + duration + velocity)
- `chord.build` (construct chord from theory)
- `scale.generate` (generate scale degrees)
- `arpeggio` (note pattern from chord)

**Transformation**:
- `transpose` (pitch shift)
- `invert` (interval inversion)
- `retrograde` (time reversal)

**MLIR Dialect**: `symbolic`
```mlir
!symbolic.pitch
!symbolic.note
!symbolic.chord
!symbolic.beat_grid
!symbolic.key

symbolic.beat_track
symbolic.chord_estimate
symbolic.note.create
symbolic.chord.build
symbolic.transpose
```

**Semantics**: Musical meaning. Notes, chords, rhythm.

---

### Layer 6: STRUCTURAL (Musical Form)

**Semantic Domain**: Song structure, sections, phrases, form

**Types**:
```python
Section:          # Labeled segment (verse, chorus, bridge)
SectionSequence:  # Ordered sections with timings
Phrase:           # Musical phrase (typically 4-8 bars)
Form:             # Overall structure (AABA, verse-chorus, etc.)

StructuralFeatures:
  - novelty       # Structural change detection
  - repetition    # Self-similarity regions
  - boundaries    # Section boundaries
```

**Operations**:

**Extraction (Analysis - INPUT)**:
- `segment_structure` (find section boundaries from SSM)
- `detect_novelty` (structural change points)
- `label_sections` (classify section types)
- `find_repetitions` (repeated segments)

**Generation (Synthesis - OUTPUT)**:
- `create_form` (AABA, verse-chorus, etc.)
- `develop_motif` (motivic development)
- `create_variation` (thematic variation)

**MLIR Dialect**: `structural`
```mlir
!structural.section
!structural.section_sequence
!structural.form

structural.segment
structural.detect_novelty
structural.label_sections
structural.create_form
```

**Semantics**: Musical architecture. How sections relate.

---

### Layer 7: COMPOSITIONAL (Musical Intent)

**Semantic Domain**: High-level musical intentions, style, constraints

**Types**:
```python
Progression:      # Chord progression with voice-leading
VoiceLeading:     # Voice movement constraints
HarmonicIntent:   # "ii-V-I", "modal interchange", etc.
Style:            # Jazz, classical, pop, etc.

Constraints:
  - smooth         # Minimal voice motion
  - common         # Common-tone retention
  - contrary       # Contrary motion
  - range          # Voice range limits
```

**Operations**:

**Analysis (INPUT)**:
- `analyze_voice_leading` (how voices move)
- `identify_progression` (common progressions)
- `classify_style` (genre/style classification)

**Generation (OUTPUT)**:
- `voice_lead` (solve voice-leading constraints)
- `harmonize` (add harmony to melody)
- `reharmonize` (substitute chords)
- `generate_progression` (create chord sequence)
- `substitute_chord` (ii for IV, tritone sub, etc.)

**Constraints**:
- `apply_range_constraints`
- `enforce_voice_leading_rules`
- `apply_style_rules`

**MLIR Dialect**: `compositional`
```mlir
!compositional.progression
!compositional.voice_leading
!compositional.constraints

compositional.voice_lead
compositional.harmonize
compositional.reharmonize
compositional.substitute
```

**Semantics**: Musical intent and creative decisions.

---

## 🔄 Bidirectional Flow Examples

### Analysis Path (INPUT): Audio → Understanding

```
Layer 1 (Physical):
  audio = load("song.wav")  # AudioBuffer(48kHz, stereo)

↓ DSP

Layer 2 (DSP):
  resampled = resample(audio, 22050)  # Downsample for analysis

↓ Spectral Transform

Layer 3 (Spectral):
  stft_matrix = stft(resampled, window=2048, hop=512)
  harmonic, percussive = hpss(stft_matrix)

↓ Feature Extraction

Layer 4 (Feature):
  mel = melspectrogram(harmonic, n_mels=128)
  chroma = chroma_features(harmonic, n_chroma=12)
  ssm = self_similarity_matrix(chroma)

↓ Symbolic Extraction

Layer 5 (Symbolic):
  beats = beat_track(percussive, resampled.sr)
  tempo = estimate_tempo(beats)
  chords = chord_estimate(chroma, beats)
  key = key_estimate(chroma)

↓ Structural Analysis

Layer 6 (Structural):
  sections = segment_structure(ssm, beats)
  form = label_sections(sections, chords)

↓ Compositional Understanding

Layer 7 (Compositional):
  progression = analyze_progression(chords, key)
  voice_leading = analyze_voice_leading(chords)
  style = classify_style(progression, beats, timbre)
```

**Result**: Rich semantic understanding of the song.

---

### Synthesis Path (OUTPUT): Intent → Audio

```
Layer 7 (Compositional):
  progression = create_progression("ii-V-I", key="C", style="jazz")
  voiced = voice_lead(progression, constraints=["smooth", "common"])

↓ Structural Generation

Layer 6 (Structural):
  form = create_form("AABA", bars=32)
  verse = apply_progression(progression, form.A)
  bridge = apply_progression(reharmonize(progression), form.B)

↓ Symbolic Realization

Layer 5 (Symbolic):
  notes = progression_to_notes(voiced, tempo=120, meter="4/4")
  melody = harmonize(notes, style="jazz")

↓ (Skip direct DSP, use synthesis)

Layer 2 (DSP):
  # For each note
  osc = sine(note.pitch, duration=note.duration)
  env = adsr(attack=0.01, decay=0.1, sustain=0.7, release=0.3)
  signal = multiply(osc, env)

  # Apply effects
  filtered = lowpass(signal, cutoff=2000, q=0.7)
  with_reverb = reverb(filtered, mix=0.3)

↓ Mix to Physical

Layer 1 (Physical):
  final = mix([bass_buffer, harmony_buffer, melody_buffer])
  normalized = normalize(final, target=0.95)
  save(normalized, "output.wav")
```

**Result**: Generated audio from musical intent.

---

### Round-Trip: Analysis → Variation → Synthesis

```
# Analyze existing song
INPUT: "original.wav" → Layers 1-7 → SongStructure

# Transform at symbolic/compositional level
progression_variant = substitute_chords(original.progression)
reharmonized = voice_lead(progression_variant, "smooth")

# Synthesize variation
OUTPUT: Layers 7-1 → "variation.wav"
```

**Result**: AI-assisted composition and arrangement.

---

## 🎯 MLIR Dialect Mapping

Each semantic layer maps to an MLIR dialect:

| Semantic Layer | MLIR Dialect | Responsibility |
|----------------|--------------|----------------|
| 7. Compositional | `compositional` | Voice-leading, progressions, style |
| 6. Structural | `structural` | Sections, form, repetition |
| 5. Symbolic | `symbolic` | Notes, chords, beats, harmony |
| 4. Feature | `features` | Mel, chroma, MFCC, perceptual |
| 3. Spectral | `spectral` | FFT, STFT, frequency domain |
| 2. DSP | `dsp` | Filters, oscillators, effects |
| 1. Physical | `physical` | Buffers, samples, I/O |

**Plus standard MLIR dialects**:
- `linalg` - Linear algebra operations
- `arith` - Arithmetic operations
- `memref` - Memory management
- `scf` - Control flow
- `llvm` - LLVM lowering
- `gpu` / `spirv` - GPU backends

---

## 🔌 Neural Network Integration

Neural models operate primarily at **Feature → Symbolic** boundary:

```
Layer 4 (Feature):
  mel = melspectrogram(audio)
  chroma = chroma_features(audio)

↓ Neural Processing (StableHLO)

StableHLO ops:
  backbone_features = ConformerBackbone(mel)

  # Parallel heads
  beat_probs = BeatHead(backbone_features)
  chord_logits = ChordHead(backbone_features)
  section_logits = SectionHead(backbone_features)

↓ Decoding to Symbolic

Layer 5 (Symbolic):
  beats = discretize_beats(beat_probs)
  chords = decode_chords(chord_logits, beats)

Layer 6 (Structural):
  sections = decode_sections(section_logits, beats)
```

**Integration Point**: `features` dialect → StableHLO → `symbolic` dialect

---

## 🧩 Composability Principles

### 1. **Vertical Composability** (Between Layers)

Layers communicate via well-defined types:

```python
# Layer 1 → Layer 3
audio: AudioBuffer → stft() → Spectrogram

# Layer 3 → Layer 4
spectrogram: Spectrogram → melspectrogram() → MelSpectrogram

# Layer 4 → Layer 5
chroma: Chroma → chord_estimate() → ChordSequence
```

**Rule**: Each layer only depends on adjacent layers.

---

### 2. **Horizontal Composability** (Within Layer)

Operations at same layer compose freely:

```python
# Layer 2 (DSP)
osc = sine(440)
filtered = lowpass(osc, 1200)
delayed = delay(filtered, 0.3)
final = reverb(delayed, 0.5)

# Layer 5 (Symbolic)
chord1 = Chord("C", "maj7")
chord2 = transpose(chord1, +5)
chord3 = invert(chord2, 1)
```

**Rule**: Operations at same layer compose via pipelines.

---

### 3. **Orthogonal Concerns**

Some concerns cut across layers:

- **Time**: All layers have temporal dimension
- **Parallelism**: GPU/CPU scheduling
- **Optimization**: Graph fusion, kernel merging
- **Provenance**: Track data lineage

**Solution**: MLIR passes handle cross-cutting concerns.

---

## 📊 Implementation Status

### Current State (Morphogen)

| Layer | Status | Operations | MLIR Dialect |
|-------|--------|------------|--------------|
| 1. Physical | ✅ Complete | 15+ ops | `physical` (partial) |
| 2. DSP | ✅ Complete | 45+ ops | `dsp` (partial) |
| 3. Spectral | ⚠️ Partial | FFT, STFT (need HPSS) | ❌ Missing |
| 4. Feature | ⚠️ Partial | Basic (need mel, chroma) | ❌ Missing |
| 5. Symbolic | ❌ Missing | Timbre only | ❌ Missing |
| 6. Structural | ❌ Missing | None | ❌ Missing |
| 7. Compositional | ❌ Missing | None | ❌ Missing |

**Python stdlib**: Layers 1-4 have 60+ operations implemented.
**MLIR dialects**: Only partial `dsp` and `audio` dialects exist.

---

### RiffStack Vision

| Layer | RiffStack Name | Status |
|-------|----------------|--------|
| 7. Compositional | Harmony DSL | 📝 Design only |
| 6. Structural | (Implicit in forms) | 📝 Design only |
| 5. Symbolic | Theory IR / Note IR | 📝 Design only |
| 2. DSP | Audio IR | ⚠️ Minimal (11 ops) |
| 1. Physical | DSP IR | 📝 Design only |

**Status**: Excellent architectural vision in docs, minimal implementation.

---

## 🚀 Integration Strategy

### Unified Platform: Morphogen as Music Foundation

**Recommendation**: Implement all 7 layers in Morphogen.

**Why**:
1. Morphogen has multi-domain architecture
2. Already has Layers 1-2 mature, 3-4 partial
3. Has MLIR infrastructure
4. Has mathematical music frameworks (group theory, topology)
5. Can serve both analysis and synthesis

**RiffStack Role**:
- High-level YAML/DSL frontend
- User-facing performance tools
- Compiles to Morphogen's compositional/symbolic layers

---

## 📋 Next Steps

1. **Complete Layer 4 (Feature)**:
   - Add `melspectrogram`, `chroma`, `hpss`, `ssm`
   - Create `morphogen/stdlib/audio_features.py`

2. **Implement Layer 5 (Symbolic)**:
   - Create `morphogen/stdlib/music_symbolic.py`
   - Beat tracking, chord estimation, key detection
   - Define `!symbolic.*` MLIR types

3. **Implement Layer 6 (Structural)**:
   - Create `morphogen/stdlib/music_structural.py`
   - Segmentation, section labeling
   - Define `!structural.*` MLIR types

4. **Design Layer 7 (Compositional)**:
   - Create `morphogen/stdlib/music_compositional.py`
   - Voice-leading solver, progression generator
   - Define `!compositional.*` MLIR types
   - Integrate RiffStack's Harmony DSL concepts

5. **Create MLIR Dialects**:
   - `mlir/dialects/features.py`
   - `mlir/dialects/symbolic.py`
   - `mlir/dialects/structural.py`
   - `mlir/dialects/compositional.py`

6. **Document Integration**:
   - ADR for music consolidation
   - Update Morphogen README with music capabilities
   - Create migration guide for RiffStack concepts

---

## 📚 Related Documents

- **[MLIR Architecture](mlir-architecture.md)** - MLIR compilation strategy
- **[Mathematical Music Frameworks](../reference/mathematical-music-frameworks.md)** - Theory foundations
- **[ADR-009: Ambient Music Domains](../adr/009-ambient-music-generative-domains.md)** - Generative audio
- **RiffStack: [Harmony DSL Vision](/home/scottsen/src/projects/riffstack/docs/HARMONY_DSL_VISION.md)** - Compositional layer design
- **RiffStack: [MLIR Architecture](/home/scottsen/src/projects/riffstack/docs/MLIR_ARCHITECTURE.md)** - Creative compiler stack

---

**Last Updated**: 2025-12-10
**Status**: Architecture design document
**Next**: Implementation plan + ADR
