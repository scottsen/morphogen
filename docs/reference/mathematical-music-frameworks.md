# Mathematical Frameworks for Musical Structure

**Version:** 1.0
**Last Updated:** 2025-11-21
**Status:** Reference Guide

---

## Overview

This document catalogs mathematical and physical frameworks that provide natural symbolic languages for representing musical structure. These domains offer notation and conceptual tools that can express musical ideas more cleanly, expressively, and computationally than traditional music theory, especially for algorithmic composition and generative systems.

**Purpose:**
- Identify mathematical domains with built-in musical mappings
- Evaluate symbolic advantages for music representation
- Guide implementation of music-theoretic operators in Morphogen
- Support cross-domain music generation (physics → audio, math → melody)

**See Also:**
- [AMBIENT_MUSIC.md](../domains/AMBIENT_MUSIC.md) - Generative audio domain
- [ACOUSTICS.md](../domains/ACOUSTICS.md) - Physical acoustics modeling
- [math-transformation-metaphors.md](./math-transformation-metaphors.md) - Pedagogical metaphors for transforms
- [AUDIO_SPECIFICATION.md](../../AUDIO_SPECIFICATION.md) - Base audio system

---

## Table of Contents

1. [Group Theory (Symmetry Groups)](#1-group-theory-symmetry-groups)
2. [Topology & Geometric Music Theory](#2-topology--geometric-music-theory)
3. [Linear Algebra / Eigenstuff](#3-linear-algebra--eigenstuff)
4. [Fourier Analysis & Wave Physics](#4-fourier-analysis--wave-physics)
5. [Dynamical Systems & Chaos Theory](#5-dynamical-systems--chaos-theory)
6. [Category Theory (High-Level Structure)](#6-category-theory-high-level-structure)
7. [Information Theory](#7-information-theory)
8. [Graph Theory (Harmonic Networks)](#8-graph-theory-harmonic-networks)
9. [Statistical Mechanics](#9-statistical-mechanics)
10. [Quantum Models (Symbolic)](#10-quantum-models-symbolic)
11. [Summary Table](#summary-table-domains--their-musical-power)
12. [Implementation Roadmap](#implementation-roadmap)

---

## 1. Group Theory (Symmetry Groups)

### Why it fits music

Music is fundamentally built on transformations:
- **Transposition** - shifting pitch up/down
- **Inversion** - mirroring around a pitch axis
- **Retrograde** - time reversal
- **Translation in time** - rhythmic displacement
- **Modulation** - key change
- **Mode shifting** - parallel/relative modes

These are naturally **elements of a group** with clear composition rules.

### The symbology advantage

Instead of verbose descriptions like "transpose the melody up a minor third," you write:

```
T₊₃(x)    # Transposition by 3 semitones
```

Inversion around a pitch axis `p`:

```
Iₚ(x) = 2p - x
```

This makes musical structures **algebraically manipulable**:

```
T₅ ∘ I₇ ∘ T₋₃(melody)    # Compose transformations
```

### What it models well

| Musical Concept | Group-Theoretic Representation |
|----------------|-------------------------------|
| **Tonal cycles** | Cyclic group ℤ₁₂ (12-tone equal temperament) |
| **12-tone composition** | Permutation groups, combinatorics |
| **Mode networks** | Dihedral groups (rotations + reflections) |
| **Chord families** | Equivalence classes under transposition |
| **Harmonic movement** | Group actions on pitch-class sets |
| **Rhythmic tilings** | Wallpaper groups, crystallographic patterns |

### Mathematical rigor

✅ **High** - Group theory provides a **clean algebra for musical structure**:
- Neo-Riemannian theory uses dihedral group D₁₂
- Set theory (Forte numbers) uses transposition/inversion equivalence classes
- Generalized interval systems (Lewin) are literally group actions

### Application in Morphogen

```python
# Group-theoretic pitch transformations
from morphogen.stdlib.music import pitch_class, transformation

# Define a melody as pitch-class set
melody = pitch_class.set([0, 4, 7])  # C major triad

# Apply group operations
transposed = transformation.transpose(melody, semitones=5)  # F major
inverted = transformation.invert(melody, axis=pitch_class.C)
retrograde = transformation.reverse(melody)

# Compose operations
transformed = transformation.compose([
    transformation.transpose(semitones=3),
    transformation.invert(axis=pitch_class.D),
    transformation.transpose(semitones=-2)
])
result = transformed(melody)
```

### References

- **David Lewin** - *Generalized Musical Intervals and Transformations* (1987)
- **Dmitri Tymoczko** - *A Geometry of Music* (2011)
- **Richard Cohn** - Neo-Riemannian operations
- **Allen Forte** - Set-theoretic analysis

---

## 2. Topology & Geometric Music Theory

### Why it fits music

Music lives on **spaces**, not just scales:
- **Pitch classes** lie on a circle (mod 12)
- **Chords** live on tori or orbifolds
- **Voice-leading** creates geometric pathways
- **Modulation** is movement through spaces

### The symbology advantage

Topology provides elegant notation:

| Concept | Topological Representation |
|---------|---------------------------|
| **Tonnetz** | Triadic lattice (neo-Riemannian space) |
| **Orbifolds** | Chord space with symmetry quotients |
| **Geodesics** | Optimal (efficient) voice-leading paths |
| **Homotopy** | "Similar" chord progressions (same class) |

### What it models well

- **Smooth vs. jagged voice leading** - Geodesic distance in chord space
- **Proximity of chords** - Metric topology on harmonic space
- **Common-tone relationships** - Intersection of pitch-class sets
- **Key neighborhood graphs** - Metric on tonal regions
- **Similarity of melodies** - Contour space topology

### Mathematical rigor

✅ **High** - Geometric music theory is mathematically rigorous:
- **Dmitri Tymoczko's orbifold construction** for chord spaces
- **Tonnetz** as a 2D triangular lattice
- **Voice-leading distance** as Euclidean metric in quotient spaces
- **Continuous transformations** between discrete harmonic objects

### Visualization opportunities

These geometric models reduce complex theory to **spatial relationships**:

```python
# Visualize chord space as orbifold
chord_space = music.chord_space(num_voices=3, pitch_range=12)
path = music.voice_leading_path(
    start_chord=[0, 4, 7],   # C major
    end_chord=[2, 5, 9],     # D minor
    optimize="smoothness"
)

# Render as 3D geometric path
visual = geometry.visualize(chord_space, path=path, color_by="distance")
```

### References

- **Dmitri Tymoczko** - *The Geometry of Musical Chords* (Science, 2006)
- **Clifton Callender, Ian Quinn, Dmitri Tymoczko** - Generalized voice-leading spaces
- **Richard Cohn** - Tonnetz and neo-Riemannian theory
- **Guerino Mazzola** - Mathematical Music Theory (topos theory)

---

## 3. Linear Algebra / Eigenstuff

### Why it fits

Music often has **low-dimensional structure**:
- Chord progressions cluster along axes
- Rhythms reduce to basis patterns
- Timbres decompose into eigen-spectra

### Symbology advantage

**Eigenvectors** give clean representations:

```
A v = λ v
```

Where:
- `A` = Musical transformation operator
- `v` = Characteristic pattern (eigenmode)
- `λ` = Scaling factor (eigenvalue)

### What it models

| Musical Phenomenon | Linear Algebra Technique |
|-------------------|-------------------------|
| **Common-practice harmony** | PCA on chord transition matrices |
| **Spectral timbre analysis** | Eigendecomposition of covariance |
| **Rhythm basis functions** | SVD of rhythm patterns |
| **Motif extraction** | Non-negative matrix factorization |
| **Tension-resolution dynamics** | Eigenvalues of Markov transition matrices |

### Example: Chord progression analysis

```python
# Analyze chord progression structure
import numpy as np
from morphogen.stdlib.music import harmony

# Chord transition matrix (which chords follow which)
transitions = harmony.transition_matrix(song)

# Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eig(transitions)

# Dominant eigenvector = most "stable" harmonic pattern
tonal_center = eigenvectors[:, 0]
```

### Mathematical rigor

✅ **High** - Linear algebra directly applicable:
- Spectral methods for audio analysis
- Matrix factorization for pattern discovery
- Dimensionality reduction (PCA/ICA) for feature extraction

### References

- **Julius O. Smith III** - *Spectral Audio Signal Processing*
- **Dan Ellis** - Music information retrieval via PCA
- **François Pachet** - Statistical learning for music generation

---

## 4. Fourier Analysis & Wave Physics

### Why it fits

**Sound is waves.**
- Timbre = spectrum
- Harmony = frequency ratios
- Consonance/dissonance = spectral roughness

### Symbology advantage

Using Fourier transforms:

```
x(t) = Σₙ aₙ e^(iωₙt)
```

Music becomes:
- **Additive** - Sum of sinusoids
- **Visual** - Spectrograms show time-frequency content
- **Analyzable** - Decompose into harmonic/inharmonic components
- **Compressible** - Sparse in frequency domain

### What it models

| Musical Concept | Fourier/Wave Representation |
|----------------|----------------------------|
| **Timbre** | Spectral envelope, harmonic partials |
| **Tuning systems** | Frequency ratios (just intonation, equal temperament) |
| **Resonance** | Peaks in frequency response |
| **Dissonance/roughness** | Critical band theory, beating frequencies |
| **Equal temperament vs. just intonation** | Rational vs. irrational frequency ratios |
| **Sound design** | Additive/subtractive synthesis |

### Wave physics gives a direct, physical vocabulary for sound

```python
# Analyze timbre via spectral decomposition
from morphogen.stdlib.audio import spectral

signal = audio.load("violin.wav")
spectrum = spectral.fft(signal, window="hann")

# Extract harmonic partials
harmonics = spectral.harmonic_peaks(spectrum, fundamental=440.0)

# Visualize spectrogram
spectrogram = spectral.stft(signal, hop_length=512)
visual.colorize(spectrogram, palette="viridis", scale="log")
```

### Mathematical rigor

✅ **High** - Wave physics is exact:
- Fourier analysis is foundational to digital audio
- Helmholtz's theory of consonance/dissonance
- Physical modeling synthesis (waveguides, modal synthesis)

### References

- **Julius O. Smith III** - *Physical Audio Signal Processing*
- **Hermann von Helmholtz** - *On the Sensations of Tone* (1863)
- **Curtis Roads** - *The Computer Music Tutorial* (1996)

---

## 5. Dynamical Systems & Chaos Theory

### Why it fits

Music evolves in time and balances:
- **Predictability** vs. **novelty**
- **Stable attractors** vs. **chaotic variation**
- **Periodic patterns** vs. **aperiodic exploration**

### Symbology advantage

Using:
- **Phase diagrams** - State-space trajectories
- **Attractors** - Stable rhythmic/harmonic patterns
- **Differential equations** - Continuous evolution laws

Gives a natural notation for:
- **Groove formation** - Limit cycles in rhythm space
- **Rhythm cycles** - Periodic orbits
- **Tension curves** - Energy gradients
- **Evolving sound textures** - Strange attractors

### What it models

| Musical Phenomenon | Dynamical System Model |
|-------------------|------------------------|
| **Generative ambient music** | Slow-drifting attractors |
| **Polyrhythms as coupled oscillators** | Phase-locked loops, entrainment |
| **Modulation as state transition** | Bifurcation in parameter space |
| **EDM builds and drops** | Energy accumulation and release |
| **Improvisation** | Chaotic exploration near attractors |

### Example: Coupled rhythm oscillators

```python
# Two rhythmic voices with weak coupling
from morphogen.stdlib.dynamics import oscillator

# Define coupled oscillators
kick = oscillator.phase(freq=2.0, phase=0.0)  # 2 Hz
snare = oscillator.phase(freq=1.5, phase=0.5)  # 1.5 Hz

# Couple with small interaction
coupled = oscillator.couple([kick, snare], coupling_strength=0.1)

# Generates polyrhythm that slowly phase-locks
rhythm = oscillator.trigger_on_phase(coupled, threshold=0.9)
```

### Mathematical rigor

✅ **High** - Dynamical systems theory is rigorous:
- **Coupled oscillators** model rhythm and groove
- **Chaos theory** explains unpredictable variation within structure
- **Bifurcation theory** models sudden musical transitions

### Allows very compact models for complex musical behavior

### References

- **Edward Large** - Oscillator models of rhythm perception
- **Steven Strogatz** - *Sync* (2003) - Coupled oscillators
- **David Temperley** - *Music and Probability* (2007)

---

## 6. Category Theory (High-Level Structure)

### Why it fits

Music is full of:
- **Mappings** - Scales to chords, motifs to variations
- **Transformations** - Compositional operations
- **Functors** between structures - Modes ↔ chords, rhythm ↔ melody

**Category theory** expresses **relationships between relationships**.

### Symbology advantage

Instead of describing specific transformations, you describe **systems of transformations** and how they compose.

**Example:**
- **Objects** = Musical structures (scales, chords, rhythms)
- **Morphisms** = Transformations (transposition, inversion, augmentation)
- **Functors** = Structure-preserving mappings (mode-to-chord, contour-to-melody)

### What it models well

- **Modular generative music systems** - Compositional pipelines
- **Abstract harmonic functions** - Tonic/dominant/subdominant as categories
- **Deep relationships between musical objects** - Universal properties
- **Transformations of transformations** - Higher-order operations

### Example: Functorial mapping

```python
# Category-theoretic music generation
from morphogen.stdlib.music import functor

# Define a functor: Scale → Chord
scale_to_chord = functor.create(
    source_category="scales",
    target_category="chords",
    mapping=lambda scale: scale.triad(degree=1)  # Tonic triad
)

# Apply functor
c_major_scale = scales.major(root=pitch_class.C)
c_major_chord = scale_to_chord(c_major_scale)

# Compose functors
melody_pipeline = functor.compose([
    scale_to_chord,
    chord_to_arpeggio,
    arpeggio_to_melody
])
```

### Mathematical rigor

⚠️ **Medium-High** - Category theory provides:
- Formal framework for compositional structure
- Abstraction over specific musical objects
- Composability guarantees

Useful for **designing powerful music-generation frameworks**.

### References

- **Guerino Mazzola** - *The Topos of Music* (2002)
- **Thomas Noll** - Categorical perspectives on music theory
- **Alexandre Popoff** - Category theory for generative music

---

## 7. Information Theory

### Why it fits

Music is partly an **information game**:
- **Predict → violate → resolve**
- **Build expectations → surprise listener**
- **Balance redundancy and novelty**

Information theory offers crisp notation:

```
H(X) = -Σ p(x) log p(x)    # Entropy (surprise)
I(X;Y) = H(X) - H(X|Y)     # Mutual information
```

### What it models

| Musical Concept | Information-Theoretic Measure |
|----------------|------------------------------|
| **Groove** | Low entropy rhythm with high syncopation uncertainty |
| **Melodic predictability** | Conditional entropy H(note | context) |
| **Harmonic expectation** | Mutual information between chords |
| **Motifs and memory** | Kolmogorov complexity |
| **Hook efficiency** | High information density (bits per second) |

### Example: Measuring melodic surprise

```python
# Information-theoretic analysis of melody
from morphogen.stdlib.music import information

melody = [60, 62, 64, 65, 67, 69, 71, 72]  # C major scale

# Entropy of pitch distribution
entropy = information.entropy(melody)

# Conditional entropy (predictability)
conditional_entropy = information.conditional_entropy(melody, order=2)

# Lower conditional entropy = more predictable
print(f"Predictability: {1 - conditional_entropy/entropy:.2%}")
```

### Mathematical rigor

✅ **High** - Information theory is rigorous:
- Shannon entropy measures surprise/uncertainty
- Mutual information captures dependencies
- Applications in music cognition and generation

### This is great for teaching an algorithm to "feel musical"

### References

- **David Huron** - *Sweet Anticipation* (2006) - Musical expectations
- **Geraint Wiggins** - Information dynamics in music
- **Marcus Pearce** - IDyOM (Information Dynamics of Music) model

---

## 8. Graph Theory (Harmonic Networks)

### Why it fits

**Harmony is a network:**
- Chords connect to **neighbors**
- Modulations are **graph traversals**
- Keys form a **circle of fifths**

Graph notation makes this clean:
- **Nodes** = Chords or keys
- **Edges** = Admissible transitions
- **Weights** = Tension/distance values

### What it models

| Musical Concept | Graph Representation |
|----------------|---------------------|
| **Functional harmony** | Directed graph: I → IV → V → I |
| **Jazz chord cycles** | Weighted graph of ii-V-I progressions |
| **Modulations** | Shortest path between keys |
| **Melody contour networks** | Graph of pitch transitions |
| **Voice-leading graphs** | Edges = minimal voice motion |

### Example: Harmonic progression graph

```python
# Build harmonic network
from morphogen.stdlib.music import graph

# Define chord nodes
chords = ["C", "Am", "F", "G", "Dm", "Em"]

# Define transitions with weights (tension)
transitions = [
    ("C", "Am", 0.2),   # Low tension
    ("C", "F", 0.5),
    ("F", "G", 0.7),
    ("G", "C", 0.1),    # Resolution
    ("Am", "Dm", 0.4),
    ("Dm", "G", 0.6),
]

# Create harmonic graph
harmony_graph = graph.create(nodes=chords, edges=transitions)

# Find optimal progression (minimize tension)
progression = graph.shortest_path(
    harmony_graph,
    start="C",
    end="C",
    optimize="weight"
)
```

### Mathematical rigor

✅ **High** - Graph theory directly applicable:
- Markov chains for chord progressions
- Shortest path algorithms for voice leading
- Network analysis for tonal relationships

### Graph symbology compresses harmonic logic beautifully

### References

- **Christopher White** - *The Harmonic Network* (2018)
- **Dmitri Tymoczko** - Voice-leading geometry and graphs
- **Jason Yust** - Graph-theoretic models of tonality

---

## 9. Statistical Mechanics

### Why it fits

Large-scale musical structure often **emerges from simple rules**:
- **Grooves** emerge from interacting micro-patterns
- **Textures** from interacting oscillators
- **Ensemble music** from many voices

Stat-mech symbology fits this **emergent behavior**.

### What it models

| Musical Concept | Statistical Mechanics Model |
|----------------|----------------------------|
| **Orchestration** | Partition function over instrument states |
| **Energy curves** | Free energy landscapes of tension/release |
| **Dynamic balance** | Equilibrium between competing patterns |
| **Spectral density** | Phonon-like modes in harmonic space |
| **Texture evolution** | Thermodynamic ensembles |

### Example: Ensemble texture as statistical distribution

```python
# Model orchestral texture as ensemble of states
from morphogen.stdlib.music import ensemble

# Define instrument states (playing/resting)
instruments = ["violin", "cello", "flute", "horn"]

# Energy function (lower = more consonant)
def energy(state):
    playing = sum(state.values())
    return abs(playing - 2)  # Prefer 2 instruments playing

# Generate texture via Boltzmann distribution
texture = ensemble.boltzmann(
    instruments=instruments,
    energy_function=energy,
    temperature=1.0  # Higher = more random
)
```

### Mathematical rigor

⚠️ **Medium** - Metaphorical but captures:
- Emergent behavior from local rules
- Temperature as "randomness" parameter
- Phase transitions (sudden texture changes)

### References

- **David Temperley** - Probabilistic models of music
- **Emergent music systems** - Agent-based models

---

## 10. Quantum Models (Symbolic)

### Why it fits

**Not because music is quantum**, but because quantum notation **perfectly captures**:

- **Superposition** - Chords containing multiple functional identities
- **Projection** - Melodies resolving into tonal centers
- **Eigenstates** - Tonic as ground state
- **Measurement** - Collapsing harmonic ambiguity into definite chord

### Symbology advantage

```
|ψ⟩ = α|I⟩ + β|V⟩ + γ|IV⟩    # Chord in harmonic superposition
```

**Projection operators**:
```
P̂_tonic |ambiguous⟩ = |resolved⟩
```

### What it models

| Musical Concept | Quantum Metaphor |
|----------------|-----------------|
| **Harmonic ambiguity** | Superposition of states |
| **Tonal resolution** | Wavefunction collapse |
| **Tonic as stable state** | Ground state eigenvalue |
| **Modulation** | Quantum tunneling between wells |
| **Functional attraction** | Potential energy landscape |

### This gives elegant metaphors for music structure

### Mathematical rigor

⚠️ **Low-Medium** - Metaphorical, but useful for:
- Describing ambiguous harmonies
- Modeling probabilistic expectations
- Visualizing tonal attraction

### References

- **Peter Beim Graben** - Quantum models of music cognition (metaphorical)
- **Guerino Mazzola** - Quantum music theory (speculative)

---

## Summary Table: Domains & Their Musical Power

| Domain | What It's Best At | Rigor | Morphogen Readiness |
|--------|------------------|-------|-------------------|
| **Group Theory** | Transpositions, inversions, symmetry | ✅ High | 🔲 Not implemented |
| **Topology** | Voice-leading, chord space, similarity | ✅ High | 🔲 Not implemented |
| **Linear Algebra** | Timbre, motifs, harmonic axes | ✅ High | ⚠️ Partial (spectral) |
| **Fourier Analysis** | Sound, timbre, tuning | ✅ High | ✅ Implemented (AUDIO) |
| **Dynamical Systems** | Rhythm, energy, evolution | ✅ High | 🔲 Not implemented |
| **Category Theory** | Meta-structure, generative models | ⚠️ Medium | 🔲 Not implemented |
| **Information Theory** | Surprise, hooks, predictability | ✅ High | 🔲 Not implemented |
| **Graph Theory** | Harmony, modulation | ✅ High | 🔲 Not implemented |
| **Statistical Mechanics** | Texture, orchestration | ⚠️ Medium | 🔲 Not implemented |
| **Quantum Formalism** | Tonal attraction, functional overlays | ⚠️ Low | 🔲 Not implemented |

**Legend:**
- ✅ High rigor = Mathematically exact
- ⚠️ Medium rigor = Strong metaphor with some formalization
- ⚠️ Low rigor = Heuristic/pedagogical metaphor

**Morphogen Status:**
- ✅ Implemented
- ⚠️ Partial (subset available)
- 🔲 Not implemented

---

## Implementation Roadmap

### Phase 1: Foundation (Current - v0.11.0)
- ✅ Fourier/spectral analysis (AUDIO domain)
- ✅ Waveguide physics (ACOUSTICS domain)
- ✅ Basic linear algebra (field operations)

### Phase 2: Music Theory Core (v0.12-0.13)
- 🎯 **Group theory operators**
  - `music.transpose(melody, semitones)`
  - `music.invert(melody, axis)`
  - `music.retrograde(melody)`
  - Pitch-class set operations
- 🎯 **Graph-theoretic harmony**
  - `harmony.graph(chords, transitions)`
  - `harmony.progression(start, end, optimize="smoothness")`
  - Circle of fifths topology

### Phase 3: Advanced Structures (v0.14-0.15)
- 🎯 **Topological chord spaces**
  - `geometry.chord_space(num_voices, pitch_range)`
  - `geometry.voice_leading_distance(chord1, chord2)`
  - Orbifold visualization
- 🎯 **Dynamical rhythm systems**
  - `dynamics.coupled_oscillators(frequencies, coupling)`
  - `dynamics.rhythm_attractor(pattern, stability)`
  - Polyrhythm phase-locking

### Phase 4: Generative Intelligence (v0.16+)
- 🎯 **Information-theoretic composition**
  - `music.entropy(sequence)`
  - `music.conditional_probability(note, context)`
  - Surprise-based generation
- 🎯 **Category-theoretic pipelines**
  - `functor.scale_to_chord`
  - `functor.compose([f, g, h])`
  - Modular composition systems

### Phase 5: Cross-Domain Integration (v1.0)
- 🎯 **Physics → Music**
  - Fluid vorticity → granular density
  - CA patterns → rhythmic sequences
  - Reaction-diffusion → spectral evolution
- 🎯 **Math → Music**
  - Fractals → harmonic structure
  - Graph traversal → chord progressions
  - Attractor dynamics → melodic contours

---

## Usage Guidelines

### When to use mathematical frameworks

**Use Group Theory when:**
- Implementing transposition/inversion operations
- Building 12-tone or serial composition tools
- Analyzing symmetries in musical patterns

**Use Topology when:**
- Designing voice-leading algorithms
- Visualizing harmonic relationships
- Measuring chord similarity

**Use Fourier Analysis when:**
- Analyzing timbre or spectrum
- Implementing frequency-domain effects
- Studying harmonic/inharmonic content

**Use Dynamical Systems when:**
- Modeling rhythm and groove
- Creating evolving textures
- Implementing generative ambient systems

**Use Information Theory when:**
- Measuring melodic/harmonic predictability
- Optimizing "hook" effectiveness
- Building expectation-based generators

**Use Graph Theory when:**
- Designing chord progression systems
- Implementing functional harmony
- Creating modulation algorithms

### Combining frameworks

The power comes from **combining** these frameworks:

```python
# Example: Group theory + Graph theory + Information theory
from morphogen.stdlib.music import pitch_class, harmony, information

# 1. Define chord progression graph (Graph Theory)
progression_graph = harmony.graph({
    "C": ["F", "Am", "Em"],
    "F": ["G", "Dm"],
    "G": ["C", "Am"],
    # ...
})

# 2. Find optimal path (minimize surprise - Information Theory)
progression = harmony.walk(
    progression_graph,
    start="C",
    length=8,
    optimize=information.minimize_entropy
)

# 3. Apply transformations (Group Theory)
transposed = [pitch_class.transpose(chord, +5) for chord in progression]
```

---

## Code Examples

### Example 1: Neo-Riemannian Transformations

```python
from morphogen.stdlib.music import chord, transformation

# Define a C major triad
c_major = chord.triad(root=0, quality="major")  # [0, 4, 7]

# Neo-Riemannian transformations
parallel = transformation.P(c_major)      # C minor [0, 3, 7]
leading_tone = transformation.L(c_major)  # E minor [4, 7, 11]
relative = transformation.R(c_major)      # A minor [9, 0, 4]

# Compose transformations: C → e → G → c → C
progression = transformation.compose([
    transformation.L,  # C → e
    transformation.P,  # e → E
    transformation.R,  # E → c♯
    transformation.L,  # c♯ → C
])(c_major)
```

### Example 2: Spectral Chord Analysis

```python
from morphogen.stdlib.audio import spectral
from morphogen.stdlib.music import harmony

# Load audio of a chord
signal = audio.load("piano_chord.wav")

# Spectral analysis (Fourier)
spectrum = spectral.fft(signal)
peaks = spectral.harmonic_peaks(spectrum, num_peaks=6)

# Identify chord from spectrum (Linear Algebra)
chord_name = harmony.identify_chord(peaks.frequencies)
print(f"Detected: {chord_name}")  # "C major 7"
```

### Example 3: Markov Melody Generator

```python
from morphogen.stdlib.music import markov, scale

# Build transition matrix from training data (Information Theory + Graph Theory)
melody_corpus = [
    [60, 62, 64, 65, 67],  # C major scale patterns
    [67, 65, 64, 62, 60],
    # ... more examples
]

transition_matrix = markov.train(melody_corpus, order=1)

# Generate new melody
c_major = scale.major(root=60)
generated = markov.generate(
    transition_matrix,
    start_note=60,
    length=16,
    quantize_to=c_major
)
```

### Example 4: Coupled Rhythm Oscillators

```python
from morphogen.stdlib.dynamics import oscillator

# Two rhythmic patterns with different periods (Dynamical Systems)
kick = oscillator.phase(freq=2.0)    # 2 Hz (120 BPM eighth notes)
hihat = oscillator.phase(freq=6.0)   # 6 Hz (16th notes)

# Weakly couple them
coupled = oscillator.kuramoto([kick, hihat], coupling=0.05)

# Generate trigger events
for t in range(1000):
    phases = coupled.step(dt=0.01)

    # Trigger on phase wrap
    if phases[0] > 0.95:
        audio.trigger("kick.wav")
    if phases[1] > 0.95:
        audio.trigger("hihat.wav")
```

---

## Validation Criteria

When implementing operators based on these frameworks:

### ✅ Mathematical Correctness
- Does the implementation match the formal definition?
- Are edge cases handled properly?
- Is numerical stability ensured?

### ✅ Musical Validity
- Does it produce musically meaningful results?
- Can it handle real-world musical examples?
- Does it align with music theory practice?

### ✅ Computational Efficiency
- Is it fast enough for real-time use?
- Can it scale to large structures?
- Are there GPU acceleration opportunities?

### ✅ Composability
- Can operators be combined cleanly?
- Do they integrate with existing domains?
- Are types compatible across domains?

---

## Contributing

To add new mathematical-music frameworks:

1. **Document the framework** in this file following the template:
   - Why it fits music
   - Symbology advantage
   - What it models well
   - Mathematical rigor assessment
   - Example applications

2. **Propose operators** in relevant domain files:
   - `morphogen/stdlib/music.py` (general music theory)
   - `morphogen/stdlib/harmony.py` (harmonic analysis)
   - `morphogen/stdlib/rhythm.py` (rhythmic structures)

3. **Add examples** demonstrating the framework:
   - Code snippets in this document
   - Full examples in `/examples/music/`

4. **Cross-reference** with related documents:
   - Link to relevant ADRs
   - Update domain documentation
   - Add to ECOSYSTEM_MAP.md if significant

---

## References

### Academic Books

- **Dmitri Tymoczko** - *A Geometry of Music* (2011)
- **David Lewin** - *Generalized Musical Intervals and Transformations* (1987)
- **Guerino Mazzola** - *The Topos of Music* (2002)
- **David Huron** - *Sweet Anticipation* (2006)
- **Julius O. Smith III** - *Spectral Audio Signal Processing* (online)

### Research Papers

- **Tymoczko** - "The Geometry of Musical Chords" (Science, 2006)
- **Richard Cohn** - "Neo-Riemannian Operations, Parsimonious Trichords, and Their Tonnetz Representations"
- **Marcus Pearce** - "The construction and evaluation of statistical models of melodic structure in music perception and composition"

### Software & Tools

- **music21** (Python) - Music theory analysis
- **OpenMusic** (Lisp) - Computer-aided composition
- **Euterpea** (Haskell) - Functional music representation
- **SuperCollider** - Sound synthesis and algorithmic composition

### Morphogen Documentation

- [AMBIENT_MUSIC.md](../domains/AMBIENT_MUSIC.md) - Generative audio domain
- [ACOUSTICS.md](../domains/ACOUSTICS.md) - Physical acoustics
- [AUDIO_SPECIFICATION.md](../../AUDIO_SPECIFICATION.md) - Audio system spec
- [math-transformation-metaphors.md](./math-transformation-metaphors.md) - Pedagogical metaphors

---

**Status Summary:**

📚 **Reference Document:** Complete
🎯 **Implementation Target:** v0.12-v1.0
🔗 **Cross-Domain Integration:** High priority
🎵 **Musical Completeness:** Foundation for comprehensive music theory support

*This document provides the theoretical foundation for implementing mathematically-grounded music operators in Morphogen, enabling cross-domain music generation and analysis.*
