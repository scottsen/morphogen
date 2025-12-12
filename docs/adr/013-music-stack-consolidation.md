---
project: morphogen
type: adr
status: proposed
date: 2025-12-10
author: TIA + Scott Senften
decision-id: ADR-013
keywords:
  - music-architecture
  - riffstack-integration
  - consolidation
  - mlir-music
beth_topics:
  - morphogen-architecture
  - music-platform
---

# ADR-013: Music Stack Consolidation in Morphogen

## Status

**Proposed** - 2025-12-10

## Context

### The Problem

We have two music-related projects with overlapping goals:

1. **Morphogen** - Multi-domain creative computation platform
   - Has 60+ audio operations (synthesis + analysis)
   - Has MLIR infrastructure (compiler, dialects, lowering)
   - Has mathematical music frameworks (group theory, topology, linear algebra)
   - Mature implementation (3,800+ lines of audio code)
   - Missing: high-level music understanding and composition

2. **RiffStack** - Audio performance language
   - Has excellent MLIR architecture vision (7-layer creative compiler)
   - Has Harmony DSL design (voice-leading, progressions, composition)
   - Minimal implementation (11 basic operations)
   - Missing: most of the envisioned stack

### The Overlap

**Significant duplication**:
- Both define audio MLIR dialects
- Both target LLVM/SPIR-V/WebGPU backends
- Both work with same audio primitives (oscillators, filters, effects)
- Both need spectral analysis (FFT, STFT)

**Complementary strengths**:
- Morphogen has implementation (bottom-up)
- RiffStack has vision (top-down)

### The Opportunity

A **new requirement emerged**: Music analysis/understanding (INPUT side)
- Mel spectrograms, chroma features, HPSS
- Beat tracking, chord recognition, structural segmentation
- Neural model integration (StableHLO)
- Rich symbolic music representation

**Question**: Where should this live? Three options:
1. New standalone project ("MusicBrain")
2. Extend RiffStack
3. Extend Morphogen

**This ADR documents the decision to consolidate into Morphogen.**

---

## Decision

**We will consolidate music work into Morphogen** as a unified music platform covering:
- **Analysis** (understanding existing music)
- **Synthesis** (generating audio)
- **Composition** (creating musical structures)

**Specifically**:

1. **Morphogen becomes the comprehensive music platform**:
   - Implement 7 semantic layers (Physical → Compositional)
   - Add missing analysis capabilities (mel, chroma, HPSS, beat tracking, chord recognition)
   - Add missing composition capabilities (voice-leading, progression generation)
   - Integrate neural models (StableHLO) for music understanding
   - Create MLIR dialects for all 7 layers

2. **RiffStack transitions to high-level frontend**:
   - YAML/DSL parser for Harmony DSL and performance language
   - User-facing CLI and performance tools
   - Compiles to Morphogen's MLIR dialects (compositional/symbolic/dsp)
   - Focus on musician UX, not DSP implementation

3. **Shared types and interfaces**:
   - Define canonical music types in Morphogen
   - RiffStack consumes these types
   - Clear API boundary

---

## Rationale

### Why Morphogen?

**1. Infrastructure Already Exists**

Morphogen has:
- ✅ MLIR compiler infrastructure (`mlir/compiler.py`, `mlir/optimizer.py`)
- ✅ Dialect system (`mlir/dialects/`)
- ✅ Lowering passes (`mlir/lowering/`)
- ✅ Multi-domain architecture (23 domains registered)
- ✅ Cross-domain integration framework
- ✅ JIT and AOT compilation

**RiffStack has**: Vision documents.

**Verdict**: Don't rebuild what exists.

---

**2. Audio Operations Already Implemented**

Morphogen has **60+ audio operations**:

| Category | Morphogen | RiffStack |
|----------|-----------|-----------|
| Oscillators | 5 (sine, saw, square, triangle, noise) | 4 |
| Filters | 10+ (lowpass, highpass, bandpass, notch, VCF, EQ) | 2 |
| Effects | 8 (delay, reverb, chorus, flanger, drive, limiter) | 2 |
| Envelopes | 3 (ADSR, AR, exp) | 0 |
| Spectral | 10 (FFT, STFT, spectral features) | 0 |
| Physical Models | 2 (string, modal) | 0 |
| I/O | 4 (play, save, load, record) | 0 |
| Signal Processing | 15+ (mix, gain, normalize, clip, resample, etc.) | 3 |
| **Total** | **60+** | **11** |

**Verdict**: Morphogen has 6x more audio code. Consolidate there.

---

**3. Mathematical Music Frameworks Exist**

Morphogen already has `docs/reference/mathematical-music-frameworks.md` covering:
- Group theory for chord symmetries
- Topology for voice-leading spaces
- Linear algebra for structural analysis
- Fourier analysis for spectral work

These are **exactly what the symbolic/compositional layers need**.

RiffStack references these concepts but doesn't have the foundation.

**Verdict**: Build on existing mathematical infrastructure.

---

**4. Analysis + Synthesis Need Same DSP**

Music understanding (INPUT) and music generation (OUTPUT) share DSP:

| Operation | Analysis Use | Synthesis Use |
|-----------|--------------|---------------|
| FFT/STFT | Feature extraction | Spectral effects |
| Filters | Frequency isolation | Sound shaping |
| Envelopes | Onset detection | Amplitude shaping |
| Resample | Normalize sample rates | Pitch shifting |

**In Morphogen**: Share code across domains.
**In separate projects**: Duplicate or coordinate.

**Verdict**: Unified platform is more efficient.

---

**5. Multi-Domain Philosophy Fits**

Morphogen is **designed** for multiple domains:
- Field (PDEs)
- Agents (swarms)
- Cellular (automata)
- Graph (networks)
- Neural (ML)
- **Audio (sound)** ← natural fit
- Vision (images)
- etc.

Music as a **multi-faceted domain** (physical → perceptual → symbolic → compositional) maps perfectly to Morphogen's architecture.

**Verdict**: Music is a domain, Morphogen hosts domains.

---

**6. MLIR Compilation Across Analysis→Synthesis Loop**

Key optimization opportunity: **joint compilation** of analysis + generation.

Example: AI-assisted composition
```
Audio → (analysis) → Structure → (transform) → Modified structure → (synthesis) → New audio
```

If analysis and synthesis are in **one MLIR compiler**:
- ✅ Fuse operations across pipeline
- ✅ Optimize memory layout end-to-end
- ✅ GPU scheduling across full pipeline
- ✅ Dead code elimination

If in separate projects:
- ❌ Serialize/deserialize between pipelines
- ❌ Separate compilation (missed optimizations)
- ❌ Coordination overhead

**Verdict**: Unified MLIR compiler enables better optimization.

---

### Why Not Keep Them Separate?

**Arguments for separation**:
1. "RiffStack is for musicians, Morphogen is for technical users"
   - **Counter**: RiffStack can still be user-facing frontend
2. "Separate projects have clear ownership"
   - **Counter**: Morphogen has clear domain structure (audio domain)
3. "Easier to iterate independently"
   - **Counter**: Shared code means bug fixes benefit both

**Arguments against separation**:
1. Duplicate MLIR infrastructure (compiler, dialects, lowering)
2. Duplicate DSP operations
3. Duplicate testing
4. Coordination overhead
5. Missed optimization opportunities

**Verdict**: Costs outweigh benefits.

---

## Architecture

### 7 Semantic Layers (All in Morphogen)

See **[MUSIC_SEMANTIC_LAYERS.md](../architecture/MUSIC_SEMANTIC_LAYERS.md)** for full specification.

```
Layer 7: COMPOSITIONAL   → compositional dialect (voice-leading, progressions)
Layer 6: STRUCTURAL      → structural dialect (sections, form)
Layer 5: SYMBOLIC        → symbolic dialect (notes, chords, beats)
Layer 4: FEATURE         → features dialect (mel, chroma, MFCC)
Layer 3: SPECTRAL        → spectral dialect (FFT, STFT, HPSS)
Layer 2: DSP             → dsp dialect (filters, oscillators, effects)
Layer 1: PHYSICAL        → physical dialect (buffers, samples)
```

**All layers implemented in Morphogen.**

---

### RiffStack as Frontend

RiffStack remains as **high-level language layer**:

```
RiffStack YAML/Harmony DSL
         ↓
    Parser/AST
         ↓
Morphogen MLIR IR (compositional/symbolic/dsp)
         ↓
   Morphogen Compiler
         ↓
   Optimized Code (CPU/GPU)
         ↓
      Audio Output
```

**RiffStack responsibilities**:
- YAML syntax and parsing
- Harmony DSL frontend
- User-facing CLI (`riffstack play`, `riffstack validate`)
- Performance tools (live mode, MIDI input)
- Documentation and examples for musicians

**RiffStack does NOT**:
- Implement DSP operations (calls Morphogen)
- Implement MLIR compiler (uses Morphogen)
- Duplicate audio infrastructure

---

### Implementation Plan

**Phase 1: Complete Feature Layer (Morphogen)**
- [ ] Create `morphogen/stdlib/audio_features.py`
- [ ] Implement: `melspectrogram`, `chroma`, `hpss`, `self_similarity_matrix`
- [ ] Create `morphogen/mlir/dialects/features.py`
- [ ] Tests matching `stdlib/audio.py` pattern

**Phase 2: Implement Symbolic Layer (Morphogen)**
- [ ] Create `morphogen/stdlib/music_symbolic.py`
- [ ] Implement: `beat_track`, `estimate_tempo`, `chord_estimate`, `key_estimate`
- [ ] Create `morphogen/mlir/dialects/symbolic.py`
- [ ] Define types: `!symbolic.beat_grid`, `!symbolic.chord_sequence`, `!symbolic.key`
- [ ] Integration with neural models (StableHLO)

**Phase 3: Implement Structural Layer (Morphogen)**
- [ ] Create `morphogen/stdlib/music_structural.py`
- [ ] Implement: `segment_structure`, `label_sections`, `detect_novelty`
- [ ] Create `morphogen/mlir/dialects/structural.py`
- [ ] Define types: `!structural.section`, `!structural.form`

**Phase 4: Implement Compositional Layer (Morphogen)**
- [ ] Create `morphogen/stdlib/music_compositional.py`
- [ ] Implement: `voice_lead`, `harmonize`, `reharmonize`, `generate_progression`
- [ ] Create `morphogen/mlir/dialects/compositional.py`
- [ ] Define types: `!compositional.progression`, `!compositional.voice_leading`
- [ ] Integrate RiffStack's Harmony DSL concepts

**Phase 5: RiffStack Frontend Integration**
- [ ] RiffStack YAML parser generates Morphogen MLIR IR
- [ ] RiffStack CLI calls Morphogen compiler
- [ ] RiffStack examples use Morphogen operations
- [ ] Migration guide for RiffStack users

---

## Consequences

### Positive

1. **Unified Music Platform**
   - One codebase for analysis + synthesis + composition
   - Shared DSP operations
   - Shared MLIR compiler
   - Shared testing infrastructure

2. **Faster Development**
   - Build on 60+ existing operations
   - Reuse MLIR infrastructure
   - Leverage multi-domain architecture
   - Focus on missing pieces (symbolic, compositional)

3. **Better Optimization**
   - Joint compilation across analysis→synthesis pipeline
   - Cross-layer optimizations
   - GPU scheduling across full stack
   - Memory layout optimization

4. **Mathematical Foundations**
   - Reuse group theory, topology, linear algebra work
   - Apply to chord recognition, voice-leading, structure analysis

5. **Clear Separation of Concerns**
   - Morphogen: Platform and implementation
   - RiffStack: User-facing language and tools

6. **Reduced Duplication**
   - One MLIR compiler
   - One audio library
   - One set of tests
   - One documentation set

### Negative

1. **Morphogen Becomes Larger**
   - More domains to maintain
   - Broader scope
   - Mitigation: Clear domain boundaries, modular structure

2. **RiffStack Identity Shift**
   - Changes from platform to frontend
   - May feel like "loss of autonomy"
   - Mitigation: RiffStack owns musician-facing UX and language design

3. **Migration Effort**
   - Need to integrate RiffStack concepts into Morphogen
   - Update RiffStack to consume Morphogen APIs
   - Documentation updates
   - Mitigation: Phased migration, clear guides

4. **Potential Performance Concerns**
   - Morphogen is Python-based (RiffStack vision was compiled)
   - Mitigation: MLIR compilation addresses this. Python is just frontend.

### Neutral

1. **Versioning and Releases**
   - Morphogen and RiffStack may version differently
   - Need coordination for music-related releases

2. **Community and Users**
   - Morphogen users get music capabilities
   - RiffStack users need to understand Morphogen dependency

3. **Documentation**
   - Need cross-project documentation
   - Clear explanation of roles

---

## Alternatives Considered

### Alternative 1: Keep Projects Separate

**Approach**: RiffStack and Morphogen remain independent, share via API/registry.

**Pros**:
- Clear project boundaries
- Independent development velocity
- Separate communities

**Cons**:
- Duplicate MLIR infrastructure
- Duplicate DSP operations
- Coordination overhead
- Missed optimization opportunities (can't fuse across projects)

**Rejected because**: Costs (duplication, coordination) outweigh benefits.

---

### Alternative 2: New "MusicBrain" Project

**Approach**: Create third project for music analysis, separate from both.

**Pros**:
- Focus solely on music understanding
- Clean scope

**Cons**:
- **Three** projects to coordinate
- Still duplicates DSP (STFT, filters needed for analysis)
- Doesn't solve RiffStack/Morphogen overlap
- Adds more complexity

**Rejected because**: Makes situation worse, not better.

---

### Alternative 3: Move Morphogen Audio Into RiffStack

**Approach**: Reverse the decision - consolidate into RiffStack.

**Pros**:
- RiffStack owns music domain

**Cons**:
- RiffStack has minimal implementation (11 ops)
- Morphogen has 60+ operations already working
- Morphogen has multi-domain infrastructure
- RiffStack not designed for multiple domains
- Would need to rebuild Morphogen's domain architecture

**Rejected because**: Moving mature code to minimal project doesn't make sense.

---

## Implementation Notes

### Morphogen Changes

**New files**:
```
morphogen/stdlib/
  audio_features.py       # Mel, chroma, HPSS, SSM
  music_symbolic.py       # Beat tracking, chord recognition
  music_structural.py     # Segmentation, sections
  music_compositional.py  # Voice-leading, progressions

morphogen/mlir/dialects/
  features.py
  symbolic.py
  structural.py
  compositional.py

morphogen/models/
  backbone.py             # Neural backbone (Conformer)
  heads/
    beat.py
    chord.py
    section.py
  stablehlo_bridge.py     # StableHLO integration
```

**Updated files**:
```
morphogen/stdlib/__init__.py  # Register new modules
morphogen/docs/DOMAINS.md     # Document music domain
morphogen/README.md           # Add music capabilities
```

---

### RiffStack Changes

**New integration**:
```
riffstack_core/
  frontend/
    parser.py         # YAML/Harmony DSL parser
    ast.py            # Abstract syntax tree
    codegen.py        # Generate Morphogen MLIR

  morphogen_bridge/
    __init__.py       # Import Morphogen operations
    runtime.py        # Morphogen runtime wrapper
```

**Updated files**:
```
riffstack_core/cli.py           # Call Morphogen compiler
riffstack_core/engine/*.py      # Use Morphogen ops instead of local
requirements.txt                # Add morphogen dependency
README.md                       # Explain Morphogen relationship
```

---

### Migration Path

**For existing RiffStack users**:

1. **Install Morphogen dependency**:
   ```bash
   pip install morphogen
   ```

2. **No YAML syntax changes** - RiffStack YAML remains the same

3. **Performance improvements** - Morphogen compiler may be faster

4. **New capabilities** - Access to 60+ operations, analysis features

**For existing Morphogen users**:

1. **New `audio.*` operations available** in stdlib

2. **New music understanding capabilities**:
   ```python
   import morphogen

   audio = morphogen.audio.load("song.wav")
   chords = morphogen.music_symbolic.chord_estimate(audio)
   structure = morphogen.music_structural.segment_structure(audio)
   ```

3. **Optional**: Use RiffStack YAML frontend for easier composition

---

## Timeline

**Q1 2025**:
- Phase 1: Complete Feature Layer ✅
- Phase 2: Implement Symbolic Layer (partial)

**Q2 2025**:
- Phase 2: Complete Symbolic Layer
- Phase 3: Implement Structural Layer

**Q3 2025**:
- Phase 4: Implement Compositional Layer
- Phase 5: RiffStack frontend integration

**Q4 2025**:
- Documentation, examples, tutorials
- Performance optimization
- Public release

---

## References

- **[MUSIC_SEMANTIC_LAYERS.md](../architecture/MUSIC_SEMANTIC_LAYERS.md)** - Complete layer specification
- **[Mathematical Music Frameworks](../reference/mathematical-music-frameworks.md)** - Theory foundations
- **[ADR-009: Ambient Music Domains](009-ambient-music-generative-domains.md)** - Related generative work
- **RiffStack: [MLIR Architecture](/home/scottsen/src/projects/riffstack/docs/MLIR_ARCHITECTURE.md)** - Original vision
- **RiffStack: [Harmony DSL Vision](/home/scottsen/src/projects/riffstack/docs/HARMONY_DSL_VISION.md)** - Compositional layer

---

## Decision Log

- **2025-12-10**: ADR created (Proposed)
- **Next**: Review and approval
- **Then**: Implementation begins

---

**Status**: Proposed
**Deciders**: Scott Senften
**Approvers**: TBD
**Implementation**: TBD
