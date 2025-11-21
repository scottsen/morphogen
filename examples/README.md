# Kairo Examples

This directory contains comprehensive examples demonstrating Kairo's capabilities across all dialects, versions, and development phases.

---

## 🎯 Quick Navigation

- [Beginner Examples](#tier-1-beginner-examples-) - Start here!
- [Intermediate Examples](#tier-2-intermediate-examples-) - Real simulations
- [Advanced Examples](#advanced-examples-) - Complex multi-domain systems
- [Cross-Domain Geometry](#cross-domain-geometry-examples-) - NEW! Geometry integration showcase
- [Python Integration](#python-integration-examples) - Runtime and tools
- [MLIR Development](#mlir-development-phase-examples) - Compiler internals
- [How to Run](#running-examples)

---

## Tier 1: Beginner Examples 🟢

Perfect for getting started - simple, clear, and immediately rewarding!

### `01_hello_heat.kairo` ⭐ **START HERE**
Your first Kairo program! Watch heat diffuse from a hot center.

```bash
morphogen run examples/01_hello_heat.kairo
```

**Demonstrates:** Field initialization, diffusion operation, visual output
**Visual:** Colorful heat spreading from center (fire palette)
**Runtime:** ~1 second
**Lines:** 58

**Key concepts:**
- `use field, visual` declarations
- `@state` for persistent field
- `flow(dt, steps)` temporal evolution
- `diffuse()` operation
- `colorize()` and `output` visualization

---

### `02_pulsing_circle.kairo`
Hypnotic animation of a circle that smoothly grows and shrinks.

```bash
morphogen run examples/02_pulsing_circle.kairo
```

**Demonstrates:** Lambda expressions, coordinate math, time-based animation
**Visual:** Pulsing circular region
**Runtime:** ~2 seconds
**Lines:** 71

---

### `03_wave_ripples.kairo`
Drop a stone in water - watch realistic wave ripples spread outward.

```bash
morphogen run examples/03_wave_ripples.kairo
```

**Demonstrates:** Wave equation (2 fields), Laplacian operator, physics simulation
**Visual:** Concentric circular waves with interference patterns
**Runtime:** ~3 seconds
**Lines:** 90

---

## Tier 2: Intermediate Examples 🟡

Real simulations demonstrating multiple Kairo features working together.

### `04_random_walk.kairo`
Watch random walkers create beautiful diffusion patterns from simple stochastic rules.

```bash
morphogen run examples/04_random_walk.kairo
```

**Demonstrates:** Random number generation, state accumulation, emergent behavior
**Visual:** Gaussian distribution emerging from Brownian motion (viridis palette)
**Runtime:** ~2 seconds
**Lines:** 80
**Tip:** Experiment with `NUM_WALKERS` to see different diffusion patterns!

---

### `05_gradient_flow.kairo`
Mesmerizing color gradients swirl and mix under a rotating velocity field.

```bash
morphogen run examples/05_gradient_flow.kairo
```

**Demonstrates:** Advection operation, vector fields, multi-channel visualization
**Visual:** Colorful swirling patterns reminiscent of paint mixing in water
**Runtime:** ~3 seconds
**Lines:** 76
**Tip:** Try different `ROTATION_SPEED` values for faster or slower mixing!

---

### `10_heat_equation.kairo`
Complete heat diffusion with hot sources and cold sinks.

```bash
morphogen run examples/10_heat_equation.kairo
```

**Demonstrates:** Boundary conditions, physical units, thermal physics
**Visual:** Temperature gradient from hot (top) to cold (bottom)
**Runtime:** ~5 seconds
**Lines:** 77

---

### `11_gray_scott.kairo` ⭐ **MUST SEE**
Stunning organic patterns from reaction-diffusion chemistry.

```bash
morphogen run examples/11_gray_scott.kairo
```

**Demonstrates:** Coupled PDEs, Laplacian operator, emergent complexity
**Visual:** Mesmerizing spots, stripes, spirals, or maze patterns
**Runtime:** ~10 seconds
**Lines:** 97
**Tip:** Experiment with F and K parameters for different patterns!

**Pattern parameters:**
- **Coral:** F=0.062, K=0.060
- **Spots:** F=0.035, K=0.065
- **Stripes:** F=0.035, K=0.058

---

## Advanced Examples 🔴

Showcasing v0.3.1 language features and complex simulations.

### `v0_3_1_complete_demo.kairo`
Complete showcase of all v0.3.1 language features.

```bash
morphogen run examples/v0_3_1_complete_demo.kairo
```

**Demonstrates:** Functions, lambdas, structs, recursion, flow blocks
**Lines:** 49

---

### `v0_3_1_lambdas_and_flow.kairo`
Lambda expressions and flow block integration.

```bash
morphogen run examples/v0_3_1_lambdas_and_flow.kairo
```

**Demonstrates:** Higher-order functions, closures, temporal evolution
**Lines:** 23

---

### `v0_3_1_recursive_factorial.kairo`
Classic recursion example computing factorial.

```bash
morphogen run examples/v0_3_1_recursive_factorial.kairo
```

**Demonstrates:** Function definitions, recursion, return statements
**Lines:** 25

---

### `v0_3_1_struct_physics.kairo`
Struct-based physics simulation with vector operations.

```bash
morphogen run examples/v0_3_1_struct_physics.kairo
```

**Demonstrates:** Struct definitions, methods, physics integration
**Lines:** 82

---

### `v0_3_1_velocity_calculation.kairo`
Velocity field calculations and gradient computations.

```bash
morphogen run examples/v0_3_1_velocity_calculation.kairo
```

**Demonstrates:** Vector fields, gradient operators
**Lines:** 22

---

## Cross-Domain Geometry Examples 🔷

Showcasing the new geometry domain with cross-domain integration. These examples demonstrate how geometry operations combine with physics, fields, agents, and procedural generation.

### `20_bouncing_spheres.kairo`
Physics simulation with 3D geometry collision detection.

```bash
morphogen run examples/20_bouncing_spheres.kairo
```

**Demonstrates:** geometry + rigidbody integration, 3D primitives (sphere, box3d), bounding boxes, collision detection
**Visual:** Spheres bouncing inside a 3D box with realistic physics
**Runtime:** ~10 seconds
**Lines:** 216

**Cross-domain features:**
- `use geometry, rigidbody, visual` multi-domain imports
- `sphere()` and `box3d()` 3D primitives
- `bounding_box()` for collision bounds
- Physics-driven geometric transformations

---

### `21_voronoi_heat.kairo`
Heat diffusion across Voronoi cells with geometric field sampling.

```bash
morphogen run examples/21_voronoi_heat.kairo
```

**Demonstrates:** geometry + field integration, Voronoi diagrams, spatial field queries, distance calculations
**Visual:** Colorful Voronoi pattern with heat diffusing between cells
**Runtime:** ~8 seconds
**Lines:** 153

**Geometric operations:**
- Voronoi cell generation from seed points
- Euclidean distance calculations
- Spatial field initialization based on geometric regions

---

### `22_delaunay_terrain.kairo`
Procedural terrain generation using Delaunay triangulation and Perlin noise.

```bash
morphogen run examples/22_delaunay_terrain.kairo
```

**Demonstrates:** geometry + noise + field integration, Delaunay triangulation, mesh generation, noise sampling
**Visual:** Animated low-poly terrain with triangular facets colored by height
**Runtime:** ~10 seconds
**Lines:** 142

**Advanced features:**
- Delaunay triangulation for terrain mesh
- Perlin noise for natural height variation
- Animated noise for evolving landscapes
- Height-based coloring (water → land → mountains)

---

### `23_geometry_patrol.kairo`
Agent navigation with convex hull patrol routes and obstacle avoidance.

```bash
morphogen run examples/23_geometry_patrol.kairo
```

**Demonstrates:** geometry + agent + field integration, convex hull, spatial queries, collision detection
**Visual:** Agents patrolling a polygon with influence fields and obstacle avoidance
**Runtime:** ~15 seconds
**Lines:** 254

**Navigation features:**
- Convex hull patrol route generation
- Point-in-polygon containment testing
- Distance to line segment calculations
- Circle-based obstacle collision detection
- Agent influence field visualization

---

### `24_mesh_morphing.kairo`
Advanced 3D mesh operations with subdivision and shape interpolation.

```bash
morphogen run examples/24_mesh_morphing.kairo
```

**Demonstrates:** 3D mesh operations, subdivision surfaces, collision meshes, volume calculations, transformations
**Visual:** 3D shape morphing from box to sphere with smooth rotation
**Runtime:** ~20 seconds
**Lines:** 143

**Advanced mesh operations:**
- `collision_mesh()` generation for physics
- `volume()` calculations
- Mesh subdivision for smooth surfaces
- 3D rotation transformations
- Shape interpolation and morphing

---

### `25_convex_hull_art.kairo`
Real-time convex hull computation with dynamic point clouds.

```bash
morphogen run examples/25_convex_hull_art.kairo
```

**Demonstrates:** convex hull algorithm, dynamic geometry, point containment, coordinate transformations
**Visual:** Mesmerizing animation of orbiting points with their convex hull boundary
**Runtime:** ~15 seconds
**Lines:** 298

**Computational geometry:**
- `convex_hull()` real-time computation
- Cartesian ↔ polar coordinate transformations
- Point-in-polygon ray casting algorithm
- Distance to polygon edge calculations
- Gradient fill based on geometric containment

---

## Python Integration Examples

Interactive Python examples demonstrating runtime integration and advanced features.

### Interactive Simulations

#### `interactive_diffusion.py`
Real-time heat diffusion with live Pygame visualization.

```bash
python examples/interactive_diffusion.py
```

**Features:** Real-time heat spreading, interactive controls, fire palette
**Controls:** Mouse to add heat, ESC to quit

---

#### `smoke_simulation.py`
Full Navier-Stokes fluid simulation with velocity and density fields.

```bash
python examples/smoke_simulation.py
```

**Features:** Incompressible flow, advection-diffusion-projection, swirling smoke
**Controls:** SPACE=pause, ←→=adjust speed, Q/ESC=quit

---

#### `reaction_diffusion.py`
Python implementation of Gray-Scott reaction-diffusion.

```bash
python examples/reaction_diffusion.py
```

**Features:** Coral/maze patterns, self-organizing structures, stunning visuals
**Use case:** Comparing Python vs Kairo implementations

---

### Audio Examples (v0.5.0+)

#### `audio_io_demo.py` ⭐
Comprehensive audio synthesis and I/O demonstrations.

```bash
# Requires audio I/O dependencies
pip install -e ".[io]"
python examples/audio_io_demo.py
```

**Demonstrates:**
- Real-time playback with `audio.play()`
- WAV/FLAC export with `audio.save()`
- Audio loading with `audio.load()`
- Microphone recording with `audio.record()`
- Oscillators (sine, saw, square, triangle)
- Filters (lowpass, highpass, bandpass)
- Effects (reverb, delay, chorus, flanger)
- Physical modeling (Karplus-Strong strings)
- Round-trip file I/O accuracy

**Lines:** 243
**Status:** ✅ Production-ready

---

#### `audio_dsp_spectral.py` ⭐
Audio DSP and spectral analysis demonstrations.

```bash
python examples/audio_dsp_spectral.py
```

**Demonstrates:**
- FFT/STFT transforms
- Spectral analysis (centroid, rolloff, flux)
- Frequency-domain processing
- Spectral gate noise reduction
- Convolution-based effects
- Buffer operations (slice, concat, resample)

**Features:** Complete audio DSP pipeline
**Status:** ✅ Production-ready

---

### Visual Examples (v0.6.0+)

#### `visual_composition_demo.py` ⭐
Multi-layer visual composition and video export.

```bash
# Requires video export dependencies
pip install -e ".[io]"
python examples/visual_composition_demo.py
```

**Demonstrates:**
- `visual.agents()` - Render particles with color-by-property
- `visual.composite()` - Multi-layer blending (add, multiply, overlay)
- `visual.video()` - MP4/GIF export
- Frame generators for memory-efficient animations
- Agent rendering styles (points, circles)
- Per-layer opacity control

**Lines:** 326
**Status:** ✅ Production-ready

---

### Utility Examples

#### `mvp_simple_test.py`
Simple Python test of Kairo runtime fundamentals.

```bash
python examples/mvp_simple_test.py
```

**Purpose:** Understanding Python-Kairo integration
**Use case:** Runtime testing and debugging

---

#### `generate_portfolio_outputs.py`
Generate portfolio images and animations for documentation.

```bash
python examples/generate_portfolio_outputs.py
```

**Purpose:** Creating documentation visuals
**Use case:** Marketing and showcase materials

---

## MLIR Development (Phase Examples)

Examples demonstrating MLIR compilation infrastructure (v0.7.0+).

> **Note:** Requires MLIR Python bindings. Install with:
> `pip install mlir -f https://github.com/makslevental/mlir-wheels/releases/expanded_assets/latest`

### `mlir_poc.py`
Initial MLIR Python bindings proof-of-concept.

```bash
python examples/mlir_poc.py
```

**Purpose:** Foundation for v0.7.0 MLIR work
**Demonstrates:** Basic MLIR integration without Kairo dialects
**Status:** Historical reference

---

### `phase2_field_operations.py`
Field operations compiled to MLIR (Phase 2 - Completed).

```bash
python examples/phase2_field_operations.py
```

**Demonstrates:**
- Field dialect operations (create, gradient, Laplacian, diffuse)
- Field-to-SCF lowering pass
- Pattern-based lowering infrastructure
- Stencil operations with boundary handling
- MLIR IR visualization (before/after lowering)
- Compiler integration

**5 complete examples:**
1. Field creation (allocation with fill value)
2. Gradient computation (central difference)
3. Laplacian computation (5-point stencil)
4. Diffusion solver (Jacobi iteration)
5. Combined workflow (create → gradient → Laplacian → diffuse)

**Status:** ✅ Complete

---

### `phase3_temporal_execution.py`
Temporal flow blocks and state management (Phase 3 - Completed).

```bash
python examples/phase3_temporal_execution.py
```

**Demonstrates:**
- Flow operations (create, run)
- State management (create, update, query)
- Temporal-to-SCF lowering
- State evolution over timesteps
- SSA-compliant transformations
- Integration with field operations

**Concepts:**
- `flow.create` → Flow metadata
- `flow.run` → `scf.for` loops with state
- `state.create/update/query` → `memref` operations

**Status:** ✅ Complete

---

### `phase4_agent_operations.py` ⭐
Agent simulations compiled to MLIR (Phase 4 - Completed).

```bash
python examples/phase4_agent_operations.py
```

**Demonstrates:**
- Agent spawn, update, query, behavior operations
- Agent-to-SCF lowering
- Multi-agent simulations (tested up to 10K+ agents)
- Property-based agent data structures
- Integration with field and temporal dialects

**8 complete examples:**
1. Basic agent spawning (allocation and initialization)
2. Agent movement (velocity-based updates)
3. Multi-agent behaviors (flocking rules)
4. Property updates (dynamic modifications)
5. Bounce behavior (boundary collision handling)
6. Agent-field integration (agents + spatial fields)
7. Temporal agent evolution (agents over timesteps)
8. Large-scale simulation (10,000+ agents, scalability test)

**Lines:** 547
**Status:** ✅ Complete

---

### `phase5_audio_operations.py` ⭐
Audio synthesis compiled to MLIR (Phase 5 - Completed).

```bash
python examples/phase5_audio_operations.py
```

**Demonstrates:**
- Audio buffer creation and management
- Oscillators (sine, square, saw, triangle)
- ADSR envelopes (attack, decay, sustain, release)
- Filters (lowpass, highpass, bandpass)
- Multi-signal mixing (stereo synthesis)
- Audio-to-SCF lowering with DSP loops

**8 complete examples:**
1. Basic oscillator (440 Hz sine wave)
2. ADSR envelope application
3. Lowpass filter frequency sweep
4. Chord mixing (C major triad, 3 oscillators)
5. Complete synth patch (OSC → ENV → FILTER → MIX)
6. Audio effects chain (reverb, delay, chorus)
7. Multi-voice synthesis (polyphony demonstration)
8. Bass synthesis with sub-oscillator layering

**Lines:** 521
**Status:** ✅ Complete

---

### `phase6_jit_aot_compilation.py` ⭐
JIT/AOT compilation with LLVM backend (Phase 6 - Completed).

```bash
python examples/phase6_jit_aot_compilation.py
```

**Demonstrates:**
- JIT compilation to native code with execution engine
- AOT compilation to binaries (shared libs, executables)
- LLVM optimization levels (O0-O3)
- Compilation caching (in-memory and persistent disk)
- ExecutionEngine unified API
- Performance benchmarking

**8 complete examples:**
1. Basic JIT compilation (simple function)
2. JIT with persistent disk caching (cache hit demonstration)
3. AOT to shared library (.so/.dylib/.dll)
4. AOT to executable binary (native ELF/PE)
5. ExecutionEngine API (context manager, unified interface)
6. Field operations JIT pipeline (full compilation chain)
7. Audio synthesis with JIT (real-time audio generation)
8. Performance benchmarking (compare O0/O1/O2/O3 optimization)

**Lines:** 521
**Status:** ✅ Complete

---

## Running Examples

### Kairo Language Files (`.kairo`)

```bash
# Basic execution
morphogen run examples/01_hello_heat.kairo

# View available options
morphogen run --help
```

### Python Files (`.py`)

```bash
# Run directly
python examples/reaction_diffusion.py

# Or make executable
chmod +x examples/reaction_diffusion.py
./examples/reaction_diffusion.py
```

### Dependencies

Some examples require optional dependencies:

```bash
# Core Kairo (required)
pip install -e .

# Audio I/O (for audio_io_demo.py, audio_dsp_spectral.py)
pip install -e ".[io]"

# MLIR (for phase*.py examples)
pip install mlir -f https://github.com/makslevental/mlir-wheels/releases/expanded_assets/latest
```

---

## Learning Path

### Path 1: Absolute Beginner → Intermediate

**Goal:** Learn Kairo language and core concepts

1. `01_hello_heat.kairo` - Understand basic syntax
2. `02_pulsing_circle.kairo` - Temporal animation
3. `10_heat_equation.kairo` - PDE simulation
4. `04_random_walk.kairo` - Agent basics
5. `11_gray_scott.kairo` - Complex patterns
6. `audio_io_demo.py` - Audio synthesis
7. `visual_composition_demo.py` - Multi-layer visuals

**Expected time:** 2-4 hours
**Outcome:** Master all four dialects (field, agent, audio, visual)

---

### Path 2: Advanced Developer → MLIR Compiler

**Goal:** Understand MLIR integration and compilation

1. `phase2_field_operations.py` - Custom MLIR dialects
2. `phase3_temporal_execution.py` - Temporal dialect and state
3. `phase4_agent_operations.py` - Agent dialect and scalability
4. `phase5_audio_operations.py` - Audio dialect and DSP
5. `phase6_jit_aot_compilation.py` - LLVM backend and optimization

**Expected time:** 4-8 hours
**Outcome:** Deep understanding of Kairo's compilation pipeline

---

## Example Categories

### By Dialect

- **Field:** `01`, `02`, `03`, `05`, `10`, `11`, `phase2`, `smoke_simulation.py`
- **Agent:** `04`, `phase4`, `v0_3_1_struct_physics.kairo`
- **Audio:** `audio_io_demo.py`, `audio_dsp_spectral.py`, `phase5`
- **Visual:** `visual_composition_demo.py`, all `.kairo` files with `output`

### By Complexity

- **Beginner (🟢):** `01`, `02`, `03` (< 100 lines, single dialect)
- **Intermediate (🟡):** `04`, `05`, `10`, `11` (100-150 lines, multiple features)
- **Advanced (🔴):** `v0_3_1_*.kairo`, all `phase*.py` (200+ lines, compiler internals)

### By Version

- **v0.3.1 Language:** `01`-`11`, `v0_3_1_*.kairo`
- **v0.4.0 Agents:** `04`, `phase4_agent_operations.py`
- **v0.5.0 Audio:** `audio_io_demo.py`, `phase5_audio_operations.py`
- **v0.6.0 I/O:** `audio_io_demo.py`, `visual_composition_demo.py`
- **v0.7.0 MLIR:** `phase2`-`phase6` examples

---

## Example Statistics

- **Total files:** 26 (12 Kairo + 14 Python)
- **Total Kairo lines:** 750 lines
- **Beginner examples:** 3 files
- **Intermediate examples:** 4 files
- **Advanced examples:** 5 files
- **Python integration:** 14 files
- **MLIR phases:** 6 files (POC + phases 2-6)
- **Audio examples:** 2 files (243 + variable lines)
- **Visual examples:** 1 file (326 lines)

---

## Troubleshooting

### Example won't run

```bash
# Check installation
kairo --version

# Reinstall if needed
pip install -e .
```

### Missing dependencies

```bash
# Install all optional dependencies
pip install -e ".[io]"

# Or install individually
pip install sounddevice soundfile scipy imageio imageio-ffmpeg
```

### MLIR examples fail

```bash
# Install MLIR Python bindings
pip install mlir -f https://github.com/makslevental/mlir-wheels/releases/expanded_assets/latest
```

If MLIR is unavailable, phase examples will gracefully skip or fall back to legacy mode.

### Import errors

```bash
# Ensure you're in the Kairo directory
cd /path/to/kairo

# Reinstall in development mode
pip install -e .
```

---

## Contributing Examples

Want to add your own example? Follow these guidelines:

### Kairo Language Examples (`.kairo`)

1. **Naming:** Use descriptive verb-noun format (`heat_diffusion.kairo`)
2. **Comments:** Explain key concepts inline
3. **Length:** Keep beginner examples < 100 lines
4. **Style:** Use consistent formatting (4-space indents)
5. **Testing:** Verify it runs successfully

### Python Examples (`.py`)

1. **Docstrings:** Add module-level docstring explaining purpose
2. **Dependencies:** List required packages in comments
3. **Focus:** Demonstrate one clear concept per example
4. **Documentation:** Add entry to this README with description

### Documentation

For each new example, add to this README:
- Category (beginner/intermediate/advanced)
- Dialect(s) used
- What it demonstrates
- Key concepts
- Runtime estimate
- Line count

---

## Additional Resources

- **[docs/GETTING_STARTED.md](../docs/GETTING_STARTED.md)** - Complete beginner's guide
- **[SPECIFICATION.md](../SPECIFICATION.md)** - Full language reference
- **[ARCHITECTURE.md](../ARCHITECTURE.md)** - System architecture
- **[docs/v0.7.0_DESIGN.md](../docs/v0.7.0_DESIGN.md)** - MLIR integration roadmap
- **[AUDIO_SPECIFICATION.md](../AUDIO_SPECIFICATION.md)** - Audio dialect details

---

**Happy coding!** 🎨🎵🔬

Explore, experiment, and create amazing simulations, sounds, and visualizations with Kairo!

---

**Last Updated:** 2025-11-15
**Version:** v0.6.0 (stable) / v0.7.0-dev (development)
