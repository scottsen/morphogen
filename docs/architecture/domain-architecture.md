# Morphogen Domain Architecture

**Version:** 1.0
**Status:** Vision Document
**Last Updated:** 2025-11-15

---

## Overview

This document presents a comprehensive, forward-looking view of the domains and layers Morphogen will eventually encompass. These domains emerge naturally from building a **deterministic, multi-domain semantic compute kernel** designed for audio, physics, graphics, AI, simulation, and analytics.

This is not aspirational fluff — these are the domains that consistently appear in successful multi-modal compute systems. Each domain is justified by real computational needs and integrated into Morphogen's unified type system, scheduler, and MLIR compilation pipeline.

### Document Purpose

- **Current Reference**: Understand what domains exist today
- **Planning Guide**: Inform roadmap prioritization
- **Architecture Vision**: Ensure coherent integration across domains
- **Engineering Resource**: Define operator requirements and dependencies

### Related Documentation

This document is part of a comprehensive domain architecture learning system:

- **[ADR-002: Cross-Domain Architectural Patterns](../adr/002-cross-domain-architectural-patterns.md)** — Battle-tested patterns from TiaCAD, RiffStack, and Strudel (reference systems, auto-anchors, operator registries, passes)
- **[Domain Implementation Guide](../guides/domain-implementation.md)** — Step-by-step guide for implementing new domains (checklists, templates, best practices)
- **[Operator Registry Expansion](../reference/operator-registry-expansion.md)** — Detailed catalog of 7 priority domains with complete operator specifications (Audio, Physics, Geometry, Finance, Graphics, Neural, Pattern)

**For domain implementers**: Start with ADR-002 for architectural principles, then follow the Domain Implementation Guide for practical steps.

---

## Domain Classification

Domains are organized into three tiers based on urgency and system maturity:

1. **Core Domains** — Essential for audio, fields, physics, graphics, or simulation. Must have.
2. **Next-Wave Domains** — Naturally emerge from a multirate, GPU/CPU-pluggable, graph-IR-based kernel. Highly likely.
3. **Advanced Domains** — Future expansion for specialized use cases. May add later.

---

## 1. Core Domains (MUST HAVE)

These domains form the bare minimum for a universal transform/simulation kernel. Several are already partially defined in `../specifications/mlir-dialects.md` and operational in v0.7.0.

---

### 1.1 Transform Dialect

**Purpose**: Domain transforms between time/frequency, space/k-space, and other spectral representations.

**Why Essential**: Audio processing, signal analysis, PDE solving, and compression all require fast, accurate transforms.

**Status**: ✅ Partially implemented (FFT, STFT, IFFT in morphogen.transform dialect)

**Operators**:
- `fft` / `ifft` — Fast Fourier Transform (1D)
- `fft2d` / `ifft2d` — 2D FFT (space → k-space)
- `stft` / `istft` — Short-Time Fourier Transform
- `dct` / `idct` — Discrete Cosine Transform
- `wavelet` — Wavelet transforms (Haar, Daubechies, etc.)
- `mel` — Mel-frequency transforms
- `cepstral` — Cepstral analysis
- `reparam` — Reparameterization (e.g., exponential → linear frequency)

**Dependencies**: Linear algebra, windowing functions

**References**: `../specifications/transform.md`, `../specifications/mlir-dialects.md`

---

### 1.2 Stochastic Dialect

**Purpose**: Random number generation, distributions, stochastic processes, Monte Carlo simulation.

**Why Essential**: Agent mutation, noise generation, probabilistic simulation, and procedural content all require deterministic, high-quality randomness.

**Status**: ⚙️ In progress (Philox RNG implemented, distribution ops planned)

**Operators**:
- `rng.init` — Initialize RNG state with seed
- `rng.uniform` — Uniform distribution [0, 1)
- `rng.normal` — Gaussian distribution (mean, stddev)
- `rng.exponential` — Exponential distribution (rate)
- `rng.poisson` — Poisson distribution (lambda)
- `monte_carlo.integrate` — Monte Carlo integration
- `sde.step` — Stochastic differential equation step (Euler-Maruyama, Milstein)

**Dependencies**: None (foundational)

**Determinism**: Strict (Philox 4×32-10 with hash-based seeding)

---

### 1.3 Fields / Grids Dialect

**Purpose**: Operations on scalar/vector/tensor fields, stencils, PDE operators, boundary conditions.

**Why Essential**: Fluid simulation, reaction-diffusion, heat transfer, and electromagnetic fields all operate on spatial grids.

**Status**: ✅ Partially implemented (morphogen.field dialect with stencil, advect, reduce)

**Operators**:
- `field.create` — Allocate field with shape, spacing, initial value
- `field.stencil` — Apply stencil (Laplacian, gradient, divergence, custom)
- `field.advect` — Advect by velocity field (semi-Lagrangian, MacCormack, BFECC)
- `field.diffuse` — Diffusion step (Jacobi, Gauss-Seidel, CG)
- `field.project` — Pressure projection (Jacobi, multigrid, PCG)
- `field.reduce` — Reduce to scalar (sum, max, min, mean)
- `field.combine` — Element-wise combination
- `field.mask` — Apply spatial mask
- `boundary.apply` — Apply boundary conditions (periodic, clamp, reflect, noSlip)

**Dependencies**: Sparse linear algebra (for solvers), stencil patterns

**References**: `../specifications/mlir-dialects.md` (morphogen.field)

---

### 1.4 Integrators Dialect

**Purpose**: Numerical integration of ordinary differential equations (ODEs) and stochastic differential equations (SDEs).

**Why Essential**: Physics simulation, agent dynamics, and control systems all require stable, accurate time-stepping.

**Status**: 🔲 Planned (currently ad-hoc in agent operations)

**Operators**:
- `integrator.euler` — Forward Euler (1st order)
- `integrator.rk2` — Runge-Kutta 2nd order (midpoint)
- `integrator.rk4` — Runge-Kutta 4th order (classic)
- `integrator.verlet` — Velocity Verlet (symplectic, for physics)
- `integrator.leapfrog` — Leapfrog integration
- `integrator.symplectic` — Symplectic split-operator methods
- `integrator.adaptive` — Adaptive step-size (Dormand-Prince, Fehlberg)

**Dependencies**: Stochastic (for SDEs)

**Determinism**: Strict (fixed timestep), Reproducible (adaptive timestep)

---

### 1.5 Audio DSP Dialect

**Purpose**: Real-time audio synthesis, filtering, effects, mixing.

**Why Essential**: Morphogen began as a creative audio kernel and must excel at low-latency, sample-accurate audio processing.

**Status**: ✅ Partially implemented (oscillators, filters, envelopes via morphogen.stream)

**Operators**:
- `osc.sine` / `osc.triangle` / `osc.sawtooth` / `osc.square` — Oscillators
- `filter.lowpass` / `filter.highpass` / `filter.bandpass` / `filter.notch` — Filters
- `envelope.adsr` — Attack-Decay-Sustain-Release envelope
- `mix` — Sum multiple streams
- `amplify` — Multiply by gain
- `delay` — Delay line (circular buffer)
- `reverb` — Reverb effects (Freeverb, Schroeder, convolution)
- `compress` — Dynamic range compression
- `distortion` — Waveshaping, clipping

**Dependencies**: Transform (for spectral effects)

**References**: `../specifications/mlir-dialects.md` (morphogen.stream)

---

### 1.6 Particles / Agents Dialect

**Purpose**: Particle-to-field transfers, field-to-particle forces, N-body dynamics, agent-based simulation.

**Why Essential**: Particle systems, swarm behavior, crowd simulation, and molecular dynamics all require agent operations.

**Status**: ⚙️ In progress (agent stdlib implemented, MLIR lowering planned)

**Operators**:
- `agent.spawn` — Create new agents
- `agent.remove` — Remove agents by predicate
- `agent.force_sum` — Calculate forces (brute force, grid, Barnes-Hut)
- `agent.integrate` — Update positions/velocities
- `agent.mutate` — Apply stochastic mutations
- `agent.to_field` — Deposit agent properties to field (particle-in-cell)
- `agent.from_field` — Sample field values at agent positions
- `agent.sort` — Sort by spatial locality (Morton order)

**Dependencies**: Fields (for coupling), Stochastic (for mutations), Integrators

**Determinism**: Strict (with stable ID ordering and deterministic force methods)

---

### 1.7 Visual / Scene Dialect

**Purpose**: Scene graph + animation system for mathematical visualization, fractal iteration, palette mapping, geometric warping, 2D/3D field rendering. Provides 3Blue1Brown-style composable math visualization capabilities.

**Why Essential**: Explanatory graphics, mathematical animations, creative visuals, procedural art, and scientific visualization all need efficient rendering with composable scene management.

**Status**: ⚙️ In progress (visual stdlib with colorization, rendering primitives; scene graph + timeline architecture designed)

**Core Components**:
- **SceneDomain** — Scene graph, MObjects, camera system
- **TimelineDomain** — Keyframe animation, easing, composition
- **Rendering** — Fractal iteration, palette mapping, compositing

**Key Operators** (Rendering Primitives):
- `fractal.mandelbrot` — Mandelbrot set iteration
- `fractal.julia` — Julia set iteration
- `fractal.ifs` — Iterated function system
- `palette.apply` — Map scalar field to color palette
- `warp.displace` — Geometric displacement by vector field
- `render.points` — Render agent positions as point sprites
- `render.layers` — Composite multiple layers with blend modes
- `filter.blur` / `filter.sharpen` — Post-processing filters

**Key Operators** (Scene Graph):
- `scene.create` / `scene.add` / `scene.remove` — Scene management
- `geo.curve` / `geo.surface` / `geo.vector_field` — Geometry creation
- `anim.move` / `anim.rotate` / `anim.fade` — Basic animations
- `anim.morph` / `anim.write_equation` — Math-specific animations
- `camera.pan` / `camera.zoom` / `camera.orbit` — Camera control

**Dependencies**: Fields (for scalar/vector data), Image/Vision (for filtering), Palette, Noise, Video

**Documentation**: See [domains/visual-scene-domain.md](domains/visual-scene-domain.md) for comprehensive architecture and [domains/visual-domain-quickref.md](domains/visual-domain-quickref.md) for quick reference

---

### 1.8 Circuit / Electrical Engineering Dialect

**Purpose**: Circuit simulation (SPICE-like), analog modeling, PCB parasitic extraction, multi-physics coupling (electrical + thermal + EM).

**Why Essential**: EE/circuit modeling is one of the most natural domains for Morphogen because it combines differential equations, spatial geometry (PCB traces), physics (EM fields), audio/analog modeling, discrete-time simulation, nonlinear systems, constraints solving, and multi-domain coupling. **No existing tool unifies circuit simulation + PCB layout + audio modeling + EM fields in one framework.**

**Status**: 🔲 Planned (ADR-003, ../specifications/circuit.md)

**Key Innovation**: Circuits as **typed operator graphs** with **reference-based composition** (no manual node numbering), seamless PCB geometry integration, and cross-domain flows (Circuit → Audio, Geometry → Circuit, Circuit → Physics).

**Operators**:

**Layer 1: Atomic Components**
- `resistor(R)` / `capacitor(C)` / `inductor(L)` — Linear passive components
- `voltage_source(V)` / `current_source(I)` — Independent sources
- `diode(Is, n)` / `bjt_npn(beta)` / `mosfet_n(Vth)` — Nonlinear semiconductors
- `op_amp(model)` / `comparator(Vref)` — Integrated components

**Layer 2: Composite Blocks**
- `voltage_divider(R1, R2)` / `rc_filter(R, C, type)` — Passive networks
- `rlc_resonator(R, L, C)` / `pi_matching(Zin, Zout)` — RF networks

**Layer 3: Circuit Constructs**
- `opamp_inverting_amp(gain, Rin)` / `sallen_key_filter(fc, Q)` — Analog circuits
- `triode_stage(tube_model, bias)` / `pentode_output_stage(tube_model)` — Tube amps
- `buck_converter(Vin, Vout, Iout, fsw)` / `ldo_regulator(Vin, Vout)` — Power electronics

**Layer 4: Circuit Presets**
- `guitar_pedal_overdrive(drive, tone, level)` — Guitar pedals (Tube Screamer, etc.)
- `tube_amp_preamp(channels, gain_stages)` — Tube amplifier preamps
- `synth_vcf_moog(cutoff, resonance)` — Synthesizer filters (Moog ladder, Roland)

**Analysis Methods**:
- `dc_operating_point(circuit)` — DC steady-state solution (Modified Nodal Analysis)
- `ac_sweep(circuit, freq_start, freq_end)` — Frequency response (Bode plot)
- `transient(circuit, duration, timestep)` — Time-domain simulation (Euler, RK4, trapezoidal)
- `harmonic_balance(circuit, fundamental_freq)` — Periodic steady-state (for oscillators, RF)
- `noise_analysis(circuit, freq_range)` — Noise spectral density
- `sensitivity_analysis(circuit, output, params)` — Parameter sensitivity (∂V/∂R)

**Circuit Reference System** (following ADR-002 pattern):
- **`CircuitRef`** — Unified reference to nodes/components/nets/ports
- **Auto-anchors**: `.port["p"]`, `.port["n"]`, `.voltage`, `.current`, `.power`, `.impedance`
- **Type-safe connections**: Can't connect incompatible ports (voltage to current source, etc.)
- **No manual node numbering**: Auto-generated from topology

**Multi-Domain Integration**:

*Circuit ↔ Geometry (PCB Layout):*
- PCB trace parasitic extraction (inductance, capacitance, resistance)
- FastHenry/FastCap algorithms or full EM solve (FDTD)
- Automatic coupling: geometry → circuit parasitics

*Circuit ↔ Audio (Analog Modeling):*
- Guitar pedal simulation (asymmetric clipping, harmonic generation)
- Tube amplifier modeling (nonlinear triode/pentode characteristics)
- Audio input → circuit simulation → audio output (WAV export)
- Oversampling for harmonic accuracy (192kHz+)

*Circuit ↔ Physics (Thermal Coupling):*
- Power dissipation in components → heat source
- Thermal model (heatsink, PCB) → temperature distribution
- Temperature feedback → circuit parameters (beta, Vbe, resistance)
- Iterative coupling until convergence

*Circuit ↔ Pattern (Modulation):*
- PWM pattern generation for switch-mode power supplies
- Audio-rate modulation for synthesis
- Control signal generation

*Circuit ↔ ML (Optimization):*
- Gradient-based circuit optimization (component values)
- Differentiable circuit simulation (JAX-like)
- Objective: minimize loss(frequency_response, target)

**Solver Architecture**:
- **Direct solvers**: LU, QR, Cholesky (dense, < 100 nodes)
- **Iterative solvers**: CG, GMRES, BiCGSTAB (sparse, > 1000 nodes)
- **Nonlinear solvers**: Newton-Raphson, damped Newton (diodes, transistors)
- **Time integrators**: Euler, backward Euler, trapezoidal, RK4
- **Auto-selection**: Based on circuit size, stiffness, linearity
- **GPU acceleration**: CUDA/HIP for large circuits

**Circuit Domain Passes**:
- **Validation**: Kirchhoff's laws (KCL, KVL), component value ranges
- **Optimization**: Series/parallel reduction, Thevenin/Norton equivalents, symbolic simplification
- **Lowering**: Circuit netlist → state-space ODE → MLIR (scf + linalg)
- **Parasitic extraction**: Geometry → inductance/capacitance/resistance

**Determinism Profile**:
- DC/AC analysis: **Strict** (same netlist → same results, bit-exact)
- Transient (fixed timestep): **Strict**
- Transient (adaptive timestep): **Reproducible** (timestep adapts, but deterministic given tolerance)
- Newton-Raphson convergence: **Reproducible** (initial guess affects path)

**Unique Capabilities (vs. SPICE, KiCad, LTspice)**:
1. **Unified circuit + PCB + audio + EM** — No other tool does this
2. **Declarative netlist** (YAML, reference-based) — Not text-based manual node numbering
3. **Type + unit safety** — Prevents mixing Ω + F, enforces dimensional analysis
4. **Cross-domain flows** — Circuit → Audio, Geometry → Circuit, Circuit → Thermal
5. **ML optimization** — Gradient descent on component values (differentiable programming)
6. **GPU acceleration** — For large circuits (> 1000 nodes)
7. **Multi-physics** — Electrical + thermal + EM + mechanical in one framework

**Example Applications**:
- Guitar pedal design (analog modeling + audio export)
- Tube amplifier design (nonlinear modeling + harmonic analysis)
- PCB signal integrity (trace parasitics + eye diagram)
- Power supply design (buck converter + thermal analysis)
- RF circuit design (S-parameters + matching networks)
- Audio synthesizer design (VCF + VCO + ADSR)
- Mixed-signal circuits (analog + digital co-simulation)

**Dependencies**:
- **Fields** (for EM field solvers, PDE coupling)
- **Integrators** (for ODE/SDE time-stepping)
- **Stochastic** (for Monte Carlo, noise analysis)
- **Geometry** (for PCB layout, parasitic extraction)
- **Audio** (for analog modeling, audio I/O)
- **Physics** (for thermal coupling, multi-physics)
- **Sparse Linear Algebra** (for large circuit matrices)
- **Autodiff** (for ML-based optimization)

**References**:
- `ADR-003` — Circuit Modeling Domain design decision
- `../specifications/circuit.md` — Complete circuit domain specification
- `examples/circuit/` — Example circuits (RC filter, op-amp, guitar pedal, PCB trace)
- ngspice, LTspice, FastHenry, FastCap (reference implementations)

**Why This is a Perfect Fit for Morphogen**:

Circuits are fundamentally **typed operator graphs** (resistor → capacitor → op-amp → speaker), which matches Morphogen's operator registry architecture exactly. Circuit simulation requires **multi-domain physics** (ODEs, linear systems, nonlinear solvers, EM fields, thermal), which Morphogen's unified kernel provides. PCB layout is **geometry with electrical constraints**, perfectly suited for Morphogen's geometry + circuit integration. Analog audio modeling is **circuit simulation + DSP**, which Morphogen unifies seamlessly.

**No existing tool can compete** with Morphogen's unified circuit + PCB + audio + EM + thermal architecture. This is Morphogen's **killer app** for the EE and analog audio communities.

---

### 1.9 Video & Audio Encoding Dialect

**Purpose**: Video encoding, audio/video filtering, sync correction, transcoding, and ffmpeg-style multimedia pipelines.

**Why Essential**: Video processing is fundamentally stream-based, operator-based, filter-based, parameterizable, batchable, GPU-accelerable, and graph-representable. **This is literally Morphogen's native shape.** ffmpeg already behaves like a domain-specific operator graph with streams, filters, and codecs — video fits Morphogen as naturally as audio, perhaps **more naturally** than any other domain.

**Status**: 🔲 Planned (../specifications/video-audio-encoding.md)

**Key Insight**: Video processing pipelines are **typed operator DAGs** with **temporal constraints** (sync, frame rate, time alignment).

```
Morphogen = operator DAG on structured data
Video = operator DAG on AV streams
```

**Subdomains**:

**VideoDomain** — Structural operations on video streams
- Decoding/encoding: `video.decode()`, `video.encode()`
- Transformation: `video.scale()`, `video.crop()`, `video.fps()`, `video.rotate()`
- Composition: `video.concat()`, `video.overlay()`, `video.blend()`
- Conversion: `video.to_audio()`, `video.from_frames()`, `video.color_convert()`

**AudioFilterDomain** — Audio processing for multimedia
- Loudness: `audio.normalize()`, `audio.loudnorm()` (EBU R128), `audio.measure_loudness()`
- Dynamics: `audio.compress()`, `audio.limiter()`, `audio.gate()`
- Timing: `audio.delay()`, `audio.trim()`, `audio.fade_in()`, `audio.fade_out()`
- EQ: `audio.equalize()`, `audio.bass_boost()`, `audio.treble_boost()`

**FilterDomain** — Visual filters (ffmpeg `-vf` equivalents)
- Spatial: `filter.blur()`, `filter.sharpen()`, `filter.unsharp()`, `filter.denoise()`
- Color: `filter.brightness()`, `filter.contrast()`, `filter.saturation()`, `filter.colorgrade()`
- Temporal: `filter.time_blend()`, `filter.deflicker()`, `filter.stabilize()`
- Quality: `filter.deband()`, `filter.deinterlace()`, `filter.upscale()`

**CodecDomain** — Codec configuration as typed operators
- Video: `codec.h264()`, `codec.h265()`, `codec.av1()`, `codec.prores()`
- Image: `codec.jpeg()`, `codec.png()`, `codec.webp()`, `codec.jpegxl()`
- Audio: `codec.aac()`, `codec.opus()`, `codec.flac()`
- GPU: `codec.h264_nvenc()`, `codec.h265_nvenc()`, `codec.h264_qsv()`

**SyncDomain** — Audio/video synchronization (Morphogen's sweet spot)
- Offset detection: `sync.detect_constant_offset()`, `sync.detect_drift()`
- Correction: `sync.apply_offset()`, `sync.timewarp()`, `sync.resample_with_drift_compensation()`
- Event alignment: `vision.detect_flash()`, `audio.detect_clap()`, `sync.align_events()`
- Lip-sync: `vision.detect_mouth_open()`, `audio.envelope()`, `sync.align_signals()`

**BatchDomain** — Parallel batch processing
- `batch.apply_to_files()` — Apply pipeline to file pattern
- `batch.parallel()` — Parallel execution with N workers
- `batch.map()` — Map function over file list

**Example Pipeline** (ffmpeg equivalent):
```morphogen
# ffmpeg -i input.mp4 -vf "scale=1920:-1, unsharp" -c:v libx264 -crf 18 output.mp4
pipeline:
  - input = video.decode("input.mp4")
  - scaled = video.scale(input, width=1920, height=-1)
  - sharpened = filter.unsharp(scaled, amount=1.5)
  - codec = codec.h264(crf=18, preset="fast")
  - video.encode(sharpened, codec, "output.mp4")
```

**Cross-Domain Integration**:
- **Video ↔ Audio**: `video.to_audio()`, `video.add_audio()`
- **Video ↔ Vision**: `video.to_frames()`, `vision.detect_objects()`, `vision.draw_bboxes()`
- **Video ↔ Geometry**: Render 3D scenes to video frames
- **Video ↔ Fields**: Overlay fluid simulations on video

**Why This Matters**:

Morphogen becomes the only platform that unifies audio synthesis, video encoding, sync correction, field simulation, agent systems, geometry, circuit modeling, and optimization — all with the same type system, scheduler, and MLIR compilation.

**Unique Capabilities**:
- ✅ **Cleaner than ffmpeg** — Typed operators, composable pipelines
- ✅ **More powerful than ffmpeg** — GPU-aware, cross-domain integration, AI upscaling
- ✅ **More deterministic than ffmpeg** — Same code → same output, always
- ✅ **More accessible than DaVinci Resolve** — Scripted, batchable, version-controllable

**Magic Operator**: `video.fix()` — One-liner to detect sync issues, denoise, stabilize, color correct, loudness normalize, upscale, and encode (auto-magic like DaVinci Resolve, but scripted and deterministic).

**Dependencies**:
- **Transform** (FFT for cross-correlation sync detection)
- **Audio** (existing Morphogen.Audio domain for DSP)
- **Visual** (frame rendering and export)
- **Stochastic** (for noise modeling in denoise)
- **Image/Vision** (for flash detection, object tracking) [future]

**Backend Strategy**:

Morphogen doesn't reimplement ffmpeg — it **orchestrates** ffmpeg as a backend:

```
Morphogen Pipeline → Graph IR → Backend Compiler → ffmpeg command
```

Advantages: Leverage ffmpeg's 20+ years of development, add type safety, enable composability, ensure determinism, optimize filter graphs, auto-select GPU codecs.

Alternative backends: GStreamer, custom C++, GPU compute shaders, hardware APIs (NVENC, VAAPI, VideoToolbox).

**References**:
- `../specifications/video-audio-encoding.md` — Complete video/audio encoding specification
- ffmpeg, DaVinci Resolve, Audacity (reference implementations)
- EBU R128 loudness standard
- NVENC, QuickSync, AMF (GPU acceleration APIs)

**Why This is a Perfect Fit for Morphogen**:

Video processing already behaves like Morphogen's operator DAG architecture. ffmpeg filter graphs map one-to-one to Morphogen pipelines. Audio/video sync leverages Morphogen's existing strength in time-domain signal processing. Batch processing exploits Morphogen's deterministic execution model. GPU acceleration fits naturally into Morphogen's MLIR compilation pipeline.

**Video belongs in Morphogen.** This is a huge new capability slice that fits perfectly with Morphogen's core architecture.

---

## 2. Next-Wave Domains (HIGHLY LIKELY)

These domains naturally emerge once you have a computational kernel that is deterministic, multirate, type+unit safe, GPU/CPU pluggable, and graph-IR based. This is where Morphogen becomes **superdomain-capable**, not just an audio/visual kernel.

---

### 2.1 Geometry & Mesh Processing

**Purpose**: Declarative geometric modeling, mesh processing, and spatial composition.

**Why Needed**: Essential for 3D modeling, CAD, robotics, physics simulation, 3D printing, computational geometry, and any domain requiring spatial reasoning.

**Status**: 🚧 In Progress (v0.9+) — **Inspired by TiaCAD v3.x**

**Key Innovation from TiaCAD**: Reference-based composition via **anchors** replaces hierarchical assemblies, making geometric composition declarative, robust, and refactor-safe.

---

#### Core Concepts (from TiaCAD)

**1. Coordinate Frames & Anchors**

Every geometric object lives in a coordinate frame and provides auto-generated anchors:

- **Frame** — Local coordinate system (origin, basis, scale)
- **Anchor** — Named reference point (`.center`, `.face_top`, `.edge_left`, etc.)
- **Placement** — Declarative composition: map anchor to anchor (not hierarchical nesting)

**Example:**
```morphogen
let base = geom.box(50mm, 30mm, 5mm)
let pillar = geom.cylinder(radius=5mm, height=50mm)

# Place pillar on top of base (declarative!)
let tower = mesh.place(
    pillar,
    anchor = pillar.anchor("bottom"),
    at = base.anchor("face_top")
)
```

**Contrast with hierarchical composition:**
- ❌ Traditional: `parent.add_child(child)` → hidden state, mutation, brittle
- ✅ TiaCAD model: `place(object, anchor, at=target)` → declarative, pure, robust

**See**: `../specifications/coordinate-frames.md` for full specification

---

#### Operator Families

**2. Primitives (3D Solids)**

```morphogen
geom.box(width, height, depth)
geom.sphere(radius)
geom.cylinder(radius, height)
geom.cone(radius_bottom, radius_top, height)
geom.torus(major_radius, minor_radius)
```

- All primitives auto-generate anchors (`.center`, `.face_{...}`, `.edge_{...}`)
- Deterministic (strict profile)

---

**3. Sketch Operations (2D → 2D)**

2D planar constructions (on XY plane):

```morphogen
sketch.rectangle(width, height)
sketch.circle(radius)
sketch.polygon(points)
sketch.regular_polygon(n_sides, radius)

# Boolean ops on sketches
sketch.union(s1, s2, ...)
sketch.difference(s1, s2)
sketch.offset(sketch, distance)
```

---

**4. Extrusion & Revolution (2D → 3D)**

```morphogen
extrude(sketch, height)
revolve(sketch, axis="z", angle=360deg)
loft(sketches, ruled=false)
sweep(profile, path, twist=0deg)
```

**Example:**
```morphogen
# Create vase by revolution
let profile = sketch.polygon([(0,0), (10,0), (8,20), (5,25)])
let vase = revolve(profile, axis="y", angle=360deg)
```

---

**5. Boolean Operations (3D)**

```morphogen
geom.union(s1, s2, ...)
geom.difference(s1, s2)
geom.intersection(s1, s2)

# Operator overloading
let result = solid_A + solid_B  # Union
let cut = solid_A - solid_B     # Difference
```

**Determinism**: Strict (within floating precision)

---

**6. Pattern Operations**

```morphogen
pattern.linear(object, direction, count, spacing)
pattern.circular(object, axis, count, angle=360deg)
pattern.grid(object, rows, cols, spacing_x, spacing_y)
```

**Example (bolt hole pattern):**
```morphogen
let hole = geom.cylinder(radius=3mm, height=10mm)
let bolts = pattern.circular(hole, axis="z", count=6)
```

---

**7. Finishing Operations**

```morphogen
geom.fillet(solid, edges, radius)    # Round edges
geom.chamfer(solid, edges, distance)  # Bevel edges
geom.shell(solid, faces, thickness)   # Hollow out
```

**Example:**
```morphogen
let box = geom.box(20mm, 20mm, 10mm)
let rounded = geom.fillet(box, edges=.edges(">Z"), radius=2mm)
```

---

**8. Mesh Operations (Discrete Geometry)**

```morphogen
mesh.from_solid(solid, tolerance=0.01mm)
mesh.subdivide(mesh, method="catmull-clark", iterations=1)
mesh.laplacian(mesh) -> SparseMatrix
mesh.sample(mesh, field: Field<T>) -> Mesh<T>
mesh.normals(mesh) -> Mesh<Vec3>
mesh.to_field(mesh, resolution) -> Field
field.to_mesh(field, isovalue) -> Mesh  # Marching cubes
```

---

**9. Measurement & Query**

```morphogen
geom.measure.volume(solid) -> f64
geom.measure.area(face) -> f64
geom.measure.bounds(object) -> BoundingBox
geom.measure.center_of_mass(solid) -> Vec3
geom.measure.distance(obj_a, obj_b) -> f64
```

---

**10. Transformations (with Explicit Origins)**

**TiaCAD principle**: All rotations/scales must specify an explicit origin (no implicit frame).

```morphogen
# ✅ Explicit origin (required!)
let rotated = transform.rotate(
    mesh,
    angle = 45 deg,
    origin = mesh.anchor("center")
)

# ❌ Implicit origin (compiler error!)
let bad = transform.rotate(mesh, 45 deg)  # ERROR: origin required
```

**Transform operators:**
```morphogen
transform.translate(object, offset)
transform.rotate(object, angle, axis, origin)
transform.scale(object, factor, origin)
transform.mirror(object, plane)
transform.affine(object, matrix)

# Coordinate conversions
transform.to_coord(field, coord_type="polar|spherical|cylindrical")
```

**See**: `../specifications/transform.md` Section 7 (Spatial Transformations)

---

#### Dependencies

- **Transform Dialect** — Spatial transformations, coordinate conversions
- **Fields** — For discretizations, SDF representations
- **Graph** — For mesh topology, adjacency
- **Sparse Linear Algebra** — For mesh Laplacian, PDE solvers
- **Type System** — Units (mm, m, deg, rad), frame types

---

#### Cross-Domain Integration

**Geometry → Fields (CFD, Heat Transfer)**
```morphogen
let solid = geom.sphere(10mm)
let sdf = field.from_solid(solid, bounds=..., resolution=(100,100,100))
let temperature = field.solve_heat(domain=sdf, ...)
```

**Geometry → Physics (Collision, Dynamics)**
```morphogen
let body = physics.rigid_body(
    shape = geom.box(10mm, 10mm, 10mm),
    mass = 1.0 kg
)
```

**Geometry → Visuals (Rendering)**
```morphogen
let rendered = visual.render(
    solid,
    camera_frame = camera.frame(),
    material = material.metal(roughness=0.2)
)
```

---

#### Backend Abstraction

Geometry operations are backend-neutral. Lowering varies by backend:

| Backend | Status | Capabilities |
|---------|--------|--------------|
| **CadQuery** | Planned | Full 3D CAD (OpenCASCADE-based) |
| **CGAL** | Future | Robust booleans, mesh processing |
| **OpenCASCADE** | Future | Industrial CAD kernel |
| **GPU SDF** | Research | Implicit surfaces (GPU-friendly) |

**Backend capabilities (operator registry):**
```yaml
operator:
  name: geom.boolean.union
  backend_caps:
    cadquery: supported
    cgal: supported
    gpu_sdf: supported (implicit conversion)
```

---

#### Use Cases

- **3D Printing** — Parametric part design, STL export
- **CAD** — Mechanical design, assemblies
- **Robotics** — Robot kinematic chains, collision geometry
- **CFD** — Mesh generation for fluid simulation
- **Physics** — Collision shapes, rigid body dynamics
- **Level-Set Methods** — Implicit surface evolution
- **Computational Geometry** — Voronoi, convex hulls, mesh analysis

---

#### Testing Strategy

**1. Determinism Tests**
```morphogen
# Primitives are bit-exact
assert_eq!(geom.box(10mm, 10mm, 10mm), geom.box(10mm, 10mm, 10mm))

# Anchors are deterministic
assert_eq!(box.anchor("face_top"), box.anchor("face_top"))
```

**2. Measurement Tests**
```morphogen
let cube = geom.box(10mm, 10mm, 10mm)
assert_approx_eq!(geom.measure.volume(cube), 1000.0 mm³, tol=1e-9)
```

**3. Transform Tests (Explicit Origins)**
```morphogen
# Rotation around center preserves center position
let rotated = transform.rotate(box, 45deg, origin=.center)
assert_vec_eq!(rotated.anchor("center").position(), box.anchor("center").position())
```

**4. Backend Equivalence**
```morphogen
@backend(cadquery)
let result_cq = geom.box(...) + geom.sphere(...)

@backend(cgal)
let result_cgal = geom.box(...) + geom.sphere(...)

assert_solid_equivalent!(result_cq, result_cgal, tol=1e-6)
```

---

#### Documentation

- **`../specifications/geometry.md`** — Full geometry domain specification
- **`../specifications/coordinate-frames.md`** — Frame/anchor system
- **`../specifications/transform.md`** — Section 7 (Spatial Transformations)
- **`../specifications/operator-registry.md`** — Layer 6b (Geometry operators)

---

#### Summary: Why TiaCAD Matters to Morphogen

TiaCAD's lessons apply **beyond geometry**:

1. **Anchors** — Unify references across domains (geometry, audio, physics, agents)
2. **Reference-based composition** — Replace hierarchies with declarative placement
3. **Explicit origins** — Prevent transform bugs, improve clarity
4. **Deterministic transforms** — Pure functions, no hidden state
5. **Backend abstraction** — Semantic operators, multiple lowering targets
6. **Parametric modeling** — Parts are pure functions (parameters → geometry)

**Key insight**: Anchors work for:
- **Geometry** — `.face_top`, `.edge_left`
- **Audio** — `.onset`, `.beat`, `.peak`
- **Physics** — `.center_of_mass`, `.joint`
- **Agents** — `.sensor`, `.waypoint`
- **Fields** — `.boundary_north`, `.gradient_max`

This unification makes Morphogen's multi-domain vision coherent and practical.

---

### 2.2 Sparse Linear Algebra

**Purpose**: Operations on sparse matrices and linear systems.

**Why Needed**: Critical for PDE solvers, graph algorithms, optimization, ML kernels, simulation.

**Status**: 🔲 Planned (currently using dense linalg for small problems)

**Operators**:
- `sparse.matmul` — Sparse matrix-vector multiply
- `sparse.solve` — Solve Ax = b (iterative solvers)
- `cg` — Conjugate Gradient
- `bicgstab` — BiConjugate Gradient Stabilized
- `sparse.cholesky` — Sparse Cholesky factorization
- `csr` / `csc` — Compressed Sparse Row/Column formats
- `sparse.transpose` — Sparse matrix transpose

**Dependencies**: None (foundational)

**Use Cases**: Poisson equation, graph Laplacian, structural analysis

**MLIR Integration**: Lower to `sparse_tensor` dialect

---

### 2.3 Optimization Domain

**Purpose**: Design discovery and parameter optimization across all Morphogen domains through comprehensive algorithm support.

**Why Critical**: Transforms Morphogen from **"simulate physics"** to **"discover new designs"**. Different optimization problems require different solvers based on continuity, smoothness, dimensionality, noise, and computational cost. Morphogen's physical domains (combustion, acoustics, circuits, motors, geometry) span all these problem types.

**Status**: 🔲 Planned (v0.10+)

**Reference**: See **[reference/optimization-algorithms.md](../reference/optimization-algorithms.md)** for complete algorithm specifications, operator signatures, and implementation roadmap.

---

#### Algorithm Categories

**1. Evolutionary / Population-Based (Global Search)**
- Best for: Messy, nonlinear, noisy, discontinuous problems
- **Genetic Algorithm (GA)** — Broad search, mixed continuous/discrete parameters
- **Differential Evolution (DE)** — Most reliable for continuous real-valued optimization
- **CMA-ES** — Gold standard for high-dimensional continuous optimization
- **Particle Swarm Optimization (PSO)** — Swarm-based cooperative search

**Use Cases**: LC filter optimization, J-tube geometry, muffler shapes, speaker EQ, motor torque ripple, PID tuning, acoustic chamber tuning, heat-transfer parameter fitting

---

**2. Local Numerical Optimization (Smooth Problems)**
- Best for: Problems with reliable gradients or smooth landscapes
- **Gradient Descent** — For differentiable objectives (requires autodiff)
- **Quasi-Newton (BFGS/L-BFGS)** — Second-order methods for faster convergence
- **Nelder-Mead (Simplex)** — Derivative-free local optimization

**Use Cases**: Filter coefficient tuning, control stability, thermodynamic equilibrium, curve fitting, impedance matching

---

**3. Surrogate / Model-Based Optimization**
- Best for: Expensive simulations (CFD, FEM) where each evaluation is costly
- **Bayesian Optimization** — Gaussian Process surrogates with intelligent sampling
- **Response Surface Modeling** — Polynomial/spline approximations
- **Kriging / RBF Surrogates** — For non-smooth high-dimensional problems

**Use Cases**: Combustion CFD optimization, expensive multi-domain simulations, gross tuning with limited budget

---

**4. Combinatorial / Discrete Optimization**
- Best for: Discrete parameter spaces (hole counts, component values, patterns)
- **Simulated Annealing** — Rugged discrete landscapes
- **Tabu Search** — Avoid revisiting poor regions
- **Beam Search / A\*** — State-space exploration with constraints

**Use Cases**: Jet hole patterns, PCB routing, discrete component selection (E12/E24 series), baffle counts, winding patterns

---

**5. Multi-Objective Optimization**
- Best for: Competing objectives (Pareto-optimal tradeoff exploration)
- **NSGA-II** — Standard multi-objective genetic algorithm
- **SPEA2** — Strength Pareto for complex tradeoff surfaces
- **Multi-Objective PSO (MOPSO)** — Swarm-based multi-objective

**Use Cases**: Minimize smoke AND maximize flame beauty, maximize torque AND minimize ripple, maximize quietness AND maintain power

---

#### Operator Contract

All optimizers share a unified interface:

**Inputs**:
- Parameter space (continuous bounds, discrete genome, or mixed)
- Objective function(s): `(T) -> f64` or `Array<(T) -> f64>` for multi-objective
- Algorithm-specific hyperparameters (population size, iterations, etc.)
- Stopping criteria (max evaluations, tolerance, time budget)
- Seed (for deterministic RNG)

**Outputs**:
- `OptResult<T>` containing:
  - Best solution found
  - Best fitness value(s)
  - Optimization history / convergence tracking
  - Algorithm-specific metadata (population, surrogate models, Pareto fronts)

**Example Operators**:
```morphogen
opt.ga<T>(genome, fitness, population_size, generations, ...) -> OptResult<T>
opt.de(bounds, fitness, population_size, generations, ...) -> OptResult<Array<f64>>
opt.cmaes(initial_mean, sigma, bounds, fitness, ...) -> OptResult<Array<f64>>
opt.bayesian(bounds, expensive_objective, n_iterations, ...) -> OptResult<Array<f64>> { gp_model }
opt.nsga2<T>(genome, objectives, population_size, ...) -> MultiObjectiveResult<T> { pareto_front }
```

---

#### Simulation Subgraph Integration

Optimizers accept **Morphogen simulation subgraphs** as objective functions:

```morphogen
# Define simulation
scene MotorTorqueRipple(winding_pattern: Array<int>) {
    let motor = motors.pmsm(winding_pattern)
    let torque = motors.compute_torque(motor, current_profile)
    out ripple = stdev(torque)
}

# Optimize winding pattern
let result = opt.de(
    bounds = [(0, 100); 12],
    fitness = |pattern| -simulate(MotorTorqueRipple(pattern)).ripple,
    population_size = 30,
    generations = 50
)
```

The subgraph is **compiled once**, then evaluated many times with different parameters — critical for performance.

---

#### Surrogate Model Storage

Surrogate models (Gaussian Processes, RBF, polynomials) are **first-class objects**:

```morphogen
# Train expensive surrogate
let result = opt.bayesian(bounds, expensive_cfd_simulation, n_iterations=50)

# Save GP model for reuse
io.save(result.gp_model, "chamber_efficiency_surrogate.gp")

# Later: load and query without re-running CFD
let gp_model = io.load<GaussianProcess>("chamber_efficiency_surrogate.gp")
let predicted_efficiency = gp_model.predict([150mm, 30mm, 250mm])

# Visualize learned landscape
viz.plot_surface_3d(gp_model, bounds, title="Predicted Efficiency")
```

---

#### Cross-Domain Applications

**Combustion Domain**:
- J-tube geometry optimization (GA for jet patterns)
- Flame shape evolution (CMA-ES for 10+ geometric parameters)
- CFD-based chamber design (Bayesian Optimization for expensive simulations)

**Acoustics Domain**:
- Muffler multi-objective design (NSGA-II: quietness vs. backpressure)
- Helmholtz resonator tuning (PSO, DE)
- Speaker crossover optimization (GA, multi-objective PSO)

**Motors Domain**:
- PID controller tuning (Differential Evolution)
- Torque ripple minimization (CMA-ES for magnet shapes)
- Winding pattern optimization (GA with discrete parameters)

**Geometry Domain (TiaCAD Integration)**:
- Parametric CAD → simulation → optimization loops
- High-dimensional parameter fitting (CMA-ES for 20+ control points)
- Multi-objective design exploration (Pareto-optimal geometries)

**Audio DSP Domain**:
- Filter parameter optimization (gradient descent with autodiff)
- EQ curve matching (L-BFGS)
- Room correction (multi-objective: flatness vs. phase)

---

#### Implementation Roadmap

**Phase 1 (v0.10)**: Core optimizers
1. Genetic Algorithm (GA) — Baseline evolutionary
2. Differential Evolution (DE) — Best general-purpose real-valued
3. CMA-ES — Gold standard for hard continuous problems
4. Nelder-Mead — Simple local optimizer
5. Simulated Annealing — Discrete + rugged landscapes

**Phase 2 (v1.0)**: Advanced methods
6. Bayesian Optimization — For expensive simulations
7. NSGA-II — Multi-objective Pareto optimization
8. L-BFGS — Quasi-Newton for smooth problems
9. Gradient Descent — Autodiff integration
10. Particle Swarm Optimization (PSO)

**Phase 3 (v1.1+)**: Complete catalog
11. SPEA2, Response Surface, Kriging, Tabu Search, Beam Search, MOPSO

---

#### Dependencies

- **Stochastic** — For mutation, crossover, initialization (evolutionary algorithms)
- **Linear Algebra** — For surrogate models (GP, RBF), covariance matrices (CMA-ES)
- **Autodiff** (Phase 2+) — For gradient-based methods
- **Sparse Linear Algebra** — For high-dimensional GP inference
- **Visualization** — Convergence plots, Pareto fronts, surrogate landscapes

---

#### Determinism

**Tier**: DETERMINISTIC (with fixed seed)

All optimizers guarantee:
- Bit-exact reproduction across platforms (with same seed)
- Enables regression testing and reproducible research
- Critical for scientific validation

```morphogen
# Same seed → identical results
let result1 = opt.ga(genome, fitness, seed=42)
let result2 = opt.ga(genome, fitness, seed=42)
assert_eq!(result1.best, result2.best)
```

---

#### What Morphogen Gains

With comprehensive optimization support, Morphogen enables:

1. **Automatic motor tuning** — Winding patterns, control loops
2. **Muffler shape evolution** — Multi-objective noise vs. backpressure
3. **Flame shape discovery** — J-tube geometry, jet patterns
4. **Speaker + room tuning** — EQ, crossover, placement
5. **Acoustic material discovery** — Perforate patterns, chamber dimensions
6. **Optimal LC filter tables** — Component value selection
7. **2-stroke expansion chamber design** — Length, diameter, taper
8. **Parametric CAD → Sim → Optimization loops** — TiaCAD integration
9. **GA-tuned control loops** — PID, MPC, LQR optimization
10. **Optimization-guided inverse problems** — Fitting recorded signals

---

**See**: **[reference/optimization-algorithms.md](../reference/optimization-algorithms.md)** for:
- Complete operator signatures for all 16 algorithms
- Detailed use cases for each Morphogen domain
- Implementation examples and testing strategy
- MLIR lowering approach
- Performance considerations and parallelization

---

### 2.4 Autodiff (Automatic Differentiation)

**Purpose**: Compute gradients, Jacobians, and Hessians automatically.

**Why Needed**: Unlocks physics simulation gradients, neural network training, differentiable graphics, differentiable audio, control optimization.

**Status**: 🔲 Planned (v0.11+)

**Operators**:
- `grad(op)` — Compute gradient of scalar function
- `jacobian` — Compute Jacobian matrix
- `hessian` — Compute Hessian matrix
- `jvp` — Jacobian-vector product (forward mode)
- `vjp` — Vector-Jacobian product (reverse mode)

**Dependencies**: None (but transforms entire graph)

**MLIR Integration**: Leverage Enzyme autodiff for MLIR

**Use Cases**: Differentiable physics, neural operators, sensitivity analysis

---

### 2.5 Graph / Network Domain

**Purpose**: Operations on graphs and networks.

**Why Needed**: Graph Laplacian transforms, spectral clustering, graph-based PDEs, network diffusion, routing/simulation, social/agent systems.

**Status**: 🔲 Planned (v0.10+)

**Operators**:
- `graph.laplacian` — Graph Laplacian matrix
- `graph.diffuse` — Diffusion on graph
- `graph.propagate` — Message propagation
- `graph.bfs` / `graph.dfs` — Breadth/depth-first search
- `graph.spectral_embed` — Spectral embedding
- `graph.pagerank` — PageRank algorithm
- `graph.shortest_path` — Dijkstra, Bellman-Ford

**Dependencies**: Sparse Linear Algebra

**Use Cases**: Social networks, circuit simulation, mesh processing

---

### 2.6 Image / Vision Ops

**Purpose**: Image processing operations (distinct from fractals and rendering).

**Why Needed**: Generic field operators + kernels for computer vision, photography, and scientific imaging.

**Status**: 🔲 Planned (v0.9+)

**Operators**:
- `blur` / `sharpen` — Convolution filters
- `edge_detect` — Sobel, Canny edge detection
- `optical_flow` — Lucas-Kanade, Farneback
- `color_transform` — RGB↔HSV, gamma correction
- `morphology.erode` / `morphology.dilate` — Morphological ops
- `histogram.equalize` — Histogram equalization
- `resize` — Image resampling (bilinear, bicubic, Lanczos)

**Dependencies**: Fields (images are 2D/3D fields), Transform (for frequency-domain filtering)

**Use Cases**: Photo processing, medical imaging, object detection

---

### 2.7 Instrument Modeling Domain

**Purpose**: Extract timbre from acoustic recordings and synthesize new notes with the same sonic character.

**Why Needed**: Convert recordings into reusable synthesis models (MIDI instruments, timbre morphing, luthier analysis, virtual acoustics).

**Status**: 🔲 Planned (v0.9-v1.0) — **The holy grail of audio DSP**

**Key Innovation**: Record acoustic guitar → extract timbre → synthesize new notes at any pitch.

---

#### Core Capabilities

**Analysis:**
- Extract fundamental frequency evolution (pitch tracking, vibrato)
- Track harmonic amplitudes over time (spectral envelope, timbre)
- Fit resonant body modes (damped sinusoids, modal analysis)
- Separate excitation from resonator (deconvolution)
- Measure decay rates per partial (T60, inharmonicity)
- Extract noise signatures (broadband residual)

**Synthesis:**
- Additive synthesis (sum of time-varying harmonics)
- Modal synthesis (damped sinusoid banks)
- Excitation generators (pluck, bow, hammer models)
- Spectral filtering (reapply timbre shape)
- Granular resynthesis (texture extension)

**Instrument Modeling:**
- Full analysis pipeline (`instrument.analyze`)
- Resynthesis at arbitrary pitch (`instrument.synthesize`)
- Timbre morphing (`instrument.morph`)
- Serialization (save/load models)

---

#### Operator Families

**Analysis Operators (13 new):**
```morphogen
pitch.autocorrelation(signal) → f32[Hz]
pitch.yin(signal, threshold) → f32[Hz]
harmonic.track_fundamental(signal) → Ctl[Hz]
harmonic.track_partials(signal, f0, num_partials) → Field2D<f32>
modal.analyze(signal, num_modes) → ModalModel
modal.extract_modes(spectrum) → Array<ModalPeak>
spectral.envelope(spectrum, smoothing) → Field1D<f32>
spectral.centroid(spectrum) → f32[Hz]
resonance.peaks(spectrum, threshold) → Array<(f32[Hz], f32)>
deconvolve(signal, f0) → (excitation: AudioSignal, body_ir: IR)
envelope.extract(signal, type) → Env
decay.fit_exponential(envelope, time) → f32[1/s]
decay.t60(envelope) → f32[s]
inharmonicity.measure(signal, f0) → f32
transient.detect(signal, threshold) → Array<f32[s]>
noise.extract_broadband(signal, harmonics) → NoiseModel
vibrato.extract(f0) → (rate: f32[Hz], depth: f32[cents], phase: f32[rad])
cepstral.transform(spectrum) → Field1D<f32>
```

**Synthesis Operators (6 new/extended):**
```morphogen
additive.synth(harmonics, f0) → AudioSignal
modal.synth(modes, excitation) → AudioSignal
excitation.pluck(type, params) → AudioSignal
spectral.filter(signal, envelope) → AudioSignal
granular.resynth(signal, grain_size, density) → AudioSignal
```

**High-Level Operators (5 new):**
```morphogen
instrument.analyze(signal) → InstrumentModel
instrument.synthesize(model, pitch, velocity) → AudioSignal
instrument.morph(model_a, model_b, blend) → InstrumentModel
instrument.save(model, path) → void
instrument.load(path) → InstrumentModel
```

---

#### Core Type: InstrumentModel

```morphogen
type InstrumentModel {
  id: String
  type: Enum  // "modal_string", "additive", etc.

  // Analysis results
  fundamental: Ctl[Hz]
  harmonics: Field2D<f32>          // Time × partial amplitudes
  modes: ModalModel                // Resonant body modes
  body_ir: IR                      // Body impulse response
  noise: NoiseModel                // Noise signature
  excitation: ExcitationModel      // Attack/pluck model
  decay_rates: Field1D<f32>        // Per-partial decay
  inharmonicity: f32               // Deviation from perfect harmonics

  // Synthesis parameters
  synth_params: Map<String, f32>
}
```

---

#### Cross-Domain Dependencies

- **Transform Domain (Layer 2)** — STFT, FFT for spectral analysis
- **Stochastic Domain (Layer 3)** — Noise modeling
- **Audio Domain (Layer 5)** — Filters, oscillators, effects
- **Physics Domain (Layer 4)** — Modal analysis (damped oscillators)

**Example workflow:**
```morphogen
// Analysis (cross-domain composition)
let spectrum = stft(recording)              // Transform domain
let f0 = harmonic.track_fundamental(spectrum)  // InstrumentModeling
let harmonics = harmonic.track_partials(spectrum, f0, 20)
let modes = modal.analyze(recording, 10)

// Build model
let guitar = InstrumentModel {
  fundamental: f0,
  harmonics: harmonics,
  modes: modes,
  ...
}

// Synthesis
let note_a3 = instrument.synthesize(guitar, pitch=220Hz, velocity=0.8)
let output = note_a3 |> reverb(0.15)        // Audio domain
```

---

#### Use Cases

| Use Case | Description | Output |
|----------|-------------|--------|
| **MIDI instrument creation** | Analyze one note → synthesize any pitch | Virtual instrument playable via MIDI |
| **Timbre morphing** | Blend two instruments | Hybrid guitar-violin timbre |
| **Luthier analysis** | Measure decay, resonance, inharmonicity | Quantitative instrument metrics |
| **Virtual acoustics** | Extract body IR → apply to other sounds | Guitar body applied to synth/drums |
| **Physics-informed synthesis** | Modal parameters → physical model | Expressive, controllable synthesis |
| **Archive preservation** | Digitize vintage instruments | Historical instrument model library |

---

#### Real-World Precedents

- **Yamaha VL1/VL70m** — Physical modeling synthesizers (1994)
- **Karplus-Strong** — Plucked string synthesis (1983)
- **Google Magenta NSynth** — Neural audio synthesis (2017)
- **IRCAM Modalys** — Modal synthesis framework
- **SPEAR, AudioSculpt** — Additive resynthesis tools

**Morphogen unifies all of these** in an extensible, GPU-accelerated, deterministic framework.

---

#### References

- **[specifications/timbre-extraction.md](../specifications/timbre-extraction.md)** — Full technical specification
- **[ADR-004: Instrument Modeling Domain](../adr/004-instrument-modeling-domain.md)** — Architectural decision
- **[reference/operator-registry-expansion.md](../reference/operator-registry-expansion.md)** — Complete operator catalog

---

### 2.8 Symbolic / Algebraic Domain

**Purpose**: Symbolic manipulation, algebraic simplification, analytic transforms.

**Why Needed**: Code generation, analytic transforms, parameter solving, optimization, constraints.

**Status**: 🔲 Planned (v0.12+)

**Operators**:
- `simplify(expr)` — Algebraic simplification
- `polynomial.fit` — Polynomial fitting
- `solve.linear` — Solve linear system symbolically
- `solve.symbolic` — Symbolic equation solving
- `diff(expr, var)` — Symbolic differentiation
- `integrate(expr, var)` — Symbolic integration

**Dependencies**: May lean on SymPy or custom MLIR dialect

**Use Cases**: Automatic kernel generation, analytic Jacobians, constraint solving

---

### 2.8 I/O & Storage Providers

**Purpose**: Load/save operations for external data (images, audio, graph snapshots).

**Why Needed**: Real-world workflows require loading IR, PNGs, WAVs, saving graph snapshots, streaming big data, mmap'ed intermediates.

**Status**: 🔲 Planned (v0.9+)

**Operators**:
- `io.load` — Load file (PNG, WAV, JSON, HDF5)
- `io.save` — Save file
- `io.stream` — Stream data (real-time or batch)
- `io.query` — Query external database
- `io.mmap` — Memory-map large file

**Dependencies**: None (runtime boundary)

**Determinism**: Nondeterministic (external I/O)

**Use Cases**: Asset loading, checkpointing, live audio input

---

### 2.9 Fluid Dynamics Domain

**Purpose**: Compressible and incompressible fluid flow, pressure wave propagation, gas dynamics, thermodynamic coupling.

**Why Needed**: Essential for exhaust acoustics, aerodynamics, combustion modeling, HVAC simulation, and any application involving gas or liquid flow. Particularly critical for multi-domain problems like 2-stroke engine exhaust systems where fluid pulses drive acoustic behavior.

**Status**: 🔲 Planned (v1.0+)

**Key Applications**:
- **2-Stroke Exhaust Systems** — Pressure pulse generation, backpressure timing, scavenging analysis
- **Aeroacoustics** — Flow-induced noise, jet noise, wind turbine acoustics
- **HVAC** — Air duct flow, ventilation system design
- **Combustion** — Engine cycles, burner design

---

#### Operator Families

**1. Compressible 1D Flow (Gas Dynamics)**

Fast, accurate simulation of wave propagation in ducts and pipes:

```morphogen
# 1D Euler equations (inviscid compressible flow)
fluid.euler_1d(
    pressure: Field1D<Pa>,
    velocity: Field1D<m/s>,
    density: Field1D<kg/m³>,
    dt: Time
) -> (pressure', velocity', density')

# Gas properties
fluid.gas_properties(
    temperature: Field1D<K>,
    composition: GasType  # "air", "exhaust", "steam", etc.
) -> GasProperties

fluid.sound_speed(gas_properties: GasProperties) -> Field1D<m/s>

# Pressure pulse generation (engine, explosion, valve opening)
fluid.pressure_pulse(
    peak_pressure: Pa,
    rise_time: Time,
    duration: Time,
    shape: "gaussian" | "square" | "exponential"
) -> PressurePulse
```

**Use Case:**
```morphogen
# 2-stroke engine combustion pulse
let pulse = fluid.pressure_pulse(
    peak_pressure = 15 bar,
    rise_time = 0.5 ms,
    duration = 2.0 ms,
    shape = "exponential"
)

# Propagate through exhaust pipe
let (p', v', rho') = fluid.euler_1d(pressure, velocity, density, dt)
```

---

**2. Incompressible Flow (Navier-Stokes)**

For liquid flow and low-speed gas flow:

```morphogen
# 2D/3D incompressible Navier-Stokes
fluid.navier_stokes(
    velocity: Field2D<Vec2<m/s>>,  # or Field3D<Vec3<m/s>>
    pressure: Field2D<Pa>,
    viscosity: f32 [Pa·s],
    density: f32 [kg/m³],
    dt: Time
) -> (velocity', pressure')

# Individual operations
fluid.advect(velocity: Field, advector: Field, dt: Time) -> Field
fluid.diffuse(field: Field, viscosity: f32, dt: Time) -> Field
fluid.project(velocity: Field, dt: Time) -> Field  # Divergence-free projection
```

**Use Case:**
```morphogen
# Classic smoke simulation
velocity = fluid.advect(velocity, velocity, dt)
velocity = fluid.diffuse(velocity, viscosity, dt)
velocity = fluid.project(velocity, dt)  # Ensure incompressibility
```

---

**3. Thermodynamic Coupling**

Temperature affects gas properties (density, viscosity, sound speed):

```morphogen
# Temperature-dependent properties
fluid.thermal_properties(
    temperature: Field<K>,
    gas_type: GasType
) -> ThermalProperties

# Convective heat transfer
fluid.convection(
    temperature: Field<K>,
    velocity: Field<m/s>,
    thermal_diffusivity: f32 [m²/s],
    dt: Time
) -> temperature'

# Buoyancy force (hot gas rises)
fluid.buoyancy_force(
    temperature: Field<K>,
    reference_temp: K,
    gravity: Vec3<m/s²>
) -> Field<Vec3<N/m³>>
```

---

**4. Boundary Conditions**

```morphogen
# Inlet boundary (fixed pressure or velocity)
fluid.inlet_bc(field: Field, inlet_region: Region, value: T) -> Field

# Outlet boundary (zero-gradient or pressure)
fluid.outlet_bc(field: Field, outlet_region: Region) -> Field

# No-slip boundary (velocity = 0 at wall)
fluid.no_slip_bc(velocity: Field, wall_region: Region) -> Field

# Slip boundary (frictionless wall)
fluid.slip_bc(velocity: Field, wall_region: Region) -> Field
```

---

**5. Engine-Specific Operators**

For 2-stroke and 4-stroke engine modeling:

```morphogen
# Combustion pulse for piston engine
engine.combustion_pulse(
    rpm: f32,
    displacement: f32 [cm³],
    compression_ratio: f32,
    fuel_type: "gasoline_2stroke" | "diesel" | "methanol"
) -> PressurePulse

# Backpressure timing analysis
engine.backpressure_timing(
    pressure_field: Field1D<Pa>,
    cycle_time: Time,
    scavenge_window: TimeRange
) -> BackpressureAnalysis {
    efficiency: f32,          # 0.0 to 1.0
    resonant_rpm: f32,
    peak_power_rpm: f32
}

# Scavenging efficiency (how much fresh mixture stays in cylinder)
engine.scavenging_efficiency(
    backpressure: Field1D<Pa>,
    port_timing: PortTiming
) -> f32  # 0.0 to 1.0
```

---

#### Dependencies

- **FieldDomain** — Fluid properties live on grids
- **IntegratorsDomain** — Time-stepping for PDEs
- **ThermalDomain** — Temperature coupling
- **GeometryDomain** — Flow through shaped ducts
- **AcousticsDomain** — Pressure waves become sound

---

#### Cross-Domain Integration

**FluidDynamics → Acoustics (2-Stroke Exhaust)**
```morphogen
# Fluid dynamics generates pressure pulse
let pulse = engine.combustion_pulse(rpm, displacement, compression_ratio)

# Acoustics propagates pulse through pipe geometry
let waveguide = acoustic.waveguide_from_geometry(exhaust_pipe)
let sound = acoustic.propagate(pulse, waveguide, dt)
```

**FluidDynamics → Thermal → FluidDynamics (Feedback)**
```morphogen
# Temperature affects gas properties
let gas_props = fluid.thermal_properties(temperature, "exhaust")

# Gas properties affect wave propagation
let wave_speed = fluid.sound_speed(gas_props)

# Flow heats up pipe walls (future: conjugate heat transfer)
```

---

#### Testing Strategy

**1. Conservation Tests**
```morphogen
# Mass conservation
assert total_mass(density_field) ≈ total_mass(density_field')

# Energy conservation (inviscid flow)
assert total_energy(p, v, rho) ≈ total_energy(p', v', rho')
```

**2. Shock Tube (Riemann Problem)**
```morphogen
# Standard gas dynamics test: discontinuity propagation
let (p_left, p_right) = (100 kPa, 10 kPa)
let shock_tube = fluid.riemann_problem(p_left, p_right)
# Verify shock speed, contact discontinuity, rarefaction wave
```

**3. Pipe Resonance**
```morphogen
# Fundamental mode of organ pipe
let resonant_freq = fluid.pipe_resonance(length, open_ends=true)
assert resonant_freq ≈ sound_speed / (2 * length)
```

---

#### References

- **"Computational Fluid Dynamics"** — Anderson (2009)
- **"Gas Dynamics"** — James E. John (1984)
- **"Two-Stroke Engine Exhaust Systems"** — Gorr, Benson
- **Ricardo WAVE** — Commercial 1D gas dynamics software

---

### 2.10 Acoustics Domain

**Purpose**: Sound wave propagation, resonance, impedance, filtering, radiation, and multi-domain coupling of mechanical/fluid systems to audio output.

**Why Needed**: Critical for musical instrument modeling, architectural acoustics, noise control, exhaust system design, speaker design, and any application where vibration or pressure waves produce sound. Bridges physics simulation to audio output.

**Status**: 🔲 Planned (v1.0+)

**Key Applications**:
- **2-Stroke Exhaust Acoustics** — Muffler design, sound prediction, backpressure tuning
- **Musical Instruments** — Physical modeling synthesis (brass, woodwinds, strings)
- **Architectural Acoustics** — Room modes, reverberation, absorption
- **Noise Control** — Muffler design, sound barriers, active noise cancellation
- **Speaker Design** — Enclosure resonance, port tuning, radiation patterns

---

#### Core Concepts

**1. Acoustic Impedance**

Ratio of pressure to volume velocity (analogous to electrical impedance):
- **High Impedance** — Hard to move air (small opening, stiff boundary)
- **Low Impedance** — Easy to move air (large opening, compliant boundary)
- **Matched Impedance** — Maximum energy transfer (no reflections)

**2. Wave Propagation**

Sound travels as pressure waves:
- **1D Waveguides** — Pipes, ducts (plane wave approximation)
- **2D/3D Fields** — Rooms, open air (spherical/cylindrical waves)
- **Modal Resonance** — Standing waves at specific frequencies

**3. Reflection & Transmission**

At acoustic discontinuities (area changes, material boundaries):
- **Reflection Coefficient** — How much wave bounces back
- **Transmission Coefficient** — How much wave passes through

---

#### Operator Families

**1. Waveguide Construction (1D Acoustics)**

Convert geometry to acoustic network (fast, accurate for pipes):

```morphogen
# Build digital waveguide from pipe geometry
acoustic.waveguide_from_geometry(
    geometry: PipeGeometry,
    discretization: Length,  # segment length
    sample_rate: Hz
) -> WaveguideNetwork

# Compute reflection coefficients at area discontinuities
acoustic.reflection_coefficients(
    waveguide: WaveguideNetwork
) -> Vec<ReflectionCoeff>  # -1.0 (open) to +1.0 (closed)

# Update waveguide with temperature-dependent properties
acoustic.update_properties(
    waveguide: WaveguideNetwork,
    wave_speed: Field1D<m/s>,
    impedance: Field1D<Pa·s/m³>
) -> WaveguideNetwork
```

**Use Case:**
```morphogen
# 2-stroke expansion chamber
let chamber = geom.expansion_chamber(inlet=40mm, belly=120mm, outlet=50mm)
let waveguide = acoustic.waveguide_from_geometry(chamber, dx=1mm, sr=44100Hz)
let reflections = acoustic.reflection_coefficients(waveguide)
```

---

**2. Waveguide Propagation**

Sample-accurate wave propagation (digital waveguide algorithm):

```morphogen
# Single time step of waveguide simulation
acoustic.waveguide_step(
    pressure_forward: Field1D<Pa>,   # right-traveling wave
    pressure_backward: Field1D<Pa>,  # left-traveling wave
    waveguide: WaveguideNetwork,
    reflections: Vec<ReflectionCoeff>,
    dt: Time
) -> (pressure_forward', pressure_backward')

# Combined pressure + velocity formulation
acoustic.waveguide_step_pv(
    pressure: Field1D<Pa>,
    velocity: Field1D<m/s>,
    waveguide: WaveguideNetwork,
    reflections: Vec<ReflectionCoeff>,
    absorption: Vec<AbsorptionCoeff>,
    radiation: RadiationImpedance,
    dt: Time
) -> (pressure', velocity')
```

---

**3. Helmholtz Resonators**

Classic acoustic component (volume + neck = tuned resonator):

```morphogen
# Compute resonant frequency of Helmholtz resonator
acoustic.helmholtz_frequency(
    volume: m³,
    neck_length: m,
    neck_area: m²
) -> Hz

# Create resonator operator
acoustic.helmholtz_resonator(
    volume: m³,
    neck_length: m,
    neck_area: m²,
    damping: f32  # 0.0 (lossless) to 1.0 (heavily damped)
) -> Resonator

# Embed in waveguide network
acoustic.attach_resonator(
    waveguide: WaveguideNetwork,
    position: Length,
    resonator: Resonator
) -> WaveguideNetwork
```

**Use Case:**
```morphogen
# Muffler with quarter-wave resonator
let resonator = acoustic.helmholtz_resonator(
    volume = 500 cm³,
    neck_length = 50 mm,
    neck_area = 20 cm²
)
# Resonant frequency ≈ 340 / (2π) * sqrt(A / (V * L)) ≈ 150 Hz
```

---

**4. Perforated Pipes & Absorption**

Muffler components:

```morphogen
# Perforated pipe (partial reflection, partial transmission)
acoustic.perforated_pipe(
    hole_diameter: Length,
    hole_spacing: Length,
    open_area_ratio: f32,  # 0.0 to 1.0
    pipe_diameter: Length
) -> AcousticImpedance

# Absorption material (fiberglass, foam, rock wool)
acoustic.absorption_material(
    type: "fiberglass" | "foam" | "rockwool",
    density: kg/m³,
    thickness: Length,
    frequency_range: (Hz, Hz)
) -> FrequencyDependentAbsorption

# Apply absorption to waveguide segment
acoustic.absorptive_segment(
    waveguide: WaveguideNetwork,
    start: Length,
    end: Length,
    absorption: FrequencyDependentAbsorption
) -> WaveguideNetwork
```

---

**5. Radiation Impedance**

Sound radiating from pipe exit to open air:

```morphogen
# Radiation impedance (frequency-dependent)
acoustic.radiation_impedance(
    diameter: Length,
    type: "unflanged" | "flanged" | "infinite_baffle"
) -> FrequencyDependentImpedance

# Radiate to specific point in space
acoustic.radiate_to_point(
    source_pressure: Pa,
    distance: Length,
    angle: Angle,  # 0° = on-axis, 90° = perpendicular
    radiation: RadiationImpedance
) -> Pa  # Sound pressure at listener position

# Directivity pattern (how sound spreads in 3D)
acoustic.directivity_pattern(
    source: AcousticSource,
    frequencies: Vec<Hz>
) -> Field2D<dB>  # Polar plot of radiation
```

**Use Case:**
```morphogen
# Tailpipe exit radiation
let radiation = acoustic.radiation_impedance(diameter=45mm, type="unflanged")
let mic_pressure = acoustic.radiate_to_point(
    source_pressure = tailpipe_pressure,
    distance = 1.0m,
    angle = 90deg,
    radiation
)
```

---

**6. FDTD Acoustics (2D/3D)**

Finite-difference time-domain for complex geometries:

```morphogen
# 2D/3D acoustic field simulation
acoustic.fdtd_step(
    pressure: Field2D<Pa>,       # or Field3D
    velocity: Field2D<Vec2<m/s>>, # or Field3D<Vec3<m/s>>
    geometry: SDF,  # Signed distance field for boundaries
    absorption: Field2D<f32>,  # Wall absorption coefficient
    dt: Time
) -> (pressure', velocity')

# Boundary conditions
acoustic.pressure_boundary(field, region, pressure_value)
acoustic.velocity_boundary(field, region, velocity_value)
acoustic.absorbing_boundary(field, region)  # Anechoic (no reflections)
```

**Use Case:**
```morphogen
# Room acoustics simulation
let room = geom.box(5m, 4m, 3m)
let sdf = field.from_solid(room, resolution=1cm)
let (p', v') = acoustic.fdtd_step(pressure, velocity, sdf, absorption, dt)
```

---

**7. Transfer Functions & Frequency Analysis**

```morphogen
# Compute acoustic transfer function (input → output)
acoustic.transfer_function(
    waveguide: WaveguideNetwork,
    freq_range: (Hz, Hz),
    resolution: Hz
) -> FrequencyResponse {
    frequencies: Vec<Hz>,
    magnitude: Vec<dB>,
    phase: Vec<rad>
}

# Find resonant frequencies (peaks in transfer function)
acoustic.resonant_frequencies(
    waveguide: WaveguideNetwork,
    threshold_db: f32
) -> Vec<Hz>

# Insertion loss (difference with/without muffler)
acoustic.insertion_loss(
    system_with_muffler: WaveguideNetwork,
    system_without_muffler: WaveguideNetwork
) -> FrequencyResponse
```

---

**8. Lumped Acoustic Networks (Circuit Analogy)**

**Concept:** Acoustics ≈ Electrical circuits
- **Acoustic Compliance** (volume) ≈ Capacitance
- **Acoustic Inertance** (pipe mass) ≈ Inductance
- **Acoustic Resistance** (friction) ≈ Resistance

```morphogen
# Convert acoustic system to lumped circuit
acoustic.to_lumped_network(
    geometry: PipeGeometry
) -> AcousticCircuit {
    nodes: Vec<Node>,
    elements: Vec<AcousticElement>
}

# Solve lumped network (fast, analytical)
acoustic.solve_lumped(
    circuit: AcousticCircuit,
    excitation: PressurePulse,
    freq_range: (Hz, Hz)
) -> FrequencyResponse
```

**Use Case:**
```morphogen
# Quick muffler design iteration
let circuit = acoustic.to_lumped_network(muffler_geometry)
let response = acoustic.solve_lumped(circuit, engine_pulse, (50Hz, 5000Hz))
# 100x faster than full waveguide simulation
```

---

#### Dependencies

- **GeometryDomain** — Pipe shapes, chambers, room geometry
- **FieldDomain** — 2D/3D acoustic fields (FDTD)
- **TransformDomain** — FFT for frequency analysis
- **FluidDynamicsDomain** — Pressure pulse generation
- **AudioDomain** — Convert pressure to audio samples

---

#### Cross-Domain Integration

**Acoustics → Audio (Sound Synthesis)**
```morphogen
# Acoustic pressure → Audio signal
let audio_sample = audio.pressure_to_sample(
    acoustic_pressure_pa,
    reference_pressure = 20e-6 Pa  # 0 dB SPL
)

# Apply microphone model
let mic_signal = audio.microphone_response(
    acoustic_signal,
    mic_type = "condenser",
    frequency_response = "flat"
)
```

**FluidDynamics → Acoustics → Audio (Complete Chain)**
```morphogen
# 1. Fluid dynamics: Combustion pulse
let pulse = engine.combustion_pulse(rpm=8000)

# 2. Acoustics: Propagate through exhaust
let waveguide = acoustic.waveguide_from_geometry(exhaust_system)
let tailpipe_pressure = acoustic.propagate(pulse, waveguide, dt)

# 3. Acoustics: Radiate to microphone
let mic_pressure = acoustic.radiate_to_point(tailpipe_pressure, distance=1m)

# 4. Audio: Convert to WAV
let audio = audio.pressure_to_sample(mic_pressure)
audio.export_wav("engine_sound.wav", audio)
```

**Geometry → Acoustics (Shape Determines Sound)**
```morphogen
# Parametric pipe design
let chamber = geom.expansion_chamber(
    diverge_angle = 12 deg,  # ← Change this
    belly_diameter = 120 mm  # ← Or this
)

# Automatically affects acoustic behavior
let waveguide = acoustic.waveguide_from_geometry(chamber)
let resonant_freqs = acoustic.resonant_frequencies(waveguide)
# Different geometry → different resonances
```

---

#### Testing Strategy

**1. Pipe Resonance Tests**
```morphogen
# Open-open pipe: f = n * c / (2L)
let pipe = geom.cylinder(diameter=50mm, length=1m)
let resonances = acoustic.resonant_frequencies(pipe)
assert resonances[0] ≈ 340 Hz / (2 * 1m) = 170 Hz
```

**2. Reflection Coefficient Tests**
```morphogen
# Area expansion: negative reflection
let expansion = geom.area_change(from=10cm², to=40cm²)
let R = acoustic.reflection_coefficient(expansion)
assert R < 0  # Negative reflection (pressure inversion)

# Open end: R ≈ -1.0
# Closed end: R ≈ +1.0
```

**3. Helmholtz Resonator Tests**
```morphogen
let resonator = acoustic.helmholtz_resonator(V=1L, L=10cm, A=1cm²)
let f_res = acoustic.resonant_frequency(resonator)
assert f_res ≈ (340 / 2π) * sqrt(1e-4 / (1e-3 * 0.1)) ≈ 54 Hz
```

**4. Energy Conservation (Lossless Waveguide)**
```morphogen
let energy_before = acoustic.total_energy(pressure, velocity)
let (p', v') = acoustic.waveguide_step(pressure, velocity, ...)
let energy_after = acoustic.total_energy(p', v')
assert energy_before ≈ energy_after  # (if no absorption)
```

---

#### Use Cases

1. **2-Stroke Exhaust System**
   - Design expansion chamber for peak power at target RPM
   - Predict exhaust sound at different RPMs
   - Optimize muffler for noise reduction without power loss

2. **Musical Instrument Modeling**
   - Brass instrument bore design (trumpet, trombone)
   - Woodwind tone hole placement (flute, clarinet)
   - Physical modeling synthesis (real-time audio)

3. **Architectural Acoustics**
   - Concert hall modal analysis
   - Room reverberation time
   - Sound absorption treatment placement

4. **Noise Control**
   - HVAC duct silencer design
   - Industrial muffler optimization
   - Active noise cancellation (predict cancellation signal)

5. **Speaker Design**
   - Bass reflex port tuning
   - Transmission line design
   - Directivity control

---

#### Documentation

- **`docs/domains/ACOUSTICS.md`** — Full acoustics domain specification
- **`docs/USE_CASES/2-stroke-muffler-modeling.md`** — Complete multi-domain example
- **`examples/acoustics/pipe_resonance.kairo`** — Simple resonance demo
- **`examples/acoustics/helmholtz_resonator.kairo`** — Resonator tuning
- **`examples/acoustics/expansion_chamber.kairo`** — 2-stroke chamber

---

#### References

- **"Acoustics: An Introduction"** — Kinsler, Frey, Coppens, Sanders
- **"Acoustic Wave Propagation in Ducts and Mufflers"** — Munjal (2014)
- **"Digital Waveguide Networks for Acoustic Modeling"** — Van Duyne, Smith (1993)
- **"Physical Audio Signal Processing"** — Julius O. Smith III (online book)
- **Yamaha VL1** — Commercial physical modeling synthesizer (uses waveguides)
- **Ricardo WAVE** — 1D engine/exhaust simulation software

---

### 2.11 Emergence Domain (Complex Systems & Artificial Life)

**Purpose**: Unified simulation of emergent systems including Cellular Automata, Agent-Based Models, Reaction-Diffusion, L-Systems, and Swarm Intelligence.

**Why Critical**: Transforms Morphogen from **"simulate physics"** to **"create artificial life and complex systems"**. Emergent systems are foundational for:
- **Creative coding** (generative art, procedural generation)
- **Biological modeling** (morphogenesis, ecology, evolution)
- **Optimization** (swarm intelligence, stigmergy)
- **Procedural content** (trees, textures, networks)

**Status**: 🔲 Proposed (v0.10-v1.0)

**Reference**: See **[ADR-005](../adr/005-emergence-domain.md)**, **[specifications/emergence.md](../specifications/emergence.md)**, and **[reference/emergence-operators.md](../reference/emergence-operators.md)** for complete specifications.

---

#### Sub-Domains

**1. Cellular Automata (CA)**

Grid-based evolution with local rules:

| Operator | Description |
|----------|-------------|
| `ca.create` | Initialize CA grid (2D/3D) |
| `ca.step` | Evolve one generation |
| `ca.rule_preset` | Common rules (Life, Brian's Brain, Wireworld, Rule 30/110) |
| `ca.lenia` | Continuous CA (smooth life-like organisms) |
| `ca.to_field` | Convert to continuous field |

**Use Cases:**
- Texture generation (biological, electronic patterns)
- Physics approximation (lattice-gas hydrodynamics)
- Electronic circuits (Wireworld)
- Procedural modeling

**2. Agent-Based Models (ABM)**

Particle systems with behavioral rules:

| Operator | Description |
|----------|-------------|
| `agent.create` | Spawn agent population |
| `agent.boids` | Flocking behavior (Reynolds boids) |
| `agent.vicsek` | Active matter physics |
| `agent.schelling` | Segregation dynamics |
| `agent.predator_prey` | Ecology simulation |
| `agent.to_field` | Rasterize to grid (particle-in-cell) |
| `agent.from_field` | Sample field at agent positions |

**Use Cases:**
- Flocking/swarming animation
- Crowd simulation
- Ecological modeling
- Social dynamics research
- Active matter physics

**3. Reaction-Diffusion (RD)**

Continuous pattern-forming PDEs:

| Operator | Description |
|----------|-------------|
| `rd.gray_scott` | Gray-Scott RD system |
| `rd.turing` | Turing pattern generator |
| `rd.to_geometry` | Extract isosurface (Marching Cubes) |

**Use Cases:**
- Biological patterns (spots, stripes, waves)
- Texture generation
- Chemical simulation (Belousov-Zhabotinsky)
- Procedural design

**4. L-Systems (Lindenmayer Systems)**

Recursive growth and morphogenesis:

| Operator | Description |
|----------|-------------|
| `lsys.create` | Define L-system grammar |
| `lsys.evolve` | Evolve string n generations |
| `lsys.to_geometry` | Turtle graphics → 3D geometry |
| `lsys.preset` | Common systems (trees, fractals) |

**Use Cases:**
- Tree/plant generation
- Fractal structures
- Coral-like forms
- Vascular systems
- Procedural architecture

**5. Swarm Intelligence**

Stigmergic optimization and network formation:

| Operator | Description |
|----------|-------------|
| `swarm.ants` | Ant Colony Optimization (ACO) |
| `swarm.slime_mold` | Physarum network optimization |
| `swarm.firefly` | Firefly algorithm |

**Use Cases:**
- Pathfinding (ACO)
- Network generation (slime mold)
- Distributed optimization
- Routing algorithms

---

#### Cross-Domain Integration

**Emergence → Geometry (Pattern → Surface)**
```morphogen
// Reaction-diffusion → 3D surface
(u, v) = rd.gray_scott(u, v, f=0.04, k=0.06)
let heightmap = v
let surface = geom.displace(plane, heightmap, scale=20mm)
```

**Emergence → Physics (Network → Structure)**
```morphogen
// Slime mold network → structural optimization
let network = swarm.slime_mold(field, food_sources=anchors)
let structure = geom.from_network(network, diameter=5mm)
let stress = physics.stress_test(structure, load=100N)
```

**Emergence → Acoustics (Swarm → Scattering)**
```morphogen
// Boids → acoustic scatterers
let positions = agent.positions(boids)
let wave = acoustic.propagate_with_scatterers(source, scatterers=positions)
```

**Emergence → Audio (Sonification)**
```morphogen
// Agent density → audio frequency
let density = agent.to_field(boids, property="density")
let freq = 200Hz + density.mean() * 800Hz
out audio = osc.sine(freq)
```

**Emergence → Optimization (Evolutionary Design)**
```morphogen
// Optimize CA parameters for structural strength
let result = opt.nsga2(
    objectives = [strength_fn, lightness_fn],
    params = [("ca_seed", "int"), ("ca_steps", "int")],
    ...
)
```

---

#### Dependencies

- **FieldDomain** — CA/RD operate on grids
- **IntegratorsDomain** — Agent dynamics (RK4, Verlet)
- **StochasticDomain** — Mutations, noise, initialization
- **GeometryDomain** — Pattern → surface conversion
- **PhysicsDomain** — Agent → rigid body, structure testing
- **AcousticsDomain** — Swarm → scattering
- **OptimizationDomain** — Swarm algorithms (PSO, ACO)

---

#### Unique Capabilities (vs. NetLogo, Golly, Processing)

**Existing tools:**
- **NetLogo:** ABM only, no cross-domain integration
- **Golly:** CA only, standalone application
- **Processing/p5.js:** Visual scripting, ad-hoc implementations
- **MATLAB/Python:** Fragmented libraries

**Morphogen EmergenceDomain:**
1. ✅ **Unified platform** — CA + ABM + RD + L-systems + Swarms in one system
2. ✅ **Cross-domain integration** — Seamless composition with Geometry, Physics, Audio, Optimization
3. ✅ **GPU acceleration** — All operators designed for parallel execution
4. ✅ **Deterministic** — Strict/repro profiles for reproducible science
5. ✅ **Type + unit safety** — Physical units tracked
6. ✅ **MLIR compilation** — JIT to CPU/GPU

**No competitor offers this.**

---

#### Example Applications

**1. Biological Morphogenesis → 3D Printing**
```morphogen
// RD pattern → geometry → stress test → STL export
(u, v) = rd.gray_scott(u, v, f=0.055, k=0.062)
let surface = geom.displace(plane, v, scale=20mm)
let stress = physics.stress_test(surface, load=100N)
io.export_stl(surface, "organic_structure.stl")
```

**2. Slime Mold Network → PCB Routing**
```morphogen
// Optimize PCB traces using slime mold
let network = swarm.slime_mold(field, food_sources=component_positions)
let traces = circuit.from_network(network, width=0.2mm)
let parasitics = circuit.extract_parasitics(traces)
```

**3. Boids → Acoustic Scattering → Audio**
```morphogen
// Swarm scatters sound waves
let positions = agent.positions(boids)
let wave = acoustic.propagate_with_scatterers(source, scatterers=positions)
out audio = acoustic.to_audio(wave, mic_position=vec3(10, 0, 0))
```

**4. CA Lattice → Structural Optimization**
```morphogen
// Optimize CA-generated lattice for strength/weight
let result = opt.nsga2(
    objectives = [maximize_strength, minimize_mass],
    design_fn = |params| ca_to_lattice_to_stress(params)
)
```

**5. L-System Trees → Wind Physics**
```morphogen
// Generate tree, simulate wind forces
let tree = lsys.to_geometry(tree_lsys.evolve(5))
let branches = physics.rigid_bodies(tree.branches)
let wind_forces = fluid.drag_force(wind_field, branches)
```

---

#### Implementation Roadmap

**Phase 1: Core Infrastructure (v0.10)**
- [ ] CAGrid2D/3D types, Agents<A> container
- [ ] Spatial indexing (grid, k-d tree)

**Phase 2: CA Operators (v0.10)**
- [ ] ca.create, ca.step, ca.rule_preset (9 presets)
- [ ] ca.lenia, ca.to_field

**Phase 3: ABM Operators (v0.10)**
- [ ] agent.create, agent.boids, agent.vicsek
- [ ] agent.to_field, agent.from_field

**Phase 4: RD Operators (v0.11)**
- [ ] rd.gray_scott, rd.turing
- [ ] rd.to_geometry (marching cubes)

**Phase 5: L-Systems (v0.11)**
- [ ] lsys.create, lsys.evolve, lsys.to_geometry

**Phase 6: Swarm Intelligence (v1.0)**
- [ ] swarm.ants, swarm.slime_mold
- [ ] Integration with OptimizationDomain

**Phase 7: Cross-Domain Examples (v1.0)**
- [ ] 10+ examples spanning all integration patterns

---

#### Testing Strategy

**Determinism Tests:**
```morphogen
// CA must be bit-exact
let ca1 = ca.create(128, 128, seed=42)
let ca2 = ca.create(128, 128, seed=42)
assert_eq!(ca1, ca2)
```

**Conservation Tests:**
```morphogen
// Boids: momentum conservation
let p_before = agent.total_momentum(boids)
boids = agent.boids(boids, dt=0.1s)
let p_after = agent.total_momentum(boids)
assert_approx_eq!(p_before, p_after)
```

**Pattern Recognition Tests:**
```morphogen
// Life: glider moves diagonally
let glider = ca.load_pattern("glider")
let evolved = ca.step_n(glider, rule=life_rule, steps=4)
assert_eq!(ca.find_pattern(evolved, "glider"), vec2(1, 1))
```

---

#### Documentation

- **[specifications/emergence.md](../specifications/emergence.md)** — Complete domain specification
- **[ADR-005](../adr/005-emergence-domain.md)** — Architectural decision record
- **[reference/emergence-operators.md](../reference/emergence-operators.md)** — Full operator catalog (45 operators)
- **[examples/emergence-cross-domain.md](../examples/emergence-cross-domain.md)** — 5 complete integration examples

---

#### Why This is a Perfect Fit for Morphogen

**Emergent systems are fundamentally graph-friendly:**
- Local rules → composable operators
- Embarrassingly parallel → GPU acceleration
- Deterministic (with fixed seeds) → reproducible science
- Cross-domain potential → Geometry, Physics, Audio, Optimization

**The killer feature:** No existing tool unifies emergence + physics + audio + geometry in one deterministic, GPU-accelerated platform.

This makes Morphogen the **universal platform for complex systems research and creative coding**.

---

### 2.12 Procedural Generation Domain (SpeedTree-Inspired Universal Synthesis)

**Purpose**: Unified procedural synthesis of complex structures including trees, terrains, cities, organic forms, fractals, and materials through rule-based, noise-driven, and grammar-based generation.

**Why Critical**: Transforms Morphogen from **"simulate existing systems"** to **"synthesize new worlds"**. This domain enables:
- **Game development** (forests, terrains, cities — rival SpeedTree, Houdini)
- **Film/VFX** (environment generation at scale)
- **Architecture** (parametric design, procedural buildings)
- **Creative coding** (generative art, organic structures)
- **3D printing** (procedural organic geometries)

**SpeedTree Inspiration**: SpeedTree is the industry-standard vegetation tool ($$$), but it's limited to trees and offline workflows. **Morphogen can exceed SpeedTree** by adding:
- **True physics-based growth** (light, gravity, structural optimization)
- **Cross-domain integration** (trees + wind + acoustics + terrain + ecology)
- **GPU real-time generation** (100k+ instances)
- **General-purpose synthesis** (not just trees — also terrains, cities, materials, fractals)

**Status**: 🔲 Proposed (v0.11-v1.0)

**Reference**: See **[ADR-008](../adr/008-procedural-generation-domain.md)**, **[specifications/procedural-generation.md](../specifications/procedural-generation.md)**, and **[reference/procedural-operators.md](../reference/procedural-operators.md)** for complete specifications.

---

#### Sub-Domains

**1. Generative Grammars (Extended L-Systems)**

Rule-based hierarchical structure generation:

| Operator | Description |
|----------|-------------|
| `lsystem.define` | Define L-system grammar |
| `lsystem.expand` | Expand grammar n iterations |
| `lsystem.parametric` | Context-sensitive rules (parameter evolution) |
| `lsystem.stochastic` | Probabilistic rules (variation) |
| `lsystem.preset` | Common grammars (koch_curve, fractal_tree, realistic_tree) |

**Use Cases:**
- Tree/plant generation
- Fractal structures
- Architectural grammars
- Recursive growth systems

**Note**: EmergenceDomain includes basic L-systems for string rewriting. ProceduralDomain **extends** this with production-quality tree generation (meshing, foliage, wind, physics integration).

---

**2. Branching and Tree Structures**

Algorithms for realistic tree generation beyond simple grammars:

| Operator | Description |
|----------|-------------|
| `branches.from_lsystem` | Convert L-system to branch hierarchy |
| `branches.to_splines` | Smooth branch curves (Bezier, Catmull-Rom) |
| `branches.to_mesh` | Generate cylindrical meshes with bark texture |
| `branches.randomize_angles` | Add noise-driven variation |
| `branches.prune` | Remove branches by depth/length/criteria |
| `growth.space_colonization` | Attraction-point-based growth (realistic crowns) |
| `growth.tropism` | Directional bias (phototropism, gravitropism) |
| `growth.gravity` | Branch bending under weight |
| `growth.collision_avoidance` | Prevent self-intersection |

**Use Cases:**
- SpeedTree-quality trees
- Physics-based growth simulation
- Biomechanical tree modeling
- Structural optimization via growth

---

**3. Noise and Stochastic Fields**

Fundamental noise operators for variation and procedural textures:

| Operator | Description |
|----------|-------------|
| `noise.perlin2d/3d` | Classic Perlin noise |
| `noise.simplex2d/3d` | Simplex noise (faster, better isotropy) |
| `noise.fbm` | Fractal Brownian Motion (multi-octave) |
| `noise.worley` | Worley/cellular noise (Voronoi) |
| `noise.curl3d` | Curl noise (divergence-free vector field) |
| `noise.turbulence` | Turbulence (absolute value noise) |
| `random.distribute` | Distribute points (uniform/poisson_disk/blue_noise) |

**Use Cases:**
- Bark displacement
- Branch curvature variation
- Leaf scatter patterns
- Terrain heightfields
- Wood grain, marble textures
- Forest placement

---

**4. Foliage and Instancing**

Efficient leaf distribution and GPU instancing for vast scenes:

| Operator | Description |
|----------|-------------|
| `foliage.scatter_on_branches` | Scatter leaves on branch endpoints |
| `foliage.align_to_normal` | Align to branch normals |
| `instancing.create` | Create GPU instance collection |
| `instancing.lod` | Level-of-detail management |
| `instancing.cull` | Frustum and occlusion culling |

**Use Cases:**
- Realistic tree foliage
- Dense forests (100k+ instances)
- Real-time rendering optimization

**Performance Target**: 100,000+ tree instances at 60 FPS

---

**5. Materials and Palettes**

Procedural texture and color generation:

| Operator | Description |
|----------|-------------|
| `palette.gradient` | Define color gradient |
| `palette.map_height` | Map height to color |
| `palette.map_curvature` | Highlight crevices, edges |
| `material.bark` | Procedural bark texture |
| `material.stone` | Stone textures (granite, marble, sandstone) |
| `material.wood` | Wood grain patterns |
| `material.pbr` | PBR material from maps |

**Use Cases:**
- Realistic tree bark
- Autumn color gradients by height
- Procedural rock formations
- Material variation

---

**6. Terrain Generation**

Heightfield-based landscapes with erosion and biomes:

| Operator | Description |
|----------|-------------|
| `terrain.fractal` | Fractal terrain (multi-octave noise) |
| `terrain.erode` | Hydraulic/thermal erosion simulation |
| `terrain.biome_map` | Moisture + temperature → biome distribution |
| `terrain.to_mesh` | Heightfield → geometry |
| `vegetation.distribute` | Place vegetation by biome/slope/moisture |

**Use Cases:**
- Realistic terrains with valleys and rivers
- Biome-based vegetation placement
- Landscape ecology modeling
- Game world generation

**Critical Feature**: Erosion simulation is **essential** for realistic terrains (valleys, rivers, weathering).

---

**7. Wind and Animation**

Dynamic tree animation and seasonal changes:

| Operator | Description |
|----------|-------------|
| `wind.simple_sway` | Sinusoidal sway |
| `wind.turbulent` | Noise-based turbulent wind |
| `wind.branch_weighting` | Mass-based bending |
| `wind.from_fluid` | Use FluidDomain wind simulation |
| `seasonal.color_transition` | Animate seasonal colors |
| `seasonal.leaf_fall` | Simulate autumn leaf falling |

**Use Cases:**
- Animated forests
- Physics-accurate wind bending
- Seasonal transitions (spring → autumn)
- Acoustic wind interaction

---

**8. Architectural and Urban**

Procedural cities and buildings:

| Operator | Description |
|----------|-------------|
| `urban.road_network` | Generate road graphs (L-system/tensor/grid) |
| `urban.lot_subdivision` | Subdivide blocks into lots |
| `urban.procedural_building` | Generate building on lot |
| `arch.facade` | Parametric building facades |
| `urban.traffic_sim` | Simulate traffic flow |

**Use Cases:**
- Procedural city generation
- Urban planning optimization
- Traffic + acoustic simulation
- Architectural parametric design

---

#### Cross-Domain Integration

**Procedural → Geometry (Trees → Meshes)**
```morphogen
// L-system → branch tree → mesh → solid geometry
let tree_string = lsystem.expand(grammar, iterations=7)
let branches = branches.from_lsystem(tree_string, angle=22.5deg)
let mesh = branches.to_mesh(branches, bark_texture=material.bark())
let solid = geom.from_mesh(mesh)  // GeometryDomain
```

**Procedural → Physics (Wind Simulation)**
```morphogen
// Generate forest, simulate wind forces
let forest = vegetation.distribute(terrain, species=[oak, pine], count=10000)
let wind_field = fluid.wind(velocity=vec3(10, 0, 0))  // FluidDomain
forest = vegetation.apply_wind(forest, wind_field)   // ProceduralDomain
```

**Procedural → Acoustics (Forest Scattering)**
```morphogen
// Forest scatters sound waves
let forest = vegetation.distribute(terrain, count=5000)
let source = acoustic.point_source(position=vec3(0, 2, 0))
let scattered = acoustic.forest_scattering(source, trees=forest, freq_range=(100Hz, 8kHz))
out audio = acoustic.listener(scattered, position=vec3(100, 2, 0))
```

**Procedural + Physics + Geometry (Structural Optimization)**
```morphogen
// Optimize tree growth for structural integrity
let tree = growth.space_colonization(root, attractors, iterations=200)
tree = growth.gravity(tree, stiffness=0.1)  // Bend under weight
let stress = physics.stress_test(tree, wind_load=100N)  // PhysicsDomain
tree = branches.reinforce(tree, stress_threshold=10MPa)  // Strengthen weak branches
```

**Procedural + Terrain + Ecology + Acoustics (Complete World)**
```morphogen
// Generate world: terrain → erosion → biomes → vegetation → wind → acoustics
let terrain = terrain.fractal(size=1000m, octaves=8, seed=42)
terrain = terrain.erode(terrain, type="hydraulic", iterations=200)
let biomes = terrain.biome_map(terrain, moisture_scale=150, seed=42)
let vegetation = vegetation.distribute(terrain, biomes, species=[oak, pine], seed=42)
let wind = fluid.wind(velocity=vec3(5, 0, 0))
vegetation = vegetation.apply_wind(vegetation, wind)
let acoustic_field = acoustic.forest_scattering(point_source, vegetation)
out audio = acoustic.binaural_forest(acoustic_field, listener_pos)
out visual = visual.render_terrain(terrain, vegetation, camera)
```

---

#### Dependencies

- **EmergenceDomain** — Basic L-systems (ProceduralDomain extends these)
- **FieldDomain** — Noise functions, heightfields
- **GeometryDomain** — Splines, meshes, solid modeling
- **StochasticDomain** — Seeded RNG for variation
- **PhysicsDomain** — Stress analysis, wind forces, gravity
- **FluidDomain** — Wind simulation for realistic animation
- **AcousticsDomain** — Forest acoustic scattering
- **MaterialDomain** — PBR textures, procedural materials
- **OptimizationDomain** — Structural optimization, growth tuning

---

#### Unique Capabilities (vs. SpeedTree, Houdini, Blender)

**Existing tools:**
- **SpeedTree:** Vegetation only, offline tool, no physics integration, expensive ($$$)
- **Houdini:** Powerful but non-deterministic, steep learning curve, expensive
- **Blender Geometry Nodes:** Great for modeling, not physics-integrated
- **Unity/Unreal ProBuilder:** Game-specific, not general-purpose
- **Substance Designer:** Materials only, not full 3D generation

**Morphogen ProceduralDomain:**
1. ✅ **Trees + terrains + cities + materials** in one unified system
2. ✅ **Cross-domain integration** — Geometry, Physics, Audio, Optimization, Fluids
3. ✅ **GPU acceleration** — Real-time generation (100k+ instances)
4. ✅ **Deterministic execution** — Reproducible results with seeds
5. ✅ **MLIR-based compilation** — JIT to CPU/GPU
6. ✅ **Type + unit safety** — Physical units tracked (m, deg, kg)
7. ✅ **YAML declarative syntax** — Simple, human-readable
8. ✅ **Physics-based growth** — Light, gravity, structural optimization
9. ✅ **Open and extensible** — Not proprietary like SpeedTree

**No competitor offers this.**

---

#### Example Applications

**1. SpeedTree-Quality Birch Tree**
```morphogen
// 20 lines of YAML → production-ready tree
procedural:
  - id: birch_tree
    steps:
      - lsystem.expand: {axiom: "A", rules: {A: "AB", B: "A[+A][-A]"}, iterations: 7}
      - branches.from_lsystem: {angle: 22.5deg, step_size: 0.5m}
      - branches.randomize_angles: {noise: simplex3d, amplitude: 10deg}
      - branches.prune: {max_depth: 12}
      - growth.tropism: {direction: vec3(0, 1, 0), weight: 0.2}  # Grow upward
      - growth.gravity: {stiffness: 0.1}  # Bend under weight
      - branches.to_mesh: {bark_texture: material.bark(scale=2.0)}
      - foliage.scatter_on_branches: {density: 0.8, leaf_mesh: geom.plane(0.1m)}
      - palette.map_height: {gradient: autumn_colors, height_range: (0m, 15m)}
      - wind.simple_sway: {strength: 0.5, frequency: 0.5Hz}
```

**2. Realistic Terrain with Ecosystem**
```morphogen
// Terrain → erosion → biomes → 10,000 trees
let terrain = terrain.fractal(size=2000m, octaves=8, seed=42)
terrain = terrain.erode(terrain, type="hydraulic", iterations=200)
let biomes = terrain.biome_map(terrain, seed=42)
let forest = vegetation.distribute(terrain, biomes, species=[oak, pine], count=10000)
```

**3. Physics-Based Tree Growth**
```morphogen
// Space colonization → structural analysis → reinforcement
let tree = growth.space_colonization(root, attractors, iterations=200)
tree = growth.tropism(tree, direction=light_direction, weight=0.3)
tree = growth.gravity(tree, stiffness=0.1)
let stress = physics.stress_test(tree, wind_load=100N)
tree = branches.reinforce(tree, stress_threshold=10MPa)
```

**4. Procedural City with Acoustics**
```morphogen
// Roads → buildings → traffic → acoustic noise map
let roads = urban.road_network(bounds, density=0.1, seed=42)
let buildings = urban.procedural_building(lots, style="commercial")
let traffic = urban.traffic_sim(roads, vehicles=1000)
let noise_map = acoustic.city_noise(traffic, buildings)
out audio = acoustic.binaural_city(noise_map, listener_pos)
```

**5. Autumn Forest Animation**
```morphogen
// Seasonal colors + wind + leaf fall
scene AutumnForest {
    step(t: Time) {
        let season = (t / 365days) % 1.0  // 0=spring, 0.5=autumn
        forest = seasonal.color_transition(forest, season, palettes)
        let (forest, falling_leaves) = seasonal.leaf_fall(forest, season)
        forest = wind.turbulent(forest, base_vel=vec3(5, 0, 0), time=t)
        out visual = visual.render_forest(forest, falling_leaves)
    }
}
```

---

#### Implementation Roadmap

**Phase 1: Foundation (v0.11)**
- [ ] Spline types (Bezier, Catmull-Rom, B-spline)
- [ ] Noise operators (perlin, simplex, fbm, worley, curl)
- [ ] L-systems (define, expand, parametric, stochastic)
- [ ] Branching (from_lsystem, to_mesh, randomize, prune)
- [ ] Space colonization algorithm
- **Deliverable:** "Morphogen Trees v1" demo (basic but functional)

**Phase 2: Production Quality (v0.12)**
- [ ] Growth algorithms (tropism, gravity, collision_avoidance)
- [ ] Foliage (scatter, instancing, LOD)
- [ ] Materials (bark, stone, wood, PBR)
- [ ] Terrain (fractal, erode, biome_map, vegetation.distribute)
- **Deliverable:** "Morphogen Trees v2" demo (SpeedTree-quality) + "Morphogen Terrain" demo

**Phase 3: Animation and Urban (v1.0)**
- [ ] Wind (simple_sway, turbulent, from_fluid)
- [ ] Seasonal (color_transition, leaf_fall)
- [ ] Urban (roads, buildings, traffic, facades)
- [ ] Cross-domain examples (10+ complete workflows)
- **Deliverable:** "Morphogen Procedural World" demo (trees + terrain + city + acoustics)

---

#### Testing Strategy

**Determinism Tests:**
```morphogen
// Grammar expansion must be bit-exact
let tree1 = lsystem.expand(grammar, iterations=7, seed=42)
let tree2 = lsystem.expand(grammar, iterations=7, seed=42)
assert_eq!(tree1, tree2)

// Noise must be deterministic
let n1 = noise.perlin3d(position=vec3(1, 2, 3), seed=42)
let n2 = noise.perlin3d(position=vec3(1, 2, 3), seed=42)
assert_eq!(n1, n2)
```

**Visual Regression Tests:**
```morphogen
// Render tree and compare to reference image
let tree = procedural.birch_tree(seed=42)
let rendered = visual.render(tree, camera=ref_camera)
assert_image_similar(rendered, "ref_birch_tree.png", threshold=0.95)
```

**Performance Tests:**
```morphogen
// 100k instances should render at 60 FPS
let forest = instancing.create(oak_mesh, count=100000)
let fps = visual.benchmark(forest, duration=10s)
assert!(fps >= 60)
```

---

#### Documentation

- **[specifications/procedural-generation.md](../specifications/procedural-generation.md)** — Complete domain specification
- **[ADR-008](../adr/008-procedural-generation-domain.md)** — Architectural decision record
- **[reference/procedural-operators.md](../reference/procedural-operators.md)** — Full operator catalog (72 operators)

---

#### Why This is a Perfect Fit for Morphogen

**SpeedTree proves the market** — It's the industry-standard vegetation tool used in AAA games and films, and it costs $$$. But it's limited to trees and offline workflows.

**Morphogen ProceduralDomain can be "SpeedTree+++":**
1. **All of SpeedTree's features** (L-systems, branching, foliage, wind, LOD)
2. **Plus terrain generation** (fractal, erosion, biomes)
3. **Plus urban/architectural** (cities, buildings, roads)
4. **Plus true physics integration** (structural optimization, wind simulation, gravity bending)
5. **Plus cross-domain workflows** (trees + terrain + ecology + acoustics + optimization)
6. **Plus GPU real-time generation** (100k+ instances)
7. **Plus deterministic, reproducible results** (seeded RNG)
8. **Plus open and extensible** (not proprietary)

**The killer feature**: No existing tool unifies procedural generation + physics + audio + geometry + optimization in one deterministic, GPU-accelerated platform.

**Market opportunity:**
- Game studios need procedural tools → Morphogen can be open alternative to expensive tools
- Film industry needs large-scale environments → Morphogen can generate massive scenes
- Architecture firms need parametric design → Morphogen offers programmable parametric workflows
- Generative artists need creative tools → Morphogen unifies art + physics + sound

This makes Morphogen the **universal platform for procedural content creation and world building**.

---

## 3. Advanced Domains (FUTURE EXPANSION)

These are "Version 2+" ideas — realistic but not urgent. They represent specialized use cases that extend Morphogen into new application areas.

---

### 3.1 Neural Operators

**Purpose**: Neural fields, neural spectral transforms, learned PDE solvers.

**Why Interesting**: Not a "deep learning framework" — but neural fields (e.g., NeRF, SDF) and neural operators (e.g., Fourier Neural Operators) fit naturally into Morphogen's field/transform model.

**Status**: 🔲 Research (v1.0+)

**Operators**:
- `mlp_field` — Neural SDF / occupancy field
- `neural_spectral` — Learned spectral transform
- `fno` — Fourier Neural Operator
- `neural_codec` — Learned audio/image compression

**Dependencies**: Autodiff, Optimization, Transform

**Use Cases**: Physics-informed ML, learned simulation, neural rendering

---

### 3.2 Probabilistic Programming

**Purpose**: Bayesian inference, sequential Monte Carlo, probabilistic models.

**Why Interesting**: Natural extension of stochastic + autodiff for probabilistic reasoning.

**Status**: 🔲 Research (v1.0+)

**Operators**:
- `sample(model)` — Sample from probabilistic model
- `condition(var, obs)` — Condition on observation
- `metropolis_step` — Metropolis-Hastings MCMC step
- `hmc_step` — Hamiltonian Monte Carlo step
- `smc.resample` — Sequential Monte Carlo resampling

**Dependencies**: Stochastic, Autodiff

**Use Cases**: Bayesian parameter estimation, uncertainty quantification, generative models

---

### 3.3 Control & Robotics

**Purpose**: Control theory operators, trajectory optimization, kinematics/dynamics.

**Why Interesting**: Morphogen's deterministic semantics make it ideal for robotic control.

**Status**: 🔲 Research (v1.1+)

**Operators**:
- `pid` — PID controller
- `mpc` — Model Predictive Control
- `trajectory.optimize` — Trajectory optimization
- `kinematics.solve` — Inverse kinematics
- `robot.dynamics` — Rigid body dynamics

**Dependencies**: Fields, Integrators, Geometry, Optimization

**Use Cases**: Drone control, robotic manipulation, motion planning

---

### 3.4 Discrete Event Simulation

**Purpose**: Agent-based discrete event systems (queues, networks, processes).

**Why Interesting**: Morphogen's event model already supports sample-accurate scheduling; extending to discrete event simulation is straightforward.

**Status**: 🔲 Research (v1.1+)

**Operators**:
- `queue.process` — Process queue events
- `event.route` — Route events through network
- `network.simulate` — Simulate packet routing

**Dependencies**: Stochastic (for arrival processes), Graph (for network topology)

**Use Cases**: Network simulation, supply chain modeling, epidemiology

---

## 4. Domains We Probably Won't Build

For completeness, here are domains that don't align with Morphogen's mission as a **semantic transform kernel**:

- **Database / Tabular** — SQL-like queries, relational algebra (better served by databases)
- **Natural Language** — Text processing, parsing, LLMs (orthogonal to Morphogen's focus)
- **Cryptography** — Hashing, encryption, signatures (security-critical, specialized)
- **Blockchain Consensus** — Proof-of-work, Byzantine agreement (niche application)
- **GUI Rendering** — Widget layout, event handling (UI frameworks handle this)

These are better addressed by specialized tools. Morphogen focuses on **numerical computation, simulation, and creative coding**.

---

## Summary: Full Domain Spectrum

Here is the likely full spectrum of domains Morphogen will eventually want:

### 1. Core (Must-Have) — v0.7-v0.8
| Domain | Status | Priority |
|--------|--------|----------|
| Transform | ✅ Partial | P0 |
| Stochastic | ⚙️ In Progress | P0 |
| Fields / PDE | ✅ Partial | P0 |
| Integrators | 🔲 Planned | P0 |
| Particles | ⚙️ In Progress | P0 |
| Audio DSP | ✅ Partial | P0 |
| Visual / Scene | ⚙️ In Progress | P0 |

### 2. Next Wave (Highly Likely) — v0.9-v1.0
| Domain | Status | Priority |
|--------|--------|----------|
| Geometry/Mesh | 🔲 Planned | P1 |
| Sparse Linear Algebra | 🔲 Planned | P1 |
| Optimization | 🔲 Planned | P1 |
| Autodiff | 🔲 Planned | P1 |
| Graph/Network | 🔲 Planned | P1 |
| Image/Vision | 🔲 Planned | P1 |
| Instrument Modeling | 🔲 Planned | P1 |
| Symbolic/Algebraic | 🔲 Planned | P2 |
| I/O & Storage | 🔲 Planned | P1 |
| Fluid Dynamics | 🔲 Planned | P1 |
| Acoustics | 🔲 Planned | P1 |
| **Emergence (CA, ABM, RD, L-Systems)** | **🔲 Proposed** | **P1** |

### 3. Advanced Future — v1.1+
| Domain | Status | Priority |
|--------|--------|----------|
| Neural Operators | 🔲 Research | P3 |
| Probabilistic Programming | 🔲 Research | P3 |
| Control & Robotics | 🔲 Research | P3 |
| Discrete Event Simulation | 🔲 Research | P3 |

**Legend**:
- ✅ Partial: Implemented but incomplete
- ⚙️ In Progress: Active development
- 🔲 Planned: Design phase
- 🔲 Research: Exploratory

---

## Design Principles

All Morphogen domains adhere to these principles:

1. **Deterministic by Default** — Operations are reproducible unless explicitly marked `@nondeterministic`
2. **Type + Unit Safe** — Physical units are tracked and validated at compile time
3. **Multirate Scheduling** — Different domains can run at different rates (audio, control, visual)
4. **GPU/CPU Pluggable** — Operations lower to MLIR and can run on any backend
5. **Minimal, Sharply Defined** — Each domain has a focused scope; lower to standard dialects ASAP
6. **Extensible** — New operators can be added without breaking existing code

---

## Integration Example: Multi-Domain Simulation

A realistic Morphogen program using multiple domains:

```morphogen
scene FluidWithParticles {
  // Fields: Velocity and pressure
  let velocity: Field2D<Vec2<m/s>> = field.create(512, 512, Vec2(0, 0))
  let pressure: Field2D<Pa> = field.create(512, 512, 0Pa)

  // Agents: Particles advected by fluid
  let particles: Agents<{pos: Vec2<m>, color: Vec3}> = agent.create(1000)

  step(dt: Time) {
    // Stochastic: Add random force
    let force_field = stochastic.perlin_noise(velocity.shape, seed=42)

    // Fields: Advect, diffuse, project velocity
    velocity = field.advect(velocity, velocity, dt, method="BFECC")
    velocity = field.diffuse(velocity, viscosity=0.01, dt, solver="CG")
    velocity = field.project(velocity, dt, solver="multigrid")

    // Particles: Update positions from velocity field
    particles = agent.from_field(particles, velocity, "velocity")
    particles = agent.integrate(particles, dt, method="RK4")

    // Image: Render particles to field
    let density = agent.to_field(particles, field.shape, "density")

    // Visual: Colorize and render
    let color_field = palette.apply(density, palette="viridis")
    out visual = render.field(color_field)

    // Audio: Sonify pressure field
    let pressure_sample = field.reduce(pressure, "mean")
    let tone = osc.sine(pressure_sample * 100Hz)
    out audio = tone
  }
}
```

**Domains Used**:
1. **Fields** — Fluid velocity and pressure
2. **Stochastic** — Perlin noise forcing
3. **Particles** — Advected by fluid
4. **Integrators** — RK4 time-stepping
5. **Image** — Particle-to-field rasterization
6. **Visual** — Palette mapping and rendering
7. **Audio** — Sonification via oscillator

This demonstrates Morphogen's **cross-domain composability** — all domains share the same type system, scheduler, and MLIR backend.

---

## Roadmap Implications

### v0.8 (Current → Next Release)
- **Complete Core Domains**: Finish Stochastic, Integrators, Particles
- **MLIR Lowering**: All core dialects lower to LLVM/GPU
- **Conformance Tests**: Determinism guarantees for all core ops

### v0.9-v0.10 (Next Wave Phase 1)
- **Add**: Geometry/Mesh, Sparse Linear Algebra, I/O & Storage, Instrument Modeling
- **Focus**: 3D simulation, large-scale PDEs, asset loading, timbre extraction & synthesis

### v1.0 (Next Wave Phase 2)
- **Add**: Optimization, Autodiff, Graph/Network, Image/Vision
- **Focus**: Differentiable programming, ML integration, vision pipelines

### v1.1+ (Advanced Domains)
- **Explore**: Neural Operators, Probabilistic Programming, Control/Robotics
- **Focus**: Research applications, novel use cases

---

## Cross-Cutting Concerns

### Determinism Across Domains
All domains support three determinism tiers:
1. **Strict** — Bit-identical (e.g., `field.diffuse`, `agent.force_sum` with deterministic methods)
2. **Reproducible** — Deterministic within precision (e.g., iterative solvers)
3. **Nondeterministic** — External I/O or adaptive termination (e.g., `io.stream(live)`)

### MLIR Dialect Strategy
- **Domain-Specific Dialects**: morphogen.stream, morphogen.field, morphogen.transform, morphogen.schedule
- **Lower to Standard Dialects ASAP**: linalg, affine, vector, arith, math, scf, memref
- **Backend Dialects**: llvm (CPU), gpu (CUDA/ROCm), spirv (Vulkan)

See `../specifications/mlir-dialects.md` for current dialect definitions.

### GPU Acceleration
All domains follow Morphogen's GPU lowering principles:
- Structured parallelism (explicit iteration spaces)
- Memory hierarchy management (global/shared/register)
- Static shape preference
- Warp-friendly execution
- Deterministic GPU semantics

See `gpu-mlir-principles.md` for details.

---

## Conclusion

Morphogen's domain architecture is designed for **long-term extensibility** while maintaining **core simplicity**. By focusing on:
- Deterministic semantics
- Type + unit safety
- Multirate scheduling
- MLIR-based compilation
- GPU/CPU portability

...we create a foundation that naturally supports audio, graphics, physics, AI, and beyond — all in a single unified system.

This document will evolve as new domains are designed, prototyped, and integrated. It serves as both a **vision** and a **contract**: every domain must justify its existence and integrate coherently with the rest of the system.

---

## References

### Core Specifications
- **../specifications/mlir-dialects.md** — Current dialect definitions (morphogen.stream, morphogen.field, morphogen.transform, morphogen.schedule)
- **overview.md** — Overall system architecture
- **gpu-mlir-principles.md** — GPU lowering design rules
- **../specifications/type-system.md** — Type system and unit tracking
- **../specifications/scheduler.md** — Multirate scheduling semantics
- **../specifications/operator-registry.md** — Operator metadata and registration
- **../specifications/coordinate-frames.md** — Unified frame and anchor system
- **../specifications/geometry.md** — Geometry domain specification (TiaCAD patterns)
- **../specifications/timbre-extraction.md** — Timbre extraction and instrument modeling specification
- **../specifications/physics-domains.md** — Physics domains for engineering modeling (FluidNetwork, ThermalODE, FluidJet, CombustionLight)

### Architectural Decision Records
- **../adr/001-unified-reference-model.md** — Decision on unified reference system
- **../adr/002-cross-domain-architectural-patterns.md** — Patterns from TiaCAD, RiffStack, and Strudel
- **../adr/004-instrument-modeling-domain.md** — Instrument modeling domain decision

### Implementation Guides
- **../guides/domain-implementation.md** — Step-by-step domain implementation guide
- **../reference/operator-registry-expansion.md** — Detailed operator catalogs for 8 priority domains (including InstrumentModeling)

### Examples & Case Studies
- **../examples/j-tube-firepit-multiphysics.md** — J-tube fire pit as multi-physics design example (validates physics domains)
- **../examples/README.md** — Comprehensive guide to Morphogen examples and case studies

---

**End of Document**
