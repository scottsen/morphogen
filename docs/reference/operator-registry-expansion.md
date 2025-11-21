# Morphogen Operator Registry Expansion: Seven Domains

**Version:** 1.0
**Date:** 2025-11-15
**Status:** Planning Document
**Related:** ADR-002, ../architecture/domain-architecture.md, DOMAIN_IMPLEMENTATION_GUIDE.md

---

## Overview

This document catalogs **seven priority domains** for Morphogen operator registry expansion, based on architectural learnings from:
- **TiaCAD** (geometry/CAD domain)
- **RiffStack** (audio DSP domain)
- **Strudel/TidalCycles** (pattern/sequencing domain)
- **Cross-domain composition** patterns

Each domain follows the unified architecture defined in **ADR-002**:
- ✅ Clean operator registry
- ✅ Unified reference types
- ✅ Self-contained builder pipelines
- ✅ Consistent execution model
- ✅ 4-layer operator hierarchy
- ✅ Domain-specific passes

---

## Domain Priority Matrix

| Domain | Priority | Complexity | Dependencies | Target Version |
|--------|----------|------------|--------------|----------------|
| **1. AudioDomain** | **P0** | Moderate | None | v0.8 (partial), v0.9 (complete) |
| **2. PhysicsDomain** | **P0** | High | CoordinateFrames | v0.9 |
| **3. GeometryDomain** | **P1** | High | CoordinateFrames | v0.8 (partial), v0.9 (complete) |
| **4. FinanceDomain** | **P1** | Moderate | Random, Linalg | v1.0 |
| **5. GraphicsDomain** | **P2** | High | Geometry | v1.0 |
| **6. NeuralDomain** | **P2** | Very High | Linalg, Tensor | v1.1+ |
| **7. PatternDomain** | **P1** | Low | Temporal | v0.9 |
| **8. InstrumentModelingDomain** | **P1** | High | Audio, Transform, Stochastic | v0.9-v1.0 |

**Priority Levels:**
- **P0:** Critical for core Morphogen platform
- **P1:** High value, enables key use cases
- **P2:** Advanced features, ecosystem expansion

---

## 1. AudioDomain (RiffStack → Morphogen.Audio)

### 1.1 Overview

**Purpose:** Real-time audio synthesis, DSP, and effects processing.

**Inspiration:** RiffStack audio framework
- Extensible operator registry
- Node-based audio graph
- Multi-rate scheduling
- Plugin architecture

**Status:** Partially implemented in `morphogen/stdlib/audio.py`
- Current: Oscillators, basic filters
- Missing: Advanced FX, samplers, comprehensive node system

---

### 1.2 Operators

#### Layer 1: Atomic Operators (Generators & Basic DSP)

| Operator | Signature | Description | Determinism |
|----------|-----------|-------------|-------------|
| `sine` | `(freq: Hz, phase: float) → AudioSignal` | Sine wave oscillator | DETERMINISTIC |
| `saw` | `(freq: Hz, phase: float) → AudioSignal` | Sawtooth oscillator | DETERMINISTIC |
| `square` | `(freq: Hz, duty: float) → AudioSignal` | Square wave with PWM | DETERMINISTIC |
| `triangle` | `(freq: Hz, phase: float) → AudioSignal` | Triangle wave | DETERMINISTIC |
| `noise` | `(seed: int, type: WhiteNoise\|PinkNoise) → AudioSignal` | Noise generator | DETERMINISTIC |
| `wavetable` | `(table: Array, freq: Hz, interp: Linear\|Cubic) → AudioSignal` | Wavetable oscillator | DETERMINISTIC |
| `delay_sample` | `(signal: AudioSignal, samples: int) → AudioSignal` | Single-sample delay | DETERMINISTIC |
| `biquad_coeff` | `(type: LPF\|HPF\|BPF, freq: Hz, Q: float) → BiquadCoeffs` | Biquad filter coefficients | DETERMINISTIC |
| `one_pole` | `(signal: AudioSignal, cutoff: Hz) → AudioSignal` | Single-pole filter | DETERMINISTIC |

#### Layer 2: Composite Operators (Filters & Effects)

| Operator | Signature | Description | Composed From |
|----------|-----------|-------------|---------------|
| `lowpass` | `(signal: AudioSignal, cutoff: Hz, Q: float) → AudioSignal` | Lowpass filter | `biquad_coeff` → IIR processor |
| `highpass` | `(signal: AudioSignal, cutoff: Hz, Q: float) → AudioSignal` | Highpass filter | `biquad_coeff` → IIR processor |
| `bandpass` | `(signal: AudioSignal, center: Hz, Q: float) → AudioSignal` | Bandpass filter | `biquad_coeff` → IIR processor |
| `notch` | `(signal: AudioSignal, center: Hz, Q: float) → AudioSignal` | Notch filter | `biquad_coeff` → IIR processor |
| `shelving` | `(signal: AudioSignal, freq: Hz, gain: dB, type: Low\|High) → AudioSignal` | Shelving EQ | `biquad_coeff` → IIR processor |
| `envelope_adsr` | `(attack: s, decay: s, sustain: 0-1, release: s, gate: bool) → EnvSignal` | ADSR envelope | Piecewise linear segments |
| `lfo` | `(rate: Hz, shape: Sine\|Triangle\|Square, depth: float) → ModSignal` | Low-frequency oscillator | Oscillator + range mapping |
| `delay_line` | `(signal: AudioSignal, delay: s, feedback: 0-1) → AudioSignal` | Feedback delay | `delay_sample` + feedback loop |

#### Layer 3: Constructs (Complex Effects & Synthesizers)

| Operator | Signature | Description | Use Case |
|----------|-----------|-------------|----------|
| `reverb_schroeder` | `(signal: AudioSignal, room_size: m, damping: 0-1) → AudioSignal` | Schroeder reverb (4 comb + 2 allpass) | Room simulation |
| `reverb_freeverb` | `(signal: AudioSignal, room_size: 0-1, damping: 0-1, wet: 0-1) → AudioSignal` | Freeverb algorithm | High-quality reverb |
| `delay_stereo` | `(signal: AudioSignal, time_L: s, time_R: s, feedback: 0-1) → AudioSignal[2]` | Stereo ping-pong delay | Spatial effects |
| `chorus` | `(signal: AudioSignal, rate: Hz, depth: ms, voices: int) → AudioSignal` | Chorus effect (multi-voice LFO delay) | Thickening |
| `flanger` | `(signal: AudioSignal, rate: Hz, depth: ms, feedback: 0-1) → AudioSignal` | Flanging (short delay + feedback) | Jet effect |
| `phaser` | `(signal: AudioSignal, rate: Hz, depth: 0-1, stages: int) → AudioSignal` | Phaser (allpass filter modulation) | Phase shifting |
| `compressor` | `(signal: AudioSignal, threshold: dB, ratio: float, attack: ms, release: ms) → AudioSignal` | Dynamics compressor | Level control |
| `limiter` | `(signal: AudioSignal, threshold: dB, release: ms) → AudioSignal` | Brick-wall limiter | Peak protection |
| `distortion` | `(signal: AudioSignal, drive: 0-1, type: Soft\|Hard\|Tube) → AudioSignal` | Nonlinear distortion | Saturation |
| `sampler` | `(sample: AudioBuffer, trigger: bool, pitch: semitones) → AudioSignal` | Sample playback engine | Drums, instruments |
| `subtractive_synth` | `(freq: Hz, filter_cutoff: Hz, resonance: 0-1, envelope: ADSR) → AudioSignal` | Classic subtractive synthesis | Analog-style sounds |

#### Layer 4: Presets (Pre-Configured Chains)

| Preset | Description | Configuration |
|--------|-------------|---------------|
| `vocal_chain` | Vocal processing (EQ + comp + reverb) | HPF 80Hz → Comp 4:1 → Reverb small room |
| `mastering_chain` | Mastering effects (EQ + multiband comp + limiter) | 3-band EQ → Multiband comp → Limiter -0.1dB |
| `guitar_amp` | Guitar amplifier simulation | Preamp → Distortion → Tone stack → Cabinet IR |
| `808_kick` | TR-808 kick drum | Sine + pitch envelope + distortion |
| `synth_lead` | Lead synthesizer preset | Saw oscillator → LP filter + resonance → delay |
| `pad_lush` | Ambient pad texture | Detuned saws → chorus → reverb (long tail) |

---

### 1.3 Reference Types

#### Primary: `NodeRef`

**Purpose:** Reference to an audio processing node in the graph.

**Auto-Anchors:**
```python
NodeRef.input[n: int] → PortRef           # Input port #n
NodeRef.output[n: int] → PortRef          # Output port #n
NodeRef.param[name: str] → ParamRef       # Named parameter (e.g., "freq", "Q")
NodeRef.sample_rate → int                 # Node's sample rate
NodeRef.block_size → int                  # Processing block size
```

**Example:**
```python
osc = sine(freq=440)
filter = lowpass(cutoff=1000, Q=0.707)

# Port-based patching
osc.output[0] >> filter.input[0]

# Parameter modulation
lfo = lfo(rate=2.0, shape="sine")
lfo.output[0] >> filter.param["cutoff"]
```

#### Secondary: `PortRef`

**Purpose:** Reference to an audio/control port.

**Types:**
- `AudioPortRef`: Full-rate audio (e.g., 48kHz)
- `ControlPortRef`: Control-rate signals (e.g., 1kHz for modulation)

**Auto-Anchors:**
```python
PortRef.sample_rate → int
PortRef.connected_to → NodeRef | None
PortRef.data_type → AudioSignal | ControlSignal
```

#### Tertiary: `GraphRef`

**Purpose:** Reference to the entire audio processing graph.

**Auto-Anchors:**
```python
GraphRef.input_nodes → List[NodeRef]       # Graph inputs
GraphRef.output_nodes → List[NodeRef]      # Graph outputs
GraphRef.topological_order → List[NodeRef] # Execution order
```

---

### 1.4 Passes

#### Validation Passes
- **SampleRateConsistency**: Ensure all connected nodes have compatible sample rates
- **PortTypeCheck**: Validate audio ports connect to audio, control to control
- **CycleDetection**: Ensure no feedback loops without explicit delay

#### Optimization Passes
- **FilterMerging**: Combine cascaded biquad filters into higher-order filters
- **NodePruning**: Remove disconnected nodes
- **GraphFlattening**: Inline single-use subgraphs
- **ConstantFolding**: Pre-compute constant parameters

#### Lowering Passes
- **AudioGraphToSCF**: Convert node graph → for loops + arithmetic
- **VectorizedDSP**: SIMD optimization for filter banks
- **CUDALowering**: GPU kernels for large parallel effect chains

---

### 1.5 Implementation Status

**Completed (v0.7):**
- ✅ Basic oscillators (sine, saw, square)
- ✅ Simple filters (one-pole)
- ✅ MLIR dialect definition
- ✅ Initial audio graph IR

**TODO (v0.8-v0.9):**
- ⬜ Complete Layer 2 filters (biquad, shelving)
- ⬜ Layer 3 effects (reverb, chorus, compressor)
- ⬜ Sampler implementation
- ⬜ Port-based patching system
- ⬜ Graph optimization passes
- ⬜ Multi-rate scheduling integration

---

## 2. PhysicsDomain (N-Body, Integrators, Forces)

### 2.1 Overview

**Purpose:** Simulate physical systems (particles, rigid bodies, forces, collisions).

**Key Applications:**
- N-body gravitational simulations (solar systems, galaxies)
- Molecular dynamics
- Rigid body physics (game engines, robotics)
- Particle systems (fluids, crowds)

**Status:** Not yet implemented

---

### 2.2 Operators

#### Layer 1: Atomic Operators

| Operator | Signature | Description | Complexity |
|----------|-----------|-------------|------------|
| `gravity_force_pair` | `(body1: BodyRef, body2: BodyRef, G: float) → Force` | Pairwise gravitational force | O(1) |
| `electrostatic_force` | `(charge1: float, charge2: float, pos1: Vec3, pos2: Vec3, k: float) → Force` | Coulomb force | O(1) |
| `spring_force` | `(pos1: Vec3, pos2: Vec3, rest_length: m, k: N/m) → Force` | Hooke's law spring | O(1) |
| `drag_force` | `(velocity: Vec3, drag_coeff: float, area: m², density: kg/m³) → Force` | Fluid drag (quadratic) | O(1) |
| `euler_step` | `(state: State, force: Force, dt: s) → State` | Euler integrator (1st order) | O(1) |
| `verlet_step` | `(state: State, force: Force, dt: s) → State` | Verlet integrator (symplectic) | O(1) |
| `rk4_substep` | `(state: State, force_fn: Fn, t: s, dt: s) → State` | Single RK4 substep | O(1) |
| `octree_insert` | `(tree: Octree, body: BodyRef) → Octree` | Insert body into octree | O(log N) |
| `aabb_intersect` | `(box1: AABB, box2: AABB) → bool` | Axis-aligned bounding box test | O(1) |

#### Layer 2: Composite Operators

| Operator | Signature | Description | Composed From |
|----------|-----------|-------------|---------------|
| `barnes_hut_force` | `(body: BodyRef, tree: Octree, theta: float, G: float) → Force` | Barnes-Hut gravity approximation | `octree_traverse` + `gravity_force_pair` |
| `rk4_integrator` | `(state: State, force_fn: Fn, dt: s) → State` | 4th-order Runge-Kutta | 4× `rk4_substep` |
| `yoshida_integrator` | `(state: State, force_fn: Fn, dt: s) → State` | Yoshida 4th-order symplectic | Multiple Verlet steps with weights |
| `leapfrog_integrator` | `(state: State, force_fn: Fn, dt: s) → State` | Leapfrog (symplectic) | `verlet_step` variant |
| `broadphase_grid` | `(bodies: List[BodyRef], cell_size: m) → List[Pair[BodyRef]]` | Grid-based broadphase collision | Spatial hashing + pair enumeration |
| `narrowphase_sat` | `(body1: BodyRef, body2: BodyRef) → Collision | None` | SAT (Separating Axis Theorem) | Convex hull tests |

#### Layer 3: Constructs

| Operator | Signature | Description | Use Case |
|----------|-----------|-------------|----------|
| `n_body_system` | `(bodies: List[BodyRef], integrator: str, forces: List[str]) → System` | Complete N-body simulation | Solar systems, galaxies |
| `rigid_body_dynamics` | `(bodies: List[RigidBody], constraints: List[Constraint]) → System` | Rigid body physics with constraints | Game engines, robotics |
| `particle_system` | `(count: int, emitter: EmitterRef, forces: List[str]) → System` | Particle simulation | Fluids, crowds, fire |
| `collision_detector` | `(bodies: List[BodyRef], broadphase: str, narrowphase: str) → CollisionSet` | Full collision detection pipeline | Physics engines |
| `sph_fluid` | `(particles: List[ParticleRef], kernel: str, viscosity: float) → System` | Smoothed Particle Hydrodynamics | Fluid simulation |

#### Layer 4: Presets

| Preset | Description | Configuration |
|--------|-------------|---------------|
| `solar_system` | Solar system with Sun + 8 planets | N-body + Verlet + realistic masses/distances |
| `molecular_dynamics` | Lennard-Jones molecular system | N-body + velocity Verlet + periodic boundaries |
| `granular_flow` | Granular material (sand, gravel) | Particles + contact forces + friction |
| `game_physics` | Standard game physics (Unity/Unreal style) | Rigid bodies + constraints + broadphase grid |

---

### 2.3 Reference Types

#### Primary: `BodyRef`

**Purpose:** Reference to a physical body (particle or rigid body).

**Auto-Anchors:**
```python
BodyRef.center_of_mass → Vec3              # Position of center of mass
BodyRef.local_axes.x → Vec3                # Local X axis (orientation)
BodyRef.local_axes.y → Vec3                # Local Y axis
BodyRef.local_axes.z → Vec3                # Local Z axis
BodyRef.velocity → Vec3                    # Linear velocity
BodyRef.angular_velocity → Vec3            # Angular velocity (rigid bodies)
BodyRef.collision_normal → Vec3            # Normal at last collision (updated dynamically)
BodyRef.bounding_box → AABB                # Axis-aligned bounding box
```

**Example:**
```python
earth = RigidBody(mass=5.972e24, shape=sphere(radius=6.371e6))
moon = RigidBody(mass=7.342e22, shape=sphere(radius=1.737e6))

# Apply force at center of mass
gravity = gravity_force_pair(earth, moon, G=6.674e-11)
gravity.apply_at(moon.center_of_mass)

# Collision detection
if earth.bounding_box.intersects(moon.bounding_box):
    collision = narrowphase(earth, moon)
    impulse = collision.normal * collision.penetration_depth
```

#### Secondary: `ForceRef`

**Purpose:** Reference to a force field or force application.

**Auto-Anchors:**
```python
ForceRef.magnitude → float                 # Force magnitude (N)
ForceRef.direction → Vec3                  # Unit direction vector
ForceRef.application_point → Vec3          # Point of application
```

#### Tertiary: `IntegratorRef`

**Purpose:** Reference to an integration scheme.

**Auto-Anchors:**
```python
IntegratorRef.timestep → s                 # Integration timestep
IntegratorRef.is_symplectic → bool         # Is symplectic? (energy conservation)
IntegratorRef.order → int                  # Order of accuracy (1, 2, 4, etc.)
```

---

### 2.4 Passes

#### Validation Passes
- **PositiveMassCheck**: Ensure all masses > 0
- **TimeStepStability**: Ensure dt < critical value for integrator
- **ForceSymmetryCheck**: Verify Newton's 3rd law (action-reaction pairs)

#### Optimization Passes
- **SymplecticEnforcement**: Replace Euler → Verlet for Hamiltonian systems
- **SpatialPartitioningOptimization**: Choose octree vs. grid vs. BVH based on density
- **BarnesHutAdaptation**: Auto-tune theta parameter for accuracy/performance trade-off

#### Lowering Passes
- **NBodyToBarnesHut**: O(N²) direct sum → O(N log N) tree code
- **IntegratorToSCF**: Integrator loops → SCF for loops
- **VectorizationPass**: SIMD for force calculations
- **CUDALowering**: GPU kernels for large particle counts (N > 10,000)

---

### 2.5 Implementation Roadmap

**v0.9:**
- ⬜ Layer 1 atomic operators (forces, integrators, spatial partitioning)
- ⬜ BodyRef with auto-anchors
- ⬜ Basic N-body simulation (direct summation)
- ⬜ Euler + Verlet integrators

**v1.0:**
- ⬜ Barnes-Hut tree code
- ⬜ RK4 + Yoshida integrators
- ⬜ Symplectic enforcement pass
- ⬜ Collision detection (broadphase + narrowphase)
- ⬜ GPU acceleration

---

## 3. GeometryDomain (TiaCAD → Morphogen.Geometry)

### 3.1 Overview

**Purpose:** Parametric CAD and computational geometry.

**Inspiration:** TiaCAD v3.x architecture
- SpatialRef unified reference system
- Auto-generated anchors (face_top, center, etc.)
- Sketch constraints
- Boolean operations

**Status:** Partially specified in `docs/../specifications/geometry.md` (RFC)
- Current: Specification complete, implementation in progress
- Missing: Full implementation of all operators

---

### 3.2 Operators

**See `../specifications/geometry.md` for complete operator catalog.**

**Summary:**
- **Layer 1 (Primitives):** sphere, box, cylinder, cone, torus
- **Layer 2 (Transforms):** translate, rotate, scale, mirror, pattern
- **Layer 3 (Boolean):** union, difference, intersection, shell
- **Layer 4 (Advanced):** loft, sweep, revolve, fillet, chamfer

---

### 3.3 Reference Types

**Primary:** `SpatialRef` (defined in `../specifications/coordinate-frames.md`)

**Auto-Anchors:**
```python
SpatialRef.face_top → SpatialRef
SpatialRef.face_bottom → SpatialRef
SpatialRef.center → SpatialRef
SpatialRef.axis_x → SpatialRef
```

**See:** `docs/../specifications/geometry.md` for full specification.

---

### 3.4 Passes

- **Mesh Simplification:** LOD generation
- **Auto-Anchor Generation:** Compute geometric anchors
- **Sketch Constraint Solving:** Solve geometric constraints
- **Boolean Tree Optimization:** CSG tree restructuring

**Implementation Roadmap:** v0.8 (partial), v0.9 (complete)

---

## 4. FinanceDomain (Monte Carlo, Stochastic Processes)

### 4.1 Overview

**Purpose:** Quantitative finance, derivatives pricing, risk simulation.

**Key Applications:**
- Monte Carlo option pricing
- Stochastic process simulation (GBM, Heston, Ornstein-Uhlenbeck)
- Yield curve construction
- VaR (Value-at-Risk) calculation
- Portfolio optimization

**Status:** Not yet implemented

---

### 4.2 Operators

#### Layer 1: Atomic Operators

| Operator | Signature | Description | Determinism |
|----------|-----------|-------------|-------------|
| `gbm_step` | `(S: float, mu: float, sigma: float, dt: s, dW: float) → float` | Geometric Brownian Motion step | DETERMINISTIC (given dW) |
| `ou_step` | `(X: float, theta: float, mu: float, sigma: float, dt: s, dW: float) → float` | Ornstein-Uhlenbeck step | DETERMINISTIC |
| `heston_step` | `(S: float, v: float, params: HestonParams, dt: s, dW: Vec2) → (float, float)` | Heston model step (S, volatility) | DETERMINISTIC |
| `normal_sample` | `(seed: int, count: int) → Array` | Normal random samples N(0,1) | DETERMINISTIC |
| `sobol_sequence` | `(dim: int, count: int, seed: int) → Array` | Quasi-random Sobol sequence | DETERMINISTIC |
| `antithetic_pair` | `(samples: Array) → Array` | Antithetic variate pairs | DETERMINISTIC |
| `call_payoff` | `(S: float, K: float) → float` | Call option payoff max(S-K, 0) | DETERMINISTIC |
| `put_payoff` | `(S: float, K: float) → float` | Put option payoff max(K-S, 0) | DETERMINISTIC |
| `barrier_check` | `(S_path: Array, barrier: float, type: UpOut\|DownOut) → bool` | Check barrier condition | DETERMINISTIC |

#### Layer 2: Composite Operators

| Operator | Signature | Description | Composed From |
|----------|-----------|-------------|---------------|
| `gbm_path` | `(S0: float, mu: float, sigma: float, T: s, steps: int, dW: Array) → Array` | Full GBM path | `gbm_step` × steps |
| `heston_path` | `(S0: float, v0: float, params: HestonParams, T: s, steps: int) → (Array, Array)` | Heston model path | `heston_step` × steps |
| `yield_curve_interp` | `(tenors: Array, rates: Array, t: s, method: Linear\|Cubic) → float` | Yield curve interpolation | Spline interpolation |
| `discount_factor` | `(rate: float, T: s) → float` | Discount factor exp(-r*T) | Exponential |
| `monte_carlo_paths` | `(process: ProcessRef, N: int, seed: int) → Array[N]` | Generate N Monte Carlo paths | Process simulator + RNG |

#### Layer 3: Constructs

| Operator | Signature | Description | Use Case |
|----------|-----------|-------------|----------|
| `european_option_mc` | `(S0: float, K: float, r: float, sigma: float, T: s, N: int, type: Call\|Put) → (float, float)` | European option price + std error | Vanilla options |
| `asian_option_mc` | `(S0: float, K: float, r: float, sigma: float, T: s, N: int, avg_type: Arithmetic\|Geometric) → float` | Asian option (path-dependent) | Exotic options |
| `barrier_option_mc` | `(S0: float, K: float, barrier: float, r: float, sigma: float, T: s, N: int) → float` | Barrier option pricing | Exotic options |
| `var_calculation` | `(portfolio: PortfolioRef, confidence: 0.95\|0.99, horizon: s, N: int) → float` | Value-at-Risk via Monte Carlo | Risk management |
| `cva_calculation` | `(portfolio: PortfolioRef, counterparty: PartyRef, N: int) → float` | Credit Valuation Adjustment | Counterparty risk |

#### Layer 4: Presets

| Preset | Description | Configuration |
|--------|-------------|---------------|
| `black_scholes` | Black-Scholes European option pricer | GBM + analytical formula (benchmark for MC) |
| `monte_carlo_engine` | Full-featured MC engine | Sobol quasi-random + antithetic + control variates |
| `portfolio_risk_suite` | VaR + CVaR + stress testing | Multiple scenarios + risk metrics |

---

### 4.3 Reference Types

#### Primary: `CurveRef`

**Purpose:** Reference to a yield curve, volatility surface, or other market data curve.

**Auto-Anchors:**
```python
CurveRef.point[t: float] → float           # Interpolated value at time t
CurveRef.tenors → Array                    # Tenor points (e.g., [1y, 2y, 5y, 10y])
CurveRef.rates → Array                     # Rate values
CurveRef.expiry → s                        # Maximum tenor (expiry)
```

**Example:**
```python
yield_curve = YieldCurve(
    tenors=[1, 2, 5, 10],  # years
    rates=[0.02, 0.025, 0.03, 0.035]
)

# Interpolate at 3 years
rate_3y = yield_curve.point[3.0]  # → 0.0275 (interpolated)

# Use in pricing
discount = exp(-rate_3y * 3.0)
```

#### Secondary: `MonteCarloRef`

**Purpose:** Reference to a Monte Carlo simulation engine.

**Auto-Anchors:**
```python
MonteCarloRef.paths → Array                # All simulated paths
MonteCarloRef.mean → float                 # Mean of paths
MonteCarloRef.std_error → float            # Standard error
MonteCarloRef.convergence_diagnostics → Diagnostics  # Convergence metrics
```

#### Tertiary: `ScenarioRef`

**Purpose:** Reference to a scenario or shock for stress testing.

**Auto-Anchors:**
```python
ScenarioRef.shocked_params → Dict          # Parameter shocks (e.g., {"sigma": +0.1})
ScenarioRef.base_params → Dict             # Original parameters
```

---

### 4.4 Passes

#### Validation Passes
- **PositiveVolatilityCheck**: Ensure all sigma > 0
- **PositivePriceCheck**: Ensure all asset prices > 0
- **TimeStepStability**: Ensure dt small enough for stiff processes (Heston)

#### Optimization Passes
- **VarianceReduction**: Replace crude MC → antithetic variates
- **QuasiRandomUpgrade**: Replace pseudo-random → Sobol/Halton
- **ControlVariates**: Add control variate adjustment for known benchmarks
- **ImportanceSampling**: Shift distribution for rare events (tail risk)

#### Lowering Passes
- **MonteCarloToVectorized**: Vectorize path generation (NumPy/Linalg)
- **MonteCarloToCUDA**: GPU kernels for N > 100,000 paths
- **AnalyticalFallback**: Use closed-form solutions when available (Black-Scholes)

---

### 4.5 Implementation Roadmap

**v1.0:**
- ⬜ Layer 1 stochastic processes (GBM, OU, Heston)
- ⬜ Random number generation (deterministic, Sobol)
- ⬜ Payoff functions (call, put, barrier)
- ⬜ Basic Monte Carlo engine

**v1.1:**
- ⬜ Variance reduction techniques
- ⬜ GPU acceleration
- ⬜ Yield curve interpolation
- ⬜ Advanced exotics (Asian, barrier, lookback)

---

## 5. GraphicsDomain (Scene Graphs, Rendering)

### 5.1 Overview

**Purpose:** 3D graphics, scene management, rendering pipelines.

**Key Applications:**
- Real-time rendering (games, simulations)
- Visualization (data, geometry, physics)
- Animation systems
- Shader pipelines

**Status:** Not yet implemented

---

### 5.2 Operators

#### Layer 1: Atomic Operators

| Operator | Signature | Description |
|----------|-----------|-------------|
| `scene_node` | `(transform: Mat4, mesh: MeshRef) → NodeRef` | Create scene node |
| `directional_light` | `(direction: Vec3, color: RGB, intensity: float) → LightRef` | Directional light source |
| `point_light` | `(position: Vec3, color: RGB, intensity: float, radius: m) → LightRef` | Point light |
| `perspective_camera` | `(fov: degrees, aspect: float, near: m, far: m) → CameraRef` | Perspective camera |
| `shader_stage` | `(stage: Vertex\|Fragment, source: GLSL) → ShaderStageRef` | Shader stage |
| `texture_2d` | `(width: px, height: px, format: RGBA\|RGB\|Depth) → TextureRef` | 2D texture |

#### Layer 2: Composite Operators

| Operator | Signature | Description |
|----------|-----------|-------------|
| `shader_program` | `(vertex: ShaderStageRef, fragment: ShaderStageRef) → ShaderRef` | Complete shader program |
| `material_pbr` | `(albedo: RGB, metallic: 0-1, roughness: 0-1) → MaterialRef` | PBR material |
| `instanced_mesh` | `(mesh: MeshRef, transforms: Array[Mat4]) → InstancedMeshRef` | Instanced rendering |

#### Layer 3: Constructs

| Operator | Signature | Description |
|----------|-----------|-------------|
| `scene_graph` | `(root: NodeRef, lights: List[LightRef], camera: CameraRef) → SceneRef` | Complete scene graph |
| `particle_renderer` | `(particles: ParticleSystemRef, shader: ShaderRef) → RenderPass` | Particle rendering |
| `post_process_chain` | `(passes: List[PostProcessPass]) → RenderPipeline` | Post-processing effects |

---

### 5.3 Reference Types

#### Primary: `NodeRef`

**Auto-Anchors:**
```python
NodeRef.bounding_box.min → Vec3
NodeRef.bounding_box.max → Vec3
NodeRef.bounding_box.center → Vec3
NodeRef.local_transform → Mat4
NodeRef.world_transform → Mat4              # Accumulated parent transforms
```

---

### 5.4 Passes

- **SceneFlattening**: Deep hierarchy → flat arrays
- **FrustumCulling**: Remove out-of-view objects
- **LODSelection**: Choose level-of-detail based on distance
- **ShaderOptimization**: GLSL AST → SPIR-V

---

## 6. NeuralDomain (ML/AI, Tensor Operations)

### 6.1 Overview

**Purpose:** Neural network construction, training, and inference.

**Status:** Not yet implemented (v1.1+)

---

### 6.2 Operators

#### Layer 1: Atomic Operators

| Operator | Signature | Description |
|----------|-----------|-------------|
| `dense` | `(input: Tensor, weights: Tensor, bias: Tensor) → Tensor` | Fully-connected layer |
| `conv2d` | `(input: Tensor, kernel: Tensor, stride: int, padding: int) → Tensor` | 2D convolution |
| `relu` | `(input: Tensor) → Tensor` | ReLU activation |
| `softmax` | `(input: Tensor, axis: int) → Tensor` | Softmax activation |
| `cross_entropy` | `(pred: Tensor, target: Tensor) → float` | Cross-entropy loss |

#### Layer 2: Composite Operators

| Operator | Signature | Description |
|----------|-----------|-------------|
| `resnet_block` | `(input: Tensor, filters: int) → Tensor` | ResNet residual block |
| `attention` | `(Q: Tensor, K: Tensor, V: Tensor) → Tensor` | Attention mechanism |

#### Layer 3: Constructs

| Operator | Signature | Description |
|----------|-----------|-------------|
| `mlp` | `(layers: List[int], activation: Fn) → ModelRef` | Multi-layer perceptron |
| `transformer` | `(embed_dim: int, num_heads: int, num_layers: int) → ModelRef` | Transformer model |

---

### 6.3 Reference Types

**Primary:** `LayerRef`

**Auto-Anchors:**
```python
LayerRef.weights → TensorRef
LayerRef.biases → TensorRef
LayerRef.activations → TensorRef
LayerRef.gradients → TensorRef
```

---

### 6.4 Passes

- **KernelFusion**: Fuse elementwise ops
- **ConstantFolding**: Pre-compute constant tensors
- **Quantization**: FP32 → INT8
- **GraphOptimization**: Remove dead nodes, merge ops

---

## 7. PatternDomain (Strudel/TidalCycles → Morphogen.Pattern)

### 7.1 Overview

**Purpose:** Temporal pattern generation and sequencing (inspired by live coding systems).

**Inspiration:** Strudel, TidalCycles
- Pattern combinators (slow, fast, repeat)
- Euclidean rhythms
- Polyrhythms and polymeters
- Pattern transformations

**Applications:**
- Audio sequencing (rhythm, melody, harmony)
- Animation sequencing (keyframes, transitions)
- Physics event scheduling (trigger forces at specific times)
- Finance event scheduling (rebalancing, dividend dates)
- UI pattern generators (loading animations, transitions)

**Status:** Not yet implemented

---

### 7.2 Operators

#### Layer 1: Atomic Operators

| Operator | Signature | Description | Determinism |
|----------|-----------|-------------|-------------|
| `cycle` | `(pattern: List[Event], period: s) → PatternRef` | Create repeating cycle | DETERMINISTIC |
| `slow` | `(pattern: PatternRef, factor: float) → PatternRef` | Slow pattern by factor | DETERMINISTIC |
| `fast` | `(pattern: PatternRef, factor: float) → PatternRef` | Speed up pattern | DETERMINISTIC |
| `take` | `(pattern: PatternRef, n: int) → PatternRef` | Take first n events | DETERMINISTIC |
| `drop` | `(pattern: PatternRef, n: int) → PatternRef` | Drop first n events | DETERMINISTIC |
| `repeat` | `(pattern: PatternRef, n: int) → PatternRef` | Repeat pattern n times | DETERMINISTIC |
| `rotate` | `(pattern: PatternRef, offset: int) → PatternRef` | Rotate pattern by offset | DETERMINISTIC |

#### Layer 2: Composite Operators

| Operator | Signature | Description | Composed From |
|----------|-----------|-------------|---------------|
| `euclidean` | `(pulses: int, steps: int) → PatternRef` | Euclidean rhythm E(k, n) | Bjorklund algorithm |
| `polyrhythm` | `(patterns: List[PatternRef]) → PatternRef` | Layer multiple rhythms | Interleave patterns |
| `shuffle` | `(pattern: PatternRef, seed: int) → PatternRef` | Shuffle event order | Deterministic shuffle |
| `arp` | `(notes: List[Note], pattern: PatternRef) → PatternRef` | Arpeggiate notes by pattern | Map pattern to notes |
| `scale_map` | `(degrees: List[int], scale: Scale) → List[Note]` | Map scale degrees to notes | Scale lookup |

#### Layer 3: Constructs

| Operator | Signature | Description | Use Case |
|----------|-----------|-------------|----------|
| `tidal_mini_notation` | `(notation: str) → PatternRef` | Parse TidalCycles mini-notation | Quick pattern entry: `"bd sd [bd bd] sd"` |
| `chord_progression` | `(chords: List[Chord], rhythm: PatternRef) → PatternRef` | Harmonic progression | Music composition |
| `drumkit_pattern` | `(kick: PatternRef, snare: PatternRef, hihat: PatternRef) → PatternRef` | Multi-track drum pattern | Rhythm programming |

#### Layer 4: Presets

| Preset | Description | Configuration |
|--------|-------------|---------------|
| `four_on_floor` | Classic 4/4 kick pattern | `E(4, 16)` on kick |
| `amen_break` | Amen break drum pattern | Iconic breakbeat rhythm |
| `euclidean_poly` | Polyrhythmic Euclidean pattern | `E(3,8)` + `E(5,8)` + `E(7,8)` |

---

### 7.3 Reference Types

#### Primary: `EventRef`

**Purpose:** Reference to a temporal event in a pattern.

**Auto-Anchors:**
```python
EventRef.time → s                          # Event time (absolute or relative)
EventRef.duration → s                      # Event duration
EventRef.value → Any                       # Event payload (note, trigger, etc.)
EventRef.cycle_number → int                # Which cycle this event belongs to
EventRef.beat_number → float               # Position within cycle (0.0 - 1.0)
```

**Example:**
```python
pattern = cycle(["C4", "E4", "G4"], period=1.0)

for event in pattern.events[0:8]:  # First 8 events
    print(f"Note: {event.value}, Time: {event.time}, Beat: {event.beat_number}")
```

#### Secondary: `PatternRef`

**Purpose:** Reference to a complete pattern (sequence of events).

**Auto-Anchors:**
```python
PatternRef.events → List[EventRef]          # All events in pattern
PatternRef.period → s                       # Pattern period (cycle length)
PatternRef.event_count → int                # Number of events per cycle
PatternRef.bpm → float                      # Tempo (beats per minute)
```

---

### 7.4 Passes

#### Validation Passes
- **TemporalConsistency**: Ensure event times are monotonically increasing
- **CyclePeriodCheck**: Ensure cycle periods are positive

#### Optimization Passes
- **EventCoalescing**: Merge adjacent identical events
- **PatternFlattening**: Inline nested pattern references
- **TemporalQuantization**: Snap events to grid (optional, for tight timing)

#### Lowering Passes
- **PatternToEventList**: Convert pattern combinators → explicit event list
- **EventScheduling**: Generate timestamped events for runtime scheduler
- **MIDILowering**: Convert note events → MIDI messages (for audio domain)

---

### 7.5 Implementation Roadmap

**v0.9:**
- ⬜ Layer 1 atomic operators (cycle, slow, fast, take, drop)
- ⬜ EventRef and PatternRef with auto-anchors
- ⬜ Basic pattern evaluation

**v1.0:**
- ⬜ Euclidean rhythms
- ⬜ Pattern combinators (polyrhythm, shuffle, arp)
- ⬜ Mini-notation parser (TidalCycles-style)
- ⬜ Integration with AudioDomain (pattern → audio sequencer)

---

## 8. InstrumentModelingDomain (Timbre Extraction → Synthesis)

### 8.1 Overview

**Purpose:** Analyze acoustic recordings, extract timbre characteristics, and synthesize new notes with the same sonic character.

**Inspiration:**
- Yamaha VL1/VL70m (physical modeling synthesizers)
- Karplus-Strong algorithm (plucked string synthesis)
- Google Magenta NSynth (neural audio synthesis)
- Modal synthesis (IRCAM Modalys)
- Additive resynthesis (SPEAR, AudioSculpt)

**Status:** Planned (v0.9+)
- Design documented in ../specifications/timbre-extraction.md and ADR-003

**Key Capability:** Record acoustic guitar → extract timbre → synthesize new notes

---

### 8.2 Operators

#### Layer 1: Atomic Operators (Analysis Primitives)

| Operator | Signature | Description | Determinism |
|----------|-----------|-------------|-------------|
| `pitch.autocorrelation` | `(signal: AudioSignal) → f32[Hz]` | Pitch detection via autocorrelation | DETERMINISTIC |
| `pitch.yin` | `(signal: AudioSignal, threshold: f32) → f32[Hz]` | YIN pitch detector (robust) | DETERMINISTIC |
| `spectral.centroid` | `(spectrum: Field2D<Complex>) → f32[Hz]` | Spectral centroid (brightness) | DETERMINISTIC |
| `spectral.envelope` | `(spectrum: Field2D<Complex>, smoothing: f32) → Field1D<f32>` | Smooth spectral envelope | DETERMINISTIC |
| `resonance.peaks` | `(spectrum: Field1D<f32>, threshold: dB) → Array<(f32[Hz], f32)>` | Detect resonant peaks | DETERMINISTIC |
| `decay.fit_exponential` | `(envelope: Field1D<f32>, time: Field1D<s>) → f32[1/s]` | Fit exponential decay rate | REPRO |
| `decay.t60` | `(envelope: Field1D<f32>) → f32[s]` | Compute T60 (time to -60dB) | DETERMINISTIC |
| `inharmonicity.measure` | `(signal: AudioSignal, f0: f32[Hz]) → f32` | Measure inharmonicity coefficient | REPRO |
| `transient.detect` | `(signal: AudioSignal, threshold: dB) → Array<f32[s]>` | Detect percussive transients | DETERMINISTIC |
| `cepstral.transform` | `(spectrum: Field1D<Complex>) → Field1D<f32>` | Cepstrum (log spectrum → IFFT) | DETERMINISTIC |

#### Layer 2: Composite Operators (Feature Extraction)

| Operator | Signature | Description | Composed From |
|----------|-----------|-------------|---------------|
| `harmonic.track_fundamental` | `(signal: AudioSignal) → Ctl[Hz]` | Track fundamental frequency over time | `stft` → `pitch.yin` → smooth |
| `harmonic.track_partials` | `(signal: AudioSignal, f0: Ctl[Hz], num_partials: int) → Field2D<f32>` | Track harmonic amplitudes (time × partial) | `stft` → peak tracking |
| `modal.extract_modes` | `(spectrum: Field1D<f32>) → Array<ModalPeak>` | Extract resonant peaks (freq, amplitude, Q) | `resonance.peaks` → fit |
| `envelope.extract` | `(signal: AudioSignal, type: Amplitude\|Spectral) → Env` | Extract amplitude or spectral envelope | `stft` → envelope follower |
| `excitation.extract` | `(signal: AudioSignal, onset: f32[s]) → AudioSignal` | Extract attack/pluck transient | `transient.detect` → window |
| `noise.extract_broadband` | `(signal: AudioSignal, harmonics: Field2D<f32>) → NoiseModel` | Extract noise residual (signal - harmonics) | Subtraction + band analysis |
| `vibrato.extract` | `(f0: Ctl[Hz]) → (rate: f32[Hz], depth: f32[cents], phase: f32[rad])` | Extract vibrato parameters | Pitch modulation analysis |

#### Layer 3: Constructs (High-Level Analysis & Synthesis)

| Operator | Signature | Description | Use Case |
|----------|-----------|-------------|----------|
| `modal.analyze` | `(signal: AudioSignal, num_modes: int) → ModalModel` | Fit damped sinusoid modes: `A*e^(-dt)*sin(2πft+φ)` | Body resonance modeling |
| `deconvolve` | `(signal: AudioSignal, f0: Ctl[Hz]) → (excitation: AudioSignal, body_ir: IR)` | Separate excitation and resonator via homomorphic deconvolution | Excitation/resonator separation |
| `additive.synth` | `(harmonics: Field2D<f32>, f0: Ctl[Hz]) → AudioSignal` | Sum of harmonics with time-varying envelopes | Harmonic resynthesis |
| `modal.synth` | `(modes: ModalModel, excitation: AudioSignal) → AudioSignal` | Damped sinusoid resonator bank | Body resonance synthesis |
| `excitation.pluck` | `(type: Karplus\|Noise, params: Map) → AudioSignal` | Pluck/noise excitation generator | Physical model excitation |
| `spectral.filter` | `(signal: AudioSignal, envelope: Field1D<f32>) → AudioSignal` | Reapply spectral envelope (formant filtering) | Timbre shaping |
| `granular.resynth` | `(signal: AudioSignal, grain_size: s, density: f32) → AudioSignal` | Granular resynthesis for textures | Extended techniques |
| `instrument.analyze` | `(signal: AudioSignal) → InstrumentModel` | Full analysis pipeline (harmonics + modes + excitation + noise) | End-to-end analysis |
| `instrument.synthesize` | `(model: InstrumentModel, pitch: f32[Hz], velocity: f32) → AudioSignal` | Generate new note from model | Resynthesis |
| `instrument.morph` | `(model_a: InstrumentModel, model_b: InstrumentModel, blend: f32) → InstrumentModel` | Morph between two timbres | Hybrid instruments |

#### Layer 4: Presets (Pre-Configured Workflows)

| Preset | Description | Configuration |
|--------|-------------|---------------|
| `guitar_model` | Acoustic guitar timbre extraction | Harmonic tracking + modal analysis + pluck detection |
| `violin_model` | Bowed string timbre extraction | Harmonic tracking + bow noise + vibrato |
| `piano_model` | Piano timbre extraction | Inharmonic partials + hammer noise + sustain pedal |
| `brass_model` | Brass instrument timbre extraction | Harmonic tracking + lip buzz + bell resonance |
| `hybrid_synth` | Morph acoustic + synth | 50% guitar model + 50% sawtooth oscillator |

---

### 8.3 Reference Types

#### Primary: `InstrumentModelRef`

**Purpose:** Reference to an extracted instrument model.

**Auto-Anchors:**
```python
InstrumentModelRef.fundamental → f32[Hz]              # Base pitch
InstrumentModelRef.harmonics → Field2D<f32>           # Time × partial amplitudes
InstrumentModelRef.modes → ModalModel                 # Resonant body modes
InstrumentModelRef.body_ir → IR                       # Body impulse response
InstrumentModelRef.excitation → ExcitationModel       # Attack/pluck model
InstrumentModelRef.noise → NoiseModel                 # Noise signature
InstrumentModelRef.decay_rates → Field1D<f32>        # Per-partial decay
InstrumentModelRef.inharmonicity → f32                # Inharmonicity coefficient
InstrumentModelRef.synth_params → Map<str, f32>      # Runtime parameters
```

**Example:**
```python
# Analyze recording
guitar = instrument.analyze(load("guitar_e3.wav"))

# Access extracted features
f0 = guitar.fundamental                  # 164.81 Hz (E3)
harmonics = guitar.harmonics             # 20 partials over time
body_modes = guitar.modes                # Guitar body resonances

# Synthesize new note
note_a3 = instrument.synthesize(guitar, pitch=220.0, velocity=0.8)
```

#### Secondary: `ModalModel`

**Purpose:** Resonant mode parameters (damped sinusoids).

```python
ModalModel.modes → Array<(freq: f32[Hz], amplitude: f32, decay: f32[1/s], phase: f32[rad])>
ModalModel.num_modes → int
```

#### Secondary: `NoiseModel`

**Purpose:** Broadband noise characteristics.

```python
NoiseModel.bands → Array<(center: f32[Hz], amplitude: f32, bandwidth: f32[Hz])>
NoiseModel.type → WhiteNoise | PinkNoise | Brownian
```

---

### 8.4 Passes

#### 8.4.1 Harmonic Tracking Pass

**Purpose:** Extract time-varying harmonic amplitudes.

**Input:** AudioSignal
**Output:** Field2D<f32> (time × partial)

**Algorithm:**
1. Compute STFT
2. Detect fundamental frequency (YIN, autocorrelation)
3. For each time frame:
   - Locate harmonic peaks (f0, 2*f0, 3*f0, ...)
   - Measure amplitude of each partial
4. Return time × partial amplitude matrix

**Determinism:** DETERMINISTIC (fixed STFT parameters)

---

#### 8.4.2 Modal Analysis Pass

**Purpose:** Fit damped sinusoid modes to spectrum.

**Input:** AudioSignal
**Output:** ModalModel (array of modes)

**Algorithm:**
1. Compute impulse response (via excitation removal)
2. Apply Prony's method or Matrix Pencil to fit modes
3. Extract (frequency, amplitude, decay rate, phase) for each mode
4. Return ModalModel

**Determinism:** REPRO (iterative least-squares fitting)

**Reference:** Morrison & Adrien (1993) - Modal Synthesis

---

#### 8.4.3 Deconvolution Pass

**Purpose:** Separate excitation and body resonance.

**Input:** AudioSignal, fundamental frequency
**Output:** (excitation: AudioSignal, body_ir: IR)

**Algorithm (Homomorphic Deconvolution):**
1. Compute cepstrum: `cepstrum = IFFT(log(|FFT(signal)|))`
2. Separate quefrency domains:
   - Low quefrency → excitation (attack/pluck)
   - High quefrency → resonator (body)
3. Reconstruct via inverse cepstrum
4. Return excitation and body IR

**Determinism:** REPRO (division in frequency domain)

**Reference:** Oppenheim & Schafer - Homomorphic Signal Processing

---

#### 8.4.4 Synthesis Pass

**Purpose:** Generate new note from InstrumentModel.

**Input:** InstrumentModel, pitch (Hz), velocity (0-1)
**Output:** AudioSignal

**Algorithm:**
1. Scale harmonics by pitch ratio: `new_freq = old_freq * (pitch / model.fundamental)`
2. Generate additive synthesis: Sum of time-varying sinusoids
3. Apply excitation (pluck/bow noise) scaled by velocity
4. Convolve with body IR
5. Add noise layer
6. Return synthesized signal

**Determinism:** DETERMINISTIC (deterministic sinusoid summation)

---

### 8.5 Integration with Other Domains

#### 8.5.1 Transform Domain (Layer 2)

**Dependency:** STFT, FFT for spectral analysis

```python
spectrum = stft(signal)                    # Transform domain
harmonics = harmonic.track_partials(spectrum)  # InstrumentModeling domain
```

---

#### 8.5.2 Audio Domain (Layer 5)

**Dependency:** Filters, oscillators, effects

```python
model = instrument.analyze(guitar_recording)
synth_signal = instrument.synthesize(model, pitch=440)
output = synth_signal |> reverb(0.15) |> lowpass(8000)  # Audio domain
```

---

#### 8.5.3 Stochastic Domain (Layer 3)

**Dependency:** Noise modeling

```python
noise = noise_model.sample(seed=42)        # Stochastic domain
combined = additive_synth + noise * 0.1    # InstrumentModeling + Stochastic
```

---

#### 8.5.4 Physics Domain (Layer 4)

**Use extracted parameters to drive physical models:**

```python
model = instrument.analyze(guitar)
modes = model.modes  # Extract modal parameters

# Use modes in physics simulation
string = physics.string(
    freq = model.fundamental,
    damping = modes[0].decay,
    stiffness = derived_from(model.inharmonicity)
)
```

---

### 8.6 Use Cases

| Use Case | Workflow | Output |
|----------|----------|--------|
| **MIDI instrument creation** | Analyze one note → synthesize any pitch | Virtual instrument playable via MIDI |
| **Timbre morphing** | `instrument.morph(guitar, violin, 0.5)` | Hybrid guitar-violin timbre |
| **Luthier analysis** | Measure decay rates, resonances, inharmonicity | Quantitative instrument metrics |
| **Virtual acoustics** | Extract body IR → apply to other sounds | Guitar body applied to synth/drums |
| **Physics-informed synthesis** | Modal parameters → physical model | Expressive, controllable synthesis |
| **Archive preservation** | Digitize vintage instruments | Historical instrument model library |

---

### 8.7 References

#### Academic Papers
- **Karplus & Strong (1983)** — Digital Synthesis of Plucked String and Drum Timbres
- **Smith (1992)** — Physical Modeling Using Digital Waveguides
- **Morrison & Adrien (1993)** — MOSAIC: A Framework for Modal Synthesis
- **Serra & Smith (1990)** — Spectral Modeling Synthesis
- **Engel et al. (2017)** — Neural Audio Synthesis (NSynth)

#### Morphogen Documentation
- **[../specifications/timbre-extraction.md](../../specifications/timbre-extraction.md)** — Full technical specification
- **[ADR-003](../../adr/003-instrument-modeling-domain.md)** — Architectural decision record
- **[../specifications/transform.md](../../specifications/transform.md)** — Transform operators (FFT, STFT)
- **[AUDIO_SPECIFICATION.md](../AUDIO_SPECIFICATION.md)** — Audio domain

---

## Cross-Domain Integration Matrix

| Source Domain | Target Domain | Flow Type | Example Use Case | Priority |
|---------------|---------------|-----------|------------------|----------|
| **Physics** | **Audio** | Sonification | Collision forces → percussion | **P1** |
| **Audio** | **Graphics** | Visualization | FFT spectrum → particle colors | **P1** |
| **Geometry** | **Physics** | Mesh Import | CAD mesh → collision geometry | **P0** |
| **Finance** | **Neural** | Data Feed | Monte Carlo paths → training data | **P2** |
| **Neural** | **Geometry** | Generation | GAN → procedural 3D shapes | **P2** |
| **Pattern** | **Audio** | Sequencing | TidalCycles → audio events | **P1** |
| **Pattern** | **Graphics** | Animation | Euclidean rhythms → keyframes | **P2** |
| **Physics** | **Graphics** | Rendering | Particle positions → instanced rendering | **P1** |
| **Audio** | **InstrumentModeling** | Analysis | Recording → timbre extraction | **P1** |
| **InstrumentModeling** | **Audio** | Synthesis | InstrumentModel → synthesized notes | **P1** |
| **InstrumentModeling** | **Physics** | Parameters | Modal analysis → physical model params | **P1** |
| **Transform** | **InstrumentModeling** | Spectral | STFT → harmonic tracking | **P1** |
| **Stochastic** | **InstrumentModeling** | Noise | Noise model → resynthesis layer | **P1** |

---

## Next Steps

### Immediate (v0.8-v0.9)

1. **Complete AudioDomain**
   - Layer 2 filters
   - Layer 3 effects (reverb, compressor)
   - Port-based patching

2. **Implement PhysicsDomain**
   - N-body simulation
   - Integrators (Euler, Verlet, RK4)
   - Basic forces

3. **Implement PatternDomain**
   - Pattern combinators
   - Euclidean rhythms
   - Integration with AudioDomain

### Medium-term (v0.9-v1.0)

4. **Implement InstrumentModelingDomain**
   - Layer 1: Analysis primitives (pitch tracking, spectral analysis)
   - Layer 2: Feature extraction (harmonic tracking, modal analysis)
   - Layer 3: Synthesis (additive, modal, deconvolution)
   - Layer 4: End-to-end workflows (analyze, synthesize, morph)

5. **Complete GeometryDomain**
   - Boolean operations
   - Advanced features (loft, sweep, fillet)

6. **Implement FinanceDomain**
   - Monte Carlo engine
   - Stochastic processes
   - Variance reduction

7. **Implement GraphicsDomain**
   - Scene graph
   - Shader pipelines
   - Integration with Geometry + Physics

### Long-term (v1.1+)

8. **Implement NeuralDomain**
   - Tensor operations
   - Layer building blocks
   - Integration with Finance (ML models)
   - Integration with InstrumentModeling (neural timbre embeddings)

---

## Conclusion

These **eight domains** form the foundation of Morphogen as a **multi-domain platform**:

1. ✅ **AudioDomain** - Proven patterns from RiffStack
2. ✅ **PhysicsDomain** - N-body, integrators, forces
3. ✅ **GeometryDomain** - TiaCAD patterns (reference implementation)
4. ✅ **FinanceDomain** - Monte Carlo, stochastic processes
5. ✅ **GraphicsDomain** - Scene graphs, rendering
6. ✅ **NeuralDomain** - ML/AI tensor operations
7. ✅ **PatternDomain** - Strudel/TidalCycles sequencing
8. ✅ **InstrumentModelingDomain** - Timbre extraction and synthesis (v0.9+)

**Each domain follows the unified architecture:**
- ✅ 4-layer operator hierarchy
- ✅ Unified reference types with auto-anchors
- ✅ Domain-specific passes
- ✅ Plugin extensibility
- ✅ Cross-domain composability

**Morphogen is not a library. Morphogen is a platform.**
