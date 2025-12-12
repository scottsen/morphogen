# Kairo 2.0 Language Specification

**Version:** 2.0.0-draft
**Status:** 🔬 Research Specification (Not Yet Implemented)
**Date:** 2025-11-21
**Authors:** Scott Sen, with Claude

---

## Executive Summary

This document specifies **Kairo 2.0**, a complete redesign of the Kairo language that synthesizes ideas from:
- **Modelica** — Physical semantics and declarative constraints
- **Faust** — Algebraic composition operators
- **Wolfram Language** — Symbolic + numeric computation
- **MLIR** — Progressive lowering and domain dialects
- **Haskell** — Type classes and algebraic laws
- **Julia** — Multiple dispatch on domain types
- **Ptolemy II** — Explicit models of computation

**Core Philosophy:** Domains, operators, and translations are first-class citizens with explicit semantics, algebraic composition, and compiler-verified properties.

**What's Different from Kairo 1.x:**
- Domains defined declaratively in `.morph` files (not Python)
- First-class translation objects with invariant preservation
- Algebraic composition operators (`∘`, `:`, `~`)
- Explicit continuous/discrete distinction
- Symbolic + numeric execution modes
- Type classes for domain properties
- Multiple dispatch on domain types

---

## Table of Contents

1. [Core Principles](#1-core-principles)
2. [Type System](#2-type-system)
3. [Domain Declaration](#3-domain-declaration)
4. [Operator Definition](#4-operator-definition)
5. [Translation System](#5-translation-system)
6. [Composition Algebra](#6-composition-algebra)
7. [Execution Models](#7-execution-models)
8. [Type Classes & Interfaces](#8-type-classes--interfaces)
9. [Constraint System](#9-constraint-system)
10. [Progressive Lowering](#10-progressive-lowering)
11. [Syntax Reference](#11-syntax-reference)
12. [Examples](#12-examples)

---

## 1. Core Principles

### 1.1 Everything is Explicit

**From:** Ptolemy II, MLIR

**Principle:** No implicit behavior. Domain properties, execution models, and transformations are declared explicitly.

```kairo
// Explicit execution model
@execution_model(continuous_time)
domain Fields { ... }

@execution_model(discrete_event)
domain StateMachine { ... }

// Explicit continuity
@continuous operator diffuse(...) { ... }
@discrete operator step(...) { ... }
```

### 1.2 Algebraic Composition

**From:** Faust, Haskell, APL

**Principle:** Operations compose algebraically with verified laws.

```kairo
// Composition operators
f ∘ g ∘ h           // Sequential composition (Haskell's .)
f : g : h            // Pipeline composition (Faust's :)
f ~ g                // Feedback composition (Faust's ~)
f <: g :> h          // Parallel split/merge (Faust's <:, :>)

// Laws enforced by compiler
(f ∘ g) ∘ h == f ∘ (g ∘ h)  // Associativity
id ∘ f == f == f ∘ id       // Identity
```

### 1.3 Declarative Constraints

**From:** Modelica

**Principle:** Physical laws and invariants are declared, not implemented.

```kairo
domain Circuit {
  // Conservation laws declared once
  constraint KirchhoffCurrent {
    forall node: Node
    => sum(node.currents_in) == sum(node.currents_out)
  }

  constraint KirchhoffVoltage {
    forall loop: Loop
    => sum(loop.voltages) == 0
  }
}
```

### 1.4 Symbolic First, Numeric Fallback

**From:** Wolfram Language

**Principle:** Compiler attempts symbolic solution before numeric.

```kairo
@solver(strategy = symbolic_first)
flow(dt = 0.01) {
  // Compiler tries to solve symbolically
  // Falls back to numeric if impossible
  temp = diffuse(temp, rate = k, dt)
}
```

### 1.5 Progressive Lowering

**From:** MLIR

**Principle:** High-level semantics progressively lower through multiple IRs.

```
Kairo Domain-Level
  ↓ lower_to
Category IR (functorial semantics)
  ↓ lower_to
MLIR Dialects
  ↓ lower_to
LLVM IR
```

---

## 2. Type System

### 2.1 Dependent Types with Physical Units

**From:** Julia (units), Idris (dependent types)

```kairo
// Physical units as refinement types
type Length = Real[meters]
type Time = Real[seconds]
type Velocity = Real[meters/seconds]

// Dependent types
type Signal<T, Domain, Rate> where {
  T: NumericType,
  Domain: DomainType,
  Rate: RateType(Domain)  // Rate depends on Domain
}

// Type-level computation
Velocity = Length / Time  // ✅ Type checker verifies units
Length + Time            // ❌ Compile error: unit mismatch
```

### 2.2 Domain Tags

**Explicit domain membership:**

```kairo
// Streams tagged with domain
signal: Stream<f32, Audio:Time, 48kHz>
field: Field2D<f32, Physics:Thermal, continuous>
particles: Agents<Particle, Physics:Discrete>

// Domain tag includes representation space
audio_freq: Stream<Complex<f32>, Audio:Frequency, 24kHz>
```

### 2.3 Continuity Tags

**Explicit continuous vs discrete:**

```kairo
type ContinuousField = Field2D<f32> @continuous
type DiscreteGrid = Grid2D<Cell> @discrete

// Operators tagged by continuity
@continuous operator diffuse(f: ContinuousField, ...) -> ContinuousField
@discrete operator step_automaton(g: DiscreteGrid, ...) -> DiscreteGrid
```

### 2.4 Effect System

**From:** Haskell (monadic effects)

```kairo
// Pure computation (default)
pure fn add(a: f32, b: f32) -> f32 = a + b

// Effectful computation
fn read_file(path: String) -> IO<String> { ... }
fn mutate_state(s: &mut State) -> Mut<()> { ... }

// Effects tracked in type
pure_result: f32 = add(1.0, 2.0)
io_result: IO<String> = read_file("data.txt")
```

---

## 3. Domain Declaration

### 3.1 Basic Domain Syntax

**Declarative domain definition in `.morph` files:**

```kairo
domain Audio {
  // Execution model
  @execution_model(continuous_time)
  @compilation_strategy(jit)

  // Representation spaces
  representations {
    Time: primary,
    Frequency: transform(fft, ifft),
    Cepstral: transform(dct, idct),
    MelScale: transform(mel, inv_mel)
  }

  // Domain entities
  entity Signal: Stream<f32, Audio:Time, Rate>
  entity Spectrum: Stream<Complex<f32>, Audio:Frequency, Rate/2>

  // Operators declared
  operator sine(freq: f32[Hz], duration: f32[s]) -> Signal
  operator filter(signal: Signal, cutoff: f32[Hz]) -> Signal
}
```

### 3.2 Domain Constraints

**From:** Modelica

```kairo
domain Circuit {
  // Physical constraints
  constraint ConservationOfCharge {
    forall node: Node
    => sum(node.in_currents) == sum(node.out_currents)
  }

  constraint PowerBalance {
    forall circuit: Circuit
    => sum(circuit.power_sources) == sum(circuit.power_sinks) + losses
  }

  // Invariants checked at runtime/compile-time
  invariant EnergyNonNegative {
    forall component: Component
    => component.stored_energy >= 0
  }
}
```

### 3.3 Domain Inheritance

```kairo
// Base domain
domain VectorSpace<T> {
  operator add(a: T, b: T) -> T
  operator scale(s: Real, v: T) -> T

  law Associativity {
    add(add(a, b), c) == add(a, add(b, c))
  }
}

// Derived domain
domain InnerProductSpace<T> extends VectorSpace<T> {
  operator inner(a: T, b: T) -> Real
  operator norm(v: T) -> Real = sqrt(inner(v, v))

  law CauchySchwarz {
    abs(inner(a, b)) <= norm(a) * norm(b)
  }
}

// Further derived
domain HilbertSpace<T> extends InnerProductSpace<T> {
  constraint Complete {
    forall sequence: CauchySequence<T>
    => exists limit: T where converges(sequence, limit)
  }
}
```

---

## 4. Operator Definition

### 4.1 Multiple Dispatch

**From:** Julia

```kairo
// Single operator, multiple implementations
operator diffuse {
  // Continuous field - spectral method
  @continuous
  in ContinuousField(f, rate, dt) -> ContinuousField {
    let laplacian = fft2d(f) * laplacian_kernel
    return ifft2d(laplacian * rate * dt) + f
  }

  // Discrete grid - finite difference
  @discrete
  in DiscreteGrid(g, rate, dt) -> DiscreteGrid {
    return finite_difference_laplacian(g) * rate * dt + g
  }

  // Stochastic field - Monte Carlo
  @stochastic
  in StochasticField(s, rate, dt) -> StochasticField {
    return monte_carlo_diffusion(s, rate, dt)
  }
}
```

### 4.2 Cross-Domain Operators

**Universal operations that work in multiple domains:**

```kairo
// Differentiation has different implementations per domain
@universal operator differentiate {
  in Time: finite_difference(h = dt),
  in Frequency: multiply(i * omega),
  in Laplace: multiply(s),

  // Properties preserved across domains
  preserves {
    linearity: true,
    order: 1
  }
}
```

### 4.3 Operator Properties

**From:** Haskell type classes

```kairo
@operator
@properties(linear, time_invariant, causal)
fn lowpass(signal: Signal, cutoff: f32[Hz]) -> Signal {
  let spectrum = fft(signal)
  let filtered = spectrum * lowpass_mask(cutoff)
  return ifft(filtered)
}

// Properties enable optimization
// Compiler knows: lowpass(a + b) == lowpass(a) + lowpass(b)
```

---

## 5. Translation System

### 5.1 First-Class Translations

**Explicit domain-crossing with verified semantics:**

```kairo
translation Fourier from Audio:Time to Audio:Frequency {
  type: bijective

  // Forward and inverse
  forward: fft(window = "hann", norm = "ortho")
  inverse: ifft(norm = "ortho")

  // Preserved invariants
  preserves {
    energy: norm(input) == norm(output),
    linearity: transform(a*x + b*y) == a*transform(x) + b*transform(y)
  }

  // Operator mappings
  operator_map {
    convolve -> multiply,
    shift -> phase_rotate,
    differentiate -> multiply_by_iw
  }

  // Verification
  verify {
    roundtrip: inverse(forward(x)) ≈ x within epsilon
  }
}
```

### 5.2 Translation with Invariants

**From:** Modelica (conservation), Your ADR-012

```kairo
translation AgentsToField from Agents to Field {
  type: approximate
  method: kernel_density_estimation

  // What's preserved (enforced by compiler/runtime)
  preserves {
    total_mass: sum(agents.mass) == integral(field),
    center_of_mass: mean(agents.position) == centroid(field)
  }

  // What's lost (documented)
  drops {
    individual_identity: "Agents collapsed to continuous density",
    velocity_distribution: "Only mean velocity preserved",
    correlations: "Spatial correlations averaged over kernel width"
  }

  // Approximation bounds
  approximation {
    metric: L2_norm,
    bound: 1e-3,
    probability: 0.95
  }

  // Parameters
  kernel: gaussian(bandwidth = 0.5)
}
```

### 5.3 Composable Translations

```kairo
// Define atomic translations
translation FFT from Time to Frequency { ... }
translation MelScale from Frequency to MelFrequency { ... }
translation LogScale from MelFrequency to LogMel { ... }

// Compose algebraically
translation MelSpectrogram = LogScale ∘ MelScale ∘ FFT

// Use composed translation
let mel_spec = MelSpectrogram(audio_signal)

// Verify composition laws
verify {
  associativity: (h ∘ g) ∘ f == h ∘ (g ∘ f)
  identity: id ∘ f == f == f ∘ id
}
```

---

## 6. Composition Algebra

### 6.1 Composition Operators

**From:** Faust, Haskell

```kairo
// Sequential composition (∘)
f ∘ g ∘ h
// Equivalent to: f(g(h(x)))

// Pipeline composition (:)
signal : filter : reverb : output
// Equivalent to: output(reverb(filter(signal)))

// Feedback composition (~)
feedback(gain) = + ~ *(gain)
// Creates feedback loop

// Parallel composition (<:, :>)
mono_signal <: (lowpass, highpass) :> stereo_mix
// Split, process in parallel, merge

// Recursive composition
recursive_filter = input : (+ ~ delay(0.1) ~ *(-0.6)) : output
```

### 6.2 Algebraic Laws

**Compiler verifies composition properties:**

```kairo
// Laws enforced by type system
law Associativity<F, G, H> {
  (f: F, g: G, h: H) => (h ∘ g) ∘ f == h ∘ (g ∘ f)
}

law Identity<F> {
  (f: F) => id ∘ f == f && f ∘ id == f
}

law Distributivity {
  f : (g <: h) == (f : g) <: (f : h)
}

// Type system prevents invalid compositions
continuous_op ∘ discrete_op  // ❌ Type error: execution model mismatch
```

### 6.3 Tacit Programming

**From:** APL, Faust

```kairo
// Define functions by composition, no intermediate variables
mean = sum ∘ length⁻¹ ∘ divide

// Pointfree style
normalize = (x => (x - mean(x)) / std(x))
// Can be written tacit:
normalize = (subtract <: mean) : divide <: std

// Audio processing chain
reverb_chain =
  delay(0.05) :
  (+ ~ *(0.6)) :  // Feedback
  lowpass(5000) :
  mix(dry = 0.3, wet = 0.7)
```

---

## 7. Execution Models

### 7.1 Explicit Models of Computation

**From:** Ptolemy II

```kairo
// Continuous time execution
@execution_model(continuous_time)
@solver(rk4, adaptive = true)
domain Fields {
  flow(dt = 0.01) {
    temp = diffuse(temp, rate = 0.1, dt)
    // ODE solver chosen based on stiffness
  }
}

// Discrete event execution
@execution_model(discrete_event)
domain StateMachine {
  on_event(event: Event) {
    state = transition(state, event)
    // Event-driven, no fixed timestep
  }
}

// Synchronous dataflow
@execution_model(synchronous_dataflow)
domain SignalProcessing {
  // Fixed rate, compile-time scheduling
  flow {
    output = fft : filter : ifft <| input
  }
}
```

### 7.2 Hybrid Execution

**Multiple models in one system:**

```kairo
@execution_model(hybrid)
scene ThermalShock {
  // Continuous thermal field
  @continuous
  @state temp: Field2D<f32> = initialize_temperature()

  // Discrete crack propagation
  @discrete
  @state cracks: Set<Crack> = empty_set()

  flow(dt = 0.01) {
    // Continuous evolution
    temp = diffuse(temp, rate = k, dt)

    // Guard-triggered discrete transition
    when max_stress(temp) > threshold:
      cracks = propagate_crack(cracks, temp)
      temp = apply_boundary_condition(temp, cracks)
  }
}
```

### 7.3 Symbolic vs Numeric Execution

**From:** Wolfram Language

```kairo
// Symbolic execution (when possible)
@solver(strategy = symbolic_first)
flow(dt) {
  // Compiler attempts symbolic solution
  velocity = integrate(acceleration, dt)
  position = integrate(velocity, dt)

  // If symbolic fails, fallback to numeric
  // Compiler chooses: Euler, RK4, adaptive, etc.
}

// Force numeric (for performance)
@solver(strategy = numeric_only)
flow(dt) {
  // Always use numeric integration
  x = rk4_step(dx_dt, x, dt)
}

// Pure symbolic (for derivation)
@solver(strategy = symbolic_only)
derive {
  // Symbolic manipulation only
  // Fails at compile-time if not solvable
  H = T + V  // Hamiltonian
  dH_dt = poisson_bracket(H, H)
}
```

---

## 8. Type Classes & Interfaces

### 8.1 Domain Interfaces

**From:** Haskell type classes

```kairo
// Interface for invertible transformations
interface Invertible<A, B> {
  operator forward(a: A) -> B
  operator inverse(b: B) -> A

  law InverseProperty {
    forall x: A => inverse(forward(x)) == x
    forall y: B => forward(inverse(y)) == y
  }
}

// Implement for FFT
implementation Invertible<Signal, Spectrum> for FFT {
  forward = fft
  inverse = ifft

  // Compiler verifies laws hold
}

// Use interface constraint
fn roundtrip<T: Invertible>(transform: T, input: T.A) -> T.A {
  return transform.inverse(transform.forward(input))
}
```

### 8.2 Domain Properties as Interfaces

```kairo
interface Conservative<T> {
  operator measure(t: T) -> Real

  law Conservation {
    forall t1, t2: T, op: Operator
    => measure(op(t1)) == measure(t1)
  }
}

interface Linear<T> {
  operator apply(a: Real, x: T, b: Real, y: T) -> T

  law Linearity {
    apply(a, x, b, y) == a * apply(1, x, 0, zero) + b * apply(0, zero, 1, y)
  }
}

// Domain declares interfaces
domain Audio implements Conservative<Signal>, Linear<Signal> {
  measure(s: Signal) = norm(s)  // Energy
  apply(a, x, b, y) = a * x + b * y
}
```

---

## 9. Constraint System

### 9.1 Declarative Constraints

**From:** Modelica

```kairo
domain FluidDynamics {
  // Physical constraints
  constraint MassConservation {
    forall cell: Cell
    => d(cell.density)/dt + divergence(cell.mass_flux) == 0
  }

  constraint MomentumConservation {
    forall cell: Cell
    => d(cell.momentum)/dt + divergence(cell.momentum_flux) ==
       cell.pressure_gradient + cell.body_forces
  }

  constraint EnergyConservation {
    forall cell: Cell
    => d(cell.energy)/dt + divergence(cell.energy_flux) ==
       cell.heat_sources - cell.work_done
  }
}
```

### 9.2 Constraint Solving

```kairo
scene FluidSimulation {
  @state fluid: FluidField

  // Compiler generates solver from constraints
  flow(dt = 0.001) {
    // Constraints are equations to solve
    solve {
      // System of equations derived from domain constraints
      MassConservation(fluid, dt),
      MomentumConservation(fluid, dt),
      EnergyConservation(fluid, dt)
    }

    // Solver chooses: direct, iterative, symbolic, etc.
  }
}
```

---

## 10. Progressive Lowering

### 10.1 Multi-Level IR

**From:** MLIR

```
Kairo Source (domain-level)
  ↓ parse & typecheck
Kairo AST (syntax tree)
  ↓ domain elaboration
Domain IR (explicit domains, translations)
  ↓ functor lowering
Category IR (functorial semantics)
  ↓ MLIR dialect lowering
MLIR (field, agent, audio, temporal dialects)
  ↓ standard lowering
LLVM IR
  ↓ codegen
Native code
```

### 10.2 Lowering Example

**High-level Kairo:**
```kairo
domain Audio {
  operator filter(signal, cutoff) = ifft(fft(signal) * mask(cutoff))
}
```

**Domain IR:**
```
%signal = audio.signal<f32, 48kHz>
%cutoff = constant 1000.0 : f32[Hz]
%filtered = audio.filter %signal, %cutoff : audio.signal
```

**Category IR (functorial):**
```
%time = audio.time_domain %signal
%freq = functor.apply @fft %time : audio.freq_domain
%masked = audio.multiply %freq, %mask : audio.freq_domain
%result = functor.apply @ifft %masked : audio.time_domain
```

**MLIR Dialect:**
```mlir
%0 = audio.fft %signal : tensor<1024xf32> -> tensor<513xcomplex<f32>>
%1 = audio.multiply %0, %mask : tensor<513xcomplex<f32>>
%2 = audio.ifft %1 : tensor<513xcomplex<f32>> -> tensor<1024xf32>
```

### 10.3 Optimization at Each Level

**Domain-level:** Algebraic simplification
```kairo
ifft(fft(signal)) => signal  // Identity elimination
```

**Category-level:** Functor fusion
```
fft ∘ ifft ∘ fft => fft  // Composition collapse
```

**MLIR-level:** Standard optimizations
```mlir
// Loop fusion, vectorization, memory coalescing
```

---

## 11. Syntax Reference

### 11.1 Keywords

```
domain          - Domain declaration
translation     - Translation declaration
operator        - Operator declaration
interface       - Type class declaration
implementation  - Interface implementation
constraint      - Physical constraint
invariant       - Runtime invariant
law             - Algebraic law
flow            - Temporal execution block
scene           - Top-level program
preserves       - Invariant preservation
drops           - Intentional information loss
verify          - Verification assertion
```

### 11.2 Operators

```
∘               - Function composition (sequential)
:               - Pipeline composition
~               - Feedback composition
<:              - Parallel split
:>              - Parallel merge
@               - Annotation marker
=>              - Implication / mapping
->              - Function type / transition
<-              - Reverse mapping
|>              - Pipe forward
<|              - Pipe backward
```

### 11.3 Annotations

```
@continuous     - Continuous execution model
@discrete       - Discrete execution model
@hybrid         - Hybrid continuous-discrete
@solver(...)    - Solver strategy
@execution_model(...)  - Model of computation
@properties(...)       - Operator properties
@state          - Stateful variable
@universal      - Cross-domain operator
@pure           - Pure function (no effects)
```

---

## 12. Examples

### 12.1 Complete Domain Definition

```kairo
domain Audio {
  // Execution model
  @execution_model(continuous_time)
  @sample_rate_invariant

  // Representation spaces
  representations {
    Time: primary,
    Frequency: invertible_via(fft, ifft),
    Cepstral: invertible_via(dct, idct),
    MelScale: approximate_via(mel_scale, inv_mel)
  }

  // Domain types
  entity Signal: Stream<f32, Audio:Time, Rate>
  entity Spectrum: Stream<Complex<f32>, Audio:Frequency, Rate/2>

  // Operators
  @continuous
  @properties(linear, time_invariant, causal)
  operator filter(signal: Signal, cutoff: f32[Hz], q: f32) -> Signal {
    let spec = fft(signal)
    let filtered = spec * biquad_response(cutoff, q)
    return ifft(filtered)
  }

  @continuous
  operator sine(freq: f32[Hz], duration: f32[s]) -> Signal {
    return generate_sine(freq, duration, sample_rate)
  }

  // Constraints
  constraint EnergyConservation {
    forall filter: LinearFilter
    => norm(filter(signal)) <= norm(signal)
  }

  // Algebraic laws
  law Linearity {
    forall f: LinearOperator, a, b: Real, x, y: Signal
    => f(a*x + b*y) == a*f(x) + b*f(y)
  }
}
```

### 12.2 Cross-Domain Translation

```kairo
translation PhysicsToAudio
  from Physics:Pressure to Audio:Signal {

  type: bijective

  forward(pressure: Field1D<f32[Pa]>) -> Signal {
    // Sample pressure field at microphone position
    let samples = sample_at(pressure, position = mic_pos)

    // Convert Pa to audio amplitude
    let normalized = samples / reference_pressure

    return Signal(normalized, sample_rate = 48kHz)
  }

  inverse(signal: Signal) -> Field1D<f32[Pa]> {
    // Reconstruct pressure field (approximate)
    let pressure_values = signal.samples * reference_pressure
    return interpolate_field(pressure_values, domain_length)
  }

  preserves {
    frequency_content: true,
    energy_proportional: true
  }

  drops {
    spatial_field_structure: "Collapsed to 1D time series",
    phase_relationships: "Single point measurement"
  }
}
```

### 12.3 Algebraic Composition

```kairo
scene AudioProcessing {
  use audio

  // Define reusable processing blocks
  let compressor =
    envelope_follower(attack = 0.01, release = 0.1) :
    gain_computer(ratio = 4.0, threshold = -20dB) :
    smooth(time = 0.005)

  let eq_chain =
    highpass(cutoff = 80Hz) :
    peak(freq = 500Hz, gain = 3dB, q = 2.0) :
    lowpass(cutoff = 12kHz)

  let reverb_send =
    delay(0.03) :
    (+ ~ *(0.7) ~ delay(0.05)) :  // Feedback comb
    lowpass(8kHz) :
    mix(dry = 0.2, wet = 0.8)

  // Compose complete chain
  let master_chain =
    eq_chain :
    compressor :
    <: (identity, reverb_send) :>  // Parallel wet/dry
    limiter(threshold = -0.3dB)

  @state input: Signal = audio.load("input.wav")

  flow {
    let processed = master_chain(input)
    audio.save(processed, "output.wav")
  }
}
```

### 12.4 Hybrid System

```kairo
scene BallBounce {
  use physics, visual

  // Continuous state
  @continuous
  @state position: f32 = 10.0  // meters
  @continuous
  @state velocity: f32 = 0.0   // m/s

  // Discrete state
  @discrete
  @state bounces: u32 = 0

  // Constants
  let gravity = 9.8  // m/s²
  let damping = 0.8
  let ground = 0.0

  // Hybrid execution
  @execution_model(hybrid)
  flow(dt = 0.001) {
    // Continuous dynamics
    velocity = velocity - gravity * dt
    position = position + velocity * dt

    // Guard-triggered discrete transition
    when position <= ground && velocity < 0 {
      // Discrete event: bounce
      position = ground
      velocity = -velocity * damping
      bounces = bounces + 1

      // Emit event
      emit BounceEvent(bounces, position, velocity)
    }

    // Visualization
    visual.draw_circle(position = (0, position), radius = 0.5)
  }
}
```

### 12.5 Symbolic Derivation

```kairo
@solver(strategy = symbolic_only)
derive HamiltonianMechanics {
  use physics

  // Declare symbolic variables
  symbol q: Position[N]  // Generalized coordinates
  symbol p: Momentum[N]  // Conjugate momenta
  symbol H: Energy       // Hamiltonian

  // Define Hamiltonian
  let T = (1/2) * sum(p[i]^2 / m[i] for i in 1..N)  // Kinetic
  let V = potential_energy(q)                        // Potential
  let H = T + V

  // Hamilton's equations (derived symbolically)
  let dq_dt = derivative(H, p)   // ∂H/∂p
  let dp_dt = -derivative(H, q)  // -∂H/∂q

  // Verify conservation
  verify {
    dH_dt = poisson_bracket(H, H) == 0  // Energy conserved
  }

  // Generate numeric solver from symbolic form
  export numeric_integrator(q_init, p_init, dt) {
    // Compiler generates code from symbolic equations
    return symplectic_integrator(dq_dt, dp_dt, q_init, p_init, dt)
  }
}
```

---

## 13. Implementation Strategy

### Phase 1: Core Language (8 weeks)

**Week 1-2: Formal Specification**
- Complete formal semantics document
- Type system formalization
- Operational semantics

**Week 3-4: Parser & AST**
- Lexer/parser for new syntax
- AST representation
- Basic type checking

**Week 5-6: Type System**
- Dependent types
- Physical units
- Domain tags
- Effect system

**Week 7-8: Code Generation**
- Compile to existing Kairo 1.x runtime
- MLIR dialect emission
- Validation suite

### Phase 2: Domain Migration (ongoing)

**Priority domains to migrate:**
1. Audio (showcase algebraic composition)
2. Fields (showcase continuous/discrete distinction)
3. Agents (showcase hybrid systems)
4. Circuit (showcase constraint solving)
5. Remaining 36 domains incrementally

### Phase 3: Optimization (4 weeks)

**Week 1-2:**
- Algebraic simplification passes
- Functor fusion
- Dead code elimination

**Week 3-4:**
- Symbolic solver integration
- Constraint solver
- Multi-method dispatch optimization

---

## 14. Comparison: Kairo 1.x vs 2.0

| Feature | Kairo 1.x | Kairo 2.0 |
|---------|-----------|-----------|
| **Domain Definition** | Python classes | Declarative `.morph` |
| **Translations** | Procedural functions | First-class objects |
| **Composition** | `\|>` pipe only | `∘`, `:`, `~`, `<:`, `:>` |
| **Continuous/Discrete** | Implicit | Explicit tags |
| **Execution Model** | Inferred | Declared |
| **Constraints** | Manual | Declarative |
| **Symbolic Mode** | None | Full support |
| **Type Classes** | None | Haskell-style |
| **Multiple Dispatch** | None | Julia-style |
| **Verification** | Tests only | Compiler-checked laws |

---

## 15. Migration Path

### Backward Compatibility

**Kairo 1.x code continues to work:**
```kairo
// Old style (still supported)
use audio

@state signal = audio.sine(440Hz, 1.0s)

flow(dt = 1.0/48000) {
  let filtered = signal |> audio.lowpass(1000Hz)
  output filtered
}
```

**Gradual adoption of 2.0 features:**
```kairo
// Mix old and new
use audio  // Old-style import

// New-style composition
let chain = audio.lowpass(1000Hz) ∘ audio.highpass(100Hz)

flow(dt = 1.0/48000) {
  let filtered = chain(signal)  // New-style application
  output filtered
}
```

### Deprecation Timeline

**v0.12.0:** Kairo 2.0 features added, 1.x fully supported
**v0.13.0:** Deprecation warnings for 1.x-only patterns
**v0.14.0:** 1.x features require explicit `@legacy` flag
**v0.15.0:** 1.x support removed (breaking change)

---

## Status

**Current:** 🔬 Research Specification
**Target:** Kairo v0.12.0+ (2026)

**Dependencies:**
- Formal semantics document
- Type system implementation
- Parser updates
- MLIR dialect extensions

**Next Steps:**
1. Review and refine this specification
2. Write formal semantics document
3. Prototype core features
4. Validate with domain migration

---

**This specification synthesizes the best ideas from existing languages while staying true to Kairo's vision of universal domain computation with explicit, verifiable semantics.**
