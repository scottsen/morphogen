# Continuous-Discrete Semantics: Dual Computational Models

**Version:** 1.0
**Status:** Architecture Specification
**Last Updated:** 2025-11-21

---

## Overview

Morphogen supports **two fundamentally different computational models** that reflect the nature of different physical and computational domains:

1. **Continuous semantics** — Differential equations, smooth evolution, spectral methods
2. **Discrete semantics** — State transitions, iterative updates, combinatorial logic

This document specifies:
- How these models differ
- How they're implemented in Morphogen
- How they interoperate (hybrid systems)
- How compilation and execution differ between them

**Prerequisites:**
- [Universal DSL Principles](../philosophy/universal-dsl-principles.md) — Design foundations
- [Operator Foundations](../philosophy/operator-foundations.md) — Mathematical background

**Related:**
- [DSL Framework Design](dsl-framework-design.md) ⭐ — Future vision for domain reasoning language
- [Transform Composition](../specifications/transform-composition.md) — Cross-representation transformations
- [Universal Domain Translation](../adr/012-universal-domain-translation.md) — Translation framework

---

## The Fundamental Divide

### Continuous Domains

**Characteristics:**
- State evolves **smoothly** over time/space
- Governed by **differential equations**
- **Transformable** (Fourier, Laplace, spectral methods work)
- **Linearizable** (Taylor expansion, local approximation)
- **Globally analyzable** (closed-form solutions possible)

**Mathematical Model:**
```
∂u/∂t = L(u)    // Evolution operator
u(t) = exp(tL)u(0)  // Solution via operator exponential
```

**Examples:**
- Temperature fields (heat equation)
- Fluid dynamics (Navier-Stokes)
- Audio signals (wave equation)
- Electromagnetic fields (Maxwell equations)

---

### Discrete Domains

**Characteristics:**
- State evolves in **discrete steps**
- Governed by **recurrence relations**
- **Non-transformable** (no Fourier analysis of logic circuits)
- **Combinatorial** (state space explosion)
- **Simulatable** (must execute to understand behavior)

**Mathematical Model:**
```
s_{n+1} = f(s_n, input)    // State transition
s(n) = f^n(s_0)            // Solution via iteration
```

**Examples:**
- Cellular automata
- State machines
- Discrete event systems
- Logic circuits
- Agent behaviors (decision trees, FSMs)

---

## Type-Level Distinction

### Continuous Type Tag

**Syntax:**
```morphogen
@continuous
operator diffuse(field: Field2D<f32>, rate: f32, dt: f32) -> Field2D<f32>
```

**Semantics:**
- Operator represents continuous evolution
- Can be solved via ODE integration
- Supports spectral methods
- Time-reversible (if conservative)

**Compilation:**
- Lower to ODE solver (RK4, adaptive, etc.)
- Enable symbolic optimization
- Use transform-based acceleration (FFT for convolution)

---

### Discrete Type Tag

**Syntax:**
```morphogen
@discrete
operator step_automaton(grid: Grid2D<Cell>, rules: Rules) -> Grid2D<Cell>
```

**Semantics:**
- Operator represents discrete update
- Must be iterated/executed
- No spectral methods
- May be irreversible

**Compilation:**
- Lower to iteration loop
- Vectorize if possible
- GPU kernel for parallel updates

---

### Hybrid Type Tag

**Syntax:**
```morphogen
@hybrid(continuous=field, discrete=events)
operator couple_thermal_shock(
    temp: Field2D<f32>,      // Continuous
    cracks: Set<Crack>       // Discrete
) -> (Field2D<f32>, Set<Crack>)
```

**Semantics:**
- Mixes continuous and discrete components
- Requires event-driven execution
- Guards trigger discrete transitions

**Compilation:**
- Hybrid ODE + event queue
- Detect zero-crossings for guards
- Adaptive time stepping around events

---

## Execution Semantics

### Continuous Execution

**Flow block with continuous operators:**

```morphogen
use field

@state temp : Field2D<f32> = random_normal(seed=42, shape=(256, 256))

@continuous
flow(dt=0.01, steps=1000) {
    // Continuous operator - differential equation
    temp = diffuse(temp, rate=0.1, dt)
}
```

**Execution:**
1. Compile `diffuse` to ODE system: `∂T/∂t = α∇²T`
2. Choose solver based on profile (Euler, RK4, adaptive)
3. Integrate from `t` to `t + dt`
4. Update state

**Key property:** Time `dt` is a **continuous parameter**, not a discrete step count.

---

### Discrete Execution

**Flow block with discrete operators:**

```morphogen
use agent

@state agents : Agents<Boid> = alloc(count=100, init=spawn_boid)

@discrete
flow(dt=0.01, steps=1000) {
    // Discrete operator - iteration
    agents = agents.map(|b| {
        vel: b.vel + flocking_force(b) * dt,
        pos: b.pos + b.vel * dt
    })
}
```

**Execution:**
1. Compile `agents.map` to iteration loop
2. For each agent, apply update function
3. Collect results into new agent collection
4. Update state

**Key property:** Time `dt` is a **discrete timestep**, updates happen instantaneously.

---

### Hybrid Execution

**Flow block with both:**

```morphogen
use field, agent

@state temp : Field2D<f32> = zeros((256, 256))
@state particles : Agents<Particle> = alloc(count=1000)

@hybrid
flow(dt=0.01, steps=1000) {
    // Continuous evolution
    temp = diffuse(temp, rate=0.1, dt)

    // Discrete update
    particles = particles.map(|p| update_particle(p, temp, dt))

    // Coupling: discrete -> continuous
    temp = temp + particle_heat_sources(particles, dt)

    // Event-driven: guard triggers discrete transition
    when temp.max() > 100.0:
        particles = particles + spawn_vapor(temp)
}
```

**Execution:**
1. ODE solver for `diffuse` (continuous)
2. Iterator for `particles.map` (discrete)
3. Coupling operators translate between domains
4. Event queue monitors guards (`when` conditions)
5. When guard fires, trigger discrete transition

---

## Compilation Strategies

### Continuous Operators → ODE Solvers

**Example: Heat diffusion**

```morphogen
temp = diffuse(temp, rate=0.1, dt)
```

**Lowering:**
```mlir
// MLIR representation
%laplacian = linalg.generic %temp : Field2D -> Field2D
%rhs = arith.mulf %laplacian, %rate : Field2D
%temp_new = arith.addf %temp, %rhs : Field2D
```

**Solver choice:**
- **Explicit Euler**: `T_{n+1} = T_n + dt * α∇²T_n` (simple, unstable for large dt)
- **Implicit Euler**: `T_{n+1} = T_n + dt * α∇²T_{n+1}` (stable, requires solve)
- **RK4**: 4th-order accurate (standard choice)
- **Adaptive**: Error control (high precision)

---

### Discrete Operators → Iteration Loops

**Example: Agent update**

```morphogen
agents = agents.map(|a| update_velocity(a, dt))
```

**Lowering:**
```mlir
// MLIR representation
scf.for %i = 0 to %num_agents {
    %agent = agent.load %agents[%i]
    %new_vel = agent.update_velocity %agent, %dt
    agent.store %new_vel -> %agents[%i]
}
```

**Optimization:**
- Vectorization (SIMD)
- GPU parallelization
- Loop unrolling

---

### Hybrid Operators → Event-Driven Execution

**Example: Phase transition**

```morphogen
flow(dt=0.01) {
    temp = diffuse(temp, rate=0.1, dt)

    when temp.max() > 100.0:
        phase = Solid -> Liquid
        temp = apply_latent_heat(temp)
}
```

**Execution:**
1. **ODE integration** for continuous evolution
2. **Event detection** at each step:
   ```
   if max(temp) > 100.0:
       trigger_event(phase_transition)
   ```
3. **Event handler** executes discrete transition
4. **Continue** ODE integration with updated state

**Implementation:**
- Event queue (priority queue by time)
- Zero-crossing detection (root-finding)
- Adaptive time stepping (refine around events)

---

## Multi-Rate Scheduling

### Problem

Different domains evolve at different rates:
- **Audio**: 48 kHz (20 μs timestep)
- **Physics**: 240 Hz (4 ms timestep)
- **Agents**: 60 Hz (16 ms timestep)
- **Graphics**: 30 Hz (33 ms timestep)

**Naive approach:** Use slowest rate for everything (wasteful)

---

### Solution: Multi-Rate Scheduler

**Declare rates explicitly:**

```morphogen
@rate(48000Hz)
@state audio : AudioBuffer = silence(duration=1.0)

@rate(60Hz)
@state agents : Agents<Particle> = alloc(count=1000)

@rate(240Hz)
@state physics : RigidBodyWorld = create_world()

@multi_rate
flow() {
    // Each domain updates at its own rate
    // Scheduler handles synchronization
}
```

**Scheduler algorithm:**

```
rates = [48000Hz, 240Hz, 60Hz]
lcm = least_common_multiple(rates)  // 48000
timesteps = [1, 200, 800]  // Relative update frequencies

for tick in 0..simulation_length:
    if tick % 1 == 0:
        update_audio(dt_audio)
    if tick % 200 == 0:
        update_physics(dt_physics)
    if tick % 800 == 0:
        update_agents(dt_agents)
```

**Coupling:** Interpolate between updates when domains interact.

---

## Bridging Continuous ↔ Discrete

### Sampling: Continuous → Discrete

**Convert smooth signal to discrete samples:**

```morphogen
@continuous
signal : Stream<f32, audio:time>

@discrete
samples : Array<f32> = sample(signal, rate=48000Hz, duration=1.0s)
```

**Semantics:**
- Sample at regular intervals
- Shannon sampling theorem applies (avoid aliasing)
- Interpolation for sub-sample queries

---

### Interpolation: Discrete → Continuous

**Reconstruct continuous signal from discrete samples:**

```morphogen
@discrete
samples : Array<f32>

@continuous
signal : Stream<f32, audio:time> = interpolate(
    samples,
    method="cubic_spline",
    sample_rate=48000Hz
)
```

**Methods:**
- Linear interpolation
- Cubic spline (smooth)
- Sinc interpolation (Shannon reconstruction)

---

### Event-Driven: Hybrid

**Continuous evolution interrupted by discrete events:**

```morphogen
@state position : f32 = 0.0
@state velocity : f32 = 10.0

@hybrid
flow(dt=0.001) {
    // Continuous: ballistic motion
    position = position + velocity * dt
    velocity = velocity - 9.8 * dt  // Gravity

    // Discrete: collision event
    when position <= 0.0:
        position = 0.0
        velocity = -0.9 * velocity  // Bounce with damping
}
```

**Semantics:**
- Continuous ODE between events
- Detect zero-crossing: `position(t) = 0`
- Apply discrete update instantaneously
- Resume continuous evolution

---

## Domain Classification

### Continuous Domains

| Domain | Continuous Aspect | Governing Equation |
|--------|------------------|-------------------|
| **Field** | Temperature, pressure, density | Heat, Navier-Stokes, reaction-diffusion |
| **Audio** | Sound pressure | Wave equation |
| **Signal** | Voltage, current | Circuit ODEs |
| **Physics (soft-body)** | Deformation, strain | Finite element |

---

### Discrete Domains

| Domain | Discrete Aspect | Update Rule |
|--------|----------------|-------------|
| **Agent** | Individual behaviors | FSM transitions, decision trees |
| **Graph** | Network topology | Edge additions/removals |
| **State Machine** | Control states | Transition function |
| **Cellular Automata** | Cell states | Neighborhood rules |

---

### Hybrid Domains

| Domain | Continuous | Discrete |
|--------|-----------|----------|
| **RigidBody** | Position, velocity | Collision events |
| **Chemistry** | Concentrations | Reaction events |
| **Neuroscience** | Membrane potential | Spike events |
| **Economics** | Market prices | Policy decisions |

---

## Implementation Status

### ✅ Currently Supported

**Continuous:**
- Field operations (diffuse, advect, project)
- Audio synthesis (oscillators, filters)
- Numerical integration (Euler, RK4)

**Discrete:**
- Agent updates (map, filter, reduce)
- State machine transitions (basic)
- Iterative solvers

**Hybrid (basic):**
- Field ↔ Agent coupling
- Discrete updates in continuous flow blocks

---

### 🚧 Planned

**Continuous:**
- Symbolic differentiation
- Adaptive ODE solvers
- Spectral methods (Chebyshev, Legendre)

**Discrete:**
- Event-driven execution
- Discrete event simulation (DES)
- Petri nets, process calculi

**Hybrid:**
- Guard-triggered events (`when` conditions)
- Multi-rate scheduling
- Zero-crossing detection
- Hybrid automata

---

## Design Guidelines

### When to Use Continuous Semantics

**Use continuous if:**
- State evolves smoothly
- Governed by differential equations
- Spectral methods apply
- Need high accuracy

**Examples:**
- Temperature diffusion
- Fluid flow
- Audio signals (synthesis)
- Electromagnetic fields

---

### When to Use Discrete Semantics

**Use discrete if:**
- State changes in steps
- Governed by logic/rules
- Combinatorial structure
- Event-driven

**Examples:**
- Cellular automata (Game of Life)
- State machines
- Agent decisions
- Discrete event systems

---

### When to Use Hybrid Semantics

**Use hybrid if:**
- Mix smooth evolution and abrupt changes
- Events trigger continuous changes
- Continuous thresholds trigger discrete transitions
- Multiple time scales

**Examples:**
- Rigid body physics (continuous motion + collision events)
- Chemical reactions (continuous concentrations + discrete reaction events)
- Neuroscience (continuous potentials + discrete spikes)
- Cyber-physical systems (continuous dynamics + discrete control)

---

## Further Reading

**Philosophy:**
- [Universal DSL Principles](../philosophy/universal-dsl-principles.md) — Design foundations
- [Operator Foundations](../philosophy/operator-foundations.md) — Mathematical background

**Specifications:**
- [Transform Composition](../specifications/transform-composition.md) — Cross-representation transforms
- [Scheduler](../specifications/scheduler.md) — Multi-rate scheduling

**ADRs:**
- [Universal Domain Translation](../adr/012-universal-domain-translation.md) — Translation semantics

**External:**
- **"Hybrid Systems: Computation and Control"** — HSCC conference
- **"Principles of Cyber-Physical Systems"** — Alur
- **"Modeling and Simulation of Dynamic Systems"** — Woods & Lawrence

---

## Summary

**Morphogen supports dual computational models:**

| Aspect | Continuous | Discrete | Hybrid |
|--------|-----------|----------|--------|
| **Evolution** | Smooth (ODEs) | Stepwise (recurrence) | Both |
| **Solver** | ODE integrator | Iterator | ODE + events |
| **Transform** | FFT, Laplace, spectral | N/A | Mixed |
| **Time** | Continuous parameter | Discrete steps | Event-driven |
| **Examples** | Fields, audio, fluids | Agents, FSMs, CA | RigidBody, chemistry |

**Key insight:** Different domains require different computational models. A universal DSL must support both and allow them to interoperate.

**Next:** See [Transform Composition](../specifications/transform-composition.md) for cross-representation transformations, or [Universal Domain Translation](../adr/012-universal-domain-translation.md) for translation semantics.
