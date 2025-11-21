# Universal DSL Principles: Design Brief for Cross-Domain Computation

**Version:** 1.0
**Status:** Foundational Design Philosophy
**Last Updated:** 2025-11-21

---

## Overview

This document distills the core design principles for building a **universal domain-specific language (DSL)** capable of defining operations across fundamentally different computational domains and translating between them.

These principles emerged from analyzing what makes domains fundamentally different, how they can interoperate, and what patterns enable reliable cross-domain translation.

### 📖 Document Purpose & Scope

**This document answers: "WHY should a universal DSL be designed this way?"**

This is a **design brief** focused on foundational principles and theoretical insights. It establishes the "why" behind design decisions, extracting universal patterns from cross-domain analysis.

**Contrast with complementary documents:**
- **This doc (Universal DSL Principles):** Design philosophy, theoretical principles, the "why"
- **[DSL Framework Design](../architecture/dsl-framework-design.md):** Implementation vision, syntax proposals, the "how"
- **Current Implementation:** What Morphogen already has today (see [Architecture Overview](../architecture/overview.md))

**Think of it this way:**
```
Universal DSL Principles (philosophy)
    ↓ informs ↓
DSL Framework Design (architecture vision)
    ↓ guides ↓
Current Implementation (code & specs)
```

**Purpose:**
- Establish design principles for universal cross-domain DSLs
- Guide Morphogen's evolution toward true domain universality
- Provide theoretical foundation for cross-domain translation
- Inform architectural decisions about hybrid systems and multi-scale modeling

**Prerequisites:**
- [Formalization and Knowledge](formalization-and-knowledge.md) — Why formalization matters
- [Operator Foundations](operator-foundations.md) — Mathematical operator theory
- [Universal Domain Frameworks](../reference/universal-domain-frameworks.md) — Theoretical foundations

**Related:**
- [DSL Framework Design](../architecture/dsl-framework-design.md) ⭐ — Future vision for domain reasoning language (implementation)
- [Continuous-Discrete Semantics](../architecture/continuous-discrete-semantics.md) — Technical implementation
- [Transform Composition](../specifications/transform-composition.md) — Composable transforms specification
- [Universal Domain Translation](../adr/012-universal-domain-translation.md) — ADR for translation framework

---

## The Eight Core Principles

### 🔶 Principle 1: Domains Differ Fundamentally in Continuity

**The structural divide you must respect:**

**Continuous domains:**
- Differentiable
- Linearizable
- Transformable (Fourier, Laplace, wavelet)
- Globally analyzable
- State evolves smoothly

**Examples:** Audio signals, temperature fields, fluid flow, electromagnetic fields

**Discrete domains:**
- Inherently stepwise
- Require Δt or explicit iteration
- Nonlinear and combinatorial
- No global transforms
- Must be simulated or executed

**Examples:** State machines, cellular automata, discrete events, logic circuits

**Design Implication:**

Your DSL needs **two different semantic backbones:**

```
Continuous Calculus: flows, derivatives, operators, transforms
Discrete Calculus: state machines, transitions, events, computations
```

**And you need a clean way to bridge them.**

This is the most important conceptual takeaway.

---

### 🔶 Principle 2: Every Domain Has Signals, States, and Operators

**The invariant ontological triad:**

**1. Signals (inputs/outputs over time/space)**

This could be:
- Voltage, current (circuits)
- Concentration, pressure (chemistry, fluids)
- Position, velocity (physics)
- Symbols, tokens (computation)
- Distributions, beliefs (probabilistic)
- Actions, rewards (RL)

**2. State (internal, persistent variables)**

- DNA sequences (biology)
- Registers (computation)
- Dynamic variables (physics)
- Memory, populations (ecology)
- Beliefs (AI)

**3. Operators (transformations)**

- Filters (signal processing)
- Transitions (state machines)
- Functions (mathematics)
- Update laws (dynamics)
- Inference rules (logic)
- Chemical reactions (chemistry)

**Design Implication:**

Across domains, the **formal shape varies** but the **concept pattern is invariant**.

Your DSL should be built around these invariants:

```morphogen
// Universal pattern
@state <state_vars>     // State
flow(dt) {              // Signal evolution
    <state> = <op>(<state>, ...)  // Operators
}
```

---

### 🔶 Principle 3: Continuous Domains Use Operators; Discrete Domains Use Update Rules

**Critical for translation:**

**In continuous systems:**
- Operators act continuously
- Described by differential equations
- Evolution is smooth

```morphogen
// Continuous operator
temp = diffuse(temp, rate=0.1, dt)  // ∂T/∂t = α∇²T
```

**In discrete systems:**
- Updates act iteratively
- Described by recurrence relations
- Evolution is stepwise

```morphogen
// Discrete update rule
state = transition(state, event)    // s_{n+1} = f(s_n, e)
```

**Design Implication:**

Your DSL should make it explicit whether an operation is:
- **Continuous operator** (flow-based calculus)
- **Discrete update rule** (step-based computation)

This dual semantics enables:
- Correct compilation (continuous → ODE solver, discrete → iterator)
- Correct translation (continuous ↔ discrete via sampling/interpolation)
- Hybrid systems (mix both seamlessly)

---

### 🔶 Principle 4: Transform Spaces Are Where Problems Become Solvable

**Every domain has some representation where things simplify:**

| Domain | Transform Space | Why It Matters |
|--------|----------------|----------------|
| **Audio** | Fourier space | Convolution → multiplication, filters = spectral masks |
| **Physics** | Phase space | Energy conservation visible, trajectories analyzable |
| **Linear Algebra** | Eigenbasis | Diagonal form, decoupled dynamics |
| **Machine Learning** | Latent embeddings | Semantic structure, interpolation |
| **Evolution** | Fitness landscapes | Selection gradients, attractors |
| **Chemistry** | Reaction networks | Mass conservation, steady states |
| **Computation** | State graphs | Reachability, deadlocks, cycles |

**Design Implication:**

Your DSL should allow each domain to **define its own natural representation space** and describe **translators between them** as mappings between representations.

```morphogen
// Domain defines natural representations
domain audio {
    representations: [time, frequency, wavelet, mel]
    natural: time

    transform time -> frequency: fft
    transform frequency -> time: ifft
    transform frequency -> mel: mel_scale
}
```

This gives your DSL **universality without forcing all domains into a single primitive**.

---

### 🔶 Principle 5: Computation = When Closed-Form Analysis Fails

**Principled distinction:**

**In continuous domains → analytics is possible:**
- Solve differential equations symbolically
- Eigendecomposition
- Transform methods (Laplace, Fourier)

**In discrete domains → analytics usually isn't:**
- State space explosion
- Combinatorial complexity
- Must simulate or execute

**Design Implication:**

Your DSL should know when an operation must be **executed vs. solved**:

```morphogen
// DSL can choose strategy
@solver(strategy="symbolic_if_possible")
flow(dt=0.01) {
    // Try symbolic solution first
    // Fall back to numeric integration if needed
    x = integrate(dx_dt, x, dt)
}
```

This enables:
- **Symbolic execution** (when possible)
- **Numeric simulation** (when necessary)
- **Hybrid reasoning** (symbolic + numeric)

---

### 🔶 Principle 6: Biological, Computational, and Physical Systems Share Universal Structure

**The minimal computation triad:**

**1. Memory**
- DNA (biology)
- Registers (computation)
- State variables (physics)

**2. Conditionals / Regulation**
- Operons, promoters (biology)
- Logic gates, branches (computation)
- State transitions (physics)

**3. Loops / Recurrence / Dynamics**
- Metabolic cycles (biology)
- Clock cycles, recursion (computation)
- ODEs, oscillators (physics)

**Design Implication:**

This **"universal minimal computation"** gives your DSL grounding for modeling:
- **Engineered systems** (circuits, software)
- **Natural systems** (biology, ecology)
- **Hybrid multi-scale systems** (synthetic biology, cyber-physical)

```morphogen
// Universal computation pattern
@state memory : <type>              // Memory
flow(dt) {
    if condition(memory):           // Conditional
        memory = update(memory)     // Loop/recurrence
}
```

It's a **unifying framework for cross-domain operations**.

---

### 🔶 Principle 7: Hybrid Systems Must Be First-Class

**Many real domains mix continuous and discrete:**

| Domain | Continuous Component | Discrete Component |
|--------|---------------------|-------------------|
| **Biology** | Metabolite concentrations | Gene switches, operons |
| **Economics** | Capital flows | Decisions, policies |
| **Cyber-Physical** | Physical dynamics | Control logic |
| **Neuroscience** | Membrane potentials | Action potentials (spikes) |
| **Evolution** | Population densities | Mutations, speciation |

**Design Implication:**

Very few DSLs do hybrid modeling well. Yours can — if you design it around:

```morphogen
// Hybrid system primitives
@continuous operators   // Smooth evolution
@discrete transitions   // Stepwise changes
@events                 // Trigger conditions
@guards                 // Conditional logic
@multi_rate             // Different time scales
```

Example:
```morphogen
@state temp : Field2D<f32>          // Continuous
@state phase : Enum<Solid|Liquid>   // Discrete

flow(dt=0.01) {
    // Continuous evolution
    temp = diffuse(temp, rate=0.1, dt)

    // Discrete transition triggered by guard
    when temp.max() > 100.0:
        phase = transition(phase, Solid -> Liquid)
        temp = apply_latent_heat(temp, phase)
}
```

This is **critical if your goal is universality**.

---

### 🔶 Principle 8: Translators Are Mappings Between Representations

**Not literal converters:**

The translator between domains shouldn't try to force equivalence; it should:

1. **Define how a structure in one domain maps into the representation space of another**
2. **Preserve relevant invariants** (energy, mass, information)
3. **Drop irrelevant details** (microscopic noise, individual identities)
4. **Express what is computable, approximate, or irreducible**

**Design Implication:**

Translation should be **explicit and declarative**:

```morphogen
// Explicit translation with semantics
translate agents -> field {
    method: kernel_density_estimation
    preserves: total_mass, center_of_mass
    drops: individual_identity, velocity_distribution
    approximate: true
    error_bound: 1e-3
}
```

This makes translation **reliable instead of lossy guesswork**.

---

## The Three Most Important Takeaways

**If you had to choose only three:**

### 1. Dual Semantics (Continuous + Discrete)

**Everything else sits on top of this divide.**

Continuous and discrete are fundamentally different computational models. Your DSL must support both with:
- Different execution semantics
- Different optimization strategies
- Clean bridging mechanisms

### 2. Representation Spaces Are First-Class

**Domains must define their natural "representation space".**

Your DSL must allow transformations between these spaces:
- FFT: time → frequency
- Eigendecomposition: standard basis → eigenbasis
- Phase space: position/velocity → Hamiltonian

This is how problems become solvable.

### 3. Hybrid Systems Are First-Class Citizens

**To reflect real-world complexity.**

Most real systems mix continuous and discrete:
- Events trigger continuous changes
- Continuous thresholds trigger discrete transitions
- Multiple time scales coexist

Your DSL must handle this natively, not as an afterthought.

---

## How Morphogen Implements These Principles

### ✅ Already Implemented

**Principle 2: Signals, States, Operators**
```morphogen
@state vel : Field2D<Vec2<f32>>     // State
flow(dt=0.01) {                      // Signals
    vel = advect(vel, vel, dt)       // Operators
}
```

**Principle 4: Transform Spaces**
```morphogen
use transform
let spectrum = fft(signal)           // Time → Frequency
let kspace = fft2d(field)            // Space → k-space
```

**Principle 7: Hybrid Systems (partial)**
```morphogen
// Field (continuous) + Agents (discrete) coupling
flow(dt=0.01) {
    force_field = agents_to_field(agents)
    temp = diffuse(temp, dt)
    agents = agents.map(|a| apply_field_force(a, force_field))
}
```

---

### ⚠️ Gaps to Address

**Principle 1: Explicit Continuous/Discrete Distinction**

**Current:** Implicit based on domain
**Needed:**
```morphogen
@continuous operator diffuse(...)
@discrete operator agents.map(...)
@hybrid operator couple_field_to_agents(...)
```

**Principle 4: Representation Spaces (formalized)**

**Current:** Transforms exist but not unified
**Needed:**
```morphogen
domain audio {
    representations: [time, frequency, wavelet]
    natural: time
    transforms: {
        time -> frequency: fft,
        frequency -> time: ifft
    }
}
```

**Principle 5: Symbolic vs. Numeric**

**Current:** Always numeric
**Needed:**
```morphogen
@solver(strategy="symbolic_if_possible")
flow(dt) {
    x = solve(dx_dt == f(x), x, dt)  // Try symbolic, else numeric
}
```

**Principle 7: Events and Guards**

**Current:** Everything time-stepped
**Needed:**
```morphogen
flow(dt=0.01) {
    system = evolve_continuous(system, dt)

    when system.temperature > 100.0:
        system = trigger_phase_change(system)
}
```

**Principle 8: Translation Semantics**

**Current:** Ad-hoc cross-domain operators
**Needed:**
```morphogen
translate agents -> field {
    method: kde
    preserves: [mass, momentum]
    drops: [individual_id]
    approximate: true
}
```

---

## Design Patterns for Universal DSLs

### Pattern 1: Representation-Polymorphic Operators

**Allow operators to work in multiple representations:**

```morphogen
// Same operator, different representations
filter(signal, cutoff=1000Hz)              // Time domain
filter(spectrum, cutoff=1000Hz)            // Frequency domain (faster)

// DSL chooses optimal representation
@optimize(representation="auto")
flow(dt) {
    filtered = signal |> filter(cutoff=1000Hz)
}
```

### Pattern 2: Multi-Scale Composition

**Support operations at different temporal/spatial scales:**

```morphogen
@multi_scale {
    molecular: dt=1e-15,    // Femtoseconds
    cellular: dt=1e-3,      // Milliseconds
    tissue: dt=0.1          // Deciseconds
}

flow() {
    molecular_state = evolve(molecular_state, molecular.dt)
    cellular_state = couple_to_molecular(cellular_state, molecular_state)
    tissue_state = couple_to_cellular(tissue_state, cellular_state)
}
```

### Pattern 3: Composable Named Transforms

**Define transforms as reusable, composable units:**

```morphogen
// Define named transform
@transform audio_to_mel_spectrogram {
    signal -> stft -> magnitude -> mel_scale -> log
}

// Use as single operation
let mel_spec = audio_to_mel_spectrogram(audio_signal)

// Inverse is automatic
let reconstructed = inv(audio_to_mel_spectrogram)(mel_spec)
```

### Pattern 4: Invariant-Preserving Translation

**Make invariants explicit in cross-domain translation:**

```morphogen
translate field<temperature> -> agents<thermal_particles> {
    method: monte_carlo_sampling
    preserves: {
        total_energy: ∫ temp dx = Σ particle.energy,
        spatial_mean: mean(temp) = mean(particle.position)
    }
    n_particles: 10000
}
```

---

## Implications for Language Design

### Type System

**Must support:**
- Continuous vs. discrete distinction
- Representation tags (time, frequency, phase, etc.)
- Invariant specifications (energy, mass, etc.)
- Multi-scale/multi-rate types

```morphogen
// Rich type system
signal : Stream<f32, audio:time, 48kHz>
spectrum : Stream<Complex<f32>, audio:frequency, 24kHz>
field : Field2D<f32 [K], continuous>
agents : Agents<Particle, discrete>
```

### Compilation

**Must support:**
- Dual code generation (ODE solvers vs. iterators)
- Representation-aware optimization
- Multi-rate scheduling
- Event-driven execution

### Runtime

**Must support:**
- Hybrid continuous-discrete simulation
- Multi-scale time stepping
- Event queues and guards
- Adaptive solvers (symbolic → numeric fallback)

---

## Further Reading

### Foundational Theory

**Philosophy:**
- [Formalization and Knowledge](formalization-and-knowledge.md) — Why formalization matters
- [Operator Foundations](operator-foundations.md) — Spectral and operator theory
- [Categorical Structure](categorical-structure.md) — Category-theoretic formalization

**Architecture:**
- [Continuous-Discrete Semantics](../architecture/continuous-discrete-semantics.md) — Technical implementation
- [Domain Architecture](../architecture/domain-architecture.md) — Domain specifications

**Specifications:**
- [Transform Composition](../specifications/transform-composition.md) — Composable transforms
- [Transform Dialect](../specifications/transform.md) — Current transform infrastructure

**ADRs:**
- [Universal Domain Translation](../adr/012-universal-domain-translation.md) — Translation framework

### External References

**Hybrid Systems:**
- **"Hybrid Systems: Computation and Control"** — HSCC conference proceedings
- **"Modeling and Simulation of Cyber-Physical Systems"** — Lee & Seshia

**Multi-Scale Modeling:**
- **"Multiscale Methods"** — Weinan E
- **"The Art of Molecular Dynamics Simulation"** — Rapaport

**Domain Translation:**
- **"Model Transformation Languages"** — ACM survey
- **"Functorial Semantics"** — Lawvere

---

## Summary

**The universal DSL requires:**

1. **Dual semantics** — Continuous and discrete as first-class
2. **Representation spaces** — Domains define natural representations
3. **Transform composition** — Named, composable, invertible transforms
4. **Hybrid systems** — Events, guards, multi-rate execution
5. **Invariant-preserving translation** — Explicit semantics for cross-domain mapping

**Morphogen already has:**
- Signals, states, operators triad
- Transform infrastructure
- Cross-domain coupling (fields ↔ agents)

**Morphogen needs:**
- Explicit continuous/discrete distinction
- Formalized representation spaces
- Event-driven execution
- Translation semantics

**The goal:**
> A universal computational substrate where **any domain can be expressed**, **any representation can be chosen**, and **any translation can be made explicit and verifiable**.

---

**Next:** See [Continuous-Discrete Semantics](../architecture/continuous-discrete-semantics.md) for technical implementation details, or [Transform Composition](../specifications/transform-composition.md) for composable transform specifications.
