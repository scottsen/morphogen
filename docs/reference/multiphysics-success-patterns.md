# ADR 005: The 12 Architectural Patterns for Multiphysics Simulation Success

**Status:** APPROVED
**Date:** 2025-11-15
**Authors:** Morphogen Architecture Team
**Supersedes:** N/A
**Related:** ADR-002 (Cross-Domain Architectural Patterns), ../architecture/gpu-mlir-principles.md

---

## Context

Building multiphysics, GPU-accelerated, simulation-heavy frameworks is notoriously difficult. Many projects fail due to architectural decisions made early that compound over time. However, large scientific codebases, game engines, VFX pipelines, and CAD kernels have independently converged on a core set of architectural patterns that dramatically increase the odds of success.

This ADR documents 12 battle-proven, highly actionable patterns directly relevant to Morphogen's goals. These are not generic platitudes—these are survival requirements for a multiphysics engine, distilled from 25+ years of collective experience across game engines (Havok, Unity DOTS, Unreal), scientific computing (OpenFOAM, PyTorch), and production systems.

**The Core Problem:**
Multiphysics systems face unique challenges:
- Multiple physical domains with different mathematical models
- CPU/GPU execution paths that must remain consistent
- Performance requirements that demand careful data layout
- Debugging complexity that scales with system interactions
- Time step stability across coupled simulations
- Deterministic reproducibility for validation

**The Core Solution:**
A specific set of architectural patterns that address these challenges systematically.

---

## Decision

Morphogen adopts the following 12 architectural patterns as **mandatory design principles** for all simulation infrastructure:

---

### Pattern 1: Keep Every Domain Pure (Never Embed Physics in Data Containers)

**Problem:** This is the #1 failure mode in multiphysics systems—when data containers know how to compute their own physics, you get unmaintainable spaghetti code.

**Bad:**
```python
class Particle:
    def compute_forces(self):  # WRONG - physics in data
        ...
    def integrate(self, dt):   # WRONG - simulation in data
        ...
```

**Good:**
```python
# Data is pure
class ParticleSet:
    positions: Array[N, 3]
    velocities: Array[N, 3]
    masses: Array[N]

# Processes are modular
class Integrator:
    def update(state, dt): ...

class ForceModel:
    def compute(state): ...

# Domain coordinates
class PhysicsDomain:
    def step(self, state, dt):
        forces = self.force_model.compute(state)
        return self.integrator.update(state, forces, dt)
```

**Morphogen Implementation:**
- `ParticleSet`, `FieldState`, `GraphState` know how to **store** data
- `Integrator`, `ForceModel`, `Solver` know how to **process** data
- `Domain` exposes **operators** for combining them

**Why This Matters:**
- Pure data = GPU-friendly, serializable, inspectable
- Pure processes = testable, composable, reusable
- Clear separation = easier debugging, easier parallelization

---

### Pattern 2: All Simulation Steps Must Be Explicit "Operators"

**Problem:** Hidden simulation steps inside other steps make debugging impossible and break determinism.

**Bad:**
```python
# integrate() secretly does many things
state = integrate(state, dt)
# Does it apply drag? Gravity? Heating? Who knows!
```

**Good:**
```python
# Every step is explicit
pipeline:
  - compute_forces
  - apply_drag
  - apply_gravity
  - integrate_symplectic(dt=0.005)
  - apply_constraints
  - check_collisions
```

**Morphogen Implementation:**
Every simulation step is an explicit operator in the execution graph. This gives you:
- **Debuggability** — Step through each operation
- **Replayability** — Record and replay exact sequences
- **Traceability** — Know what happened when
- **Composability** — Rearrange, disable, or swap steps

**Result:** This ONE principle is worth 10× reduced debugging time.

---

### Pattern 3: Keep CPU and GPU Execution as Symmetric as Possible

**Problem:** The more your CPU and GPU code paths diverge, the more pain you will feel.

**Bad:**
```python
# CPU path does things GPU can't
if backend == "cpu":
    result = complex_cpu_only_function(data)
else:
    result = simplified_gpu_version(data)  # Different behavior!
```

**Good:**
```python
# Operators implement both with identical signatures
class DiffusionOp(Operator):
    def execute_cpu(self, field, dt): ...
    def execute_gpu(self, field, dt): ...
    # Same interface, same semantics, different implementation
```

**Morphogen Design:**
```
Domain → Operator → Backend Selection
                  ↓
            CPU or GPU (identical semantics)
```

**Key Requirements:**
- Ops implement `.cpu()` and `.gpu()` forms with identical signatures
- Data structures are backend-agnostic
- Memory layout is known to the operator but not to the pipeline
- Backend selection happens at execution time, not compile time

**Why This Matters:**
- Easier testing (validate on CPU, deploy on GPU)
- Debugging (run on CPU with full debugger)
- Portability (works everywhere)
- Correctness (same results regardless of backend)

---

### Pattern 4: All Data Should Be Columnar (Struct-of-Arrays, Not Array-of-Structs)

**Problem:** Array-of-Structs (AoS) layout kills GPU performance and vectorization.

**Bad (Array-of-Structs):**
```python
class Particle:
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    mass: float

particles = [Particle(...) for _ in range(N)]  # Terrible for cache
```

**Good (Struct-of-Arrays):**
```python
class ParticleSet:
    positions: Array[N, 3]    # Contiguous
    velocities: Array[N, 3]   # Contiguous
    masses: Array[N]          # Contiguous
```

**Why SoA is Critical:**
- GPU loves contiguous arrays (coalesced memory access)
- Vectorization works automatically (SIMD-friendly)
- Memory is coherent (cache-friendly)
- Warp efficiency increases dramatically
- Cache misses drop dramatically

**Morphogen Rule:**
**ALL** simulation state is stored in columnar format. No exceptions.

**This is one of the single most important choices you can make early.**

---

### Pattern 5: Use "Field Accessors" Instead of Naked Arrays

**Problem:** Direct array access is fragile, error-prone, and hard to evolve.

**Bad:**
```python
positions = state["pos"]  # What type? What units? No validation!
velocities = state["vel"]
```

**Good:**
```python
class FieldAccessor:
    def __getitem__(self, key: str) -> Field:
        field = self._data[key]
        return Field(data=field, metadata=self._metadata[key])

state.field["position"]  # Returns typed, validated Field object
state.field["velocity"]  # Includes units, bounds, backend info
state.field["temperature"]
```

**Benefits:**
- HDF5/NetCDF serialization (automatic schema)
- Subsetting and views (safe slicing)
- GPU vs CPU mapping (automatic transfer)
- Lazy evaluation (deferred computation)
- Bounds checking (runtime safety)
- Unit validation (physical correctness)

**Morphogen Implementation:**
Every domain exposes state through field accessors, not raw arrays.

**Result:** Raw NumPy arrays = footguns. Field accessors = safe, flexible, inspectable.

---

### Pattern 6: Define Strict Simulation Phases

**Problem:** Arbitrary operator execution order leads to race conditions and non-deterministic results.

**Solution:** All serious simulation engines (Havok, Unity DOTS, Unreal, OpenFOAM, PyTorch) use the same trick: strict phase boundaries.

**Standard Simulation Loop:**
```
BEGIN STEP
  Phase 1: Gather Inputs      (read sensors, user input, previous state)
  Phase 2: Compute Forces      (calculate all forces/flows/reactions)
  Phase 3: Integrate           (update positions/velocities/fields)
  Phase 4: Apply Constraints   (collision resolution, boundary conditions)
  Phase 5: Write Outputs       (visualization, logging, checkpoints)
END STEP
```

**Morphogen Implementation:**
```python
class SimulationScheduler:
    phases = [
        Phase.INPUT,
        Phase.COMPUTE,
        Phase.INTEGRATE,
        Phase.CONSTRAIN,
        Phase.OUTPUT
    ]

    def execute_step(self, operators):
        for phase in self.phases:
            phase_ops = [op for op in operators if op.phase == phase]
            self.execute_phase(phase_ops)
```

**Key Rules:**
- Operators declare their phase
- Phases execute in strict order
- No operator can run in arbitrary order
- Cross-phase data dependencies are explicit

**Why This Matters:**
- Prevents one operator from corrupting another's assumptions
- Enables parallel execution within phases
- Guarantees deterministic execution
- Makes debugging tractable

---

### Pattern 7: Use a Functional Core, Imperative Shell Model

**Problem:** Side effects and mutable state make simulation code untestable and non-deterministic.

**Solution:** Keep the simulation core purely functional, with imperative wrappers only at the edges.

**Functional Core:**
```python
# Pure function - no side effects
def simulate_step(current_state: State, dt: float) -> State:
    forces = compute_forces(current_state)
    next_state = integrate(current_state, forces, dt)
    return apply_constraints(next_state)
```

**Imperative Shell:**
```python
# Outer layers handle I/O, rendering, UI
class SimulationEngine:
    def run(self):
        while not done:
            self.state = simulate_step(self.state, self.dt)  # Pure!
            self.render(self.state)        # Imperative
            self.log(self.state)           # Imperative
            self.handle_input()            # Imperative
```

**Benefits:**
- Pure math is **testable** (no mocks needed)
- Pure math is **parallelizable** (no race conditions)
- Pure math is **backend-agnostic** (CPU/GPU identical)
- Pure math is **cacheable** (memoization possible)
- Pure math is **deterministic** (same inputs = same outputs)

**Morphogen Principle:**
Simulation operators are pure functions. Side effects only at domain boundaries.

---

### Pattern 8: Never Mix Units — Require Explicit Domain Units

**Problem:** Unit mismatches are the source of 90% of multiphysics disasters.

**Famous Failures:**
- Mars Climate Orbiter: pounds-force vs. newtons ($327M loss)
- Gimli Glider: pounds vs. kilograms (near-fatal)
- Countless simulation bugs from mixing units

**Solution:** Force domains to declare their unit system.

**Morphogen Implementation:**
```python
class PhysicsDomain:
    units = {
        "length": "m",
        "time": "s",
        "mass": "kg",
        "temperature": "K"
    }
    coordinate_system = "cartesian"
    constants = {
        "g": 9.81,  # m/s²
        "c": 299792458  # m/s
    }
```

**Operator Type Safety:**
```python
@operator(domain="physics")
def apply_force(
    body: BodyRef,
    force: Vector3[units="N"],  # Type-checked!
    point: Vector3[units="m"]
) -> None:
    ...

# This will fail type checking:
apply_force(body, force=Vector3(1, 2, 3, units="lbf"), point=...)
```

**Cross-Domain Unit Conversion:**
```python
# Explicit conversion at domain boundaries
thermal_energy = PhysicsDomain.convert_to(
    HeatDomain.temperature,
    from_units="K",
    to_units="J"
)
```

**Result:** Unit enforcement is an **insane win** for correctness.

---

### Pattern 9: Domains Should Be Layered, Not Entangled

**Problem:** Direct dependencies between domains create unmaintainable coupling.

**Bad (Entangled):**
```python
# HeatDomain directly calls FluidDomain internals
class HeatDomain:
    def convection(self):
        velocity = FluidDomain._internal_velocity_field  # WRONG!
```

**Good (Layered):**
```python
# Domains query each other through interfaces
class HeatDomain:
    def convection(self, fluid_interface: FluidInterface):
        velocity = fluid_interface.get_velocity_field()  # Clean!
```

**Morphogen Domain Hierarchy:**
```
┌─────────────────────────────────────┐
│  High-Level Domains                 │
│  (Acoustics, Electromagnetics)      │
└──────────────┬──────────────────────┘
               │ queries
┌──────────────▼──────────────────────┐
│  Mid-Level Domains                  │
│  (Fluid, Heat, Structures)          │
└──────────────┬──────────────────────┘
               │ queries
┌──────────────▼──────────────────────┐
│  Base-Level Domains                 │
│  (Geometry, Fields, Particles)      │
└─────────────────────────────────────┘
```

**Key Rules:**
- Higher-level domains **query** lower-level domains
- Never direct access to internals
- Interfaces define contracts
- Dependencies flow downward only

**Examples:**
- `HeatDomain` asks `GeometryDomain` for surface area
- `FluidDomain` asks `GeometryDomain` for cross-sections
- `AcousticsDomain` asks `FluidDomain` for impedance curves

**Result:** This is the essence of **decoupled multiphysics**.

---

### Pattern 10: Separation of "Model Space" and "Visualization Space"

**Problem:** Using the same data structures for simulation and rendering chokes performance.

**Don't Do What Game Engines Do Wrong:**
```python
# Single mesh for both simulation and rendering - SLOW!
mesh = HighPolyMesh(vertices=1000000)
physics.simulate(mesh)   # Too expensive!
renderer.draw(mesh)      # Overkill!
```

**Morphogen Separation:**
```python
# Simulation data (coarse, fast)
sim_mesh = CoarseMesh(vertices=1000)
sim_particles = ParticleSet(N=10000)
sim_field = Field3D(resolution=(64, 64, 64))

# Visualization data (pretty, high-res)
vis_mesh = HighPolyMesh.from_simulation(sim_mesh, subdivision=4)
vis_particles = InstancedSprites.from_simulation(sim_particles)
vis_field = Field3D.upsample(sim_field, target=(256, 256, 256))
```

**Examples:**
| Simulation Space | Visualization Space |
|-----------------|-------------------|
| 64³ grid | 256³ upsampled field |
| 1K coarse mesh | 100K subdivision surface |
| 10K particles | 10K instanced sprites |
| Physics timestep (dt=0.001) | Render frames (dt=0.016) |

**Benefits:**
- Simulation runs fast (coarse data)
- Visuals look good (fine data)
- No performance tradeoff
- Different update rates possible

**Morphogen Rule:** Never assume simulation data = visualization data.

---

### Pattern 11: Never Assume Anything About Time Step Stability

**Problem:** Bad time step logic will kill the credibility of everything else.

**The Reality:**
- Fixed time steps: Simple but can miss events
- Adaptive time steps: Accurate but complex
- Different domains need different dt
- Stiff equations need special handling
- Stability depends on physics and numerics

**Morphogen Requirements:**

1. **Support Multiple Time Step Strategies:**
   ```python
   # Fixed time step
   scheduler.set_timestep(dt=0.001, mode="fixed")

   # Adaptive time step with error control
   scheduler.set_timestep(dt_initial=0.001, mode="adaptive",
                         error_tol=1e-6, dt_min=1e-6, dt_max=0.01)

   # Domain-specific time steps
   physics_domain.set_timestep(dt=0.001)
   audio_domain.set_timestep(dt=1/48000)  # 48kHz sample rate
   ```

2. **Enforce Stability Constraints:**
   ```python
   # CFL condition for fluid dynamics
   dt_max = CFL * dx / max_velocity

   # Symplectic integrators for Hamiltonian systems
   integrator = SymplecticIntegrator(order=4, dt=0.001)
   ```

3. **Provide Error Estimators:**
   ```python
   # Embedded Runge-Kutta methods
   state_next, error = rk45_step(state, dt)
   if error > tolerance:
       dt = dt * 0.5  # Reduce time step
   ```

4. **Allow Per-Domain Time Steps:**
   ```python
   # Different domains, different time scales
   fluid.step(dt=0.001)    # Fast dynamics
   heat.step(dt=0.01)      # Slower diffusion
   structure.step(dt=0.1)  # Even slower deformation
   ```

**Key Insight:** Time step stability is physics-specific. The framework must be flexible.

---

### Pattern 12: Everything Must Be Serializable & Deterministic

**Problem:** Non-reproducible simulations are useless for validation, debugging, and optimization.

**Critical Requirements:**

1. **Bit-Exact Reproducibility:**
   ```python
   # Same seed = same results, always
   sim1 = Simulation(seed=42)
   sim1.run(steps=1000)

   sim2 = Simulation(seed=42)
   sim2.run(steps=1000)

   assert sim1.state == sim2.state  # Bit-exact match
   ```

2. **Full State Serialization:**
   ```python
   # Save complete state
   state.save("checkpoint_t1000.h5")

   # Resume exactly
   state = State.load("checkpoint_t1000.h5")
   simulation.resume(state)
   ```

3. **Deterministic Random Number Generation:**
   ```python
   # Reproducible RNG with hierarchical seeding
   seed = hash64(global_seed, particle_id, timestep, operation_id)
   rng = PhiloxRNG(seed)  # Counter-based RNG
   ```

4. **Operation Ordering:**
   ```python
   # Stable ordering for parallel reductions
   particles.sort(key=lambda p: (p.id, p.creation_index))
   forces = sum_forces(particles)  # Always same order
   ```

**Benefits:**
- **Debugging:** Reproduce bugs exactly
- **Comparisons:** Validate changes
- **Regression tests:** Ensure consistency
- **Long-running sims:** Checkpoint and resume
- **Hyperparameter sweeps:** Compare configurations
- **Optimization:** Genetic algorithms, gradient-free methods

**Morphogen Commitment:**
Every simulation is reproducible bit-for-bit with a seed. No exceptions.

---

## The 3 Highest-Leverage Patterns

If Morphogen does only these three things right, everything else becomes easier:

### (1) Typed Operator Registry
Cross-domain typed operators remove chaos. Every operation is:
- Discoverable (registry inspection)
- Type-safe (compile-time checking)
- Composable (operators combine cleanly)
- Documented (metadata includes docs)

### (2) Universal Columnar State Model
Every domain benefits from SoA layout:
- GPU performance (coalesced access)
- CPU vectorization (SIMD-friendly)
- Serialization (HDF5/NetCDF compatible)
- Introspection (field metadata)

### (3) Stable Simulation Scheduler with Explicit Phases
This is what 25 years of game/physics/HPC experience all agree on:
- Deterministic execution order
- Parallel execution within phases
- Clear data dependencies
- Debuggable, traceable, replayable

---

## Consequences

### Positive

1. **Architectural Stability**
   - Proven patterns from production systems
   - Reduces risk of catastrophic refactoring
   - Clear guidelines for all developers

2. **Performance by Design**
   - GPU-friendly from day one (SoA, symmetry)
   - Optimization opportunities explicit (phases, operators)
   - No performance cliffs from bad abstractions

3. **Debuggability**
   - Explicit operators = step-through debugging
   - Pure functions = reproducible tests
   - Serialization = checkpoint debugging

4. **Correctness**
   - Unit checking prevents disasters
   - Determinism enables validation
   - Type safety catches errors early

5. **Composability**
   - Pure domains = clean interfaces
   - Layered architecture = no entanglement
   - Explicit operators = rearrangeable pipelines

6. **Scalability**
   - Columnar data = GPU-ready
   - Functional core = parallelizable
   - Separate sim/vis spaces = performance headroom

### Negative

1. **Learning Curve**
   - Developers must understand all 12 patterns
   - Upfront design effort before coding
   - Discipline required to maintain purity

2. **Implementation Overhead**
   - Field accessors vs raw arrays (more code)
   - Explicit operators vs implicit steps (more boilerplate)
   - Unit systems vs naked floats (more typing)

3. **Potential Over-Engineering**
   - Simple prototypes may feel heavyweight
   - Temptation to skip patterns for quick hacks
   - Risk of gold-plating if not balanced

### Mitigations

1. **Documentation & Training**
   - Comprehensive pattern guide (this ADR)
   - Reference implementations for each pattern
   - Code review checklist for pattern compliance

2. **Tooling Support**
   - Linters to enforce patterns (unit checking, SoA layout)
   - Code generators for boilerplate (field accessors, operators)
   - Templates for new domains (`kairo new-domain`)

3. **Progressive Adoption**
   - Core domains adopt all patterns (Geometry, Physics, Audio)
   - New domains follow by example
   - Pattern violations caught in review

4. **Pragmatism**
   - Allow exceptions with explicit justification
   - Prototype mode for exploration (with technical debt markers)
   - Balance purity with shipping

---

## Summary: The Architecture for Success

These patterns distill decades of hard-won experience into actionable principles:

| Pattern | Core Principle | Result |
|---------|---------------|--------|
| 1. Pure Domains | Data ≠ Logic | Stable, testable |
| 2. Explicit Operators | No hidden steps | Debuggable |
| 3. CPU/GPU Symmetry | Same semantics | Scalable |
| 4. Columnar Layout | SoA not AoS | Fast |
| 5. Field Accessors | Structured access | Safe |
| 6. Simulation Phases | Ordered execution | Predictable |
| 7. Functional Core | Pure functions | Reliable |
| 8. Unit Enforcement | Physical correctness | Correct |
| 9. Domain Layering | Queries not calls | Modular |
| 10. Sim/Vis Separation | Different data | Efficient |
| 11. Time Step Flexibility | Multiple strategies | Robust |
| 12. Deterministic Serialization | Reproducible | Trustworthy |

**These patterns aren't ideas — they are survival requirements for a multiphysics engine.**

---

## References

- **ADR-002:** Cross-Domain Architectural Patterns
- **../architecture/gpu-mlir-principles.md:** GPU execution patterns
- **../specifications/operator-registry.md:** Operator type system
- **../specifications/scheduler.md:** Simulation phase scheduling
- **../specifications/type-system.md:** Unit system and type safety
- **../architecture/domain-architecture.md:** Multi-domain vision

### External References

- **Game Engines:** Havok Physics, Unity DOTS, Unreal Engine
- **Scientific Computing:** OpenFOAM, PyTorch, JAX
- **Visualization:** Houdini, ParaView, VTK
- **CAD Kernels:** OpenCASCADE, CGAL, Rhino3D

---

## Adoption Checklist

For each new domain or major subsystem, verify:

- [ ] Pattern 1: Is data pure? No physics in containers?
- [ ] Pattern 2: Are all steps explicit operators?
- [ ] Pattern 3: Do CPU and GPU paths have identical semantics?
- [ ] Pattern 4: Is all state in SoA (columnar) format?
- [ ] Pattern 5: Are field accessors used instead of raw arrays?
- [ ] Pattern 6: Do operators declare their phase?
- [ ] Pattern 7: Is the simulation core purely functional?
- [ ] Pattern 8: Are units declared and enforced?
- [ ] Pattern 9: Do domains query (not call) each other?
- [ ] Pattern 10: Are simulation and visualization data separate?
- [ ] Pattern 11: Is time step handling flexible and stable?
- [ ] Pattern 12: Is state fully serializable and deterministic?

**If you can check all 12 boxes, you're building a system that will scale.**

---

**Final Word:**

These patterns emerge from necessity. They are the distilled wisdom of thousands of engineers who built systems that survived contact with reality.

Morphogen doesn't have to make the same mistakes. Follow these patterns, and you **greatly increase your odds of success**.
