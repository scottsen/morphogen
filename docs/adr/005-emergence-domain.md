# ADR-004: Emergence Domain for Complex Systems and Artificial Life

**Date:** 2025-11-15
**Status:** Proposed
**Authors:** Claude, User
**Related:** ADR-002 (Cross-Domain Patterns), ../specifications/emergence.md

---

## Context

Morphogen has successfully evolved from an audio/visual creative coding kernel to a multi-domain physics simulation platform. We have:

- ✅ Audio DSP (oscillators, filters, effects)
- ✅ Visual/Fractal rendering
- ✅ Transform operations (FFT, STFT, wavelets)
- ✅ Fields/PDE operators
- ✅ Physics domains (FluidNetwork, ThermalODE, Combustion)
- ✅ Circuit modeling
- ✅ Geometry/CAD (TiaCAD-inspired)

However, Morphogen currently lacks support for **emergent systems** — a major class of computational models that sit naturally between:
- **Physics** (continuous PDEs, particles)
- **Optimization** (swarm algorithms)
- **Graphics** (procedural generation)
- **Artificial Life** (complex adaptive systems)

Emergent systems include:
1. **Cellular Automata (CA)** — Conway's Life, Wolfram rules, Lenia
2. **Agent-Based Models (ABM)** — Boids, Vicsek, Schelling segregation
3. **Reaction-Diffusion (RD)** — Turing patterns, Gray-Scott
4. **L-Systems** — Fractal trees, plants, morphogenesis
5. **Swarm Intelligence** — Ant colonies, slime mold, particle swarm optimization
6. **Complex Adaptive Systems** — Multi-agent evolution, stigmergy

These systems are widely used in:
- **Science:** Biology, ecology, physics (active matter), chemistry
- **Engineering:** Robotics, network design, structural optimization
- **Art:** Generative design, procedural textures, animation
- **Games:** Procedural content, AI behaviors, simulations

**The Problem:** No existing tool unifies emergence with physics, audio, geometry, and optimization. Users must stitch together:
- NetLogo (ABM)
- Golly (CA)
- Processing/openFrameworks (visuals)
- Custom code (everything else)

This fragmentation prevents cross-domain workflows like:
- Reaction-diffusion → geometry → stress testing
- Slime mold network → circuit layout → EM simulation
- Boids → acoustic scattering → audio output
- L-system trees → wind physics → visualization

---

## Decision

**We will add an EmergenceDomain to Morphogen as a core domain (v0.10-v1.0).**

The EmergenceDomain will provide:

### 1. Cellular Automata (CA)
- **Grid-based evolution** (2D/3D)
- **Rule presets:** Life, Brian's Brain, Wireworld, Wolfram rules
- **Continuous CA:** Lenia
- **CA → Field conversion** (for visualization, geometry)

### 2. Agent-Based Models (ABM)
- **Flocking:** Boids (separation, alignment, cohesion)
- **Active matter:** Vicsek model
- **Social dynamics:** Schelling segregation
- **Ecology:** Predator-prey
- **Agent ↔ Field coupling** (particle-in-cell)

### 3. Reaction-Diffusion (RD)
- **Gray-Scott system**
- **Turing patterns**
- **RD → Geometry conversion** (isosurface extraction)

### 4. L-Systems
- **String evolution** (recursive grammar)
- **Turtle graphics interpretation** → 3D geometry
- **Parametric growth** (trees, corals, fractals)

### 5. Swarm Intelligence
- **Ant Colony Optimization (ACO)** — pathfinding, routing
- **Slime Mold (Physarum)** — network optimization
- **Firefly Algorithm** — optimization

### 6. Cross-Domain Integration
- **Emergence → Geometry** (patterns → surfaces, L-systems → meshes)
- **Emergence → Physics** (agents → rigid bodies, networks → structures)
- **Emergence → Acoustics** (swarms → scatterers)
- **Emergence → Optimization** (PSO integration, ACO routing)
- **Emergence → Audio** (sonification of density, position)

---

## Why EmergenceDomain Fits Morphogen

### 1. Emergent Systems Are Graph-Friendly

**CA, ABM, RD, L-systems are fundamentally local and composable** — exactly Morphogen's operator model:

```morphogen
// Natural composition
let ca = ca.create(512, 512, init="random")
let evolved = ca.step(ca, rule=life_rule)
let field = ca.to_field(evolved)
let geometry = geom.extrude(field, height=field * 10mm)
```

This is **impossible** in traditional tools (NetLogo, Golly, etc.).

---

### 2. Cross-Domain Workflows Unlock Innovation

**Example 1: Biological Morphogenesis → 3D Printing**
```morphogen
// Reaction-diffusion pattern
(u, v) = rd.gray_scott(u, v, f=0.04, k=0.06)

// Convert to geometry
let heightmap = v
let surface = geom.displace(plane, heightmap, scale=20mm)

// Stress test structure
let stress = physics.stress_test(surface, load=100N)

// Export for 3D printing
io.export_stl(surface, "organic_structure.stl")
```

**Example 2: Slime Mold Network → Circuit Layout**
```morphogen
// Generate optimal network
let network = swarm.slime_mold(field, food_sources=component_positions)

// Convert to PCB traces
let traces = circuit.from_network(network, width=0.2mm)

// EM simulation
let fields = circuit.em_solve(traces, frequency=2.4GHz)
```

**Example 3: Boids → Acoustic Scattering → Audio**
```morphogen
scene SwarmAcoustics {
    let boids = agent.boids(boids, count=1000)
    let positions = agent.positions(boids)

    // Acoustic wave scattering
    let wave = acoustic.propagate_with_scatterers(
        source = point_source,
        scatterers = positions,
        radius = 0.1m
    )

    // Audio output
    out audio = acoustic.to_audio(wave, mic_position=vec3(10, 0, 0))

    // Visual output
    out visual = visual.render_agents(boids, color_by=agent.density)
}
```

**No existing tool can do these workflows.**

---

### 3. GPU Acceleration Is Natural

All emergence operators are embarrassingly parallel:

**CA:**
- Each cell update is independent
- Stencil access → shared memory optimization
- Double-buffering → no synchronization

**ABM:**
- Spatial indexing → parallel neighbor search
- Force accumulation → reduction
- Position updates → fully parallel

**RD:**
- Laplacian operator → stencil (already in FieldDomain)
- Integration → parallel per-cell

**L-Systems:**
- String evolution → parallel character replacement
- Turtle interpretation → sequential (CPU) or batched (GPU)

---

### 4. Determinism Is Achievable

**Strict Determinism:**
- CA evolution (deterministic rules)
- L-system string rewriting (deterministic grammar)
- Agent updates (with stable ordering + deterministic RNG)

**Reproducible Determinism:**
- RD integration (iterative solvers with fixed tolerance)
- Swarm algorithms (deterministic RNG, stable force ordering)

**This matches Morphogen's determinism profiles perfectly.**

---

### 5. Unique Differentiator

**Existing tools:**
- **NetLogo:** ABM only, no physics/audio/geometry
- **Golly:** CA only, no integration
- **Processing/p5.js:** Visual scripting, ad-hoc implementations
- **MATLAB/Python:** Fragmented libraries, no unified pipeline
- **Game engines (Unity/Unreal):** Not deterministic, hard to extend

**Morphogen + EmergenceDomain:**
- ✅ CA + ABM + RD + L-systems + Swarms in one system
- ✅ Cross-domain integration (Geometry, Physics, Audio, Optimization)
- ✅ GPU acceleration
- ✅ Deterministic execution
- ✅ MLIR-based compilation
- ✅ Type + unit safety

**No competitor offers this.**

---

## Consequences

### Positive

1. **New User Communities**
   - Generative artists
   - Biological modelers
   - Game developers (procedural content)
   - Swarm robotics researchers
   - Complexity scientists

2. **Novel Research Applications**
   - Differentiable emergence (autodiff + CA/ABM)
   - Physics-informed emergence (RD + heat equation)
   - Evolutionary design (L-systems + optimization + physics)

3. **Portfolio Projects**
   - Biological morphogenesis → 3D printing
   - Swarm intelligence network design
   - Acoustic swarm simulation
   - Generative architecture

4. **Academic Impact**
   - "Morphogen: A Unified Platform for Emergent Systems and Multi-Physics Simulation"
   - First deterministic, GPU-accelerated emergence platform
   - Cross-domain workflows unprecedented in literature

---

### Negative / Risks

1. **Implementation Complexity**
   - 6 sub-domains (CA, ABM, RD, L-systems, swarms, hybrids)
   - ~50 new operators
   - Spatial indexing infrastructure (k-d tree, grid)
   - Extensive testing (determinism, conservation, patterns)

   **Mitigation:** Phased rollout (v0.10: CA + ABM, v0.11: RD + L-systems, v1.0: Swarms)

2. **Performance Expectations**
   - Large grids (1024×1024 CA) → memory bandwidth
   - Many agents (100k+ boids) → neighbor search O(N²)

   **Mitigation:** GPU acceleration, spatial indexing (O(N log N)), tiling strategies

3. **Documentation Burden**
   - Each sub-domain needs specification, examples, tutorials
   - Cross-domain examples require deep domain knowledge

   **Mitigation:** Incremental documentation, example-driven approach

4. **Scope Creep**
   - Users may request every CA rule variant
   - Every swarm algorithm (bee colonies, fish schools, etc.)

   **Mitigation:** Start with core algorithms, extensible operator registry

---

## Alternatives Considered

### Alternative 1: Don't Add EmergenceDomain

**Pros:**
- Less implementation work
- Narrower scope (stay focused on physics/audio)

**Cons:**
- ❌ Misses huge creative coding community
- ❌ Limits Morphogen to "serious engineering" (excludes art, biology, games)
- ❌ Competitors (Processing, TouchDesigner) dominate generative art space
- ❌ No differentiator for procedural generation

**Rejected:** EmergenceDomain is too valuable for creative + scientific applications.

---

### Alternative 2: Integrate Existing Libraries (NetLogo, Golly APIs)

**Pros:**
- Leverage existing CA/ABM implementations
- Faster time-to-market

**Cons:**
- ❌ No determinism guarantees
- ❌ Can't integrate with Morphogen's type system
- ❌ No GPU acceleration
- ❌ No cross-domain composition (NetLogo can't export to Morphogen geometry)
- ❌ Dependency hell (FFI, version conflicts)

**Rejected:** Breaks Morphogen's unified compilation model.

---

### Alternative 3: EmergenceDomain as Plugin

**Pros:**
- Optional for users who don't need it
- Doesn't bloat core

**Cons:**
- ❌ Fragments ecosystem (core vs. plugin)
- ❌ Harder to maintain cross-domain integration
- ❌ Less polished than built-in domain

**Rejected:** Emergence is too fundamental (affects Geometry, Physics, Optimization).

---

## Implementation Plan

### Phase 1: Core Infrastructure (v0.10)
- [ ] `CAGrid2D<T>` / `CAGrid3D<T>` types
- [ ] `Agents<A>` container type
- [ ] Spatial indexing (grid-based, k-d tree)
- [ ] Boundary conditions (periodic, fixed, mirror)

### Phase 2: CA Operators (v0.10)
- [ ] `ca.create`, `ca.step`, `ca.step_n`
- [ ] `ca.rule_preset` (Life, Brian's Brain, Wireworld, Rule 30/110)
- [ ] `ca.lenia` (continuous CA)
- [ ] `ca.to_field`
- [ ] Tests: determinism, pattern recognition (gliders, oscillators)
- [ ] Example: texture generation, Game of Life animation

### Phase 3: ABM Operators (v0.10)
- [ ] `agent.create`, `agent.remove`
- [ ] `agent.boids` (separation, alignment, cohesion)
- [ ] `agent.vicsek` (active matter)
- [ ] `agent.schelling` (social dynamics)
- [ ] `agent.to_field` / `agent.from_field` (particle-in-cell)
- [ ] Tests: momentum conservation, determinism
- [ ] Example: flocking animation, crowd simulation

### Phase 4: RD Operators (v0.11)
- [ ] `rd.gray_scott`
- [ ] `rd.turing`
- [ ] `rd.to_geometry` (marching cubes)
- [ ] Tests: pattern formation, convergence
- [ ] Example: biological patterns, texture → geometry

### Phase 5: L-Systems (v0.11)
- [ ] `lsys.create`, `lsys.evolve`
- [ ] `lsys.to_geometry` (turtle graphics)
- [ ] Presets: fractal trees, Koch curve, Hilbert curve
- [ ] Tests: string evolution correctness
- [ ] Example: tree generation, coral structures

### Phase 6: Swarm Intelligence (v1.0)
- [ ] `swarm.ants` (ACO)
- [ ] `swarm.slime_mold` (Physarum network)
- [ ] `swarm.firefly` (optimization)
- [ ] Integration with OptimizationDomain (PSO already there)
- [ ] Tests: convergence, optimality
- [ ] Example: pathfinding, network generation

### Phase 7: Cross-Domain Examples (v1.0)
- [ ] Emergence → Geometry (RD heightmap, L-system tree)
- [ ] Emergence → Physics (agent → rigid body, network → structure)
- [ ] Emergence → Acoustics (boids → scatterers)
- [ ] Emergence → Audio (sonification)
- [ ] Portfolio project: biological morphogenesis → 3D print

---

## Success Metrics

**Technical:**
- [ ] 50+ emergence operators implemented
- [ ] 100% determinism for strict-tier operators
- [ ] GPU acceleration for CA (10x+ speedup vs. CPU)
- [ ] Spatial indexing (O(N log N) for 100k+ agents)

**Creative:**
- [ ] 10+ cross-domain examples
- [ ] Gallery of generative art outputs
- [ ] 3D-printable procedural models

**Community:**
- [ ] "Show me a cool Morphogen emergence example" becomes common demo
- [ ] Artists adopt Morphogen for generative work
- [ ] Academic papers cite EmergenceDomain

---

## Related Decisions

- **ADR-002:** Cross-Domain Architectural Patterns (anchors, references, auto-composition)
  - EmergenceDomain follows same patterns for integration
- **ADR-003:** Circuit Modeling Domain
  - Shows precedent for adding complex multi-operator domains
- **../specifications/geometry.md:** Geometry domain (TiaCAD patterns)
  - Emergence → Geometry is critical integration point
- **../specifications/physics-domains.md:** Physics domains
  - Emergence → Physics enables agent-based structural optimization

---

## References

### Academic

1. **Wolfram, S.** "A New Kind of Science" (2002) — CA foundations
2. **Reynolds, C.** "Flocks, Herds, and Schools" (1987) — Boids
3. **Turing, A.** "The Chemical Basis of Morphogenesis" (1952) — RD
4. **Chan, B.** "Lenia: Biology of Artificial Life" (2019) — Continuous CA
5. **Tero, A. et al.** "Rules for Biologically Inspired Adaptive Network Design" (2010) — Slime mold

### Tools

- **NetLogo** (ABM platform)
- **Golly** (CA simulator)
- **Processing / p5.js** (creative coding)
- **TouchDesigner** (visual creative platform)

### Morphogen Specs

- **../specifications/emergence.md** — Complete domain specification
- **../reference/emergence-operators.md** — Operator catalog

---

## Decision Outcome

**✅ APPROVED for v0.10-v1.0 implementation**

**Rationale:**
1. Natural fit for Morphogen's operator model
2. Unique cross-domain integration (no competitor offers this)
3. Unlocks major user communities (artists, biologists, game devs)
4. GPU-accelerable, deterministic, composable
5. Phased rollout mitigates implementation risk

**Next Steps:**
1. ✅ Create ../specifications/emergence.md (completed)
2. ✅ Create ADR-005 (this document)
3. ✅ Create ../reference/emergence-operators.md (completed)
4. ⏳ Update ../architecture/domain-architecture.md (add EmergenceDomain to Next-Wave tier)
5. ⏳ Create example: CA → Geometry → Visualization
6. ⏳ Implement Phase 1 (v0.10): Core infrastructure + CA
7. ⏳ Implement Phase 2 (v0.10): ABM
8. ⏳ Implement Phases 3-6 (v0.11-v1.0)

---

**Authors:** Claude (AI assistant), User (domain expert)
**Review Status:** Pending community review
**Implementation Target:** v0.10 (CA + ABM), v0.11 (RD + L-systems), v1.0 (Swarms)

---

**End of ADR-004**
