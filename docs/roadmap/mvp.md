# Creative Computation DSL - MVP Roadmap

## Overview

This document outlines the Minimum Viable Product (MVP) path for Creative Computation DSL v0.2.2, focusing on delivering core functionality that demonstrates the language's unique value proposition while establishing a solid foundation for future growth.

## MVP Vision

**Goal:** Enable creative developers to write expressive, deterministic simulations that compile to efficient code through a seamless Python-based toolchain.

**Target Users:**
- Creative coders exploring generative art and simulations
- Researchers building reproducible computational experiments
- Educators teaching computational physics and creative coding

**Success Criteria:**
- Users can write and run simple field-based simulations
- Compilation pipeline works end-to-end (lexer → parser → type checker)
- At least one complete example runs successfully
- Documentation enables first-time users to get started in under 30 minutes

## MVP Scope

### Phase 1: Core Language Frontend (Weeks 1-2)

**Objective:** Establish a working parser and type system for core DSL features.

**Deliverables:**
- ✅ Lexer tokenizes all DSL syntax
- ✅ Parser builds AST for:
  - Basic types (f32, Field2D, Vec2)
  - Step blocks
  - Variable assignments
  - Function calls (field operations)
- ✅ Type checker validates:
  - Type compatibility
  - Basic unit checking (no complex unit arithmetic yet)
  - Symbol resolution
- ✅ Error reporting with line numbers and helpful messages

**Out of Scope:**
- Agents, signals, visual operations
- Advanced features (modules, compose, iterate)
- Complex unit conversions

**Example Code to Support:**
```dsl
set profile = medium
set dt = 0.01

@double_buffer vel : Field2D<f32>

vel = field.advect(vel, vel, dt)
vel = field.diffuse(vel, rate=0.1, dt)
vel = field.project(vel, method="jacobi", iter=20)
```

### Phase 2: Field Operations Runtime (Weeks 3-4)

**Objective:** Implement core field operations with NumPy-based execution (MLIR deferred to post-MVP).

**Deliverables:**
- Field allocation and memory management
- Essential field operations:
  - `field.alloc` - Create new fields
  - `field.advect` - Semi-Lagrangian advection (single method)
  - `field.diffuse` - Jacobi solver only
  - `field.project` - Pressure projection with Jacobi
  - `field.combine` - Element-wise operations
  - `field.map` - Custom functions
  - `field.boundary` - Reflect and periodic only
- Double-buffering support via @double_buffer decorator
- Step-by-step execution model

**Out of Scope:**
- Multiple solver methods (CG, multigrid, etc.)
- GPU acceleration
- Solver profiles (use defaults only)
- Performance optimization

**Success Metric:** Run a basic fluid simulation (advection + diffusion + projection) and output field values.

### Phase 3: Simple Visualization (Week 5)

**Objective:** Enable users to see simulation results visually.

**Deliverables:**
- Simple field visualization:
  - `visual.colorize` - Map field values to colors
  - `visual.output` - Save to PNG/image file
- Basic color palettes (fire, grayscale, viridis)
- No real-time rendering (file output only)

**Out of Scope:**
- Layer composition and blend modes
- Point sprite rendering for agents
- Post-processing filters
- Real-time display

**Example Code:**
```dsl
visual.output(
  visual.colorize(temp, palette="fire"),
  path="output.png"
)
```

### Phase 4: Documentation & Examples (Week 6)

**Objective:** Make the MVP accessible and usable for early adopters.

**Deliverables:**
- **Getting Started Guide:**
  - Installation instructions
  - First simulation walkthrough
  - Common patterns and idioms
- **Complete Examples:**
  - Heat diffusion (simplest)
  - Reaction-diffusion (Gray-Scott)
  - Basic fluid simulation (Navier-Stokes)
- **API Reference:** Auto-generated from code
- **Troubleshooting Guide:** Common errors and solutions

**Out of Scope:**
- Video tutorials
- Interactive playground
- Advanced examples (agents, audio, hybrid systems)

### Phase 5: Testing & Polish (Week 7)

**Objective:** Ensure reliability and smooth user experience.

**Deliverables:**
- Unit tests for:
  - Lexer tokenization
  - Parser correctness
  - Type checking rules
  - Field operations accuracy
- Integration tests:
  - End-to-end example execution
  - Determinism verification (same code → same results)
- Error message improvements
- Performance profiling for field operations

**Success Metric:** All examples run successfully, tests pass, error messages are clear and actionable.

## MVP Feature Matrix

| Domain | MVP Features | Post-MVP |
|--------|-------------|----------|
| **Types** | f32, f64, Field2D, Vec2, BoundarySpec | i32, Field3D, Vec3, Complex types |
| **Structure** | step blocks, @double_buffer | substep, module, compose, iterate |
| **Field Ops** | alloc, advect, diffuse, project, combine, map, boundary | stencil, sample_grad, integrate, laplacian, gradient |
| **Agents** | *(none)* | All agent operations |
| **Signals** | *(none)* | All signal/audio operations |
| **Visual** | colorize, output (PNG only) | points, layer, filter, coord_warp, real-time display |
| **I/O** | Basic file output | Streaming, load_field, multiple formats |
| **Solvers** | Jacobi only | CG, multigrid, preconditioners |
| **Profiles** | Default profile | Low/medium/high profiles, custom solvers |
| **Backend** | NumPy interpreter | MLIR lowering, GPU support |

## Technical Architecture (MVP)

```
┌─────────────────┐
│  .ccdsl Source  │
└────────┬────────┘
         ↓
    ┌────────┐
    │ Lexer  │
    └────┬───┘
         ↓
    ┌────────┐
    │ Parser │ → AST
    └────┬───┘
         ↓
  ┌──────────────┐
  │ Type Checker │ → Typed AST
  └──────┬───────┘
         ↓
  ┌──────────────────┐
  │ NumPy Interpreter│ → Results
  └──────┬───────────┘
         ↓
  ┌──────────────┐
  │ Visualization│ → PNG Output
  └──────────────┘
```

**MLIR Lowering:** Deferred to post-MVP. The interpreter validates the language design before investing in compilation infrastructure.

## Non-MVP Features (Explicitly Deferred)

These features are valuable but not required for initial release:

### Agent-Based Systems
- All agent operations (force_sum, integrate, spawn, remove, mutate, reproduce)
- Deterministic RNG (Philox)
- Barnes-Hut acceleration

*Rationale:* Field operations demonstrate core value; agents can be added once foundation is solid.

### Signal Processing & Audio
- All signal operations
- Audio synthesis and output
- Block-based rendering

*Rationale:* Requires different execution model; better as separate module.

### Advanced Solvers
- Conjugate Gradient (CG)
- Multigrid
- Preconditioners (ILU, Jacobi)

*Rationale:* Jacobi solver sufficient for MVP; advanced solvers needed for production performance.

### MLIR Lowering & Optimization
- MLIR dialect generation
- Operation fusion
- GPU offloading
- SIMD vectorization

*Rationale:* Interpreter validates language semantics; compilation is optimization step.

### Advanced Visual Features
- Real-time rendering
- Layer composition
- Post-processing filters
- Point sprites for agents

*Rationale:* File output sufficient for validation; interactive rendering adds complexity.

### Language Features
- Modules and composition
- Iterate loops
- Link dependencies
- Solver profiles and registries

*Rationale:* Nice-to-have for code organization; not essential for basic simulations.

## Success Metrics

### Quantitative Metrics
- [ ] 3+ complete working examples
- [ ] 80%+ test coverage for frontend
- [ ] Parse + type-check < 100ms for typical programs
- [ ] Field operations match reference implementation accuracy (1e-4 tolerance)
- [ ] Documentation covers 100% of MVP features

### Qualitative Metrics
- [ ] First-time user can run example in < 30 minutes
- [ ] Error messages are clear and actionable (user testing)
- [ ] Code feels natural and expressive (developer feedback)
- [ ] Performance adequate for interactive development (<1s per frame for 256×256 grid)

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Type system too complex | High | Start with simple types only; defer unit arithmetic |
| Field operations too slow | Medium | Profile early; NumPy is fast enough for MVP |
| Scope creep | High | Strict adherence to MVP scope; defer everything else |
| Parser bugs hard to debug | Medium | Comprehensive test suite + clear error messages |
| Users want agents/audio | Low | Clear MVP communication; roadmap for future features |

## Post-MVP Roadmap (High-Level)

### v0.3 - Agent Systems
- Add agent types and operations
- Implement deterministic RNG
- Barnes-Hut force calculation
- Agent examples (boids, evolutionary algorithms)

### v0.4 - Signal Processing
- Signal types and operations
- Audio synthesis
- Block-based rendering
- Audio examples (FM synthesis, procedural music)

### v0.5 - MLIR Lowering
- MLIR dialect generation
- Basic optimization passes
- Performance benchmarking
- Comparison with interpreted version

### v0.6 - Advanced Features
- Modules and composition
- Solver profiles
- Advanced visual features
- Comprehensive optimization

## Launch Plan

### Pre-Launch (Week 8)
- Internal testing with 3-5 external beta users
- Bug fixes from beta feedback
- Polish documentation
- Prepare release notes

### Launch (Week 9)
- Tag v0.2.2-mvp release
- Publish to PyPI (pip install creative-computation-dsl)
- Announce on relevant communities:
  - r/creativecoding
  - Processing Foundation forum
  - Academic mailing lists
- Create demo video (5 minutes)

### Post-Launch
- Monitor issues and user feedback
- Weekly bug fix releases
- Begin v0.3 development
- Gather feature requests for prioritization

## Development Guidelines

### Code Quality
- Follow existing code style
- Comprehensive docstrings
- Type hints for all public APIs
- Clear error messages with suggestions

### Testing
- Write tests before features (TDD encouraged)
- Test both success and failure cases
- Keep tests fast (<1s total for unit tests)
- Integration tests use known-good outputs

### Documentation
- Update docs with every feature
- Code examples for all operations
- Keep language reference in sync
- Write for beginners

## Timeline Summary

| Week | Phase | Focus |
|------|-------|-------|
| 1-2 | Frontend | Lexer, parser, type checker |
| 3-4 | Runtime | Field operations with NumPy |
| 5 | Visual | Simple visualization output |
| 6 | Docs | Examples and documentation |
| 7 | Polish | Testing, bug fixes, refinement |
| 8 | Beta | External testing |
| 9 | Launch | Release v0.2.2-mvp |

**Total: 9 weeks to MVP**

## Appendix: MVP Example Programs

### Example 1: Heat Diffusion
```dsl
set dt = 0.01

@double_buffer temp : Field2D<f32>

# Initialize with hot center
temp = field.map(temp, fn=init_heat)

# Diffuse heat over time
temp = field.diffuse(temp, rate=0.1, dt)
temp = field.boundary(temp, spec="reflect")

visual.output(
  visual.colorize(temp, palette="fire"),
  path="heat.png"
)
```

### Example 2: Reaction-Diffusion (Gray-Scott)
```dsl
set dt = 1.0

@double_buffer u, v : Field2D<f32>

# Initialize random seed
u = field.random(shape=(256, 256), seed=42)
v = field.random(shape=(256, 256), seed=43)

# Reaction-diffusion step
u = field.diffuse(u, rate=0.16, dt)
v = field.diffuse(v, rate=0.08, dt)

# Reaction term (via combine + map)
reaction_u = field.combine(u, v, fn=react_u)
reaction_v = field.combine(u, v, fn=react_v)

u = field.combine(u, reaction_u, fn=add)
v = field.combine(v, reaction_v, fn=add)

visual.output(
  visual.colorize(v, palette="viridis"),
  path="rd.png"
)
```

### Example 3: Simple Fluid Simulation
```dsl
set dt = 0.016  # ~60 FPS

@double_buffer vel : Field2D<Vec2>
@double_buffer density : Field2D<f32>

# Advect velocity by itself
vel = field.advect(vel, vel, dt)

# Diffuse velocity (viscosity)
vel = field.diffuse(vel, rate=0.0001, dt)

# Make incompressible
vel = field.project(vel, method="jacobi", iter=20)

# Advect and diffuse density
density = field.advect(density, vel, dt)
density = field.diffuse(density, rate=0.00001, dt)
density = field.boundary(density, spec="reflect")

visual.output(
  visual.colorize(density, palette="grayscale"),
  path="fluid.png"
)
```

---

**Document Version:** 1.0
**Last Updated:** 2025-11-05
**Status:** Proposed
**Owner:** Development Team
