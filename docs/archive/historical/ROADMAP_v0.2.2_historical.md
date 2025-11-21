# Creative Computation DSL — Development Roadmap

**Vision:** A production-grade DSL for deterministic, multi-domain simulations with MLIR compilation.

---

## Current Status (January 2025)

### ✅ Completed

**v0.2.2-alpha** (Initial Foundation)
- Language specification (comprehensive)
- Complete documentation suite
- Lexer and parser (full AST generation)
- Type system with physical units
- Type checker with error reporting
- Project structure and packaging
- CLI interface skeleton
- Example programs (5 complete examples)

**Status:** Foundation complete, ready for MVP implementation

---

## Roadmap Overview

```
v0.2.2-mvp  →  v0.3.0  →  v0.4.0  →  v0.5.0  →  v0.6.0  →  v1.0.0
 (MVP)     (Perf)   (Multi)   (Audio)   (MLIR)  (Production)
   3w         8w       8w        8w       12w        12w
```

**Total timeline:** ~1 year to v1.0.0

---

## v0.2.2-mvp — Minimum Viable Product
**Target:** Q1 2025 (3 weeks)
**Focus:** Field operations work flawlessly

### Goals
- Execute field-based programs
- Real-time visualization
- Deterministic execution
- 2-3 working examples
- Good developer experience

### Features

**Runtime (P0 - Critical)**
- [ ] Execution engine
- [ ] Expression evaluator
- [ ] Variable/state management
- [ ] Step execution loop
- [ ] Double-buffer swapping

**Field Operations (P0 - Critical)**
- [ ] Field2D data structure (NumPy-based)
- [ ] field.alloc() — Allocation
- [ ] field.map() — Element-wise operations
- [ ] field.combine() — Binary operations
- [ ] field.random() — Random initialization
- [ ] field.advect() — Semi-Lagrangian advection
- [ ] field.diffuse() — Jacobi diffusion
- [ ] field.project() — Jacobi projection
- [ ] field.laplacian() — 5-point stencil
- [ ] field.gradient() — Central difference
- [ ] field.divergence() — Divergence operator
- [ ] field.sample() — Bilinear interpolation
- [ ] field.boundary() — Periodic, clamp, reflect

**Visualization (P0 - Critical)**
- [ ] visual.colorize() — Scalar field to RGB
- [ ] visual.output() — Display window (Pygame)
- [ ] Palette support (viridis, plasma, fire, grayscale)
- [ ] Window controls (pause, step, quit)

**Determinism (P0 - Critical)**
- [ ] Philox-based RNG (or NumPy PCG64)
- [ ] Seeded field initialization
- [ ] Reproducibility tests

**Developer Experience (P1 - Important)**
- [ ] Helpful error messages with line numbers
- [ ] Parameter override via CLI
- [ ] Getting started guide
- [ ] Installation instructions

**Examples (P0 - Critical)**
- [ ] Simple diffusion
- [ ] Smoke simulation
- [ ] Reaction-diffusion (Gray-Scott)

### Success Criteria
✅ Smoke simulation runs at 30+ FPS (256×256)
✅ Same seed produces identical results
✅ User can install and run in <15 minutes
✅ Documentation is clear and complete

### Deliverables
- Working CCDSL implementation (fields only)
- 3 example programs
- GETTING_STARTED.md guide
- PyPI package (v0.2.2-mvp)

---

## v0.3.0 — Performance & Better Solvers
**Target:** Q2 2025 (8 weeks)
**Focus:** Production-grade performance for field operations

### Goals
- 10x performance improvement
- Better numerical methods
- Larger problem sizes
- GPU acceleration (basic)

### Features

**Optimization (P0)**
- [ ] Numba JIT compilation for hot paths
- [ ] Operation fusion (combine → map → diffuse)
- [ ] In-place operations where safe
- [ ] Memory pooling
- [ ] Profiling tools

**Advanced Solvers (P0)**
- [ ] MacCormack advection (higher accuracy)
- [ ] Conjugate Gradient solver (diffusion, projection)
- [ ] Multigrid solver (large grids)
- [ ] Preconditioners (Jacobi, ILU0)
- [ ] Convergence monitoring

**Field Operations (P1)**
- [ ] field.stencil() — Custom stencil kernels
- [ ] field.sample_grad() — Sample with gradient
- [ ] field.integrate() — Temporal integration
- [ ] field.react() — Reaction terms
- [ ] field.mask() — Masking operations
- [ ] field.threshold() — Thresholding

**3D Support (P1)**
- [ ] Field3D data structure
- [ ] 3D visualization (volume rendering)
- [ ] 3D examples

**GPU Support (P2)**
- [ ] CuPy backend for GPU arrays
- [ ] GPU-accelerated field operations
- [ ] Automatic CPU/GPU selection
- [ ] Performance benchmarks

### Success Criteria
✅ 512×512 smoke simulation at 60+ FPS
✅ 1024×1024 simulation feasible
✅ GPU version 5-10x faster than CPU
✅ CG solver converges in <10 iterations

### Deliverables
- High-performance field operations
- GPU support (CUDA)
- Advanced solver examples
- Performance benchmarking suite

---

## v0.4.0 — Multi-Domain (Agents + Fields)
**Target:** Q3 2025 (8 weeks)
**Focus:** Agent-based computing with field coupling

### Goals
- Agent-based simulations work
- Field-agent coupling
- Hybrid examples
- Evolutionary algorithms

### Features

**Agent System (P0)**
- [ ] Agent data structure (stable IDs)
- [ ] agent.alloc() — Allocation
- [ ] agent.map() — Per-agent transformations
- [ ] agent.force_sum() — Pairwise forces
  - [ ] Brute force method
  - [ ] Grid acceleration
  - [ ] Barnes-Hut tree
- [ ] agent.integrate() — Position/velocity updates
- [ ] agent.spawn() — Spawn new agents
- [ ] agent.remove() — Remove agents
- [ ] agent.reduce() — Reductions (sum, count, etc.)

**Field-Agent Coupling (P0)**
- [ ] agent.sample_field() — Agents read from fields
- [ ] agent.deposit() — Agents write to fields
- [ ] Two-way coupling examples

**Evolutionary Computing (P1)**
- [ ] agent.mutate() — Probabilistic mutation
- [ ] agent.reproduce() — Offspring creation
- [ ] Fitness-based selection

**Agent Visualization (P0)**
- [ ] visual.points() — Point sprites
- [ ] Color by property
- [ ] Size by property
- [ ] Trails/motion blur

**Examples (P0)**
- [ ] Boids flocking
- [ ] N-body gravity
- [ ] Predator-prey
- [ ] Physarum (slime mold)
- [ ] Evolutionary fluid hybrid

### Success Criteria
✅ 10,000 agents at 60+ FPS
✅ Barnes-Hut works correctly
✅ Field-agent coupling is seamless
✅ Deterministic agent evolution

### Deliverables
- Complete agent system
- Field-agent coupling
- 5 agent-based examples
- Agent tutorial

---

## v0.5.0 — Kairo.Audio Implementation
**Target:** Q4 2025 (8 weeks)
**Focus:** Implement Kairo.Audio dialect specification

### Goals
- Kairo.Audio dialect fully implemented
- Compositional audio synthesis
- Physical modeling primitives
- Deterministic polyphony
- RiffStack integration

### Features

**Core Audio Types (P0)**
- [ ] Sig (audio-rate stream) type
- [ ] Ctl (control-rate stream) type
- [ ] Evt<A> (event stream) type
- [ ] scene and module constructs
- [ ] Rate model and cross-rate communication

**Oscillators & Synthesis (P0)**
- [ ] sine, saw, square, tri oscillators
- [ ] Band-limited synthesis (BLEP/PolyBLEP)
- [ ] Noise generators (white, pink, brown)
- [ ] Deterministic RNG (Philox)

**Filters & Effects (P0)**
- [ ] lpf, hpf, bpf, svf filters
- [ ] delay, reverb, chorus, flanger
- [ ] drive, limiter effects
- [ ] Convolution (IR-based)

**Envelopes & Control (P0)**
- [ ] adsr, ar envelope generators
- [ ] envexp, linseg envelopes
- [ ] Expressive control (vibrato, bend)

**Physical Modeling (P1)**
- [ ] string — Karplus-Strong waveguide
- [ ] membrane — 2D waveguide
- [ ] bodyIR — Resonant body modeling
- [ ] pickup — Pickup simulation
- [ ] amp, cab — Amplification modeling

**Event System (P0)**
- [ ] score — Event sequences
- [ ] loop — Deterministic looping
- [ ] spawn — Polyphonic voice allocation
- [ ] Sample-accurate event timing

**Audio I/O (P0)**
- [ ] Real-time audio output
- [ ] Profile-based quality control
- [ ] Buffer management
- [ ] Latency handling

**RiffStack Integration (P1)**
- [ ] YAML patch import/export
- [ ] Operator registry sharing
- [ ] Live performance mode

**Examples (P0)**
- [ ] Simple pluck synthesis
- [ ] FM synthesis
- [ ] Polyphonic sequencer
- [ ] Physical modeling demo
- [ ] Audio-reactive visuals

### Success Criteria
✅ Deterministic audio rendering (bit-exact)
✅ Real-time synthesis at 44.1-96kHz
✅ Polyphonic voice allocation works
✅ Physical models sound realistic
✅ RiffStack integration functional

### Deliverables
- Complete Kairo.Audio implementation
- Audio synthesis examples
- Physical modeling examples
- RiffStack integration
- Audio dialect tutorial
- AUDIO_SPECIFICATION.md (complete)

---

## v0.6.0 — MLIR Lowering
**Target:** Q1 2026 (12 weeks)
**Focus:** Production compilation pipeline

### Goals
- MLIR compilation works
- Optimal performance
- Multi-device support
- Production quality

### Features

**MLIR Backend (P0)**
- [ ] AST → MLIR IR conversion
- [ ] Dialect selection (linalg, scf, arith, etc.)
- [ ] MLIR optimization passes
- [ ] Code generation
- [ ] JIT compilation

**Optimization (P0)**
- [ ] Operation fusion in MLIR
- [ ] Loop optimization (tiling, unrolling)
- [ ] Vectorization (SIMD)
- [ ] Memory layout optimization
- [ ] Dead code elimination

**Multi-Device (P1)**
- [ ] CPU backend (LLVM)
- [ ] GPU backend (CUDA/ROCm)
- [ ] Multi-GPU support
- [ ] Heterogeneous execution

**Advanced MLIR (P2)**
- [ ] Custom MLIR dialects for CCDSL
- [ ] Polyhedral optimization
- [ ] Autotuning
- [ ] Cross-device compilation

**Performance (P0)**
- [ ] 100x speedup over Python
- [ ] Performance profiling tools
- [ ] Optimization guides

### Success Criteria
✅ Compiled code 100x faster than interpreted
✅ Multi-GPU scaling is linear
✅ Can handle 4096×4096 grids
✅ Production-ready reliability

### Deliverables
- MLIR compilation pipeline
- Performance benchmarks
- Multi-device examples
- Optimization guide

---

## v1.0.0 — Production Release
**Target:** Q2 2026 (12 weeks)
**Focus:** Polish, stability, ecosystem

### Goals
- Production-ready
- Complete feature set
- Excellent documentation
- Growing community

### Features

**Language Completeness (P0)**
- [ ] All v0.2.2 spec features implemented
- [ ] iterate() — Dynamic loops
- [ ] link() — Dependency metadata
- [ ] compose() — Parallel execution
- [ ] Module system
- [ ] Adaptive timestep

**Visual Domain (P0)**
- [ ] visual.layer() — Layer composition
- [ ] visual.filter() — Post-processing
- [ ] visual.coord_warp() — Geometric warps
- [ ] visual.retime() — Temporal effects
- [ ] visual.text() — Text overlay
- [ ] Blend modes (alpha, add, multiply)

**I/O System (P0)**
- [ ] io.load_field() — Load from files
- [ ] io.save_field() — Save to files
- [ ] io.stream() — Real-time streams
- [ ] Video output (H.264)
- [ ] Image sequences

**Tooling (P1)**
- [ ] Interactive debugger
- [ ] Visual graph editor
- [ ] Parameter tuning UI
- [ ] Live reload
- [ ] Profiler UI

**Documentation (P0)**
- [ ] Complete API reference
- [ ] Video tutorials
- [ ] Cookbook (common patterns)
- [ ] Case studies
- [ ] Research papers

**Testing (P0)**
- [ ] Comprehensive test suite (>90% coverage)
- [ ] Conformance tests
- [ ] Determinism tests
- [ ] Cross-platform CI/CD
- [ ] Performance regression tests

**Ecosystem (P1)**
- [ ] Package repository
- [ ] Community examples
- [ ] Third-party integrations
- [ ] VSCode extension
- [ ] Jupyter kernel

### Success Criteria
✅ All specification features work
✅ Production users in multiple domains
✅ Active community contributing
✅ Stable API (semantic versioning)
✅ Publication-quality results

### Deliverables
- v1.0.0 release
- Complete documentation
- Production examples
- Research paper
- Conference presentation

---

## Beyond v1.0.0 — Future Vision

### Advanced Features
- Automatic differentiation (for ML/optimization)
- Symbolic manipulation
- Quantum simulation support
- Distributed execution (MPI)
- Cloud deployment

### Performance
- TPU support
- Custom hardware backends
- WebGPU for browser
- Mobile deployment

### Applications
- Scientific computing packages
- Creative coding framework
- Game engine integration
- VFX pipeline tools
- Educational platform

### Research
- Novel optimization techniques
- Domain-specific optimizations
- Program synthesis
- AI-assisted coding

---

## Development Principles

### Throughout All Versions

**Quality Over Speed**
- Correctness first
- Performance second
- Features third

**User Experience**
- Clear error messages
- Helpful documentation
- Smooth learning curve
- Fast iteration time

**Determinism**
- Reproducible by default
- Bit-exact when possible
- Documented nondeterminism

**Performance**
- Profile before optimizing
- Measure everything
- Maintain benchmarks

**Community**
- Open development
- Responsive to feedback
- Clear contribution process
- Inclusive environment

---

## Version Summary

| Version | Focus | Duration | Key Deliverable |
|---------|-------|----------|-----------------|
| v0.2.2-mvp | Fields work | 3 weeks | Smoke simulation |
| v0.3.0 | Performance | 8 weeks | GPU support |
| v0.4.0 | Multi-domain | 8 weeks | Agent-field coupling |
| v0.5.0 | Kairo.Audio | 8 weeks | Audio synthesis & physical modeling |
| v0.6.0 | MLIR | 12 weeks | Production compiler |
| v1.0.0 | Production | 12 weeks | Complete system |
| **Total** | | **~1 year** | Production DSL |

---

## How to Contribute

### Current Phase (MVP)
See [MVP.md](MVP.md) for detailed task breakdown.

**High Priority:**
- Runtime engine implementation
- Field operations (NumPy-based)
- Visualization (Pygame)
- Example programs
- Testing

**How to Help:**
1. Check [MVP.md](MVP.md) for open tasks
2. Pick a task that interests you
3. Discuss approach in GitHub issue
4. Submit PR with tests
5. Iterate based on feedback

### Future Contributions
As we progress through versions, we'll need help with:
- Advanced solvers
- GPU kernels
- MLIR lowering
- Documentation
- Example programs
- Testing

---

**Last Updated:** January 2025
**Status:** Active Development (MVP Phase)
**Next Milestone:** v0.2.2-mvp (3 weeks)
