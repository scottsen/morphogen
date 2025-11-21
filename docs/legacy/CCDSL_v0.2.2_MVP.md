# Creative Computation DSL v0.2.2 — MVP Definition

**Goal:** Deliver a working, usable implementation that demonstrates the core value proposition of CCDSL while being achievable in a reasonable timeframe.

---

## Table of Contents

1. [MVP Vision](#mvp-vision)
2. [Success Criteria](#success-criteria)
3. [Core Features (Must Have)](#core-features-must-have)
4. [Deferred Features (Later Versions)](#deferred-features-later-versions)
5. [Implementation Phases](#implementation-phases)
6. [Target Examples](#target-examples)
7. [Testing Requirements](#testing-requirements)
8. [Performance Targets](#performance-targets)
9. [Documentation Requirements](#documentation-requirements)
10. [Development Timeline](#development-timeline)

---

## MVP Vision

### What We're Building

A **proof-of-concept DSL implementation** that:
- Demonstrates deterministic simulation with strong types
- Shows CCDSL's expressiveness for at least one domain
- Provides working examples users can run and modify
- Validates the language design and compilation approach

### What Success Looks Like

A user can:
1. Write a simple CCDSL program
2. Type-check it with `ccdsl check`
3. Run it with `ccdsl run` and see visual output
4. Get deterministic, reproducible results
5. Modify parameters and see effects

### MVP Philosophy

**"One domain done well beats four domains done poorly"**

Focus on making **field operations** (PDE toolkit) work flawlessly first, then expand to other domains. This provides immediate value for fluid dynamics, reaction-diffusion, and scientific computing users.

---

## Success Criteria

### Critical (Must Pass)

✅ **SC-1: Basic Program Execution**
- Parse and execute a simple field program
- Display visual output in a window
- Support at least 10 consecutive timesteps

✅ **SC-2: Type Safety**
- Detect type mismatches at compile time
- Enforce unit compatibility
- Provide helpful error messages

✅ **SC-3: Determinism**
- Same seed → identical results (bit-exact)
- Cross-platform reproducibility (within floating-point tolerance)
- Stable execution across runs

✅ **SC-4: Field Operations**
- Advection (semi-Lagrangian method)
- Diffusion (Jacobi solver)
- Projection (Jacobi solver)
- Basic element-wise operations (map, combine)
- Boundary conditions (periodic, clamp)

✅ **SC-5: Example Programs**
- At least 2 runnable example programs
- Clear, commented code
- Demonstrate key features
- Run at interactive frame rates (>10 FPS)

### Important (Should Pass)

⚠️ **SC-6: Performance**
- 256×256 grid at >30 FPS (CPU)
- Reasonable memory usage (<1GB)
- No memory leaks

⚠️ **SC-7: Developer Experience**
- Informative error messages with line numbers
- Clear documentation for getting started
- Easy installation process

⚠️ **SC-8: Visual Quality**
- Field colorization with multiple palettes
- Smooth rendering
- Window resize support

### Nice to Have (May Defer)

💡 **SC-9: Advanced Solvers**
- Conjugate Gradient for diffusion/projection
- MacCormack advection
- Multiple preconditioners

💡 **SC-10: Multi-Domain**
- Basic agent operations
- Signal processing basics
- Field-agent coupling

---

## Core Features (Must Have)

### Phase 1: Foundation ✅ (COMPLETE)

**Status:** Implemented in initial commit

- [x] Lexer with full tokenization
- [x] Parser generating complete AST
- [x] Type system with units
- [x] Type checker with error reporting
- [x] AST visitors (type checker, printer)
- [x] CLI interface structure
- [x] Project structure and packaging

### Phase 2: Runtime Core (CRITICAL PATH)

**Priority:** P0 - Blocks all execution

#### 2.1 Execution Engine
```python
# creative_computation/runtime/engine.py
class ExecutionEngine:
    - Execute parsed AST
    - Manage timesteps
    - Handle double buffering
    - Coordinate operations
```

**Required capabilities:**
- Execute step blocks
- Evaluate expressions
- Manage variable scope
- Handle state persistence

#### 2.2 Field Data Structure
```python
# creative_computation/runtime/field.py
class Field2D:
    - Store dense 2D arrays (NumPy-based)
    - Support different dtypes (f32, f64, Vec2)
    - Efficient indexing
    - Boundary handling
```

#### 2.3 Memory Management
- Double-buffer swapping
- Temporary buffer allocation
- Memory reuse where possible

### Phase 3: Core Field Operations (CRITICAL PATH)

**Priority:** P0 - Required for any useful program

#### 3.1 Basic Operations (MUST HAVE)
```python
# creative_computation/stdlib/field_ops.py

def alloc(dtype, size, init=0.0):
    """Allocate new field"""

def map(field, fn):
    """Apply function element-wise"""

def combine(field_a, field_b, fn):
    """Combine two fields element-wise"""

def sample(field, pos, interp='linear', out_of_bounds='clamp'):
    """Sample at continuous position"""

def boundary(field, spec):
    """Apply boundary conditions"""
```

#### 3.2 PDE Operations (MUST HAVE)
```python
def advect_semilagrangian(field, velocity, dt):
    """Semi-Lagrangian advection (stable, simple)"""

def diffuse_jacobi(field, rate, dt, iterations=20):
    """Jacobi diffusion solver"""

def project_jacobi(velocity, iterations=40):
    """Jacobi projection (Poisson solve for pressure)"""

def laplacian(field):
    """5-point stencil Laplacian"""

def gradient(field):
    """Central difference gradient"""

def divergence(vector_field):
    """Divergence of vector field"""
```

**Why these methods?**
- **Semi-Lagrangian:** Unconditionally stable, simple to implement
- **Jacobi:** Easy to implement, good enough for MVP, fully parallel
- We can add better methods (MacCormack, CG, Multigrid) later

### Phase 4: Visualization (CRITICAL PATH)

**Priority:** P0 - Required to see results

#### 4.1 Field Rendering
```python
# creative_computation/stdlib/visual_ops.py

def colorize(field, palette='viridis'):
    """Convert scalar field to RGB image"""
    # Use matplotlib colormaps

def render_to_window(visual):
    """Display in window using pygame/pyglet"""

def save_frame(visual, path):
    """Save frame as PNG"""
```

**Supported palettes (MVP):**
- viridis (default)
- plasma
- grayscale
- fire (black-red-yellow-white)

#### 4.2 Display Backend
Use **Pygame** for MVP (easy to install, cross-platform):
```python
# creative_computation/runtime/display.py

class DisplayWindow:
    - Create window
    - Display frames
    - Handle close event
    - Basic keyboard input (pause, step, quit)
```

### Phase 5: Deterministic RNG

**Priority:** P1 - Important for determinism

```python
# creative_computation/runtime/rng.py

class PhiloxRNG:
    """Philox 4×32-10 counter-based RNG"""

    def __init__(self, seed):
        self.seed = seed

    def random_field(shape, seed):
        """Generate random field deterministically"""

    def random_float(seed):
        """Generate single random float"""
```

**Implementation notes:**
- Use NumPy's Generator with PCG64 for MVP (similar guarantees)
- Can swap to true Philox later if needed
- Critical: same seed → same sequence

---

## Deferred Features (Later Versions)

### Post-MVP: Performance Optimization

**Defer to v0.3.0:**
- MLIR lowering (use Python interpreter for MVP)
- GPU acceleration (CPU-only for MVP)
- Operation fusion
- Lazy evaluation
- JIT compilation

**Rationale:** Get correctness first, optimize later

### Post-MVP: Advanced Field Operations

**Defer to v0.3.x:**
- MacCormack advection
- BFECC advection
- Conjugate Gradient solver
- Multigrid solver
- ILU preconditioners
- 3D fields (Field3D)
- Adaptive mesh refinement

**Rationale:** Jacobi solver is sufficient to demonstrate language

### Post-MVP: Agent-Based Computing

**Defer to v0.4.0:**
- Agent allocation and lifecycle
- Force calculations
- Field sampling by agents
- Agent-field coupling
- Barnes-Hut acceleration

**Rationale:** Focus on one domain (fields) for MVP

### Post-MVP: Signal Processing

**Defer to v0.5.0:**
- Oscillators
- Filters
- Envelopes
- Audio I/O
- FFT operations

**Rationale:** Not critical for demonstrating core language features

### Post-MVP: Advanced Features

**Defer to v0.6.0+:**
- `iterate` loops
- `link` metadata
- `compose` parallel execution
- Module system
- Adaptive timestep
- Hot reload

**Rationale:** These are polish features, not MVP-critical

---

## Implementation Phases

### Phase 1: Foundation ✅ (COMPLETE)
**Estimated time:** 2 days
**Status:** Done (initial commit)

- [x] Project structure
- [x] Lexer and parser
- [x] Type system
- [x] Basic CLI

### Phase 2: Runtime Engine (CURRENT)
**Estimated time:** 3-4 days
**Deliverable:** Execute simple programs

**Tasks:**
1. Create ExecutionEngine class
2. Implement expression evaluation
3. Add variable/state management
4. Handle step execution
5. Test with simple programs (no fields yet)

**Milestone:** Can execute `x = 5 + 3` and print result

### Phase 3: Field Data Structure
**Estimated time:** 2 days
**Deliverable:** Field creation and basic operations

**Tasks:**
1. Implement Field2D class (NumPy wrapper)
2. Add allocation and initialization
3. Implement element-wise operations (map, combine)
4. Add boundary condition handling
5. Write unit tests

**Milestone:** Can create fields and do element-wise math

### Phase 4: Core PDE Operations
**Estimated time:** 4-5 days
**Deliverable:** Working fluid simulation

**Tasks:**
1. Implement advection (semi-Lagrangian)
2. Implement diffusion (Jacobi)
3. Implement projection (Jacobi)
4. Implement gradient, divergence, Laplacian
5. Add field.sample() with interpolation
6. Write integration tests

**Milestone:** Smoke simulation works (no visualization yet)

### Phase 5: Visualization
**Estimated time:** 2-3 days
**Deliverable:** See simulation results

**Tasks:**
1. Set up Pygame window
2. Implement field colorization
3. Add palette support
4. Implement visual.output()
5. Add basic controls (pause, step, quit)

**Milestone:** Can see smoke simulation running in real-time

### Phase 6: Polish and Examples
**Estimated time:** 2-3 days
**Deliverable:** MVP ready for users

**Tasks:**
1. Write 2-3 example programs
2. Improve error messages
3. Add parameter tuning via CLI
4. Write getting started guide
5. Test on multiple platforms
6. Fix critical bugs

**Milestone:** User can download, install, and run examples

### Phase 7: Documentation
**Estimated time:** 1-2 days
**Deliverable:** Users can learn the system

**Tasks:**
1. Update README with installation
2. Write tutorial for MVP features
3. Document supported operations
4. Add troubleshooting guide
5. Create demo video (optional)

**Milestone:** New user can go from zero to running simulation in 30 minutes

---

## Target Examples

### Example 1: Simple Diffusion (Week 1 Target)
```dsl
# diffusion.ccdsl - Simplest possible example

set dt = 0.1

@double_buffer field : Field2D<f32>
field = step.state(field.random(shape=[128, 128], seed=42))

step {
  field = field.diffuse(field, rate=0.1, dt, method="jacobi", iter=20)
  field = field.boundary(field, spec="periodic")

  visual.output(visual.colorize(field, palette="viridis"))
}
```

**Requirements:**
- field.random()
- field.diffuse() (Jacobi)
- field.boundary()
- visual.colorize()
- visual.output()

**Expected behavior:**
- Random noise slowly smooths out
- Colorful visualization
- 30+ FPS at 128×128

### Example 2: Smoke Simulation (Week 2 Target)
```dsl
# smoke.ccdsl - Classic fluid sim

set profile = medium
set dt = 0.016

@double_buffer velocity : Field2D<Vec2[m/s]>
@double_buffer density : Field2D<f32>

velocity = step.state(field.alloc(Vec2[m/s], size=[256, 256]))
density = step.state(field.random(shape=[256, 256], seed=42))

step {
  # Advect
  velocity = field.advect(velocity, velocity, dt, method="semilagrangian")
  density = field.advect(density, velocity, dt, method="semilagrangian")

  # Diffuse
  velocity = field.diffuse(velocity, rate=0.0001, dt, method="jacobi", iter=20)
  density = field.diffuse(density, rate=0.00001, dt, method="jacobi", iter=20)

  # Project
  velocity = field.project(velocity, method="jacobi", iter=40)

  # Boundaries
  velocity = field.boundary(velocity, spec="periodic")
  density = field.boundary(density, spec="clamp")

  # Visualize
  visual.output(visual.colorize(density, palette="viridis"))
}
```

**Requirements:**
- Vec2 fields
- field.advect() (semi-Lagrangian)
- field.diffuse() (Jacobi)
- field.project() (Jacobi)
- Boundary conditions
- Visualization

**Expected behavior:**
- Swirling fluid motion
- Density advects with velocity
- Incompressible (divergence-free)
- 20-30 FPS at 256×256

### Example 3: Reaction-Diffusion (Week 2 Target)
```dsl
# gray_scott.ccdsl - Pattern formation

set dt = 1.0

@double_buffer u : Field2D<f32>
@double_buffer v : Field2D<f32>

@param f : f32 = 0.055
@param k : f32 = 0.062
@param du : f32 = 0.16
@param dv : f32 = 0.08

u = step.state(field.alloc(f32, size=[256, 256], init=1.0))
v = step.state(field.random(shape=[256, 256], seed=42))

step {
  lap_u = field.laplacian(u)
  lap_v = field.laplacian(v)

  # Gray-Scott reaction terms
  uvv = field.combine(u, v, fn=multiply_v_squared)

  u = field.combine(u, lap_u, fn=add_diffusion(du))
  u = field.combine(u, uvv, fn=subtract)
  u = field.combine(u, u, fn=add_feed(f))

  v = field.combine(v, lap_v, fn=add_diffusion(dv))
  v = field.combine(v, uvv, fn=add)
  v = field.combine(v, v, fn=subtract_kill(f, k))

  u = field.boundary(u, spec="periodic")
  v = field.boundary(v, spec="periodic")

  visual.output(visual.colorize(v, palette="magma"))
}
```

**Requirements:**
- field.laplacian()
- field.combine() with custom functions
- Parameter system
- Visualization

**Expected behavior:**
- Organic pattern formation
- Stable patterns emerge
- Different parameters → different patterns

---

## Testing Requirements

### Unit Tests (Required)

**Lexer tests:**
- Token generation
- Keyword recognition
- Operator parsing
- String/number literals
- Error cases

**Parser tests:**
- Assignment parsing
- Function calls
- Field access
- Type annotations
- Error recovery

**Type checker tests:**
- Type inference
- Unit compatibility
- Error detection
- Symbol table

**Field operations tests:**
- Allocation
- Element-wise operations
- Boundary conditions
- Interpolation
- Each PDE operation

**Target:** 80%+ code coverage

### Integration Tests (Required)

**End-to-end tests:**
```python
def test_simple_diffusion():
    """Test complete program execution"""
    program = parse("diffusion.ccdsl")
    result = execute(program, steps=10)
    assert result.success
    assert result.frames == 10

def test_determinism():
    """Test reproducibility"""
    result1 = execute("smoke.ccdsl", seed=42, steps=100)
    result2 = execute("smoke.ccdsl", seed=42, steps=100)
    assert np.array_equal(result1.final_state, result2.final_state)

def test_visual_output():
    """Test visualization pipeline"""
    program = parse("diffusion.ccdsl")
    result = execute(program, steps=1)
    assert result.has_visual_output
    assert result.image.shape == (128, 128, 3)
```

### Manual Tests (Required)

**Installation test:**
1. Fresh Python environment
2. `pip install -e .`
3. `ccdsl --version`
4. Should complete without errors

**Example execution test:**
1. `ccdsl run examples/fluids/diffusion.ccdsl`
2. Window opens
3. Visualization appears
4. Runs smoothly
5. Can close cleanly

**Parameter tuning test:**
1. `ccdsl run examples/fluids/smoke.ccdsl --param viscosity=0.01`
2. Behavior changes appropriately
3. Visual feedback is clear

---

## Performance Targets

### Minimum Acceptable Performance

**128×128 grid:**
- Target: 60 FPS
- Minimum: 30 FPS

**256×256 grid:**
- Target: 30 FPS
- Minimum: 15 FPS

**512×512 grid:**
- Target: 10 FPS
- Minimum: 5 FPS

### Performance Optimization Strategy

**Phase 1 (MVP):** Pure Python + NumPy
- Simple, correct implementation
- No premature optimization
- Profile to find bottlenecks

**Phase 2 (Post-MVP):** Numba JIT
- Add `@numba.jit` to hot loops
- Easy wins with minimal code change

**Phase 3 (Future):** MLIR
- Full compilation pipeline
- GPU support
- Production performance

### Performance Testing

```python
@benchmark
def benchmark_diffusion():
    """Benchmark diffusion operation"""
    field = Field2D(np.random.rand(256, 256))
    for _ in range(100):
        field = diffuse_jacobi(field, rate=0.1, dt=0.01, iter=20)

# Target: <100ms for 100 iterations on reference hardware
```

**Reference hardware:**
- Intel i5-8250U (typical laptop CPU)
- 8GB RAM
- Integrated graphics

---

## Documentation Requirements

### Required Documentation

**For Users:**
1. **README.md** ✅ (Done)
   - Installation instructions
   - Quick start
   - Link to full spec

2. **GETTING_STARTED.md** (TODO)
   - Step-by-step first program
   - Explanation of concepts
   - Troubleshooting

3. **MVP_FEATURES.md** (TODO)
   - What's implemented
   - What's not implemented
   - Known limitations

4. **EXAMPLES.md** ✅ (Done in examples/README.md)
   - Description of each example
   - How to run
   - What to expect

**For Contributors:**
1. **CONTRIBUTING.md** (TODO)
   - How to set up dev environment
   - Coding standards
   - How to run tests
   - PR process

2. **ARCHITECTURE.md** ✅ (Done)
   - System design
   - Component interaction
   - Extension points

3. **ROADMAP.md** (TODO)
   - MVP definition (this document)
   - Post-MVP plans
   - Version timeline

### Documentation Standards

**All code examples must:**
- Be runnable (no pseudocode)
- Include expected output
- Have clear comments
- Show best practices

**All operations must document:**
- Purpose
- Parameters (with types)
- Return type
- Example usage
- Performance characteristics

---

## Development Timeline

### Week 1: Runtime + Basic Fields
**Days 1-2:** Runtime engine
- [ ] Expression evaluation
- [ ] Variable management
- [ ] Step execution

**Days 3-4:** Field data structure
- [ ] Field2D class
- [ ] Basic operations (map, combine)
- [ ] Boundary conditions

**Day 5:** Testing + integration
- [ ] Unit tests
- [ ] First example working (diffusion)

**Milestone:** Simple diffusion simulation runs

### Week 2: PDE Operations + Visualization
**Days 1-2:** PDE operations
- [ ] Advection (semi-Lagrangian)
- [ ] Diffusion (Jacobi)
- [ ] Gradient, Laplacian, divergence

**Days 3-4:** Projection + visualization
- [ ] Projection (Jacobi)
- [ ] Field colorization
- [ ] Pygame display window

**Day 5:** Testing + smoke example
- [ ] Integration tests
- [ ] Smoke simulation working

**Milestone:** Fluid simulation runs with visualization

### Week 3: Polish + Release
**Days 1-2:** Additional examples
- [ ] Reaction-diffusion
- [ ] Wave equation
- [ ] Documentation for examples

**Days 3-4:** Bug fixes + polish
- [ ] Better error messages
- [ ] Parameter system
- [ ] CLI improvements
- [ ] Cross-platform testing

**Day 5:** Release prep
- [ ] GETTING_STARTED.md
- [ ] MVP_FEATURES.md
- [ ] Demo video
- [ ] Announcement draft

**Milestone:** MVP v0.2.2 released

---

## Success Metrics

### Technical Metrics

✅ **Correctness**
- All unit tests pass
- All integration tests pass
- Examples run without errors

✅ **Performance**
- Meets minimum FPS targets
- Memory usage < 1GB
- No memory leaks

✅ **Determinism**
- Bit-exact reproducibility (same seed)
- Cross-platform consistency (within tolerance)

### User Experience Metrics

✅ **Time to First Success**
- New user can run first example in < 15 minutes
- Clear error messages when things go wrong

✅ **Learning Curve**
- User can write custom program in < 1 hour
- Documentation is clear and complete

✅ **Satisfaction**
- Users can create interesting results
- System feels responsive and polished
- Examples are impressive

---

## Risk Assessment

### High Risk

**⚠️ Risk 1: Performance too slow**
- **Impact:** High (unusable if <10 FPS)
- **Likelihood:** Medium
- **Mitigation:** Profile early, optimize hot paths, use Numba if needed
- **Fallback:** Reduce default grid size, add performance warnings

**⚠️ Risk 2: Solver instability**
- **Impact:** High (incorrect physics)
- **Likelihood:** Medium
- **Mitigation:** Use well-tested algorithms, validate against known solutions
- **Fallback:** Reduce timestep, increase iterations, add stability warnings

### Medium Risk

**⚠️ Risk 3: Cross-platform issues**
- **Impact:** Medium (blocks some users)
- **Likelihood:** Medium
- **Mitigation:** Test on Windows, Mac, Linux, Use portable libraries
- **Fallback:** Document known issues, provide workarounds

**⚠️ Risk 4: Installation difficulties**
- **Impact:** Medium (users can't try it)
- **Likelihood:** Low (simple dependencies)
- **Mitigation:** Clear installation docs, test on fresh systems
- **Fallback:** Provide pre-built binaries or Docker image

### Low Risk

**⚠️ Risk 5: Scope creep**
- **Impact:** Low (delays release)
- **Likelihood:** Medium
- **Mitigation:** Strict MVP definition (this document), regular scope reviews
- **Fallback:** Cut non-essential features, push to next version

---

## Definition of Done

### MVP is complete when:

✅ **Functional Requirements:**
- [ ] All Phase 1-6 tasks completed
- [ ] All target examples run successfully
- [ ] All unit tests pass (>80% coverage)
- [ ] All integration tests pass

✅ **Performance Requirements:**
- [ ] Meets minimum FPS targets
- [ ] No critical performance issues
- [ ] Profiling shows reasonable resource usage

✅ **Quality Requirements:**
- [ ] No critical bugs
- [ ] No memory leaks
- [ ] Determinism verified
- [ ] Cross-platform tested (Win/Mac/Linux)

✅ **Documentation Requirements:**
- [ ] README updated with installation
- [ ] GETTING_STARTED.md written
- [ ] MVP_FEATURES.md written
- [ ] All examples documented

✅ **User Validation:**
- [ ] 3+ external users can install and run
- [ ] Users can create custom programs
- [ ] Feedback incorporated

### Release Checklist

- [ ] Version bumped to v0.2.2-mvp
- [ ] CHANGELOG.md updated
- [ ] Git tagged
- [ ] PyPI package uploaded
- [ ] GitHub release created
- [ ] Announcement posted
- [ ] Demo video uploaded

---

## Post-MVP Roadmap

### v0.3.0 - Performance (Q2 2025)
- Numba JIT compilation
- Operation fusion
- Better solvers (CG, MacCormack)
- GPU support (basic)

### v0.4.0 - Multi-Domain (Q3 2025)
- Agent-based computing
- Field-agent coupling
- Agent examples (boids, particles)

### v0.5.0 - Audio (Q4 2025)
- Signal processing
- Audio synthesis
- Audio-reactive examples

### v0.6.0 - MLIR (Q1 2026)
- Full MLIR lowering
- Production optimization
- Multi-device support

### v1.0.0 - Production (Q2 2026)
- All features from specification
- Production-grade performance
- Comprehensive testing
- Industrial-quality documentation

---

## Conclusion

This MVP focuses on **depth over breadth** — doing field operations exceptionally well rather than doing all domains poorly. By delivering a polished experience for fluid dynamics and PDE-based simulations, we:

1. **Validate the language design** with real, working programs
2. **Demonstrate the value proposition** with impressive examples
3. **Build a foundation** that other domains can build upon
4. **Ship something useful** in a reasonable timeframe

**The MVP is successful if:** Users can create beautiful, deterministic, fluid simulations in CCDSL and feel excited about what's possible.

---

**Document Version:** 1.0
**Last Updated:** 2025-01-05
**Status:** Draft for Review
