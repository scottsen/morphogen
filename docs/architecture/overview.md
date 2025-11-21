# Creative Computation DSL Architecture

## Overview

The Creative Computation DSL is designed as a multi-stage compilation pipeline that transforms high-level DSL code into efficient executable code through MLIR lowering.

## Compilation Pipeline

```
Source Code (.ccdsl)
    ↓
[Lexer] → Tokens
    ↓
[Parser] → AST
    ↓
[Type Checker] → Typed AST
    ↓
[MLIR Lowering] → MLIR IR
    ↓
[MLIR Optimization Passes]
    ↓
[Code Generation] → Executable
```

## Core Components

### 1. Frontend (Lexer + Parser)

**Lexer** (`creative_computation/lexer/`)
- Tokenizes source code into a stream of tokens
- Handles comments, strings, numbers, identifiers, operators
- Tracks source locations for error reporting

**Parser** (`creative_computation/parser/`)
- Recursive descent parser
- Builds Abstract Syntax Tree (AST) from tokens
- Supports:
  - Step/substep blocks
  - Module definitions
  - Field, agent, signal, and visual operations
  - Decorators (@double_buffer, @benchmark, etc.)
  - Type annotations with units

### 2. Type System

**Type Definitions** (`creative_computation/ast/types.py`)
- Scalar types: f32, f64, f16, i32, i64, u32, u64, bool
- Vector types: Vec2, Vec3 (with units)
- Field types: Field2D, Field3D (with element types and units)
- Agent types: Agents<RecordType>
- Signal types: Signal<T>
- Visual type (opaque)

**Unit System** (`creative_computation/types/units.py`)
- Physical unit tracking (m, m/s, kg, etc.)
- Unit compatibility checking
- Safe promotions, error on lossy casts
- Override with @allow_unit_cast decorator

**Type Checker** (`creative_computation/ast/visitors.py`)
- Traverses AST to infer and check types
- Maintains symbol table
- Reports type mismatches and unit errors
- Validates function signatures

### 3. Abstract Syntax Tree (AST)

**Node Types** (`creative_computation/ast/nodes.py`)

Expressions:
- `Literal` — Numeric, string, boolean literals
- `Identifier` — Variable references
- `BinaryOp` — Binary operations (+, -, *, /, etc.)
- `UnaryOp` — Unary operations (-, !)
- `Call` — Function/method calls
- `FieldAccess` — Field/method access (object.field)

Statements:
- `Assignment` — Variable assignment with optional type annotation
- `Step` — Single timestep block
- `Substep` — Subdivided timestep block
- `Module` — Reusable subsystem definition
- `Compose` — Parallel module composition

**Visitor Pattern**
- `ASTVisitor` — Base visitor for tree traversal
- `TypeChecker` — Type checking visitor
- `ASTPrinter` — Debug pretty-printing

### 4. MLIR Lowering

**Target Dialects**
- `linalg` — Linear algebra operations (field operations)
- `affine` — Affine loop transformations (stencils)
- `scf` — Structured control flow (loops, conditionals)
- `arith` — Arithmetic operations
- `math` — Mathematical functions
- `vector` — SIMD vector operations
- `gpu` — GPU kernel launches
- `async` — Asynchronous operations
- `memref` — Memory reference operations

**Lowering Strategy**

Field Operations:
```
field.stencil(x, fn, radius)
  ↓
linalg.generic + affine loops
  ↓
Fused neighborhood access pattern
```

Agent Operations:
```
agent.force_sum(A, rule, method="barnes_hut")
  ↓
scf.for loops + Barnes-Hut tree construction
  ↓
Morton-ordered spatial acceleration
```

Signal Operations:
```
signal.osc(freq, shape)
  ↓
arith + math ops
  ↓
Vectorized waveform generation
```

**GPU Lowering Principles**

Morphogen's MLIR lowering follows structured patterns that ensure efficient GPU execution while maintaining determinism. The compiler pipeline implements:

1. **Structured Parallelism** — All operations expose explicit iteration spaces for GPU block/thread mapping
2. **Memory Hierarchy Management** — Explicit modeling of global/shared/register memory tiers
3. **Static Shape Preference** — Compile-time constants enable optimal tiling and vectorization
4. **Warp-Friendly Execution** — Uniform control flow to minimize divergence penalties
5. **Deterministic GPU Semantics** — Profile-driven guarantees for strict/repro/live execution modes

For detailed GPU design rules, operator metadata requirements, and lowering patterns, see [GPU & MLIR Principles](./gpu-mlir-principles.md).

### 5. Runtime System

**Execution Context** (`creative_computation/runtime/`)
- Manages state across timesteps
- Handles double-buffered resources
- Implements deterministic RNG (Philox 4×32-10)
- Coordinates solver profiles

**Resource Management**
- Memory allocation for fields and agents
- Buffer swapping for @double_buffer
- Lazy evaluation and operation fusion

**Determinism**
- Stable agent ordering by (id, creation_index)
- Reproducible RNG seeding: hash64(global_seed, id, tick, seed)
- Morton ordering for spatial algorithms

### 6. Standard Library

**Field Operations** (`creative_computation/stdlib/field.py`)
- Advection: Semi-Lagrangian, MacCormack, BFECC
- Diffusion: Jacobi, Gauss-Seidel, Conjugate Gradient
- Projection: Jacobi, Multigrid, Preconditioned CG
- Stencils: Laplacian, Gradient, Divergence
- Boundary conditions: reflect, periodic, noSlip, clamp

**Agent Operations** (`creative_computation/stdlib/agent.py`)
- Force calculation: brute force, grid, Barnes-Hut
- Integration: Euler, Verlet, RK4
- Spawn/remove with stable ID management
- Mutation and reproduction with deterministic RNG

**Signal Operations** (`creative_computation/stdlib/signal.py`)
- Oscillators: sine, triangle, sawtooth, square
- Noise generators: white, pink, Perlin
- Filters: lowpass, highpass, bandpass, notch
- Envelopes: ADSR, custom curves

**Visual Operations** (`creative_computation/stdlib/visual.py`)
- Colorization with palettes
- Point sprite rendering
- Layer composition with blend modes
- Post-processing filters

## Solver Configuration

### Profiles

Global performance/precision profiles:
- `low`: f16 precision, Jacobi solvers, iter=10
- `medium`: f32 precision, CG solvers, iter=20
- `high`: f64 precision, Multigrid, iter=40

### Precedence
Per-op > Solver alias > Module profile > Global profile

### Preconditioners
- Jacobi: Diagonal scaling
- ILU(0): Incomplete LU factorization
- Multigrid smoother: V-cycle with Jacobi smoother

## Determinism Model

### Tiers

1. **Strict** — Bit-identical across devices and runs
   - field.diffuse, agent.force_sum (with deterministic methods)
   - All RNG-based operations with fixed seed

2. **Reproducible** — Deterministic within precision
   - field.project (iterative solvers may vary slightly)
   - visual.filter (floating-point rounding)

3. **Nondeterministic** — External I/O or adaptive termination
   - io.stream(live) — Real-time input
   - iterate(unbounded) — May vary iteration count

Use `@nondeterministic` annotation to document intentionally nondeterministic code.

## Optimization Strategies

### Operation Fusion
Consecutive field operations are fused when possible:
```
field.combine(a, b) → field.mask(...) → field.diffuse(...)
```
Generates a single fused kernel to minimize memory traffic.

### Lazy Evaluation
Expressions are not evaluated immediately; instead, they build a computation graph that is optimized before execution.

### SIMD Vectorization
Scalar operations are vectorized using MLIR's vector dialect:
- Field element-wise operations
- Signal block processing
- Agent force calculations

### GPU Offloading
Large-scale operations are offloaded to GPU when available:
- Field advection/diffusion
- Agent force summation
- Visual rendering

## Error Handling

### Compile-Time Errors
- Type mismatches
- Unit incompatibilities
- Undefined symbols
- Invalid solver configurations

### Runtime Errors
- NaN/Inf detection (with @check_finite decorator)
- Out-of-bounds access (controlled by out_of_bounds parameter)
- Resource allocation failures

### Lints and Warnings
- Unused state variables
- Potentially inefficient patterns
- Unit cast warnings
- Nondeterministic operations

## Testing Strategy

### Unit Tests
- Lexer tokenization
- Parser correctness
- Type system rules
- MLIR lowering correctness

### Integration Tests
- End-to-end compilation
- Determinism verification
- Solver accuracy

### Conformance Tests
See Language Reference Section 13:
- Unit mismatch → compile error
- Spawn/remove → identical runs
- Barnes-Hut → bit-exact forces
- Fused/unfused operations → identical results

## Future Extensions

### Planned Features
- Multi-GPU support
- Distributed execution
- Interactive debugger
- Visual graph editor
- Live parameter tuning
- Automatic differentiation
- Backward compatibility mode

### Research Directions
- Machine learning integration
- Quantum circuit simulation
- Probabilistic programming
- Symbolic manipulation
