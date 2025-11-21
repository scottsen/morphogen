# Creative Computation DSL v0.2.2 — Complete Specification

**A typed, semantics-first domain-specific language for expressive, deterministic simulations and generative computation.**

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Design Principles](#2-design-principles)
3. [Getting Started](#3-getting-started)
4. [Type System](#4-type-system)
5. [Program Structure](#5-program-structure)
6. [Field Operations](#6-field-operations)
7. [Agent-Based Computation](#7-agent-based-computation)
8. [Signal Processing](#8-signal-processing)
9. [Visual Domain](#9-visual-domain)
10. [I/O and Streams](#10-io-and-streams)
11. [Determinism and Reproducibility](#11-determinism-and-reproducibility)
12. [Solver Configuration](#12-solver-configuration)
13. [Performance and Optimization](#13-performance-and-optimization)
14. [MLIR Lowering](#14-mlir-lowering)
15. [Advanced Features](#15-advanced-features)
16. [Best Practices](#16-best-practices)
17. [Complete Examples](#17-complete-examples)
18. [Language Reference](#18-language-reference)

---

## 1. Introduction

### 1.1 What is Creative Computation DSL?

Creative Computation DSL (CCDSL) is a domain-specific language designed to bridge the gap between artistic expression and scientific rigor in computational simulations. It combines:

- **Expressive syntax** for natural description of physical phenomena
- **Strong type system** with physical unit tracking
- **Deterministic semantics** for reproducible results
- **High performance** through MLIR compilation
- **Multi-domain integration** of fields, agents, signals, and visuals

### 1.2 Why a New DSL?

Existing tools for simulation and generative art often force users to choose between:
- **Ease of use** vs. **performance**
- **Flexibility** vs. **correctness**
- **Determinism** vs. **expressiveness**

CCDSL provides all of these simultaneously by:
- Compiling to MLIR for optimal performance across CPUs and GPUs
- Using strong types and units to catch errors at compile time
- Guaranteeing bit-exact reproducibility with explicit RNG seeding
- Offering a composable vocabulary that works across domains

### 1.3 Use Cases

**Scientific Computing**
- Computational fluid dynamics
- Reaction-diffusion systems
- N-body simulations
- Climate modeling

**Creative Coding**
- Generative art and animation
- Procedural content generation
- Interactive installations
- Audio-visual performances

**Research and Education**
- Reproducible research in computational science
- Teaching computational physics
- Exploring emergent behaviors
- Prototyping new algorithms

---

## 2. Design Principles

### 2.1 Pure Per-Step Graphs, Explicit Cross-Step State

Each `step` defines a pure computation graph:
```dsl
step {
  # All operations here form a DAG
  # No hidden state mutations
  velocity = field.advect(velocity, velocity, dt)
  density = field.diffuse(density, rate, dt)
}
```

State that persists across steps must be explicit:
```dsl
# Declare persistent state
@double_buffer velocity : Field2D<Vec2[m/s]>

# Initialize once
particles = step.state(agent.alloc(Particle, count=1000))
```

**Why?** This enables:
- Automatic parallelization
- Operation fusion
- Clear data flow
- Easier debugging

### 2.2 Deterministic Semantics

Every operation is deterministic by default:

**RNG:** Philox 4×32-10 with explicit seeding
```dsl
field.random(shape, seed=42)  # Same result every time
```

**Agent ordering:** Stable (id, creation_index) sorting
```dsl
agents = agent.spawn(...)  # Order is deterministic
```

**Floating point:** Reproducible accumulation order
```dsl
agent.reduce(agents, fn=sum, init=0.0)  # Deterministic sum
```

**Why?** Enables:
- Reproducible research
- Debugging and testing
- Collaborative development
- Version control of results

### 2.3 Composability and Clarity

Small vocabulary, maximum reuse:

```dsl
# Same combine operator works on fields and signals
field.combine(a, b, fn=add)
signal.combine(s1, s2, fn=multiply)

# Same integrate pattern across domains
field.integrate(x, rate, dt)
signal.integrate(sig, dt)
agent.integrate(agents, forces, dt)
```

**Why?** Reduces cognitive load and enables transfer of knowledge between domains.

### 2.4 MLIR-Oriented Lowering

Every operation maps cleanly to MLIR dialects:

| CCDSL Op | MLIR Dialect | Purpose |
|----------|--------------|---------|
| field.stencil | linalg + affine | Fused neighborhood access |
| agent.force_sum | scf + gpu | Parallel force calculation |
| signal.filter | arith + math | Digital signal processing |
| iterate | scf.while | Dynamic loops |

**Why?** Enables:
- State-of-art optimization
- Multi-target compilation (CPU, GPU, TPU)
- Integration with existing tools
- Future extensibility

### 2.5 Live Creativity

Hot-reload and tunable parameters for interactive development:

```dsl
@param viscosity : f32 = 0.001 @range(0, 0.1) @doc "Fluid viscosity"

set profile = medium  # Switch between performance profiles
```

**Why?** Supports creative exploration and rapid prototyping.

---

## 3. Getting Started

### 3.1 Hello, World!

```dsl
# hello.ccdsl - Your first CCDSL program

step {
  # Create a signal
  sine_wave = signal.osc(freq=440.0, shape="sin")

  # Output to console (via visual text)
  visual.output(
    visual.text("Hello, Creative Computation!", fmt="default")
  )

  # Output audio
  io.output(sine_wave, target="audio")
}
```

Run it:
```bash
ccdsl run hello.ccdsl
```

### 3.2 Simple Fluid Simulation

```dsl
# smoke.ccdsl - Basic smoke simulation

set profile = medium
set dt = 0.016  # 60 fps

# State variables
@double_buffer density : Field2D<f32>
@double_buffer velocity : Field2D<Vec2[m/s]>

step {
  # Advect density through velocity field
  density = field.advect(density, velocity, dt)

  # Diffuse density
  density = field.diffuse(density, rate=0.0001, dt)

  # Visualize
  visual.output(
    visual.colorize(density, palette="viridis")
  )
}
```

### 3.3 Particle System

```dsl
# particles.ccdsl - Simple particle system

type Particle = {
  id: u64,
  pos: Vec2[m],
  vel: Vec2[m/s],
  life: f32
}

set dt = 0.016

particles = step.state(agent.alloc(Particle, count=100))

step {
  # Apply gravity
  gravity = agent.map(particles, fn=apply_gravity)

  # Update positions
  particles = agent.integrate(particles, gravity, dt)

  # Remove dead particles
  particles = agent.remove(particles, pred=is_dead)

  # Spawn new particles
  particles = agent.spawn(particles, template=new_particle)

  # Render
  visual.output(
    visual.points(particles, color="white")
  )
}
```

### 3.4 Development Workflow

1. **Write** — Create .ccdsl file
2. **Check** — Validate types: `ccdsl check program.ccdsl`
3. **Parse** — Inspect AST: `ccdsl parse program.ccdsl`
4. **Run** — Execute: `ccdsl run program.ccdsl`
5. **Profile** — Benchmark: `ccdsl run program.ccdsl --profile high`
6. **Debug** — Add `@benchmark` decorators for timing

---

## 4. Type System

### 4.1 Scalar Types

| Type | Description | Range | Precision |
|------|-------------|-------|-----------|
| `f32` | 32-bit float | ±3.4e38 | ~7 digits |
| `f64` | 64-bit float | ±1.7e308 | ~15 digits |
| `f16` | 16-bit float | ±65504 | ~3 digits |
| `i32` | 32-bit signed int | ±2.1e9 | Exact |
| `i64` | 64-bit signed int | ±9.2e18 | Exact |
| `u32` | 32-bit unsigned int | 0 to 4.3e9 | Exact |
| `u64` | 64-bit unsigned int | 0 to 1.8e19 | Exact |
| `bool` | Boolean | true/false | N/A |

**Usage:**
```dsl
x : f32 = 3.14
count : u32 = 100
flag : bool = true
```

### 4.2 Vector Types

**Vec2** — 2D coordinates
```dsl
position : Vec2[m] = {x: 1.0, y: 2.0}
velocity : Vec2[m/s] = {x: 0.5, y: -0.3}
```

**Vec3** — 3D coordinates
```dsl
point : Vec3[m] = {x: 1.0, y: 2.0, z: 3.0}
color : Vec3 = {r: 1.0, g: 0.5, b: 0.0}  # No unit
```

**Operations:**
```dsl
# Component-wise operations
v3 = v1 + v2
v4 = v1 * 2.0

# Dot product
dot_product = vec2.dot(v1, v2) : f32

# Length
length = vec2.length(v1) : f32[m]

# Normalization
direction = vec2.normalize(v1) : Vec2  # Unit-less
```

### 4.3 Field Types

**Field2D<T>** — 2D dense grid
```dsl
density : Field2D<f32>
velocity : Field2D<Vec2[m/s]>
pressure : Field2D<f32[Pa]>
```

**Field3D<T>** — 3D dense grid
```dsl
volume : Field3D<f32>
vector_field : Field3D<Vec3[m/s]>
```

**Allocation:**
```dsl
# Create field with size
field = field.alloc(f32, size=[256, 256])

# Create with initial value
field = field.alloc(f32, size=[128, 128], init=0.0)

# Create from random data
field = field.random(shape=[512, 512], seed=42)
```

### 4.4 Agent Types

Agents are collections of records with stable IDs:

```dsl
type Boid = {
  id: u64,           # Unique identifier (automatic)
  pos: Vec2[m],      # Position
  vel: Vec2[m/s],    # Velocity
  color: i32,        # Custom data
  energy: f32        # Custom data
}

# Allocate agent collection
boids : Agents<Boid> = agent.alloc(Boid, count=500)
```

**Required fields:**
- `id: u64` — Unique identifier (managed automatically)

**Custom fields:**
- Any scalar, vector, or composite type

### 4.5 Signal Types

Time-varying values:

```dsl
audio : Signal<f32> = signal.osc(freq=440.0)
control : Signal<f32[Hz]> = signal.rate(60.0)
stereo : Signal<Vec2> = signal.mix([left, right])
```

### 4.6 Visual Type

Opaque renderable (linear RGB):

```dsl
vis : Visual = visual.colorize(field, palette="fire")
composed : Visual = visual.layer([bg, fg], blend="alpha")
```

### 4.7 Physical Units

**Supported units:**
- Length: `m`, `cm`, `km`
- Time: `s`, `ms`
- Mass: `kg`, `g`
- Force: `N` (Newton)
- Pressure: `Pa` (Pascal)
- Frequency: `Hz`
- Composite: `m/s`, `m/s^2`, `kg*m/s^2`

**Unit checking:**
```dsl
position : Vec2[m] = {x: 1.0, y: 2.0}
time : f32[s] = 0.1
velocity : Vec2[m/s] = position / time  # ✓ Type checks

# Error: Unit mismatch
acceleration : Vec2[m/s^2] = velocity  # ✗ Compile error
```

**Unit conversion:**
```dsl
# Automatic safe conversions
distance_m : f32[m] = 1000.0
distance_km : f32[km] = distance_m  # ✓ Automatic conversion

# Explicit lossy casts
@allow_unit_cast
velocity_no_unit : f32 = velocity[m/s]  # ⚠ Warning
```

### 4.8 Type Inference

Types can often be inferred:

```dsl
# Explicit type
x : f32 = 3.14

# Inferred type
y = 3.14  # Inferred as f32

# Inferred from operation
z = field.advect(velocity, velocity, dt)  # Inferred as Field2D<Vec2[m/s]>
```

---

## 5. Program Structure

### 5.1 Step Blocks

A `step` represents one timestep:

```dsl
step {
  # All operations execute once per timestep
  velocity = field.advect(velocity, velocity, dt)
  velocity = field.diffuse(velocity, rate, dt)
  velocity = field.project(velocity)
}
```

**Execution model:**
1. Evaluate all expressions in topological order
2. Swap double-buffered resources
3. Output visuals/audio
4. Advance to next timestep

### 5.2 Substeps

Subdivide timestep for stability:

```dsl
step {
  # Run 4 times with dt/4 each
  substep(4) {
    particles = agent.integrate(particles, forces, dt)
  }
}
```

**When to use:**
- High-velocity particles
- Stiff differential equations
- Explicit integration schemes

### 5.3 Modules

Reusable subsystems:

```dsl
module FluidSolver(
  vel: Field2D<Vec2[m/s]>,
  viscosity: f32,
  dt: f32[s]
) {
  vel = field.advect(vel, vel, dt)
  vel = field.diffuse(vel, rate=viscosity, dt)
  vel = field.project(vel, method="cg", iter=40)

  # Return value (last expression)
  vel
}

# Usage
step {
  velocity = FluidSolver(velocity, 0.001, dt)
}
```

### 5.4 Composition

Parallel execution of independent modules:

```dsl
module UpdateFluid(...) { ... }
module UpdateParticles(...) { ... }
module UpdateAudio(...) { ... }

step {
  compose(
    UpdateFluid(velocity, density),
    UpdateParticles(particles),
    UpdateAudio(audio_buffer)
  )
}
```

**Benefits:**
- Explicit parallelism
- Modular code organization
- Independent timesteps per module

### 5.5 State Management

**Persistent state** — Survives across steps:
```dsl
# Declare with step.state()
particles = step.state(agent.alloc(Particle, count=100))

step {
  # Update persistent state
  particles = agent.integrate(particles, forces, dt)
}
```

**Local variables** — Scoped to step:
```dsl
step {
  # Local computation
  laplacian = field.laplacian(density)
  gradient = field.gradient(density)
  # These don't persist
}
```

**Double buffering** — Automatic read/write buffers:
```dsl
@double_buffer density : Field2D<f32>

step {
  # Reads from current buffer, writes to next
  density = field.diffuse(density, rate, dt)
}
```

### 5.6 Configuration

Global settings:

```dsl
# Set execution profile
set profile = medium

# Set adaptive timestep
set dt = adaptive_dt(cfl=0.5, max_dt=0.02, min_dt=0.002)

# Set global seed
set seed = 42

# Set solver defaults
set solver = field.diffuse(method="cg", iter=30)
```

---

## 6. Field Operations

Fields represent dense grids of values, ideal for PDEs and continuous phenomena.

### 6.1 Creation and Allocation

```dsl
# Allocate with size
field = field.alloc(f32, size=[256, 256])

# Allocate with initial value
field = field.alloc(f32, size=[512, 512], init=1.0)

# Random initialization
field = field.random(shape=[256, 256], seed=42)

# Load from file
field = io.load_field("data/initial.png", format="grayscale")
```

### 6.2 Advection

Transport values through a velocity field:

```dsl
# Semi-Lagrangian (stable, diffusive)
density = field.advect(density, velocity, dt, method="semilagrangian")

# MacCormack (higher accuracy)
density = field.advect(density, velocity, dt, method="maccormack")

# BFECC (best accuracy, more expensive)
density = field.advect(density, velocity, dt, method="bfecc")
```

**Parameters:**
- `x: FieldND<T>` — Field to advect
- `v: FieldND<VecN>` — Velocity field
- `dt: f32[s]` — Timestep
- `method: str` — Advection scheme

**Example:**
```dsl
step {
  # Self-advection (fluid velocity)
  velocity = field.advect(velocity, velocity, dt, method="maccormack")

  # Passive scalar transport
  density = field.advect(density, velocity, dt, method="maccormack")
}
```

### 6.3 Diffusion

Smooth values over time:

```dsl
# Jacobi iteration (simple, stable)
field = field.diffuse(x, rate=0.1, dt, method="jacobi", iter=20)

# Gauss-Seidel (faster convergence)
field = field.diffuse(x, rate=0.1, dt, method="gauss_seidel", iter=10)

# Conjugate Gradient (best for large systems)
field = field.diffuse(x, rate=0.1, dt, method="cg", iter=20, tol=1e-6)
```

**Parameters:**
- `x: FieldND<T>` — Field to diffuse
- `rate: f32` — Diffusion rate (higher = more smoothing)
- `dt: f32[s]` — Timestep
- `method: str` — Solver method
- `iter: u32` — Maximum iterations
- `tol: f32` — Convergence tolerance (for CG)

**Physical interpretation:**
```dsl
# Heat equation: ∂T/∂t = α∇²T
temperature = field.diffuse(temperature, rate=α, dt)

# Viscous diffusion: ∂v/∂t = ν∇²v
velocity = field.diffuse(velocity, rate=ν, dt)
```

### 6.4 Projection

Make a vector field divergence-free (incompressible):

```dsl
# Jacobi (simple)
velocity = field.project(velocity, method="jacobi", iter=40)

# Conjugate Gradient (accurate)
velocity = field.project(velocity, method="cg", tol=1e-4, iter=50)

# Multigrid (fastest for large grids)
velocity = field.project(velocity, method="multigrid", iter=5)
```

**Use case — Incompressible fluids:**
```dsl
step {
  # 1. Advect
  velocity = field.advect(velocity, velocity, dt)

  # 2. Add forces
  velocity = field.combine(velocity, external_forces, fn=add)

  # 3. Diffuse (viscosity)
  velocity = field.diffuse(velocity, rate=viscosity, dt)

  # 4. Project (enforce incompressibility)
  velocity = field.project(velocity, method="cg", iter=40)
}
```

### 6.5 Reaction

Couple multiple fields through reaction terms:

```dsl
# Gray-Scott reaction-diffusion
temp = field.react(u, v, Params{
  f: 0.055,  # Feed rate
  k: 0.062   # Kill rate
})
```

### 6.6 Stencil Operations

Apply custom function to neighborhoods:

```dsl
# 3×3 neighborhood (radius=1)
result = field.stencil(field, fn=my_kernel, radius=1)

# 5×5 neighborhood (radius=2)
result = field.stencil(field, fn=my_kernel, radius=2)

# Custom stencil function
fn my_kernel(center: f32, neighbors: Array<f32>) -> f32 {
  sum = 0.0
  for n in neighbors {
    sum = sum + n
  }
  return center + sum * 0.1
}
```

**Built-in stencils:**
```dsl
# Laplacian: ∇²f
laplacian = field.laplacian(field)

# Gradient: ∇f
gradient = field.gradient(field)  # Returns Field2D<Vec2>

# Divergence: ∇·v
divergence = field.divergence(velocity)  # Returns Field2D<f32>

# Curl (2D): ∂v_y/∂x - ∂v_x/∂y
curl = field.curl(velocity)  # Returns Field2D<f32>
```

### 6.7 Sampling

Read values at arbitrary positions:

```dsl
# Nearest neighbor
value = field.sample(field, pos={x: 100.5, y: 50.3}, interp="nearest")

# Bilinear interpolation
value = field.sample(field, pos={x: 100.5, y: 50.3}, interp="linear")

# Bicubic interpolation
value = field.sample(field, pos={x: 100.5, y: 50.3}, interp="cubic")

# With gradient
grad = field.sample_grad(field, pos, interp="linear")  # Returns VecN
```

**Out-of-bounds behavior:**
```dsl
# Use boundary condition
value = field.sample(field, pos, interp="linear", out_of_bounds="boundary")

# Return zero
value = field.sample(field, pos, interp="linear", out_of_bounds="zero")

# Clamp to edges
value = field.sample(field, pos, interp="linear", out_of_bounds="clamp")
```

### 6.8 Element-wise Operations

```dsl
# Map function
squared = field.map(field, fn=square)

# Combine two fields
sum = field.combine(a, b, fn=add)
product = field.combine(a, b, fn=multiply)

# Threshold
mask = field.threshold(field, threshold=0.5)  # Returns Field<bool>

# Mask application
masked = field.mask(field, mask)

# Integration over time
accumulated = field.integrate(field, rate=1.0, dt)
```

### 6.9 Boundary Conditions

```dsl
# Reflective (mirror)
field = field.boundary(field, spec="reflect")

# Periodic (wrap)
field = field.boundary(field, spec="periodic")

# No-slip (zero at boundaries)
velocity = field.boundary(velocity, spec="noSlip")

# Free-slip (tangent preserved)
velocity = field.boundary(velocity, spec="freeSlip")

# Clamp (extend edges)
field = field.boundary(field, spec="clamp")
```

### 6.10 Resize and Transform

```dsl
# Resize with interpolation
large = field.resize(field, size=[512, 512], interp="linear")
small = field.resize(field, size=[64, 64], interp="linear")

# Rotate (not yet implemented in v0.2.2)
# rotated = field.rotate(field, angle=45.0)

# Translate (not yet implemented in v0.2.2)
# translated = field.translate(field, offset={x: 10, y: 20})
```

---

## 7. Agent-Based Computation

Agents represent sparse, discrete entities with individual properties and behaviors.

### 7.1 Agent Definition

```dsl
type Particle = {
  id: u64,              # Required: unique identifier
  pos: Vec2[m],         # Position
  vel: Vec2[m/s],       # Velocity
  mass: f32[kg],        # Mass
  charge: f32,          # Electric charge
  color: i32,           # Visualization
  age: f32[s],          # Lifetime
  energy: f32           # Custom property
}
```

**Required fields:**
- `id: u64` — Unique identifier (managed automatically)

**Common patterns:**
- Position: `Vec2[m]` or `Vec3[m]`
- Velocity: `Vec2[m/s]` or `Vec3[m/s]`
- Acceleration: `Vec2[m/s^2]` or `Vec3[m/s^2]`

### 7.2 Creation and Initialization

```dsl
# Allocate empty collection
particles = agent.alloc(Particle, count=1000)

# Allocate with template
particles = agent.alloc(Particle, count=1000, template={
  pos: {x: 0.0, y: 0.0},
  vel: {x: 0.0, y: 0.0},
  mass: 1.0,
  charge: 0.0,
  color: 0xFFFFFF,
  age: 0.0,
  energy: 100.0
})

# Allocate with function
particles = agent.alloc(Particle, count=1000, init_fn=random_particle)

fn random_particle(id: u64) -> Particle {
  return {
    id: id,
    pos: random_in_circle(radius=10.0, seed=id),
    vel: {x: 0.0, y: 0.0},
    mass: 1.0,
    charge: random_float(seed=id+1000),
    color: random_color(seed=id+2000),
    age: 0.0,
    energy: 100.0
  }
}
```

### 7.3 Per-Agent Transformations

Apply function to each agent:

```dsl
# Update velocities
particles = agent.map(particles, fn=update_velocity)

fn update_velocity(p: Particle) -> Particle {
  # Apply drag
  p.vel = p.vel * 0.99

  # Age
  p.age = p.age + dt

  # Decay energy
  p.energy = p.energy - 0.1

  return p
}
```

**Built-in transformations:**
```dsl
# Clamp positions to bounds
particles = agent.map(particles, fn=clamp_position(min, max))

# Apply uniform force
particles = agent.map(particles, fn=add_force(force_vec))

# Update age
particles = agent.map(particles, fn=increment_age(dt))
```

### 7.4 Force Calculations

Compute pairwise interactions:

```dsl
# Brute force (O(n²), accurate)
forces = agent.force_sum(particles, rule=gravitational, method="brute")

# Grid acceleration (O(n), approximate)
forces = agent.force_sum(particles, rule=gravitational, method="grid")

# Barnes-Hut tree (O(n log n), good balance)
forces = agent.force_sum(particles, rule=gravitational, method="barnes_hut")
```

**Force rules:**
```dsl
# Gravitational attraction
fn gravitational(a: Particle, b: Particle) -> Vec2[N] {
  r = b.pos - a.pos
  dist = vec2.length(r)
  if dist < 0.1 { return {x: 0.0, y: 0.0} }  # Avoid singularity

  dir = vec2.normalize(r)
  magnitude = G * a.mass * b.mass / (dist * dist)
  return dir * magnitude
}

# Electrostatic (Coulomb)
fn electrostatic(a: Particle, b: Particle) -> Vec2[N] {
  r = b.pos - a.pos
  dist = vec2.length(r)
  if dist < 0.1 { return {x: 0.0, y: 0.0} }

  dir = vec2.normalize(r)
  magnitude = k_e * a.charge * b.charge / (dist * dist)
  return dir * magnitude
}

# Lennard-Jones potential
fn lennard_jones(a: Particle, b: Particle) -> Vec2[N] {
  r = b.pos - a.pos
  dist = vec2.length(r)
  if dist < 0.1 { return {x: 0.0, y: 0.0} }

  dir = vec2.normalize(r)
  # σ = equilibrium distance, ε = interaction strength
  ratio = σ / dist
  r6 = ratio^6
  r12 = r6 * r6
  magnitude = 24.0 * ε * (2.0 * r12 - r6) / dist
  return dir * magnitude
}
```

### 7.5 Integration

Update agent positions based on forces:

```dsl
# Euler integration (simple, fast)
particles = agent.integrate(particles, forces, dt, method="euler")

# Verlet integration (energy-conserving)
particles = agent.integrate(particles, forces, dt, method="verlet")

# Runge-Kutta 4 (accurate)
particles = agent.integrate(particles, forces, dt, method="rk4")
```

**Manual integration:**
```dsl
particles = agent.map(particles, fn=integrate_manual)

fn integrate_manual(p: Particle, f: Vec2[N]) -> Particle {
  # a = F / m
  acc = f / p.mass

  # v = v + a * dt
  p.vel = p.vel + acc * dt

  # x = x + v * dt
  p.pos = p.pos + p.vel * dt

  return p
}
```

### 7.6 Field Sampling

Agents read from fields:

```dsl
# Sample scalar field
temperatures = agent.sample_field(particles, temperature_field)

# Sample with gradient
gradients = agent.sample_field(particles, density_field, grad=true)

# Use sampled values
particles = agent.map(particles, fn=respond_to_field)

fn respond_to_field(p: Particle, temp: f32) -> Particle {
  # Move away from high temperature
  if temp > 0.5 {
    p.vel = p.vel * 1.1
  }
  return p
}
```

### 7.7 Field Deposition

Agents write to fields:

```dsl
# Point deposition
density = agent.deposit(particles, density_field, kernel="point")

# Gaussian kernel
density = agent.deposit(particles, density_field, kernel="gaussian", radius=2.0)

# Linear interpolation
density = agent.deposit(particles, density_field, kernel="linear")
```

**Use case — Two-way coupling:**
```dsl
step {
  # Agents sample velocity from field
  velocities = agent.sample_field(particles, velocity_field)
  particles = agent.map(particles, fn=apply_fluid_velocity)

  # Agents deposit influence back to field
  influence = agent.deposit(particles, field.alloc(f32, size), kernel="gaussian")
  velocity_field = field.combine(velocity_field, influence, fn=add)
}
```

### 7.8 Spawning and Removal

**Spawning:**
```dsl
# Spawn with template
particles = agent.spawn(particles, template={
  pos: {x: 0.0, y: 0.0},
  vel: {x: 1.0, y: 0.0},
  mass: 1.0,
  energy: 100.0
})

# Spawn with function
particles = agent.spawn(particles, fn=spawn_particle)

fn spawn_particle(parent: Particle) -> Particle {
  return {
    id: new_id(),  # Automatic
    pos: parent.pos + random_offset(),
    vel: parent.vel * 0.5,
    mass: parent.mass * 0.5,
    energy: parent.energy * 0.5
  }
}

# Conditional spawning
particles = agent.when(
  particles,
  pred=should_spawn,
  then=agent.spawn(template=offspring)
)
```

**Removal:**
```dsl
# Remove dead particles
particles = agent.remove(particles, pred=is_dead)

fn is_dead(p: Particle) -> bool {
  return p.energy <= 0.0 || p.age > max_lifetime
}

# Remove out of bounds
particles = agent.remove(particles, pred=out_of_bounds)

fn out_of_bounds(p: Particle) -> bool {
  return p.pos.x < 0.0 || p.pos.x > 100.0 ||
         p.pos.y < 0.0 || p.pos.y > 100.0
}
```

### 7.9 Mutation

Probabilistic changes:

```dsl
# Mutate with fixed rate
particles = agent.mutate(
  particles,
  fn=mutate_energy,
  rate=0.05,  # 5% chance per particle
  seed=42
)

fn mutate_energy(p: Particle) -> Particle {
  # Random perturbation
  p.energy = p.energy * random_float(0.5, 1.5, seed=p.id)
  return p
}
```

### 7.10 Reproduction

Create offspring:

```dsl
# Reproduce with fixed rate
particles = agent.reproduce(
  particles,
  template=offspring_template,
  rate=0.02  # 2% chance per particle
)

# Reproduce with custom function
particles = agent.reproduce(
  particles,
  fn=create_offspring,
  rate=0.02
)

fn create_offspring(parent: Particle) -> Particle {
  return {
    id: new_id(),
    pos: parent.pos + random_offset(radius=1.0, seed=parent.id),
    vel: parent.vel + random_velocity(seed=parent.id),
    mass: parent.mass,
    energy: parent.energy * 0.5  # Split energy
  }
}
```

### 7.11 Reductions

Aggregate information:

```dsl
# Count agents
count = agent.reduce(particles, fn=count, init=0)

# Sum property
total_energy = agent.reduce(particles, fn=sum_energy, init=0.0)

fn sum_energy(acc: f32, p: Particle) -> f32 {
  return acc + p.energy
}

# Find center of mass
com = agent.reduce(particles, fn=center_of_mass, init={x: 0.0, y: 0.0})

fn center_of_mass(acc: Vec2, p: Particle) -> Vec2 {
  return acc + p.pos * p.mass
}
```

### 7.12 Conditional Operations

```dsl
# Apply operation only when condition is met
particles = agent.when(
  particles,
  pred=has_enough_energy,
  then=agent.spawn(template=offspring)
)

fn has_enough_energy(p: Particle) -> bool {
  return p.energy > 50.0
}
```

---

## 8. Signal Processing

Signals represent time-varying values, primarily for audio synthesis and control.

### 8.1 Oscillators

```dsl
# Sine wave
sine = signal.osc(freq=440.0, phase=0.0, shape="sin")

# Triangle wave
triangle = signal.osc(freq=220.0, shape="tri")

# Sawtooth wave
sawtooth = signal.osc(freq=110.0, shape="saw")

# Square wave
square = signal.osc(freq=55.0, shape="square")

# Variable frequency
freq_signal = signal.rate(440.0)
variable_sine = signal.osc(freq=freq_signal, shape="sin")
```

### 8.2 Noise Generators

```dsl
# White noise
white = signal.noise(freq=1.0, seed=42)

# Pink noise (1/f spectrum)
pink = signal.noise(freq=1.0, seed=42, color="pink")

# Brown noise (1/f² spectrum)
brown = signal.noise(freq=1.0, seed=42, color="brown")
```

### 8.3 Envelopes

```dsl
# ADSR envelope
envelope = signal.env(
  attack=0.01,    # Attack time (s)
  decay=0.1,      # Decay time (s)
  sustain=0.7,    # Sustain level (0-1)
  release=0.3     # Release time (s)
)

# Apply envelope
output = signal.map(oscillator, fn=multiply(envelope))
```

### 8.4 Filters

```dsl
# Low-pass filter
filtered = signal.filter(
  signal,
  type="lowpass",
  cutoff=1000.0,  # Hz
  resonance=0.7
)

# High-pass filter
filtered = signal.filter(signal, type="highpass", cutoff=500.0)

# Band-pass filter
filtered = signal.filter(signal, type="bandpass", cutoff=1000.0, resonance=2.0)

# Notch filter
filtered = signal.filter(signal, type="notch", cutoff=60.0)
```

### 8.5 Signal Mixing

```dsl
# Mix multiple signals
mixed = signal.mix([sig1, sig2, sig3])

# Mix with clipping
mixed = signal.mix([sig1, sig2, sig3], clip=true)

# Weighted mix
weighted = signal.combine(sig1, sig2, fn=weighted_sum(0.7, 0.3))
```

### 8.6 Signal Transformation

```dsl
# Map function
doubled = signal.map(signal, fn=multiply(2.0))

# Clamp
clamped = signal.map(signal, fn=clamp(-1.0, 1.0))

# Distortion
distorted = signal.map(signal, fn=tanh)
```

### 8.7 Delays and Integration

```dsl
# Delay by 1 sample
delayed = signal.delay(signal, n=1)

# Delay by N samples
echo = signal.delay(signal, n=44100)  # 1 second at 44.1kHz

# Integration
integrated = signal.integrate(signal, dt)
```

### 8.8 Triggers and Events

```dsl
# Trigger on rising edge
triggered = signal.trigger(event, fn=spawn_note)

fn spawn_note() -> Signal<f32> {
  return signal.osc(freq=random_freq(), shape="sin")
}
```

### 8.9 Sample Rate

```dsl
# Get sample rate
sr = signal.sample_rate()  # e.g., 44100.0 Hz

# Set fixed rate
rate_signal = signal.rate(60.0)  # 60 Hz
```

### 8.10 Block Processing

```dsl
# Render block of samples
block = signal.block(signal, n_samples=512)

# Process block
processed_block = process_audio_block(block)
```

---

## 9. Visual Domain

### 9.1 Field Visualization

```dsl
# Colorize scalar field
vis = visual.colorize(density, palette="viridis")

# Available palettes
vis = visual.colorize(field, palette="fire")      # Black-red-yellow-white
vis = visual.colorize(field, palette="ice")       # Black-blue-cyan-white
vis = visual.colorize(field, palette="rainbow")   # Full spectrum
vis = visual.colorize(field, palette="grayscale") # Black-white
vis = visual.colorize(field, palette="magma")     # Perceptually uniform
vis = visual.colorize(field, palette="viridis")   # Perceptually uniform
```

### 9.2 Agent Visualization

```dsl
# Point sprites with uniform color
vis = visual.points(particles, color="white")

# Color by property
vis = visual.points(particles, color=particle_color_fn)

fn particle_color_fn(p: Particle) -> Vec3 {
  # Map energy to color
  r = p.energy / 100.0
  g = 1.0 - r
  b = 0.5
  return {r: r, g: g, b: b}
}

# Size by property
vis = visual.points(particles, color="white", size=particle_size_fn)
```

### 9.3 Layer Composition

```dsl
# Simple overlay
composed = visual.layer([background, foreground])

# Alpha blending
composed = visual.layer([bg, fg], blend="alpha")

# Additive blending
composed = visual.layer([layer1, layer2, layer3], blend="add")

# Multiply blending
composed = visual.layer([base, overlay], blend="multiply")
```

### 9.4 Post-Processing

```dsl
# Blur
blurred = visual.filter(vis, fn=blur(radius=2.0))

# Sharpen
sharpened = visual.filter(vis, fn=sharpen(amount=1.5))

# Color adjustment
adjusted = visual.filter(vis, fn=adjust_brightness(1.2))

# Custom filter
custom = visual.filter(vis, fn=my_filter)
```

### 9.5 Coordinate Warping

```dsl
# Distortion
warped = visual.coord_warp(vis, fn=radial_distortion)

fn radial_distortion(coord: Vec2) -> Vec2 {
  center = {x: 0.5, y: 0.5}
  offset = coord - center
  dist = vec2.length(offset)
  factor = 1.0 + dist * 0.5
  return center + offset * factor
}
```

### 9.6 Temporal Effects

```dsl
# Motion blur / trail effect
trailed = visual.retime(vis, t_signal=decay_signal)

# Feedback effect
feedback = visual.layer([current, previous * 0.95])
```

### 9.7 Text Overlay

```dsl
# Simple text
text_vis = visual.text("Frame: 123", fmt="default")

# Formatted text
stats = visual.text(
  "FPS: ${fps}\nParticles: ${count}",
  fmt="monospace"
)

# Composite with scene
final = visual.layer([scene, stats])
```

### 9.8 Metadata Tags

```dsl
# Tag for debugging/organization
vis = visual.tag(vis, layer="foreground", type="particles")

# Multiple tags
vis = visual.tag(vis, frame=current_frame, source="simulation")
```

### 9.9 Output

```dsl
# Display to window
visual.output(vis, target="window", format="rgb")

# Save to file
visual.output(vis, target="output.png", format="png")

# Stream to video
visual.output(vis, target="video", format="h264")
```

---

## 10. I/O and Streams

### 10.1 Loading Data

```dsl
# Load field from image
field = io.load_field("data/heightmap.png", format="grayscale")
field = io.load_field("data/texture.png", format="rgb")

# Load from binary
field = io.load_field("data/sim_state.bin", format="f32")

# Load configuration
params = io.load_config("config.json")
```

### 10.2 Saving Data

```dsl
# Save visual
io.save_visual(vis, "output/frame_0001.png")

# Save field
io.save_field(field, "output/state.bin", format="f32")

# Save agents (CSV)
io.save_agents(particles, "output/particles.csv")
```

### 10.3 Real-Time Streams

```dsl
# Audio input stream (microphone)
audio_in = io.stream<f32>("audio_input")

# Video input stream (camera)
video_in = io.stream<Visual>("video_input")

# Control stream (MIDI, OSC)
control = io.stream<f32>("midi_control")

# Note: Streams are nondeterministic unless recorded
```

### 10.4 Output Streams

```dsl
# Audio output
io.output(audio_signal, target="audio", format="f32")

# Video output
io.output(visual, target="video", format="h264", fps=60)

# Network stream
io.output(data, target="network://localhost:8080", format="json")
```

### 10.5 Recording and Playback

```dsl
# Record stream to file
recorded = io.record(stream, path="recording.dat")

# Playback recorded stream (deterministic)
playback = io.playback("recording.dat")
```

---

## 11. Determinism and Reproducibility

### 11.1 Determinism Tiers

| Tier | Definition | Examples |
|------|------------|----------|
| **Strict** | Bit-identical across devices | field.diffuse, agent.force_sum(barnes_hut) |
| **Reproducible** | Deterministic within precision | field.project(cg), signal.filter |
| **Nondeterministic** | External I/O or adaptive stop | io.stream(live), iterate(unbounded) |

### 11.2 RNG System

**Philox 4×32-10 Counter-Based RNG:**
```dsl
# Global seed
set seed = 42

# Seeding formula: hash64(global_seed, agent_id, timestep, local_seed)
random_value = random(seed=local_seed)
```

**Properties:**
- Deterministic given same seed
- Parallel-safe (no shared state)
- Reproducible across platforms
- Fast generation

**Usage:**
```dsl
# Random field
field = field.random(shape=[256, 256], seed=42)

# Random agent initialization
particles = agent.alloc(Particle, count=1000, init_fn=random_init)

fn random_init(id: u64) -> Particle {
  # Each agent gets unique but deterministic random values
  pos = random_vec2(seed=id)
  vel = random_vec2(seed=id + 1000)
  ...
}

# Random mutation
particles = agent.mutate(particles, fn=mutate, rate=0.05, seed=42)
```

### 11.3 Ordering Guarantees

**Agent ordering:**
- Sorted by `(id, creation_index)`
- Stable across operations
- Morton ordering for spatial acceleration

**Field operations:**
- Deterministic accumulation order
- Reproducible reductions
- Consistent stencil traversal

**Parallel execution:**
- Order-independent operations only
- Deterministic scheduling
- No race conditions

### 11.4 Floating-Point Reproducibility

**Strategies:**
- Kahan summation for reductions
- Fixed accumulation order
- Consistent rounding modes
- Optional extended precision

**Caveats:**
- Different architectures may vary slightly
- Use `profile=high` with `precision=f64` for maximum reproducibility
- Iterative solvers may vary within tolerance

### 11.5 Marking Nondeterministic Code

```dsl
@nondeterministic
step {
  # Real-time audio input
  audio_in = io.stream<f32>("microphone")

  # Process audio
  processed = signal.filter(audio_in, type="lowpass", cutoff=1000.0)

  # Output
  io.output(processed, target="audio")
}
```

**Effect:**
- Compiler warning if used with reproducibility checks
- Excluded from determinism tests
- Documented in metadata

### 11.6 Reproducibility Checklist

✅ **Always deterministic:**
- Fixed global seed
- All operations use provided seeds
- No external input streams
- Convergent iterative solvers

⚠️ **Potentially nondeterministic:**
- Adaptive timesteps (use fixed if needed)
- Real-time input streams
- Unbounded iteration
- Platform-specific optimizations

❌ **Never deterministic:**
- Live microphone input
- Wall-clock time
- Network I/O
- True hardware RNG

---

## 12. Solver Configuration

### 12.1 Solver Methods

**Iterative Methods:**

| Method | Convergence | Speed | Memory | Best For |
|--------|-------------|-------|--------|----------|
| Jacobi | Slow | Fast/iter | Low | Simple problems |
| Gauss-Seidel | Medium | Medium | Low | Medium problems |
| CG | Fast | Slow/iter | Medium | Large symmetric |
| Multigrid | Fastest | Medium | High | Very large grids |

### 12.2 Per-Operation Configuration

```dsl
# Explicit solver specification
field = field.diffuse(
  x,
  rate=0.1,
  dt,
  method="cg",
  iter=30,
  tol=1e-6,
  precond="jacobi"
)

# Projection with multigrid
velocity = field.project(
  velocity,
  method="multigrid",
  iter=5,
  smoother="jacobi",
  levels=4
)
```

### 12.3 Solver Registry

Define reusable solver configurations:

```dsl
# Register custom solver
solver high_quality_diffusion = field.diffuse(
  method="cg",
  iter=50,
  tol=1e-8,
  precond="ilu0"
)

# Use registered solver
field = high_quality_diffusion(field, rate=0.1, dt)
```

### 12.4 Profile System

**Predefined profiles:**

```dsl
# Low precision, fast iteration
set profile = low

# Balanced (default)
set profile = medium

# High precision, slow iteration
set profile = high
```

**Profile definitions:**

```dsl
profile low {
  precision = f16
  field.diffuse = jacobi(iter=10)
  field.project = jacobi(iter=20)
  agent.force_sum = grid
}

profile medium {
  precision = f32
  field.diffuse = gauss_seidel(iter=20)
  field.project = cg(iter=40, tol=1e-4)
  agent.force_sum = barnes_hut
}

profile high {
  precision = f64
  field.diffuse = cg(iter=50, tol=1e-8)
  field.project = multigrid(iter=5, levels=4)
  agent.force_sum = barnes_hut
}
```

### 12.5 Precedence Rules

Solver selection follows this precedence:

1. **Per-operation specification** (highest priority)
   ```dsl
   field = field.diffuse(x, rate, dt, method="cg", iter=30)
   ```

2. **Solver registry alias**
   ```dsl
   solver my_diffuse = field.diffuse(method="cg", iter=30)
   field = my_diffuse(x, rate, dt)
   ```

3. **Module profile**
   ```dsl
   module FluidSolver @profile(high) { ... }
   ```

4. **Global profile** (lowest priority)
   ```dsl
   set profile = medium
   ```

### 12.6 Preconditioners

Improve convergence of iterative solvers:

```dsl
# Jacobi preconditioner (diagonal)
field = field.diffuse(x, rate, dt, method="cg", precond="jacobi")

# ILU(0) preconditioner (incomplete LU)
field = field.diffuse(x, rate, dt, method="cg", precond="ilu0")

# Multigrid as preconditioner
field = field.diffuse(x, rate, dt, method="cg", precond="multigrid")
```

### 12.7 Convergence Monitoring

```dsl
# Enable convergence tracking
@monitor_convergence
field = field.diffuse(x, rate, dt, method="cg", iter=50, tol=1e-6)

# Access convergence info
conv_info = field.convergence_info(field)
# Returns: {iterations: u32, residual: f32, converged: bool}
```

### 12.8 Adaptive Solver Selection

```dsl
# Automatically select solver based on problem size
field = field.diffuse(x, rate, dt, method="auto")

# Heuristics:
# - Small grids (<64²): Jacobi
# - Medium grids (64²-512²): CG
# - Large grids (>512²): Multigrid
```

---

## 13. Performance and Optimization

### 13.1 Operation Fusion

**Automatic fusion:**

```dsl
# This pipeline:
result = field.combine(a, b, fn=add)
result = field.mask(result, mask)
result = field.diffuse(result, rate, dt)

# Compiles to single fused kernel (when possible)
```

**Fusion benefits:**
- Reduced memory bandwidth
- Better cache utilization
- Fewer kernel launches (GPU)
- Lower overhead

### 13.2 Lazy Evaluation

Operations build a computation graph:

```dsl
# These don't execute immediately:
step1 = field.advect(x, v, dt)
step2 = field.diffuse(step1, rate, dt)
step3 = field.project(step2)

# Execution happens at:
# - step boundary
# - visual.output()
# - explicit evaluation
```

**Benefits:**
- Global optimization view
- Dead code elimination
- Operation reordering
- Memory planning

### 13.3 Memory Management

**Double buffering:**
```dsl
@double_buffer field : Field2D<f32>

# Automatically managed:
# - Read from current buffer
# - Write to next buffer
# - Swap at step boundary
```

**Explicit buffers:**
```dsl
# When you need control
@storage(device="gpu", layout="row_major")
field : Field2D<f32>
```

**Memory hints:**
```dsl
# Temporary (can be discarded)
@temporary
intermediate = field.laplacian(x)

# Persistent (must be saved)
@persistent
state = field.integrate(x, rate, dt)
```

### 13.4 GPU Offloading

**Automatic offloading:**
```dsl
# Large operations automatically use GPU
field = field.advect(velocity, velocity, dt)  # GPU
particles = agent.force_sum(particles, rule, method="barnes_hut")  # GPU
```

**Manual control:**
```dsl
# Force GPU execution
@device(gpu)
step {
  field = field.diffuse(field, rate, dt)
}

# Force CPU execution
@device(cpu)
step {
  small_field = field.map(small_field, fn=custom)
}
```

**Hybrid execution:**
```dsl
# Compose CPU and GPU work
compose(
  UpdateFluidGPU(velocity),    # GPU
  UpdateParticlesCPU(particles)  # CPU
)
```

### 13.5 Benchmarking

```dsl
@benchmark(name="fluid_step")
step {
  velocity = field.advect(velocity, velocity, dt)
  velocity = field.project(velocity, method="cg", iter=40)
}

# Outputs:
# fluid_step: 12.3ms (avg), 11.8ms (min), 15.2ms (max)
```

**Profiling specific operations:**
```dsl
@benchmark(name="advection")
velocity = field.advect(velocity, velocity, dt)

@benchmark(name="projection")
velocity = field.project(velocity)
```

### 13.6 Performance Tuning Tips

**Field operations:**
- Use appropriate grid size (power of 2 is often faster)
- Prefer MacCormack over semi-Lagrangian for accuracy
- Use multigrid for large grids (>256²)
- Enable operation fusion

**Agent operations:**
- Use grid acceleration for dense systems
- Use Barnes-Hut for sparse systems
- Batch agent operations when possible
- Minimize field sampling (expensive)

**Signal operations:**
- Use block processing for audio
- Avoid per-sample operations in inner loops
- Prefer vectorized operations

**General:**
- Profile before optimizing
- Use appropriate precision (f16/f32/f64)
- Minimize host-device transfers
- Leverage composability for parallelism

### 13.7 Scalability

**Problem size guidelines:**

| Grid Size | Agents | Method | Device | Time/Step |
|-----------|--------|--------|--------|-----------|
| 64² | 100 | Jacobi | CPU | <1ms |
| 256² | 1K | CG | GPU | ~5ms |
| 512² | 10K | Multigrid | GPU | ~10ms |
| 1024² | 100K | Multigrid | Multi-GPU | ~50ms |

---

## 14. MLIR Lowering

### 14.1 Compilation Pipeline

```
DSL Source
    ↓ [Parser]
Abstract Syntax Tree (AST)
    ↓ [Type Checker]
Typed AST
    ↓ [MLIR Lowering]
High-Level MLIR (Custom Dialects)
    ↓ [Dialect Conversion]
Mid-Level MLIR (linalg, scf, arith)
    ↓ [Optimization Passes]
Optimized MLIR
    ↓ [Target Lowering]
LLVM IR / GPU Kernels
    ↓ [Code Generation]
Native Executable / GPU Binaries
```

### 14.2 Dialect Mapping

| DSL Operation | MLIR Dialects | Notes |
|---------------|---------------|-------|
| field.stencil | linalg, affine | Fused neighborhood loops |
| field.advect | linalg, vector | SIMD-optimized interpolation |
| field.diffuse | linalg, scf | Iterative solver with loops |
| field.project | linalg, scf | Poisson solver |
| agent.map | scf.parallel | Parallel agent updates |
| agent.force_sum | scf, gpu | Tree-based force calculation |
| signal.osc | arith, math | Vectorized waveform generation |
| signal.filter | arith, vector | IIR/FIR filter implementation |
| iterate | scf.while | Dynamic loop with condition |
| io.output | async, memref | Asynchronous host callback |

### 14.3 Example Lowering

**DSL Code:**
```dsl
result = field.stencil(field, fn=laplacian, radius=1)
```

**MLIR IR:**
```mlir
%result = linalg.generic {
  indexing_maps = [
    affine_map<(i, j) -> (i-1, j)>,
    affine_map<(i, j) -> (i+1, j)>,
    affine_map<(i, j) -> (i, j-1)>,
    affine_map<(i, j) -> (i, j+1)>,
    affine_map<(i, j) -> (i, j)>,
    affine_map<(i, j) -> (i, j)>
  ],
  iterator_types = ["parallel", "parallel"]
}
ins(%field, %field, %field, %field, %field : memref<?x?xf32>, ...)
outs(%result : memref<?x?xf32>) {
^bb0(%left: f32, %right: f32, %up: f32, %down: f32, %center: f32, %out: f32):
  %sum = arith.addf %left, %right : f32
  %sum2 = arith.addf %sum, %up : f32
  %sum3 = arith.addf %sum2, %down : f32
  %c4 = arith.constant 4.0 : f32
  %prod = arith.mulf %center, %c4 : f32
  %laplacian = arith.subf %sum3, %prod : f32
  linalg.yield %laplacian : f32
}
```

### 14.4 Optimization Passes

**Applied optimizations:**
1. Dead code elimination
2. Common subexpression elimination
3. Loop fusion and tiling
4. Vectorization (SIMD)
5. Memory layout optimization
6. Constant folding
7. Algebraic simplification

**GPU-specific:**
1. Kernel fusion
2. Memory coalescing
3. Shared memory utilization
4. Warp-level optimizations
5. Occupancy maximization

### 14.5 Custom Operations

**Lowering custom functions:**

DSL:
```dsl
fn my_kernel(center: f32, neighbors: Array<f32>) -> f32 {
  sum = 0.0
  for n in neighbors {
    sum = sum + n
  }
  return center + sum * 0.1
}

result = field.stencil(field, fn=my_kernel, radius=1)
```

MLIR:
```mlir
func.func @my_kernel(%center: f32, %neighbors: memref<?xf32>) -> f32 {
  %c0 = arith.constant 0.0 : f32
  %sum = scf.for %i = %c0 to %count step %c1 iter_args(%acc = %c0) -> (f32) {
    %n = memref.load %neighbors[%i] : memref<?xf32>
    %new_acc = arith.addf %acc, %n : f32
    scf.yield %new_acc : f32
  }
  %c_factor = arith.constant 0.1 : f32
  %scaled = arith.mulf %sum, %c_factor : f32
  %result = arith.addf %center, %scaled : f32
  return %result : f32
}
```

---

## 15. Advanced Features

### 15.1 Iterate — Dynamic Loops

```dsl
# Iterate until convergence
converged = iterate(
  expr=improve_solution(x),
  until=is_converged,
  max_iter=1000
)

fn improve_solution(x: Field2D<f32>) -> Field2D<f32> {
  return field.diffuse(x, rate=0.1, dt=0.01)
}

fn is_converged(prev: Field2D<f32>, curr: Field2D<f32>) -> bool {
  diff = field.combine(prev, curr, fn=absolute_difference)
  max_diff = field.reduce(diff, fn=max, init=0.0)
  return max_diff < 1e-6
}
```

**Use cases:**
- Custom iterative solvers
- Adaptive refinement
- Fixed-point iteration
- Game loops

### 15.2 Link — Dependency Metadata

```dsl
# Document dependencies
dependency = link(
  velocity -> density,
  mode="oneway"
)

# Bidirectional coupling
coupling = link(
  particles <-> field,
  mode="bidirectional"
)
```

**Benefits:**
- Graph visualization
- Dependency tracking
- Documentation generation
- No runtime cost (metadata only)

### 15.3 Adaptive Timestep

```dsl
# CFL-based adaptive timestep
set dt = adaptive_dt(
  cfl=0.5,        # CFL condition
  max_dt=0.02,    # Maximum timestep
  min_dt=0.002    # Minimum timestep
)

step {
  # dt automatically adjusted for stability
  velocity = field.advect(velocity, velocity, dt)
}
```

### 15.4 Custom Types

```dsl
# Define complex agent types
type Cell = {
  id: u64,
  pos: Vec2[m],
  nutrients: f32,
  waste: f32,
  dna: Array<i32>,
  generation: u32
}

# Nested types
type Ecosystem = {
  cells: Agents<Cell>,
  field: Field2D<f32>,
  parameters: EcoParams
}

type EcoParams = {
  diffusion_rate: f32,
  consumption_rate: f32,
  reproduction_threshold: f32
}
```

### 15.5 Metadata and Documentation

```dsl
@metadata(key="category", value="fluid_simulation")
@metadata(key="author", value="John Doe")
@metadata(key="version", value="1.0.0")
module NavierStokes(...) {
  @doc("Solve incompressible Navier-Stokes equations")
  @param viscosity : f32 = 0.001 @range(0, 0.1) @doc("Kinematic viscosity")
  @param dt : f32[s] = 0.016 @doc("Timestep duration")

  ...
}
```

### 15.6 Conditional Compilation

```dsl
@if(feature="gpu")
@device(gpu)
step {
  # GPU-specific implementation
  field = field.diffuse(field, rate, dt, method="multigrid")
}

@if(feature="cpu")
@device(cpu)
step {
  # CPU-specific implementation
  field = field.diffuse(field, rate, dt, method="gauss_seidel")
}
```

---

## 16. Best Practices

### 16.1 Code Organization

**Modular structure:**
```dsl
# physics.ccdsl
module FluidPhysics(...) { ... }
module ParticlePhysics(...) { ... }

# rendering.ccdsl
module FluidRenderer(...) { ... }
module ParticleRenderer(...) { ... }

# main.ccdsl
import physics
import rendering

step {
  compose(
    physics.FluidPhysics(velocity, density),
    physics.ParticlePhysics(particles),
    rendering.FluidRenderer(density),
    rendering.ParticleRenderer(particles)
  )
}
```

### 16.2 Parameter Management

```dsl
# Use @param for tunable values
@param viscosity : f32 = 0.001 @range(0, 0.1)
@param diffusion : f32 = 0.0001 @range(0, 0.01)
@param gravity : Vec2[m/s^2] = {x: 0.0, y: -9.81}

# Document parameters
@param particle_count : u32 = 1000 @doc("Number of particles in simulation")
```

### 16.3 Type Safety

```dsl
# Use explicit types with units
position : Vec2[m] = {x: 0.0, y: 0.0}
velocity : Vec2[m/s] = {x: 1.0, y: 0.0}

# Avoid unit-less when units make sense
# Bad: position : Vec2 = ...
# Good: position : Vec2[m] = ...
```

### 16.4 Deterministic Testing

```dsl
# Fix all seeds for reproducibility
set seed = 42

# Use fixed timestep for testing
set dt = 0.01

# Document nondeterministic sections
@nondeterministic
step {
  audio_in = io.stream<f32>("microphone")
  ...
}
```

### 16.5 Performance

**Do:**
- Profile before optimizing
- Use appropriate solver methods
- Leverage composition for parallelism
- Enable operation fusion

**Don't:**
- Premature optimization
- Excessive field sampling from agents
- Deep nesting of operations
- Ignoring memory layout

### 16.6 Documentation

```dsl
# Document modules
@doc("Simulate incompressible fluid flow using Navier-Stokes equations")
module FluidSolver(...) {
  # Document complex operations
  @doc("Project velocity field to enforce incompressibility")
  velocity = field.project(velocity, method="cg", iter=40)
}

# Document custom functions
@doc("Calculate gravitational force between two particles")
fn gravitational_force(a: Particle, b: Particle) -> Vec2[N] {
  ...
}
```

---

## 17. Complete Examples

### 17.1 Smoke Simulation

```dsl
# smoke.ccdsl - Complete smoke simulation with external forces

set profile = medium
set dt = 0.016  # 60 fps

# Physical parameters
@param viscosity : f32 = 0.0001 @range(0, 0.01) @doc "Kinematic viscosity"
@param diffusion : f32 = 0.00001 @range(0, 0.001) @doc "Density diffusion"
@param buoyancy : f32 = 0.1 @range(0, 1.0) @doc "Buoyancy strength"
@param vorticity : f32 = 0.05 @range(0, 0.5) @doc "Vorticity confinement"

# State variables
@double_buffer velocity : Field2D<Vec2[m/s]>
@double_buffer density : Field2D<f32>
@double_buffer temperature : Field2D<f32>

# Initialize
velocity = step.state(field.alloc(Vec2[m/s], size=[256, 256], init={x: 0.0, y: 0.0}))
density = step.state(field.random(shape=[256, 256], seed=42))
temperature = step.state(field.random(shape=[256, 256], seed=43))

# Add smoke source
fn add_source(field: Field2D<f32>, pos: Vec2[m], radius: f32[m], amount: f32) -> Field2D<f32> {
  # Add Gaussian blob at position
  ...
}

step {
  # Add smoke at source location
  density = add_source(density, pos={x: 128.0, y: 32.0}, radius=10.0, amount=1.0)
  temperature = add_source(temperature, pos={x: 128.0, y: 32.0}, radius=10.0, amount=1.0)

  # Buoyancy force (hot rises)
  buoyancy_force = field.map(temperature, fn=compute_buoyancy)
  velocity = field.combine(velocity, buoyancy_force, fn=add_force)

  # Vorticity confinement (maintain swirls)
  curl = field.curl(velocity)
  vorticity_force = field.gradient(curl)
  velocity = field.combine(velocity, vorticity_force, fn=add_scaled_force(vorticity))

  # Advect velocity through itself
  velocity = field.advect(velocity, velocity, dt, method="maccormack")

  # Viscous diffusion
  velocity = field.diffuse(velocity, rate=viscosity, dt, method="cg", iter=20)

  # Enforce incompressibility
  velocity = field.project(velocity, method="cg", iter=40, tol=1e-4)

  # Apply boundary conditions
  velocity = field.boundary(velocity, spec="noSlip")

  # Advect density
  density = field.advect(density, velocity, dt, method="maccormack")
  density = field.diffuse(density, rate=diffusion, dt, method="jacobi", iter=20)
  density = field.boundary(density, spec="clamp")

  # Advect temperature
  temperature = field.advect(temperature, velocity, dt, method="maccormack")
  temperature = field.diffuse(temperature, rate=diffusion, dt, method="jacobi", iter=20)

  # Dissipation
  density = field.map(density, fn=dissipate(0.995))
  temperature = field.map(temperature, fn=dissipate(0.998))

  # Visualize
  visual.output(
    visual.layer([
      visual.colorize(density, palette="viridis"),
      visual.colorize(temperature, palette="fire")
    ], blend="add")
  )
}
```

### 17.2 Flocking with Predators

```dsl
# predator_prey.ccdsl - Boids with predator-prey dynamics

type Boid = {
  id: u64,
  pos: Vec2[m],
  vel: Vec2[m/s],
  is_predator: bool,
  energy: f32,
  age: f32[s]
}

set profile = medium
set dt = 0.016

# Parameters
@param boid_count : u32 = 500
@param predator_count : u32 = 20
@param perception_radius : f32[m] = 2.0
@param separation_weight : f32 = 1.5
@param alignment_weight : f32 = 1.0
@param cohesion_weight : f32 = 1.0
@param flee_weight : f32 = 3.0

# Initialize
boids = step.state(agent.alloc(Boid, count=boid_count + predator_count, init_fn=init_boid))

fn init_boid(id: u64) -> Boid {
  is_pred = id < predator_count
  return {
    id: id,
    pos: random_vec2(seed=id, bounds={min: -50.0, max: 50.0}),
    vel: random_vec2(seed=id+1000, bounds={min: -2.0, max: 2.0}),
    is_predator: is_pred,
    energy: if is_pred { 100.0 } else { 50.0 },
    age: 0.0
  }
}

step {
  # Separate boids into prey and predators
  prey = agent.filter(boids, pred=is_prey)
  predators = agent.filter(boids, pred=is_predator)

  # Prey behavior
  prey_forces = compose(
    agent.force_sum(prey, rule=separation, method="grid"),
    agent.force_sum(prey, rule=alignment, method="grid"),
    agent.force_sum(prey, rule=cohesion, method="grid"),
    agent.force_sum(prey, predators, rule=flee, method="grid")
  )

  # Predator behavior
  predator_forces = compose(
    agent.force_sum(predators, prey, rule=chase, method="grid"),
    agent.force_sum(predators, rule=separation, method="grid")
  )

  # Integrate
  prey = agent.integrate(prey, prey_forces, dt, method="verlet")
  predators = agent.integrate(predators, predator_forces, dt, method="verlet")

  # Wrap boundaries
  prey = agent.map(prey, fn=wrap_periodic)
  predators = agent.map(predators, fn=wrap_periodic)

  # Energy dynamics
  prey = agent.map(prey, fn=decay_energy(rate=0.1))
  predators = agent.map(predators, fn=decay_energy(rate=0.5))

  # Predation
  prey = agent.interaction(prey, predators, rule=predation, radius=1.0)

  # Remove dead agents
  prey = agent.remove(prey, pred=is_dead)
  predators = agent.remove(predators, pred=is_dead)

  # Reproduction
  prey = agent.reproduce(prey, fn=spawn_prey, rate=0.01)
  predators = agent.reproduce(predators, fn=spawn_predator, rate=0.005)

  # Merge back
  boids = agent.merge([prey, predators])

  # Visualize
  visual.output(
    visual.layer([
      visual.points(prey, color="blue", size=2.0),
      visual.points(predators, color="red", size=4.0)
    ])
  )
}
```

### 17.3 Audio-Reactive Particles

```dsl
# audio_particles.ccdsl - Particles react to audio input

type Particle = {
  id: u64,
  pos: Vec2[m],
  vel: Vec2[m/s],
  frequency_band: i32,  # Which frequency band controls this particle
  energy: f32
}

set profile = low
set dt = 0.016

# Audio analysis
audio_in = io.stream<f32>("audio_input")
spectrum = signal.fft(audio_in, size=512)
bands = signal.band_split(spectrum, bands=[
  {low: 20, high: 200},     # Bass
  {low: 200, high: 2000},   # Mids
  {low: 2000, high: 20000}  # Highs
])

# Particles
particles = step.state(agent.alloc(Particle, count=1000, init_fn=init_particle))

step {
  # Extract energy from frequency bands
  bass_energy = signal.reduce(bands[0], fn=rms)
  mid_energy = signal.reduce(bands[1], fn=rms)
  high_energy = signal.reduce(bands[2], fn=rms)

  # Update particle energy based on frequency bands
  particles = agent.map(particles, fn=update_from_audio)

  fn update_from_audio(p: Particle) -> Particle {
    if p.frequency_band == 0 {
      p.energy = bass_energy * 100.0
    } else if p.frequency_band == 1 {
      p.energy = mid_energy * 100.0
    } else {
      p.energy = high_energy * 100.0
    }
    return p
  }

  # Apply forces based on energy
  particles = agent.map(particles, fn=apply_audio_force)

  fn apply_audio_force(p: Particle) -> Particle {
    # Radial force from center based on energy
    center = {x: 0.0, y: 0.0}
    dir = vec2.normalize(p.pos - center)
    force = dir * p.energy * 0.1
    p.vel = p.vel + force * dt
    return p
  }

  # Integration
  particles = agent.integrate(particles, {x: 0.0, y: 0.0}, dt, method="euler")

  # Apply drag
  particles = agent.map(particles, fn=apply_drag(0.95))

  # Boundary
  particles = agent.map(particles, fn=bounce_boundary)

  # Color by frequency band
  visual.output(
    visual.points(particles, color=color_by_band)
  )

  fn color_by_band(p: Particle) -> Vec3 {
    if p.frequency_band == 0 {
      return {r: 1.0, g: 0.0, b: 0.0}  # Red for bass
    } else if p.frequency_band == 1 {
      return {r: 0.0, g: 1.0, b: 0.0}  # Green for mids
    } else {
      return {r: 0.0, g: 0.0, b: 1.0}  # Blue for highs
    }
  }
}
```

---

## 18. Language Reference

### 18.1 Keywords

```
step, substep, module, compose, type, set, profile, solver, iterate, link
fn, return, if, else, for, while
true, false
```

### 18.2 Operators

**Arithmetic:** `+`, `-`, `*`, `/`, `%`, `^`
**Comparison:** `==`, `!=`, `<`, `<=`, `>`, `>=`
**Logical:** `&&`, `||`, `!`
**Assignment:** `=`
**Type annotation:** `:`
**Field access:** `.`
**Array index:** `[]`
**Function call:** `()`

### 18.3 Decorators

- `@double_buffer` — Enable double buffering
- `@param` — Tunable parameter
- `@doc` — Documentation string
- `@range` — Parameter range
- `@benchmark` — Measure timing
- `@metadata` — Attach metadata
- `@nondeterministic` — Mark nondeterministic code
- `@device` — Specify execution device
- `@storage` — Memory layout hints
- `@allow_unit_cast` — Allow lossy unit conversion

### 18.4 Built-in Functions

See sections 6-9 for complete lists of:
- Field operations (section 6)
- Agent operations (section 7)
- Signal operations (section 8)
- Visual operations (section 9)

### 18.5 Grammar Summary

```ebnf
program = statement*

statement = step_block
          | substep_block
          | module_def
          | compose_stmt
          | assignment
          | set_stmt
          | type_def

step_block = "step" "{" statement* "}"

substep_block = "substep" "(" expr ")" "{" statement* "}"

module_def = decorator* "module" IDENT "(" params ")" "{" statement* "}"

assignment = decorator* IDENT [":" type_annotation] "=" expr

expr = literal
     | identifier
     | binary_op
     | unary_op
     | call
     | field_access
     | "(" expr ")"

type_annotation = IDENT ["<" type_annotation ("," type_annotation)* ">"] ["[" UNIT "]"]
```

---

## Appendix A: Migration Guide

### From v0.2.1 to v0.2.2

**New features:**
- `iterate` for dynamic loops
- `link` for dependency metadata
- `field.stencil`, `field.sample_grad`, `field.integrate`
- `agent.mutate`, `agent.reproduce`
- `signal.block`, `io.output(audio)`
- `@benchmark`, `visual.tag`, `@metadata`

**Breaking changes:**
None. v0.2.2 is fully backward compatible with v0.2.1.

---

## Appendix B: Error Messages

**Common compilation errors:**

1. **Unit mismatch**
   ```
   Error: Cannot assign Vec2[m/s] to Vec2[m]
   Hint: Check unit annotations or use @allow_unit_cast
   ```

2. **Type mismatch**
   ```
   Error: Expected Field2D<f32>, got Field2D<Vec2>
   Hint: Use field.map() to transform element types
   ```

3. **Undefined symbol**
   ```
   Error: Undefined identifier 'velocty'
   Hint: Did you mean 'velocity'?
   ```

4. **Invalid solver configuration**
   ```
   Error: Unknown method 'jacoby' for field.diffuse
   Hint: Valid methods are: jacobi, gauss_seidel, cg
   ```

---

## Appendix C: Performance Benchmarks

Typical performance on reference hardware (NVIDIA RTX 3080, Intel i9-11900K):

| Operation | Grid Size | Agents | Time/Step | Throughput |
|-----------|-----------|--------|-----------|------------|
| Navier-Stokes | 256² | - | 3.2ms | 312 FPS |
| Navier-Stokes | 512² | - | 8.5ms | 117 FPS |
| Gray-Scott RD | 512² | - | 5.1ms | 196 FPS |
| Boids (grid) | - | 10K | 1.8ms | 555 FPS |
| Boids (Barnes-Hut) | - | 50K | 6.2ms | 161 FPS |
| Hybrid (fluid+agents) | 256² | 5K | 6.7ms | 149 FPS |

---

## Appendix D: Community Resources

**Official:**
- Documentation: https://ccdsl.org/docs
- GitHub: https://github.com/ccdsl/ccdsl
- Discord: https://discord.gg/ccdsl

**Learning:**
- Tutorial series: https://ccdsl.org/tutorials
- Example gallery: https://ccdsl.org/gallery
- Research papers: https://ccdsl.org/papers

**Contributing:**
- Issue tracker: https://github.com/ccdsl/ccdsl/issues
- Contributing guide: https://github.com/ccdsl/ccdsl/CONTRIBUTING.md
- Development roadmap: https://github.com/ccdsl/ccdsl/ROADMAP.md

---

## Glossary

**Advection** — Transport of a quantity through a velocity field

**Agent** — Discrete entity with individual properties and behaviors

**Barnes-Hut** — O(n log n) algorithm for approximate n-body forces using octrees

**CFL Condition** — Courant-Friedrichs-Lewy stability criterion for explicit timestepping

**Conjugate Gradient (CG)** — Iterative solver for linear systems

**Divergence** — Measure of "outflow" from a point in a vector field

**Double buffering** — Technique using separate read/write buffers to avoid conflicts

**Field** — Dense grid of values representing continuous phenomena

**Laplacian** — Second-order differential operator (∇²)

**MLIR** — Multi-Level Intermediate Representation, compilation framework

**Multigrid** — Fast solver using hierarchy of grid resolutions

**Navier-Stokes** — Equations governing incompressible fluid flow

**Philox** — Counter-based random number generator

**Projection** — Operation to make vector field divergence-free

**RK4** — Fourth-order Runge-Kutta integration method

**Stencil** — Pattern for accessing neighboring grid points

**Verlet integration** — Symplectic integration method conserving energy

**Vorticity** — Measure of local rotation in a fluid

---

**Creative Computation DSL v0.2.2**
*Expressive. Deterministic. Performant.*
