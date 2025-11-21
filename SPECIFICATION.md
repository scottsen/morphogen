# Morphogen v0.10.0 Language Specification

**A Language of Creative Determinism**

*Where computation becomes composition*

---

## Document Information

- **Version**: 0.10.0
- **Date**: 2025-11-16
- **Status**: Draft Specification
- **Authors**: Scott Sen, with Claude
- **Target Audience**: Implementors and Language Designers

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Design Principles](#2-design-principles)
3. [Language Overview](#3-language-overview)
4. [Type System](#4-type-system)
5. [Syntax Reference](#5-syntax-reference)
6. [Temporal Model](#6-temporal-model)
7. [State Management](#7-state-management)
8. [Determinism and RNG](#8-determinism-and-rng)
9. [Field Dialect](#9-field-dialect)
10. [Agent Dialect](#10-agent-dialect)
11. [Audio Dialect (Morphogen.Audio)](#11-audio-dialect-kairoaudio)
12. [Visual Dialect](#12-visual-dialect)
13. [Profile System](#13-profile-system)
14. [Module System](#14-module-system)
15. [I/O and Interop](#15-io-and-interop)
16. [Runtime Model](#16-runtime-model)
17. [MLIR Lowering](#17-mlir-lowering)
18. [Complete Examples](#18-complete-examples)
19. [Implementation Notes](#19-implementation-notes)

---

## 1. Introduction

### 1.1 What is Morphogen?

**Morphogen** is a typed, deterministic domain-specific language for creative computation. It unifies the domains of **simulation**, **sound**, **visualization**, and **procedural design** within a single, reproducible execution model.

Morphogen programs describe time-evolving systems through:
- **Explicit temporal structure** via `flow` blocks
- **Declarative state management** via `@state` annotations
- **Deterministic randomness** via explicit RNG objects
- **Unified semantics** across fields, agents, signals, and visuals

### 1.2 Why Morphogen?

| Problem | Traditional Approach | Morphogen Approach |
|---------|---------------------|----------------|
| **Reproducibility** | Random seeds scattered everywhere | Explicit RNG with deterministic policy |
| **Time** | Hidden global timestep | Explicit `flow(dt)` blocks |
| **State** | Mutable variables everywhere | Declarative `@state` with clear scope |
| **Domains** | Separate frameworks per domain | Unified language across all domains |
| **Performance** | Manual optimization | MLIR lowering with automatic fusion |

### 1.3 Evolution from v0.2.2

Morphogen has evolved from Creative Computation DSL v0.2.2 through several iterations:

| Aspect | v0.2.2 | v0.3.0 | v0.3.1 | v0.6.0 |
|--------|--------|--------|--------|--------|
| **Temporal** | `step` blocks | `flow(dt, substeps)` | `flow(dt, substeps)` ✓ | `flow(dt, substeps)` ✓ |
| **State** | `step.state()` | `@state` | `@state` ✓ | `@state` ✓ |
| **Types** | Explicit domains | Abstract Flow<T> | **Explicit domains** ✓ | **Explicit domains** ✓ |
| **Modules** | Clear syntax | Unclear compose | **Clear syntax** ✓ | **Clear syntax** ✓ |
| **Functions** | Implicit | Unclear | **Explicit fn/lambda** ✓ | **Explicit fn/lambda** ✓ |
| **RNG** | Implicit | Explicit | Explicit ✓ | Explicit ✓ |
| **Audio I/O** | - | - | - | **play/save/load/record** ✓ |
| **Visual I/O** | Basic | Basic | Basic | **agents/composite/video** ✓ |

**Key Milestones**:
- v0.3.1 = v0.3.0 semantics + v0.2.2 completeness
- v0.6.0 = Complete multimedia I/O pipeline with production-ready features

---

## 2. Design Principles

### 2.1 Core Values

1. **Determinism by Default**
   - Every operation yields identical results given identical inputs
   - Across platforms, runs, and time
   - Nondeterminism is explicit and contained

2. **Time is Explicit**
   - No hidden global clock
   - All temporal evolution happens in `flow` blocks
   - Timestep and iteration count are always visible

3. **State is Declarative**
   - `@state` declarations make persistence explicit
   - Clear distinction between per-step computation and cross-step state
   - Enables hot-reload and analysis

4. **Composability**
   - Functions and modules compose algebraically
   - Same patterns work across domains
   - Build complex systems from simple parts

5. **Transparency**
   - Randomness is explicit (RNG objects)
   - Units are part of the type system
   - Solver configuration is visible

6. **Form Follows Semantics**
   - Syntax mirrors mathematical structure
   - Implementation details are hidden
   - Elegance is a design constraint

### 2.2 Non-Goals

- **Not a general-purpose language** - Focused on creative computation
- **Not dynamically typed** - Strong static types with inference
- **Not hardware-specific** - Portable across CPU/GPU/accelerators
- **Not production-first** - Prioritizes clarity and reproducibility

---

## 3. Language Overview

### 3.1 Hello, World

```morphogen
# hello.kairo - Your first Morphogen program

use visual

flow(dt=1.0, steps=1) {
    output text("Hello, Morphogen!")
}
```

### 3.2 Simple Diffusion

```morphogen
# diffusion.kairo - Heat spreading over time

use field, visual

@state temp : Field2D<f32 [K]> = random_normal(seed=42, shape=(128, 128))

flow(dt=0.1, steps=100) {
    temp = diffuse(temp, rate=0.2, dt)
    output colorize(temp, palette="fire")
}
```

### 3.3 Particle System

```morphogen
# particles.kairo - Simple particle physics

use agent, visual

struct Particle {
    pos: Vec2<f32 [m]>
    vel: Vec2<f32 [m/s]>
    age: u32
}

@state particles : Agents<Particle> = alloc(count=1000, init=spawn_particle)

fn spawn_particle(id: u32, rng: RNG) -> Particle {
    return Particle {
        pos: rng.uniform_vec2(min=(0, 0), max=(100, 100)),
        vel: rng.normal_vec2(mean=(0, 0), std=(1, 1)),
        age: 0
    }
}

flow(dt=0.01, steps=1000) {
    # Apply gravity
    let gravity = Vec2(0.0, -9.8)
    particles = particles.map(|p| {
        vel: p.vel + gravity * dt,
        pos: p.pos + p.vel * dt,
        age: p.age + 1
    })

    # Bounce off walls
    particles = particles.map(|p| {
        vel: if p.pos.y < 0.0 { Vec2(p.vel.x, -p.vel.y * 0.8) } else { p.vel },
        pos: if p.pos.y < 0.0 { Vec2(p.pos.x, 0.0) } else { p.pos }
    })

    # Render
    output points(particles, color="white", size=2.0)
}
```

### 3.4 Key Concepts

- **`use` statements** - Import dialect modules
- **`@state` declarations** - Persistent variables across timesteps
- **`flow` blocks** - Temporal scope with explicit dt
- **Type annotations** - Optional but recommended for clarity
- **Physical units** - `[K]`, `[m]`, `[m/s]` are part of the type system
- **Lambdas** - `|args| expr` for inline functions

---

## 4. Type System

### 4.1 Scalar Types

| Type | Description | Size | Range |
|------|-------------|------|-------|
| `bool` | Boolean | 1 bit | true/false |
| `i32` | Signed integer | 32 bits | -2³¹ to 2³¹-1 |
| `i64` | Signed integer | 64 bits | -2⁶³ to 2⁶³-1 |
| `u32` | Unsigned integer | 32 bits | 0 to 2³²-1 |
| `u64` | Unsigned integer | 64 bits | 0 to 2⁶⁴-1 |
| `f32` | Floating point | 32 bits | IEEE 754 single |
| `f64` | Floating point | 64 bits | IEEE 754 double |

### 4.2 Vector Types

Fixed-length numeric vectors:

```morphogen
Vec2<f32>      # 2D vector
Vec3<f64>      # 3D vector
Vec4<i32>      # 4D integer vector
```

Operations:
```morphogen
a : Vec2<f32> = Vec2(1.0, 2.0)
b : Vec2<f32> = Vec2(3.0, 4.0)

c = a + b              # Component-wise addition
d = a * 2.0            # Scalar multiplication
e = dot(a, b)          # Dot product → f32
f = length(a)          # Magnitude → f32
g = normalize(a)       # Unit vector → Vec2<f32>
```

### 4.3 Field Types

Dense grid data over 2D or 3D space:

```morphogen
Field2D<T>     # 2D grid of type T
Field3D<T>     # 3D grid of type T
```

Examples:
```morphogen
temp : Field2D<f32 [K]>              # Temperature field
vel : Field2D<Vec2<f32 [m/s]>>       # Velocity field
density : Field3D<f32 [kg/m³]>       # 3D density
```

Creation:
```morphogen
zeros((256, 256))                    # All zeros
ones((256, 256))                     # All ones
fill((256, 256), value=42.0)         # Fill with value
random_uniform(seed=42, shape=(256, 256), min=0.0, max=1.0)
random_normal(seed=42, shape=(256, 256), mean=0.0, std=1.0)
```

### 4.4 Agent Types

Sparse collections of structured records:

```morphogen
Agents<T>      # Collection of agent records of type T
```

Agents must be defined as structs:
```morphogen
struct Boid {
    pos: Vec2<f32 [m]>
    vel: Vec2<f32 [m/s]>
    energy: f32
}

agents : Agents<Boid>
```

### 4.5 Signal Types

Time-domain functions for audio and control:

```morphogen
Signal<T>      # Time-varying signal of type T
```

Examples:
```morphogen
audio : Signal<f32>              # Audio signal
control : Signal<Vec2<f32>>      # 2D control signal
```

### 4.6 Visual Type

Opaque renderable objects:

```morphogen
Visual         # Image, video frame, or visual composition
```

Visuals are created by dialect operations and composed via layers.

### 4.7 Physical Units

Types can carry physical units:

```morphogen
temp : f32 [K]                   # Temperature in Kelvin
pos : Vec2<f32 [m]>              # Position in meters
vel : Vec2<f32 [m/s]>            # Velocity in m/s
force : Vec2<f32 [N]>            # Force in Newtons
```

**Unit promotion** is safe:
```morphogen
a : f32 [m] = 10.0
b : f32 [m] = 20.0
c = a + b                        # OK: m + m = m
```

**Unit conversion** must be explicit:
```morphogen
time : f32 [s] = 10.0
freq : f32 [Hz] = 1.0 / time     # OK: 1/s = Hz

# Error: cannot mix incompatible units
temp : f32 [K] = 300.0
dist : f32 [m] = 100.0
x = temp + dist                  # ERROR: K + m is invalid
```

### 4.8 Type Inference

Types can be inferred from context:

```morphogen
# Explicit
temp : Field2D<f32> = zeros((256, 256))

# Inferred from initialization
temp = zeros((256, 256))                    # Inferred: Field2D<f32>

# Inferred from operation
vel = zeros((256, 256))
vel2 = advect(vel, vel, dt=0.1)             # Inferred: same type as vel
```

### 4.9 Type Constructors

| Constructor | Example | Description |
|-------------|---------|-------------|
| Scalar | `42.0`, `true`, `100` | Literal values |
| Vector | `Vec2(1.0, 2.0)` | Vector constructor |
| Struct | `Particle { pos: p, vel: v }` | Record constructor |
| Field | `zeros((256, 256))` | Field allocation |
| Agent | `alloc(count=100)` | Agent collection |

---

## 5. Syntax Reference

### 5.1 Comments

```morphogen
# Single-line comment

# Multi-line comments
# span multiple lines
# with # at the start of each line
```

### 5.2 Declarations

#### Variables
```morphogen
x = 42.0                         # Inferred type
y : f32 = 42.0                   # Explicit type
z : f32 [m/s] = 10.0             # With units
```

#### State
```morphogen
@state temp : Field2D<f32> = zeros((256, 256))
@state agents : Agents<Particle> = alloc(count=100)
```

#### Constants
```morphogen
const GRAVITY : f32 [m/s²] = 9.8
const GRID_SIZE : u32 = 256
```

### 5.3 Functions

```morphogen
# Simple function
fn double(x: f32) -> f32 {
    return x * 2.0
}

# Multiple parameters
fn clamp(x: f32, min: f32, max: f32) -> f32 {
    return max(min, min(x, max))
}

# Generic function (future)
fn interpolate<T>(a: T, b: T, t: f32) -> T {
    return a * (1.0 - t) + b * t
}

# No return type (returns unit)
fn print_stats(field: Field2D<f32>) {
    print("Mean: ", mean(field))
    print("Max: ", max(field))
}
```

### 5.4 Lambdas

```morphogen
# Single expression
field.map(|x| x * 2.0)

# Multiple parameters
combine(a, b, |x, y| x + y)

# Struct construction
agents.map(|a| {
    vel: a.vel * 0.99,
    pos: a.pos + a.vel * dt
})

# Multiple statements (block form)
agents.map(|a| {
    let new_vel = a.vel + force * dt
    let new_pos = a.pos + new_vel * dt
    return { vel: new_vel, pos: new_pos }
})
```

### 5.5 Control Flow

#### If/Else Expressions
```morphogen
# Simple if/else
color = if temp > 100.0 { "red" } else { "blue" }

# Nested
speed = if vel > 10.0 {
    "fast"
} else if vel > 5.0 {
    "medium"
} else {
    "slow"
}

# Multi-line
result = if condition {
    # Complex computation
    let x = compute_x()
    let y = compute_y()
    x + y
} else {
    default_value
}
```

#### Iterate (Dynamic Loops)
```morphogen
# Iterate until convergence
pressure = iterate(max_iter=100, tolerance=1e-6) {
    let p_next = relax(pressure)
    let residual = norm(p_next - pressure)
    continue_if(residual > 1e-6, p_next)
}
```

### 5.6 Operators

#### Arithmetic
```morphogen
a + b          # Addition
a - b          # Subtraction
a * b          # Multiplication
a / b          # Division
a % b          # Modulo
-a             # Negation
```

#### Comparison
```morphogen
a == b         # Equal
a != b         # Not equal
a < b          # Less than
a <= b         # Less than or equal
a > b          # Greater than
a >= b         # Greater than or equal
```

#### Logical
```morphogen
a && b         # Logical AND
a || b         # Logical OR
!a             # Logical NOT
```

#### Field Access
```morphogen
particle.pos          # Field access
particle.vel.x        # Nested field access
```

### 5.7 Structs

```morphogen
# Definition
struct Particle {
    pos: Vec2<f32 [m]>
    vel: Vec2<f32 [m/s]>
    mass: f32 [kg]
    active: bool
}

# Construction
p = Particle {
    pos: Vec2(0.0, 0.0),
    vel: Vec2(1.0, 0.0),
    mass: 1.0,
    active: true
}

# Update (immutable - creates new instance)
p2 = Particle { pos: Vec2(1.0, 1.0), ..p }  # Update pos, keep rest
```

### 5.8 Use Statements

```morphogen
use field                        # Import field dialect
use field, agent, visual         # Multiple imports
use signal as sig                # Aliased import (future)
```

---

## 6. Temporal Model

### 6.1 Flow Blocks

**Syntax:**
```morphogen
flow(dt, steps, substeps) {
    # body
}
```

**Parameters:**
- `dt` - Base timestep (required)
- `steps` - Number of iterations (optional, default: infinite/interactive)
- `substeps` - Inner iterations per step (optional, default: 1)

**Examples:**
```morphogen
# Fixed number of steps
flow(dt=0.01, steps=1000) {
    temp = diffuse(temp, rate=0.1, dt)
}

# Infinite loop (interactive mode)
flow(dt=0.016) {  # ~60 FPS
    temp = diffuse(temp, rate=0.1, dt)
    output colorize(temp, palette="fire")
}

# With substeps for stability
flow(dt=0.1, substeps=10) {  # Inner dt = 0.01
    vel = advect(vel, vel, dt / substeps)
}
```

### 6.2 Nested Flows

Flows can be nested for hierarchical time:

```morphogen
flow(dt=0.1, steps=100) {
    # Outer timestep

    # Fast inner physics
    flow(dt=0.01, steps=10) {
        particles = integrate(particles, forces, dt)
    }

    # Slow visualization
    if step % 10 == 0 {
        output render(particles)
    }
}
```

### 6.3 Timestep Access

Within a flow block:
```morphogen
flow(dt=0.01, steps=100) {
    step       # Current iteration number (0-based)
    time       # Current simulation time (step * dt)
    dt         # Timestep value
}
```

---

## 7. State Management

### 7.1 State Declarations

**Syntax:**
```morphogen
@state name : Type = initializer
```

**Examples:**
```morphogen
@state temp : Field2D<f32> = zeros((256, 256))
@state vel : Field2D<Vec2<f32>> = random_normal(seed=1, shape=(256, 256))
@state agents : Agents<Particle> = alloc(count=1000)
@state energy : f32 = 100.0
```

### 7.2 State Semantics

- **Double-buffered** - Reads from current, writes to next
- **Immutable per-step** - State values don't change mid-step
- **Explicit updates** - Must reassign to update:
  ```morphogen
  @state x : f32 = 0.0

  flow(dt=0.1) {
      x = x + 1.0          # Updates x for next step
  }
  ```

### 7.3 Local Variables

Non-state variables are **local to each flow iteration**:

```morphogen
flow(dt=0.1) {
    # Local - recomputed each step
    let dx = gradient(temp)
    let laplacian = div(dx)

    # State - persists across steps
    temp = temp + laplacian * dt
}
```

### 7.4 State Initialization

State can be initialized from:
```morphogen
# Literal values
@state count : u32 = 0

# Constructor functions
@state temp : Field2D<f32> = zeros((256, 256))

# Random distributions
@state noise : Field2D<f32> = random_uniform(seed=42, shape=(256, 256))

# Loaded data
@state initial : Field2D<f32> = load_field("data/initial.png")

# Custom initialization functions
@state agents : Agents<Particle> = alloc(count=100, init=spawn_random)

fn spawn_random(id: u32, rng: RNG) -> Particle {
    return Particle {
        pos: rng.uniform_vec2(min=(0, 0), max=(100, 100)),
        vel: Vec2(0.0, 0.0)
    }
}
```

---

## 8. Determinism and RNG

### 8.1 Determinism Guarantee

Morphogen guarantees **bitwise-identical** results when:
1. Same source code
2. Same input data
3. Same profile settings
4. Same RNG seeds

This holds across:
- Multiple runs
- Different machines (same architecture)
- Different compilers (with same profile)

### 8.2 RNG Objects

**All randomness is explicit** via RNG objects:

```morphogen
# Create RNG with seed
rng = random(seed=42)

# Generate random values
x = rng.uniform(min=0.0, max=1.0)
y = rng.normal(mean=0.0, std=1.0)
v = rng.uniform_vec2(min=(0, 0), max=(10, 10))
```

### 8.3 RNG Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `uniform` | `(min, max) -> f32` | Uniform distribution |
| `normal` | `(mean, std) -> f32` | Gaussian distribution |
| `uniform_vec2` | `(min, max) -> Vec2<f32>` | 2D uniform |
| `normal_vec2` | `(mean, std) -> Vec2<f32>` | 2D Gaussian |
| `uniform_vec3` | `(min, max) -> Vec3<f32>` | 3D uniform |
| `normal_vec3` | `(mean, std) -> Vec3<f32>` | 3D Gaussian |
| `choice` | `(options: [T]) -> T` | Random element |
| `shuffle` | `(list: [T]) -> [T]` | Shuffle list |

### 8.4 Field Random Initialization

```morphogen
# Uniform distribution
field = random_uniform(seed=42, shape=(256, 256), min=0.0, max=1.0)

# Normal distribution
field = random_normal(seed=42, shape=(256, 256), mean=0.0, std=1.0)
```

### 8.5 Agent Random Initialization

```morphogen
@state agents : Agents<Particle> = alloc(count=100, init=spawn_random)

fn spawn_random(id: u32, rng: RNG) -> Particle {
    return Particle {
        pos: rng.uniform_vec2(min=(0, 0), max=(100, 100)),
        vel: rng.normal_vec2(mean=(0, 0), std=(1, 1))
    }
}
```

Each agent gets a **unique deterministic RNG** derived from:
- Global seed
- Agent ID
- Timestep

### 8.6 RNG Algorithm

Morphogen uses **Philox 4×32-10** (counter-based RNG):
- Deterministic
- Parallel-friendly
- No shared state
- Fast on GPU

---

## 9. Field Dialect

Fields represent dense grids over 2D or 3D space.

### 9.1 Creation

```morphogen
use field

# Allocation
zeros((256, 256))                               # All zeros
ones((256, 256))                                # All ones
fill((256, 256), value=42.0)                    # Fill with value

# Random
random_uniform(seed=42, shape=(256, 256), min=0.0, max=1.0)
random_normal(seed=42, shape=(256, 256), mean=0.0, std=1.0)

# From function
from_fn((256, 256), |x, y| sin(x * 0.1) * cos(y * 0.1))

# Load from file
load_field("data/heightmap.png")
```

### 9.2 Element-wise Operations

```morphogen
# Map (unary operation)
field.map(|x| x * 2.0)
field.map(|x| sin(x))
field.map(|x| if x > 0.5 { 1.0 } else { 0.0 })

# Combine (binary operation)
combine(field_a, field_b, |a, b| a + b)
combine(field_a, field_b, |a, b| max(a, b))

# Common operations
field + scalar                     # Add scalar
field * scalar                     # Multiply by scalar
field + other_field                # Add fields
field * other_field                # Multiply fields
```

### 9.3 PDE Operations

#### Diffusion
```morphogen
# Basic diffusion
diffuse(field, rate, dt)

# With options
diffuse(field, rate, dt,
    method="jacobi",        # "jacobi" | "gauss-seidel" | "cg"
    iterations=20,
    boundary="reflect"      # "reflect" | "periodic" | "clamp"
)
```

**Methods:**
- `jacobi` - Simple, parallel, stable
- `gauss-seidel` - Faster convergence, sequential
- `cg` - Conjugate gradient, best for large systems

#### Advection
```morphogen
# Semi-Lagrangian advection
advect(field, velocity, dt)

# With options
advect(field, velocity, dt,
    method="semilagrangian",   # "semilagrangian" | "maccormack" | "bfecc"
    interpolation="bilinear",  # "nearest" | "bilinear" | "bicubic"
    boundary="reflect"
)
```

**Methods:**
- `semilagrangian` - Stable, somewhat diffusive
- `maccormack` - Higher accuracy, more expensive
- `bfecc` - Best accuracy, most expensive

#### Projection (Divergence-Free)
```morphogen
# Make velocity field divergence-free
velocity = project(velocity)

# With options
velocity = project(velocity,
    method="cg",           # "jacobi" | "cg" | "multigrid"
    tolerance=1e-6,
    max_iterations=100,
    boundary="reflect"
)
```

### 9.4 Stencil Operations

```morphogen
# Built-in stencils
gradient(field)                    # ∇f → Field2D<Vec2>
divergence(vector_field)           # ∇·v → Field2D<f32>
laplacian(field)                   # ∇²f → Field2D<f32>
curl(vector_field)                 # ∇×v → Field2D<f32> (2D) or Field3D<Vec3> (3D)

# Custom stencil
stencil(field, radius=1, |neighbors, center| {
    # neighbors: 3×3 array for radius=1
    # Return: new value for center
    let sum = 0.0
    for n in neighbors {
        sum = sum + n
    }
    return sum / neighbors.len()
})
```

### 9.5 Sampling

```morphogen
# Sample at normalized coordinates
sample(field, pos=(0.5, 0.5))              # Returns: T
sample(field, pos=(0.5, 0.5),
    interpolation="bilinear",              # "nearest" | "bilinear" | "bicubic"
    boundary="reflect"
)

# Sample with gradient
sample_grad(field, pos=(0.5, 0.5))         # Returns: (value, gradient)
```

### 9.6 Reduction Operations

```morphogen
sum(field)                         # Sum all elements
mean(field)                        # Average value
min(field)                         # Minimum value
max(field)                         # Maximum value
norm(field)                        # L2 norm
```

### 9.7 Boundary Conditions

```morphogen
@boundary(field) = reflect         # Mirror at edges
@boundary(field) = periodic        # Wrap around
@boundary(field) = clamp           # Extend edge values
@boundary(field) = value(0.0)      # Fixed value at boundary
```

### 9.8 Example: Fluid Simulation

```morphogen
use field, visual

@state vel : Field2D<Vec2<f32 [m/s]>> = zeros((256, 256))
@state density : Field2D<f32> = zeros((256, 256))

const VISCOSITY : f32 = 0.001
const DIFFUSION : f32 = 0.0001

flow(dt=0.01, steps=1000) {
    # Add force
    vel = vel + force_field * dt

    # Advect velocity
    vel = advect(vel, vel, dt, method="maccormack")

    # Diffuse velocity (viscosity)
    vel = diffuse(vel, rate=VISCOSITY, dt, iterations=20)

    # Project (incompressibility)
    vel = project(vel, method="cg", max_iterations=50)

    # Advect and diffuse density
    density = advect(density, vel, dt)
    density = diffuse(density, rate=DIFFUSION, dt)

    # Dissipation
    density = density * 0.995

    # Visualize
    output colorize(density, palette="viridis")
}
```

---

## 10. Agent Dialect

Agents are sparse collections of structured records.

### 10.1 Agent Definition

```morphogen
struct Particle {
    pos: Vec2<f32 [m]>
    vel: Vec2<f32 [m/s]>
    mass: f32 [kg]
    age: u32
}
```

### 10.2 Creation

```morphogen
use agent

# Allocate empty
agents = alloc(count=1000)

# Allocate with template
agents = alloc(count=1000, template=Particle {
    pos: Vec2(0.0, 0.0),
    vel: Vec2(0.0, 0.0),
    mass: 1.0,
    age: 0
})

# Allocate with function
agents = alloc(count=1000, init=spawn_random)

fn spawn_random(id: u32, rng: RNG) -> Particle {
    return Particle {
        pos: rng.uniform_vec2(min=(0, 0), max=(100, 100)),
        vel: rng.normal_vec2(mean=(0, 0), std=(1, 1)),
        mass: rng.uniform(min=0.5, max=2.0),
        age: 0
    }
}
```

### 10.3 Per-Agent Transformations

```morphogen
# Map - transform each agent
agents = agents.map(|a| {
    vel: a.vel + force * dt,
    pos: a.pos + a.vel * dt,
    age: a.age + 1
})

# Filter - remove agents
agents = agents.filter(|a| a.age < 1000)

# Conditional update
agents = agents.map(|a| {
    vel: if a.pos.y < 0.0 {
        Vec2(a.vel.x, -a.vel.y * 0.8)  # Bounce
    } else {
        a.vel
    }
})
```

### 10.4 Force Calculations

```morphogen
# Compute pairwise forces
forces = force_sum(agents, rule=gravity_force)

fn gravity_force(a: Particle, b: Particle) -> Vec2<f32> {
    let r = b.pos - a.pos
    let dist = length(r)
    if dist < 0.1 { return Vec2(0.0, 0.0) }

    let G = 6.674e-11
    let force_mag = G * a.mass * b.mass / (dist * dist)
    return normalize(r) * force_mag
}

# Apply forces
agents = integrate(agents, forces, dt, method="verlet")
```

**Methods:**
- `brute` - O(n²), exact
- `grid` - O(n), approximate
- `barnes_hut` - O(n log n), good balance

### 10.5 Field Interaction

```morphogen
# Sample field at agent positions
agents = sample_field(agents, temp, |a, t| {
    energy: a.energy + t * dt
})

# With gradient
agents = sample_field_grad(agents, temp, |a, t, grad_t| {
    vel: a.vel - grad_t * 0.1  # Move away from heat
})

# Deposit to field
density_field = deposit(agents, shape=(256, 256),
    value=|a| a.mass,
    kernel="gaussian",
    radius=2.0
)
```

### 10.6 Spawning and Removal

```morphogen
# Spawn new agents
agents = spawn(agents, count=10, init=spawn_particle)

# Conditional spawning
if energy > threshold {
    agents = spawn(agents, count=5, init=spawn_particle)
}

# Remove agents
agents = agents.filter(|a| a.age < max_age)
agents = agents.filter(|a| in_bounds(a.pos))
```

### 10.7 Reductions

```morphogen
# Count
n = agents.count()

# Sum property
total_mass = agents.sum(|a| a.mass)

# Average
avg_speed = agents.mean(|a| length(a.vel))

# Center of mass
com = agents.sum(|a| a.pos * a.mass) / agents.sum(|a| a.mass)
```

### 10.8 Example: Flocking

```morphogen
use agent, visual

struct Boid {
    pos: Vec2<f32 [m]>
    vel: Vec2<f32 [m/s]>
}

@state boids : Agents<Boid> = alloc(count=200, init=spawn_boid)

fn spawn_boid(id: u32, rng: RNG) -> Boid {
    return Boid {
        pos: rng.uniform_vec2(min=(0, 0), max=(100, 100)),
        vel: rng.normal_vec2(mean=(0, 0), std=(1, 1))
    }
}

flow(dt=0.01, steps=1000) {
    # Flocking rules
    boids = boids.map(|b| {
        let neighbors = nearby(boids, b.pos, radius=5.0)

        # Separation
        let sep = separation(b, neighbors)

        # Alignment
        let align = alignment(b, neighbors)

        # Cohesion
        let coh = cohesion(b, neighbors)

        # Combine
        let force = sep * 1.5 + align * 1.0 + coh * 1.0

        return {
            vel: b.vel + force * dt,
            pos: b.pos + b.vel * dt
        }
    })

    # Render
    output points(boids, color="white", size=2.0)
}
```

---

## 11. Audio Dialect (Morphogen.Audio)

**Morphogen.Audio** is a compositional, deterministic audio language built as a first-class dialect of the Morphogen kernel. It provides stream-based audio computation with physical modeling primitives, expressive control, and deterministic execution semantics.

### 11.1 Core Concepts

Morphogen.Audio extends Morphogen with audio-specific types and constructs:

- **Stream Types**: `Sig` (audio-rate), `Ctl` (control-rate), `Evt<A>` (events)
- **Compositional Structure**: `scene` and `module` constructs
- **Physical Modeling**: Waveguides, resonant bodies, amplification
- **Deterministic Polyphony**: Reproducible voice allocation and scheduling
- **Profile-Based Quality**: Tunable precision and oversampling

### 11.2 Quick Example

```morphogen
scene PluckDemo {
  let note = note("D3")
  let env  = adsr(5ms, 60ms, 0.6, 200ms)
  let exc  = noise(seed=1) |> lpf(6kHz) |> envexp(10ms)
  out stereo = string(note, 1.2s) exc |> reverb(0.1)
}
```

### 11.3 Key Features

**Oscillators and Synthesis:**
```morphogen
sine(freq=440Hz, phase=0)
saw(freq=440Hz, blep=true)
square(freq=440Hz, pwm=0.5)
noise(type="white", seed=0)
```

**Filters and Effects:**
```morphogen
lpf(cutoff=2kHz, q=0.707)
reverb(mix=0.12, size=0.8)
delay(time=300ms, feedback=0.3)
drive(amount=0.5, shape="tanh")
```

**Physical Modeling:**
```morphogen
string(freq, t60=1.5s, damp=0.3) (exc: Sig)
amp(model="brown", drive=0.6)
cab(ir="4x12.ir", mic="sm57")
```

**Event Scheduling:**
```morphogen
let seq = score [
  at 0s note("A3",1,0.5s),
  at 0.5s note("C4",0.8,0.5s)
] |> loop(2s)
let poly = spawn(seq, voice, max_voices=8)
```

### 11.4 Complete Specification

For the complete Morphogen.Audio language specification, including:
- Detailed type system and rate model
- Comprehensive operator reference
- Physical modeling extensions
- MLIR lowering details
- Conformance tests and examples

See **[AUDIO_SPECIFICATION.md](AUDIO_SPECIFICATION.md)**

### 11.5 Audio I/O Operations (v0.6.0)

**Real-time playback and file I/O** for audio buffers:

```morphogen
# Play audio in real-time
audio.play(buffer, blocking=true)

# Save audio to file
audio.save(buffer, "output.wav")           # WAV format
audio.save(buffer, "output.flac")          # FLAC lossless
audio.save(buffer, "output.wav", format="wav")  # Explicit format

# Load audio from file
buffer = audio.load("input.wav")           # Auto-detect format
buffer = audio.load("input.flac")          # FLAC support

# Record from microphone
recording = audio.record(
    duration=5.0,        # seconds
    sample_rate=44100,
    channels=1           # 1=mono, 2=stereo
)
```

**Supported Formats:**
- **WAV**: Uncompressed (via soundfile or scipy.io.wavfile)
- **FLAC**: Lossless compression (via soundfile)

**Dependencies:**
- `sounddevice`: For playback and recording
- `soundfile`: For FLAC and high-quality WAV I/O
- `scipy`: Fallback for WAV support

**Example Workflow:**
```morphogen
# Synthesize audio
tone = audio.sine(freq=440.0, duration=1.0)
processed = audio.reverb(tone, mix=0.2)

# Export to file
audio.save(processed, "output.flac")

# Play back
audio.play(processed)

# Load and process
loaded = audio.load("input.wav")
filtered = audio.lowpass(loaded, cutoff=2000.0)
audio.save(filtered, "filtered.wav")
```

### 11.6 Integration with Morphogen Core

Morphogen.Audio seamlessly integrates with other Morphogen dialects:
- Audio can drive visual parameters via cross-rate sampling
- Field data can modulate audio parameters
- Agent systems can trigger audio events
- Audio I/O enables real-time playback and file-based workflows
- Unified deterministic execution across all domains

---

## 12. Visual Dialect

Visual operations convert numeric data to images and support multi-layer composition and video export.

### 12.1 Field Visualization

```morphogen
use visual

# Colorize scalar field
visual.colorize(field, palette="viridis")

# Available palettes
# "grayscale", "fire", "viridis", "coolwarm"

# With value range
visual.colorize(field, palette="fire", vmin=0.0, vmax=100.0)
```

### 12.2 Agent Visualization (v0.6.0)

**Render agents as particles** with property-based styling:

```morphogen
# Basic agent rendering
visual.agents(
    agents,
    width=512,
    height=512,
    pos_property="pos",      # Agent property for position
    color=(255, 255, 255),   # White particles
    size=2.0,                # Particle radius
    background=(0, 0, 0)     # Black background
)

# Color by property with palette
visual.agents(
    agents,
    width=512,
    height=512,
    color_property="energy",   # Color particles by energy
    palette="fire",            # Use fire palette for mapping
    size=3.0
)

# Size by property (variable particle sizes)
visual.agents(
    agents,
    width=512,
    height=512,
    size_property="mass",      # Size particles by mass
    size_scale=2.0,            # Scale factor
    color=(100, 200, 255)
)

# Custom bounds (default: auto-compute from agent positions)
visual.agents(
    agents,
    width=512,
    height=512,
    bounds=((0.0, 100.0), (0.0, 100.0))  # ((x_min, x_max), (y_min, y_max))
)
```

**Agent Property Mapping:**
- `pos_property`: Agent property containing Vec2 position (default: "pos")
- `color_property`: Property to map to colors via palette
- `size_property`: Property to map to particle sizes
- `palette`: Color palette for property mapping ("grayscale", "fire", "viridis", "coolwarm")

### 12.3 Layer System (v0.6.0)

**Create and compose visual layers:**

```morphogen
# Create empty layer
layer = visual.layer(width=512, height=512, background=(0, 0, 0))

# Convert existing visual to layer
layer = visual.layer(visual=existing_visual)

# Composite multiple layers
result = visual.composite(
    layer1,
    layer2,
    layer3,
    mode="over"      # Blending mode
)

# Available blend modes:
# - "over": Standard alpha compositing (default)
# - "add": Additive blending
# - "multiply": Multiply blending
# - "screen": Screen blending
# - "overlay": Overlay blending
```

### 12.4 Layer Composition Examples

```morphogen
# Field + Agents composition
field_visual = visual.colorize(temperature, palette="fire")
agent_visual = visual.agents(
    particles,
    width=512,
    height=512,
    color_property="velocity",
    palette="coolwarm"
)
result = visual.composite(field_visual, agent_visual, mode="add")

# Multi-layer with different blend modes
background = visual.colorize(density, palette="viridis")
particles = visual.agents(boids, width=512, height=512, color=(255, 255, 255))
overlay = visual.layer(width=512, height=512, background=(255, 0, 0))

final = visual.composite(
    background,              # Base layer
    particles,               # Additive particles
    overlay,                 # Overlay effect
    mode="over"
)
```

### 12.5 Video Export (v0.6.0)

**Export animations as video files:**

```morphogen
# Export from frame list
frames = [frame1, frame2, frame3, ...]
visual.video(
    frames,
    path="output.mp4",
    fps=30,
    format="mp4"     # or "gif"
)

# Export from generator (memory-efficient)
def frame_generator():
    for step in range(100):
        # Generate frame
        yield visual.colorize(compute_frame(step), palette="fire")

visual.video(
    frame_generator(),
    path="animation.gif",
    fps=15,
    format="gif",
    max_frames=100
)

# Auto-detect format from extension
visual.video(frames, path="output.mp4", fps=30)  # MP4
visual.video(frames, path="output.gif", fps=15)  # GIF
```

**Supported Formats:**
- **MP4**: H.264 codec via imageio + ffmpeg
- **GIF**: Infinite loop, good for short animations

**Dependencies:**
- `imageio`: Core library for video export
- `imageio-ffmpeg`: FFmpeg plugin for MP4 support

### 12.6 Output Operations

```morphogen
# Display in interactive window (Pygame)
visual.display(visual, window_size=(512, 512))

# Save single frame
visual.output(visual, path="output.png")           # PNG
visual.output(visual, path="output.jpg")           # JPEG
visual.output(visual, path="output.png", format="png")  # Explicit
```

### 12.7 Complete Animation Example

```morphogen
use field, agent, visual

@state temp : Field2D<f32> = random_normal(seed=42, shape=(256, 256))
@state particles : Agents<Particle> = alloc(count=500, init=spawn_particle)

flow(dt=0.01, steps=100) {
    # Update physics
    temp = diffuse(temp, rate=0.1, dt)
    particles = integrate(particles, forces, dt)

    # Create layered visualization
    field_vis = visual.colorize(temp, palette="fire", vmin=0.0, vmax=1.0)
    agent_vis = visual.agents(
        particles,
        width=256,
        height=256,
        color_property="energy",
        palette="coolwarm",
        size_property="mass",
        size_scale=2.0
    )

    # Composite layers
    frame = visual.composite(field_vis, agent_vis, mode="add")

    # Display and save
    visual.display(frame)
    visual.output(frame, path="output/frame_{step:04d}.png")
}

# After flow: export video
frames = [load_frame(i) for i in range(100)]
visual.video(frames, path="simulation.mp4", fps=30)
```

---

## 13. Profile System

Profiles define solver configuration and precision policy.

### 13.1 Built-in Profiles

```morphogen
# Fast - Low precision, fast iteration
profile fast {
    precision = f32
    solver.diffuse.method = "jacobi"
    solver.diffuse.iterations = 10
    solver.project.method = "jacobi"
    solver.project.iterations = 20
}

# Balanced - Default profile
profile balanced {
    precision = f32
    solver.diffuse.method = "jacobi"
    solver.diffuse.iterations = 20
    solver.project.method = "cg"
    solver.project.iterations = 50
}

# Accurate - High precision, slow
profile accurate {
    precision = f64
    solver.diffuse.method = "cg"
    solver.diffuse.iterations = 50
    solver.project.method = "multigrid"
    solver.project.iterations = 100
}
```

### 13.2 Custom Profiles

```morphogen
profile my_profile {
    precision = f32
    determinism = "bitexact"

    solver.diffuse.method = "cg"
    solver.diffuse.iterations = 30
    solver.diffuse.tolerance = 1e-6

    solver.project.method = "multigrid"
    solver.project.iterations = 100

    parallel.deterministic = true
}
```

### 13.3 Using Profiles

```morphogen
# Module-level
@profile(accurate)
module fluid_sim {
    # All operations use 'accurate' profile
}

# Block-level
@profile(fast)
flow(dt=0.01) {
    # Fast profile for this flow
}
```

### 13.4 Profile Keys

| Key | Values | Description |
|-----|--------|-------------|
| `precision` | `f32`, `f64` | Floating-point precision |
| `determinism` | `"bitexact"`, `"reproducible"` | Determinism level |
| `solver.<op>.method` | Solver name | Default solver for operation |
| `solver.<op>.iterations` | Integer | Max iterations |
| `solver.<op>.tolerance` | Float | Convergence tolerance |
| `parallel.deterministic` | Boolean | Force deterministic parallelism |

---

## 14. Module System

### 14.1 Module Declaration

```morphogen
module fluid_sim

use field, visual

# Module content
```

### 14.2 Exports

```morphogen
module math_utils

# Private function (not exported)
fn helper(x: f32) -> f32 {
    return x * 2.0
}

# Public function (exported)
export fn square(x: f32) -> f32 {
    return x * x
}

export fn cube(x: f32) -> f32 {
    return x * x * x
}
```

### 14.3 Imports

```morphogen
module main

use field, visual
use math_utils

flow(dt=0.1) {
    let x = math_utils.square(5.0)     # Use exported function
}
```

### 14.4 Module Parameterization

```morphogen
module fluid_sim

# Module parameters
@param viscosity : f32 = 0.001
@param diffusion : f32 = 0.0001

export fn simulate(vel: Field2D<Vec2<f32>>, dt: f32) -> Field2D<Vec2<f32>> {
    let v = diffuse(vel, rate=viscosity, dt)
    return project(v)
}
```

Usage:
```morphogen
use fluid_sim with { viscosity: 0.01, diffusion: 0.001 }

vel = fluid_sim.simulate(vel, dt=0.01)
```

---

## 15. I/O and Interop

### 15.1 Loading Data

```morphogen
# Load field from image
temp = load_field("data/initial.png")

# Load field from binary
temp = load_field("data/initial.bin", format="raw", shape=(256, 256))

# Load configuration
config = load_config("config.toml")
```

### 15.2 Saving Data

```morphogen
# Save visual
save_visual(visual, "output/frame.png")
save_visual(visual, "output/frame.exr", format="exr")

# Save field
save_field(field, "output/field.bin")
save_field(field, "output/field.png", normalize=true)

# Save animation
flow(dt=0.01, steps=100) {
    # ... computation ...
    save_visual(visual, "output/frame_{step:04d}.png")
}
```

### 15.3 Console I/O

```morphogen
# Print values
print("Temperature:", mean(temp))
print("Step {step}, Time {time:.2f}")

# Assertions
assert(mean(temp) > 0.0, "Temperature must be positive")
```

---

## 16. Runtime Model

### 16.1 Execution Model

1. **Initialization** - Evaluate all `@state` initializers
2. **Flow Execution** - Execute flow blocks in order
3. **Step Iteration**:
   - Read current state (buffer A)
   - Execute flow body
   - Write next state (buffer B)
   - Swap buffers
4. **Termination** - When steps complete or user interrupts

### 16.2 Memory Model

- **Fields** - Dense buffers (row-major)
- **Agents** - Structure-of-Arrays (SoA)
- **Signals** - Ring buffers or block-based
- **Visuals** - GPU textures or CPU images

### 16.3 Deterministic Parallelism

- Fixed work-group sizes
- Deterministic reduction (pairwise tree)
- Ordered atomic emulation where needed
- Floating-point associativity controlled by profile

### 16.4 Hot-Reload

For interactive development:
1. Track state symbols by name and type
2. On code change, recompile
3. Link new IR to existing state buffers
4. Resume at next tick boundary

---

## 17. MLIR Lowering

### 17.1 Dialect Mapping

| Morphogen | MLIR Dialect | Purpose |
|-------|--------------|---------|
| `flow` | `scf.for`, `scf.while` | Loop structure |
| Field ops | `linalg`, `affine` | Dense tensor ops |
| Agent ops | `scf`, `gpu` | Sparse iteration |
| Signal ops | `async`, `memref` | Streaming buffers |
| Visual ops | Custom `visual` dialect | Render graph |
| RNG | `math`, `arith` | Philox implementation |

### 17.2 Lowering Pipeline

```
Morphogen AST
    ↓ Type checking
Typed AST
    ↓ Lowering
MLIR (high-level dialects)
    ↓ Optimization passes
MLIR (low-level dialects)
    ↓ Backend codegen
LLVM IR / SPIR-V / Metal
    ↓ Compilation
Native code
```

### 17.3 Optimization Opportunities

- **Operation fusion** - Combine consecutive field ops
- **Dead code elimination** - Remove unused computations
- **Loop unrolling** - Substep loops
- **Vectorization** - SIMD operations
- **GPU offload** - Large field/agent operations

---

## 18. Complete Examples

### 18.1 Heat Diffusion

```morphogen
# diffusion.kairo - Simple heat diffusion

use field, visual

@state temp : Field2D<f32 [K]> = random_normal(
    seed=42,
    shape=(128, 128),
    mean=300.0,
    std=50.0
)

const KAPPA : f32 [m²/s] = 0.1

flow(dt=0.01, steps=500) {
    temp = diffuse(temp, rate=KAPPA, dt, iterations=20)
    output colorize(temp, palette="fire", min=250.0, max=350.0)
}
```

### 18.2 Smoke Simulation

```morphogen
# smoke.kairo - Incompressible fluid with density

use field, visual

@state vel : Field2D<Vec2<f32 [m/s]>> = zeros((256, 256))
@state density : Field2D<f32> = zeros((256, 256))
@state temp : Field2D<f32 [K]> = fill((256, 256), value=300.0)

const VISCOSITY : f32 = 0.001
const DIFFUSION : f32 = 0.0001
const BUOYANCY : f32 = 0.1

flow(dt=0.01, steps=1000) {
    # Add smoke source
    if step < 100 {
        density = add_source(density, pos=(128, 220), radius=10.0, amount=1.0)
        temp = add_source(temp, pos=(128, 220), radius=10.0, amount=50.0)
    }

    # Buoyancy force (hot air rises)
    let buoyancy_force = temp.map(|t| Vec2(0.0, (t - 300.0) * BUOYANCY))
    vel = vel + buoyancy_force * dt

    # Advect velocity
    vel = advect(vel, vel, dt, method="maccormack")

    # Diffuse velocity (viscosity)
    vel = diffuse(vel, rate=VISCOSITY, dt, iterations=20)

    # Project (incompressibility)
    vel = project(vel, method="cg", max_iterations=50)

    # Advect density and temperature
    density = advect(density, vel, dt)
    temp = advect(temp, vel, dt)

    # Diffuse density
    density = diffuse(density, rate=DIFFUSION, dt, iterations=10)

    # Dissipation
    density = density * 0.995
    temp = temp * 0.999

    # Visualize
    output colorize(density, palette="viridis")
}

fn add_source(field: Field2D<f32>, pos: (f32, f32), radius: f32, amount: f32) -> Field2D<f32> {
    return from_fn(field.shape(), |x, y| {
        let dx = x - pos.0
        let dy = y - pos.1
        let dist = sqrt(dx * dx + dy * dy)
        let value = if dist < radius { amount } else { 0.0 }
        return sample(field, pos=(x, y)) + value
    })
}
```

### 18.3 Reaction-Diffusion (Gray-Scott)

```morphogen
# gray_scott.kairo - Pattern formation

use field, visual

@state u : Field2D<f32> = ones((256, 256))
@state v : Field2D<f32> = zeros((256, 256))

const Du : f32 = 0.16
const Dv : f32 = 0.08
const F : f32 = 0.060
const K : f32 = 0.062

flow(dt=1.0, steps=10000) {
    # Add initial perturbation
    if step == 0 {
        v = add_circle(v, center=(128, 128), radius=20.0, value=0.5)
    }

    # Gray-Scott reaction
    let uvv = u * v * v
    let du_dt = Du * laplacian(u) - uvv + F * (1.0 - u)
    let dv_dt = Dv * laplacian(v) + uvv - (F + K) * v

    u = u + du_dt * dt
    v = v + dv_dt * dt

    # Visualize
    output colorize(v, palette="viridis", min=0.0, max=1.0)
}

fn add_circle(field: Field2D<f32>, center: (f32, f32), radius: f32, value: f32) -> Field2D<f32> {
    return from_fn(field.shape(), |x, y| {
        let dx = x - center.0
        let dy = y - center.1
        let dist = sqrt(dx * dx + dy * dy)
        if dist < radius {
            return value
        } else {
            return sample(field, pos=(x, y))
        }
    })
}
```

### 18.4 Flocking (Boids)

```morphogen
# boids.kairo - Flocking behavior

use agent, visual

struct Boid {
    pos: Vec2<f32 [m]>
    vel: Vec2<f32 [m/s]>
}

@state boids : Agents<Boid> = alloc(count=200, init=spawn_boid)

const MAX_SPEED : f32 = 5.0
const SEPARATION_RADIUS : f32 = 5.0
const ALIGNMENT_RADIUS : f32 = 10.0
const COHESION_RADIUS : f32 = 10.0

fn spawn_boid(id: u32, rng: RNG) -> Boid {
    return Boid {
        pos: rng.uniform_vec2(min=(0, 0), max=(100, 100)),
        vel: rng.normal_vec2(mean=(0, 0), std=(1, 1))
    }
}

flow(dt=0.01, steps=1000) {
    boids = boids.map(|b| {
        let neighbors = nearby(boids, b.pos, radius=COHESION_RADIUS)

        # Separation - avoid crowding
        let sep = Vec2(0.0, 0.0)
        let sep_count = 0
        for n in neighbors {
            if distance(b.pos, n.pos) < SEPARATION_RADIUS {
                sep = sep + (b.pos - n.pos)
                sep_count = sep_count + 1
            }
        }
        if sep_count > 0 {
            sep = sep / sep_count
        }

        # Alignment - steer towards average heading
        let align = Vec2(0.0, 0.0)
        for n in neighbors {
            align = align + n.vel
        }
        if neighbors.len() > 0 {
            align = align / neighbors.len()
        }

        # Cohesion - steer towards center of mass
        let coh = Vec2(0.0, 0.0)
        for n in neighbors {
            coh = coh + n.pos
        }
        if neighbors.len() > 0 {
            coh = coh / neighbors.len()
            coh = coh - b.pos
        }

        # Combine forces
        let force = sep * 1.5 + align * 1.0 + coh * 1.0
        let new_vel = b.vel + force * dt

        # Limit speed
        let speed = length(new_vel)
        if speed > MAX_SPEED {
            new_vel = normalize(new_vel) * MAX_SPEED
        }

        return {
            vel: new_vel,
            pos: b.pos + new_vel * dt
        }
    })

    # Wrap boundaries
    boids = boids.map(|b| {
        pos: wrap(b.pos, min=(0, 0), max=(100, 100))
    })

    # Render
    output points(boids, color="white", size=2.0)
}
```

---

## 19. Implementation Notes

### 19.1 Compiler Architecture

```
morphogen/
├── frontend/
│   ├── lexer.rs           # Tokenization
│   ├── parser.rs          # AST construction
│   └── ast.rs             # AST definitions
├── types/
│   ├── checker.rs         # Type checking
│   ├── inference.rs       # Type inference
│   └── units.rs           # Physical units
├── mlir/
│   ├── lowering.rs        # AST → MLIR
│   ├── dialects.rs        # Dialect definitions
│   └── passes.rs          # Optimization passes
├── runtime/
│   ├── engine.rs          # Flow scheduler
│   ├── state.rs           # State management
│   └── rng.rs             # Philox RNG
└── stdlib/
    ├── field.rs           # Field operations
    ├── agent.rs           # Agent operations
    ├── signal.rs          # Signal operations
    └── visual.rs          # Visual operations
```

### 19.2 Key Implementation Challenges

1. **Deterministic Parallelism**
   - Use fixed work-group sizes
   - Implement tree-based reductions
   - Control floating-point associativity

2. **Hot-Reload**
   - Track state by symbol name + type hash
   - Serialize/deserialize state buffers
   - Rebuild IR without losing state

3. **Profile System**
   - Profiles must affect codegen, not just runtime
   - Need compile-time and runtime components
   - Allow per-operation overrides

4. **Unit Checking**
   - Dimensional analysis at type-check time
   - Unit inference for operations
   - Error messages with unit hints

### 19.3 MVP Implementation Order

**Phase 1: Frontend (2 weeks)**
- Lexer
- Parser
- AST
- Type system (without units)

**Phase 2: Core Runtime (2 weeks)**
- Flow scheduler
- State management
- RNG (Philox)

**Phase 3: Field Dialect (2 weeks)**
- Field data structure
- Basic operations (map, combine)
- PDE operations (diffuse, advect, project)

**Phase 4: Visual Dialect (1 week)**
- Colorization
- Display window
- Frame output

**Phase 5: Polish (1 week)**
- Error messages
- Examples
- Documentation

**Total: 8 weeks to MVP**

### 19.4 Testing Strategy

- **Unit tests** - Per-module, test individual functions
- **Integration tests** - End-to-end examples
- **Determinism tests** - Run same program multiple times, check bitwise equality
- **Performance tests** - Benchmark against target metrics
- **Visual tests** - Generate reference images, compare

---

## Appendix A: Grammar Summary

```ebnf
program = { use_stmt | const_decl | struct_decl | fn_decl | state_decl | flow_block }

use_stmt = "use" ident { "," ident }

const_decl = "const" ident ":" type "=" expr

struct_decl = "struct" ident "{" { field_decl } "}"
field_decl = ident ":" type

fn_decl = "fn" ident "(" params ")" [ "->" type ] block
params = [ param { "," param } ]
param = ident ":" type

state_decl = "@state" ident ":" type "=" expr

flow_block = "flow" "(" flow_params ")" block
flow_params = "dt" "=" expr [ "," "steps" "=" expr ] [ "," "substeps" "=" expr ]

block = "{" { stmt } "}"

stmt = let_stmt | assign_stmt | expr_stmt | flow_block | if_stmt | return_stmt

let_stmt = "let" ident [ ":" type ] "=" expr
assign_stmt = ident "=" expr
expr_stmt = expr
if_stmt = "if" expr block [ "else" ( if_stmt | block ) ]
return_stmt = "return" expr

expr = lambda_expr | binary_expr | unary_expr | primary_expr

lambda_expr = "|" [ params ] "|" ( expr | block )

binary_expr = expr op expr
unary_expr = op expr

primary_expr = literal | ident | call_expr | field_access | vec_constructor | struct_constructor | paren_expr

call_expr = ident "(" [ args ] ")"
args = expr { "," expr }

field_access = expr "." ident

vec_constructor = "Vec2" | "Vec3" | "Vec4" "(" args ")"
struct_constructor = ident "{" { field_init } "}"
field_init = ident ":" expr

paren_expr = "(" expr ")"

type = scalar_type | vec_type | field_type | agent_type | signal_type | "Visual"
scalar_type = ( "bool" | "i32" | "i64" | "u32" | "u64" | "f32" | "f64" ) [ "[" unit "]" ]
vec_type = ( "Vec2" | "Vec3" | "Vec4" ) "<" type ">"
field_type = ( "Field2D" | "Field3D" ) "<" type ">"
agent_type = "Agents" "<" ident ">"
signal_type = "Signal" "<" type ">"
```

---

## Appendix B: Built-in Functions

### Math
- `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`
- `sqrt`, `pow`, `exp`, `log`, `log2`, `log10`
- `abs`, `min`, `max`, `clamp`
- `floor`, `ceil`, `round`, `fract`

### Vector
- `dot(a, b)` - Dot product
- `length(v)` - Magnitude
- `normalize(v)` - Unit vector
- `distance(a, b)` - Euclidean distance

### Field (See Section 9)

### Agent (See Section 10)

### Signal (See Section 11)

### Visual (See Section 12)

---

## Appendix C: Comparison Matrix

| Feature | Morphogen v0.3.1 | Python+NumPy | GLSL | Faust |
|---------|--------------|--------------|------|-------|
| **Deterministic** | ✅ Yes | ⚠️ Partial | ❌ No | ✅ Yes |
| **Multi-domain** | ✅ Fields+Agents+Signals | ⚠️ Via libraries | ❌ Graphics only | ❌ Audio only |
| **Type safety** | ✅ Strong static | ❌ Dynamic | ✅ Static | ✅ Static |
| **Units** | ✅ Built-in | ❌ No | ❌ No | ❌ No |
| **Hot-reload** | ✅ Yes | ⚠️ Partial | ❌ No | ❌ No |
| **MLIR** | ✅ Yes | ❌ No | ❌ No | ❌ No |
| **Learning curve** | Medium | Easy | Medium | Hard |

---

## Appendix D: Future Extensions (v0.4+)

### Space Abstraction
```morphogen
space fluid = Space(dim=2, size=(256, 256), boundary=reflect)

@state temp : Field<f32> in fluid
```

### Streaming I/O
```morphogen
audio_in = stream<Signal<f32>>("microphone")
video_out = stream<Visual>("display")
```

### Generic Types
```morphogen
fn interpolate<T: Numeric>(a: T, b: T, t: f32) -> T {
    return a * (1.0 - t) + b * t
}
```

### Error Handling
```morphogen
fn load_field(path: str) -> Result<Field2D<f32>, IOError> {
    # ...
}

field = try load_field("data.png") catch {
    zeros((256, 256))
}
```

### Match Expressions
```morphogen
state = match agent.state {
    Idle => wander(),
    Hunting => chase(),
    Fleeing => run()
}
```

---

## Document History

- **v0.10.0** (2025-11-16): Five New Computational Domains
  - Graph/Network domain with 15+ algorithms
  - Signal Processing domain with FFT/STFT and spectral analysis
  - State Machine domain with FSM and behavior trees
  - Terrain Generation domain with erosion and biome classification
  - Computer Vision domain with edge detection and feature extraction
  - Updated domain count to 23 implemented domains
  - RigidBody Physics domain (v0.8.2)
  - Cellular Automata domain (v0.9.1)
  - Cross-domain composition infrastructure (v0.9.0)

- **v0.6.0** (2025-11-14): Audio I/O and Visual Extensions
  - Audio I/O operations: play(), save(), load(), record()
  - Visual extensions: agents(), layer(), composite(), video()
  - Complete agent visualization with property-based styling
  - Layer composition system with multiple blend modes
  - Video export (MP4, GIF) with generator support
  - Updated examples and complete API documentation

- **v0.3.1** (2025-11-06): Refined specification
  - Explicit domain types (not Flow<T>)
  - Clear module system
  - Function and lambda syntax
  - Complete dialect documentation
  - Comprehensive examples

- **v0.3.0** (2025-11-06): Initial ChatGPT specification
  - Core Flow/Space/Time model
  - Profile system
  - MLIR lowering

- **v0.2.2** (2025-11-05): Creative Computation DSL
  - step blocks
  - Comprehensive stdlib
  - Complete examples

---

**End of Morphogen v0.6.0 Specification**
