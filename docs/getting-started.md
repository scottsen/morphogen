# Getting Started with Morphogen

Welcome to Morphogen! This guide will help you get up and running in under 30 minutes.

## What is Morphogen?

**Morphogen** is a typed, deterministic domain-specific language for creative computation. It unifies **simulation**, **sound**, **visualization**, and **procedural design** within a single, reproducible execution model.

### Key Features

- ✅ **Deterministic by default** - Bitwise-identical results across runs and platforms
- ✅ **Explicit temporal model** - Time evolution via `flow(dt)` blocks
- ✅ **Declarative state** - `@state` annotations make persistence clear
- ✅ **Multi-domain** - Fields, agents, signals, and visuals in one language
- ✅ **Hot-reload ready** - Interactive development with live code updates
- ✅ **MLIR-based** - Compiles to optimized native code (v0.7.0+)

---

## Installation

### Prerequisites

- **Python 3.9 or higher**
- **pip package manager**

### Install from Source

```bash
# Clone the repository
git clone https://github.com/scottsen/morphogen.git
cd kairo

# Install the package
pip install -e .
```

This will install Morphogen and its core dependencies:
- **numpy** - For numerical operations
- **pillow** - For image output

### Optional I/O Dependencies

For audio I/O and video export (v0.6.0+ features):

```bash
pip install -e ".[io]"
```

This adds:
- **sounddevice** - Real-time audio playback/recording
- **soundfile** - WAV/FLAC file I/O
- **scipy** - Audio processing utilities
- **imageio** - Video export (MP4, GIF)

### Verify Installation

```bash
# Check version
kairo --version

# You should see:
# Morphogen v0.6.0 (stable) / v0.7.0-dev (development)
```

---

## Your First Program

Let's create a simple heat diffusion simulation to understand the basics.

### Example: Heat Diffusion

Create a new file called `hello.kairo`:

```morphogen
# hello.kairo - Heat diffusion simulation

use field, visual

@state temp : Field2D<f32 [K]> = random_normal(
    seed=42,
    shape=(128, 128),
    mean=300.0,
    std=50.0
)

const KAPPA : f32 [m²/s] = 0.1

flow(dt=0.01, steps=100) {
    temp = diffuse(temp, rate=KAPPA, dt, iterations=20)
    output colorize(temp, palette="fire", min=250.0, max=350.0)
}
```

Run it:

```bash
morphogen run hello.kairo
```

You should see a visualization of heat spreading across the field, smoothing out over 100 timesteps.

---

## Core Concepts

### 1. Temporal Model - `flow` blocks

Morphogen programs describe time-evolving systems through `flow` blocks:

```morphogen
flow(dt=0.01, steps=1000) {
    # This block executes 1000 times with timestep 0.01
    temp = diffuse(temp, rate=0.1, dt)
    output colorize(temp, palette="fire")
}
```

**Parameters:**
- `dt` - Timestep duration (in seconds or dimensionless)
- `steps` - Number of iterations to execute

### 2. State Management - `@state`

Persistent variables are declared with `@state`:

```morphogen
@state vel : Field2D<Vec2<f32>> = zeros((256, 256))
@state agents : Agents<Particle> = alloc(count=1000)

flow(dt=0.01) {
    vel = advect(vel, vel, dt)      # Updates vel for next step
    agents = integrate(agents, dt)   # Updates agents for next step
}
```

**Without `@state`**, variables are local to each timestep.

### 3. Type System with Physical Units

Types can carry dimensional information:

```morphogen
temp : Field2D<f32 [K]>           # Temperature in Kelvin
pos : Vec2<f32 [m]>               # Position in meters
vel : Vec2<f32 [m/s]>             # Velocity in m/s

# Unit checking (annotations, not enforced yet)
dist : f32 [m] = 10.0
time : f32 [s] = 2.0
speed = dist / time               # Implicitly: f32 [m/s]
```

### 4. Deterministic Randomness

All randomness is explicit via seeded functions:

```morphogen
@state field : Field2D<f32> = random_normal(
    seed=42,      # Explicit seed
    shape=(100, 100),
    mean=0.0,
    std=1.0
)

# Same seed → same output every time
```

---

## Four Dialects

### 1. Field Dialect - Dense Grid Operations

For simulations on spatial grids (PDEs, fluid dynamics, reaction-diffusion):

```morphogen
use field

@state temp : Field2D<f32> = random_normal(seed=42, shape=(256, 256))

flow(dt=0.1, steps=100) {
    # PDE operations
    temp = diffuse(temp, rate=0.2, dt, iterations=20)
    temp = advect(temp, velocity, dt)

    # Stencil operations
    let grad = gradient(temp)
    let lap = laplacian(temp)

    # Element-wise operations
    temp = temp.map(|x| clamp(x, 0.0, 1.0))
}
```

**Common operations:**
- `diffuse()` - Heat/mass diffusion
- `advect()` - Transport along velocity field
- `project()` - Incompressibility constraint
- `gradient()`, `laplacian()`, `divergence()` - Differential operators

### 2. Agent Dialect - Sparse Particle Systems

For agent-based simulations (particles, boids, crowds):

```morphogen
use agent

struct Boid {
    pos: Vec2<f32>
    vel: Vec2<f32>
}

@state boids : Agents<Boid> = alloc(count=200, init=spawn_boid)

fn spawn_boid(id: u32, rng: RNG) -> Boid {
    return Boid {
        pos: rng.uniform_vec2(min=(0, 0), max=(100, 100)),
        vel: rng.normal_vec2(mean=(0, 0), std=(1, 1))
    }
}

flow(dt=0.01, steps=1000) {
    boids = boids.map(|b| {
        vel: b.vel + flocking_force(b) * dt,
        pos: b.pos + b.vel * dt
    })
}
```

**Status:** ✅ Production-ready as of v0.4.0

### 3. Audio Dialect - Sound Synthesis and Processing

For audio synthesis and processing:

```morphogen
use audio

# Simple synthesis
let pluck = noise(seed=1) |> lowpass(6000)
let string = string(pluck, freq=220, t60=1.5)
let final = string |> reverb(mix=0.12)

# Real-time playback (v0.6.0+)
audio.play(final)

# Export to file (v0.6.0+)
audio.save(final, "output.wav")
```

**Status:** ✅ Production-ready as of v0.5.0 (synthesis) and v0.6.0 (I/O)

**Features:**
- Oscillators (sine, saw, square, triangle, noise)
- Filters (lowpass, highpass, bandpass, EQ)
- Envelopes (ADSR, AR, exponential decay)
- Effects (delay, reverb, chorus, flanger, limiter)
- Physical modeling (Karplus-Strong strings, modal synthesis)

### 4. Visual Dialect - Rendering and Composition

For visualization and video export:

```morphogen
use visual

# Colorize fields
let field_vis = colorize(temp, palette="viridis")

# Render agents (v0.6.0+)
let agent_vis = visual.agents(
    particles,
    width=256,
    height=256,
    color_property='vel',
    palette='fire',
    size=3.0
)

# Layer composition (v0.6.0+)
let combined = visual.composite(
    field_vis,
    agent_vis,
    mode="add",
    opacity=[1.0, 0.7]
)

# Video export (v0.6.0+)
visual.video(frames, "animation.mp4", fps=30)

output combined
```

**Palettes:** `grayscale`, `fire`, `viridis`, `coolwarm`

---

## Complete Examples

### Example 1: Reaction-Diffusion (Gray-Scott)

Create `grayscott.kairo`:

```morphogen
use field, visual

@state u : Field2D<f32> = ones((256, 256))
@state v : Field2D<f32> = zeros((256, 256))

const Du : f32 = 0.16
const Dv : f32 = 0.08
const F : f32 = 0.060
const K : f32 = 0.062

flow(dt=1.0, steps=10000) {
    # Gray-Scott reaction
    let uvv = u * v * v
    let du_dt = Du * laplacian(u) - uvv + F * (1.0 - u)
    let dv_dt = Dv * laplacian(v) + uvv - (F + K) * v

    u = u + du_dt * dt
    v = v + dv_dt * dt

    output colorize(v, palette="viridis")
}
```

Run with:
```bash
morphogen run grayscott.kairo
```

### Example 2: Particle System with Gravity

Create `particles.kairo`:

```morphogen
use agent, visual

struct Particle {
    pos: Vec2<f32 [m]>
    vel: Vec2<f32 [m/s]>
    age: u32
}

@state particles : Agents<Particle> = alloc(count=1000, init=spawn)

fn spawn(id: u32, rng: RNG) -> Particle {
    return Particle {
        pos: rng.uniform_vec2(min=(0, 0), max=(100, 100)),
        vel: rng.normal_vec2(mean=(0, 0), std=(1, 1)),
        age: 0
    }
}

const GRAVITY : Vec2<f32 [m/s²]> = Vec2(0.0, -9.8)

flow(dt=0.01, steps=1000) {
    # Apply gravity
    particles = particles.map(|p| {
        vel: p.vel + GRAVITY * dt,
        pos: p.pos + p.vel * dt,
        age: p.age + 1
    })

    # Bounce off floor
    particles = particles.map(|p| {
        vel: if p.pos.y < 0.0 { Vec2(p.vel.x, -p.vel.y * 0.8) } else { p.vel },
        pos: if p.pos.y < 0.0 { Vec2(p.pos.x, 0.0) } else { p.pos }
    })

    output visual.agents(particles, width=512, height=512, size=2.0)
}
```

### Example 3: Simple Audio Synthesis

Create `synth.kairo`:

```morphogen
use audio

# Generate a plucked string sound
let excitation = noise(seed=7) |> lowpass(cutoff=6000) |> envexp(time=5ms)
let string_tone = string(excitation, freq=220, t60=1.5)
let final = string_tone |> reverb(mix=0.12) |> limiter(threshold=-1dB)

# Play it (requires audio I/O dependencies)
audio.play(final)

# Or save to file
audio.save(final, "pluck.wav")
```

---

## Project Structure

A typical Morphogen project:

```
my-project/
├── main.kairo           # Main program
├── lib/
│   ├── forces.kairo     # Custom force functions
│   └── visuals.kairo    # Custom visualizations
├── examples/
│   ├── 01_simple.kairo
│   └── 02_advanced.kairo
└── output/
    ├── frames/          # Rendered frames
    └── audio/           # Exported audio
```

---

## Running Morphogen Programs

### Basic Execution

```bash
morphogen run program.kairo
```

### With Arguments (future)

```bash
morphogen run program.kairo --steps 10000 --dt 0.001
```

### Interactive Mode (future)

```bash
kairo repl
```

---

## Next Steps

### 1. Explore Examples

Check out the `examples/` directory for:
- **Beginner**: `01_hello_heat.kairo`, `02_pulsing_circle.kairo`
- **Intermediate**: `10_heat_equation.kairo`, `11_gray_scott.kairo`
- **Advanced**: `v0_3_1_complete_demo.kairo`, MLIR phase examples

See `examples/README.md` for a complete guide.

### 2. Read the Specification

For complete language reference:
- **[SPECIFICATION.md](../SPECIFICATION.md)** - Full language specification
- **[LANGUAGE_REFERENCE.md](../LANGUAGE_REFERENCE.md)** - Quick reference guide
- **[AUDIO_SPECIFICATION.md](../AUDIO_SPECIFICATION.md)** - Audio dialect details

### 3. Understand the Architecture

For implementors and advanced users:
- **[ARCHITECTURE.md](../ARCHITECTURE.md)** - Morphogen Stack architecture
- **[docs/v0.7.0_DESIGN.md](v0.7.0_DESIGN.md)** - MLIR integration roadmap

### 4. Join the Community

- **GitHub**: https://github.com/scottsen/kairo
- **Issues**: https://github.com/scottsen/morphogen/issues
- **Discussions**: Share your creations and get help

---

## Performance Tips

### Field Operations

1. **Field Size**: Start with 128×128 or 256×256 for experimentation
   - Larger fields (512×512+) require more computation
   - v0.7.0+ MLIR compilation significantly improves performance

2. **Iteration Count**: For diffusion and projection:
   - **Quick preview**: 10 iterations
   - **Good quality**: 20 iterations (default)
   - **High accuracy**: 40+ iterations

3. **Timestep Selection**:
   - Smaller `dt` = more stable but slower
   - Larger `dt` = faster but may diverge
   - Typical range: 0.001 to 0.1

### Agent Operations

1. **Agent Count**: Performance scales linearly
   - 1,000 agents: Near-instant
   - 10,000 agents: ~0.01s per frame
   - 100,000+ agents: Consider spatial hashing optimizations

2. **Force Calculations**: Use spatial hashing for N-body forces
   ```morphogen
   forces = compute_pairwise_forces(
       agents,
       radius=5.0,  # Interaction radius
       force_func=gravity
   )
   ```

---

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError: No module named 'kairo'`:

```bash
# Reinstall with dependencies
pip install -e .
```

### Audio I/O Not Working

If `audio.play()` or `audio.save()` fail:

```bash
# Install I/O dependencies
pip install -e ".[io]"
```

### MLIR Features Not Available (v0.7.0+)

MLIR compilation requires additional setup:

```bash
# Install MLIR Python bindings (optional)
pip install mlir -f https://github.com/makslevental/mlir-wheels/releases/expanded_assets/latest
```

If MLIR is not available, Morphogen falls back to Python NumPy interpreter.

### Simulation Too Slow

- Reduce field size: `(256, 256)` → `(128, 128)`
- Reduce iterations: `iterations=40` → `iterations=20`
- Enable MLIR compilation for 10-100x speedup (v0.7.0+)

---

## Current Limitations

### v0.6.0 (Stable)

- ✅ Field operations (production-ready)
- ✅ Agent operations (production-ready)
- ✅ Audio synthesis (production-ready)
- ✅ Audio/visual I/O (production-ready)
- ⏳ Physical unit checking (annotations only, not enforced)
- ⏳ Hot-reload (designed, not implemented)
- ⏳ GPU acceleration (planned for v0.7.0 MLIR phases)

### v0.7.0 (Development)

- ✅ MLIR integration foundation (Phase 1)
- ✅ Field operations dialect (Phase 2)
- ✅ Temporal execution (Phase 3)
- ✅ Agent operations dialect (Phase 4)
- ✅ Audio operations dialect (Phase 5)
- ✅ JIT/AOT compilation (Phase 6)
- ⏳ GPU compilation (Phase 7, in progress)

See [docs/v0.7.0_DESIGN.md](v0.7.0_DESIGN.md) for the complete roadmap.

---

## Getting Help

- **Documentation**: Check `docs/` directory for detailed guides
- **Examples**: Browse `examples/` for working code
- **Issues**: Report bugs or request features at https://github.com/scottsen/morphogen/issues
- **Specification**: See `SPECIFICATION.md` for language details

---

**Congratulations!** You're now ready to create your own simulations, sounds, and visualizations with Morphogen. Happy coding! 🎨🎵🔬

---

**Version**: v0.6.0 (stable) / v0.7.0-dev (development)
**Last Updated**: 2025-11-15
