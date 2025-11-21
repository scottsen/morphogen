# Kairo Example Portfolio Plan

**Date:** 2025-11-07
**Purpose:** Strategic plan for compelling example programs
**Goal:** Demonstrate Kairo's unique features across complexity levels and domains

---

## Current State Analysis

### Existing Examples (5 total)
1. ✅ `v0_3_1_velocity_calculation.kairo` - Functions, physical units, if/else
2. ✅ `v0_3_1_struct_physics.kairo` - Structs, physics, bouncing ball
3. ✅ `v0_3_1_lambdas_and_flow.kairo` - Lambdas, closures, substeps
4. ✅ `v0_3_1_recursive_factorial.kairo` - Recursion
5. ✅ `v0_3_1_complete_demo.kairo` - Feature integration

### Gap Analysis

**What's Working:**
- Good coverage of language features (functions, structs, lambdas, recursion)
- Physical units demonstrated
- Flow blocks shown

**What's Missing:** 🚨
- ❌ **Visual examples** (stdlib has visual ops but NO examples using them!)
- ❌ **Field operations** (core Kairo feature, not demonstrated)
- ❌ **Real PDE simulations** (heat, wave, fluid)
- ❌ **"Hello World"** style beginner intro
- ❌ **Visually compelling** demonstrations
- ❌ **Multi-domain** examples (fields + agents)

---

## Example Portfolio Strategy

### Design Principles
1. **Visual First** - Every example should produce compelling visual output
2. **Progressive Complexity** - Clear path from beginner to advanced
3. **Domain Diversity** - Physics, biology, art, mathematics
4. **Feature Showcase** - Each highlights specific Kairo strengths
5. **"Wow Factor"** - Examples that make people excited to try Kairo

### Target Portfolio (20 examples)
- **Tier 1 (Beginner):** 5 examples - Simple, clear, immediate results
- **Tier 2 (Intermediate):** 8 examples - Real simulations, multiple features
- **Tier 3 (Advanced):** 5 examples - Complex interactions, production-quality
- **Tier 4 (Showcase):** 2 examples - "Flagship" demonstrations

---

## 🟢 TIER 1: Beginner Examples (5)
**Goal:** Get started in 5 minutes, see immediate visual results

### 1.1 Hello Heat (⭐ HIGHEST PRIORITY)
**File:** `01_hello_heat.kairo`
**Tagline:** "Your first Kairo program - watch heat diffuse"
**Visual:** Colorful heat diffusion from hot center
**Lines of Code:** ~15

**Demonstrates:**
- Field allocation and initialization
- Single field operation (diffuse)
- Visual output (colorize + output)
- Flow blocks with dt and steps

**Why Important:** This should be THE first example everyone runs. Simple, fast, visually satisfying.

**Implementation Sketch:**
```kairo
use field, visual

@state temp : Field2D<f32> = zeros((128, 128))

# Set hot spot in center
# (need stdlib function for this, or use map)

flow(dt=0.1, steps=100) {
    temp = diffuse(temp, rate=0.1, dt, iterations=10)
    output colorize(temp, palette="fire", min=0.0, max=100.0)
}
```

---

### 1.2 Pulsing Circle
**File:** `02_pulsing_circle.kairo`
**Tagline:** "Animated visualization with lambda expressions"
**Visual:** Circle that grows and shrinks
**Lines of Code:** ~20

**Demonstrates:**
- Field map operations with lambdas
- Coordinate math
- Time-based animation
- Visual palettes

**Implementation Sketch:**
```kairo
use field, visual

@state time = 0.0

flow(dt=0.05, steps=200) {
    radius = 20.0 + 10.0 * sin(time)

    # Create circular field using map with coordinates
    field = zeros((128, 128)).map(|value, x, y| {
        dx = x - 64.0
        dy = y - 64.0
        dist = sqrt(dx * dx + dy * dy)
        return if dist < radius then 1.0 else 0.0
    })

    output colorize(field, palette="viridis")
    time = time + dt
}
```

---

### 1.3 Wave Ripples
**File:** `03_wave_ripples.kairo`
**Tagline:** "Simple wave equation - drop a stone in water"
**Visual:** Concentric circular waves
**Lines of Code:** ~25

**Demonstrates:**
- Two-field wave equation
- Laplacian operator
- Physical simulation
- Temporal dynamics

**Implementation Sketch:**
```kairo
use field, visual

@state u : Field2D<f32> = zeros((128, 128))
@state v : Field2D<f32> = zeros((128, 128))

const C : f32 = 0.5  # Wave speed

# Initialize with central disturbance
# u[64, 64] = 1.0

flow(dt=0.1, steps=300) {
    # Wave equation: d²u/dt² = c² ∇²u
    # Using two fields: u (displacement), v (velocity)

    lap = laplacian(u)
    v = v + lap * C * C * dt
    u = u + v * dt

    # Damping
    v = v * 0.995

    output colorize(u, palette="coolwarm", min=-1.0, max=1.0)
}
```

---

### 1.4 Random Walk Visualization
**File:** `04_random_walk.kairo`
**Tagline:** "Brownian motion creates beautiful patterns"
**Visual:** Accumulating density from random walks
**Lines of Code:** ~30

**Demonstrates:**
- RNG with determinism
- State accumulation
- Random field operations
- Long-term behavior

---

### 1.5 Color Gradient Flow
**File:** `05_gradient_flow.kairo`
**Tagline:** "Watch gradients flow and mix"
**Visual:** Flowing color gradients
**Lines of Code:** ~20

**Demonstrates:**
- Advection operation
- Vector fields
- Color visualization
- Smooth motion

---

## 🟡 TIER 2: Intermediate Examples (8)
**Goal:** Real simulations demonstrating multiple Kairo features

### 2.1 Heat Equation (Full) (⭐ HIGH PRIORITY)
**File:** `10_heat_equation.kairo`
**Tagline:** "Complete heat diffusion with sources and sinks"
**Visual:** Heat map with hot/cold regions
**Lines of Code:** ~40

**Demonstrates:**
- Diffusion with sources
- Boundary conditions
- Physical units (temperature in Kelvin)
- Multiple field operations

**Implementation Sketch:**
```kairo
use field, visual

@state temp : Field2D<f32 [K]> = random_normal(
    seed=42,
    shape=(256, 256),
    mean=300.0,
    std=20.0
)

const KAPPA : f32 [m²/s] = 0.1  # Thermal diffusivity

flow(dt=0.01, steps=500) {
    # Apply heat source at top
    # (need stdlib function to set region)

    # Apply cold sink at bottom

    # Diffuse
    temp = diffuse(temp, rate=KAPPA, dt, iterations=20)

    output colorize(temp, palette="fire", min=250.0, max=350.0)
}
```

---

### 2.2 Gray-Scott Reaction-Diffusion (⭐ HIGHEST PRIORITY)
**File:** `11_gray_scott.kairo`
**Tagline:** "Complex patterns from simple rules"
**Visual:** Stunning organic patterns (spots, stripes, spirals)
**Lines of Code:** ~50

**Demonstrates:**
- Two coupled PDEs
- Laplacian operator
- Parameter-driven behavior
- Emergent complexity

**Why Important:** This is a visually stunning example that will attract creative coders and scientists. Gray-Scott patterns are mesmerizing.

**Implementation Sketch:**
```kairo
use field, visual

@state u : Field2D<f32> = ones((256, 256))
@state v : Field2D<f32> = zeros((256, 256))

# Initialize with small random perturbation in center
# (need stdlib function for region initialization)

const Du : f32 = 0.16
const Dv : f32 = 0.08
const F : f32 = 0.060  # Feed rate
const K : f32 = 0.062  # Kill rate

flow(dt=1.0, steps=10000) {
    # Gray-Scott reaction: U + 2V → 3V, V → P
    uvv = u * v * v

    du_dt = Du * laplacian(u) - uvv + F * (1.0 - u)
    dv_dt = Dv * laplacian(v) + uvv - (F + K) * v

    u = u + du_dt * dt
    v = v + dv_dt * dt

    output colorize(v, palette="viridis", min=0.0, max=1.0)
}
```

---

### 2.3 Smoke Simulation
**File:** `12_smoke_simulation.kairo`
**Tagline:** "Classic fluid simulation - smoke rising"
**Visual:** Realistic smoke plume with buoyancy
**Lines of Code:** ~60

**Demonstrates:**
- Advection (semi-Lagrangian)
- Diffusion
- Buoyancy forces
- Divergence-free projection
- Multi-field coupling

---

### 2.4 Fluid Flow (Navier-Stokes)
**File:** `13_fluid_flow.kairo`
**Tagline:** "Full incompressible fluid simulation"
**Visual:** Swirling fluid with vortices
**Lines of Code:** ~70

**Demonstrates:**
- Vector field operations
- Pressure projection
- Viscosity
- Complex PDE system

---

### 2.5 Turing Patterns
**File:** `14_turing_patterns.kairo`
**Tagline:** "How leopards get their spots"
**Visual:** Animal coat patterns emerging
**Lines of Code:** ~45

**Demonstrates:**
- Reaction-diffusion variant
- Pattern formation
- Biological modeling
- Parameter exploration

---

### 2.6 Spring Network
**File:** `15_spring_network.kairo`
**Tagline:** "Interconnected springs with lambdas"
**Visual:** Bouncing spring mesh
**Lines of Code:** ~80

**Demonstrates:**
- Struct arrays (particles)
- Lambda-based force calculations
- Higher-order functions
- Physics integration

---

### 2.7 Perlin Noise Flow Field
**File:** `16_flow_field.kairo`
**Tagline:** "Particles following noise-based vector field"
**Visual:** Organic flowing particle trails
**Lines of Code:** ~55

**Demonstrates:**
- Procedural generation
- Field sampling
- Long-term visualization
- Artistic applications

---

### 2.8 Oscillator Grid
**File:** `17_oscillator_grid.kairo`
**Tagline:** "Coupled oscillators create waves"
**Visual:** Grid of oscillators with wave propagation
**Lines of Code:** ~50

**Demonstrates:**
- Nearest-neighbor coupling
- Synchronization phenomena
- Phase dynamics
- Complex systems

---

## 🔴 TIER 3: Advanced Examples (5)
**Goal:** Production-quality demonstrations of Kairo's power

### 3.1 Kelvin-Helmholtz Instability (⭐ HIGH PRIORITY)
**File:** `20_kelvin_helmholtz.kairo`
**Tagline:** "Beautiful fluid instability phenomenon"
**Visual:** Characteristic vortex rollup patterns
**Lines of Code:** ~90

**Demonstrates:**
- Advanced fluid dynamics
- Instability physics
- High-resolution simulation
- Publication-quality results

**Why Important:** This is a famous physics phenomenon that looks amazing and demonstrates serious computational capability.

---

### 3.2 Multi-Scale Turbulence
**File:** `21_turbulence.kairo`
**Tagline:** "Cascading energy across scales"
**Visual:** Turbulent flow field
**Lines of Code:** ~100

**Demonstrates:**
- High Reynolds number flow
- Energy cascade
- Vortex dynamics
- Performance at scale

---

### 3.3 Cahn-Hilliard Phase Separation
**File:** `22_phase_separation.kairo`
**Tagline:** "Oil and water don't mix - watch why"
**Visual:** Spinodal decomposition patterns
**Lines of Code:** ~70

**Demonstrates:**
- 4th-order PDE (biharmonic)
- Phase field methods
- Materials science
- Long-time dynamics

---

### 3.4 Mandelbrot Zoom Animation
**File:** `23_mandelbrot_zoom.kairo`
**Tagline:** "Journey into infinite complexity"
**Visual:** Animated zoom into Mandelbrot set
**Lines of Code:** ~60

**Demonstrates:**
- Complex number operations
- Iterative algorithms
- Coordinate transformations
- Mathematical beauty

---

### 3.5 Double Pendulum Chaos
**File:** `24_double_pendulum.kairo`
**Tagline:** "Deterministic chaos in action"
**Visual:** Chaotic pendulum with trajectory trails
**Lines of Code:** ~80

**Demonstrates:**
- Coupled ODEs
- Chaotic dynamics
- Struct-based state
- Deterministic sensitivity

---

## 🌟 TIER 4: Flagship Examples (2)
**Goal:** Mind-blowing demonstrations for marketing/demos

### 4.1 Interactive Fluid Painting (⭐ SHOWCASE)
**File:** `30_fluid_painting.kairo`
**Tagline:** "Paint with fluid dynamics"
**Visual:** Interactive fluid simulation with mouse input
**Lines of Code:** ~120

**Demonstrates:**
- Full Navier-Stokes
- User interaction
- Multiple coupled fields (velocity, density, color)
- Real-time performance
- Artistic applications

**Why Important:** This is a demo that sells the language. People love interactive fluid sims.

---

### 4.2 Multi-Domain Ecosystem (⭐ SHOWCASE)
**File:** `31_ecosystem.kairo`
**Tagline:** "Fields, agents, and emergent behavior"
**Visual:** Agents navigating fields with complex interactions
**Lines of Code:** ~150

**Demonstrates:**
- Field + Agent coupling (when agents implemented)
- Multiple domains
- Emergent complexity
- Production-scale simulation

**Note:** Requires agent dialect, save for later milestone

---

## Implementation Priority

### Phase 1: Quick Wins (1 week)
**Goal:** Get 5-7 strong examples immediately
1. 🔥 **01_hello_heat.kairo** (beginner intro)
2. 🔥 **11_gray_scott.kairo** (visual wow factor)
3. 🔥 **03_wave_ripples.kairo** (physics + pretty)
4. **10_heat_equation.kairo** (complete physics)
5. **02_pulsing_circle.kairo** (simple animation)
6. **12_smoke_simulation.kairo** (classic demo)
7. **20_kelvin_helmholtz.kairo** (advanced physics)

**Why These?**
- Mix of complexity levels
- All visually compelling
- Cover different domains
- Demonstrate core Kairo features (fields + visuals)

---

### Phase 2: Portfolio Expansion (2 weeks)
Add 5-8 more examples covering:
- Turing patterns
- Fluid flow
- Spring networks
- Random walks
- Gradient flow
- Oscillator grid
- Perlin flow field

---

### Phase 3: Advanced Showcase (3-4 weeks)
Complete the portfolio with:
- Advanced physics examples
- Flagship interactive demos
- Multi-domain examples (when agents ready)

---

## Example Template Structure

Each example should follow this structure:

```kairo
# [Number]_[name].kairo
# Kairo Example: [Full Title]
#
# Description: [What this demonstrates in 1-2 sentences]
#
# Demonstrates:
#   - Feature 1
#   - Feature 2
#   - Feature 3
#
# Expected Output: [What user should see]
#
# Parameters you can experiment with:
#   - [param1]: [what it does]
#   - [param2]: [what it does]
#
# Author: Kairo Team
# Date: 2025-11-07

use field, visual  # or other dialects

# Constants with clear physical meaning
const PARAM1 : f32 = 0.1

# State variables with initialization
@state field1 : Field2D<f32> = zeros((256, 256))

# Main simulation
flow(dt=0.01, steps=1000) {
    # Clear, commented logic

    # Visual output
    output colorize(field1, palette="viridis")
}
```

---

## Visual Output Strategy

Each example should support:

1. **Static output** - PNG snapshots
2. **Animation** - Frame sequences or video
3. **Interactive** - Real-time window (when supported)

**Palette Guide:**
- **fire**: Heat, energy, intensity
- **viridis**: General-purpose, perceptually uniform
- **coolwarm**: Diverging data (negative/positive)
- **plasma**: High-contrast patterns
- **grayscale**: Publication figures

---

## Documentation Per Example

Each example needs:

1. **README entry** - One-line description
2. **Code comments** - What's happening and why
3. **Parameter guide** - What to experiment with
4. **Expected output** - Image showing result
5. **Variations** - Suggested modifications

---

## Testing Checklist

For each example verify:
- ✅ Runs without errors
- ✅ Produces expected output
- ✅ Completes in reasonable time (<2 minutes)
- ✅ Deterministic (same seed = same result)
- ✅ Comments are clear
- ✅ Parameters are well-chosen
- ✅ Visual output is compelling

---

## Success Metrics

**Example Portfolio Success = When:**
1. New users can run first example in <5 minutes
2. Each complexity tier has clear progression
3. Visual outputs are "shareable" (social media worthy)
4. Examples cover all major Kairo features
5. Users say "I want to build something like that!"

---

## Future Example Ideas (Backlog)

### When Agent Dialect is Ready:
- Boids flocking
- Predator-prey dynamics
- Particle life
- Physarum slime mold
- Swarm intelligence
- N-body gravity

### When Signal Dialect is Ready:
- FM synthesis
- Audio-reactive particles
- Spectrum analyzer
- Wave interference (audio)
- Granular synthesis

### Cross-Domain:
- Audio-reactive fluid
- Agent-field coupling
- Sound-driven visuals
- Multi-modal synthesis

### When FluidDynamics & Acoustics Domains are Ready:
- **2-Stroke Engine & Muffler Acoustics** (⭐ FLAGSHIP MULTI-DOMAIN SHOWCASE)
  - Complete exhaust system simulation
  - Fluid dynamics → Acoustics → Audio pipeline
  - Geometry-driven acoustic behavior
  - Realistic engine sound synthesis
  - Backpressure timing optimization
  - See: `docs/USE_CASES/2-stroke-muffler-modeling.md`
- Pipe organ modeling
- Brass/woodwind instrument design
- Architectural room acoustics
- Speaker enclosure design
- HVAC duct acoustics

---

## Next Steps

1. **Review with maintainer** - Get feedback on priorities
2. **Start with Phase 1** - Implement top 5-7 examples
3. **Create example gallery** - Visual showcase on README
4. **Video demos** - Screen capture of each running
5. **User testing** - Watch someone try examples cold

---

**Priority Actions:**
1. 🔥 Implement `01_hello_heat.kairo` (THE starter example)
2. 🔥 Implement `11_gray_scott.kairo` (THE visual showcase)
3. 🔥 Implement `03_wave_ripples.kairo` (physics + beauty)
4. Create example gallery in README with images
5. Video recording of examples running

---

**Questions to Resolve:**
1. Do we have stdlib functions for region initialization? (setting center pixel, rectangles, etc.)
2. Is `random_normal()` for fields implemented?
3. Does `colorize()` support all palettes mentioned?
4. Can we save PNG output from `output()`?
5. Is there a way to set individual field values or regions?

---

**End of Plan - Ready for Implementation** ✨
