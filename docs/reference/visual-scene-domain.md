# Visual & Scene Domain Architecture

## Overview

The Visual/Scene domain provides Morphogen with **composable math-visualization capabilities** inspired by 3Blue1Brown's Manim library, but built natively into Morphogen's multi-domain architecture. This creates a powerful system for explanatory graphics, mathematical animations, and interactive visualizations.

> **Core Concept**: A composable 2D/3D visual DSL with scene graph + animation/timeline engine that plays nicely with math.

## What Makes This Special

Unlike traditional visualization tools, Morphogen's approach integrates:

1. **Scene graph architecture** - Composable visual objects (curves, text, surfaces, arrows)
2. **Parametric everything** - Functions, transforms, and paths as first-class primitives
3. **Timeline system** - Time as a domain, with easing and sequencing
4. **Camera system** - Pans, zooms, rotations, tracking
5. **Multi-domain integration** - Direct connection to simulation, noise, palette, and field domains

### Why Morphogen > Manim for Scientific Computing

**Direct Simulation Integration**
- Visualize differential equations from Morphogen's ODE solver
- Show emergent systems (CA, agents) with live field updates
- Animate optimization surfaces (GA/PSO results)
- Display J-tube simulations (pressure, flow, velocity)

**Shared Infrastructure**
- Plasma, Mandelbrot, CA, reaction-diffusion as visual sources
- Vector fields, fractals, noise - all GPU-accelerated
- Unified palette and color management

**Performance**
- MLIR → CPU/GPU compilation for all visual operations
- High-resolution (4K+) rendering of complex scenes
- Real-time preview for iterative development

---

## Architecture

### Domain Hierarchy

```
VisualDomain (top-level)
├── SceneDomain        - Scene graph, objects, camera
├── TimelineDomain     - Animation, keyframes, easing
├── GeometryDomain     - 2D/3D primitives, curves, surfaces
├── PaletteDomain      - Color mapping, gradients (existing)
├── NoiseDomain        - Procedural patterns (existing)
├── FieldDomain        - Scalar/vector fields (existing)
└── VideoDomain        - Encoding, composition (existing)
```

### Data Flow

```
Scene Definition
    ↓
Scene Graph (MObjects)
    ↓
Timeline (Animations)
    ↓
Time-Parametrized Transforms
    ↓
Per-Frame Rendering
    ↓
Video Encoding
```

---

## SceneDomain

### Core Types

**Scene**
- Container for all visual objects and animations
- Manages camera, background, and rendering settings
- Tracks timeline state

**MObject (Base Visual Object)**
- Position, rotation, scale transforms
- Color, opacity, stroke properties
- Parent-child relationships for hierarchical transforms

**Specific MObject Types**
```
Geometric:
- Curve2D, Curve3D       - Parametric curves
- Surface, Mesh          - 3D surfaces and meshes
- PointCloud             - Point-based visualization
- Arrow, Line, Ray       - Directional indicators
- Dot, Circle, Rectangle - Basic shapes

Mathematical:
- Axes, Grid             - Coordinate systems
- NumberPlane            - 2D coordinate plane
- VectorField            - Vector field visualization
- ParametricCurve        - f(t) curves
- ParametricSurface      - f(u,v) surfaces

Text & Annotation:
- Text                   - Basic text objects
- Equation               - LaTeX/MathML equations
- Label                  - Object annotations
```

### Core Operators

**Scene Management**
```julia
scene.create(name, objects=[])
scene.add(object)
scene.remove(object)
scene.set_background(color | gradient | field)
scene.set_camera(position, target, fov)
scene.render(t)  # Render at time t
```

**Geometry Creation**
```julia
geo.curve(f(t), t_range)
geo.parametric_surface(f(u,v), u_range, v_range)
geo.vector_field(F(x,y), bounds)
geo.axes(x_range, y_range, [z_range])
geo.grid(spacing, bounds)
geo.arrow(start, end)
geo.dot(position, radius)
```

**Object Properties**
```julia
obj.set_position(x, y, [z])
obj.set_rotation(angle | quaternion)
obj.set_scale(factor | [x, y, z])
obj.set_color(color)
obj.set_opacity(alpha)
obj.set_stroke(width, color)
obj.set_fill(color)
```

---

## TimelineDomain

### Core Concept

Time is treated as a domain parameter. Any numeric property (position, opacity, color, control points) becomes a function of `t`.

### Types

**Animation**
- Base type for all animations
- Has: target object, property, duration, easing

**Keyframe**
- Time + value pairs for property animation
- Interpolation method (linear, bezier, custom)

**Easing**
- Built-in functions: linear, ease_in, ease_out, ease_in_out
- Custom easing functions: f(t) → [0,1]

**Timeline**
- Container for animations
- Supports sequencing and parallelism

### Core Operators

**Basic Animations**
```julia
anim.move(object, to=position, duration, easing)
anim.rotate(object, angle, duration, easing)
anim.scale(object, factor, duration, easing)
anim.fade_in(object, duration)
anim.fade_out(object, duration)
anim.highlight(object, color, duration)
```

**Transform Animations**
```julia
anim.morph(objectA, to=objectB, duration)
anim.grow(object, duration)           # Scale from 0
anim.shrink(object, duration)         # Scale to 0
anim.transform(object, matrix, duration)
```

**Math-Specific Animations**
```julia
anim.write(text_object, duration)            # Text reveal
anim.write_equation(equation, duration)      # Equation reveal
anim.transform_equation(eq1, eq2, duration)  # Equation morph
anim.show_substitution(eq, var, value)       # Variable replacement
anim.draw_curve(curve, duration)             # Curve drawing
```

**Timeline Composition**
```julia
timeline.sequence([anim1, anim2, anim3])  # Sequential
timeline.parallel([anim1, anim2, anim3])  # Parallel
timeline.stagger([anims], delay)          # Staggered start
timeline.loop(anim, count | Inf)          # Repeat
```

**Camera Animations**
```julia
camera.pan(to=point, duration, easing)
camera.zoom(factor, duration, easing)
camera.orbit(angle, duration, easing)
camera.follow(object, [offset])           # Track object
```

---

## Integration with Existing Domains

### PaletteDomain

Colors and gradients for visual objects:

```julia
# Static coloring
obj.set_color(palette.get("inferno", 0.5))

# Animated color mapping
anim.color_map(
    surface,
    field=scalar_field,
    palette="viridis",
    duration=5.0
)

# Palette cycling
anim.cycle_palette(
    object,
    palette="rainbow",
    speed=1.0
)
```

### NoiseDomain

Procedural motion and backgrounds:

```julia
# Animated background
scene.background(
    noise.fbm2d(x, y, t, octaves=4)
)

# Noise-driven motion
anim.perturb(
    object,
    noise=noise.simplex3d,
    amplitude=0.1,
    frequency=2.0
)

# Procedural patterns
texture = noise.cellular2d(x, y)
surface.set_texture(texture)
```

### FieldDomain

Scalar and vector field visualization:

```julia
# Vector field display
vfield = field.vector(F(x,y))
scene.add(geo.vector_field(vfield))

# Animate field evolution
anim.evolve_field(vfield, duration=10.0)

# Integral curves
curve = field.integral_curve(vfield, start_point)
anim.draw_curve(curve, duration=2.0)

# Field-based coloring
surface.color_by_field(
    field=field.gradient(scalar),
    palette="coolwarm"
)
```

### VideoDomain

Rendering to video:

```julia
# Scene to video
video = video.from_scene(
    scene,
    fps=60,
    duration=10.0,
    resolution=(1920, 1080)
)

# Add audio
video.add_audio(track, sync=true)

# Encode
video.encode(
    codec="h264",
    crf=18,
    preset="slow"
)

# Compositing
final = video.composite([
    layer1,
    layer2.with_blend("overlay")
])
```

### EmergenceDomain

Live simulation visualization:

```julia
# Cellular automata
ca = emergence.cellular_automaton(rules)
anim.evolve_ca(ca, steps=100, fps=30)

# Agent systems
agents = emergence.agent_system(behavior)
anim.show_agents(agents, duration=10.0)

# Reaction-diffusion
rd = emergence.reaction_diffusion(params)
surface.texture_from_field(rd.concentration)
anim.evolve(rd, duration=20.0)
```

---

## Example Use Cases

### 1. Vector Field Visualization

```julia
# Define vector field
F(x, y) = [-y, x]  # Rotation field

# Create scene
scene = scene.create("vector_field")
vfield = geo.vector_field(F, bounds=(-5,5,-5,5))
scene.add(vfield)

# Add integral curves
for start in sample_points:
    curve = field.integral_curve(F, start)
    scene.add(curve)
    anim.draw_curve(curve, duration=2.0)

# Render
video = video.from_scene(scene, duration=5.0, fps=60)
```

### 2. Differential Equation Animation

```julia
# Solve ODE
solution = ode.solve(f, u0, tspan)

# Visualize solution curve
scene = scene.create("ode_solution")
axes = geo.axes((-1,10), (-2,2))
curve = geo.curve(t -> solution(t), (0,10))

scene.add(axes)
scene.add(curve)

# Animate drawing
anim.draw_curve(curve, duration=5.0)

# Show trajectory point
dot = geo.dot(solution(0), radius=0.1)
anim.move_along_curve(dot, curve, duration=5.0)
```

### 3. Mathematical Explanation

```julia
# Start with equation
eq1 = equation("∫₀¹ x² dx")
scene.add(eq1)
anim.write_equation(eq1, duration=1.0)

# Show area visualization
axes = geo.axes((0,1), (0,1))
curve = geo.curve(x -> x^2, (0,1))
area = geo.region_under_curve(curve)

scene.add(axes)
scene.add(curve)
anim.sequence([
    anim.fade_in(axes, 0.5),
    anim.draw_curve(curve, 1.0),
    anim.fill_region(area, color="blue", opacity=0.3, duration=1.0)
])

# Show result
eq2 = equation("= ⅓")
anim.transform_equation(eq1, eq2, duration=1.0)
```

### 4. Fractal Exploration

```julia
# Mandelbrot set
mandelbrot = noise.mandelbrot(
    bounds=(-2,1,-1.5,1.5),
    iterations=100
)

surface = geo.surface_from_field(mandelbrot)
scene.add(surface)

# Color by iteration count
surface.color_by_field(
    mandelbrot,
    palette="inferno"
)

# Zoom animation
camera.zoom(100, duration=10.0, easing="ease_in_out")
camera.pan(to=(-0.5, 0), duration=10.0)
```

---

## Implementation Roadmap

### Phase 1: Math/Scene Foundation (MVP)

**Goal**: Basic scene graph and rendering

**Components**:
- SceneDomain type system
- MObject base types: Axes, Curve2D, Dot, Arrow, Text
- Simple transform system (position, rotation, scale)
- Basic TimelineDomain with linear interpolation
- CPU renderer (matplotlib or cairo backend)

**Deliverables**:
- Can create scenes with basic geometry
- Can animate simple properties
- Can render to PNG sequence

**Estimate**: 2-3 weeks

---

### Phase 2: Palette + Noise + Field Integration

**Goal**: Rich visual effects and procedural content

**Components**:
- PaletteDomain integration (colormaps)
- NoiseDomain integration (Perlin, simplex, FBM)
- FieldDomain integration (scalar/vector fields)
- Field-based rendering (contours, streamlines)
- Texture mapping from fields

**Deliverables**:
- Can color objects with palettes
- Can generate procedural backgrounds
- Can visualize vector fields
- Can animate field evolution

**Estimate**: 2 weeks

---

### Phase 3: Video + Audio Production

**Goal**: Complete video pipeline

**Components**:
- VideoDomain: scene → frames → encode
- Frame buffer management
- Video codec integration (ffmpeg)
- AudioDomain integration
- Audio-visual synchronization
- Basic compositing (layers, blending)

**Deliverables**:
- Can render scenes to MP4/MOV
- Can add audio tracks
- Can composite multiple layers
- Can export high-quality videos

**Estimate**: 2 weeks

---

### Phase 4: Advanced Animation & Math Sugar

**Goal**: 3Blue1Brown-style ergonomics

**Components**:
- Equation objects (LaTeX integration)
- Math-specific animations:
  - `write_equation()`
  - `transform_equation()`
  - `show_substitution()`
- Advanced easing functions
- Curve morphing
- Smart object alignment
- Animation templates

**Deliverables**:
- Can create mathematical explanations
- Can morph between equations
- Can create polished animations quickly

**Estimate**: 2-3 weeks

---

### Phase 5: Performance & GPU Acceleration

**Goal**: Real-time preview and 4K rendering

**Components**:
- GPU-accelerated rendering (via MLIR)
- Shader-based effects
- Real-time preview mode
- Parallel frame rendering
- Memory optimization for large scenes

**Deliverables**:
- Interactive scene editing
- Fast iteration cycles
- 4K rendering at reasonable speeds

**Estimate**: 3-4 weeks

---

### Phase 6: Advanced 3D & Simulation

**Goal**: Full 3D capabilities and live simulation

**Components**:
- 3D primitives (Surface, Mesh, PointCloud)
- Lighting and shading models
- 3D camera controls (orbit, pan, zoom)
- Direct emergence domain hooks:
  - Live CA visualization
  - Agent system rendering
  - Reaction-diffusion display
- Simulation state → visual state binding

**Deliverables**:
- Can create 3D mathematical visualizations
- Can show live simulations
- Can render complex 3D scenes

**Estimate**: 4 weeks

---

## Total Estimated Timeline

**Minimum Viable Product (Phases 1-3)**: 6-7 weeks
**Full Feature Set (Phases 1-6)**: 15-18 weeks

---

## Technical Considerations

### Rendering Backend

**Phase 1-2**: CPU-based (matplotlib, cairo)
- Easier to implement
- Good for prototyping
- Sufficient for basic scenes

**Phase 5+**: GPU-based (MLIR → Vulkan/Metal)
- Required for real-time preview
- Necessary for 4K rendering
- Enables shader effects

### Scene Graph Structure

**Hierarchical Transforms**:
- Parent transforms affect children
- Allows complex object grouping
- Simplifies camera operations

**Spatial Indexing**:
- Quad/octree for large scenes
- Frustum culling for performance
- Necessary for complex visualizations

### Animation System

**Property Interpolation**:
- Support for all numeric types
- Custom interpolators for special types (colors, quaternions)
- Keyframe optimization

**Timeline Management**:
- Efficient seeking
- State caching for fast preview
- Dependency tracking for property changes

---

## Design Principles

### 1. Composability First

Every visual element should compose naturally:
```julia
# Objects compose
scene.add([axes, curve, labels])

# Animations compose
timeline.parallel([anim1, anim2])

# Transforms compose
obj.transform = translate * rotate * scale
```

### 2. Time as a Domain

Time is not special - it's just another parameter:
```julia
# Position as function of time
obj.position = f(t)

# Color as function of time and space
obj.color = f(x, y, t)

# Any property can be time-dependent
obj.opacity = ease_in_out(t)
```

### 3. Math-First API

Embrace mathematical notation:
```julia
# Functions are first-class
f(x) = x^2
curve = geo.curve(f, (0,10))

# Fields are natural
F(x,y) = [cos(x), sin(y)]
vfield = geo.vector_field(F)

# Parametric definitions
surface = geo.parametric_surface(
    (u,v) -> [u*cos(v), u*sin(v), u],
    u=(0,2), v=(0,2π)
)
```

### 4. Multi-Domain Integration

Visual domain should leverage all other domains:
```julia
# Noise → texture
texture = noise.fbm2d(x, y)

# Field → color
color = palette.map(field.magnitude(F(x,y)))

# Simulation → animation
state = emergence.step(ca)
visual.update_from_state(state)
```

---

## Comparison to Manim

| Feature | Manim | Morphogen Visual |
|---------|-------|--------------|
| **Core Language** | Python | Julia / Domain IR |
| **Architecture** | Monolithic | Multi-domain, composable |
| **Rendering** | Cairo (CPU) | CPU → GPU (MLIR) |
| **Simulation** | External | Native (ODE, CA, agents) |
| **Fields** | Manual | FieldDomain integration |
| **Noise** | External libs | NoiseDomain (GPU) |
| **Performance** | Single-threaded | Parallel, GPU-accelerated |
| **Extension** | Python plugins | Domain operators |
| **Type System** | Dynamic | Static + inference |
| **Timeline** | Imperative | Declarative + imperative |

### What Morphogen Does Better

1. **Unified computation + visualization**
   - Simulate and visualize in one graph
   - No data transfer overhead

2. **Performance**
   - GPU acceleration throughout
   - Parallel rendering
   - MLIR optimization

3. **Composability**
   - Everything is an operator
   - Cross-domain reuse
   - Type safety

4. **Scientific computing**
   - Native ODE/PDE support
   - Field operations
   - Emergence simulations

### What Manim Does Better (Currently)

1. **Maturity**
   - Years of development
   - Large user community
   - Extensive examples

2. **Ease of use**
   - Python simplicity
   - Well-documented
   - Lots of tutorials

3. **Features**
   - More built-in objects
   - More animation presets
   - LaTeX integration

**Morphogen's Goal**: Match Manim's ergonomics while exceeding its capabilities for scientific computing.

---

## Next Steps

1. **Start with Phase 1** - Get basic scene graph working
2. **Create example gallery** - Show what's possible
3. **Gather feedback** - From potential users
4. **Iterate on API** - Before locking in design
5. **Build incrementally** - Each phase should be usable

---

## References

- 3Blue1Brown: https://www.3blue1brown.com/
- Manim: https://github.com/3b1brown/manim
- Manim Community: https://www.manim.community/
- Grant Sanderson's videos on animation: https://www.youtube.com/c/3blue1brown

---

## Related Documentation

- [EmergenceDomain](./emergence-domain.md) - For simulation integration
- [PaletteDomain](./palette-domain.md) - For color management
- [NoiseDomain](./noise-domain.md) - For procedural content
- [FieldDomain](./field-domain.md) - For field operations
- [VideoDomain](./video-domain.md) - For encoding/composition

---

*This document represents the architectural vision for Morphogen's visual/scene capabilities. Implementation details may evolve as the system develops.*
