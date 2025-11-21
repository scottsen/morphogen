# Kairo Showcase Examples 🎨

**High-impact cross-domain demonstrations showing the power of Kairo**

These examples go beyond simple feature demonstrations to showcase how Kairo's domains work together to solve real-world problems and create stunning visualizations. Each example integrates multiple domains to demonstrate the full power of the Kairo ecosystem.

---

## 🎯 Overview

The showcase examples demonstrate **cross-domain integration** - the key value proposition of Kairo. Rather than using each domain in isolation, these examples show how combining domains creates capabilities greater than the sum of their parts.

### What Makes These Examples Special?

- **Portfolio Quality**: Publication-ready visualizations and demonstrations
- **Real-World Applications**: Practical use cases across multiple domains
- **Best Practices**: Demonstrates optimal patterns for combining domains
- **Performance**: Shows how to efficiently use Kairo for complex tasks
- **Educational**: Teaches both the "how" and the "why"

---

## 📚 Examples

### 1. Fractal Explorer 🌀

**File**: `01_fractal_explorer.py`

**Demonstrates**: Noise + Palette + Color + Image + Field

**What it does**: Creates stunning visualizations of Mandelbrot and Julia sets using advanced coloring techniques that go far beyond traditional fractal renderers.

**Domains integrated**:
- **Field**: Gradient computation for orbit trap effects
- **Noise**: Texture overlays and FBM modulation
- **Palette**: Sophisticated color gradients and scientific colormaps
- **Color**: Advanced blend modes (overlay, screen)
- **Image**: Composition, filtering, and sharpening

**Key features**:
- Smooth iteration coloring (eliminates banding)
- Noise texture overlays for organic appearance
- Multi-palette composition with blend modes
- Deep zoom visualization with dynamic palettes
- Orbit trap coloring techniques
- FBM-based organic coloring
- Animated palette cycling

**Run it**:
```bash
python examples/showcase/01_fractal_explorer.py
```

**Output**: 6 different fractal visualizations demonstrating various techniques

**Best for**: Artists, mathematicians, anyone interested in algorithmic art

---

### 2. Physics Visualizer ⚡

**File**: `02_physics_visualizer.py`

**Demonstrates**: Integrators + Field + Palette + Image

**What it does**: Simulates and visualizes physical systems with publication-quality scientific visualization.

**Domains integrated**:
- **Integrators**: Time evolution using RK4 integration
- **Field**: Spatial derivatives (Laplacian, gradient)
- **Palette**: Physical quantity visualization (temperature, pressure, etc.)
- **Image**: Rendering temporal evolution

**Systems simulated**:
1. **Heat Diffusion**: Multiple sources and sinks with diffusion
2. **Wave Interference**: 2D wave equation with multiple sources
3. **Reaction-Diffusion**: Gray-Scott pattern formation (spots, stripes)
4. **Coupled Oscillators**: Network of harmonic oscillators
5. **N-Body Gravity**: Gravitational dynamics visualization

**Run it**:
```bash
python examples/showcase/02_physics_visualizer.py
```

**Output**: Animated sequences showing temporal evolution of physical systems

**Best for**: Physicists, engineers, computational scientists, educators

---

### 3. Procedural Art Generator 🎨

**File**: `03_procedural_art.py`

**Demonstrates**: Noise + Image + Color + Palette + Field

**What it does**: Creates stunning generative art using procedural techniques and multi-layer composition.

**Domains integrated**:
- **Noise**: Perlin, Worley, FBM, marble, turbulence
- **Image**: Layer composition, blend modes, filters
- **Color**: Sophisticated color schemes and gradients
- **Palette**: Gradient generation (linear, cosine)
- **Field**: Mathematical transformations

**Art styles created**:
1. **Organic Abstract**: Layered noise compositions
2. **Geometric Noise**: Math + noise hybrid patterns
3. **Flow Field Art**: Vector field visualizations
4. **Layered Composition**: Complex multi-layer artwork
5. **Glitch Art**: Distortion and scanline effects
6. **Gradient Exploration**: Procedural color palettes
7. **Abstract Terrain**: Non-photorealistic landscapes

**Run it**:
```bash
python examples/showcase/03_procedural_art.py
```

**Output**: 7 unique generative art pieces

**Best for**: Digital artists, creative coders, game developers

---

### 4. Scientific Visualization Suite 🔬

**File**: `04_scientific_visualization.py`

**Demonstrates**: Sparse Linear Algebra + Field + Palette + Image + I/O

**What it does**: Solves partial differential equations and creates publication-quality scientific visualizations.

**Domains integrated**:
- **Sparse Linear Algebra**: PDE discretization and solving
- **Field**: Gradient, Laplacian, and field analysis
- **Palette**: Scientific colormaps (diverging, sequential)
- **Image**: Publication-quality rendering
- **I/O**: Simulation checkpointing (conceptual)

**PDEs solved**:
1. **Poisson Equation**: Electrostatic potential fields
2. **Laplace Equation**: Steady-state heat distribution
3. **Helmholtz Equation**: Acoustic resonance
4. **Eigenvalue Problem**: Vibration modes
5. **Time-Dependent Heat**: With checkpointing
6. **PDE Comparison**: Side-by-side solver comparison

**Numerical methods**:
- Finite difference discretization
- Sparse matrix assembly
- Conjugate gradient solver
- Implicit time integration
- Eigenvalue computation

**Run it**:
```bash
python examples/showcase/04_scientific_visualization.py
```

**Output**: Multiple PDE solutions with scientific visualizations

**Best for**: Scientists, engineers, numerical analysts, researchers

---

### 5. Audio Visualizer 🎵

**File**: `05_audio_visualizer.py`

**Demonstrates**: Audio + Field + Cellular + Palette + Visual

**What it does**: Creates stunning audio-reactive visualizations combining synthesis, spectral analysis, and multi-domain integration.

**Domains integrated**:
- **Audio**: Synthesis and processing
- **Field**: Audio-reactive diffusion effects
- **Cellular**: Audio-driven cellular automata
- **Palette**: Color mapping for visualizations
- **Visual**: Video output with embedded audio

**Visualizations created**:
1. **Spectrum Analyzer**: FFT-based frequency analysis
2. **Waveform Display**: Temporal audio visualization
3. **Audio-Reactive CA**: Cellular automata driven by audio amplitude
4. **Beat-Synchronized Patterns**: Energy-based pattern generation
5. **Field Diffusion**: Audio spectrum driving heat diffusion
6. **Video Export**: MP4/GIF with synchronized audio

**Run it**:
```bash
python examples/showcase/05_audio_visualizer.py
```

**Output**: Multiple visualizations and video files with embedded audio

**Best for**: Audio engineers, VJ artists, creative coders, educators

---

### 6. Interactive Physics Sandbox ⚛️

**File**: `06_physics_sandbox.py`

**Demonstrates**: RigidBody + Field + Genetic + Cellular + Visual

**What it does**: Creates interactive physics simulations with emergent behavior through multi-domain integration.

**Domains integrated**:
- **RigidBody**: Particle collisions and dynamics (simplified implementation)
- **Field**: Gravity wells, force fields, and diffusion
- **Genetic**: Evolutionary optimization of physics parameters
- **Cellular**: Emergent structures from physics interactions
- **Palette/Visual**: Beautiful real-time visualizations

**Simulations created**:
1. **Particle Collisions**: N-body collision sandbox with realistic physics
2. **Gravity Fields**: Particles in multi-attractor gravity systems
3. **Emergent Structures**: Physics + cellular automata creating patterns
4. **Genetic Optimization**: Evolving optimal particle configurations

**Physics features**:
- Collision detection and response
- Force field integration
- Friction and damping
- Energy conservation
- Emergent clustering behavior

**Run it**:
```bash
python examples/showcase/06_physics_sandbox.py
```

**Output**: Multiple physics simulation frames showing collision, gravity, and emergent behavior

**Best for**: Game developers, physicists, educators, creative coders

---

### 7. Physical Modeling Instrument 🎸

**File**: `07_physical_instrument.py`

**Demonstrates**: Audio + Acoustics + Field + Signal + Visual

**What it does**: Synthesizes musical instruments using physical models that simulate real-world vibrating systems.

**Domains integrated**:
- **Audio**: Sound synthesis and export
- **Acoustics**: Waveguide models and resonance
- **Field**: Vibration visualization
- **Signal**: Spectral analysis
- **Visual**: String vibration and mode visualization

**Instruments modeled**:
1. **String Instruments**: Karplus-Strong plucked strings (guitar, violin)
2. **Drums**: Modal synthesis with inharmonic overtones
3. **Bells**: Complex modal patterns
4. **Multi-String**: Chord synthesis
5. **Rhythmic Patterns**: Drum sequencing

**Physical modeling techniques**:
- Karplus-Strong algorithm
- Modal synthesis (sum of damped sine waves)
- Waveguide synthesis
- Spectral analysis and visualization
- Mode shape visualization

**Run it**:
```bash
python examples/showcase/07_physical_instrument.py
```

**Output**: WAV audio files and spectrograms showing synthesized instruments

**Best for**: Audio engineers, instrument designers, musicians, signal processing enthusiasts

---

### 8. Digital Twin 🏭

**File**: `08_digital_twin.py`

**Demonstrates**: Field + Integrators + SparseLinalg + Visual + I/O

**What it does**: Creates digital twins - virtual representations of physical systems that simulate real-world behavior for monitoring, optimization, and control.

**Domains integrated**:
- **Field**: Thermal dynamics and fluid flow
- **Integrators**: Time evolution of physical systems
- **Sparse Linear Algebra**: PDE solving for complex physics
- **Visual**: Real-time monitoring visualizations
- **I/O**: Data logging and checkpointing

**Digital twins created**:
1. **Thermal Manufacturing**: Heat treatment process with quality metrics
2. **Heat Exchanger**: Counter-flow heat exchanger with efficiency tracking
3. **Active Cooling System**: Electronics cooling with feedback control
4. **Multi-Physics**: Thermal-structural coupling (expansion and stress)

**Industrial applications**:
- Process optimization and monitoring
- Predictive maintenance
- Quality control
- System design validation
- Real-time performance tracking

**Run it**:
```bash
python examples/showcase/08_digital_twin.py
```

**Output**: Thermal field visualizations and metrics for industrial processes

**Best for**: Engineers, industrial designers, manufacturing professionals, systems engineers

---

## 🎓 Learning Path

### For Beginners
Start with **Fractal Explorer** - it's visually stunning and easier to understand than physics simulations.

### For Scientists/Engineers
Go straight to **Physics Visualizer**, **Scientific Visualization Suite**, or **Digital Twin** to see practical applications.

### For Artists
Check out **Procedural Art Generator** first, then explore **Fractal Explorer**.

### For Audio Enthusiasts
Start with **Audio Visualizer** and **Physical Modeling Instrument** to see cross-domain audio synthesis.

### For Game Developers
**Interactive Physics Sandbox** demonstrates emergent behavior and optimization.

### For Full Understanding
Work through all eight examples in order - each builds on concepts from the previous ones.

---

## 🔑 Key Concepts Demonstrated

### Cross-Domain Integration

The showcase examples demonstrate that **Kairo's true power emerges from domain combinations**:

| Domain | Provides | Used By |
|--------|----------|---------|
| **Noise** | Organic texture | All visual examples |
| **Field** | Spatial operations | Physics, PDEs, fractals, digital twins |
| **Palette** | Color mapping | All visualization |
| **Image** | Composition | All visual output |
| **Integrators** | Time evolution | Physics systems, digital twins |
| **Sparse Linear Algebra** | PDE solving | Scientific computing, digital twins |
| **I/O** | Data persistence | Scientific workflows, digital twins |
| **Color** | Blend modes | Art, composition |
| **Audio** | Sound synthesis | Audio visualizer, instruments |
| **RigidBody** | Physics simulation | Physics sandbox |
| **Genetic** | Optimization | Physics sandbox |
| **Cellular** | Pattern formation | Emergent structures, audio viz |

### Design Patterns

**Pattern 1: Compute → Normalize → Colorize → Render**
```python
# Compute (sparse linalg, integrators, noise, etc.)
data = compute_something()

# Normalize
normalized = field.normalize(Field2D(data), 0.0, 1.0)

# Colorize
pal = palette.viridis()

# Render
img = image.from_field(normalized.data, pal)
```

**Pattern 2: Multi-Layer Composition**
```python
# Create layers
layer1 = create_layer_1()
layer2 = create_layer_2()
layer3 = create_layer_3()

# Composite with blend modes
temp = image.blend(layer1, layer2, mode="screen", opacity=0.5)
result = image.blend(temp, layer3, mode="overlay", opacity=0.4)
```

**Pattern 3: Temporal Evolution**
```python
# Setup integrator
integrator = integrators.create_integrator('rk4', derivatives, dt)

# Time loop
for step in range(n_steps):
    state = integrator.step(t, state)

    # Visualize
    if step % save_freq == 0:
        visualize(state)
```

---

## 🚀 Advanced Usage

### Saving Output

All examples output Image objects. To save them:

```python
from morphogen.stdlib import io_storage

# After running an example
img = fractal_explorer.mandelbrot_with_noise_overlay()

# Save as PNG
io_storage.save(img, "fractal.png", format="png")

# Save as JPEG
io_storage.save(img, "fractal.jpg", format="jpeg", quality=95)
```

### Creating Animations

```python
from morphogen.stdlib import io_storage

# Generate frames
frames = []
for i in range(100):
    frame = generate_frame(i)
    frames.append(frame)

# Save as animated GIF
io_storage.save_animation(frames, "animation.gif", fps=30)

# Or as MP4 video
io_storage.save_video(frames, "animation.mp4", fps=30, codec="h264")
```

### Parameter Exploration

Each example has tunable parameters. Try modifying:

**Fractal Explorer**:
- `max_iter`: More iterations = finer detail (but slower)
- `x_min, x_max, y_min, y_max`: Zoom coordinates
- Palette parameters for different color schemes

**Physics Visualizer**:
- `alpha`, `k`, `F`: Physical parameters
- `dt`, `n_steps`: Time resolution
- Grid size for performance vs quality tradeoff

**Procedural Art**:
- `scale`, `octaves`, `persistence`: Noise characteristics
- Blend mode opacities for different aesthetics
- Palette gradients for color schemes

**Scientific Visualization**:
- Grid resolution for accuracy
- Solver tolerance for convergence
- Boundary conditions for different scenarios

---

## 📊 Performance Notes

### Computational Complexity

| Example | Complexity | Typical Runtime | Memory |
|---------|-----------|-----------------|--------|
| Fractal Explorer | O(n²) per iteration | 5-30s | Moderate |
| Physics Visualizer | O(n² × steps) | 10-60s | Moderate-High |
| Procedural Art | O(n²) per layer | 5-20s | Moderate |
| Scientific Viz | O(n² log n) (sparse solve) | 10-40s | High |

### Optimization Tips

1. **Start small**: Use smaller grid sizes (128×128) for testing
2. **Profile first**: Time individual demos to find bottlenecks
3. **Parallel potential**: Many operations are embarrassingly parallel
4. **Caching**: Reuse computed fields when possible
5. **Sparse matrices**: Essential for large PDEs

---

## 🎯 Applications

### Research & Science
- **Computational Physics**: Simulate physical systems
- **Numerical Analysis**: Solve PDEs efficiently
- **Scientific Visualization**: Publication-quality figures

### Art & Design
- **Generative Art**: Algorithmic artwork
- **Procedural Textures**: Game assets, VFX
- **Album Covers**: Unique visual designs

### Education
- **Physics Education**: Interactive simulations
- **Mathematics**: Visualize abstract concepts
- **Computer Graphics**: Teach rendering techniques

### Industry
- **Game Development**: Procedural content generation
- **Engineering**: CFD, heat transfer, acoustics
- **Data Visualization**: Scientific data presentation

---

## 🔧 Extending the Examples

### Adding Your Own Demo

Use this template:

```python
"""My Custom Demo - Cross-Domain Showcase

Demonstrates integration of:
- Domain 1
- Domain 2
- Domain 3
"""

import numpy as np
from morphogen.stdlib import domain1, domain2, domain3


def my_demo():
    """Demo: Custom visualization."""
    print("Demo: My Custom Visualization")
    print("-" * 60)

    # Setup
    print("  - Setting up...")

    # Computation
    print("  - Computing...")

    # Visualization
    print("  - Visualizing...")

    print(f"  ✓ Generated visualization")
    print()

    return result


def main():
    """Run demo."""
    print("=" * 60)
    print("MY CUSTOM SHOWCASE")
    print("=" * 60)
    print()

    my_demo()

    print("=" * 60)
    print("DEMO COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## 📖 Related Documentation

- **[Main Examples README](../README.md)**: Overview of all examples
- **[SPECIFICATION.md](../../SPECIFICATION.md)**: Language reference
- **[ARCHITECTURE.md](../../ARCHITECTURE.md)**: System architecture
- **Domain Documentation**:
  - `morphogen/stdlib/noise.py`: Noise generation
  - `morphogen/stdlib/palette.py`: Color palettes
  - `morphogen/stdlib/field.py`: Field operations
  - `morphogen/stdlib/integrators.py`: Time integration
  - `morphogen/stdlib/sparse_linalg.py`: Sparse linear algebra
  - `morphogen/stdlib/image.py`: Image processing
  - `morphogen/stdlib/io_storage.py`: I/O operations

---

## 🤝 Contributing

Want to add your own showcase example? We'd love to see it!

**Requirements**:
- Must integrate **at least 3 domains**
- Should demonstrate **practical applications**
- Include comprehensive **documentation**
- Follow the **template structure**

Submit a PR to the `examples/showcase/` directory!

---

## 📝 License

All examples are licensed under the same license as Kairo (see [LICENSE](../../LICENSE)).

Feel free to use, modify, and build upon these examples for your own projects!

---

## 🌟 Gallery

*(In production, this section would include screenshots/renders from each example)*

**Fractal Explorer**: Stunning Mandelbrot and Julia sets with advanced coloring

**Physics Visualizer**: Real-time physical simulations with scientific accuracy

**Procedural Art**: Gallery-quality generative artwork

**Scientific Visualization**: Publication-ready PDE solutions

**Audio Visualizer**: Real-time audio-reactive visualizations with video export

**Interactive Physics Sandbox**: Particle collisions, gravity fields, emergent behavior

**Physical Modeling Instrument**: Realistic instrument synthesis using physical models

**Digital Twin**: Industrial process simulation and monitoring

---

**Happy exploring!** 🚀

These examples represent the cutting edge of what's possible when you combine Kairo's domains. Use them as inspiration, learning resources, or starting points for your own projects.

**Last Updated**: 2025-11-19
**Kairo Version**: v0.10.0+
**Examples**: 8 showcase demonstrations
