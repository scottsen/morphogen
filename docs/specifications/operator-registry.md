# SPEC: Morphogen Operator Registry

**Version:** 2.0 Draft
**Status:** RFC
**Last Updated:** 2025-11-15

---

## Overview

The **Operator Registry** is the single source of truth for all operations in Morphogen. It defines:

1. **Operator signatures** — Inputs, outputs, parameters with types/units
2. **Determinism metadata** — Tier (strict/repro/live) and behavior
3. **Numeric properties** — Order, symplectic, conservative, etc.
4. **Transform metadata** — Domain changes (time→frequency, etc.)
5. **Lowering hints** — MLIR tiling, vectorization, memory patterns
6. **Profile overrides** — Per-profile behavior customization
7. **Layered architecture** — 7 semantic layers from kernel to domain-specific ops

**Design Principle:** If it's in the registry, it's documented, validated, and ready for codegen. The registry is organized into 7 semantic layers, from foundational kernel operations to domain-specific applications (audio, physics, finance, fractals, etc.).

---

## Registry Schema (JSON)

### Top-Level Structure

```json
{
  "version": "1.0",
  "operators": [
    {
      /* Operator definition */
    }
  ]
}
```

---

## Operator Definition

### Minimal Example

```json
{
  "name": "sine",
  "category": "oscillator",
  "description": "Sine wave oscillator",
  "inputs": [],
  "outputs": [
    {"name": "out", "type": "Stream<f32,time,audio>", "description": "Audio output"}
  ],
  "params": [
    {
      "name": "freq",
      "type": "f32<Hz>",
      "default": "440Hz",
      "description": "Frequency in Hertz"
    },
    {
      "name": "phase",
      "type": "f32<rad>",
      "default": "0rad",
      "description": "Initial phase"
    }
  ],
  "determinism": "strict",
  "rate": "audio"
}
```

---

### Full Operator Schema

```json
{
  "name": "string (required)",
  "category": "string (required)",
  "description": "string (required)",

  "inputs": [
    {
      "name": "string (required)",
      "type": "string (required)",
      "description": "string (optional)",
      "optional": "bool (default: false)"
    }
  ],

  "outputs": [
    {
      "name": "string (required)",
      "type": "string (required)",
      "description": "string (optional)"
    }
  ],

  "params": [
    {
      "name": "string (required)",
      "type": "string (required)",
      "default": "string (optional)",
      "description": "string (optional)",
      "range": "[min, max] (optional)",
      "enum": "['val1', 'val2', ...] (optional)"
    }
  ],

  "determinism": "strict | repro | live (required)",
  "rate": "audio | control | visual | sim (required)",

  "numeric_properties": {
    "order": "int (optional)",
    "symplectic": "bool (default: false)",
    "conservative": "bool (default: false)",
    "reversible": "bool (default: false)"
  },

  "transform_metadata": {
    "input_domain": "string (optional)",
    "output_domain": "string (optional)",
    "transform_type": "string (optional)"
  },

  "lowering_hints": {
    "tile_sizes": "[int, int, ...] (optional)",
    "vectorize": "bool (default: true)",
    "parallelize": "bool (default: true)",
    "memory_pattern": "string (optional)"
  },

  "profile_overrides": {
    "strict": {/* Profile-specific settings */},
    "repro": {/* Profile-specific settings */},
    "live": {/* Profile-specific settings */}
  },

  "implementation": {
    "python": "string (module path)",
    "mlir": "string (dialect.op)",
    "lowering_template": "string (optional)"
  },

  "tests": [
    {
      "name": "string",
      "inputs": {},
      "params": {},
      "expected_outputs": {},
      "tolerance": "float (default: 0)"
    }
  ]
}
```

---

## Layered Operator Architecture

Morphogen's operator registry is organized into **7 semantic layers**, from foundational kernel operations to domain-specific applications. Each layer builds on the layers below it, creating a coherent operator universe.

### Layer 1: Kernel Core Operators

**Foundational, domain-agnostic operations** that form the base of all higher-level operations.

| Operator | Category | Description |
|----------|----------|-------------|
| `cast` | core | Type conversion between numeric types |
| `unit.cast` | core | Unit domain conversion (Hz↔rad/s, dB↔linear, etc.) |
| `shape` | core | Query shape/dimensions of data |
| `rate.change` | core | Change sample rate or temporal resolution |
| `domain.change` | core | Trivial domain changes (not transforms like FFT) |

**Example Metadata:**
```json
{
  "name": "cast",
  "category": "core",
  "layer": 1,
  "inputs": [{"name": "x", "type": "Any"}],
  "params": {"to": {"type": "Type"}},
  "determinism": "strict",
  "lowering": {"dialect": "morphogen.core", "template": "cast_generic"}
}
```

---

### Layer 2: Transform Operators

**First-class domain transforms** — Fourier-family operations and coordinate mappings.

| Operator | Transform Type | Domain Change |
|----------|----------------|---------------|
| `fft` | Fourier | time → frequency |
| `ifft` | Fourier | frequency → time |
| `stft` | Fourier | time → time-frequency |
| `istft` | Fourier | time-frequency → time |
| `dct` | Cosine | time → frequency |
| `idct` | Cosine | frequency → time |
| `wavelet` | Wavelet | time → time-scale |
| `iwavelet` | Wavelet | time-scale → time |
| `space.to_kspace` | Spatial | space → k-space (reciprocal) |
| `kspace.to_space` | Spatial | k-space → space |
| `laplacian.spectral` | Spectral | PDE in frequency domain |
| `transform.reparam` | Coordinate | Warp/scale/translate coordinates |
| `mel` | Perception | frequency → mel scale |
| `mel.inverse` | Perception | mel scale → frequency |

**Example Metadata:**
```json
{
  "name": "fft",
  "category": "transform",
  "layer": 2,
  "inputs": [{"name": "sig", "type": "Stream<f32,time>"}],
  "params": {
    "window": {"type": "Enum", "default": "hann"},
    "normalize": {"type": "Bool", "default": true}
  },
  "domain_change": {"from": "time", "to": "frequency"},
  "determinism": "strict",
  "lowering": {"dialect": "morphogen.transform", "template": "fft_1d"},
  "numeric_properties": {"invertible": true, "inverse_op": "ifft"}
}
```

See **[transform.md](transform.md)** for complete transform dialect specification.

---

### Layer 3: Stochastic Operators

**Randomness and Monte Carlo machinery** — used across physics, finance, graphics, and audio.

| Operator | Type | Description |
|----------|------|-------------|
| `rng.uniform` | RNG | Uniform random numbers |
| `rng.normal` | RNG | Gaussian random numbers |
| `rng.poisson` | RNG | Poisson process |
| `rng.bernoulli` | RNG | Bernoulli trials |
| `stochastic.brownian` | SDE | Brownian motion process |
| `stochastic.geometric_bm` | SDE | Geometric Brownian motion |
| `stochastic.ou` | SDE | Ornstein-Uhlenbeck process |
| `stochastic.jump_diffusion` | SDE | Jump diffusion process |
| `mc.sample` | Monte Carlo | Sample from distribution |
| `mc.expectation` | Monte Carlo | Compute expectation |
| `mc.path` | Monte Carlo | Generate sample paths |
| `mc.antithetic` | Monte Carlo | Antithetic variance reduction |

**Example Metadata:**
```json
{
  "name": "stochastic.brownian",
  "category": "stochastic",
  "layer": 3,
  "inputs": [],
  "params": {
    "sigma": {"type": "Ctl<f32>", "default": 1.0},
    "dt": {"type": "Rate", "default": "1ms"},
    "seed": {"type": "u64", "required": true}
  },
  "outputs": [{"type": "Stream<f32,time>"}],
  "determinism": "repro",
  "lowering": {"dialect": "morphogen.stream", "template": "brownian_step"}
}
```

---

### Layer 4: Physics & Field Operators

**PDE solvers, integrators, and spatial operations.**

#### 4a. Integrators (ODE/SDE)

| Operator | Order | Symplectic | Description |
|----------|-------|------------|-------------|
| `integrate.euler` | 1 | No | Explicit Euler |
| `integrate.verlet` | 2 | Yes | Velocity Verlet (symplectic) |
| `integrate.rk4` | 4 | No | 4th-order Runge-Kutta |
| `integrate.split` | — | — | Operator splitting |

**Metadata includes:**
- `order`: Accuracy order
- `symplectic`: Energy conservation property
- `stability_region`: Timestep stability bounds

**Example:**
```json
{
  "name": "integrate.verlet",
  "category": "integrator",
  "layer": 4,
  "params": {
    "dt": {"type": "Rate"},
    "force": {"type": "Fn"}
  },
  "numeric_properties": {
    "order": 2,
    "symplectic": true,
    "conservative": true
  },
  "lowering": {"dialect": "morphogen.stream", "template": "verlet_step"}
}
```

#### 4b. PDE Field Operators

| Operator | Description |
|----------|-------------|
| `field.gradient` | Compute spatial gradient |
| `field.divergence` | Compute divergence |
| `field.laplacian` | Compute Laplacian |
| `field.convolve` | Spatial convolution |
| `field.boundary.apply` | Apply boundary conditions |

#### 4c. Particle/Grid Coupling

| Operator | Description |
|----------|-------------|
| `particle.update` | Update particle positions |
| `particle.to_field` | Scatter particles to grid |
| `field.sample_at` | Sample field at particle positions |

---

### Layer 5: Audio / DSP Operators

**Classic audio synthesis and processing operations.**

#### 5a. Oscillators

| Operator | Waveform |
|----------|----------|
| `sine` | Sine wave |
| `saw` | Sawtooth wave |
| `square` | Square wave |
| `triangle` | Triangle wave |
| `noise` | White noise (seeded) |

#### 5b. Filters

| Operator | Type |
|----------|------|
| `lpf` | Low-pass filter |
| `hpf` | High-pass filter |
| `bpf` | Band-pass filter |
| `svf` | State-variable filter |
| `peq` | Parametric EQ |

#### 5c. Time-Domain Effects

| Operator | Effect |
|----------|--------|
| `delay` | Delay line |
| `reverb` | Reverb (FDN/convolution) |
| `compressor` | Dynamics compressor |
| `limiter` | Peak limiter |

#### 5d. Spectral Operations

| Operator | Description |
|----------|-------------|
| `spectral.sharpen` | Sharpen spectral peaks |
| `spectral.morph` | Morph between spectra |

**Example:**
```json
{
  "name": "lpf",
  "category": "filter",
  "layer": 5,
  "inputs": [{"name": "in", "type": "Stream<f32,time,audio>"}],
  "params": {
    "cutoff": {"type": "f32<Hz>", "default": "1000Hz"},
    "resonance": {"type": "f32", "default": 0.707}
  },
  "outputs": [{"type": "Stream<f32,time,audio>"}],
  "determinism": "strict",
  "lowering": {"dialect": "morphogen.audio", "template": "biquad_lpf"}
}
```

---

### Layer 6: Fractal / Visual / Geometry Operators

**Fractal generation, field visualization, geometric modeling, and mesh processing.**

**Extended with TiaCAD-inspired geometry operators (v0.9+)**

#### 6a. Coordinate Mapping

| Operator | Description |
|----------|-------------|
| `fractal.map_plane` | Map complex plane coordinates |
| `field.reparam` | Warp field coordinates |

#### 6b. Geometry Primitives (3D Solids)

**TiaCAD-inspired declarative CAD operators**

| Operator | Description | Anchors Generated |
|----------|-------------|-------------------|
| `geom.box` | Rectangular box (width, height, depth) | `.center`, `.face_{top,bottom,left,right,front,back}`, `.corner_{...}`, `.edge_{...}` |
| `geom.sphere` | Sphere (radius) | `.center`, `.pole_{north,south}`, `.equator` |
| `geom.cylinder` | Cylinder (radius, height) | `.center`, `.face_{top,bottom}`, `.axis`, `.edge_{top,bottom}` |
| `geom.cone` | Cone (radius_bottom, radius_top, height) | `.center`, `.face_{top,bottom}`, `.apex` |
| `geom.torus` | Torus (major_radius, minor_radius) | `.center`, `.axis` |

**Determinism:** Strict

#### 6c. Sketch Operations (2D → 2D)

| Operator | Description |
|----------|-------------|
| `sketch.rectangle` | Rectangle (width, height) |
| `sketch.circle` | Circle (radius) |
| `sketch.polygon` | Polygon from points |
| `sketch.regular_polygon` | Regular n-sided polygon |
| `sketch.arc` | Circular arc (radius, start_angle, end_angle) |
| `sketch.spline` | Spline curve from control points |
| `sketch.union` | Boolean union of sketches |
| `sketch.difference` | Boolean difference of sketches |
| `sketch.offset` | Parallel offset (expand/contract) |

**Determinism:** Strict

#### 6d. Extrusion & Revolution (2D → 3D)

| Operator | Description |
|----------|-------------|
| `extrude` | Extrude sketch vertically |
| `revolve` | Revolve sketch around axis |
| `loft` | Loft between multiple sketches |
| `sweep` | Sweep profile along path |

**Determinism:** Strict (extrude, revolve), Repro (loft, sweep - spline fitting)

#### 6e. Boolean Operations (3D)

| Operator | Description |
|----------|-------------|
| `geom.union` | Combine solids (addition) |
| `geom.difference` | Subtract solid from another |
| `geom.intersection` | Common volume of solids |
| `geom.symmetric_difference` | XOR of solids |

**Operator overloading supported:**
- `solid_A + solid_B` → union
- `solid_A - solid_B` → difference
- `solid_A & solid_B` → intersection

**Determinism:** Strict (within floating precision)

#### 6f. Pattern Operations

| Operator | Description |
|----------|-------------|
| `pattern.linear` | Linear array (direction, count, spacing) |
| `pattern.circular` | Circular pattern (axis, count, angle) |
| `pattern.grid` | 2D grid pattern (rows, cols, spacing) |
| `pattern.along_path` | Distribute along curve |

**Determinism:** Strict

#### 6g. Finishing Operations

| Operator | Description |
|----------|-------------|
| `geom.fillet` | Round edges (radius) |
| `geom.chamfer` | Bevel edges (distance) |
| `geom.shell` | Hollow out solid (thickness) |
| `geom.draft` | Taper faces (angle, neutral_plane) |
| `geom.offset` | Offset surface (expand/contract) |

**Determinism:** Repro (iterative solvers involved)

#### 6h. Mesh Operations (Discrete Geometry)

| Operator | Description |
|----------|-------------|
| `mesh.from_solid` | Tessellate solid to triangle mesh |
| `mesh.subdivide` | Mesh subdivision (Catmull-Clark, Loop) |
| `mesh.laplacian` | Compute mesh Laplacian matrix |
| `mesh.sample` | Sample field at mesh vertices |
| `mesh.normals` | Compute vertex/face normals |
| `mesh.simplify` | Decimate mesh (edge collapse) |
| `mesh.smooth` | Laplacian smoothing |
| `mesh.to_field` | Rasterize mesh to field |
| `field.to_mesh` | Extract isosurface (Marching Cubes) |

**Dependencies:** Sparse Linear Algebra (for Laplacian), Fields (for rasterization)

**Determinism:** Repro (tessellation, interpolation)

#### 6i. Measurement & Query

| Operator | Description | Returns |
|----------|-------------|---------|
| `geom.measure.volume` | Compute solid volume | `f64[m³]` |
| `geom.measure.area` | Compute face area | `f64[m²]` |
| `geom.measure.length` | Compute edge/wire length | `f64[m]` |
| `geom.measure.bounds` | Compute bounding box | `BoundingBox` |
| `geom.measure.center_of_mass` | Compute COM | `Vec3[m]` |
| `geom.measure.normal` | Compute face normal | `Vec3` (unit vector) |
| `geom.measure.distance` | Distance between objects | `f64[m]` |

**Determinism:** Strict

#### 6j. Anchor Operations (TiaCAD Concept)

| Operator | Description |
|----------|-------------|
| `anchor.create` | Define custom anchor on object |
| `anchor.resolve` | Query anchor by name/pattern |
| `anchor.position` | Get anchor position (Vec3) |
| `anchor.orientation` | Get anchor orientation (Mat3) |
| `anchor.frame` | Get anchor's coordinate frame |
| `object.place` | Place object using anchor alignment |
| `object.align` | Align multiple objects along axis |

**See:** `docs/coordinate-frames.md` for full anchor system specification

**Determinism:** Strict

#### 6k. Fractal Iteration Functions

| Operator | Fractal Type |
|----------|--------------|
| `fractal.mandelbrot` | Mandelbrot set |
| `fractal.julia` | Julia set |
| `fractal.escape_time` | Escape-time algorithm |

#### 6l. Palette / Color Transforms

| Operator | Description |
|----------|-------------|
| `color.smooth` | Smooth color gradients |
| `color.palette` | Apply color palette lookup |

---

### Layer 7: Finance / Quantitative Operators

**Built on stochastic and field layers for quantitative finance.**

#### 7a. Models

| Operator | Model |
|----------|-------|
| `model.black_scholes` | Black-Scholes SDE |
| `model.heston` | Heston stochastic volatility |
| `model.sabr` | SABR model |

#### 7b. Payoffs

| Operator | Instrument |
|----------|------------|
| `payoff.call` | Call option |
| `payoff.put` | Put option |
| `payoff.barrier` | Barrier option |
| `payoff.binary` | Binary/digital option |

#### 7c. Pricing

| Operator | Method |
|----------|--------|
| `price.mc` | Monte Carlo pricing |
| `price.pde_step` | PDE solver step |
| `price.fourier` | Fourier pricing |

**Example:**
```json
{
  "name": "model.heston",
  "category": "finance",
  "layer": 7,
  "params": {
    "kappa": {"type": "f32", "description": "Mean reversion speed"},
    "theta": {"type": "f32", "description": "Long-run variance"},
    "sigma": {"type": "f32", "description": "Volatility of volatility"},
    "rho": {"type": "f32", "description": "Correlation"}
  },
  "outputs": [{"type": "Stream<Vec2<f32>,time>", "description": "[price, variance]"}],
  "determinism": "repro",
  "lowering": {"dialect": "morphogen.stochastic", "template": "heston_euler"}
}
```

---

## Layer Summary Table

| Layer | Operator Types | Examples |
|-------|----------------|----------|
| **1. Core** | cast, domain, rate, shape | `cast`, `rate.change` |
| **2. Transforms** | FFT-family, reparam, spectral, affine | `fft`, `transform.rotate`, `transform.to_coord` |
| **3. Stochastic** | RNG, processes, Monte Carlo | `rng.normal`, `mc.path` |
| **4. Physics/Fields** | integrators, PDEs, grids | `integrate.verlet`, `field.laplacian` |
| **5. Audio** | filters, oscillators, FX | `lpf`, `reverb` |
| **6. Fractals/Visuals/Geometry** | iteration, palette, mapping, CAD, mesh | `fractal.mandelbrot`, `geom.box`, `mesh.laplacian`, `anchor.resolve` |
| **7. Finance** | models, payoffs, pricing | `model.heston`, `price.mc` |

---

## Legacy Category Table

For backward compatibility, operators also have traditional **categories**:

| Category | Description | Examples |
|----------|-------------|----------|
| `oscillator` | Waveform generators | sine, saw, square, triangle, noise |
| `filter` | Frequency filters | lpf, hpf, bpf, notch, allpass |
| `envelope` | Amplitude envelopes | adsr, ar, exp_decay |
| `effect` | Audio effects | reverb, delay, chorus, flanger |
| `transform` | Domain transforms | fft, ifft, stft, istft, dct |
| `math` | Mathematical ops | add, mul, sin, cos, exp, log |
| `field` | Spatial field ops | advect, diffuse, project, laplacian |
| `agent` | Particle/agent ops | spawn, force_sum, integrate |
| `visual` | Rendering ops | colorize, render, blend |
| `control` | Control flow | gate, switch, seq |
| `utility` | Utilities | resample, delay, mix |

---

## Type System Integration

### Input/Output Types

Types follow the **type-system.md** definitions:

```json
{
  "inputs": [
    {"name": "in", "type": "Stream<f32,time,audio>"}
  ],
  "outputs": [
    {"name": "out", "type": "Stream<f32,time,audio>"}
  ]
}
```

**Supported Types:**
- `Stream<T,Domain,Rate>` — Time-varying signals
- `Field<T,Domain>` — Spatial fields
- `Evt<A>` — Event streams
- Scalar types: `f32`, `f64`, `i32`, `bool`, etc.
- Vector types: `Vec2<f32>`, `Vec3<f32>`, etc.
- Complex types: `Complex<f32>`, `Complex<f64>`

---

### Parameter Types with Units

Parameters must include **unit annotations**:

```json
{
  "params": [
    {"name": "freq", "type": "f32<Hz>", "default": "440Hz"},
    {"name": "cutoff", "type": "f32<Hz>", "default": "2kHz"},
    {"name": "time", "type": "f32<s>", "default": "0.5s"},
    {"name": "gain", "type": "f32<dB>", "default": "-6dB"},
    {"name": "phase", "type": "f32<rad>", "default": "0rad"},
    {"name": "ratio", "type": "f32", "default": "0.5"}  // Unitless
  ]
}
```

**Validation:**
```python
def validate_param_value(param_def, value):
    """Validate parameter value against definition."""
    value_numeric, value_unit = parse_unit(value)
    param_type, param_unit = parse_type_unit(param_def["type"])

    if param_unit and value_unit != param_unit:
        raise ValueError(f"Unit mismatch: {value_unit} != {param_unit}")

    if "range" in param_def:
        min_val, max_val = param_def["range"]
        if not (min_val <= value_numeric <= max_val):
            raise ValueError(f"Value {value_numeric} out of range [{min_val}, {max_val}]")
```

---

## Determinism Metadata

Every operator declares its **determinism tier**:

```json
{
  "determinism": "strict",
  "determinism_rationale": "Uses Philox RNG with explicit seed",
  "profile_overrides": {
    "strict": {
      "rng": "philox",
      "precision": "f64"
    },
    "repro": {
      "rng": "philox",
      "precision": "f32"
    },
    "live": {
      "rng": "philox_fast",
      "precision": "f32"
    }
  }
}
```

**Determinism Tiers:**
- **strict** — Bit-exact across devices/runs
- **repro** — Deterministic within FP precision
- **live** — Replayable but not bit-exact

---

## Numeric Properties

For numerical algorithms (integrators, solvers), declare **numeric properties**:

```json
{
  "name": "rk4",
  "category": "integrator",
  "numeric_properties": {
    "order": 4,              // 4th-order accurate
    "symplectic": false,     // Not symplectic
    "conservative": false,   // Not energy-conserving
    "reversible": false,     // Not time-reversible
    "explicit": true,        // Explicit method
    "adaptive": false        // Fixed timestep
  }
}
```

**Use Cases:**
- Inform users about algorithm properties
- Enable validation (e.g., "use symplectic for Hamiltonian systems")
- Guide optimizer selection

---

## Transform Metadata

Operators that change domains (FFT, STFT, etc.) declare **transform metadata**:

```json
{
  "name": "fft",
  "category": "transform",
  "transform_metadata": {
    "input_domain": "time",
    "output_domain": "frequency",
    "transform_type": "fourier",
    "invertible": true,
    "inverse_op": "ifft"
  },
  "params": [
    {
      "name": "window",
      "type": "string",
      "default": "hann",
      "enum": ["hann", "hamming", "blackman", "kaiser", "rectangular"]
    },
    {
      "name": "norm",
      "type": "string",
      "default": "ortho",
      "enum": ["ortho", "forward", "backward"]
    }
  ]
}
```

**Validation:**
```python
def validate_transform(op, input_type):
    """Validate transform is legal for input type."""
    if input_type.domain != op.transform_metadata["input_domain"]:
        raise ValueError(
            f"Transform {op.name} expects domain={op.transform_metadata['input_domain']}, "
            f"got {input_type.domain}"
        )
```

---

## Lowering Hints

Operators provide **lowering hints** to guide MLIR code generation:

```json
{
  "name": "convolution",
  "category": "effect",
  "lowering_hints": {
    "tile_sizes": [16, 16],        // Tile spatial dims
    "vectorize": true,             // Enable vectorization
    "parallelize": true,           // Enable parallelization
    "memory_pattern": "streaming", // "streaming", "random", "stencil"
    "prefer_fft": true,            // Use FFT for large IRs
    "partition_size": 8192         // FFT partition size
  }
}
```

**Memory Patterns:**
- `streaming` — Sequential access (enable prefetch)
- `random` — Random access (disable prefetch)
- `stencil` — Neighborhood access (tile for cache locality)

---

## Profile Overrides

Operators can customize behavior per profile:

```json
{
  "name": "reverb",
  "profile_overrides": {
    "strict": {
      "fft_provider": "reference",
      "ir_cache": false,
      "precision": "f64"
    },
    "repro": {
      "fft_provider": "fftw",
      "ir_cache": true,
      "precision": "f32"
    },
    "live": {
      "fft_provider": "fastest",
      "ir_cache": true,
      "precision": "f32",
      "partition_size": 2048  // Smaller for low latency
    }
  }
}
```

---

## Implementation References

### Python Implementation

```json
{
  "implementation": {
    "python": "morphogen.stdlib.oscillators.sine"
  }
}
```

**Python function signature:**
```python
def sine(freq: f32<Hz>, phase: f32<rad> = 0.0) -> Stream<f32, time, audio>:
    """Sine wave oscillator."""
    ...
```

---

### MLIR Dialect

```json
{
  "implementation": {
    "mlir": "morphogen.signal.sine"
  }
}
```

**MLIR operation:**
```mlir
%out = morphogen.signal.sine %freq, %phase : (f32, f32) -> !morphogen.stream<f32>
```

---

### Lowering Template

For custom lowering logic:

```json
{
  "implementation": {
    "lowering_template": "templates/fft_lowering.mlir.j2"
  }
}
```

**Template (Jinja2):**
```mlir
func.func @fft_{{op.id}}(%input: tensor<{{size}}xf32>) -> tensor<{{size}}xcomplex<f32>> {
  // FFT-specific lowering
  {{#if profile.strict}}
    %result = fft.reference %input : tensor<{{size}}xf32> -> tensor<{{size}}xcomplex<f32>>
  {{else}}
    %result = fft.vendor %input : tensor<{{size}}xf32> -> tensor<{{size}}xcomplex<f32>>
  {{/if}}
  return %result : tensor<{{size}}xcomplex<f32>>
}
```

---

## Golden Test Vectors

Every operator must include **golden test vectors** for validation:

```json
{
  "tests": [
    {
      "name": "sine_440hz_strict",
      "params": {"freq": "440Hz", "phase": "0rad"},
      "duration": "1s",
      "sample_rate": 48000,
      "profile": "strict",
      "expected_output_hash": "sha256:abc123...",
      "expected_output_samples": [0.0, 0.0574, 0.1144, ...],  // First 10 samples
      "tolerance": 0.0  // Bit-exact
    },
    {
      "name": "sine_440hz_repro",
      "params": {"freq": "440Hz", "phase": "0rad"},
      "duration": "1s",
      "sample_rate": 48000,
      "profile": "repro",
      "expected_output_hash": "sha256:def456...",
      "tolerance": 1e-7  // Within FP precision
    }
  ]
}
```

**Validation:**
```python
def run_golden_test(op_def, test):
    """Run golden test for operator."""
    op = instantiate_operator(op_def, test["params"])
    output = execute_operator(op, test["duration"], test["sample_rate"], test["profile"])

    if test["tolerance"] == 0:
        # Bit-exact comparison
        expected = np.array(test["expected_output_samples"])
        assert np.array_equal(output[:len(expected)], expected)
    else:
        # Floating-point comparison
        expected = np.array(test["expected_output_samples"])
        assert np.allclose(output[:len(expected)], expected, rtol=test["tolerance"])
```

---

## Registry Operations

### Loading Registry

```python
def load_registry(path="morphogen/registry/operators.json"):
    """Load operator registry from JSON."""
    with open(path) as f:
        data = json.load(f)

    registry = OperatorRegistry(version=data["version"])

    for op_def in data["operators"]:
        op = OperatorDefinition.from_dict(op_def)
        registry.register(op)

    return registry
```

---

### Querying Registry

```python
# Get operator by name
op = registry.get("sine")

# Get operators by category
oscillators = registry.get_by_category("oscillator")

# Get operators by determinism tier
strict_ops = registry.get_by_determinism("strict")

# Search operators
results = registry.search("filter", category="filter")
```

---

### Validating Operator Invocation

```python
def validate_operator_call(op_def, params, inputs):
    """Validate operator call against registry definition."""

    # Check all required params provided
    for param_def in op_def.params:
        if "default" not in param_def and param_def["name"] not in params:
            raise ValueError(f"Missing required parameter: {param_def['name']}")

    # Check parameter types and units
    for param_name, param_value in params.items():
        param_def = op_def.get_param(param_name)
        validate_param_value(param_def, param_value)

    # Check input types
    for input_def in op_def.inputs:
        if input_def["name"] in inputs:
            input_value = inputs[input_def["name"]]
            if not types_compatible(input_value.type, input_def["type"]):
                raise TypeError(
                    f"Input type mismatch: {input_value.type} != {input_def['type']}"
                )
```

---

## Codegen from Registry

### Generate Python Stubs

```python
def generate_python_stub(op_def):
    """Generate Python function stub from operator definition."""

    params_sig = ", ".join(
        f"{p['name']}: {p['type']}" + (f" = {p['default']}" if "default" in p else "")
        for p in op_def.params
    )

    inputs_sig = ", ".join(f"{i['name']}: {i['type']}" for i in op_def.inputs)

    outputs_sig = ", ".join(o["type"] for o in op_def.outputs)
    if len(op_def.outputs) == 1:
        return_type = outputs_sig
    else:
        return_type = f"Tuple[{outputs_sig}]"

    all_params = (inputs_sig + ", " + params_sig) if inputs_sig and params_sig else (inputs_sig or params_sig)

    return f"""
def {op_def.name}({all_params}) -> {return_type}:
    \"\"\"{op_def.description}\"\"\"
    # Implementation here
    pass
"""
```

---

### Generate MLIR Stubs

```python
def generate_mlir_stub(op_def):
    """Generate MLIR operation stub from operator definition."""

    inputs_mlir = ", ".join(
        f"%{i['name']}: {mlir_type(i['type'])}" for i in op_def.inputs
    )

    params_mlir = ", ".join(
        f"{p['name']}: {mlir_type(p['type'])}" for p in op_def.params
    )

    outputs_mlir = ", ".join(mlir_type(o["type"]) for o in op_def.outputs)

    return f"""
def {op_def.category}.{op_def.name}({inputs_mlir}, {params_mlir}) -> ({outputs_mlir})
"""
```

---

### Generate Documentation

```python
def generate_markdown_docs(registry):
    """Generate markdown documentation from registry."""

    md = "# Morphogen Operator Reference\n\n"

    for category in registry.categories:
        md += f"## {category.title()}\n\n"

        for op in registry.get_by_category(category):
            md += f"### {op.name}\n\n"
            md += f"{op.description}\n\n"

            # Parameters table
            md += "**Parameters:**\n\n"
            md += "| Name | Type | Default | Description |\n"
            md += "|------|------|---------|-------------|\n"
            for param in op.params:
                default = param.get("default", "—")
                desc = param.get("description", "")
                md += f"| {param['name']} | {param['type']} | {default} | {desc} |\n"
            md += "\n"

            # Example
            md += "**Example:**\n\n"
            md += f"```morphogen\nlet output = {op.name}("
            md += ", ".join(f"{p['name']}={p.get('default', '...')}" for p in op.params)
            md += ")\n```\n\n"

    return md
```

---

## Example: Complete Operator Definition

```json
{
  "name": "lpf",
  "category": "filter",
  "description": "Second-order lowpass filter (Butterworth)",

  "inputs": [
    {"name": "in", "type": "Stream<f32,time,audio>", "description": "Input signal"}
  ],

  "outputs": [
    {"name": "out", "type": "Stream<f32,time,audio>", "description": "Filtered output"}
  ],

  "params": [
    {
      "name": "cutoff",
      "type": "f32<Hz>",
      "default": "1kHz",
      "description": "Cutoff frequency",
      "range": [20, 20000]
    },
    {
      "name": "q",
      "type": "f32",
      "default": "0.707",
      "description": "Resonance (Q factor)",
      "range": [0.1, 10.0]
    }
  ],

  "determinism": "repro",
  "rate": "audio",

  "numeric_properties": {
    "order": 2,
    "stable": true
  },

  "lowering_hints": {
    "vectorize": true,
    "memory_pattern": "streaming"
  },

  "profile_overrides": {
    "strict": {"precision": "f64"},
    "repro": {"precision": "f32"},
    "live": {"precision": "f32"}
  },

  "implementation": {
    "python": "morphogen.stdlib.filters.lpf",
    "mlir": "morphogen.signal.lpf"
  },

  "tests": [
    {
      "name": "lpf_1khz_sine",
      "params": {"cutoff": "1kHz", "q": "0.707"},
      "input": "sine(440Hz)",
      "duration": "1s",
      "sample_rate": 48000,
      "profile": "repro",
      "expected_magnitude_at_440hz": 0.95,
      "expected_magnitude_at_2khz": 0.25,
      "tolerance": 0.05
    }
  ]
}
```

---

## Implementation Checklist

### Phase 1: Registry Parser
- [ ] JSON schema definition
- [ ] Registry loader (JSON → Python objects)
- [ ] Validation (required fields, type checking)

### Phase 2: Operator Definitions
- [ ] Define 50+ core operators (oscillators, filters, effects, etc.)
- [ ] Add golden test vectors for each operator
- [ ] Validate all definitions

### Phase 3: Codegen
- [ ] Python stub generator
- [ ] MLIR stub generator
- [ ] Documentation generator (markdown)

### Phase 4: Runtime Integration
- [ ] Operator instantiation from registry
- [ ] Parameter validation
- [ ] Type checking

---

## Summary

The Operator Registry provides:

✅ **Single source of truth** — All operators defined in one place
✅ **Type-safe definitions** — Inputs, outputs, params with units
✅ **Determinism metadata** — Explicit tiers and guarantees
✅ **Lowering hints** — Guide MLIR code generation
✅ **Golden tests** — Validate correctness
✅ **Codegen-ready** — Generate stubs, docs, validation

This makes **adding new operators trivial** and ensures consistency across frontends, kernel, and docs.

---

## References

- `type-system.md` — Type definitions used in registry
- `profiles.md` — Profile overrides
- `transform.md` — Transform metadata
- `graph-ir.md` — Graph IR uses operator names from registry
