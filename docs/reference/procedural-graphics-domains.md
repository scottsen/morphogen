# Procedural Graphics Domains

**Version:** 0.8.1
**Date:** 2025-11-15
**Status:** ✅ IMPLEMENTED

This document describes the four new procedural graphics domains added to Morphogen in v0.8.1:
- **NoiseDomain** - Procedural noise generation
- **PaletteDomain** - Scalar-to-color mapping
- **ColorDomain** - Color manipulation and conversion
- **ImageDomain** - Image operations and compositing
- **FieldDomain (Extended)** - Additional field operations for graphics

---

## Overview

These domains unlock comprehensive procedural graphics capabilities for Morphogen, enabling:
- Fractal visualization (Mandelbrot, Julia sets)
- Plasma effects and retro graphics
- Audio-reactive visualizers
- Scientific data rendering (heatmaps, spectrograms)
- Procedural texture generation
- Simulation visualization (fluids, cellular automata, physics)
- Post-processing effects

---

## 1. NoiseDomain

**Purpose:** Generate procedural noise for textures, terrain, and randomization.

### Layer 1: Basic Noise Types

```python
from morphogen.stdlib import noise

# Perlin noise (smooth gradient noise)
perlin = noise.perlin2d(
    shape=(256, 256),
    scale=0.02,          # Spatial frequency
    octaves=4,           # Number of layers
    persistence=0.5,     # Amplitude decay
    lacunarity=2.0,      # Frequency increase
    seed=42
)

# Simplex noise (improved Perlin, fewer artifacts)
simplex = noise.simplex2d((256, 256), scale=0.05, octaves=3)

# Value noise (interpolated random values)
value = noise.value2d((256, 256), scale=0.03, octaves=4)

# Worley/Voronoi noise (cellular patterns)
worley = noise.worley(
    shape=(256, 256),
    num_points=20,
    distance_metric="euclidean",  # or "manhattan", "chebyshev"
    feature="F1",                  # "F1", "F2", or "F2-F1"
    seed=42
)
```

### Layer 2: Fractal Noise Patterns

```python
# Fractional Brownian Motion (layered noise)
fbm = noise.fbm(
    shape=(512, 512),
    scale=0.01,
    octaves=6,
    persistence=0.5,
    lacunarity=2.0,
    noise_type="perlin",  # or "simplex", "value"
    seed=123
)

# Ridged multifractal (sharp ridges for mountains)
ridged = noise.ridged_fbm((256, 256), scale=0.02, octaves=6)

# Turbulence (swirling patterns)
turbulence = noise.turbulence((256, 256), scale=0.03, octaves=5)

# Marble (sine waves + turbulence)
marble = noise.marble((256, 256), scale=0.05, turbulence_power=3.0)
```

### Layer 3: Vector Fields & Advanced

```python
# 2D vector field (for flow fields, particle advection)
vx, vy = noise.vector_field((256, 256), scale=0.05, octaves=4)

# Gradient field (spatial derivatives)
grad_x, grad_y = noise.gradient_field((256, 256), scale=0.03, octaves=4)

# Plasma (diamond-square algorithm)
plasma = noise.plasma((257, 257), seed=42)
```

### Use Cases
- Terrain generation
- Procedural textures (wood, marble, clouds)
- Turbulence fields for fluids
- Randomized simulation parameters
- Audio-reactive visual effects

---

## 2. PaletteDomain

**Purpose:** Map scalar values to colors for visualization.

### Layer 1: Palette Creation

```python
from morphogen.stdlib import palette

# Scientific colormaps (perceptually uniform)
viridis = palette.viridis(resolution=256)
inferno = palette.inferno(resolution=256)
plasma = palette.plasma(resolution=256)
magma = palette.magma(resolution=256)

# Basic palettes
greyscale = palette.greyscale(resolution=256)
rainbow = palette.rainbow(resolution=256)
hsv_wheel = palette.hsv_wheel(resolution=256)

# Thematic palettes
fire = palette.fire(resolution=256)      # Black → Red → Orange → Yellow → White
ice = palette.ice(resolution=256)        # Black → Blue → Cyan → White

# Custom gradient
custom = palette.from_gradient([
    (0.0, (0, 0, 0)),      # Black at start
    (0.5, (1, 0, 0)),      # Red at middle
    (1.0, (1, 1, 1))       # White at end
], resolution=256, name="custom")

# Procedural cosine gradients (IQ style)
sunset = palette.cosine(
    a=(0.5, 0.5, 0.5),
    b=(0.5, 0.5, 0.5),
    c=(1.0, 1.0, 1.0),
    d=(0.0, 0.10, 0.20),
    resolution=256
)
```

### Layer 2: Palette Transformations

```python
# Shift palette cyclically
shifted = palette.shift(viridis, amount=0.25)

# Cycle over time (for animation)
cycled = palette.cycle(viridis, speed=1.0, time=0.5)

# Flip/reverse
flipped = palette.flip(viridis)

# Interpolate between palettes
mixed = palette.lerp(fire, ice, t=0.5)

# Adjust saturation and brightness
saturated = palette.saturate(viridis, factor=1.5)
brighter = palette.brightness(viridis, factor=1.2)
```

### Layer 3: Palette Application

```python
# Map scalar field to colors
from morphogen.stdlib import noise, image

noise_field = noise.perlin2d((256, 256))
pal = palette.viridis()
rgb_image = palette.map(pal, noise_field.data)

# Cyclic mapping (for phase, angles)
phase_image = palette.map_cyclic(pal, phase_field, frequency=2.0)
```

### Use Cases
- Fractal coloring
- Heatmaps
- Spectrogram visualization
- Scientific data rendering
- Procedural art

---

## 3. ColorDomain

**Purpose:** Color space conversion and manipulation.

### Layer 1: Color Space Conversions

```python
from morphogen.stdlib import color

# RGB ↔ HSV
hsv = color.rgb_to_hsv((1.0, 0.0, 0.0))  # Red → (0.0, 1.0, 1.0)
rgb = color.hsv_to_rgb((0.0, 1.0, 1.0))  # → (1.0, 0.0, 0.0)

# RGB ↔ HSL
hsl = color.rgb_to_hsl((0.5, 0.5, 1.0))
rgb = color.hsl_to_rgb((0.67, 1.0, 0.75))

# Hex ↔ RGB
rgb = color.hex_to_rgb("#FF0000")        # → (1.0, 0.0, 0.0)
hex_str = color.rgb_to_hex((1.0, 0.0, 0.0))  # → "#FF0000"

# Temperature (Kelvin) → RGB
candle = color.temperature_to_rgb(1850)    # Warm orange
daylight = color.temperature_to_rgb(5500)  # Neutral white
blue_sky = color.temperature_to_rgb(10000) # Cool blue
```

### Layer 2: Color Manipulation

```python
# Basic operations
sum_color = color.add((1, 0, 0), (0, 1, 0))        # → (1, 1, 0) yellow
product = color.multiply((1, 0.5, 0), (0.5, 1, 1)) # → (0.5, 0.5, 0)
mixed = color.mix((1, 0, 0), (0, 0, 1), t=0.5)     # → (0.5, 0, 0.5) purple

# Adjustments
brighter = color.brightness((0.5, 0.5, 0.5), factor=1.5)
saturated = color.saturate((0.8, 0.5, 0.5), factor=2.0)
gamma_corrected = color.gamma_correct(linear_rgb, gamma=2.2)
```

### Layer 3: Blend Modes

```python
# Photoshop-style blend modes
overlay = color.blend_overlay(base, blend)
screen = color.blend_screen(base, blend)
multiply = color.blend_multiply(base, blend)
difference = color.blend_difference(base, blend)
soft_light = color.blend_soft_light(base, blend)
```

### Layer 4: Utility Operations

```python
# Posterize (reduce color levels)
posterized = color.posterize((0.7, 0.3, 0.9), levels=4)

# Threshold to black/white
thresholded = color.threshold((0.8, 0.3, 0.5), threshold_value=0.5)
```

### Use Cases
- Color grading
- Visual effects
- Temperature-based lighting
- Procedural coloring
- Simulation visualization

---

## 4. ImageDomain

**Purpose:** Image creation, transformation, filtering, and compositing.

### Layer 1: Image Creation

```python
from morphogen.stdlib import image, noise, palette

# Blank image
blank = image.blank(width=256, height=256, channels=3, fill_value=0.0)

# Solid color
red_img = image.rgb(r=1.0, g=0.0, b=0.0, width=256, height=256)

# From scalar field
noise_field = noise.perlin2d((256, 256))
pal = palette.viridis()
img = image.from_field(noise_field.data, pal)

# Compose from separate channels
r = noise.perlin2d((256, 256), seed=0).data
g = noise.perlin2d((256, 256), seed=1).data
b = noise.perlin2d((256, 256), seed=2).data
img = image.compose(r, g, b)
```

### Layer 2: Transformations

```python
# Scale
scaled = image.scale(img, factor=2.0, method="bilinear")

# Rotate
rotated = image.rotate(img, angle=45, reshape=True)

# Warp using displacement field
vx, vy = noise.vector_field((256, 256))
warped = image.warp(img, (vy.data * 10, vx.data * 10))
```

### Layer 3: Filters

```python
# Blur
blurred = image.blur(img, sigma=2.0)

# Sharpen
sharpened = image.sharpen(img, strength=1.5)

# Edge detection
edges = image.edge_detect(img, method="sobel")  # or "prewitt", "laplacian"

# Morphological operations
eroded = image.erode(img, iterations=2)
dilated = image.dilate(img, iterations=2)
```

### Layer 4: Compositing

```python
# Blend with modes
blended = image.blend(img_a, img_b, mode="overlay", opacity=0.7)
# Modes: "normal", "multiply", "screen", "overlay", "difference", "soft_light"

# Overlay with mask
result = image.overlay(base_img, overlay_img, mask=mask_array)

# Alpha compositing
composited = image.alpha_composite(background, foreground)
```

### Layer 5: Procedural Effects

```python
# Apply palette to image channel
colored = image.apply_palette(img, pal, channel="luminance")
# Channels: "luminance", "r", "g", "b", "saturation"

# Generate normal map from height field
terrain = noise.fbm((256, 256), octaves=6)
normals = image.normal_map_from_heightfield(terrain.data, strength=2.0)

# Gradient map (Photoshop-style)
gradient_mapped = image.gradient_map(img, gradient_palette)
```

### Use Cases
- Procedural texture generation
- Fractal visualization
- Post-processing effects
- Simulation rendering
- Image-based effects

---

## 5. FieldDomain (Extended)

**Purpose:** Field operations for graphics and simulation.

The existing FieldDomain has been extended with additional operations for procedural graphics work.

### New Operations

```python
from morphogen.stdlib import field
from morphogen.stdlib.field import Field2D

# Gradient (spatial derivatives)
scalar_field = Field2D(noise.perlin2d((256, 256)).data)
grad_x, grad_y = field.gradient(scalar_field)

# Divergence (measure of "outflow")
velocity_field = Field2D(velocity_data)  # 2-channel field
div = field.divergence(velocity_field)

# Curl (vorticity, rotation)
curl = field.curl(velocity_field)

# Smoothing
smoothed = field.smooth(noisy_field, iterations=3, method="gaussian")
# Methods: "gaussian", "box"

# Normalization
normalized = field.normalize(field, target_min=0.0, target_max=1.0)

# Thresholding
binary = field.threshold(field, threshold_value=0.5, low_value=0.0, high_value=1.0)

# Sampling at arbitrary positions
positions = np.array([[10.5, 20.3], [50.1, 60.7]])
values = field.sample(field, positions, method="bilinear")
# Methods: "nearest", "bilinear"

# Utility operations
clamped = field.clamp(field, min_value=0.0, max_value=1.0)
absolute = field.abs(field)
speed = field.magnitude(velocity_field)
```

### Use Cases
- Flow field visualization
- Vector field analysis
- Gradient-based effects
- Field smoothing and processing

---

## Complete Example: Procedural Terrain

Here's a complete example combining all domains to create procedural terrain:

```python
from morphogen.stdlib import noise, palette, image, field

# 1. Generate terrain using ridged multifractal
terrain = noise.ridged_fbm(
    shape=(512, 512),
    scale=0.02,
    octaves=6,
    persistence=0.6,
    lacunarity=2.2,
    seed=42
)

# 2. Create terrain-appropriate palette
terrain_pal = palette.from_gradient([
    (0.0, (0.0, 0.2, 0.5)),     # Deep water (blue)
    (0.3, (0.0, 0.5, 0.8)),     # Shallow water
    (0.4, (0.8, 0.7, 0.4)),     # Beach (sand)
    (0.6, (0.2, 0.6, 0.2)),     # Grass (green)
    (0.8, (0.5, 0.4, 0.3)),     # Mountain (brown)
    (1.0, (1.0, 1.0, 1.0))      # Snow (white)
], resolution=256, name="terrain")

# 3. Map terrain to colors
terrain_img = image.from_field(terrain.data, terrain_pal)

# 4. Generate normal map for lighting
normals = image.normal_map_from_heightfield(terrain.data, strength=2.0)

# 5. Add atmospheric perspective (blend with sky color)
sky_img = image.rgb(r=0.5, g=0.7, b=1.0, width=512, height=512)
final = image.blend(terrain_img, sky_img, mode="screen", opacity=0.2)

# 6. Apply slight blur for realism
final = image.blur(final, sigma=0.5)
```

---

## Performance Notes

All domains use NumPy for numerical operations, providing:
- **Vectorization:** Fast array operations
- **Memory efficiency:** Contiguous memory layout
- **Determinism:** Seeded RNGs for reproducibility

Future optimizations planned:
- MLIR lowering for JIT compilation
- GPU acceleration for large-scale generation
- Lazy evaluation for pipeline composition

---

## Architecture Patterns

All domains follow Morphogen's standard patterns:

1. **Operations Class Pattern:**
   ```python
   class NoiseOperations:
       @staticmethod
       def perlin2d(...): ...

   noise = NoiseOperations()  # Singleton instance
   ```

2. **4-Layer Hierarchy:**
   - Layer 1: Atomic operations
   - Layer 2: Basic compositions
   - Layer 3: Complex patterns
   - Layer 4: Presets & utilities

3. **Determinism Guarantees:**
   - All noise functions accept `seed` parameter
   - Bit-exact repeatability with same seed
   - Strict determinism profile

---

## Future Extensions

Planned additions for v0.9:

- **ShaderDomain:** GLSL-like shader pipelines
- **VisionDomain:** Real image processing (convolution, detection)
- **GraphicsDomain:** High-level rendering pipelines
- **3D Support:** Extension to 3D noise, fields, images

---

## References

- **ADR-002:** Cross-domain architectural patterns
- **../architecture/domain-architecture.md:** Overall domain vision
- **Example:** `/examples/procedural_graphics/demo_all_domains.py`

---

**Status:** ✅ Fully implemented and tested
**Next Steps:** Integration with MLIR for GPU acceleration
