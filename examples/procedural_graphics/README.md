# Procedural Graphics Examples

This directory contains examples demonstrating Kairo's procedural graphics domains.

## Quick Start

Run the comprehensive demo:

```bash
python demo_all_domains.py
```

This demo showcases:
- **NoiseDomain:** Perlin, fBm, ridged multifractal, marble, turbulence
- **PaletteDomain:** Scientific colormaps, custom gradients, cosine palettes
- **ColorDomain:** Color space conversions, blend modes, temperature
- **ImageDomain:** Image creation, filters, transformations, compositing
- **FieldDomain:** Gradient, divergence, curl, smoothing

## Demo Output

The demo runs 8 different scenarios:

1. **Basic Noise with Palette** - Simple Perlin noise colored with Viridis
2. **Fractal Brownian Motion** - Multi-octave noise with fire palette
3. **Marble Pattern** - Sine waves + turbulence with post-processing
4. **Procedural Terrain** - Ridged fBm with elevation-based coloring + normal maps
5. **Color Manipulation** - Blend modes and color mixing
6. **Field Operations** - Vector fields, divergence, curl, magnitude
7. **Animated Palette** - Palette cycling simulation
8. **Cosine Gradients** - IQ-style procedural gradients

## Key Concepts

### Noise Generation
- **Perlin/Simplex:** Smooth gradient noise for natural textures
- **fBm:** Layered noise for terrain and clouds
- **Worley:** Cellular patterns for stone, water caustics
- **Ridged:** Sharp ridges for mountains
- **Turbulence:** Swirling patterns for marble, fire

### Palette Mapping
- Map scalar fields (noise, simulation data) to colors
- Scientific colormaps (Viridis, Inferno, Plasma, Magma)
- Custom gradients with color stops
- Procedural palettes (cosine gradients)
- Palette cycling for animation

### Image Processing
- Create images from fields or separate channels
- Apply filters (blur, sharpen, edge detection)
- Blend images with Photoshop-style modes
- Warp using displacement fields
- Generate normal maps from height fields

### Field Operations
- Compute gradients, divergence, curl
- Smooth fields using Gaussian or box filters
- Normalize and threshold fields
- Sample at arbitrary positions

## Use Cases

**Fractals:**
```python
mandelbrot_data = compute_mandelbrot((512, 512), max_iter=100)
pal = palette.inferno()
img = image.from_field(mandelbrot_data, pal)
```

**Procedural Textures:**
```python
wood = noise.turbulence((256, 256), scale=0.1, octaves=6)
wood_pal = palette.from_gradient([...])  # Brown gradient
texture = image.from_field(wood.data, wood_pal)
```

**Terrain Generation:**
```python
terrain = noise.ridged_fbm((1024, 1024), scale=0.01, octaves=8)
pal = palette.from_gradient([water, beach, grass, mountain, snow])
heightmap = image.from_field(terrain.data, pal)
normals = image.normal_map_from_heightfield(terrain.data)
```

**Audio Visualization:**
```python
spectrogram = compute_fft(audio_data)  # Returns 2D array
pal = palette.viridis()
vis = palette.map(pal, spectrogram)
```

## Next Steps

- Explore each domain's full API in `/morphogen/stdlib/`
- Read `/docs/PROCEDURAL_GRAPHICS_DOMAINS.md` for complete documentation
- Experiment with different noise types, palettes, and filters
- Combine domains to create complex procedural systems
