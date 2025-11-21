# Kairo Example Tools 🛠️

**Utilities for working with Kairo examples and generating outputs**

This directory contains helper scripts and tools for running, testing, and generating outputs from Kairo examples.

---

## Available Tools

### `generate_showcase_outputs.py`

**Comprehensive output generator for creating portfolio-quality outputs from Kairo examples**

#### Features

- ✅ Multi-format export (PNG, MP4, GIF, WAV, FLAC)
- ✅ Quality presets (draft, web, production, print)
- ✅ Cross-domain outputs (audio + visual)
- ✅ Organized output directories with metadata
- ✅ Progress reporting and timing
- ✅ Deterministic seeding for reproducibility

#### Usage

```bash
# List available examples and presets
python examples/tools/generate_showcase_outputs.py --list

# Generate specific example with web preset (default)
python examples/tools/generate_showcase_outputs.py --example reaction_diffusion

# Generate with production quality
python examples/tools/generate_showcase_outputs.py --example fractal --preset production

# Generate with specific formats only
python examples/tools/generate_showcase_outputs.py --example physics --formats mp4,gif

# Generate all examples with draft preset (quick preview)
python examples/tools/generate_showcase_outputs.py --all --preset draft

# Specify custom output directory
python examples/tools/generate_showcase_outputs.py --example fractal --output-dir ./my_outputs

# Use specific seed for reproducibility
python examples/tools/generate_showcase_outputs.py --example reaction_diffusion --seed 123
```

#### Quality Presets

| Preset | Resolution | FPS | Duration | Use Case |
|--------|-----------|-----|----------|----------|
| **draft** | 512×512 | 15 | 5s | Quick preview, testing parameters |
| **web** | 1280×720 | 30 | 30s | Social media, online sharing |
| **production** | 1920×1080 | 30 | 60s | Portfolio, YouTube, high quality |
| **print** | 3840×2160 | 60 | 30s | Publication, 4K displays, print |

#### Output Structure

```
examples/outputs/
└── reaction_diffusion_web_20251116_143052/
    ├── reaction_diffusion_thumbnail.png
    ├── reaction_diffusion_keyframe_0.png
    ├── reaction_diffusion_keyframe_1.png
    ├── reaction_diffusion_keyframe_2.png
    ├── reaction_diffusion_keyframe_3.png
    ├── reaction_diffusion_keyframe_4.png
    ├── reaction_diffusion.mp4
    ├── reaction_diffusion_loop.gif
    ├── reaction_diffusion.wav (if audio generated)
    └── metadata.json
```

#### Metadata File

Each output includes a `metadata.json` file with complete generation information:

```json
{
  "example": "reaction_diffusion",
  "algorithm": "Gray-Scott",
  "parameters": {
    "Du": 0.16,
    "Dv": 0.08,
    "F": 0.06,
    "K": 0.062
  },
  "frames": 900,
  "fps": 30,
  "duration_seconds": 30.0,
  "resolution": [1280, 720],
  "seed": 42,
  "generated_at": "2025-11-16T14:30:52.123456",
  "preset": "web",
  "preset_config": {
    "resolution": [1280, 720],
    "fps": 30,
    "max_duration": 30,
    "audio_sr": 44100,
    "description": "Social media optimized"
  },
  "generation_time_seconds": 45.3
}
```

#### Available Examples

Currently supported examples:

- **reaction_diffusion** - Gray-Scott reaction-diffusion patterns
- **physics** - Physics-based particle simulation
- **fractal** - Mandelbrot fractal zoom animation

*More examples coming soon!*

#### Adding New Examples

To add a new example to the generator:

1. Create a generator function with signature:
   ```python
   def generate_my_example(
       generator: OutputGenerator,
       seed: int = 42,
       duration_seconds: float = None
   ) -> Tuple[List, Optional[np.ndarray], Dict]:
       """
       Generate my example.

       Returns:
           (frames, audio_data, metadata)
       """
       # Your implementation
       return frames, audio_data, metadata
   ```

2. Add to `EXAMPLE_GENERATORS` registry:
   ```python
   EXAMPLE_GENERATORS = {
       'my_example': generate_my_example,
       # ...
   }
   ```

3. Run with:
   ```bash
   python examples/tools/generate_showcase_outputs.py --example my_example
   ```

---

## Post-Processing Tips

### Combining Video + Audio

```bash
# Use ffmpeg to combine MP4 video with WAV audio
ffmpeg -i output.mp4 -i output.wav -c:v copy -c:a aac -strict experimental combined.mp4
```

### GIF Optimization

```bash
# Create optimized GIF with custom palette
ffmpeg -i input.mp4 -vf "palettegen" palette.png
ffmpeg -i input.mp4 -i palette.png -lavfi "paletteuse" optimized.gif
```

### Extract Still Frames

```bash
# Extract frame at specific timestamp
ffmpeg -i video.mp4 -ss 00:00:05.000 -vframes 1 frame.png

# Extract all frames
ffmpeg -i video.mp4 frames/frame_%04d.png
```

### Video Compression

```bash
# Compress for web (reduce file size)
ffmpeg -i input.mp4 -vcodec libx264 -crf 28 compressed.mp4

# CRF values: 18=visually lossless, 23=high quality, 28=good quality (smaller)
```

---

## See Also

- **[Output Generation Guide](../../docs/guides/output-generation.md)** - Comprehensive guide to creating compelling outputs
- **[Showcase Examples](../showcase/README.md)** - Advanced cross-domain demonstrations
- **[Examples README](../README.md)** - Overview of all examples

---

**Happy Creating!** 🎨

*Last Updated: 2025-11-16*
